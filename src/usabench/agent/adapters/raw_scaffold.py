"""The default adapter: wraps the reference scaffold for the harness.

This is the ``type: raw_scaffold`` agent of ``configs/agents/scaffold_default.yaml``
-- the no-external-dependency, fully-reproducible default agent-under-test
(``docs/infra.md`` ``agent/adapters/raw_scaffold.py``).

Why an adapter at all
---------------------
The harness drives every agent through one uniform interface (``reset`` /
``step`` / ``run``) and one hard invariant: **exactly one
:class:`~usabench.agent.base.AgentAction` per step** so the trace is totally
ordered and budgets debit deterministically (``docs/protocol.md`` §1.2, §5.2).
The reference :class:`~usabench.agent.scaffold.ReActScaffold` already satisfies
that invariant, so this adapter is a thin pass-through that also (a) carries the
:class:`~usabench.agent.scaffold.ScaffoldConfig`, and (b) gives the harness a
single, stable construction point (:func:`build_default_agent`).

How an EXTERNAL framework would be adapted (the unroll contract)
----------------------------------------------------------------
A native agent framework (SWE-agent, OpenHands, a bare provider tool-calling
loop) typically advances a *whole turn* at once -- one model call that may emit
several tool calls, executed internally, with results fed straight back in. To
fit this benchmark, such a framework must be **unrolled into one action per
step** so the harness (not the framework) owns the sandbox, the oracle channel,
the budgets, and the trace. The adapter contract is:

1. **Intercept tools, do not execute them.** The framework's tool executors are
   replaced by stubs that *record* the requested call and hand control back to
   the adapter instead of running it. The harness executes the action and returns
   the :class:`~usabench.agent.base.Observation`.
2. **Queue, then drain.** When the framework's turn yields N tool calls, convert
   each to one :class:`AgentAction`, tag them with a shared ``meta["batch_id"]``,
   enqueue them, and return them one per ``step`` (FIFO). Only call the framework
   again once the queue is empty -- exactly what :class:`ReActScaffold` already
   does internally (see :meth:`ReActScaffold._refill_queue`).
3. **Feed observations back the framework's way.** Each :class:`Observation` the
   harness returns must be threaded back to the matching tool call (by
   ``meta["raw_tool_call_id"]``) before the framework's next turn, so its internal
   transcript stays consistent.
4. **Map control actions.** The framework's "I'm done" / "submit" maps to
   ``declare_done``; its "I'm stuck / give up" maps to ``give_up``. A request for
   human help maps to ``ask_user`` (it routes to the oracle, never a real human).
5. **Never let the framework reach the network or host.** All side effects go
   through the harness-injected sandbox; ``ask_user``/``message_user`` go through
   the harness InteractionBus. The adapter must not let the framework open its own
   sockets or subprocesses.

A concrete external adapter would subclass nothing here -- it would implement the
same ``reset`` / ``step`` surface, holding its own queue and a coroutine/iterator
over the framework's turn loop that is suspended at each tool boundary.
"""

from __future__ import annotations

from collections.abc import Iterator

from usabench.agent.base import AgentAction, Observation, OracleChannel, ToolHandle
from usabench.agent.scaffold import ReActScaffold, ScaffoldConfig
from usabench.agent.tools import DEFAULT_TOOL_NAMES, build_tool_specs
from usabench.core.schema import AgentTaskView
from usabench.llm.client import LLMClient, ToolSpec
from usabench.logging_setup import get_logger

__all__ = ["RawScaffoldAdapter", "build_default_agent"]

logger = get_logger(__name__)


class RawScaffoldAdapter:
    """Default adapter wrapping :class:`ReActScaffold` for uniform harness driving.

    Conforms to the :class:`usabench.agent.base.Agent` protocol. It owns a
    :class:`ReActScaffold` (which already enforces one-action-per-step and native
    multi-tool unrolling) and exposes the harness-facing ``reset`` / ``step`` /
    ``run`` surface plus a stable ``id``/``adapter`` identity for the run manifest.

    Attributes:
        id: A stable agent id recorded in ``episode_start.agent`` of the trace.
    """

    #: Adapter kind string recorded in the run manifest (``docs/infra.md``).
    adapter_type = "raw_scaffold"

    def __init__(
        self,
        client: LLMClient,
        *,
        config: ScaffoldConfig | None = None,
        tool_names: list[str] | None = None,
        agent_id: str = "scaffold-default",
    ) -> None:
        """Build the adapter around an LLM client.

        Args:
            client: The model-under-test client (fake or real).
            config: Optional scaffold config (max_steps, decoding, native tools).
            tool_names: Subset/order of tools to expose; defaults to all.
            agent_id: Stable agent id for the manifest / ``episode_start``.
        """
        self.id = agent_id
        self._client = client
        self._config = config or ScaffoldConfig()
        self._tool_names = list(tool_names) if tool_names is not None else list(DEFAULT_TOOL_NAMES)
        self._scaffold = ReActScaffold(client, self._config)
        self._tool_specs: list[ToolSpec] = build_tool_specs(self._tool_names)

    # --- identity --------------------------------------------------------- #

    def manifest_entry(self) -> dict[str, object]:
        """Return the ``episode_start.agent`` sub-document for the trace/manifest.

        Returns:
            A JSON-able dict describing the agent for reproducibility.
        """
        return {
            "id": self.id,
            "adapter": self.adapter_type,
            "provider": str(getattr(self._client, "provider", "")),
            "decoding": {
                "temperature": self._config.temperature,
                "max_tokens": self._config.max_tokens,
                "native_tools": self._config.native_tools,
            },
            "tools": list(self._tool_names),
            "max_steps": self._config.max_steps,
        }

    def tool_specs(self) -> list[ToolSpec]:
        """Return the tool specs this agent is offered (a defensive copy)."""
        return [t.model_copy(deep=True) for t in self._tool_specs]

    # --- Agent protocol --------------------------------------------------- #

    def reset(self, task_view: AgentTaskView, tools: list[ToolSpec] | None = None) -> None:
        """Reset the wrapped scaffold for a new episode.

        Args:
            task_view: The gold-stripped task projection.
            tools: Tool specs to offer; if ``None``, the adapter's own default
                specs (from its configured ``tool_names``) are used.
        """
        offered = tools if tools is not None else self._tool_specs
        self._scaffold.reset(task_view, offered)
        logger.info("adapter_reset", agent_id=self.id, task_id=task_view.id)

    def step(self, observation: Observation) -> AgentAction:
        """Return exactly one action from the wrapped scaffold.

        Args:
            observation: The rendered result of the previous action.

        Returns:
            The single next :class:`AgentAction`.
        """
        return self._scaffold.step(observation)

    def run(
        self,
        task_view: AgentTaskView,
        tools: ToolHandle,
        oracle_channel: OracleChannel,
    ) -> Iterator[AgentAction]:
        """Drive a full episode via the scaffold's local iterator (smoke-test path).

        Delegates to :meth:`ReActScaffold.run`; the harness normally prefers
        ``step`` so it can interleave budgeting and tracing.

        Args:
            task_view: The gold-stripped task projection.
            tools: The sandbox tool surface.
            oracle_channel: The mediated oracle channel.

        Yields:
            One :class:`AgentAction` at a time.
        """
        self._scaffold.reset(task_view, self._tool_specs)
        yield from self._scaffold.run(task_view, tools, oracle_channel)


def build_default_agent(
    client: LLMClient,
    *,
    max_steps: int = 40,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    native_tools: bool = True,
    tool_names: list[str] | None = None,
    agent_id: str = "scaffold-default",
    system_prompt: str | None = None,
) -> RawScaffoldAdapter:
    """Construct the default :class:`RawScaffoldAdapter` from plain parameters.

    This is the single stable entry point the harness's agent factory calls when a
    run config specifies ``type: raw_scaffold`` (``configs/agents/scaffold_default.yaml``).

    Args:
        client: The model-under-test client.
        max_steps: Scaffold safety cap on emitted actions.
        temperature: Decoding temperature.
        max_tokens: Max completion tokens per call.
        native_tools: Whether to offer/prefer native tool-calling.
        tool_names: Subset/order of tools to expose; ``None`` = all defaults.
        agent_id: Stable agent id for the manifest.
        system_prompt: Optional system-prompt override.

    Returns:
        A ready-to-drive :class:`RawScaffoldAdapter`.
    """
    cfg = ScaffoldConfig(
        max_steps=max_steps,
        temperature=temperature,
        max_tokens=max_tokens,
        native_tools=native_tools,
        system_prompt=system_prompt,
    )
    return RawScaffoldAdapter(
        client, config=cfg, tool_names=tool_names, agent_id=agent_id
    )
