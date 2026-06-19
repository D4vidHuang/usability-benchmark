"""The reference ReAct scaffold (no external agent-framework dependency).

This is the default, fully-reproducible agent-under-test wrapper
(``docs/infra.md`` ``agent/scaffold.py``; ``configs/agents/scaffold_default.yaml``).
It drives an :class:`~usabench.llm.client.LLMClient` in a Reason+Act loop and
emits **exactly one** :class:`~usabench.agent.base.AgentAction` per
:meth:`ReActScaffold.step` -- the one-action-per-step contract that gives the
trace a total order and makes budgets debit deterministically
(``docs/protocol.md`` §1.2).

How one model turn maps to actions
----------------------------------
A single LLM completion may contain several tool calls (native function-calling)
or a single ReAct ``Action:`` block. The scaffold normalizes both into a FIFO
**action queue** and returns one action per ``step``:

* **Native tool calls** -- each :class:`~usabench.llm.client.ToolCall` becomes one
  queued :class:`AgentAction`; they share a ``batch_id`` in ``meta`` so the
  harness can reconstruct that they came from one turn (``docs/protocol.md`` §1.2
  "unrolled ... sharing a ``batch_id``").
* **ReAct text** -- when the provider returns no native tool calls, the scaffold
  parses a ``Thought:`` / ``Action:`` / ``Action Input:`` block out of the text.

The model is only re-prompted once its queued actions are exhausted. The harness
calls ``step(observation)`` repeatedly; the observation is appended to the
transcript as a ``tool`` (or ``user``) message before the next model call.

This module has only core-dep imports (pydantic, structlog) -- it talks to the
model exclusively through the :class:`LLMClient` *protocol*, so it works against
the fake client in tests and any real provider in production.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Iterator
from typing import Any

from usabench.agent.base import AgentAction, Observation, OracleChannel, ToolHandle
from usabench.core.enums import QueryClass
from usabench.core.ids import short_hash
from usabench.core.schema import AgentTaskView
from usabench.llm.client import Completion, LLMClient, Message, ToolCall, ToolSpec
from usabench.logging_setup import get_logger

__all__ = [
    "ReActScaffold",
    "ScaffoldConfig",
    "render_task_prompt",
    "tool_call_to_action",
    "parse_react_action",
]

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #


class ScaffoldConfig:
    """Decoding + loop parameters for the reference scaffold.

    Mirrors ``configs/agents/scaffold_default.yaml``. Kept as a plain dataclass-
    like object (not pydantic) so it stays import-cheap and trivially copyable.

    Attributes:
        max_steps: Hard cap on emitted actions before the scaffold gives up
            (a safety net; the harness also enforces budgets).
        temperature: Sampling temperature for the model-under-test.
        max_tokens: Max completion tokens per model call.
        native_tools: If True, offer tools natively and prefer parsed tool calls;
            if False, force the text ReAct protocol.
        system_prompt: Optional override for the system prompt.
    """

    __slots__ = ("max_steps", "temperature", "max_tokens", "native_tools", "system_prompt")

    def __init__(
        self,
        *,
        max_steps: int = 40,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        native_tools: bool = True,
        system_prompt: str | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.native_tools = native_tools
        self.system_prompt = system_prompt


#: The frozen system prompt for the reference ReAct scaffold. It documents the
#: tool vocabulary and the ask-don't-guess norm that the benchmark measures.
DEFAULT_SYSTEM_PROMPT = """\
You are an autonomous software engineering agent working in a sandboxed workspace.
Your job is to build exactly what the user asked for and nothing more.

You act by calling tools. The available tools are:
  - write_file(path, content)  : create/overwrite a workspace-relative file
  - read_file(path)            : read a workspace-relative file
  - run_cmd(cmd, timeout_s?)   : run a shell command in the sandbox
  - ask_user(text, query_class): ask the human user a question (do not guess when
                                 the request is ambiguous -- ask)
  - message_user(text)         : send the user a short status update
And two control actions, emitted as tools or in the ReAct format below:
  - declare_done(summary, entrypoint): you believe the task is complete; submit it
  - give_up(reason)            : you cannot proceed

Norms:
  * The request is intentionally under-specified. Prefer asking a focused question
    over guessing on load-bearing decisions (output format, scope, constraints).
  * Verify your work by running it before declaring done.
  * Keep the solution minimal; do not add features the user did not ask for.

If native tool-calling is unavailable, respond in the ReAct format:
  Thought: <your reasoning>
  Action: <one of write_file|read_file|run_cmd|ask_user|message_user|declare_done|give_up>
  Action Input: <a single-line JSON object of the action's arguments>
Emit exactly ONE Action per message.
"""


# --------------------------------------------------------------------------- #
# Prompt rendering                                                             #
# --------------------------------------------------------------------------- #


def render_task_prompt(task_view: AgentTaskView) -> str:
    """Render the initial user message from the gold-stripped task projection.

    Only agent-visible fields are used (the projection already excludes gold), so
    this can never leak the hidden spec.

    Args:
        task_view: The agent-visible task projection.

    Returns:
        A prompt string presenting the goal, environment, and capability hints.
    """
    lines: list[str] = [f"# Task: {task_view.title}", "", task_view.user_goal, ""]
    if task_view.user_goal_persona_note:
        lines.append(f"(Note on the user: {task_view.user_goal_persona_note})")
        lines.append("")
    lines.append(f"Deliverable type: {task_view.deliverable_type}")
    lines.append(f"Domain: {task_view.domain}")
    if task_view.required_capabilities:
        lines.append("Relevant capabilities: " + ", ".join(task_view.required_capabilities))
    env = task_view.env
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- base image: {env.base_image}")
    lines.append(f"- network: {env.network}")
    if env.allowlist:
        lines.append(f"- allowlisted hosts: {', '.join(env.allowlist)}")
    if env.allowed_reqs:
        lines.append(f"- allowed dependencies: {', '.join(env.allowed_reqs)}")
    if env.entrypoint_hint:
        lines.append(f"- entrypoint hint: {env.entrypoint_hint}")
    else:
        lines.append("- entrypoint: you decide (not specified)")
    lines.append("")
    lines.append("Begin by deciding whether you need to clarify anything, then build it.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Mapping model output -> AgentAction                                          #
# --------------------------------------------------------------------------- #

#: Tool names that map onto control actions rather than sandbox effects.
_CONTROL_TOOLS = {"declare_done", "give_up"}
#: Tool names the scaffold understands (sandbox + user + control).
_KNOWN_TOOLS = {
    "write_file",
    "read_file",
    "run_cmd",
    "ask_user",
    "message_user",
    "declare_done",
    "give_up",
}


def _coerce_args(arguments: Any) -> dict[str, Any]:
    """Coerce a provider's tool arguments into a plain dict.

    Providers may hand back a parsed dict (the normalized path) or, defensively, a
    JSON string. A non-JSON string is wrapped as ``{"text": ...}`` so a misbehaving
    model still produces a runnable action instead of crashing the loop.
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"text": arguments}
    return {}


def tool_call_to_action(call: ToolCall, *, batch_id: str | None = None) -> AgentAction:
    """Convert one normalized :class:`ToolCall` into one :class:`AgentAction`.

    Args:
        call: The model's normalized tool call.
        batch_id: Optional id shared across all calls from the same model turn,
            stamped into ``meta`` so the harness can reconstruct native batches
            (``docs/protocol.md`` §1.2).

    Returns:
        Exactly one :class:`AgentAction`. Unknown tool names become a
        ``message_user`` action reporting the error so the loop never dies on a
        malformed call.
    """
    args = _coerce_args(call.arguments)
    meta: dict[str, Any] = {"raw_tool_call_id": call.id}
    if batch_id is not None:
        meta["batch_id"] = batch_id
    name = call.name

    if name == "write_file":
        return AgentAction.write_file(
            str(args.get("path", "")), str(args.get("content", "")), **meta
        )
    if name == "read_file":
        return AgentAction.read_file(str(args.get("path", "")), **meta)
    if name == "run_cmd":
        meta_cmd = dict(meta)
        if args.get("timeout_s") is not None:
            meta_cmd["timeout_s"] = int(args["timeout_s"])
        return AgentAction.run_cmd(str(args.get("cmd", "")), **meta_cmd)
    if name == "ask_user":
        qc = _coerce_query_class(args.get("query_class"))
        return AgentAction.ask_user(str(args.get("text", "")), query_class=qc, **meta)
    if name == "message_user":
        return AgentAction.message_user(str(args.get("text", "")), **meta)
    if name == "declare_done":
        return AgentAction.declare_done(
            summary=str(args.get("summary", "")),
            entrypoint=args.get("entrypoint"),
            **meta,
        )
    if name == "give_up":
        return AgentAction.give_up(str(args.get("reason", "")), **meta)

    return AgentAction.message_user(f"[scaffold] unknown tool requested: {name!r}", **meta)


def _coerce_query_class(value: Any) -> QueryClass:
    """Coerce an arbitrary value into a :class:`QueryClass`, defaulting safely."""
    if isinstance(value, QueryClass):
        return value
    try:
        return QueryClass(str(value))
    except (ValueError, TypeError):
        return QueryClass.CLARIFICATION


# ReAct text parsing. Tolerant to spacing/casing variations.
_ACTION_RE = re.compile(r"action\s*:\s*([a-z_]+)", re.IGNORECASE)
_INPUT_RE = re.compile(r"action\s*input\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_react_action(text: str) -> AgentAction | None:
    """Parse a single ReAct ``Action`` / ``Action Input`` block from model text.

    Used only when the provider returns no native tool calls. The parser is
    forgiving: it accepts a JSON object as the action input, and falls back to
    treating the trailing text as the free-text payload for text-shaped actions.

    Args:
        text: The raw model completion text.

    Returns:
        One :class:`AgentAction`, or ``None`` if no parseable action is present
        (the scaffold treats that as a think-aloud ``message_user``).
    """
    if not text:
        return None
    m = _ACTION_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip().lower()
    if name not in _KNOWN_TOOLS:
        return None

    args: dict[str, Any] = {}
    im = _INPUT_RE.search(text)
    if im:
        raw_input = im.group(1).strip()
        # Stop at the next 'Thought:'/'Observation:' marker if the model rambled.
        raw_input = re.split(r"\n\s*(?:thought|observation)\s*:", raw_input, flags=re.IGNORECASE)[0]
        raw_input = raw_input.strip().strip("`").strip()
        try:
            parsed = json.loads(raw_input)
            if isinstance(parsed, dict):
                args = parsed
            else:
                args = {"text": raw_input}
        except json.JSONDecodeError:
            args = {"text": raw_input}

    synthetic = ToolCall(id=f"react-{short_hash(text, 8)}", name=name, arguments=args)
    return tool_call_to_action(synthetic)


# --------------------------------------------------------------------------- #
# The scaffold                                                                 #
# --------------------------------------------------------------------------- #


class ReActScaffold:
    """A minimal, dependency-free ReAct agent implementing :class:`Agent`.

    The scaffold owns a private chat transcript and an action queue. On each
    ``step`` it (a) drains the queue if it is non-empty, else (b) appends the
    latest observation to the transcript, calls the model, and refills the queue
    from the model's tool calls (or parsed ReAct action). It always returns
    exactly one :class:`AgentAction`.

    It conforms structurally to the :class:`usabench.agent.base.Agent` protocol
    (``reset`` / ``step`` / ``run``). The harness owns budgets and termination;
    the scaffold only proposes the next action.
    """

    def __init__(self, client: LLMClient, config: ScaffoldConfig | None = None) -> None:
        """Build the scaffold around an LLM client.

        Args:
            client: Any :class:`LLMClient`-conforming backend (fake or real).
            config: Scaffold decoding/loop config; defaults applied if ``None``.
        """
        self._client = client
        self._cfg = config or ScaffoldConfig()
        self._messages: list[Message] = []
        self._tools: list[ToolSpec] = []
        self._queue: deque[AgentAction] = deque()
        self._n_actions = 0
        self._started = False

    # --- Agent protocol --------------------------------------------------- #

    def reset(self, task_view: AgentTaskView, tools: list[ToolSpec]) -> None:
        """Initialize the scaffold for a fresh episode.

        Seeds the transcript with the system prompt and the rendered task prompt,
        clears the action queue, and stores the offered tool specs.

        Args:
            task_view: The gold-stripped task projection.
            tools: The tool specs available this episode (native tool-calling).
        """
        system = self._cfg.system_prompt or DEFAULT_SYSTEM_PROMPT
        self._messages = [
            Message(role="system", content=system),
            Message(role="user", content=render_task_prompt(task_view)),
        ]
        self._tools = list(tools)
        self._queue.clear()
        self._n_actions = 0
        self._started = True
        logger.info(
            "scaffold_reset",
            task_id=task_view.id,
            n_tools=len(tools),
            native_tools=self._cfg.native_tools,
        )

    def step(self, observation: Observation) -> AgentAction:
        """Emit exactly one action, calling the model only when the queue is empty.

        Args:
            observation: The rendered result of the previous action. On the very
                first step (queue empty, transcript only has the prompt) the
                observation text, if any, is ignored as redundant.

        Returns:
            The single :class:`AgentAction` to execute next.
        """
        if not self._started:
            raise RuntimeError("ReActScaffold.step called before reset()")

        # Drain any actions already queued from a prior multi-tool turn.
        if self._queue:
            return self._pop()

        # Feed the latest observation back into the transcript, then re-prompt.
        self._ingest_observation(observation)

        if self._n_actions >= self._cfg.max_steps:
            logger.warning("scaffold_max_steps", max_steps=self._cfg.max_steps)
            return AgentAction.give_up(f"reached max_steps={self._cfg.max_steps}")

        completion = self._call_model()
        self._record_assistant_turn(completion)
        self._refill_queue(completion)

        if not self._queue:
            # The model produced neither a tool call nor a parseable action: treat
            # its text as a think-aloud status update so the loop makes progress.
            text = completion.text.strip() or "(no actionable output)"
            return AgentAction.message_user(text)
        return self._pop()

    def run(
        self,
        task_view: AgentTaskView,
        tools: ToolHandle,
        oracle_channel: OracleChannel,
    ) -> Iterator[AgentAction]:
        """Drive a full episode as an iterator, executing tools locally.

        Convenience API for callers that do not interpose their own loop: the
        scaffold both *proposes* and *executes* each action (against the supplied
        sandbox + oracle) and yields it. The harness normally uses ``step``
        instead so it can interleave budget accounting and tracing; ``run`` is for
        smoke tests and ad-hoc local driving.

        Args:
            task_view: The gold-stripped task projection.
            tools: A sandbox tool surface (``ToolHandle``).
            oracle_channel: The mediated oracle channel.

        Yields:
            One :class:`AgentAction` per iteration until the agent declares done
            or gives up (or the safety step cap is hit).
        """
        if not self._started:
            self.reset(task_view, self._tools)
        obs = Observation(text="")
        for _ in range(self._cfg.max_steps * 4):  # generous hard ceiling
            action = self.step(obs)
            yield action
            if action.kind in ("declare_done", "give_up"):
                return
            obs = self._execute_locally(action, tools, oracle_channel)

    # --- internals -------------------------------------------------------- #

    def _pop(self) -> AgentAction:
        """Pop the next queued action and bump the action counter."""
        action = self._queue.popleft()
        self._n_actions += 1
        return action

    def _ingest_observation(self, observation: Observation) -> None:
        """Append the previous action's observation to the transcript.

        The first step has nothing to ingest (the prompt is already seeded). After
        that, observations enter as ``tool`` messages so native tool-calling
        providers correctly thread results to their calls; providers that ignore
        tool roles still see the text.
        """
        # Only the initial step has exactly the seeded [system,user] transcript.
        if len(self._messages) <= 2 and not observation.text:
            return
        content = observation.text or ""
        if observation.oracle_text is not None:
            content = f"User says: {observation.oracle_text}"
        self._messages.append(Message(role="tool", content=content, name="observation"))

    def _call_model(self) -> Completion:
        """Invoke the LLM client with the current transcript and tool specs."""
        tools = self._tools if (self._cfg.native_tools and self._tools) else None
        return self._client.chat(
            self._messages,
            tools=tools,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
        )

    def _record_assistant_turn(self, completion: Completion) -> None:
        """Append the model's turn to the transcript for the next round."""
        text = completion.text or ""
        if completion.tool_calls:
            names = ", ".join(c.name for c in completion.tool_calls)
            text = (text + f"\n[tool_calls: {names}]").strip()
        self._messages.append(Message(role="assistant", content=text or "(tool call)"))

    def _refill_queue(self, completion: Completion) -> None:
        """Fill the action queue from one model turn (native calls or ReAct text).

        Native tool calls are unrolled in order, sharing a ``batch_id`` so the
        harness can group them back into one turn. If there are no native calls,
        the scaffold parses a single ReAct action from the text.
        """
        if completion.tool_calls:
            batch_id = (
                f"b-{short_hash(completion.text + completion.tool_calls[0].id, 8)}"
                if len(completion.tool_calls) > 1
                else None
            )
            for call in completion.tool_calls:
                self._queue.append(tool_call_to_action(call, batch_id=batch_id))
            return
        action = parse_react_action(completion.text)
        if action is not None:
            self._queue.append(action)

    def _execute_locally(
        self, action: AgentAction, tools: ToolHandle, oracle_channel: OracleChannel
    ) -> Observation:
        """Execute one action against the local tool/oracle surfaces (``run`` only).

        Returns:
            The :class:`Observation` to feed into the next step. Control actions
            (``declare_done`` / ``give_up``) and pure messages return acks.
        """
        kind = action.kind
        if kind == "write_file":
            return tools.write_file(action.path or "", action.content or "")
        if kind == "read_file":
            return tools.read_file(action.path or "")
        if kind == "run_cmd":
            timeout = action.meta.get("timeout_s")
            if timeout is not None:
                return tools.run_cmd(action.cmd or "", timeout_s=int(timeout))
            return tools.run_cmd(action.cmd or "")
        if kind == "ask_user":
            qc = action.query_class or QueryClass.CLARIFICATION
            return oracle_channel.ask(action.text or "", qc)
        if kind == "message_user":
            return Observation(text="ack")
        return Observation(text="ack")
