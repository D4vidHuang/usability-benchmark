"""The Agent protocol and the typed :class:`AgentAction` it emits.

The agent-under-test is a black box behind a thin adapter. The harness drives it
one action at a time (``step``) or pulls an iterator of actions (``run``), and the
agent emits exactly **one** :class:`AgentAction` per step -- the one-action-per-step
contract that gives the trace a total order and makes budgets debit deterministically
(``docs/protocol.md`` §1.2).

An :class:`AgentAction` is a tagged union over a small closed set of action kinds:
``write_file`` | ``read_file`` | ``run_cmd`` | ``ask_user`` | ``message_user`` |
``declare_done`` | ``give_up``. Each carries a typed payload.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from usabench.core.enums import QueryClass
from usabench.core.schema import AgentTaskView

__all__ = [
    "ActionKind",
    "AgentAction",
    "Observation",
    "ToolHandle",
    "OracleChannel",
    "Agent",
]


class ActionKind:
    """The closed set of action-kind string constants (not an Enum to keep payload
    typing simple; values mirror the agent-tool vocabulary)."""

    WRITE_FILE = "write_file"
    READ_FILE = "read_file"
    RUN_CMD = "run_cmd"
    ASK_USER = "ask_user"
    MESSAGE_USER = "message_user"
    DECLARE_DONE = "declare_done"
    GIVE_UP = "give_up"

    ALL: tuple[str, ...] = (
        WRITE_FILE,
        READ_FILE,
        RUN_CMD,
        ASK_USER,
        MESSAGE_USER,
        DECLARE_DONE,
        GIVE_UP,
    )


class AgentAction(BaseModel):
    """One action the agent emits per step (a tagged union over ``kind``).

    Only the fields relevant to ``kind`` are populated; the rest stay ``None``.
    Validation enforces that the required payload for each kind is present.

    Attributes:
        kind: One of :class:`ActionKind` values.
        path: Target path for ``write_file`` / ``read_file``.
        content: File content for ``write_file``.
        cmd: Command for ``run_cmd``.
        text: Free text for ``ask_user`` / ``message_user`` / ``declare_done`` /
            ``give_up``.
        query_class: Classification when ``kind == ask_user`` (clarification, ...).
        summary: Optional submission summary for ``declare_done``.
        entrypoint: Declared run entrypoint for ``declare_done``.
        meta: Adapter-specific passthrough (e.g. batch_id, raw tool-call id).
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., description="Action kind (see ActionKind).")
    path: str | None = Field(None, description="File path for write/read.")
    content: str | None = Field(None, description="File content for write_file.")
    cmd: str | None = Field(None, description="Shell command for run_cmd.")
    text: str | None = Field(None, description="Text for ask/message/declare/give_up.")
    query_class: QueryClass | None = Field(None, description="Class for ask_user queries.")
    summary: str | None = Field(None, description="Submission summary for declare_done.")
    entrypoint: str | None = Field(None, description="Run entrypoint for declare_done.")
    meta: dict[str, Any] = Field(default_factory=dict, description="Adapter passthrough.")

    def model_post_init(self, __context: Any) -> None:
        """Validate the per-kind payload contract after construction."""
        k = self.kind
        if k not in ActionKind.ALL:
            raise ValueError(f"unknown action kind: {k!r} (allowed: {ActionKind.ALL})")
        if k == ActionKind.WRITE_FILE and (self.path is None or self.content is None):
            raise ValueError("write_file requires 'path' and 'content'")
        if k == ActionKind.READ_FILE and self.path is None:
            raise ValueError("read_file requires 'path'")
        if k == ActionKind.RUN_CMD and not self.cmd:
            raise ValueError("run_cmd requires 'cmd'")
        if k == ActionKind.ASK_USER and not self.text:
            raise ValueError("ask_user requires 'text'")

    # --- ergonomic constructors -------------------------------------------- #

    @classmethod
    def write_file(cls, path: str, content: str, **meta: Any) -> AgentAction:
        """Construct a ``write_file`` action."""
        return cls(kind=ActionKind.WRITE_FILE, path=path, content=content, meta=meta)

    @classmethod
    def read_file(cls, path: str, **meta: Any) -> AgentAction:
        """Construct a ``read_file`` action."""
        return cls(kind=ActionKind.READ_FILE, path=path, meta=meta)

    @classmethod
    def run_cmd(cls, cmd: str, **meta: Any) -> AgentAction:
        """Construct a ``run_cmd`` action."""
        return cls(kind=ActionKind.RUN_CMD, cmd=cmd, meta=meta)

    @classmethod
    def ask_user(
        cls, text: str, query_class: QueryClass = QueryClass.CLARIFICATION, **meta: Any
    ) -> AgentAction:
        """Construct an ``ask_user`` (oracle query) action."""
        return cls(kind=ActionKind.ASK_USER, text=text, query_class=query_class, meta=meta)

    @classmethod
    def message_user(cls, text: str, **meta: Any) -> AgentAction:
        """Construct a ``message_user`` (status update) action."""
        return cls(kind=ActionKind.MESSAGE_USER, text=text, meta=meta)

    @classmethod
    def declare_done(
        cls, summary: str = "", entrypoint: str | None = None, **meta: Any
    ) -> AgentAction:
        """Construct a ``declare_done`` (submit) action."""
        return cls(kind=ActionKind.DECLARE_DONE, summary=summary, entrypoint=entrypoint, meta=meta)

    @classmethod
    def give_up(cls, reason: str = "", **meta: Any) -> AgentAction:
        """Construct a ``give_up`` action."""
        return cls(kind=ActionKind.GIVE_UP, text=reason, meta=meta)


class Observation(BaseModel):
    """What the harness hands back to the agent after each action.

    Attributes:
        text: A rendered, human/agent-readable result of the previous action.
        exit_code: Exit code if the previous action was a command.
        oracle_text: The oracle's reply text if the previous action asked it.
        truncated: True if the result was truncated for size.
        data: Structured passthrough (e.g. file contents, budget snapshot).
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field("", description="Rendered result of the last action.")
    exit_code: int | None = Field(None, description="Exit code for run_cmd results.")
    oracle_text: str | None = Field(None, description="Oracle reply for ask_user results.")
    truncated: bool = Field(False, description="Whether the result was truncated.")
    data: dict[str, Any] = Field(default_factory=dict, description="Structured passthrough.")


@runtime_checkable
class ToolHandle(Protocol):
    """A minimal sandbox tool surface the agent uses to act on the workspace."""

    def write_file(self, path: str, content: str) -> Observation:
        """Write ``content`` to ``path`` in the sandbox; return the result."""
        ...

    def read_file(self, path: str) -> Observation:
        """Read ``path`` from the sandbox; return its content in the observation."""
        ...

    def run_cmd(self, cmd: str, *, timeout_s: int = 120) -> Observation:
        """Execute ``cmd`` in the sandbox under a timeout; return the result."""
        ...


@runtime_checkable
class OracleChannel(Protocol):
    """The single mediated channel through which the agent reaches the oracle."""

    def ask(self, text: str, query_class: QueryClass) -> Observation:
        """Send a query to the oracle and return its (mediated) response."""
        ...


@runtime_checkable
class Agent(Protocol):
    """The agent-under-test interface.

    An adapter conforming to this protocol can be driven either step-by-step
    (``step``) or as a generator (``run``). Both must honor one-action-per-step:
    each call yields/returns exactly one :class:`AgentAction`.
    """

    def reset(self, task_view: AgentTaskView, tools: list[ToolSpec]) -> None:
        """Initialize the agent for a new episode with the agent-visible task.

        Args:
            task_view: The gold-stripped task projection.
            tools: The tool specs available this episode.
        """
        ...

    def step(self, observation: Observation) -> AgentAction:
        """Emit exactly one action given the latest observation.

        Args:
            observation: The rendered result of the previous action (or the
                initial prompt observation on the first step).

        Returns:
            The single :class:`AgentAction` to execute next.
        """
        ...

    def run(
        self,
        task_view: AgentTaskView,
        tools: ToolHandle,
        oracle_channel: OracleChannel,
    ) -> Iterator[AgentAction]:
        """Drive the full episode as an iterator of actions (convenience API).

        Args:
            task_view: The gold-stripped task projection.
            tools: The sandbox tool surface.
            oracle_channel: The mediated oracle channel.

        Yields:
            One :class:`AgentAction` at a time until the agent declares done or
            gives up.
        """
        ...


# Late import to avoid a hard import cost at module load; ToolSpec is only needed
# for the ``reset`` signature annotation above.
from usabench.llm.client import ToolSpec  # noqa: E402
