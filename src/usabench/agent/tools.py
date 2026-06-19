"""Sandboxed agent tools and the user/oracle channel interface.

The reference scaffold acts on the world through a small, closed toolset:

* ``write_file(path, content)`` -- create/overwrite a file in the sandbox.
* ``read_file(path)`` -- read a file back from the sandbox.
* ``run_cmd(cmd, timeout_s=...)`` -- execute a shell command **inside the
  provided sandbox handle** (never on the host).
* ``ask_user(text, query_class)`` -- pose a question to the simulated user; this
  is the single mediated path to the oracle.
* ``message_user(text)`` -- emit a status / think-aloud update to the user.

Two channels are *injected* by the harness, never imported here:

* a :class:`SandboxHandle` -- the file/exec surface (``docs/protocol.md`` §5.2,
  ``SandboxBackend`` ABC). ``run_cmd`` runs strictly inside it.
* a :class:`UserChannel` -- the harness's ``InteractionBus`` gateway to the
  oracle. ``ask_user`` and ``message_user`` route through this callback so every
  such message is typed, timestamped, and logged by the harness
  (``docs/infra.md`` ``interaction_bus.py``; ``configs/agents/scaffold_default.yaml``
  ``ask_user_is_interaction: true``).

To avoid an import cycle (``harness`` imports ``agent``, so ``agent`` must NOT
import ``harness``) both injected surfaces are *structural* ``Protocol`` types:
the harness passes any object that quacks correctly. The scaffold depends on the
interface, not the implementation.

Path safety: ``write_file``/``read_file`` reject absolute paths and ``..``
traversal so the agent cannot escape the ``/work`` workspace
(``docs/protocol.md`` §5.3). The sandbox enforces the real boundary; this is a
cheap first line of defence kept on the agent side.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from usabench.agent.base import Observation, ToolHandle
from usabench.core.enums import QueryClass
from usabench.core.errors import SandboxError
from usabench.llm.client import ToolSpec
from usabench.logging_setup import get_logger

__all__ = [
    "SandboxHandle",
    "UserChannel",
    "ExecOutcome",
    "SandboxToolset",
    "DEFAULT_TOOL_NAMES",
    "build_tool_specs",
    "is_safe_relpath",
    "MAX_OBSERVATION_CHARS",
]

logger = get_logger(__name__)

#: Cap on rendered observation text so a single huge stdout cannot blow the
#: context window. The harness content-addresses the full blob; the scaffold only
#: ever sees a truncated view.
MAX_OBSERVATION_CHARS = 8_000

#: The default tool vocabulary (mirrors ``configs/agents/scaffold_default.yaml``).
DEFAULT_TOOL_NAMES: tuple[str, ...] = (
    "write_file",
    "read_file",
    "run_cmd",
    "ask_user",
    "message_user",
)


# --------------------------------------------------------------------------- #
# Injected interfaces (structural; the harness supplies the concrete objects)  #
# --------------------------------------------------------------------------- #


@runtime_checkable
class SandboxHandle(Protocol):
    """The sandbox file/exec surface the toolset acts through.

    Structurally compatible with the harness ``SandboxBackend`` (``docs/protocol.md``
    §5.2). The toolset never touches the host filesystem or runs subprocesses
    directly -- every effect goes through this handle so isolation, the network
    policy, and resource limits are enforced in one place.
    """

    def exec(self, cmd: str, *, timeout_s: int = 120) -> ExecOutcome:
        """Run ``cmd`` in the sandbox under ``timeout_s`` seconds; return outcome."""
        ...

    def read_file(self, path: str) -> str:
        """Read ``path`` (relative to the workspace) and return its text."""
        ...

    def write_file(self, path: str, content: str) -> None:
        """Write ``content`` to ``path`` (relative to the workspace)."""
        ...


@runtime_checkable
class UserChannel(Protocol):
    """The mediated channel to the simulated user / oracle (the InteractionBus).

    Structurally compatible with :class:`usabench.agent.base.OracleChannel`. The
    harness injects an object exposing ``ask`` (a blocking round-trip that returns
    the oracle reply) and ``notify`` (a fire-and-forget status update). Keeping
    this as a Protocol is what prevents the ``agent -> harness`` import cycle.
    """

    def ask(self, text: str, query_class: QueryClass) -> Observation:
        """Send a question to the oracle and return its mediated reply."""
        ...

    def notify(self, text: str) -> Observation:
        """Emit a one-way status update to the user (no oracle reply expected)."""
        ...


class ExecOutcome(Protocol):
    """Minimal structural shape of a sandbox exec result.

    Both the reference sandbox and a Docker/Apptainer backend return an object
    with at least these attributes (``docs/protocol.md`` ``tool_result`` payload).
    """

    exit_code: int
    stdout: str
    stderr: str


# --------------------------------------------------------------------------- #
# Path safety                                                                  #
# --------------------------------------------------------------------------- #


def is_safe_relpath(path: str) -> bool:
    """Return True iff ``path`` is a workspace-relative path with no escape.

    Rejects absolute paths and any component that is ``..`` so the agent cannot
    read/write outside the sandbox workspace. The sandbox enforces the real
    boundary; this is a fast pre-check that keeps obvious traversal out of the
    trace (``docs/protocol.md`` §5.3).

    Args:
        path: A candidate file path supplied by the agent.

    Returns:
        True if the path is safe to pass to the sandbox, False otherwise.
    """
    if not path or path.strip() == "":
        return False
    norm = path.replace("\\", "/")
    if norm.startswith("/") or (len(norm) > 1 and norm[1] == ":"):
        return False  # absolute (posix or windows-drive)
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    return ".." not in parts


def _truncate(text: str, limit: int = MAX_OBSERVATION_CHARS) -> tuple[str, bool]:
    """Truncate ``text`` to ``limit`` chars, returning ``(text, was_truncated)``."""
    if len(text) <= limit:
        return text, False
    head = text[: limit - 64]
    return f"{head}\n... [truncated {len(text) - len(head)} chars]", True


# --------------------------------------------------------------------------- #
# Tool specs (provider-agnostic JSON-schema, fed to the LLM as native tools)    #
# --------------------------------------------------------------------------- #


def build_tool_specs(tool_names: list[str] | None = None) -> list[ToolSpec]:
    """Build the provider-agnostic :class:`ToolSpec` list for the scaffold.

    These are the same tools the LLM is offered (when native tool-calling is on)
    and the same names the ReAct text parser recognizes, so both code paths share
    one vocabulary.

    Args:
        tool_names: Subset/order of :data:`DEFAULT_TOOL_NAMES` to expose; ``None``
            means all default tools.

    Returns:
        A list of :class:`ToolSpec` in the requested order.
    """
    selected = list(tool_names) if tool_names is not None else list(DEFAULT_TOOL_NAMES)
    specs: list[ToolSpec] = []
    for name in selected:
        spec = _TOOL_SPEC_TABLE.get(name)
        if spec is None:
            raise SandboxError(f"unknown tool name: {name!r}")
        specs.append(spec.model_copy(deep=True))
    return specs


_TOOL_SPEC_TABLE: dict[str, ToolSpec] = {
    "write_file": ToolSpec(
        name="write_file",
        description="Create or overwrite a file in the workspace with the given text content.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "content": {"type": "string", "description": "Full file content to write."},
            },
            "required": ["path", "content"],
        },
    ),
    "read_file": ToolSpec(
        name="read_file",
        description="Read a file from the workspace and return its text content.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path."}
            },
            "required": ["path"],
        },
    ),
    "run_cmd": ToolSpec(
        name="run_cmd",
        description=(
            "Run a shell command inside the sandboxed workspace and return its "
            "exit code, stdout, and stderr. Network is denied unless allowlisted."
        ),
        parameters={
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to execute."},
                "timeout_s": {
                    "type": "integer",
                    "description": "Per-command timeout in seconds.",
                    "default": 120,
                },
            },
            "required": ["cmd"],
        },
    ),
    "ask_user": ToolSpec(
        name="ask_user",
        description=(
            "Ask the human user a question (clarification, confirmation, or a hint "
            "request). Use this to resolve ambiguity instead of guessing. Every "
            "ask_user is a counted interaction."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The question to ask the user."},
                "query_class": {
                    "type": "string",
                    "enum": [c.value for c in QueryClass],
                    "description": "Kind of question being asked.",
                    "default": QueryClass.CLARIFICATION.value,
                },
            },
            "required": ["text"],
        },
    ),
    "message_user": ToolSpec(
        name="message_user",
        description="Send a brief status update or plan to the user. No reply is expected.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Status update text."}
            },
            "required": ["text"],
        },
    ),
}


# --------------------------------------------------------------------------- #
# The toolset: dispatches one tool call to the injected sandbox / user channel  #
# --------------------------------------------------------------------------- #


class SandboxToolset(ToolHandle):
    """Concrete tool surface binding the scaffold to an injected sandbox + user.

    Implements :class:`usabench.agent.base.ToolHandle` (``write_file`` /
    ``read_file`` / ``run_cmd``) and additionally routes ``ask_user`` /
    ``message_user`` through the injected :class:`UserChannel`. Every method
    returns a rendered :class:`Observation` (truncated) that the scaffold feeds
    straight back into the model.

    The harness constructs this with the real sandbox and InteractionBus; tests
    pass fakes. Because both dependencies are :class:`Protocol` types, this module
    never imports the harness (no cycle).
    """

    def __init__(
        self,
        sandbox: SandboxHandle,
        user_channel: UserChannel | None = None,
        *,
        default_timeout_s: int = 120,
        max_observation_chars: int = MAX_OBSERVATION_CHARS,
    ) -> None:
        """Bind the toolset to its sandbox and (optional) user channel.

        Args:
            sandbox: The file/exec surface (the harness ``SandboxBackend``).
            user_channel: The mediated oracle gateway. If ``None``, ``ask_user``/
                ``message_user`` raise (configurations without an oracle).
            default_timeout_s: Default per-command timeout for ``run_cmd``.
            max_observation_chars: Truncation cap for rendered observations.
        """
        self._sandbox = sandbox
        self._user = user_channel
        self._default_timeout_s = default_timeout_s
        self._max_chars = max_observation_chars

    # --- ToolHandle: sandbox file/exec surface ---------------------------- #

    def write_file(self, path: str, content: str) -> Observation:
        """Write ``content`` to ``path`` inside the sandbox; return an ack."""
        if not is_safe_relpath(path):
            return Observation(text=f"ERROR: unsafe or non-relative path: {path!r}", exit_code=1)
        try:
            self._sandbox.write_file(path, content)
        except Exception as exc:  # noqa: BLE001 - normalize backend failures
            logger.warning("write_file_failed", path=path, error=str(exc))
            return Observation(text=f"ERROR writing {path}: {exc}", exit_code=1)
        n_lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        return Observation(
            text=f"Wrote {len(content)} bytes ({n_lines} lines) to {path}.",
            exit_code=0,
            data={"path": path, "bytes": len(content)},
        )

    def read_file(self, path: str) -> Observation:
        """Read ``path`` from the sandbox; return its (possibly truncated) text."""
        if not is_safe_relpath(path):
            return Observation(text=f"ERROR: unsafe or non-relative path: {path!r}", exit_code=1)
        try:
            content = self._sandbox.read_file(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("read_file_failed", path=path, error=str(exc))
            return Observation(text=f"ERROR reading {path}: {exc}", exit_code=1)
        rendered, truncated = _truncate(content, self._max_chars)
        return Observation(
            text=rendered,
            exit_code=0,
            truncated=truncated,
            data={"path": path, "bytes": len(content)},
        )

    def run_cmd(self, cmd: str, *, timeout_s: int | None = None) -> Observation:
        """Execute ``cmd`` inside the sandbox under a timeout; return the result.

        Args:
            cmd: The shell command to run.
            timeout_s: Per-command timeout; falls back to the toolset default.

        Returns:
            An :class:`Observation` carrying the rendered stdout/stderr and exit
            code. Backend failures are normalized into a non-zero exit code rather
            than raised, so the scaffold can observe and recover from them.
        """
        to = self._default_timeout_s if timeout_s is None else timeout_s
        try:
            outcome = self._sandbox.exec(cmd, timeout_s=to)
        except Exception as exc:  # noqa: BLE001
            logger.warning("run_cmd_failed", cmd=cmd, error=str(exc))
            return Observation(text=f"ERROR running command: {exc}", exit_code=1)
        stdout = getattr(outcome, "stdout", "") or ""
        stderr = getattr(outcome, "stderr", "") or ""
        exit_code = int(getattr(outcome, "exit_code", 0))
        body = stdout
        if stderr:
            body = f"{stdout}\n[stderr]\n{stderr}" if stdout else f"[stderr]\n{stderr}"
        rendered, truncated = _truncate(body, self._max_chars)
        text = f"$ {cmd}\n[exit={exit_code}]\n{rendered}"
        return Observation(
            text=text,
            exit_code=exit_code,
            truncated=truncated,
            data={"cmd": cmd, "exit_code": exit_code},
        )

    # --- User channel: the single mediated path to the oracle ------------- #

    def ask_user(
        self, text: str, query_class: QueryClass = QueryClass.CLARIFICATION
    ) -> Observation:
        """Pose ``text`` to the oracle via the injected channel; return its reply.

        Args:
            text: The question to ask.
            query_class: The kind of question (clarification, hint_request, ...).

        Returns:
            An :class:`Observation` whose ``oracle_text`` holds the mediated reply.

        Raises:
            SandboxError: If no user channel was injected.
        """
        if self._user is None:
            raise SandboxError("ask_user called but no UserChannel was injected")
        obs = self._user.ask(text, query_class)
        # Normalize: surface the oracle reply as the primary observation text.
        reply = obs.oracle_text if obs.oracle_text is not None else obs.text
        rendered, truncated = _truncate(reply or "", self._max_chars)
        return Observation(
            text=rendered,
            oracle_text=reply,
            truncated=truncated or obs.truncated,
            data={**obs.data, "query_class": str(query_class)},
        )

    def message_user(self, text: str) -> Observation:
        """Send a one-way status update to the user via the injected channel.

        Args:
            text: The status update.

        Returns:
            An :class:`Observation` acknowledging the update.

        Raises:
            SandboxError: If no user channel was injected.
        """
        if self._user is None:
            raise SandboxError("message_user called but no UserChannel was injected")
        obs = self._user.notify(text)
        return Observation(text=obs.text or "ack", data=obs.data)

    # --- Generic dispatch (used by the scaffold to run one parsed action) - #

    def dispatch(self, name: str, args: dict[str, Any]) -> Observation:
        """Dispatch a single named tool call to the right method.

        Used by the scaffold to execute exactly one parsed tool call per step
        (the one-action-per-step contract). Unknown tools yield an error
        observation rather than raising, so the model can self-correct.

        Args:
            name: Tool name (one of :data:`DEFAULT_TOOL_NAMES`).
            args: Parsed argument dict for the tool.

        Returns:
            The rendered :class:`Observation` of running the tool.
        """
        if name == "write_file":
            return self.write_file(str(args.get("path", "")), str(args.get("content", "")))
        if name == "read_file":
            return self.read_file(str(args.get("path", "")))
        if name == "run_cmd":
            timeout = args.get("timeout_s")
            return self.run_cmd(
                str(args.get("cmd", "")),
                timeout_s=int(timeout) if timeout is not None else None,
            )
        if name == "ask_user":
            qc_raw = args.get("query_class", QueryClass.CLARIFICATION.value)
            try:
                qc = QueryClass(qc_raw)
            except ValueError:
                qc = QueryClass.CLARIFICATION
            return self.ask_user(str(args.get("text", "")), qc)
        if name == "message_user":
            return self.message_user(str(args.get("text", "")))
        return Observation(text=f"ERROR: unknown tool {name!r}", exit_code=1)
