"""The episode runner: owns the loop, the clock, termination, and the trace.

``run_episode`` implements the single-threaded, deterministic interaction loop of
``docs/protocol.md`` §1.2. One agent action per step gives the trace a total order
and makes budgets debit deterministically. The runner:

* emits ``episode_start`` (seq 0) with everything outcome-affecting;
* drives the agent one :class:`~usabench.agent.base.AgentAction` at a time;
* routes file/exec actions to the sandbox and *derives* ``file_edit`` events by
  diffing workspace snapshots (never trusting the agent);
* routes ``ask_user`` through the :class:`~usabench.harness.interaction_bus.InteractionBus`
  to the oracle;
* on ``declare_done`` triggers a ``verification_run`` + oracle review;
* enforces the five budget ceilings and forces a final verification on budget/give-up;
* emits ``episode_end`` as the last line with the scorer-derivable severity cache.

Every metric is a pure offline function of the resulting ``trace.jsonl`` plus the
frozen gold (``DESIGN.md`` invariant 4); the runner's only job is to produce a
complete, hash-chained, replayable trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from usabench.agent.base import ActionKind, Agent, AgentAction, Observation
from usabench.core.enums import (
    Actor,
    QueryClass,
    RunStatus,
    Severity,
    TerminatedReason,
    Verdict,
)
from usabench.core.errors import BudgetExceeded, SandboxError, UsabenchError
from usabench.core.ids import sha256_hex
from usabench.core.schema import (
    AgentMessage,
    BudgetDebit,
    CodeRun,
    EpisodeEnd,
    EpisodeStart,
    FileEdit,
    HandoffRequest,
    MessageToUser,
    RunManifest,
    RunResult,
    Task,
    Usage,
    VerificationRun,
)
from usabench.harness.budget import BudgetDebitRecord, BudgetMeter
from usabench.harness.interaction_bus import InteractionBus
from usabench.harness.sandbox import SandboxBackend
from usabench.logging_setup import bind_run, get_logger

if TYPE_CHECKING:
    from usabench.llm.client import ToolSpec

__all__ = ["VerifierLike", "AgentUsageLike", "run_episode"]

_log = get_logger(__name__)

#: Hard cap on loop iterations as a safety net independent of budgets.
_LOOP_SAFETY_CAP = 100_000


@runtime_checkable
class VerifierLike(Protocol):
    """Structural protocol for the deterministic acceptance verifier.

    The real verifier lives outside the harness subtree (``usabench.eval``); the
    runner only needs the ``verify`` entry point. Defined structurally so the
    harness has no hard import dependency on the eval package.
    """

    def verify(
        self,
        task: Task,
        sandbox: SandboxBackend,
        *,
        trigger: str,
        entrypoint: str | None,
    ) -> VerificationRun:
        """Run acceptance checks against the workspace; return a VerificationRun."""
        ...


@runtime_checkable
class AgentUsageLike(Protocol):
    """Optional structural protocol: an agent that reports per-call token usage."""

    def usage(self) -> Usage:
        """Return cumulative AUT usage since the last call (delta accounting)."""
        ...


def _agent_usage_delta(agent: Agent) -> Usage:
    """Return the agent's usage delta if it implements :class:`AgentUsageLike`."""
    fn = getattr(agent, "usage", None)
    if callable(fn):
        try:
            result = fn()
            if isinstance(result, Usage):
                return result
        except Exception:  # pragma: no cover - defensive; usage is best-effort
            _log.warning("runner.agent_usage_failed")
    return Usage()


def _stdout_hash(text: str) -> str | None:
    """Return the sha256 of ``text`` for the trace, or ``None`` if empty."""
    return sha256_hex(text) if text else None


def _trunc(text: str, limit: int = 4000) -> str:
    """Truncate ``text`` to ``limit`` chars for the trace's inline preview."""
    return text if len(text) <= limit else text[:limit]


def run_episode(
    task: Task,
    agent: Agent,
    *,
    sandbox: SandboxBackend,
    bus: InteractionBus,
    budget: BudgetMeter,
    seed: int,
    manifest: RunManifest,
    verifier: VerifierLike | None = None,
    episode_start_extra: dict[str, dict[str, Any]] | None = None,
    max_oracle_per_turn: int = 1,
) -> RunResult:
    """Run one full (task, agent, oracle) episode and write its trace.

    Args:
        task: The frozen benchmark task (gold included; only ``agent_view`` is
            handed to the agent).
        agent: The agent-under-test (driven via ``reset`` + ``step``).
        sandbox: A set-up :class:`SandboxBackend` for the agent's workspace.
        bus: An *open* :class:`InteractionBus` writing this run's ``trace.jsonl``
            (with its oracle configured).
        budget: The :class:`BudgetMeter` enforcing the five ceilings.
        seed: The replica seed (recorded in ``episode_start``).
        manifest: The pre-built :class:`RunManifest` for this run.
        verifier: Optional verifier; if ``None``, verification runs are skipped and
            ``declare_done`` accepts on the oracle review alone.
        episode_start_extra: Optional extra fields merged into ``episode_start``
            descriptors (agent/oracle/sandbox dicts).
        max_oracle_per_turn: Cap on oracle queries serviced within a single turn
            (defaults to 1, matching one-action-per-step).

    Returns:
        A :class:`RunResult` summarizing the run (a view over the trace).
    """
    bind_run(manifest.run_id, task_id=task.id, seed=seed)
    extra = episode_start_extra or {}

    # --- seq 0: episode_start ---------------------------------------------- #
    start_payload = EpisodeStart(
        task_id=task.id,
        task_version=task.schema_version,
        hidden_spec_sha256=_hidden_spec_hash(task),
        agent={**manifest.agent, **dict(extra.get("agent", {}))},
        oracle={**manifest.oracle, **dict(extra.get("oracle", {}))},
        budgets=manifest.budgets or budget.limits.as_payload(),
        seed=seed,
        sandbox={
            "network": str(sandbox.network),
            **manifest.sandbox,
            **dict(extra.get("sandbox", {})),
        },
        harness_version=manifest.package_version,
        git_commit=manifest.git_sha,
    )
    bus.emit(Actor.HARNESS, start_payload, t_turn=None, budgets_after=budget.snapshot())

    # --- agent reset -------------------------------------------------------- #
    tool_specs = _default_tool_specs()
    try:
        agent.reset(task.agent_view(), tool_specs)
    except Exception as exc:  # pragma: no cover - adapter failure is rare
        return _finish_error(task, bus, budget, manifest, seed, f"agent.reset failed: {exc}")

    observation = Observation(text=task.user_goal)
    turn = 0
    terminated_reason: TerminatedReason | None = None
    accepted = False
    final_score: float | None = None
    status = RunStatus.RUNNING
    invalid = False
    invalid_reason: str | None = None

    try:
        for _ in range(_LOOP_SAFETY_CAP):
            # Budget gate BEFORE asking the agent for another action.
            if budget.is_exhausted():
                terminated_reason = budget.exhausted_reason or TerminatedReason.BUDGET_TURNS
                break

            turn += 1
            try:
                action = agent.step(observation)
            except Exception as exc:  # pragma: no cover - adapter failure
                invalid = True
                invalid_reason = f"agent.step raised: {exc}"
                terminated_reason = TerminatedReason.ERROR
                break

            # Debit the turn (may raise BudgetExceeded).
            _debit(bus, budget, budget.debit_turn(reason=action.kind), turn)

            observation, ctrl = _dispatch_action(
                action=action,
                task=task,
                agent=agent,
                sandbox=sandbox,
                bus=bus,
                budget=budget,
                turn=turn,
                verifier=verifier,
                max_oracle_per_turn=max_oracle_per_turn,
            )
            if ctrl.terminated_reason is not None:
                terminated_reason = ctrl.terminated_reason
                accepted = ctrl.accepted
                final_score = ctrl.final_score
                break
        else:  # pragma: no cover - safety cap is effectively unreachable
            terminated_reason = TerminatedReason.ERROR
            invalid = True
            invalid_reason = "loop safety cap reached"
    except BudgetExceeded as exc:
        terminated_reason = budget.exhausted_reason or _reason_for_budget(exc.kind)
    except SandboxError as exc:
        invalid = True
        invalid_reason = f"sandbox error: {exc}"
        terminated_reason = TerminatedReason.ERROR
    except UsabenchError as exc:  # pragma: no cover - defensive catch-all
        invalid = True
        invalid_reason = f"harness error: {exc}"
        terminated_reason = TerminatedReason.ERROR

    if terminated_reason is None:
        terminated_reason = TerminatedReason.ERROR
        invalid = True
        invalid_reason = invalid_reason or "loop exited without a termination reason"

    # --- forced final verification on non-accept terminations -------------- #
    if not accepted and not invalid and verifier is not None and terminated_reason in {
        TerminatedReason.BUDGET_TURNS,
        TerminatedReason.BUDGET_WALL,
        TerminatedReason.BUDGET_TOKENS,
        TerminatedReason.BUDGET_COST,
        TerminatedReason.BUDGET_ORACLE,
        TerminatedReason.GIVE_UP,
    }:
        vrun = _safe_verify(verifier, task, sandbox, trigger="forced_final", entrypoint=None)
        if vrun is not None:
            bus.emit(Actor.ENV, vrun, t_turn=turn, budgets_after=budget.snapshot())
            final_score = vrun.rubric_score

    status = _status_for(terminated_reason, accepted, invalid)

    # --- episode_end (last line) ------------------------------------------- #
    sev_counts = {str(k): v for k, v in bus.severity_counts.items() if v}
    end_payload = EpisodeEnd(
        terminated_reason=terminated_reason,
        accepted=accepted,
        final_weighted_score=final_score,
        interventions_by_severity=sev_counts,
        invalid=invalid,
        invalid_reason=invalid_reason,
    )
    bus.emit(Actor.HARNESS, end_payload, t_turn=None, budgets_after=budget.snapshot())

    return RunResult(
        run_id=manifest.run_id,
        task_id=task.id,
        seed=seed,
        status=status,
        terminated_reason=terminated_reason,
        accepted=accepted,
        n_events=bus.seq,
        wall_clock_s=budget.snapshot().wall_s,
        agent_usage=Usage(
            prompt_tokens=0,
            completion_tokens=int(budget.used("token")),
            cost_usd=round(budget.used("cost"), 8),
        ),
        oracle_usage=Usage(),
        interventions_by_severity=sev_counts,
        trace_path=str(bus.trace_path),
        manifest=manifest,
        invalid=invalid,
        invalid_reason=invalid_reason,
    )


# --------------------------------------------------------------------------- #
# Action dispatch                                                             #
# --------------------------------------------------------------------------- #


class _Control:
    """Loop-control flags returned by :func:`_dispatch_action`."""

    __slots__ = ("terminated_reason", "accepted", "final_score")

    def __init__(
        self,
        terminated_reason: TerminatedReason | None = None,
        accepted: bool = False,
        final_score: float | None = None,
    ) -> None:
        self.terminated_reason = terminated_reason
        self.accepted = accepted
        self.final_score = final_score


def _dispatch_action(
    *,
    action: AgentAction,
    task: Task,
    agent: Agent,
    sandbox: SandboxBackend,
    bus: InteractionBus,
    budget: BudgetMeter,
    turn: int,
    verifier: VerifierLike | None,
    max_oracle_per_turn: int,
) -> tuple[Observation, _Control]:
    """Execute one agent action, emit its trace events, return next observation."""
    kind = action.kind

    if kind == ActionKind.WRITE_FILE:
        return _do_write(action, sandbox, bus, budget, turn), _Control()

    if kind == ActionKind.READ_FILE:
        return _do_read(action, sandbox), _Control()

    if kind == ActionKind.RUN_CMD:
        return _do_run_cmd(action, task, sandbox, bus, budget, turn), _Control()

    if kind == ActionKind.ASK_USER:
        return _do_ask_user(action, bus, budget, turn), _Control()

    if kind == ActionKind.MESSAGE_USER:
        bus.emit(
            Actor.AGENT,
            MessageToUser(text=action.text or ""),
            t_turn=turn,
            budgets_after=budget.snapshot(),
        )
        _accrue_agent_usage(agent, bus, budget, turn)
        return Observation(text="ack"), _Control()

    if kind == ActionKind.DECLARE_DONE:
        return _do_declare_done(action, task, sandbox, bus, budget, turn, verifier)

    if kind == ActionKind.GIVE_UP:
        bus.emit(
            Actor.AGENT,
            HandoffRequest(reason=action.text or "agent gave up"),
            t_turn=turn,
            budgets_after=budget.snapshot(),
        )
        return Observation(text="acknowledged give_up"), _Control(
            terminated_reason=TerminatedReason.GIVE_UP
        )

    # Unknown kinds are prevented by AgentAction validation, but be defensive.
    return Observation(text=f"unknown action kind: {kind}"), _Control()  # pragma: no cover


def _do_write(
    action: AgentAction,
    sandbox: SandboxBackend,
    bus: InteractionBus,
    budget: BudgetMeter,
    turn: int,
) -> Observation:
    """Apply a write, diff the workspace, and emit derived ``file_edit`` events."""
    path = action.path or ""
    before = sandbox.snapshot()
    try:
        sandbox.write(path, action.content or "")
    except SandboxError as exc:
        return Observation(text=f"write denied: {exc}", exit_code=1)
    after = sandbox.snapshot()
    edits: list[FileEdit] = sandbox.diff(before, after)
    for edit in edits:
        bus.emit(Actor.ENV, edit, t_turn=turn, budgets_after=budget.snapshot())
    msg = f"wrote {path} ({len(edits)} file change(s) derived)"
    return Observation(text=msg, exit_code=0)


def _do_read(action: AgentAction, sandbox: SandboxBackend) -> Observation:
    """Read a file; reads have no side effect so no trace event is derived."""
    path = action.path or ""
    try:
        content = sandbox.read(path)
    except SandboxError as exc:
        return Observation(text=f"read failed: {exc}", exit_code=1)
    truncated = len(content) > 8000
    return Observation(
        text=content[:8000], exit_code=0, truncated=truncated, data={"path": path}
    )


def _do_run_cmd(
    action: AgentAction,
    task: Task,
    sandbox: SandboxBackend,
    bus: InteractionBus,
    budget: BudgetMeter,
    turn: int,
) -> Observation:
    """Execute a command, derive any ``file_edit`` it caused, emit a ``code_run``."""
    cmd = action.cmd or ""
    timeout = int(action.meta.get("timeout_s", 120)) if action.meta else 120
    before = sandbox.snapshot()
    result = sandbox.exec(cmd, timeout_s=timeout)
    after = sandbox.snapshot()
    for edit in sandbox.diff(before, after):
        bus.emit(Actor.ENV, edit, t_turn=turn, budgets_after=budget.snapshot())

    is_test = any(tok in cmd for tok in ("pytest", "unittest", "test", "tox"))
    code_run = CodeRun(
        call_id=str(action.meta.get("call_id")) if action.meta.get("call_id") else None,
        cmd=cmd,
        exit_code=result.exit_code,
        stdout_sha256=_stdout_hash(result.stdout),
        stdout_trunc=_trunc(result.stdout),
        stderr_trunc=_trunc(result.stderr),
        wall_ms=result.wall_ms,
        is_test=is_test,
        self_test_passed=(result.exit_code == 0) if is_test else None,
    )
    bus.emit(Actor.ENV, code_run, t_turn=turn, budgets_after=budget.snapshot())
    return Observation(
        text=_trunc(result.stdout or result.stderr),
        exit_code=result.exit_code,
        truncated=result.truncated,
    )


def _do_ask_user(
    action: AgentAction,
    bus: InteractionBus,
    budget: BudgetMeter,
    turn: int,
) -> Observation:
    """Route an ``ask_user`` action through the oracle channel + debit a query."""
    _debit(bus, budget, budget.debit_oracle_query(reason="ask_user"), turn)
    qclass = action.query_class or QueryClass.CLARIFICATION
    if isinstance(qclass, str):
        qclass = QueryClass(qclass)
    _q_env, r_env = bus.ask_oracle(
        action.text or "",
        qclass,
        t_turn=turn,
        context_refs=list(action.meta.get("context_refs", [])) if action.meta else [],
        agent_blocked=bool(action.meta.get("agent_blocked", False)) if action.meta else False,
        budgets_after=budget.snapshot(),
    )
    oracle_text = getattr(r_env.payload, "text", "")
    return Observation(text=oracle_text, oracle_text=oracle_text, exit_code=0)


def _do_declare_done(
    action: AgentAction,
    task: Task,
    sandbox: SandboxBackend,
    bus: InteractionBus,
    budget: BudgetMeter,
    turn: int,
    verifier: VerifierLike | None,
) -> tuple[Observation, _Control]:
    """Handle a submission: verification_run + oracle review; accept or continue."""
    entrypoint = action.entrypoint
    if verifier is None:
        # No verifier wired: the oracle review alone decides acceptance.
        bus.oracle_review(
            Verdict.ACCEPT,
            severity=Severity.NONE,
            text="accepted (no verifier configured)",
            t_turn=turn,
            budgets_after=budget.snapshot(),
        )
        return (
            Observation(text="submission accepted", exit_code=0),
            _Control(terminated_reason=TerminatedReason.ACCEPT, accepted=True, final_score=None),
        )

    vrun = _safe_verify(verifier, task, sandbox, trigger="submit", entrypoint=entrypoint)
    if vrun is None:
        # Verifier crashed -> treat the submission as a non-accepting reject, loop on.
        bus.oracle_review(
            Verdict.REJECT,
            severity=Severity.TRIVIAL_CLARIFICATION,
            text="verification failed to run",
            t_turn=turn,
            budgets_after=budget.snapshot(),
        )
        return Observation(text="verification error; please revise", exit_code=1), _Control()

    bus.emit(Actor.ENV, vrun, t_turn=turn, budgets_after=budget.snapshot())

    if vrun.all_must_pass:
        bus.oracle_review(
            Verdict.ACCEPT,
            severity=Severity.NONE,
            text="behavior matches intent; accepted",
            cited_criteria=[c.id for c in vrun.must_have],
            t_turn=turn,
            budgets_after=budget.snapshot(),
        )
        return (
            Observation(text="submission accepted", exit_code=0),
            _Control(
                terminated_reason=TerminatedReason.ACCEPT,
                accepted=True,
                final_score=vrun.rubric_score,
            ),
        )

    # Reject with named feedback: an intervention that does not terminate the run.
    failed = [c.id for c in vrun.must_have if c.passed is False]
    bus.oracle_review(
        Verdict.REJECT,
        severity=Severity.TRIVIAL_CLARIFICATION,
        text="some must-have criteria still fail; please revise",
        cited_criteria=failed,
        t_turn=turn,
        budgets_after=budget.snapshot(),
    )
    return (
        Observation(text=f"rejected: criteria failing: {', '.join(failed) or 'unknown'}", exit_code=1),
        _Control(),
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _accrue_agent_usage(agent: Agent, bus: InteractionBus, budget: BudgetMeter, turn: int) -> None:
    """Pull an agent usage delta (if any) and debit token/cost budgets."""
    usage = _agent_usage_delta(agent)
    if usage.total_tokens or usage.cost_usd:
        records = budget.debit_usage(usage.total_tokens, usage.cost_usd, reason="agent_message")
        for rec in records:
            _emit_debit(bus, rec, turn, budget)
        bus.emit(
            Actor.AGENT,
            AgentMessage(text="", tokens=usage),
            t_turn=turn,
            budgets_after=budget.snapshot(),
        )


def _debit(bus: InteractionBus, budget: BudgetMeter, record: BudgetDebitRecord, turn: int) -> None:
    """Emit a ``budget_debit`` event for an already-applied debit record."""
    _emit_debit(bus, record, turn, budget)


def _emit_debit(bus: InteractionBus, record: BudgetDebitRecord, turn: int, budget: BudgetMeter) -> None:
    """Translate a :class:`BudgetDebitRecord` into a ``budget_debit`` trace event."""
    bus.emit(
        Actor.HARNESS,
        BudgetDebit(kind=record.kind, amount=record.amount, reason=record.reason),
        t_turn=turn,
        budgets_after=budget.snapshot(),
    )


def _safe_verify(
    verifier: VerifierLike,
    task: Task,
    sandbox: SandboxBackend,
    *,
    trigger: str,
    entrypoint: str | None,
) -> VerificationRun | None:
    """Call the verifier, swallowing its failures into ``None`` (never crash the run)."""
    try:
        return verifier.verify(task, sandbox, trigger=trigger, entrypoint=entrypoint)
    except Exception as exc:  # pragma: no cover - verifier is external
        _log.warning("runner.verify_failed", trigger=trigger, error=str(exc))
        return None


def _hidden_spec_hash(task: Task) -> str:
    """Return a stable sha256 of the task's frozen hidden spec (provenance)."""
    from usabench.core.ids import canonical_json

    return sha256_hex(canonical_json(task.hidden.model_dump(mode="json")))


def _reason_for_budget(kind: str) -> TerminatedReason:
    """Map a budget kind string onto its :class:`TerminatedReason`."""
    return {
        "turn": TerminatedReason.BUDGET_TURNS,
        "wall": TerminatedReason.BUDGET_WALL,
        "token": TerminatedReason.BUDGET_TOKENS,
        "cost": TerminatedReason.BUDGET_COST,
        "oracle_query": TerminatedReason.BUDGET_ORACLE,
    }.get(kind, TerminatedReason.BUDGET_TURNS)


def _status_for(reason: TerminatedReason, accepted: bool, invalid: bool) -> RunStatus:
    """Derive the run status from the termination reason and flags."""
    if invalid:
        return RunStatus.INVALID
    if accepted or reason == TerminatedReason.ACCEPT:
        return RunStatus.ACCEPTED
    if reason == TerminatedReason.GIVE_UP:
        return RunStatus.AGENT_GAVE_UP
    if reason == TerminatedReason.ORACLE_TAKEOVER:
        return RunStatus.ORACLE_TAKEOVER
    if reason.is_budget:
        return RunStatus.BUDGET_EXHAUSTED
    if reason == TerminatedReason.ERROR:
        return RunStatus.ERROR
    return RunStatus.ERROR


def _finish_error(
    task: Task,
    bus: InteractionBus,
    budget: BudgetMeter,
    manifest: RunManifest,
    seed: int,
    reason: str,
) -> RunResult:
    """Emit a terminal ``episode_end`` for an early/invalid failure and return a result."""
    end_payload = EpisodeEnd(
        terminated_reason=TerminatedReason.ERROR,
        accepted=False,
        invalid=True,
        invalid_reason=reason,
    )
    bus.emit(Actor.HARNESS, end_payload, t_turn=None, budgets_after=budget.snapshot())
    return RunResult(
        run_id=manifest.run_id,
        task_id=task.id,
        seed=seed,
        status=RunStatus.INVALID,
        terminated_reason=TerminatedReason.ERROR,
        accepted=False,
        n_events=bus.seq,
        trace_path=str(bus.trace_path),
        manifest=manifest,
        invalid=True,
        invalid_reason=reason,
    )


def _default_tool_specs() -> list[ToolSpec]:
    """Return the default tool specs handed to the agent on reset.

    These mirror the sandbox surface (``write_file``/``read_file``/``run_cmd``) plus
    the oracle channel (``ask_user``). Kept minimal and JSON-schema-typed so the
    agent adapter can present them to any provider.
    """
    from usabench.llm.client import ToolSpec

    return [
        ToolSpec(
            name="write_file",
            description="Create or overwrite a file in the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
        ToolSpec(
            name="read_file",
            description="Read a file from the workspace.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        ToolSpec(
            name="run_cmd",
            description="Run a shell command in the sandboxed workspace.",
            parameters={
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        ),
        ToolSpec(
            name="ask_user",
            description="Ask the user (oracle) a clarifying question.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
    ]
