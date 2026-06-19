"""End-to-end harness smoke tests (no network, no API, no paid calls).

These tests drive the real :func:`usabench.harness.runner.run_episode` loop with a
scripted fake agent + fake oracle + a trivial verifier over the *working*
:class:`~usabench.harness.sandbox.LocalSubprocessSandbox`. They prove that the
smoke path runs end-to-end on a laptop and that the produced ``trace.jsonl`` is a
complete, totally-ordered, hash-chained artifact whose integrity invariants hold
(``docs/protocol.md`` §4.3) -- the harness's one and only runtime job.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from usabench.agent.base import AgentAction, Observation
from usabench.core.enums import (
    InteractionType,
    QueryClass,
    RunStatus,
    Severity,
    TerminatedReason,
    Verdict,
)
from usabench.core.ids import GENESIS_HASH, next_hash
from usabench.core.schema import (
    AcceptanceCriterion,
    CriterionResult,
    HiddenSpec,
    OracleResponse,
    Task,
    TaskEnv,
    VerificationRun,
    parse_event,
)
from usabench.harness import (
    BudgetLimits,
    BudgetMeter,
    InteractionBus,
    LocalSubprocessSandbox,
    build_manifest,
    is_run_complete,
    run_episode,
)
from usabench.harness.interaction_bus import OracleQueryContext

# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


class ScriptedAgent:
    """A deterministic agent that replays a fixed list of actions then declares done."""

    def __init__(self, actions: list[AgentAction]) -> None:
        self._actions = list(actions)
        self._i = 0

    def reset(self, task_view: object, tools: list[object]) -> None:  # noqa: D401
        self._i = 0

    def step(self, observation: Observation) -> AgentAction:
        if self._i < len(self._actions):
            action = self._actions[self._i]
            self._i += 1
            return action
        return AgentAction.declare_done(summary="done", entrypoint="python main.py")

    def run(self, task_view: object, tools: object, oracle_channel: object) -> Iterator[AgentAction]:
        yield from self._actions


class FakeOracle:
    """A scripted oracle: returns a fixed severity + text for every query."""

    def __init__(self, severity: Severity = Severity.NONE, text: str = "a table, please") -> None:
        self.severity = severity
        self.text = text
        self.calls = 0

    def answer(self, ctx: OracleQueryContext) -> OracleResponse:
        self.calls += 1
        return OracleResponse(
            severity=self.severity,
            text=self.text,
            verdict=Verdict.NA,
            reveals=["HP1"] if self.severity == Severity.NONE else [],
        )


class PassingVerifier:
    """A verifier that asserts a known file exists in the workspace -> all pass."""

    def __init__(self, required_file: str) -> None:
        self.required_file = required_file

    def verify(
        self,
        task: Task,
        sandbox: object,
        *,
        trigger: str,
        entrypoint: object,
    ) -> VerificationRun:
        ws = Path(sandbox.workspace)  # type: ignore[attr-defined]
        ok = (ws / self.required_file).is_file()
        return VerificationRun(
            trigger=trigger,  # type: ignore[arg-type]
            entrypoint=str(entrypoint) if entrypoint else None,
            must_have=[CriterionResult(id="MH1", passed=ok)],
            all_must_pass=ok,
            rubric_score=1.0 if ok else 0.0,
        )


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _make_task() -> Task:
    return Task(
        id="ub-smoke-0001",
        title="Tiny CLI",
        user_goal="Build a tiny script that prints hello.",
        domain="cli-util",
        difficulty="T1",  # type: ignore[arg-type]
        deliverable_type="script",  # type: ignore[arg-type]
        env=TaskEnv(),
        hidden=HiddenSpec(
            summary="A script printing hello, output as a table.",
            acceptance_criteria=[
                AcceptanceCriterion(id="MH1", text="prints hello", is_core=True, check_kind="func"),  # type: ignore[arg-type]
            ],
        ),
    )


def _make_manifest(task: Task, seed: int) -> object:
    return build_manifest(
        task_id=task.id,
        seed=seed,
        config={"run": "smoke", "api_key": "sk-shouldberedacted00000000"},
        package_version="0.1.0",
        git_sha_value="deadbeef",
        budgets=BudgetLimits().as_payload(),
        agent={"id": "fake-agent"},
        oracle={"id": "fake-oracle", "persona": "non_expert_user"},
        sandbox={"network": "deny"},
    )


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


def test_local_sandbox_write_read_exec(tmp_path: Path) -> None:
    """The local sandbox actually writes, reads, execs, and diffs on a Mac."""
    with LocalSubprocessSandbox(task_id="t", root=tmp_path) as sb:
        before = sb.snapshot()
        sb.write("main.py", "print('hello')\n")
        after = sb.snapshot()
        edits = sb.diff(before, after)
        assert any(e.path == "main.py" and e.op == "create" for e in edits)
        assert sb.read("main.py") == "print('hello')\n"
        res = sb.exec("python3 main.py")
        assert res.exit_code == 0
        assert "hello" in res.stdout


def test_local_sandbox_path_escape_denied(tmp_path: Path) -> None:
    """Writes that escape the workspace root are denied."""
    from usabench.core.errors import SandboxError

    with LocalSubprocessSandbox(task_id="t", root=tmp_path) as sb:
        with pytest.raises(SandboxError):
            sb.write("../escape.txt", "nope")


def test_run_episode_accept_path(tmp_path: Path) -> None:
    """A full accepting episode produces a complete, valid, hash-chained trace."""
    task = _make_task()
    manifest = _make_manifest(task, seed=7)
    trace_path = tmp_path / "runs" / manifest.run_id / "trace.jsonl"  # type: ignore[attr-defined]

    agent = ScriptedAgent(
        [
            AgentAction.ask_user("table or json?", query_class=QueryClass.CLARIFICATION),
            AgentAction.write_file("main.py", "print('hello')\n"),
            AgentAction.run_cmd("python main.py"),
            AgentAction.message_user("I built it; demoing now."),
            AgentAction.declare_done(summary="prints hello", entrypoint="python main.py"),
        ]
    )
    oracle = FakeOracle(severity=Severity.NONE)
    verifier = PassingVerifier(required_file="main.py")

    sandbox = LocalSubprocessSandbox(task_id=task.id, root=tmp_path / "ws")
    sandbox.setup()
    bus = InteractionBus(trace_path, run_id=manifest.run_id, oracle=oracle).open()
    budget = BudgetMeter(BudgetLimits())
    try:
        result = run_episode(
            task,
            agent,
            sandbox=sandbox,
            bus=bus,
            budget=budget,
            seed=7,
            manifest=manifest,
            verifier=verifier,
        )
    finally:
        bus.close()
        sandbox.teardown()

    assert result.accepted is True
    assert result.status == RunStatus.ACCEPTED
    assert result.terminated_reason == TerminatedReason.ACCEPT
    assert oracle.calls == 1
    assert trace_path.is_file()
    assert is_run_complete(trace_path.parent)

    _assert_trace_integrity(trace_path, run_id=manifest.run_id)


def test_run_episode_budget_exhaustion(tmp_path: Path) -> None:
    """When the turn budget is tiny the run terminates as budget-exhausted."""
    task = _make_task()
    manifest = _make_manifest(task, seed=1)
    trace_path = tmp_path / "trace.jsonl"

    # Agent keeps writing forever; only 2 turns are allowed.
    agent = ScriptedAgent([AgentAction.write_file(f"f{i}.txt", "x") for i in range(50)])
    oracle = FakeOracle()
    verifier = PassingVerifier(required_file="f0.txt")

    sandbox = LocalSubprocessSandbox(task_id=task.id, root=tmp_path / "ws")
    sandbox.setup()
    bus = InteractionBus(trace_path, run_id=manifest.run_id, oracle=oracle).open()
    budget = BudgetMeter(BudgetLimits(max_turns=2))
    try:
        result = run_episode(
            task, agent, sandbox=sandbox, bus=bus, budget=budget, seed=1,
            manifest=manifest, verifier=verifier,
        )
    finally:
        bus.close()
        sandbox.teardown()

    assert result.accepted is False
    assert result.status == RunStatus.BUDGET_EXHAUSTED
    # StrictModel uses use_enum_values=True, so terminated_reason serializes to a str.
    assert TerminatedReason(result.terminated_reason).is_budget
    # A forced-final verification must still have run so partial credit exists.
    types = [parse_event(json.loads(ln)).type for ln in trace_path.read_text().splitlines() if ln.strip()]
    assert InteractionType.VERIFICATION_RUN.value in [str(t) for t in types]
    _assert_trace_integrity(trace_path, run_id=manifest.run_id)


def test_run_episode_give_up(tmp_path: Path) -> None:
    """A give_up terminates the run as agent_gave_up."""
    task = _make_task()
    manifest = _make_manifest(task, seed=2)
    trace_path = tmp_path / "trace.jsonl"

    agent = ScriptedAgent([AgentAction.give_up("too hard")])
    oracle = FakeOracle()

    sandbox = LocalSubprocessSandbox(task_id=task.id, root=tmp_path / "ws")
    sandbox.setup()
    bus = InteractionBus(trace_path, run_id=manifest.run_id, oracle=oracle).open()
    budget = BudgetMeter(BudgetLimits())
    try:
        result = run_episode(
            task, agent, sandbox=sandbox, bus=bus, budget=budget, seed=2, manifest=manifest,
        )
    finally:
        bus.close()
        sandbox.teardown()

    assert result.status == RunStatus.AGENT_GAVE_UP
    assert result.terminated_reason == TerminatedReason.GIVE_UP
    _assert_trace_integrity(trace_path, run_id=manifest.run_id)


def test_oracle_severity_counted_in_episode_end(tmp_path: Path) -> None:
    """An L2 hint is tallied into the episode_end interventions_by_severity cache."""
    task = _make_task()
    manifest = _make_manifest(task, seed=3)
    trace_path = tmp_path / "trace.jsonl"

    agent = ScriptedAgent(
        [
            AgentAction.ask_user("any hint?", query_class=QueryClass.HINT_REQUEST),
            AgentAction.write_file("main.py", "print('hello')\n"),
            AgentAction.declare_done(summary="done", entrypoint="python main.py"),
        ]
    )
    oracle = FakeOracle(severity=Severity.SUBSTANTIVE_SPEC_INFO, text="check time parsing")
    verifier = PassingVerifier(required_file="main.py")

    sandbox = LocalSubprocessSandbox(task_id=task.id, root=tmp_path / "ws")
    sandbox.setup()
    bus = InteractionBus(trace_path, run_id=manifest.run_id, oracle=oracle).open()
    budget = BudgetMeter(BudgetLimits())
    try:
        run_episode(
            task, agent, sandbox=sandbox, bus=bus, budget=budget, seed=3,
            manifest=manifest, verifier=verifier,
        )
    finally:
        bus.close()
        sandbox.teardown()

    end = _read_episode_end(trace_path)
    # severity 2 once (the hint); the accept review is severity 0.
    assert end.payload.interventions_by_severity.get("2") == 1  # type: ignore[attr-defined]


def test_manifest_redacts_secrets(tmp_path: Path) -> None:
    """The manifest must not contain the raw API key passed in the config."""
    task = _make_task()
    manifest = _make_manifest(task, seed=9)
    blob = json.dumps(manifest.model_dump(mode="json"))  # type: ignore[attr-defined]
    assert "sk-shouldberedacted" not in blob


# --------------------------------------------------------------------------- #
# Integrity helpers                                                            #
# --------------------------------------------------------------------------- #


def _read_episode_end(trace_path: Path):  # type: ignore[no-untyped-def]
    lines = [ln for ln in trace_path.read_text().splitlines() if ln.strip()]
    return parse_event(json.loads(lines[-1]))


def _assert_trace_integrity(trace_path: Path, *, run_id: str) -> None:
    """Assert the canonical trace.jsonl integrity invariants (docs/protocol.md §4.3)."""
    lines = [ln for ln in trace_path.read_text().splitlines() if ln.strip()]
    assert lines, "trace is empty"

    prev = GENESIS_HASH
    seqs: list[int] = []
    types: list[str] = []
    tool_call_ids: set[str] = set()
    for raw in lines:
        obj = json.loads(raw)
        env = parse_event(obj)
        assert env.run_id == run_id
        seqs.append(env.seq)
        types.append(str(env.type))
        # Hash chain: prev_hash links, and hash is recomputable.
        assert env.prev_hash == prev, f"broken chain at seq {env.seq}"
        recomputed = next_hash(env.prev_hash, env.canonical_without_hash())
        assert env.hash == recomputed, f"bad hash at seq {env.seq}"
        prev = env.hash
        if str(env.type) == InteractionType.TOOL_CALL.value:
            tool_call_ids.add(obj["payload"]["call_id"])

    # seq strictly increasing from 0 with no gaps.
    assert seqs == list(range(len(seqs))), f"non-monotonic seq: {seqs}"
    # Exactly one episode_start (first) and one episode_end (last).
    assert types[0] == InteractionType.EPISODE_START.value
    assert types[-1] == InteractionType.EPISODE_END.value
    assert types.count(InteractionType.EPISODE_START.value) == 1
    assert types.count(InteractionType.EPISODE_END.value) == 1
