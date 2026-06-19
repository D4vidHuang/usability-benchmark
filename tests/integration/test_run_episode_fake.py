"""Full FakeLLM episode through ``harness.run_episode``, scored end-to-end.

This is the core regression test: the real :class:`ReActScaffold` agent, driven by
a deterministic :class:`FakeLLMClient` scripted to emit native tool calls, runs a
complete episode against the real :class:`LocalSubprocessSandbox` + a fake oracle +
an inline verifier. We then assert:

* the run terminates ACCEPTED with the expected reason;
* the produced ``trace.jsonl`` is a complete, totally-ordered, hash-chained,
  schema-valid artifact (the ONE canonical artifact, ``DESIGN.md`` invariant 4);
* every A-G metric, the geometric composite, and the integrity flags compute
  cleanly off that trace -- i.e. the harness and scorer agree on the same file.

No network, no API, no paid calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from usabench.agent.scaffold import ReActScaffold, ScaffoldConfig
from usabench.core.enums import RunStatus, Severity, TerminatedReason, Verdict
from usabench.core.ids import GENESIS_HASH, next_hash
from usabench.core.schema import (
    CriterionResult,
    OracleResponse,
    Task,
    VerificationRun,
    parse_event,
)
from usabench.eval import compute_all, compute_composite, compute_integrity
from usabench.eval.composite import CompositeInputs
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
from usabench.llm.client import Completion, ToolCall
from usabench.llm.fake import FakeLLMClient

ROOT = Path(__file__).resolve().parents[2]
TRACE_SCHEMA = json.loads((ROOT / "schemas" / "trace.schema.json").read_text())


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


def _tool_completion(name: str, args: dict, *, call_id: str) -> Completion:
    """A canned model turn that issues exactly one native tool call."""
    return Completion(
        text="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
        finish_reason="tool_calls",
    )


def _scripted_fake_llm() -> FakeLLMClient:
    """A FakeLLM that drives a tiny build: write -> run -> declare_done."""
    script = [
        _tool_completion("write_file", {"path": "main.py", "content": "print('hello')\n"}, call_id="c1"),
        _tool_completion("run_cmd", {"cmd": "python main.py"}, call_id="c2"),
        _tool_completion("message_user", {"text": "Built it; the script prints hello."}, call_id="c3"),
        _tool_completion("declare_done", {"summary": "prints hello", "entrypoint": "python main.py"}, call_id="c4"),
    ]
    return FakeLLMClient(script=script)


class _FakeOracle:
    """A scripted oracle: a fixed severity/verdict for every query."""

    def __init__(self, severity: Severity = Severity.NONE) -> None:
        self.severity = severity
        self.calls = 0

    def answer(self, ctx: OracleQueryContext) -> OracleResponse:
        self.calls += 1
        return OracleResponse(severity=self.severity, text="sure", verdict=Verdict.NA)


class _FileExistsVerifier:
    """Passes iff a required file exists in the workspace (all must-have met)."""

    def __init__(self, required_file: str) -> None:
        self.required_file = required_file

    def verify(self, task: Task, sandbox: object, *, trigger: str, entrypoint: object) -> VerificationRun:
        ws = Path(sandbox.workspace)  # type: ignore[attr-defined]
        ok = (ws / self.required_file).is_file()
        return VerificationRun(
            trigger=trigger,  # type: ignore[arg-type]
            entrypoint=str(entrypoint) if entrypoint else None,
            must_have=[CriterionResult(id="MH1", passed=ok)],
            all_must_pass=ok,
            rubric_score=1.0 if ok else 0.0,
        )


def _make_task() -> Task:
    from usabench.core.schema import AcceptanceCriterion, HiddenSpec, TaskEnv

    return Task(
        id="ub-smoke-int",
        title="Tiny CLI",
        user_goal="Build a tiny script that prints hello.",
        domain="cli-util",
        difficulty="T1",  # type: ignore[arg-type]
        deliverable_type="script",  # type: ignore[arg-type]
        env=TaskEnv(),
        hidden=HiddenSpec(
            summary="A script printing hello.",
            acceptance_criteria=[
                AcceptanceCriterion(id="MH1", text="prints hello", is_core=True, is_hard=True, check_kind="func"),  # type: ignore[arg-type]
            ],
        ),
    )


# --------------------------------------------------------------------------- #
# The end-to-end test                                                          #
# --------------------------------------------------------------------------- #


def _run(tmp_path: Path):  # type: ignore[no-untyped-def]
    task = _make_task()
    manifest = build_manifest(
        task_id=task.id,
        seed=7,
        config={"run": "fake-int", "api_key": "sk-shouldberedacted00000000"},
        package_version="0.1.0",
        git_sha_value="deadbeef",
        budgets=BudgetLimits().as_payload(),
        agent={"id": "react-fake"},
        oracle={"id": "fake-oracle"},
        sandbox={"network": "deny"},
    )
    trace_path = tmp_path / "runs" / manifest.run_id / "trace.jsonl"

    agent = ReActScaffold(_scripted_fake_llm(), ScaffoldConfig(max_steps=20, native_tools=True))
    oracle = _FakeOracle(severity=Severity.NONE)
    verifier = _FileExistsVerifier(required_file="main.py")

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
    return task, manifest, trace_path, result


def test_fake_episode_runs_and_accepts(tmp_path: Path) -> None:
    _, manifest, trace_path, result = _run(tmp_path)
    assert result.status == RunStatus.ACCEPTED
    assert result.accepted is True
    assert result.terminated_reason == TerminatedReason.ACCEPT
    assert trace_path.is_file()
    assert is_run_complete(trace_path.parent)


def test_fake_episode_trace_is_complete_and_chained(tmp_path: Path) -> None:
    _, manifest, trace_path, _ = _run(tmp_path)
    lines = [ln for ln in trace_path.read_text().splitlines() if ln.strip()]
    assert lines, "trace is empty"

    prev = GENESIS_HASH
    seqs: list[int] = []
    types: list[str] = []
    for raw in lines:
        obj = json.loads(raw)
        # Each line is schema-valid against trace.schema.json.
        jsonschema.validate(instance=obj, schema=TRACE_SCHEMA)
        env = parse_event(obj)
        assert env.run_id == manifest.run_id
        seqs.append(env.seq)
        types.append(str(env.type))
        assert env.prev_hash == prev, f"broken chain at seq {env.seq}"
        assert env.hash == next_hash(env.prev_hash, env.canonical_without_hash())
        prev = env.hash

    # Total order: seq is 0..N-1 with no gaps; bookended by start/end.
    assert seqs == list(range(len(seqs)))
    assert types[0] == "episode_start"
    assert types[-1] == "episode_end"
    assert types.count("episode_start") == 1
    assert types.count("episode_end") == 1
    # The agent's file write was diffed into a file_edit by the harness.
    assert "file_edit" in types
    assert "verification_run" in types


def test_fake_episode_scores_end_to_end(tmp_path: Path) -> None:
    task, _, trace_path, _ = _run(tmp_path)
    trace = [parse_event(json.loads(ln)) for ln in trace_path.read_text().splitlines() if ln.strip()]

    # Every A-G metric computes off the same canonical trace.
    metrics = compute_all(trace, task)
    assert metrics["A1_success_binary"] == 1
    assert metrics["A2_criteria_score"] == pytest.approx(1.0)
    # No oracle help was solicited -> zero assistance cost.
    assert metrics["C1_assistance_cost"] == pytest.approx(0.0)
    assert metrics["B1_n_interventions"] >= 0

    # Integrity flags compute and are clean for an honest solved run.
    flags = compute_integrity(trace, task)
    assert flags.fake_done is False
    assert flags.hard_pass_frac == pytest.approx(1.0)

    # The geometric headline composite is well-defined and high for S=1, no help.
    comp = compute_composite(
        CompositeInputs(
            s_core=metrics["A3_core_criteria_score"],
            assistance_cost=metrics["C1_assistance_cost"],
            autonomy=metrics["D1_autonomy_ratio"],
            robustness=1.0,
            success_binary=metrics["A1_success_binary"],
            n_clarifications=metrics["B2_n_clarifications"],
            goal_drift=metrics["A4_goal_drift"],
            fake_done=flags.fake_done,
        )
    )
    assert comp.usability_geometric == pytest.approx(1.0)


def test_fake_episode_manifest_redacts_secret(tmp_path: Path) -> None:
    _, manifest, _, _ = _run(tmp_path)
    assert "sk-shouldberedacted" not in manifest.model_dump_json()
