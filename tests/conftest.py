"""Shared pytest fixtures for the usability-benchmark test suite.

These fixtures give every test a deterministic, zero-cost, offline foundation:

* :func:`fake_llm` -- a scripted :class:`~usabench.llm.fake.FakeLLMClient` (no
  network, $0 cost) that satisfies the ``LLMClient`` protocol.
* :func:`sandbox` -- a real, working :class:`LocalSubprocessSandbox` rooted under
  pytest's ``tmp_path`` (auto torn down).
* :func:`sample_task` -- a small but complete :class:`~usabench.core.schema.Task`
  with hidden gold (acceptance criteria, ambiguity points, info units) so the
  metric / gold / integrity functions have real material to chew on.
* :func:`sample_gold` -- the :class:`~usabench.eval.gold.Gold` accessor over it.
* :func:`trace_builder` -- a tiny, hash-chained ``trace.jsonl`` builder that
  produces schema-valid envelopes in total ``seq`` order (the ONE canonical
  artifact), used to hand-build episode fixtures for the metric tests.
* :func:`sample_trace_path` -- a hand-built, known-good ``trace.jsonl`` on disk.

Nothing here imports an optional/heavy dependency, so the whole suite runs with
the base + dev install.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from usabench.core.enums import Actor, InteractionType
from usabench.core.ids import GENESIS_HASH, next_hash
from usabench.core.schema import (
    AcceptanceCriterion,
    AgentMessage,
    AmbiguityPoint,
    BudgetSnapshot,
    Checkpoint,
    CodeRun,
    CriterionResult,
    EpisodeEnd,
    EpisodeStart,
    FileEdit,
    HiddenSpec,
    InfoUnit,
    MessageToUser,
    OracleQuery,
    OracleResponse,
    Task,
    TaskEnv,
    TraceEnvelope,
    TraceEvent,
    VerificationRun,
)
from usabench.eval.gold import Gold, as_gold
from usabench.harness.sandbox import LocalSubprocessSandbox
from usabench.llm.fake import FakeLLMClient

# Re-export the most useful payload constructors so tests can build events
# without a long import list. (Imported above; named here for discoverability.)
__all__ = [
    "AgentMessage",
    "Checkpoint",
    "CodeRun",
    "CriterionResult",
    "EpisodeEnd",
    "EpisodeStart",
    "FileEdit",
    "MessageToUser",
    "OracleQuery",
    "OracleResponse",
    "VerificationRun",
    "TraceBuilder",
]


# --------------------------------------------------------------------------- #
# LLM                                                                          #
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    """A deterministic, zero-cost scripted LLM client (echoes the last user msg)."""
    return FakeLLMClient.echo(prefix="")


# --------------------------------------------------------------------------- #
# Sandbox                                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def sandbox(tmp_path: Path) -> Any:
    """A real, working local subprocess sandbox rooted under ``tmp_path``.

    Yields a *set-up* sandbox and tears it down afterwards.
    """
    sb = LocalSubprocessSandbox(task_id="t-fixture", root=tmp_path / "ws")
    sb.setup()
    try:
        yield sb
    finally:
        sb.teardown()


# --------------------------------------------------------------------------- #
# Task / gold                                                                  #
# --------------------------------------------------------------------------- #


def _build_sample_task() -> Task:
    """Construct a small but complete calendar-summarizer task with gold."""
    return Task(
        id="ub-cal-0007",
        title="Calendar workload summarizer",
        user_goal="build me a tool that analyzes my calendar",
        domain="data-analysis",
        difficulty="T2",  # type: ignore[arg-type]
        deliverable_type="cli-tool",  # type: ignore[arg-type]
        env=TaskEnv(),
        accept_threshold=0.80,
        hidden=HiddenSpec(
            summary="weekly time-allocation breakdown from an ICS file",
            acceptance_criteria=[
                AcceptanceCriterion(
                    id="AC1",
                    text="parses an ICS file",
                    weight=2.0,
                    is_core=True,
                    is_hard=True,
                    check_kind="func",  # type: ignore[arg-type]
                ),
                AcceptanceCriterion(
                    id="AC2",
                    text="aggregates hours per category per week",
                    weight=2.0,
                    is_core=True,
                    is_hard=False,
                    check_kind="func",  # type: ignore[arg-type]
                ),
                AcceptanceCriterion(
                    id="AC3",
                    text="renders a table",
                    weight=1.0,
                    is_core=False,
                    is_hard=False,
                    check_kind="rubric_auto",  # type: ignore[arg-type]
                ),
            ],
            ambiguity_points=[
                AmbiguityPoint(
                    id="AP1",
                    question="all-day events: count as 0 hours or 24?",
                    gold="all-day events count as 0 hours",
                    severity="medium",
                ),
            ],
            info_units=[
                InfoUnit(id="AC1", klass="requirement", desc="must parse ICS"),
                InfoUnit(id="AC2", klass="requirement", desc="weekly aggregation"),
                InfoUnit(id="pref:table", klass="preference", desc="prefers a table"),
            ],
        ),
    )


@pytest.fixture
def sample_task() -> Task:
    """A small but complete :class:`Task` with hidden gold."""
    return _build_sample_task()


@pytest.fixture
def sample_gold(sample_task: Task) -> Gold:
    """The :class:`Gold` accessor over :func:`sample_task`."""
    return as_gold(sample_task)


# --------------------------------------------------------------------------- #
# Trace builder                                                               #
# --------------------------------------------------------------------------- #


class TraceBuilder:
    """A minimal, hash-chained ``trace.jsonl`` builder for fixtures.

    Mirrors the :class:`~usabench.harness.interaction_bus.InteractionBus` writer
    contract (monotonic ``seq`` from 0, deterministic wall clock, ``next_hash``
    chain, append-only) WITHOUT requiring a sandbox or oracle, so metric tests can
    hand-assemble a precise episode and assert exact metric values.
    """

    def __init__(self, run_id: str = "r-test", *, t0: float = 1_718_800_000.0) -> None:
        self.run_id = run_id
        self._t0 = t0
        self._seq = 0
        self._prev = GENESIS_HASH
        self.events: list[TraceEnvelope] = []

    def add(
        self,
        actor: Actor,
        payload: TraceEvent,
        *,
        t_turn: int | None = None,
        dt: float = 1.0,
        budgets_after: BudgetSnapshot | None = None,
    ) -> TraceEnvelope:
        """Append one chained event and return it.

        Args:
            actor: Who produced the event.
            payload: A typed trace payload.
            t_turn: Agent-turn index (or ``None`` for harness events).
            dt: Seconds to advance the deterministic clock for this event.
            budgets_after: Optional budget snapshot for this event.
        """
        env = TraceEnvelope(
            run_id=self.run_id,
            event_id=f"e{self._seq}",
            seq=self._seq,
            ts=self._t0 + self._seq * dt,
            t_turn=t_turn,
            actor=actor,
            type=InteractionType(payload.type),
            payload=payload,
            budgets_after=budgets_after or BudgetSnapshot(),
            prev_hash=self._prev,
        )
        env.hash = next_hash(self._prev, env.canonical_without_hash())
        self._prev = env.hash
        self._seq += 1
        self.events.append(env)
        return env

    def write(self, path: Path) -> Path:
        """Serialize the built events as a byte-stable ``trace.jsonl`` and return path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for env in self.events:
                line = json.dumps(
                    env.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
                )
                fh.write(line + "\n")
        return path


@pytest.fixture
def trace_builder() -> Callable[..., TraceBuilder]:
    """Factory returning a fresh :class:`TraceBuilder` (optionally with a run id)."""

    def _make(run_id: str = "r-test") -> TraceBuilder:
        return TraceBuilder(run_id=run_id)

    return _make


def build_known_good_trace(run_id: str = "r-good") -> TraceBuilder:
    """A hand-built, known-GOOD episode: solved, no help, all core criteria met.

    The episode: start -> a working checkpoint at full score -> a passing
    verification (all must-have met) -> a bare-accept oracle review (sev 0) ->
    final_acceptance accepted -> episode_end. No oracle queries, no interventions.
    """
    tb = TraceBuilder(run_id=run_id)
    tb.add(
        Actor.HARNESS,
        EpisodeStart(
            task_id="ub-cal-0007",
            hidden_spec_sha256="0" * 64,
            seed=7,
        ),
        t_turn=None,
    )
    tb.add(
        Actor.AGENT,
        AgentMessage(text="I'll parse the ICS and print a weekly table."),
        t_turn=0,
    )
    tb.add(
        Actor.AGENT,
        FileEdit(path="main.py", op="create", added=40, removed=0, loc_after=40),
        t_turn=1,
    )
    tb.add(
        Actor.AGENT,
        CodeRun(cmd="python -m pytest", exit_code=0, is_test=True, self_test_passed=True),
        t_turn=2,
    )
    tb.add(
        Actor.HARNESS,
        Checkpoint(
            weighted_score=1.0,
            is_working_version=True,
            criteria_state={"AC1": True, "AC2": True, "AC3": True},
            criteria_passed=3,
            criteria_total=3,
        ),
        t_turn=2,
    )
    tb.add(
        Actor.AGENT,
        MessageToUser(text="The tool is ready to use; here is a demo."),
        t_turn=3,
    )
    tb.add(
        Actor.HARNESS,
        VerificationRun(
            trigger="submit",
            entrypoint="python main.py",
            must_have=[
                CriterionResult(id="AC1", passed=True),
                CriterionResult(id="AC2", passed=True),
            ],
            should_have=[CriterionResult(id="AC3", score=1.0)],
            all_must_pass=True,
            rubric_score=1.0,
        ),
        t_turn=3,
    )
    tb.add(
        Actor.ORACLE,
        OracleResponse(responds_to=None, severity=0, text="looks complete", verdict="accept"),
        t_turn=3,
    )
    tb.add(
        Actor.HARNESS,
        EpisodeEnd(
            terminated_reason="accept",
            accepted=True,
            final_weighted_score=1.0,
            interventions_by_severity={"0": 1},
        ),
        t_turn=None,
    )
    return tb


@pytest.fixture
def sample_trace_path(tmp_path: Path) -> Path:
    """A hand-built known-good ``trace.jsonl`` written to disk."""
    tb = build_known_good_trace()
    return tb.write(tmp_path / "runs" / "r-good" / "trace.jsonl")
