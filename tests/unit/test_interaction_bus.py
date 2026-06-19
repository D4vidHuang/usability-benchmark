"""InteractionBus: trace-writer typing, hash chaining, and oracle-severity counting.

The bus is the single funnel for the canonical artifact AND the sole agent<->oracle
channel (``docs/protocol.md`` §4, ``docs/infra.md``). These tests prove:

* every emitted line is a chained, monotonic, schema-derived envelope;
* ``ask_oracle`` logs both sides and authoritatively links the response to its
  query (so the scorer's ref-integrity invariant holds regardless of the oracle);
* oracle-response severities are tallied into the ``episode_end`` cache exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from usabench.core.enums import (
    Actor,
    InteractionType,
    QueryClass,
    Severity,
    Verdict,
)
from usabench.core.ids import GENESIS_HASH, next_hash
from usabench.core.schema import (
    AgentMessage,
    OracleResponse,
    parse_event,
)
from usabench.harness.interaction_bus import InteractionBus, OracleQueryContext


class _ScriptedOracle:
    """A scripted oracle returning a fixed severity/verdict per query."""

    def __init__(self, severity: Severity, *, verdict: Verdict = Verdict.NA) -> None:
        self.severity = severity
        self.verdict = verdict
        self.calls = 0
        self.last_ctx: OracleQueryContext | None = None

    def answer(self, ctx: OracleQueryContext) -> OracleResponse:
        self.calls += 1
        self.last_ctx = ctx
        # Deliberately set a WRONG responds_to to prove the bus overrides it.
        return OracleResponse(
            responds_to="bogus-id",
            severity=self.severity,
            text="answer",
            verdict=self.verdict,
            info_units_revealed=["AC1"] if int(self.severity) >= 2 else [],
        )


def _clock_factory():  # type: ignore[no-untyped-def]
    """A deterministic monotonically-increasing clock."""
    state = {"t": 1_000.0}

    def _clock() -> float:
        state["t"] += 1.0
        return state["t"]

    return _clock


# --------------------------------------------------------------------------- #
# Writer typing + chaining                                                     #
# --------------------------------------------------------------------------- #


def test_emit_assigns_monotonic_seq_and_chains_hash(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    with InteractionBus(path, run_id="r1", clock=_clock_factory()) as bus:
        e0 = bus.emit(Actor.AGENT, AgentMessage(text="a"), t_turn=0)
        e1 = bus.emit(Actor.AGENT, AgentMessage(text="b"), t_turn=1)

    assert e0.seq == 0 and e1.seq == 1
    assert e0.prev_hash == GENESIS_HASH
    assert e1.prev_hash == e0.hash
    # The envelope type is derived from the payload discriminator.
    assert str(e0.type) == InteractionType.AGENT_MESSAGE.value
    # Recompute the chain from disk.
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    prev = GENESIS_HASH
    for raw in lines:
        env = parse_event(json.loads(raw))
        assert env.prev_hash == prev
        assert env.hash == next_hash(env.prev_hash, env.canonical_without_hash())
        prev = env.hash


def test_emit_before_open_raises(tmp_path: Path) -> None:
    bus = InteractionBus(tmp_path / "t.jsonl", run_id="r1")
    with pytest.raises(RuntimeError):
        bus.emit(Actor.AGENT, AgentMessage(text="x"), t_turn=0)


def test_written_line_is_byte_stable_sorted_keys(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    with InteractionBus(path, run_id="r1", clock=_clock_factory()) as bus:
        bus.emit(Actor.AGENT, AgentMessage(text="x"), t_turn=0)
    line = path.read_text().splitlines()[0]
    # sorted-keys + compact separators: 'event_id' precedes 'run_id', no spaces.
    assert ", " not in line and '": ' not in line
    obj_keys = list(json.loads(line).keys())
    assert obj_keys == sorted(obj_keys)


# --------------------------------------------------------------------------- #
# Oracle channel: query/response logging + responds_to linkage                 #
# --------------------------------------------------------------------------- #


def test_ask_oracle_logs_both_sides_and_links_response(tmp_path: Path) -> None:
    oracle = _ScriptedOracle(Severity.SUBSTANTIVE_SPEC_INFO)
    path = tmp_path / "trace.jsonl"
    with InteractionBus(path, run_id="r1", oracle=oracle, clock=_clock_factory()) as bus:
        q_env, r_env = bus.ask_oracle(
            "all-day events: 0 or 24 hours?",
            QueryClass.CLARIFICATION,
            t_turn=2,
        )

    assert oracle.calls == 1
    # The bus authoritatively links the response to the query, overriding the
    # oracle's (deliberately bogus) responds_to.
    assert r_env.payload.responds_to == q_env.event_id  # type: ignore[attr-defined]
    assert q_env.payload.query_class == QueryClass.CLARIFICATION  # type: ignore[attr-defined]
    # The context carried the originating query event id to the oracle.
    assert oracle.last_ctx is not None
    assert oracle.last_ctx.query_event_id == q_env.event_id


def test_ask_oracle_without_backend_raises(tmp_path: Path) -> None:
    with InteractionBus(tmp_path / "t.jsonl", run_id="r1") as bus:
        with pytest.raises(RuntimeError):
            bus.ask_oracle("q?", QueryClass.CLARIFICATION, t_turn=0)


# --------------------------------------------------------------------------- #
# Severity counting                                                            #
# --------------------------------------------------------------------------- #


def test_severity_counts_accumulate_across_responses(tmp_path: Path) -> None:
    oracle = _ScriptedOracle(Severity.DIRECTIONAL_HINT)  # sev 3
    path = tmp_path / "trace.jsonl"
    with InteractionBus(path, run_id="r1", oracle=oracle, clock=_clock_factory()) as bus:
        bus.ask_oracle("hint?", QueryClass.HINT_REQUEST, t_turn=1)
        bus.ask_oracle("hint again?", QueryClass.HINT_REQUEST, t_turn=2)
        # A bare-accept review is severity 0.
        bus.oracle_review(Verdict.ACCEPT, severity=Severity.NONE, t_turn=3)

    counts = bus.severity_counts
    assert counts[3] == 2
    assert counts[0] == 1
    assert counts[5] == 0


def test_oracle_review_reject_carries_severity(tmp_path: Path) -> None:
    """A reject that names a fix is itself assistance and is tallied with its severity."""
    path = tmp_path / "trace.jsonl"
    with InteractionBus(path, run_id="r1", clock=_clock_factory()) as bus:
        env = bus.oracle_review(
            Verdict.REJECT,
            severity=Severity.SUBSTANTIVE_SPEC_INFO,
            text="you missed the weekly aggregation",
            cited_criteria=["AC2"],
            t_turn=4,
        )
    assert str(env.actor) == Actor.ORACLE.value
    assert env.payload.verdict == Verdict.REJECT  # type: ignore[attr-defined]
    assert int(env.payload.severity) == 2  # type: ignore[attr-defined]
    assert bus.severity_counts[2] == 1
