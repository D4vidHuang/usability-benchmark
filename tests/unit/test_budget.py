"""Multi-ceiling budget metering (``docs/protocol.md`` §1.3).

The :class:`BudgetMeter` enforces five independent ceilings -- turns, wall, tokens,
cost, oracle queries -- and terminates on the *first* one breached. These tests
exercise each ceiling in isolation plus the exhausted-reason mapping, using an
injected deterministic clock so the wall-clock ceiling is testable without sleeping.
"""

from __future__ import annotations

import pytest

from usabench.core.enums import TerminatedReason
from usabench.core.errors import BudgetExceeded
from usabench.harness.budget import BudgetLimits, BudgetMeter


class _FakeClock:
    """A controllable monotonic clock for the wall-budget tests."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# --------------------------------------------------------------------------- #
# Per-ceiling cutoffs                                                          #
# --------------------------------------------------------------------------- #


def test_turn_ceiling_raises_on_overflow() -> None:
    meter = BudgetMeter(BudgetLimits(max_turns=2))
    meter.debit_turn()
    meter.debit_turn()  # exactly at the ceiling -- no raise yet (used==limit)
    with pytest.raises(BudgetExceeded):
        meter.debit_turn()  # crosses the ceiling
    assert meter.exhausted_reason == TerminatedReason.BUDGET_TURNS


def test_oracle_query_ceiling() -> None:
    meter = BudgetMeter(BudgetLimits(max_oracle_queries=1))
    meter.debit_oracle_query()
    with pytest.raises(BudgetExceeded):
        meter.debit_oracle_query()
    assert meter.exhausted_reason == TerminatedReason.BUDGET_ORACLE


def test_token_ceiling_via_usage() -> None:
    meter = BudgetMeter(BudgetLimits(max_tokens=100))
    meter.debit_usage(tokens=80, cost_usd=0.0)
    with pytest.raises(BudgetExceeded):
        meter.debit_usage(tokens=50, cost_usd=0.0)
    assert meter.exhausted_reason == TerminatedReason.BUDGET_TOKENS


def test_cost_ceiling_via_usage() -> None:
    meter = BudgetMeter(BudgetLimits(max_cost_usd=1.0, max_tokens=10**9))
    meter.debit_usage(tokens=1, cost_usd=0.6)
    with pytest.raises(BudgetExceeded):
        meter.debit_usage(tokens=1, cost_usd=0.6)
    assert meter.exhausted_reason == TerminatedReason.BUDGET_COST


def test_wall_ceiling_with_injected_clock() -> None:
    clock = _FakeClock()
    meter = BudgetMeter(BudgetLimits(max_wall_s=10.0), clock=clock)
    clock.t = 5.0
    assert not meter.is_exhausted()
    clock.t = 11.0  # past the ceiling
    assert meter.is_exhausted()
    with pytest.raises(BudgetExceeded):
        meter.check()
    assert meter.exhausted_reason == TerminatedReason.BUDGET_WALL


# --------------------------------------------------------------------------- #
# First-ceiling-wins + snapshot accounting                                     #
# --------------------------------------------------------------------------- #


def test_token_checked_before_cost_in_usage() -> None:
    """When a single call breaches both, the token ceiling is reported first."""
    meter = BudgetMeter(BudgetLimits(max_tokens=10, max_cost_usd=0.01))
    with pytest.raises(BudgetExceeded):
        meter.debit_usage(tokens=100, cost_usd=10.0)
    assert meter.exhausted_reason == TerminatedReason.BUDGET_TOKENS


def test_independent_ceilings_do_not_interfere() -> None:
    """Spending all tokens does not falsely exhaust the turn ceiling."""
    meter = BudgetMeter(BudgetLimits(max_turns=80, max_tokens=1000))
    meter.debit_usage(tokens=1000, cost_usd=0.0)  # exactly at token ceiling
    assert meter.is_exhausted()  # token kind is at limit
    # But the turn ceiling is untouched.
    assert meter.used("turn") == 0.0


def test_snapshot_reflects_usage() -> None:
    clock = _FakeClock()
    meter = BudgetMeter(BudgetLimits(), clock=clock)
    meter.debit_turn()
    meter.debit_usage(tokens=120, cost_usd=0.5)
    meter.debit_oracle_query()
    clock.t = 7.0
    snap = meter.snapshot()
    assert snap.turns == 1
    assert snap.tokens == 120
    assert snap.cost_usd == pytest.approx(0.5)
    assert snap.oracle_queries == 1
    assert snap.wall_s == pytest.approx(7.0)


def test_negative_and_unknown_debits_rejected() -> None:
    meter = BudgetMeter(BudgetLimits())
    with pytest.raises(ValueError):
        meter.debit("turn", -1.0)
    with pytest.raises(ValueError):
        meter.debit("does_not_exist", 1.0)


def test_debit_returns_record() -> None:
    meter = BudgetMeter(BudgetLimits())
    rec = meter.debit_turn(reason="agent_action")
    assert rec.kind == "turn"
    assert rec.amount == 1.0
    assert rec.reason == "agent_action"
