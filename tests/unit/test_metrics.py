"""The A-G metric registry, the geometric composite, and the pass^k estimator.

Each test hand-assembles a precise episode via the :class:`TraceBuilder` fixture so
exact metric values can be asserted (the registry functions are PURE offline
functions of ``trace.jsonl`` + frozen gold, ``DESIGN.md`` invariant 4). The
severity weights come from ``usability_score.yaml`` (``w=[0,1,3,6,12,25]``) -- the
single source of truth -- so the assistance-cost assertions also pin the spec.
"""

from __future__ import annotations

import math

import pytest

# Type alias only for readability in the helper below.
from tests.conftest import TraceBuilder  # noqa: E402
from usabench.core.enums import Actor, QueryClass, Verdict
from usabench.core.schema import (
    AgentMessage,
    Checkpoint,
    CodeRun,
    CriterionResult,
    EpisodeEnd,
    EpisodeStart,
    FileEdit,
    MessageToUser,
    OracleQuery,
    OracleResponse,
    VerificationRun,
)
from usabench.eval import metrics
from usabench.eval.aggregate import pass_at_k, pass_hat_k
from usabench.eval.composite import (
    CompositeInputs,
    compute_composite,
    geometric_usability,
)
from usabench.eval.spec import get_severity_weights

# --------------------------------------------------------------------------- #
# Episode fixtures                                                             #
# --------------------------------------------------------------------------- #


def _episode_solved_with_help(run_id: str = "r-help") -> TraceBuilder:
    """A solved episode that needed a clarification (sev1) and a hint (sev2).

    Hand-built so the assistance metrics are exactly computable:
      * 1 clarification query answered at severity 1,
      * 1 hint query answered at severity 2 (reveals info unit AC2),
      * 1 bare-accept review at severity 0.
    The verification meets all three criteria (weighted score 1.0).
    """
    tb = TraceBuilder(run_id=run_id)
    tb.add(Actor.HARNESS, EpisodeStart(task_id="ub-cal-0007", hidden_spec_sha256="0" * 64, seed=7), t_turn=None)
    tb.add(Actor.AGENT, AgentMessage(text="planning"), t_turn=0)

    # Turn 1: a clarification (answered sev 1).
    tb.add(Actor.AGENT, OracleQuery(query_class=QueryClass.CLARIFICATION, text="all-day events?"), t_turn=1)
    tb.add(
        Actor.ORACLE,
        OracleResponse(responds_to="e2", severity=1, text="count all-day as 0h", verdict=Verdict.NA),
        t_turn=1,
    )

    tb.add(Actor.AGENT, FileEdit(path="main.py", op="create", added=30, removed=0, loc_after=30), t_turn=2)
    tb.add(Actor.AGENT, CodeRun(cmd="pytest", exit_code=1, is_test=True, self_test_passed=False), t_turn=3)

    # Turn 4: a hint (answered sev 2, reveals AC2).
    tb.add(Actor.AGENT, OracleQuery(query_class=QueryClass.HINT_REQUEST, text="why failing?"), t_turn=4)
    tb.add(
        Actor.ORACLE,
        OracleResponse(
            responds_to="e6",
            severity=2,
            text="you missed weekly aggregation",
            verdict=Verdict.NA,
            info_units_revealed=["AC2"],
            cited_criteria=["AC2"],
        ),
        t_turn=4,
    )

    tb.add(Actor.AGENT, FileEdit(path="main.py", op="modify", added=10, removed=2, loc_after=38), t_turn=5)
    tb.add(Actor.AGENT, CodeRun(cmd="pytest", exit_code=0, is_test=True, self_test_passed=True), t_turn=6)
    tb.add(
        Actor.HARNESS,
        Checkpoint(weighted_score=1.0, is_working_version=True, criteria_state={"AC1": True, "AC2": True, "AC3": True}),
        t_turn=6,
    )
    tb.add(Actor.AGENT, MessageToUser(text="It works; here is a demo of the table."), t_turn=7)
    tb.add(
        Actor.HARNESS,
        VerificationRun(
            trigger="submit",
            entrypoint="python main.py",
            must_have=[CriterionResult(id="AC1", passed=True), CriterionResult(id="AC2", passed=True)],
            should_have=[CriterionResult(id="AC3", score=1.0)],
            all_must_pass=True,
            rubric_score=1.0,
        ),
        t_turn=7,
    )
    tb.add(Actor.ORACLE, OracleResponse(responds_to=None, severity=0, text="accept", verdict=Verdict.ACCEPT), t_turn=7)
    tb.add(
        Actor.HARNESS,
        EpisodeEnd(terminated_reason="accept", accepted=True, final_weighted_score=1.0, interventions_by_severity={"0": 1, "1": 1, "2": 1}),
        t_turn=None,
    )
    return tb


# --------------------------------------------------------------------------- #
# Dimension A -- goal achievement                                             #
# --------------------------------------------------------------------------- #


def test_A_dimension_on_solved_episode(sample_gold) -> None:  # type: ignore[no-untyped-def]
    trace = _episode_solved_with_help().events
    assert metrics.A1_success_binary(trace, sample_gold) == 1
    assert metrics.A2_criteria_score(trace, sample_gold) == pytest.approx(1.0)
    assert metrics.A3_core_criteria_score(trace, sample_gold) == pytest.approx(1.0)
    assert metrics.A4_goal_drift(trace, sample_gold) == pytest.approx(0.0)
    assert metrics.A5_regression_free(trace, sample_gold) == pytest.approx(1.0)


def test_A2_partial_credit_when_a_core_criterion_unmet(sample_gold) -> None:  # type: ignore[no-untyped-def]
    """Failing AC2 (weight 2 of total 5) drops A2 and A3 below 1."""
    tb = TraceBuilder(run_id="r-partial")
    tb.add(Actor.HARNESS, EpisodeStart(task_id="ub-cal-0007", hidden_spec_sha256="0" * 64), t_turn=None)
    tb.add(
        Actor.HARNESS,
        VerificationRun(
            trigger="forced_final",
            must_have=[CriterionResult(id="AC1", passed=True), CriterionResult(id="AC2", passed=False)],
            should_have=[CriterionResult(id="AC3", score=1.0)],
            all_must_pass=False,
            rubric_score=0.6,
        ),
        t_turn=1,
    )
    tb.add(Actor.HARNESS, EpisodeEnd(terminated_reason="budget_turns", accepted=False), t_turn=None)
    trace = tb.events
    # Total weight = AC1(2)+AC2(2)+AC3(1) = 5; met = AC1(2)+AC3(1) = 3 -> 3/5.
    assert metrics.A2_criteria_score(trace, sample_gold) == pytest.approx(3.0 / 5.0)
    # Core weight = AC1(2)+AC2(2) = 4; met core = AC1(2) -> 2/4.
    assert metrics.A3_core_criteria_score(trace, sample_gold) == pytest.approx(2.0 / 4.0)
    assert metrics.A1_success_binary(trace, sample_gold) == 0


# --------------------------------------------------------------------------- #
# Dimension B -- interaction load                                            #
# --------------------------------------------------------------------------- #


def test_B_dimension_counts(sample_gold) -> None:  # type: ignore[no-untyped-def]
    trace = _episode_solved_with_help().events
    assert metrics.B1_n_interventions(trace, sample_gold) == 3  # 3 oracle responses
    assert metrics.B2_n_clarifications(trace, sample_gold) == 1
    assert metrics.B3_n_hint_requests(trace, sample_gold) == 1
    assert metrics.B4_n_corrections(trace, sample_gold) == 1  # the unsolicited accept review
    assert metrics.B5_n_handoffs(trace, sample_gold) == 0
    # First working checkpoint is at turn 6.
    assert metrics.B6_turns_to_first_working(trace, sample_gold) == pytest.approx(6.0)


def test_B9_query_class_entropy(sample_gold) -> None:  # type: ignore[no-untyped-def]
    """Two queries of distinct classes -> entropy = ln(2) nats."""
    trace = _episode_solved_with_help().events
    assert metrics.B9_mean_query_class_entropy(trace, sample_gold) == pytest.approx(math.log(2))


# --------------------------------------------------------------------------- #
# Dimension C -- assistance amount & severity (the spec-pinned core)          #
# --------------------------------------------------------------------------- #


def test_C1_assistance_cost_uses_spec_weights(sample_gold) -> None:  # type: ignore[no-untyped-def]
    """AC = w[1] + w[2] + w[0] with canonical convex weights [0,1,3,6,12,25]."""
    weights = get_severity_weights()
    assert weights == [0, 1, 3, 6, 12, 25]
    trace = _episode_solved_with_help().events
    expected = weights[1] + weights[2] + weights[0]  # 1 + 3 + 0 = 4
    assert metrics.C1_assistance_cost(trace, sample_gold) == pytest.approx(float(expected))


def test_C1_convexity_one_severe_dominates_many_mild(sample_gold) -> None:  # type: ignore[no-untyped-def]
    """A single sev-5 (25) outweighs ten sev-1 (10) -- convex weighting."""
    weights = get_severity_weights()

    def _one(sev: int) -> TraceBuilder:
        tb = TraceBuilder()
        tb.add(Actor.HARNESS, EpisodeStart(task_id="t", hidden_spec_sha256="0" * 64), t_turn=None)
        tb.add(Actor.ORACLE, OracleResponse(responds_to=None, severity=sev, text="x"), t_turn=0)  # type: ignore[arg-type]
        tb.add(Actor.HARNESS, EpisodeEnd(terminated_reason="accept"), t_turn=None)
        return tb

    severe = metrics.C1_assistance_cost(_one(5).events, sample_gold)
    assert severe == pytest.approx(float(weights[5]))  # 25
    assert severe > 10 * float(weights[1])  # 25 > 10


def test_C2_C3_severity_histogram(sample_gold) -> None:  # type: ignore[no-untyped-def]
    trace = _episode_solved_with_help().events
    assert metrics.C2_max_severity(trace, sample_gold) == 2
    assert metrics.C3_severity_histogram(trace, sample_gold) == [1, 1, 1, 0, 0, 0]


def test_C4_spec_info_transferred_normalised(sample_gold) -> None:  # type: ignore[no-untyped-def]
    """Only AC2 (a requirement) was revealed; normalised by 3 hidden units."""
    trace = _episode_solved_with_help().events
    # 1 graded (requirement) info unit revealed / 3 hidden units.
    assert metrics.C4_spec_info_transferred(trace, sample_gold) == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------- #
# Dimension D / E / G spot checks                                            #
# --------------------------------------------------------------------------- #


def test_D3_self_recovery(sample_gold) -> None:  # type: ignore[no-untyped-def]
    """A test failure later fixed within the window counts as a recovery."""
    trace = _episode_solved_with_help().events
    # One detected failure (turn 3); recovered (turn 6) WITH a sev2 hint between ->
    # NOT an unaided recovery, so the rate is 0/1 = 0.0.
    assert metrics.D3_self_recovery_rate(trace, sample_gold) == pytest.approx(0.0)


def test_E_efficiency_counts(sample_gold) -> None:  # type: ignore[no-untyped-def]
    trace = _episode_solved_with_help().events
    assert metrics.E4_n_tool_calls(trace, sample_gold) == 0  # no tool_call events here
    # Edit churn = (30+0) + (10+2) = 42.
    assert metrics.E5_edit_churn(trace, sample_gold) == 42
    # Two code_run -> file_edit cycles? runs at t3,t6; edits at t2,t5. Order:
    # edit(t2), run(t3), edit(t5), run(t6) -> one run(t3)->edit(t5) cycle.
    assert metrics.E6_iterations(trace, sample_gold) == 1


def test_G2_redundant_query_rate(sample_gold) -> None:  # type: ignore[no-untyped-def]
    """A literally-repeated query is flagged redundant."""
    tb = TraceBuilder()
    tb.add(Actor.HARNESS, EpisodeStart(task_id="t", hidden_spec_sha256="0" * 64), t_turn=None)
    tb.add(Actor.AGENT, OracleQuery(query_class=QueryClass.CLARIFICATION, text="Which format?"), t_turn=0)
    tb.add(Actor.AGENT, OracleQuery(query_class=QueryClass.CLARIFICATION, text="which format?"), t_turn=1)
    tb.add(Actor.HARNESS, EpisodeEnd(terminated_reason="accept"), t_turn=None)
    assert metrics.G2_redundant_query_rate(tb.events, sample_gold) == pytest.approx(0.5)


def test_G4_false_confidence_on_unaccepted_completion_claim(sample_gold) -> None:  # type: ignore[no-untyped-def]
    """Claiming 'done' while not accepted is a false-confidence event."""
    tb = TraceBuilder()
    tb.add(Actor.HARNESS, EpisodeStart(task_id="t", hidden_spec_sha256="0" * 64), t_turn=None)
    tb.add(Actor.AGENT, MessageToUser(text="All done, the task is complete!"), t_turn=0)
    tb.add(
        Actor.HARNESS,
        VerificationRun(must_have=[CriterionResult(id="AC1", passed=False)], all_must_pass=False, rubric_score=0.0),
        t_turn=1,
    )
    tb.add(Actor.HARNESS, EpisodeEnd(terminated_reason="give_up", accepted=False), t_turn=None)
    assert metrics.G4_false_confidence(tb.events, sample_gold) == pytest.approx(1.0)


def test_compute_all_covers_full_registry(sample_gold) -> None:  # type: ignore[no-untyped-def]
    """compute_all returns one value for every registry id."""
    trace = _episode_solved_with_help().events
    out = metrics.compute_all(trace, sample_gold)
    assert set(out.keys()) == set(metrics.registry().keys())
    # Spot-check a couple of representative values.
    assert out["A1_success_binary"] == 1
    assert out["C1_assistance_cost"] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# Geometric composite                                                         #
# --------------------------------------------------------------------------- #


def test_geometric_zero_factor_kills_score() -> None:
    """Either S or H -> 0 drives the geometric headline to 0."""
    assert geometric_usability(0.0, 1.0, 1.0, 1.0) == pytest.approx(0.0)
    assert geometric_usability(1.0, 0.0, 1.0, 1.0) == pytest.approx(0.0)


def test_geometric_all_ones_is_one() -> None:
    assert geometric_usability(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)


def test_geometric_matches_closed_form() -> None:
    """The headline equals its weighted-geometric-mean closed form for known inputs."""
    from usabench.eval._common import spec_get

    alpha = float(spec_get("composite", "alpha"))
    beta = float(spec_get("composite", "beta"))
    gamma = float(spec_get("composite", "gamma"))
    delta = float(spec_get("composite", "delta"))
    s, h, e, r = 0.8, 0.5, 0.9, 0.7
    core = math.exp((alpha * math.log(s) + beta * math.log(h)) / (alpha + beta))
    expected = core * (e ** gamma) * (r ** delta)
    assert geometric_usability(s, h, e, r) == pytest.approx(expected)


def test_compute_composite_under_ask_penalty_fires() -> None:
    """A failed run that asked nothing AND drifted gets its H penalised."""
    # success_binary=0, n_clarifications=0, goal_drift > tau (0.5 default) -> fire.
    penalised = compute_composite(
        CompositeInputs(
            s_core=0.0,
            assistance_cost=0.0,  # zero help -> H would be 1.0
            success_binary=0,
            n_clarifications=0,
            goal_drift=0.9,
        )
    )
    assert penalised.under_ask_penalised is True
    # H was knocked below 1 by the rho haircut.
    assert penalised.h < 1.0

    not_penalised = compute_composite(
        CompositeInputs(
            s_core=0.0,
            assistance_cost=0.0,
            success_binary=0,
            n_clarifications=1,  # it DID ask -> no trap
            goal_drift=0.9,
        )
    )
    assert not_penalised.under_ask_penalised is False
    assert not_penalised.h == pytest.approx(1.0)


def test_compute_composite_reports_three_variants() -> None:
    res = compute_composite(CompositeInputs(s_core=0.9, assistance_cost=0.0, autonomy=0.8, robustness=1.0))
    # All three composites are populated and in [0,1].
    for v in (res.usability_geometric, res.usability_multiplicative, res.usability_linear):
        assert 0.0 <= v <= 1.0


# --------------------------------------------------------------------------- #
# pass^k estimator -- exact C(c,k)/C(n,k)                                     #
# --------------------------------------------------------------------------- #


def test_pass_hat_k_exact_combinatorics() -> None:
    """pass^k = C(c,k)/C(n,k): probability ALL k of a random k-subset succeed."""
    # c=3 successes of n=5, k=2 -> C(3,2)/C(5,2) = 3/10.
    assert pass_hat_k(3, 5, 2) == pytest.approx(3.0 / 10.0)
    # c=4 of n=5, k=2 -> C(4,2)/C(5,2) = 6/10.
    assert pass_hat_k(4, 5, 2) == pytest.approx(6.0 / 10.0)
    # c=2 of n=4, k=2 -> C(2,2)/C(4,2) = 1/6.
    assert pass_hat_k(2, 4, 2) == pytest.approx(1.0 / 6.0)


def test_pass_hat_k_boundaries() -> None:
    assert pass_hat_k(5, 5, 5) == pytest.approx(1.0)  # all succeed
    assert pass_hat_k(1, 5, 2) == pytest.approx(0.0)  # c<k -> cannot draw k successes
    assert pass_hat_k(3, 3, 4) == pytest.approx(0.0)  # k>n -> 0
    assert pass_hat_k(0, 0, 1) == pytest.approx(0.0)  # empty


def test_pass_at_k_is_the_anyofk_estimator() -> None:
    """pass@k = 1 - C(n-c,k)/C(n,k): probability >=1 of a k-subset succeeds."""
    # c=1 of n=5, k=2 -> 1 - C(4,2)/C(5,2) = 1 - 6/10 = 0.4.
    assert pass_at_k(1, 5, 2) == pytest.approx(0.4)
    # All fail -> 0; all succeed -> 1.
    assert pass_at_k(0, 5, 2) == pytest.approx(0.0)
    assert pass_at_k(5, 5, 2) == pytest.approx(1.0)


def test_pass_hat_k_le_pass_at_k() -> None:
    """The strict all-of-k estimator never exceeds the any-of-k estimator."""
    for c in range(0, 6):
        assert pass_hat_k(c, 5, 2) <= pass_at_k(c, 5, 2) + 1e-12
