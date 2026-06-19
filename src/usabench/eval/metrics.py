"""The A-G metric registry (``docs/metrics.md``) as pure offline functions.

Every public function here is named ``<ID>_<snake_name>`` to match the registry
IDs in ``docs/metrics.md`` **exactly** (e.g. :func:`A1_success_binary`,
:func:`G4_false_confidence`). Each takes ``(trace, gold)`` where:

* ``trace`` is an iterable of :class:`~usabench.core.schema.TraceEnvelope`
  (the parsed lines of the ONE canonical artifact ``trace.jsonl``), and
* ``gold`` is anything :func:`usabench.eval.gold.as_gold` accepts (a
  :class:`~usabench.core.schema.Task`, :class:`~usabench.core.schema.HiddenSpec`,
  ``dict``, or :class:`~usabench.eval.gold.Gold`).

All functions are **pure**: same inputs -> same output, no I/O, no globals beyond
the cached spec. All numeric constants (severity weights, ``epsilon``,
``accept_threshold``) come from ``usability_score.yaml`` via
:mod:`usabench.eval.spec` -- nothing is hardcoded.

:func:`compute_all` returns the full ``{metric_id: value}`` dict.

Where a metric needs a signal the canonical trace cannot fully carry offline
(e.g. solution-provenance spans for C5, embedding cosine for A4), we compute a
correct, well-defined *proxy* from the logged fields and mark it with a ``TODO``
in the docstring; the function still imports and runs and is unit-testable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from usabench.core.enums import QueryClass, Severity
from usabench.core.schema import TraceEnvelope
from usabench.eval._common import (
    as_value,
    checkpoints,
    clip01,
    code_runs,
    episode_end,
    event_actor,
    event_type,
    final_acceptance,
    iter_events,
    messages_to_user,
    oracle_queries,
    oracle_responses,
    payload_severity,
    safe_div,
    spec_get,
    verification_runs,
)
from usabench.eval.gold import Gold, as_gold

__all__ = [
    # Dimension A -- goal achievement
    "A1_success_binary",
    "A2_criteria_score",
    "A3_core_criteria_score",
    "A4_goal_drift",
    "A5_regression_free",
    # Dimension B -- interaction load
    "B1_n_interventions",
    "B2_n_clarifications",
    "B3_n_hint_requests",
    "B4_n_corrections",
    "B5_n_handoffs",
    "B6_turns_to_first_working",
    "B7_turns_to_acceptance",
    "B8_interventions_to_acceptance",
    "B9_mean_query_class_entropy",
    "B10_time_blocked_fraction",
    # Dimension C -- assistance amount & severity
    "C1_assistance_cost",
    "C2_max_severity",
    "C3_severity_histogram",
    "C4_spec_info_transferred",
    "C5_solution_leakage",
    "C6_assistance_efficiency",
    # Dimension D -- autonomy
    "D1_autonomy_ratio",
    "D2_unaided_progress_fraction",
    "D3_self_recovery_rate",
    "D4_blocked_resolution_self",
    "D5_proactive_inference",
    # Dimension E -- efficiency
    "E1_wall_clock_s",
    "E2_tokens_total",
    "E3_cost_usd_total",
    "E4_n_tool_calls",
    "E5_edit_churn",
    "E6_iterations",
    "E7_cost_per_progress",
    "E8_tokens_per_progress",
    # Dimension F -- robustness (computed at aggregate level; see aggregate.py)
    # Dimension G -- agent UX quality
    "G1_question_quality",
    "G2_redundant_query_rate",
    "G3_status_transparency",
    "G4_false_confidence",
    # rollup
    "compute_all",
]

# Number of follow-up turns within which a self-recovery may occur (D3).
_SELF_RECOVERY_WINDOW_TURNS = 5


# --------------------------------------------------------------------------- #
# Internal helpers (acceptance reconstruction from the trace)                  #
# --------------------------------------------------------------------------- #


def _last_verification(trace: list[TraceEnvelope]) -> Any:
    """Return the payload of the last ``verification_run`` event, or ``None``."""
    runs = verification_runs(trace)
    return runs[-1].payload if runs else None


def _criterion_result_map(payload: Any) -> dict[str, Any]:
    """Map criterion id -> CriterionResult from a verification_run payload."""
    out: dict[str, Any] = {}
    if payload is None:
        return out
    for bucket in ("must_have", "should_have"):
        for cr in getattr(payload, bucket, []) or []:
            out[cr.id] = cr
    return out


def _criterion_value(cr: Any) -> float:
    """Score a single CriterionResult in ``[0,1]`` (``score`` else ``passed``)."""
    if cr is None:
        return 0.0
    score = getattr(cr, "score", None)
    if score is not None:
        return clip01(float(score))
    passed = getattr(cr, "passed", None)
    return 1.0 if passed else 0.0


def _weighted_over(criteria: list[Any], results: dict[str, Any]) -> float:
    """Weighted fraction met over ``criteria`` using ``results`` (0 if empty)."""
    total_w = sum(float(c.weight) for c in criteria)
    if total_w <= 0:
        return 0.0
    acc = 0.0
    for c in criteria:
        acc += float(c.weight) * _criterion_value(results.get(c.id))
    return clip01(acc / total_w)


def _acceptance_view(trace: list[TraceEnvelope], gold: Gold) -> dict[str, Any]:
    """Reconstruct A1/A2/A3/hard_pass_frac from the trace + gold.

    Prefers per-criterion results from the last ``verification_run``; falls back
    to the cached ``final_acceptance`` / ``episode_end`` weighted score when no
    per-criterion detail is present.
    """
    payload = _last_verification(trace)
    results = _criterion_result_map(payload)
    criteria = gold.criteria

    if criteria and results:
        weighted = _weighted_over(criteria, results)
        core = _weighted_over(gold.core_criteria, results) if gold.core_criteria else weighted
        hard = gold.hard_criteria
        if hard:
            hard_frac = sum(
                1.0 for c in hard if _criterion_value(results.get(c.id)) >= 1.0
            ) / len(hard)
        else:
            hard_frac = 1.0
    else:
        # Fallback: use cached scalar totals; no per-criterion detail available.
        weighted = _cached_weighted_score(trace)
        core = weighted
        hard_frac = 1.0 if weighted >= gold.accept_threshold else 0.0

    accepted = _cached_accepted(trace)
    if accepted is None:
        accepted = weighted >= gold.accept_threshold

    return {
        "weighted_score": weighted,
        "core_criteria_score": core,
        "hard_pass_frac": clip01(hard_frac),
        "accepted": bool(accepted),
    }


def _cached_weighted_score(trace: list[TraceEnvelope]) -> float:
    """Best cached weighted score from final_acceptance/episode_end/verification."""
    fa = final_acceptance(trace)
    if fa is not None:
        return clip01(float(getattr(fa.payload, "weighted_score", 0.0)))
    ee = episode_end(trace)
    if ee is not None:
        fws = getattr(ee.payload, "final_weighted_score", None)
        if fws is not None:
            return clip01(float(fws))
    payload = _last_verification(trace)
    if payload is not None:
        return clip01(float(getattr(payload, "rubric_score", 0.0)))
    return 0.0


def _cached_accepted(trace: list[TraceEnvelope]) -> Any:
    """Cached ``accepted`` flag from final_acceptance/episode_end, or ``None``."""
    fa = final_acceptance(trace)
    if fa is not None:
        return bool(getattr(fa.payload, "accepted", False))
    ee = episode_end(trace)
    if ee is not None:
        return bool(getattr(ee.payload, "accepted", False))
    return None


# --------------------------------------------------------------------------- #
# Dimension A -- goal achievement / problem-solving success                    #
# --------------------------------------------------------------------------- #


def A1_success_binary(trace: Iterable[TraceEnvelope], gold: Any) -> int:
    """A1 ``success_binary``: 1 iff the run's final acceptance accepted.

    Args:
        trace: Parsed ``trace.jsonl`` events.
        gold: Task gold (see :func:`usabench.eval.gold.as_gold`).

    Returns:
        ``1`` if accepted, else ``0``.
    """
    g = as_gold(gold)
    return 1 if _acceptance_view(iter_events(trace), g)["accepted"] else 0


def A2_criteria_score(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """A2 ``criteria_score``: weighted fraction of all acceptance criteria met."""
    g = as_gold(gold)
    return float(_acceptance_view(iter_events(trace), g)["weighted_score"])


def A3_core_criteria_score(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """A3 ``core_criteria_score``: A2 restricted to ``is_core`` criteria.

    This is the capped success signal ``S`` that feeds the headline composite --
    the main anti-stuffing lever (peripheral criteria can lift A2 but not A3).
    """
    g = as_gold(gold)
    return float(_acceptance_view(iter_events(trace), g)["core_criteria_score"])


def A4_goal_drift(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """A4 ``goal_drift``: ``1 - similarity(delivered, gold_goal)`` in ``[0,1]``.

    The registry defines this as an oracle-judged cosine distance between the
    delivered artifact's capability summary and the gold goal embedding. That
    judgment is produced by V3 and logged as an ``oracle_judgment`` criterion;
    offline we use the best available proxy: ``1 - A2`` (the unmet-fraction of
    the weighted criteria), which is monotone in true drift and bounded in
    ``[0,1]``.

    TODO: when a dedicated ``goal_alignment`` judge criterion is present in the
    final ``verification_run`` (channel ``oracle_judgment``), prefer its score.
    """
    g = as_gold(gold)
    trace_l = iter_events(trace)
    payload = _last_verification(trace_l)
    results = _criterion_result_map(payload)
    # Prefer an explicit alignment criterion if the task defines one.
    for cid in ("goal_alignment", "alignment", "A4", "goal_drift"):
        if cid in results:
            return clip01(1.0 - _criterion_value(results[cid]))
    return clip01(1.0 - _acceptance_view(trace_l, g)["weighted_score"])


def A5_regression_free(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """A5 ``regression_free``: fraction of once-passing checks still passing.

    Operationalised over the checkpoint history: for every criterion id that was
    ever observed ``passed`` in any checkpoint's ``criteria_state``, the fraction
    that is still passing in the final checkpoint. Returns ``1.0`` if no
    regression signal is available (nothing ever passed to regress).
    """
    as_gold(gold)  # validates shape; A5 needs only the trace
    cps = checkpoints(iter_events(trace))
    if not cps:
        return 1.0
    ever_passed: set[str] = set()
    for cp in cps:
        state = getattr(cp.payload, "criteria_state", {}) or {}
        for cid, val in state.items():
            if _truthy_pass(val):
                ever_passed.add(cid)
    if not ever_passed:
        return 1.0
    final_state = getattr(cps[-1].payload, "criteria_state", {}) or {}
    still = sum(1 for cid in ever_passed if _truthy_pass(final_state.get(cid)))
    return clip01(safe_div(float(still), float(len(ever_passed)), default=1.0))


def _truthy_pass(val: Any) -> bool:
    """Interpret a checkpoint criteria_state value as pass/fail."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val >= 1.0
    if isinstance(val, dict):
        if "passed" in val:
            return bool(val["passed"])
        if "score" in val:
            return float(val["score"]) >= 1.0
    if isinstance(val, str):
        return val.lower() in {"pass", "passed", "true", "ok"}
    return False


# --------------------------------------------------------------------------- #
# Dimension B -- interaction / intervention load                               #
# --------------------------------------------------------------------------- #


def B1_n_interventions(trace: Iterable[TraceEnvelope], gold: Any) -> int:
    """B1 ``n_interventions``: count of ``oracle_response`` events."""
    return len(oracle_responses(iter_events(trace)))


def _query_class(payload: Any) -> str:
    """Return an oracle_query's class as a string value."""
    qc = getattr(payload, "query_class", None)
    return str(as_value(qc)) if qc is not None else ""


def B2_n_clarifications(trace: Iterable[TraceEnvelope], gold: Any) -> int:
    """B2 ``n_clarifications``: ``oracle_query`` with class=clarification."""
    return sum(
        1
        for ev in oracle_queries(iter_events(trace))
        if _query_class(ev.payload) == QueryClass.CLARIFICATION.value
    )


def B3_n_hint_requests(trace: Iterable[TraceEnvelope], gold: Any) -> int:
    """B3 ``n_hint_requests``: ``oracle_query`` with class=hint_request."""
    return sum(
        1
        for ev in oracle_queries(iter_events(trace))
        if _query_class(ev.payload) == QueryClass.HINT_REQUEST.value
    )


def B4_n_corrections(trace: Iterable[TraceEnvelope], gold: Any) -> int:
    """B4 ``n_corrections``: UNSOLICITED ``oracle_response`` (``responds_to`` null)."""
    return sum(
        1
        for ev in oracle_responses(iter_events(trace))
        if getattr(ev.payload, "responds_to", None) is None
    )


def B5_n_handoffs(trace: Iterable[TraceEnvelope], gold: Any) -> int:
    """B5 ``n_handoffs``: handoff-class queries + any oracle-takeover termination."""
    trace_l = iter_events(trace)
    n = sum(
        1
        for ev in oracle_queries(trace_l)
        if _query_class(ev.payload) == QueryClass.HANDOFF_REQUEST.value
    )
    ee = episode_end(trace_l)
    if ee is not None:
        reason = str(as_value(getattr(ee.payload, "terminated_reason", "")))
        if reason == "oracle_takeover":
            n += 1
    return n


def B6_turns_to_first_working(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """B6 ``turns_to_first_working``: turn of first working checkpoint (inf if none)."""
    for cp in checkpoints(iter_events(trace)):
        if getattr(cp.payload, "is_working_version", False):
            return float(cp.t_turn) if cp.t_turn is not None else 0.0
    return math.inf


def B7_turns_to_acceptance(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """B7 ``turns_to_acceptance``: turn of first checkpoint at/above threshold."""
    g = as_gold(gold)
    thr = g.accept_threshold
    for cp in checkpoints(iter_events(trace)):
        if float(getattr(cp.payload, "weighted_score", 0.0)) >= thr:
            return float(cp.t_turn) if cp.t_turn is not None else 0.0
    return math.inf


def B8_interventions_to_acceptance(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """B8 ``interventions_to_acceptance``: # oracle_responses before acceptance.

    Counts ``oracle_response`` events with ``seq`` before the first accepting
    checkpoint. Returns the total intervention count if acceptance never occurs.
    """
    g = as_gold(gold)
    trace_l = iter_events(trace)
    thr = g.accept_threshold
    accept_seq = None
    for cp in checkpoints(trace_l):
        if float(getattr(cp.payload, "weighted_score", 0.0)) >= thr:
            accept_seq = cp.seq
            break
    resp = oracle_responses(trace_l)
    if accept_seq is None:
        return float(len(resp))
    return float(sum(1 for ev in resp if ev.seq < accept_seq))


def B9_mean_query_class_entropy(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """B9 ``mean_query_class_entropy``: Shannon entropy (nats) of query classes."""
    counts: dict[str, int] = {}
    for ev in oracle_queries(iter_events(trace)):
        qc = _query_class(ev.payload)
        counts[qc] = counts.get(qc, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p)
    return ent


def B10_time_blocked_fraction(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """B10 ``time_blocked_fraction``: wall-clock in blocked segments / total.

    A blocked segment opens at an ``agent_blocked`` (``blocked=True``) event and
    closes at the next ``oracle_response`` or ``agent_blocked(blocked=False)`` (or
    episode end). Fraction is the summed blocked duration over total wall time.
    """
    trace_l = iter_events(trace)
    if len(trace_l) < 2:
        return 0.0
    t0 = trace_l[0].ts
    t1 = trace_l[-1].ts
    total = t1 - t0
    if total <= 0:
        return 0.0
    blocked = 0.0
    open_ts: float | None = None
    for ev in trace_l:
        etype = event_type(ev)
        if etype == "agent_blocked":
            if getattr(ev.payload, "blocked", True):
                if open_ts is None:
                    open_ts = ev.ts
            else:
                if open_ts is not None:
                    blocked += ev.ts - open_ts
                    open_ts = None
        elif etype == "oracle_response" and open_ts is not None:
            blocked += ev.ts - open_ts
            open_ts = None
    if open_ts is not None:
        blocked += t1 - open_ts
    return clip01(blocked / total)


# --------------------------------------------------------------------------- #
# Dimension C -- human-assistance amount & severity                            #
# --------------------------------------------------------------------------- #


def C1_assistance_cost(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """C1 ``assistance_cost`` (AC): convex-weighted sum over oracle severities.

    ``AC = Σ_responses w[sev]`` with the canonical convex weights
    ``w = [0,1,3,6,12,25]`` read from ``usability_score.yaml`` (NOT hardcoded).
    Convexity means one sev-5 (25) dominates many sev-1 (1).
    """
    weights = [float(w) for w in spec_get("severity_weights", default=[0, 1, 3, 6, 12, 25])]
    total = 0.0
    for ev in oracle_responses(iter_events(trace)):
        sev = payload_severity(ev.payload)
        if 0 <= sev < len(weights):
            total += weights[sev]
    return total


def C2_max_severity(trace: Iterable[TraceEnvelope], gold: Any) -> int:
    """C2 ``max_severity``: the maximum severity reached in the episode."""
    sevs = [payload_severity(ev.payload) for ev in oracle_responses(iter_events(trace))]
    return max(sevs) if sevs else 0


def C3_severity_histogram(trace: Iterable[TraceEnvelope], gold: Any) -> list[int]:
    """C3 ``severity_histogram``: counts per severity level 0..5."""
    hist = [0] * 6
    for ev in oracle_responses(iter_events(trace)):
        sev = payload_severity(ev.payload)
        if 0 <= sev <= 5:
            hist[sev] += 1
    return hist


def C4_spec_info_transferred(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """C4 ``spec_info_transferred``: distinct requirement/constraint units the
    oracle had to reveal, normalised by ``n_hidden_spec_units``.

    Only counts info units whose class is ``requirement`` or ``constraint`` (a
    revealed *preference* is not the same failure-to-infer). Normalised to
    ``[0,1]`` by the task's hidden-spec unit count.
    """
    g = as_gold(gold)
    graded_ids = {
        u.id
        for u in g.info_units
        if str(as_value(u.klass)) in {"requirement", "constraint"}
    }
    revealed: set[str] = set()
    for ev in oracle_responses(iter_events(trace)):
        for uid in getattr(ev.payload, "info_units_revealed", []) or []:
            if uid in graded_ids:
                revealed.add(uid)
    denom = g.n_hidden_spec_units
    if denom <= 0:
        return 0.0
    return clip01(len(revealed) / denom)


def C5_solution_leakage(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """C5 ``solution_leakage``: fraction of accepted-solution components whose
    origin traces to a sev>=4 oracle response (provenance tagging).

    Offline proxy: among the *met* acceptance criteria, the fraction whose id is
    ``cited`` by, or whose info unit is ``revealed`` by, a sev>=4 oracle response
    (``provenance_tag`` present). Bounded in ``[0,1]``. If no criterion-level
    provenance is recoverable, falls back to the share of high-severity responses
    among all responses (a conservative upper proxy).

    TODO: replace the fallback with true code-span provenance once the harness
    tags oracle-contributed spans in ``file_edit`` payloads.
    """
    as_gold(gold)
    trace_l = iter_events(trace)
    payload = _last_verification(trace_l)
    results = _criterion_result_map(payload)
    met_ids = {cid for cid, cr in results.items() if _criterion_value(cr) >= 1.0}

    leaked_targets: set[str] = set()
    high_sev_responses = 0
    total_responses = 0
    for ev in oracle_responses(trace_l):
        total_responses += 1
        sev = payload_severity(ev.payload)
        if sev >= int(Severity.PARTIAL_SOLUTION):
            high_sev_responses += 1
            for cid in getattr(ev.payload, "cited_criteria", []) or []:
                leaked_targets.add(cid)
            for uid in getattr(ev.payload, "info_units_revealed", []) or []:
                leaked_targets.add(uid)

    if met_ids and (leaked_targets & met_ids):
        return clip01(len(leaked_targets & met_ids) / len(met_ids))
    if met_ids and not leaked_targets:
        return 0.0
    # Fallback: high-severity share of interventions (conservative).
    return clip01(safe_div(float(high_sev_responses), float(total_responses), default=0.0))


def C6_assistance_efficiency(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """C6 ``assistance_efficiency``: A2 gained per unit AC = ``A2 / (AC + 1)``."""
    g = as_gold(gold)
    trace_l = iter_events(trace)
    a2 = _acceptance_view(trace_l, g)["weighted_score"]
    ac = C1_assistance_cost(trace_l, g)
    return safe_div(a2, ac + 1.0, default=a2)


# --------------------------------------------------------------------------- #
# Dimension D -- autonomy / self-sufficiency                                   #
# --------------------------------------------------------------------------- #


def D1_autonomy_ratio(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """D1 ``autonomy_ratio``: ``A2 * (1 - C5)`` -- success net of leaked share."""
    g = as_gold(gold)
    trace_l = iter_events(trace)
    a2 = _acceptance_view(trace_l, g)["weighted_score"]
    leak = C5_solution_leakage(trace_l, g)
    return clip01(a2 * (1.0 - leak))


def _sev_at_least(payload: Any, level: int) -> bool:
    """True if an oracle_response payload's severity is ``>= level``."""
    return payload_severity(payload) >= level


def D2_unaided_progress_fraction(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """D2 ``unaided_progress_fraction``: forward progress made in help-free spans.

    Walks the checkpoint sequence; for each consecutive pair, positive deltas in
    ``weighted_score`` are 'unaided' when no sev>=2 ``oracle_response`` occurred
    between the two checkpoints' ``seq``. Returns unaided positive delta / total
    positive delta.
    """
    as_gold(gold)
    trace_l = iter_events(trace)
    cps = checkpoints(trace_l)
    if len(cps) < 2:
        return 1.0
    resp = oracle_responses(trace_l)
    total_pos = 0.0
    unaided_pos = 0.0
    for prev, cur in zip(cps, cps[1:], strict=False):
        delta = float(getattr(cur.payload, "weighted_score", 0.0)) - float(
            getattr(prev.payload, "weighted_score", 0.0)
        )
        if delta <= 0:
            continue
        total_pos += delta
        aided = any(
            prev.seq < ev.seq <= cur.seq
            and _sev_at_least(ev.payload, int(Severity.SUBSTANTIVE_SPEC_INFO))
            for ev in resp
        )
        if not aided:
            unaided_pos += delta
    if total_pos <= 0:
        return 1.0
    return clip01(unaided_pos / total_pos)


def D3_self_recovery_rate(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """D3 ``self_recovery_rate``: agent-detected failures fixed without sev>=2 help.

    A failure is a ``code_run`` with ``self_test_passed is False`` (or non-zero
    exit on a test). A recovery is a later ``code_run`` with ``self_test_passed
    is True`` within :data:`_SELF_RECOVERY_WINDOW_TURNS` turns and with no
    intervening sev>=2 ``oracle_response``. Returns recovered / detected.
    """
    as_gold(gold)
    trace_l = iter_events(trace)
    runs = code_runs(trace_l)
    resp = oracle_responses(trace_l)

    def _failed(p: Any) -> bool:
        stp = getattr(p, "self_test_passed", None)
        if stp is not None:
            return stp is False
        return getattr(p, "is_test", False) and getattr(p, "exit_code", 0) != 0

    def _passed(p: Any) -> bool:
        stp = getattr(p, "self_test_passed", None)
        if stp is not None:
            return stp is True
        return getattr(p, "is_test", False) and getattr(p, "exit_code", 0) == 0

    detected = 0
    recovered = 0
    for i, ev in enumerate(runs):
        if not _failed(ev.payload):
            continue
        detected += 1
        fail_turn = ev.t_turn if ev.t_turn is not None else 0
        for later in runs[i + 1 :]:
            lt = later.t_turn if later.t_turn is not None else 0
            if lt - fail_turn > _SELF_RECOVERY_WINDOW_TURNS:
                break
            if _passed(later.payload):
                aided = any(
                    ev.seq < r.seq <= later.seq
                    and _sev_at_least(r.payload, int(Severity.SUBSTANTIVE_SPEC_INFO))
                    for r in resp
                )
                if not aided:
                    recovered += 1
                break
    if detected == 0:
        return 1.0
    return clip01(recovered / detected)


def D4_blocked_resolution_self(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """D4 ``blocked_resolution_self``: blocked segments exited without sev>=3 help.

    A blocked segment runs from an ``agent_blocked(blocked=True)`` to its
    resolution (next ``agent_blocked(blocked=False)`` or ``oracle_response`` or
    episode end). The segment is *self-resolved* if no sev>=3 ``oracle_response``
    occurred inside it. Returns self-resolved / total blocked segments.
    """
    as_gold(gold)
    trace_l = iter_events(trace)
    resp = oracle_responses(trace_l)
    segments: list[tuple[int, int]] = []
    open_seq: int | None = None
    for ev in trace_l:
        etype = event_type(ev)
        if etype == "agent_blocked":
            if getattr(ev.payload, "blocked", True):
                if open_seq is None:
                    open_seq = ev.seq
            elif open_seq is not None:
                segments.append((open_seq, ev.seq))
                open_seq = None
        elif etype == "oracle_response" and open_seq is not None:
            segments.append((open_seq, ev.seq))
            open_seq = None
    if open_seq is not None:
        segments.append((open_seq, trace_l[-1].seq))
    if not segments:
        return 1.0
    self_resolved = 0
    for lo, hi in segments:
        aided = any(
            lo < r.seq <= hi and _sev_at_least(r.payload, int(Severity.DIRECTIONAL_HINT))
            for r in resp
        )
        if not aided:
            self_resolved += 1
    return clip01(self_resolved / len(segments))


def D5_proactive_inference(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """D5 ``proactive_inference``: hidden-spec units satisfied but NEVER revealed.

    The positive complement of C4: of the ``n_hidden_spec_units``, the fraction
    the agent satisfied (its info-unit id appears as a met acceptance criterion or
    in a passing checkpoint) WITHOUT the oracle having revealed it. Bounded
    ``[0,1]``.
    """
    g = as_gold(gold)
    trace_l = iter_events(trace)
    denom = g.n_hidden_spec_units
    if denom <= 0:
        return 0.0
    revealed: set[str] = set()
    for ev in oracle_responses(trace_l):
        for uid in getattr(ev.payload, "info_units_revealed", []) or []:
            revealed.add(uid)

    payload = _last_verification(trace_l)
    results = _criterion_result_map(payload)
    met_ids = {cid for cid, cr in results.items() if _criterion_value(cr) >= 1.0}

    inferred = 0
    for unit in g.info_units:
        if unit.id in revealed:
            continue
        if unit.id in met_ids:
            inferred += 1
    return clip01(inferred / denom)


# --------------------------------------------------------------------------- #
# Dimension E -- efficiency                                                     #
# --------------------------------------------------------------------------- #


def E1_wall_clock_s(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """E1 ``wall_clock_s``: last event ts - first event ts."""
    trace_l = iter_events(trace)
    if len(trace_l) < 2:
        return 0.0
    return max(0.0, trace_l[-1].ts - trace_l[0].ts)


def _usage_tokens(usage: Any) -> int:
    """Total tokens from a Usage-like payload field (or 0)."""
    if usage is None:
        return 0
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    return int(pt) + int(ct)


def _usage_cost(usage: Any) -> float:
    """Cost in USD from a Usage-like payload field (or 0.0)."""
    if usage is None:
        return 0.0
    return float(getattr(usage, "cost_usd", 0.0) or 0.0)


def E2_tokens_total(trace: Iterable[TraceEnvelope], gold: Any) -> int:
    """E2 ``tokens_total``: agent-side token total (oracle tokens excluded).

    Sums ``tokens`` Usage from agent-actor events; oracle-side tokens are tracked
    separately (a cheap agent must not look efficient by offloading onto the
    oracle), so events with ``actor == oracle`` are skipped.
    """
    total = 0
    for ev in iter_events(trace):
        if event_actor(ev) == "oracle":
            continue
        total += _usage_tokens(getattr(ev.payload, "tokens", None))
    return total


def E3_cost_usd_total(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """E3 ``cost_usd_total``: agent-side cost in USD (oracle cost excluded)."""
    total = 0.0
    for ev in iter_events(trace):
        if event_actor(ev) == "oracle":
            continue
        total += _usage_cost(getattr(ev.payload, "tokens", None))
    return total


def E4_n_tool_calls(trace: Iterable[TraceEnvelope], gold: Any) -> int:
    """E4 ``n_tool_calls``: count of ``tool_call`` events."""
    return sum(1 for ev in iter_events(trace) if event_type(ev) == "tool_call")


def E5_edit_churn(trace: Iterable[TraceEnvelope], gold: Any) -> int:
    """E5 ``edit_churn``: total added+removed LOC across ``file_edit`` events."""
    churn = 0
    for ev in iter_events(trace):
        if event_type(ev) == "file_edit":
            churn += int(getattr(ev.payload, "added", 0) or 0)
            churn += int(getattr(ev.payload, "removed", 0) or 0)
    return churn


def E6_iterations(trace: Iterable[TraceEnvelope], gold: Any) -> int:
    """E6 ``iterations``: number of code_run -> file_edit cycles.

    Counts ordered transitions where a ``code_run`` is followed (later in the
    trace) by a ``file_edit`` -- i.e. each run that triggered a subsequent edit.
    """
    trace_l = iter_events(trace)
    iters = 0
    seen_run = False
    for ev in trace_l:
        etype = event_type(ev)
        if etype == "code_run":
            seen_run = True
        elif etype == "file_edit" and seen_run:
            iters += 1
            seen_run = False
    return iters


def E7_cost_per_progress(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """E7 ``cost_per_progress``: ``cost_usd_total / max(A2, epsilon)`` (headline)."""
    g = as_gold(gold)
    trace_l = iter_events(trace)
    eps = float(spec_get("epsilon", default=1e-6))
    a2 = float(_acceptance_view(trace_l, g)["weighted_score"])
    return E3_cost_usd_total(trace_l, g) / max(a2, eps)


def E8_tokens_per_progress(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """E8 ``tokens_per_progress``: ``tokens_total / max(A2, epsilon)``."""
    g = as_gold(gold)
    trace_l = iter_events(trace)
    eps = float(spec_get("epsilon", default=1e-6))
    a2 = float(_acceptance_view(trace_l, g)["weighted_score"])
    return E2_tokens_total(trace_l, g) / max(a2, eps)


# --------------------------------------------------------------------------- #
# Dimension G -- UX quality of the agent's OWN communication                   #
# --------------------------------------------------------------------------- #


def G1_question_quality(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """G1 ``question_quality``: mean judge score of the agent's questions.

    The registry grades this with an LLM judge over ``oracle_query`` texts. The
    judge score is not in the canonical trace, so offline we surface the cached
    judge score when the harness logged it on each query payload
    (``meta.question_quality``), else return ``1.0`` when there are no questions
    to grade (vacuously good) and a neutral ``0.5`` when questions exist but no
    judge score is present.

    TODO: wire to :mod:`usabench.eval.scoring.v3_judge` to compute live scores.
    """
    as_gold(gold)
    queries = oracle_queries(iter_events(trace))
    if not queries:
        return 1.0
    scores: list[float] = []
    for ev in queries:
        meta = getattr(ev.payload, "context_refs", None)  # placeholder channel
        del meta
        s = _logged_judge_score(ev.payload, "question_quality")
        if s is not None:
            scores.append(s)
    if not scores:
        return 0.5
    return clip01(sum(scores) / len(scores))


def _logged_judge_score(payload: Any, key: str) -> float | None:
    """Extract a pre-logged judge score from a payload's optional meta, if any."""
    meta = getattr(payload, "meta", None)
    if isinstance(meta, dict) and key in meta:
        try:
            return clip01(float(meta[key]))
        except (TypeError, ValueError):
            return None
    return None


def G2_redundant_query_rate(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """G2 ``redundant_query_rate``: fraction of queries duplicating a prior ask.

    Offline lexical proxy: a query is redundant if its normalised text matches a
    strictly earlier query's normalised text exactly. Returns redundant / total.
    """
    as_gold(gold)
    queries = oracle_queries(iter_events(trace))
    if not queries:
        return 0.0
    seen: set[str] = set()
    redundant = 0
    for ev in queries:
        norm = " ".join(str(getattr(ev.payload, "text", "")).lower().split())
        if norm and norm in seen:
            redundant += 1
        else:
            seen.add(norm)
    return clip01(redundant / len(queries))


def G3_status_transparency(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """G3 ``status_transparency``: did the agent keep the user informed?

    LLM-judged in the registry. Offline proxy: a normalised count of
    ``message_to_user`` events relative to the number of agent turns, saturating
    at 1.0 (more status updates -> more transparent, with diminishing returns).
    """
    as_gold(gold)
    trace_l = iter_events(trace)
    msgs = len(messages_to_user(trace_l))
    if msgs == 0:
        return 0.0
    turns = _max_turn(trace_l)
    if turns <= 0:
        return 1.0 if msgs > 0 else 0.0
    # One informative update roughly every few turns is healthy; saturate at 1.
    return clip01(msgs / max(1.0, turns / 4.0))


def _max_turn(trace: list[TraceEnvelope]) -> int:
    """Highest ``t_turn`` observed in the trace (0 if none)."""
    turns = [ev.t_turn for ev in trace if ev.t_turn is not None]
    return max(turns) if turns else 0


def G4_false_confidence(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """G4 ``false_confidence``: rate of agent assertions later contradicted.

    Offline proxy: the fraction of ``code_run`` self-tests the agent ran that it
    asserted/expected to pass but that *failed* (``self_test_passed is False``),
    among all self-tested runs -- a logged mis-calibration signal. Augmented by
    any ``message_to_user`` claiming completion while the final acceptance is not
    accepted (one strong false-confidence event), folded in as an extra failed
    unit. Returns a rate in ``[0,1]``.
    """
    g = as_gold(gold)
    trace_l = iter_events(trace)
    tested = [
        ev
        for ev in code_runs(trace_l)
        if getattr(ev.payload, "self_test_passed", None) is not None
    ]
    failed = sum(1 for ev in tested if getattr(ev.payload, "self_test_passed", None) is False)
    denom = len(tested)

    accepted = _acceptance_view(trace_l, g)["accepted"]
    claim_done = any(
        _claims_completion(str(getattr(ev.payload, "text", "")))
        for ev in messages_to_user(trace_l)
    )
    if claim_done and not accepted:
        failed += 1
        denom += 1
    if denom == 0:
        return 0.0
    return clip01(failed / denom)


_COMPLETION_MARKERS = (
    "done",
    "complete",
    "finished",
    "ready to use",
    "all set",
    "fully working",
    "it works",
    "task is complete",
)


def _claims_completion(text: str) -> bool:
    """Heuristic: does a user-facing message assert the task is finished?"""
    t = text.lower()
    return any(marker in t for marker in _COMPLETION_MARKERS)


# --------------------------------------------------------------------------- #
# Rollup                                                                        #
# --------------------------------------------------------------------------- #

#: Registry id -> metric function. Ordered A..G to match ``docs/metrics.md``.
_REGISTRY: dict[str, Any] = {
    "A1_success_binary": A1_success_binary,
    "A2_criteria_score": A2_criteria_score,
    "A3_core_criteria_score": A3_core_criteria_score,
    "A4_goal_drift": A4_goal_drift,
    "A5_regression_free": A5_regression_free,
    "B1_n_interventions": B1_n_interventions,
    "B2_n_clarifications": B2_n_clarifications,
    "B3_n_hint_requests": B3_n_hint_requests,
    "B4_n_corrections": B4_n_corrections,
    "B5_n_handoffs": B5_n_handoffs,
    "B6_turns_to_first_working": B6_turns_to_first_working,
    "B7_turns_to_acceptance": B7_turns_to_acceptance,
    "B8_interventions_to_acceptance": B8_interventions_to_acceptance,
    "B9_mean_query_class_entropy": B9_mean_query_class_entropy,
    "B10_time_blocked_fraction": B10_time_blocked_fraction,
    "C1_assistance_cost": C1_assistance_cost,
    "C2_max_severity": C2_max_severity,
    "C3_severity_histogram": C3_severity_histogram,
    "C4_spec_info_transferred": C4_spec_info_transferred,
    "C5_solution_leakage": C5_solution_leakage,
    "C6_assistance_efficiency": C6_assistance_efficiency,
    "D1_autonomy_ratio": D1_autonomy_ratio,
    "D2_unaided_progress_fraction": D2_unaided_progress_fraction,
    "D3_self_recovery_rate": D3_self_recovery_rate,
    "D4_blocked_resolution_self": D4_blocked_resolution_self,
    "D5_proactive_inference": D5_proactive_inference,
    "E1_wall_clock_s": E1_wall_clock_s,
    "E2_tokens_total": E2_tokens_total,
    "E3_cost_usd_total": E3_cost_usd_total,
    "E4_n_tool_calls": E4_n_tool_calls,
    "E5_edit_churn": E5_edit_churn,
    "E6_iterations": E6_iterations,
    "E7_cost_per_progress": E7_cost_per_progress,
    "E8_tokens_per_progress": E8_tokens_per_progress,
    "G1_question_quality": G1_question_quality,
    "G2_redundant_query_rate": G2_redundant_query_rate,
    "G3_status_transparency": G3_status_transparency,
    "G4_false_confidence": G4_false_confidence,
}


def registry() -> dict[str, Any]:
    """Return the ``{metric_id: function}`` registry (copy)."""
    return dict(_REGISTRY)


def compute_all(trace: Iterable[TraceEnvelope], gold: Any) -> dict[str, Any]:
    """Compute every per-episode metric (A..G) for one trace.

    Args:
        trace: Parsed ``trace.jsonl`` events for a single episode.
        gold: Task gold (see :func:`usabench.eval.gold.as_gold`).

    Returns:
        A dict ``{metric_id: value}`` covering every row of ``docs/metrics.md``
        that is per-episode computable (Dimension F is seed-level and lives in
        :mod:`usabench.eval.aggregate`).
    """
    g = as_gold(gold)
    trace_l = iter_events(trace)
    out: dict[str, Any] = {}
    for mid, fn in _REGISTRY.items():
        out[mid] = fn(trace_l, g)
    return out
