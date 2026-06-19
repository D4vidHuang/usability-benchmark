"""Degenerate-strategy integrity flags (``docs/scoring.md`` §8).

The benchmark's whole point is *appropriate* interaction, so three degenerate
strategies must be actively detected and surfaced. This module computes the
per-(task, seed) integrity flags as **pure offline functions of ``trace.jsonl`` +
the frozen task gold** -- no I/O, no LLM, no heavy deps:

* ``fake_done`` -- the agent asserted completion but the artifact does not deliver
  (a completion-claim classifier over the trace AND ``hard_pass_frac < 1``).
* ``never_ask`` -- the agent shipped an *under-specified* task without resolving a
  single ambiguity (no clarification queries on a task that HAS ambiguity points).
* ``over_ask`` -- the agent offloaded onto the oracle (interaction-count over a
  per-task panel ceiling AND/OR a high redundant+high-severity assistance share).
* ``oracle_attributed_credit_frac`` -- the fraction of *met* acceptance criteria
  whose satisfaction is traceable to a high-severity (sev>=4) oracle response
  (the attribution check; the composite discounts this).
* ``discovery_failed`` -- the entrypoint could not be discovered, so the artifact
  never validly executed (``valid_exec == 0`` with a recorded discovery failure).

Only ``fake_done`` carries a *score* consequence in the headline composite (via
:func:`usabench.eval.composite.multiplicative_usability`); the rest are reported
as integrity statistics and feed the leaderboard's integrity columns
(``docs/scoring.md`` §10.3). ``never_ask`` / ``over_ask`` are the UC-side
classifications the metrics module surfaces; we compute them from the same trace.

The numeric thresholds (``over_ask`` panel multiplier, sev>=4 cutoff) are derived
from the spec where the spec defines them (the severity scale via
:class:`~usabench.core.enums.Severity`); the per-task interaction ceiling is a
caller-supplied panel statistic (there is no global constant to hardcode).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from usabench.core.enums import QueryClass, Severity
from usabench.core.schema import TraceEnvelope
from usabench.eval._common import (
    as_value,
    clip01,
    episode_end,
    event_type,
    iter_events,
    messages_to_user,
    oracle_queries,
    oracle_responses,
    payload_severity,
    safe_div,
    verification_runs,
)
from usabench.eval.gold import Gold, as_gold

__all__ = [
    "IntegrityFlags",
    "fake_done",
    "never_ask",
    "over_ask",
    "oracle_attributed_credit_frac",
    "discovery_failed",
    "claims_completion",
    "compute_integrity",
]

#: Lexical markers a user-facing message uses to assert the task is finished.
#: Mirrors :data:`usabench.eval.metrics._COMPLETION_MARKERS` (kept local so the
#: two modules stay independently importable / testable).
_COMPLETION_MARKERS: tuple[str, ...] = (
    "done",
    "complete",
    "completed",
    "finished",
    "ready to use",
    "all set",
    "fully working",
    "it works",
    "task is complete",
    "i have finished",
    "should be working now",
)

#: Default multiplier on the panel-mean interaction count above which a run is
#: classified as over-asking, when the caller supplies a panel baseline but no
#: explicit ceiling. (When no panel baseline is supplied, over_ask cannot fire on
#: the count signal alone -- only the assistance-share signal can.)
_OVER_ASK_PANEL_MULTIPLIER: float = 2.0

#: Fraction of interventions that must be redundant-or-high-severity for the
#: assistance-share signal of over_ask to fire (a content-based fallback that does
#: not need a panel baseline).
_OVER_ASK_SHARE_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class IntegrityFlags:
    """The per-(task, seed) integrity-flag bundle (``docs/scoring.md`` §8.4)."""

    fake_done: bool
    never_ask: bool
    over_ask: bool
    discovery_failed: bool
    oracle_attributed_credit_frac: float
    n_interventions: int = 0
    n_clarifications: int = 0
    hard_pass_frac: float = 1.0
    detail: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #


def claims_completion(text: str) -> bool:
    """Heuristic completion-claim classifier over a user-facing message.

    Args:
        text: A ``message_to_user`` / agent message body.

    Returns:
        ``True`` if the message asserts the task is finished.
    """
    t = (text or "").lower()
    return any(marker in t for marker in _COMPLETION_MARKERS)


def _last_verification_payload(trace_l: list[TraceEnvelope]) -> Any:
    """Return the last ``verification_run`` payload, or ``None``."""
    runs = verification_runs(trace_l)
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


def _criterion_passed(cr: Any) -> bool:
    """True if a CriterionResult counts as fully met (score>=1 or passed True)."""
    if cr is None:
        return False
    score = getattr(cr, "score", None)
    if score is not None:
        return float(score) >= 1.0
    return bool(getattr(cr, "passed", False))


def _hard_pass_frac(trace_l: list[TraceEnvelope], gold: Gold) -> float:
    """Fraction of gold hard criteria fully met in the last verification_run.

    Falls back to the cached ``episode_end.accepted`` (1.0 if accepted else 0.0)
    when no per-criterion verification detail is present.
    """
    payload = _last_verification_payload(trace_l)
    results = _criterion_result_map(payload)
    hard = gold.hard_criteria
    if hard and results:
        met = sum(1.0 for c in hard if _criterion_passed(results.get(c.id)))
        return clip01(met / len(hard))
    # No per-criterion hard detail: use the cached accepted flag as a proxy.
    ee = episode_end(trace_l)
    if ee is not None:
        return 1.0 if bool(getattr(ee.payload, "accepted", False)) else 0.0
    return 1.0


def _n_clarifications(trace_l: list[TraceEnvelope]) -> int:
    """Count ``oracle_query`` events of class ``clarification``."""
    n = 0
    for ev in oracle_queries(trace_l):
        qc = getattr(ev.payload, "query_class", None)
        if str(as_value(qc)) == QueryClass.CLARIFICATION.value:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Individual flags                                                             #
# --------------------------------------------------------------------------- #


def fake_done(trace: Iterable[TraceEnvelope], gold: Any) -> bool:
    """``fake_done``: agent claimed completion but the artifact does not deliver.

    Fires when a completion-claim is detected in any ``message_to_user`` (or an
    ``episode_end`` reporting ``accepted`` while hard constraints are unmet is a
    structural contradiction) **and** ``hard_pass_frac < 1`` (``docs/scoring.md``
    §8.1). Confident prose cannot move the execution-grounded ``hard_pass_frac``,
    so this is a trust-integrity flag, not a quality measure.

    Args:
        trace: Parsed ``trace.jsonl`` events.
        gold: Task gold (see :func:`usabench.eval.gold.as_gold`).

    Returns:
        ``True`` if the fake-done condition holds.
    """
    g = as_gold(gold)
    trace_l = iter_events(trace)
    hpf = _hard_pass_frac(trace_l, g)
    if hpf >= 1.0:
        return False
    claimed = any(
        claims_completion(str(getattr(ev.payload, "text", "")))
        for ev in messages_to_user(trace_l)
    )
    # An episode_end that records accepted=True while hard constraints are unmet is
    # itself a completion claim contradicted by execution.
    ee = episode_end(trace_l)
    if ee is not None and bool(getattr(ee.payload, "accepted", False)):
        claimed = True
    return bool(claimed)


def never_ask(trace: Iterable[TraceEnvelope], gold: Any) -> bool:
    """``never_ask``: shipped an under-specified task without any clarification.

    Fires when the task HAS ambiguity points (it is genuinely under-specified) and
    the agent made **zero** clarification queries (``docs/scoring.md`` §8.2). On a
    fully-specified task (no ambiguity points) never-ask cannot fire -- not asking
    is correct there.

    Args:
        trace: Parsed ``trace.jsonl`` events.
        gold: Task gold.

    Returns:
        ``True`` if the never-ask condition holds.
    """
    g = as_gold(gold)
    if not g.ambiguity_points:
        return False
    trace_l = iter_events(trace)
    return _n_clarifications(trace_l) == 0


def over_ask(
    trace: Iterable[TraceEnvelope],
    gold: Any,
    *,
    panel_mean_interventions: float | None = None,
    panel_ceiling: float | None = None,
) -> bool:
    """``over_ask``: the agent offloaded work onto the oracle.

    Two independent signals (either suffices), per ``docs/scoring.md`` §8.3:

    * **count signal** -- the run's intervention count exceeds a per-task ceiling.
      The ceiling is ``panel_ceiling`` if given, else
      ``_OVER_ASK_PANEL_MULTIPLIER * panel_mean_interventions`` when a panel mean
      is supplied. With no panel context the count signal cannot fire (there is no
      defensible global threshold to hardcode).
    * **assistance-share signal** -- a high fraction of interventions are
      *redundant or high-severity* (sev>=substantive). A content-based fallback
      that needs no panel baseline.

    Args:
        trace: Parsed ``trace.jsonl`` events.
        gold: Task gold (validated for shape; not otherwise used).
        panel_mean_interventions: Mean intervention count over the reference panel
            for this task, if available.
        panel_ceiling: Explicit per-task intervention ceiling, if available.

    Returns:
        ``True`` if either over-ask signal fires.
    """
    as_gold(gold)  # validate shape
    trace_l = iter_events(trace)
    responses = oracle_responses(trace_l)
    n = len(responses)

    # Count signal.
    ceiling: float | None = None
    if panel_ceiling is not None:
        ceiling = float(panel_ceiling)
    elif panel_mean_interventions is not None:
        ceiling = _OVER_ASK_PANEL_MULTIPLIER * float(panel_mean_interventions)
    count_signal = ceiling is not None and n > ceiling

    # Assistance-share signal: redundant OR high-severity interventions dominate.
    if n > 0:
        substantive = int(Severity.SUBSTANTIVE_SPEC_INFO)
        heavy = sum(1 for ev in responses if payload_severity(ev.payload) >= substantive)
        redundant = _redundant_query_count(trace_l)
        share = safe_div(float(heavy + redundant), float(n), default=0.0)
        share_signal = share >= _OVER_ASK_SHARE_THRESHOLD
    else:
        share_signal = False

    return bool(count_signal or share_signal)


def _redundant_query_count(trace_l: list[TraceEnvelope]) -> int:
    """Count queries whose normalised text exactly repeats an earlier query."""
    seen: set[str] = set()
    redundant = 0
    for ev in oracle_queries(trace_l):
        norm = " ".join(str(getattr(ev.payload, "text", "")).lower().split())
        if norm and norm in seen:
            redundant += 1
        elif norm:
            seen.add(norm)
    return redundant


def oracle_attributed_credit_frac(trace: Iterable[TraceEnvelope], gold: Any) -> float:
    """``oracle_attributed_credit_frac``: met-criteria share traceable to sev>=4 help.

    The attribution check (``docs/scoring.md`` §8.3): among the acceptance criteria
    that ended up *met*, the fraction whose id is ``cited_criteria`` of, or whose
    info unit is ``info_units_revealed`` by, a sev>=4 (partial-solution / takeover)
    oracle response. High values mean the oracle materially built the passing parts.

    Args:
        trace: Parsed ``trace.jsonl`` events.
        gold: Task gold.

    Returns:
        A fraction in ``[0,1]``; ``0.0`` if nothing was met.
    """
    as_gold(gold)
    trace_l = iter_events(trace)
    payload = _last_verification_payload(trace_l)
    results = _criterion_result_map(payload)
    met_ids = {cid for cid, cr in results.items() if _criterion_passed(cr)}
    if not met_ids:
        return 0.0

    high = int(Severity.PARTIAL_SOLUTION)
    attributed: set[str] = set()
    for ev in oracle_responses(trace_l):
        if payload_severity(ev.payload) < high:
            continue
        for cid in getattr(ev.payload, "cited_criteria", []) or []:
            if cid in met_ids:
                attributed.add(cid)
        for uid in getattr(ev.payload, "info_units_revealed", []) or []:
            if uid in met_ids:
                attributed.add(uid)
    return clip01(len(attributed) / len(met_ids))


def discovery_failed(trace: Iterable[TraceEnvelope], gold: Any) -> bool:
    """``discovery_failed``: the entrypoint could not be discovered, so it never ran.

    A discovery failure is recorded when the final ``verification_run`` has no
    resolvable ``entrypoint`` AND evaluated no functional criteria (the artifact's
    interface could not be found), or when any ``code_run`` is explicitly tagged as
    a discovery failure in its truncated stderr. This maps to ``valid_exec == 0``
    not attributable to a crashing artifact (``docs/scoring.md`` §3.2 / §8.4).

    Args:
        trace: Parsed ``trace.jsonl`` events.
        gold: Task gold.

    Returns:
        ``True`` if a discovery failure is detected.
    """
    as_gold(gold)
    trace_l = iter_events(trace)
    payload = _last_verification_payload(trace_l)
    if payload is None:
        # No verification ran at all -> nothing could be discovered/executed.
        # Only flag when the run actually ended (not a truncated trace).
        return episode_end(trace_l) is not None
    entrypoint = getattr(payload, "entrypoint", None)
    results = _criterion_result_map(payload)
    no_func_eval = len(results) == 0
    if not entrypoint and no_func_eval:
        return True
    # Explicit discovery-failure marker on any env-side code_run.
    for ev in trace_l:
        if event_type(ev) == "code_run":
            stderr = str(getattr(ev.payload, "stderr_trunc", "") or "").lower()
            if "discovery_failed" in stderr or "no entrypoint" in stderr:
                return True
    return False


# --------------------------------------------------------------------------- #
# Rollup                                                                       #
# --------------------------------------------------------------------------- #


def compute_integrity(
    trace: Iterable[TraceEnvelope],
    gold: Any,
    *,
    panel_mean_interventions: float | None = None,
    panel_ceiling: float | None = None,
) -> IntegrityFlags:
    """Compute the full :class:`IntegrityFlags` bundle for one episode.

    Args:
        trace: Parsed ``trace.jsonl`` events for one episode.
        gold: Task gold (see :func:`usabench.eval.gold.as_gold`).
        panel_mean_interventions: Optional per-task panel mean intervention count
            (enables the over-ask count signal).
        panel_ceiling: Optional explicit per-task intervention ceiling.

    Returns:
        An :class:`IntegrityFlags` with every flag and the supporting counts.
    """
    g = as_gold(gold)
    trace_l = iter_events(trace)
    hpf = _hard_pass_frac(trace_l, g)
    n_interventions = len(oracle_responses(trace_l))
    n_clar = _n_clarifications(trace_l)
    return IntegrityFlags(
        fake_done=fake_done(trace_l, g),
        never_ask=never_ask(trace_l, g),
        over_ask=over_ask(
            trace_l,
            g,
            panel_mean_interventions=panel_mean_interventions,
            panel_ceiling=panel_ceiling,
        ),
        discovery_failed=discovery_failed(trace_l, g),
        oracle_attributed_credit_frac=oracle_attributed_credit_frac(trace_l, g),
        n_interventions=n_interventions,
        n_clarifications=n_clar,
        hard_pass_frac=hpf,
        detail={
            "n_ambiguity_points": len(g.ambiguity_points),
            "panel_mean_interventions": panel_mean_interventions,
            "panel_ceiling": panel_ceiling,
        },
    )
