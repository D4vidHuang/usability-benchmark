"""V2 -- frozen rubric / acceptance-criteria checklist channel (``docs/scoring.md`` §4).

The rubric is authored offline from ``hidden_spec`` and frozen (per-run synthesis
is the dominant rubric-variance source). Each acceptance criterion is one item
with a ``weight`` and a routing ``check_kind``; V2 aggregates the *soft*
(non-hard) items whose channel is deterministic (``func`` / ``rubric_auto``)::

    V2 = ( Σ_{i in soft} w_i * s_i ) / ( Σ_{i in soft} w_i )

Hard constraints are NOT a weighted V2 term -- they form the GA gate via
``hard_pass_frac = Σ_{i in hard} s_i / |hard|`` (``docs/scoring.md`` §4.4), which
:mod:`usabench.eval.scoring.ga` consumes.

Item scores ``s_i in {0, 0.5, 1}``; deterministic checks return ``{0,1}`` only.
This module scores from explicit item results OR recovers them from the last
``verification_run`` in the canonical trace, so it is a pure function of
``trace.jsonl`` + gold.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from usabench.core.schema import TraceEnvelope
from usabench.eval._common import as_value, clip01, verification_runs
from usabench.eval.gold import Gold, as_gold

__all__ = ["V2Result", "score_v2", "rubric_auto_item_score", "hard_pass_frac_from_results"]


@dataclass(frozen=True)
class V2Result:
    """The V2 channel score plus the gating ``hard_pass_frac`` (all ``[0,1]``)."""

    score: float
    hard_pass_frac: float
    n_soft: int = 0
    n_hard: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


def rubric_auto_item_score(value: Any) -> float:
    """Coerce an item outcome into ``s_i in {0, 0.5, 1}``.

    Accepts a bool, a numeric score in ``[0,1]`` (snapped to {0,0.5,1}), a
    :class:`~usabench.core.schema.CriterionResult`-like object, or ``None``
    (treated as 0 / absent).
    """
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return _snap(float(value))
    # CriterionResult-like
    score = getattr(value, "score", None)
    if score is not None:
        return _snap(float(score))
    passed = getattr(value, "passed", None)
    if passed is not None:
        return 1.0 if passed else 0.0
    return 0.0


def _snap(x: float) -> float:
    """Snap a continuous score to the {0, 0.5, 1} item scale."""
    x = clip01(x)
    if x >= 0.75:
        return 1.0
    if x >= 0.25:
        return 0.5
    return 0.0


def hard_pass_frac_from_results(gold: Gold, results: Mapping[str, Any]) -> float:
    """Fraction of hard criteria fully met (``s_i == 1``) given item results."""
    hard = gold.hard_criteria
    if not hard:
        return 1.0
    passed = sum(1.0 for c in hard if rubric_auto_item_score(results.get(c.id)) >= 1.0)
    return clip01(passed / len(hard))


def score_v2(
    trace_or_results: Iterable[TraceEnvelope] | Mapping[str, Any],
    gold: Any,
    *,
    results: Mapping[str, Any] | None = None,
) -> V2Result:
    """Compute the V2 score (soft rubric items) and ``hard_pass_frac``.

    Two calling conventions:

    * ``score_v2(trace, gold)`` -- recover item results from the last
      ``verification_run`` event in the trace.
    * ``score_v2(results_mapping, gold)`` or ``score_v2(trace, gold,
      results=mapping)`` -- score an explicit ``{criterion_id: outcome}`` mapping.

    'Soft' items are criteria with ``is_hard == False`` whose channel is
    deterministic (``func`` or ``rubric_auto``); judge-channel items are scored by
    :mod:`usabench.eval.scoring.v3_judge` (V3) and excluded here.

    Args:
        trace_or_results: A trace iterable, or a ``{id: outcome}`` mapping.
        gold: Task gold.
        results: Optional explicit results mapping (overrides trace recovery).

    Returns:
        A :class:`V2Result`.
    """
    g = as_gold(gold)
    res_map = _resolve_results(trace_or_results, results)

    soft = [
        c
        for c in g.criteria
        if not c.is_hard and str(as_value(c.check_kind)) in {"func", "rubric_auto"}
    ]
    total_w = sum(float(c.weight) for c in soft)
    if total_w > 0:
        acc = sum(float(c.weight) * rubric_auto_item_score(res_map.get(c.id)) for c in soft)
        v2 = clip01(acc / total_w)
    else:
        v2 = 0.0
    hpf = hard_pass_frac_from_results(g, res_map)
    return V2Result(
        score=v2,
        hard_pass_frac=hpf,
        n_soft=len(soft),
        n_hard=len(g.hard_criteria),
        detail={"soft_ids": [c.id for c in soft]},
    )


def _resolve_results(
    trace_or_results: Iterable[TraceEnvelope] | Mapping[str, Any],
    results: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the ``{criterion_id: outcome}`` mapping from the chosen source."""
    if results is not None:
        return dict(results)
    if isinstance(trace_or_results, Mapping):
        return dict(trace_or_results)
    # Treat as a trace; recover from the last verification_run.
    runs = verification_runs(list(trace_or_results))
    if not runs:
        return {}
    payload = runs[-1].payload
    out: dict[str, Any] = {}
    for bucket in ("must_have", "should_have"):
        for cr in getattr(payload, bucket, []) or []:
            out[cr.id] = cr
    return out
