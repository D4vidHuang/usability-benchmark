"""Task QC stage 5: intervention calibration + the discriminative gate.

A shipped task must be *empirically* shown to require and reward assistance
(``docs/tasks.md`` §6.4, §8.5). This module consumes the pilot-run results -- a
small panel of agents x trials, run twice: once with the oracle LIVE and once with
interaction DISABLED -- and:

1. **Calibrates** ``expected_interventions``: the median / p10 / p90 number of
   oracle interventions over the live trials, plus a by-severity-class breakdown,
   to fill the task's ``expected_interventions`` block.

2. **Gates on discrimination**: a task is *discriminative* iff disabling
   interaction materially drops the score -- i.e. on >= ``min_models_drop`` of the
   panel models, the no-interaction score falls by at least ``min_score_drop``
   below the interactive score (a high-severity ambiguity was load-bearing). Tasks
   where interventions are always ~0 *and* the score is already high are rejected;
   tasks no model can pass even with max help are flagged for gold/AC review.

This stage is a PURE offline function of pilot ``trace.jsonl`` summaries: it counts
``oracle_response`` events by their logged 0-5 severity and compares cached
acceptance scores. Where it needs constants (the success threshold ``tau_ga``) it
reads them from ``usabench/eval/spec/usability_score.yaml`` via the foundation
loader -- NO scoring constant is hardcoded here (DESIGN frozen decision #2).
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import structlog

from usabench.core.enums import InteractionType, Severity
from usabench.core.schema import TraceEnvelope, parse_event

__all__ = [
    "TrialSummary",
    "CalibrationResult",
    "summarize_trace",
    "calibrate",
    "DEFAULT_MIN_SCORE_DROP",
    "DEFAULT_MIN_MODELS_DROP",
]

log = structlog.get_logger(__name__)

#: A task must drop at least this much score when interaction is disabled.
DEFAULT_MIN_SCORE_DROP = 0.15
#: ...on at least this many distinct panel models, to be called discriminative.
DEFAULT_MIN_MODELS_DROP = 2


@dataclass(slots=True)
class TrialSummary:
    """A compact summary of one pilot trial (derived from its ``trace.jsonl``).

    Attributes:
        model: The agent model id for this trial.
        interactive: True if the oracle was live; False for the no-interaction arm.
        score: The trial's final weighted/GA score in ``[0, 1]``.
        accepted: Whether the run was accepted.
        n_interventions: Count of oracle responses with severity >= 1 (assistance).
        n_oracle_events: Count of all oracle responses (includes L0 elicitation).
        by_severity: Histogram of oracle-response severities, keyed by label.
    """

    model: str
    interactive: bool
    score: float = 0.0
    accepted: bool = False
    n_interventions: int = 0
    n_oracle_events: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class CalibrationResult:
    """The calibration + discrimination verdict for one task.

    Attributes:
        task_id: The task under calibration.
        discriminative: True if disabling interaction materially hurt the score.
        expected_interventions: The ``expected_interventions`` block to write back
            onto the task (median/p10/p90 + by-type + provenance).
        models_with_drop: Models whose no-interaction score dropped past threshold.
        mean_interactive_score: Mean live-oracle score across trials.
        mean_noninteractive_score: Mean no-interaction score across trials.
        flagged_unsolvable: True if no trial passed even with the oracle live
            (likely a broken gold/AC set; flag for review).
        errors: Hard problems (e.g. no usable trials).
    """

    task_id: str
    discriminative: bool = False
    expected_interventions: dict[str, Any] = field(default_factory=dict)
    models_with_drop: list[str] = field(default_factory=list)
    mean_interactive_score: float = 0.0
    mean_noninteractive_score: float = 0.0
    flagged_unsolvable: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if the task passes the calibration gate (discriminative, no errors)."""
        return self.discriminative and not self.errors

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "task_id": self.task_id,
            "ok": self.ok,
            "discriminative": self.discriminative,
            "expected_interventions": dict(self.expected_interventions),
            "models_with_drop": list(self.models_with_drop),
            "mean_interactive_score": self.mean_interactive_score,
            "mean_noninteractive_score": self.mean_noninteractive_score,
            "flagged_unsolvable": self.flagged_unsolvable,
            "errors": list(self.errors),
        }


def _coerce_event(line: Any) -> TraceEnvelope | None:
    """Best-effort coerce a raw line/dict/envelope into a :class:`TraceEnvelope`."""
    if isinstance(line, TraceEnvelope):
        return line
    if isinstance(line, dict):
        try:
            return parse_event(line)
        except Exception:  # noqa: BLE001 - skip unparseable lines defensively
            return None
    return None


def summarize_trace(
    events: Iterable[Any],
    *,
    model: str,
    interactive: bool,
) -> TrialSummary:
    """Summarize one trial's trace events into a :class:`TrialSummary`.

    Counts oracle interventions by their logged 0-5 severity, and reads the final
    score / acceptance from the ``episode_end`` event (the scorer-derivable cache).

    Args:
        events: An iterable of trace lines (dicts) or :class:`TraceEnvelope`s.
        model: The agent model id for this trial.
        interactive: Whether the oracle was live in this trial.

    Returns:
        A :class:`TrialSummary`.
    """
    summary = TrialSummary(model=model, interactive=interactive)
    for raw in events:
        ev = _coerce_event(raw)
        if ev is None:
            continue
        etype = ev.type.value if hasattr(ev.type, "value") else str(ev.type)
        if etype == InteractionType.ORACLE_RESPONSE.value:
            payload = ev.payload
            sev_raw = getattr(payload, "severity", 0)
            try:
                sev = Severity(int(sev_raw))
            except (ValueError, TypeError):
                sev = Severity.NONE
            summary.n_oracle_events += 1
            summary.by_severity[sev.label] = summary.by_severity.get(sev.label, 0) + 1
            if Severity.is_assistance(int(sev)):
                summary.n_interventions += 1
        elif etype == InteractionType.EPISODE_END.value:
            payload = ev.payload
            fws = getattr(payload, "final_weighted_score", None)
            if fws is not None:
                summary.score = float(fws)
            summary.accepted = bool(getattr(payload, "accepted", False))
    return summary


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile of a list (``q`` in ``[0, 1]``)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _by_type_breakdown(trials: list[TrialSummary]) -> dict[str, int]:
    """Median per-severity-class intervention counts across the interactive trials."""
    classes: set[str] = set()
    for t in trials:
        classes.update(t.by_severity.keys())
    out: dict[str, int] = {}
    for cls in classes:
        counts = [float(t.by_severity.get(cls, 0)) for t in trials]
        out[cls] = int(round(statistics.median(counts))) if counts else 0
    return out


def calibrate(
    trials: Iterable[TrialSummary],
    *,
    task_id: str,
    tau_ga: float | None = None,
    min_score_drop: float = DEFAULT_MIN_SCORE_DROP,
    min_models_drop: int = DEFAULT_MIN_MODELS_DROP,
) -> CalibrationResult:
    """Compute ``expected_interventions`` and the discriminative-gate verdict.

    Args:
        trials: All pilot :class:`TrialSummary` records (both arms).
        task_id: The task id.
        tau_ga: Success threshold; defaults to ``tau_ga`` from the frozen spec.
        min_score_drop: Minimum per-model score drop (interactive - none) to count
            a model as showing the ambiguity is load-bearing.
        min_models_drop: Minimum number of models that must show the drop.

    Returns:
        A :class:`CalibrationResult`.
    """
    if tau_ga is None:
        from usabench.eval.spec import load_spec

        tau_ga = float(load_spec().get("tau_ga", 0.80))

    trials = list(trials)
    result = CalibrationResult(task_id=task_id)
    if not trials:
        result.errors.append("no pilot trials provided")
        return result

    live = [t for t in trials if t.interactive]
    none = [t for t in trials if not t.interactive]
    if not live:
        result.errors.append("no interactive (live-oracle) trials provided")
        return result

    # --- Calibrate expected_interventions from the live arm. ---
    intervention_counts = [float(t.n_interventions) for t in live]
    result.expected_interventions = {
        "median": int(round(statistics.median(intervention_counts))),
        "p10": int(round(_percentile(intervention_counts, 0.10))),
        "p90": int(round(_percentile(intervention_counts, 0.90))),
        "by_type": _by_type_breakdown(live),
        "calibrated_with": sorted({t.model for t in live}),
        "n_trials": len(live),
    }

    # --- Solvability: did anything pass with the oracle live? ---
    result.mean_interactive_score = round(
        statistics.fmean([t.score for t in live]) if live else 0.0, 6
    )
    result.flagged_unsolvable = not any(t.score >= tau_ga or t.accepted for t in live)
    if result.flagged_unsolvable:
        result.errors.append(
            f"no live trial reached tau_ga={tau_ga} (gold/AC likely broken; review)"
        )

    # --- Discrimination: compare per-model live vs no-interaction. ---
    if none:
        result.mean_noninteractive_score = round(
            statistics.fmean([t.score for t in none]), 6
        )
        models = {t.model for t in live} & {t.model for t in none}
        dropped: list[str] = []
        for m in sorted(models):
            live_m = statistics.fmean([t.score for t in live if t.model == m])
            none_m = statistics.fmean([t.score for t in none if t.model == m])
            if (live_m - none_m) >= min_score_drop:
                dropped.append(m)
        result.models_with_drop = dropped
        result.discriminative = len(dropped) >= min_models_drop
        if not result.discriminative:
            result.errors.append(
                f"not discriminative: only {len(dropped)} model(s) dropped >= "
                f"{min_score_drop} when interaction was disabled (need {min_models_drop})"
            )
    else:
        # No no-interaction arm => cannot prove discrimination here.
        result.discriminative = False
        result.errors.append(
            "no no-interaction trials provided; cannot evaluate the discriminative gate"
        )

    result.expected_interventions["discriminative"] = result.discriminative
    return result
