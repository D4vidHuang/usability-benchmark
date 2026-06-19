"""Per-task difficulty normalisation against a reference panel (``docs/metrics.md`` §7.1).

Raw counts (turns, AC, cost) scale with task difficulty, so cross-task aggregates
use **difficulty-normalised z-scores** computed against a frozen **reference agent
panel** P run once per task::

    m_tilde = (m - mu_{P,t}) / sigma_{P,t}            # higher z = better, after direction-correction

Direction-correction flips the sign for 'lower-is-better' metrics so that higher
normalised values always mean 'better'. For the assistance composite we also
expose the **difficulty-relative AC** ``AC / E[AC_P,t]`` (1.0 = average help for
this task), which feeds ``H`` in :mod:`usabench.eval.composite`.

This module is pure: it takes the panel's raw metric values for a task plus the
candidate's raw value and returns normalised numbers; it performs no I/O.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from usabench.eval._common import safe_div

__all__ = [
    "PanelStats",
    "panel_stats",
    "zscore",
    "difficulty_relative",
    "LOWER_IS_BETTER",
    "normalize_metric",
    "Normalizer",
]

#: Registry metrics where a *lower* raw value is better (direction-corrected so
#: higher z = better). Aligned with the 'dir' column of ``docs/metrics.md``.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "A4_goal_drift",
        "B1_n_interventions",
        "B2_n_clarifications",
        "B3_n_hint_requests",
        "B4_n_corrections",
        "B5_n_handoffs",
        "B6_turns_to_first_working",
        "B7_turns_to_acceptance",
        "B8_interventions_to_acceptance",
        "B10_time_blocked_fraction",
        "C1_assistance_cost",
        "C2_max_severity",
        "C4_spec_info_transferred",
        "C5_solution_leakage",
        "E1_wall_clock_s",
        "E2_tokens_total",
        "E3_cost_usd_total",
        "E4_n_tool_calls",
        "E5_edit_churn",
        "E6_iterations",
        "E7_cost_per_progress",
        "E8_tokens_per_progress",
        "G2_redundant_query_rate",
        "G4_false_confidence",
    }
)


@dataclass(frozen=True)
class PanelStats:
    """Mean/std (and mean for ratios) of one metric over the panel on one task."""

    mean: float
    std: float
    n: int
    values: tuple[float, ...] = field(default_factory=tuple)


def panel_stats(values: Iterable[float]) -> PanelStats:
    """Compute :class:`PanelStats` from the panel's raw values for a metric/task.

    Args:
        values: Finite raw metric values from the reference panel on this task.

    Returns:
        A :class:`PanelStats`; ``std`` is the population std (0 if <2 values).
    """
    vals = [float(v) for v in values if v == v and v not in (float("inf"), float("-inf"))]
    if not vals:
        return PanelStats(mean=0.0, std=0.0, n=0, values=tuple())
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) >= 2 else 0.0
    return PanelStats(mean=mean, std=std, n=len(vals), values=tuple(vals))


def zscore(value: float, stats: PanelStats, *, lower_is_better: bool = False) -> float:
    """Direction-corrected z-score of ``value`` against ``stats``.

    When ``stats.std == 0`` (degenerate panel), returns ``0.0`` (the candidate is
    treated as exactly average -- no spurious signal from a single panel point).

    Args:
        value: The candidate's raw metric value.
        stats: The panel statistics for this metric/task.
        lower_is_better: If True, the sign is flipped so higher z = better.

    Returns:
        The (possibly sign-flipped) z-score.
    """
    if stats.std == 0.0:
        return 0.0
    z = (float(value) - stats.mean) / stats.std
    return -z if lower_is_better else z


def difficulty_relative(value: float, panel_mean: float) -> float:
    """``value / panel_mean`` (1.0 = average for this task); 0 if mean ~0."""
    return safe_div(float(value), float(panel_mean), default=0.0)


def normalize_metric(
    metric_id: str, value: float, panel_values: Iterable[float]
) -> float:
    """Direction-corrected z-score of a named metric against its panel values.

    Looks up the metric's direction in :data:`LOWER_IS_BETTER` automatically.
    """
    stats = panel_stats(panel_values)
    return zscore(value, stats, lower_is_better=metric_id in LOWER_IS_BETTER)


class Normalizer:
    """Holds a per-task reference panel and normalises candidate metrics against it.

    The panel is ``{metric_id: [panel raw values]}`` for ONE task. Build a
    :class:`Normalizer` per task; call :meth:`z` / :meth:`relative` per metric.
    """

    def __init__(self, panel: Mapping[str, Iterable[float]]) -> None:
        """Precompute panel statistics for every provided metric.

        Args:
            panel: Mapping ``metric_id -> iterable of panel raw values`` for one task.
        """
        self._stats: dict[str, PanelStats] = {
            mid: panel_stats(vals) for mid, vals in panel.items()
        }

    def stats_for(self, metric_id: str) -> PanelStats | None:
        """Return the cached :class:`PanelStats` for ``metric_id`` (or ``None``)."""
        return self._stats.get(metric_id)

    def z(self, metric_id: str, value: float) -> float:
        """Direction-corrected z-score of ``value`` for ``metric_id``."""
        stats = self._stats.get(metric_id)
        if stats is None:
            return 0.0
        return zscore(value, stats, lower_is_better=metric_id in LOWER_IS_BETTER)

    def relative(self, metric_id: str, value: float) -> float:
        """Difficulty-relative ratio ``value / panel_mean`` for ``metric_id``."""
        stats = self._stats.get(metric_id)
        if stats is None:
            return 0.0
        return difficulty_relative(value, stats.mean)

    def panel_mean(self, metric_id: str) -> float:
        """Panel mean for ``metric_id`` (0.0 if unknown)."""
        stats = self._stats.get(metric_id)
        return stats.mean if stats is not None else 0.0
