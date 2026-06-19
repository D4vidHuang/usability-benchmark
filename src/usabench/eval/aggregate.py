"""Seed/task aggregation, reliability estimators, and bootstrap CIs (``docs/metrics.md`` §7, ``docs/scoring.md`` §7).

This module turns per-(task,seed) numbers into the published, robust statistics:

* **median +/- IQR over seeds** -- every headline number is reported this way
  (a single-seed number is never published, ``docs/metrics.md`` §7.2).
* **Dimension F robustness:** ``pass^k`` (the strict 'all-of-k' reliability
  estimator ``C(c,k)/C(n,k)``) and ``pass@k`` (the unbiased 'any-of-k'
  estimator ``1 - C(n-c,k)/C(n,k)``), plus ``success_rate`` (F3), ``score_cv``
  (F4), ``assistance_cv`` (F5), ``usability_iqr`` (F6).
* **cluster bootstrap CI** -- resample tasks (clusters) then seeds within task to
  respect within-task correlation (``docs/scoring.md`` §7.2/§7.4).
* **paired comparison** -- a paired cluster-bootstrap difference test between two
  agents over shared tasks.

All constants (``stats.bootstrap_resamples``, ``stats.ci_level``, ``pass_k.k``,
``tau_ga``) come from ``usability_score.yaml``. Pure: no I/O, deterministic given
an explicit RNG seed.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from usabench.eval._common import clip01, spec_get

__all__ = [
    "MedianIQR",
    "median_iqr",
    "comb",
    "pass_hat_k",
    "pass_at_k",
    "success_rate",
    "coefficient_of_variation",
    "bootstrap_ci",
    "cluster_bootstrap_ci",
    "CIResult",
    "PairedResult",
    "paired_cluster_bootstrap",
    "SeedAggregate",
    "aggregate_seeds",
]


@dataclass(frozen=True)
class MedianIQR:
    """A median with its inter-quartile range and sample size."""

    median: float
    q1: float
    q3: float
    iqr: float
    n: int


def median_iqr(values: Sequence[float]) -> MedianIQR:
    """Median, Q1, Q3, and IQR of ``values`` (empty -> all zeros)."""
    vals = [float(v) for v in values]
    if not vals:
        return MedianIQR(0.0, 0.0, 0.0, 0.0, 0)
    s = sorted(vals)
    med = float(statistics.median(s))
    if len(s) == 1:
        return MedianIQR(med, s[0], s[0], 0.0, 1)
    q1, q3 = _quartiles(s)
    return MedianIQR(med, q1, q3, q3 - q1, len(s))


def _quartiles(sorted_vals: list[float]) -> tuple[float, float]:
    """Linear-interpolated Q1/Q3 (type-7, matching numpy default)."""

    def _percentile(p: float) -> float:
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        idx = p * (len(sorted_vals) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return sorted_vals[int(idx)]
        frac = idx - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    return _percentile(0.25), _percentile(0.75)


def comb(n: int, k: int) -> int:
    """Binomial coefficient ``C(n, k)`` (0 for invalid ``k``)."""
    if k < 0 or k > n or n < 0:
        return 0
    return math.comb(n, k)


def pass_hat_k(c: int, n: int, k: int) -> float:
    """Unbiased ``pass^k`` estimator: probability ALL k of k reruns succeed.

    ``pass^k = C(c, k) / C(n, k)`` with ``c`` successes among ``n`` reruns
    (``docs/metrics.md`` §7.2 / ``docs/scoring.md`` §7.2). Returns 0 if ``k > n``;
    returns 0 if ``c < k`` (cannot draw k successes).

    Args:
        c: Number of successful reruns.
        n: Total number of reruns (seeds).
        k: Subset size (``k <= n``).

    Returns:
        The probability that a random size-``k`` subset is all-successful.
    """
    if n <= 0 or k <= 0 or k > n:
        return 0.0
    denom = comb(n, k)
    if denom == 0:
        return 0.0
    return clip01(comb(c, k) / denom)


def pass_at_k(c: int, n: int, k: int) -> float:
    """Unbiased ``pass@k`` estimator: probability >=1 of k reruns succeeds.

    ``pass@k = 1 - C(n-c, k) / C(n, k)`` (the HumanEval estimator). Diagnostic
    only -- the benchmark headlines ``pass^k``, not ``pass@k``.
    """
    if n <= 0 or k <= 0 or k > n:
        return 0.0
    denom = comb(n, k)
    if denom == 0:
        return 0.0
    return clip01(1.0 - comb(n - c, k) / denom)


def success_rate(successes: Sequence[bool]) -> float:
    """F3 ``success_rate``: mean of boolean success over seeds."""
    if not successes:
        return 0.0
    return sum(1.0 for x in successes if x) / len(successes)


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Coefficient of variation ``std/|mean|`` (0 if mean ~0 or <2 values)."""
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return 0.0
    mean = statistics.fmean(vals)
    if mean == 0:
        return 0.0
    return statistics.pstdev(vals) / abs(mean)


@dataclass(frozen=True)
class CIResult:
    """A point estimate with a bootstrap confidence interval."""

    estimate: float
    lo: float
    hi: float
    level: float
    n_resamples: int


def _ci_bounds(samples: list[float], level: float) -> tuple[float, float]:
    """Percentile CI bounds at ``level`` from bootstrap ``samples``."""
    if not samples:
        return 0.0, 0.0
    s = sorted(samples)
    alpha = (1.0 - level) / 2.0
    lo = _interp_percentile(s, alpha)
    hi = _interp_percentile(s, 1.0 - alpha)
    return lo, hi


def _interp_percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolated percentile of a sorted list."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = p * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[int(idx)]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int | None = None,
    level: float | None = None,
    rng: random.Random | None = None,
) -> CIResult:
    """Simple (non-clustered) percentile bootstrap CI of the mean.

    Args:
        values: Flat sample.
        resamples: Bootstrap resamples (default ``stats.bootstrap_resamples``).
        level: Confidence level (default ``stats.ci_level``).
        rng: Optional seeded RNG for determinism.

    Returns:
        A :class:`CIResult` (estimate = sample mean).
    """
    vals = [float(v) for v in values]
    n_resamples = int(spec_get("stats", "bootstrap_resamples", default=10000)) if resamples is None else resamples
    lvl = float(spec_get("stats", "ci_level", default=0.95)) if level is None else level
    if not vals:
        return CIResult(0.0, 0.0, 0.0, lvl, n_resamples)
    r = rng or random.Random()
    est = statistics.fmean(vals)
    boots = [statistics.fmean(r.choices(vals, k=len(vals))) for _ in range(n_resamples)]
    lo, hi = _ci_bounds(boots, lvl)
    return CIResult(est, lo, hi, lvl, n_resamples)


def cluster_bootstrap_ci(
    clusters: Mapping[str, Sequence[float]],
    *,
    resamples: int | None = None,
    level: float | None = None,
    rng: random.Random | None = None,
) -> CIResult:
    """Cluster-bootstrap CI of the grand mean: resample tasks, then seeds.

    Respects within-task correlation by resampling whole tasks (clusters) with
    replacement, then resampling seeds within each chosen task (``docs/scoring.md``
    §7.2). The point estimate is the mean of per-cluster means.

    Args:
        clusters: Mapping ``task_id -> [per-seed values]``.
        resamples: Bootstrap resamples (default from spec).
        level: Confidence level (default from spec).
        rng: Optional seeded RNG.

    Returns:
        A :class:`CIResult`.
    """
    cluster_ids = [cid for cid, vals in clusters.items() if len(vals) > 0]
    n_resamples = int(spec_get("stats", "bootstrap_resamples", default=10000)) if resamples is None else resamples
    lvl = float(spec_get("stats", "ci_level", default=0.95)) if level is None else level
    if not cluster_ids:
        return CIResult(0.0, 0.0, 0.0, lvl, n_resamples)
    r = rng or random.Random()
    per_cluster_mean = {cid: statistics.fmean([float(v) for v in clusters[cid]]) for cid in cluster_ids}
    est = statistics.fmean([per_cluster_mean[cid] for cid in cluster_ids])

    boots: list[float] = []
    n_clusters = len(cluster_ids)
    for _ in range(n_resamples):
        chosen = r.choices(cluster_ids, k=n_clusters)
        means: list[float] = []
        for cid in chosen:
            seeds = [float(v) for v in clusters[cid]]
            resampled = r.choices(seeds, k=len(seeds))
            means.append(statistics.fmean(resampled))
        boots.append(statistics.fmean(means))
    lo, hi = _ci_bounds(boots, lvl)
    return CIResult(est, lo, hi, lvl, n_resamples)


@dataclass(frozen=True)
class PairedResult:
    """A paired difference (A - B) with a bootstrap CI and a two-sided p-value."""

    delta: float
    lo: float
    hi: float
    p_value: float
    level: float
    n_resamples: int


def paired_cluster_bootstrap(
    a: Mapping[str, Sequence[float]],
    b: Mapping[str, Sequence[float]],
    *,
    resamples: int | None = None,
    level: float | None = None,
    rng: random.Random | None = None,
) -> PairedResult:
    """Paired cluster-bootstrap test of ``mean(A) - mean(B)`` over shared tasks.

    Tasks present in both ``a`` and ``b`` are paired; clusters (tasks) are
    resampled with replacement, seeds within each task resampled, and the
    per-task A-vs-B difference of means averaged. The two-sided bootstrap p-value
    is ``2 * min(P(delta*<=0), P(delta*>=0))`` (``docs/scoring.md`` §7.4).

    Args:
        a: Agent A's ``task_id -> [per-seed statistic]``.
        b: Agent B's ``task_id -> [per-seed statistic]``.

    Returns:
        A :class:`PairedResult` for the A - B difference.
    """
    shared = [t for t in a if t in b and len(a[t]) > 0 and len(b[t]) > 0]
    n_resamples = int(spec_get("stats", "bootstrap_resamples", default=10000)) if resamples is None else resamples
    lvl = float(spec_get("stats", "ci_level", default=0.95)) if level is None else level
    if not shared:
        return PairedResult(0.0, 0.0, 0.0, 1.0, lvl, n_resamples)
    r = rng or random.Random()

    def _delta(task_ids: list[str], resample_seeds: bool) -> float:
        diffs: list[float] = []
        for t in task_ids:
            av = [float(v) for v in a[t]]
            bv = [float(v) for v in b[t]]
            if resample_seeds:
                av = r.choices(av, k=len(av))
                bv = r.choices(bv, k=len(bv))
            diffs.append(statistics.fmean(av) - statistics.fmean(bv))
        return statistics.fmean(diffs)

    point = _delta(shared, resample_seeds=False)
    boots: list[float] = []
    for _ in range(n_resamples):
        chosen = r.choices(shared, k=len(shared))
        boots.append(_delta(chosen, resample_seeds=True))
    lo, hi = _ci_bounds(boots, lvl)
    le = sum(1 for d in boots if d <= 0.0) / len(boots)
    ge = sum(1 for d in boots if d >= 0.0) / len(boots)
    p = min(1.0, 2.0 * min(le, ge))
    return PairedResult(point, lo, hi, p, lvl, n_resamples)


@dataclass(frozen=True)
class SeedAggregate:
    """The Dimension-F robustness block + central tendency over a task's seeds."""

    n_seeds: int
    n_success: int
    success_rate: float          # F3
    pass_hat_k: float            # F1 at k = pass_k.k
    pass_at_k: float             # F2 at k = pass_k.k (diagnostic)
    score_cv: float              # F4 (CV of A2)
    assistance_cv: float         # F5 (CV of C1)
    usability_median_iqr: MedianIQR  # F6 source (IQR of the headline)
    detail: dict[str, float] = field(default_factory=dict)


def aggregate_seeds(
    *,
    successes: Sequence[bool],
    a2_scores: Sequence[float],
    assistance_costs: Sequence[float],
    usability_scores: Sequence[float],
    k: int | None = None,
) -> SeedAggregate:
    """Aggregate one task's per-seed results into the Dimension-F block.

    Args:
        successes: Per-seed A1 success booleans (pass^k success definition).
        a2_scores: Per-seed A2 weighted scores (for F4 score CV).
        assistance_costs: Per-seed C1 assistance costs (for F5 assistance CV).
        usability_scores: Per-seed headline Usability Scores (for F6 IQR).
        k: ``pass^k`` subset size; defaults to ``pass_k.k`` from the spec.

    Returns:
        A :class:`SeedAggregate`.
    """
    n = len(successes)
    c = sum(1 for x in successes if x)
    kk = int(spec_get("pass_k", "k", default=2)) if k is None else k
    # If k exceeds available seeds, clamp to n so the estimator is still defined.
    k_eff = min(kk, n) if n > 0 else kk
    return SeedAggregate(
        n_seeds=n,
        n_success=c,
        success_rate=success_rate(successes),
        pass_hat_k=pass_hat_k(c, n, k_eff),
        pass_at_k=pass_at_k(c, n, k_eff),
        score_cv=coefficient_of_variation(a2_scores),
        assistance_cv=coefficient_of_variation(assistance_costs),
        usability_median_iqr=median_iqr(usability_scores),
        detail={"k": float(k_eff)},
    )
