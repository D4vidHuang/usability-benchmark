"""Composite Usability Scores (``docs/metrics.md`` §8 / ``docs/scoring.md`` §9).

Two composites, both reading the SAME success signal ``S = A3_core_criteria_score``
and the SAME normalised assistance ``AC = C1`` from ``usability_score.yaml``:

* **GEOMETRIC Usability Score (HEADLINE):**
    ``USABILITY = (S^alpha * H^beta)^(1/(alpha+beta)) * E^gamma * R^delta``
  Geometric coupling of success ``S`` and assistance-lightness ``H`` forces an
  agent to *both* succeed *and* do it with little help: either factor -> 0 kills
  the score.

* **MULTIPLICATIVE Usability (SECONDARY, documented):**
    ``U = GA * (1 - lambda*AC_norm) * (1 - fake_done_penalty?)``
  Reported for interpretability/ablation; reads the same GA(core)=S and the same
  normalised AC.

* **LINEAR variant (documented):** ``w . [S, H, A, E, R]``.

Inputs (all in ``[0,1]``, direction-corrected):
    S = A3 core-criteria score (capped success).
    H = 1 - min(1, AC_rel),  AC_rel = C1 / (kappa * E[C1]_panel);  H<-H*(1-rho) under the UnderAsk trap.
    A = D1 autonomy ratio.
    E = 1 / (1 + cost_per_progress / median_cost_per_progress_panel).
    R = F1 pass^k at k = pass_k.k.

Every constant (alpha, beta, gamma, delta, lambda, kappa, rho, tau, linear
weights, fake_done_penalty) is read from the spec; nothing is hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from usabench.eval._common import clip01, safe_div, spec_get

__all__ = [
    "CompositeInputs",
    "CompositeResult",
    "assistance_lightness",
    "efficiency_score",
    "apply_under_ask_penalty",
    "geometric_usability",
    "multiplicative_usability",
    "linear_usability",
    "compute_composite",
]


@dataclass(frozen=True)
class CompositeInputs:
    """The per-(task,seed) raw inputs to the composites.

    Attributes:
        s_core: ``S`` = A3 core-criteria score (capped success).
        assistance_cost: ``AC`` = C1 raw convex-weighted assistance cost (points).
        panel_mean_ac: ``E[C1]`` over the reference panel on this task (>=0). When
            unknown, pass the run's own ``assistance_cost`` and rely on ``kappa``.
        autonomy: ``A`` = D1 autonomy ratio.
        cost_per_progress: ``E7`` (USD per unit A2) for this run.
        panel_median_cost_per_progress: median of E7 over the reference panel.
        robustness: ``R`` = pass^k at ``k = pass_k.k`` (precomputed by aggregate).
        ga: Goal-Achievement (for the multiplicative secondary). Defaults to
            ``s_core`` if not supplied (both read the same core success).
        success_binary: A1; used by the UnderAsk trap.
        n_clarifications: B2; used by the UnderAsk trap.
        goal_drift: A4; used by the UnderAsk trap.
        fake_done: integrity flag; applies the multiplicative haircut.
    """

    s_core: float
    assistance_cost: float
    panel_mean_ac: float | None = None
    autonomy: float = 0.0
    cost_per_progress: float | None = None
    panel_median_cost_per_progress: float | None = None
    robustness: float = 1.0
    ga: float | None = None
    success_binary: int = 0
    n_clarifications: int = 0
    goal_drift: float = 0.0
    fake_done: bool = False


@dataclass(frozen=True)
class CompositeResult:
    """The headline + secondary composite outputs and the resolved sub-inputs."""

    usability_geometric: float
    usability_multiplicative: float
    usability_linear: float
    s: float
    h: float
    a: float
    e: float
    r: float
    under_ask_penalised: bool = False
    detail: dict[str, Any] = field(default_factory=dict)


def assistance_lightness(
    assistance_cost: float, panel_mean_ac: float | None
) -> float:
    """``H = 1 - min(1, AC_rel)`` with ``AC_rel = AC / (kappa * E[AC]_panel)``.

    When the panel mean is missing or zero, AC_rel is undefined; we treat a run
    with zero assistance as fully light (``H=1``) and any positive assistance as
    fully heavy relative to a zero baseline (``H=0``) only if no panel context is
    available -- callers that have a panel should always pass ``panel_mean_ac``.
    """
    kappa = float(spec_get("kappa", default=2.0))
    if assistance_cost <= 0:
        return 1.0
    if not panel_mean_ac or panel_mean_ac <= 0:
        # No panel baseline: any help with no reference scale saturates H to 0.
        return 0.0
    ac_rel = assistance_cost / (kappa * panel_mean_ac)
    return clip01(1.0 - min(1.0, ac_rel))


def efficiency_score(
    cost_per_progress: float | None,
    panel_median_cost_per_progress: float | None,
) -> float:
    """``E = 1 / (1 + cost_per_progress / median_cost_per_progress_panel)``.

    Returns ``1.0`` (no efficiency penalty) when either input is missing -- the
    efficiency discount is only meaningful relative to a panel baseline.
    """
    if cost_per_progress is None or not panel_median_cost_per_progress:
        return 1.0
    ratio = safe_div(cost_per_progress, panel_median_cost_per_progress, default=0.0)
    return clip01(1.0 / (1.0 + ratio))


def apply_under_ask_penalty(
    h: float, *, success_binary: int, n_clarifications: int, goal_drift: float
) -> tuple[float, bool]:
    """Apply the UnderAsk anti-gaming penalty to ``H`` (``docs/metrics.md`` §8.3).

    If ``success_binary == 0`` AND ``n_clarifications == 0`` AND ``goal_drift >
    tau``, scale ``H <- H * (1 - rho)`` -- you cannot bank the 'no help' reward
    while having built the wrong thing.

    Returns:
        ``(h_after, fired)`` where ``fired`` indicates the trap triggered.
    """
    rho = float(spec_get("under_ask_penalty", "rho", default=0.5))
    tau = float(spec_get("under_ask_penalty", "tau", default=0.5))
    if success_binary == 0 and n_clarifications == 0 and goal_drift > tau:
        return clip01(h * (1.0 - rho)), True
    return clip01(h), False


def geometric_usability(s: float, h: float, e: float, r: float) -> float:
    """Headline geometric Usability Score.

    ``(S^alpha * H^beta)^(1/(alpha+beta)) * E^gamma * R^delta`` with exponents from
    the spec. The S/H core is a weighted geometric mean (so either -> 0 kills it);
    E and R are multiplicative discounts.
    """
    alpha = float(spec_get("composite", "alpha", default=0.55))
    beta = float(spec_get("composite", "beta", default=0.45))
    gamma = float(spec_get("composite", "gamma", default=0.20))
    delta = float(spec_get("composite", "delta", default=0.20))
    s, h, e, r = clip01(s), clip01(h), clip01(e), clip01(r)
    denom = alpha + beta
    if denom <= 0:
        core = 0.0
    else:
        # Weighted geometric mean via exp/log; 0-safe (any zero factor -> 0).
        if s <= 0.0 or h <= 0.0:
            core = 0.0
        else:
            import math

            core = math.exp((alpha * math.log(s) + beta * math.log(h)) / denom)
    return clip01(core * (e ** gamma) * (r ** delta))


def multiplicative_usability(
    ga: float, ac_norm: float, *, fake_done: bool
) -> float:
    """Secondary multiplicative Usability ``U = GA*(1-lambda*AC)*(1-penalty?)``.

    Args:
        ga: Goal-Achievement (core success), in ``[0,1]``.
        ac_norm: Normalised assistance in ``[0,1]`` (1 = maximal help). This is
            ``1 - H`` so both composites share the same assistance scale.
        fake_done: If True, apply the flat ``fake_done_penalty`` haircut.

    Returns:
        ``U`` in ``[0,1]``.
    """
    lam = float(spec_get("multiplicative", "lambda", default=0.5))
    penalty = float(spec_get("multiplicative", "fake_done_penalty", default=0.25)) if fake_done else 0.0
    return clip01(clip01(ga) * (1.0 - lam * clip01(ac_norm)) * (1.0 - penalty))


def linear_usability(s: float, h: float, a: float, e: float, r: float) -> float:
    """Documented linear variant ``w . [S, H, A, E, R]`` (weights from spec)."""
    weights = [float(w) for w in spec_get("linear", "weights", default=[0.35, 0.30, 0.15, 0.10, 0.10])]
    vec = [clip01(s), clip01(h), clip01(a), clip01(e), clip01(r)]
    n = min(len(weights), len(vec))
    return clip01(sum(weights[i] * vec[i] for i in range(n)))


def compute_composite(inputs: CompositeInputs) -> CompositeResult:
    """Compute all three composites from one :class:`CompositeInputs`.

    Resolves the derived sub-inputs (H with the UnderAsk penalty, E, R), then
    evaluates the geometric headline, the multiplicative secondary, and the linear
    variant. The geometric and multiplicative forms read the SAME ``S`` and the
    SAME normalised assistance.

    Args:
        inputs: The per-(task,seed) raw inputs.

    Returns:
        A :class:`CompositeResult`.
    """
    s = clip01(inputs.s_core)
    h_raw = assistance_lightness(inputs.assistance_cost, inputs.panel_mean_ac)
    h, fired = apply_under_ask_penalty(
        h_raw,
        success_binary=inputs.success_binary,
        n_clarifications=inputs.n_clarifications,
        goal_drift=inputs.goal_drift,
    )
    a = clip01(inputs.autonomy)
    e = efficiency_score(inputs.cost_per_progress, inputs.panel_median_cost_per_progress)
    r = clip01(inputs.robustness)
    ga = clip01(inputs.ga) if inputs.ga is not None else s

    geo = geometric_usability(s, h, e, r)
    # Both composites share the assistance scale: ac_norm = 1 - H.
    mult = multiplicative_usability(ga, 1.0 - h, fake_done=inputs.fake_done)
    lin = linear_usability(s, h, a, e, r)
    return CompositeResult(
        usability_geometric=geo,
        usability_multiplicative=mult,
        usability_linear=lin,
        s=s,
        h=h,
        a=a,
        e=e,
        r=r,
        under_ask_penalised=fired,
        detail={"h_before_under_ask": h_raw, "ga": ga},
    )
