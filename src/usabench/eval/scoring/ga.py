"""Goal-Achievement (GA) blend + hard-constraint gate (``docs/scoring.md`` §5).

GA combines the three verification channels and caps the result by a smooth
hard-constraint gate::

    GA_raw = w_v1*V1 + w_v2*V2 + w_v3*V3        (channel weights sum to 1)
    GA     = GA_raw * gate(hard_pass_frac)
    gate(h)= floor + slope*h                    (floor=0.30, slope=0.70)

All five constants are read from ``usability_score.yaml`` (``ga_channels.*`` and
``gate.*``) -- none is hardcoded. ``gate(0) = floor`` (missing all hard
constraints caps GA at 30% of GA_raw, smooth not a cliff); ``gate(1) = 1.0`` (no
penalty when every hard constraint is met).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from usabench.eval._common import clip01, spec_get

__all__ = ["GAResult", "ga_channel_weights", "gate", "compute_ga"]


@dataclass(frozen=True)
class GAResult:
    """The composed Goal-Achievement score and its inputs."""

    ga: float
    ga_raw: float
    v1: float
    v2: float
    v3: float
    hard_pass_frac: float
    gate: float
    weights: dict[str, float] = field(default_factory=dict)


def ga_channel_weights() -> tuple[float, float, float]:
    """Return ``(w_v1, w_v2, w_v3)`` from the spec (defaults 0.40/0.35/0.25)."""
    w1 = float(spec_get("ga_channels", "v1", default=0.40))
    w2 = float(spec_get("ga_channels", "v2", default=0.35))
    w3 = float(spec_get("ga_channels", "v3", default=0.25))
    return w1, w2, w3


def gate(hard_pass_frac: float) -> float:
    """The smooth hard-constraint gate ``floor + slope*h``.

    Args:
        hard_pass_frac: Fraction of hard constraints met, ``h in [0,1]``.

    Returns:
        The multiplicative gate value in ``[floor, floor+slope]`` (defaults
        ``[0.30, 1.00]``).
    """
    floor = float(spec_get("gate", "floor", default=0.30))
    slope = float(spec_get("gate", "slope", default=0.70))
    h = clip01(hard_pass_frac)
    return floor + slope * h


def compute_ga(
    v1: float,
    v2: float,
    v3: float,
    hard_pass_frac: float,
    *,
    weights: tuple[float, float, float] | None = None,
) -> GAResult:
    """Blend V1/V2/V3 and apply the hard-constraint gate.

    Args:
        v1: V1 functional/sandbox score in ``[0,1]``.
        v2: V2 rubric score in ``[0,1]``.
        v3: V3 judge-jury score in ``[0,1]``.
        hard_pass_frac: Fraction of hard constraints met (drives the gate).
        weights: Optional ``(w_v1, w_v2, w_v3)`` override; defaults to the spec.

    Returns:
        A :class:`GAResult` with ``ga``, ``ga_raw``, the gate, and the inputs.
    """
    w1, w2, w3 = weights if weights is not None else ga_channel_weights()
    v1, v2, v3 = clip01(v1), clip01(v2), clip01(v3)
    ga_raw = clip01(w1 * v1 + w2 * v2 + w3 * v3)
    g = gate(hard_pass_frac)
    ga = clip01(ga_raw * g)
    return GAResult(
        ga=ga,
        ga_raw=ga_raw,
        v1=v1,
        v2=v2,
        v3=v3,
        hard_pass_frac=clip01(hard_pass_frac),
        gate=g,
        weights={"v1": w1, "v2": w2, "v3": w3},
    )
