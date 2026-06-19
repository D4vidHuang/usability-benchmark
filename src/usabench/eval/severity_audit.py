"""Offline oracle-severity re-grading + inter-rater agreement (``docs/metrics.md`` §7 / §3.1).

The oracle policy self-declares a 0-5 assistance severity on every
``oracle_response`` (the convex-weighted scale that drives C1). That self-label is
a *validity risk*: an oracle that systematically under-reports severity would
inflate every agent's assistance-lightness. We therefore re-grade the severities
**offline** with an independent rater and report the agreement between the two
label streams as a **benchmark-validity number** -- specifically **Cohen's kappa**
(``docs/metrics.md`` §7 / ``docs/scoring.md`` ``oracle_severity_audit``).

This module is pure and dependency-free. It provides:

* :func:`extract_severity_labels` -- pull the oracle's self-declared severities
  (and the response texts / reveals) from ``trace.jsonl`` in order.
* :func:`cohen_kappa` -- Cohen's kappa between two equal-length label streams
  (quadratic-weighted option for the ordinal 0-5 scale), implemented from first
  principles (no scipy/sklearn).
* :func:`SeverityRater` protocol + :class:`RubricSeverityRater` -- a deterministic,
  LLM-free re-grader that maps a response's *observable* reveal content onto a 0-5
  severity using the same rubric the oracle is bound to (``docs/metrics.md`` §3.1).
  A live LLM re-grader can satisfy the same protocol; the SDK is never imported
  here.
* :func:`audit_severities` -- run a rater over a trace and return the agreement
  report (kappa + confusion + disagreement detail).

The audit is what flags an oracle for redesign, not what scores an agent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from usabench.core.enums import Severity
from usabench.core.schema import TraceEnvelope
from usabench.eval._common import (
    clip01,
    iter_events,
    oracle_responses,
    payload_severity,
)

__all__ = [
    "SeverityLabel",
    "extract_severity_labels",
    "cohen_kappa",
    "weighted_cohen_kappa",
    "SeverityRater",
    "RubricSeverityRater",
    "SeverityAuditResult",
    "audit_severities",
]


@dataclass(frozen=True)
class SeverityLabel:
    """One oracle response's auditable fields (the unit the rater re-grades)."""

    event_id: str
    seq: int
    declared: int
    text: str = ""
    reveals: tuple[str, ...] = field(default_factory=tuple)
    info_units_revealed: tuple[str, ...] = field(default_factory=tuple)
    provenance_tag: str | None = None


def extract_severity_labels(trace: Iterable[TraceEnvelope]) -> list[SeverityLabel]:
    """Pull the oracle's self-declared severity labels from a trace, in order.

    Args:
        trace: Parsed ``trace.jsonl`` events.

    Returns:
        One :class:`SeverityLabel` per ``oracle_response`` event, ordered by
        ``seq``.
    """
    out: list[SeverityLabel] = []
    for ev in oracle_responses(iter_events(trace)):
        p = ev.payload
        out.append(
            SeverityLabel(
                event_id=str(ev.event_id),
                seq=int(ev.seq),
                declared=payload_severity(p),
                text=str(getattr(p, "text", "") or ""),
                reveals=tuple(getattr(p, "reveals", []) or []),
                info_units_revealed=tuple(getattr(p, "info_units_revealed", []) or []),
                provenance_tag=getattr(p, "provenance_tag", None),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Cohen's kappa                                                               #
# --------------------------------------------------------------------------- #


def cohen_kappa(a: Sequence[int], b: Sequence[int], *, n_categories: int = 6) -> float | None:
    """Cohen's kappa (nominal) between two equal-length label streams.

    ``kappa = (p_o - p_e) / (1 - p_e)`` where ``p_o`` is observed agreement and
    ``p_e`` is chance agreement from the marginal rater distributions. Implemented
    from first principles -- no scipy/sklearn.

    Args:
        a: First rater's labels (ints in ``[0, n_categories)``).
        b: Second rater's labels (same length as ``a``).
        n_categories: Number of label categories (severity scale -> 6).

    Returns:
        Kappa in ``[-1, 1]`` (1 = perfect agreement). ``None`` if the streams are
        empty or length-mismatched. When ``p_e == 1`` (both raters constant on the
        same single category), returns ``1.0`` if they fully agree else ``0.0``.
    """
    if len(a) != len(b) or len(a) == 0:
        return None
    n = len(a)
    agree = sum(1 for x, y in zip(a, b, strict=False) if int(x) == int(y))
    p_o = agree / n

    count_a = [0] * n_categories
    count_b = [0] * n_categories
    for x, y in zip(a, b, strict=False):
        ix, iy = int(x), int(y)
        if 0 <= ix < n_categories:
            count_a[ix] += 1
        if 0 <= iy < n_categories:
            count_b[iy] += 1
    p_e = sum((count_a[c] / n) * (count_b[c] / n) for c in range(n_categories))

    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def weighted_cohen_kappa(
    a: Sequence[int], b: Sequence[int], *, n_categories: int = 6
) -> float | None:
    """Quadratic-weighted Cohen's kappa for the ordinal 0-5 severity scale.

    Quadratic weights ``w_ij = (i-j)^2 / (n_categories-1)^2`` penalise larger
    severity disagreements more (a declared-5/regraded-1 gap matters far more than
    declared-2/regraded-1). ``kappa_w = 1 - (Σ w_ij O_ij) / (Σ w_ij E_ij)``.

    Args:
        a: First rater's labels.
        b: Second rater's labels.
        n_categories: Number of ordinal categories (severity scale -> 6).

    Returns:
        Weighted kappa, or ``None`` for empty/mismatched input. Returns ``1.0``
        when there is no expected weighted disagreement and observed is also 0.
    """
    if len(a) != len(b) or len(a) == 0:
        return None
    n = len(a)
    nc = n_categories
    denom_scale = float((nc - 1) ** 2) if nc > 1 else 1.0

    observed = [[0.0] * nc for _ in range(nc)]
    count_a = [0] * nc
    count_b = [0] * nc
    for x, y in zip(a, b, strict=False):
        ix, iy = int(x), int(y)
        if 0 <= ix < nc and 0 <= iy < nc:
            observed[ix][iy] += 1.0
            count_a[ix] += 1
            count_b[iy] += 1

    num = 0.0
    den = 0.0
    for i in range(nc):
        for j in range(nc):
            w = ((i - j) ** 2) / denom_scale
            o_ij = observed[i][j] / n
            e_ij = (count_a[i] / n) * (count_b[j] / n)
            num += w * o_ij
            den += w * e_ij
    if den == 0.0:
        return 1.0 if num == 0.0 else 0.0
    return 1.0 - (num / den)


# --------------------------------------------------------------------------- #
# Re-graders                                                                   #
# --------------------------------------------------------------------------- #


class SeverityRater(Protocol):
    """An independent re-grader of an oracle response's severity (0-5)."""

    def regrade(self, label: SeverityLabel) -> int:
        """Return a re-graded severity in ``[0, 5]`` for one response."""
        ...


class RubricSeverityRater:
    """A deterministic, LLM-free re-grader keyed off *observable* reveal content.

    It applies the §3.1 rubric mechanically from the logged fields, so the audit
    can run fully offline and is itself unit-testable:

    * a ``provenance_tag`` marking an oracle-contributed solution span, or a reveal
      id namespaced ``solution:``/``code:``/``takeover:`` -> sev 4-5;
    * a revealed *requirement* / *constraint* info unit -> sev 2 (substantive);
    * a revealed hint id (``hint:``/``pitfall:``/``approach:``) -> sev 3;
    * a revealed *preference* / a non-empty answer with no new spec -> sev 1;
    * an empty / acknowledgement-only response with no reveals -> sev 0.

    Where multiple rules apply, the **maximum** severity wins (the rubric forbids
    under-reporting). The result is the rater's independent label, NOT the oracle's
    self-declared one -- disagreement with the declared label is exactly the audit
    signal.

    A live LLM re-grader satisfying :class:`SeverityRater` can be swapped in; this
    class is the documented default that needs no provider SDK.
    """

    #: Reveal-id prefixes that imply a solution fragment / takeover (sev 4+).
    _SOLUTION_PREFIXES: tuple[str, ...] = ("solution:", "code:", "takeover:", "patch:")
    #: Reveal-id prefixes that imply a directional hint (sev 3).
    _HINT_PREFIXES: tuple[str, ...] = ("hint:", "pitfall:", "approach:", "concept:")
    #: Reveal-id prefixes that imply substantive spec info (sev 2).
    _SPEC_PREFIXES: tuple[str, ...] = ("req:", "requirement:", "con:", "constraint:")
    #: Reveal-id prefixes that imply trivial clarification of a preference (sev 1).
    _PREF_PREFIXES: tuple[str, ...] = ("pref:", "preference:", "clarify:")

    def regrade(self, label: SeverityLabel) -> int:
        """Re-grade one response's severity from its observable reveal content."""
        sev = 0

        if label.provenance_tag:
            sev = max(sev, int(Severity.PARTIAL_SOLUTION))

        ids = [str(x).lower() for x in (*label.reveals, *label.info_units_revealed)]
        for rid in ids:
            if rid.startswith(self._SOLUTION_PREFIXES):
                sev = max(sev, int(Severity.PARTIAL_SOLUTION))
            elif rid.startswith(self._HINT_PREFIXES):
                sev = max(sev, int(Severity.DIRECTIONAL_HINT))
            elif rid.startswith(self._SPEC_PREFIXES):
                sev = max(sev, int(Severity.SUBSTANTIVE_SPEC_INFO))
            elif rid.startswith(self._PREF_PREFIXES):
                sev = max(sev, int(Severity.TRIVIAL_CLARIFICATION))
            else:
                # An unclassifiable reveal still transferred *some* information.
                sev = max(sev, int(Severity.SUBSTANTIVE_SPEC_INFO))

        if sev == 0 and label.text.strip():
            # A non-empty answer with no structured reveal is at least a trivial
            # clarification (it restated/disambiguated something).
            sev = max(sev, int(Severity.TRIVIAL_CLARIFICATION))
        return int(min(sev, int(Severity.TAKEOVER)))


@dataclass(frozen=True)
class SeverityAuditResult:
    """The oracle-severity audit report for one trace (or a pooled set)."""

    n: int
    kappa: float | None
    weighted_kappa: float | None
    observed_agreement: float
    mean_declared: float
    mean_regraded: float
    declared: tuple[int, ...]
    regraded: tuple[int, ...]
    confusion: tuple[tuple[int, ...], ...]
    disagreements: tuple[tuple[str, int, int], ...]
    detail: dict[str, Any] = field(default_factory=dict)


def audit_severities(
    trace: Iterable[TraceEnvelope],
    rater: SeverityRater | None = None,
) -> SeverityAuditResult:
    """Re-grade a trace's oracle severities and report agreement.

    Args:
        trace: Parsed ``trace.jsonl`` events.
        rater: An independent :class:`SeverityRater`; defaults to
            :class:`RubricSeverityRater` (deterministic, no LLM).

    Returns:
        A :class:`SeverityAuditResult` with Cohen's kappa, quadratic-weighted
        kappa, the 6x6 confusion matrix, and per-event disagreements.
    """
    r = rater if rater is not None else RubricSeverityRater()
    labels = extract_severity_labels(trace)
    declared = [lab.declared for lab in labels]
    regraded = [int(r.regrade(lab)) for lab in labels]

    n = len(labels)
    confusion = [[0] * 6 for _ in range(6)]
    disagreements: list[tuple[str, int, int]] = []
    agree = 0
    for lab, d, g in zip(labels, declared, regraded, strict=False):
        if 0 <= d <= 5 and 0 <= g <= 5:
            confusion[d][g] += 1
        if d == g:
            agree += 1
        else:
            disagreements.append((lab.event_id, d, g))

    return SeverityAuditResult(
        n=n,
        kappa=cohen_kappa(declared, regraded),
        weighted_kappa=weighted_cohen_kappa(declared, regraded),
        observed_agreement=clip01(agree / n) if n else 0.0,
        mean_declared=(sum(declared) / n) if n else 0.0,
        mean_regraded=(sum(regraded) / n) if n else 0.0,
        declared=tuple(declared),
        regraded=tuple(regraded),
        confusion=tuple(tuple(row) for row in confusion),
        disagreements=tuple(disagreements),
        detail={"rater": type(r).__name__},
    )
