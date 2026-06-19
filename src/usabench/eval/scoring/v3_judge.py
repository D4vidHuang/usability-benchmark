"""V3 -- LLM-as-judge jury channel (``docs/scoring.md`` §6).

V3 grades the qualitative (``oracle_judgment`` / ``judge``-channel) acceptance
criteria a script cannot check. It is the noisiest channel, so it gets the most
safeguards and the lowest GA weight. This module implements the **aggregation and
bias-control logic** as pure functions over structured judge verdicts; the actual
LLM calls are produced by a :class:`JudgePanel` that *lazily* uses an
:class:`~usabench.llm.client.LLMClient` (so importing this module pulls in no
provider SDK).

Pipeline (``usability_score.yaml`` ``judge.*``):

1. **Pairwise vs reference, mapped to a point score** ``s_i in {0, 0.5, 1}``:
   ``comparable``/``better`` -> 1, ``worse_but_functional`` -> 0.5,
   ``missing``/``broken`` -> 0.
2. **Position-swap calibration** -- each pairwise judgment is run twice with the
   candidate/reference order swapped; keep the verdict only if consistent, else
   force ``s_i = 0.5`` and raise a ``judge_position_conflict`` flag.
3. **Jury of J=3 heterogeneous judges** -- per item take the **median** ``s_i``.
4. **Self-preference guard** -- a judge of the same model family as the agent is
   dropped from the panel for that run.
5. **Krippendorff's alpha** -- inter-judge agreement reported per task
   (:func:`krippendorff_alpha`), with a documented simple-agreement fallback.

V3 inherits the same rubric weights as V2; the V2/V3 split is purely *which
verifier produced the item score*::

    V3 = ( Σ_{i in judge} w_i * median_j(s_{i,j}) ) / ( Σ_{i in judge} w_i )
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from usabench.eval._common import clip01, spec_get
from usabench.eval.gold import as_gold

__all__ = [
    "PairwiseVerdict",
    "JudgeItemScore",
    "V3Result",
    "verdict_to_score",
    "resolve_position_swap",
    "jury_item_score",
    "krippendorff_alpha",
    "score_v3",
    "Judge",
    "JudgePanel",
]

#: Mapping from a pairwise verdict label to the point score s_i in {0,0.5,1}.
_VERDICT_SCORE: dict[str, float] = {
    "better": 1.0,
    "comparable": 1.0,
    "worse_but_functional": 0.5,
    "worse": 0.5,
    "missing": 0.0,
    "broken": 0.0,
}


@dataclass(frozen=True)
class PairwiseVerdict:
    """One judge's pairwise verdict on one rubric item, in one ordering.

    Attributes:
        item_id: The rubric/criterion id being judged.
        judge_id: Stable id of the judge model (e.g. ``"anthropic-judge"``).
        verdict: A label in :data:`_VERDICT_SCORE` (candidate vs reference).
        swapped: ``True`` if this judgment was made with reference shown first.
        rationale: Optional short justification (evidence quote).
    """

    item_id: str
    judge_id: str
    verdict: str
    swapped: bool = False
    rationale: str | None = None


@dataclass(frozen=True)
class JudgeItemScore:
    """The resolved per-item, per-judge score after position-swap calibration."""

    item_id: str
    judge_id: str
    score: float
    position_conflict: bool = False


@dataclass(frozen=True)
class V3Result:
    """The V3 channel score plus jury diagnostics."""

    score: float
    alpha: float | None
    n_judge_items: int = 0
    n_position_conflicts: int = 0
    per_item: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


def verdict_to_score(verdict: str) -> float:
    """Map a pairwise verdict label to ``s_i in {0, 0.5, 1}`` (unknown -> 0.5)."""
    return _VERDICT_SCORE.get(str(verdict).strip().lower(), 0.5)


def resolve_position_swap(verdicts: Sequence[PairwiseVerdict]) -> JudgeItemScore:
    """Resolve a judge's (possibly swapped) verdicts for ONE item into a score.

    If both orderings are present and agree, use their (equal) score. If they
    disagree, force ``0.5`` and flag a position conflict (``docs/scoring.md``
    §6.3). With a single ordering, take it at face value.

    Args:
        verdicts: All :class:`PairwiseVerdict` records for one (item, judge) pair.

    Returns:
        A :class:`JudgeItemScore`.

    Raises:
        ValueError: If ``verdicts`` is empty or spans multiple items/judges.
    """
    if not verdicts:
        raise ValueError("resolve_position_swap requires >=1 verdict")
    item_ids = {v.item_id for v in verdicts}
    judge_ids = {v.judge_id for v in verdicts}
    if len(item_ids) != 1 or len(judge_ids) != 1:
        raise ValueError("verdicts must share one item_id and one judge_id")
    item_id = next(iter(item_ids))
    judge_id = next(iter(judge_ids))

    normal = [v for v in verdicts if not v.swapped]
    swapped = [v for v in verdicts if v.swapped]
    if normal and swapped:
        s_n = verdict_to_score(normal[-1].verdict)
        s_s = verdict_to_score(swapped[-1].verdict)
        if s_n == s_s:
            return JudgeItemScore(item_id, judge_id, s_n, position_conflict=False)
        return JudgeItemScore(item_id, judge_id, 0.5, position_conflict=True)
    # Single ordering available.
    s = verdict_to_score(verdicts[-1].verdict)
    return JudgeItemScore(item_id, judge_id, s, position_conflict=False)


def jury_item_score(scores: Sequence[float]) -> float:
    """Median of the per-judge item scores (the jury aggregate, §6.3)."""
    if not scores:
        return 0.0
    return float(statistics.median(scores))


def krippendorff_alpha(
    ratings: Mapping[str, Mapping[str, float]],
    *,
    fallback_simple_agreement: bool = True,
) -> float | None:
    """Krippendorff's alpha (nominal) over judge ratings; pairwise-agreement based.

    ``ratings`` maps ``item_id -> {judge_id: score}``. We compute the nominal-scale
    alpha via the pairwise-coincidence form::

        alpha = 1 - D_o / D_e

    where ``D_o`` is observed disagreement and ``D_e`` is expected disagreement
    under chance, both over all judge-pairs within items. Scores are treated as
    nominal categories (values in ``{0, 0.5, 1}``).

    If ``scipy``/``numpy`` are unavailable we still compute alpha from first
    principles (no heavy deps required). When the design is too degenerate for a
    defined alpha (e.g. a single rated unit, or all judges constant), we fall back
    to **mean pairwise simple agreement** if ``fallback_simple_agreement`` else
    return ``None`` -- this is the documented 'simple agreement if scipy absent'
    behaviour, generalised to 'when alpha is undefined'.

    Returns:
        Alpha in ``(-inf, 1]`` (1 = perfect agreement), or a simple-agreement
        proxy in ``[0, 1]``, or ``None`` if nothing can be computed.
    """
    # Collect per-item judge values for units with >=2 ratings.
    units: list[list[float]] = []
    for _item, judge_map in ratings.items():
        vals = [float(v) for v in judge_map.values()]
        if len(vals) >= 2:
            units.append(vals)
    if not units:
        return None

    # All observed values -> nominal categories.
    all_vals: list[float] = [v for u in units for v in u]
    n_total = len(all_vals)
    if n_total < 2:
        return None

    # Observed disagreement: average over within-unit ordered pairs of [c != c'].
    obs_pairs = 0
    obs_disagree = 0
    for u in units:
        m = len(u)
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                obs_pairs += 1
                if u[a] != u[b]:
                    obs_disagree += 1
    if obs_pairs == 0:
        return None
    d_o = obs_disagree / obs_pairs

    # Expected disagreement: probability two random values (overall) differ.
    cat_counts: dict[float, int] = {}
    for v in all_vals:
        cat_counts[v] = cat_counts.get(v, 0) + 1
    same = 0
    for c in cat_counts.values():
        same += c * (c - 1)
    total_ordered = n_total * (n_total - 1)
    d_e = 1.0 - (same / total_ordered) if total_ordered > 0 else 0.0

    if d_e == 0.0:
        # All values identical => perfect agreement (alpha defined as 1.0).
        if d_o == 0.0:
            return 1.0
        if fallback_simple_agreement:
            return _mean_pairwise_agreement(units)
        return None
    alpha = 1.0 - (d_o / d_e)
    return float(alpha)


def _mean_pairwise_agreement(units: list[list[float]]) -> float:
    """Mean within-unit fraction of agreeing ordered judge-pairs."""
    agree = 0
    pairs = 0
    for u in units:
        m = len(u)
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                pairs += 1
                if u[a] == u[b]:
                    agree += 1
    return clip01(agree / pairs) if pairs else 1.0


def score_v3(
    verdicts: Iterable[PairwiseVerdict],
    gold: Any,
    *,
    item_weights: Mapping[str, float] | None = None,
) -> V3Result:
    """Compute the V3 jury score from raw pairwise verdicts.

    Args:
        verdicts: Every :class:`PairwiseVerdict` produced by the panel (all
            judges, both orderings, all judge-channel items).
        gold: Task gold; supplies the judge-channel criteria and their weights.
        item_weights: Optional override of per-item weights; defaults to the gold
            criteria weights.

    Returns:
        A :class:`V3Result`. Items with no verdicts are dropped from the
        denominator (abstention handling, ``docs/scoring.md`` §6.3), not scored 0.
    """
    g = as_gold(gold)
    judge_items = {c.id: float(c.weight) for c in g.criteria_by_kind("oracle_judgment")}
    if item_weights:
        judge_items.update({k: float(v) for k, v in item_weights.items()})

    # Group verdicts by (item, judge).
    by_pair: dict[tuple[str, str], list[PairwiseVerdict]] = {}
    for v in verdicts:
        by_pair.setdefault((v.item_id, v.judge_id), []).append(v)

    # Resolve each (item, judge) via position-swap, then jury-median per item.
    per_item_judge: dict[str, dict[str, float]] = {}
    n_conflicts = 0
    for (item_id, judge_id), vs in by_pair.items():
        resolved = resolve_position_swap(vs)
        if resolved.position_conflict:
            n_conflicts += 1
        per_item_judge.setdefault(item_id, {})[judge_id] = resolved.score

    per_item: dict[str, float] = {}
    for item_id, judge_map in per_item_judge.items():
        per_item[item_id] = jury_item_score(list(judge_map.values()))

    # Weighted aggregate over scored judge items only (abstention => excluded).
    num = 0.0
    den = 0.0
    for item_id, s in per_item.items():
        w = judge_items.get(item_id, 1.0)
        num += w * s
        den += w
    score = clip01(num / den) if den > 0 else 0.0

    alpha = krippendorff_alpha(per_item_judge)
    return V3Result(
        score=score,
        alpha=alpha,
        n_judge_items=len(per_item),
        n_position_conflicts=n_conflicts,
        per_item=per_item,
        detail={"judge_item_ids": list(judge_items)},
    )


# --------------------------------------------------------------------------- #
# Live judge panel (lazy LLM use)                                              #
# --------------------------------------------------------------------------- #


class Judge:
    """One judge model wrapped around an :class:`~usabench.llm.client.LLMClient`.

    The LLM is only invoked inside :meth:`judge_item`; constructing a Judge pulls
    in no provider SDK. The judge emits a single verdict label from
    :data:`_VERDICT_SCORE` for a (candidate, reference, item) triple.
    """

    def __init__(self, judge_id: str, client: Any, *, family: str | None = None) -> None:
        """Store the judge id, the LLM client, and its model family.

        Args:
            judge_id: Stable id for jury aggregation / alpha.
            client: An object satisfying the ``LLMClient`` protocol.
            family: Model family (e.g. ``"anthropic"``); used by the
                self-preference guard in :class:`JudgePanel`.
        """
        self.judge_id = judge_id
        self.client = client
        self.family = family

    def judge_item(
        self,
        *,
        item_id: str,
        item_text: str,
        candidate_view: str,
        reference_view: str,
        swapped: bool,
        temperature: float | None = None,
    ) -> PairwiseVerdict:
        """Ask the judge for ONE pairwise verdict (lazy LLM call).

        Args:
            item_id: The rubric item id.
            item_text: The item criterion text.
            candidate_view / reference_view: Rendered artifact summaries (+ exec
                output). In a swapped ordering the reference is presented first.
            swapped: Whether this is the order-swapped pass.
            temperature: Decoding temperature; defaults to ``judge.temperature``.

        Returns:
            A :class:`PairwiseVerdict`.
        """
        from usabench.llm.client import Message  # lazy import

        temp = float(spec_get("judge", "temperature", default=0.2)) if temperature is None else temperature
        first, second = (reference_view, candidate_view) if swapped else (candidate_view, reference_view)
        first_label, second_label = ("A", "B")
        prompt = (
            "You are grading a software artifact against ONE acceptance criterion. "
            "Ignore length, formatting flourish and confident tone; grade only the "
            "criterion against observed behavior + code.\n\n"
            f"Criterion: {item_text}\n\n"
            f"Artifact {first_label}:\n{first}\n\n"
            f"Artifact {second_label}:\n{second}\n\n"
            "Reply with exactly one of: better, comparable, worse_but_functional, "
            "missing, broken -- describing the CANDIDATE relative to the REFERENCE."
        )
        messages = [Message(role="user", content=prompt)]
        completion = self.client.chat(messages, temperature=temp, max_tokens=64)
        verdict = _parse_verdict(getattr(completion, "text", "") or "")
        return PairwiseVerdict(
            item_id=item_id,
            judge_id=self.judge_id,
            verdict=verdict,
            swapped=swapped,
            rationale=None,
        )


def _parse_verdict(text: str) -> str:
    """Extract the first recognised verdict label from raw judge text."""
    low = text.strip().lower()
    for label in _VERDICT_SCORE:
        if label in low:
            return label
    return "comparable"


class JudgePanel:
    """A heterogeneous jury of judges with the self-preference guard.

    Calling :meth:`collect_verdicts` runs every eligible judge over every
    judge-channel item in both orderings (when ``judge.position_swap``), producing
    the raw verdict list that :func:`score_v3` consumes. LLM use is entirely
    inside the judges' :meth:`Judge.judge_item`.
    """

    def __init__(self, judges: Sequence[Judge]) -> None:
        """Store the panel of judges (size need not equal ``judge.n``)."""
        self.judges = list(judges)

    def eligible_judges(self, agent_family: str | None) -> list[Judge]:
        """Drop judges whose family matches the agent under test (§6.3 guard)."""
        if not agent_family:
            return list(self.judges)
        return [j for j in self.judges if (j.family or "") != agent_family]

    def collect_verdicts(
        self,
        *,
        items: Mapping[str, str],
        candidate_view: str,
        reference_view: str,
        agent_family: str | None = None,
    ) -> list[PairwiseVerdict]:
        """Run the eligible jury over all items (both orderings) and collect verdicts.

        Args:
            items: ``{item_id: item_text}`` for the judge-channel criteria.
            candidate_view / reference_view: Rendered artifact summaries.
            agent_family: The agent-under-test's model family (for the guard).

        Returns:
            A flat list of :class:`PairwiseVerdict`.
        """
        swap = bool(spec_get("judge", "position_swap", default=True))
        orderings = [False, True] if swap else [False]
        out: list[PairwiseVerdict] = []
        for judge in self.eligible_judges(agent_family):
            for item_id, item_text in items.items():
                for swapped in orderings:
                    out.append(
                        judge.judge_item(
                            item_id=item_id,
                            item_text=item_text,
                            candidate_view=candidate_view,
                            reference_view=reference_view,
                            swapped=swapped,
                        )
                    )
        return out
