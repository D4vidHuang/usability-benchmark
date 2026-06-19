"""Task QC stage 4: grader-sanity (does the grader actually discriminate?).

A task is only useful if its grader separates good work from bad. This stage runs
the task's acceptance check against three *probe solutions* and asserts the grader
responds correctly (``docs/tasks.md`` §8.4):

* a **known-good** probe -> scores ~1.0 (>= ``good_min``);
* a **known-bad/empty** probe -> scores ~0.0 (<= ``bad_max``);
* an **ambiguity-trap** probe -> *fails the specific acceptance criteria tied to a
  high-severity ambiguity point*. This is the load-bearing check: it proves that
  ignoring a high-severity ambiguity (i.e. not asking the oracle and guessing
  wrong) actually breaks the grade, which is exactly the usability signal the
  benchmark measures.

Because the real functional grader runs inside the hermetic sandbox (owned by the
verification workstream, not the dataset workstream), this module works against a
**grader callable** abstraction: a ``Callable[[Solution, Task], AcceptanceResult]``.
The dataset side ships the *probes* and the *expectation table*; whoever owns the
sandbox passes in the real grader. A deterministic in-memory ``rubric_grader`` is
provided so the sanity logic is testable offline (it scores by keyword-matching the
probe's declared satisfied-criterion ids -- enough to exercise the discrimination
assertions without a container).

Pure offline; depends only on the foundation models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from usabench.core.schema import (
    AcceptanceResult,
    CriterionResult,
    HiddenSpec,
    Task,
)

__all__ = [
    "ProbeSolution",
    "GraderSanityResult",
    "Grader",
    "make_probe_set",
    "rubric_grader",
    "run_grader_sanity",
    "ambiguity_linked_criteria",
]

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class ProbeSolution:
    """A synthetic candidate solution used to probe the grader.

    Attributes:
        label: Probe name (``known_good`` | ``known_bad`` | ``ambiguity_trap``).
        satisfied_criteria: The acceptance-criterion ids this probe claims to pass
            (the in-memory :func:`rubric_grader` consults this; a real sandbox
            grader ignores it and executes the artifact instead).
        files: Optional file map (path -> content) for a real sandbox grader.
        note: Human description of what the probe represents.
    """

    label: str
    satisfied_criteria: set[str] = field(default_factory=set)
    files: dict[str, str] = field(default_factory=dict)
    note: str = ""


@dataclass(slots=True)
class GraderSanityResult:
    """The outcome of the three-probe grader-sanity check.

    Attributes:
        task_id: The task under test.
        good_score: Weighted score of the known-good probe.
        bad_score: Weighted score of the known-bad probe.
        trap_failed_linked: True if the ambiguity-trap probe failed the criteria
            linked to a high-severity ambiguity point (the desired behavior).
        linked_criteria: The criterion ids tied to high-severity ambiguities.
        errors: Hard discrimination failures.
        ok: True if all three probes behaved as required.
    """

    task_id: str
    good_score: float = 0.0
    bad_score: float = 0.0
    trap_failed_linked: bool = False
    linked_criteria: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if no hard discrimination errors were recorded."""
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "task_id": self.task_id,
            "ok": self.ok,
            "good_score": self.good_score,
            "bad_score": self.bad_score,
            "trap_failed_linked": self.trap_failed_linked,
            "linked_criteria": list(self.linked_criteria),
            "errors": list(self.errors),
        }


#: A grader: maps a probe solution + task to an :class:`AcceptanceResult`.
Grader = Callable[[ProbeSolution, Task], AcceptanceResult]


def ambiguity_linked_criteria(hidden: HiddenSpec) -> list[str]:
    """Find acceptance-criterion ids linked to a high-severity ambiguity point.

    Linkage is inferred two ways (in priority order):

    1. An explicit ``AP{n} -> AC{m}`` mapping in ``hidden.reveal_rules`` whose value
       names criterion ids.
    2. A naming convention: a high-severity ``AP{k}`` links to ``AC{k}`` when that
       criterion exists (the worked calendar example uses exactly this alignment,
       e.g. AP1 recurrence <-> AC3, AP2 timezone <-> AC5 -- so we also accept any
       criterion whose ``source`` or ``check_ref`` references the ambiguity id).

    Args:
        hidden: The task's hidden spec.

    Returns:
        A sorted, de-duplicated list of linked criterion ids.
    """
    ac_ids = {ac.id for ac in hidden.acceptance_criteria}
    high_aps = [ap.id for ap in hidden.ambiguity_points if ap.severity == "high"]
    linked: set[str] = set()

    # Resolve linkage PER high-severity ambiguity point, in priority order: an
    # explicit reveal_rules mapping or a source/check_ref back-reference is
    # authoritative; only when an AP has NO explicit link do we fall back to the
    # AP{k} <-> AC{k} naming convention (avoids over-linking when the author has
    # already wired the gold mapping, e.g. AP1->AC3, AP2->AC5).
    for ap_id in high_aps:
        explicit: set[str] = set()

        # 1. Explicit reveal_rules mapping (value may list criterion ids).
        rule = hidden.reveal_rules.get(ap_id)
        if isinstance(rule, str):
            explicit.update(tok for tok in rule.replace(",", " ").split() if tok in ac_ids)

        # 2. source/check_ref back-references from a criterion to this AP id.
        for ac in hidden.acceptance_criteria:
            refs = " ".join(filter(None, [ac.source, ac.check_ref]))
            if ap_id and ap_id in refs:
                explicit.add(ac.id)

        if explicit:
            linked.update(explicit)
            continue

        # 3. Convention fallback (only when nothing explicit linked this AP).
        digits = "".join(c for c in ap_id if c.isdigit())
        if digits:
            cand = f"AC{digits}"
            if cand in ac_ids:
                linked.add(cand)

    return sorted(linked)


def make_probe_set(task: Task) -> list[ProbeSolution]:
    """Build the canonical three-probe set for a task.

    Args:
        task: A finished task with authored acceptance criteria.

    Returns:
        ``[known_good, known_bad, ambiguity_trap]``.
    """
    all_ids = {ac.id for ac in task.hidden.acceptance_criteria}
    linked = set(ambiguity_linked_criteria(task.hidden))
    return [
        ProbeSolution(
            label="known_good",
            satisfied_criteria=set(all_ids),
            note="A correct reference solution; should pass everything.",
        ),
        ProbeSolution(
            label="known_bad",
            satisfied_criteria=set(),
            note="An empty/broken solution; should fail everything.",
        ),
        ProbeSolution(
            label="ambiguity_trap",
            # Passes everything EXCEPT the high-severity-ambiguity-linked criteria,
            # modeling an agent that guessed wrong instead of asking.
            satisfied_criteria=set(all_ids) - linked,
            note="Ignores a high-severity ambiguity; must fail its linked criteria.",
        ),
    ]


def rubric_grader(probe: ProbeSolution, task: Task) -> AcceptanceResult:
    """A deterministic in-memory grader for offline sanity testing.

    Scores by treating ``probe.satisfied_criteria`` as ground truth (a real sandbox
    grader executes the artifact instead). Computes the weighted score, the
    core-criteria score, and the hard-constraint pass fraction the same way the real
    :class:`AcceptanceResult` is shaped.

    Args:
        probe: The probe solution.
        task: The task whose criteria are graded.

    Returns:
        An :class:`AcceptanceResult`.
    """
    acs = task.hidden.acceptance_criteria
    results: list[CriterionResult] = []
    total_w = 0.0
    got_w = 0.0
    core_total = 0.0
    core_got = 0.0
    hard_total = 0
    hard_pass = 0
    for ac in acs:
        passed = ac.id in probe.satisfied_criteria
        results.append(CriterionResult(id=ac.id, passed=passed, score=1.0 if passed else 0.0))
        w = float(ac.weight)
        total_w += w
        if passed:
            got_w += w
        if ac.is_core:
            core_total += w
            if passed:
                core_got += w
        if ac.is_hard:
            hard_total += 1
            if passed:
                hard_pass += 1
    weighted = (got_w / total_w) if total_w else 0.0
    core = (core_got / core_total) if core_total else weighted
    hard_frac = (hard_pass / hard_total) if hard_total else 1.0
    return AcceptanceResult(
        criteria=results,
        weighted_score=round(weighted, 6),
        core_criteria_score=round(core, 6),
        hard_pass_frac=round(hard_frac, 6),
        accepted=weighted >= float(task.accept_threshold),
    )


def run_grader_sanity(
    task: Task,
    *,
    grader: Grader | None = None,
    good_min: float = 0.99,
    bad_max: float = 0.10,
) -> GraderSanityResult:
    """Run the three-probe grader-sanity check (``docs/tasks.md`` §8.4).

    Asserts the grader discriminates: good ~1.0, bad ~0.0, and the ambiguity-trap
    fails its high-severity-linked criteria. With no ``grader`` supplied, the
    deterministic :func:`rubric_grader` is used so the logic is testable offline.

    Args:
        task: A finished task with authored acceptance criteria.
        grader: Optional real grader callable; defaults to :func:`rubric_grader`.
        good_min: Minimum weighted score the known-good probe must reach.
        bad_max: Maximum weighted score the known-bad probe may reach.

    Returns:
        A :class:`GraderSanityResult`.
    """
    g = grader or rubric_grader
    result = GraderSanityResult(task_id=task.id)
    result.linked_criteria = ambiguity_linked_criteria(task.hidden)

    if not task.hidden.acceptance_criteria:
        result.errors.append("no acceptance_criteria to grade")
        return result

    probes = {p.label: p for p in make_probe_set(task)}
    good = g(probes["known_good"], task)
    bad = g(probes["known_bad"], task)
    trap = g(probes["ambiguity_trap"], task)

    result.good_score = good.weighted_score
    result.bad_score = bad.weighted_score

    if good.weighted_score < good_min:
        result.errors.append(
            f"known-good scored {good.weighted_score:.3f} < good_min {good_min}"
        )
    if bad.weighted_score > bad_max:
        result.errors.append(f"known-bad scored {bad.weighted_score:.3f} > bad_max {bad_max}")

    if not result.linked_criteria:
        result.errors.append(
            "no acceptance criterion is linked to a high-severity ambiguity "
            "(task may not be intervention-bearing / discriminative)"
        )
    else:
        passed_in_trap = {cr.id for cr in trap.criteria if cr.passed}
        still_passing = [c for c in result.linked_criteria if c in passed_in_trap]
        result.trap_failed_linked = not still_passing
        if still_passing:
            result.errors.append(
                "ambiguity-trap probe still passed linked criteria "
                f"{still_passing} (grader does not punish wrong guess)"
            )

    return result
