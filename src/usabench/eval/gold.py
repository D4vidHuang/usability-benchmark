"""A thin, uniform accessor over a task's frozen *gold* knowledge.

Metrics and scoring functions need read-only access to a handful of gold fields
(the weighted acceptance criteria, the hidden-spec info units, the ambiguity
points, the accept threshold). Callers may hold a full
:class:`~usabench.core.schema.Task`, a bare
:class:`~usabench.core.schema.HiddenSpec`, or a raw dict (e.g. freshly parsed
``task.yaml``). :class:`Gold` normalises all three into one interface so every
metric takes ``(trace, gold)`` without caring which shape it received.

The accessor never mutates its source and never exposes anything to the agent --
it is for the offline grader only.
"""

from __future__ import annotations

from typing import Any

from usabench.core.schema import (
    AcceptanceCriterion,
    AmbiguityPoint,
    HiddenSpec,
    InfoUnit,
    Task,
)

__all__ = ["Gold", "as_gold"]


class Gold:
    """Read-only adapter exposing the gold fields metrics consume.

    Construct via :func:`as_gold` (which is idempotent and accepts ``Task`` /
    ``HiddenSpec`` / ``dict`` / ``Gold``).
    """

    def __init__(
        self,
        *,
        hidden: HiddenSpec,
        accept_threshold: float = 0.80,
        task_id: str | None = None,
        difficulty: str | None = None,
    ) -> None:
        """Store the normalised gold.

        Args:
            hidden: The frozen :class:`HiddenSpec`.
            accept_threshold: Weighted-score acceptance threshold for this task.
            task_id: Optional task id (for diagnostics).
            difficulty: Optional difficulty tier string (T1..T4), for normalisation.
        """
        self.hidden = hidden
        self.accept_threshold = float(accept_threshold)
        self.task_id = task_id
        self.difficulty = difficulty

    # -- acceptance criteria views ---------------------------------------- #

    @property
    def criteria(self) -> list[AcceptanceCriterion]:
        """All weighted acceptance criteria (gold)."""
        return list(self.hidden.acceptance_criteria)

    @property
    def core_criteria(self) -> list[AcceptanceCriterion]:
        """Criteria flagged ``is_core`` (functional must-haves -> A3)."""
        return [c for c in self.hidden.acceptance_criteria if c.is_core]

    @property
    def hard_criteria(self) -> list[AcceptanceCriterion]:
        """Criteria flagged ``is_hard`` (gating constraints -> GA gate)."""
        return [c for c in self.hidden.acceptance_criteria if c.is_hard]

    def criteria_by_kind(self, kind: str) -> list[AcceptanceCriterion]:
        """Criteria whose ``check_kind`` value equals ``kind``."""
        out: list[AcceptanceCriterion] = []
        for c in self.hidden.acceptance_criteria:
            ck = c.check_kind.value if hasattr(c.check_kind, "value") else c.check_kind
            if str(ck) == kind:
                out.append(c)
        return out

    # -- hidden-spec views ------------------------------------------------ #

    @property
    def info_units(self) -> list[InfoUnit]:
        """Discrete hidden-spec info units (denominator for C4 / D5)."""
        return list(self.hidden.info_units)

    @property
    def n_hidden_spec_units(self) -> int:
        """Number of hidden-spec info units (>=0)."""
        return self.hidden.n_hidden_spec_units

    @property
    def ambiguity_points(self) -> list[AmbiguityPoint]:
        """Under-specifications the agent should surface by asking."""
        return list(self.hidden.ambiguity_points)

    def info_unit_ids(self) -> set[str]:
        """The set of all hidden-spec info-unit ids."""
        return {u.id for u in self.hidden.info_units}


def as_gold(source: Any) -> Gold:
    """Coerce ``source`` into a :class:`Gold` accessor (idempotent).

    Args:
        source: A :class:`Gold`, :class:`~usabench.core.schema.Task`,
            :class:`~usabench.core.schema.HiddenSpec`, or a raw ``dict`` shaped
            like ``task.yaml`` (with a nested ``hidden`` mapping) or like a
            ``HiddenSpec`` mapping directly.

    Returns:
        A :class:`Gold` accessor.

    Raises:
        TypeError: If ``source`` cannot be interpreted as gold.
    """
    if isinstance(source, Gold):
        return source
    if isinstance(source, Task):
        return Gold(
            hidden=source.hidden,
            accept_threshold=source.accept_threshold,
            task_id=source.id,
            difficulty=str(source.difficulty.value if hasattr(source.difficulty, "value") else source.difficulty),
        )
    if isinstance(source, HiddenSpec):
        return Gold(hidden=source)
    if isinstance(source, dict):
        if "hidden" in source and isinstance(source["hidden"], dict):
            hidden = HiddenSpec.model_validate(source["hidden"])
            return Gold(
                hidden=hidden,
                accept_threshold=float(source.get("accept_threshold", 0.80)),
                task_id=source.get("id"),
                difficulty=source.get("difficulty"),
            )
        # Treat the dict itself as a HiddenSpec mapping.
        hidden = HiddenSpec.model_validate(source)
        return Gold(hidden=hidden)
    raise TypeError(f"cannot interpret {type(source).__name__} as gold")
