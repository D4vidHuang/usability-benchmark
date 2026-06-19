"""Multi-ceiling budget metering.

The :class:`BudgetMeter` enforces the five independent ceilings of
``docs/protocol.md`` §1.3 -- turns, wall-clock seconds, AUT tokens, USD cost, and
oracle queries. The run terminates when *any* ceiling is hit. Every debit is
recorded so the scorer can reconstruct remaining budget at any event purely from
the trace (``budgets_after`` snapshots) -- nothing the scorer needs lives only in
memory (``DESIGN.md`` invariant 4).

The meter never writes the trace itself; the runner pulls debits/snapshots from it
and persists them via the :class:`~usabench.harness.interaction_bus.InteractionBus`.
This keeps budget accounting a pure, testable function of the debits applied.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from usabench.core.enums import TerminatedReason
from usabench.core.errors import BudgetExceeded
from usabench.core.schema import BudgetSnapshot

__all__ = ["BudgetLimits", "BudgetDebitRecord", "BudgetMeter"]


#: The five budget kinds, in a stable order matching ``BudgetSnapshot`` fields.
_KINDS: tuple[str, ...] = ("turn", "wall", "token", "cost", "oracle_query")

#: Map a budget kind onto the ``TerminatedReason`` raised when it is exhausted.
_KIND_TO_REASON: dict[str, TerminatedReason] = {
    "turn": TerminatedReason.BUDGET_TURNS,
    "wall": TerminatedReason.BUDGET_WALL,
    "token": TerminatedReason.BUDGET_TOKENS,
    "cost": TerminatedReason.BUDGET_COST,
    "oracle_query": TerminatedReason.BUDGET_ORACLE,
}


class BudgetLimits(BaseModel):
    """The five per-run budget ceilings (``docs/protocol.md`` §1.3 defaults).

    Attributes:
        max_turns: Maximum agent actions.
        max_wall_s: Maximum wall-clock seconds.
        max_tokens: Maximum AUT prompt+completion tokens.
        max_cost_usd: Maximum AUT cost in USD.
        max_oracle_queries: Maximum oracle exchanges.
    """

    model_config = ConfigDict(extra="forbid")

    max_turns: int = Field(80, ge=0)
    max_wall_s: float = Field(1800.0, ge=0.0)
    max_tokens: int = Field(600_000, ge=0)
    max_cost_usd: float = Field(3.0, ge=0.0)
    max_oracle_queries: int = Field(25, ge=0)

    def limit_for(self, kind: str) -> float:
        """Return the ceiling for a given budget ``kind``.

        Args:
            kind: One of ``turn|wall|token|cost|oracle_query``.

        Returns:
            The configured ceiling as a float.

        Raises:
            ValueError: If ``kind`` is not a known budget kind.
        """
        mapping = {
            "turn": float(self.max_turns),
            "wall": float(self.max_wall_s),
            "token": float(self.max_tokens),
            "cost": float(self.max_cost_usd),
            "oracle_query": float(self.max_oracle_queries),
        }
        if kind not in mapping:
            raise ValueError(f"unknown budget kind: {kind!r}")
        return mapping[kind]

    def as_payload(self) -> dict[str, float]:
        """Return a plain dict for the ``episode_start`` budgets payload."""
        return {
            "max_turns": self.max_turns,
            "max_wall_s": self.max_wall_s,
            "max_tokens": self.max_tokens,
            "max_cost_usd": self.max_cost_usd,
            "max_oracle_queries": self.max_oracle_queries,
        }


class BudgetDebitRecord(BaseModel):
    """A single applied debit, returned to the runner so it can log a trace event.

    Attributes:
        kind: The budget kind debited (``token|cost|turn|wall|oracle_query``).
        amount: The amount debited (delta).
        reason: A short human-readable reason (e.g. the action kind).
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    amount: float
    reason: str | None = None


class BudgetMeter:
    """Stateful, monotonic budget accumulator with five ceilings.

    The meter accumulates usage per kind and raises :class:`BudgetExceeded` the
    moment any ceiling is crossed. Wall-clock is tracked from a start timestamp so
    it advances even while no other debit happens; it is sampled lazily on every
    :meth:`snapshot` and :meth:`check`.

    Example:
        >>> meter = BudgetMeter(BudgetLimits(max_turns=2))
        >>> _ = meter.debit_turn()
        >>> _ = meter.debit_turn()
        >>> meter.is_exhausted()
        True
    """

    def __init__(self, limits: BudgetLimits, *, clock: Callable[[], float] | None = None) -> None:
        """Initialize the meter.

        Args:
            limits: The five ceilings to enforce.
            clock: Optional callable returning monotonic-ish seconds (for tests);
                defaults to :func:`time.monotonic`.
        """
        self.limits = limits
        self._used: dict[str, float] = {k: 0.0 for k in _KINDS}
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._start = float(self._clock())
        self._exhausted_reason: TerminatedReason | None = None

    # --- internal helpers --------------------------------------------------- #

    def _now(self) -> float:
        return float(self._clock())

    def _refresh_wall(self) -> None:
        """Sync the wall-clock usage to elapsed time since start."""
        self._used["wall"] = max(self._used["wall"], self._now() - self._start)

    def _enforce(self, kind: str) -> None:
        """Raise :class:`BudgetExceeded` if ``kind`` has crossed its ceiling."""
        limit = self.limits.limit_for(kind)
        used = self._used[kind]
        if used > limit:
            self._exhausted_reason = _KIND_TO_REASON[kind]
            raise BudgetExceeded(kind, limit, used)

    # --- debits ------------------------------------------------------------- #

    def debit(self, kind: str, amount: float, reason: str | None = None) -> BudgetDebitRecord:
        """Apply a debit of ``amount`` to ``kind`` and enforce its ceiling.

        Args:
            kind: One of ``token|cost|turn|wall|oracle_query``.
            amount: Non-negative delta to add to usage.
            reason: Optional reason logged with the debit.

        Returns:
            A :class:`BudgetDebitRecord` describing the applied debit (for tracing).

        Raises:
            ValueError: If ``kind`` is unknown or ``amount`` is negative.
            BudgetExceeded: If applying the debit crosses the ceiling for ``kind``.
        """
        if kind not in self._used:
            raise ValueError(f"unknown budget kind: {kind!r}")
        if amount < 0:
            raise ValueError(f"debit amount must be non-negative: {amount!r}")
        self._used[kind] += float(amount)
        record = BudgetDebitRecord(kind=kind, amount=float(amount), reason=reason)
        self._enforce(kind)
        return record

    def debit_turn(self, reason: str = "agent_action") -> BudgetDebitRecord:
        """Debit one agent turn."""
        return self.debit("turn", 1.0, reason)

    def debit_oracle_query(self, reason: str = "oracle_query") -> BudgetDebitRecord:
        """Debit one oracle exchange."""
        return self.debit("oracle_query", 1.0, reason)

    def debit_usage(
        self, tokens: int, cost_usd: float, reason: str = "agent_message"
    ) -> list[BudgetDebitRecord]:
        """Debit AUT token + cost usage from one LLM call.

        Tokens and cost are debited as two records (the trace keeps them as
        independent ``budget_debit`` kinds). Either may breach independently.

        Args:
            tokens: AUT prompt+completion tokens for the call.
            cost_usd: AUT USD cost for the call.
            reason: Reason logged with both debits.

        Returns:
            A list of the applied :class:`BudgetDebitRecord` (token first, then cost).

        Raises:
            BudgetExceeded: If either ceiling is crossed (token is checked first).
        """
        records: list[BudgetDebitRecord] = []
        if tokens:
            records.append(self.debit("token", float(tokens), reason))
        if cost_usd:
            records.append(self.debit("cost", float(cost_usd), reason))
        return records

    # --- queries ------------------------------------------------------------ #

    def check(self) -> None:
        """Re-evaluate every ceiling (incl. wall-clock) without applying a debit.

        Raises:
            BudgetExceeded: If any ceiling is currently exceeded.
        """
        self._refresh_wall()
        for kind in _KINDS:
            self._enforce(kind)

    def is_exhausted(self) -> bool:
        """Return True if any ceiling is currently at/over its limit (no raise)."""
        self._refresh_wall()
        for kind in _KINDS:
            if self._used[kind] >= self.limits.limit_for(kind):
                self._exhausted_reason = _KIND_TO_REASON[kind]
                return True
        return False

    @property
    def exhausted_reason(self) -> TerminatedReason | None:
        """The :class:`TerminatedReason` for the breached ceiling, if any."""
        return self._exhausted_reason

    def used(self, kind: str) -> float:
        """Return the amount used for ``kind`` (wall is refreshed first)."""
        if kind == "wall":
            self._refresh_wall()
        return self._used[kind]

    def snapshot(self) -> BudgetSnapshot:
        """Return a :class:`BudgetSnapshot` of current usage (``budgets_after``).

        Wall-clock is refreshed so the snapshot is current at call time.
        """
        self._refresh_wall()
        return BudgetSnapshot(
            turns=int(self._used["turn"]),
            wall_s=round(self._used["wall"], 6),
            tokens=int(self._used["token"]),
            cost_usd=round(self._used["cost"], 8),
            oracle_queries=int(self._used["oracle_query"]),
        )
