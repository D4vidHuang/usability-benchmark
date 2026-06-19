"""Token + cost accounting for LLM calls.

Every concrete :class:`~usabench.llm.client.LLMClient` records the tokens and cost
of each call here. The two public abstractions are:

* :class:`PriceTable` -- converts ``(prompt_tokens, completion_tokens)`` into a USD
  cost from a per-million-token price (read from the model config's
  ``price_per_mtok``). vLLM/local models price at ``0.0``.
* :class:`UsageMeter` -- a running accumulator that rolls per-call
  :class:`~usabench.core.schema.Usage` deltas into agent-vs-oracle totals and
  exposes a :class:`~usabench.harness.budget.BudgetMeter`-compatible surface
  (``tokens``/``cost_usd`` totals + a ``charge`` hook). The harness owns the real
  ``BudgetMeter``; this meter is the thing the LLM layer mutates so cost is
  attributed at the call site, split by :class:`Channel`.

Nothing here imports a provider SDK; it is pure arithmetic over the normalized
:class:`Usage` model, so it is safe to import with only core deps.
"""

from __future__ import annotations

import enum
import threading
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from usabench.core.schema import Usage

__all__ = [
    "Channel",
    "PriceTable",
    "ChannelTotals",
    "UsageMeter",
    "estimate_cost_usd",
]


class Channel(enum.StrEnum):
    """Which role a billed LLM call belongs to.

    The benchmark must separate the cost/tokens the *agent under test* spends from
    what the *oracle* (simulated user) and *judges* spend, because assistance and
    budgets are scored per channel (``docs/scoring.md``).
    """

    AGENT = "agent"
    ORACLE = "oracle"
    JUDGE = "judge"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class PriceTable(BaseModel):
    """Per-million-token USD prices for one model.

    Mirrors the ``price_per_mtok: {input, output}`` block in a model config. Local
    vLLM models leave both at ``0.0`` (their cost is GPU wall-time, tracked
    elsewhere).

    Attributes:
        input_per_mtok: USD per 1,000,000 prompt tokens.
        output_per_mtok: USD per 1,000,000 completion tokens.
    """

    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float = Field(0.0, ge=0.0)
    output_per_mtok: float = Field(0.0, ge=0.0)

    @classmethod
    def from_config(cls, price_per_mtok: dict[str, Any] | None) -> PriceTable:
        """Build a price table from a config's ``price_per_mtok`` mapping.

        Args:
            price_per_mtok: A mapping with ``input``/``output`` keys (either may be
                absent, defaulting to ``0.0``). ``None`` yields a free table.

        Returns:
            A :class:`PriceTable`.
        """
        if not price_per_mtok:
            return cls()
        return cls(
            input_per_mtok=float(price_per_mtok.get("input", 0.0)),
            output_per_mtok=float(price_per_mtok.get("output", 0.0)),
        )

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute the USD cost of a single call.

        Args:
            prompt_tokens: Number of prompt/input tokens billed.
            completion_tokens: Number of completion/output tokens billed.

        Returns:
            The USD cost (``0.0`` for free/local models).
        """
        return (
            prompt_tokens * self.input_per_mtok
            + completion_tokens * self.output_per_mtok
        ) / 1_000_000.0


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    price_per_mtok: dict[str, Any] | None,
) -> float:
    """Convenience one-shot cost estimate from a raw config price block.

    Args:
        prompt_tokens: Prompt token count.
        completion_tokens: Completion token count.
        price_per_mtok: The model config's ``price_per_mtok`` mapping (or ``None``).

    Returns:
        The USD cost.
    """
    return PriceTable.from_config(price_per_mtok).cost_usd(prompt_tokens, completion_tokens)


class ChannelTotals(BaseModel):
    """Accumulated usage for one :class:`Channel`."""

    model_config = ConfigDict(extra="forbid")

    calls: int = Field(0, ge=0)
    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)
    cost_usd: float = Field(0.0, ge=0.0)

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens for this channel."""
        return self.prompt_tokens + self.completion_tokens

    def add(self, usage: Usage) -> None:
        """Fold one call's :class:`Usage` into the running totals."""
        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.cost_usd += usage.cost_usd


#: A budget-charge hook: ``(channel, usage) -> None``. The harness passes a closure
#: that debits its real BudgetMeter and may raise ``BudgetExceeded``.
ChargeHook = Callable[[Channel, Usage], None]


class UsageMeter:
    """Thread-safe accumulator of per-channel token + cost usage.

    The meter is created once per episode and shared by all clients used in that
    episode. Each client calls :meth:`record` with the call's normalized
    :class:`Usage` and its :class:`Channel`; the meter folds it into per-channel
    totals and, if a :class:`ChargeHook` was registered, forwards the delta to the
    harness budget (which may raise :class:`~usabench.core.errors.BudgetExceeded`).

    This is intentionally *not* the BudgetMeter itself -- it is the LLM-layer view
    that the harness budget consumes. Keeping it here means the clients have no
    import-time dependency on ``usabench.harness``.
    """

    def __init__(self, *, charge_hook: ChargeHook | None = None) -> None:
        """Initialize an empty meter.

        Args:
            charge_hook: Optional callback invoked on every :meth:`record` with the
                channel and the call's usage. Typically wired to a BudgetMeter so a
                ceiling breach raises before the next call.
        """
        self._lock = threading.Lock()
        self._charge_hook = charge_hook
        self._totals: dict[Channel, ChannelTotals] = {
            ch: ChannelTotals() for ch in Channel
        }

    def set_charge_hook(self, hook: ChargeHook | None) -> None:
        """Register (or clear) the budget-charge hook after construction."""
        self._charge_hook = hook

    def record(self, usage: Usage, *, channel: Channel = Channel.AGENT) -> None:
        """Record one call's usage and forward it to the charge hook.

        Args:
            usage: The normalized usage for the completed call.
            channel: Which role spent it (agent/oracle/judge).

        Raises:
            BudgetExceeded: If a registered charge hook rejects the debit.
        """
        with self._lock:
            self._totals[channel].add(usage)
        # Charge hook runs OUTSIDE the lock: it may raise BudgetExceeded, and we do
        # not want to hold the meter lock while harness budget logic runs.
        if self._charge_hook is not None:
            self._charge_hook(channel, usage)

    def totals(self, channel: Channel) -> ChannelTotals:
        """Return a snapshot copy of one channel's totals."""
        with self._lock:
            return self._totals[channel].model_copy(deep=True)

    @property
    def total_cost_usd(self) -> float:
        """Total USD spent across all channels."""
        with self._lock:
            return sum(t.cost_usd for t in self._totals.values())

    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + completion) across all channels."""
        with self._lock:
            return sum(t.total_tokens for t in self._totals.values())

    def as_usage(self, channel: Channel | None = None) -> Usage:
        """Collapse totals into a single :class:`Usage` rollup.

        Args:
            channel: If given, only that channel; otherwise the sum of all channels.

        Returns:
            A :class:`Usage` aggregate (useful for manifests / run results).
        """
        with self._lock:
            if channel is not None:
                t = self._totals[channel]
                return Usage(
                    prompt_tokens=t.prompt_tokens,
                    completion_tokens=t.completion_tokens,
                    cost_usd=t.cost_usd,
                )
            return Usage(
                prompt_tokens=sum(t.prompt_tokens for t in self._totals.values()),
                completion_tokens=sum(t.completion_tokens for t in self._totals.values()),
                cost_usd=sum(t.cost_usd for t in self._totals.values()),
            )

    def snapshot(self) -> dict[str, ChannelTotals]:
        """Return a full per-channel snapshot keyed by channel value."""
        with self._lock:
            return {ch.value: t.model_copy(deep=True) for ch, t in self._totals.items()}

    def reset(self) -> None:
        """Zero all channel totals (does not clear the charge hook)."""
        with self._lock:
            self._totals = {ch: ChannelTotals() for ch in Channel}
