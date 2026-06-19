"""Retry / backoff policy for LLM provider calls.

Wraps a single provider call with exponential backoff + jitter on transient
failures (HTTP ``429`` rate limits and ``5xx`` server errors / timeouts), capped at
``max_attempts``. When the provider returns a ``Retry-After`` header (or the SDK
exception carries one), we honor it instead of the computed backoff.

``tenacity`` is imported **lazily inside functions** so the base package imports
with only core deps. If tenacity is somehow unavailable at call time, we fall back
to a small hand-rolled retry loop with identical semantics, so the LLM layer never
hard-fails on a missing optional import.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from usabench.core.errors import ProviderError
from usabench.logging_setup import get_logger

__all__ = ["RetryConfig", "is_retryable", "retry_after_seconds", "call_with_retry"]

_log = get_logger("usabench.llm.retry")

T = TypeVar("T")

#: HTTP status codes we retry on (rate limit + transient server errors).
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class RetryConfig(BaseModel):
    """Backoff parameters, mirroring a model config's ``retry`` block.

    Attributes:
        max_attempts: Total attempts including the first (``>=1``).
        base_delay_s: Initial backoff delay; doubles each attempt.
        max_delay_s: Cap on any single backoff delay.
        jitter_s: Uniform jitter added to each delay to de-correlate retries.
        respect_retry_after: Honor a provider ``Retry-After`` over computed backoff.
    """

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(5, ge=1)
    base_delay_s: float = Field(1.0, ge=0.0)
    max_delay_s: float = Field(60.0, ge=0.0)
    jitter_s: float = Field(0.5, ge=0.0)
    respect_retry_after: bool = True

    @classmethod
    def from_config(cls, retry: dict[str, Any] | None) -> RetryConfig:
        """Build from a config ``retry`` mapping (unknown keys ignored)."""
        if not retry:
            return cls()
        allowed = {k: retry[k] for k in retry if k in cls.model_fields}
        return cls(**allowed)


def _status_of(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from a provider exception."""
    for attr in ("status_code", "status", "http_status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    # openai/anthropic SDK errors expose ``.response`` with ``.status_code``.
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc
    return None


def is_retryable(exc: BaseException) -> bool:
    """Return ``True`` if a provider exception is worth retrying.

    Retries cover rate limits, transient ``5xx``, connection/timeout errors, and
    anything that self-reports a retryable status. A wrapped
    :class:`~usabench.core.errors.ProviderError` is inspected via its ``.status``.

    Args:
        exc: The exception raised by the provider call.

    Returns:
        Whether the call should be retried.
    """
    if isinstance(exc, ProviderError):
        return exc.status is None or exc.status in _RETRYABLE_STATUS
    status = _status_of(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    # SDK exception class names we treat as transient even without a status.
    transient_markers = (
        "timeout",
        "connection",
        "ratelimit",
        "apiconnection",
        "internalserver",
        "serviceunavailable",
        "overloaded",
    )
    return any(m in name for m in transient_markers)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a ``Retry-After`` hint (seconds) from a provider exception, if any.

    Looks at common SDK shapes: ``exc.retry_after``, response headers
    ``Retry-After`` / ``retry-after`` / ``x-ratelimit-reset``.

    Args:
        exc: The provider exception.

    Returns:
        Seconds to wait, or ``None`` if no hint is present / parseable.
    """
    direct = getattr(exc, "retry_after", None)
    if isinstance(direct, (int, float)):
        return float(direct)

    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        getter = headers.get  # mapping-like
    except AttributeError:
        return None
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        raw = getter(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _compute_delay(attempt: int, cfg: RetryConfig) -> float:
    """Exponential backoff with jitter for a 1-based ``attempt`` index."""
    delay: float = min(cfg.base_delay_s * float(2 ** (attempt - 1)), cfg.max_delay_s)
    if cfg.jitter_s:
        delay += random.uniform(0.0, cfg.jitter_s)
    return delay


def call_with_retry(
    fn: Callable[[], T],
    cfg: RetryConfig,
    *,
    provider: str = "unknown",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Invoke ``fn`` with retry/backoff and ``Retry-After`` handling.

    On the final failed attempt the underlying exception is re-raised wrapped in a
    :class:`~usabench.core.errors.ProviderError` (unless it already is one).

    Args:
        fn: A zero-arg callable performing exactly one provider request.
        cfg: The :class:`RetryConfig` to apply.
        provider: Provider name for error/log context.
        sleep: Sleep function (injectable for tests).

    Returns:
        The value returned by ``fn`` on the first successful attempt.

    Raises:
        ProviderError: If all attempts fail (transient) or the error is fatal.
    """
    try:
        return _call_with_tenacity(fn, cfg, provider=provider, sleep=sleep)
    except _TenacityUnavailable:
        return _call_with_loop(fn, cfg, provider=provider, sleep=sleep)


class _TenacityUnavailable(Exception):
    """Internal signal that tenacity could not be imported; use the fallback."""


def _wrap_fatal(exc: BaseException, provider: str) -> ProviderError:
    """Wrap a non-ProviderError into a ProviderError preserving status."""
    if isinstance(exc, ProviderError):
        return exc
    return ProviderError(str(exc), provider=provider, status=_status_of(exc))


def _call_with_loop(
    fn: Callable[[], T],
    cfg: RetryConfig,
    *,
    provider: str,
    sleep: Callable[[float], None],
) -> T:
    """Hand-rolled retry loop used when tenacity is unavailable."""
    last: BaseException | None = None
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we classify below
            last = exc
            if attempt >= cfg.max_attempts or not is_retryable(exc):
                raise _wrap_fatal(exc, provider) from exc
            delay = _compute_delay(attempt, cfg)
            if cfg.respect_retry_after:
                hinted = retry_after_seconds(exc)
                if hinted is not None:
                    delay = min(max(hinted, 0.0), cfg.max_delay_s)
            _log.warning(
                "llm_retry",
                provider=provider,
                attempt=attempt,
                max_attempts=cfg.max_attempts,
                delay_s=round(delay, 3),
                error=type(exc).__name__,
            )
            sleep(delay)
    # Unreachable in practice, but keeps the type checker honest.
    raise _wrap_fatal(last or ProviderError("retry exhausted", provider=provider), provider)


def _call_with_tenacity(
    fn: Callable[[], T],
    cfg: RetryConfig,
    *,
    provider: str,
    sleep: Callable[[float], None],
) -> T:
    """Retry via tenacity; raises :class:`_TenacityUnavailable` if not importable."""
    try:
        import tenacity
    except ImportError as exc:  # pragma: no cover - exercised only without tenacity
        raise _TenacityUnavailable() from exc

    def _before_sleep(retry_state: Any) -> None:
        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None else None
        _log.warning(
            "llm_retry",
            provider=provider,
            attempt=retry_state.attempt_number,
            max_attempts=cfg.max_attempts,
            error=type(exc).__name__ if exc else None,
        )

    def _wait(retry_state: Any) -> float:
        exc = None
        if retry_state.outcome is not None:
            exc = retry_state.outcome.exception()
        if cfg.respect_retry_after and exc is not None:
            hinted = retry_after_seconds(exc)
            if hinted is not None:
                return min(max(hinted, 0.0), cfg.max_delay_s)
        return _compute_delay(retry_state.attempt_number, cfg)

    retrying = tenacity.Retrying(
        stop=tenacity.stop_after_attempt(cfg.max_attempts),
        wait=_wait,
        retry=tenacity.retry_if_exception(is_retryable),
        reraise=True,
        before_sleep=_before_sleep,
        sleep=sleep,
    )
    try:
        return retrying(fn)
    except Exception as exc:  # noqa: BLE001 - normalize to ProviderError
        raise _wrap_fatal(exc, provider) from exc
