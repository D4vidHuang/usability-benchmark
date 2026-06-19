"""Typed exception hierarchy for the benchmark.

A small, closed set of exceptions so every package can ``except`` precisely and
the harness can map failures onto the right :class:`~usabench.core.enums.RunStatus`
and :class:`~usabench.core.enums.TerminatedReason` without string matching.
"""

from __future__ import annotations

__all__ = [
    "UsabenchError",
    "BudgetExceeded",
    "OracleProtocolError",
    "ProviderError",
    "SchemaViolation",
    "SandboxError",
    "ConfigError",
    "TraceIntegrityError",
]


class UsabenchError(Exception):
    """Base class for all benchmark-specific errors."""


class BudgetExceeded(UsabenchError):
    """Raised when a run hits a budget ceiling (turns/wall/tokens/cost/oracle).

    Attributes:
        kind: Which budget was exhausted (e.g. ``"tokens"``, ``"oracle_queries"``).
        limit: The configured ceiling.
        used: The amount consumed at the moment of the breach.
    """

    def __init__(self, kind: str, limit: float, used: float) -> None:
        self.kind = kind
        self.limit = limit
        self.used = used
        super().__init__(f"budget '{kind}' exceeded: used {used} > limit {limit}")


class OracleProtocolError(UsabenchError):
    """Raised when the oracle violates its protocol contract.

    Examples: a non-JSON response, a self-declared ``level`` outside ``[0,5]``, a
    reject without a citable criterion, or an unsolicited reveal (leak).
    """


class ProviderError(UsabenchError):
    """Raised when an LLM provider call fails after retries are exhausted.

    Attributes:
        provider: The provider name (e.g. ``"anthropic"``).
        status: Optional HTTP/status code if available.
    """

    def __init__(self, message: str, *, provider: str | None = None, status: int | None = None) -> None:
        self.provider = provider
        self.status = status
        super().__init__(message)


class SchemaViolation(UsabenchError):
    """Raised when a record fails JSON-Schema or pydantic validation.

    Attributes:
        pointer: Optional JSON pointer / field path that failed.
    """

    def __init__(self, message: str, *, pointer: str | None = None) -> None:
        self.pointer = pointer
        super().__init__(message)


class SandboxError(UsabenchError):
    """Raised on sandbox backend failures (image pull, exec crash, path escape)."""


class ConfigError(UsabenchError):
    """Raised on malformed/unresolvable configuration (bad YAML, missing ``${ENV}``)."""


class TraceIntegrityError(UsabenchError):
    """Raised when ``trace.jsonl`` fails an integrity invariant.

    Examples: a broken hash chain, non-monotonic ``seq``, a ``tool_result`` with
    no matching ``tool_call``, or an intervention-severity sum that disagrees with
    the cached ``run_end`` totals (``docs/protocol.md`` §4.3).
    """
