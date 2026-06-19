"""structlog JSON logging configuration with secret redaction.

One logger for the whole package, emitting JSON lines, optionally bound to a
``run_id`` so every log line for an episode is greppable. A processor redacts
secret-shaped values (API keys, tokens) so logs are safe to ship.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

__all__ = ["configure_logging", "get_logger", "bind_run", "redact_processor"]

#: Keys whose values are scrubbed in log events.
_SECRET_KEY_PATTERN = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization|bearer)")
#: Values that look like opaque credentials.
_SECRET_VALUE_PATTERN = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|ghp_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,})")

_REDACTED = "***REDACTED***"

_CONFIGURED = False


def redact_processor(
    _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor that redacts secret-shaped keys and values.

    Args:
        _logger: The wrapped logger (unused).
        _method_name: The log method name (unused).
        event_dict: The mutable event dict being built.

    Returns:
        The event dict with secrets replaced by a redaction marker.
    """
    for key in list(event_dict.keys()):
        value = event_dict[key]
        if isinstance(key, str) and _SECRET_KEY_PATTERN.search(key):
            event_dict[key] = _REDACTED
            continue
        if isinstance(value, str):
            event_dict[key] = _SECRET_VALUE_PATTERN.sub(_REDACTED, value)
    return event_dict


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure structlog + stdlib logging for the whole process (idempotent).

    Args:
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
        json_output: If True, emit JSON lines; otherwise a console renderer.
    """
    global _CONFIGURED

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=numeric_level,
    )

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger, configuring logging on first use.

    Args:
        name: Optional logger name (typically ``__name__``).

    Returns:
        A bound structlog logger.
    """
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


def bind_run(run_id: str, **extra: Any) -> None:
    """Bind ``run_id`` (and any extra context) to all subsequent log lines.

    Uses contextvars so the binding is scoped to the current execution context.

    Args:
        run_id: The run id to attach to every log line.
        **extra: Additional key/value context to bind.
    """
    if not _CONFIGURED:
        configure_logging()
    structlog.contextvars.bind_contextvars(run_id=run_id, **extra)
