"""Config hashing -- stable digests feeding ``run_id``.

Thin wrapper over :mod:`usabench.core.ids` that knows how to coerce pydantic
models / dicts into a canonical, hashable form and (optionally) redact secrets
before hashing so a key rotation does not change a ``config_hash``.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from usabench.core.ids import canonical_json
from usabench.core.ids import config_hash as _config_hash

__all__ = ["config_hash", "to_hashable", "redact_secrets"]

#: Keys whose values are redacted before hashing/logging.
_SECRET_KEY_PATTERN = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization|bearer)")
#: Values that look like secrets (long opaque tokens).
_SECRET_VALUE_PATTERN = re.compile(r"^(sk-|ghp_|gho_|github_pat_)[A-Za-z0-9_\-]{8,}$")

_REDACTED = "***REDACTED***"


def to_hashable(cfg: Any) -> Any:
    """Coerce ``cfg`` into a plain JSON-serializable structure.

    Pydantic models are dumped in JSON mode; dicts/lists are recursed; scalars
    pass through. This guarantees the same logical config hashes identically
    regardless of whether it arrived as a model or a dict.

    Args:
        cfg: A pydantic model, dict, list, or scalar.

    Returns:
        A plain dict/list/scalar structure.
    """
    if isinstance(cfg, BaseModel):
        return cfg.model_dump(mode="json")
    if isinstance(cfg, dict):
        return {k: to_hashable(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [to_hashable(v) for v in cfg]
    return cfg


def redact_secrets(cfg: Any) -> Any:
    """Return a copy of ``cfg`` with secret-shaped keys/values redacted.

    Args:
        cfg: A JSON-serializable structure (call :func:`to_hashable` first).

    Returns:
        The structure with secrets replaced by a redaction marker.
    """
    if isinstance(cfg, dict):
        out: dict[str, Any] = {}
        for k, v in cfg.items():
            if isinstance(k, str) and _SECRET_KEY_PATTERN.search(k):
                out[k] = _REDACTED
            else:
                out[k] = redact_secrets(v)
        return out
    if isinstance(cfg, list):
        return [redact_secrets(v) for v in cfg]
    if isinstance(cfg, str) and _SECRET_VALUE_PATTERN.match(cfg):
        return _REDACTED
    return cfg


def config_hash(cfg: Any, *, redact: bool = True) -> str:
    """Compute a stable SHA-256 hex digest of a config.

    Args:
        cfg: A pydantic model, dict, list, or scalar.
        redact: If True (default), redact secret-shaped fields before hashing so
            the hash is invariant under key rotation.

    Returns:
        64-character hex digest of the canonical JSON of the (redacted) config.
    """
    hashable = to_hashable(cfg)
    if redact:
        hashable = redact_secrets(hashable)
    return _config_hash(hashable)


def canonical(cfg: Any, *, redact: bool = True) -> str:
    """Return the canonical-JSON string used for hashing (handy for debugging)."""
    hashable = to_hashable(cfg)
    if redact:
        hashable = redact_secrets(hashable)
    return canonical_json(hashable)
