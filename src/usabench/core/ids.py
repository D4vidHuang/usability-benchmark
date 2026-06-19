"""Deterministic identity & integrity helpers.

This module is the single source of truth for:

* ``canonical_json`` -- a stable, sorted-key serialization used everywhere a
  hash must be reproducible (config hashing, run ids, the trace hash-chain).
* ``sha256_hex`` -- the hash primitive.
* ``run_id`` -- ``sha256(config_hash + task_id + seed + git_sha)`` per ``DESIGN.md``.
* ``next_hash`` -- the per-event chaining function that makes ``trace.jsonl``
  tamper-evident and replayable (``docs/protocol.md`` §4).

All functions are pure and dependency-free (stdlib only), so the trace writer,
the scorer, and tests share *identical* hashing semantics.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "canonical_json",
    "sha256_hex",
    "config_hash",
    "run_id",
    "event_hash",
    "next_hash",
    "GENESIS_HASH",
    "short_hash",
]

#: The ``prev_hash`` value used for the first event in a chain (seq 0).
GENESIS_HASH: str = "0" * 64


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to a canonical JSON string.

    The serialization is stable across runs and machines: keys are sorted,
    separators are compact (no insignificant whitespace), non-ASCII is preserved
    (``ensure_ascii=False``) so byte-identical inputs hash identically, and
    ``NaN``/``Infinity`` are rejected (``allow_nan=False``) because they are not
    valid JSON and would break cross-language re-hashing.

    Args:
        obj: Any JSON-serializable object (dicts, lists, scalars). Pydantic
            models should be dumped with ``model_dump(mode="json")`` first.

    Returns:
        A deterministic JSON string suitable for hashing.

    Raises:
        ValueError: If ``obj`` contains ``NaN``/``Infinity``.
        TypeError: If ``obj`` is not JSON-serializable.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(data: str | bytes) -> str:
    """Return the lowercase hex SHA-256 digest of ``data``.

    Args:
        data: A string (UTF-8 encoded) or raw bytes.

    Returns:
        64-character lowercase hexadecimal digest.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def short_hash(data: str | bytes, length: int = 12) -> str:
    """Return a truncated SHA-256 hex digest (for human-readable ids/filenames).

    Args:
        data: Input to hash.
        length: Number of leading hex chars to keep.

    Returns:
        The first ``length`` characters of the hex digest.
    """
    return sha256_hex(data)[:length]


def config_hash(cfg: Any) -> str:
    """Hash a (resolved) config object/dict into a stable digest.

    Args:
        cfg: A JSON-serializable config (dict or already-dumped model).

    Returns:
        64-character hex SHA-256 of the canonical JSON of ``cfg``.
    """
    return sha256_hex(canonical_json(cfg))


def run_id(
    config_hash_value: str,
    task_id: str,
    seed: int,
    git_sha: str,
) -> str:
    """Compute the canonical, reproducible ``run_id`` for one episode.

    ``run_id = sha256(canonical_json({config_hash, task_id, seed, git_sha}))`` --
    identical inputs always yield the identical id, which makes batches idempotent
    and resumable (``DESIGN.md`` §6 invariant 3).

    Args:
        config_hash_value: The :func:`config_hash` of the resolved run config.
        task_id: The task's stable id.
        seed: The integer seed for this replica.
        git_sha: The repo git commit the run was launched from.

    Returns:
        64-character hex run id.
    """
    payload = {
        "config_hash": config_hash_value,
        "task_id": task_id,
        "seed": int(seed),
        "git_sha": git_sha,
    }
    return sha256_hex(canonical_json(payload))


def event_hash(event_canonical: str) -> str:
    """Hash the canonical form of a single event payload+envelope.

    Args:
        event_canonical: The canonical-JSON string of the event (excluding the
            ``hash`` field itself).

    Returns:
        64-character hex digest.
    """
    return sha256_hex(event_canonical)


def next_hash(prev_hash: str, event_canonical: str) -> str:
    """Compute the chained hash for the next trace event.

    The chain binds each event to its predecessor so that any reordering or
    tampering is detectable: ``hash_i = sha256(prev_hash || sha256(event_i))``.
    The first event uses :data:`GENESIS_HASH` as ``prev_hash``.

    Args:
        prev_hash: The ``hash`` of the previous event (or :data:`GENESIS_HASH`).
        event_canonical: Canonical-JSON of the current event *without* its own
            ``hash``/``prev_hash`` fields.

    Returns:
        The 64-character hex chain hash for the current event.
    """
    return sha256_hex(prev_hash + event_hash(event_canonical))
