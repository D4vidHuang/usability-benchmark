"""Single-source-of-truth loader for the usability-score spec.

The ONLY way other modules should obtain scoring constants is via
:func:`load_spec` / :func:`get_severity_weights` here -- no constant is duplicated
in Python (DESIGN.md frozen build decision #2). The spec file lives next to this
module so it ships inside the installed package.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

__all__ = ["SPEC_PATH", "load_spec", "get_severity_weights", "reload_spec"]

#: Absolute path to the frozen spec, resolved relative to this file.
SPEC_PATH: Path = Path(__file__).resolve().parent / "usability_score.yaml"


@functools.lru_cache(maxsize=1)
def load_spec() -> dict[str, Any]:
    """Load and cache the usability-score spec as a plain dict.

    Returns:
        The parsed ``usability_score.yaml`` contents.

    Raises:
        FileNotFoundError: If the spec file is missing.
        ValueError: If the spec is not a mapping.
    """
    if not SPEC_PATH.is_file():
        raise FileNotFoundError(f"usability score spec not found: {SPEC_PATH}")
    data = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"usability score spec must be a mapping: {SPEC_PATH}")
    return data


def reload_spec() -> dict[str, Any]:
    """Clear the cache and reload the spec (for tests that mutate it)."""
    load_spec.cache_clear()
    return load_spec()


def get_severity_weights() -> list[float]:
    """Return the canonical convex severity weight vector ``w[0..5]``.

    Returns:
        A 6-element list of floats, one per severity level 0..5.

    Raises:
        ValueError: If the spec's ``severity_weights`` is not length 6.
    """
    weights = load_spec()["severity_weights"]
    if len(weights) != 6:
        raise ValueError(f"severity_weights must have 6 entries (0..5), got {len(weights)}")
    return [float(w) for w in weights]
