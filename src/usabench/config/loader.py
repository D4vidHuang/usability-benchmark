"""YAML config loading with ``${ENV}`` interpolation.

Loads a YAML file, recursively interpolates ``${VAR}`` / ``${VAR:-default}``
references from the process environment, and optionally validates the result into
a pydantic model. Used by every config under ``configs/`` (models, oracle, agents,
runs, daic).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from usabench.core.errors import ConfigError

__all__ = ["load_yaml", "load_yaml_config", "interpolate_env"]

T = TypeVar("T", bound=BaseModel)

#: Matches ``${VAR}`` and ``${VAR:-default}``.
_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def interpolate_env(value: Any, env: dict[str, str] | None = None) -> Any:
    """Recursively interpolate ``${ENV}`` references in a loaded YAML structure.

    Supports ``${VAR}`` (required) and ``${VAR:-default}`` (with fallback). A
    required variable that is unset raises :class:`ConfigError`.

    Args:
        value: A scalar, list, or dict from parsed YAML.
        env: Optional environment mapping; defaults to ``os.environ``.

    Returns:
        The same structure with all string ``${...}`` references resolved.

    Raises:
        ConfigError: If a required ``${VAR}`` (no default) is unset.
    """
    environ = os.environ if env is None else env

    if isinstance(value, dict):
        return {k: interpolate_env(v, environ) for k, v in value.items()}  # type: ignore[arg-type]
    if isinstance(value, list):
        return [interpolate_env(v, environ) for v in value]  # type: ignore[arg-type]
    if isinstance(value, str):
        return _interpolate_str(value, environ)
    return value


def _interpolate_str(s: str, environ: Any) -> str:
    """Resolve all ``${...}`` references inside a single string."""

    def _replace(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        if name in environ:
            return str(environ[name])
        if default is not None:
            return default
        raise ConfigError(f"required environment variable not set: ${{{name}}}")

    return _ENV_PATTERN.sub(_replace, s)


def load_yaml(path: str | Path, *, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Load a YAML file and interpolate ``${ENV}`` references.

    Args:
        path: Path to the YAML file.
        env: Optional environment override for interpolation.

    Returns:
        The parsed, env-interpolated dict.

    Raises:
        ConfigError: If the file is missing, unparseable, or not a mapping.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"failed to parse YAML {p}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}: {p}")
    result = interpolate_env(raw, env)
    assert isinstance(result, dict)
    return result


def load_yaml_config(
    path: str | Path,
    model: type[T] | None = None,
    *,
    env: dict[str, str] | None = None,
) -> Any:
    """Load a YAML config and optionally validate it into a pydantic model.

    Args:
        path: Path to the YAML file.
        model: Optional pydantic model class to validate the loaded dict into. If
            ``None``, the raw interpolated dict is returned.
        env: Optional environment override for interpolation.

    Returns:
        A validated ``model`` instance, or the raw dict if ``model`` is ``None``.

    Raises:
        ConfigError: On load/parse failure or model validation failure.
    """
    data = load_yaml(path, env=env)
    if model is None:
        return data
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"config {path} failed validation for {model.__name__}: {exc}") from exc
