"""Run-manifest construction (``docs/infra.md`` §6.1).

The manifest captures everything that affects a run's outcome -- git sha, config
hash, lockfile hash, seeds, image digest, oracle-prompt hash, host/SLURM context --
with secrets redacted, so two leaderboard numbers compare iff their manifests
agree. It is written once per run to ``runs/<run_id>/manifest.json`` and is also
mirrored into the ``episode_start`` trace payload by the runner.

Everything here is pure/offline and degrades gracefully off a git checkout (the
build environment may be a plain directory), so importing and calling it never
fails for want of ``git``.
"""

from __future__ import annotations

import os
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from usabench.config.hashing import config_hash as _config_hash
from usabench.config.hashing import redact_secrets, to_hashable
from usabench.core.ids import run_id as _run_id
from usabench.core.ids import sha256_hex
from usabench.core.schema import RunManifest
from usabench.logging_setup import get_logger

__all__ = [
    "git_sha",
    "file_sha256",
    "oracle_prompt_hash",
    "utc_now_iso",
    "build_manifest",
]

_log = get_logger(__name__)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (timezone-aware)."""
    return datetime.now(UTC).isoformat()


def git_sha(repo_root: str | Path | None = None, *, short: bool = False) -> str:
    """Return the current git commit SHA, or a sentinel off a git checkout.

    Args:
        repo_root: Directory to resolve the SHA in (defaults to cwd).
        short: If True, return the abbreviated SHA.

    Returns:
        The commit SHA, with a ``-dirty`` suffix if the tree has uncommitted
        changes, or ``"unknown"`` if not a git repo / git is unavailable.
    """
    cwd = str(repo_root) if repo_root is not None else None
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        sha = subprocess.run(  # noqa: S603,S607 - fixed argv
            args, cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603,S607 - fixed argv
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return "unknown"


def file_sha256(path: str | Path | None) -> str | None:
    """Return the sha256 hex of a file's bytes, or ``None`` if absent.

    Used for the lockfile hash so an install drift is detectable in the manifest.

    Args:
        path: Path to the file (e.g. ``requirements.lock``), or ``None``.

    Returns:
        The 64-char hex digest, or ``None`` if ``path`` is missing/unset.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return sha256_hex(p.read_bytes())


def oracle_prompt_hash(prompt_template: str) -> str:
    """Hash the frozen oracle system-prompt template (``run_start`` provenance).

    Args:
        prompt_template: The rendered/static oracle system-prompt text.

    Returns:
        The sha256 hex of the template (``docs/protocol.md`` §2.6 #1).
    """
    return sha256_hex(prompt_template)


def build_manifest(
    *,
    task_id: str,
    seed: int,
    config: Any,
    package_version: str,
    git_sha_value: str | None = None,
    repo_root: str | Path | None = None,
    requirements_lock: str | Path | None = None,
    agent: dict[str, Any] | None = None,
    oracle: dict[str, Any] | None = None,
    budgets: dict[str, Any] | None = None,
    sandbox: dict[str, Any] | None = None,
    run_id_value: str | None = None,
    started_at: str | None = None,
) -> RunManifest:
    """Assemble a :class:`RunManifest` for one run, with secrets redacted.

    The ``config_hash`` and ``run_id`` are computed from the *redacted* config so a
    key rotation does not change them (``docs/infra.md`` §6.1).

    Args:
        task_id: The task's stable id.
        seed: The replica seed.
        config: The resolved run config (pydantic model or dict).
        package_version: ``usabench.__version__``.
        git_sha_value: Pre-resolved git SHA; if ``None``, resolved via :func:`git_sha`.
        repo_root: Repo root for git resolution.
        requirements_lock: Path to the lockfile to hash (optional).
        agent: Agent descriptor (model id, decoding) -- secrets redacted.
        oracle: Oracle descriptor (model, persona, prompt hash) -- secrets redacted.
        budgets: Budget ceilings dict.
        sandbox: Sandbox descriptor (image digest, network policy).
        run_id_value: Pre-computed run id; if ``None``, derived deterministically.
        started_at: ISO start time; defaults to now.

    Returns:
        A fully-populated, redacted :class:`RunManifest`.
    """
    sha = git_sha_value if git_sha_value is not None else git_sha(repo_root)
    cfg_hash = _config_hash(config, redact=True)
    rid = run_id_value or _run_id(cfg_hash, task_id, seed, sha)

    def _clean(d: dict[str, Any] | None) -> dict[str, Any]:
        cleaned: dict[str, Any] = redact_secrets(to_hashable(d or {}))
        return cleaned

    return RunManifest(
        run_id=rid,
        task_id=task_id,
        seed=int(seed),
        git_sha=sha,
        config_hash=cfg_hash,
        package_version=package_version,
        requirements_lock_sha256=file_sha256(requirements_lock),
        agent=_clean(agent),
        oracle=_clean(oracle),
        budgets=_clean(budgets),
        sandbox=_clean(sandbox),
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        started_at=started_at or utc_now_iso(),
        ended_at=None,
    )
