"""Repo metadata + README harvesting at a pinned commit.

Given a :class:`~collect.sources.RepoRef`, this source fetches the repository's
metadata, resolves and **pins** the default-branch HEAD SHA (no floating refs --
``docs/tasks.md`` §2.1), retrieves the README *at that commit*, and detects the
presence of tests / CI (a verifiability hint, §5.1). It returns the loosely-typed
pieces that :func:`collect.normalize.normalize_repo` assembles into a record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from collect.github_client import GitHubClient, GitHubError
from collect.sources import RepoRef

__all__ = ["RepoEnrichment", "resolve_head_sha", "detect_tests_ci", "enrich_repo"]

log = structlog.get_logger(__name__)

#: Paths probed to infer the presence of a test suite.
_TEST_PATHS: tuple[str, ...] = ("tests", "test", "spec")
#: Paths probed to infer the presence of CI.
_CI_PATHS: tuple[str, ...] = (".github/workflows", ".circleci", ".gitlab-ci.yml", "tox.ini")


@dataclass(slots=True)
class RepoEnrichment:
    """The enriched, pin-resolved pieces for one repo.

    Attributes:
        repo: The raw repo metadata payload.
        head_sha: The resolved default-branch HEAD commit SHA (pinned).
        readme: Decoded README text at ``head_sha`` (or ``None``).
        has_tests: Whether a test directory was detected.
        has_ci: Whether a CI config was detected.
        source_list: Provenance: the awesome list this repo came from, if any.
    """

    repo: dict[str, Any]
    head_sha: str | None
    readme: str | None
    has_tests: bool = False
    has_ci: bool = False
    source_list: str | None = None


def resolve_head_sha(client: GitHubClient, owner: str, repo: str, branch: str) -> str | None:
    """Resolve the current HEAD commit SHA of a repo's default branch.

    Args:
        client: An initialized :class:`GitHubClient`.
        owner: Repo owner.
        repo: Repo name.
        branch: The default branch name.

    Returns:
        The pinned commit SHA, or ``None`` if it could not be resolved.
    """
    if not branch:
        return None
    try:
        ref = client.get(f"/repos/{owner}/{repo}/commits/{branch}")
    except GitHubError as exc:
        if exc.status == 404:
            return None
        raise
    if isinstance(ref, dict):
        return ref.get("sha")
    return None


def detect_tests_ci(
    client: GitHubClient, owner: str, repo: str, *, ref: str | None = None
) -> tuple[bool, bool]:
    """Probe for the presence of tests and CI in a repo.

    Args:
        client: An initialized :class:`GitHubClient`.
        owner: Repo owner.
        repo: Repo name.
        ref: Optional pinned ref to probe against.

    Returns:
        ``(has_tests, has_ci)``.
    """
    has_tests = any(client.path_exists(owner, repo, p, ref=ref) for p in _TEST_PATHS)
    has_ci = any(client.path_exists(owner, repo, p, ref=ref) for p in _CI_PATHS)
    return has_tests, has_ci


def enrich_repo(
    client: GitHubClient,
    ref: RepoRef,
    *,
    fetch_readme: bool = True,
    probe_tests_ci: bool = True,
) -> RepoEnrichment | None:
    """Fetch metadata + pinned README + tests/CI flags for one repo reference.

    Args:
        client: An initialized :class:`GitHubClient`.
        ref: The repo reference to enrich.
        fetch_readme: Whether to retrieve the README at the pinned commit.
        probe_tests_ci: Whether to probe for tests/CI presence.

    Returns:
        A :class:`RepoEnrichment`, or ``None`` if the repo is unavailable (404).
    """
    try:
        repo = client.get_repo(ref.owner, ref.repo)
    except GitHubError as exc:
        if exc.status in (404, 451):  # gone / DMCA / unavailable
            log.info("collect.readmes.unavailable", repo=ref.full_name, status=exc.status)
            return None
        raise
    if not isinstance(repo, dict):
        return None

    branch = str(repo.get("default_branch") or "")
    head_sha = resolve_head_sha(client, ref.owner, ref.repo, branch)
    readme = (
        client.get_readme(ref.owner, ref.repo, ref=head_sha) if fetch_readme else None
    )
    has_tests, has_ci = (
        detect_tests_ci(client, ref.owner, ref.repo, ref=head_sha)
        if probe_tests_ci
        else (False, False)
    )
    return RepoEnrichment(
        repo=repo,
        head_sha=head_sha,
        readme=readme,
        has_tests=has_tests,
        has_ci=has_ci,
        source_list=ref.source_list,
    )
