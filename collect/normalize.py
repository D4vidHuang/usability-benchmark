"""Normalize raw GitHub API payloads into ``raw_harvest`` records.

The collector's sources return loosely-typed dicts straight off the REST/GraphQL
APIs. This module maps them onto the provenance-bearing record shape validated by
``schemas/raw_harvest.schema.json`` (``docs/tasks.md`` §5.5): links + pinned SHA +
SPDX license + short attributed excerpts ONLY -- never a vendored copy of code.

Every record carries: ``owner/repo``, the pinned ``head_sha`` at harvest time, the
SPDX ``license_spdx`` and a derived ``redistributable`` flag, the ``fetched_at``
timestamp, and a ``harvest_provenance_id`` so downstream stages (drafting, QC) can
trace gold material back to its source.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import structlog

from collect.filters import is_redistributable

__all__ = [
    "READExcerpt_MAX",
    "ISSUE_BODY_MAX",
    "now_iso",
    "make_provenance_id",
    "normalize_repo",
    "normalize_issue",
    "guess_domain",
    "tier_size_proxy",
]

log = structlog.get_logger(__name__)

#: Max README characters retained (fair-use-scale excerpt; schema caps at 8192).
READExcerpt_MAX = 4096
#: Max issue-body characters retained.
ISSUE_BODY_MAX = 2048

#: Topic/keyword hints -> domain (best-effort; refined later by the drafter).
_DOMAIN_HINTS: dict[str, list[str]] = {
    "cli-util": ["cli", "command-line", "argparse", "click", "terminal", "cobra", "console"],
    "data-analysis": ["data", "analytics", "pandas", "statistics", "report", "csv", "ics", "plot", "chart"],
    "web-dashboard": ["flask", "fastapi", "dashboard", "web", "frontend", "express", "django", "webapp"],
    "api-integration": ["api", "client", "sdk", "integration", "rest", "graphql", "webhook", "oauth"],
    "automation": ["automation", "cron", "scheduler", "watcher", "bot", "glue", "pipeline", "workflow"],
    "dev-tooling": ["lint", "formatter", "codegen", "git", "tooling", "linter", "devtool", "generator"],
}

#: Coarse repo-size (KB) -> tier proxy. Refined/confirmed by a human later.
_SIZE_TIER_BANDS: list[tuple[int, str]] = [
    (300, "T1"),
    (1500, "T2"),
    (6000, "T3"),
]


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 ``date-time`` string."""
    return _dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_provenance_id(seq: int = 0, *, when: _dt.datetime | None = None) -> str:
    """Build a harvest provenance id like ``hv-2026-06-19-0312``.

    Args:
        seq: A monotonically increasing counter within a harvest run; rendered as
            a zero-padded 4-digit suffix.
        when: Override timestamp (defaults to now, UTC).

    Returns:
        A stable provenance id string.
    """
    ts = when or _dt.datetime.now(tz=_dt.UTC)
    return f"hv-{ts.strftime('%Y-%m-%d')}-{seq % 10000:04d}"


def _spdx_of(repo: dict[str, Any]) -> str:
    """Extract an SPDX id from a REST/GraphQL repo payload, defaulting to NOASSERTION."""
    lic = repo.get("license") or repo.get("licenseInfo")
    if isinstance(lic, dict):
        spdx = lic.get("spdx_id") or lic.get("spdxId")
        if spdx:
            return str(spdx)
    return "NOASSERTION"


def _topics_of(repo: dict[str, Any]) -> list[str]:
    """Extract a topic list from REST (``topics``) or GraphQL (``repositoryTopics``)."""
    if isinstance(repo.get("topics"), list):
        return [str(t) for t in repo["topics"]]
    rt = repo.get("repositoryTopics")
    if isinstance(rt, dict):
        nodes = rt.get("nodes") or []
        out: list[str] = []
        for n in nodes:
            topic = (n or {}).get("topic") or {}
            name = topic.get("name")
            if name:
                out.append(str(name))
        return out
    return []


def guess_domain(repo: dict[str, Any]) -> str | None:
    """Heuristically classify a repo into one of the six domains.

    Scores each domain by keyword hits across topics + description + name, then
    returns the best-scoring domain (or ``None`` if nothing matches).

    Args:
        repo: A repo payload.

    Returns:
        A domain enum value (``cli-util`` ...), or ``None``.
    """
    haystack = " ".join(
        [
            str(repo.get("name") or ""),
            str(repo.get("description") or ""),
            " ".join(_topics_of(repo)),
        ]
    ).lower()
    best: str | None = None
    best_score = 0
    for domain, hints in _DOMAIN_HINTS.items():
        score = sum(1 for h in hints if h in haystack)
        if score > best_score:
            best_score, best = score, domain
    return best


def tier_size_proxy(size_kb: int | None) -> str | None:
    """Map repo size (KB) to a coarse difficulty-tier proxy (``T1``..``T4``).

    Args:
        size_kb: Repo size in kilobytes (REST ``size`` field).

    Returns:
        ``T1``/``T2``/``T3``/``T4``, or ``None`` if size is unknown.
    """
    if size_kb is None:
        return None
    for ceiling, tier in _SIZE_TIER_BANDS:
        if size_kb <= ceiling:
            return tier
    return "T4"


def normalize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single issue payload into a ``candidate_issues`` entry.

    Args:
        issue: A REST issue dict.

    Returns:
        A dict with ``number``, ``title``, ``labels``, ``url``, ``body_excerpt``.
    """
    labels_raw = issue.get("labels") or []
    labels: list[str] = []
    for lab in labels_raw:
        if isinstance(lab, dict) and lab.get("name"):
            labels.append(str(lab["name"]))
        elif isinstance(lab, str):
            labels.append(lab)
    body = issue.get("body") or ""
    return {
        "number": int(issue.get("number", 0)),
        "title": str(issue.get("title") or ""),
        "labels": labels,
        "url": issue.get("html_url") or issue.get("url"),
        "body_excerpt": body[:ISSUE_BODY_MAX] if body else None,
    }


def normalize_repo(
    repo: dict[str, Any],
    *,
    fetched_at: str | None = None,
    provenance_id: str | None = None,
    head_sha: str | None = None,
    readme: str | None = None,
    readme_sha: str | None = None,
    has_tests: bool = False,
    has_ci: bool = False,
    candidate_issues: list[dict[str, Any]] | None = None,
    source_list: str | None = None,
) -> dict[str, Any]:
    """Normalize a repo payload into a ``raw_harvest`` record dict.

    The result validates against ``schemas/raw_harvest.schema.json``. Stores links
    + metadata + short attributed excerpts only.

    Args:
        repo: A REST/GraphQL repo payload.
        fetched_at: ISO-8601 fetch timestamp (defaults to now).
        provenance_id: Harvest provenance id (defaults to a generated one).
        head_sha: Pinned commit SHA at harvest time (REST has no ``head_sha`` field;
            callers pass the resolved default-branch HEAD).
        readme: Optional decoded README text (truncated to ``READExcerpt_MAX``).
        readme_sha: Optional README blob SHA.
        has_tests: Whether a tests directory was detected.
        has_ci: Whether a CI workflow was detected.
        candidate_issues: Pre-normalized ``candidate_issues`` entries.
        source_list: Name of the awesome-* list it was discovered via, if any.

    Returns:
        A ``raw_harvest`` record dict.
    """
    owner = (
        (repo.get("owner") or {}).get("login")
        if isinstance(repo.get("owner"), dict)
        else repo.get("owner")
    )
    full = repo.get("full_name") or ""
    if not owner and "/" in str(full):
        owner = str(full).split("/", 1)[0]
    name = repo.get("name") or (str(full).split("/", 1)[1] if "/" in str(full) else "")

    spdx = _spdx_of(repo)
    size_kb = repo.get("size") if repo.get("size") is not None else repo.get("size_kb")
    primary_language = repo.get("language")
    if primary_language is None and isinstance(repo.get("primaryLanguage"), dict):
        primary_language = repo["primaryLanguage"].get("name")

    excerpt = readme[:READExcerpt_MAX] if readme else None

    record: dict[str, Any] = {
        "harvest_provenance_id": provenance_id or make_provenance_id(),
        "fetched_at": fetched_at or now_iso(),
        "owner": str(owner or ""),
        "repo": str(name or ""),
        "url": repo.get("html_url") or repo.get("url") or f"https://github.com/{owner}/{name}",
        "default_branch": str(repo.get("default_branch") or repo.get("defaultBranchRef", {}).get("name", "") or ""),
        "head_sha": str(head_sha or repo.get("head_sha") or ""),
        "description": repo.get("description"),
        "topics": _topics_of(repo),
        "primary_language": primary_language,
        "size_kb": int(size_kb) if size_kb is not None else None,
        "stars": int(repo.get("stargazers_count") or repo.get("stargazerCount") or 0),
        "pushed_at": repo.get("pushed_at") or repo.get("pushedAt"),
        "archived": bool(repo.get("archived") or repo.get("isArchived") or False),
        "fork": bool(repo.get("fork") or repo.get("isFork") or False),
        "license_spdx": spdx,
        "redistributable": is_redistributable(spdx),
        "readme_excerpt": excerpt,
        "readme_sha": readme_sha,
        "has_tests": bool(has_tests),
        "has_ci": bool(has_ci),
        "candidate_issues": candidate_issues or [],
        "source_list": source_list,
        "domain_guess": guess_domain(repo),
        "tier_size_proxy": tier_size_proxy(int(size_kb) if size_kb is not None else None),
        "dedup_cluster_id": None,
        "dedup_representative": True,
    }
    return record
