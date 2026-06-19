"""Awesome-list discovery: find ``awesome-*`` lists and parse repo links out.

``awesome-*`` README files are curated, real, idea-grade pools of projects
(``docs/tasks.md`` §5.2). This source:

1. searches for ``awesome <domain> in:name`` repositories, then
2. fetches each list's README and regex-extracts ``github.com/owner/repo`` links,
   enqueuing those repos as candidates (tagged with the list name for provenance).

Self-references (the awesome list pointing at itself), ``sponsors``/``topics``/
``orgs`` and other non-repo GitHub paths are filtered out.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import structlog

from collect.github_client import GitHubClient
from collect.sources import RepoRef

__all__ = ["GITHUB_REPO_RE", "extract_repo_links", "discover_awesome_lists"]

log = structlog.get_logger(__name__)

#: Matches ``github.com/owner/repo`` inside markdown/HTML link contexts.
GITHUB_REPO_RE = re.compile(
    r"github\.com/(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9\-]{0,38})?)/(?P<repo>[A-Za-z0-9._\-]+)"
)

#: GitHub path prefixes that are NOT repositories.
_NON_REPO_OWNERS: frozenset[str] = frozenset(
    {
        "sponsors",
        "topics",
        "orgs",
        "collections",
        "marketplace",
        "features",
        "about",
        "pricing",
        "settings",
        "notifications",
        "explore",
        "search",
        "login",
        "join",
        "apps",
    }
)


def _clean_repo_name(name: str) -> str:
    """Strip trailing ``.git``, punctuation, and URL fragments from a repo name."""
    name = name.split("#", 1)[0].split("?", 1)[0]
    if name.endswith(".git"):
        name = name[:-4]
    return name.rstrip("/.,)\"'")


def extract_repo_links(
    readme: str, *, source_list: str | None = None, exclude: set[str] | None = None
) -> list[RepoRef]:
    """Extract unique repo references from an awesome-list README.

    Args:
        readme: The decoded README text.
        source_list: The awesome list's name (provenance tag on each ref).
        exclude: Lowercased ``owner/repo`` slugs to drop (e.g. the list itself).

    Returns:
        A de-duplicated list of :class:`RepoRef` (insertion order preserved).
    """
    excl = {e.lower() for e in (exclude or set())}
    seen: set[str] = set()
    out: list[RepoRef] = []
    for m in GITHUB_REPO_RE.finditer(readme or ""):
        owner = m.group("owner")
        repo = _clean_repo_name(m.group("repo"))
        if not owner or not repo or owner.lower() in _NON_REPO_OWNERS:
            continue
        full = f"{owner}/{repo}".lower()
        if full in seen or full in excl:
            continue
        seen.add(full)
        out.append(RepoRef(owner=owner, repo=repo, source_list=source_list))
    return out


def discover_awesome_lists(
    client: GitHubClient,
    domains: list[str],
    *,
    max_lists_per_domain: int = 3,
    max_repos_per_list: int | None = None,
    min_stars: int = 50,
) -> Iterator[RepoRef]:
    """Discover candidate repos via ``awesome-*`` lists for each domain keyword.

    Args:
        client: An initialized :class:`GitHubClient`.
        domains: Domain keywords to search awesome lists for (e.g. ``cli``).
        max_lists_per_domain: How many awesome lists to mine per domain.
        max_repos_per_list: Optional cap on repos extracted per list.
        min_stars: Minimum stars for an awesome list to be considered.

    Yields:
        :class:`RepoRef` objects extracted from the lists' READMEs.
    """
    seen_global: set[str] = set()
    for domain in domains:
        query = f"awesome {domain} in:name stars:>={min_stars}"
        log.info("collect.awesome.search", domain=domain, query=query)
        lists: list[dict[str, Any]] = client.search_repositories(
            query, sort="stars", order="desc", max_results=max_lists_per_domain
        )
        for lst in lists:
            full = lst.get("full_name") or ""
            owner, _, name = full.partition("/")
            if not owner:
                continue
            readme = client.get_readme(owner, name)
            if not readme:
                continue
            refs = extract_repo_links(readme, source_list=name, exclude={full})
            count = 0
            for ref in refs:
                if ref.full_name.lower() in seen_global:
                    continue
                seen_global.add(ref.full_name.lower())
                yield ref
                count += 1
                if max_repos_per_list is not None and count >= max_repos_per_list:
                    break
