"""Topic / stars repository discovery, sharded past the 1000-result search cap.

The GitHub search API hard-caps any single query at 1000 results. To page past
that we *shard* a query across non-overlapping ``stars:`` ranges (and optionally
``pushed:`` windows), so each shard stays under the cap while their union covers
the whole space (``docs/tasks.md`` §5.2). Each shard is a normal repo search;
results across shards are de-overlapped by ``owner/repo``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import structlog

from collect.github_client import GitHubClient
from collect.sources import RepoRef

__all__ = ["star_shards", "build_query", "discover_topics", "discover_query"]

log = structlog.get_logger(__name__)

#: Default star-range shard edges; each consecutive pair becomes one shard.
DEFAULT_STAR_EDGES: tuple[int, ...] = (50, 100, 200, 500, 1000, 2000, 5000, 10000)


def star_shards(edges: Iterable[int] = DEFAULT_STAR_EDGES) -> list[str]:
    """Build ``stars:`` qualifier shards from a sorted list of edge values.

    Consecutive edges ``[a, b]`` produce ``stars:a..{b-1}``; the final edge ``z``
    produces an open-ended ``stars:>=z`` so nothing is dropped.

    Args:
        edges: Ascending star thresholds.

    Returns:
        A list of non-overlapping ``stars:`` qualifier strings.
    """
    e = sorted(set(int(x) for x in edges))
    shards: list[str] = []
    for i in range(len(e) - 1):
        lo, hi = e[i], e[i + 1] - 1
        if hi < lo:
            hi = lo
        shards.append(f"stars:{lo}..{hi}")
    if e:
        shards.append(f"stars:>={e[-1]}")
    return shards


def build_query(
    *,
    topic: str | None = None,
    language: str | None = None,
    pushed_after: str | None = None,
    extra: str | None = None,
) -> str:
    """Assemble a GitHub repo-search ``q`` string from common qualifiers.

    Args:
        topic: A ``topic:`` qualifier value.
        language: A ``language:`` qualifier value.
        pushed_after: A date (``YYYY-MM-DD``) for a ``pushed:>=`` recency filter.
        extra: Any additional raw qualifier fragment.

    Returns:
        The space-joined query string (without the star shard, which is appended
        per shard by :func:`discover_query`).
    """
    parts: list[str] = []
    if topic:
        parts.append(f"topic:{topic}")
    if language:
        parts.append(f"language:{language}")
    if pushed_after:
        parts.append(f"pushed:>={pushed_after}")
    if extra:
        parts.append(extra)
    return " ".join(parts)


def discover_query(
    client: GitHubClient,
    base_query: str,
    *,
    star_edges: Iterable[int] = DEFAULT_STAR_EDGES,
    per_shard: int = 100,
    domain_hint: str | None = None,
    max_repos: int | None = None,
) -> Iterator[RepoRef]:
    """Discover repos for a base query by iterating star-range shards.

    Args:
        client: An initialized :class:`GitHubClient`.
        base_query: The shared qualifier string (topic/language/recency).
        star_edges: Star-range shard edges (see :func:`star_shards`).
        per_shard: Max results to pull per shard.
        domain_hint: Optional domain to tag each discovered ref with.
        max_repos: Optional global cap across all shards.

    Yields:
        Unique :class:`RepoRef` objects (de-overlapped by ``owner/repo``).
    """
    seen: set[str] = set()
    emitted = 0
    for shard in star_shards(star_edges):
        q = f"{base_query} {shard}".strip()
        log.info("collect.topics.shard", query=q)
        repos: list[dict[str, Any]] = client.search_repositories(
            q, sort="stars", order="desc", max_results=per_shard
        )
        for repo in repos:
            full = repo.get("full_name") or ""
            if not full or full.lower() in seen:
                continue
            seen.add(full.lower())
            owner, _, name = full.partition("/")
            yield RepoRef(owner=owner, repo=name, source_list=None, domain_hint=domain_hint)
            emitted += 1
            if max_repos is not None and emitted >= max_repos:
                return


def discover_topics(
    client: GitHubClient,
    seeds: list[dict[str, Any]],
    *,
    star_edges: Iterable[int] = DEFAULT_STAR_EDGES,
    per_shard: int = 100,
    pushed_after: str | None = None,
    max_repos_per_seed: int | None = None,
) -> Iterator[RepoRef]:
    """Discover repos across a matrix of ``(topic, language, domain)`` seeds.

    Each seed is a dict with optional keys ``topic``, ``language``, ``domain``,
    ``extra``. Recency (``pushed_after``) is applied uniformly to bias toward fresh
    repos (a contamination mitigation, ``docs/tasks.md`` §9.4).

    Args:
        client: An initialized :class:`GitHubClient`.
        seeds: The seed matrix (from ``harvest.yaml``).
        star_edges: Star-range shard edges.
        per_shard: Max results per shard.
        pushed_after: Optional recency cutoff date applied to every seed.
        max_repos_per_seed: Optional cap per seed.

    Yields:
        :class:`RepoRef` objects, deduped within each seed.
    """
    for seed in seeds:
        base = build_query(
            topic=seed.get("topic"),
            language=seed.get("language"),
            pushed_after=pushed_after or seed.get("pushed_after"),
            extra=seed.get("extra"),
        )
        yield from discover_query(
            client,
            base,
            star_edges=star_edges,
            per_shard=per_shard,
            domain_hint=seed.get("domain"),
            max_repos=max_repos_per_seed,
        )
