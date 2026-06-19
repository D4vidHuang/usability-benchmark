"""Enhancement / feature-request / good-first-issue harvesting.

Under-specified issues phrased like a user's ask are prime *ambiguity material*
for the drafter (``docs/tasks.md`` §5.1-§5.2): an "it would be nice if..." feature
request mirrors exactly the kind of vague goal we want a task's ``user_goal`` to
read like. This source pulls labeled issues for a repo and normalizes them into
``candidate_issues`` entries.
"""

from __future__ import annotations

from typing import Any

import structlog

from collect.github_client import GitHubClient, GitHubError
from collect.normalize import normalize_issue
from collect.sources import RepoRef

__all__ = ["AMBIGUITY_LABELS", "harvest_issues"]

log = structlog.get_logger(__name__)

#: Labels that tend to mark under-specified, user-voiced asks.
AMBIGUITY_LABELS: tuple[str, ...] = (
    "enhancement",
    "feature request",
    "feature-request",
    "good first issue",
    "good-first-issue",
    "help wanted",
)


def harvest_issues(
    client: GitHubClient,
    ref: RepoRef,
    *,
    labels: list[str] | None = None,
    max_issues: int = 10,
) -> list[dict[str, Any]]:
    """Harvest labeled, under-specified issues for a repo as ambiguity material.

    Tries each candidate label; the issues API treats multiple labels as an AND,
    so we query labels one at a time and merge by issue number. Pull requests are
    excluded by :meth:`GitHubClient.list_issues`.

    Args:
        client: An initialized :class:`GitHubClient`.
        ref: The repo to harvest issues from.
        labels: Override label set (defaults to :data:`AMBIGUITY_LABELS`).
        max_issues: Cap on the total returned issues.

    Returns:
        A list of normalized ``candidate_issues`` entries (deduped by number).
    """
    label_set = list(labels or AMBIGUITY_LABELS)
    by_number: dict[int, dict[str, Any]] = {}
    for label in label_set:
        if len(by_number) >= max_issues:
            break
        try:
            issues = client.list_issues(
                ref.owner, ref.repo, labels=[label], state="open", max_results=max_issues
            )
        except GitHubError as exc:
            if exc.status in (404, 410):
                return []
            log.warning("collect.issues.error", repo=ref.full_name, label=label, status=exc.status)
            continue
        for issue in issues:
            norm = normalize_issue(issue)
            num = norm["number"]
            if num and num not in by_number:
                by_number[num] = norm
            if len(by_number) >= max_issues:
                break
    return list(by_number.values())[:max_issues]
