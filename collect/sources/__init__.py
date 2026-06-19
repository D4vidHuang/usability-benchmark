"""Discovery sources for the GitHub collector.

Each source turns a configured query/seed into a stream of *repo references*
(``owner``, ``repo``, and the awesome-list name it came from, if any). The
:mod:`collect.pipeline` then enriches those refs (README, issues, tests/CI),
normalizes them (:mod:`collect.normalize`), filters them (:mod:`collect.filters`),
and writes ``raw_harvest.jsonl``.

The four sources mirror ``docs/tasks.md`` §5.2:

* :mod:`collect.sources.topics` -- topic/stars search discovery, sharded past the
  1000-result search cap.
* :mod:`collect.sources.awesome_lists` -- discover ``awesome-*`` lists and parse
  ``github.com/owner/repo`` links out of their READMEs.
* :mod:`collect.sources.readmes` -- fetch + pin a repo's README at a commit.
* :mod:`collect.sources.issues` -- harvest enhancement / feature-request /
  good-first-issue items for ambiguity material.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RepoRef", "SOURCE_NAMES"]

#: The registered source names exposed by ``usabench-collect list-sources``.
SOURCE_NAMES: tuple[str, ...] = ("topics", "awesome_lists", "readmes", "issues")


@dataclass(frozen=True, slots=True)
class RepoRef:
    """A lightweight reference to a candidate repository.

    Attributes:
        owner: Repository owner login.
        repo: Repository name.
        source_list: Name of the awesome-* list it was discovered via, if any.
        domain_hint: Optional domain guess carried from the discovery query.
    """

    owner: str
    repo: str
    source_list: str | None = None
    domain_hint: str | None = None

    @property
    def full_name(self) -> str:
        """``owner/repo`` slug."""
        return f"{self.owner}/{self.repo}"

    @classmethod
    def from_full_name(cls, full: str, **kwargs: object) -> RepoRef:
        """Build a :class:`RepoRef` from an ``owner/repo`` string."""
        owner, _, repo = full.partition("/")
        return cls(owner=owner, repo=repo, **kwargs)  # type: ignore[arg-type]
