"""GitHub data-collection ETL package.

Depends only on :mod:`usabench.core` types and the *core* runtime deps (httpx,
pyyaml, jsonschema, structlog, typer). Heavy/optional deps (``datasketch`` for
MinHash dedup) are imported **lazily** behind try/except so this package imports
with the base install (DESIGN frozen decision #6). Concrete modules:

* :mod:`collect.github_client` -- REST + GraphQL client (rate-limit + ETag aware).
* :mod:`collect.cache` -- stdlib-sqlite ETag HTTP cache.
* :mod:`collect.sources` -- topic/awesome/readme/issue discovery sources.
* :mod:`collect.normalize` -- API payloads -> ``raw_harvest`` records.
* :mod:`collect.filters` -- license/quality gates, dedup, PII/secret scrub.
* :mod:`collect.pipeline` -- resumable source->normalize->filter->JSONL pipeline.
* :mod:`collect.cli` -- the ``usabench-collect`` Typer app.

Submodules are imported on demand (not eagerly here) so ``import collect`` stays
cheap and side-effect-free.
"""

from __future__ import annotations

__all__ = [
    "GitHubClient",
    "GitHubConfig",
    "HttpCache",
    "CollectionPipeline",
    "PipelineConfig",
    "RepoRef",
]


def __getattr__(name: str) -> object:  # PEP 562 lazy attribute access
    """Lazily resolve the package's public API without eager submodule imports."""
    if name in ("GitHubClient", "GitHubConfig"):
        from collect import github_client

        return getattr(github_client, name)
    if name == "HttpCache":
        from collect.cache import HttpCache

        return HttpCache
    if name in ("CollectionPipeline", "PipelineConfig"):
        from collect import pipeline

        return getattr(pipeline, name)
    if name == "RepoRef":
        from collect.sources import RepoRef

        return RepoRef
    raise AttributeError(f"module 'collect' has no attribute {name!r}")
