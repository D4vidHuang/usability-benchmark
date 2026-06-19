"""Task-drafting package (LLM-assisted draft authoring from harvested candidates).

Stage 2 of the dataset pipeline (``docs/tasks.md`` §4, §6.1): turns a normalized
``candidates`` record into a draft ``task.json`` (lay ``user_goal`` + ambiguity
points, NO gold). Depends only on :mod:`usabench.core` and :mod:`usabench.llm`;
the concrete :mod:`drafting.draft_tasks` is imported on demand so this package
stays cheap to import.
"""

from __future__ import annotations

__all__ = [
    "DraftConfig",
    "DraftResult",
    "draft_from_candidate",
    "draft_file",
]


def __getattr__(name: str) -> object:  # PEP 562 lazy attribute access
    """Lazily resolve the package's public API without eager submodule imports."""
    if name in __all__:
        from drafting import draft_tasks

        return getattr(draft_tasks, name)
    raise AttributeError(f"module 'drafting' has no attribute {name!r}")
