"""Task QC stage 6: task-level diversity (embedding dedup + domain/tier quotas).

The final task set must be diverse, not twelve near-identical "todo CLI" tasks
(``docs/tasks.md`` §5.4, §8.6). This stage operates on finished/draft task records:

1. **Near-twin dedup** -- embed each task's ``user_goal`` and drop any task whose
   cosine similarity to an earlier-kept task exceeds ``sim_threshold`` (default
   0.92, per ``docs/tasks.md`` §5.4). Embeddings come from ``sentence-transformers``
   if installed; otherwise we **degrade gracefully** to a deterministic hashed
   bag-of-words (character/word n-gram) vector, so the gate still runs (and is
   testable) with only core deps. The fallback is coarser but conservative.

2. **Domain / tier quotas** -- track coverage across the six domains and four tiers
   against target quotas and report which cells are under/over-filled, so curation
   can steer drafting toward balanced coverage.

Pure offline; no network. Heavy embedding deps are imported LAZILY behind
try/except (DESIGN frozen decision #6).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import structlog

__all__ = [
    "DOMAINS",
    "TIERS",
    "DiversityReport",
    "QuotaReport",
    "embeddings_available",
    "embed_texts",
    "cosine",
    "dedup_tasks",
    "quota_report",
    "run_diversity",
]

log = structlog.get_logger(__name__)

#: The six task domains (mirrors ``schemas/task.schema.json``).
DOMAINS: tuple[str, ...] = (
    "cli-util",
    "data-analysis",
    "web-dashboard",
    "api-integration",
    "automation",
    "dev-tooling",
)
#: The four difficulty tiers.
TIERS: tuple[str, ...] = ("T1", "T2", "T3", "T4")

#: Dimensionality of the deterministic hashed-bag fallback embedding.
_FALLBACK_DIM = 512


@dataclass(slots=True)
class DiversityReport:
    """The outcome of near-twin dedup over a task set.

    Attributes:
        kept_ids: Task ids retained (non-twins).
        dropped: Pairs ``(dropped_id, kept_id, similarity)`` for each removed twin.
        used_embeddings: True if a real embedding model was used (else fallback).
    """

    kept_ids: list[str] = field(default_factory=list)
    dropped: list[tuple[str, str, float]] = field(default_factory=list)
    used_embeddings: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "kept_ids": list(self.kept_ids),
            "dropped": [
                {"dropped_id": d, "near_to": k, "similarity": round(s, 4)}
                for (d, k, s) in self.dropped
            ],
            "used_embeddings": self.used_embeddings,
            "n_kept": len(self.kept_ids),
            "n_dropped": len(self.dropped),
        }


@dataclass(slots=True)
class QuotaReport:
    """Domain x tier coverage against target quotas.

    Attributes:
        counts: Observed counts keyed by ``"<domain>/<tier>"``.
        per_domain: Observed counts per domain.
        per_tier: Observed counts per tier.
        under_filled: Cells below their target (``"domain/tier"`` -> deficit).
        target_per_cell: The per-cell target used.
    """

    counts: dict[str, int] = field(default_factory=dict)
    per_domain: dict[str, int] = field(default_factory=dict)
    per_tier: dict[str, int] = field(default_factory=dict)
    under_filled: dict[str, int] = field(default_factory=dict)
    target_per_cell: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "counts": dict(self.counts),
            "per_domain": dict(self.per_domain),
            "per_tier": dict(self.per_tier),
            "under_filled": dict(self.under_filled),
            "target_per_cell": self.target_per_cell,
        }


# --------------------------------------------------------------------------- #
# Embeddings                                                                   #
# --------------------------------------------------------------------------- #


def embeddings_available() -> bool:
    """Return True if ``sentence-transformers`` is importable."""
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens of ``text``."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _hashed_bag_vector(text: str, dim: int = _FALLBACK_DIM) -> list[float]:
    """Deterministic L2-normalized hashed bag-of-words vector (no heavy deps).

    Uses word unigrams + bigrams hashed into ``dim`` buckets with TF weighting.
    Stable across processes (hashing is SHA-256-based, not Python's salted hash).

    Args:
        text: Input text.
        dim: Vector dimensionality.

    Returns:
        An L2-normalized list of ``dim`` floats.
    """
    toks = _tokens(text)
    grams = list(toks)
    grams += [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
    vec = [0.0] * dim
    for g, count in Counter(grams).items():
        h = int.from_bytes(hashlib.sha256(g.encode("utf-8")).digest()[:4], "big")
        vec[h % dim] += float(count)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def embed_texts(texts: list[str]) -> tuple[list[list[float]], bool]:
    """Embed a list of texts, preferring a real model and falling back gracefully.

    Args:
        texts: The strings to embed (e.g. ``user_goal``s).

    Returns:
        ``(vectors, used_model)`` -- ``used_model`` is True iff a real embedding
        model produced the vectors (False for the hashed-bag fallback).
    """
    if not texts:
        return [], False
    if embeddings_available():
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            raw = model.encode(texts, normalize_embeddings=True)
            vectors = [list(map(float, row)) for row in raw]
            return vectors, True
        except Exception as exc:  # pragma: no cover - degrade gracefully
            log.warning("qc.diversity.embed_model_failed", error=str(exc))
    return [_hashed_bag_vector(t) for t in texts], False


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is zero)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Dedup + quotas                                                               #
# --------------------------------------------------------------------------- #


def _goal_of(task: dict[str, Any]) -> str:
    """Extract the ``user_goal`` from a task or a ``{"task": ...}`` draft wrapper."""
    if "task" in task and isinstance(task["task"], dict):
        task = task["task"]
    return str(task.get("user_goal") or "")


def _id_of(task: dict[str, Any]) -> str:
    """Extract the task id from a task or a draft wrapper."""
    if "task" in task and isinstance(task["task"], dict):
        task = task["task"]
    return str(task.get("id") or "<unknown>")


def dedup_tasks(
    tasks: list[dict[str, Any]], *, sim_threshold: float = 0.92
) -> DiversityReport:
    """Drop near-twin tasks by ``user_goal`` embedding cosine similarity.

    Greedy: iterate in input order, keep a task unless it is too similar to an
    already-kept task (``docs/tasks.md`` §5.4 uses cosine >= 0.92).

    Args:
        tasks: Task dicts (or ``{"task": ...}`` draft wrappers).
        sim_threshold: Cosine similarity at/above which a task is a near-twin.

    Returns:
        A :class:`DiversityReport`.
    """
    report = DiversityReport()
    if not tasks:
        return report
    goals = [_goal_of(t) for t in tasks]
    ids = [_id_of(t) for t in tasks]
    vectors, used = embed_texts(goals)
    report.used_embeddings = used

    kept_idx: list[int] = []
    for i in range(len(tasks)):
        twin_of: int | None = None
        best_sim = 0.0
        for j in kept_idx:
            sim = cosine(vectors[i], vectors[j])
            if sim >= sim_threshold and sim > best_sim:
                best_sim, twin_of = sim, j
        if twin_of is None:
            kept_idx.append(i)
        else:
            report.dropped.append((ids[i], ids[twin_of], best_sim))
    report.kept_ids = [ids[i] for i in kept_idx]
    return report


def quota_report(
    tasks: list[dict[str, Any]], *, target_per_cell: int = 2
) -> QuotaReport:
    """Compute domain x tier coverage and under-filled cells against a quota.

    Args:
        tasks: Task dicts (or draft wrappers).
        target_per_cell: Desired minimum tasks per ``(domain, tier)`` cell.

    Returns:
        A :class:`QuotaReport`.
    """
    report = QuotaReport(target_per_cell=target_per_cell)
    cell_counts: Counter[str] = Counter()
    dom_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()

    for t in tasks:
        inner = t["task"] if "task" in t and isinstance(t["task"], dict) else t
        domain = str(inner.get("domain") or "")
        tier = str(inner.get("difficulty") or "")
        if domain in DOMAINS:
            dom_counts[domain] += 1
        if tier in TIERS:
            tier_counts[tier] += 1
        if domain in DOMAINS and tier in TIERS:
            cell_counts[f"{domain}/{tier}"] += 1

    report.counts = dict(cell_counts)
    report.per_domain = {d: dom_counts.get(d, 0) for d in DOMAINS}
    report.per_tier = {t: tier_counts.get(t, 0) for t in TIERS}

    for domain in DOMAINS:
        for tier in TIERS:
            cell = f"{domain}/{tier}"
            have = cell_counts.get(cell, 0)
            if have < target_per_cell:
                report.under_filled[cell] = target_per_cell - have
    return report


def run_diversity(
    tasks: list[dict[str, Any]],
    *,
    sim_threshold: float = 0.92,
    target_per_cell: int = 2,
) -> dict[str, Any]:
    """Run the full diversity gate: dedup then quota coverage.

    Args:
        tasks: Task dicts (or ``{"task": ...}`` draft wrappers).
        sim_threshold: Near-twin cosine threshold.
        target_per_cell: Quota target per ``(domain, tier)`` cell.

    Returns:
        A dict with ``dedup`` and ``quota`` sub-reports plus an ``ok`` flag (True
        when nothing was dropped and no cell is under target).
    """
    dedup = dedup_tasks(tasks, sim_threshold=sim_threshold)
    # Quota over the surviving (kept) tasks only.
    kept = set(dedup.kept_ids)
    survivors = [t for t in tasks if _id_of(t) in kept] if kept else tasks
    quota = quota_report(survivors, target_per_cell=target_per_cell)
    return {
        "dedup": dedup.as_dict(),
        "quota": quota.as_dict(),
        "ok": not dedup.dropped and not quota.under_filled,
    }
