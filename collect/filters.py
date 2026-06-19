"""Quality gates, license allowlist, dedup, and PII/secret scrubbing.

Three responsibilities (``docs/tasks.md`` §5.1, §5.3-§5.4, §9.3):

1. **License + quality gates** -- an SPDX allowlist for the *redistributable* pool
   plus stars/recency/size/activity gates. ``passes_quality_gates`` returns a
   boolean and the list of human-readable reasons (for the ``candidates.jsonl``
   curator fields).
2. **Dedup** -- exact ``owner/repo`` keys, fork-collapse, and near-duplicate
   clustering over ``description + README`` via MinHash (``datasketch`` if present)
   with a deterministic shingled-hash fallback so the base install still dedups.
3. **PII / secret scrub** -- redact emails and common credential patterns from any
   stored excerpt before it lands in ``raw_harvest.jsonl``.

``datasketch`` is imported **lazily** behind ``try/except`` so the package imports
with only core deps (DESIGN frozen decision #6).
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import structlog

__all__ = [
    "PERMISSIVE_SPDX",
    "GateConfig",
    "is_redistributable",
    "passes_quality_gates",
    "scrub_text",
    "scrub_record",
    "minhash_available",
    "near_dup_clusters",
    "dedup_records",
    "shingles",
]

log = structlog.get_logger(__name__)

#: SPDX ids permitted into the *redistributable* pool (``docs/tasks.md`` §5.1, §9.2).
PERMISSIVE_SPDX: frozenset[str] = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Unlicense",
        "MPL-2.0",
        "0BSD",
        "BSD-3-Clause-Clear",
    }
)

# --- PII / secret scrub patterns ------------------------------------------- #
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_SECRET_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI-style keys
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),  # Slack tokens
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google API keys
)


@dataclass(slots=True)
class GateConfig:
    """Thresholds for the harvest-time quality gates (``docs/tasks.md`` §5.1).

    Attributes:
        min_stars: Minimum stargazers.
        max_age_months: Reject repos not pushed within this many months (recency
            bias is also a contamination mitigation, §9.4).
        allow_archived: Permit archived repos.
        allow_forks: Permit forks (normally collapsed to upstream instead).
        min_size_kb: Minimum repo size (filters empty stubs).
        max_size_kb: Maximum repo size (keeps tasks modest in scope).
        require_redistributable: Require a permissive license for the pool.
        require_description: Require a non-empty description.
    """

    min_stars: int = 50
    max_age_months: int = 18
    allow_archived: bool = False
    allow_forks: bool = False
    min_size_kb: int = 5
    max_size_kb: int = 200_000
    require_redistributable: bool = True
    require_description: bool = True


def is_redistributable(spdx: str | None) -> bool:
    """Return True if ``spdx`` is in the permissive redistributable allowlist.

    Args:
        spdx: An SPDX license id (case-insensitive match against the allowlist).

    Returns:
        Whether the license permits redistribution of derived material.
    """
    if not spdx:
        return False
    norm = spdx.strip()
    # Allowlist comparison is case-insensitive on the canonical SPDX spelling.
    return any(norm.lower() == lic.lower() for lic in PERMISSIVE_SPDX)


def _months_since(iso_ts: str | None, *, now: _dt.datetime | None = None) -> float | None:
    """Months elapsed since an ISO-8601 timestamp, or ``None`` if unparseable."""
    if not iso_ts:
        return None
    ref = now or _dt.datetime.now(tz=_dt.UTC)
    try:
        ts = _dt.datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.UTC)
    return (ref - ts).days / 30.4375


def passes_quality_gates(
    record: dict[str, Any], config: GateConfig | None = None, *, now: _dt.datetime | None = None
) -> tuple[bool, list[str]]:
    """Evaluate the quality/license/activity gates for a harvest record.

    Args:
        record: A normalized ``raw_harvest`` record dict.
        config: Gate thresholds (defaults applied if omitted).
        now: Reference time for recency (defaults to now, UTC).

    Returns:
        ``(passed, reasons)`` where ``reasons`` lists every failed gate (empty when
        ``passed`` is True).
    """
    cfg = config or GateConfig()
    reasons: list[str] = []

    stars = int(record.get("stars") or 0)
    if stars < cfg.min_stars:
        reasons.append(f"stars<{cfg.min_stars} (got {stars})")

    if record.get("archived") and not cfg.allow_archived:
        reasons.append("archived")
    if record.get("fork") and not cfg.allow_forks:
        reasons.append("fork")

    months = _months_since(record.get("pushed_at"), now=now)
    if months is not None and months > cfg.max_age_months:
        reasons.append(f"stale (pushed {months:.1f}mo ago > {cfg.max_age_months})")

    size_kb = record.get("size_kb")
    if size_kb is not None:
        if size_kb < cfg.min_size_kb:
            reasons.append(f"too small ({size_kb}KB < {cfg.min_size_kb})")
        if size_kb > cfg.max_size_kb:
            reasons.append(f"too large ({size_kb}KB > {cfg.max_size_kb})")

    if cfg.require_redistributable and not record.get("redistributable"):
        reasons.append(f"non-permissive license ({record.get('license_spdx')})")

    if cfg.require_description and not (record.get("description") or "").strip():
        reasons.append("no description")

    return (len(reasons) == 0, reasons)


# --- PII / secret scrub ----------------------------------------------------- #


def scrub_text(text: str | None) -> str | None:
    """Redact emails and common credential patterns from a string.

    Args:
        text: Arbitrary text (e.g. a README or issue excerpt), or ``None``.

    Returns:
        The text with PII/secrets replaced by ``[REDACTED-*]`` markers, or ``None``.
    """
    if not text:
        return text
    out = text
    for pattern in _SECRET_RES:
        out = pattern.sub("[REDACTED-SECRET]", out)
    out = _EMAIL_RE.sub("[REDACTED-EMAIL]", out)
    return out


def scrub_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``record`` with stored excerpts scrubbed of PII/secrets.

    Args:
        record: A ``raw_harvest`` record dict.

    Returns:
        A scrubbed copy (the input is not mutated).
    """
    out = dict(record)
    out["readme_excerpt"] = scrub_text(out.get("readme_excerpt"))
    if out.get("description"):
        out["description"] = scrub_text(out["description"])
    issues = []
    for issue in out.get("candidate_issues") or []:
        issue = dict(issue)
        issue["body_excerpt"] = scrub_text(issue.get("body_excerpt"))
        issue["title"] = scrub_text(issue.get("title"))
        issues.append(issue)
    out["candidate_issues"] = issues
    return out


# --- Dedup ------------------------------------------------------------------ #


def minhash_available() -> bool:
    """Return True if ``datasketch`` is importable (enables MinHash dedup)."""
    try:
        import datasketch  # noqa: F401
    except Exception:
        return False
    return True


def shingles(text: str, k: int = 5) -> set[str]:
    """Produce the set of ``k``-word shingles of ``text`` (lowercased, normalized).

    Args:
        text: Input text.
        k: Shingle width in words.

    Returns:
        A set of space-joined ``k``-grams; falls back to individual tokens if the
        text is shorter than ``k`` words.
    """
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(tokens) < k:
        return set(tokens)
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _signature_text(record: dict[str, Any]) -> str:
    """Build the dedup signature text: description + first 2KB of README."""
    desc = record.get("description") or ""
    readme = (record.get("readme_excerpt") or "")[:2048]
    return f"{desc}\n{readme}"


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two sets (0.0 for two empty sets)."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _fallback_clusters(records: list[dict[str, Any]], threshold: float) -> list[list[int]]:
    """Deterministic shingled-Jaccard near-dup clustering (no datasketch).

    Greedy single-link clustering: each record joins the first existing cluster
    whose representative exceeds ``threshold``, else opens a new cluster.
    """
    sigs = [shingles(_signature_text(r)) for r in records]
    clusters: list[list[int]] = []
    reps: list[set[str]] = []
    for i, sig in enumerate(sigs):
        placed = False
        for c, rep in enumerate(reps):
            if _jaccard(sig, rep) >= threshold:
                clusters[c].append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
            reps.append(sig)
    return clusters


def _minhash_clusters(records: list[dict[str, Any]], threshold: float) -> list[list[int]]:
    """Near-dup clustering using ``datasketch`` MinHash + LSH (lazy import)."""
    from datasketch import MinHash, MinHashLSH

    num_perm = 128
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mh_by_idx: dict[int, Any] = {}
    for i, rec in enumerate(records):
        mh = MinHash(num_perm=num_perm)
        for sh in shingles(_signature_text(rec)):
            mh.update(sh.encode("utf-8"))
        mh_by_idx[i] = mh
        lsh.insert(str(i), mh)

    # Union-find over LSH-adjacent records to form connected clusters.
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(len(records)):
        for j_str in lsh.query(mh_by_idx[i]):
            union(i, int(j_str))

    groups: dict[int, list[int]] = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def near_dup_clusters(
    records: list[dict[str, Any]], *, threshold: float = 0.8
) -> list[list[int]]:
    """Cluster records by near-duplicate similarity of description + README.

    Uses MinHash/LSH when ``datasketch`` is installed, else a deterministic
    shingled-Jaccard fallback so the base install still dedups.

    Args:
        records: Normalized harvest records.
        threshold: Jaccard similarity at/above which two records are near-dups.

    Returns:
        A list of clusters; each cluster is a list of record indices.
    """
    if not records:
        return []
    if minhash_available():
        try:
            return _minhash_clusters(records, threshold)
        except Exception as exc:  # pragma: no cover - degrade gracefully
            log.warning("collect.dedup.minhash_failed", error=str(exc))
    return _fallback_clusters(records, threshold)


def dedup_records(
    records: Iterable[dict[str, Any]], *, threshold: float = 0.8
) -> list[dict[str, Any]]:
    """Dedup harvest records: exact key, fork-collapse, then near-dup clustering.

    Each output record gains ``dedup_cluster_id`` and ``dedup_representative``
    (the highest-starred member of its cluster). Exact ``owner/repo`` repeats are
    dropped first; the survivors are clustered by near-duplicate similarity.

    Args:
        records: An iterable of normalized harvest records.
        threshold: Near-dup Jaccard threshold.

    Returns:
        The deduped, annotated records (one per unique ``owner/repo``).
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for rec in records:
        key = f"{rec.get('owner')}/{rec.get('repo')}".lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(rec))

    clusters = near_dup_clusters(unique, threshold=threshold)
    for c, cluster in enumerate(clusters):
        cluster_id = f"cl_{c:04d}"
        # Representative = highest-starred member.
        rep_idx = max(cluster, key=lambda i: int(unique[i].get("stars") or 0))
        for i in cluster:
            unique[i]["dedup_cluster_id"] = cluster_id
            unique[i]["dedup_representative"] = i == rep_idx
    return unique
