"""The resumable collection pipeline: source -> normalize -> filter -> JSONL.

This ties the sources together into one idempotent, restartable harvest
(``docs/tasks.md`` §4, §5.3). For each discovered :class:`~collect.sources.RepoRef`
it:

1. skips repos already processed (a ``seen.sqlite`` cursor keyed by ``owner/repo``),
2. enriches it (metadata + pinned README + tests/CI) via :mod:`collect.sources.readmes`,
3. harvests ambiguity-bearing issues via :mod:`collect.sources.issues`,
4. normalizes it into a ``raw_harvest`` record (:mod:`collect.normalize`),
5. scrubs PII/secrets and evaluates quality/license gates (:mod:`collect.filters`),
6. appends the record to ``raw_harvest.jsonl`` (only repos passing the gates by
   default, but the curator can keep rejects with ``keep_rejects=True``).

A final pass dedups the written records (exact + near-dup) and emits a
``candidates.jsonl`` with curator-facing fields. The pipeline never re-spends
budget on a repo it has already seen, so a killed run resumes cleanly.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from collect.cache import HttpCache
from collect.filters import GateConfig, dedup_records, passes_quality_gates, scrub_record
from collect.github_client import GitHubClient, GitHubConfig
from collect.normalize import make_provenance_id, normalize_repo
from collect.sources import RepoRef
from collect.sources.awesome_lists import discover_awesome_lists
from collect.sources.issues import harvest_issues
from collect.sources.readmes import enrich_repo
from collect.sources.topics import discover_topics

__all__ = ["SeenCursor", "PipelineConfig", "HarvestStats", "CollectionPipeline"]

log = structlog.get_logger(__name__)

_SEEN_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen (
    full_name  TEXT PRIMARY KEY,
    status     TEXT,
    seen_at    REAL
);
"""


class SeenCursor:
    """A SQLite cursor of processed ``owner/repo`` keys for resumability."""

    def __init__(self, path: str | Path) -> None:
        """Open (creating if needed) the seen-cursor database."""
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.executescript(_SEEN_SCHEMA)
        self._conn.commit()

    def is_seen(self, full_name: str) -> bool:
        """Return True if ``full_name`` (``owner/repo``) was already processed."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM seen WHERE full_name = ?", (full_name.lower(),)
            ).fetchone()
        return row is not None

    def mark(self, full_name: str, status: str) -> None:
        """Record ``full_name`` as processed with a terminal ``status``."""
        import time

        with self._lock:
            self._conn.execute(
                "INSERT INTO seen (full_name, status, seen_at) VALUES (?, ?, ?) "
                "ON CONFLICT(full_name) DO UPDATE SET status=excluded.status, seen_at=excluded.seen_at",
                (full_name.lower(), status, time.time()),
            )
            self._conn.commit()

    def count(self) -> int:
        """Return the number of repos processed so far."""
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0])

    def close(self) -> None:
        """Close the cursor database."""
        with self._lock:
            self._conn.close()


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for one harvest run.

    Attributes:
        out_dir: Directory for ``raw_harvest.jsonl``, ``candidates.jsonl``, sqlite.
        gates: Quality/license gate thresholds.
        seeds: The topic/language/domain seed matrix (for the ``topics`` source).
        domains: Domain keywords for awesome-list discovery.
        sources: Which sources to run (subset of ``topics``/``awesome_lists``).
        pushed_after: Recency cutoff date applied to topic discovery.
        max_repos: Global cap on repos discovered.
        max_issues: Issues harvested per repo.
        keep_rejects: Persist gate-failing repos too (with ``passes_quality_gates``).
        dedup_threshold: Near-dup Jaccard threshold for the final dedup pass.
    """

    out_dir: Path = field(default_factory=lambda: Path("data"))
    gates: GateConfig = field(default_factory=GateConfig)
    seeds: list[dict[str, Any]] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=lambda: ["topics", "awesome_lists"])
    pushed_after: str | None = None
    max_repos: int | None = None
    max_issues: int = 8
    keep_rejects: bool = False
    dedup_threshold: float = 0.8


@dataclass(slots=True)
class HarvestStats:
    """Counters describing the outcome of a harvest run."""

    discovered: int = 0
    skipped_seen: int = 0
    unavailable: int = 0
    written: int = 0
    rejected: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return the stats as a plain dict."""
        return {
            "discovered": self.discovered,
            "skipped_seen": self.skipped_seen,
            "unavailable": self.unavailable,
            "written": self.written,
            "rejected": self.rejected,
        }


class CollectionPipeline:
    """Orchestrates discovery -> enrichment -> normalize -> filter -> JSONL.

    The pipeline owns a :class:`GitHubClient` (with an ETag cache) and a
    :class:`SeenCursor`. It is constructed from a :class:`PipelineConfig`; call
    :meth:`run` to harvest, then :meth:`build_candidates` to emit the deduped
    ``candidates.jsonl``.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        client: GitHubClient | None = None,
        gh_config: GitHubConfig | None = None,
    ) -> None:
        """Initialize the pipeline and its output paths.

        Args:
            config: The harvest configuration.
            client: Optional pre-built GitHub client (else one is created with an
                ETag cache under ``out_dir``).
            gh_config: Optional GitHub client tunables (used if ``client`` is None).
        """
        self.config = config
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.out_dir / "raw_harvest.jsonl"
        self.candidates_path = self.out_dir / "candidates.jsonl"
        self.seen = SeenCursor(self.out_dir / "seen.sqlite")
        self._cache = HttpCache(self.out_dir / "http_cache.sqlite")
        self.client = client or GitHubClient(gh_config, cache=self._cache)
        self.provenance_seq = 0

    # -- discovery ---------------------------------------------------------- #

    def discover(self) -> Iterator[RepoRef]:
        """Yield repo references from the configured sources, deduped by name.

        Yields:
            Unique :class:`RepoRef` objects across all enabled sources.
        """
        seen_names: set[str] = set()
        emitted = 0

        def _emit(ref: RepoRef) -> RepoRef | None:
            nonlocal emitted
            key = ref.full_name.lower()
            if not ref.owner or not ref.repo or key in seen_names:
                return None
            seen_names.add(key)
            emitted += 1
            return ref

        if "topics" in self.config.sources and self.config.seeds:
            for ref in discover_topics(
                self.client,
                self.config.seeds,
                pushed_after=self.config.pushed_after,
                max_repos_per_seed=self.config.max_repos,
            ):
                out = _emit(ref)
                if out is not None:
                    yield out
                if self.config.max_repos is not None and emitted >= self.config.max_repos:
                    return

        if "awesome_lists" in self.config.sources and self.config.domains:
            for ref in discover_awesome_lists(self.client, self.config.domains):
                out = _emit(ref)
                if out is not None:
                    yield out
                if self.config.max_repos is not None and emitted >= self.config.max_repos:
                    return

    # -- per-repo processing ------------------------------------------------ #

    def process_ref(self, ref: RepoRef) -> dict[str, Any] | None:
        """Enrich, normalize, scrub, and gate a single repo reference.

        Args:
            ref: The repo reference to process.

        Returns:
            A scrubbed ``raw_harvest`` record dict, or ``None`` if the repo was
            unavailable (its caller marks the seen-cursor regardless).
        """
        enr = enrich_repo(self.client, ref)
        if enr is None:
            return None
        issues = harvest_issues(self.client, ref, max_issues=self.config.max_issues)
        self.provenance_seq += 1
        record = normalize_repo(
            enr.repo,
            provenance_id=make_provenance_id(self.provenance_seq),
            head_sha=enr.head_sha,
            readme=enr.readme,
            has_tests=enr.has_tests,
            has_ci=enr.has_ci,
            candidate_issues=issues,
            source_list=enr.source_list,
        )
        return scrub_record(record)

    # -- run ---------------------------------------------------------------- #

    def run(self, *, refs: Iterable[RepoRef] | None = None) -> HarvestStats:
        """Run the full harvest, appending passing records to ``raw_harvest.jsonl``.

        Args:
            refs: Optional explicit ref stream (defaults to :meth:`discover`).

        Returns:
            A :class:`HarvestStats` summary.
        """
        stats = HarvestStats()
        stream = refs if refs is not None else self.discover()
        with self.raw_path.open("a", encoding="utf-8") as fh:
            for ref in stream:
                stats.discovered += 1
                if self.seen.is_seen(ref.full_name):
                    stats.skipped_seen += 1
                    continue
                record = self.process_ref(ref)
                if record is None:
                    stats.unavailable += 1
                    self.seen.mark(ref.full_name, "unavailable")
                    continue
                passed, reasons = passes_quality_gates(record, self.config.gates)
                if passed or self.config.keep_rejects:
                    if not passed:
                        record["_gate_reasons"] = reasons
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    stats.written += 1 if passed else 0
                if not passed:
                    stats.rejected += 1
                self.seen.mark(ref.full_name, "kept" if passed else "rejected")
                log.info(
                    "collect.pipeline.processed",
                    repo=ref.full_name,
                    passed=passed,
                    reasons=reasons or None,
                )
        return stats

    # -- candidate building ------------------------------------------------- #

    def build_candidates(self) -> int:
        """Dedup ``raw_harvest.jsonl`` and write ``candidates.jsonl``.

        Reads every harvested record, applies exact + near-dup dedup, and writes a
        curator-facing ``candidates.jsonl`` adding ``passes_quality_gates``,
        ``gate_reasons``, ``suitability_prefilter_score`` and ``draft_status``.

        Returns:
            The number of candidate records written.
        """
        records = list(self._read_jsonl(self.raw_path))
        deduped = dedup_records(records, threshold=self.config.dedup_threshold)
        written = 0
        with self.candidates_path.open("w", encoding="utf-8") as fh:
            for rec in deduped:
                passed, reasons = passes_quality_gates(rec, self.config.gates)
                cand = dict(rec)
                cand["passes_quality_gates"] = passed
                cand["gate_reasons"] = reasons
                cand["suitability_prefilter_score"] = suitability_prefilter_score(rec)
                cand["draft_status"] = "pending"
                fh.write(json.dumps(cand, ensure_ascii=False) + "\n")
                written += 1
        log.info("collect.pipeline.candidates", written=written)
        return written

    @staticmethod
    def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
        """Yield decoded JSON objects from a ``.jsonl`` file (skipping blanks)."""
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def close(self) -> None:
        """Release the client, cache, and seen-cursor handles."""
        self.client.close()
        self._cache.close()
        self.seen.close()

    def __enter__(self) -> CollectionPipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def suitability_prefilter_score(record: dict[str, Any]) -> float:
    """Heuristic 0-1 suitability prefilter score (``docs/tasks.md`` §5.6).

    Rewards a feature-list-bearing README, tests/CI presence, enhancement issues,
    and a size in a task-appropriate band. This is a cheap pre-rank for curators,
    not the calibration gate.

    Args:
        record: A normalized harvest record.

    Returns:
        A score in ``[0, 1]``.
    """
    score = 0.0
    readme = (record.get("readme_excerpt") or "").lower()
    if readme:
        score += 0.25
    if any(tok in readme for tok in ("usage", "features", "install", "example", "options")):
        score += 0.20
    if record.get("has_tests"):
        score += 0.20
    if record.get("has_ci"):
        score += 0.10
    if record.get("candidate_issues"):
        score += 0.15
    size = record.get("size_kb")
    if size is not None and 50 <= size <= 6000:
        score += 0.10
    return round(min(score, 1.0), 3)
