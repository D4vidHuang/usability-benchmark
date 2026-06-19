"""A tiny stdlib-only HTTP response cache for conditional GitHub requests.

The cache is a single SQLite file keyed by request URL. For each URL it stores
the last ``ETag`` (and ``Last-Modified``) header plus the JSON-decoded body and a
fetch timestamp. The collector uses it to issue conditional requests
(``If-None-Match`` / ``If-Modified-Since``); on a ``304 Not Modified`` GitHub
charges **zero** rate-limit budget and we serve the cached body instead
(``docs/tasks.md`` §5.3).

We deliberately use the stdlib :mod:`sqlite3` so the base ``collect`` install
needs no extra dependency (DESIGN frozen decision: ``cache.py`` uses stdlib
sqlite3 to avoid a hard dep).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

__all__ = ["CachedResponse", "HttpCache"]

log = structlog.get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    url           TEXT PRIMARY KEY,
    etag          TEXT,
    last_modified TEXT,
    status        INTEGER,
    body          TEXT,
    fetched_at    REAL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass(slots=True)
class CachedResponse:
    """A cached HTTP response entry.

    Attributes:
        url: The request URL (cache key).
        etag: The stored ``ETag`` header, if any.
        last_modified: The stored ``Last-Modified`` header, if any.
        status: The HTTP status of the cached response.
        body: The JSON-decoded response body.
        fetched_at: Epoch seconds when this entry was last written.
    """

    url: str
    etag: str | None
    last_modified: str | None
    status: int
    body: Any
    fetched_at: float


class HttpCache:
    """A thread-safe SQLite-backed ETag cache for GitHub API responses.

    The cache is opened lazily and is safe to share across threads (it serializes
    access behind a lock and uses ``check_same_thread=False``). It is intentionally
    small: callers ask for the conditional headers for a URL, then store the body
    and validators after a ``200``.
    """

    def __init__(self, path: str | Path) -> None:
        """Open (creating if needed) the cache database at ``path``.

        Args:
            path: Filesystem path to the SQLite cache file. Parent directories are
                created if missing. Use ``":memory:"`` for an ephemeral cache.
        """
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, url: str) -> CachedResponse | None:
        """Return the cached entry for ``url`` or ``None`` if absent.

        Args:
            url: The request URL.

        Returns:
            The :class:`CachedResponse` for ``url``, or ``None``.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT url, etag, last_modified, status, body, fetched_at "
                "FROM http_cache WHERE url = ?",
                (url,),
            ).fetchone()
        if row is None:
            return None
        body = json.loads(row[4]) if row[4] is not None else None
        return CachedResponse(
            url=row[0],
            etag=row[1],
            last_modified=row[2],
            status=int(row[3]),
            body=body,
            fetched_at=float(row[5]),
        )

    def conditional_headers(self, url: str) -> dict[str, str]:
        """Build conditional-request headers for ``url`` from the cached validators.

        Args:
            url: The request URL.

        Returns:
            A dict possibly containing ``If-None-Match`` and/or
            ``If-Modified-Since``; empty if the URL is not cached.
        """
        entry = self.get(url)
        headers: dict[str, str] = {}
        if entry is None:
            return headers
        if entry.etag:
            headers["If-None-Match"] = entry.etag
        if entry.last_modified:
            headers["If-Modified-Since"] = entry.last_modified
        return headers

    def store(
        self,
        url: str,
        *,
        etag: str | None,
        last_modified: str | None,
        status: int,
        body: Any,
    ) -> None:
        """Upsert a cache entry after a successful (``200``) fetch.

        Args:
            url: The request URL (cache key).
            etag: The response ``ETag`` header, if present.
            last_modified: The response ``Last-Modified`` header, if present.
            status: The HTTP status code.
            body: The JSON-decoded response body.
        """
        payload = json.dumps(body, ensure_ascii=False) if body is not None else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO http_cache (url, etag, last_modified, status, body, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(url) DO UPDATE SET "
                "etag=excluded.etag, last_modified=excluded.last_modified, "
                "status=excluded.status, body=excluded.body, fetched_at=excluded.fetched_at",
                (url, etag, last_modified, int(status), payload, time.time()),
            )
            self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        """Read a free-form metadata value (e.g. a resume cursor)."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        """Write a free-form metadata value."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> HttpCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
