"""A real GitHub REST + GraphQL client over :mod:`httpx`.

This is the network spine of the collector (``docs/tasks.md`` §5.2-§5.3). It:

* authenticates with a token from ``$GITHUB_TOKEN`` (scopes: public read),
* reads ``X-RateLimit-Remaining`` / ``X-RateLimit-Reset`` from every response and
  **sleeps until reset** when the primary budget runs low,
* honors the *secondary* (abuse) rate limit: respects ``Retry-After`` and applies
  exponential backoff with jitter on ``403``/``429``/``5xx``,
* issues conditional requests via an ETag :class:`~collect.cache.HttpCache` so
  re-runs are nearly free (a ``304`` costs no primary budget),
* paginates ``Link``-header REST collections and the GitHub search API (which
  caps at 1000 results -- callers shard around that, see :mod:`collect.sources`).

The client is synchronous and politely paced (a small inter-request delay): the
collector is strictly read-only and runs well under GitHub's limits.

Only :mod:`httpx`, :mod:`tenacity`-style backoff (implemented inline to avoid a
hard ordering dependency), and the stdlib are used. ``httpx`` is a *core* dep, so
this module imports cleanly with the base install.
"""

from __future__ import annotations

import os
import random
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import httpx
import structlog

from collect.cache import HttpCache
from usabench.core.errors import ProviderError

__all__ = [
    "GitHubConfig",
    "RateLimitState",
    "GitHubResponse",
    "GitHubClient",
    "GitHubError",
]

log = structlog.get_logger(__name__)

#: Default REST API root.
REST_ROOT = "https://api.github.com"
#: GraphQL endpoint.
GRAPHQL_URL = "https://api.github.com/graphql"
#: API version pin per GitHub's documented header.
API_VERSION = "2022-11-28"

#: Parses ``<url>; rel="next"`` segments of a REST ``Link`` header.
_LINK_RE = re.compile(r'<(?P<url>[^>]+)>;\s*rel="(?P<rel>[^"]+)"')


class GitHubError(ProviderError):
    """A GitHub API failure surfaced after retries/rate-limit handling."""


@dataclass(slots=True)
class GitHubConfig:
    """Tunables for the GitHub client.

    Attributes:
        token: OAuth/PAT token; falls back to ``$GITHUB_TOKEN``.
        rest_root: REST API base URL (overridable for GitHub Enterprise).
        graphql_url: GraphQL endpoint URL.
        per_request_delay_s: Polite inter-request sleep (``<=1 req/200ms`` default).
        rate_floor: Sleep-until-reset when remaining primary budget drops below this.
        max_retries: Max retries for transient/abuse failures.
        backoff_base_s: Base seconds for exponential backoff.
        backoff_cap_s: Max single backoff sleep.
        timeout_s: Per-request HTTP timeout.
        user_agent: Sent as the ``User-Agent`` header.
    """

    token: str | None = None
    rest_root: str = REST_ROOT
    graphql_url: str = GRAPHQL_URL
    per_request_delay_s: float = 0.2
    rate_floor: int = 50
    max_retries: int = 6
    backoff_base_s: float = 2.0
    backoff_cap_s: float = 300.0
    timeout_s: float = 30.0
    user_agent: str = "usability-benchmark-collector/0.1 (+https://github.com/D4vidHuang/usability-benchmark)"


@dataclass(slots=True)
class RateLimitState:
    """The most recent rate-limit headers observed.

    Attributes:
        limit: ``X-RateLimit-Limit``.
        remaining: ``X-RateLimit-Remaining``.
        reset_at: Epoch seconds of ``X-RateLimit-Reset``.
        resource: ``X-RateLimit-Resource`` (e.g. ``core``, ``search``, ``graphql``).
        graphql_cost: Last GraphQL ``rateLimit.cost`` if requested.
        graphql_remaining: Last GraphQL ``rateLimit.remaining`` if requested.
    """

    limit: int = 0
    remaining: int = 0
    reset_at: float = 0.0
    resource: str = ""
    graphql_cost: int = 0
    graphql_remaining: int = 0


@dataclass(slots=True)
class GitHubResponse:
    """A normalized response from :meth:`GitHubClient.request`.

    Attributes:
        status_code: HTTP status code (``304`` indicates a cache hit served).
        data: JSON-decoded body (dict or list), possibly served from cache.
        from_cache: True if the body came from the ETag cache (``304``).
        headers: The raw response headers (empty on a pure cache serve).
        etag: The response ``ETag``, if any.
    """

    status_code: int
    data: Any
    from_cache: bool = False
    headers: dict[str, str] = field(default_factory=dict)
    etag: str | None = None


class GitHubClient:
    """A synchronous, rate-limit-aware GitHub REST + GraphQL client.

    Construct once and reuse; it owns an :class:`httpx.Client` and an optional
    :class:`~collect.cache.HttpCache`. All methods are blocking and may sleep to
    respect rate limits. The client never raises on a normal ``304`` -- it serves
    the cached body transparently.
    """

    def __init__(self, config: GitHubConfig | None = None, cache: HttpCache | None = None) -> None:
        """Initialize the client.

        Args:
            config: Client tunables. A default :class:`GitHubConfig` is used if
                omitted; its ``token`` falls back to ``$GITHUB_TOKEN``.
            cache: Optional ETag cache enabling conditional requests.
        """
        self.config = config or GitHubConfig()
        token = self.config.token or os.environ.get("GITHUB_TOKEN")
        self._token = token
        self.cache = cache
        self.rate = RateLimitState()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": self.config.user_agent,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(headers=headers, timeout=self.config.timeout_s)

    # -- lifecycle ---------------------------------------------------------- #

    @property
    def has_token(self) -> bool:
        """True if a token is configured (required for non-trivial collection)."""
        return bool(self._token)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- rate-limit accounting --------------------------------------------- #

    def _record_rate(self, response: httpx.Response) -> None:
        """Update :attr:`rate` from a response's ``X-RateLimit-*`` headers."""
        h = response.headers
        try:
            if "X-RateLimit-Remaining" in h:
                self.rate.remaining = int(h["X-RateLimit-Remaining"])
            if "X-RateLimit-Limit" in h:
                self.rate.limit = int(h["X-RateLimit-Limit"])
            if "X-RateLimit-Reset" in h:
                self.rate.reset_at = float(h["X-RateLimit-Reset"])
            if "X-RateLimit-Resource" in h:
                self.rate.resource = h["X-RateLimit-Resource"]
        except (ValueError, TypeError):  # pragma: no cover - defensive
            pass

    def _respect_primary_limit(self) -> None:
        """Sleep until reset if the primary budget is below the safety floor."""
        if self.rate.remaining and self.rate.remaining < self.config.rate_floor:
            sleep_s = max(0.0, self.rate.reset_at - time.time()) + 1.0
            if sleep_s > 0:
                log.warning(
                    "github.rate_limit.sleep_until_reset",
                    remaining=self.rate.remaining,
                    floor=self.config.rate_floor,
                    sleep_s=round(sleep_s, 1),
                    resource=self.rate.resource,
                )
                time.sleep(min(sleep_s, self.config.backoff_cap_s))

    def _backoff_sleep(self, attempt: int, retry_after: str | None) -> None:
        """Sleep before a retry, honoring ``Retry-After`` then exponential backoff."""
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = self.config.backoff_base_s
        else:
            delay = self.config.backoff_base_s * (2 ** attempt)
        delay = min(delay, self.config.backoff_cap_s)
        delay += random.uniform(0, min(1.0, delay * 0.1))  # jitter
        log.warning("github.backoff", attempt=attempt, sleep_s=round(delay, 1))
        time.sleep(delay)

    @staticmethod
    def _is_secondary_limit(response: httpx.Response) -> bool:
        """Detect a secondary (abuse) rate-limit response."""
        if response.status_code not in (403, 429):
            return False
        if "Retry-After" in response.headers:
            return True
        body = response.text.lower()
        return "secondary rate limit" in body or "abuse" in body

    # -- core request ------------------------------------------------------- #

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        use_cache: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> GitHubResponse:
        """Perform one GitHub request with rate-limit + ETag handling.

        Absolute URLs are used as-is; bare paths (``/search/repositories``) are
        joined onto :attr:`GitHubConfig.rest_root`. On ``304 Not Modified`` the
        cached body is returned with ``from_cache=True``. Transient (``5xx``) and
        secondary-rate-limit (``403``/``429``) responses are retried with backoff.

        Args:
            method: HTTP method (``GET``/``POST``).
            url: Absolute URL or REST path.
            params: Query parameters.
            json_body: JSON request body (for GraphQL POST).
            use_cache: Whether to send/store conditional validators.
            extra_headers: Additional per-request headers.

        Returns:
            A :class:`GitHubResponse`.

        Raises:
            GitHubError: On a non-retryable error or after retries are exhausted.
        """
        full_url = url if url.startswith("http") else f"{self.config.rest_root}{url}"
        headers: dict[str, str] = dict(extra_headers or {})
        if use_cache and self.cache is not None and method.upper() == "GET":
            headers.update(self.cache.conditional_headers(full_url))

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            self._respect_primary_limit()
            if self.config.per_request_delay_s:
                time.sleep(self.config.per_request_delay_s)
            try:
                response = self._client.request(
                    method, full_url, params=params, json=json_body, headers=headers
                )
            except httpx.HTTPError as exc:  # network/timeout
                last_exc = exc
                if attempt < self.config.max_retries:
                    self._backoff_sleep(attempt, None)
                    continue
                raise GitHubError(
                    f"network error for {method} {full_url}: {exc}", provider="github"
                ) from exc

            self._record_rate(response)

            if response.status_code == 304:
                cached = self.cache.get(full_url) if self.cache else None
                return GitHubResponse(
                    status_code=304,
                    data=cached.body if cached else None,
                    from_cache=True,
                    etag=cached.etag if cached else None,
                )

            if self._is_secondary_limit(response):
                if attempt < self.config.max_retries:
                    self._backoff_sleep(attempt, response.headers.get("Retry-After"))
                    continue
                raise GitHubError(
                    "secondary rate limit not cleared after retries",
                    provider="github",
                    status=response.status_code,
                )

            # Primary-rate-limit exhaustion: remaining==0 on a 403.
            if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
                self._respect_primary_limit()
                if attempt < self.config.max_retries:
                    continue

            if response.status_code >= 500:
                if attempt < self.config.max_retries:
                    self._backoff_sleep(attempt, response.headers.get("Retry-After"))
                    continue

            if response.status_code >= 400:
                raise GitHubError(
                    f"GitHub {method} {full_url} -> {response.status_code}: {response.text[:500]}",
                    provider="github",
                    status=response.status_code,
                )

            data = self._decode(response)
            etag = response.headers.get("ETag")
            if use_cache and self.cache is not None and method.upper() == "GET":
                self.cache.store(
                    full_url,
                    etag=etag,
                    last_modified=response.headers.get("Last-Modified"),
                    status=response.status_code,
                    body=data,
                )
            return GitHubResponse(
                status_code=response.status_code,
                data=data,
                from_cache=False,
                headers=dict(response.headers),
                etag=etag,
            )

        # Exhausted loop without returning (only reachable via continues).
        raise GitHubError(
            f"request failed after {self.config.max_retries} retries: {method} {full_url}"
            + (f" ({last_exc})" if last_exc else ""),
            provider="github",
        )

    @staticmethod
    def _decode(response: httpx.Response) -> Any:
        """Decode a response body as JSON, tolerating empty bodies."""
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    # -- convenience wrappers ---------------------------------------------- #

    def get(self, url: str, *, params: dict[str, Any] | None = None, use_cache: bool = True) -> Any:
        """GET a single resource and return its decoded body.

        Args:
            url: REST path or absolute URL.
            params: Query parameters.
            use_cache: Whether to use conditional caching.

        Returns:
            The decoded JSON body.
        """
        return self.request("GET", url, params=params, use_cache=use_cache).data

    def paginate(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        per_page: int = 100,
        max_pages: int = 10,
        use_cache: bool = True,
    ) -> Iterator[Any]:
        """Yield items across a ``Link``-paginated REST collection.

        Handles both list bodies (``/issues``) and search-style ``{items: [...]}``
        bodies. Follows ``rel="next"`` until exhausted or ``max_pages`` is hit.

        Args:
            url: REST path or absolute URL of the first page.
            params: Query parameters for the first page.
            per_page: Page size (capped at 100 by GitHub).
            max_pages: Safety cap on pages followed.
            use_cache: Whether to use conditional caching.

        Yields:
            Individual items from each page.
        """
        query = dict(params or {})
        query.setdefault("per_page", min(per_page, 100))
        next_url: str | None = url
        next_params: dict[str, Any] | None = query
        pages = 0
        while next_url and pages < max_pages:
            resp = self.request("GET", next_url, params=next_params, use_cache=use_cache)
            pages += 1
            body = resp.data
            items = body.get("items") if isinstance(body, dict) else body
            if isinstance(items, list):
                yield from items
            elif items is not None:
                yield items
            # `Link` header is absent on pure cache serves; stop there.
            link = resp.headers.get("Link", "") if resp.headers else ""
            next_url = self._next_link(link)
            next_params = None  # the next link already carries the cursor

    @staticmethod
    def _next_link(link_header: str) -> str | None:
        """Extract the ``rel="next"`` URL from a ``Link`` header, or ``None``."""
        for m in _LINK_RE.finditer(link_header or ""):
            if m.group("rel") == "next":
                return m.group("url")
        return None

    def search_repositories(
        self, query: str, *, sort: str | None = None, order: str = "desc", max_results: int = 100
    ) -> list[dict[str, Any]]:
        """Run a repository search, paging up to ``max_results`` (<=1000 per GitHub).

        Args:
            query: The GitHub search qualifier string (``q``).
            sort: Optional sort field (``stars``, ``updated``, ...).
            order: ``asc`` or ``desc``.
            max_results: Cap on returned repos (GitHub hard-caps a query at 1000).

        Returns:
            A list of repo dicts (the search ``items``).
        """
        params: dict[str, Any] = {"q": query, "order": order}
        if sort:
            params["sort"] = sort
        out: list[dict[str, Any]] = []
        pages = max(1, (min(max_results, 1000) + 99) // 100)
        for item in self.paginate(
            "/search/repositories", params=params, per_page=100, max_pages=pages
        ):
            out.append(item)
            if len(out) >= max_results:
                break
        return out

    def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        """Fetch a single repository's metadata."""
        return cast("dict[str, Any]", self.get(f"/repos/{owner}/{repo}"))

    def get_readme(self, owner: str, repo: str, *, ref: str | None = None) -> str | None:
        """Fetch and decode a repository README, or ``None`` if absent.

        Args:
            owner: Repo owner.
            repo: Repo name.
            ref: Optional git ref (commit/branch) to pin the README to.

        Returns:
            The decoded README text, or ``None`` if there is no README.
        """
        import base64

        params = {"ref": ref} if ref else None
        try:
            body = self.get(f"/repos/{owner}/{repo}/readme", params=params)
        except GitHubError as exc:
            if exc.status == 404:
                return None
            raise
        if not isinstance(body, dict) or "content" not in body:
            return None
        try:
            return base64.b64decode(body["content"]).decode("utf-8", errors="replace")
        except (ValueError, KeyError):  # pragma: no cover - defensive
            return None

    def list_issues(
        self,
        owner: str,
        repo: str,
        *,
        labels: list[str] | None = None,
        state: str = "open",
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """List issues (excluding PRs) for a repo, optionally filtered by labels.

        Args:
            owner: Repo owner.
            repo: Repo name.
            labels: Optional label filter (comma-joined for the API).
            state: ``open`` | ``closed`` | ``all``.
            max_results: Cap on returned issues.

        Returns:
            A list of issue dicts with ``pull_request`` entries removed.
        """
        params: dict[str, Any] = {"state": state}
        if labels:
            params["labels"] = ",".join(labels)
        out: list[dict[str, Any]] = []
        for item in self.paginate(
            f"/repos/{owner}/{repo}/issues", params=params, per_page=50, max_pages=2
        ):
            if isinstance(item, dict) and "pull_request" in item:
                continue  # issues endpoint also returns PRs
            out.append(item)
            if len(out) >= max_results:
                break
        return out

    def path_exists(self, owner: str, repo: str, path: str, *, ref: str | None = None) -> bool:
        """Return True if ``path`` exists in the repo (for tests/CI detection)."""
        params = {"ref": ref} if ref else None
        try:
            body = self.get(f"/repos/{owner}/{repo}/contents/{path}", params=params)
        except GitHubError as exc:
            if exc.status == 404:
                return False
            raise
        return body is not None

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a GraphQL query and return its ``data`` object.

        Records the GraphQL ``rateLimit`` cost/remaining onto :attr:`rate` when the
        query selects them. Raises on GraphQL ``errors``.

        Args:
            query: The GraphQL document.
            variables: Optional variables map.

        Returns:
            The ``data`` object of the response.

        Raises:
            GitHubError: On transport failure or GraphQL ``errors``.
        """
        body = {"query": query, "variables": variables or {}}
        resp = self.request("POST", self.config.graphql_url, json_body=body, use_cache=False)
        payload = resp.data
        if not isinstance(payload, dict):
            raise GitHubError(f"unexpected GraphQL response: {payload!r}", provider="github")
        if payload.get("errors"):
            raise GitHubError(f"GraphQL errors: {payload['errors']}", provider="github")
        data = payload.get("data") or {}
        rl = data.get("rateLimit") if isinstance(data, dict) else None
        if isinstance(rl, dict):
            self.rate.graphql_cost = int(rl.get("cost", 0) or 0)
            self.rate.graphql_remaining = int(rl.get("remaining", 0) or 0)
        return data
