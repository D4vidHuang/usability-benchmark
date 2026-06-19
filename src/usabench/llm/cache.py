"""Content-addressed on-disk response cache for LLM calls.

A debugging/CI aid only -- **OFF by default**. When enabled (``USABENCH_LLM_CACHE=1``
or an explicit :class:`ResponseCache` passed to a client), identical requests
(same model, messages, tools, and decoding params) return the previously stored
:class:`~usabench.llm.client.Completion` without spending tokens.

Per ``docs/infra.md`` §6.1 the cache is deliberately disabled for *measurement*
runs because cache hits would distort variance and cost. The cache key is the
SHA-256 of the canonical JSON of the request, so it is stable across machines and
process restarts; entries are plain JSON files under a sharded directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from usabench.core.ids import canonical_json, sha256_hex
from usabench.llm.client import Completion, Message, ToolSpec
from usabench.logging_setup import get_logger

__all__ = ["ResponseCache", "cache_key", "default_cache_from_env"]

_log = get_logger("usabench.llm.cache")

#: Env var that, when truthy, enables a default file cache rooted at ``$USABENCH_LLM_CACHE_DIR``.
_ENABLE_ENV = "USABENCH_LLM_CACHE"
_DIR_ENV = "USABENCH_LLM_CACHE_DIR"
_DEFAULT_DIR = ".cache/llm"


def cache_key(
    *,
    model: str,
    provider: str,
    messages: list[Message],
    tools: list[ToolSpec] | None,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    stop: list[str] | None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Compute the content-addressed cache key for a request.

    The key hashes everything that can change the response. ``extra`` carries any
    provider-specific passthrough kwargs so e.g. a different ``top_p`` is a cache
    miss.

    Args:
        model: Model id.
        provider: Provider value (``anthropic``/``openai``/``vllm``/``fake``).
        messages: The request messages.
        tools: Optional tool specs.
        temperature: Sampling temperature.
        max_tokens: Max completion tokens.
        seed: Optional seed.
        stop: Optional stop sequences.
        extra: Optional provider passthrough kwargs.

    Returns:
        A 64-char hex digest usable as a filename stem.
    """
    payload = {
        "model": model,
        "provider": provider,
        "messages": [m.model_dump(mode="json", exclude_none=True) for m in messages],
        "tools": (
            [t.model_dump(mode="json") for t in tools] if tools else None
        ),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "stop": stop,
        "extra": extra or {},
    }
    return sha256_hex(canonical_json(payload))


class ResponseCache:
    """A simple sharded JSON file cache for :class:`Completion` objects.

    Entries are stored at ``<root>/<key[:2]>/<key>.json``. The cache is
    process-safe for reads and uses an atomic write (temp file + ``os.replace``) so
    a concurrent reader never sees a partial file. Missing/corrupt entries simply
    behave as a miss.
    """

    def __init__(self, root: str | Path) -> None:
        """Initialize the cache rooted at ``root`` (created on first write)."""
        self.root = Path(root)

    def _path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Completion | None:
        """Return the cached :class:`Completion` for ``key`` or ``None`` on miss.

        A corrupt or unparseable entry is treated as a miss (and logged), never an
        error, so a bad cache file can never break a run.
        """
        path = self._path_for(key)
        if not path.is_file():
            return None
        try:
            data = path.read_text(encoding="utf-8")
            completion = Completion.model_validate_json(data)
        except Exception as exc:  # noqa: BLE001 - corrupt entry => miss
            _log.warning("llm_cache_corrupt", key=key, error=str(exc))
            return None
        _log.debug("llm_cache_hit", key=key)
        return completion

    def set(self, key: str, completion: Completion) -> None:
        """Atomically store ``completion`` under ``key``."""
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(completion.model_dump_json(), encoding="utf-8")
        os.replace(tmp, path)
        _log.debug("llm_cache_store", key=key)

    def clear(self) -> None:
        """Delete all cache entries (best-effort)."""
        if not self.root.exists():
            return
        for child in self.root.glob("**/*.json"):
            try:
                child.unlink()
            except OSError:  # pragma: no cover - best effort
                pass


def default_cache_from_env(env: dict[str, str] | None = None) -> ResponseCache | None:
    """Return a :class:`ResponseCache` if caching is enabled via env, else ``None``.

    Caching is OFF unless ``USABENCH_LLM_CACHE`` is truthy
    (``1``/``true``/``yes``/``on``). When on, the root is ``$USABENCH_LLM_CACHE_DIR``
    or ``./.cache/llm``.

    Args:
        env: Optional environment mapping (defaults to ``os.environ``).

    Returns:
        A cache instance when enabled, otherwise ``None``.
    """
    environ = os.environ if env is None else env
    flag = str(environ.get(_ENABLE_ENV, "")).strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return None
    root = environ.get(_DIR_ENV) or _DEFAULT_DIR
    _log.info("llm_cache_enabled", root=root)
    return ResponseCache(root)
