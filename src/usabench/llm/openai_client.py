"""OpenAI-compatible LLM client (serves BOTH OpenAI and vLLM).

This single client drives any backend that exposes the OpenAI ``/v1/chat/completions``
shape: the hosted OpenAI API *and* a locally-served vLLM model. The only
differences between the two are ``base_url`` and ``api_key`` -- the uniformity
decision in ``docs/infra.md`` §3. The provider enum (``openai`` vs ``vllm``) is
carried through purely for reporting/cost attribution.

The ``openai`` SDK is imported **lazily inside methods** so the base package
imports with only core deps; install the ``[api]`` extra to actually call OpenAI,
and point ``base_url`` at a vLLM server for open-weight models.
"""

from __future__ import annotations

import json
import os
from typing import Any

from usabench.core.enums import Provider
from usabench.core.errors import ConfigError, ProviderError
from usabench.core.schema import Usage
from usabench.llm.cache import ResponseCache, cache_key
from usabench.llm.client import Completion, Message, ToolCall, ToolSpec
from usabench.llm.retry import RetryConfig, call_with_retry
from usabench.llm.usage import Channel, PriceTable, UsageMeter
from usabench.logging_setup import get_logger

__all__ = ["OpenAIClient"]

_log = get_logger("usabench.llm.openai")


def _messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert normalized :class:`Message` objects to OpenAI chat dicts."""
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.name is not None:
            d["name"] = m.name
        if m.tool_call_id is not None:
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out


def _tools_to_openai(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
    """Convert normalized tool specs to the OpenAI ``tools`` array (function form)."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


class OpenAIClient:
    """Concrete :class:`~usabench.llm.client.LLMClient` for OpenAI / vLLM backends.

    Attributes:
        provider: ``Provider.OPENAI`` or ``Provider.VLLM`` (reporting only).
        model: The served model id.
    """

    def __init__(
        self,
        *,
        model: str,
        provider: Provider = Provider.OPENAI,
        api_key: str | None = None,
        base_url: str | None = None,
        price: PriceTable | None = None,
        retry: RetryConfig | None = None,
        usage_meter: UsageMeter | None = None,
        channel: Channel = Channel.AGENT,
        cache: ResponseCache | None = None,
        default_params: dict[str, Any] | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        """Construct an OpenAI-compatible client.

        Args:
            model: Served model id (must match vLLM ``--served-model-name``).
            provider: ``OPENAI`` or ``VLLM`` -- only base_url/api_key differ.
            api_key: API key (or a local dummy for vLLM). Required at call time.
            base_url: Override endpoint (vLLM: ``http://host:port/v1``).
            price: Token price table for cost accounting.
            retry: Retry/backoff policy.
            usage_meter: Meter to record tokens + cost into.
            channel: Which channel (agent/oracle/judge) usage is billed to.
            cache: Optional on-disk response cache (off by default).
            default_params: Sampling defaults merged under per-call overrides.
            timeout_s: Per-request timeout passed to the SDK.
        """
        if not provider.is_openai_shaped:
            raise ConfigError(f"OpenAIClient cannot serve provider {provider!r}")
        self.provider = provider
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._price = price or PriceTable()
        self._retry = retry or RetryConfig()
        self._usage_meter = usage_meter
        self._channel = channel
        self._cache = cache
        self._default_params = default_params or {}
        self._timeout_s = timeout_s
        self._client: Any = None  # lazily constructed SDK client

    # -- SDK plumbing ------------------------------------------------------- #

    def _ensure_client(self) -> Any:
        """Lazily build and cache the underlying ``openai`` SDK client."""
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - requires [api] extra
            raise ProviderError(
                "the 'openai' package is required for OpenAI/vLLM clients; "
                "install with: pip install 'usability-benchmark[api]'",
                provider=str(self.provider),
            ) from exc

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # vLLM accepts any non-empty key; use a dummy so the SDK is happy.
            api_key = "local-dummy" if self.provider is Provider.VLLM else None
        if not api_key:
            raise ConfigError(
                "no API key for OpenAI client (set api_key or OPENAI_API_KEY)"
            )

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self._timeout_s}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)
        return self._client

    # -- protocol ----------------------------------------------------------- #

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Completion:
        """Run one chat completion against the OpenAI-compatible endpoint.

        Args mirror the :class:`~usabench.llm.client.LLMClient` protocol. Cache is
        consulted (if configured) before any network call; on a miss the call is
        retried per :attr:`_retry` and usage is recorded into the meter.

        Returns:
            A normalized :class:`Completion`.

        Raises:
            ProviderError: If the call fails after retries.
        """
        params = {**self._default_params}
        params.update(kwargs)
        key: str | None = None
        if self._cache is not None:
            key = cache_key(
                model=self.model,
                provider=str(self.provider),
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                stop=stop,
                extra=params,
            )
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        client = self._ensure_client()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_to_openai(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        oa_tools = _tools_to_openai(tools)
        if oa_tools is not None:
            request["tools"] = oa_tools
        if seed is not None:
            request["seed"] = seed
        if stop:
            request["stop"] = stop
        request.update(params)

        def _do_call() -> Any:
            return client.chat.completions.create(**request)

        raw = call_with_retry(_do_call, self._retry, provider=str(self.provider))
        completion = self._normalize(raw)

        if self._usage_meter is not None:
            self._usage_meter.record(completion.usage, channel=self._channel)
        if self._cache is not None and key is not None:
            self._cache.set(key, completion)
        return completion

    # -- normalization ------------------------------------------------------ #

    def _normalize(self, raw: Any) -> Completion:
        """Map a provider SDK response object into a :class:`Completion`."""
        try:
            payload = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
        except Exception:  # noqa: BLE001 - keep raw best-effort
            payload = {}

        choice = raw.choices[0] if getattr(raw, "choices", None) else None
        message = getattr(choice, "message", None)
        text = getattr(message, "content", None) or ""
        finish_reason = getattr(choice, "finish_reason", None)

        tool_calls: list[ToolCall] = []
        for tc in getattr(message, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") if fn else ""
            raw_args = getattr(fn, "arguments", "") if fn else ""
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", "") or "",
                    name=name,
                    arguments=_parse_arguments(raw_args),
                )
            )

        usage = self._usage_from_raw(raw)
        return Completion(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            model=getattr(raw, "model", None) or self.model,
            finish_reason=finish_reason,
            provider=self.provider,
            raw=payload if isinstance(payload, dict) else {},
        )

    def _usage_from_raw(self, raw: Any) -> Usage:
        """Extract token counts from the SDK response and compute cost."""
        u = getattr(raw, "usage", None)
        prompt_tokens = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
        completion_tokens = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
        cost = self._price.cost_usd(prompt_tokens, completion_tokens)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )


def _parse_arguments(raw_args: Any) -> dict[str, Any]:
    """Parse a tool-call ``arguments`` value (JSON string or dict) into a dict."""
    if isinstance(raw_args, dict):
        return raw_args
    if not raw_args:
        return {}
    try:
        parsed = json.loads(raw_args)
    except (TypeError, ValueError):
        return {"_raw": str(raw_args)}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}
