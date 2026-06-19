"""Anthropic Messages API client, normalized to the uniform ``Completion`` shape.

Wraps the Anthropic ``messages.create`` endpoint and maps its response (content
blocks, ``tool_use`` blocks, ``input_tokens``/``output_tokens``) onto the single
:class:`~usabench.llm.client.Completion` shape every other backend produces, so the
harness/oracle/judge code paths are provider-agnostic.

The ``anthropic`` SDK is imported **lazily inside methods**; install the ``[api]``
extra to use this client. The benchmark's oracle is always an API model, so this
is the default oracle backend.

Key shape differences handled here:

* Anthropic takes the system prompt as a top-level ``system`` arg, not a message;
  we lift any ``role="system"`` messages out of the list.
* Anthropic requires ``max_tokens`` and rejects ``temperature`` outside ``[0, 1]``;
  we clamp.
* Tool results come back to the model as ``tool_result`` content blocks on a
  ``user`` message; we translate ``role="tool"`` messages accordingly.
"""

from __future__ import annotations

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

__all__ = ["AnthropicClient"]

_log = get_logger("usabench.llm.anthropic")


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Lift ``system`` messages into a single system string (Anthropic shape).

    Args:
        messages: The normalized conversation.

    Returns:
        ``(system_prompt_or_None, remaining_messages)``. Multiple system messages
        are concatenated with blank-line separators.
    """
    system_parts: list[str] = []
    rest: list[Message] = []
    for m in messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
        else:
            rest.append(m)
    system = "\n\n".join(system_parts) if system_parts else None
    return system, rest


def _messages_to_anthropic(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert non-system messages to Anthropic ``messages`` blocks.

    ``role="tool"`` messages become a ``user`` message carrying a ``tool_result``
    content block (linked by ``tool_call_id``). Plain text messages become a single
    text content block.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id or "",
                            "content": m.content,
                        }
                    ],
                }
            )
        else:
            role = "assistant" if m.role == "assistant" else "user"
            out.append({"role": role, "content": m.content})
    return out


def _tools_to_anthropic(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
    """Convert normalized tool specs to Anthropic ``tools`` (input_schema form)."""
    if not tools:
        return None
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters or {"type": "object", "properties": {}},
        }
        for t in tools
    ]


class AnthropicClient:
    """Concrete :class:`~usabench.llm.client.LLMClient` for the Anthropic API.

    Attributes:
        provider: Always :data:`Provider.ANTHROPIC`.
        model: The Anthropic model id.
    """

    provider: Provider = Provider.ANTHROPIC

    def __init__(
        self,
        *,
        model: str,
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
        """Construct an Anthropic client.

        Args:
            model: Anthropic model id (e.g. ``claude-opus-4-8``).
            api_key: API key; falls back to ``ANTHROPIC_API_KEY`` at call time.
            base_url: Optional endpoint override (proxies/gateways).
            price: Token price table for cost accounting.
            retry: Retry/backoff policy.
            usage_meter: Meter to record tokens + cost into.
            channel: Which channel (agent/oracle/judge) usage is billed to.
            cache: Optional on-disk response cache (off by default).
            default_params: Sampling defaults merged under per-call overrides.
            timeout_s: Per-request timeout passed to the SDK.
        """
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
        self._client: Any = None

    # -- SDK plumbing ------------------------------------------------------- #

    def _ensure_client(self) -> Any:
        """Lazily build and cache the underlying ``anthropic`` SDK client."""
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - requires [api] extra
            raise ProviderError(
                "the 'anthropic' package is required for the Anthropic client; "
                "install with: pip install 'usability-benchmark[api]'",
                provider=str(self.provider),
            ) from exc

        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigError(
                "no API key for Anthropic client (set api_key or ANTHROPIC_API_KEY)"
            )
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self._timeout_s}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = Anthropic(**kwargs)
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
        """Run one Anthropic message completion, normalized to :class:`Completion`.

        Note:
            Anthropic does not honor a ``seed``; it is recorded in the cache key and
            ignored by the API. ``temperature`` is clamped to ``[0, 1]``.

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
        system, rest = _split_system(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_to_anthropic(rest),
            "max_tokens": max_tokens,
            "temperature": max(0.0, min(1.0, temperature)),
        }
        if system:
            request["system"] = system
        an_tools = _tools_to_anthropic(tools)
        if an_tools is not None:
            request["tools"] = an_tools
        if stop:
            request["stop_sequences"] = stop
        # Anthropic ignores 'seed'; pass only recognized passthrough params.
        request.update(params)

        def _do_call() -> Any:
            return client.messages.create(**request)

        raw = call_with_retry(_do_call, self._retry, provider=str(self.provider))
        completion = self._normalize(raw)

        if self._usage_meter is not None:
            self._usage_meter.record(completion.usage, channel=self._channel)
        if self._cache is not None and key is not None:
            self._cache.set(key, completion)
        return completion

    # -- normalization ------------------------------------------------------ #

    def _normalize(self, raw: Any) -> Completion:
        """Map an Anthropic Messages response into a :class:`Completion`."""
        try:
            payload = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
        except Exception:  # noqa: BLE001
            payload = {}

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in getattr(raw, "content", None) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", "") or "",
                        name=getattr(block, "name", "") or "",
                        arguments=_coerce_input(getattr(block, "input", None)),
                    )
                )

        usage = self._usage_from_raw(raw)
        return Completion(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            model=getattr(raw, "model", None) or self.model,
            finish_reason=getattr(raw, "stop_reason", None),
            provider=self.provider,
            raw=payload if isinstance(payload, dict) else {},
        )

    def _usage_from_raw(self, raw: Any) -> Usage:
        """Extract token counts from the Anthropic response and compute cost."""
        u = getattr(raw, "usage", None)
        prompt_tokens = int(getattr(u, "input_tokens", 0) or 0) if u else 0
        completion_tokens = int(getattr(u, "output_tokens", 0) or 0) if u else 0
        cost = self._price.cost_usd(prompt_tokens, completion_tokens)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )


def _coerce_input(value: Any) -> dict[str, Any]:
    """Coerce a ``tool_use`` input (already a dict for Anthropic) into a dict."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"_value": value}
