"""Uniform LLM access layer.

The protocol + normalized data models, the concrete clients, the factory, and the
usage/retry/cache helpers are all exported here. Concrete clients import their
heavy SDKs (``anthropic`` / ``openai``) **lazily inside methods**, so importing
``usabench.llm`` (and every name below) stays cheap and works with only core deps
installed. ``tenacity`` is likewise imported lazily by :mod:`usabench.llm.retry`.
"""

from __future__ import annotations

from usabench.llm.anthropic_client import AnthropicClient
from usabench.llm.cache import ResponseCache, cache_key, default_cache_from_env
from usabench.llm.client import (
    Completion,
    LLMClient,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)
from usabench.llm.factory import ModelConfig, build_client
from usabench.llm.fake import FakeLLMClient, make_completion
from usabench.llm.openai_client import OpenAIClient
from usabench.llm.retry import RetryConfig, call_with_retry
from usabench.llm.usage import (
    Channel,
    ChannelTotals,
    PriceTable,
    UsageMeter,
    estimate_cost_usd,
)

__all__ = [
    # protocol + models
    "Completion",
    "LLMClient",
    "Message",
    "ToolCall",
    "ToolSpec",
    "Usage",
    # factory
    "ModelConfig",
    "build_client",
    # concrete clients
    "AnthropicClient",
    "OpenAIClient",
    "FakeLLMClient",
    "make_completion",
    # usage / pricing
    "Channel",
    "ChannelTotals",
    "PriceTable",
    "UsageMeter",
    "estimate_cost_usd",
    # retry
    "RetryConfig",
    "call_with_retry",
    # cache
    "ResponseCache",
    "cache_key",
    "default_cache_from_env",
]
