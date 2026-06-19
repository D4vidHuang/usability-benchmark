"""The uniform LLM client interface.

EVERYTHING that talks to a model -- agent-under-test, oracle, and LLM judges --
goes through the :class:`LLMClient` protocol. Two concrete impls satisfy it (an
Anthropic wrapper and an OpenAI/vLLM wrapper); they normalize provider responses
into the single :class:`Completion` shape defined here.

This module is deliberately dependency-light: it defines the *interface* and the
normalized data models only. Concrete clients (which import ``anthropic`` /
``openai`` lazily) live in sibling modules so importing this one is cheap.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from usabench.core.enums import Provider
from usabench.core.schema import Usage

__all__ = [
    "Message",
    "ToolSpec",
    "ToolCall",
    "Completion",
    "Usage",
    "LLMClient",
]


class Message(BaseModel):
    """A single chat message in the normalized request format.

    Attributes:
        role: One of ``system`` | ``user`` | ``assistant`` | ``tool``.
        content: The text content of the message.
        name: Optional name (e.g. tool name for ``role="tool"``).
        tool_call_id: Optional id linking a ``tool`` message to its call.
    """

    model_config = ConfigDict(extra="forbid")

    role: str = Field(..., description="system | user | assistant | tool")
    content: str = Field("", description="Message text content.")
    name: str | None = Field(None, description="Optional name (e.g. tool name).")
    tool_call_id: str | None = Field(None, description="Links a tool result to its call.")


class ToolSpec(BaseModel):
    """A tool/function the model may call, in provider-agnostic JSON-schema form."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Tool name.")
    description: str = Field("", description="What the tool does.")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="JSON-Schema for the tool arguments."
    )


class ToolCall(BaseModel):
    """A normalized tool/function call emitted by the model.

    Attributes:
        id: Provider call id (echoed back when returning the tool result).
        name: Name of the tool being called.
        arguments: Parsed argument dict (providers return JSON strings; clients
            parse them into a dict here).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Provider tool-call id.")
    name: str = Field(..., description="Tool name.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Parsed call arguments.")


class Completion(BaseModel):
    """The single normalized response shape every provider is mapped into.

    Attributes:
        text: The assistant's text output (empty if it only returned tool calls).
        tool_calls: Any tool/function calls the model requested.
        usage: Token + cost accounting for this call.
        model: The model id that produced the completion.
        finish_reason: Provider finish reason (stop | length | tool_calls | ...).
        provider: Which backend produced it.
        raw: The raw provider payload, retained for debugging only.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field("", description="Assistant text output.")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Requested tool calls.")
    usage: Usage = Field(default_factory=Usage, description="Token + cost accounting.")
    model: str = Field("", description="Model id that produced the completion.")
    finish_reason: str | None = Field(None, description="Provider finish reason.")
    provider: Provider | None = Field(None, description="Backend provider.")
    raw: dict[str, Any] = Field(default_factory=dict, description="Raw provider payload (debug).")


@runtime_checkable
class LLMClient(Protocol):
    """The uniform model-access interface implemented by every backend.

    A client maps a list of :class:`Message` (plus optional tools and decoding
    params) to a normalized :class:`Completion`. Implementations are responsible
    for retry/backoff, usage accounting, and (optionally) on-disk caching, but
    must NOT leak provider-specific shapes past the :class:`Completion` boundary.
    """

    #: The provider this client speaks to.
    provider: Provider

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
        """Run one chat completion and return a normalized :class:`Completion`.

        Args:
            messages: The conversation so far.
            tools: Optional tool specs the model may call.
            temperature: Sampling temperature (0.0 = greedy where supported).
            max_tokens: Maximum completion tokens.
            seed: Optional seed for providers that honor it (recorded regardless).
            stop: Optional stop sequences.
            **kwargs: Provider-specific passthrough (e.g. ``top_p``).

        Returns:
            A normalized :class:`Completion`.

        Raises:
            ProviderError: If the call fails after retries are exhausted.
        """
        ...
