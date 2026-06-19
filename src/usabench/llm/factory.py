"""Build a concrete :class:`~usabench.llm.client.LLMClient` from a model config.

``build_client(model_cfg)`` dispatches on the ``provider`` enum to the right
implementation, wiring in pricing, retry policy, the shared usage meter, the
channel, and an optional response cache. It accepts either a raw config dict (as
loaded from ``configs/models/*.yaml`` via
:func:`usabench.config.loader.load_yaml`) or a pre-validated :class:`ModelConfig`.

Config shape (``docs/infra.md`` §3)::

    id: claude-opus
    provider: anthropic            # anthropic | openai | vllm | fake
    model: claude-opus-4-8
    api_key_env: ANTHROPIC_API_KEY # env var holding the key
    base_url: ...                  # OpenAI/vLLM: explicit endpoint, or
    base_url_env: USABENCH_VLLM_BASE_URL   # ... resolved from env at build time
    api_key: ...                   # discouraged inline key (env preferred)
    params: { temperature: 0.7, max_tokens: 4096, seed: 7 }
    price_per_mtok: { input: 15.0, output: 75.0 }
    retry: { max_attempts: 6, base_delay_s: 1.5, max_delay_s: 60 }

For provider ``fake`` the config may carry a ``script``/``keyed``/``default`` block
that is forwarded to :class:`~usabench.llm.fake.FakeLLMClient`, powering the smoke
path with zero cost.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from usabench.core.enums import Provider
from usabench.core.errors import ConfigError
from usabench.llm.cache import ResponseCache
from usabench.llm.client import LLMClient
from usabench.llm.retry import RetryConfig
from usabench.llm.usage import Channel, PriceTable, UsageMeter
from usabench.logging_setup import get_logger

__all__ = ["ModelConfig", "build_client"]

_log = get_logger("usabench.llm.factory")


class ModelConfig(BaseModel):
    """Validated view of a ``configs/models/*.yaml`` record.

    Unknown keys are tolerated (``extra="ignore"``) so configs can carry serving
    hints (e.g. ``serving:``) the LLM layer ignores.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field("", description="Human-readable config id.")
    provider: Provider = Field(..., description="Backend provider.")
    model: str = Field(..., description="Served model id.")
    api_key: str | None = Field(None, description="Inline key (env preferred).")
    api_key_env: str | None = Field(None, description="Env var holding the key.")
    base_url: str | None = Field(None, description="Explicit endpoint.")
    base_url_env: str | None = Field(None, description="Env var holding the endpoint.")
    params: dict[str, Any] = Field(default_factory=dict, description="Sampling defaults.")
    price_per_mtok: dict[str, Any] | None = Field(None, description="Token prices.")
    retry: dict[str, Any] | None = Field(None, description="Retry/backoff block.")
    timeout_s: float = Field(120.0, ge=0.0, description="Per-request timeout.")
    # fake-only scripting (ignored by real providers)
    script: list[Any] | None = Field(None, description="FakeLLMClient sequence.")
    keyed: dict[str, Any] | None = Field(None, description="FakeLLMClient keyed map.")
    default: Any | None = Field(None, description="FakeLLMClient fallback.")

    def resolve_api_key(self, env: dict[str, str] | None = None) -> str | None:
        """Resolve the API key from inline value or ``api_key_env``."""
        if self.api_key:
            return self.api_key
        environ = os.environ if env is None else env
        if self.api_key_env:
            return environ.get(self.api_key_env)
        return None

    def resolve_base_url(self, env: dict[str, str] | None = None) -> str | None:
        """Resolve the base URL from inline value or ``base_url_env``."""
        if self.base_url:
            return self.base_url
        environ = os.environ if env is None else env
        if self.base_url_env:
            return environ.get(self.base_url_env)
        return None


def _coerce_config(model_cfg: ModelConfig | dict[str, Any]) -> ModelConfig:
    """Accept a dict or a :class:`ModelConfig` and return a validated config."""
    if isinstance(model_cfg, ModelConfig):
        return model_cfg
    try:
        return ModelConfig.model_validate(model_cfg)
    except Exception as exc:  # noqa: BLE001 - normalize to ConfigError
        raise ConfigError(f"invalid model config: {exc}") from exc


def build_client(
    model_cfg: ModelConfig | dict[str, Any],
    *,
    usage_meter: UsageMeter | None = None,
    channel: Channel = Channel.AGENT,
    cache: ResponseCache | None = None,
    env: dict[str, str] | None = None,
) -> LLMClient:
    """Build a concrete :class:`LLMClient` from a model config.

    Args:
        model_cfg: A model config dict or :class:`ModelConfig`.
        usage_meter: Shared usage meter to record tokens/cost into.
        channel: The channel (agent/oracle/judge) this client bills to.
        cache: Optional response cache (off by default).
        env: Optional environment override for key/URL resolution.

    Returns:
        A client satisfying the :class:`LLMClient` protocol.

    Raises:
        ConfigError: On an invalid/unsupported config.
    """
    cfg = _coerce_config(model_cfg)
    price = PriceTable.from_config(cfg.price_per_mtok)
    retry = RetryConfig.from_config(cfg.retry)

    _log.debug(
        "build_client",
        id=cfg.id,
        provider=str(cfg.provider),
        model=cfg.model,
        channel=str(channel),
    )

    if cfg.provider is Provider.FAKE:
        # Local import keeps the dependency graph tidy; FakeLLMClient is core-only.
        from usabench.llm.fake import FakeLLMClient

        return FakeLLMClient(
            script=cfg.script,
            keyed=cfg.keyed,
            default=cfg.default,
            model=cfg.model or "fake-model",
            usage_meter=usage_meter,
            channel=channel,
        )

    if cfg.provider is Provider.ANTHROPIC:
        from usabench.llm.anthropic_client import AnthropicClient

        return AnthropicClient(
            model=cfg.model,
            api_key=cfg.resolve_api_key(env),
            base_url=cfg.resolve_base_url(env),
            price=price,
            retry=retry,
            usage_meter=usage_meter,
            channel=channel,
            cache=cache,
            default_params=cfg.params,
            timeout_s=cfg.timeout_s,
        )

    if cfg.provider.is_openai_shaped:
        from usabench.llm.openai_client import OpenAIClient

        return OpenAIClient(
            model=cfg.model,
            provider=cfg.provider,
            api_key=cfg.resolve_api_key(env),
            base_url=cfg.resolve_base_url(env),
            price=price,
            retry=retry,
            usage_meter=usage_meter,
            channel=channel,
            cache=cache,
            default_params=cfg.params,
            timeout_s=cfg.timeout_s,
        )

    raise ConfigError(f"unsupported provider: {cfg.provider!r}")  # pragma: no cover
