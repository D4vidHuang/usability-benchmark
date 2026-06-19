"""Tests for the uniform LLM client layer (``usabench.llm``).

Covers the FakeLLMClient scripting modes, factory dispatch, usage/cost accounting,
the budget charge hook, retry/backoff semantics (provider-error wrapping and the
no-tenacity fallback), response-cache round-trip, and provider-response
normalization for the OpenAI/vLLM and Anthropic clients (using duck-typed mock SDK
objects, so no network and no SDK installs are required).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from usabench.core.enums import Provider
from usabench.core.errors import BudgetExceeded, ProviderError
from usabench.llm import (
    Channel,
    Completion,
    FakeLLMClient,
    LLMClient,
    Message,
    PriceTable,
    ResponseCache,
    RetryConfig,
    UsageMeter,
    build_client,
    cache_key,
    call_with_retry,
    estimate_cost_usd,
)
from usabench.llm.anthropic_client import (
    AnthropicClient,
    _messages_to_anthropic,
    _split_system,
)
from usabench.llm.openai_client import (
    OpenAIClient,
    _messages_to_openai,
    _parse_arguments,
)


@pytest.fixture
def msg() -> list[Message]:
    return [Message(role="user", content="hi")]


# -- FakeLLMClient ---------------------------------------------------------- #


def test_fake_satisfies_protocol_and_sequence(msg: list[Message]) -> None:
    fc = FakeLLMClient(script=["hello", "world"])
    assert isinstance(fc, LLMClient)
    assert fc.provider is Provider.FAKE
    assert fc.chat(msg).text == "hello"
    assert fc.chat(msg).text == "world"
    assert fc.chat(msg).text == ""  # exhausted -> empty default
    assert fc.call_count == 3


def test_fake_round_robin_and_reset(msg: list[Message]) -> None:
    rr = FakeLLMClient(script=["a", "b"], round_robin=True)
    assert [rr.chat(msg).text for _ in range(5)] == ["a", "b", "a", "b", "a"]
    rr.reset()
    assert rr.call_count == 0


def test_fake_keyed_with_tool_calls(msg: list[Message]) -> None:
    keyed = {
        "q1": {
            "text": "",
            "tool_calls": [{"id": "t1", "name": "write_file", "arguments": {"path": "x"}}],
        }
    }
    kc = FakeLLMClient(keyed=keyed, key_fn=lambda _m: "q1")
    comp = kc.chat(msg)
    assert comp.tool_calls[0].name == "write_file"
    assert comp.finish_reason == "tool_calls"
    assert comp.provider is Provider.FAKE


def test_fake_echo_responder() -> None:
    echo = FakeLLMClient.echo(prefix="> ")
    assert echo.chat([Message(role="user", content="ping")]).text == "> ping"


# -- usage / pricing -------------------------------------------------------- #


def test_usage_meter_splits_channels(msg: list[Message]) -> None:
    meter = UsageMeter()
    agent = FakeLLMClient(
        default="x", usage_meter=meter, channel=Channel.AGENT,
        prompt_tokens_per_call=10, completion_tokens_per_call=5,
    )
    oracle = FakeLLMClient(
        default="y", usage_meter=meter, channel=Channel.ORACLE,
        prompt_tokens_per_call=3, completion_tokens_per_call=2,
    )
    agent.chat(msg)
    agent.chat(msg)
    oracle.chat(msg)
    assert meter.totals(Channel.AGENT).total_tokens == 30
    assert meter.totals(Channel.ORACLE).total_tokens == 5
    assert meter.total_cost_usd == 0.0
    assert meter.total_tokens == 35


def test_price_table_cost() -> None:
    pt = PriceTable(input_per_mtok=15.0, output_per_mtok=75.0)
    assert pt.cost_usd(1_000_000, 0) == pytest.approx(15.0)
    assert pt.cost_usd(0, 1_000_000) == pytest.approx(75.0)
    assert estimate_cost_usd(2_000_000, 0, {"input": 5.0, "output": 15.0}) == pytest.approx(10.0)
    assert PriceTable.from_config(None).cost_usd(1000, 1000) == 0.0


def test_charge_hook_can_raise_budget(msg: list[Message]) -> None:
    spent = {"tok": 0}

    def hook(_ch: Channel, usage: Any) -> None:
        spent["tok"] += usage.total_tokens
        if spent["tok"] > 20:
            raise BudgetExceeded("tokens", 20, spent["tok"])

    meter = UsageMeter(charge_hook=hook)
    client = FakeLLMClient(default="z", usage_meter=meter, prompt_tokens_per_call=15)
    client.chat(msg)  # 15 ok
    with pytest.raises(BudgetExceeded):
        client.chat(msg)  # 30 -> breach


# -- factory ---------------------------------------------------------------- #


def test_factory_dispatch(msg: list[Message]) -> None:
    fb = build_client({"provider": "fake", "model": "fk", "script": ["s1"]})
    assert isinstance(fb, FakeLLMClient)
    assert fb.chat(msg).text == "s1"
    assert build_client({"provider": "anthropic", "model": "c", "api_key": "k"}).provider is Provider.ANTHROPIC
    assert build_client({"provider": "openai", "model": "g", "api_key": "k"}).provider is Provider.OPENAI
    assert build_client({"provider": "vllm", "model": "Q", "base_url_env": "X"}).provider is Provider.VLLM


# -- retry ------------------------------------------------------------------ #


class _Transient(Exception):
    status_code = 503


class _Fatal(Exception):
    status_code = 400


def test_retry_success_after_transient() -> None:
    n = {"i": 0}

    def flaky() -> str:
        n["i"] += 1
        if n["i"] < 3:
            raise _Transient()
        return "done"

    out = call_with_retry(
        flaky, RetryConfig(max_attempts=5, base_delay_s=0, jitter_s=0), sleep=lambda _s: None
    )
    assert out == "done" and n["i"] == 3


def test_retry_wraps_fatal_and_exhaustion() -> None:
    with pytest.raises(ProviderError) as ei:
        call_with_retry(lambda: (_ for _ in ()).throw(_Fatal()), RetryConfig(), sleep=lambda _s: None)
    assert ei.value.status == 400

    with pytest.raises(ProviderError) as ej:
        call_with_retry(
            lambda: (_ for _ in ()).throw(_Transient()),
            RetryConfig(max_attempts=2, base_delay_s=0, jitter_s=0),
            sleep=lambda _s: None,
        )
    assert ej.value.status == 503


# -- cache ------------------------------------------------------------------ #


def test_cache_round_trip(tmp_path: Any, msg: list[Message]) -> None:
    cache = ResponseCache(tmp_path)
    key = cache_key(
        model="m", provider="fake", messages=msg, tools=None,
        temperature=0.0, max_tokens=10, seed=None, stop=None,
    )
    assert cache.get(key) is None
    cache.set(key, Completion(text="cached!"))
    assert cache.get(key).text == "cached!"


# -- normalization (mock SDK objects) --------------------------------------- #


def test_openai_normalize_and_cost() -> None:
    client = OpenAIClient(model="gpt-x", price=PriceTable(input_per_mtok=5, output_per_mtok=15))
    raw = SimpleNamespace(
        model="gpt-x",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="hi there",
                    tool_calls=[
                        SimpleNamespace(
                            id="c1",
                            function=SimpleNamespace(name="run_cmd", arguments='{"cmd":"ls"}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=2_000_000),
    )
    comp = client._normalize(raw)
    assert comp.text == "hi there"
    assert comp.tool_calls[0].name == "run_cmd"
    assert comp.tool_calls[0].arguments == {"cmd": "ls"}
    assert comp.provider is Provider.OPENAI
    assert comp.usage.cost_usd == pytest.approx(35.0)


def test_openai_request_shaping() -> None:
    om = _messages_to_openai(
        [
            Message(role="system", content="sys"),
            Message(role="tool", content="42", tool_call_id="c1", name="run_cmd"),
        ]
    )
    assert om[0] == {"role": "system", "content": "sys"}
    assert om[1]["tool_call_id"] == "c1" and om[1]["name"] == "run_cmd"
    assert _parse_arguments('{"a":1}') == {"a": 1}
    assert _parse_arguments("not json") == {"_raw": "not json"}


def test_anthropic_normalize_and_cost() -> None:
    client = AnthropicClient(model="claude-x", price=PriceTable(input_per_mtok=15, output_per_mtok=75))
    raw = SimpleNamespace(
        model="claude-x",
        content=[
            SimpleNamespace(type="text", text="hello "),
            SimpleNamespace(type="text", text="world"),
            SimpleNamespace(type="tool_use", id="tu1", name="ask_user", input={"q": "tz?"}),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=1_000_000),
    )
    comp = client._normalize(raw)
    assert comp.text == "hello world"
    assert comp.tool_calls[0].name == "ask_user"
    assert comp.tool_calls[0].arguments == {"q": "tz?"}
    assert comp.finish_reason == "tool_use"
    assert comp.provider is Provider.ANTHROPIC
    assert comp.usage.cost_usd == pytest.approx(90.0)


def test_anthropic_system_split_and_tool_result() -> None:
    system, rest = _split_system(
        [
            Message(role="system", content="A"),
            Message(role="system", content="B"),
            Message(role="user", content="hi"),
        ]
    )
    assert system == "A\n\nB" and len(rest) == 1
    am = _messages_to_anthropic(
        [
            Message(role="tool", content="res", tool_call_id="tu1"),
            Message(role="assistant", content="ok"),
        ]
    )
    assert am[0]["content"][0]["type"] == "tool_result"
    assert am[0]["content"][0]["tool_use_id"] == "tu1"
    assert am[1]["role"] == "assistant"


def test_missing_sdk_raises_clear_error(msg: list[Message]) -> None:
    # When the provider SDK is absent, the lazy import inside the client must raise
    # a ProviderError with install guidance rather than a bare ImportError. If the
    # SDK happens to be installed in this environment, skip (the network call would
    # otherwise fail for unrelated reasons).
    import importlib.util

    for sdk, client in (
        ("openai", OpenAIClient(model="m", api_key="k")),
        ("anthropic", AnthropicClient(model="m", api_key="k")),
    ):
        if importlib.util.find_spec(sdk) is not None:
            continue  # SDK present -> not exercising the missing-SDK path
        with pytest.raises(ProviderError) as ei:
            client.chat(msg)
        assert "install" in str(ei.value).lower()
