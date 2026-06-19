"""Deterministic, scriptable :class:`~usabench.llm.client.LLMClient` for tests.

``FakeLLMClient`` is the backbone of the zero-cost smoke path and the unit/
integration tests: it returns *canned* completions with **no network and zero
cost**, fully reproducibly. It supports three scripting styles, checked in order:

1. **Keyed** -- a mapping from a *prompt key* to a canned response. The key may be
   a request hash (see :func:`~usabench.llm.cache.cache_key`) or any user-chosen
   string matched by a ``key_fn`` you supply (default: a hash of the last user
   message's text). Most precise; use for asserting exact prompt -> response.
2. **Sequence** -- an ordered list consumed round-robin (or once-through). Easiest
   for "first call returns X, second returns Y" scenarios.
3. **Default / responder** -- a fallback constant response, or a ``responder``
   callable ``(messages, **params) -> str | Completion`` for dynamic fakes (e.g. an
   agent that echoes, or an oracle that always answers "level 1").

Every entry may be a plain ``str`` (becomes ``Completion.text``), a
:class:`~usabench.llm.client.Completion`, or a ``dict`` validated into one. Tool
calls are supported by providing a ``Completion`` with ``tool_calls`` set, or a
dict with a ``tool_calls`` list.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from usabench.core.enums import Provider
from usabench.core.ids import sha256_hex
from usabench.core.schema import Usage
from usabench.llm.client import Completion, Message, ToolSpec
from usabench.llm.usage import Channel, UsageMeter

__all__ = ["FakeLLMClient", "ScriptEntry", "make_completion"]

#: Anything that can be coerced into a :class:`Completion`.
ScriptEntry = str | Completion | dict[str, Any]

#: A dynamic responder: receives the messages + call kwargs, returns an entry.
Responder = Callable[..., ScriptEntry]

#: Maps a request to a string key for the keyed-script lookup.
KeyFn = Callable[[list[Message]], str]


def make_completion(
    entry: ScriptEntry,
    *,
    model: str = "fake",
    provider: Provider = Provider.FAKE,
) -> Completion:
    """Coerce a script entry into a fully-formed :class:`Completion`.

    Args:
        entry: A string (-> ``text``), a :class:`Completion`, or a dict.
        model: Model id to stamp when the entry does not specify one.
        provider: Provider to stamp when the entry does not specify one.

    Returns:
        A normalized :class:`Completion` with ``provider``/``model`` filled in.

    Raises:
        TypeError: If ``entry`` is not a supported type.
    """
    if isinstance(entry, Completion):
        completion = entry.model_copy(deep=True)
    elif isinstance(entry, str):
        completion = Completion(text=entry)
    elif isinstance(entry, dict):
        completion = Completion.model_validate(entry)
    else:  # pragma: no cover - defensive
        raise TypeError(f"unsupported fake script entry type: {type(entry).__name__}")

    if not completion.model:
        completion.model = model
    if completion.provider is None:
        completion.provider = provider
    if completion.finish_reason is None:
        completion.finish_reason = "tool_calls" if completion.tool_calls else "stop"
    return completion


def _default_key_fn(messages: list[Message]) -> str:
    """Default keyed-lookup key: SHA-256 of the last user message's content."""
    last_user = ""
    for msg in reversed(messages):
        if msg.role == "user":
            last_user = msg.content
            break
    if not last_user and messages:
        last_user = messages[-1].content
    return sha256_hex(last_user)


class FakeLLMClient:
    """A deterministic, zero-cost LLM client implementing the LLMClient protocol.

    The client satisfies the structural ``LLMClient`` protocol (it has a
    ``provider`` attribute and a matching ``chat`` method). It records every call
    for assertions (:attr:`calls`) and bills ``$0`` into an optional
    :class:`~usabench.llm.usage.UsageMeter`.

    Example:
        Sequence script::

            client = FakeLLMClient(script=["hello", "world"])
            client.chat([Message(role="user", content="hi")]).text   # -> "hello"
            client.chat([Message(role="user", content="hi")]).text   # -> "world"

        Keyed script (oracle that always answers a clarification)::

            client = FakeLLMClient(keyed={"q1": '{"level": 1, "text": "use UTC"}'},
                                   key_fn=lambda msgs: "q1")
    """

    #: Provider value reported to the harness/factory.
    provider: Provider = Provider.FAKE

    def __init__(
        self,
        *,
        script: list[ScriptEntry] | None = None,
        keyed: dict[str, ScriptEntry] | None = None,
        default: ScriptEntry | None = None,
        responder: Responder | None = None,
        key_fn: KeyFn | None = None,
        round_robin: bool = False,
        model: str = "fake-model",
        usage_meter: UsageMeter | None = None,
        channel: Channel = Channel.AGENT,
        prompt_tokens_per_call: int = 0,
        completion_tokens_per_call: int = 0,
    ) -> None:
        """Configure the fake client's scripting.

        Args:
            script: Ordered responses consumed by call order.
            keyed: Mapping from a key (see ``key_fn``) to a response.
            default: Fallback response when nothing else matches.
            responder: Dynamic fallback callable ``(messages, **params) -> entry``.
            key_fn: Maps messages to a lookup key for ``keyed`` (default: hash of
                the last user message).
            round_robin: If True, a consumed ``script`` wraps around instead of
                falling through to ``default``/``responder`` once exhausted.
            model: Model id stamped on returned completions.
            usage_meter: Optional meter to record (zero-cost) usage into.
            channel: Which channel usage is recorded under.
            prompt_tokens_per_call: Synthetic prompt-token count to report (lets
                tests exercise token-budget paths with zero cost).
            completion_tokens_per_call: Synthetic completion-token count to report.
        """
        self._script = list(script) if script else []
        self._keyed = dict(keyed) if keyed else {}
        self._default = default
        self._responder = responder
        self._key_fn = key_fn or _default_key_fn
        self._round_robin = round_robin
        self.model = model
        self._usage_meter = usage_meter
        self._channel = channel
        self._prompt_tokens = prompt_tokens_per_call
        self._completion_tokens = completion_tokens_per_call

        self._lock = threading.Lock()
        self._cursor = 0
        #: Recorded calls (message lists + kwargs) for test assertions.
        self.calls: list[dict[str, Any]] = []

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
        """Return the next canned completion for ``messages`` (deterministic).

        Resolution order: keyed lookup, then sequence, then ``responder``, then
        ``default``. If nothing matches, an empty completion is returned.

        Args mirror the :class:`~usabench.llm.client.LLMClient` protocol; all
        decoding params are recorded but only affect output via a custom
        ``responder``.

        Returns:
            A zero-cost normalized :class:`Completion`.
        """
        with self._lock:
            self.calls.append(
                {
                    "messages": [m.model_dump(exclude_none=True) for m in messages],
                    "tools": [t.name for t in tools] if tools else None,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "seed": seed,
                    "stop": stop,
                    "kwargs": kwargs,
                }
            )
            entry = self._resolve(messages, temperature=temperature, max_tokens=max_tokens,
                                   seed=seed, stop=stop, **kwargs)

        completion = make_completion(entry, model=self.model, provider=self.provider)
        completion.usage = Usage(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            cost_usd=0.0,
        )
        if self._usage_meter is not None:
            self._usage_meter.record(completion.usage, channel=self._channel)
        return completion

    # -- resolution --------------------------------------------------------- #

    def _resolve(self, messages: list[Message], **params: Any) -> ScriptEntry:
        """Pick the response entry for the current call (lock already held)."""
        # 1) keyed lookup
        if self._keyed:
            key = self._key_fn(messages)
            if key in self._keyed:
                return self._keyed[key]

        # 2) sequence
        if self._script:
            n = len(self._script)
            if self._round_robin:
                entry = self._script[self._cursor % n]
                self._cursor += 1
                return entry
            if self._cursor < n:
                entry = self._script[self._cursor]
                self._cursor += 1
                return entry

        # 3) responder
        if self._responder is not None:
            return self._responder(messages, **params)

        # 4) default / empty
        if self._default is not None:
            return self._default
        return Completion(text="")

    # -- convenience -------------------------------------------------------- #

    def reset(self) -> None:
        """Rewind the sequence cursor and clear recorded calls."""
        with self._lock:
            self._cursor = 0
            self.calls.clear()

    @property
    def call_count(self) -> int:
        """Number of :meth:`chat` calls made so far."""
        return len(self.calls)

    @classmethod
    def echo(cls, prefix: str = "", **kwargs: Any) -> FakeLLMClient:
        """Build a fake that echoes the last user message (optionally prefixed).

        Handy for trivial smoke agents.

        Args:
            prefix: String prepended to the echoed text.
            **kwargs: Forwarded to :class:`FakeLLMClient`.

        Returns:
            A configured :class:`FakeLLMClient`.
        """

        def _responder(messages: list[Message], **_params: Any) -> str:
            text = messages[-1].content if messages else ""
            return f"{prefix}{text}"

        return cls(responder=_responder, **kwargs)
