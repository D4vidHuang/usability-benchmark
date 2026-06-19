"""Shared, dependency-free helpers for the evaluation subtree.

Every metric / scoring function in :mod:`usabench.eval` is a *pure* offline
function of ``trace.jsonl`` (a list of :class:`~usabench.core.schema.TraceEnvelope`)
plus the frozen task gold (a :class:`~usabench.core.schema.Task` or its
:class:`~usabench.core.schema.HiddenSpec`). This module centralises the small
amount of plumbing those functions share so no logic is duplicated:

* spec-constant access (always via :func:`spec` -> ``usability_score.yaml``);
* event-stream filtering by ``type``/``actor``;
* robust extraction of enum-or-value fields (the on-disk envelope stores
  ``actor``/``type`` as raw strings because the models use ``use_enum_values``,
  while ``payload.severity`` stays a :class:`~usabench.core.enums.Severity`);
* numeric guards (``clip01``, ``safe_div``).

Nothing here imports heavy/optional deps, so the whole eval package imports with
core deps only.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from usabench.core.enums import InteractionType, Severity
from usabench.core.schema import TraceEnvelope
from usabench.eval.spec import load_spec

__all__ = [
    "spec",
    "spec_get",
    "clip01",
    "safe_div",
    "as_value",
    "event_type",
    "event_actor",
    "payload_severity",
    "iter_events",
    "events_of_type",
    "oracle_responses",
    "oracle_queries",
    "checkpoints",
    "verification_runs",
    "code_runs",
    "file_edits",
    "agent_messages",
    "messages_to_user",
    "final_acceptance",
    "episode_end",
]


def spec() -> dict[str, Any]:
    """Return the cached usability-score spec dict (single source of truth)."""
    return load_spec()


def spec_get(*path: str, default: Any = None) -> Any:
    """Read a nested key from the spec, e.g. ``spec_get("composite", "alpha")``.

    Args:
        *path: A sequence of keys to descend.
        default: Returned if any key on the path is missing.

    Returns:
        The value at ``path`` in :func:`spec`, or ``default``.
    """
    cur: Any = load_spec()
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def clip01(x: float) -> float:
    """Clamp ``x`` into the closed unit interval ``[0, 1]``."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def safe_div(num: float, den: float, *, default: float = 0.0) -> float:
    """Divide ``num`` by ``den``, returning ``default`` when ``den`` is ~0."""
    if den == 0:
        return default
    return num / den


def as_value(x: Any) -> Any:
    """Return ``x.value`` if ``x`` is an enum member, else ``x`` unchanged."""
    return x.value if hasattr(x, "value") else x


def event_type(ev: TraceEnvelope) -> str:
    """Return the event-type string of an envelope (enum-or-str safe)."""
    return str(as_value(ev.type))


def event_actor(ev: TraceEnvelope) -> str:
    """Return the actor string of an envelope (enum-or-str safe)."""
    return str(as_value(ev.actor))


def payload_severity(payload: Any) -> int:
    """Return an oracle response's severity as a plain ``int`` in ``[0, 5]``.

    Handles both the :class:`~usabench.core.enums.Severity` enum and a raw int.
    """
    sev = getattr(payload, "severity", 0)
    if isinstance(sev, Severity):
        return int(sev)
    return int(sev)


def iter_events(trace: Iterable[TraceEnvelope]) -> list[TraceEnvelope]:
    """Materialise ``trace`` into a list (defensive for single-pass iterables)."""
    return list(trace)


def events_of_type(
    trace: Iterable[TraceEnvelope], etype: InteractionType | str
) -> list[TraceEnvelope]:
    """Return all envelopes whose ``type`` equals ``etype``."""
    target = str(as_value(etype))
    return [ev for ev in trace if event_type(ev) == target]


def oracle_responses(trace: Iterable[TraceEnvelope]) -> list[TraceEnvelope]:
    """All ``oracle_response`` envelopes (solicited + unsolicited)."""
    return events_of_type(trace, InteractionType.ORACLE_RESPONSE)


def oracle_queries(trace: Iterable[TraceEnvelope]) -> list[TraceEnvelope]:
    """All ``oracle_query`` envelopes (agent -> oracle)."""
    return events_of_type(trace, InteractionType.ORACLE_QUERY)


def checkpoints(trace: Iterable[TraceEnvelope]) -> list[TraceEnvelope]:
    """All ``checkpoint`` envelopes, in trace order."""
    return events_of_type(trace, InteractionType.CHECKPOINT)


def verification_runs(trace: Iterable[TraceEnvelope]) -> list[TraceEnvelope]:
    """All ``verification_run`` envelopes, in trace order."""
    return events_of_type(trace, InteractionType.VERIFICATION_RUN)


def code_runs(trace: Iterable[TraceEnvelope]) -> list[TraceEnvelope]:
    """All ``code_run`` envelopes (test/command executions)."""
    return events_of_type(trace, InteractionType.CODE_RUN)


def file_edits(trace: Iterable[TraceEnvelope]) -> list[TraceEnvelope]:
    """All ``file_edit`` envelopes (harness-diffed mutations)."""
    return events_of_type(trace, InteractionType.FILE_EDIT)


def agent_messages(trace: Iterable[TraceEnvelope]) -> list[TraceEnvelope]:
    """All ``agent_message`` envelopes (think-aloud / plan)."""
    return events_of_type(trace, InteractionType.AGENT_MESSAGE)


def messages_to_user(trace: Iterable[TraceEnvelope]) -> list[TraceEnvelope]:
    """All ``message_to_user`` envelopes (agent -> user status)."""
    return events_of_type(trace, InteractionType.MESSAGE_TO_USER)


def final_acceptance(trace: Iterable[TraceEnvelope]) -> TraceEnvelope | None:
    """The last ``final_acceptance`` envelope, or ``None`` if absent."""
    evs = events_of_type(trace, InteractionType.FINAL_ACCEPTANCE)
    return evs[-1] if evs else None


def episode_end(trace: Iterable[TraceEnvelope]) -> TraceEnvelope | None:
    """The ``episode_end`` envelope, or ``None`` if the trace is truncated."""
    evs = events_of_type(trace, InteractionType.EPISODE_END)
    return evs[-1] if evs else None
