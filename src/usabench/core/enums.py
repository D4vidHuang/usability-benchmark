"""Canonical enumerations shared across the whole benchmark.

These string/int enums are the closed vocabularies every other package imports.
They are intentionally dependency-free (only stdlib + ``__future__``) so importing
``usabench.core.enums`` never pulls in heavy optional deps.

The vocabularies are aligned with ``docs/protocol.md`` (interaction loop + trace
events), ``docs/metrics.md`` (0-5 assistance-severity scale, query classes), and
``DESIGN.md`` (the canonical synthesis that wins where docs disagree).
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

__all__ = [
    "InteractionType",
    "Actor",
    "Severity",
    "QueryClass",
    "RunStatus",
    "Provider",
    "TerminatedReason",
    "Verdict",
    "CheckKind",
    "VerifierChannel",
    "Difficulty",
    "RevealRule",
    "DeliverableType",
    "NetworkPolicy",
]


class InteractionType(StrEnum):
    """The closed set of trace-event ``type`` values.

    One enum value == one ``type`` string in ``trace.jsonl``. Aligned with the
    interaction loop and the event payloads documented in ``docs/protocol.md``
    and the Episode/Event view in ``docs/metrics.md``. The scorer rejects any
    event whose ``type`` is not in this set.
    """

    # Run lifecycle (harness-emitted bookends).
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"

    # Agent-side actions.
    AGENT_MESSAGE = "agent_message"
    TOOL_CALL = "tool_call"
    FILE_EDIT = "file_edit"
    CODE_RUN = "code_run"
    MESSAGE_TO_USER = "message_to_user"
    AGENT_BLOCKED = "agent_blocked"
    HANDOFF_REQUEST = "handoff_request"

    # Oracle channel.
    ORACLE_QUERY = "oracle_query"
    ORACLE_RESPONSE = "oracle_response"

    # Harness / verifier / accounting.
    CHECKPOINT = "checkpoint"
    VERIFICATION_RUN = "verification_run"
    BUDGET_DEBIT = "budget_debit"
    FINAL_ACCEPTANCE = "final_acceptance"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @classmethod
    def values(cls) -> list[str]:
        """Return every event-type string value (stable order)."""
        return [m.value for m in cls]

    @classmethod
    def is_oracle_event(cls, value: str) -> bool:
        """True if ``value`` is an oracle-channel event type."""
        return value in {cls.ORACLE_QUERY.value, cls.ORACLE_RESPONSE.value}


class Actor(StrEnum):
    """Who produced a trace event.

    The trace envelope's ``actor`` field. ``agent`` is the system-under-test;
    ``oracle`` is the simulated user; ``harness`` is the orchestrator; ``env`` is
    the sandbox/verifier execution environment.
    """

    AGENT = "agent"
    ORACLE = "oracle"
    HARNESS = "harness"
    ENV = "env"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class Severity(IntEnum):
    """The canonical 0-5 assistance-severity scale (``docs/metrics.md`` §3.1).

    The *same* scale the oracle's hint ladder uses (L0-L5), so the oracle's
    self-declared ``level`` is mechanically checkable against the logged severity.
    Convex weights ``w = [0, 1, 3, 6, 12, 25]`` are NOT defined here -- they live
    only in ``usabench/eval/spec/usability_score.yaml`` (single source of truth).
    """

    NONE = 0
    TRIVIAL_CLARIFICATION = 1
    SUBSTANTIVE_SPEC_INFO = 2
    DIRECTIONAL_HINT = 3
    PARTIAL_SOLUTION = 4
    TAKEOVER = 5

    @property
    def label(self) -> str:
        """Human-readable label for this severity level."""
        return _SEVERITY_LABELS[int(self)]

    @classmethod
    def from_level(cls, level: int) -> Severity:
        """Map an oracle hint-ladder level (0-5) onto a :class:`Severity`.

        Args:
            level: Oracle-declared hint level in ``[0, 5]``.

        Returns:
            The matching :class:`Severity` member.

        Raises:
            ValueError: If ``level`` is outside ``[0, 5]``.
        """
        if not 0 <= level <= 5:
            raise ValueError(f"severity level out of range [0,5]: {level!r}")
        return cls(level)

    @classmethod
    def is_assistance(cls, sev: int) -> bool:
        """True if severity ``sev`` counts as *assistance* (L1+), not elicitation.

        L0 events are *counted* (an interaction happened) but are spec-elicitation,
        not assistance burden -- see ``docs/protocol.md`` §2.5.
        """
        return int(sev) >= 1


_SEVERITY_LABELS: dict[int, str] = {
    0: "none",
    1: "trivial_clarification",
    2: "substantive_spec_info",
    3: "directional_hint",
    4: "partial_solution",
    5: "takeover",
}


class QueryClass(StrEnum):
    """Classification of an agent's outbound oracle query.

    Aligned with ``docs/protocol.md`` (qtype: clarify|hint_request|handoff|confirm)
    and ``docs/metrics.md`` query_class. ``out_of_scope`` captures off-spec asks the
    oracle answers with no information but still logs as an interaction.
    """

    CLARIFICATION = "clarification"
    HINT_REQUEST = "hint_request"
    CONFIRMATION = "confirmation"
    HANDOFF_REQUEST = "handoff_request"
    OUT_OF_SCOPE = "out_of_scope"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class RunStatus(StrEnum):
    """Lifecycle status of a single run/episode."""

    PENDING = "pending"
    RUNNING = "running"
    ACCEPTED = "accepted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    AGENT_GAVE_UP = "agent_gave_up"
    ORACLE_TAKEOVER = "oracle_takeover"
    ERROR = "error"
    INVALID = "invalid"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @property
    def is_terminal(self) -> bool:
        """True if this status represents a finished run."""
        return self not in {RunStatus.PENDING, RunStatus.RUNNING}

    @property
    def is_scorable(self) -> bool:
        """True if a run in this status should be included in scoring."""
        return self not in {RunStatus.PENDING, RunStatus.RUNNING, RunStatus.ERROR, RunStatus.INVALID}


class Provider(StrEnum):
    """LLM backend providers.

    ``vllm`` is reached through the OpenAI-compatible code path (only ``base_url``
    differs); ``fake`` is the deterministic zero-cost smoke client.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    VLLM = "vllm"
    FAKE = "fake"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @property
    def is_openai_shaped(self) -> bool:
        """True if this provider uses the OpenAI-compatible client path."""
        return self in {Provider.OPENAI, Provider.VLLM}


class TerminatedReason(StrEnum):
    """Why a run ended (trace ``termination``/``episode_end`` payload).

    Mirrors ``docs/protocol.md`` §1.4 termination conditions plus the per-budget
    breakdown so the scorer can attribute which ceiling was hit.
    """

    ACCEPT = "accept"
    BUDGET_TURNS = "budget_turns"
    BUDGET_WALL = "budget_wall"
    BUDGET_TOKENS = "budget_tokens"
    BUDGET_COST = "budget_cost"
    BUDGET_ORACLE = "budget_oracle"
    GIVE_UP = "give_up"
    ORACLE_TAKEOVER = "oracle_takeover"
    ERROR = "error"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @property
    def is_budget(self) -> bool:
        """True if this reason is a budget-ceiling termination."""
        return self.value.startswith("budget_")


class Verdict(StrEnum):
    """Oracle review / acceptance verdict."""

    ACCEPT = "accept"
    REJECT = "reject"
    NA = "na"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class CheckKind(StrEnum):
    """How a single acceptance criterion is verified.

    Each criterion routes to exactly one channel (``docs/scoring.md`` §1 invariant):
    ``func`` -> deterministic functional checker (V1), ``rubric_auto`` -> scripted
    rubric checker (V2), ``oracle_judgment`` -> LLM-judge (V3).
    """

    FUNC = "func"
    RUBRIC_AUTO = "rubric_auto"
    ORACLE_JUDGMENT = "oracle_judgment"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class VerifierChannel(StrEnum):
    """The three verification channels that combine into Goal Achievement (GA)."""

    V1 = "v1"  # functional / sandbox execution (deterministic)
    V2 = "v2"  # frozen rubric / acceptance-criteria checklist
    V3 = "v3"  # LLM-as-judge jury

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class Difficulty(StrEnum):
    """Task difficulty tier (``docs/tasks.md`` §1.2)."""

    T1 = "T1"  # trivial-shaped
    T2 = "T2"  # standard
    T3 = "T3"  # composite
    T4 = "T4"  # open-scope

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class RevealRule(StrEnum):
    """When the oracle may release a hidden ambiguity point / preference."""

    ON_ASK = "on_ask"
    ON_HINT = "on_hint"
    NEVER_VOLUNTEER = "never_volunteer"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class DeliverableType(StrEnum):
    """The kind of artifact a task asks the agent to produce (``docs/tasks.md``)."""

    CLI_TOOL = "cli-tool"
    SCRIPT = "script"
    WEB_APP = "web-app"
    LIBRARY = "library"
    NOTEBOOK = "notebook"
    SERVICE = "service"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class NetworkPolicy(StrEnum):
    """Sandbox network policy. Scored runs default to ``deny`` (hermetic)."""

    DENY = "deny"
    ALLOWLIST = "allowlist"
    MOCKED = "mocked"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value
