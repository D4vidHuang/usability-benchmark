"""Pydantic v2 data models -- the shared contracts every package imports.

This module defines three families of models:

1. **Task definition** (:class:`AcceptanceCriterion`, :class:`HiddenSpec`,
   :class:`TaskEnv`, :class:`Task`) -- the frozen benchmark content. The
   ``Task.agent_view`` helper returns the agent-visible projection with every
   gold/hidden field stripped (the structural two-tier visibility guarantee from
   ``docs/tasks.md`` §2.1).

2. **Trace events** -- a discriminated union (:data:`TraceEvent`) of typed
   payloads sharing a common envelope (:class:`TraceEnvelope`). One union member
   per :class:`~usabench.core.enums.InteractionType`. ``trace.jsonl`` is the ONE
   canonical artifact (``DESIGN.md`` invariant 1); ``parse_event`` reconstructs a
   typed event from a raw dict by its ``type`` discriminator.

3. **Run aggregates** (:class:`RunManifest`, :class:`RunResult`,
   :class:`AcceptanceResult`, :class:`OracleResponse`) -- summaries that are
   *views/caches* derived from the trace, never independent sources of truth.

All models use ``from __future__ import annotations`` and are mypy-friendly.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from usabench.core.enums import (
    Actor,
    CheckKind,
    DeliverableType,
    Difficulty,
    InteractionType,
    NetworkPolicy,
    QueryClass,
    RevealRule,
    RunStatus,
    Severity,
    TerminatedReason,
    Verdict,
)
from usabench.core.ids import GENESIS_HASH, canonical_json, next_hash

__all__ = [
    "SCHEMA_VERSION",
    "StrictModel",
    # Task definition
    "AcceptanceCriterion",
    "AmbiguityPoint",
    "InfoUnit",
    "ReferenceRepo",
    "HiddenSpec",
    "TaskEnv",
    "Task",
    "AgentTaskView",
    # Trace envelope + payloads
    "TraceEnvelope",
    "EpisodeStart",
    "EpisodeEnd",
    "AgentMessage",
    "ToolCall",
    "ToolResult",
    "FileEdit",
    "CodeRun",
    "MessageToUser",
    "AgentBlocked",
    "HandoffRequest",
    "OracleQuery",
    "OracleResponse",
    "Checkpoint",
    "VerificationRun",
    "CriterionResult",
    "BudgetDebit",
    "BudgetSnapshot",
    "FinalAcceptance",
    "TraceEvent",
    "TraceEventModel",
    "parse_event",
    "chain_event",
    # Run aggregates
    "Usage",
    "AcceptanceResult",
    "RunManifest",
    "RunResult",
]

#: Bumped on any breaking change to the trace/task schemas.
SCHEMA_VERSION = "1.0.0"


class StrictModel(BaseModel):
    """Base model: forbid unknown fields, validate on assignment, use enum values."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        frozen=False,
    )


# --------------------------------------------------------------------------- #
# Task definition models                                                       #
# --------------------------------------------------------------------------- #


class AcceptanceCriterion(StrictModel):
    """A single weighted, independently-checkable acceptance criterion.

    Each criterion routes to exactly one verification channel via ``check_kind``
    (``docs/scoring.md`` §1 invariant). ``is_core`` marks functional must-haves
    that feed the anti-stuffing ``A3_core_criteria_score``; ``is_hard`` marks
    gating constraints that drive the GA hard-constraint cap (``docs/scoring.md``
    §4.4).
    """

    id: str = Field(..., description="Stable criterion id, e.g. 'AC2'.")
    text: str = Field(..., description="Human-readable description of the criterion.")
    weight: float = Field(1.0, ge=0.0, description="Relative weight within its kind.")
    is_core: bool = Field(False, description="True if a functional must-have (feeds A3).")
    is_hard: bool = Field(False, description="True if a gating hard constraint (feeds the GA gate).")
    check_kind: CheckKind = Field(..., description="Which channel verifies this criterion.")
    check_ref: str | None = Field(
        None, description="Pointer to the checker script / test (for func/rubric_auto)."
    )
    source: str | None = Field(
        None, description="Provenance span justifying this criterion (path:lines or issue:#N)."
    )


class AmbiguityPoint(StrictModel):
    """An under-specification the agent *should* surface by asking the oracle.

    Resolving the high-severity points cheaply (by asking) rather than guessing
    wrong is the core usability signal (``docs/tasks.md`` §3.4).
    """

    id: str = Field(..., description="Stable id, e.g. 'AP1'.")
    question: str = Field(..., description="The decision the agent must make / ask about.")
    gold: str = Field(..., description="The correct resolution (oracle-private).")
    reveal: RevealRule = Field(RevealRule.ON_ASK, description="When the oracle may reveal it.")
    severity: Literal["low", "medium", "high"] = Field(
        "medium", description="How load-bearing this ambiguity is."
    )


class InfoUnit(StrictModel):
    """A discrete unit of hidden-spec information (requirement/constraint/preference).

    Used to compute ``C4_spec_info_transferred`` (spec the agent failed to elicit
    or infer) and ``D5_proactive_inference`` (spec correctly inferred unaided).
    """

    id: str = Field(..., description="Stable id, e.g. 'req:recurring'.")
    klass: Literal["requirement", "constraint", "preference"] = Field(
        ..., alias="class", description="Information class."
    )
    desc: str = Field(..., description="What the unit conveys.")

    model_config = ConfigDict(populate_by_name=True, extra="forbid", use_enum_values=True)


class ReferenceRepo(StrictModel):
    """A real OSS repo that grounds a task's gold intent (links + metadata only)."""

    url: str = Field(..., description="Repository URL.")
    commit: str = Field(..., description="Pinned commit SHA (no floating refs).")
    license: str = Field(..., description="SPDX license id, e.g. 'MIT'.")
    role: Literal["primary", "secondary", "inspiration"] = Field(
        "primary", description="How central this repo is to the gold."
    )
    why: str | None = Field(None, description="What this repo justifies in the spec.")


class HiddenSpec(StrictModel):
    """Oracle-private gold knowledge. NEVER shown to the agent.

    Loaded only by the grader and the oracle process. Mirrors the structured
    ``hidden_spec`` of ``docs/protocol.md`` §2.1 and ``docs/tasks.md`` §2.
    """

    summary: str = Field(..., description="1-3 sentence statement of what the user actually wants.")
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        default_factory=list, description="Full weighted criteria checklist (gold)."
    )
    ambiguity_points: list[AmbiguityPoint] = Field(
        default_factory=list, description="Under-specifications to surface by asking."
    )
    info_units: list[InfoUnit] = Field(
        default_factory=list, description="Discrete hidden-spec info units for C4/D5."
    )
    reveal_rules: dict[str, str] = Field(
        default_factory=dict, description="Optional per-id reveal-rule overrides."
    )
    oracle_persona: str = Field(
        "non_expert_user", description="Persona archetype: non_expert_user | maintainer."
    )
    known_pitfalls: list[str] = Field(
        default_factory=list, description="Pitfalls dispensed only as graded hints, never volunteered."
    )
    out_of_scope: list[str] = Field(
        default_factory=list, description="Asks the oracle answers with 'not needed'."
    )
    constraints: list[str] = Field(
        default_factory=list, description="Constraints the oracle states only if asked."
    )

    @property
    def n_hidden_spec_units(self) -> int:
        """Number of hidden-spec info units (denominator for C4/D5 normalization)."""
        return len(self.info_units)


class TaskEnv(StrictModel):
    """The frozen, reproducible execution environment for a task.

    Scored runs are hermetic: ``network`` defaults to ``deny`` with a per-task
    ``allowlist`` (``DESIGN.md`` invariant 5).
    """

    dockerfile: str | None = Field(None, description="Path to a Dockerfile, or null for base_image.")
    base_image: str = Field("python:3.11-slim", description="Pinned base image.")
    network: NetworkPolicy = Field(NetworkPolicy.DENY, description="Network policy (hermetic default).")
    allowlist: list[str] = Field(
        default_factory=list, description="Hosts permitted when network==allowlist."
    )
    fixtures: list[str] = Field(default_factory=list, description="Starter files given to the agent.")
    allowed_reqs: list[str] = Field(
        default_factory=list, description="Allowlisted dependency specifiers (requirements.allowed.txt)."
    )
    setup: list[str] = Field(default_factory=list, description="Setup commands (e.g. pip install).")
    entrypoint_hint: str | None = Field(
        None, description="Optional entrypoint hint; null = agent decides (more ambiguity)."
    )


class Task(StrictModel):
    """A complete benchmark task: agent-visible fields + oracle-private ``hidden``.

    The agent-visible projection is produced by :meth:`agent_view`, which strips
    every gold field so nothing the agent sees can leak the answer.
    """

    id: str = Field(..., description="Stable slug, e.g. 'ub-cal-0007'.")
    schema_version: str = Field(SCHEMA_VERSION, description="Task schema version.")
    title: str = Field(..., description="Short human title.")
    user_goal: str = Field(..., description="The under-specified, lay-phrased goal shown to the agent.")
    domain: str = Field(..., description="Domain enum value (cli-util, data-analysis, ...).")
    difficulty: Difficulty = Field(..., description="Difficulty tier T1..T4.")
    deliverable_type: DeliverableType = Field(..., description="Kind of artifact to produce.")
    required_capabilities: list[str] = Field(
        default_factory=list, description="Controlled-vocab capabilities the task exercises."
    )
    reference_repos: list[ReferenceRepo] = Field(
        default_factory=list, description="Real OSS repos grounding the gold (oracle-private detail)."
    )
    env: TaskEnv = Field(default_factory=TaskEnv, description="Frozen execution environment.")
    hidden: HiddenSpec = Field(..., description="Oracle-private gold knowledge. Never shown to agent.")
    expected_interventions: dict[str, Any] | None = Field(
        None, description="Calibration targets (median/p10/p90 + by-type) from pilot runs."
    )
    contamination_label: str | None = Field(
        None, description="Contamination-risk label: low|medium|high."
    )
    accept_threshold: float = Field(
        0.80, ge=0.0, le=1.0, description="Weighted-score threshold for acceptance."
    )
    user_goal_persona_note: str | None = Field(
        None, description="Note on how the user phrases things (e.g. 'non-expert')."
    )
    harvest_provenance_id: str | None = Field(
        None, description="Link back to the raw harvest record."
    )

    def agent_view(self) -> AgentTaskView:
        """Return the agent-visible projection with all gold/hidden fields removed.

        This is the structural two-tier visibility guarantee: the returned object
        contains *no* field that could leak the gold answer (``docs/tasks.md``
        §2.1). Reference-repo *detail* is omitted entirely; only the public goal,
        environment, and capability hints remain.

        Returns:
            An :class:`AgentTaskView` safe to hand to the agent-under-test.
        """
        return AgentTaskView(
            id=self.id,
            title=self.title,
            user_goal=self.user_goal,
            domain=self.domain,
            difficulty=self.difficulty,
            deliverable_type=self.deliverable_type,
            required_capabilities=list(self.required_capabilities),
            env=self.env,
            user_goal_persona_note=self.user_goal_persona_note,
        )


class AgentTaskView(StrictModel):
    """The agent-visible projection of a :class:`Task`. Contains NO gold fields."""

    id: str
    title: str
    user_goal: str
    domain: str
    difficulty: Difficulty
    deliverable_type: DeliverableType
    required_capabilities: list[str] = Field(default_factory=list)
    env: TaskEnv = Field(default_factory=TaskEnv)
    user_goal_persona_note: str | None = None


# --------------------------------------------------------------------------- #
# Trace event models (discriminated union over `type`)                         #
# --------------------------------------------------------------------------- #


class BudgetSnapshot(StrictModel):
    """Remaining/used budget snapshot carried on every event (``budgets_after``).

    Lets the scorer reconstruct remaining budget at any point without recomputing
    from scratch. Monotonic non-decreasing per kind (``docs/protocol.md`` §4.3).
    """

    turns: int = Field(0, ge=0)
    wall_s: float = Field(0.0, ge=0.0)
    tokens: int = Field(0, ge=0)
    cost_usd: float = Field(0.0, ge=0.0)
    oracle_queries: int = Field(0, ge=0)


class Usage(StrictModel):
    """Normalized token + cost accounting for a single LLM call or rollup."""

    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)
    cost_usd: float = Field(0.0, ge=0.0)

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens


# --- Typed payloads (one per InteractionType) ------------------------------- #


class EpisodeStart(StrictModel):
    """``episode_start`` (seq 0) payload: everything that affects outcomes."""

    type: Literal[InteractionType.EPISODE_START] = InteractionType.EPISODE_START
    task_id: str
    task_version: str = SCHEMA_VERSION
    hidden_spec_sha256: str
    agent: dict[str, Any] = Field(default_factory=dict)
    oracle: dict[str, Any] = Field(default_factory=dict)
    budgets: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0
    sandbox: dict[str, Any] = Field(default_factory=dict)
    harness_version: str | None = None
    git_commit: str | None = None


class EpisodeEnd(StrictModel):
    """``episode_end`` (last line): final snapshot + scorer-derivable cache."""

    type: Literal[InteractionType.EPISODE_END] = InteractionType.EPISODE_END
    terminated_reason: TerminatedReason
    accepted: bool = False
    final_weighted_score: float | None = None
    interventions_by_severity: dict[str, int] = Field(default_factory=dict)
    invalid: bool = False
    invalid_reason: str | None = None

    @field_validator("terminated_reason")
    @classmethod
    def _coerce_terminated_reason(cls, v: Any) -> TerminatedReason:
        """Restore a real :class:`TerminatedReason` after ``use_enum_values``.

        ``StrictModel`` sets ``use_enum_values=True``, which would otherwise store
        this field as a bare ``str``. Coercing back to the enum lets callers use
        ``.is_budget`` (and friends) directly, while ``model_dump(mode="json")``
        still emits the plain string -- so ``trace.jsonl`` serialization is
        unchanged. The same contract holds on :class:`RunResult`.
        """
        return TerminatedReason(v)


class AgentMessage(StrictModel):
    """``agent_message``: agent think-aloud / plan with no side effect."""

    type: Literal[InteractionType.AGENT_MESSAGE] = InteractionType.AGENT_MESSAGE
    text: str
    tokens: Usage | None = None
    raw_finish_reason: str | None = None


class ToolCall(StrictModel):
    """``tool_call``: an agent action on the sandbox channel."""

    type: Literal[InteractionType.TOOL_CALL] = InteractionType.TOOL_CALL
    call_id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    args_sha256: str | None = None
    batch_id: str | None = None


class ToolResult(StrictModel):
    """``tool_result``-style payload from the sandbox (carried as code_run/env)."""

    type: Literal[InteractionType.CODE_RUN] = InteractionType.CODE_RUN
    call_id: str
    exit_code: int = 0
    stdout_sha256: str | None = None
    stdout_trunc: str | None = None
    stderr_trunc: str | None = None
    wall_ms: int = 0
    truncated: bool = False


class CodeRun(StrictModel):
    """``code_run``: explicit code/test execution result."""

    type: Literal[InteractionType.CODE_RUN] = InteractionType.CODE_RUN
    call_id: str | None = None
    cmd: str = ""
    exit_code: int = 0
    stdout_sha256: str | None = None
    stdout_trunc: str | None = None
    stderr_trunc: str | None = None
    wall_ms: int = 0
    is_test: bool = False
    self_test_passed: bool | None = None


class FileEdit(StrictModel):
    """``file_edit``: a filesystem mutation, derived by the harness via diffing.

    Never trusted from the agent -- the harness diffs sandbox snapshots.
    """

    type: Literal[InteractionType.FILE_EDIT] = InteractionType.FILE_EDIT
    path: str
    op: Literal["create", "modify", "delete", "rename"]
    pre_sha256: str | None = None
    post_sha256: str | None = None
    added: int = 0
    removed: int = 0
    unified_diff_sha256: str | None = None
    loc_after: int = 0


class MessageToUser(StrictModel):
    """``message_to_user``: agent communicates status/plan to the user (UX dim G)."""

    type: Literal[InteractionType.MESSAGE_TO_USER] = InteractionType.MESSAGE_TO_USER
    text: str


class AgentBlocked(StrictModel):
    """``agent_blocked``: agent declares itself stuck (feeds B10/D4 + stuck-probe)."""

    type: Literal[InteractionType.AGENT_BLOCKED] = InteractionType.AGENT_BLOCKED
    reason: str | None = None
    blocked: bool = True


class HandoffRequest(StrictModel):
    """``handoff_request``: agent asks the human to take over (severity-5 territory)."""

    type: Literal[InteractionType.HANDOFF_REQUEST] = InteractionType.HANDOFF_REQUEST
    reason: str | None = None


class OracleQuery(StrictModel):
    """``oracle_query``: the agent solicits the oracle. THIS is what we count."""

    type: Literal[InteractionType.ORACLE_QUERY] = InteractionType.ORACLE_QUERY
    query_class: QueryClass
    text: str
    context_refs: list[str] = Field(default_factory=list)
    agent_blocked: bool = False


class OracleResponse(StrictModel):
    """``oracle_response``: oracle answer, labeled with a 0-5 severity.

    The same LLM call that answers also self-declares its hint ``level``/severity
    against the rubric, which the offline classifier checks (``docs/protocol.md``
    §2.6, ``docs/metrics.md`` §3.1). ``responds_to=None`` marks an UNSOLICITED
    intervention (a proactive correction).
    """

    type: Literal[InteractionType.ORACLE_RESPONSE] = InteractionType.ORACLE_RESPONSE
    responds_to: str | None = Field(
        None, description="event_id of the originating oracle_query; null = unsolicited."
    )
    severity: Severity = Field(..., description="0-5 assistance-severity self-declared by the oracle.")
    severity_rationale: str | None = None
    text: str = ""
    reveals: list[str] = Field(default_factory=list, description="hidden_spec ids disclosed.")
    info_units_revealed: list[str] = Field(default_factory=list)
    verdict: Verdict = Verdict.NA
    cited_criteria: list[str] = Field(default_factory=list)
    refusals: list[str] = Field(default_factory=list)
    provenance_tag: str | None = Field(
        None, description="Tag marking oracle-contributed solution spans (for C5 leakage)."
    )
    oracle_tokens: Usage | None = None
    latency_ms: int = 0

    @field_validator("severity")
    @classmethod
    def _coerce_severity(cls, v: Any) -> Severity:
        """Allow ints in [0,5] to be coerced to :class:`Severity`."""
        return Severity(int(v))


class CriterionResult(StrictModel):
    """Per-criterion verification outcome inside a :class:`VerificationRun`."""

    id: str
    passed: bool | None = None
    score: float | None = Field(None, ge=0.0, le=1.0)
    detail_sha256: str | None = None
    channel: str | None = None


class VerificationRun(StrictModel):
    """``verification_run``: deterministic acceptance evaluation of the artifact."""

    type: Literal[InteractionType.VERIFICATION_RUN] = InteractionType.VERIFICATION_RUN
    trigger: Literal["submit", "forced_final"] = "submit"
    entrypoint: str | None = None
    must_have: list[CriterionResult] = Field(default_factory=list)
    should_have: list[CriterionResult] = Field(default_factory=list)
    all_must_pass: bool = False
    rubric_score: float = Field(0.0, ge=0.0, le=1.0)
    wall_ms: int = 0
    runner_image_digest: str | None = None


class Checkpoint(StrictModel):
    """``checkpoint``: an acceptance snapshot over time (feeds B6/B7/D2/D3)."""

    type: Literal[InteractionType.CHECKPOINT] = InteractionType.CHECKPOINT
    weighted_score: float = Field(0.0, ge=0.0, le=1.0)
    is_working_version: bool = False
    criteria_state: dict[str, Any] = Field(default_factory=dict)
    criteria_passed: int = 0
    criteria_total: int = 0


class BudgetDebit(StrictModel):
    """``budget_debit``: a single budget debit (one per action or coalesced per turn)."""

    type: Literal[InteractionType.BUDGET_DEBIT] = InteractionType.BUDGET_DEBIT
    kind: Literal["token", "cost", "turn", "wall", "oracle_query"]
    amount: float
    reason: str | None = None


class FinalAcceptance(StrictModel):
    """``final_acceptance``: the gold acceptance result the oracle review concluded."""

    type: Literal[InteractionType.FINAL_ACCEPTANCE] = InteractionType.FINAL_ACCEPTANCE
    accepted: bool = False
    weighted_score: float = Field(0.0, ge=0.0, le=1.0)
    verdict: Verdict = Verdict.NA
    cited_criteria: list[str] = Field(default_factory=list)


#: The discriminated-union of all event payloads, keyed on ``type``.
TraceEvent = Annotated[
    EpisodeStart | EpisodeEnd | AgentMessage | ToolCall | FileEdit | CodeRun | MessageToUser | AgentBlocked | HandoffRequest | OracleQuery | OracleResponse | Checkpoint | VerificationRun | BudgetDebit | FinalAcceptance,
    Field(discriminator="type"),
]

#: Mapping from event-type string -> payload model, used by :func:`parse_event`.
_PAYLOAD_BY_TYPE: dict[str, type[StrictModel]] = {
    InteractionType.EPISODE_START.value: EpisodeStart,
    InteractionType.EPISODE_END.value: EpisodeEnd,
    InteractionType.AGENT_MESSAGE.value: AgentMessage,
    InteractionType.TOOL_CALL.value: ToolCall,
    InteractionType.FILE_EDIT.value: FileEdit,
    InteractionType.CODE_RUN.value: CodeRun,
    InteractionType.MESSAGE_TO_USER.value: MessageToUser,
    InteractionType.AGENT_BLOCKED.value: AgentBlocked,
    InteractionType.HANDOFF_REQUEST.value: HandoffRequest,
    InteractionType.ORACLE_QUERY.value: OracleQuery,
    InteractionType.ORACLE_RESPONSE.value: OracleResponse,
    InteractionType.CHECKPOINT.value: Checkpoint,
    InteractionType.VERIFICATION_RUN.value: VerificationRun,
    InteractionType.BUDGET_DEBIT.value: BudgetDebit,
    InteractionType.FINAL_ACCEPTANCE.value: FinalAcceptance,
}


class TraceEnvelope(StrictModel):
    """One line of ``trace.jsonl``: common envelope + a typed ``payload``.

    Append-only, totally ordered by ``seq``, hash-chained via ``prev_hash``/``hash``
    (``docs/protocol.md`` §4). The ``payload`` is one member of :data:`TraceEvent`,
    discriminated on its own ``type`` field which mirrors the envelope ``type``.
    """

    schema_version: str = Field(SCHEMA_VERSION)
    run_id: str
    event_id: str = Field(..., description="Unique id for this event (e.g. UUID).")
    seq: int = Field(..., ge=0, description="Monotonic total-order index.")
    ts: float = Field(..., description="Epoch seconds wall clock.")
    t_turn: int | None = Field(None, description="Agent-turn index; null for harness events.")
    actor: Actor
    type: InteractionType
    payload: TraceEvent
    budgets_after: BudgetSnapshot = Field(default_factory=BudgetSnapshot)
    prev_hash: str = Field(GENESIS_HASH, description="Hash of the previous event in the chain.")
    hash: str | None = Field(None, description="Chain hash of this event; set by chain_event.")

    @field_validator("type")
    @classmethod
    def _type_matches_payload(cls, v: InteractionType, info: Any) -> InteractionType:
        """Envelope ``type`` must agree with the payload's discriminator."""
        payload = info.data.get("payload")
        if payload is not None:
            payload_type = getattr(payload, "type", None)
            pt_val = getattr(payload_type, "value", payload_type)
            v_val = v.value if hasattr(v, "value") else v
            if pt_val is not None and pt_val != v_val:
                raise ValueError(f"envelope type {v_val!r} != payload type {pt_val!r}")
        return v

    def canonical_without_hash(self) -> str:
        """Canonical-JSON of this event excluding the ``hash`` field (for chaining)."""
        data = self.model_dump(mode="json")
        data.pop("hash", None)
        return canonical_json(data)

    def compute_hash(self) -> str:
        """Compute (without mutating) this event's chain hash from ``prev_hash``."""
        return next_hash(self.prev_hash, self.canonical_without_hash())


# Backwards/forwards-friendly alias: the envelope IS the on-disk event model.
TraceEventModel = TraceEnvelope


def parse_event(raw: dict[str, Any]) -> TraceEnvelope:
    """Reconstruct a typed :class:`TraceEnvelope` from a raw trace-line dict.

    The ``payload`` dict is dispatched to the correct model by its ``type``
    discriminator, so callers get a fully-typed event without manual branching.

    Args:
        raw: A decoded ``trace.jsonl`` line (envelope with nested ``payload``).

    Returns:
        A validated :class:`TraceEnvelope`.

    Raises:
        SchemaViolation: If ``type`` is unknown or the payload fails validation.
    """
    from usabench.core.errors import SchemaViolation

    payload = raw.get("payload")
    etype = raw.get("type")
    if isinstance(payload, dict) and etype not in payload:
        # Ensure the payload carries its discriminator so the union resolves.
        payload = {**payload, "type": etype}
        raw = {**raw, "payload": payload}
    model = _PAYLOAD_BY_TYPE.get(str(etype))
    if model is None:
        raise SchemaViolation(f"unknown trace event type: {etype!r}", pointer="/type")
    try:
        return TraceEnvelope.model_validate(raw)
    except Exception as exc:  # pragma: no cover - re-raise as typed
        raise SchemaViolation(str(exc), pointer="/payload") from exc


def chain_event(prev_hash: str, envelope: TraceEnvelope) -> TraceEnvelope:
    """Return a copy of ``envelope`` with ``prev_hash`` set and ``hash`` computed.

    Args:
        prev_hash: The previous event's ``hash`` (or ``GENESIS_HASH`` for seq 0).
        envelope: The event to chain.

    Returns:
        A new :class:`TraceEnvelope` with ``prev_hash`` and ``hash`` populated.
    """
    ev = envelope.model_copy(update={"prev_hash": prev_hash})
    ev.hash = ev.compute_hash()
    return ev


# --------------------------------------------------------------------------- #
# Run aggregates (views/caches derived from the trace)                         #
# --------------------------------------------------------------------------- #


class AcceptanceResult(StrictModel):
    """The gold acceptance check: weighted per-criterion outcome + accept flag.

    A *view* recomputed from the trace's verification events plus the frozen gold;
    the scorer asserts equality against the cached ``episode_end`` totals.
    """

    criteria: list[CriterionResult] = Field(default_factory=list)
    weighted_score: float = Field(0.0, ge=0.0, le=1.0)
    core_criteria_score: float = Field(0.0, ge=0.0, le=1.0)
    hard_pass_frac: float = Field(0.0, ge=0.0, le=1.0)
    accepted: bool = False


class RunManifest(StrictModel):
    """Reproducibility manifest written per run (``docs/infra.md`` §6.1).

    Captures everything outcome-affecting so two numbers compare iff their
    manifests (and the enclosing ``release.lock``) agree.
    """

    run_id: str
    task_id: str
    seed: int
    git_sha: str
    config_hash: str
    package_version: str
    requirements_lock_sha256: str | None = None
    agent: dict[str, Any] = Field(default_factory=dict)
    oracle: dict[str, Any] = Field(default_factory=dict)
    budgets: dict[str, Any] = Field(default_factory=dict)
    sandbox: dict[str, Any] = Field(default_factory=dict)
    hostname: str | None = None
    slurm_job_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None


class RunResult(StrictModel):
    """Top-level summary of one finished run (a view over ``trace.jsonl``)."""

    run_id: str
    task_id: str
    seed: int
    status: RunStatus
    terminated_reason: TerminatedReason | None = None
    accepted: bool = False

    @field_validator("terminated_reason")
    @classmethod
    def _coerce_terminated_reason(cls, v: Any) -> TerminatedReason | None:
        """Restore a real :class:`TerminatedReason` after ``use_enum_values``.

        Mirrors :meth:`EpisodeEnd._coerce_terminated_reason` so ``.is_budget`` works
        directly on a stored result; JSON serialization still round-trips as a str.
        """
        return None if v is None else TerminatedReason(v)
    acceptance: AcceptanceResult | None = None
    n_events: int = 0
    wall_clock_s: float = 0.0
    agent_usage: Usage = Field(default_factory=Usage)
    oracle_usage: Usage = Field(default_factory=Usage)
    interventions_by_severity: dict[str, int] = Field(default_factory=dict)
    trace_path: str | None = None
    manifest: RunManifest | None = None
    invalid: bool = False
    invalid_reason: str | None = None
