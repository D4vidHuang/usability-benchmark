"""The simulated-user oracle: an LLM playing the human user/maintainer.

:class:`SimulatedUserOracle` is the runtime heart of the "how much human help?"
measurement (``docs/protocol.md`` §2). It is **truthful, bounded, graded, and
reproducible**:

* It holds the task's :class:`~usabench.core.schema.HiddenSpec` (oracle-private)
  and answers only through a single typed channel.
* :meth:`SimulatedUserOracle.answer` renders the frozen persona+hidden-spec
  template (``prompts/system_user.j2``), calls an
  :class:`~usabench.llm.client.LLMClient` at a low, config-pulled temperature, and
  returns a structured :class:`~usabench.core.schema.OracleResponse` that
  **self-labels** its severity (0-5) and ``info_units_revealed``.
* :meth:`SimulatedUserOracle.review` renders ``prompts/grading_user.j2`` to decide
  accept/reject on intent (the deterministic verifier owns objective gating).
* Every self-label is validated / overridden by
  :class:`~usabench.oracle.classifier.InteractionClassifier` and bounded by
  :class:`~usabench.oracle.policy.OraclePolicy` so an over-helpful or
  protocol-violating oracle is *detectable*, not silent.

The oracle stays API-based and constant across the agent grid; its decoding
temperature and seed come from config so behavior is reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from usabench.core.enums import QueryClass, Severity, Verdict
from usabench.core.errors import OracleProtocolError
from usabench.core.ids import sha256_hex
from usabench.core.schema import (
    HiddenSpec,
    OracleQuery,
    OracleResponse,
    Usage,
    VerificationRun,
)
from usabench.oracle.classifier import Classification, InteractionClassifier
from usabench.oracle.policy import MAX_AUTO_LEVEL, Helpfulness, OraclePolicy, PolicyConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from usabench.llm.client import Completion, LLMClient

__all__ = [
    "OracleConfig",
    "OracleReview",
    "AnswerResult",
    "SimulatedUserOracle",
    "PROMPTS_DIR",
]

_log = structlog.get_logger("usabench.oracle.oracle")

#: Directory holding the frozen Jinja2 prompt templates.
PROMPTS_DIR: Path = Path(__file__).resolve().parent / "prompts"

_SYSTEM_TEMPLATE = "system_user.j2"
_GRADING_TEMPLATE = "grading_user.j2"


@dataclass
class OracleConfig:
    """Reproducible oracle decoding + behavior knobs (pulled from run config).

    Attributes:
        model: Pinned oracle model id (recorded into ``episode_start``).
        temperature: Decoding temperature; kept low (<=0.3) for reproducibility
            (``docs/scoring.md`` §8). Default 0.0 (greedy where supported).
        seed: Optional decoding seed, yoked to the run seed.
        max_tokens: Max completion tokens for an oracle turn.
        persona: ``non_expert_user`` | ``maintainer``.
        helpfulness: ``strict`` | ``standard`` | ``lenient`` (escalation thresholds).
        hint_budget: Graded hints before extra hints become rescues.
        proactive_stuck_help: Whether stuck-offers are enabled.
        stuck_offer_cooldown_turns: Min turns between stuck-offers.
        max_level: Hard ceiling on the hint level (never L5 in auto runs).
    """

    model: str = "oracle"
    temperature: float = 0.0
    seed: int | None = None
    max_tokens: int = 1024
    persona: str = "non_expert_user"
    helpfulness: str = Helpfulness.STANDARD
    hint_budget: int = 3
    proactive_stuck_help: bool = True
    stuck_offer_cooldown_turns: int = 20
    max_level: int = MAX_AUTO_LEVEL

    def policy_config(self) -> PolicyConfig:
        """Project this config onto a :class:`PolicyConfig` for the policy layer."""
        return PolicyConfig(
            helpfulness=self.helpfulness,
            hint_budget=self.hint_budget,
            proactive_stuck_help=self.proactive_stuck_help,
            stuck_offer_cooldown_turns=self.stuck_offer_cooldown_turns,
            max_level=self.max_level,
        )


@dataclass
class OracleReview:
    """The oracle's accept/reject decision on a submission (``docs/protocol.md`` §2.7).

    Attributes:
        verdict: ``accept`` | ``reject``.
        feedback: In-character behavioral feedback (non-leaking on reject).
        cited_criteria: hidden_spec ids the verdict is grounded in (required on
            reject -- a reject without a citable id is flagged).
        classification: The classifier's labeling of this review (severity etc.).
        usage: Oracle token/cost accounting for the review call.
        response: The underlying :class:`OracleResponse` event for the trace.
    """

    verdict: Verdict
    feedback: str
    cited_criteria: list[str] = field(default_factory=list)
    classification: Classification | None = None
    usage: Usage = field(default_factory=Usage)
    response: OracleResponse | None = None

    @property
    def accepted(self) -> bool:
        """True iff the oracle accepted the submission."""
        return _verdict_value(self.verdict) == Verdict.ACCEPT.value


@dataclass
class AnswerResult:
    """An oracle answer plus its classification (returned by :meth:`answer`).

    Attributes:
        response: The structured :class:`OracleResponse` (goes into the trace).
        classification: The :class:`Classification` (severity/leak/conflict flags).
    """

    response: OracleResponse
    classification: Classification


class SimulatedUserOracle:
    """An LLM-backed simulated user, bounded by policy and audited by the classifier.

    The oracle is *strictly reactive*: it answers a query, reviews a submission, or
    responds to an accepted stuck-offer. It never volunteers gold and never writes
    code beyond an authorized L4 snippet.

    Args:
        hidden: The frozen oracle-private gold for this task.
        client: The :class:`LLMClient` the oracle speaks through.
        config: Decoding + behavior knobs.
        judge: Optional independent judge client for the classifier's Stage B
            (leak detection / names_fix). If ``None``, the classifier falls back
            to conservative rule-based defaults.
    """

    def __init__(
        self,
        hidden: HiddenSpec,
        client: LLMClient,
        config: OracleConfig | None = None,
        *,
        judge: LLMClient | None = None,
    ) -> None:
        self.hidden = hidden
        self.client = client
        self.config = config or OracleConfig(persona=hidden.oracle_persona)
        self.policy = OraclePolicy(hidden=hidden, config=self.config.policy_config())
        self.classifier = InteractionClassifier(hidden=hidden, judge=judge)
        self._env = _make_jinja_env()
        self._system_template = self._env.get_template(_SYSTEM_TEMPLATE)
        self._grading_template = self._env.get_template(_GRADING_TEMPLATE)
        #: sha256 of the rendered system prompt (-> episode_start.oracle.prompt_sha256).
        self.system_prompt: str = self._render_system_prompt()
        self.prompt_sha256: str = sha256_hex(self.system_prompt)

    # --- public API --------------------------------------------------------- #

    def answer(
        self,
        query: OracleQuery,
        *,
        repeat_requests: int = 0,
        prior_failures: int = 0,
        from_accepted_offer: bool = False,
    ) -> AnswerResult:
        """Answer an agent oracle-query, returning a graded :class:`OracleResponse`.

        The policy picks the lowest ladder level that addresses the query; the LLM
        produces the in-character text and a self-declared level; the classifier
        validates/overrides the severity and flags leaks/conflicts. The final
        severity stamped on the response is the policy-effective severity (which
        also enforces the hint budget).

        Args:
            query: The agent's oracle query.
            repeat_requests: Repeat hint-requests after a failure (drives escalation).
            prior_failures: Demonstrated failures preceding this query.
            from_accepted_offer: True if this stems from an accepted stuck-offer.

        Returns:
            An :class:`AnswerResult` with the response and its classification.

        Raises:
            OracleProtocolError: If the model output is unparseable or violates the
                level contract.
        """
        qclass = _query_class(query.query_class)

        # 1. Policy: deterministic ceiling on how much help this query may unlock.
        target_level = self.policy.target_level(
            qclass,
            requested_hint=qclass in (QueryClass.HINT_REQUEST,),
            repeat_requests=repeat_requests,
            prior_failures=prior_failures,
            from_accepted_offer=from_accepted_offer,
        )
        scope = self.policy.classify_scope(query.text)

        # 2. Ask the model, constrained to <= target_level.
        completion = self._chat(
            self.system_prompt
            + "\n\nThe agent just asked you:\n"
            + query.text
            + f"\n\n(You may answer at hint level 0..{target_level}. "
            f"Out-of-scope match: {scope.out_of_scope_match or 'none'}.)"
        )
        parsed = _parse_oracle_json(completion.text)
        declared_level = _clamp_level(int(parsed.get("level", target_level)), self.config.max_level)

        # 3. Bound the declared level by the policy ceiling, then apply hint budget.
        bounded_level = min(declared_level, target_level)
        effective_severity = self.policy.register_dispensed(bounded_level)

        response = self._build_response(
            query=query,
            parsed=parsed,
            severity=effective_severity,
            completion=completion,
            scope_out_of_scope=scope.is_out_of_scope,
        )

        classification = self.classifier.classify_response(
            response,
            query=query,
            from_offer=from_accepted_offer,
            offer_accepted=True,
        )
        # Surface a declared-vs-bounded mismatch as a conflict for the audit trail.
        if declared_level != bounded_level:
            classification.classifier_conflict = True
            classification.rationale += (
                f" ; declared L{declared_level} bounded to L{bounded_level} by policy"
            )
        _log.debug(
            "oracle_answer",
            qclass=qclass.value,
            target_level=target_level,
            declared_level=declared_level,
            severity=int(effective_severity),
            leak=classification.leak_flag,
        )
        return AnswerResult(response=response, classification=classification)

    def review(self, verification: VerificationRun, *, entrypoint: str | None = None) -> OracleReview:
        """Review a submission against the verifier report and decide accept/reject.

        The oracle accepts iff all must-haves pass AND behavior matches intent; a
        reject must cite a hidden_spec id (a reject without one is flagged). The
        deterministic verifier owns objective gating -- the oracle only judges
        intent (``docs/protocol.md`` §2.7).

        Args:
            verification: The deterministic :class:`VerificationRun` report.
            entrypoint: The demonstrated run entrypoint, if any.

        Returns:
            An :class:`OracleReview` with verdict, feedback, citations, and the
            classifier's labeling.

        Raises:
            OracleProtocolError: If the model output is unparseable.
        """
        prompt = self._render_grading_prompt(verification, entrypoint=entrypoint)
        completion = self._chat(prompt)
        parsed = _parse_oracle_json(completion.text)

        verdict = _coerce_verdict(parsed.get("verdict", "reject"))
        feedback = str(parsed.get("text", "")).strip()
        cited = [str(c) for c in parsed.get("cited_criteria", []) if c]
        review_severity = _clamp_level(int(parsed.get("severity", 0)), 5)

        # A reject MUST cite a criterion id; an uncited reject is a protocol breach
        # -- we keep the run going but flag it via an empty-citation review.
        if verdict == Verdict.REJECT and not cited:
            _log.warning("reject_without_citation", feedback=feedback[:120])

        response = OracleResponse(
            responds_to=None,
            severity=Severity.from_level(0 if verdict == Verdict.ACCEPT else review_severity),
            severity_rationale="submission review",
            text=feedback,
            reveals=[],
            info_units_revealed=[],
            verdict=verdict,
            cited_criteria=cited,
            refusals=[],
            oracle_tokens=completion.usage,
            latency_ms=0,
        )
        classification = self.classifier.classify_review(response)
        return OracleReview(
            verdict=verdict,
            feedback=feedback,
            cited_criteria=cited,
            classification=classification,
            usage=completion.usage,
            response=response,
        )

    def offer_stuck_help(self, turn: int, reason: str) -> bool:
        """Decide whether to emit a stuck-offer at ``turn`` (rate-limited, §2.4).

        The harness asks this when the agent is objectively stuck. The decision is
        deterministic (policy-driven), not an LLM call -- the offer text itself is
        a fixed level-0 nudge produced by :meth:`stuck_offer_text`.

        Args:
            turn: The current agent-turn index.
            reason: The trace-computable stuck reason (for logging).

        Returns:
            True iff an offer should be made now.
        """
        if not self.policy.may_offer_stuck_help(turn):
            return False
        self.policy.record_offer(turn)
        _log.info("oracle_stuck_offer", turn=turn, reason=reason)
        return True

    @staticmethod
    def stuck_offer_text(reason: str) -> str:
        """The fixed level-0 nudge text offered when the agent is stuck."""
        return (
            "It looks like you might be stuck. I can give you a small hint if you'd "
            "like -- just ask."
        )

    # --- prompt rendering --------------------------------------------------- #

    def _render_system_prompt(self) -> str:
        """Render the frozen persona+hidden-spec system prompt."""
        h = self.hidden
        must_haves = [
            {"id": c.id, "text": c.text}
            for c in h.acceptance_criteria
            if c.is_core or c.is_hard
        ] or [{"id": c.id, "text": c.text} for c in h.acceptance_criteria]
        ambiguity = [
            {
                "id": ap.id,
                "question": ap.question,
                "gold": ap.gold,
                "reveal": _enum_value(ap.reveal),
                "severity": ap.severity,
            }
            for ap in h.ambiguity_points
        ]
        info_units = [{"id": u.id, "klass": u.klass, "desc": u.desc} for u in h.info_units]
        persona = self.config.persona or h.oracle_persona
        rendered = self._system_template.render(
            persona_type=persona,
            domain_knowledge="low" if persona == "non_expert_user" else "high",
            intent=h.summary,
            must_haves=must_haves,
            ambiguity_points=ambiguity,
            info_units=info_units,
            constraints=list(h.constraints),
            out_of_scope=list(h.out_of_scope),
            known_pitfalls=list(h.known_pitfalls),
            helpfulness=self.config.helpfulness,
            escalation_repeats=self.policy.config.escalation_repeats,
            max_level=self.config.max_level,
        )
        return str(rendered)

    def _render_grading_prompt(
        self, verification: VerificationRun, *, entrypoint: str | None
    ) -> str:
        """Render the oracle-as-judge submission-review prompt."""
        h = self.hidden
        must_haves = [
            {"id": c.id, "text": c.text} for c in h.acceptance_criteria if c.is_core or c.is_hard
        ] or [{"id": c.id, "text": c.text} for c in h.acceptance_criteria]
        soft = [{"id": ap.id, "text": f"{ap.question} -> {ap.gold}"} for ap in h.ambiguity_points]
        rendered = self._grading_template.render(
            persona_type=self.config.persona or h.oracle_persona,
            intent=h.summary,
            must_haves=must_haves,
            soft_prefs=soft,
            all_must_pass=verification.all_must_pass,
            entrypoint=entrypoint or verification.entrypoint,
            must_results=[{"id": r.id, "passed": bool(r.passed)} for r in verification.must_have],
            should_results=[
                {"id": r.id, "score": (r.score if r.score is not None else 0.0)}
                for r in verification.should_have
            ],
            rubric_score=verification.rubric_score,
        )
        return str(rendered)

    # --- model plumbing ----------------------------------------------------- #

    def _chat(self, prompt: str) -> Completion:
        """Call the LLM client with the oracle's pinned decoding params.

        Raises:
            OracleProtocolError: If the provider call fails.
        """
        from usabench.llm.client import Message

        try:
            return self.client.chat(
                [Message(role="user", content=prompt)],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                seed=self.config.seed,
            )
        except Exception as exc:  # pragma: no cover - provider failure path
            raise OracleProtocolError(f"oracle LLM call failed: {exc}") from exc

    def _build_response(
        self,
        *,
        query: OracleQuery,
        parsed: dict[str, Any],
        severity: Severity,
        completion: Completion,
        scope_out_of_scope: bool,
    ) -> OracleResponse:
        """Assemble a validated :class:`OracleResponse` from parsed model output."""
        reveals = [str(r) for r in parsed.get("reveals", []) if r]
        info_units = [str(u) for u in parsed.get("info_units_revealed", []) if u]
        refusals = [str(r) for r in parsed.get("refusals", []) if r]
        if scope_out_of_scope and not refusals:
            refusals = ["out_of_scope"]
        provenance = "oracle" if int(severity) >= int(Severity.PARTIAL_SOLUTION) else None
        return OracleResponse(
            responds_to=None,  # the harness backfills the originating query event_id
            severity=severity,
            severity_rationale=str(parsed.get("severity_rationale", "")) or None,
            text=str(parsed.get("text", "")).strip(),
            reveals=reveals,
            info_units_revealed=info_units,
            verdict=Verdict.NA,
            cited_criteria=[str(c) for c in parsed.get("cited_criteria", []) if c],
            refusals=refusals,
            provenance_tag=provenance,
            oracle_tokens=completion.usage,
            latency_ms=0,
        )


# --------------------------------------------------------------------------- #
# Module-level helpers.                                                         #
# --------------------------------------------------------------------------- #


def _make_jinja_env() -> Any:
    """Build a sandboxed Jinja2 environment rooted at the prompts directory.

    Jinja2 is a core runtime dependency, but it is imported here (function scope)
    to keep the module-import surface minimal and uniform with the rest of the
    package's lazy-import discipline.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )


def _parse_oracle_json(text: str) -> dict[str, Any]:
    """Parse the oracle's JSON output, tolerating code fences and surrounding prose.

    Args:
        text: The raw model output.

    Returns:
        The parsed object as a dict.

    Raises:
        OracleProtocolError: If no JSON object can be recovered (R: structured
            oracle output is mandatory, ``docs/protocol.md`` §2.6).
    """
    if not text:
        raise OracleProtocolError("empty oracle response (expected JSON)")
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError) as exc:
        raise OracleProtocolError(f"oracle response was not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise OracleProtocolError("oracle response JSON must be an object")
    return obj


def _clamp_level(level: int, max_level: int) -> int:
    """Clamp ``level`` into ``[0, max_level]``."""
    return max(0, min(int(level), int(max_level)))


def _query_class(qc: Any) -> QueryClass:
    """Normalize a query-class value (enum or string) to :class:`QueryClass`."""
    if isinstance(qc, QueryClass):
        return qc
    try:
        return QueryClass(str(qc))
    except ValueError:
        return QueryClass.CLARIFICATION


def _coerce_verdict(v: Any) -> Verdict:
    """Coerce a verdict value to :class:`Verdict`, defaulting to ``reject``."""
    if isinstance(v, Verdict):
        return v
    s = str(v).strip().lower()
    if s == Verdict.ACCEPT.value:
        return Verdict.ACCEPT
    if s == Verdict.NA.value:
        return Verdict.NA
    return Verdict.REJECT


def _enum_value(v: Any) -> str:
    """Return the ``.value`` of an enum, or ``str(v)`` for a plain value."""
    return v.value if hasattr(v, "value") else str(v)


def _verdict_value(v: Any) -> str:
    """Normalize a verdict (enum or string) to its string value."""
    return v.value if isinstance(v, Verdict) else str(v)
