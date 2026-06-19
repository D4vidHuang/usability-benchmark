"""Two-stage interaction classifier (``docs/protocol.md`` §3.3, metrics.md §3).

Every oracle exchange is classified into a closed **intervention type** and a 0-5
**severity**. The classification is a *pure offline function of the trace* so it is
reproducible and re-runnable (``DESIGN.md`` invariant 4).

Stage A -- **rule-based, authoritative where deterministic.**
    Type and severity are read directly from the structured fields already in the
    trace (the oracle's self-declared ``level``/``verdict``/``reveals``). E.g.
    ``oracle_response.level == 2`` -> ``hint_directional``, ``S = 2``. A consistency
    assertion checks the self-declared level against the level implied by
    ``reveals``/``verdict``; a mismatch raises ``classifier_conflict``.

Stage B -- **LLM-judge fallback, only for genuinely linguistic decisions.**
    Used to (a) set ``submission_reject`` severity from ``names_fix``, (b) detect
    R2 leaks (an unsolicited reveal of a hidden-spec item the query did not
    trigger), and (c) override Stage A only when Stage A returns ``ambiguous``.
    The judge runs at a low, pinned temperature (read from the frozen spec).

The classifier never *decides* what the oracle said -- it only labels what is
already in the trace, so the oracle and the classifier are mutually checkable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from usabench.core.enums import QueryClass, Severity, Verdict
from usabench.core.schema import HiddenSpec, OracleQuery, OracleResponse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from usabench.llm.client import LLMClient

__all__ = [
    "InterventionType",
    "Classification",
    "InteractionClassifier",
    "type_for_level",
]

_log = structlog.get_logger("usabench.oracle.classifier")


class InterventionType:
    """The closed set of intervention-type strings (``docs/protocol.md`` §3.1).

    The scorer rejects any type outside this set. Severity is a separate 0-5 axis;
    the mapping from ladder level to type is fixed by :func:`type_for_level`.
    """

    CLARIFICATION = "clarification"                  # L0 clarify, S0
    PREFERENCE_REVEAL = "preference_reveal"          # triggered hidden preference, S0
    SCOPE_CONFIRMATION = "scope_confirmation"        # in/out of scope, S0
    HINT_NUDGE = "hint_nudge"                         # L1, S1
    HINT_DIRECTIONAL = "hint_directional"            # L2, S2
    HINT_CONCRETE = "hint_concrete"                  # L3, S3
    RESCUE_SNIPPET = "rescue_snippet"                # L4, S4
    TAKEOVER = "takeover"                             # L5, S5
    PROACTIVE_OFFER_ACCEPTED = "proactive_offer_accepted"
    PROACTIVE_OFFER_DECLINED = "proactive_offer_declined"
    SUBMISSION_REJECT = "submission_reject"          # reject; S = 1 + names_fix
    ERROR_RECOVERY_HINT = "error_recovery_hint"      # S = ladder level used
    HUMAN_FALLBACK = "human_fallback"                # S5, flagged loudly

    ALL: tuple[str, ...] = (
        CLARIFICATION,
        PREFERENCE_REVEAL,
        SCOPE_CONFIRMATION,
        HINT_NUDGE,
        HINT_DIRECTIONAL,
        HINT_CONCRETE,
        RESCUE_SNIPPET,
        TAKEOVER,
        PROACTIVE_OFFER_ACCEPTED,
        PROACTIVE_OFFER_DECLINED,
        SUBMISSION_REJECT,
        ERROR_RECOVERY_HINT,
        HUMAN_FALLBACK,
    )


#: Ladder level -> the intervention type that level implies (the plain hint path).
_LEVEL_TYPE: dict[int, str] = {
    0: InterventionType.CLARIFICATION,
    1: InterventionType.HINT_NUDGE,
    2: InterventionType.HINT_DIRECTIONAL,
    3: InterventionType.HINT_CONCRETE,
    4: InterventionType.RESCUE_SNIPPET,
    5: InterventionType.TAKEOVER,
}


def type_for_level(level: int) -> str:
    """Map a ladder/severity level (0-5) to its plain intervention type.

    Args:
        level: Ladder level in ``[0, 5]``.

    Returns:
        The corresponding :class:`InterventionType` string.

    Raises:
        ValueError: If ``level`` is outside ``[0, 5]``.
    """
    if level not in _LEVEL_TYPE:
        raise ValueError(f"level out of range [0,5]: {level!r}")
    return _LEVEL_TYPE[level]


@dataclass
class Classification:
    """The result of classifying one oracle exchange.

    Attributes:
        type: One of :class:`InterventionType` values.
        severity: The 0-5 :class:`Severity` for this exchange.
        level_declared: The oracle's self-declared ladder level.
        level_classified: The level the classifier concluded (== severity int).
        classifier_conflict: True if Stage-A consistency assertions failed.
        leaked_ids: hidden_spec ids the oracle revealed *unsolicited* (R2 leak).
        leak_flag: True if any unsolicited reveal was detected.
        names_fix: For rejects, whether the feedback named the concrete fix.
        needs_review: True if Stage A and Stage B disagreed (human-audit sample).
        stage_a: Stage-A method tag ("rule" | "ambiguous").
        stage_b: Stage-B method tag (None if the judge did not run).
        rationale: Short free-text explanation (debug / audit).
    """

    type: str
    severity: Severity
    level_declared: int
    level_classified: int
    classifier_conflict: bool = False
    leaked_ids: list[str] = field(default_factory=list)
    leak_flag: bool = False
    names_fix: bool | None = None
    needs_review: bool = False
    stage_a: str = "rule"
    stage_b: str | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the ``intervention`` payload shape used in the trace."""
        return {
            "type": self.type,
            "severity": int(self.severity),
            "level_declared": self.level_declared,
            "level_classified": self.level_classified,
            "classifier_conflict": self.classifier_conflict,
            "leaked_ids": list(self.leaked_ids),
            "leak_flag": self.leak_flag,
            "names_fix": self.names_fix,
            "needs_review": self.needs_review,
            "classified_by": {"stageA": self.stage_a, "stageB": self.stage_b},
            "rationale": self.rationale,
        }


@dataclass
class InteractionClassifier:
    """Two-stage classifier over oracle exchanges (rule-first, judge-second).

    Stage A is always run and is authoritative for severity wherever the
    structured signal suffices. Stage B (the LLM judge) runs only when a judge
    client is supplied AND a linguistic decision is genuinely required (a reject's
    ``names_fix``, leak detection, or an ``ambiguous`` Stage-A result).

    Attributes:
        hidden: The frozen oracle-private gold (leak detection reference).
        judge: Optional LLM client for Stage B. If ``None``, Stage B is skipped
            and free-text decisions fall back to conservative rule defaults.
    """

    hidden: HiddenSpec
    judge: LLMClient | None = None

    # --- public API --------------------------------------------------------- #

    def classify_response(
        self,
        response: OracleResponse,
        query: OracleQuery | None = None,
        *,
        from_offer: bool = False,
        offer_accepted: bool = True,
    ) -> Classification:
        """Classify a single ``oracle_response`` exchange.

        Args:
            response: The oracle's structured response event.
            query: The originating ``oracle_query``, if any (None = unsolicited).
            from_offer: True if this response stems from a stuck-offer.
            offer_accepted: For ``from_offer``, whether the offer was accepted.

        Returns:
            A :class:`Classification` ready to be written as an ``intervention``.
        """
        level = int(response.severity)

        # Stuck-offer bookkeeping (§3.1): a declined offer transfers no assistance.
        if from_offer and not offer_accepted:
            return Classification(
                type=InterventionType.PROACTIVE_OFFER_DECLINED,
                severity=Severity.NONE,
                level_declared=level,
                level_classified=0,
                stage_a="rule",
                rationale="stuck-offer declined; no information transferred",
            )

        # Stage A: type from the structured level + reveals.
        a_type, a_conflict, a_rationale = self._stage_a_type(response, level)
        if from_offer and offer_accepted:
            a_type = InterventionType.PROACTIVE_OFFER_ACCEPTED

        cls = Classification(
            type=a_type,
            severity=Severity.from_level(level),
            level_declared=level,
            level_classified=level,
            classifier_conflict=a_conflict,
            stage_a="rule",
            rationale=a_rationale,
        )

        # Stage B: leak detection (always attempted when a judge exists, since a
        # leak is an unsolicited reveal regardless of level).
        self._detect_leak(cls, response, query)
        return cls

    def classify_review(self, response: OracleResponse) -> Classification:
        """Classify a ``submit`` review (accept / reject) exchange.

        An accept transfers no assistance (S0). A reject is ``submission_reject``
        with severity ``1 + names_fix`` (``docs/protocol.md`` §3.1): a reject that
        names the concrete fix is more help, hence higher severity. ``names_fix``
        is decided by Stage B when a judge is available, else from the oracle's
        self-declared review severity.

        Args:
            response: The oracle's review response (carries ``verdict``).

        Returns:
            A :class:`Classification` for the review.
        """
        verdict = _verdict_value(response.verdict)
        if verdict == Verdict.ACCEPT.value:
            return Classification(
                type=InterventionType.CLARIFICATION,  # accept != assistance
                severity=Severity.NONE,
                level_declared=int(response.severity),
                level_classified=0,
                stage_a="rule",
                rationale="submission accepted; no assistance",
            )

        # Reject: severity = 1 + (named-fix specificity).
        names_fix = self._stage_b_names_fix(response)
        if names_fix is None:
            # No judge: infer from self-declared severity (>=2 implies a named fix).
            names_fix = int(response.severity) >= 2
        sev_int = min(1 + (1 if names_fix else 0), 5)
        if int(response.severity) > sev_int:
            # Honor a higher self-declared severity (oracle named a deep fix).
            sev_int = min(int(response.severity), 5)
        return Classification(
            type=InterventionType.SUBMISSION_REJECT,
            severity=Severity.from_level(sev_int),
            level_declared=int(response.severity),
            level_classified=sev_int,
            names_fix=names_fix,
            stage_a="rule",
            stage_b="judge" if self.judge is not None else None,
            rationale="submission rejected; severity = 1 + names_fix",
        )

    def classify_query(self, query: OracleQuery) -> QueryClass:
        """Return the (already-typed) class of an agent query.

        The query class is structured in the trace, so this is a pass-through that
        normalizes a possibly-string value into the :class:`QueryClass` enum.

        Args:
            query: The agent's oracle query.

        Returns:
            The normalized :class:`QueryClass`.
        """
        qc = query.query_class
        if isinstance(qc, QueryClass):
            return qc
        try:
            return QueryClass(str(qc))
        except ValueError:
            _log.warning("unknown_query_class", value=qc)
            return QueryClass.CLARIFICATION

    # --- Stage A internals -------------------------------------------------- #

    def _stage_a_type(self, response: OracleResponse, level: int) -> tuple[str, bool, str]:
        """Derive intervention type + conflict flag from structured fields.

        Returns:
            ``(type, classifier_conflict, rationale)``.
        """
        conflict = False
        rationale = f"level={level} -> {_LEVEL_TYPE.get(level, '??')}"

        if level == 0:
            # A triggered preference reveal vs a scope confirmation vs plain clarify.
            if response.reveals:
                return InterventionType.PREFERENCE_REVEAL, conflict, "L0 with reveals"
            if response.refusals:
                return InterventionType.SCOPE_CONFIRMATION, conflict, "L0 refusal/scope"
            return InterventionType.CLARIFICATION, conflict, rationale

        # Consistency assertion (§3.3): an accept/reject verdict on a non-review
        # response, or a reveal at a hint level, is suspicious.
        if _verdict_value(response.verdict) != Verdict.NA.value:
            conflict = True
            rationale += " ; verdict set on a non-review response (conflict)"

        a_type = _LEVEL_TYPE.get(level, InterventionType.HINT_DIRECTIONAL)
        return a_type, conflict, rationale

    # --- Stage B internals (LLM judge) ------------------------------------- #

    def _detect_leak(
        self,
        cls: Classification,
        response: OracleResponse,
        query: OracleQuery | None,
    ) -> None:
        """Flag unsolicited reveals (R2 leaks) and set ``leak_flag`` in place.

        A leak is the disclosure of a hidden-spec item that the query did NOT
        trigger. Stage B (a judge) is authoritative when present; otherwise a
        conservative rule fires only on an *unsolicited* response that still
        carried reveals (``responds_to is None`` with non-empty ``reveals``).
        """
        if self.judge is not None:
            leaked = self._stage_b_leaks(response, query)
            if leaked:
                cls.leaked_ids = leaked
                cls.leak_flag = True
                cls.stage_b = "judge"
                cls.needs_review = True
                _log.warning("oracle_leak_detected", leaked=leaked)
            return

        # No judge: a reveal is a clear rule-detectable leak only when there was no
        # originating query to trigger it (a truly unsolicited disclosure). When a
        # query IS present the reveal is solicited and only the LLM judge can decide
        # whether the *specific* item was actually triggered, so we abstain.
        if query is None and response.reveals:
            cls.leaked_ids = list(response.reveals)
            cls.leak_flag = True
            cls.needs_review = True
            _log.warning("oracle_unsolicited_reveal", reveals=response.reveals)

    def _stage_b_leaks(
        self, response: OracleResponse, query: OracleQuery | None
    ) -> list[str]:
        """Ask the judge which hidden ids were revealed without being triggered."""
        prompt = _LEAK_JUDGE_PROMPT.format(
            query=(query.text if query else "(unsolicited -- no query)"),
            response=response.text,
            hidden=_render_hidden_for_judge(self.hidden),
        )
        data = self._judge_json(prompt)
        leaked = data.get("leaked_ids", []) if isinstance(data, dict) else []
        return [str(x) for x in leaked] if isinstance(leaked, list) else []

    def _stage_b_names_fix(self, response: OracleResponse) -> bool | None:
        """Ask the judge whether a reject's feedback named the concrete fix."""
        if self.judge is None:
            return None
        prompt = _NAMES_FIX_JUDGE_PROMPT.format(feedback=response.text)
        data = self._judge_json(prompt)
        if isinstance(data, dict) and "names_fix" in data:
            return bool(data["names_fix"])
        return None

    def _judge_json(self, prompt: str) -> dict[str, Any]:
        """Run the judge at the spec temperature and parse its JSON reply.

        Returns ``{}`` on any failure so classification degrades gracefully to the
        rule-based defaults rather than crashing the offline pass.
        """
        if self.judge is None:
            return {}
        from usabench.eval.spec import load_spec
        from usabench.llm.client import Message

        temperature = float(load_spec().get("judge", {}).get("temperature", 0.2))
        try:
            completion = self.judge.chat(
                [Message(role="user", content=prompt)],
                temperature=temperature,
                max_tokens=512,
            )
        except Exception as exc:  # pragma: no cover - provider failure path
            _log.warning("judge_call_failed", error=str(exc))
            return {}
        return _safe_json(completion.text)


# --------------------------------------------------------------------------- #
# Stage-B judge prompts + small JSON helpers.                                  #
# --------------------------------------------------------------------------- #

_LEAK_JUDGE_PROMPT = """\
You audit a simulated user (oracle) for information leaks. The oracle must NOT
reveal a hidden-spec item unless the agent's question triggered it.

AGENT QUERY:
{query}

ORACLE RESPONSE:
{response}

HIDDEN SPEC ITEMS (id: description):
{hidden}

List ONLY the ids whose content the oracle disclosed WITHOUT the query asking for
them. Respond as JSON: {{"leaked_ids": ["<id>", ...]}}. Empty list if none.
"""

_NAMES_FIX_JUDGE_PROMPT = """\
A simulated user rejected an agent's submission with this feedback:

{feedback}

Did the feedback NAME THE CONCRETE FIX (a specific change/approach to make), as
opposed to merely saying something is wrong? Respond as JSON:
{{"names_fix": true|false}}.
"""


def _render_hidden_for_judge(hidden: HiddenSpec) -> str:
    """Render hidden-spec ids+descriptions as a compact list for the leak judge."""
    lines: list[str] = []
    for ap in hidden.ambiguity_points:
        lines.append(f"{ap.id}: {ap.question} -> {ap.gold}")
    for u in hidden.info_units:
        lines.append(f"{u.id}: {u.desc}")
    for p in hidden.known_pitfalls:
        lines.append(f"pitfall: {p}")
    return "\n".join(lines) if lines else "(none)"


def _safe_json(text: str) -> dict[str, Any]:
    """Best-effort parse of a JSON object from possibly-fenced model output."""
    if not text:
        return {}
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
    except (json.JSONDecodeError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _verdict_value(v: Any) -> str:
    """Normalize a verdict (enum or string) to its string value."""
    return v.value if isinstance(v, Verdict) else str(v)
