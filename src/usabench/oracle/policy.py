"""Oracle disclosure policy: reveal rules R1-R7 and the hint ladder L0-L5.

This module is the *deterministic, dependency-light* core of the oracle. It owns:

* the **hint ladder** (``docs/protocol.md`` §2.5): levels L0-L5 mapped onto the
  canonical 0-5 :class:`~usabench.core.enums.Severity` scale, with monotonic
  one-step escalation keyed on repeat-requests / repeated failure;
* the **reveal rules** R1-R7 (``docs/protocol.md`` §2.3) as machine-checkable
  predicates -- in particular which :class:`~usabench.core.schema.AmbiguityPoint`
  a given query may trigger, honoring each point's
  :class:`~usabench.core.enums.RevealRule`;
* the **hint budget** (``docs/tasks.md`` §3.1: ``oracle_persona.hint_budget``):
  once exceeded, an additional hint is reclassified as a *rescue* (severity is
  bumped), so an over-helpful trace is detectable;
* **out-of-scope refusal** (R6) and the **stuck-offer** rate-limiting (§2.4).

Nothing here calls an LLM. The LLM-backed :class:`~usabench.oracle.oracle.SimulatedUserOracle`
consults this policy to pick the *target level*, bound the model, and post-validate
the model's self-declared level. Severity weights are NOT defined here -- when the
policy needs the convex weights it reads them from the frozen spec via
:func:`usabench.eval.spec.get_severity_weights`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from usabench.core.enums import QueryClass, RevealRule, Severity
from usabench.core.schema import AmbiguityPoint, HiddenSpec

__all__ = [
    "Helpfulness",
    "HINT_LADDER",
    "MAX_AUTO_LEVEL",
    "PolicyConfig",
    "RevealDecision",
    "OraclePolicy",
    "escalation_repeats_for",
]

_log = structlog.get_logger("usabench.oracle.policy")

#: Hard ceiling on the hint level in automated runs. L5 (takeover) is a human
#: fallback only -- the oracle must never reach it on its own (R4, §2.5).
MAX_AUTO_LEVEL: int = 4

#: Human-readable name for each ladder level, indexed by level 0..5. Mirrors the
#: ladder table in ``docs/protocol.md`` §2.5 and the severity labels in metrics.md.
HINT_LADDER: dict[int, str] = {
    0: "clarification",      # restate/confirm scope; reveal a triggered preference
    1: "nudge",              # point at the area, no answer
    2: "directional_hint",   # name the concept / pitfall
    3: "concrete_guidance",  # describe the fix in prose, no code
    4: "rescue_snippet",     # one small snippet / exact API call
    5: "takeover",           # oracle supplies a gating impl (human fallback only)
}


class Helpfulness:
    """The three oracle-helpfulness presets (``docs/protocol.md`` §2.6).

    Helpfulness changes *escalation thresholds only*, never truthfulness. The
    default scored configuration is ``standard``.
    """

    STRICT = "strict"
    STANDARD = "standard"
    LENIENT = "lenient"

    ALL: tuple[str, ...] = (STRICT, STANDARD, LENIENT)


#: How many explicit repeat-requests (after a failure) are required before the
#: oracle is allowed to escalate one level, per helpfulness preset.
_ESCALATION_REPEATS: dict[str, int] = {
    Helpfulness.STRICT: 2,
    Helpfulness.STANDARD: 1,
    Helpfulness.LENIENT: 0,
}


def escalation_repeats_for(helpfulness: str) -> int:
    """Return the repeat-request count required to escalate, for a helpfulness preset.

    Args:
        helpfulness: One of :class:`Helpfulness` values.

    Returns:
        The number of post-failure repeat requests required before the oracle may
        climb one ladder level. Unknown presets fall back to ``standard``.
    """
    return _ESCALATION_REPEATS.get(helpfulness, _ESCALATION_REPEATS[Helpfulness.STANDARD])


@dataclass
class PolicyConfig:
    """Tunable, reproducible knobs for the disclosure policy.

    Attributes:
        helpfulness: One of :class:`Helpfulness` values (escalation thresholds).
        hint_budget: Max graded hints (L1+) before extra hints become rescues
            (``oracle_persona.hint_budget``, ``docs/tasks.md`` §3.1).
        proactive_stuck_help: Whether the harness may offer stuck-nudges (§2.4).
        stuck_offer_cooldown_turns: Min turns between stuck-offers (rate limit).
        max_level: Hard ceiling on the hint level (never L5 in auto runs).
    """

    helpfulness: str = Helpfulness.STANDARD
    hint_budget: int = 3
    proactive_stuck_help: bool = True
    stuck_offer_cooldown_turns: int = 20
    max_level: int = MAX_AUTO_LEVEL

    def __post_init__(self) -> None:
        if self.helpfulness not in Helpfulness.ALL:
            _log.warning("unknown_helpfulness_preset", preset=self.helpfulness)
            self.helpfulness = Helpfulness.STANDARD
        self.max_level = max(0, min(int(self.max_level), MAX_AUTO_LEVEL))

    @property
    def escalation_repeats(self) -> int:
        """Repeat-requests required before a one-level escalation."""
        return escalation_repeats_for(self.helpfulness)


@dataclass
class RevealDecision:
    """The policy's verdict on whether a query may trigger a hidden reveal.

    Attributes:
        triggered_ids: Ambiguity-point / info-unit ids the query is allowed to
            unlock (subject to each point's :class:`RevealRule`).
        is_out_of_scope: True if the query is about an out-of-scope item (R6).
        out_of_scope_match: The matched out-of-scope phrase, if any.
        max_reveal_level: The highest ladder level at which a reveal is permitted
            for the triggered points (``on_ask`` -> L0, ``on_hint`` -> L1+).
    """

    triggered_ids: list[str] = field(default_factory=list)
    is_out_of_scope: bool = False
    out_of_scope_match: str | None = None
    max_reveal_level: int = 0


@dataclass
class OraclePolicy:
    """Stateful, per-episode disclosure policy.

    Tracks the current ladder level and the number of graded hints already spent
    so escalation stays monotonic and the hint budget is enforceable. The policy
    is consulted by the oracle for every query; it does not itself call the model.

    Attributes:
        hidden: The frozen oracle-private gold (drives reveal/scope decisions).
        config: The tunable :class:`PolicyConfig`.
    """

    hidden: HiddenSpec
    config: PolicyConfig = field(default_factory=PolicyConfig)

    #: Highest ladder level reached so far (monotonic non-decreasing per query line).
    _current_level: int = field(default=0, init=False)
    #: Number of graded (L1+) hints dispensed so far this episode.
    _hints_spent: int = field(default=0, init=False)
    #: Turn index of the last stuck-offer, for cooldown rate-limiting (-inf-ish).
    _last_offer_turn: int = field(default=-(10**9), init=False)

    # --- reveal / scope decisions ------------------------------------------ #

    def classify_scope(self, query_text: str) -> RevealDecision:
        """Decide what a free-text query may unlock and whether it is out-of-scope.

        Applies R2 (no volunteering) and R6 (scope guarding) deterministically:
        an :class:`AmbiguityPoint` is *triggerable* only when the query text
        overlaps its question keywords AND its :class:`RevealRule` permits a
        release at the current interaction stage.

        Args:
            query_text: The agent's raw oracle-query text.

        Returns:
            A :class:`RevealDecision` describing triggered ids and scope status.
        """
        text = (query_text or "").lower()
        decision = RevealDecision()

        # R6: out-of-scope match -> answer "not needed", reveal nothing.
        for oos in self.hidden.out_of_scope:
            if _phrase_overlaps(text, oos):
                decision.is_out_of_scope = True
                decision.out_of_scope_match = oos
                _log.debug("query_out_of_scope", match=oos)
                return decision

        # R2: a hidden preference is unlocked only if its reveal rule allows it.
        for ap in self.hidden.ambiguity_points:
            if ap.reveal == RevealRule.NEVER_VOLUNTEER:
                continue
            if _query_triggers_point(text, ap):
                decision.triggered_ids.append(ap.id)
                lvl = 0 if ap.reveal == RevealRule.ON_ASK else 1
                decision.max_reveal_level = max(decision.max_reveal_level, lvl)

        return decision

    def may_reveal(self, ap: AmbiguityPoint, at_level: int) -> bool:
        """Whether ambiguity point ``ap`` may be revealed at hint level ``at_level``.

        Args:
            ap: The candidate ambiguity point.
            at_level: The ladder level at which the reveal would occur.

        Returns:
            True iff the point's :class:`RevealRule` permits release at this level.
        """
        if ap.reveal == RevealRule.NEVER_VOLUNTEER:
            return False
        if ap.reveal == RevealRule.ON_ASK:
            return True  # askable at any level, including L0
        # ON_HINT: only once the agent is in graded-hint territory.
        return at_level >= 1

    # --- hint-ladder level selection --------------------------------------- #

    def target_level(
        self,
        query_class: QueryClass,
        *,
        requested_hint: bool,
        repeat_requests: int,
        prior_failures: int,
        from_accepted_offer: bool = False,
    ) -> int:
        """Pick the LOWEST ladder level that addresses the query (R4).

        Escalation is monotonic and at most one level per qualifying event: the
        agent must re-ask (``repeat_requests``) after a demonstrated failure
        (``prior_failures``), or have accepted a stuck-offer.

        Args:
            query_class: The class of the agent's query.
            requested_hint: True if the agent explicitly asked for a hint.
            repeat_requests: Number of repeat hint requests after a failure.
            prior_failures: Number of demonstrated failures (failed verifications /
                failed tool runs) preceding this query.
            from_accepted_offer: True if this stems from an accepted stuck-offer.

        Returns:
            The target ladder level in ``[0, config.max_level]``.
        """
        # Pure clarification / scope / confirmation never leaves L0.
        if query_class in (QueryClass.CLARIFICATION, QueryClass.CONFIRMATION):
            return self._bump_floor(0)
        if query_class == QueryClass.OUT_OF_SCOPE:
            return self._bump_floor(0)
        if query_class == QueryClass.HANDOFF_REQUEST:
            # A handoff is a takeover request; in auto runs we cap at max_level and
            # let the harness flag it. The oracle still never *writes* the solution.
            return self.config.max_level

        # Hint requests: start at L1, escalate at most one step per qualifying repeat.
        if requested_hint or query_class == QueryClass.HINT_REQUEST or from_accepted_offer:
            base = max(1, self._current_level)
            qualifies = from_accepted_offer or (
                repeat_requests >= self.config.escalation_repeats and prior_failures >= 1
            )
            target = base + 1 if (qualifies and self._current_level >= 1) else base
            return self._bump_floor(min(target, self.config.max_level))

        # Default: treat as a clarification (no assistance).
        return self._bump_floor(0)

    def _bump_floor(self, level: int) -> int:
        """Record ``level`` as the new monotonic floor and return it (clamped)."""
        level = max(0, min(int(level), self.config.max_level))
        self._current_level = max(self._current_level, level)
        return level

    def register_dispensed(self, level: int) -> Severity:
        """Account for a dispensed response and return its *effective* severity.

        Enforces the hint budget (``docs/tasks.md`` §3.1): the first
        ``hint_budget`` graded hints (L1+) keep their ladder severity; once the
        budget is exhausted, an additional graded hint is reclassified as a
        *rescue* -- its severity is bumped by one level (capped at L4) so the
        over-helpful trace is detectable downstream.

        Args:
            level: The ladder level the oracle actually used for the response.

        Returns:
            The effective :class:`Severity` to stamp on the ``oracle_response``.
        """
        level = max(0, min(int(level), 5))
        effective = level
        if Severity.is_assistance(level):
            self._hints_spent += 1
            if self._hints_spent > self.config.hint_budget:
                effective = min(level + 1, MAX_AUTO_LEVEL)
                _log.info(
                    "hint_budget_exceeded_rescue",
                    spent=self._hints_spent,
                    budget=self.config.hint_budget,
                    declared_level=level,
                    effective_level=effective,
                )
        return Severity.from_level(effective)

    # --- stuck-offer rate limiting (§2.4) ---------------------------------- #

    def may_offer_stuck_help(self, turn: int) -> bool:
        """Whether a stuck-offer is permitted at agent-turn ``turn``.

        Honors both the ``proactive_stuck_help`` run-condition switch and the
        per-offer cooldown so the oracle cannot babysit the agent.

        Args:
            turn: The current agent-turn index.

        Returns:
            True iff proactive help is enabled and the cooldown has elapsed.
        """
        if not self.config.proactive_stuck_help:
            return False
        return (turn - self._last_offer_turn) >= self.config.stuck_offer_cooldown_turns

    def record_offer(self, turn: int) -> None:
        """Record that a stuck-offer was emitted at agent-turn ``turn``."""
        self._last_offer_turn = turn

    # --- introspection ------------------------------------------------------ #

    @property
    def current_level(self) -> int:
        """The highest ladder level reached so far this episode."""
        return self._current_level

    @property
    def hints_spent(self) -> int:
        """Number of graded (L1+) hints dispensed so far this episode."""
        return self._hints_spent

    @property
    def hint_budget_remaining(self) -> int:
        """Graded hints remaining before extra hints become rescues (>=0)."""
        return max(0, self.config.hint_budget - self._hints_spent)


# --------------------------------------------------------------------------- #
# Lightweight keyword-overlap helpers (deterministic, no LLM).                  #
# --------------------------------------------------------------------------- #

#: Tokens that carry no topical signal -- ignored when matching query <-> point.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "do", "does", "did", "should", "would",
        "could", "can", "i", "you", "it", "this", "that", "to", "of", "for",
        "and", "or", "in", "on", "as", "be", "by", "with", "what", "how",
        "want", "need", "use", "using", "my", "your", "me", "we", "they",
        "have", "has", "will", "shall", "if", "when", "which", "whether",
    }
)


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, drop stopwords -> a content-token set."""
    out: set[str] = set()
    for raw in text.lower().split():
        tok = "".join(ch for ch in raw if ch.isalnum())
        if tok and tok not in _STOPWORDS and len(tok) > 2:
            out.add(tok)
    return out


def _phrase_overlaps(haystack_text: str, phrase: str) -> bool:
    """True if ``phrase`` substring-matches or shares >=1 content token with text."""
    p = phrase.lower().strip()
    if not p:
        return False
    if p in haystack_text:
        return True
    return bool(_tokenize(haystack_text) & _tokenize(phrase))


def _query_triggers_point(query_text: str, ap: AmbiguityPoint) -> bool:
    """Heuristic: does a query topically match an ambiguity point's question?

    Deterministic keyword overlap between the (already-lowercased) query text and
    the ambiguity point's question. The LLM oracle makes the *final* call inside
    its persona, but this gate prevents the policy from ever pre-marking a reveal
    the agent didn't actually probe (a defensive R2 backstop).
    """
    q_tokens = _tokenize(query_text)
    if not q_tokens:
        return False
    point_tokens = _tokenize(ap.question)
    return bool(q_tokens & point_tokens)
