"""The simulated-user oracle subtree.

Public surface:

* :class:`~usabench.oracle.oracle.SimulatedUserOracle` -- the LLM-backed user that
  answers queries (:meth:`~usabench.oracle.oracle.SimulatedUserOracle.answer`) and
  reviews submissions (:meth:`~usabench.oracle.oracle.SimulatedUserOracle.review`),
  self-labeling each response with a 0-5 severity validated by the classifier.
* :class:`~usabench.oracle.policy.OraclePolicy` -- the deterministic disclosure
  policy: reveal rules R1-R7, the L0-L5 hint ladder, hint budget, out-of-scope
  refusal, and stuck-offer rate limiting.
* :class:`~usabench.oracle.classifier.InteractionClassifier` -- the two-stage
  (rule-based + LLM-judge) interaction classifier.

Heavy/optional deps (jinja2 templates, the LLM client) are imported lazily inside
the relevant call sites so importing this subtree is cheap.
"""

from __future__ import annotations

from usabench.oracle.classifier import (
    Classification,
    InteractionClassifier,
    InterventionType,
    type_for_level,
)
from usabench.oracle.oracle import (
    AnswerResult,
    OracleConfig,
    OracleReview,
    SimulatedUserOracle,
)
from usabench.oracle.policy import (
    HINT_LADDER,
    MAX_AUTO_LEVEL,
    Helpfulness,
    OraclePolicy,
    PolicyConfig,
    RevealDecision,
)

__all__ = [
    # oracle
    "SimulatedUserOracle",
    "OracleConfig",
    "OracleReview",
    "AnswerResult",
    # policy
    "OraclePolicy",
    "PolicyConfig",
    "RevealDecision",
    "Helpfulness",
    "HINT_LADDER",
    "MAX_AUTO_LEVEL",
    # classifier
    "InteractionClassifier",
    "Classification",
    "InterventionType",
    "type_for_level",
]
