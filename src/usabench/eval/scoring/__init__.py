"""Goal-Achievement (GA) scoring channels V1/V2/V3 and their composition.

``docs/scoring.md`` defines GA as a weighted blend of three verification
channels, capped by a hard-constraint gate:

* **V1** -- functional / sandbox execution (deterministic; highest trust).
* **V2** -- frozen rubric / acceptance-criteria checklist (semi-objective).
* **V3** -- LLM-as-judge jury (noisiest; lowest weight; most safeguarded).

``GA = (w_v1*V1 + w_v2*V2 + w_v3*V3) * gate(hard_pass_frac)`` with channel
weights and ``gate(h) = floor + slope*h`` read from ``usability_score.yaml``.

Every score function in this subpackage is a pure function of structured inputs
(channel results that the harness/sandbox produced, or the canonical
``trace.jsonl``) plus the frozen task gold; nothing runs Docker or an LLM at
import time.
"""

from __future__ import annotations

from usabench.eval.scoring.ga import GAResult, compute_ga, gate
from usabench.eval.scoring.v1_functional import V1Result, score_v1, score_v1_from_outcomes
from usabench.eval.scoring.v2_rubric import V2Result, score_v2
from usabench.eval.scoring.v3_judge import V3Result, score_v3

__all__ = [
    "V1Result",
    "score_v1",
    "score_v1_from_outcomes",
    "V2Result",
    "score_v2",
    "V3Result",
    "score_v3",
    "GAResult",
    "compute_ga",
    "gate",
]
