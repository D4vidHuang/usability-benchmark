"""The harness: orchestration that runs one (task, agent, oracle) episode.

This package owns the *runtime* contract of the benchmark (``docs/protocol.md``):
it drives the interaction loop, mediates the single agent<->oracle channel, runs
the sandbox, enforces budgets, builds the run manifest, and -- above all -- writes
the ONE canonical artifact, the hash-chained ``trace.jsonl``. Every metric is a
pure offline function of that trace plus the frozen gold; the harness never holds
scorer-needed state only in memory (``DESIGN.md`` invariant 4).

Public surface:

* :func:`~usabench.harness.runner.run_episode` -- the loop + trace writer.
* :class:`~usabench.harness.interaction_bus.InteractionBus` -- trace writer + oracle gateway.
* :class:`~usabench.harness.sandbox.SandboxBackend` and the working
  :class:`~usabench.harness.sandbox.LocalSubprocessSandbox` (+ an Apptainer stub).
* :class:`~usabench.harness.budget.BudgetMeter` / :class:`~usabench.harness.budget.BudgetLimits`.
* :func:`~usabench.harness.manifest.build_manifest`.
* :func:`~usabench.harness.batch.run_batch` and the planning helpers.
"""

from __future__ import annotations

from usabench.harness.batch import (
    BatchCell,
    BatchPlan,
    is_run_complete,
    iter_cells,
    plan_batch,
    run_batch,
)
from usabench.harness.budget import (
    BudgetDebitRecord,
    BudgetLimits,
    BudgetMeter,
)
from usabench.harness.interaction_bus import (
    InteractionBus,
    OracleLike,
    OracleQueryContext,
)
from usabench.harness.manifest import build_manifest, git_sha, oracle_prompt_hash
from usabench.harness.runner import VerifierLike, run_episode
from usabench.harness.sandbox import (
    ApptainerSandbox,
    ExecResult,
    LocalSubprocessSandbox,
    SandboxBackend,
    WorkspaceSnapshot,
)

__all__ = [
    # runner
    "run_episode",
    "VerifierLike",
    # interaction bus
    "InteractionBus",
    "OracleLike",
    "OracleQueryContext",
    # sandbox
    "SandboxBackend",
    "LocalSubprocessSandbox",
    "ApptainerSandbox",
    "ExecResult",
    "WorkspaceSnapshot",
    # budget
    "BudgetMeter",
    "BudgetLimits",
    "BudgetDebitRecord",
    # manifest
    "build_manifest",
    "git_sha",
    "oracle_prompt_hash",
    # batch
    "run_batch",
    "plan_batch",
    "iter_cells",
    "is_run_complete",
    "BatchCell",
    "BatchPlan",
]
