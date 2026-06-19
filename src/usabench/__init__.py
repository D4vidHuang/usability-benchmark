"""usabench -- a benchmark for the *usability* of AI coding agents.

The benchmark gives an agent an open-ended, under-specified software goal, places
an LLM simulated-user oracle in the loop, and counts + severity-grades every
interaction. The headline number rewards building the right thing with minimal
human babysitting.

This top-level module keeps imports lightweight: it re-exports the most-used core
contracts (enums + a few schema models) but does NOT import any heavy optional
dependency (anthropic, openai, vllm, datasketch, ...). Those live behind lazy
imports in their own subpackages so ``import usabench`` is cheap and side-effect
free.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Lightweight, dependency-free re-exports. Importing these does not pull in any
# heavy/optional dependency.
from usabench.core.enums import (
    Actor,
    InteractionType,
    Provider,
    QueryClass,
    RunStatus,
    Severity,
    TerminatedReason,
)
from usabench.core.errors import (
    BudgetExceeded,
    OracleProtocolError,
    ProviderError,
    SandboxError,
    SchemaViolation,
    UsabenchError,
)
from usabench.core.schema import (
    SCHEMA_VERSION,
    AcceptanceCriterion,
    HiddenSpec,
    OracleResponse,
    Task,
    TaskEnv,
    TraceEnvelope,
    parse_event,
)

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    # enums
    "Actor",
    "InteractionType",
    "Provider",
    "QueryClass",
    "RunStatus",
    "Severity",
    "TerminatedReason",
    # errors
    "UsabenchError",
    "BudgetExceeded",
    "OracleProtocolError",
    "ProviderError",
    "SandboxError",
    "SchemaViolation",
    # schema
    "AcceptanceCriterion",
    "HiddenSpec",
    "OracleResponse",
    "Task",
    "TaskEnv",
    "TraceEnvelope",
    "parse_event",
]
