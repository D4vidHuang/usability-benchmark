"""Evaluation: scoring channels, metric registry, composites, statistics.

This package holds the scientific core (the A-G metric registry, GA channels, and
the geometric Usability Score). Every function is a PURE offline function of the
canonical ``trace.jsonl`` plus the frozen task gold, and every numeric constant is
read from :mod:`usabench.eval.spec` (``usability_score.yaml``) -- nothing is
hardcoded.

The lightweight re-exports below are the package's public entrypoints. Heavier
leaves (the V1/V2/V3 scoring channels, the metric registry) are imported from
their submodules on demand to keep ``import usabench.eval`` cheap; they pull in no
optional provider deps either way.
"""

from __future__ import annotations

from usabench.eval.composite import CompositeInputs, CompositeResult, compute_composite
from usabench.eval.integrity import IntegrityFlags, compute_integrity
from usabench.eval.metrics import compute_all, registry
from usabench.eval.severity_audit import SeverityAuditResult, audit_severities
from usabench.eval.spec import get_severity_weights, load_spec

__all__ = [
    "load_spec",
    "get_severity_weights",
    "compute_all",
    "registry",
    "compute_composite",
    "CompositeInputs",
    "CompositeResult",
    "compute_integrity",
    "IntegrityFlags",
    "audit_severities",
    "SeverityAuditResult",
]
