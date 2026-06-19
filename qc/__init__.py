"""Task quality-control package (schema validation, grader sanity, calibration).

The curation gates of ``docs/tasks.md`` §8, one module per stage:

* :mod:`qc.validate` -- schema validation + visibility-partition lint + SHA pinning.
* :mod:`qc.grader_sanity` -- proves the grader discriminates (good/bad/trap probes).
* :mod:`qc.calibrate` -- intervention calibration + the discriminative gate.
* :mod:`qc.diversity` -- embedding near-twin dedup + domain/tier quotas.

Depends only on :mod:`usabench.core`, the JSON schemas under ``schemas/``, and the
frozen spec in :mod:`usabench.eval.spec`. Submodules are imported on demand so
``import qc`` stays cheap and free of optional deps.
"""

from __future__ import annotations

__all__ = [
    "validate_task_dict",
    "validate_task_file",
    "ValidationReport",
    "run_grader_sanity",
    "GraderSanityResult",
    "calibrate",
    "summarize_trace",
    "CalibrationResult",
    "TrialSummary",
    "run_diversity",
    "dedup_tasks",
    "quota_report",
    "DiversityReport",
]

_MODULE_BY_NAME = {
    "validate_task_dict": "validate",
    "validate_task_file": "validate",
    "ValidationReport": "validate",
    "run_grader_sanity": "grader_sanity",
    "GraderSanityResult": "grader_sanity",
    "calibrate": "calibrate",
    "summarize_trace": "calibrate",
    "CalibrationResult": "calibrate",
    "TrialSummary": "calibrate",
    "run_diversity": "diversity",
    "dedup_tasks": "diversity",
    "quota_report": "diversity",
    "DiversityReport": "diversity",
}


def __getattr__(name: str) -> object:  # PEP 562 lazy attribute access
    """Lazily resolve the package's public API without eager submodule imports."""
    mod = _MODULE_BY_NAME.get(name)
    if mod is not None:
        import importlib

        return getattr(importlib.import_module(f"qc.{mod}"), name)
    raise AttributeError(f"module 'qc' has no attribute {name!r}")
