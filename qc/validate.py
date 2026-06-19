"""Task QC stage 1: schema validation + visibility-partition lint + SHA pinning.

This is the first gate in the curation pipeline (``docs/tasks.md`` §8.1). For a
finished ``task.json`` it asserts three independent invariants:

1. **Schema validity** -- the record validates against
   ``schemas/task.schema.json`` AND round-trips through the authoritative pydantic
   model :class:`usabench.core.schema.Task` (the schema and the model are kept in
   lock-step by the foundation tests; we check both so a drift is caught here).

2. **Visibility partition** (the load-bearing one) -- the agent-visible projection
   (:meth:`Task.agent_view`) must contain NO gold/hidden content. We diff the
   serialized agent view against the gold fields (``hidden``, ``reference_repos``,
   per-ambiguity ``gold``) and fail if any gold string leaks into a field the agent
   can see. This is the structural two-tier guarantee of ``docs/tasks.md`` §2.1.

3. **Pinned SHAs** -- every ``reference_repos[].commit`` must be a real pinned
   commit SHA, never a floating ref (``main``/``HEAD``/``latest`` or an empty
   string) (``docs/tasks.md`` §2.1).

It also lints the suitability pre-conditions that are cheaply checkable here
(``docs/tasks.md`` §8.2): >=2 ambiguity points with >=1 high-severity, >=1
reference repo, and the ">=80% of acceptance-criterion weight is auto-checkable"
rule (``check_kind in {func, rubric_auto}``).

Pure offline: depends only on ``jsonschema`` (a core dep), the foundation models,
and the JSON schemas under ``schemas/``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from usabench.core.errors import SchemaViolation
from usabench.core.schema import Task

__all__ = [
    "ValidationReport",
    "FLOATING_REFS",
    "SHA_RE",
    "load_task_schema",
    "validate_task_dict",
    "validate_task_file",
    "lint_visibility_partition",
    "check_pinned_shas",
    "lint_suitability",
]

log = structlog.get_logger(__name__)

#: Refs we reject as un-pinned (a commit SHA is required instead).
FLOATING_REFS: frozenset[str] = frozenset({"", "main", "master", "head", "latest", "trunk", "develop"})
#: A 7-40 char lowercase hex commit SHA (abbreviated or full).
SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

#: Gold/oracle-private top-level keys that must NEVER appear in the agent view.
_GOLD_TOP_KEYS: frozenset[str] = frozenset({"hidden", "reference_repos", "expected_interventions"})


@dataclass(slots=True)
class ValidationReport:
    """The accumulated outcome of validating one task.

    Attributes:
        task_id: The task id (``"<unknown>"`` if it could not be read).
        errors: Hard failures that block promotion.
        warnings: Soft issues (suitability heuristics that merely flag).
        checks: Per-check pass/fail map (for structured reporting).
    """

    task_id: str = "<unknown>"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if there are no hard errors."""
        return not self.errors

    def add_error(self, check: str, message: str) -> None:
        """Record a failed check and its message."""
        self.checks[check] = False
        self.errors.append(f"[{check}] {message}")

    def add_pass(self, check: str) -> None:
        """Record a passed check (idempotent; does not overwrite a prior failure)."""
        self.checks.setdefault(check, True)

    def add_warning(self, message: str) -> None:
        """Record a soft warning."""
        self.warnings.append(message)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the report to a plain dict."""
        return {
            "task_id": self.task_id,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": dict(self.checks),
        }


def _default_schema_path() -> Path:
    """Locate ``schemas/task.schema.json`` relative to the repo root."""
    here = Path(__file__).resolve()
    # qc/validate.py -> repo root is two levels up.
    return here.parent.parent / "schemas" / "task.schema.json"


def load_task_schema(path: str | Path | None = None) -> dict[str, Any]:
    """Load the ``task.schema.json`` document.

    Args:
        path: Optional explicit schema path; defaults to the repo ``schemas/`` copy.

    Returns:
        The parsed JSON schema document.

    Raises:
        FileNotFoundError: If the schema file is missing.
    """
    p = Path(path) if path else _default_schema_path()
    if not p.is_file():
        raise FileNotFoundError(f"task schema not found: {p}")
    doc: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return doc


# --------------------------------------------------------------------------- #
# Individual checks                                                            #
# --------------------------------------------------------------------------- #


def check_pinned_shas(task: dict[str, Any], report: ValidationReport) -> None:
    """Assert every reference-repo commit is a pinned SHA (no floating refs).

    Args:
        task: A task dict.
        report: The report to record results on.
    """
    refs = task.get("reference_repos") or []
    bad: list[str] = []
    for ref in refs:
        commit = str((ref or {}).get("commit") or "").strip().lower()
        if commit in FLOATING_REFS or not SHA_RE.match(commit):
            bad.append(f"{(ref or {}).get('url', '?')} -> {commit!r}")
    if bad:
        report.add_error("pinned_shas", "un-pinned reference commits: " + "; ".join(bad))
    else:
        report.add_pass("pinned_shas")


def lint_visibility_partition(task_model: Task, report: ValidationReport) -> None:
    """Assert no gold/hidden string leaks into the agent-visible projection.

    The agent view is serialized and scanned: if any non-trivial gold string
    (hidden summary, ambiguity ``gold``, reference-repo url/why, info-unit desc)
    appears verbatim inside it, the partition is broken. We also assert the agent
    view simply has no top-level gold keys at all.

    Args:
        task_model: A validated :class:`Task`.
        report: The report to record results on.
    """
    view = task_model.agent_view()
    view_dump = view.model_dump(mode="json")

    # Structural: no gold top-level keys may exist on the view object.
    leaked_keys = _GOLD_TOP_KEYS & set(view_dump.keys())
    if leaked_keys:
        report.add_error("visibility_keys", f"agent view exposes gold keys: {sorted(leaked_keys)}")
    else:
        report.add_pass("visibility_keys")

    view_blob = json.dumps(view_dump, ensure_ascii=False).lower()

    # Content: collect gold strings and ensure none appear verbatim in the view.
    gold_strings: list[str] = []
    hidden = task_model.hidden
    if hidden.summary and not hidden.summary.lower().startswith("todo"):
        gold_strings.append(hidden.summary)
    for ap in hidden.ambiguity_points:
        if ap.gold and not ap.gold.lower().startswith("todo"):
            gold_strings.append(ap.gold)
    for ac in hidden.acceptance_criteria:
        gold_strings.append(ac.text)
    for iu in hidden.info_units:
        gold_strings.append(iu.desc)
    for rr in task_model.reference_repos:
        gold_strings.append(rr.url)
        if rr.why:
            gold_strings.append(rr.why)

    leaks: list[str] = []
    for gold in gold_strings:
        needle = (gold or "").strip().lower()
        # Only flag substantive strings to avoid false positives on short tokens.
        if len(needle) >= 12 and needle in view_blob:
            leaks.append(gold[:60])
    if leaks:
        report.add_error("visibility_content", "gold leaked into agent view: " + "; ".join(leaks))
    else:
        report.add_pass("visibility_content")


def lint_suitability(task: dict[str, Any], report: ValidationReport) -> None:
    """Lint the cheaply-checkable suitability pre-conditions (``docs/tasks.md`` §8.2).

    These are *warnings* on a draft (gold may be absent) and become hard
    requirements only on a finished task; we record them as warnings so the report
    stays usable across pipeline stages.

    Args:
        task: A task dict.
        report: The report to record results on.
    """
    hidden = task.get("hidden") or {}
    aps = hidden.get("ambiguity_points") or []
    n_high = sum(1 for ap in aps if (ap or {}).get("severity") == "high")
    if len(aps) < 2:
        report.add_warning(f"suitability: only {len(aps)} ambiguity points (<2)")
    if n_high < 1:
        report.add_warning("suitability: no high-severity ambiguity point")
    if not (task.get("reference_repos") or []):
        report.add_warning("suitability: no reference_repos pinned (gold ungrounded)")

    acs = hidden.get("acceptance_criteria") or []
    if acs:
        total_w = sum(float((ac or {}).get("weight", 1.0)) for ac in acs)
        auto_w = sum(
            float((ac or {}).get("weight", 1.0))
            for ac in acs
            if (ac or {}).get("check_kind") in {"func", "rubric_auto"}
        )
        if total_w > 0 and auto_w / total_w < 0.80:
            report.add_warning(
                f"suitability: only {auto_w / total_w:.0%} of AC weight is auto-checkable (<80%)"
            )
    else:
        report.add_warning("suitability: no acceptance_criteria authored yet (draft)")


# --------------------------------------------------------------------------- #
# Top-level validation                                                         #
# --------------------------------------------------------------------------- #


def validate_task_dict(
    task: dict[str, Any],
    *,
    schema: dict[str, Any] | None = None,
    require_gold: bool = False,
) -> ValidationReport:
    """Validate a single task dict and return a structured report.

    Runs: JSON-Schema validation, pydantic round-trip, visibility-partition lint,
    pinned-SHA check, and the suitability lint.

    Args:
        task: A task record dict.
        schema: Optional pre-loaded ``task.schema.json`` (loaded on demand if None).
        require_gold: If True, missing gold (placeholder ``TODO`` summary, empty
            acceptance criteria) is promoted from a warning to an error -- used to
            gate a *finished* task vs. an early *draft*.

    Returns:
        A :class:`ValidationReport`.
    """
    import jsonschema

    report = ValidationReport(task_id=str(task.get("id") or "<unknown>"))
    schema_doc = schema or load_task_schema()

    # 1. JSON-Schema.
    validator = jsonschema.Draft202012Validator(schema_doc)
    errs = sorted(validator.iter_errors(task), key=lambda e: list(e.path))
    if errs:
        for e in errs[:10]:
            loc = "/".join(str(p) for p in e.path) or "<root>"
            report.add_error("json_schema", f"{loc}: {e.message}")
    else:
        report.add_pass("json_schema")

    # 2. Pydantic round-trip (authoritative model). Only attempt if schema passed
    #    structurally enough to construct (the model is stricter on some fields).
    task_model: Task | None = None
    try:
        task_model = Task.model_validate(task)
        report.add_pass("pydantic_model")
    except Exception as exc:  # noqa: BLE001 - surface as a structured error
        report.add_error("pydantic_model", str(exc).splitlines()[0])

    # 3. Pinned SHAs (dict-level; independent of model validity).
    check_pinned_shas(task, report)

    # 4. Visibility partition (needs the model).
    if task_model is not None:
        lint_visibility_partition(task_model, report)

    # 5. Suitability.
    lint_suitability(task, report)

    # 6. Gold-presence gate for finished tasks.
    if require_gold:
        hidden = task.get("hidden") or {}
        summary = str(hidden.get("summary") or "")
        if not summary or summary.lower().startswith("todo"):
            report.add_error("gold_present", "hidden.summary is a placeholder/TODO")
        if not (hidden.get("acceptance_criteria") or []):
            report.add_error("gold_present", "no acceptance_criteria authored")
        if not (task.get("reference_repos") or []):
            report.add_error("gold_present", "no reference_repos pinned")

    return report


def validate_task_file(
    path: str | Path,
    *,
    schema: dict[str, Any] | None = None,
    require_gold: bool = False,
) -> ValidationReport:
    """Load and validate a ``task.json`` file.

    Args:
        path: Path to a ``task.json``.
        schema: Optional pre-loaded schema.
        require_gold: Whether to require finished gold (see :func:`validate_task_dict`).

    Returns:
        A :class:`ValidationReport`.

    Raises:
        SchemaViolation: If the file cannot be read or parsed as JSON.
    """
    p = Path(path)
    try:
        task = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaViolation(f"cannot read task {p}: {exc}") from exc
    return validate_task_dict(task, schema=schema, require_gold=require_gold)
