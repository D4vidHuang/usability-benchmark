"""The concrete acceptance verifier wired into the episode runner.

:class:`FunctionalVerifier` implements the harness ``VerifierLike`` protocol
(:class:`usabench.harness.runner.VerifierLike`): on every ``declare_done`` (and on
forced-final verification) the runner calls :meth:`FunctionalVerifier.verify`,
which grades the agent's *delivered artifact* in the sandbox workspace and returns
a :class:`~usabench.core.schema.VerificationRun`. The runner reads
``vrun.must_have`` / ``vrun.all_must_pass`` / ``vrun.rubric_score`` to accept or
reject (``runner._do_declare_done``).

Grading reuses the established grader contract (``tasks/<id>/grader/grade.py``;
see ``tasks/ub-cal-0007/grader/grade.py``): ``python grade.py --artifact
<artifact>`` runs the artifact on a hermetic fixture and prints a JSON report to
stdout::

    {"task_id", "criteria":[{"id","passed","score","detail",...}],
     "weighted_score", "accepted"}

with exit code 0 meaning "grading ran" (pass/fail lives in the JSON) and nonzero
meaning the grader itself failed. This module is the *only* place that knows how
to locate and invoke that grader; the harness stays free of any eval dependency.

The verifier is deliberately robust: a missing grader, a crashed grader, or a
malformed report degrades to a non-accepting (but non-crashing)
:class:`VerificationRun` so a single bad task can never abort an episode. The
authoritative criterion *weights*, *ids*, ``is_core`` and ``is_hard`` flags come
from the task's frozen ``hidden.acceptance_criteria`` -- the grader's own report
supplies only per-criterion pass/score, never the gold weighting.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from usabench.core.ids import sha256_hex
from usabench.core.schema import CriterionResult, Task, VerificationRun
from usabench.eval.gold import as_gold
from usabench.harness.sandbox import SandboxBackend
from usabench.logging_setup import get_logger

__all__ = ["FunctionalVerifier"]

_log = get_logger(__name__)

#: Default artifact filename when ``declare_done`` does not name an entrypoint.
_DEFAULT_ENTRYPOINT = "main.py"
#: Wall-clock cap on the grader subprocess (seconds).
_GRADER_TIMEOUT_S = 90
#: Extracts a ``*.py`` path token from a declared entrypoint, which agents often
#: phrase as a whole command ("python word_freq.py <file>") rather than a filename.
_PY_TOKEN_RE = re.compile(r"[\w./-]+\.py\b")
#: Workspace dirs that never hold the agent's deliverable.
_IGNORE_DIR_PARTS = {"__pycache__", ".git", "grader", ".venv", "node_modules"}


class FunctionalVerifier:
    """Grade the agent's artifact via the task's ``grader/grade.py`` (stdlib-only).

    The verifier is constructed once per run with the *taskset root* (the ``tasks/``
    directory that contains one ``<task.id>/grader/grade.py`` per gradable task).
    On each :meth:`verify` it resolves the artifact in the sandbox workspace, shells
    out to the grader with the workspace as the working directory, parses the JSON
    report, and folds it into a :class:`VerificationRun` keyed by the task's
    authoritative acceptance criteria.
    """

    def __init__(self, tasks_root: str | Path) -> None:
        """Store the taskset root used to locate per-task graders.

        Args:
            tasks_root: Path to the directory holding ``<task.id>/grader/grade.py``
                subtrees (typically the repo's ``tasks/`` directory).
        """
        self.tasks_root = Path(tasks_root)

    # -- VerifierLike ------------------------------------------------------- #

    def verify(
        self,
        task: Task,
        sandbox: SandboxBackend,
        *,
        trigger: str,
        entrypoint: str | None,
    ) -> VerificationRun:
        """Grade the workspace artifact and return a :class:`VerificationRun`.

        Args:
            task: The frozen task (supplies the authoritative acceptance criteria).
            sandbox: The set-up sandbox whose workspace holds the agent's files.
            trigger: ``"submit"`` (a ``declare_done``) or ``"forced_final"``.
            entrypoint: The agent-declared entrypoint, or ``None`` (defaults to
                ``main.py``).

        Returns:
            A :class:`VerificationRun`. If no grader exists, or grading fails to
            run/parse, the run *defers*: ``all_must_pass`` reflects "no functional
            criteria evaluated" rather than crashing the episode.
        """
        trig = "forced_final" if trigger == "forced_final" else "submit"
        workspace = Path(sandbox.workspace)
        grader = self._grader_path(task)
        artifact = self._resolve_artifact(workspace, entrypoint)
        artifact_name = artifact.name if artifact is not None else (entrypoint or _DEFAULT_ENTRYPOINT)

        if grader is None or not grader.is_file():
            _log.info("verifier.no_grader", task_id=task.id, tasks_root=str(self.tasks_root))
            return self._deferred_run(task, trig, artifact_name)

        if artifact is None:
            # The agent never wrote a runnable Python artifact: every criterion fails.
            return self._failed_run(task, trig, artifact_name, detail="no python artifact found")

        report, wall_ms = self._run_grader(grader, artifact, cwd=workspace)
        if report is None:
            # Grader itself failed (nonzero exit / unparseable). Treat as a reject.
            return self._failed_run(task, trig, artifact_name, detail="grader failed to run")

        return self._run_from_report(task, trig, artifact_name, report, wall_ms)

    # -- grader location + invocation -------------------------------------- #

    def _grader_path(self, task: Task) -> Path | None:
        """Resolve ``<tasks_root>/<task.id>/grader/grade.py`` if it exists.

        Returns an ABSOLUTE path: the grader subprocess runs with the sandbox
        workspace as ``cwd``, so a relative grader path would resolve against the
        workspace (where it does not exist) and fail to open.
        """
        candidate = self.tasks_root / task.id / "grader" / "grade.py"
        return candidate.resolve() if candidate.is_file() else None

    def _resolve_artifact(self, workspace: Path, entrypoint: str | None) -> Path | None:
        """Resolve the agent's deliverable Python file inside the workspace.

        Open-ended tasks do not dictate a filename, so agents name their tool
        freely ("word_freq.py") and often declare the entrypoint as a whole command
        ("python word_freq.py <file>"). We therefore resolve in order:

        1. any ``*.py`` token parsed out of the declared ``entrypoint`` that exists;
        2. the conventional ``main.py``;
        3. discovery -- the single ``*.py`` the agent wrote (excluding fixtures /
           caches / the grader); with several, prefer one whose stem appears in the
           declared entrypoint, else the largest.

        Returns the absolute path (guaranteed under ``workspace``) or ``None``.
        """
        root = workspace.resolve()

        def _under(name: str) -> Path | None:
            cand = (root / name.lstrip("./")).resolve()
            try:
                cand.relative_to(root)
            except ValueError:
                return None
            return cand if cand.is_file() else None

        named: list[str] = list(_PY_TOKEN_RE.findall(entrypoint)) if entrypoint else []
        for name in [*named, _DEFAULT_ENTRYPOINT]:
            hit = _under(name)
            if hit is not None:
                return hit

        discovered = sorted(
            p.resolve()
            for p in root.rglob("*.py")
            if p.is_file() and not (_IGNORE_DIR_PARTS & set(p.parts))
        )
        if not discovered:
            return None
        if len(discovered) == 1:
            return discovered[0]
        if entrypoint:
            for p in discovered:
                if p.stem and p.stem in entrypoint:
                    return p
        return max(discovered, key=lambda p: p.stat().st_size)

    def _run_grader(
        self, grader: Path, artifact: Path, *, cwd: Path
    ) -> tuple[dict[str, Any] | None, int]:
        """Run the grader subprocess; return ``(report_dict|None, wall_ms)``.

        The grader is invoked with the same interpreter running the harness and
        the sandbox workspace as ``cwd``. A nonzero exit, a timeout, or output that
        does not parse into a JSON object yields ``(None, wall_ms)`` -- the caller
        turns that into a non-accepting run.
        """
        import time as _time

        start = _time.monotonic()
        try:
            proc = subprocess.run(  # noqa: S603 - argv list, no shell, hermetic
                [sys.executable, str(grader), "--artifact", str(artifact)],
                capture_output=True,
                text=True,
                timeout=_GRADER_TIMEOUT_S,
                cwd=str(cwd),
                check=False,
            )
        except subprocess.TimeoutExpired:
            wall_ms = int((_time.monotonic() - start) * 1000)
            _log.warning("verifier.grader_timeout", grader=str(grader))
            return None, wall_ms
        except OSError as exc:  # pragma: no cover - environment dependent
            wall_ms = int((_time.monotonic() - start) * 1000)
            _log.warning("verifier.grader_oserror", grader=str(grader), error=str(exc))
            return None, wall_ms
        wall_ms = int((_time.monotonic() - start) * 1000)

        if proc.returncode != 0:
            _log.warning(
                "verifier.grader_nonzero",
                grader=str(grader),
                exit_code=proc.returncode,
                stderr=proc.stderr[-300:],
            )
            return None, wall_ms

        report = self._parse_report(proc.stdout)
        if report is None:
            _log.warning("verifier.grader_unparseable", grader=str(grader))
        return report, wall_ms

    @staticmethod
    def _parse_report(stdout: str) -> dict[str, Any] | None:
        """Parse the grader's JSON report object from stdout (tolerant of preamble)."""
        text = stdout.strip()
        if not text:
            return None
        # The contract is a single JSON object on stdout; accept a trailing object
        # even if the grader prefixed diagnostics by scanning for the last {...}.
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    # -- report -> VerificationRun ----------------------------------------- #

    def _run_from_report(
        self,
        task: Task,
        trigger: str,
        entrypoint: str,
        report: dict[str, Any],
        wall_ms: int,
    ) -> VerificationRun:
        """Fold a parsed grader report into a :class:`VerificationRun`.

        Per-criterion pass/score is taken from the report; the *partitioning* into
        ``must_have`` (core/hard) vs ``should_have`` and the ``rubric_score`` use
        the task's authoritative criteria and weights, never the grader's own
        (which are only a standalone convenience).
        """
        g = as_gold(task)
        by_id = self._report_results_by_id(report)

        must_have: list[CriterionResult] = []
        should_have: list[CriterionResult] = []
        earned = 0.0
        total_w = 0.0
        for crit in g.criteria:
            cr = by_id.get(crit.id) or CriterionResult(id=crit.id, passed=None, score=None)
            value = _criterion_value(cr)
            total_w += float(crit.weight)
            earned += float(crit.weight) * value
            if crit.is_core or crit.is_hard:
                must_have.append(cr)
            else:
                should_have.append(cr)

        all_must_pass = bool(must_have) and all(_passed(cr) for cr in must_have)
        rubric_score = round(earned / total_w, 6) if total_w > 0 else 0.0

        return VerificationRun(
            trigger=trigger,
            entrypoint=entrypoint,
            must_have=must_have,
            should_have=should_have,
            all_must_pass=all_must_pass,
            rubric_score=min(1.0, max(0.0, rubric_score)),
            wall_ms=wall_ms,
        )

    @staticmethod
    def _report_results_by_id(report: dict[str, Any]) -> dict[str, CriterionResult]:
        """Map criterion id -> CriterionResult from a grader report's ``criteria``."""
        out: dict[str, CriterionResult] = {}
        for raw in report.get("criteria", []) or []:
            if not isinstance(raw, dict):
                continue
            cid = raw.get("id")
            if not isinstance(cid, str):
                continue
            passed = raw.get("passed")
            score = raw.get("score")
            detail = raw.get("detail")
            out[cid] = CriterionResult(
                id=cid,
                passed=bool(passed) if passed is not None else None,
                score=_clip_score(score),
                detail_sha256=sha256_hex(str(detail)) if detail else None,
                channel="func",
            )
        return out

    # -- degraded / non-accepting runs ------------------------------------- #

    def _deferred_run(self, task: Task, trigger: str, entrypoint: str) -> VerificationRun:
        """A run for a task with no grader: no func criteria evaluated, defer.

        ``all_must_pass`` is ``False`` only if there ARE must-have criteria that
        went unevaluated; for a task with no gradable criteria it stays ``False``
        and the runner falls through to the oracle review.
        """
        g = as_gold(task)
        must_have = [
            CriterionResult(id=c.id, passed=None, score=None, channel="func")
            for c in g.criteria
            if c.is_core or c.is_hard
        ]
        should_have = [
            CriterionResult(id=c.id, passed=None, score=None, channel="func")
            for c in g.criteria
            if not (c.is_core or c.is_hard)
        ]
        return VerificationRun(
            trigger=trigger,
            entrypoint=entrypoint,
            must_have=must_have,
            should_have=should_have,
            all_must_pass=False,
            rubric_score=0.0,
        )

    def _failed_run(
        self, task: Task, trigger: str, entrypoint: str, *, detail: str
    ) -> VerificationRun:
        """A run where every criterion fails (artifact missing or grader broke)."""
        g = as_gold(task)
        detail_hash = sha256_hex(detail)
        must_have = [
            CriterionResult(id=c.id, passed=False, score=0.0, detail_sha256=detail_hash, channel="func")
            for c in g.criteria
            if c.is_core or c.is_hard
        ]
        should_have = [
            CriterionResult(id=c.id, passed=False, score=0.0, detail_sha256=detail_hash, channel="func")
            for c in g.criteria
            if not (c.is_core or c.is_hard)
        ]
        return VerificationRun(
            trigger=trigger,
            entrypoint=entrypoint,
            must_have=must_have,
            should_have=should_have,
            all_must_pass=False,
            rubric_score=0.0,
        )


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #


def _clip_score(score: Any) -> float | None:
    """Coerce a report score into ``[0,1]`` or ``None`` if absent/uncoercible."""
    if score is None:
        return None
    try:
        f = float(score)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, f))


def _criterion_value(cr: CriterionResult) -> float:
    """Score a CriterionResult in ``[0,1]`` (``score`` preferred, else ``passed``)."""
    if cr.score is not None:
        return min(1.0, max(0.0, float(cr.score)))
    return 1.0 if cr.passed else 0.0


def _passed(cr: CriterionResult) -> bool:
    """A must-have criterion counts as passing iff value reaches 1.0."""
    return _criterion_value(cr) >= 1.0
