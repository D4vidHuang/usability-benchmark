"""V1 -- functional / sandbox execution channel (``docs/scoring.md`` §3).

V1 is the deterministic, LLM-free backbone of Goal Achievement. It produces three
sub-signals and blends them with weights from ``usability_score.yaml``::

    V1 = v1_install*install_ok + v1_valid*valid_exec + v1_func*func_criteria

with the structural cascade ``install_ok=0 => valid_exec=0 => func_criteria=0``
(you cannot run what will not install). Because V1 is deterministic given the
artifact, it carries the highest trust weight in GA and anchors the fake-done
integrity check.

This module is the **pure scoring layer**. The *actual* sandbox build/install/
smoke/checker execution lives in the sandbox subtree (out of this agent's scope);
this layer either:

1. scores a :class:`CheckOutcome` set the sandbox already produced
   (:func:`score_v1_from_outcomes`), or
2. recovers V1 from the last ``verification_run`` event in the canonical trace
   (:func:`score_v1`), so the whole pipeline remains a pure function of
   ``trace.jsonl`` + gold.

A minimal, deterministic in-process :class:`SubprocessChecker` is provided so
functional checkers can be driven locally for tests without Docker; it shells out
with network left to the caller's environment (hermetic enforcement is the
sandbox subtree's job).
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from usabench.core.schema import TraceEnvelope
from usabench.eval._common import clip01, safe_div, spec_get, verification_runs
from usabench.eval.gold import as_gold

__all__ = [
    "CheckOutcome",
    "V1Result",
    "score_v1_from_outcomes",
    "score_v1",
    "SubprocessChecker",
    "v1_subsignal_weights",
]


@dataclass(frozen=True)
class CheckOutcome:
    """The result of one objective check (install / smoke / functional criterion).

    Attributes:
        check_id: Stable id (``"install"``, ``"smoke"``, or a criterion id).
        passed: Whether the check passed.
        kind: ``"install" | "smoke" | "functional"``.
        detail: Optional short diagnostic (e.g. exit code, stderr tail).
    """

    check_id: str
    passed: bool
    kind: str = "functional"
    detail: str | None = None


@dataclass(frozen=True)
class V1Result:
    """The V1 channel score plus its three sub-signals (all in ``[0,1]``)."""

    install_ok: float
    valid_exec: float
    func_criteria: float
    score: float
    n_functional: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


def v1_subsignal_weights() -> tuple[float, float, float]:
    """Return ``(install, valid_exec, func_criteria)`` weights from the spec."""
    w_install = float(spec_get("v1_subsignals", "install", default=0.25))
    w_valid = float(spec_get("v1_subsignals", "valid_exec", default=0.25))
    w_func = float(spec_get("v1_subsignals", "func_criteria", default=0.50))
    return w_install, w_valid, w_func


def _blend_v1(install_ok: float, valid_exec: float, func_criteria: float, n_func: int) -> V1Result:
    """Apply the cascade + weighted blend to the three sub-signals."""
    install_ok = clip01(install_ok)
    # Cascade: can't validly execute what didn't install; can't pass functional
    # criteria on something that doesn't run.
    valid_exec = clip01(valid_exec) if install_ok >= 1.0 else 0.0
    func_criteria = clip01(func_criteria) if valid_exec >= 1.0 else 0.0
    w_i, w_v, w_f = v1_subsignal_weights()
    score = clip01(w_i * install_ok + w_v * valid_exec + w_f * func_criteria)
    return V1Result(
        install_ok=install_ok,
        valid_exec=valid_exec,
        func_criteria=func_criteria,
        score=score,
        n_functional=n_func,
        detail={"weights": {"install": w_i, "valid_exec": w_v, "func_criteria": w_f}},
    )


def score_v1_from_outcomes(outcomes: Iterable[CheckOutcome]) -> V1Result:
    """Compute V1 from a set of objective :class:`CheckOutcome` records.

    Args:
        outcomes: The install, smoke, and functional-criterion check results the
            sandbox produced for one artifact.

    Returns:
        A :class:`V1Result` with the three sub-signals and the blended score.
    """
    outs = list(outcomes)
    install = [o for o in outs if o.kind == "install"]
    smoke = [o for o in outs if o.kind == "smoke"]
    func = [o for o in outs if o.kind == "functional"]

    install_ok = 1.0 if install and all(o.passed for o in install) else 0.0
    valid_exec = 1.0 if smoke and all(o.passed for o in smoke) else 0.0
    if func:
        func_criteria = safe_div(sum(1.0 for o in func if o.passed), float(len(func)))
    else:
        func_criteria = 0.0
    res = _blend_v1(install_ok, valid_exec, func_criteria, len(func))
    res.detail["outcomes"] = {o.check_id: o.passed for o in outs}
    return res


def score_v1(trace: Iterable[TraceEnvelope], gold: Any) -> V1Result:
    """Recover V1 from the last ``verification_run`` event in the trace.

    Functional criteria are those whose gold ``check_kind == 'func'``. ``install``
    and ``valid_exec`` sub-signals are read from the verification payload's
    ``must_have`` results carrying ids ``"install"`` / ``"smoke"`` if present,
    else inferred (``install_ok=1`` and ``valid_exec=1`` whenever any functional
    criterion was evaluated at all, which means the artifact ran).

    Args:
        trace: Parsed ``trace.jsonl`` events.
        gold: Task gold.

    Returns:
        A :class:`V1Result`.
    """
    g = as_gold(gold)
    runs = verification_runs(list(trace))
    if not runs:
        return _blend_v1(0.0, 0.0, 0.0, 0)
    payload = runs[-1].payload
    results: dict[str, Any] = {}
    for bucket in ("must_have", "should_have"):
        for cr in getattr(payload, bucket, []) or []:
            results[cr.id] = cr

    func_ids = [c.id for c in g.criteria_by_kind("func")]
    func_vals = [_passed01(results.get(cid)) for cid in func_ids if cid in results]

    install_cr = results.get("install")
    smoke_cr = results.get("smoke")
    ran_at_all = bool(func_vals) or bool(getattr(payload, "entrypoint", None))
    install_ok = _passed01(install_cr) if install_cr is not None else (1.0 if ran_at_all else 0.0)
    valid_exec = _passed01(smoke_cr) if smoke_cr is not None else (1.0 if ran_at_all else 0.0)
    func_criteria = safe_div(sum(func_vals), float(len(func_vals))) if func_vals else 0.0
    return _blend_v1(install_ok, valid_exec, func_criteria, len(func_vals))


def _passed01(cr: Any) -> float:
    """1.0 if a CriterionResult passed (score>=1 or passed True), else 0.0."""
    if cr is None:
        return 0.0
    score = getattr(cr, "score", None)
    if score is not None:
        return 1.0 if float(score) >= 1.0 else 0.0
    return 1.0 if getattr(cr, "passed", False) else 0.0


class SubprocessChecker:
    """A minimal deterministic checker that drives an artifact via a shell command.

    This exists so functional checker scripts can be exercised in unit tests
    without the full container sandbox. It is **not** the hermetic runner -- the
    sandbox subtree owns network isolation and resource caps. Here we only capture
    exit code / stdout for an assertion callback.
    """

    def __init__(self, *, timeout_s: float = 120.0, cwd: str | None = None) -> None:
        """Store run parameters.

        Args:
            timeout_s: Per-command wall timeout; a timeout counts as failure.
            cwd: Working directory for the command.
        """
        self.timeout_s = timeout_s
        self.cwd = cwd

    def run(self, cmd: Sequence[str]) -> tuple[int, str, str]:
        """Run ``cmd`` and return ``(exit_code, stdout, stderr)``.

        A timeout or OS error is reported as a non-zero exit with the error text
        on stderr (never raised), matching the 'timeout => check fails, never a
        harness crash' rule from ``docs/scoring.md`` §3.1.
        """
        try:
            proc = subprocess.run(
                list(cmd),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=self.cwd,
                check=False,
            )
            stdout = proc.stdout if isinstance(proc.stdout, str) else ""
            stderr = proc.stderr if isinstance(proc.stderr, str) else ""
            return proc.returncode, stdout, stderr
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout if isinstance(exc.stdout, str) else ""
            return 124, partial, f"timeout after {self.timeout_s}s"
        except OSError as exc:  # pragma: no cover - environment dependent
            return 127, "", str(exc)

    def check(self, check_id: str, cmd: Sequence[str], *, kind: str = "functional") -> CheckOutcome:
        """Run ``cmd`` and return a pass/fail :class:`CheckOutcome` on exit code 0."""
        code, _out, err = self.run(cmd)
        return CheckOutcome(
            check_id=check_id,
            passed=code == 0,
            kind=kind,
            detail=None if code == 0 else f"exit={code} {err[:200]}",
        )
