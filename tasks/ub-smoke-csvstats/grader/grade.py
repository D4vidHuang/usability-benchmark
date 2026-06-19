#!/usr/bin/env python3
"""Deterministic grader for the smoke task ``ub-smoke-csvstats``.

Contract (identical to ``tasks/ub-cal-0007/grader/grade.py``)::

    python grade.py --artifact <agent_tool.py>

The grader builds a tiny self-contained CSV fixture (with a blank and a
non-numeric cell in the target column), runs the agent artifact selecting a column
by header name, and emits a per-criterion JSON report to stdout::

    {"task_id", "criteria":[{"id","passed","score","detail",...}],
     "weighted_score", "accepted"}

Exit code 0 means the *grading run* succeeded; nonzero means the grader itself
failed. Stdlib only; hermetic; no network.

The expected tool reads a CSV with a header row, selects a numeric column by its
header name (passed as the second positional arg or via ``--column``), skips
blank/non-numeric cells, and prints count, min, max, and mean for that column.
Reported numbers are scraped tolerantly by label.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TASK_ID = "ub-smoke-csvstats"

DEFAULT_WEIGHTS = {"AC1": 1.0, "AC2": 1.5, "AC3": 2.0, "AC4": 1.0}

# Fixture: column "value" has a blank and a non-numeric cell that must be skipped.
# Numeric values: 10, 20, 30, 40  -> count=4 min=10 max=40 mean=25.
FIXTURE_CSV = (
    "name,value,other\n"
    "a,10,x\n"
    "b,20,y\n"
    "c,,z\n"          # blank -> skipped
    "d,notnum,w\n"    # non-numeric -> skipped
    "e,30,v\n"
    "f,40,u\n"
)
TARGET_COLUMN = "value"
GOLD = {"count": 4, "min": 10.0, "max": 40.0, "mean": 25.0}
# A control column to prove the tool selects by name (not a fixed column).
OTHER_COLUMN = "other"  # non-numeric -> count of numerics is 0


def _run_artifact(artifact: Path, args: list[str], timeout: float = 30.0) -> tuple[int, str, str]:
    """Run ``python <artifact> <args>`` and capture (exit_code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            [sys.executable, str(artifact), *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except OSError as exc:  # pragma: no cover - defensive
        return 125, "", f"{type(exc).__name__}: {exc}"


def _run_column(artifact: Path, csv_path: Path, column: str) -> tuple[int, str, str]:
    """Invoke the tool selecting ``column`` (positional first, then ``--column``)."""
    code, out, err = _run_artifact(artifact, [str(csv_path), column])
    if code != 0 or not out.strip():
        code, out, err = _run_artifact(artifact, [str(csv_path), "--column", column])
    if code != 0 or not out.strip():
        code, out, err = _run_artifact(artifact, ["--column", column, str(csv_path)])
    return code, out, err


def _scrape(stdout: str, *labels: str) -> float | None:
    """First number on a line mentioning any of ``labels`` (case-insensitive)."""
    for line in stdout.splitlines():
        low = line.lower()
        if any(lbl in low for lbl in labels):
            m = re.search(r"(-?\d+(?:\.\d+)?)", line.replace(",", ""))
            if m:
                return float(m.group(1))
    return None


def _close(got: float | None, gold: float, tol: float = 1e-6) -> bool:
    return got is not None and abs(got - gold) <= tol


def _criterion(cid: str, passed: bool, detail: str, weight: float) -> dict[str, Any]:
    return {
        "id": cid, "passed": bool(passed), "score": 1.0 if passed else 0.0,
        "weight": weight, "channel": "func", "detail": detail,
    }


def grade(artifact: Path, weights: dict[str, float]) -> dict[str, Any]:
    """Run every check and return the full per-criterion report."""
    criteria: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "data.csv"
        csv_path.write_text(FIXTURE_CSV, encoding="utf-8")

        code, out, err = _run_column(artifact, csv_path, TARGET_COLUMN)

        # AC1: runs without crashing on a well-formed CSV with a header.
        ran_ok = code == 0 and bool(out.strip())
        criteria.append(_criterion(
            "AC1", ran_ok,
            f"exit={code}" + (f" stderr={err[:160]!r}" if not ran_ok else ""),
            weights["AC1"],
        ))

        got_count = _scrape(out, "count", "n=", "number of", "rows")
        got_min = _scrape(out, "min")
        got_max = _scrape(out, "max")
        got_mean = _scrape(out, "mean", "average", "avg")

        # AC3: count, min, max, mean correct (with the bad cells skipped).
        stats_ok = (
            _close(got_count, GOLD["count"]) and _close(got_min, GOLD["min"])
            and _close(got_max, GOLD["max"]) and _close(got_mean, GOLD["mean"])
        )
        ac3 = _criterion(
            "AC3", stats_ok,
            f"count={got_count} min={got_min} max={got_max} mean={got_mean} gold={GOLD}",
            weights["AC3"],
        )

        # AC2: selects by header NAME. Selecting "other" (non-numeric) must yield a
        # different result than "value" -- proving it is not hard-coded to a column.
        oc, oout, _oe = _run_column(artifact, csv_path, OTHER_COLUMN)
        other_mean = _scrape(oout, "mean", "average", "avg") if oc == 0 else None
        # The two columns must not report the same mean; "value" must match gold.
        by_name_ok = _close(got_mean, GOLD["mean"]) and (
            other_mean is None or not _close(other_mean, GOLD["mean"])
        )
        criteria.append(_criterion(
            "AC2", by_name_ok,
            f"value_mean={got_mean} other_mean={other_mean} (must differ)",
            weights["AC2"],
        ))
        criteria.append(ac3)

        # AC4: skips blank/non-numeric cells rather than crashing. Count==4 (not 6)
        # AND the run did not crash proves the bad cells were skipped, not fatal.
        skip_ok = ran_ok and _close(got_count, GOLD["count"])
        criteria.append(_criterion(
            "AC4", skip_ok,
            f"count={got_count} (gold 4; 2 bad cells skipped) ran_ok={ran_ok}",
            weights["AC4"],
        ))

    total_w = sum(weights[c["id"]] for c in criteria)
    earned = sum(c["weight"] * c["score"] for c in criteria)
    weighted = earned / total_w if total_w else 0.0
    return {
        "task_id": TASK_ID,
        "criteria": criteria,
        "weighted_score": round(weighted, 6),
        "accepted": weighted >= 0.80,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Grader for {TASK_ID}.")
    parser.add_argument("--artifact", type=Path, required=True,
                        help="Path to the agent's CLI tool (a .py file).")
    args = parser.parse_args(argv)
    if not args.artifact.exists():
        print(json.dumps({"error": f"artifact not found: {args.artifact}"}), file=sys.stderr)
        return 2
    report = grade(args.artifact, dict(DEFAULT_WEIGHTS))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
