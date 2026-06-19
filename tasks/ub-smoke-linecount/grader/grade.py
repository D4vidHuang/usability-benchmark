#!/usr/bin/env python3
"""Deterministic grader for the smoke task ``ub-smoke-linecount``.

Contract (identical to ``tasks/ub-cal-0007/grader/grade.py``)::

    python grade.py --artifact <agent_tool.py>

The grader builds a tiny self-contained UTF-8 fixture with a known number of
lines, words, and characters, runs the agent artifact on it, and emits a
per-criterion JSON report to stdout::

    {"task_id", "criteria":[{"id","passed","score","detail",...}],
     "weighted_score", "accepted"}

Exit code 0 means the *grading run* succeeded; nonzero means the grader itself
failed. Stdlib only; hermetic; no network.

The expected tool is a tiny ``wc``: read a text file (positional arg) or stdin and
print the number of lines, words, and characters (``wc`` semantics: lines are
newline-terminated). Reported numbers are scraped tolerantly by label.
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

TASK_ID = "ub-smoke-linecount"

DEFAULT_WEIGHTS = {"AC1": 1.0, "AC2": 1.0, "AC3": 1.0, "AC4": 1.0}

# Fixture with three newline-terminated lines. wc semantics:
#   lines = number of '\n' = 3
#   words = whitespace-separated tokens = 9
#   chars = total bytes/characters including newlines = len(text)
FIXTURE_TEXT = "the quick brown\nfox jumps over\nthe lazy dog\n"
GOLD = {
    "lines": FIXTURE_TEXT.count("\n"),          # 3
    "words": len(FIXTURE_TEXT.split()),         # 9
    "chars": len(FIXTURE_TEXT),                 # 44 (incl. the 3 newlines)
}


def _run_artifact(artifact: Path, args: list[str], *, stdin: str | None = None,
                  timeout: float = 30.0) -> tuple[int, str, str]:
    """Run ``python <artifact> <args>`` and capture (exit_code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            [sys.executable, str(artifact), *args],
            capture_output=True, text=True, timeout=timeout, input=stdin,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except OSError as exc:  # pragma: no cover - defensive
        return 125, "", f"{type(exc).__name__}: {exc}"


def _scrape(stdout: str, *labels: str) -> int | None:
    """First integer on a line mentioning any of ``labels`` (case-insensitive)."""
    for line in stdout.splitlines():
        low = line.lower()
        if any(lbl in low for lbl in labels):
            m = re.search(r"(\d+)", line.replace(",", ""))
            if m:
                return int(m.group(1))
    return None


def _scrape_positional(stdout: str) -> list[int]:
    """Fallback: the integers on the first line carrying >=3 numbers (wc-style)."""
    for line in stdout.splitlines():
        nums = [int(n) for n in re.findall(r"\d+", line.replace(",", ""))]
        if len(nums) >= 3:
            return nums
    nums = [int(n) for n in re.findall(r"\d+", stdout.replace(",", ""))]
    return nums


def _criterion(cid: str, passed: bool, detail: str, weight: float) -> dict[str, Any]:
    return {
        "id": cid, "passed": bool(passed), "score": 1.0 if passed else 0.0,
        "weight": weight, "channel": "func", "detail": detail,
    }


def grade(artifact: Path, weights: dict[str, float]) -> dict[str, Any]:
    """Run every check and return the full per-criterion report."""
    criteria: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        fixture = Path(td) / "sample.txt"
        fixture.write_text(FIXTURE_TEXT, encoding="utf-8")

        code, out, err = _run_artifact(artifact, [str(fixture)])
        if code != 0 or not out.strip():
            code2, out2, err2 = _run_artifact(artifact, [], stdin=FIXTURE_TEXT)
            if code2 == 0 and out2.strip():
                code, out, err = code2, out2, err2

        # AC1: runs without crashing on a UTF-8 text file.
        ran_ok = code == 0 and bool(out.strip())
        criteria.append(_criterion(
            "AC1", ran_ok,
            f"exit={code}" + (f" stderr={err[:160]!r}" if not ran_ok else ""),
            weights["AC1"],
        ))

        # Prefer labelled numbers; fall back to positional "lines words chars".
        got_lines = _scrape(out, "line")
        got_words = _scrape(out, "word")
        got_chars = _scrape(out, "char", "byte")
        if got_lines is None or got_words is None or got_chars is None:
            pos = _scrape_positional(out)
            if len(pos) >= 3:
                got_lines = got_lines if got_lines is not None else pos[0]
                got_words = got_words if got_words is not None else pos[1]
                got_chars = got_chars if got_chars is not None else pos[2]

        criteria.append(_criterion(
            "AC2", got_lines == GOLD["lines"],
            f"lines={got_lines} gold={GOLD['lines']}", weights["AC2"],
        ))
        criteria.append(_criterion(
            "AC3", got_words == GOLD["words"],
            f"words={got_words} gold={GOLD['words']}", weights["AC3"],
        ))
        criteria.append(_criterion(
            "AC4", got_chars == GOLD["chars"],
            f"chars={got_chars} gold={GOLD['chars']}", weights["AC4"],
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
