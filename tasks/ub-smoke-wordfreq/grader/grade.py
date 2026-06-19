#!/usr/bin/env python3
"""Deterministic grader for the smoke task ``ub-smoke-wordfreq``.

Contract (identical to ``tasks/ub-cal-0007/grader/grade.py``)::

    python grade.py --artifact <agent_tool.py>

The grader builds a tiny, self-contained UTF-8 fixture, runs the agent artifact on
it, and emits a per-criterion pass/fail report as JSON to stdout::

    {"task_id", "criteria":[{"id","passed","score","detail",...}],
     "weighted_score", "accepted"}

Exit code 0 means the *grading run* succeeded (pass/fail is in the JSON); a
nonzero exit means the grader itself failed. Stdlib only; hermetic; no network.

The expected tool reads a text file path (positional arg) or stdin, tokenizes into
lowercase alphanumeric words, and prints the top-N words by descending frequency,
most frequent first, with a default of N=10. Output is scraped tolerantly: any
line containing a known word and its count is accepted, in either ``word count``
or ``word: count`` orderings.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

TASK_ID = "ub-smoke-wordfreq"

# Authoritative weights mirror the task's hidden.acceptance_criteria. Duplicated
# here only for standalone scoring convenience; the harness verifier re-weights
# from the frozen task gold.
DEFAULT_WEIGHTS = {"AC1": 1.0, "AC2": 2.0, "AC3": 1.0, "AC4": 1.0}

# A deterministic fixture. "the" appears most often (6x, mixed case exercises the
# case-insensitive requirement); "fox" appears 4x; "dog" 2x. The expected counts
# are derived clean-room from this text below, so the comment is illustrative only.
FIXTURE_TEXT = (
    "The the THE quick brown fox\n"
    "fox FOX jumps over the lazy dog\n"
    "the dog and the Fox ran\n"
)
# Expected lowercase counts derived clean-room from the fixture.
EXPECTED = Counter(re.findall(r"[a-z0-9]+", FIXTURE_TEXT.lower()))


def _run_artifact(artifact: Path, args: list[str], *, stdin: str | None = None,
                  timeout: float = 30.0) -> tuple[int, str, str]:
    """Run ``python <artifact> <args>`` and capture (exit_code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            [sys.executable, str(artifact), *args],
            capture_output=True, text=True, timeout=timeout,
            input=stdin,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except OSError as exc:  # pragma: no cover - defensive
        return 125, "", f"{type(exc).__name__}: {exc}"


def _count_for(stdout: str, word: str) -> int | None:
    """Scrape the reported count for ``word`` from stdout (order-tolerant)."""
    pat_word_first = re.compile(rf"\b{re.escape(word)}\b\D*?(\d+)", re.IGNORECASE)
    pat_count_first = re.compile(rf"(\d+)\D*?\b{re.escape(word)}\b", re.IGNORECASE)
    for line in stdout.splitlines():
        low = line.lower()
        if word in low:
            m = pat_word_first.search(line) or pat_count_first.search(line)
            if m:
                return int(m.group(1))
    return None


def _ranked_words(stdout: str) -> list[str]:
    """Return the words in the order they first appear in stdout (top-first)."""
    order: list[str] = []
    seen: set[str] = set()
    for line in stdout.splitlines():
        for tok in re.findall(r"[a-zA-Z0-9]+", line):
            low = tok.lower()
            if low in EXPECTED and low not in seen:
                seen.add(low)
                order.append(low)
    return order


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

        # AC1: runs without crashing on a UTF-8 text file.
        code, out, err = _run_artifact(artifact, [str(fixture)])
        if code != 0 or not out.strip():
            # Fall back to stdin if the tool reads stdin rather than a path arg.
            code2, out2, err2 = _run_artifact(artifact, [], stdin=FIXTURE_TEXT)
            if code2 == 0 and out2.strip():
                code, out, err = code2, out2, err2
        ran_ok = code == 0 and bool(out.strip())
        criteria.append(_criterion(
            "AC1", ran_ok,
            f"exit={code}" + (f" stderr={err[:160]!r}" if not ran_ok else ""),
            weights["AC1"],
        ))

        # AC2: case-insensitive counts. "the"->5, "fox"->3, "dog"->2.
        got_the = _count_for(out, "the")
        got_fox = _count_for(out, "fox")
        ci_ok = got_the == EXPECTED["the"] and got_fox == EXPECTED["fox"]
        criteria.append(_criterion(
            "AC2", ci_ok,
            f"the={got_the}(gold {EXPECTED['the']}) fox={got_fox}(gold {EXPECTED['fox']})",
            weights["AC2"],
        ))

        # AC3: most frequent word ("the") printed first.
        ranked = _ranked_words(out)
        top_ok = bool(ranked) and ranked[0] == "the"
        criteria.append(_criterion(
            "AC3", top_ok, f"ranked={ranked[:4]} expected_first=the", weights["AC3"],
        ))

        # AC4: honors a top-N argument (N=1 -> only the top word reported).
        c4, out4, _e4 = _run_artifact(artifact, [str(fixture), "1"])
        if c4 != 0:
            c4, out4, _e4 = _run_artifact(artifact, [str(fixture), "--top", "1"])
        if c4 != 0:
            c4, out4, _e4 = _run_artifact(artifact, [str(fixture), "-n", "1"])
        ranked1 = _ranked_words(out4) if c4 == 0 else []
        topn_ok = c4 == 0 and ranked1[:1] == ["the"] and len(ranked1) == 1
        criteria.append(_criterion(
            "AC4", topn_ok, f"top1_exit={c4} ranked={ranked1[:3]}", weights["AC4"],
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
