"""Reporting: leaderboard assembly + single-episode trace inspection.

This package turns *already-scored* runs into the two human-facing artifacts:

* :mod:`usabench.report.leaderboard` -- assemble per-agent leaderboard rows from
  the aggregates produced by :mod:`usabench.eval.aggregate` (Usability Score + CI,
  GA, GA_norm, ``pass^1``, ``pass^n``, AC, integrity rates, judge alpha, and the
  agent/oracle cost split), then emit a JSONL file and a Markdown table whose
  exact column set matches ``docs/scoring.md`` §10.3.

* :mod:`usabench.report.trace_view` -- render one episode's ``trace.jsonl`` (the
  ONE canonical artifact, ``DESIGN.md`` invariant 4) to a standalone, dependency-
  free HTML page: a timeline of agent actions, oracle queries/responses colored by
  the 0-5 assistance severity, and acceptance checkpoints.

Both modules are *pure consumers*: every number is read from the trace or from
``eval.aggregate``, and every scoring constant/label is read from the single
source of truth (:mod:`usabench.eval.spec`). Nothing here recomputes a score or
hardcodes a weight.
"""

from __future__ import annotations

from usabench.report.leaderboard import (
    LEADERBOARD_COLUMNS,
    LeaderboardRow,
    build_leaderboard,
    rows_to_jsonl,
    rows_to_markdown,
    write_leaderboard,
)
from usabench.report.trace_view import render_trace_html, write_trace_html

__all__ = [
    "LEADERBOARD_COLUMNS",
    "LeaderboardRow",
    "build_leaderboard",
    "rows_to_jsonl",
    "rows_to_markdown",
    "write_leaderboard",
    "render_trace_html",
    "write_trace_html",
]
