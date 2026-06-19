"""A genuinely-solving FakeLLM agent for the ``v0_smoke`` taskset (hermetic).

The smoke acceptance gate drives the real :func:`usabench.harness.run_episode`
loop with a :class:`~usabench.llm.fake.FakeLLMClient`-backed agent. For the gate
to be meaningful the fake agent must *actually solve* each smoke task: it must
``write_file`` a correct ``main.py`` and then ``declare_done`` with that
entrypoint, so the artifact passes the task's grader (``tasks/<id>/grader/grade.py``)
and Goal Achievement is genuinely 1.0 -- not rubber-stamped.

This module provides :func:`build_smoke_responder`, a deterministic
:class:`~usabench.llm.fake.Responder` that:

1. identifies the smoke task from the rendered agent prompt (the task title is
   present in the first user message -- see
   :func:`usabench.agent.scaffold.render_task_prompt`), and
2. emits ReAct actions in sequence -- first a ``write_file`` of the matching
   correct solution, then a ``declare_done`` with ``entrypoint: "main.py"`` --
   stepping by inspecting the transcript so it works turn-over-turn without any
   external state.

The solutions are intentionally small, stdlib-only, and deterministic; each
satisfies every acceptance criterion of its task. An unrecognised prompt falls
back to a plain ``declare_done`` so the agent never hangs the loop.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from usabench.llm.client import Message

__all__ = ["build_smoke_responder", "SMOKE_SOLUTIONS"]


# --------------------------------------------------------------------------- #
# Correct, stdlib-only solutions (one per smoke task).                          #
# --------------------------------------------------------------------------- #

_WORDFREQ_SOLUTION = '''\
"""Top-N word frequency counter (stdlib only, deterministic)."""
import re
import sys
from collections import Counter


def main(argv):
    args = [a for a in argv if not a.startswith("-")]
    top_n = 10
    flags = {}
    i = 0
    while i < len(argv):
        if argv[i] in ("--top", "-n") and i + 1 < len(argv):
            flags["top"] = argv[i + 1]
            i += 2
            continue
        i += 1
    path = args[0] if args else None
    if len(args) >= 2:
        try:
            top_n = int(args[1])
        except ValueError:
            pass
    if "top" in flags:
        top_n = int(flags["top"])
    text = open(path, encoding="utf-8").read() if path else sys.stdin.read()
    words = re.findall(r"[a-z0-9]+", text.lower())
    for word, count in Counter(words).most_common(top_n):
        print(f"{word} {count}")


if __name__ == "__main__":
    main(sys.argv[1:])
'''

_CSVSTATS_SOLUTION = '''\
"""Per-column CSV statistics: count/min/max/mean (stdlib only, deterministic)."""
import csv
import sys


def main(argv):
    column = None
    positional = []
    i = 0
    while i < len(argv):
        if argv[i] == "--column" and i + 1 < len(argv):
            column = argv[i + 1]
            i += 2
            continue
        if not argv[i].startswith("-"):
            positional.append(argv[i])
        i += 1
    path = positional[0]
    if column is None and len(positional) > 1:
        column = positional[1]
    values = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cell = row.get(column, "")
            if cell is None or cell.strip() == "":
                continue
            try:
                values.append(float(cell))
            except ValueError:
                continue
    count = len(values)
    print(f"count: {count}")
    if count:
        print(f"min: {min(values)}")
        print(f"max: {max(values)}")
        print(f"mean: {sum(values) / count}")
    else:
        print("min: n/a")
        print("max: n/a")
        print("mean: n/a")


if __name__ == "__main__":
    main(sys.argv[1:])
'''

_LINECOUNT_SOLUTION = '''\
"""A tiny wc: count lines, words, and characters (stdlib only, deterministic)."""
import sys


def main(argv):
    args = [a for a in argv if not a.startswith("-")]
    text = open(args[0], encoding="utf-8").read() if args else sys.stdin.read()
    lines = text.count("\\n")
    words = len(text.split())
    chars = len(text)
    print(f"lines: {lines}")
    print(f"words: {words}")
    print(f"chars: {chars}")


if __name__ == "__main__":
    main(sys.argv[1:])
'''

#: Map a lowercase identifying keyword (present in the task title/goal) to its
#: correct ``main.py`` source. Order matters only for disambiguation, which is not
#: needed here (the keywords are mutually exclusive across the three smoke tasks).
SMOKE_SOLUTIONS: dict[str, str] = {
    "word frequency": _WORDFREQ_SOLUTION,
    "csv column": _CSVSTATS_SOLUTION,
    "line and word": _LINECOUNT_SOLUTION,
}

#: Fallback keyword probes if the title text shifts (matched against the prompt).
_KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
    "word frequency": ("word frequency", "most often", "words show up"),
    "csv column": ("csv", "average and the min", "column"),
    "line and word": ("like 'wc'", "lines, words", "line and word"),
}


def _first_user_text(messages: list[Message]) -> str:
    """Return the first user message's content (the rendered task prompt)."""
    for msg in messages:
        if msg.role == "user":
            return msg.content
    return messages[0].content if messages else ""


def _select_solution(prompt: str) -> str | None:
    """Pick the matching solution source for a rendered agent prompt, if any."""
    low = prompt.lower()
    for key, aliases in _KEYWORD_ALIASES.items():
        if any(alias in low for alias in aliases):
            return SMOKE_SOLUTIONS[key]
    return None


def _already_wrote_main(messages: list[Message]) -> bool:
    """True if a prior assistant turn in this episode already wrote ``main.py``."""
    for msg in messages:
        if msg.role == "assistant" and "write_file" in msg.content and "main.py" in msg.content:
            return True
    return False


def _react(action: str, payload: dict[str, Any]) -> str:
    """Render a single ReAct ``Action`` / ``Action Input`` block (one line of JSON)."""
    return f"Action: {action}\nAction Input: {json.dumps(payload)}"


def build_smoke_responder() -> Callable[..., str]:
    """Build a deterministic responder that solves each smoke task end-to-end.

    Returns:
        A callable ``(messages, **params) -> str`` suitable for
        :class:`~usabench.llm.fake.FakeLLMClient`'s ``responder``. On the first
        turn for a task it returns a ``write_file`` of the correct ``main.py``; on
        the next turn it returns a ``declare_done`` with ``entrypoint: "main.py"``.
    """

    def _responder(messages: list[Message], **_params: Any) -> str:
        prompt = _first_user_text(messages)
        solution = _select_solution(prompt)
        if solution is None:
            # Unknown task: submit so the loop terminates rather than hanging.
            return _react("declare_done", {"summary": "smoke stub", "entrypoint": "main.py"})
        if _already_wrote_main(messages):
            return _react(
                "declare_done",
                {"summary": "implemented main.py", "entrypoint": "main.py"},
            )
        return _react("write_file", {"path": "main.py", "content": solution})

    return _responder
