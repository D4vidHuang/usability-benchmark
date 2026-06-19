"""The :class:`FunctionalVerifier` must genuinely grade the workspace artifact.

This is the regression guard for the bug where the verifier shelled out to the
grader with a *relative* path while ``cwd`` was the sandbox workspace, so every
criterion silently failed (GA=0) even for a correct solution. The benchmark is
worthless if grading does not discriminate, so we assert both directions:

* a correct ``main.py`` -> ``all_must_pass`` and ``rubric_score == 1.0``;
* a wrong / missing ``main.py`` -> not accepted and ``rubric_score < 1.0``.

Hermetic: the smoke graders are stdlib-only and run on a tiny in-grader fixture,
so this needs no network and no model calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from usabench.core.schema import Task
from usabench.eval.verifier import FunctionalVerifier
from usabench.harness import LocalSubprocessSandbox
from usabench.llm.smoke_agent import SMOKE_SOLUTIONS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASKS_ROOT = _REPO_ROOT / "tasks"
_SMOKE_TASKSET = _TASKS_ROOT / "curated" / "v0_smoke.jsonl"

#: A solution that runs but satisfies none of the acceptance criteria.
_WRONG_SOLUTION = "import sys\n\n\nif __name__ == '__main__':\n    print('not what was asked')\n"


def _load_task(task_id: str) -> Task:
    """Load one task record from the committed smoke taskset by id."""
    for line in _SMOKE_TASKSET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("id") == task_id:
            return Task.model_validate(record)
    raise AssertionError(f"task {task_id!r} not found in {_SMOKE_TASKSET}")


@pytest.fixture
def wordfreq_task() -> Task:
    return _load_task("ub-smoke-wordfreq")


def test_correct_artifact_is_accepted(wordfreq_task: Task) -> None:
    """A correct main.py passes every must-have criterion with full credit."""
    verifier = FunctionalVerifier(_TASKS_ROOT)
    with LocalSubprocessSandbox(task_id=wordfreq_task.id) as sandbox:
        sandbox.write("main.py", SMOKE_SOLUTIONS["word frequency"])
        vrun = verifier.verify(
            wordfreq_task, sandbox, trigger="submit", entrypoint="main.py"
        )
    assert vrun.all_must_pass is True
    assert vrun.rubric_score == pytest.approx(1.0)
    assert vrun.must_have and all(c.passed for c in vrun.must_have)


def test_wrong_artifact_is_rejected(wordfreq_task: Task) -> None:
    """A running-but-incorrect main.py is graded as a non-accepting failure."""
    verifier = FunctionalVerifier(_TASKS_ROOT)
    with LocalSubprocessSandbox(task_id=wordfreq_task.id) as sandbox:
        sandbox.write("main.py", _WRONG_SOLUTION)
        vrun = verifier.verify(
            wordfreq_task, sandbox, trigger="submit", entrypoint="main.py"
        )
    assert vrun.all_must_pass is False
    assert vrun.rubric_score < 1.0


def test_freely_named_artifact_is_resolved(wordfreq_task: Task) -> None:
    """The agent may name its tool freely and declare a command-string entrypoint.

    Open-ended tasks do not dictate a filename; a capable agent writes e.g.
    ``word_freq.py`` and declares ``python word_freq.py <file>``. The verifier must
    still find and grade it (regression for the entrypoint-resolution fix).
    """
    verifier = FunctionalVerifier(_TASKS_ROOT)
    with LocalSubprocessSandbox(task_id=wordfreq_task.id) as sandbox:
        sandbox.write("word_freq.py", SMOKE_SOLUTIONS["word frequency"])
        vrun = verifier.verify(
            wordfreq_task, sandbox, trigger="submit", entrypoint="python word_freq.py <file>"
        )
    assert vrun.all_must_pass is True
    assert vrun.rubric_score == pytest.approx(1.0)


def test_missing_artifact_is_rejected(wordfreq_task: Task) -> None:
    """Declaring done without writing the entrypoint fails every criterion."""
    verifier = FunctionalVerifier(_TASKS_ROOT)
    with LocalSubprocessSandbox(task_id=wordfreq_task.id) as sandbox:
        vrun = verifier.verify(
            wordfreq_task, sandbox, trigger="submit", entrypoint="absent.py"
        )
    assert vrun.all_must_pass is False
    assert vrun.rubric_score == pytest.approx(0.0)
