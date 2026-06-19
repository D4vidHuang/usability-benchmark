"""Resumable batch fan-out: N seeds x M tasks x K models.

The batch layer enumerates every ``(task, model, seed)`` cell of an experiment,
computes its deterministic ``run_id`` (so a cell is addressable before it runs),
and writes each run under ``runs/<run_id>/``. It is **resumable**: a cell whose
output directory already contains a complete ``trace.jsonl`` (one ending in an
``episode_end`` line) is skipped, so a killed batch can be re-launched and only the
missing cells run (``docs/infra.md`` §6.1 -- idempotent, resumable batches).

The actual per-cell execution is delegated to a caller-supplied ``run_one``
callable so the batch layer stays free of agent/oracle/verifier wiring (which lives
in the CLI / scaffolding). This keeps the fan-out a pure, testable scheduler.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from usabench.config.hashing import config_hash as _config_hash
from usabench.core.ids import run_id as _run_id
from usabench.core.schema import RunResult, Task
from usabench.logging_setup import get_logger

__all__ = ["BatchCell", "BatchPlan", "RunOne", "iter_cells", "plan_batch", "run_batch"]

_log = get_logger(__name__)

#: The canonical trace filename written under each run directory.
TRACE_FILENAME = "trace.jsonl"
#: The per-run result summary filename.
RESULT_FILENAME = "result.json"


@dataclass(frozen=True)
class BatchCell:
    """One ``(task, model, seed)`` unit of work with its resolved identity.

    Attributes:
        task_id: The task's stable id.
        model_id: The model/agent config id.
        seed: The replica seed.
        run_id: The deterministic run id for this cell.
        run_dir: Absolute path to this cell's output directory.
    """

    task_id: str
    model_id: str
    seed: int
    run_id: str
    run_dir: Path

    @property
    def trace_path(self) -> Path:
        """Path to this cell's canonical ``trace.jsonl``."""
        return self.run_dir / TRACE_FILENAME

    @property
    def result_path(self) -> Path:
        """Path to this cell's ``result.json`` summary."""
        return self.run_dir / RESULT_FILENAME


@dataclass(frozen=True)
class BatchPlan:
    """A fully-enumerated batch: every cell + the output root.

    Attributes:
        cells: All planned :class:`BatchCell` units (every task x model x seed).
        output_root: The directory under which ``runs/<run_id>/`` are written.
    """

    cells: tuple[BatchCell, ...]
    output_root: Path

    def pending(self) -> list[BatchCell]:
        """Return the cells whose runs are not yet complete (resume set)."""
        return [c for c in self.cells if not is_run_complete(c.run_dir)]


#: A caller-supplied per-cell executor: takes the cell + task + model id, runs the
#: episode into ``cell.run_dir``, and returns the :class:`RunResult`.
RunOne = Callable[["BatchCell", Task, str], RunResult]


def _model_id(model_cfg: object) -> str:
    """Best-effort extraction of a stable model id from a model config object."""
    if isinstance(model_cfg, str):
        return model_cfg
    if isinstance(model_cfg, dict):
        return str(model_cfg.get("id") or model_cfg.get("model") or "model")
    mid = getattr(model_cfg, "id", None) or getattr(model_cfg, "model", None)
    return str(mid) if mid is not None else "model"


def is_run_complete(run_dir: str | Path) -> bool:
    """Return True if ``run_dir`` holds a complete trace (ends in ``episode_end``).

    A run is considered complete iff its ``trace.jsonl`` exists, is non-empty, and
    its final non-blank line is an ``episode_end`` event. This is the resume
    predicate -- it tolerates partial/crashed traces (which are *not* complete and
    will be re-run).

    Args:
        run_dir: The run's output directory.

    Returns:
        True if the run finished and need not be repeated.
    """
    trace = Path(run_dir) / TRACE_FILENAME
    if not trace.is_file() or trace.stat().st_size == 0:
        return False
    last = ""
    with trace.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                last = stripped
    if not last:
        return False
    try:
        obj = json.loads(last)
    except json.JSONDecodeError:
        return False
    return bool(isinstance(obj, dict) and obj.get("type") == "episode_end")


def iter_cells(
    tasks: Iterable[Task],
    models: Iterable[object],
    seeds: Iterable[int],
    *,
    output_root: str | Path,
    config: object,
    git_sha: str,
) -> Iterator[BatchCell]:
    """Yield one :class:`BatchCell` per ``(task, model, seed)`` with resolved ids.

    The ``run_id`` is derived from ``config_hash(config + model_id) x task_id x seed
    x git_sha`` so the same experiment cell always maps to the same id and directory
    (idempotent batches). The model id is folded into the config hash so two models
    in the same batch get distinct run ids.

    Args:
        tasks: The tasks to run.
        models: The model/agent configs to run (str id, dict, or model object).
        seeds: The replica seeds.
        output_root: Where ``<run_id>/`` directories are created.
        config: The shared run config (hashed into the run id).
        git_sha: The repo git sha (recorded in the run id).

    Yields:
        One :class:`BatchCell` per cell, in (task, model, seed) order.
    """
    root = Path(output_root)
    models_list = list(models)
    seeds_list = list(seeds)
    base = _to_hashable_config(config)
    for task in tasks:
        for model_cfg in models_list:
            mid = _model_id(model_cfg)
            cfg_hash = _config_hash({"run": base, "model": _to_hashable_config(model_cfg)})
            for seed in seeds_list:
                rid = _run_id(cfg_hash, task.id, int(seed), git_sha)
                yield BatchCell(
                    task_id=task.id,
                    model_id=mid,
                    seed=int(seed),
                    run_id=rid,
                    run_dir=root / rid,
                )


def _to_hashable_config(cfg: object) -> object:
    """Coerce a config (model/dict/str) into a JSON-hashable structure."""
    from usabench.config.hashing import to_hashable

    return to_hashable(cfg)


def plan_batch(
    tasks: Iterable[Task],
    models: Iterable[object],
    seeds: Iterable[int],
    *,
    output_root: str | Path,
    config: object,
    git_sha: str,
) -> BatchPlan:
    """Enumerate a full batch into a :class:`BatchPlan` (no execution).

    Args:
        tasks: Tasks to run.
        models: Model/agent configs.
        seeds: Replica seeds.
        output_root: Output root for run directories.
        config: Shared run config (hashed into run ids).
        git_sha: Repo git sha.

    Returns:
        A :class:`BatchPlan` with every cell enumerated.
    """
    cells = tuple(
        iter_cells(
            tasks, models, seeds, output_root=output_root, config=config, git_sha=git_sha
        )
    )
    return BatchPlan(cells=cells, output_root=Path(output_root))


def run_batch(
    plan: BatchPlan,
    tasks_by_id: dict[str, Task],
    run_one: RunOne,
    *,
    resume: bool = True,
    max_cells: int | None = None,
) -> list[RunResult]:
    """Execute a planned batch, skipping already-complete cells when resuming.

    For each pending cell, this creates the run directory, calls ``run_one`` to
    execute the episode (which is responsible for writing ``trace.jsonl`` there),
    persists the returned :class:`RunResult` to ``result.json``, and collects it.
    A failure in one cell is logged and recorded but never aborts the batch.

    Args:
        plan: The enumerated :class:`BatchPlan`.
        tasks_by_id: Lookup from task id to the frozen :class:`Task`.
        run_one: The per-cell executor callable.
        resume: If True (default), skip cells whose runs are already complete.
        max_cells: Optional cap on how many cells to execute this invocation.

    Returns:
        The list of :class:`RunResult` for the cells executed this invocation
        (skipped/complete cells are not re-summarized).
    """
    results: list[RunResult] = []
    cells = plan.pending() if resume else list(plan.cells)
    if max_cells is not None:
        cells = cells[:max_cells]
    total = len(cells)
    _log.info("batch.start", total=total, output_root=str(plan.output_root))

    for i, cell in enumerate(cells, start=1):
        task = tasks_by_id.get(cell.task_id)
        if task is None:
            _log.warning("batch.task_missing", task_id=cell.task_id, run_id=cell.run_id)
            continue
        cell.run_dir.mkdir(parents=True, exist_ok=True)
        _log.info(
            "batch.cell_start",
            i=i,
            total=total,
            task_id=cell.task_id,
            model_id=cell.model_id,
            seed=cell.seed,
            run_id=cell.run_id,
        )
        try:
            result = run_one(cell, task, cell.model_id)
        except Exception as exc:  # pragma: no cover - executor failures are logged
            _log.error(
                "batch.cell_failed", run_id=cell.run_id, error=str(exc), task_id=cell.task_id
            )
            continue
        _write_result(cell.result_path, result)
        results.append(result)

    _log.info("batch.done", executed=len(results), total=total)
    return results


def _write_result(path: Path, result: RunResult) -> None:
    """Persist a :class:`RunResult` summary as pretty JSON next to its trace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
