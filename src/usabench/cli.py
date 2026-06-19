"""The ``usabench`` Typer CLI: the operator-facing entry point.

Subcommands wire the foundation, harness, llm, oracle, and eval packages into the
end-to-end pipeline (``DESIGN.md``; ``docs/infra.md`` §6):

* ``collect``      -- delegate to the GitHub harvester (``collect.cli``).
* ``run``          -- load a run config, build agent + oracle clients via
  :func:`usabench.llm.factory.build_client`, fan out episodes through
  :func:`usabench.harness.run_episode`, and write ``runs/<run_id>/``.
* ``score``        -- read ``runs/``, compute the A-G metric registry + GA channels
  + the geometric/multiplicative composites (all from ``usability_score.yaml``),
  and write per-run + per-agent aggregate scores.
* ``leaderboard``  -- assemble + render the leaderboard (``usabench.report``).
* ``smoke``        -- the ACCEPTANCE GATE: run the ``v0_smoke`` tasks with the
  :class:`~usabench.llm.fake.FakeLLMClient` agent + oracle end-to-end
  (run -> score -> leaderboard) with ZERO network/API, exiting 0.
* ``serve-check``  -- ping a vLLM/OpenAI-shaped endpoint for liveness.
* ``estimate``     -- a dry token/cost estimate for a planned batch.

Every numeric constant in scoring is read from
:mod:`usabench.eval.spec` (``usability_score.yaml``) -- the CLI hardcodes none.
Heavy/optional deps (anthropic, openai, vllm) stay behind the lazy imports in
:mod:`usabench.llm.factory`; the smoke path uses only core deps.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer

from usabench import __version__
from usabench.eval.spec import SPEC_PATH, load_spec
from usabench.logging_setup import get_logger

app = typer.Typer(add_completion=False, help="usability-benchmark CLI.")
_log = get_logger(__name__)

#: Default smoke taskset shipped in the repo (``tasks/curated/v0_smoke.jsonl``).
_SMOKE_TASKSET = "tasks/curated/v0_smoke.jsonl"
#: The canonical trace filename written under each run directory.
_TRACE_FILENAME = "trace.jsonl"
#: The per-run result summary filename (mirrors ``harness.batch.RESULT_FILENAME``).
_RESULT_FILENAME = "result.json"
#: The per-run scored-metrics filename written by ``score``.
_SCORE_FILENAME = "score.json"


# --------------------------------------------------------------------------- #
# version / validate-spec                                                      #
# --------------------------------------------------------------------------- #


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("validate-spec")
def validate_spec() -> None:
    """Load and echo the usability-score spec (single source of truth)."""
    spec = load_spec()
    typer.echo(f"loaded {SPEC_PATH}")
    typer.echo(
        json.dumps(
            {"severity_weights": spec["severity_weights"], "keys": sorted(spec.keys())}
        )
    )


# --------------------------------------------------------------------------- #
# collect (delegate to the GitHub harvester)                                   #
# --------------------------------------------------------------------------- #


@app.command(
    "collect",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def collect(ctx: typer.Context) -> None:
    """Delegate to the ``usabench-collect`` GitHub harvester CLI.

    All arguments after ``collect`` are forwarded verbatim to
    :mod:`collect.cli` (e.g. ``usabench collect dry-run --limit 3``). Heavy
    collection deps are imported lazily inside that module.
    """
    from collect.cli import app as collect_app

    # Re-dispatch the residual args through the collector's own Typer app.
    collect_app(args=list(ctx.args), prog_name="usabench collect", standalone_mode=True)


# --------------------------------------------------------------------------- #
# run                                                                          #
# --------------------------------------------------------------------------- #


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a run config YAML."),
    output_root: Path | None = typer.Option(
        None, "--output-root", "-o", help="Override the config's run output root."
    ),
    fake_llm: bool = typer.Option(
        False, "--fake-llm", help="Force every model + oracle onto the FakeLLMClient (hermetic)."
    ),
    max_cells: int | None = typer.Option(
        None, "--max-cells", help="Cap on (task,model,seed) cells executed this invocation."
    ),
    no_resume: bool = typer.Option(
        False, "--no-resume", help="Re-run complete cells instead of skipping them."
    ),
) -> None:
    """Run a batch of episodes from a run config and write ``runs/<run_id>/``.

    Loads the run config, builds the agent + oracle clients (via
    :func:`usabench.llm.factory.build_client`), fans out one episode per
    ``(task, model, seed)`` cell through :func:`usabench.harness.run_episode`,
    and writes each cell's ``trace.jsonl`` + ``result.json``. Resumable: complete
    cells are skipped unless ``--no-resume``.
    """
    plan = _load_run_config(config, output_root=output_root, force_fake=fake_llm)
    results = _execute_run(plan, max_cells=max_cells, resume=not no_resume)
    accepted = sum(1 for r in results if getattr(r, "accepted", False))
    typer.echo(
        json.dumps(
            {
                "run_id_namespace": plan.run_label,
                "output_root": str(plan.output_root),
                "cells_executed": len(results),
                "accepted": accepted,
            }
        )
    )


# --------------------------------------------------------------------------- #
# score                                                                        #
# --------------------------------------------------------------------------- #


@app.command()
def score(
    runs_dir: Path = typer.Option(..., "--runs", "-r", help="Directory holding runs/<run_id>/."),
    tasks: Path = typer.Option(..., "--tasks", "-t", help="Taskset JSONL with the frozen gold."),
    scores_dir: Path | None = typer.Option(
        None, "--scores", "-s", help="Output dir for per-agent aggregates (default: <runs>/../scores)."
    ),
) -> None:
    """Score every complete run under ``runs_dir`` against the frozen task gold.

    For each run's ``trace.jsonl`` this computes the full A-G metric registry
    (:func:`usabench.eval.compute_all`), the GA channels (V1/V2 from the trace),
    and the geometric + multiplicative composites
    (:func:`usabench.eval.compute_composite`) -- every constant read from
    ``usability_score.yaml``. Per-run ``score.json`` files are written next to each
    trace, and per-agent aggregates are written to the scores dir for the
    leaderboard.
    """
    out_dir = scores_dir or (runs_dir.parent / "scores")
    summary = _score_runs(runs_dir, tasks_path=tasks, scores_dir=out_dir)
    typer.echo(json.dumps(summary))


# --------------------------------------------------------------------------- #
# leaderboard                                                                  #
# --------------------------------------------------------------------------- #


@app.command()
def leaderboard(
    scores_dir: Path = typer.Option(..., "--scores", "-s", help="Dir of per-agent aggregate JSON."),
    out_dir: Path | None = typer.Option(
        None, "--out", "-o", help="Where to write leaderboard.{jsonl,md} (default: <scores>)."
    ),
    title: str = typer.Option("Usability Benchmark Leaderboard", "--title", help="Markdown title."),
    show: bool = typer.Option(True, "--show/--no-show", help="Print the Markdown table to stdout."),
) -> None:
    """Build and render the leaderboard from per-agent score aggregates."""
    from usabench.report.leaderboard import (
        build_leaderboard,
        load_agent_aggregates,
        rows_to_markdown,
        write_leaderboard,
    )

    aggregates = load_agent_aggregates(scores_dir)
    rows = build_leaderboard(aggregates)
    target = out_dir or scores_dir
    paths = write_leaderboard(rows, target, title=title)
    if show:
        typer.echo(rows_to_markdown(rows, title=title))
    typer.echo(
        json.dumps(
            {"n_agents": len(rows), "jsonl": str(paths["jsonl"]), "markdown": str(paths["markdown"])}
        )
    )


# --------------------------------------------------------------------------- #
# smoke (the acceptance gate)                                                  #
# --------------------------------------------------------------------------- #


@app.command()
def smoke(
    work_dir: Path | None = typer.Option(
        None, "--work-dir", help="Working dir for runs/scores (default: a temp dir under ./_smoke)."
    ),
    taskset: Path = typer.Option(
        Path(_SMOKE_TASKSET), "--taskset", help="Smoke taskset JSONL (default: v0_smoke)."
    ),
    keep: bool = typer.Option(False, "--keep", help="Keep the work dir instead of using a temp dir."),
) -> None:
    """Run the v0_smoke tasks end-to-end with the FakeLLM agent + oracle (ZERO network).

    This is the acceptance gate (``docs/infra.md`` §6.5): it drives the real
    :func:`usabench.harness.run_episode` loop with a :class:`FakeLLMClient`-backed
    agent and a :class:`FakeLLMClient`-backed oracle against
    ``tasks/curated/v0_smoke.jsonl``, scores the resulting traces with
    :mod:`usabench.eval`, and prints a leaderboard. It must exit 0 with no paid API
    calls; any nonzero exit means the wiring is broken.
    """
    base = _resolve_smoke_workdir(work_dir, keep=keep)
    runs_dir = base / "runs"
    scores_dir = base / "scores"

    typer.echo(f"[smoke] work dir: {base}")
    typer.echo(f"[smoke] taskset:  {taskset}")

    # --- 1) RUN: FakeLLM agent + FakeLLM oracle through run_episode ----------- #
    plan = _build_smoke_plan(taskset=taskset, output_root=runs_dir)
    results = _execute_run(plan, max_cells=None, resume=False)
    if not results:
        typer.secho("[smoke] FAILED: no episodes executed", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    accepted = sum(1 for r in results if getattr(r, "accepted", False))
    typer.echo(f"[smoke] ran {len(results)} episode(s); accepted={accepted}")

    # --- 2) SCORE: pure offline metrics + composites from each trace.jsonl ---- #
    summary = _score_runs(runs_dir, tasks_path=taskset, scores_dir=scores_dir)
    typer.echo(
        f"[smoke] scored {summary['n_runs_scored']} run(s) "
        f"across {summary['n_agents']} agent(s)"
    )

    # --- 3) LEADERBOARD: assemble + render -----------------------------------#
    from usabench.report.leaderboard import (
        build_leaderboard,
        load_agent_aggregates,
        rows_to_markdown,
        write_leaderboard,
    )

    aggregates = load_agent_aggregates(scores_dir)
    rows = build_leaderboard(aggregates)
    write_leaderboard(rows, scores_dir, title="Smoke Leaderboard")
    typer.echo(rows_to_markdown(rows, title="Smoke Leaderboard"))

    # --- gate assertions: the wiring produced a complete pipeline ------------ #
    if not rows:
        typer.secho("[smoke] FAILED: empty leaderboard", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(
        f"[smoke] OK: run -> score -> leaderboard, {len(rows)} agent row(s), zero network.",
        fg=typer.colors.GREEN,
    )


# --------------------------------------------------------------------------- #
# serve-check                                                                  #
# --------------------------------------------------------------------------- #


@app.command("serve-check")
def serve_check(
    base_url: str = typer.Option(..., "--base-url", "-u", help="vLLM/OpenAI endpoint base URL."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model id to expect (optional)."),
    timeout_s: float = typer.Option(10.0, "--timeout", help="Request timeout in seconds."),
) -> None:
    """Ping a vLLM/OpenAI-shaped endpoint's ``/v1/models`` for liveness.

    Uses ``httpx`` (a core dep) directly -- it never imports vLLM. Exits 0 when the
    endpoint answers (and, if ``--model`` is given, lists that model), else 1.
    """
    import httpx

    url = base_url.rstrip("/")
    models_url = url + ("/models" if url.endswith("/v1") else "/v1/models")
    try:
        resp = httpx.get(models_url, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - report any failure as a dead endpoint
        typer.secho(f"serve-check FAILED: {models_url}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    served = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
    typer.echo(json.dumps({"endpoint": models_url, "served_models": served}))
    if model is not None and model not in served:
        typer.secho(
            f"serve-check: model {model!r} not served (have: {served})",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)
    typer.secho("serve-check OK", fg=typer.colors.GREEN)


# --------------------------------------------------------------------------- #
# estimate                                                                     #
# --------------------------------------------------------------------------- #


@app.command()
def estimate(
    config: Path = typer.Option(..., "--config", "-c", help="Run config YAML to estimate."),
    tokens_per_episode: int = typer.Option(
        20000, "--tokens-per-episode", help="Assumed agent tokens per episode (prompt+completion)."
    ),
    prompt_fraction: float = typer.Option(
        0.7, "--prompt-fraction", help="Fraction of tokens that are prompt (priced as input)."
    ),
) -> None:
    """Print a dry token/cost estimate for a planned batch (no episodes run).

    Enumerates the ``tasks x models x seeds`` grid from the config and multiplies
    by ``--tokens-per-episode``, pricing each model from its ``price_per_mtok``
    block via :func:`usabench.llm.usage.estimate_cost_usd`. Fake/local models
    price at ``$0``.
    """
    from usabench.llm.usage import estimate_cost_usd

    cfg = _load_raw_run_config(config)
    tasks = _load_tasks(_resolve_taskset(config, cfg))
    seeds = _resolve_seeds(cfg)
    models = _resolve_models(cfg, force_fake=False)

    n_tasks = len(tasks)
    n_seeds = len(seeds)
    per_model: list[dict[str, Any]] = []
    total_cost = 0.0
    total_tokens = 0
    prompt_t = int(tokens_per_episode * prompt_fraction)
    completion_t = tokens_per_episode - prompt_t
    for m in models:
        cells = n_tasks * n_seeds
        tokens = cells * tokens_per_episode
        cost = cells * estimate_cost_usd(prompt_t, completion_t, m.get("price_per_mtok"))
        total_cost += cost
        total_tokens += tokens
        per_model.append(
            {
                "model": m.get("id") or m.get("model") or "model",
                "provider": str(m.get("provider")),
                "cells": cells,
                "tokens": tokens,
                "cost_usd": round(cost, 4),
            }
        )

    typer.echo(
        json.dumps(
            {
                "n_tasks": n_tasks,
                "n_seeds": n_seeds,
                "n_models": len(models),
                "tokens_per_episode": tokens_per_episode,
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 4),
                "per_model": per_model,
            },
            indent=2,
        )
    )


# =========================================================================== #
# Internal wiring                                                              #
# =========================================================================== #


@dataclass
class _RunPlan:
    """A resolved, runnable batch plan (config + grid + budgets + wiring)."""

    run_label: str
    output_root: Path
    tasks: list[Any]
    models: list[dict[str, Any]]
    seeds: list[int]
    budget_limits: Any
    oracle_cfg: dict[str, Any]
    agent_cfg: dict[str, Any]
    tasks_root: Path = field(default_factory=lambda: Path("tasks"))
    force_fake: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


def _load_raw_run_config(config: Path) -> dict[str, Any]:
    """Load and env-interpolate a run config YAML into a dict."""
    from usabench.config.loader import load_yaml

    return load_yaml(config)


def _resolve_taskset(config: Path, cfg: dict[str, Any]) -> Path:
    """Resolve the taskset path (relative paths are relative to the config's dir or cwd)."""
    raw = cfg.get("taskset") or _SMOKE_TASKSET
    p = Path(raw)
    if p.is_file():
        return p
    # Try relative to the config file's directory.
    alt = config.parent / raw
    if alt.is_file():
        return alt
    # Fall back to cwd-relative (Typer already resolved cwd); error surfaces on load.
    return p


def _resolve_tasks_root(cfg: dict[str, Any], taskset: Path) -> Path:
    """Resolve the directory holding ``<task.id>/grader/grade.py`` subtrees.

    Honours an explicit ``tasks_root`` in the config; otherwise derives it from the
    taskset path by convention: tasksets live at ``tasks/<group>/<file>.jsonl`` while
    graders live at ``tasks/<task.id>/grader/grade.py``, so the grader root is the
    taskset's grandparent (``tasks/``) when nested, else its parent.
    """
    explicit = cfg.get("tasks_root")
    if explicit:
        return Path(str(explicit))
    parent = taskset.parent
    # ``tasks/curated/v0_smoke.jsonl`` -> ``tasks/``; a flat ``tasks/x.jsonl`` -> ``tasks/``.
    grandparent = parent.parent
    return grandparent if grandparent != Path("") and grandparent != parent else parent


def _resolve_seeds(cfg: dict[str, Any]) -> list[int]:
    """Resolve the seed list from ``seeds`` (preferred) or ``repeats``."""
    seeds = cfg.get("seeds")
    if isinstance(seeds, list) and seeds:
        return [int(s) for s in seeds]
    repeats = int(cfg.get("repeats", 1) or 1)
    return list(range(1, repeats + 1))


def _resolve_models(cfg: dict[str, Any], *, force_fake: bool) -> list[dict[str, Any]]:
    """Resolve the model configs, forcing the fake provider when requested."""
    models = cfg.get("models") or []
    out: list[dict[str, Any]] = []
    for m in models:
        md = dict(m) if isinstance(m, dict) else {"id": str(m)}
        if force_fake:
            md = _coerce_fake_model(md)
        out.append(md)
    if not out:
        # Default to a single fake model so a config without ``models`` still runs.
        out.append(_coerce_fake_model({"id": "fake-agent"}))
    return out


def _coerce_fake_model(md: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a model config onto the deterministic fake provider (hermetic)."""
    fake = dict(md)
    fake["provider"] = "fake"
    fake.setdefault("model", "fake-model")
    # If no script is configured, give it a self-contained declare_done so the
    # episode terminates promptly with an acceptance.
    if not any(k in fake for k in ("script", "keyed", "default")):
        fake["default"] = _DEFAULT_FAKE_AGENT_SCRIPT
    return fake


#: A ReAct ``declare_done`` the FakeLLM agent emits when no script is configured.
#: ``ReActScaffold`` parses this via :func:`usabench.agent.scaffold.parse_react_action`.
_DEFAULT_FAKE_AGENT_SCRIPT = (
    'Action: declare_done\n'
    'Action Input: {"summary": "smoke stub", "entrypoint": "main.py"}'
)


def _load_run_config(
    config: Path, *, output_root: Path | None, force_fake: bool
) -> _RunPlan:
    """Build a fully-resolved :class:`_RunPlan` from a run config YAML."""
    cfg = _load_raw_run_config(config)
    taskset = _resolve_taskset(config, cfg)
    tasks = _load_tasks(taskset)
    models = _resolve_models(cfg, force_fake=force_fake)
    seeds = _resolve_seeds(cfg)
    out_root = Path(output_root) if output_root is not None else Path(cfg.get("output_root", "./runs"))
    any_fake = force_fake or any(str(m.get("provider")) == "fake" for m in models)
    return _RunPlan(
        run_label=str(cfg.get("id", "run")),
        output_root=out_root,
        tasks=tasks,
        models=models,
        seeds=seeds,
        budget_limits=_resolve_budget(cfg, free_provider=any_fake),
        oracle_cfg=dict(cfg.get("oracle") or {}),
        agent_cfg=_resolve_agent_cfg(config, cfg),
        tasks_root=_resolve_tasks_root(cfg, taskset),
        force_fake=force_fake,
        raw=cfg,
    )


def _resolve_agent_cfg(config: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve the agent scaffold config (inline mapping or a path to a YAML)."""
    agent = cfg.get("agent")
    if isinstance(agent, dict):
        return agent
    if isinstance(agent, str):
        from usabench.config.loader import load_yaml

        candidate = Path(agent)
        if not candidate.is_file():
            alt = config.parent / agent
            candidate = alt if alt.is_file() else candidate
        if candidate.is_file():
            return load_yaml(candidate)
    return {}


def _resolve_budget(cfg: dict[str, Any], *, free_provider: bool = False) -> Any:
    """Build a :class:`BudgetLimits` from the config's ``budget`` block (or defaults).

    The budget gate treats ``used >= limit`` as exhausted, so a ``0.0`` cost (or
    ``0`` token) ceiling pre-empts the very first action. For free providers (the
    fake/local path that bills ``$0``) we lift a zero cost/token ceiling to a tiny
    positive headroom so a hermetic run is not blocked -- the spend stays ~0.
    """
    from usabench.harness import BudgetLimits

    budget = cfg.get("budget") or {}
    if not isinstance(budget, dict):
        return BudgetLimits()
    limits = BudgetLimits.model_validate(budget)
    if free_provider:
        patch: dict[str, Any] = {}
        if limits.max_cost_usd <= 0.0:
            patch["max_cost_usd"] = 1.0
        if limits.max_tokens <= 0:
            patch["max_tokens"] = 1_000_000
        if patch:
            limits = limits.model_copy(update=patch)
    return limits


def _build_smoke_plan(*, taskset: Path, output_root: Path) -> _RunPlan:
    """Build the hermetic smoke plan: one fake agent, the smoke oracle, seed 1."""
    from usabench.harness import BudgetLimits

    tasks = _load_tasks(taskset)
    # The smoke agent must genuinely SOLVE each task so the grader accepts it; mark
    # the fake model with ``agent_role: smoke`` so ``_build_agent`` wires the
    # solving responder (see usabench.llm.smoke_agent).
    smoke_model = _coerce_fake_model({"id": "fake-agent", "agent_role": "smoke"})
    return _RunPlan(
        run_label="smoke",
        output_root=output_root,
        tasks=tasks,
        models=[smoke_model],
        seeds=[1],
        # NOTE: a zero cost/token ceiling is "exhausted" at used==0 (the budget gate
        # is ``used >= limit``), which would pre-empt the very first action. The fake
        # provider bills $0, so we give the hermetic smoke a tiny positive headroom
        # rather than 0.0 -- any real spend on the fake path is still effectively 0.
        budget_limits=BudgetLimits(
            max_turns=8, max_wall_s=120.0, max_tokens=20000, max_cost_usd=1.0, max_oracle_queries=5
        ),
        oracle_cfg={
            "id": "oracle-fake",
            "client": {
                "provider": "fake",
                "model": "fake-oracle",
                "default": '{"level": 0, "text": "ok", "verdict": "accept", "cited_criteria": []}',
            },
            "temperature": 0.0,
            "persona": "non_expert_user",
            "helpfulness": "standard",
            "hint_budget": 3,
            "max_level": 4,
            "proactive_stuck_help": False,
        },
        agent_cfg={"max_steps": 8, "temperature": 0.0, "native_tools": False},
        tasks_root=_resolve_tasks_root({}, taskset),
        force_fake=True,
        raw={"id": "smoke"},
    )


def _load_tasks(taskset: Path) -> list[Any]:
    """Load and parse a JSONL taskset into :class:`~usabench.core.schema.Task` objects."""
    from usabench.core.schema import Task

    if not taskset.is_file():
        raise typer.BadParameter(f"taskset not found: {taskset}")
    tasks: list[Any] = []
    with taskset.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(Task.model_validate(json.loads(line)))
            except Exception as exc:  # noqa: BLE001 - report which line is bad
                raise typer.BadParameter(f"{taskset}:{i}: invalid task: {exc}") from exc
    if not tasks:
        raise typer.BadParameter(f"taskset is empty: {taskset}")
    return tasks


def _execute_run(plan: _RunPlan, *, max_cells: int | None, resume: bool) -> list[Any]:
    """Fan out ``plan`` through the batch scheduler + the real episode runner."""
    from usabench.harness import git_sha, plan_batch, run_batch

    tasks_by_id = {t.id: t for t in plan.tasks}
    batch_plan = plan_batch(
        plan.tasks,
        plan.models,
        plan.seeds,
        output_root=plan.output_root,
        config={"id": plan.run_label, "budget": plan.budget_limits.as_payload()},
        git_sha=git_sha(),
    )
    models_by_id = {(m.get("id") or m.get("model") or "model"): m for m in plan.models}

    def _run_one(cell: Any, task: Any, model_id: str) -> Any:
        model_cfg = models_by_id.get(model_id, plan.models[0])
        return _run_episode_for_cell(cell, task, model_cfg, plan)

    return run_batch(
        batch_plan, tasks_by_id, _run_one, resume=resume, max_cells=max_cells
    )


def _run_episode_for_cell(cell: Any, task: Any, model_cfg: dict[str, Any], plan: _RunPlan) -> Any:
    """Build the agent + oracle + sandbox for one cell and run the episode."""
    from usabench.harness import (
        BudgetMeter,
        InteractionBus,
        LocalSubprocessSandbox,
        build_manifest,
        run_episode,
    )
    from usabench.llm.factory import build_client
    from usabench.llm.usage import Channel

    # 1) Agent: a ReAct scaffold over the (fake) model client.
    agent = _build_agent(model_cfg, plan.agent_cfg)

    # 2) Oracle: a SimulatedUserOracle over the (fake) oracle client, adapted to
    #    the bus's structural OracleLike protocol.
    oracle_client = build_client(
        _oracle_client_cfg(plan.oracle_cfg, force_fake=plan.force_fake), channel=Channel.ORACLE
    )
    oracle = _build_oracle(task, oracle_client, plan.oracle_cfg)

    # 3) Sandbox: a real local workspace (network deny by default; hermetic).
    network = _task_network(task)
    sandbox = LocalSubprocessSandbox(
        task_id=task.id,
        run_id=cell.run_id,
        network=network,
        allowlist=list(getattr(task.env, "allowlist", []) or []),
    )

    manifest = build_manifest(
        task_id=task.id,
        seed=cell.seed,
        config={"id": plan.run_label, "model": model_cfg.get("id")},
        package_version=__version__,
        budgets=plan.budget_limits.as_payload(),
        agent={"id": model_cfg.get("id"), "provider": str(model_cfg.get("provider"))},
        oracle={"id": plan.oracle_cfg.get("id", "oracle"), "prompt_sha256": oracle.prompt_sha256},
        sandbox={"network": str(network), "backend": "local"},
        run_id_value=cell.run_id,
    )

    # 4) Verifier: grade the delivered artifact via the task's grader on every
    #    declare_done (and on forced-final verification). EVERY run grades.
    from usabench.eval.verifier import FunctionalVerifier

    verifier = FunctionalVerifier(plan.tasks_root)

    sandbox.setup()
    bus = InteractionBus(cell.trace_path, run_id=cell.run_id, oracle=oracle).open()
    budget = BudgetMeter(plan.budget_limits)
    try:
        result = run_episode(
            task,
            agent,
            sandbox=sandbox,
            bus=bus,
            budget=budget,
            seed=cell.seed,
            manifest=manifest,
            verifier=verifier,
        )
    finally:
        bus.close()
        sandbox.teardown()
    return result


def _build_agent(model_cfg: dict[str, Any], agent_cfg: dict[str, Any]) -> Any:
    """Build a :class:`ReActScaffold` agent over a model client from config."""
    from usabench.agent.scaffold import ReActScaffold, ScaffoldConfig
    from usabench.llm.factory import build_client
    from usabench.llm.usage import Channel

    if str(model_cfg.get("agent_role")) == "smoke":
        # The smoke agent must genuinely solve each task. A responder cannot pass
        # through the (serialisable) model config, so build the FakeLLMClient with
        # the solving responder directly here rather than via the factory.
        from usabench.llm.fake import FakeLLMClient
        from usabench.llm.smoke_agent import build_smoke_responder

        client: Any = FakeLLMClient(
            responder=build_smoke_responder(),
            model=str(model_cfg.get("model", "fake-model")),
            channel=Channel.AGENT,
        )
    else:
        client = build_client(model_cfg, channel=Channel.AGENT)
    scaffold_cfg = ScaffoldConfig(
        max_steps=int(agent_cfg.get("max_steps", 40)),
        temperature=float(agent_cfg.get("temperature", 0.0)),
        max_tokens=int(agent_cfg.get("max_tokens", 4096)),
        native_tools=bool(agent_cfg.get("native_tools", False)),
        system_prompt=agent_cfg.get("system_prompt") or None,
    )
    return ReActScaffold(client, scaffold_cfg)


def _oracle_client_cfg(oracle_cfg: dict[str, Any], *, force_fake: bool) -> dict[str, Any]:
    """Extract (and optionally fake-ify) the oracle's LLM client config."""
    client = dict(oracle_cfg.get("client") or {})
    if not client:
        client = {"provider": "fake", "model": "fake-oracle"}
    if force_fake:
        client["provider"] = "fake"
        client.setdefault("model", "fake-oracle")
        if not any(k in client for k in ("script", "keyed", "default")):
            client["default"] = (
                '{"level": 0, "text": "ok", "verdict": "accept", "cited_criteria": []}'
            )
    return client


def _build_oracle(task: Any, oracle_client: Any, oracle_cfg: dict[str, Any]) -> Any:
    """Build a bus-compatible oracle (``OracleLike``) over a SimulatedUserOracle."""
    from usabench.oracle.oracle import OracleConfig, SimulatedUserOracle

    persona = oracle_cfg.get("persona") or task.hidden.oracle_persona
    config = OracleConfig(
        model=str((oracle_cfg.get("client") or {}).get("model", "oracle")),
        temperature=float(oracle_cfg.get("temperature", 0.0)),
        persona=persona,
        helpfulness=str(oracle_cfg.get("helpfulness", "standard")),
        hint_budget=int(oracle_cfg.get("hint_budget", 3)),
        proactive_stuck_help=bool(oracle_cfg.get("proactive_stuck_help", False)),
        max_level=int(oracle_cfg.get("max_level", 4)),
    )
    sim = SimulatedUserOracle(task.hidden, oracle_client, config)
    return _BusOracleAdapter(sim)


class _BusOracleAdapter:
    """Adapt a :class:`SimulatedUserOracle` to the bus's structural ``OracleLike``.

    The bus calls ``answer(ctx) -> OracleResponse``; the simulated oracle's
    ``answer(query, ...) -> AnswerResult`` returns a richer object. This adapter
    unwraps the :class:`~usabench.core.schema.OracleResponse`, degrading gracefully
    to a refusing level-0 response if the (fake/real) model output is unparseable
    so a single bad oracle turn never crashes the harness loop.
    """

    def __init__(self, sim: Any) -> None:
        self._sim = sim
        #: Surfaced so the manifest can record the frozen oracle prompt hash.
        self.prompt_sha256 = getattr(sim, "prompt_sha256", "")

    def answer(self, ctx: Any) -> Any:
        from usabench.core.enums import Severity, Verdict
        from usabench.core.errors import OracleProtocolError
        from usabench.core.schema import OracleResponse

        try:
            result = self._sim.answer(ctx.query)
            return result.response
        except OracleProtocolError as exc:
            _log.warning("oracle.unparseable_falling_back", error=str(exc))
            return OracleResponse(
                severity=Severity.NONE,
                text="(could not parse oracle reply; no help given)",
                verdict=Verdict.NA,
            )


def _task_network(task: Any) -> Any:
    """Resolve the sandbox :class:`NetworkPolicy` for a task (default deny)."""
    from usabench.core.enums import NetworkPolicy

    net = getattr(task.env, "network", None)
    if net is None:
        return NetworkPolicy.DENY
    try:
        return NetworkPolicy(str(net.value if hasattr(net, "value") else net))
    except ValueError:
        return NetworkPolicy.DENY


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #


def _score_runs(runs_dir: Path, *, tasks_path: Path, scores_dir: Path) -> dict[str, Any]:
    """Score every complete run under ``runs_dir`` and write per-agent aggregates.

    Returns a small summary dict (counts) for the CLI to echo.
    """
    tasks_by_id = {t.id: t for t in _load_tasks(tasks_path)}
    scores_dir.mkdir(parents=True, exist_ok=True)

    # agent_id -> task_id -> list of per-seed scored records.
    per_agent: dict[str, dict[str, list[dict[str, Any]]]] = {}
    n_runs_scored = 0

    for run_dir in sorted(_iter_run_dirs(runs_dir)):
        scored = _score_one_run(run_dir, tasks_by_id)
        if scored is None:
            continue
        n_runs_scored += 1
        agent_id = scored["agent"]
        task_id = scored["task_id"]
        per_agent.setdefault(agent_id, {}).setdefault(task_id, []).append(scored)

    # Build + write one aggregate file per agent.
    for agent_id, by_task in per_agent.items():
        aggregate = _aggregate_agent(agent_id, by_task)
        out_path = scores_dir / f"{_safe_name(agent_id)}.json"
        out_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "n_runs_scored": n_runs_scored,
        "n_agents": len(per_agent),
        "scores_dir": str(scores_dir),
    }


def _iter_run_dirs(runs_dir: Path) -> Iterable[Path]:
    """Yield candidate run directories (those containing a ``trace.jsonl``)."""
    if not runs_dir.is_dir():
        return []
    out: list[Path] = []
    for child in runs_dir.iterdir():
        if child.is_dir() and (child / _TRACE_FILENAME).is_file():
            out.append(child)
    return out


def _score_one_run(run_dir: Path, tasks_by_id: dict[str, Any]) -> dict[str, Any] | None:
    """Compute metrics + composites for one run; write ``score.json``; return it.

    Returns ``None`` (and logs) when the run is incomplete or its task gold is
    missing, so a partial batch never aborts scoring.
    """
    from usabench.eval import compute_all, compute_composite
    from usabench.eval.composite import CompositeInputs
    from usabench.eval.scoring.v1_functional import score_v1
    from usabench.harness.batch import is_run_complete

    if not is_run_complete(run_dir):
        _log.info("score.skip_incomplete", run_dir=str(run_dir))
        return None

    trace = _read_trace(run_dir / _TRACE_FILENAME)
    if not trace:
        return None

    task_id, agent_id, seed = _run_identity(run_dir, trace)
    task = tasks_by_id.get(task_id)
    if task is None:
        _log.warning("score.task_missing", task_id=task_id, run_dir=str(run_dir))
        return None

    metrics = compute_all(trace, task)
    # V1 (functional/sandbox GA channel) recovered from the trace's verification_run.
    v1 = score_v1(trace, task)

    # The two composites read the SAME S = A3 (core criteria) and AC = C1 from the
    # trace + the spec constants -- no constant is hardcoded here.
    composite_inputs = CompositeInputs(
        s_core=float(metrics.get("A3_core_criteria_score", 0.0)),
        assistance_cost=float(metrics.get("C1_assistance_cost", 0.0)),
        panel_mean_ac=float(metrics.get("C1_assistance_cost", 0.0)) or None,
        autonomy=float(metrics.get("D1_autonomy_ratio", 0.0)),
        cost_per_progress=float(metrics.get("E7_cost_per_progress", 0.0)),
        robustness=1.0,
        ga=float(metrics.get("A3_core_criteria_score", 0.0)),
        success_binary=int(metrics.get("A1_success_binary", 0)),
        n_clarifications=int(metrics.get("B2_n_clarifications", 0)),
        goal_drift=float(metrics.get("A4_goal_drift", 0.0)),
        fake_done=False,
    )
    composite = compute_composite(composite_inputs)

    record = {
        "run_id": run_dir.name,
        "task_id": task_id,
        "agent": agent_id,
        "seed": seed,
        "accepted": bool(metrics.get("A1_success_binary", 0)),
        "v1": v1.score,
        "metrics": _jsonable(metrics),
        "composite": {
            "usability_geometric": composite.usability_geometric,
            "usability_multiplicative": composite.usability_multiplicative,
            "usability_linear": composite.usability_linear,
            "s": composite.s,
            "h": composite.h,
            "a": composite.a,
            "e": composite.e,
            "r": composite.r,
            "under_ask_penalised": composite.under_ask_penalised,
        },
    }
    (run_dir / _SCORE_FILENAME).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    return record


def _aggregate_agent(agent_id: str, by_task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Roll up one agent's per-(task,seed) scores into a leaderboard aggregate.

    Uses :mod:`usabench.eval.aggregate` for the robustness (pass^k) block and a
    simple macro-average for the headline composites. Shapes the output to the
    field aliases :func:`usabench.report.leaderboard.normalize_aggregate` accepts.
    """
    from usabench.eval.aggregate import aggregate_seeds

    all_records = [r for recs in by_task.values() for r in recs]
    n_seeds = max((len(recs) for recs in by_task.values()), default=0)

    def _mean(key_path: tuple[str, ...]) -> float:
        vals = [_dig(r, key_path) for r in all_records]
        vals = [float(v) for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    # Per-task pass^k over seeds (success = A1), then macro-average across tasks.
    pass_k_vals: list[float] = []
    success_rates: list[float] = []
    for recs in by_task.values():
        successes = [bool(r.get("accepted", False)) for r in recs]
        a2 = [float(_dig(r, ("metrics", "A2_criteria_score")) or 0.0) for r in recs]
        ac = [float(_dig(r, ("metrics", "C1_assistance_cost")) or 0.0) for r in recs]
        us = [float(_dig(r, ("composite", "usability_geometric")) or 0.0) for r in recs]
        agg = aggregate_seeds(
            successes=successes, a2_scores=a2, assistance_costs=ac, usability_scores=us
        )
        pass_k_vals.append(agg.pass_hat_k)
        success_rates.append(agg.success_rate)

    pass_n = sum(pass_k_vals) / len(pass_k_vals) if pass_k_vals else 0.0
    pass_1 = sum(success_rates) / len(success_rates) if success_rates else 0.0

    return {
        "agent": agent_id,
        "n_seeds": n_seeds,
        "n_tasks": len(by_task),
        "usability_score": _mean(("composite", "usability_geometric")),
        "usability_score_mult": _mean(("composite", "usability_multiplicative")),
        "ga": _mean(("metrics", "A3_core_criteria_score")),
        "v1": _mean(("v1",)),
        "pass_1": pass_1,
        "pass_n": pass_n,
        "ac": _mean(("composite", "h")),  # normalised assistance-lightness proxy
        "help_severity": _mean(("metrics", "C1_assistance_cost")),
        "n_interactions_per_task": _mean(("metrics", "B1_n_interventions")),
        "cost_agent_usd_per_task": _mean(("metrics", "E3_cost_usd_total")),
        "tokens_per_task": _mean(("metrics", "E2_tokens_total")),
        "wall_s_per_task": _mean(("metrics", "E1_wall_clock_s")),
        "release_lock": "",
    }


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #


def _read_trace(trace_path: Path) -> list[Any]:
    """Parse a ``trace.jsonl`` into a list of :class:`TraceEnvelope` (skip bad lines)."""
    from usabench.core.schema import parse_event

    out: list[Any] = []
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(parse_event(json.loads(line)))
            except Exception as exc:  # noqa: BLE001 - one bad line must not kill scoring
                _log.warning("score.bad_trace_line", error=str(exc))
    return out


def _run_identity(run_dir: Path, trace: list[Any]) -> tuple[str, str, int]:
    """Recover (task_id, agent_id, seed) from result.json, the trace, or the dir name."""
    result_path = run_dir / _RESULT_FILENAME
    task_id = ""
    agent_id = ""
    seed = 0
    if result_path.is_file():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            task_id = str(data.get("task_id") or "")
            seed = int(data.get("seed") or 0)
            agent_id = str((data.get("manifest") or {}).get("agent", {}).get("id") or "")
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass
    # Fall back to the episode_start payload for any missing field.
    if not task_id or not agent_id:
        start = next((e for e in trace if str(getattr(e, "type", "")) == "episode_start"), None)
        if start is not None:
            payload = start.payload
            task_id = task_id or str(getattr(payload, "task_id", "") or "")
            agent_id = agent_id or str((getattr(payload, "agent", {}) or {}).get("id") or "")
            seed = seed or int(getattr(payload, "seed", 0) or 0)
    return task_id, (agent_id or "agent"), seed


def _dig(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Walk a nested dict by ``path`` keys, returning ``None`` on any miss."""
    cur: Any = record
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _jsonable(metrics: dict[str, Any]) -> dict[str, Any]:
    """Coerce metric values into JSON-serialisable forms (inf/NaN -> string/null)."""
    import math

    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, float):
            if math.isinf(v):
                out[k] = "inf" if v > 0 else "-inf"
            elif math.isnan(v):
                out[k] = None
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _safe_name(name: str) -> str:
    """Make an agent id safe for use as a filename."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "agent"


def _resolve_smoke_workdir(work_dir: Path | None, *, keep: bool) -> Path:
    """Resolve (and create) the smoke working directory."""
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    if keep:
        base = Path("_smoke")
        base.mkdir(parents=True, exist_ok=True)
        return base
    import tempfile

    base = Path("_smoke")
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=str(base)))


def main() -> None:
    """Module entry point for the ``usabench`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
