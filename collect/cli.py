"""The ``usabench-collect`` CLI: ``run | list-sources | validate | dry-run``.

This is the operator-facing entry point for the GitHub harvester
(``docs/tasks.md`` §5). The most important subcommand for smoke-testing is
``dry-run``: with just ``$GITHUB_TOKEN`` set it hits the live API read-only with a
tiny cap and prints a few normalized records -- proving the client, sources,
normalize, and filters all work end-to-end against real GitHub.

Heavy collection logic lives in :mod:`collect.pipeline`; this module only parses
flags, loads ``harvest.yaml``, wires the pieces, and prints results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog
import typer

from collect.filters import GateConfig, passes_quality_gates, scrub_record
from collect.github_client import GitHubClient, GitHubConfig, GitHubError
from collect.normalize import normalize_repo
from collect.sources import SOURCE_NAMES, RepoRef
from collect.sources.readmes import enrich_repo
from collect.sources.topics import build_query
from usabench.core.errors import ConfigError

app = typer.Typer(add_completion=False, help="usability-benchmark GitHub collector.")
log = structlog.get_logger(__name__)

#: Default tiny seed used by dry-run so it works with no config file.
_DRY_RUN_QUERY = "topic:cli language:python"


def _load_harvest_yaml(path: Path | None) -> dict[str, Any]:
    """Load an optional ``harvest.yaml`` config, returning ``{}`` if unset/missing."""
    if path is None:
        return {}
    from usabench.config.loader import load_yaml

    if not path.is_file():
        raise typer.BadParameter(f"config not found: {path}")
    try:
        return load_yaml(path)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _gate_config(cfg: dict[str, Any]) -> GateConfig:
    """Build a :class:`GateConfig` from the ``gates`` block of a harvest config."""
    gates = cfg.get("gates") or {}
    base = GateConfig()
    return GateConfig(
        min_stars=int(gates.get("min_stars", base.min_stars)),
        max_age_months=int(gates.get("max_age_months", base.max_age_months)),
        allow_archived=bool(gates.get("allow_archived", base.allow_archived)),
        allow_forks=bool(gates.get("allow_forks", base.allow_forks)),
        min_size_kb=int(gates.get("min_size_kb", base.min_size_kb)),
        max_size_kb=int(gates.get("max_size_kb", base.max_size_kb)),
        require_redistributable=bool(
            gates.get("require_redistributable", base.require_redistributable)
        ),
        require_description=bool(gates.get("require_description", base.require_description)),
    )


@app.command("list-sources")
def list_sources() -> None:
    """List the registered discovery sources."""
    for src in SOURCE_NAMES:
        typer.echo(src)


@app.command("validate")
def validate(
    path: Path = typer.Argument(..., help="Path to raw_harvest.jsonl or candidates.jsonl."),
    schema: Path | None = typer.Option(
        None, help="Path to raw_harvest.schema.json (defaults to repo schemas/)."
    ),
    limit: int = typer.Option(0, help="Validate only the first N lines (0 = all)."),
) -> None:
    """Validate a harvest JSONL file against ``raw_harvest.schema.json``.

    Reports the count of valid/invalid records and the first few errors. Exits
    non-zero if any record is invalid.
    """
    import jsonschema

    schema_path = schema or _default_schema_path()
    if not schema_path.is_file():
        raise typer.BadParameter(f"schema not found: {schema_path}")
    schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema_doc)

    if not path.is_file():
        raise typer.BadParameter(f"file not found: {path}")

    total = 0
    invalid = 0
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if limit and total >= limit:
                break
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid += 1
                errors.append(f"line {i + 1}: invalid JSON: {exc}")
                continue
            # Strip pipeline-internal annotation fields before schema validation.
            record = {k: v for k, v in record.items() if not k.startswith("_") and k not in {
                "passes_quality_gates", "gate_reasons", "suitability_prefilter_score", "draft_status"
            }}
            errs = sorted(validator.iter_errors(record), key=lambda e: list(e.path))
            if errs:
                invalid += 1
                if len(errors) < 10:
                    loc = "/".join(str(p) for p in errs[0].path) or "<root>"
                    errors.append(f"line {i + 1}: {loc}: {errs[0].message}")

    typer.echo(f"validated {total} records: {total - invalid} valid, {invalid} invalid")
    for e in errors:
        typer.echo(f"  - {e}")
    if invalid:
        raise typer.Exit(code=1)


@app.command("dry-run")
def dry_run(
    query: str = typer.Option(_DRY_RUN_QUERY, help="GitHub repo-search qualifier string."),
    limit: int = typer.Option(3, help="Max repos to fetch (kept tiny for a smoke test)."),
    pushed_after: str | None = typer.Option(
        None, help="Recency cutoff date YYYY-MM-DD (adds pushed:>=)."
    ),
    show_readme: bool = typer.Option(False, help="Include a README excerpt snippet."),
) -> None:
    """Hit the live GitHub API read-only with a tiny cap and print records.

    Requires ``$GITHUB_TOKEN`` (a public-read PAT). This is the canonical smoke
    test: it discovers a couple of repos, enriches + normalizes them, runs the
    quality gates, and prints the resulting ``raw_harvest`` records as JSON.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        typer.secho(
            "GITHUB_TOKEN is not set. Export a public-read PAT and retry:\n"
            "  export GITHUB_TOKEN=ghp_xxx\n"
            "  usabench-collect dry-run",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    full_query = query
    if pushed_after:
        full_query = f"{query} {build_query(pushed_after=pushed_after)}".strip()

    typer.echo(f"# dry-run query: {full_query!r} (limit={limit})", err=True)
    with GitHubClient(GitHubConfig(token=token)) as client:
        try:
            repos = client.search_repositories(
                full_query, sort="stars", order="desc", max_results=limit
            )
        except GitHubError as exc:
            typer.secho(f"GitHub error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

        if not repos:
            typer.echo("# no repositories matched.", err=True)
            raise typer.Exit(code=0)

        printed = 0
        for repo in repos[:limit]:
            full = repo.get("full_name") or ""
            owner, _, name = full.partition("/")
            ref = RepoRef(owner=owner, repo=name)
            try:
                enr = enrich_repo(client, ref, fetch_readme=show_readme, probe_tests_ci=False)
            except GitHubError as exc:
                typer.secho(f"# skip {full}: {exc}", fg=typer.colors.YELLOW, err=True)
                continue
            if enr is None:
                continue
            record = scrub_record(
                normalize_repo(
                    enr.repo,
                    head_sha=enr.head_sha,
                    readme=enr.readme if show_readme else None,
                    has_tests=enr.has_tests,
                    has_ci=enr.has_ci,
                )
            )
            passed, reasons = passes_quality_gates(record)
            record["_passes_quality_gates"] = passed
            record["_gate_reasons"] = reasons
            if not show_readme:
                record.pop("readme_excerpt", None)
            typer.echo(json.dumps(record, ensure_ascii=False, indent=2))
            printed += 1

        rate = client.rate
        typer.echo(
            f"# done: printed {printed} record(s). "
            f"rate: remaining={rate.remaining}/{rate.limit} resource={rate.resource or 'search'}",
            err=True,
        )


@app.command("run")
def run(
    config: Path | None = typer.Option(None, help="Path to harvest.yaml."),
    out_dir: Path = typer.Option(Path("data"), help="Output directory for JSONL + sqlite."),
    max_repos: int | None = typer.Option(None, help="Global cap on repos discovered."),
    sources: str = typer.Option("topics,awesome_lists", help="Comma-separated sources to run."),
    pushed_after: str | None = typer.Option(None, help="Recency cutoff date YYYY-MM-DD."),
    keep_rejects: bool = typer.Option(False, help="Persist gate-failing repos too."),
    build_candidates: bool = typer.Option(
        True, help="After harvest, dedup and write candidates.jsonl."
    ),
) -> None:
    """Run the full, resumable harvest into ``out_dir``.

    Requires ``$GITHUB_TOKEN``. Reads an optional ``harvest.yaml`` for the seed
    matrix, domains, and gate thresholds. Resumable: a re-run skips repos already
    in ``seen.sqlite`` and reuses the ETag cache.
    """
    if not os.environ.get("GITHUB_TOKEN"):
        typer.secho("GITHUB_TOKEN is not set.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    from collect.pipeline import CollectionPipeline, PipelineConfig

    cfg = _load_harvest_yaml(config)
    pipeline_cfg = PipelineConfig(
        out_dir=out_dir,
        gates=_gate_config(cfg),
        seeds=cfg.get("seeds", []),
        domains=cfg.get("domains", []),
        sources=[s.strip() for s in sources.split(",") if s.strip()],
        pushed_after=pushed_after or cfg.get("pushed_after"),
        max_repos=max_repos if max_repos is not None else cfg.get("max_repos"),
        keep_rejects=keep_rejects,
    )
    with CollectionPipeline(pipeline_cfg) as pipeline:
        stats = pipeline.run()
        typer.echo(json.dumps({"harvest": stats.as_dict()}))
        if build_candidates:
            n = pipeline.build_candidates()
            typer.echo(json.dumps({"candidates_written": n}))


def _default_schema_path() -> Path:
    """Locate ``schemas/raw_harvest.schema.json`` relative to the repo root."""
    here = Path(__file__).resolve()
    # collect/cli.py -> repo root is two levels up.
    return here.parent.parent / "schemas" / "raw_harvest.schema.json"


def main() -> None:
    """Module entry point for the ``usabench-collect`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
