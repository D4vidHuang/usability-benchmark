"""Leaderboard assembly from scored-run aggregates.

The leaderboard is a *pure view* over per-agent aggregates produced upstream by
:mod:`usabench.eval.aggregate` (the metrics/statistics agent owns that module).
This module's only jobs are:

1. **Normalize** an aggregate (however it arrives -- a pydantic model, a plain
   dataclass, or a dict) into a flat, fully-typed :class:`LeaderboardRow`.
2. **Order & sort** rows per ``docs/scoring.md`` §10.3 (Usability Score desc,
   ties broken by lower AC).
3. **Emit** the two artifacts: a machine-readable ``leaderboard.jsonl`` (one row
   per line) and a human-readable Markdown table with the exact column set.

Every threshold/label needed for rendering (e.g. the ``tau_ga`` used in the
``pass^k`` success definition, the ``k`` reported for ``pass^n``, CI level) is
read from the single source of truth :mod:`usabench.eval.spec` -- no constant is
duplicated here (``DESIGN.md`` frozen build decision #2).

The exact leaderboard columns (``docs/scoring.md`` §10.3 / ``docs/metrics.md``
§7-8) are, in order:

    Usability Score (headline) + 95% CI, GA, GA_norm, pass^1, pass^n, AC,
    n_interactions/task, help_severity, V1/V2/V3 (diagnostic), fake_done %,
    never_ask %, over_ask %, judge alpha, $cost/tokens/wall per task
    (agent + oracle split), n_seeds, release.lock hash.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from usabench.eval.spec import load_spec
from usabench.logging_setup import get_logger

__all__ = [
    "LEADERBOARD_COLUMNS",
    "LeaderboardRow",
    "build_leaderboard",
    "rows_to_jsonl",
    "rows_to_markdown",
    "write_leaderboard",
    "load_agent_aggregates",
]

_log = get_logger(__name__)

#: Sentinel printed for a missing scalar in the Markdown table.
_NA = "--"


# --------------------------------------------------------------------------- #
# Column registry (single declaration of the leaderboard's public schema)      #
# --------------------------------------------------------------------------- #


class _Column(BaseModel):
    """One leaderboard column: how to extract it and how to render it.

    Attributes:
        key: The :class:`LeaderboardRow` attribute name.
        header: Markdown table header text.
        kind: Rendering hint -- ``score`` (3 dp), ``pct`` (percentage),
            ``int`` (integer), ``cost`` (USD, 4 dp), ``str`` (verbatim),
            ``ci`` (a (lo, hi) interval), or ``sci`` (scientific for big tokens).
        primary: Marked with a star in the header (a headline column).
    """

    model_config = ConfigDict(frozen=True)

    key: str
    header: str
    kind: str = "score"
    primary: bool = False


#: The canonical, ordered column schema (``docs/scoring.md`` §10.3).
LEADERBOARD_COLUMNS: tuple[_Column, ...] = (
    _Column(key="agent", header="Agent", kind="str"),
    _Column(key="usability_score", header="Usability", kind="score", primary=True),
    _Column(key="usability_ci", header="95% CI", kind="ci", primary=True),
    _Column(key="ga", header="GA", kind="score", primary=True),
    _Column(key="ga_norm", header="GA_norm", kind="score"),
    _Column(key="pass_1", header="pass^1", kind="score", primary=True),
    _Column(key="pass_n", header="pass^n", kind="score", primary=True),
    _Column(key="ac", header="AC", kind="score", primary=True),
    _Column(key="n_interactions_per_task", header="Interactions/task", kind="score"),
    _Column(key="help_severity", header="HelpSev", kind="score"),
    _Column(key="v1", header="V1", kind="score"),
    _Column(key="v2", header="V2", kind="score"),
    _Column(key="v3", header="V3", kind="score"),
    _Column(key="fake_done_rate", header="fake_done%", kind="pct", primary=True),
    _Column(key="never_ask_rate", header="never_ask%", kind="pct"),
    _Column(key="over_ask_rate", header="over_ask%", kind="pct"),
    _Column(key="judge_alpha", header="judge α", kind="score"),
    _Column(key="cost_agent_usd_per_task", header="$agent/task", kind="cost"),
    _Column(key="cost_oracle_usd_per_task", header="$oracle/task", kind="cost"),
    _Column(key="tokens_per_task", header="tok/task", kind="sci"),
    _Column(key="wall_s_per_task", header="wall_s/task", kind="score"),
    _Column(key="n_seeds", header="n_seeds", kind="int"),
    _Column(key="release_lock", header="release.lock", kind="str"),
)


# --------------------------------------------------------------------------- #
# The flat leaderboard row                                                     #
# --------------------------------------------------------------------------- #


class LeaderboardRow(BaseModel):
    """One agent's row in the leaderboard -- the public reporting contract.

    Every field maps to a column in :data:`LEADERBOARD_COLUMNS`. Optional fields
    are ``None`` when the upstream aggregate did not supply them (e.g. an agent
    run without the judge channel has ``judge_alpha=None``); renderers print a
    sentinel rather than a fabricated zero.

    The cost is split agent vs oracle on purpose (``docs/metrics.md`` §5): the
    oracle is a fixed harness cost, so an agent that offloads thinking onto an
    expensive oracle must not look cheap.
    """

    model_config = ConfigDict(extra="forbid")

    # Identity / provenance ------------------------------------------------- #
    agent: str = Field(..., description="Agent (model) identifier under test.")
    n_seeds: int = Field(0, ge=0, description="Reruns/seeds per task behind these numbers.")
    n_tasks: int = Field(0, ge=0, description="Distinct tasks aggregated.")
    release_lock: str = Field("", description="release.lock content hash (comparability key).")

    # Headline composites + CI ---------------------------------------------- #
    usability_score: float | None = Field(
        None, description="Geometric Usability Score headline, macro-avg over tiers."
    )
    usability_ci: tuple[float, float] | None = Field(
        None, description="Cluster-bootstrap CI (lo, hi) for the Usability Score."
    )
    usability_score_mult: float | None = Field(
        None, description="Secondary multiplicative U variant (reported, not headline)."
    )

    # Goal achievement ------------------------------------------------------ #
    ga: float | None = Field(None, description="Mean goal-achievement, macro-avg over tiers.")
    ga_norm: float | None = Field(None, description="Reference-relative GA (difficulty-adjusted).")
    ga_by_tier: dict[str, float] = Field(
        default_factory=dict, description="GA broken down by difficulty tier (diagnostic)."
    )

    # Reliability ----------------------------------------------------------- #
    pass_1: float | None = Field(None, description="Mean per-attempt success (pass^1).")
    pass_n: float | None = Field(None, description="All-of-n reliability (pass^k at k=n).")
    pass_k_k: int | None = Field(None, description="The k reported as pass^n (==n_seeds).")

    # Assistance / interaction ---------------------------------------------- #
    ac: float | None = Field(None, description="Mean assistance cost in [0,1] (lower better).")
    n_interactions_per_task: float | None = Field(
        None, description="Mean oracle interactions per task."
    )
    help_severity: float | None = Field(
        None, description="Mean severity-weighted help (convex weights)."
    )

    # Verification channels (diagnostic) ------------------------------------ #
    v1: float | None = Field(None, description="Mean V1 functional/sandbox channel score.")
    v2: float | None = Field(None, description="Mean V2 rubric channel score.")
    v3: float | None = Field(None, description="Mean V3 judge channel score.")

    # Integrity flags ------------------------------------------------------- #
    fake_done_rate: float | None = Field(None, description="Rate of fake-done runs (claim≠deliver).")
    never_ask_rate: float | None = Field(None, description="Rate of never-ask runs.")
    over_ask_rate: float | None = Field(None, description="Rate of over-ask runs.")

    # Judge diagnostics ----------------------------------------------------- #
    judge_alpha: float | None = Field(None, description="Mean inter-judge agreement (Krippendorff α).")

    # Efficiency (cost split) ----------------------------------------------- #
    cost_agent_usd_per_task: float | None = Field(None, description="Agent-side $ per task.")
    cost_oracle_usd_per_task: float | None = Field(None, description="Oracle-side $ per task (covariate).")
    tokens_per_task: float | None = Field(None, description="Agent tokens per task.")
    wall_s_per_task: float | None = Field(None, description="Wall-clock seconds per task.")

    def cost_total_usd_per_task(self) -> float | None:
        """Return agent+oracle $/task, or ``None`` if neither side is known."""
        a, o = self.cost_agent_usd_per_task, self.cost_oracle_usd_per_task
        if a is None and o is None:
            return None
        return (a or 0.0) + (o or 0.0)


# --------------------------------------------------------------------------- #
# Aggregate -> row normalization                                              #
# --------------------------------------------------------------------------- #

#: Accepted shapes for one agent's aggregate: a mapping or any attr-bearing obj.
AggregateLike = Mapping[str, Any] | Any

#: Alternative source keys we accept for each row field (defensive against the
#: exact attribute names ``eval.aggregate`` settles on; first hit wins).
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "agent": ("agent", "agent_id", "model", "name"),
    "n_seeds": ("n_seeds", "seeds", "n_seed"),
    "n_tasks": ("n_tasks", "tasks", "n_task"),
    "release_lock": ("release_lock", "release_lock_hash", "release", "lock_hash"),
    "usability_score": ("usability_score", "usability", "usability_geom", "score"),
    "usability_ci": ("usability_ci", "usability_score_ci", "ci", "score_ci"),
    "usability_score_mult": ("usability_score_mult", "usability_mult", "u_multiplicative"),
    "ga": ("ga", "ga_mean", "goal_achievement"),
    "ga_norm": ("ga_norm", "ga_normalized", "ga_rel"),
    "ga_by_tier": ("ga_by_tier", "ga_tier", "ga_per_tier"),
    "pass_1": ("pass_1", "pass1", "pass_hat_1", "pass^1", "success_rate"),
    "pass_n": ("pass_n", "passn", "pass_hat_n", "pass^n", "pass_hat_k"),
    "pass_k_k": ("pass_k_k", "pass_k", "k"),
    "ac": ("ac", "assistance_cost", "ac_mean"),
    "n_interactions_per_task": ("n_interactions_per_task", "n_interactions", "interactions_per_task"),
    "help_severity": ("help_severity", "severity_weighted_help", "assistance_cost_points"),
    "v1": ("v1", "v1_mean"),
    "v2": ("v2", "v2_mean"),
    "v3": ("v3", "v3_mean"),
    "fake_done_rate": ("fake_done_rate", "fake_done", "fake_done_pct"),
    "never_ask_rate": ("never_ask_rate", "never_ask", "never_ask_pct"),
    "over_ask_rate": ("over_ask_rate", "over_ask", "over_ask_pct"),
    "judge_alpha": ("judge_alpha", "judge_a", "alpha", "krippendorff_alpha"),
    "cost_agent_usd_per_task": ("cost_agent_usd_per_task", "cost_agent", "agent_cost_usd", "cost_usd_agent"),
    "cost_oracle_usd_per_task": ("cost_oracle_usd_per_task", "cost_oracle", "oracle_cost_usd", "cost_usd_oracle"),
    "tokens_per_task": ("tokens_per_task", "tokens", "agent_tokens_per_task"),
    "wall_s_per_task": ("wall_s_per_task", "wall_s", "wall_clock_s_per_task"),
}


def _get(src: AggregateLike, *names: str) -> Any:
    """Return the first present, non-None attribute/key among ``names``.

    Args:
        src: A mapping or an attribute-bearing object (e.g. a pydantic model).
        *names: Candidate source keys, tried in order.

    Returns:
        The first value found, else ``None``.
    """
    for name in names:
        if isinstance(src, Mapping):
            if name in src and src[name] is not None:
                return src[name]
        else:
            val = getattr(src, name, None)
            if val is not None:
                return val
    return None


def _as_ci(value: Any) -> tuple[float, float] | None:
    """Coerce a CI-like value into a ``(lo, hi)`` float tuple, or ``None``.

    Accepts a 2-sequence ``[lo, hi]`` or a mapping with ``lo``/``hi`` (or
    ``low``/``high``) keys.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        lo = value.get("lo", value.get("low"))
        hi = value.get("hi", value.get("high"))
        if lo is None or hi is None:
            return None
        return (float(lo), float(hi))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    return None


def normalize_aggregate(src: AggregateLike) -> LeaderboardRow:
    """Normalize one agent's aggregate (any accepted shape) into a row.

    This is the single adapter between whatever :mod:`usabench.eval.aggregate`
    emits and the leaderboard's public :class:`LeaderboardRow`. It is permissive
    by design (tries several source-key aliases per field) so the report layer
    does not break if the aggregator renames a field; missing values stay
    ``None`` and render as a sentinel rather than a fabricated ``0``.

    Args:
        src: One agent's aggregate as a mapping or attribute-bearing object.

    Returns:
        A populated :class:`LeaderboardRow`.
    """
    data: dict[str, Any] = {}
    for field, aliases in _FIELD_ALIASES.items():
        data[field] = _get(src, *aliases)

    agent = data.get("agent") or "<unknown-agent>"
    n_seeds = int(data.get("n_seeds") or 0)

    row = LeaderboardRow(
        agent=str(agent),
        n_seeds=n_seeds,
        n_tasks=int(data.get("n_tasks") or 0),
        release_lock=str(data.get("release_lock") or ""),
        usability_score=_to_float(data.get("usability_score")),
        usability_ci=_as_ci(data.get("usability_ci")),
        usability_score_mult=_to_float(data.get("usability_score_mult")),
        ga=_to_float(data.get("ga")),
        ga_norm=_to_float(data.get("ga_norm")),
        ga_by_tier={str(k): float(v) for k, v in (data.get("ga_by_tier") or {}).items()},
        pass_1=_to_float(data.get("pass_1")),
        pass_n=_to_float(data.get("pass_n")),
        pass_k_k=_to_int(data.get("pass_k_k")) or (n_seeds or None),
        ac=_to_float(data.get("ac")),
        n_interactions_per_task=_to_float(data.get("n_interactions_per_task")),
        help_severity=_to_float(data.get("help_severity")),
        v1=_to_float(data.get("v1")),
        v2=_to_float(data.get("v2")),
        v3=_to_float(data.get("v3")),
        fake_done_rate=_to_float(data.get("fake_done_rate")),
        never_ask_rate=_to_float(data.get("never_ask_rate")),
        over_ask_rate=_to_float(data.get("over_ask_rate")),
        judge_alpha=_to_float(data.get("judge_alpha")),
        cost_agent_usd_per_task=_to_float(data.get("cost_agent_usd_per_task")),
        cost_oracle_usd_per_task=_to_float(data.get("cost_oracle_usd_per_task")),
        tokens_per_task=_to_float(data.get("tokens_per_task")),
        wall_s_per_task=_to_float(data.get("wall_s_per_task")),
    )
    return row


def _to_float(value: Any) -> float | None:
    """Coerce to float, returning ``None`` for ``None``/NaN/uncoercible input."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _to_int(value: Any) -> int | None:
    """Coerce to int, returning ``None`` for ``None``/uncoercible input."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Source: pull aggregates from eval.aggregate (lazy, optional)                 #
# --------------------------------------------------------------------------- #


def load_agent_aggregates(results_dir: str | Path) -> list[Any]:
    """Load per-agent aggregates by delegating to :mod:`usabench.eval.aggregate`.

    The aggregation module is owned by the metrics/statistics agent and is
    imported *lazily* so this report package imports cleanly even before that
    module lands. We probe a few conventional entry-point names and fall back to
    reading any ``*.json`` aggregate files directly so the leaderboard is usable
    today.

    Args:
        results_dir: Directory holding scored-run aggregates (per-agent JSON or
            whatever ``eval.aggregate`` writes).

    Returns:
        A list of aggregate objects/dicts, one per agent, suitable for
        :func:`build_leaderboard`.
    """
    results_path = Path(results_dir)

    # eval.aggregate is owned by the metrics/statistics agent and may not exist
    # yet; import it lazily and treat it as Any so this module type-checks and
    # imports cleanly regardless. Once present, its loader functions are used.
    _aggregate: Any
    try:
        import importlib

        _aggregate = importlib.import_module("usabench.eval.aggregate")
    except Exception as exc:  # pragma: no cover - module not present yet
        _log.info("eval.aggregate unavailable; falling back to JSON scan", error=str(exc))
        _aggregate = None

    if _aggregate is not None:
        for fn_name in ("load_leaderboard_aggregates", "load_agent_aggregates", "aggregate_agents"):
            fn = getattr(_aggregate, fn_name, None)
            if callable(fn):
                _log.info("loading aggregates via eval.aggregate", fn=fn_name)
                result = fn(results_path)
                return list(result)

    # Fallback: each agent is a top-level *.json aggregate file.
    out: list[Any] = []
    if results_path.is_dir():
        for path in sorted(results_path.glob("*.json")):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:  # pragma: no cover - bad file
                _log.warning("skipping unparseable aggregate", path=str(path), error=str(exc))
    return out


# --------------------------------------------------------------------------- #
# Build + sort                                                                 #
# --------------------------------------------------------------------------- #


def _sort_key(row: LeaderboardRow) -> tuple[float, float]:
    """Sort key implementing ``docs/scoring.md`` §10.3.

    Usability Score descending (so we negate it), then *lower* AC as the
    tiebreak. Missing values sort last (treated as the worst).
    """
    us = row.usability_score
    ac = row.ac
    primary = -us if us is not None else float("inf")
    secondary = ac if ac is not None else float("inf")
    return (primary, secondary)


def build_leaderboard(
    aggregates: Iterable[AggregateLike],
    *,
    sort: bool = True,
) -> list[LeaderboardRow]:
    """Build sorted leaderboard rows from per-agent aggregates.

    Args:
        aggregates: An iterable of per-agent aggregate objects/dicts (as produced
            by :mod:`usabench.eval.aggregate` or :func:`load_agent_aggregates`).
        sort: When ``True`` (default), sort by Usability Score desc, ties broken
            by lower AC (``docs/scoring.md`` §10.3).

    Returns:
        A list of :class:`LeaderboardRow`, one per agent.
    """
    rows = [normalize_aggregate(a) for a in aggregates]
    if sort:
        rows.sort(key=_sort_key)
    _log.info("leaderboard built", n_rows=len(rows))
    return rows


# --------------------------------------------------------------------------- #
# Emit: JSONL                                                                  #
# --------------------------------------------------------------------------- #


def rows_to_jsonl(rows: Sequence[LeaderboardRow]) -> str:
    """Serialize rows to newline-delimited JSON (one row per line).

    Args:
        rows: Leaderboard rows to serialize.

    Returns:
        A JSONL string (trailing newline included when non-empty).
    """
    lines = [json.dumps(r.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) for r in rows]
    return ("\n".join(lines) + "\n") if lines else ""


# --------------------------------------------------------------------------- #
# Emit: Markdown                                                               #
# --------------------------------------------------------------------------- #


def _fmt_cell(row: LeaderboardRow, col: _Column) -> str:
    """Render one cell of the Markdown table for ``col`` of ``row``."""
    value = getattr(row, col.key, None)
    if col.kind == "str":
        return str(value) if value not in (None, "") else _NA
    if col.kind == "int":
        return str(int(value)) if value is not None else _NA
    if col.kind == "ci":
        ci = _as_ci(value)
        if ci is None:
            return _NA
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]"
    if value is None:
        return _NA
    if col.kind == "pct":
        return f"{float(value) * 100:.1f}%"
    if col.kind == "cost":
        return f"${float(value):.4f}"
    if col.kind == "sci":
        f = float(value)
        return f"{f:.3g}" if f else "0"
    # default: score
    return f"{float(value):.3f}"


def rows_to_markdown(
    rows: Sequence[LeaderboardRow],
    *,
    title: str = "Usability Benchmark Leaderboard",
    columns: Sequence[_Column] = LEADERBOARD_COLUMNS,
) -> str:
    """Render rows as a GitHub-flavored Markdown leaderboard table.

    Headline columns (``primary``) are starred. A short footer documents the
    sort order and the ``pass^k`` success definition, with the threshold read
    from the spec (single source of truth).

    Args:
        rows: Sorted leaderboard rows.
        title: H2 title placed above the table.
        columns: Column schema (defaults to :data:`LEADERBOARD_COLUMNS`).

    Returns:
        A Markdown document string.
    """
    spec = load_spec()
    tau_ga = spec.get("tau_ga")
    ci_level = spec.get("stats", {}).get("ci_level")

    headers = [(f"{c.header} ★" if c.primary else c.header) for c in columns]
    sep = ["---"] * len(columns)

    def _md_row(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    lines: list[str] = [f"## {title}", ""]
    lines.append(_md_row(headers))
    lines.append(_md_row(sep))
    for row in rows:
        lines.append(_md_row([_fmt_cell(row, c) for c in columns]))

    n = len(rows)
    n_seeds = rows[0].n_seeds if rows else 0
    lock = next((r.release_lock for r in rows if r.release_lock), "")
    lines += [
        "",
        f"_Sorted by Usability Score (desc); ties broken by lower AC. {n} agent(s)._  ",
        (
            f"_pass^n = pass^k at k=n_seeds; success = GA ≥ {tau_ga} AND hard_pass_frac = 1._  "
            if tau_ga is not None
            else ""
        ),
        (f"_CIs are cluster-bootstrap at level {ci_level}._  " if ci_level is not None else ""),
        "_Cost is split agent vs oracle (oracle is a fixed harness cost, not the agent's)._  ",
        (f"_n_seeds = {n_seeds}; release.lock = `{lock}`._" if lock else f"_n_seeds = {n_seeds}._"),
    ]
    return "\n".join(line for line in lines if line is not None) + "\n"


# --------------------------------------------------------------------------- #
# Emit: both, to disk                                                          #
# --------------------------------------------------------------------------- #


def write_leaderboard(
    rows: Sequence[LeaderboardRow],
    out_dir: str | Path,
    *,
    basename: str = "leaderboard",
    title: str = "Usability Benchmark Leaderboard",
) -> dict[str, Path]:
    """Write ``<basename>.jsonl`` and ``<basename>.md`` to ``out_dir``.

    Args:
        rows: Sorted leaderboard rows.
        out_dir: Output directory (created if absent).
        basename: Filename stem for both artifacts.
        title: Markdown table title.

    Returns:
        A mapping ``{"jsonl": Path, "markdown": Path}`` of the files written.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path / f"{basename}.jsonl"
    md_path = out_path / f"{basename}.md"
    jsonl_path.write_text(rows_to_jsonl(rows), encoding="utf-8")
    md_path.write_text(rows_to_markdown(rows, title=title), encoding="utf-8")
    _log.info("leaderboard written", jsonl=str(jsonl_path), markdown=str(md_path), n_rows=len(rows))
    return {"jsonl": jsonl_path, "markdown": md_path}
