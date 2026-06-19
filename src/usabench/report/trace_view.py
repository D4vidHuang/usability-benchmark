"""Render one episode's ``trace.jsonl`` to a standalone inspectable HTML page.

``trace.jsonl`` is the ONE canonical artifact (``DESIGN.md`` invariant 4): an
append-only, totally-ordered, hash-chained sequence of typed events. This module
turns a single episode's trace into a self-contained HTML page (no external CSS/JS,
no network) so a reviewer can eyeball *what the agent did and how much it leaned on
the human*:

* a header card with the run id, task, terminated reason, and headline totals
  (turns, oracle queries, assistance cost, acceptance);
* a vertical **timeline** of every event in ``seq`` order;
* oracle queries/responses colored by the **0-5 assistance severity**, with the
  color intensity driven by the convex severity weights in the single source of
  truth (:mod:`usabench.eval.spec`) -- so a sev-5 *takeover* is visually as loud as
  it is costly;
* acceptance **checkpoints** drawn as score milestones.

The renderer is a pure function of the trace (+ the frozen spec): it parses lines
with :func:`usabench.core.schema.parse_event`, never recomputes a score, and reads
every severity weight from the spec rather than hardcoding one.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from usabench.core.enums import InteractionType, Severity
from usabench.core.schema import TraceEnvelope, parse_event
from usabench.eval.spec import get_severity_weights, load_spec
from usabench.logging_setup import get_logger

__all__ = [
    "render_trace_html",
    "write_trace_html",
    "load_trace",
]

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #


def load_trace(trace_path: str | Path) -> list[TraceEnvelope]:
    """Parse a ``trace.jsonl`` file into typed, seq-ordered events.

    Blank lines are skipped. Each line is parsed via
    :func:`usabench.core.schema.parse_event`, giving fully-typed payloads. The
    result is sorted by ``seq`` so the timeline is total-ordered even if the file
    was concatenated out of order.

    Args:
        trace_path: Path to the episode's ``trace.jsonl``.

    Returns:
        The parsed events sorted by ``seq``.

    Raises:
        SchemaViolation: If a line carries an unknown event type or invalid
            payload (propagated from :func:`parse_event`).
    """
    path = Path(trace_path)
    events: list[TraceEnvelope] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(parse_event(json.loads(line)))
    events.sort(key=lambda e: e.seq)
    _log.info("trace loaded", path=str(path), n_events=len(events))
    return events


# --------------------------------------------------------------------------- #
# Severity color (intensity ∝ convex weight from the spec)                     #
# --------------------------------------------------------------------------- #

#: Fixed hue ramp for the 6 severity levels (green->red); intensity comes from
#: the spec weights, the *names/labels* come from :class:`Severity`.
_SEVERITY_HUES: tuple[str, ...] = (
    "#6b7280",  # 0 none      -- grey
    "#15803d",  # 1 trivial   -- green
    "#65a30d",  # 2 substantive -- lime
    "#ca8a04",  # 3 hint      -- amber
    "#ea580c",  # 4 partial   -- orange
    "#b91c1c",  # 5 takeover  -- red
)


def _severity_palette() -> list[dict[str, Any]]:
    """Build the per-severity palette: label + color + spec weight.

    Returns one dict per level 0..5 with ``level``, ``label`` (from
    :class:`Severity`), ``color`` (hue ramp), and ``weight`` (the convex weight
    read from the spec single-source-of-truth, used for the legend and to size
    the visual emphasis).
    """
    weights = get_severity_weights()
    palette: list[dict[str, Any]] = []
    for level in range(6):
        sev = Severity(level)
        palette.append(
            {
                "level": level,
                "label": sev.label,
                "color": _SEVERITY_HUES[level],
                "weight": weights[level],
            }
        )
    return palette


def _severity_of(env: TraceEnvelope) -> int | None:
    """Return the integer severity of an oracle_response event, else ``None``."""
    if env.type != InteractionType.ORACLE_RESPONSE.value:
        return None
    sev = getattr(env.payload, "severity", None)
    if sev is None:
        return None
    return int(sev)


# --------------------------------------------------------------------------- #
# Per-event rendering                                                          #
# --------------------------------------------------------------------------- #


def _esc(value: Any) -> str:
    """HTML-escape a value's string form (None -> empty)."""
    return html.escape("" if value is None else str(value))


def _trunc(text: str | None, limit: int = 600) -> str:
    """Escape and length-cap free text for inline display."""
    if not text:
        return ""
    t = text if len(text) <= limit else text[:limit] + " …"
    return _esc(t)


def _event_summary(env: TraceEnvelope) -> str:
    """Produce a short, human-readable summary line for one event's payload."""
    p = env.payload
    t = env.type
    if t == InteractionType.EPISODE_START.value:
        return f"task <b>{_esc(getattr(p, 'task_id', ''))}</b>, seed {_esc(getattr(p, 'seed', ''))}"
    if t == InteractionType.EPISODE_END.value:
        acc = "accepted" if getattr(p, "accepted", False) else "not accepted"
        return f"{_esc(getattr(p, 'terminated_reason', ''))} — {acc}"
    if t == InteractionType.AGENT_MESSAGE.value:
        return _trunc(getattr(p, "text", ""))
    if t == InteractionType.MESSAGE_TO_USER.value:
        return _trunc(getattr(p, "text", ""))
    if t == InteractionType.TOOL_CALL.value:
        return f"<code>{_esc(getattr(p, 'tool', ''))}</code> {_esc(getattr(p, 'args', ''))}"
    if t == InteractionType.CODE_RUN.value:
        ec = getattr(p, "exit_code", 0)
        cmd = getattr(p, "cmd", "") or getattr(p, "call_id", "")
        tag = "ok" if ec == 0 else f"exit {ec}"
        out = _trunc(getattr(p, "stdout_trunc", ""), 300)
        return f"<code>{_esc(cmd)}</code> <span class='pill'>{tag}</span>{('<br>' + out) if out else ''}"
    if t == InteractionType.FILE_EDIT.value:
        return (
            f"<code>{_esc(getattr(p, 'path', ''))}</code> "
            f"({_esc(getattr(p, 'op', ''))}, +{_esc(getattr(p, 'added', 0))}/-{_esc(getattr(p, 'removed', 0))})"
        )
    if t == InteractionType.AGENT_BLOCKED.value:
        return _trunc(getattr(p, "reason", "")) or "agent declared itself blocked"
    if t == InteractionType.HANDOFF_REQUEST.value:
        return _trunc(getattr(p, "reason", "")) or "agent requested human takeover"
    if t == InteractionType.ORACLE_QUERY.value:
        qc = _esc(getattr(p, "query_class", ""))
        return f"<span class='pill'>{qc}</span> {_trunc(getattr(p, 'text', ''))}"
    if t == InteractionType.ORACLE_RESPONSE.value:
        return _oracle_response_summary(p)
    if t == InteractionType.CHECKPOINT.value:
        ws = getattr(p, "weighted_score", 0.0)
        wv = "working" if getattr(p, "is_working_version", False) else "not-working"
        passed = getattr(p, "criteria_passed", 0)
        total = getattr(p, "criteria_total", 0)
        return f"score <b>{float(ws):.2f}</b> ({passed}/{total} criteria) — {wv}"
    if t == InteractionType.VERIFICATION_RUN.value:
        return f"rubric_score <b>{float(getattr(p, 'rubric_score', 0.0)):.2f}</b> ({_esc(getattr(p, 'trigger', ''))})"
    if t == InteractionType.BUDGET_DEBIT.value:
        return f"{_esc(getattr(p, 'kind', ''))} −{_esc(getattr(p, 'amount', 0))} ({_trunc(getattr(p, 'reason', ''), 120)})"
    if t == InteractionType.FINAL_ACCEPTANCE.value:
        acc = "ACCEPTED" if getattr(p, "accepted", False) else "REJECTED"
        return f"<b>{acc}</b> — weighted {float(getattr(p, 'weighted_score', 0.0)):.2f}"
    return _esc(t)


def _oracle_response_summary(payload: Any) -> str:
    """Summarize an oracle_response with its severity badge and revealed info."""
    sev = int(getattr(payload, "severity", 0))
    label = Severity(sev).label
    unsolicited = getattr(payload, "responds_to", None) is None
    badge = f"<span class='sevbadge' data-sev='{sev}'>sev {sev} · {_esc(label)}</span>"
    if unsolicited:
        badge += " <span class='pill warn'>unsolicited</span>"
    reveals = getattr(payload, "info_units_revealed", None) or getattr(payload, "reveals", None) or []
    reveal_html = ""
    if reveals:
        chips = " ".join(f"<span class='chip'>{_esc(r)}</span>" for r in reveals)
        reveal_html = f"<div class='reveals'>reveals: {chips}</div>"
    text = _trunc(getattr(payload, "text", ""))
    return f"{badge}<div class='otext'>{text}</div>{reveal_html}"


def _actor_class(env: TraceEnvelope) -> str:
    """Return a CSS class for an event based on its actor + type."""
    actor = env.actor.value if hasattr(env.actor, "value") else str(env.actor)
    base = f"actor-{actor}"
    if env.type in (InteractionType.ORACLE_QUERY.value, InteractionType.ORACLE_RESPONSE.value):
        return base + " oracle"
    if env.type == InteractionType.CHECKPOINT.value:
        return base + " checkpoint"
    if env.type in (
        InteractionType.EPISODE_START.value,
        InteractionType.EPISODE_END.value,
        InteractionType.FINAL_ACCEPTANCE.value,
    ):
        return base + " lifecycle"
    return base


def _render_event(env: TraceEnvelope) -> str:
    """Render a single timeline entry (one <li>)."""
    sev = _severity_of(env)
    sev_attr = f" data-sev='{sev}'" if sev is not None else ""
    turn = f"t{env.t_turn}" if env.t_turn is not None else "—"
    cls = _actor_class(env)
    summary = _event_summary(env)
    actor = _esc(env.actor.value if hasattr(env.actor, "value") else env.actor)
    return (
        f"<li class='evt {cls}'{sev_attr}>"
        f"<div class='meta'><span class='seq'>#{env.seq}</span>"
        f"<span class='turn'>{turn}</span>"
        f"<span class='type'>{_esc(env.type)}</span>"
        f"<span class='who'>{actor}</span></div>"
        f"<div class='body'>{summary}</div>"
        f"</li>"
    )


# --------------------------------------------------------------------------- #
# Header / totals (read off the trace; never recomputed scores)                #
# --------------------------------------------------------------------------- #


def _episode_totals(events: list[TraceEnvelope]) -> dict[str, Any]:
    """Collect headline totals straight off the trace for the header card.

    All quantities are *read* from logged events, not re-scored: oracle-query
    count, the severity histogram + spec-weighted assistance-cost points, the
    final checkpoint score, and the terminated reason / acceptance from the
    ``episode_end`` / ``final_acceptance`` events.
    """
    weights = get_severity_weights()
    sev_hist = {i: 0 for i in range(6)}
    n_queries = 0
    n_responses = 0
    assistance_points = 0.0
    last_checkpoint: float | None = None
    terminated_reason = "—"
    accepted: bool | None = None
    final_score: float | None = None
    run_id = ""
    task_id = ""
    seed: Any = ""
    max_turn = 0

    for env in events:
        run_id = run_id or env.run_id
        if env.t_turn is not None:
            max_turn = max(max_turn, env.t_turn)
        t = env.type
        p = env.payload
        if t == InteractionType.EPISODE_START.value:
            task_id = getattr(p, "task_id", "") or task_id
            seed = getattr(p, "seed", seed)
        elif t == InteractionType.ORACLE_QUERY.value:
            n_queries += 1
        elif t == InteractionType.ORACLE_RESPONSE.value:
            n_responses += 1
            sev = int(getattr(p, "severity", 0))
            sev = max(0, min(5, sev))
            sev_hist[sev] += 1
            assistance_points += weights[sev]
        elif t == InteractionType.CHECKPOINT.value:
            last_checkpoint = float(getattr(p, "weighted_score", 0.0))
        elif t == InteractionType.EPISODE_END.value:
            terminated_reason = str(getattr(p, "terminated_reason", "—"))
            accepted = bool(getattr(p, "accepted", False))
            fs = getattr(p, "final_weighted_score", None)
            if fs is not None:
                final_score = float(fs)
        elif t == InteractionType.FINAL_ACCEPTANCE.value:
            accepted = bool(getattr(p, "accepted", accepted))
            final_score = float(getattr(p, "weighted_score", final_score or 0.0))

    return {
        "run_id": run_id,
        "task_id": task_id,
        "seed": seed,
        "n_events": len(events),
        "n_turns": max_turn,
        "n_queries": n_queries,
        "n_responses": n_responses,
        "severity_histogram": sev_hist,
        "assistance_points": assistance_points,
        "max_severity": max((s for s, c in sev_hist.items() if c), default=0),
        "last_checkpoint": last_checkpoint,
        "final_score": final_score,
        "terminated_reason": terminated_reason,
        "accepted": accepted,
    }


# --------------------------------------------------------------------------- #
# Page assembly                                                                #
# --------------------------------------------------------------------------- #


def _css() -> str:
    """Return the embedded stylesheet (severity colors injected per-level)."""
    palette = _severity_palette()
    sev_rules = "\n".join(
        f".evt[data-sev='{p['level']}'] {{ border-left-color: {p['color']}; }}\n"
        f".sevbadge[data-sev='{p['level']}'] {{ background: {p['color']}; }}"
        for p in palette
    )
    return f"""
:root {{ --bg:#0f172a; --card:#1e293b; --ink:#e2e8f0; --muted:#94a3b8; --line:#334155; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; padding:24px; background:var(--bg); color:var(--ink);
  font:14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }}
h1 {{ font-size:18px; margin:0 0 4px; }}
.sub {{ color:var(--muted); margin-bottom:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:14px 18px; margin-bottom:18px; }}
.totals {{ display:flex; flex-wrap:wrap; gap:18px; }}
.totals .kv {{ min-width:120px; }}
.totals .kv b {{ display:block; font-size:20px; }}
.totals .kv span {{ color:var(--muted); font-size:12px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; }}
.legend .li {{ display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); }}
.legend .sw {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
ul.timeline {{ list-style:none; margin:0; padding:0; position:relative; }}
li.evt {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--line);
  border-radius:8px; padding:8px 12px; margin:8px 0; }}
li.evt.oracle {{ background:#1f2937; }}
li.evt.checkpoint {{ border-left-color:#38bdf8; }}
li.evt.lifecycle {{ border-left-color:#a78bfa; }}
.evt .meta {{ display:flex; gap:10px; font-size:11px; color:var(--muted); margin-bottom:2px; }}
.evt .meta .type {{ color:var(--ink); }}
.evt .body {{ word-break:break-word; }}
.evt .otext {{ margin-top:4px; }}
code {{ background:#0b1220; padding:1px 5px; border-radius:4px; }}
.pill {{ background:#334155; border-radius:10px; padding:0 7px; font-size:11px; }}
.pill.warn {{ background:#7c2d12; }}
.sevbadge {{ color:#fff; border-radius:10px; padding:1px 8px; font-size:11px; font-weight:700; }}
.chip {{ background:#0b3b2e; border:1px solid #115e45; border-radius:10px; padding:0 7px; font-size:11px; }}
.reveals {{ margin-top:4px; color:var(--muted); font-size:12px; }}
.ok {{ color:#34d399; }} .bad {{ color:#f87171; }}
{sev_rules}
"""


def _legend_html() -> str:
    """Render the severity legend (label + spec weight per level)."""
    palette = _severity_palette()
    items = "".join(
        f"<span class='li'><span class='sw' style='background:{p['color']}'></span>"
        f"{p['level']} {_esc(p['label'])} (w={_esc(p['weight'])})</span>"
        for p in palette
    )
    return f"<div class='legend'>{items}</div>"


def _header_html(totals: dict[str, Any]) -> str:
    """Render the header card with run identity + headline totals."""
    acc = totals["accepted"]
    acc_html = (
        "<b class='ok'>accepted</b>"
        if acc
        else ("<b class='bad'>not accepted</b>" if acc is False else "<b>—</b>")
    )
    score = totals["final_score"]
    score_str = f"{score:.2f}" if score is not None else "—"
    cp = totals["last_checkpoint"]
    cp_str = f"{cp:.2f}" if cp is not None else "—"
    return f"""
<div class="card">
  <div class="totals">
    <div class="kv"><b>{totals['n_turns']}</b><span>turns</span></div>
    <div class="kv"><b>{totals['n_queries']}</b><span>oracle queries</span></div>
    <div class="kv"><b>{totals['n_responses']}</b><span>oracle responses</span></div>
    <div class="kv"><b>{totals['assistance_points']:.0f}</b><span>assistance cost (pts)</span></div>
    <div class="kv"><b>{totals['max_severity']}</b><span>max severity</span></div>
    <div class="kv"><b>{cp_str}</b><span>last checkpoint</span></div>
    <div class="kv"><b>{score_str}</b><span>final weighted</span></div>
    <div class="kv"><b>{_esc(totals['terminated_reason'])}</b><span>terminated</span></div>
    <div class="kv">{acc_html}<span>acceptance</span></div>
  </div>
  {_legend_html()}
</div>
"""


def render_trace_html(
    events_or_path: str | Path | list[TraceEnvelope],
    *,
    title: str | None = None,
) -> str:
    """Render an episode trace to a standalone HTML document string.

    The output is fully self-contained (inline CSS, no scripts, no external
    requests), so it can be opened directly or attached to a report. Oracle
    events are color-coded by the 0-5 assistance severity, with the legend
    showing the convex weight each level carries in the score (read from the
    spec single source of truth).

    Args:
        events_or_path: Either a path to ``trace.jsonl`` or a pre-parsed list of
            :class:`TraceEnvelope` events.
        title: Optional page title; defaults to ``run <run_id>``.

    Returns:
        A complete HTML document as a string.
    """
    if isinstance(events_or_path, (str, Path)):
        events = load_trace(events_or_path)
    else:
        events = sorted(events_or_path, key=lambda e: e.seq)

    totals = _episode_totals(events)
    # Confirm the spec is loadable so a broken spec fails loudly here, not silently.
    _ = load_spec()
    page_title = title or f"trace · {totals['run_id'] or 'episode'}"
    subtitle = (
        f"task <b>{_esc(totals['task_id'])}</b> · seed {_esc(totals['seed'])} · "
        f"run <code>{_esc(totals['run_id'])}</code> · {totals['n_events']} events"
    )
    timeline = "\n".join(_render_event(e) for e in events)

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(page_title)}</title>
<style>{_css()}</style>
</head><body>
<h1>{_esc(page_title)}</h1>
<div class="sub">{subtitle}</div>
{_header_html(totals)}
<ul class="timeline">
{timeline}
</ul>
</body></html>
"""


def write_trace_html(
    trace_path: str | Path,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Render ``trace_path`` and write the HTML to ``out_path``.

    Args:
        trace_path: Path to the episode's ``trace.jsonl``.
        out_path: Destination ``.html`` file (parent dirs created).
        title: Optional page title.

    Returns:
        The written output path.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    html_doc = render_trace_html(trace_path, title=title)
    out.write_text(html_doc, encoding="utf-8")
    _log.info("trace HTML written", trace=str(trace_path), out=str(out))
    return out
