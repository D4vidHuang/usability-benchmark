#!/usr/bin/env python3
"""Deterministic grader for task ``ub-cal-0007`` (calendar workload summarizer).

This is the task's ``verification.harness_entry``. It is a *pure, deterministic*
function of (the agent's artifact, the frozen fixture, the committed gold) and
emits a per-criterion pass/fail report as JSON to stdout. It has **no third-party
dependencies** (stdlib only) so it runs hermetically in the verification sandbox
and so the gold remains reproducible byte-for-byte (``docs/tasks.md`` §8.3).

Pipeline per invocation::

    python grade.py --artifact <agent_tool.py> [--ics <sample_calendar.ics>]

1. Invoke the agent artifact on the fixture and on ``--help`` and capture stdout.
2. Extract the agent's summary numbers tolerantly: prefer machine-readable JSON
   (if the tool honors a ``--json`` flag or simply prints a JSON object), else
   scrape labelled numbers from the human-readable table.
3. Run the timezone property probe (``probes/tz_probe.py``).
4. Compare every extracted value against ``gold/cal_summary_gold.json`` within the
   documented tolerances and emit one ``CriterionResult``-shaped record per AC.
5. Print a JSON report ``{"task_id", "criteria":[...], "weighted_score", "accepted"}``.

The grader's own exit code is ``0`` on a successful *grading run* regardless of
whether the artifact passed; a non-zero exit means the grader itself failed
(e.g. the gold file is missing). This matches the harness contract: pass/fail
lives in the JSON report, not in the grader exit code.

An internal :func:`reference_summary` re-derives the gold from the fixture using
only stdlib; it is the gold-reproduction path used by ``qc/`` and lets the grader
self-check without the agent (``python grade.py --self-test``).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# --------------------------------------------------------------------------- #
# Paths (resolved relative to this file so the grader is location-independent). #
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parent
DEFAULT_ICS = TASK_DIR / "env" / "sample_calendar.ics"
GOLD_PATH = HERE / "gold" / "cal_summary_gold.json"
TZ_PROBE = HERE / "probes" / "tz_probe.py"

REPORT_TZ = "Europe/Amsterdam"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Per-criterion weights mirror task.json hidden.acceptance_criteria. They are
# duplicated here ONLY for the standalone scoring convenience of the grader; the
# authoritative weights live in task.json. The harness passes the canonical
# weights in; when run standalone these defaults are used.
DEFAULT_WEIGHTS = {
    "AC1": 1.0,
    "AC2": 2.0,
    "AC3": 2.0,
    "AC4": 1.0,
    "AC5": 1.5,
    "AC6": 1.0,
    "AC7": 0.5,
    "AC8": 1.0,
}


# --------------------------------------------------------------------------- #
# Minimal, dependency-free RFC5545 reader (clean-room; reference path for gold). #
# --------------------------------------------------------------------------- #
def _parse_vtimezones(text: str) -> dict[str, tzinfo]:
    """Return TZID -> tzinfo using :mod:`zoneinfo`, with a fixed-offset fallback.

    We trust the IANA database via :class:`zoneinfo.ZoneInfo` for any TZID it
    knows (the fixture uses canonical IANA ids). Unknown TZIDs fall back to UTC.
    """
    tzids = set(re.findall(r"^TZID:(.+)$", text, flags=re.MULTILINE))
    out: dict[str, tzinfo] = {}
    for tzid in tzids:
        tzid = tzid.strip()
        try:
            out[tzid] = ZoneInfo(tzid)
        except Exception:  # pragma: no cover - defensive
            out[tzid] = UTC
    return out


def _unfold(text: str) -> list[str]:
    """Unfold RFC5545 line continuations (a line starting with space/tab)."""
    lines: list[str] = []
    for raw in text.splitlines():
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _parse_dt(value: str, params: dict[str, str], tzmap: dict[str, tzinfo]) -> tuple[Any, bool]:
    """Parse a DTSTART/DTEND value. Returns (datetime|date, is_all_day)."""
    if params.get("VALUE") == "DATE" or re.fullmatch(r"\d{8}", value):
        return date(int(value[0:4]), int(value[4:6]), int(value[6:8])), True
    m = re.fullmatch(r"(\d{8})T(\d{6})(Z)?", value)
    if not m:
        raise ValueError(f"unparseable datetime: {value!r}")
    d, t, z = m.groups()
    dt = datetime(
        int(d[0:4]), int(d[4:6]), int(d[6:8]),
        int(t[0:2]), int(t[2:4]), int(t[4:6]),
    )
    if z == "Z":
        return dt.replace(tzinfo=UTC), False
    tzid = params.get("TZID")
    if tzid and tzid in tzmap:
        return dt.replace(tzinfo=tzmap[tzid]), False
    # Floating time -> interpret in the report timezone.
    return dt.replace(tzinfo=ZoneInfo(REPORT_TZ)), False


def _expand_rrule(start: datetime, end: datetime, rrule: str) -> list[tuple[datetime, datetime]]:
    """Expand a small subset of RRULE (FREQ=WEEKLY;BYDAY=..;COUNT=..) used by the fixture.

    This deliberately supports only the recurrence shapes present in the frozen
    fixture; a real agent solution is expected to use a full library. The grader
    only needs to reproduce the gold, so a minimal expander is correct here.
    """
    parts = dict(kv.split("=", 1) for kv in rrule.split(";") if "=" in kv)
    freq = parts.get("FREQ")
    count = int(parts.get("COUNT", "1"))
    occ: list[tuple[datetime, datetime]] = []
    dur = end - start
    if freq == "WEEKLY":
        interval = int(parts.get("INTERVAL", "1"))
        for i in range(count):
            s = start + timedelta(weeks=i * interval)
            occ.append((s, s + dur))
    elif freq == "DAILY":
        interval = int(parts.get("INTERVAL", "1"))
        for i in range(count):
            s = start + timedelta(days=i * interval)
            occ.append((s, s + dur))
    else:  # FREQ unsupported by this minimal expander -> single occurrence.
        occ.append((start, end))
    return occ


def _read_events(ics_path: Path) -> list[dict[str, Any]]:
    """Parse the fixture into a flat list of event dicts (stdlib only)."""
    text = ics_path.read_text(encoding="utf-8")
    tzmap = _parse_vtimezones(text)
    lines = _unfold(text)
    events: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    in_tz = False
    for line in lines:
        if line == "BEGIN:VTIMEZONE":
            in_tz = True
            continue
        if line == "END:VTIMEZONE":
            in_tz = False
            continue
        if in_tz:
            continue
        if line == "BEGIN:VEVENT":
            cur = {"attendees": 0, "rrule": None, "status": None, "conf": False}
            continue
        if line == "END:VEVENT":
            if cur is not None:
                events.append(cur)
            cur = None
            continue
        if cur is None or ":" not in line:
            continue
        name_params, _, value = line.partition(":")
        name, *param_list = name_params.split(";")
        params = {}
        for p in param_list:
            if "=" in p:
                k, v = p.split("=", 1)
                params[k.upper()] = v
        name = name.upper()
        if name == "DTSTART":
            cur["start"], cur["all_day"] = _parse_dt(value, params, tzmap)
        elif name == "DTEND":
            cur["end"], _ = _parse_dt(value, params, tzmap)
        elif name == "RRULE":
            cur["rrule"] = value
        elif name == "STATUS":
            cur["status"] = value.upper()
        elif name == "ATTENDEE":
            cur["attendees"] += 1
        elif name in ("SUMMARY",):
            cur["summary"] = value
        elif name in ("LOCATION", "X-CONFERENCE-URL", "CONFERENCE", "URL"):
            # A conferencing link is an actual URL in a URL-bearing field. We do
            # NOT scan DESCRIPTION prose (the word "meeting" would false-match a
            # naive host regex). Require a real http(s) URL or a conferencing host
            # that appears with a URL-like path.
            if re.search(r"https?://(\S*\.)?(zoom|meet|teams|webex|whereby|jitsi)\b", value, re.IGNORECASE) \
                    or re.search(r"https?://\S+", value):
                cur["conf"] = True
    return events


def reference_summary(ics_path: Path = DEFAULT_ICS) -> dict[str, Any]:
    """Re-derive the gold summary from the fixture using stdlib only.

    This is the canonical reference path: the committed ``cal_summary_gold.json``
    was produced by this function and ``qc`` re-runs it to prove gold
    reproducibility. Semantics (RRULE expansion, TZ normalization, all-day &
    CANCELLED exclusion, the meeting heuristic) are authored clean-room.

    Args:
        ics_path: Path to the frozen ``.ics`` fixture.

    Returns:
        A summary dict with the same shape as the committed gold.
    """
    report_tz = ZoneInfo(REPORT_TZ)
    events = _read_events(ics_path)

    total = 0.0
    durations: list[float] = []
    per_wd = {w: 0.0 for w in WEEKDAYS}
    event_count = 0
    timed_count = 0
    allday_count = 0
    cancelled = 0
    meeting_count = 0
    meeting_hours = 0.0
    non_meeting_count = 0
    non_meeting_hours = 0.0

    for ev in events:
        if ev.get("status") == "CANCELLED":
            cancelled += 1
            continue
        if ev.get("all_day"):
            allday_count += 1
            event_count += 1
            continue
        start: datetime = ev["start"]
        end: datetime = ev["end"]
        occurrences = (
            _expand_rrule(start, end, ev["rrule"]) if ev.get("rrule") else [(start, end)]
        )
        is_meeting = (ev.get("attendees", 0) >= 2) or bool(ev.get("conf"))
        for s, e in occurrences:
            s = s.astimezone(report_tz)
            e = e.astimezone(report_tz)
            hours = (e - s).total_seconds() / 3600.0
            total += hours
            durations.append(hours)
            per_wd[WEEKDAYS[s.weekday()]] += hours
            event_count += 1
            timed_count += 1
            if is_meeting:
                meeting_count += 1
                meeting_hours += hours
            else:
                non_meeting_count += 1
                non_meeting_hours += hours

    durations.sort()
    n = len(durations)
    mean = (sum(durations) / n) if n else 0.0
    median = 0.0
    if n:
        median = durations[n // 2] if n % 2 else (durations[n // 2 - 1] + durations[n // 2]) / 2
    busiest_wd = max(per_wd, key=lambda k: per_wd[k]) if any(per_wd.values()) else "Mon"

    return {
        "totals": {
            "total_scheduled_hours": round(total, 4),
            "event_count": event_count,
            "timed_event_count": timed_count,
            "allday_event_count": allday_count,
            "cancelled_excluded_count": cancelled,
        },
        "durations_hours": {"mean": round(mean, 4), "median": round(median, 4)},
        "meeting_split": {
            "meeting_count": meeting_count,
            "meeting_hours": round(meeting_hours, 4),
            "non_meeting_count": non_meeting_count,
            "non_meeting_hours": round(non_meeting_hours, 4),
        },
        "per_weekday_hours": {k: round(v, 4) for k, v in per_wd.items()},
        "busiest_day": {"weekday": busiest_wd, "hours": round(per_wd[busiest_wd], 4)},
    }


# --------------------------------------------------------------------------- #
# Tolerant extraction of an agent artifact's reported summary.                  #
# --------------------------------------------------------------------------- #
def _run_artifact(artifact: Path, args: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
    """Run ``python <artifact> <args>`` and capture (exit_code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            [sys.executable, str(artifact), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(TASK_DIR),
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as exc:  # pragma: no cover - defensive
        return 125, "", f"{type(exc).__name__}: {exc}"


def _scrape_number(stdout: str, *keywords: str) -> float | None:
    """Find the first number on a line mentioning any of ``keywords`` (case-insensitive)."""
    for line in stdout.splitlines():
        low = line.lower()
        if any(k.lower() in low for k in keywords):
            m = re.search(r"(-?\d+(?:\.\d+)?)", line.replace(",", ""))
            if m:
                return float(m.group(1))
    return None


def _scrape_weekday(stdout: str, *keywords: str) -> str | None:
    """Find a weekday token on a line mentioning any of ``keywords``."""
    longnames = {
        "monday": "Mon", "tuesday": "Tue", "wednesday": "Wed", "thursday": "Thu",
        "friday": "Fri", "saturday": "Sat", "sunday": "Sun",
    }
    for line in stdout.splitlines():
        low = line.lower()
        if any(k.lower() in low for k in keywords):
            for long, short in longnames.items():
                if long in low or short.lower() in low:
                    return short
    return None


def extract_agent_summary(artifact: Path, ics_path: Path) -> dict[str, Any]:
    """Best-effort extraction of the agent tool's summary into the gold shape.

    Prefers a JSON object on stdout (whether emitted by default or under a
    ``--json`` flag); otherwise scrapes labelled numbers from the human table.
    The output format is itself an ambiguity point (AP5, low severity), so the
    grader is tolerant: it accepts either medium.
    """
    summary: dict[str, Any] = {}
    # Try --json first, then a bare run.
    for extra in (["--json"], []):
        code, out, _err = _run_artifact(artifact, [str(ics_path), *extra])
        # Look for a JSON object anywhere in stdout.
        for blob in re.findall(r"\{.*\}", out, flags=re.DOTALL):
            try:
                summary["_json"] = json.loads(blob)
                break
            except Exception:
                continue
        summary["_exit_default"] = code if not extra else summary.get("_exit_default")
        summary.setdefault("_stdout", out)
        if "_json" in summary:
            break

    out = summary.get("_stdout", "")
    summary["total_scheduled_hours"] = _scrape_number(out, "total", "scheduled hours", "hours")
    summary["event_count"] = _scrape_number(out, "events", "event count", "total events")
    summary["allday"] = _scrape_number(out, "all-day", "all day", "allday")
    summary["meeting_count"] = _scrape_number(out, "meeting")
    summary["busiest_day"] = _scrape_weekday(out, "busiest", "busy")
    return summary


def _get(summary: dict[str, Any], json_path: list[str], scrape_key: str) -> float | None:
    """Read a value from the agent summary, preferring the parsed JSON object."""
    j = summary.get("_json")
    if isinstance(j, dict):
        cur: Any = j
        for key in json_path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                cur = None
                break
        if isinstance(cur, (int, float)):
            return float(cur)
    val = summary.get(scrape_key)
    return float(val) if isinstance(val, (int, float)) else None


# --------------------------------------------------------------------------- #
# Criterion checks.                                                             #
# --------------------------------------------------------------------------- #
def _within_pct(got: float | None, gold: float, tol_pct: float) -> bool:
    if got is None:
        return False
    if gold == 0:
        return abs(got) <= 1e-9
    return abs(got - gold) / abs(gold) * 100.0 <= tol_pct


def _within_abs(got: float | None, gold: float, tol_abs: float) -> bool:
    return got is not None and abs(got - gold) <= tol_abs


def _criterion(cid: str, passed: bool, detail: str, weight: float) -> dict[str, Any]:
    return {
        "id": cid,
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "weight": weight,
        "channel": "func",
        "detail": detail,
    }


def grade(artifact: Path, ics_path: Path, gold: dict[str, Any],
          weights: dict[str, float]) -> dict[str, Any]:
    """Run every check and return the full per-criterion report."""
    criteria: list[dict[str, Any]] = []

    # AC1: parses + runs without crashing.
    code, _out, err = _run_artifact(artifact, [str(ics_path)])
    criteria.append(_criterion(
        "AC1", code == 0,
        f"exit_code={code}" + (f" stderr={err[:200]!r}" if code != 0 else ""),
        weights["AC1"],
    ))

    agent = extract_agent_summary(artifact, ics_path)

    # AC2: total scheduled hours within +/-2%.
    g_hours = gold["totals"]["total_scheduled_hours"]
    got_hours = _get(agent, ["totals", "total_scheduled_hours"], "total_scheduled_hours")
    criteria.append(_criterion(
        "AC2", _within_pct(got_hours, g_hours, gold["tolerances"]["total_scheduled_hours"]["tol_pct"]),
        f"got={got_hours} gold={g_hours} tol=2%", weights["AC2"],
    ))

    # AC3: recurring events expanded -> event_count matches exactly.
    g_count = gold["totals"]["event_count"]
    got_count = _get(agent, ["totals", "event_count"], "event_count")
    criteria.append(_criterion(
        "AC3", _within_abs(got_count, g_count, gold["tolerances"]["event_count"]["tol_abs"]),
        f"got={got_count} gold={g_count} (RRULE must expand to 4 standups)", weights["AC3"],
    ))

    # AC4: all-day excluded from hours (event_count includes it but hours do not).
    g_allday = gold["totals"]["allday_event_count"]
    got_allday = _get(agent, ["totals", "allday_event_count"], "allday")
    # Pass if the all-day count is reported AND the hour total already excludes it
    # (AC2 passing implies the 24h all-day block was not summed into hours).
    allday_ok = _within_abs(got_allday, g_allday, 0) and _within_pct(
        got_hours, g_hours, gold["tolerances"]["total_scheduled_hours"]["tol_pct"]
    )
    criteria.append(_criterion(
        "AC4", allday_ok,
        f"allday_reported={got_allday} gold={g_allday}; hours_exclude_allday={_within_pct(got_hours, g_hours, 2.0)}",
        weights["AC4"],
    ))

    # AC5: times normalized to one timezone -> property probe.
    probe_pass, probe_detail = run_tz_probe(artifact, ics_path)
    criteria.append(_criterion("AC5", probe_pass, probe_detail, weights["AC5"]))

    # AC6: CANCELLED excluded. With one cancelled 1h meeting, including it would
    # push hours to 8.0 (+14%), failing the 2% tol. So AC2 passing => AC6 holds.
    cancelled_ok = _within_pct(got_hours, g_hours, gold["tolerances"]["total_scheduled_hours"]["tol_pct"])
    criteria.append(_criterion(
        "AC6", cancelled_ok,
        f"hours={got_hours} would be {g_hours + 1.0} if the cancelled 1h event were counted",
        weights["AC6"],
    ))

    # AC7: --help with a documented invocation.
    hc, hout, _herr = _run_artifact(artifact, ["--help"])
    help_ok = hc == 0 and bool(re.search(r"usage", hout, re.IGNORECASE))
    criteria.append(_criterion(
        "AC7", help_ok, f"--help exit={hc} contains_usage={bool(re.search('usage', hout, re.I))}",
        weights["AC7"],
    ))

    # AC8: busiest weekday matches gold exactly (TZ-sensitive).
    g_busy = gold["busiest_day"]["weekday"]
    got_busy = None
    j = agent.get("_json")
    if isinstance(j, dict):
        bd = j.get("busiest_day")
        if isinstance(bd, dict):
            got_busy = bd.get("weekday")
        elif isinstance(bd, str):
            got_busy = bd
    if got_busy is None:
        got_busy = agent.get("busiest_day")
    busy_ok = isinstance(got_busy, str) and got_busy[:3].title() == g_busy
    criteria.append(_criterion(
        "AC8", busy_ok, f"got={got_busy} gold={g_busy} (TZ-sensitive: wrong TZ -> Wed)",
        weights["AC8"],
    ))

    total_w = sum(weights[c["id"]] for c in criteria)
    earned = sum(c["weight"] * c["score"] for c in criteria)
    weighted = earned / total_w if total_w else 0.0
    return {
        "task_id": "ub-cal-0007",
        "criteria": criteria,
        "weighted_score": round(weighted, 6),
        "accepted": weighted >= 0.80,
    }


def run_tz_probe(artifact: Path, ics_path: Path) -> tuple[bool, str]:
    """Invoke ``probes/tz_probe.py`` and return (passed, detail)."""
    if not TZ_PROBE.exists():
        return False, "tz_probe.py missing"
    proc = subprocess.run(
        [sys.executable, str(TZ_PROBE), "--artifact", str(artifact), "--ics", str(ics_path)],
        capture_output=True, text=True, timeout=90,
    )
    detail = (proc.stdout.strip() or proc.stderr.strip())[:300]
    return proc.returncode == 0, detail


# --------------------------------------------------------------------------- #
# CLI.                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grader for ub-cal-0007.")
    parser.add_argument("--artifact", type=Path, help="Path to the agent's CLI tool (a .py file).")
    parser.add_argument("--ics", type=Path, default=DEFAULT_ICS, help="Path to the fixture .ics.")
    parser.add_argument("--self-test", action="store_true",
                        help="Re-derive the gold from the fixture and diff against the committed gold.")
    args = parser.parse_args(argv)

    if not GOLD_PATH.exists():
        print(json.dumps({"error": f"gold not found: {GOLD_PATH}"}), file=sys.stderr)
        return 2
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    if args.self_test:
        ref = reference_summary(args.ics)
        ok = True
        diffs: dict[str, Any] = {}
        for k in ("total_scheduled_hours", "event_count", "allday_event_count", "cancelled_excluded_count"):
            if abs(ref["totals"][k] - gold["totals"][k]) > 1e-9:
                ok = False
                diffs[k] = {"reference": ref["totals"][k], "gold": gold["totals"][k]}
        if ref["busiest_day"]["weekday"] != gold["busiest_day"]["weekday"]:
            ok = False
            diffs["busiest_day"] = {"reference": ref["busiest_day"], "gold": gold["busiest_day"]}
        print(json.dumps({"self_test_passed": ok, "diffs": diffs, "reference": ref}, indent=2))
        return 0 if ok else 1

    if not args.artifact:
        parser.error("--artifact is required unless --self-test is given")
    if not args.artifact.exists():
        print(json.dumps({"error": f"artifact not found: {args.artifact}"}), file=sys.stderr)
        return 2

    report = grade(args.artifact, args.ics, gold, dict(DEFAULT_WEIGHTS))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
