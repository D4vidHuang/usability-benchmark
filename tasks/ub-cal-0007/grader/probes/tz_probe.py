#!/usr/bin/env python3
"""Property probe for timezone normalization (criterion AC5 of ub-cal-0007).

The check is a *property*, not a golden value: a correct tool must normalize all
event start times to a single report timezone **before** bucketing them by
weekday. The fixture is engineered so this property is observable and
load-bearing -- the cross-timezone "Quarterly Review with NY Office" is scheduled
``2026-06-03 20:00`` America/New_York, which is ``2026-06-04 02:00``
Europe/Amsterdam. A tool that normalizes correctly attributes that hour to
**Thursday**; a tool that reads the New-York wall clock attributes it to
**Wednesday**, which flips the busiest day.

The probe exercises the property two ways and passes if *either* establishes
correct normalization (tools differ in which knobs they expose -- the output
format is itself an ambiguity point):

1. **Default-report-tz path.** Run the tool with no timezone flag. If it reports a
   busiest day of ``Thu`` (the TZ-normalized answer) it has normalized; ``Wed``
   indicates naive wall-clock reading.

2. **Override path.** If the tool accepts a ``--tz`` override (AP2 gold: "allow
   ``--tz``"), run it under two different report zones and assert the per-weekday
   distribution *shifts coherently* -- i.e. the same UTC instants are re-bucketed,
   not re-parsed as floating local times.

Exit code ``0`` = property holds (AC5 pass); non-zero = fail. A one-line JSON
rationale is printed to stdout either way.

Stdlib only, so it runs hermetically in the verification sandbox.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

WEEKDAY_LONG = {
    "monday": "Mon", "tuesday": "Tue", "wednesday": "Wed", "thursday": "Thu",
    "friday": "Fri", "saturday": "Sat", "sunday": "Sun",
}
# The TZ-normalized (correct) busiest day vs. the naive-wall-clock (wrong) answer.
EXPECTED_BUSIEST_NORMALIZED = "Thu"
NAIVE_BUSIEST = "Wed"


def _run(artifact: Path, args: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
    """Run ``python <artifact> <args>`` returning (exit_code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            [sys.executable, str(artifact), *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as exc:  # pragma: no cover - defensive
        return 125, "", f"{type(exc).__name__}: {exc}"


def _busiest_weekday(stdout: str) -> str | None:
    """Extract the reported busiest weekday from JSON or human-readable stdout."""
    for blob in re.findall(r"\{.*\}", stdout, flags=re.DOTALL):
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        bd = obj.get("busiest_day")
        if isinstance(bd, dict) and isinstance(bd.get("weekday"), str):
            return bd["weekday"][:3].title()
        if isinstance(bd, str):
            return bd[:3].title()
    for line in stdout.splitlines():
        low = line.lower()
        if "busiest" in low or "busy" in low:
            for long, short in WEEKDAY_LONG.items():
                if long in low or short.lower() in low:
                    return short
    return None


def _weekday_distribution(stdout: str) -> dict[str, float]:
    """Best-effort per-weekday hour distribution from a JSON object on stdout."""
    for blob in re.findall(r"\{.*\}", stdout, flags=re.DOTALL):
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        for key in ("per_weekday_hours", "per_weekday", "weekday_hours", "by_weekday"):
            d = obj.get(key)
            if isinstance(d, dict):
                out: dict[str, float] = {}
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        out[str(k)[:3].title()] = float(v)
                if out:
                    return out
    return {}


def probe(artifact: Path, ics: Path) -> tuple[bool, dict[str, object]]:
    """Run the two normalization checks. Returns (passed, rationale)."""
    rationale: dict[str, object] = {}

    # --- Path 1: default report tz, inspect the busiest day. ---
    code, out, err = _run(artifact, [str(ics)])
    if code != 0:
        return False, {"reason": "artifact crashed on default run", "exit": code, "stderr": err[:200]}
    busiest = _busiest_weekday(out)
    rationale["default_busiest_day"] = busiest
    if busiest == EXPECTED_BUSIEST_NORMALIZED:
        rationale["path1"] = "pass: busiest day == Thu (TZ-normalized)"
        return True, rationale
    if busiest == NAIVE_BUSIEST:
        rationale["path1"] = "fail-signal: busiest day == Wed (naive NY wall clock, not normalized)"

    # --- Path 2: --tz override coherence. ---
    code_a, out_a, _ = _run(artifact, [str(ics), "--tz", "Europe/Amsterdam"])
    code_b, out_b, _ = _run(artifact, [str(ics), "--tz", "America/New_York"])
    if code_a == 0 and code_b == 0:
        dist_a = _weekday_distribution(out_a)
        dist_b = _weekday_distribution(out_b)
        rationale["tz_ams"] = dist_a
        rationale["tz_ny"] = dist_b
        if dist_a and dist_b:
            total_a = round(sum(dist_a.values()), 3)
            total_b = round(sum(dist_b.values()), 3)
            # Same instants -> identical TOTAL hours under any report tz...
            totals_match = abs(total_a - total_b) <= 0.05
            # ...but the per-weekday buckets must differ (re-bucketing, not re-parsing).
            buckets_differ = dist_a != dist_b
            rationale["totals_match_across_tz"] = totals_match
            rationale["buckets_shift_across_tz"] = buckets_differ
            if totals_match and buckets_differ:
                rationale["path2"] = "pass: total hours invariant, weekday buckets shift coherently"
                return True, rationale
            rationale["path2"] = "fail: --tz did not re-bucket the same instants coherently"
        else:
            rationale["path2"] = "inconclusive: no machine-readable weekday distribution"
    else:
        rationale["path2"] = "inconclusive: tool does not accept --tz override"

    return False, rationale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Timezone-normalization property probe (AC5).")
    parser.add_argument("--artifact", type=Path, required=True, help="Agent CLI tool (.py).")
    parser.add_argument("--ics", type=Path, required=True, help="Fixture .ics path.")
    args = parser.parse_args(argv)

    if not args.artifact.exists():
        print(json.dumps({"passed": False, "reason": f"artifact not found: {args.artifact}"}))
        return 2

    passed, rationale = probe(args.artifact, args.ics)
    print(json.dumps({"passed": passed, **rationale}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
