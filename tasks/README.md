# `tasks/` — the benchmark task corpus

This directory holds the **frozen benchmark content**: the under-specified, interactive
tasks the agent-under-test must solve, plus the oracle-private gold knowledge and the
deterministic graders that score them. It is the data side of *usability-benchmark*
(import name `usabench`). The harness, oracle, and scorer are owned by other workstreams;
this directory owns the *content* they consume.

Everything here is authored to the design in [`docs/tasks.md`](../docs/tasks.md) and
validated against [`schemas/task.schema.json`](../schemas/task.schema.json), which mirrors
the pydantic model `usabench.core.schema.Task`.

---

## 1. Layout

```
tasks/
  README.md                     # this file
  curated/
    v0_smoke.jsonl              # 2–3 tiny, stdlib-only tasks for the FakeLLM smoke path
    v1.jsonl                    # starter set: ub-cal-0007 (full) + hand-written stubs
  ub-cal-0007/                  # the fully worked example (one task = one directory)
    task.json                   # the task record (agent-visible + hidden gold)
    env/
      Dockerfile                # frozen, hermetic build (network DENY at run time)
      requirements.allowed.txt  # the ONLY deps the agent may install
      sample_calendar.ics       # synthetic fixture exercising every acceptance criterion
    grader/
      grade.py                  # deterministic grader entrypoint (stdlib only)
      gold/cal_summary_gold.json# committed gold summary (reproduced by grade.py --self-test)
      probes/tz_probe.py        # property probe for timezone normalization (AC5)
```

Two shapes of task ship here, both validating against the *same* schema:

- **Directory tasks** (`tasks/<id>/task.json` + `env/` + `grader/`) — the canonical,
  fully-equipped form. `ub-cal-0007` is the reference example: it carries a real fixture,
  a deterministic grader, a committed gold artifact, and a property probe.
- **Curated JSONL sets** (`curated/*.jsonl`) — one task object per line, for quickly
  registering a *set* of tasks. A line may be a complete directory task copied inline
  (e.g. `ub-cal-0007` appears in both `tasks/ub-cal-0007/task.json` and the first line of
  `v1.jsonl`, byte-for-byte equal) or a lightweight stub whose gold/grader is authored
  later (`source` fields read `"starter stub (gold to be authored)"`).

---

## 2. The task schema (agent-visible vs. oracle-private)

A task is one JSON object with a structurally enforced **two-tier visibility** split
(`docs/tasks.md` §2.1). The loader projects an *agent view* (`Task.agent_view()`) that
strips every gold field, so nothing the agent sees can leak the answer.

| Surface | Fields | Who sees it |
|---|---|---|
| **Agent-visible** | `id`, `title`, `user_goal`, `user_goal_persona_note`, `domain`, `difficulty`, `deliverable_type`, `required_capabilities`, `env` | the agent-under-test |
| **Oracle-private** | `reference_repos`, `hidden` (`summary`, `acceptance_criteria`, `ambiguity_points`, `info_units`, `reveal_rules`, `oracle_persona`, `known_pitfalls`, `out_of_scope`, `constraints`), `expected_interventions` | the grader + the oracle process only |

Key invariants (checked in CI by `qc/`, and re-verified for every line of every JSONL
file by the validation snippet in §6):

- The `user_goal` is **under-specified**: it omits ≥2 decisions a competent dev must make.
  Those omissions are enumerated in `hidden.ambiguity_points`, each with a `severity`
  (`low`/`medium`/`high`) and a `reveal` rule (`on_ask`/`on_hint`/`never_volunteer`).
  A good task has ≥1 **high**-severity ambiguity that is *load-bearing* — guessing it wrong
  fails a criterion (see §3).
- `hidden.acceptance_criteria[]` are weighted, independently checkable, and routed to a
  verification channel via `check_kind` (`func` → deterministic checker, `rubric_auto` →
  scripted rubric, `oracle_judgment` → LLM judge). `is_core` marks functional must-haves;
  `is_hard` marks gating hard constraints. ≥80% of AC weight should be `func`/`rubric_auto`
  (auto-gradable) so the headline metric needs no human.
- `hidden.info_units[]` are discrete spec units (`requirement`/`constraint`/`preference`)
  used by the metrics workstream for spec-elicitation / proactive-inference scoring.
- `env.network` defaults to `deny` (hermetic; `DESIGN.md` invariant 5). `env.allowed_reqs`
  is the **only** dependency set the agent may install.

The agent-visible surface is produced programmatically — never hand-maintained — so it
cannot drift out of sync with the gold:

```python
from usabench.core.schema import Task
import json
task = Task.model_validate(json.load(open("tasks/ub-cal-0007/task.json")))
agent_view = task.agent_view()   # contains NO hidden/reference_repos/gold fields
```

---

## 3. The worked example: `ub-cal-0007` (calendar workload summarizer)

A T2 `data-analysis` task. The user, a non-expert, asks for "a little tool" that summarizes
how they spend their time from a Google-Calendar `.ics` export. The phrasing deliberately
omits every decision that makes the task hard; those omissions are the `ambiguity_points`.

**Fixture (`env/sample_calendar.ics`) — engineered to exercise every criterion.** It is
fully synthetic (no real person's calendar; `docs/tasks.md` §9.3) and contains:

| Event | Property it exercises | Criterion |
|---|---|---|
| Weekly "Team Standup" (`RRULE:FREQ=WEEKLY;COUNT=4`) | recurrence expansion | AC3, AC8 |
| "Quarterly Review with NY Office" (scheduled `America/New_York`) | cross-timezone normalization — NY Wed 20:00 EDT → Amsterdam **Thu** 02:00 CEST | AC5, AC8 |
| "PTO – Out of Office" (`VALUE=DATE`, all-day) | all-day excluded from hours, counted separately | AC4 |
| "Vendor Sync" (`STATUS:CANCELLED`) | cancelled events excluded from every total | AC6 |
| "Project Kickoff" (3 attendees + conferencing URL) | meeting heuristic (≥2 attendees OR conf URL) | meeting split |
| "Focus Block" (1 attendee, no link) | non-meeting | meeting split |
| "1:1 with Manager" (2 attendees, no link) | meeting via attendee count | meeting split |

The cross-timezone review is the load-bearing trap: an agent that reads the New-York wall
clock without normalizing attributes that hour to **Wednesday**, flipping the busiest day
and failing AC5 (`tz_probe.py`) and AC8. This is exactly the usability signal — *asking the
high-severity timezone question (AP2) cheaply beats guessing wrong*.

**Gold (`grader/gold/cal_summary_gold.json`).** The frozen expected summary
(total 7.0 scheduled hours, 9 events, busiest day Thu @ 3.0h, …). It is authored
clean-room — semantics cross-checked against the reference repos, no code copied — and is
**reproduced byte-for-byte** by the grader's stdlib reference path. Do not regenerate it
casually: any change is a gold change and must be re-reviewed (`docs/tasks.md` §8.3).

**Grader (`grader/grade.py`).** The task's `verification.harness_entry`. A pure,
deterministic, **stdlib-only** function of (agent artifact, fixture, gold). It runs the
artifact, tolerantly extracts its summary (preferring machine-readable JSON, else scraping
the human table — the output format is itself a low-severity ambiguity, AP5), runs the
timezone probe, compares every value against the gold within documented tolerances, and
prints a per-criterion pass/fail report. Its own exit code is `0` on a successful grading
run regardless of the artifact's score; the verdict lives in the JSON report.

```bash
# Reproduce the gold from the fixture and diff it against the committed gold:
python tasks/ub-cal-0007/grader/grade.py --self-test

# Grade an agent artifact (a .py CLI tool):
python tasks/ub-cal-0007/grader/grade.py --artifact path/to/agent_tool.py
```

**Probe (`grader/probes/tz_probe.py`).** A *property* probe for AC5: it asserts the tool
normalizes to one report timezone *before* weekday bucketing, either by checking the
default busiest day (`Thu`, not `Wed`) or by verifying that a `--tz` override re-buckets the
same instants coherently (total hours invariant, per-weekday buckets shift).

The grader is verified to **discriminate** (`docs/tasks.md` §8.4): a known-good solution
scores 1.0; an empty/bad solution scores ~0.1 (only the "runs at all" criterion); and an
ambiguity-trap solution that ignores the timezone question scores 0.75 — below the 0.80
`accept_threshold` — because it fails AC5 and AC8.

---

## 4. Curated sets

- **`curated/v0_smoke.jsonl`** — 2–3 *tiny*, T1, `cli-util` tasks (`ub-smoke-wordfreq`,
  `ub-smoke-csvstats`, `ub-smoke-linecount`). Every criterion is `func` and
  self-contained, every constraint is "stdlib only", and they carry no `env` fixtures.
  These exist so the zero-cost `FakeLLM` smoke path (`make smoke` / `usabench smoke`, see
  `docs/infra.md` §6.5) can drive schema → harness → sandbox → acceptance → scoring →
  leaderboard end-to-end **with no external deps and no paid API calls**.
- **`curated/v1.jsonl`** — the starter set: `ub-cal-0007` (the full worked example, copied
  inline) plus a couple of hand-written stubs (`ub-log-0001` access-log summarizer,
  `ub-jsonflat-0001` JSON→CSV flattener) whose gold/grader are authored in a later pass.
  The stubs are schema-valid and demonstrate the shape; they are flagged via
  `source: "starter stub (gold to be authored)"` and must clear the §8 QC gates of
  `docs/tasks.md` before they ship in a scored release.

---

## 5. Provenance, licensing, ethics, contamination

These follow `docs/tasks.md` §9 and are summarized here because they bind every file in
this directory.

- **Links + metadata, not blobs.** `reference_repos[]` records only the URL, a **pinned**
  commit SHA, and the SPDX `license`. We store no vendored copy of reference code. All
  gold implementations and fixtures in this tree are **authored by us, clean-room** from
  the *behavior/spec*, not copied — so the redistributable artifact is ours.
  - In `ub-cal-0007` the reference repos are
    [`waldbaer/icalendar-events-cli`](https://github.com/waldbaer/icalendar-events-cli) (MIT, primary),
    [`pimutils/khal`](https://github.com/pimutils/khal) (MIT, secondary), and
    [`loteoo/icsp`](https://github.com/loteoo/icsp) (Unlicense, inspiration).
  - **Commit pins are placeholders** (`0000…`, `1111…`, `2222…`) in this build. The human
    freezing a release replaces them with the real harvested HEAD SHAs and records them,
    with licenses, in the repo-root `ATTRIBUTION.md`. CI's pinned-SHA lint will fail a
    *scored release* on a floating/placeholder ref, by design.
- **License allowlist.** Only permissive licenses populate the redistributable reference
  pool: MIT, Apache-2.0, BSD-2/3, ISC, Unlicense, MPL-2.0. GPL/AGPL/unlicensed repos may
  be referenced *by link only* (we author all derived material independently).
- **Synthetic fixtures.** `sample_calendar.ics` and all smoke inputs are generated by us;
  they contain no PII and no real calendar export. No secrets/keys appear in any task.
- **Hermetic by default.** `env.network = deny`; agents cannot reach third parties during a
  scored run. The build installs `env.allowed_reqs` while the network is available, then
  scored runs execute offline.
- **Contamination labeling.** Each task carries `contamination_label`
  (`low`/`medium`/`high`). The novelty defenses are: a non-expert `user_goal`, fixtures we
  synthesize, an AC set recombined across multiple reference repos, and surface-form
  perturbation — so verbatim recall of a known repo does not satisfy the grader. A fraction
  of finished tasks is held out (never published) for the canonical leaderboard.

---

## 6. Validating this directory

Every task object — `tasks/<id>/task.json` and **every line** of `curated/*.jsonl` — must
validate against both `schemas/task.schema.json` and the pydantic `Task` model, and must
have a leak-free agent view. Run:

```bash
python - <<'PY'
import glob, json
from jsonschema import Draft202012Validator
from usabench.core.schema import Task

schema = json.load(open("schemas/task.schema.json"))
validator = Draft202012Validator(schema)

def objects():
    yield "tasks/ub-cal-0007/task.json", json.load(open("tasks/ub-cal-0007/task.json"))
    for path in sorted(glob.glob("tasks/curated/*.jsonl")):
        for i, line in enumerate(open(path), 1):
            line = line.strip()
            if line:
                yield f"{path}:{i}", json.loads(line)

GOLD_KEYS = {"hidden", "reference_repos", "acceptance_criteria", "ambiguity_points", "info_units"}
ok = True
for where, obj in objects():
    errs = list(validator.iter_errors(obj))
    task = Task.model_validate(obj)               # raises on a pydantic mismatch
    leak = GOLD_KEYS & set(task.agent_view().model_dump())
    status = "OK" if not errs and not leak else "FAIL"
    if status != "OK":
        ok = False
    print(f"{status:4} {where} (schema_errors={len(errs)}, agent_view_leak={sorted(leak)})")
assert ok, "validation failed"
print("all task objects valid, agent views leak-free")
PY

# And prove the worked-example gold still reproduces:
python tasks/ub-cal-0007/grader/grade.py --self-test
```

Both must pass before a task is promoted toward a scored release. The full QC pipeline
(schema lint, suitability checks, gold reproducibility, grader discrimination, calibration,
diversity/dedup, human review, held-out partition) lives in `qc/` and is described in
`docs/tasks.md` §8.
