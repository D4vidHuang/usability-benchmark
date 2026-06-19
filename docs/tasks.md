I have all the grounding I need. Writing the specification document now.

# usability-benchmark — Task Dataset & Data Collection Specification

**Owner:** Task-Dataset & Data-Collection workstream
**Status:** Design v1 (buildable)
**Scope:** Task taxonomy, task JSON schema, one fully worked example, GitHub harvesting methodology + script design, output schemas, curation/QC pipeline, licensing/ethics/contamination.

This document specifies *what a task is*, *how we manufacture tasks at scale from public GitHub*, and *how we keep them clean, legal, and contamination-resistant*. It is written to be turned directly into code under `D4vidHuang/usability-benchmark`.

---

## 0. Design context & prior art (named)

We deliberately diverge from **SWE-bench** (Jimenez et al., 2024 — "resolve this issue, pass the hidden `FAIL_TO_PASS`/`PASS_TO_PASS` tests"). SWE-bench is *closed-form, single-shot, fully-specified*. Our tasks are *open-ended, under-specified, and interactive*, and the headline measurement is **how much human assistance the agent needed**, not just terminal success.

Borrowed ideas, with attribution:
- **τ-bench** (Yao et al., 2024, arXiv:2406.12045): an LLM **simulated user** drives a dynamic conversation; success is graded by comparing **final environment/DB state to an annotated gold state**, and `pass^k` measures reliability across trials. We reuse the simulated-user oracle, state-based grading, and the variance metric. ([arXiv](https://arxiv.org/abs/2406.12045))
- **SWE-bench** (Jimenez et al., 2024): the *reference-repo-grounds-the-task* idea, hidden test verification, and license/contamination hygiene around scraped repos.
- **GAIA** (Mialon et al., 2023) and **AgentBench** (Liu et al., 2023): difficulty tiering by required-capability count and tool-step depth.
- **τ-bench follow-up caution** ("Lost in Simulation," arXiv:2601.17087): LLM-simulated users are imperfect proxies; we mitigate with a constrained oracle persona + a small human-validated calibration subset (see §6.4, §8.5).

A note we carry through the whole design: because we *measure intervention*, the task material must be engineered so that intervention is *necessary and discriminative* — a task the agent can one-shot from the prompt is a bad task for us, even if it's a great task for SWE-bench.

---

## 1. Task Taxonomy

### 1.1 Domains (top-level categories)

| Code | Domain | Typical deliverable | Why it's good for us |
|------|--------|--------------------|----------------------|
| `cli-util` | CLI utilities | argparse/click/cobra tool + `--help` | crisp I/O contract; ambiguity in flags/format |
| `data-analysis` | Data-analysis / reporting tools | script or notebook that ingests a file and emits stats/plots | ambiguity in *what to measure*, *grouping*, *output medium* |
| `web-dashboard` | Small web apps / dashboards | Flask/FastAPI/Express + minimal frontend | ambiguity in scope, persistence, framework |
| `api-integration` | 3rd-party API / file-format integrations | client wrapper, sync tool, parser | ambiguity in auth, which endpoints, error handling |
| `automation` | Automation / glue scripts | cron-able script, file watcher, batch job | ambiguity in triggers, idempotency, side-effects |
| `dev-tooling` | Developer tooling | linter/formatter/codegen/git helper | ambiguity in rules, config surface, output format |

Each domain maps to a pool of reference OSS projects (the harvest, §5) so a task draft is always *backed by a real, runnable implementation*.

### 1.2 Difficulty tiers

Tiers are defined by **objective, measurable proxies**, not vibes. A task's tier is the **max** of its component scores, then human-confirmed.

| Tier | Name | Required capabilities | Distinct deliverable units¹ | Expected oracle interventions² | Reference-repo size proxy | Bounded session budget³ |
|------|------|----------------------|-----------------------------|-------------------------------|---------------------------|--------------------------|
| T1 | **Trivial-shaped** | 1 | 1 | 0–1 | ≤300 LoC core | ≤15 agent turns / 10 min |
| T2 | **Standard** | 2–3 | 2–3 | 1–3 | 300–1.5k LoC | ≤40 turns / 30 min |
| T3 | **Composite** | 3–5 | 3–6 | 2–5 | 1.5k–6k LoC | ≤80 turns / 60 min |
| T4 | **Open-scope** | ≥5 | ≥6 (some optional) | 4–8 | >6k LoC or multi-repo | ≤150 turns / 120 min |

¹ *Distinct deliverable units* = independently verifiable acceptance criteria (a file produced, a flag honored, a route that returns 200, a test that passes).
² *Expected oracle interventions* = calibration target (§6.4); used to normalize the assistance score, not as a hard cap.
³ Budgets are caps for the runner; not all are consumed.

**Required-capability vocabulary** (controlled list, used for `required_capabilities` and for tier scoring): `file-io`, `arg-parsing`, `data-parsing(csv/json/ics/xml)`, `datetime-handling`, `aggregation-stats`, `plotting/viz`, `http-client`, `auth-handling`, `web-server`, `frontend-render`, `db-persistence`, `recurrence-rules`, `timezone-handling`, `error-handling/retries`, `packaging/entrypoint`, `testing`, `concurrency`, `streaming/large-input`.

### 1.3 What makes a task **suitable** (admission filter)

A draft is admissible only if **all five** hold (these become automated checks in §8):

1. **Under-specified, but groundable.** The `user_goal` omits ≥2 decisions a competent dev must make (the `ambiguity_points`), yet a defensible "right answer" exists *because* a reference repo made those decisions.
2. **Real OSS reference.** ≥1 `reference_repos` entry with a permissive/known license, pinned commit, and a README or issue corpus that justifies the acceptance criteria.
3. **Runnable & verifiable.** The deliverable can be executed in a pinned container and graded by a deterministic harness (file diff, CLI golden output, HTTP probe, or a held-out test) — i.e. there is a `verification` spec that does not require a human.
4. **Modest scope.** Buildable within the tier's session budget by a strong human dev; we are not benchmarking marathon projects.
5. **Intervention-bearing.** Calibration runs (§6.4) show the median agent needs ≥1 oracle interaction OR makes ≥1 ambiguity-driven wrong assumption when interaction is disabled. If interaction never helps and the agent never errs, the task is **rejected** — it doesn't discriminate on our axis.

### 1.4 What we explicitly **exclude**

- Pure bug-fix / "make this failing test pass" tasks (that's SWE-bench's job).
- Tasks requiring paid APIs, secrets, or network resources we can't sandbox (mock them or drop).
- Tasks whose only verification is subjective aesthetics with no objective floor.
- Tasks where the reference repo's license forbids redistribution of the *derived material we need* (we still only store links + metadata; see §9).

---

## 2. Task Schema (JSON)

One task = one JSON object, validated by JSON Schema (`schema/task.schema.json`). Tasks ship as a directory `tasks/<id>/task.json` plus a frozen `tasks/<id>/env/` (Dockerfile, fixtures). The oracle-private fields (`hidden_spec`, `acceptance_criteria`, parts of `ambiguity_points`, `oracle_persona`) are **never** shown to the agent — they are loaded only by the grader and the oracle process.

```jsonc
{
  "id": "ub-cal-0007",                       // stable slug: ub-<domain>-<n>
  "schema_version": "1.0",
  "title": "Calendar workload summarizer",
  "domain": "data-analysis",                 // enum from §1.1
  "difficulty": "T2",                        // T1..T4
  "created_utc": "2026-06-19T00:00:00Z",
  "harvest_provenance_id": "hv-2026-06-19-0312",  // links to raw harvest record

  // ---------- SHOWN TO AGENT ----------
  "user_goal": "I export my Google Calendar as an .ics file every week and I want a little tool I can run that tells me how I'm actually spending my time — like how many hours of meetings I had, which days are busiest, that kind of thing. Can you build that?",
  "user_goal_persona_note": "Non-expert; describes outcomes, not implementation.",
  "deliverable_type": "cli-tool",            // enum: cli-tool|script|web-app|library|notebook|service
  "starter_files": ["env/sample_calendar.ics"],   // given to agent
  "environment": {
    "base_image": "python:3.11-slim",
    "setup": ["pip install -r env/requirements.allowed.txt"],   // allowlisted deps
    "network": "offline",                    // offline|mocked|login-node-online
    "entrypoint_hint": null                  // null = agent decides (more ambiguity)
  },
  "required_capabilities": ["data-parsing(ics)","datetime-handling","timezone-handling",
                            "recurrence-rules","aggregation-stats","arg-parsing","file-io"],

  // ---------- ORACLE-PRIVATE (gold knowledge) ----------
  "reference_repos": [
    {"url": "https://github.com/waldbaer/icalendar-events-cli",
     "commit": "PINNED_SHA", "license": "MIT", "role": "primary",
     "why": "RFC5545 parsing + filtering + JSON/human output; defines sane field semantics"},
    {"url": "https://github.com/pimutils/khal",
     "commit": "PINNED_SHA", "license": "MIT", "role": "secondary",
     "why": "recurrence + timezone handling reference; what 'an event' means"},
    {"url": "https://github.com/loteoo/icsp",
     "commit": "PINNED_SHA", "license": "Unlicense", "role": "inspiration",
     "why": "ics->tabular reduction; supports the 'tabular stats' framing"}
  ],
  "hidden_spec": "The gold tool reads one or more .ics files, expands recurring events (RRULE) within the analysis window, normalizes all times to a single timezone, and reports: total events, total scheduled hours, per-weekday hour distribution, busiest day, mean/median event duration, and a meeting-vs-non-meeting split heuristic. All-day events are excluded from hour totals but counted separately. Declined/cancelled events (STATUS:CANCELLED) are excluded.",
  "acceptance_criteria": [
    {"id":"AC1","kind":"capability","text":"Parses a valid RFC5545 .ics and runs without crashing","weight":1.0,"auto":true},
    {"id":"AC2","kind":"correctness","text":"Total scheduled hours within ±2% of gold for sample_calendar.ics","weight":2.0,"auto":true},
    {"id":"AC3","kind":"correctness","text":"Recurring events expanded across the window (count matches gold ±0)","weight":2.0,"auto":true},
    {"id":"AC4","kind":"correctness","text":"All-day events excluded from hour total","weight":1.0,"auto":true},
    {"id":"AC5","kind":"correctness","text":"Times normalized to one timezone before aggregation","weight":1.5,"auto":true},
    {"id":"AC6","kind":"correctness","text":"CANCELLED events excluded","weight":1.0,"auto":true},
    {"id":"AC7","kind":"usability","text":"Has --help and a documented invocation","weight":0.5,"auto":true},
    {"id":"AC8","kind":"correctness","text":"Per-weekday busiest-day output matches gold","weight":1.0,"auto":true}
  ],

  "ambiguity_points": [
    {"id":"AP1","question":"Should recurring events be expanded, or counted once?","gold":"Expanded within window","reveal":"on_ask","severity":"high"},
    {"id":"AP2","question":"What timezone should hours be reported in?","gold":"Normalize to the calendar's primary TZ; allow --tz override","reveal":"on_ask","severity":"high"},
    {"id":"AP3","question":"Are all-day events counted as hours?","gold":"No, reported separately","reveal":"on_ask","severity":"medium"},
    {"id":"AP4","question":"How is a 'meeting' defined vs other events?","gold":"≥2 attendees OR has a conferencing URL; else non-meeting","reveal":"on_ask","severity":"medium"},
    {"id":"AP5","question":"Output format: text table, JSON, or chart?","gold":"Human-readable table by default; --json optional","reveal":"on_ask","severity":"low"},
    {"id":"AP6","question":"What analysis window?","gold":"Full span of file unless --from/--to given","reveal":"on_ask","severity":"medium"}
  ],

  "verification": {
    "method": "harness",
    "harness_entry": "grader/grade.py",
    "checks": [
      {"ac":"AC1","type":"exit_code","cmd":"python {ARTIFACT} env/sample_calendar.ics","expect_exit":0},
      {"ac":"AC2","type":"numeric_from_stdout","gold_field":"total_hours","tol_pct":2},
      {"ac":"AC3","type":"numeric_from_stdout","gold_field":"event_count","tol_abs":0},
      {"ac":"AC5","type":"property_probe","probe":"grader/probes/tz_probe.py"},
      {"ac":"AC7","type":"stdout_contains","cmd":"python {ARTIFACT} --help","needle":"usage"}
    ],
    "gold_artifact": "grader/gold/cal_summary_gold.json",   // computed once, committed
    "scoring":"weighted_fraction_of_AC_passed"
  },

  "oracle_persona": {
    "name":"busy_nonexpert",
    "system_prompt_ref":"oracle/personas/busy_nonexpert.md",
    "knowledge_scope":["hidden_spec","acceptance_criteria","ambiguity_points"],
    "answer_policy":"Answer clarifying questions truthfully but minimally; never volunteer the spec; do not write code; if asked an out-of-scope question, say you don't care/decide yourself.",
    "hint_budget": 3,                          // max proactive hints before counted as 'rescue'
    "style":"casual, non-technical vocabulary"
  },

  "expected_interventions": {                  // calibration (§6.4), filled after pilot runs
    "median": 3, "p10": 1, "p90": 6,
    "by_type": {"clarification":2,"hint":1,"rescue":0},
    "calibrated_with":"claude + gpt + 1 open-weight, 5 trials each",
    "discriminative": true
  },

  "labels": {"contamination_risk":"medium","redistributable":"links_only","reviewed_by":"<gh-user>"}
}
```

### 2.1 Field notes / invariants

- **Two-tier visibility is enforced structurally.** The loader emits an *agent view* (whitelist of fields) and a *grader/oracle view*. There is no field the agent sees that leaks the gold answer.
- `ambiguity_points[].reveal ∈ {on_ask, on_hint, never_volunteer}` is the contract the oracle obeys (§6).
- `acceptance_criteria[].auto` must be `true` for ≥80% of weight; non-auto criteria require a rubric and are excluded from the headline metric (kept for human-validation subset only).
- `commit` SHAs are **mandatory and pinned** — no floating `main`.
- Everything is reproducible from `env/` + pinned image + pinned deps allowlist (`requirements.allowed.txt`) to control stochastic dependency drift.

---

## 3. Worked example, end-to-end: "analyze my calendar/schedule"

This is the `ub-cal-0007` object above, expanded into the four artifacts a task author must produce.

### 3.1 The goal as the user phrases it (shown)
> "I export my Google Calendar as an .ics file every week and I want a little tool I can run that tells me how I'm actually spending my time — like how many hours of meetings I had, which days are busiest, that kind of thing. Can you build that?"

Deliberately omits: recurrence handling, timezone, all-day treatment, "meeting" definition, output format, window. Those omissions are the `ambiguity_points`.

### 3.2 Reference repos to mine (gold-side)
- **`waldbaer/icalendar-events-cli`** (MIT, Python 3.10+) — primary. RFC5545/jCal parsing via `icalendar` + `recurring-ical-events`; filtering by date range / summary / location; JSON & human output. Defines defensible field semantics and the "filter then report" shape. ([repo](https://github.com/waldbaer/icalendar-events-cli))
- **`pimutils/khal`** (MIT, Python) — secondary. Standards-tracking CLI calendar; our reference for *recurrence expansion and timezone correctness* and "what counts as an event." ([repo](https://github.com/pimutils/khal))
- **`loteoo/icsp`** (Unlicense, bash/awk) — inspiration. `.ics` → TSV/CSV reduction; supports the "turn the calendar into rows, then aggregate" framing and shows a minimal-dependency path. ([repo](https://github.com/loteoo/icsp))

These are mined for: README invocation examples, the `icalendar`/`recurring-ical-events` semantics (RRULE expansion, all-day vs timed, CANCELLED status), and issues discussing timezone pitfalls — which become `hidden_spec` clauses and acceptance criteria. We store **links + commit SHAs + extracted snippets we author ourselves**, never a fork of their code (§9).

### 3.3 Acceptance criteria (oracle-private)
AC1–AC8 above. The **gold artifact** (`cal_summary_gold.json`) is produced *once* by us by running a reference implementation (authored by the task author, semantics cross-checked against the reference repos) on the frozen `sample_calendar.ics`, then committed. The fixture is hand-built to exercise every AC: it contains a weekly RRULE standup, a one-off cross-timezone meeting, an all-day "PTO", a `STATUS:CANCELLED` event, and a 3-attendee meeting with a conferencing URL.

### 3.4 Ambiguity points the agent *should* surface (and severity)
AP1 recurrence (high), AP2 timezone (high), AP3 all-day (medium), AP4 meeting definition (medium), AP5 output format (low), AP6 window (medium). An agent that builds the tool **without asking any of the high-severity ones** and guesses wrong will fail AC2/AC3/AC5 — this is exactly the signal we want: *good usability = asking the right questions cheaply rather than failing silently.*

### 3.5 What the oracle reveals **only if asked**
- Asked "should repeating events be counted each time they occur?" → "Yes, count each occurrence in the period." (resolves AP1)
- Asked "what timezone should the hours be in?" → "Use whatever the calendar's main timezone is; let me override it if I want." (AP2)
- Asked "do you want all-day things like vacation in the hour totals?" → "No, keep those separate." (AP3)
- Asked nothing about output → oracle never volunteers; agent's reasonable default passes AC7 regardless (AP5 is low-severity by design).
- Asked an off-spec question ("what Python version?") → persona answers "I don't know, you pick." (no information, but still counts as an interaction event in the assistance log).

The interaction log (who asked what, severity, whether it was a clarification vs a rescue) is the raw material for the **assistance score** owned by the metrics workstream; our job is to make the `ambiguity_points` + `oracle_persona` produce a *clean, gradable* interaction trace.

---

## 4. Pipeline overview (harvest → draft → final)

```
        GitHub REST/GraphQL                  LLM drafter             human + auto QC
raw_harvest.jsonl  ──────►  candidates.jsonl  ─────►  task_drafts.jsonl  ─────►  tasks/<id>/task.json
   (signals,                 (filtered,                (schema-valid,            (reviewed, calibrated,
    metadata,                 deduped,                  gold authored,            frozen env,
    links only)              license-OK)                AC + ambiguity)          discriminative)
```

Four stages, four artifacts, each idempotent and re-runnable. Stage boundaries are JSONL files so the pipeline is resumable and auditable.

---

## 5. Data-collection methodology & script design

### 5.1 What signals we harvest (and why)

| Signal | Source | What it tells us |
|--------|--------|------------------|
| repo description, topics | REST `GET /repos`, search | domain classification, candidate `user_goal` seed |
| README (root) | REST `GET /repos/{o}/{r}/readme` | feature list → acceptance criteria; invocation examples |
| stars, pushed_at, archived | search/repo | quality + recency (anti-contamination) filters |
| license (SPDX id) | repo `license` field | redistribution legality (§9) |
| `good-first-issue`, `enhancement`, `feature request` issues | REST `GET /issues`, GraphQL | under-specified asks phrased like users; ambiguity mining |
| `awesome-*` list entries | search `awesome <domain> in:name` + README parse | curated, real, idea-grade project pools |
| primary language, size (KB) | repo | tier size proxy, environment image choice |
| has_tests / CI presence | contents API (`/tests`, `.github/workflows`) | verifiability hint |

**Quality gates at harvest time** (configurable in `harvest.yaml`):
`stars >= 50`, `pushed_at within last 18 months`, `not archived`, `not fork`, `size_kb in [tier-appropriate band]`, `license in ALLOWLIST` (MIT, Apache-2.0, BSD-2/3, ISC, Unlicense, MPL-2.0; **exclude** GPL/AGPL/no-license/`other` from the *redistributable* pool — they may still be referenced as links-only, see §9).

### 5.2 Endpoints & queries (concrete)

- **Repo discovery (REST search):** `GET /search/repositories?q=topic:ics+language:python+stars:>50+pushed:>2024-12-01&sort:updated` — one query per `(domain, language, topic)` cell from a seed matrix in `harvest.yaml`. Search API caps at 1000 results/query → we *shard by star range and date window* to page past the cap.
- **Awesome lists:** `GET /search/repositories?q=awesome+in:name+<domain>` then fetch READMEs and extract `github.com/owner/repo` links via regex → enqueue those repos.
- **Bulk metadata + issues (GraphQL):** one query batches `repository(owner,name){ description stars pushedAt licenseInfo{spdxId} primaryLanguage object(README) issues(first:20, labels:["good first issue","enhancement"]) }` for up to ~50 repos/request to conserve rate budget. GraphQL points cost is computed and logged.
- **README & file probes (REST):** `GET /repos/{o}/{r}/readme`, `GET /repos/{o}/{r}/contents/{path}` for tests/CI detection. Conditional requests with `If-None-Match` (ETag) to avoid spending budget on unchanged repos.

### 5.3 Rate-limit & robustness handling

- Token: `repo+workflow` scope, REST 5,000 req/h, GraphQL 5,000 points/h. Read `X-RateLimit-Remaining`/`Reset` from every response; **sleep-until-reset** when remaining < safety floor (e.g. 50).
- **Secondary-rate-limit** (abuse) handling: honor `Retry-After`; exponential backoff with jitter on 403/429/5xx (base 2s, cap 5 min, max 6 retries).
- **ETags / conditional requests** to make re-runs nearly free and resumable.
- **Checkpointing:** append-only `raw_harvest.jsonl` + a `seen.sqlite` cursor (key = `owner/repo@endpoint`) so a killed job resumes without re-spending budget.
- **Where it runs:** discovery + metadata harvest run on the **DAIC login node** (has internet) or a tiny CPU job; outputs land on `/tudelft.net/staff-umbrella/CoReFusion/usability-benchmark/data/`. No GPU. Single-threaded with polite pacing (≤1 req/200ms) — we are read-only and well under limits.

### 5.4 Dedup

- **Exact:** drop repeated `owner/repo` (sqlite unique).
- **Near-dup repos:** MinHash/SimHash over `(description + README first 2KB)`; cluster at Jaccard ≥ 0.8; keep the highest-starred representative — prevents 12 near-identical "todo CLI" tasks.
- **Cross-fork collapse:** if `fork == true` or `source` field present, map to upstream.
- **Task-level dedup (later):** after drafting, embed `user_goal` (sentence-transformer) and drop cosine-≥0.92 near-twins so the final set is diverse across domains/tiers.

### 5.5 Raw harvest schema (`raw_harvest.jsonl`)

```jsonc
{
  "harvest_provenance_id": "hv-2026-06-19-0312",
  "fetched_utc": "2026-06-19T03:12:44Z",
  "owner": "waldbaer", "repo": "icalendar-events-cli",
  "url": "https://github.com/waldbaer/icalendar-events-cli",
  "default_branch": "main", "head_sha": "PINNED_SHA",
  "description": "Command-line tool to read and filter events from iCalendar (RFC 5545)...",
  "topics": ["icalendar","cli","rfc5545"],
  "primary_language": "Python", "size_kb": 412,
  "stars": 73, "pushed_at": "2026-02-10T...", "archived": false, "fork": false,
  "license_spdx": "MIT", "redistributable": true,
  "readme_excerpt": "<first 4KB, stored for drafting; ours to quote under license>",
  "readme_sha": "blob_sha", "has_tests": true, "has_ci": true,
  "candidate_issues": [
    {"number": 12, "title":"Support filtering recurring events by window",
     "labels":["enhancement"], "url":"...", "body_excerpt":"<2KB>"}
  ],
  "source_list": "awesome-ics",                 // if discovered via an awesome-* list
  "domain_guess": "data-analysis", "tier_size_proxy": "T2",
  "dedup_cluster_id": "cl_0091", "dedup_representative": true
}
```

### 5.6 Candidate schema (`candidates.jsonl`)

Adds curator-facing fields: `passes_quality_gates: bool`, `gate_reasons: []`, `suitability_prefilter_score` (heuristic 0–1 from: has feature-list README, has tests/CI, has enhancement issues, size in band), and a `draft_status: pending|drafted|rejected`.

---

## 6. From candidate → task draft (LLM drafting + oracle calibration)

### 6.1 Drafting (automated, LLM-assisted, then human-gated)

For each admitted candidate, a **drafter LLM** is prompted with the README excerpt + feature list + enhancement issues and asked to emit a *draft* `task.json` minus gold-graded artifacts: it proposes `user_goal` (rephrased into non-expert voice), `domain`, tentative `difficulty`, `required_capabilities`, candidate `acceptance_criteria`, and `ambiguity_points`. The drafter is explicitly instructed to (a) phrase the goal as an outcome, not an implementation, and (b) surface ≥2 high-severity ambiguities.

### 6.2 Gold authoring (human-in-the-loop)
A human author (or a strong model under human review) writes the **reference implementation** and the frozen `env/` fixtures, runs them to produce the committed `gold_artifact`, and finalizes `hidden_spec` + the auto checks. This is the expensive step; it is *not* fully automated, because gold correctness is the foundation of the whole benchmark.

### 6.3 Environment freezing
Pin `base_image`, generate `requirements.allowed.txt` from the reference impl, build and smoke-test the container, and store the image digest. Network is set `offline` unless a task genuinely needs a mocked endpoint.

### 6.4 Intervention calibration (fills `expected_interventions`)
Run a small pilot: **N=3 agents** (one Claude, one OpenAI, one open-weight via vLLM on a GPU node) × **5 trials** each, with the oracle live. Log interaction counts/types. We then require:
- **Discriminative check:** with interaction *disabled*, the agent's score drops materially (≥1 AC fails due to an un-asked high-severity ambiguity) on ≥2 of 3 models — proves the ambiguity is load-bearing.
- **Calibration:** store median/p10/p90 interventions and the by-type breakdown. Tasks where interventions are always 0 *and* score is already high → **rejected** (not discriminative). Tasks where even max assistance can't get any model to pass → flagged for gold/AC review (likely broken).

This directly closes the loop with the locked design: every shipped task is *empirically* shown to require and reward assistance, and stochasticity is measured up front (`pass^k`/variance, after τ-bench).

---

## 7. Oracle persona contract (data-side responsibilities)

The metrics/runner workstreams own the live oracle loop; the dataset owns the **content** the oracle is given. Per task we ship:
- `oracle_persona.system_prompt_ref` → a persona file (`oracle/personas/busy_nonexpert.md`) parameterized with the task's `hidden_spec`, `acceptance_criteria`, and `ambiguity_points`.
- **Answer policy** baked in: truthful, minimal, never volunteers gold, never writes code, obeys each `ambiguity_point.reveal` flag, has a `hint_budget`; exceeding it = a "rescue" (high-severity intervention). Off-scope questions get a no-information persona answer but still log as an interaction.
- This is the lever against the "LLM-simulated users are unreliable" risk (arXiv:2601.17087): a *tightly scripted, scoped* oracle is far more reproducible than an open-ended one. A small human-played subset (§8.5) validates that the LLM oracle's answers match what a human maintainer would say.

---

## 8. Curation & quality-control pipeline

Stages, each gating promotion to the next. CI-enforced.

1. **Schema validation** — `task.json` validates against `task.schema.json`; required gold fields present; commit SHAs pinned; visibility partition lints clean (no gold field in the agent whitelist).
2. **Suitability auto-checks** (§1.3): ≥2 ambiguity points, ≥1 high-severity; ≥1 reference repo; ≥80% of AC weight is `auto`; env builds; license OK.
3. **Gold reproducibility** — fresh container build from digest reproduces `gold_artifact` byte-for-byte (or within declared tolerances). Flaky gold ⇒ block.
4. **Grader sanity** — a *known-good* solution scores 1.0; a *known-bad/empty* solution scores ~0; an *ambiguity-trap* solution (ignores high-severity AP) fails the corresponding AC. This proves the grader actually discriminates.
5. **Calibration gate** (§6.4): `expected_interventions.discriminative == true`.
6. **Diversity/dedup gate** (§5.4 task-level): goal embedding not a near-twin of an existing task; domain/tier quotas tracked (target balanced coverage across the 6 domains and 4 tiers).
7. **Human review** — a reviewer signs `labels.reviewed_by`, confirms: goal reads like a real non-expert ask, ambiguities are genuine, gold matches the reference repos' intent, no license/ethics violation. Two reviewers for T3/T4.
8. **Held-out partition** — assign each finished task to `public` or `private_heldout` (§9.4) before publishing.

### 8.5 Human-validation subset
~10–15% of tasks get a **human-played oracle** transcript and human grading of any non-auto criteria, to (a) validate the LLM oracle as a proxy and (b) anchor the auto-grader. Disagreements feed back into persona/AC fixes.

---

## 9. Licensing, ethics, contamination

### 9.1 Only public, store links + metadata not blobs
We harvest **only public repos**. We persist URLs, pinned SHAs, SPDX license, and *short, attributed excerpts* (README/issue snippets ≤4KB) used to author tasks — never a vendored copy of the reference code. The benchmark's gold implementations and fixtures are **authored by us** (clean-room from the *behavior/specs*, not copied), so the redistributable artifact is ours.

### 9.2 Respect licenses
- `reference_repos[].license` is recorded with each task; an `ATTRIBUTION.md` lists every referenced repo + license + commit.
- Excerpt storage is limited to fair-use-scale snippets with attribution; permissive-license repos (MIT/BSD/Apache/ISC/Unlicense/MPL) populate the `redistributable` pool. GPL/AGPL/unlicensed repos may still be *referenced by link* but we author all derived material independently and store no derivative of their code.
- The benchmark repo itself ships under a permissive license for the harness and CC-BY-style for task data, with the attribution file.

### 9.3 Ethics
- Read-only API use, polite rate-limiting, no scraping of private/abuse-flagged content, no PII. The sample `.ics` and all fixtures are **synthetic** (we generate them), never a real person's exported calendar.
- No secrets/keys in tasks; network `offline`/`mocked` by default so agents can't exfiltrate or hit third parties during runs.

### 9.4 Contamination / data-leakage (frontier models may have trained on these repos)
This is the sharpest risk: if a model memorized `khal`, it can "solve" without interacting — exactly the behavior we're trying to *measure*. Mitigations, layered:
1. **Recency bias in harvest** — prefer repos/issues created or pushed *recently* (rolling window, e.g. last ≤18 months, refreshed each release) so reference material post-dates common training cutoffs.
2. **Novel recombination** — the *task* is not "reimplement repo X." We recombine: a `user_goal` framed by a non-expert, fixtures we synthesize, and an AC set that mixes features across multiple reference repos. The deliverable contract differs from any single repo's exact API/output, so verbatim recall doesn't pass the grader.
3. **Surface-form perturbation** — paraphrase goals, randomize fixture contents (seeded), and vary output-format defaults so memorized snippets don't satisfy `gold_artifact`.
4. **Private held-out set** — a fraction (target ~25%) of finished tasks are **never published**; they live only on `/tudelft.net/staff-umbrella/CoReFusion` and are used for the canonical leaderboard. Public tasks are for development/iteration; held-out tasks guard against overfitting and undetected leakage.
5. **Contamination labeling + audit** — each task carries `labels.contamination_risk`; for high-risk tasks we run a *no-interaction recall probe* (can the model produce the gold without any task material?) and prefer tasks where it cannot.
6. **Refreshable design** — because tasks are *generated from a pipeline*, the benchmark can be re-harvested on newer repos each cycle, producing a fresh held-out wave to outrun training-data creep.

---

## 10. Concrete file layout (to hand to the repo build)

```
usability-benchmark/
  harvest/
    harvest.yaml                 # seed matrix, quality gates, license allowlist
    collect_repos.py             # REST search + awesome-list expansion, sharded
    enrich_graphql.py            # batched metadata+README+issues, ETag-aware
    rate_limit.py                # remaining/reset + backoff + secondary-limit
    dedup.py                     # minhash + fork-collapse + embedding twin-drop
    -> data/raw_harvest.jsonl, data/candidates.jsonl, seen.sqlite
  drafting/
    draft_tasks.py               # LLM drafter -> task_drafts.jsonl (no gold)
    personas/busy_nonexpert.md
  tasks/
    ub-cal-0007/
      task.json
      env/{Dockerfile, requirements.allowed.txt, sample_calendar.ics}
      grader/{grade.py, gold/cal_summary_gold.json, probes/tz_probe.py}
  schema/{task.schema.json, raw_harvest.schema.json}
  qc/{validate.py, grader_sanity.py, calibrate.py, diversity.py}
  ATTRIBUTION.md
  LICENSE
```

---

## 11. Open decisions to confirm with the team (non-blocking)

1. Exact quality-gate thresholds (stars/recency) per domain — set in `harvest.yaml`, tunable; defaults above.
2. Held-out fraction (proposed 25%) and refresh cadence.
3. Whether `notebook` deliverables are in-scope for v1 (recommend defer; harder to auto-grade).
4. Calibration model set for `expected_interventions` (proposed: 1 Claude + 1 OpenAI + 1 open-weight via vLLM, 5 trials) — coordinate with the infra/metrics workstreams.

---

**Key files this workstream will own (absolute paths once the repo is cloned to DAIC):**
`/tudelft.net/staff-umbrella/CoReFusion/usability-benchmark/harvest/`, `/.../schema/task.schema.json`, `/.../tasks/ub-cal-0007/task.json`, `/.../qc/`.

**Sources:** [τ-bench (arXiv:2406.12045)](https://arxiv.org/abs/2406.12045) · ["Lost in Simulation" (arXiv:2601.17087)](https://arxiv.org/pdf/2601.17087) · [icalendar-events-cli](https://github.com/waldbaer/icalendar-events-cli) · [pimutils/khal](https://github.com/pimutils/khal) · [loteoo/icsp](https://github.com/loteoo/icsp) · [GitHub topic: ics-files](https://github.com/topics/ics-files)