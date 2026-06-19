I have everything I need. Here is the specification document.

---

# `usability-benchmark` — Runtime Protocol Specification
**Component: Simulated-User Oracle & Agent Harness**
Version 0.1 · Owner: oracle/harness workstream · Status: buildable draft

This document specifies the *runtime contract* of the benchmark: the interaction loop, the oracle, the intervention taxonomy, the canonical event trace, and a reference harness architecture. It is the interface between **task definitions** (produced by the task-curation workstream: `task_id`, `prompt`, `hidden_spec`, `reference_repo`, `acceptance_criteria`) and the **scorer** (which consumes only the trace). The guiding invariant:

> **All metrics are computed offline, purely from the trace. The harness's only job at runtime is to produce a complete, unambiguous, replayable trace.** Nothing the scorer needs may live only in memory.

---

## 0. Terminology & roles

| Term | Definition |
|---|---|
| **AUT** (agent-under-test) | The coding agent being benchmarked. Treated as a black box behind an adapter. May be an API model (Claude/OpenAI) or an open-weight model served by vLLM. |
| **Harness** | The orchestrator. Owns the loop, the sandbox, the clock/budgets, the trace writer, the tool router, and the oracle gateway. |
| **Oracle** | An LLM playing the human user/maintainer. Holds private `hidden_spec`. Reachable by the AUT only through a single typed channel. |
| **Sandbox** | Isolated, reproducible execution environment for the AUT's file/exec/test actions (one per run). |
| **Verifier** | Deterministic acceptance-criteria runner (tests, smoke checks, rubric probes). Distinct from the oracle. |
| **Run** | One (task, agent, seed) execution producing one trace file. |
| **Episode** | A run plus its replicas (seeds) for variance estimation. |
| **Intervention** | Any oracle-side contribution of information or correction beyond the initial prompt. Counted and severity-graded. |

The benchmark borrows its *interactive-clarification* framing from **τ-bench / τ²-bench** (tool-agent-user simulation), **InterCode** and **SWE-agent**'s agent-computer interface (ACI) for the sandboxed tool loop, **MINT** for measuring multi-turn use of feedback/tools, **GAIA**/**WebArena** for goal-grounded acceptance, and **CodeRL/LLM-as-judge** lines for rubric grading. We deliberately invert SWE-bench: the dependent variable is **assistance required**, not pass/fail alone.

---

## 1. The interaction loop

### 1.1 Channels (who can call whom)

The AUT has exactly **three** outbound channels, all mediated by the harness. It can never reach the oracle or the host directly.

```
                 ┌───────────────────────────────────────────┐
                 │                  HARNESS                   │
                 │  clock · budgets · trace writer · router   │
                 │                                            │
  ┌─────────┐    │   ┌────────────┐  ┌──────────┐  ┌────────┐ │   ┌──────────┐
  │   AUT   │◀──▶│──▶│  SANDBOX   │  │  ORACLE  │  │VERIFIER│ │◀─▶│  ORACLE  │
  │ (agent) │    │   │ fs/exec    │  │ gateway  │  │ runner │ │   │  (LLM)   │
  └─────────┘    │   └────────────┘  └──────────┘  └────────┘ │   └──────────┘
       ▲         │         ▲              ▲             ▲      │
       │         └─────────┼──────────────┼─────────────┼──────┘
       │   (1) tool_call   │  (2) oracle  │  (3) submit │
       └───────────────────┴───── query ──┴─────────────┘
```

1. **Sandbox/tool channel** — `tool_call` (read/write file, exec command, run tests, http within allowlist). Synchronous; returns `tool_result`.
2. **Oracle channel** — `oracle_query` of a fixed message type (`clarify` | `hint_request` | `handoff` | `confirm`). The harness routes it to the oracle LLM and returns `oracle_response`. Every such exchange is an *intervention candidate* (classified per §3).
3. **Submission channel** — `submit` (agent declares done) triggers a `verification_run` and an `oracle_review`.

The oracle is **strictly reactive**: it cannot interrupt the AUT, cannot push messages, and is only ever invoked by a harness-routed query, a `submit` review, or a harness-triggered *stuck-probe* (§2.4). This keeps the AUT in control of how much help it pulls, which is the quantity we measure.

### 1.2 Turn structure

A **turn** is one AUT action emission plus its synchronous result. Actions are a tagged union:

```
AgentAction =
  | { kind: "tool_call",   tool, args }            // sandbox channel
  | { kind: "oracle_query", qtype, text, context } // oracle channel
  | { kind: "message",     text }                   // think-aloud / plan, no side effect
  | { kind: "submit",      summary, entrypoint }    // submission channel
  | { kind: "give_up",     reason }                  // voluntary termination
```

The loop (single-threaded, deterministic ordering):

```
init:   write task.prompt as the first user message; start clock; budgets := task.budgets
loop:
  action  := AUT.step(observation)        # one action per step
  log(agent_action)
  switch action.kind:
    tool_call    -> result := sandbox.run(action); log(tool_call,tool_result,file_edit?)
    oracle_query -> resp := oracle.handle(action); log(oracle_query,oracle_response,intervention)
    message      -> result := ack
    submit       -> v := verifier.run(); r := oracle.review(v); log(verification_run,oracle_review)
                    if r.accept or budgets.exhausted: break
                    else: result := r.feedback (counts as intervention)
    give_up      -> break
  observation := render(result)
  budgets := budgets.debit(action)
  if budgets.exhausted: log(termination,"budget"); break
```

**One action per step** (no parallel tool batches at the protocol level) so the trace has a total order and budgets debit deterministically. Adapters that natively batch tool calls are unrolled by the harness into sequential `tool_call` events sharing a `batch_id`.

### 1.3 Budgets

Budgets are per-task constants in `task.budgets`, enforced by the harness, and **every debit is logged** so the scorer can reconstruct remaining budget at any event. We track five independent ceilings; the run terminates when *any* is hit:

| Budget | Unit | Default | Rationale |
|---|---|---|---|
| `max_turns` | agent actions | 80 | bounds wall-clock & log size |
| `max_wall_s` | seconds | 1800 | kills hung sandboxes |
| `max_tokens` | prompt+completion of AUT | 600k | normalizes across models |
| `max_cost_usd` | USD (priced per model) | 3.00 | cross-model fairness |
| `max_oracle_queries` | oracle exchanges | 25 | prevents "interview the oracle" degeneracy |

Token/cost are read from the model adapter's usage accounting (API headers for API models; vLLM `usage` for local). The **oracle's** own tokens are tracked separately (`oracle_tokens`, `oracle_cost`) and never count against the AUT — but they ARE logged, because oracle cost is part of "assistance cost". Budgets are reported in the trace at `run_start` so the scorer never hard-codes them.

### 1.4 Termination conditions

A run ends on the first of:

1. **Accept** — `submit` → `verification_run` passes the *gating* acceptance criteria AND `oracle_review.verdict == accept`. (Both required: tests guard objective behavior, oracle guards under-specified intent.)
2. **Budget exhausted** — any budget ceiling hit. The harness still triggers one final forced `verification_run` on the current sandbox state so partial-credit/quality metrics exist.
3. **Voluntary give-up** — AUT emits `give_up`. Forced final verification as in (2).
4. **Hard error** — sandbox crash, adapter failure after retries. Logged as `termination{reason:"error"}`; run is marked `invalid` and excluded from scoring (but retained for debugging).

`oracle_review` may also **reject with required revisions**; a reject does *not* terminate — it returns feedback (an intervention) and the loop continues until a budget is hit or a later submit is accepted. The count of reject→resubmit cycles is itself a usability signal.

---

## 2. The oracle

The oracle is the heart of the "how much human help?" measurement. Its behavior must be **truthful, bounded, graded, and reproducible**.

### 2.1 Private knowledge construction

The oracle is initialized from the task's `hidden_spec` — a structured object built once at curation time from the reference repo, never shown to the AUT:

```yaml
hidden_spec:
  intent: >                       # 1-3 sentences: what the user actually wants
  reference_repo: { url, commit_sha, license }
  must_have:                      # gating acceptance criteria (objective, testable)
    - id: MH1
      desc: "CLI accepts a path to a .ics file and prints events grouped by day"
      check: { type: "test", ref: "tests/test_cli.py::test_groups_by_day" }
  should_have:                    # graded rubric criteria (quality, partial credit)
    - id: SH1
      desc: "handles recurring events (RRULE)"
      check: { type: "rubric", weight: 0.2 }
  hidden_preferences:             # the "user taste" the AUT must elicit, not guess
    - id: HP1
      desc: "output as a table, not JSON, because user reads it in terminal"
      reveal_if: "agent asks about output format OR proposes a format"
  constraints:
    - "Python 3.11, stdlib + `icalendar` only, no network"
  known_pitfalls:                 # used to grade hints, NOT volunteered
    - "all-day events have no time component; naive .time() crashes"
  out_of_scope:                   # oracle says 'not needed' if asked
    - "calendar editing / writing back"
  persona: { type: "non_expert_user", domain_knowledge: "low", ... }
```

Construction pipeline (curation-time, scripted on the DAIC login node via the GitHub API):
1. Pull README, top issues/feature-requests, `docs/`, and the file tree at a pinned `commit_sha`.
2. An extraction LLM proposes `intent`, `must_have`, `should_have`, `hidden_preferences`, `constraints`, `known_pitfalls`. **Each item must cite a source span** (`source: {path, line_range}` or `issue:#N`) so the spec is auditable and defensible.
3. A human curator (or a second-pass verifier LLM with disagreement flagging) ratifies. `must_have` items must each compile to a runnable `check`.
4. Freeze `hidden_spec` + `commit_sha`. The reference repo provides the **gold reference implementation** used only for offline scorer calibration, never exposed at runtime.

### 2.2 Persona

Two persona archetypes, chosen per task and recorded in `run_start`:

- **`non_expert_user`** (default for "build me a tool…" tasks): describes goals in outcome terms, *cannot* read code fluently, answers "does this do what you want?" by behavior not implementation, has taste (the `hidden_preferences`) but doesn't volunteer it, gets confused by jargon. This persona makes the AUT do the work of translation and is the realistic usability setting.
- **`maintainer`** (for tasks grounded in a real repo's contribution norms): knows the codebase conventions, can answer "where does X live?", expects tests, will reject on style/scope violations. More precise, less ambiguous.

Persona only changes *tone and what counts as a reasonable question* — it never changes ground truth. Both personas draw answers from the same `hidden_spec`.

### 2.3 Disclosure rules (what may be revealed, and when)

The oracle operates under a strict information-release policy, enforced both by the system prompt and by a post-hoc **leakage check** (§2.6):

| Rule | Policy |
|---|---|
| **R1 Truthful** | Never lie. If a fact is in `hidden_spec`, answers consistent with it; if unknown/out-of-scope, says so. |
| **R2 No volunteering** | Never proactively reveals `hidden_preferences`, `known_pitfalls`, or solution structure. Only releases on a triggering query (`reveal_if`) or graded hint. |
| **R3 Reactive only** | Speaks only when queried, on `submit` review, or on a harness stuck-probe. |
| **R4 Graded hints** | Hints follow the severity ladder (§2.5): start vague, escalate only on repeated explicit requests or repeated failure. Never hands over a full solution. |
| **R5 Behavior over code** | `non_expert_user` answers acceptance questions from observed program behavior (verifier output / demoed run), not by reading the diff. |
| **R6 Scope guarding** | If asked about `out_of_scope`, answers "not needed for this" — this is a *correct, non-leaking* clarification. |
| **R7 No grading disclosure** | Never tells the AUT its current score, how many criteria remain, or budget remaining. |

### 2.4 Stuck detection (the one place the harness probes the oracle without an AUT query)

To model a real user who notices a floundering agent, the harness may *offer* a hint when the AUT is **objectively stuck**, defined by trace-computable signals (no LLM judgment needed to trigger):

```
stuck := (k_consecutive_failed_verifications >= 2)
      OR (no file_edit in last N=15 turns AND >0 failed tool_calls)
      OR (oscillation: same file reverted to a prior hash >=3 times)
      OR (>= 6 consecutive tool errors)
```

On `stuck`, the harness emits a **single** `oracle_offer` (a `level-0` nudge: "It looks like you're stuck on X — want a hint?"). The AUT may accept (→ becomes a graded hint, counted as intervention) or decline (logged, costs nothing). Offers are rate-limited (≤1 per `cooldown=20` turns) so the oracle can't babysit. Whether offers are enabled is a **run condition** (`oracle.proactive_stuck_help ∈ {off, on}`) so we can ablate it — see §2.7.

### 2.5 Graded hint / intervention ladder

Hint *strength* maps onto the severity scale (§3). The oracle picks the **lowest** level that addresses the query, and escalates only when the AUT re-asks after failing. This monotonic-escalation rule is what keeps "amount of help" meaningful and comparable.

| Level | Name | What the oracle may say | Severity (S) |
|---|---|---|---|
| **L0** | Clarification | Restate/disambiguate the goal; reveal a triggered `hidden_preference`; confirm scope (R6). *Information the user always knew and would naturally state.* | **0** (not penalized as "help" — it's spec elicitation, but still counted) |
| **L1** | Nudge | Point at the *area* without the answer: "the problem is in how you parse times." | **1** |
| **L2** | Directional hint | Name the concept/pitfall: "all-day events have no time; you're calling `.time()` on a date." | **2** |
| **L3** | Concrete guidance | Describe the fix in prose / pseudo-steps, no final code: "guard for all-day, branch on `isinstance(v, date)`." | **3** |
| **L4** | Near-solution / rescue | Provide a code snippet, exact API call, or do a sub-step. Strong rescue. | **4** |
| **L5** | Takeover / failure-equivalent | Oracle (or human fallback) supplies the working implementation of a gating criterion. The run's autonomy is effectively void for that criterion. | **5** |

L0 is special: it is **counted** (interaction happened) but is the *expected* cost of an under-specified task; it primarily feeds the "clarification efficiency" metric (did the agent ask the *right* questions early?). L1–L5 are **assistance** and feed the "intervention burden" metric. The split is recorded so both can be reported.

The oracle is told the ladder and instructed to stay at L0 unless (a) the AUT explicitly requests a hint, or (b) a stuck-offer was accepted, or (c) it is reviewing a rejected submission and a `must_have` is failing for an identifiable, nameable reason. Escalation requires an explicit *repeat* request or a *repeated* failure — never two levels at once.

### 2.6 Reproducibility & controlling helpfulness

LLM oracles are stochastic and drift in "helpfulness." Controls:

1. **Fixed oracle model + decoding**: oracle uses one pinned API model id, `temperature=0`, fixed `seed` where supported, and a **frozen system prompt template** (§2.8) hashed into `run_start.oracle_prompt_sha256`.
2. **Structured oracle output**: the oracle must return JSON (`level`, `reveals[]`, `text`, `refusals[]`). The `level` it self-declares is checked against the rule-based classifier (§3.3); mismatches are flagged.
3. **Leakage check (offline)**: a separate judge compares each `oracle_response.text` against `hidden_spec` items NOT triggered by the query; any unsolicited reveal raises `leak_flag` on the event. Runs with leaks above a threshold are quarantined. This makes "oracle too helpful" *detectable*, not silent.
4. **Helpfulness as an explicit axis, not noise**: we run a small grid of `oracle.helpfulness ∈ {strict, standard, lenient}` realized as different *escalation thresholds* (how many repeat-requests before going up a level) — NOT different truthfulness. The default scored configuration is `standard`. Reporting helpfulness sensitivity is part of the benchmark, mirroring how τ-bench reports user-simulator variance.
5. **Variance budget**: every (task, agent) is run with `R` seeds (default `R=5`). The oracle's seed is yoked to the run seed. Scorer reports mean ± 95% CI (bootstrap) and `pass^k` style stability (fraction of seeds that accept), à la τ-bench's `pass@k`/`pass^k`.
6. **Replay mode**: oracle exchanges are content-addressed (`hash(system_prompt + hidden_spec + query_context)`); in `replay` runs the harness can serve cached oracle responses, giving bit-stable re-scoring and cheap scorer iteration.

### 2.7 Oracle as gateway, not omniscient grader

The oracle decides **accept/reject on intent**, but objective `must_have` gating is the **verifier's** job. The oracle's `review` therefore receives the verifier report and may only:
- accept iff all `must_have` checks pass AND behavior matches `intent`;
- reject with *named, non-leaking* feedback if a `must_have` fails or a `hidden_preference` is violated;
- it may **not** invent new requirements not in `hidden_spec` (prevents goalpost-moving). A reject must cite a `hidden_spec` id; rejects without a citable id are themselves flagged in the trace.

### 2.8 Oracle system-prompt template

```text
You are role-playing a {{persona.type}} who asked an AI coding agent to build something for you.
You are NOT the agent. You never write code unless this prompt's hint rules explicitly authorize a
Level-4 snippet. You speak in the first person as the user.

WHAT YOU WANT (private — never paste verbatim, never dump):
  Intent: {{hidden_spec.intent}}
  Must-haves (objective): {{must_have summaries}}
  Things you care about but will NOT mention unless triggered:
    {{hidden_preferences: desc + reveal_if}}
  Constraints you'll state if asked: {{constraints}}
  Out of scope (say "I don't need that"): {{out_of_scope}}
  Pitfalls you are AWARE of as hints to dispense ONLY per the ladder: {{known_pitfalls}}

PERSONA: domain knowledge = {{persona.domain_knowledge}}. {{persona-specific tone block:
  - non_expert_user: you can't read code; judge by what the program DOES; get confused by jargon;
    you have taste but don't volunteer it; ask the agent to "just show me it working."
  - maintainer: you know the repo conventions; expect tests; reject scope creep and bad style.}}

RULES:
  R1 Always truthful & consistent with WHAT YOU WANT. If asked something not covered, say you don't
     know / don't care, in character.
  R2 Never volunteer hidden_preferences, pitfalls, or solution structure. Reveal a hidden_preference
     ONLY when its reveal_if condition is met by the agent's query.
  R3 Speak only in response to the agent's message, a submission review, or an offered nudge.
  R4 HINT LADDER — give the LOWEST level that answers the question. Escalate one level only if the
     agent re-asks after failing or accepts a stuck-offer. Never jump levels. Never reach Level 5.
       L0 clarify/restate intent or reveal a triggered preference
       L1 point at the area, no answer
       L2 name the concept/pitfall
       L3 describe the fix in prose, NO code
       L4 (only if explicitly begged AND already failed at L3) one small snippet/exact API call
  R5 Judge acceptance by observed BEHAVIOR (the run output you're shown), not by reading the diff.
  R6 If asked about out_of_scope, say it's not needed.
  R7 Never reveal scores, how many criteria remain, or remaining budget.

When reviewing a submission you will be given the verifier report. Accept only if all must-haves pass
AND behavior matches the intent. If you reject, name the specific thing that's wrong by referencing one
of the must-haves/preferences ABOVE — do not invent new requirements.

Respond ONLY as JSON:
{ "level": <0-4>, "text": "<what you say to the agent, in character>",
  "reveals": ["<hidden_spec id you disclosed, if any>"],
  "verdict": "<accept|reject|na>", "cited_criteria": ["<id>"], "refusals": ["<what you declined>"] }
```

---

## 3. Intervention taxonomy

Every oracle exchange, every forced offer, every reject, and certain harness-side events are classified into a **type** and a **severity S ∈ {0..5}**. The taxonomy is closed (the scorer rejects unknown types) and dual-classified: a **rule-based** pass (deterministic, primary for severity-from-level) plus an **LLM-judge** pass (for ambiguous text), with disagreements flagged.

### 3.1 Type enumeration

| `intervention.type` | Trigger | Default S | Notes |
|---|---|---|---|
| `clarification` | AUT `oracle_query{clarify}` answered at L0 | 0 | spec elicitation; counted, not "assistance" |
| `preference_reveal` | a `hidden_preference` released via `reveal_if` | 0 | the agent earned it by asking |
| `scope_confirmation` | oracle confirms in/out of scope | 0 | |
| `hint_nudge` | L1 hint | 1 | |
| `hint_directional` | L2 hint | 2 | |
| `hint_concrete` | L3 hint | 3 | |
| `rescue_snippet` | L4 code/API given | 4 | |
| `takeover` | L5 oracle/human supplies gating impl | 5 | autonomy void for that criterion |
| `proactive_offer_accepted` | stuck-offer accepted → becomes a hint | = resulting hint level | offer itself is S0; accepted hint carries its level |
| `proactive_offer_declined` | stuck-offer declined | 0 | logged, no assistance delivered |
| `submission_reject` | oracle rejects a `submit` | 1 + (named-fix specificity) | reject that *names the fix* is more help → higher S |
| `error_recovery_hint` | oracle helps interpret a failing tool/test on request | 1–3 | severity = ladder level used |
| `human_fallback` | out-of-band human operator intervenes (escape hatch) | 5 | should be ~never in automated runs; flagged loudly |

Severity is **not** simply additive across a run; the scorer reports the full distribution (count per S level), a weighted **Intervention Burden** `IB = Σ w_S · count_S` (default weights `w = [0,1,2,4,8,16]`, super-linear so a single L5 dominates many L1s), and **autonomy** `A = 1 − IB/IB_cap`. L0 events are reported separately as **Clarification Count** and feed a **Clarification Efficiency** metric (did early questions reduce later interventions?).

### 3.2 Mapping to the severity scale

Severity S is the *same* scale the oracle uses for its hint ladder (§2.5), so type→severity is mostly a table lookup. The only computed severities are `submission_reject` (depends on how specific the named fix is) and the two `error_recovery_hint`/`proactive_offer_accepted` cases (inherit the ladder level actually used). This keeps the oracle's self-declared `level` and the logged `severity` mutually checkable.

### 3.3 Auto-classification

Two-stage, both run **offline** from the trace so classification is reproducible and re-runnable:

**Stage A — rule-based (authoritative for severity where possible).**
- `type` is determined by the event's channel + the oracle's structured `level`/`verdict`/`reveals` fields (which are already in the trace). E.g. `oracle_response.level==2` → `hint_directional`, `S=2`.
- Consistency assertions: oracle's self-declared `level` must equal the level implied by `reveals`/`verdict`; otherwise `classifier_conflict=true`.

**Stage B — LLM-judge (only for free-text ambiguity & leakage).**
- A judge model reads `oracle_query.text` + `oracle_response.text` + `hidden_spec` and outputs `{type, severity, leaked_ids[], names_fix: bool}`.
- Used to (a) set `submission_reject` severity via `names_fix`, (b) detect R2 leaks (§2.6), (c) override Stage A only when Stage A returns `ambiguous`.
- Judge runs at `temperature=0`, pinned model, with a rubric prompt; its own tokens are logged. Disagreements between Stage A and Stage B set `needs_review=true`; a small human-audited sample calibrates judge agreement (report Cohen's κ).

This rule-first / judge-second design follows MINT/τ-bench practice: deterministic where the structured signal suffices, LLM-judge only for the genuinely linguistic decisions.

---

## 4. Canonical trace / event log schema (JSON Lines)

One run → one `trace.jsonl`. **Append-only, totally ordered by `seq`.** Every event shares an envelope; `payload` is type-specific. The scorer reads nothing but these files (+ the frozen `hidden_spec` for judging). Hashes make the trace replayable and tamper-evident.

### 4.1 Common envelope (every line)

```json
{
  "schema_version": "1.0.0",
  "run_id": "uuid",
  "seq": 42,                          // monotonic int, total order
  "t_wall": 1718800000.123,           // epoch seconds
  "t_turn": 17,                       // agent-turn index (null for harness-internal events)
  "actor": "agent|harness|oracle|verifier|sandbox|judge",
  "type": "agent_message|tool_call|tool_result|file_edit|oracle_query|oracle_response|intervention|verification_run|oracle_review|budget|termination|run_start|run_end|...",
  "payload": { ... },
  "budgets_after": {                  // snapshot so scorer never recomputes from scratch
    "turns": 17, "wall_s": 312.0, "tokens": 84120, "cost_usd": 0.41, "oracle_queries": 3
  },
  "ev_hash": "sha256(...)"            // hash of canonicalized payload; chain via prev_hash optional
}
```

### 4.2 Event payloads

**`run_start`** (seq 0):
```json
{ "task_id": "...", "task_version": "...", "hidden_spec_sha256": "...",
  "reference_repo": {"url":"...","commit_sha":"..."},
  "agent": {"id":"claude-...|vllm:Qwen...","adapter":"...","decoding":{"temperature":0.0,"top_p":1.0},"endpoint":"api|vllm"},
  "oracle": {"model":"...","prompt_sha256":"...","helpfulness":"standard","proactive_stuck_help":"on","persona":"non_expert_user"},
  "budgets": {"max_turns":80,"max_wall_s":1800,"max_tokens":600000,"max_cost_usd":3.0,"max_oracle_queries":25},
  "seed": 7, "sandbox": {"image_digest":"sha256:...","cpu":4,"mem_gb":8,"network":"allowlist"},
  "harness_version":"...", "git_commit":"..." }
```

**`agent_message`**: `{ "text": "...", "tokens": {"prompt":1200,"completion":300}, "raw_finish_reason":"..." }`

**`tool_call`**: `{ "call_id":"c-19","batch_id":null,"tool":"exec|read_file|write_file|run_tests|http_get","args":{...},"args_sha256":"..." }`

**`tool_result`**: `{ "call_id":"c-19","exit_code":0,"stdout_sha256":"...","stdout_trunc":"first 4KB...","stderr_trunc":"...","wall_ms":820,"truncated":true }`
(Large blobs are content-addressed into a side `blobs/` store; the trace keeps the hash + a truncation so it stays diff-able and small.)

**`file_edit`** (emitted whenever the sandbox fs mutates, derived by the harness, not trusted from the agent):
```json
{ "path":"src/cli.py","op":"create|modify|delete|rename",
  "pre_sha256":"...","post_sha256":"...","added":34,"removed":2,
  "unified_diff_sha256":"...","loc_after":120 }
```

**`oracle_query`**: `{ "qtype":"clarify|hint_request|handoff|confirm","text":"...","context_refs":["seq:38","path:src/cli.py"] }`

**`oracle_response`**: `{ "level":2,"text":"...","reveals":["HP1"],"verdict":"na","cited_criteria":[],"refusals":["..."],"oracle_tokens":{"prompt":900,"completion":80},"oracle_cost_usd":0.006,"latency_ms":700 }`

**`intervention`** (one per assist-bearing exchange; emitted by harness right after the `oracle_response`/reject it summarizes):
```json
{ "ref_seq":57, "channel":"oracle|stuck_offer|review",
  "type":"hint_directional", "severity":2,
  "level_declared":2, "level_classified":2, "classifier_conflict":false,
  "reveals":["..."], "leak_flag":false, "names_fix":null,
  "classified_by":{"stageA":"rule","stageB":null}, "needs_review":false }
```

**`oracle_offer`** (stuck-probe): `{ "reason":"2_failed_verifications","accepted":true|false,"resulting_level":2|null }`

**`verification_run`**:
```json
{ "trigger":"submit|forced_final","entrypoint":"python -m cli ...",
  "must_have":[{"id":"MH1","passed":true,"detail_sha256":"..."}],
  "should_have":[{"id":"SH1","score":0.5,"detail_sha256":"..."}],
  "all_must_pass":true, "rubric_score":0.5, "wall_ms":4200, "runner_image_digest":"sha256:..." }
```

**`oracle_review`**: `{ "ref_verification":SEQ,"verdict":"accept|reject","feedback":"...","cited_criteria":["MH1"],"oracle_tokens":{...} }`

**`budget`** (debit log; one per action OR coalesced per turn): `{ "kind":"token|cost|turn|wall|oracle_query","amount":300,"reason":"agent_message" }`

**`termination`**: `{ "reason":"accept|budget_turns|budget_tokens|budget_cost|budget_wall|budget_oracle|give_up|error","detail":"..." }`

**`run_end`** (last line): final budget snapshot, `accepted: bool`, `final_rubric_score`, counts of interventions by severity (a *cache* of scorer-derivable values — the scorer recomputes and asserts equality as an integrity check), `invalid: bool`, `invalid_reason`.

### 4.3 Schema invariants the scorer enforces

- `seq` strictly increasing, no gaps; exactly one `run_start` and one `run_end` (unless `invalid:error`).
- Every `tool_result.call_id` matches a prior `tool_call.call_id`.
- Every `intervention.ref_seq` points at an `oracle_response`/`oracle_review`/`oracle_offer`.
- `budgets_after` monotonic non-decreasing per kind.
- Sum of `intervention` severities reconstructs `IB`; mismatch with `run_end` cache ⇒ trace rejected.
- Any `leak_flag:true` or `human_fallback` ⇒ run flagged for human audit before inclusion.

This schema is the **contract**: if a metric can't be computed from these fields, either the metric or the schema is wrong — fix the schema, don't add runtime state.

---

## 5. Reference harness architecture

Agent-framework-agnostic, driving both API agents and a vLLM-served local agent. Python, asyncio, single process per run for determinism.

### 5.1 Components

```
usability_harness/
  runner.py            # owns the loop (§1.2), clock, termination, writes trace
  budgets.py           # multi-ceiling debit + snapshot
  trace.py             # JSONL writer, envelope, ev_hash, blob store
  adapters/
    base.py            # AgentAdapter ABC
    anthropic.py       # API agent (Claude) — usage from response headers
    openai.py          # API agent (OpenAI)
    vllm_openai.py     # local open-weight via vLLM's OpenAI-compatible server
    react_shim.py      # wraps a tool-using ReAct/agent loop into one-action-per-step
  sandbox/
    base.py            # SandboxBackend ABC: run(cmd), read/write, snapshot, diff
    docker.py          # rootless Docker / Podman backend
    apptainer.py       # Apptainer/Singularity backend for DAIC compute nodes
  tools/
    registry.py        # tool allowlist + arg schemas (exec, fs, run_tests, http_allowlist)
  oracle/
    gateway.py         # OracleGateway: routes query->LLM, caches (replay), enforces JSON
    prompt.py          # system-prompt template (§2.8)
  verifier/
    runner.py          # executes must_have/should_have checks in a fresh sandbox snapshot
  classify/            # OFFLINE: ruleA + judgeB (§3.3); not in the runtime hot path
  config/
    run.yaml           # (task, agent, oracle, budgets, seed) — fully declares a run
```

### 5.2 Key interfaces

```python
class AgentAdapter(ABC):
    async def reset(self, task_prompt: str, tool_schemas: list[ToolSchema]) -> None: ...
    async def step(self, observation: Observation) -> AgentAction: ...      # exactly one action
    def usage(self) -> Usage: ...   # {prompt_tokens, completion_tokens, cost_usd}

class SandboxBackend(ABC):
    async def exec(self, cmd: list[str], timeout_s: int) -> ExecResult: ...
    async def read(self, path: str) -> bytes: ...
    async def write(self, path: str, data: bytes) -> FileEdit: ...          # returns diff event
    async def snapshot(self) -> SnapshotId: ...                              # for verifier isolation
    async def restore(self, snap: SnapshotId) -> None: ...

class OracleGateway:
    async def handle(self, query: OracleQuery, ctx: RunContext) -> OracleResponse: ...
    async def review(self, vrun: VerificationRun, ctx: RunContext) -> OracleReview: ...
    async def maybe_offer(self, ctx: RunContext) -> Optional[OracleOffer]: ...  # stuck-probe
```

The **adapter is the only framework-specific code.** A SWE-agent-style, an OpenHands-style, or a bare-tool-calling agent each get a thin adapter that conforms to `step()` returning one `AgentAction`. Native multi-tool-call turns are unrolled by `react_shim` into sequential `tool_call`s sharing `batch_id`, preserving the one-action-per-step trace invariant.

### 5.3 Sandboxing the agent's file/exec actions

- **Isolation**: each run gets a fresh container from a pinned image (`image_digest` in `run_start`). Backends: **Docker/Podman** on the login/CPU node; **Apptainer** on DAIC compute nodes (rootless, SLURM-friendly, the standard HPC choice). The same `SandboxBackend` ABC abstracts both.
- **Filesystem**: agent works in `/work` (overlay/tmpfs). The harness derives `file_edit` events by diffing snapshots — it never trusts the agent's claim of what it wrote. Path traversal outside `/work` is denied by the tool layer.
- **Exec**: commands run as an unprivileged user, `timeout_s` enforced, output captured + content-addressed. CPU/mem cgroup limits from `run_start.sandbox`.
- **Network**: default **deny**, with a per-task **allowlist** (e.g. only PyPI mirror for installs, no general internet) — this matters on DAIC where compute nodes are already network-restricted; the allowlist is honored on the login node too for parity. Network policy is recorded so runs are comparable.
- **Verifier isolation**: verification runs on a *snapshot* of `/work` in a *fresh* container so a misbehaving build can't poison grading, and so the verifier image (`runner_image_digest`) is pinned independently of the dev image.

### 5.4 Driving both API and local (vLLM) agents

- **API agents**: `anthropic.py` / `openai.py` call the hosted API from the **login/CPU node** (has internet). Usage/cost from response metadata; cost priced via a per-model `prices.yaml`.
- **Local agents**: a vLLM **OpenAI-compatible server** is launched on a GPU node (`--gres=gpu:a40:1` etc.); `vllm_openai.py` is literally the OpenAI adapter pointed at the local base-url. Decoding (`temperature`, `seed`) pinned in `run_start`. The **oracle stays API-based** for consistency regardless of which agent is under test — so oracle behavior is a constant across the agent grid.
- **DAIC topology**: data collection + API agent + oracle on login/CPU node; vLLM server + local-agent runs as a GPU SLURM job; shared state on `/tudelft.net/staff-umbrella/CoReFusion` (traces, blobs, hidden_specs); home dir kept clean. Replay-mode oracle caching means re-scoring needs no GPU and no API spend.

### 5.5 Determinism & reproducibility hooks

- Everything that affects outcomes is captured in `run_start` (image digests, model ids, decoding, seed, budgets, oracle prompt hash) and `run.yaml`.
- `temperature=0` + fixed seeds where the backend honors them; `R` replicas otherwise quantify residual stochasticity.
- Oracle replay cache (content-addressed) for bit-stable re-scoring.
- Offline classification/scoring re-runnable from `trace.jsonl` + frozen `hidden_spec` with zero runtime dependency.

---

## 6. End-to-end example (abridged trace)

```jsonl
{"seq":0,"actor":"harness","type":"run_start","payload":{"task_id":"cal-001","agent":{"id":"claude-opus-4-8"},"oracle":{"persona":"non_expert_user","helpfulness":"standard","proactive_stuck_help":"on"},"budgets":{"max_turns":80,"max_oracle_queries":25}},"t_turn":null}
{"seq":1,"actor":"agent","type":"agent_message","payload":{"text":"I'll build a CLI that reads an .ics file. What format do you want the output in?"},"t_turn":1}
{"seq":2,"actor":"agent","type":"oracle_query","payload":{"qtype":"clarify","text":"Do you want JSON, a table, or plain text output?"},"t_turn":1}
{"seq":3,"actor":"oracle","type":"oracle_response","payload":{"level":0,"text":"A table I can read in my terminal, please.","reveals":["HP1"],"verdict":"na"},"t_turn":1}
{"seq":4,"actor":"harness","type":"intervention","payload":{"ref_seq":3,"type":"preference_reveal","severity":0,"level_declared":0,"level_classified":0,"leak_flag":false},"t_turn":1}
{"seq":12,"actor":"sandbox","type":"file_edit","payload":{"path":"src/cli.py","op":"create","added":40,"post_sha256":"..."},"t_turn":4}
{"seq":20,"actor":"agent","type":"tool_call","payload":{"call_id":"c-20","tool":"run_tests","args":{"k":"test_groups_by_day"}},"t_turn":7}
{"seq":21,"actor":"sandbox","type":"tool_result","payload":{"call_id":"c-20","exit_code":1,"stderr_trunc":"AttributeError: 'date' object has no attribute 'time'"},"t_turn":7}
{"seq":34,"actor":"harness","type":"oracle_offer","payload":{"reason":"2_failed_verifications","accepted":true,"resulting_level":2},"t_turn":11}
{"seq":35,"actor":"oracle","type":"oracle_response","payload":{"level":2,"text":"All-day events don't have a time of day — you're treating them like timed events."},"t_turn":11}
{"seq":36,"actor":"harness","type":"intervention","payload":{"ref_seq":35,"type":"hint_directional","severity":2,"level_declared":2,"level_classified":2},"t_turn":11}
{"seq":50,"actor":"agent","type":"submit","payload":{"summary":"CLI groups events by day, table output","entrypoint":"python -m cli sample.ics"},"t_turn":18}
{"seq":51,"actor":"verifier","type":"verification_run","payload":{"trigger":"submit","must_have":[{"id":"MH1","passed":true}],"all_must_pass":true,"rubric_score":0.8},"t_turn":18}
{"seq":52,"actor":"oracle","type":"oracle_review","payload":{"verdict":"accept","cited_criteria":["MH1"]},"t_turn":18}
{"seq":53,"actor":"harness","type":"termination","payload":{"reason":"accept"},"t_turn":18}
{"seq":54,"actor":"harness","type":"run_end","payload":{"accepted":true,"final_rubric_score":0.8,"interventions_by_severity":{"0":1,"2":1},"IB":2,"clarification_count":1,"invalid":false}}
```

Scorer-derived headline metrics for this run: **accepted=1**, **IB=2** (one L2 hint), **autonomy A=1−2/IB_cap**, **clarification_count=1** (asked the right format question up front → efficient), **rubric=0.8**, **turns=18/80**, **oracle_queries=2/25**.

---

## 7. Open knobs & defaults (for the experiment-design workstream)

| Knob | Default | Why it's a knob |
|---|---|---|
| `R` (seeds/episode) | 5 | variance vs. compute |
| `oracle.helpfulness` | standard | sensitivity reporting (strict/standard/lenient) |
| `oracle.proactive_stuck_help` | on | ablate "does a watchful user help usability?" |
| persona | per-task | non_expert_user vs maintainer comparison |
| severity weights `w_S` | [0,1,2,4,8,16] | super-linear; tunable in scorer, not runtime |
| budgets | §1.3 table | per-task scaled by reference repo size |

Nothing in §7 changes the trace schema — only which runs we launch and how the scorer weights them. That separation (runtime produces traces; scorer/experiment design interprets them) is the core architectural commitment of this protocol.

---

**Deliverable files (for repo integration):** this spec maps to `docs/protocol.md`, the schema in §4 to `schemas/trace.schema.json`, the oracle template in §2.8 to `usability_harness/oracle/prompt.py`, and the interfaces in §5.2 to `usability_harness/{adapters,sandbox,oracle}/base.py`. The interface between this workstream and the task-curation workstream is `hidden_spec` (§2.1); the interface to the scorer is `trace.jsonl` + frozen `hidden_spec` (§4).