I have enough to produce the deliverable. The task is a self-contained design document — I'll return the markdown directly as my final response rather than writing a file, per the instructions.

# Usability-Benchmark: Metric Suite Specification

**Component:** Usability metrics (the scientific core)
**Status:** v0.1 — designed to be turned directly into `eval/metrics.py` + `schemas/run_log.schema.json`
**Audience:** benchmark engineers, reviewers, downstream analysts

---

## 0. Preliminaries: the data model everything is computed from

All metrics are pure functions of a **per-episode run log** (one task × one model × one seed = one *episode*). To make metric definitions unambiguous, we first fix the event schema. Every metric below cites the exact fields it consumes. If a metric can't be traced to a logged field, it doesn't ship.

### 0.1 Episode object

```jsonc
{
  "episode_id": "uuid",
  "task_id": "calendar-analyzer-easy-003",
  "model": "claude-opus-4-8",          // agent under test
  "oracle_model": "claude-opus-4-8",   // simulated user (fixed across all episodes)
  "seed": 7,
  "difficulty": 2,                      // 1..5, from task card (see §6.2)
  "started_at": "...", "ended_at": "...",
  "wall_clock_s": 1843.2,
  "terminated_reason": "accepted | budget_exhausted | agent_gaveup | oracle_takeover | crash",
  "events": [ Event, ... ],            // ordered, monotonic ts
  "checkpoints": [ Checkpoint, ... ],  // acceptance evaluations over time
  "final_acceptance": AcceptanceResult
}
```

### 0.2 Event types (the interaction trace)

Every turn boundary and every agent↔oracle exchange is an `Event`. The two that carry the novel signal are `oracle_query` (agent → oracle) and `oracle_response` (oracle → agent, **labeled with a severity**).

```jsonc
// Agent does internal work (no human contact)
{ "type": "agent_action", "ts": ..., "turn": 14,
  "subtype": "tool_call | file_edit | code_run | plan | message_to_user",
  "tokens_in": 5120, "tokens_out": 880, "cost_usd": 0.041,
  "tool_name": "bash", "edit_loc": 23, "test_invocation": true,
  "self_test_passed": false }

// Agent solicits the human (oracle). THIS is what we count.
{ "type": "oracle_query", "ts": ..., "turn": 15,
  "query_text": "...",
  "query_class": "clarification | spec_request | hint_request | confirmation | handoff_request | bug_report",
  "agent_blocked": true }     // did the agent declare itself stuck/blocked?

// Oracle answers, and the SAME LLM call that answers also emits a severity label
// against the rubric in §3.1 (graded by the oracle policy, audited offline).
{ "type": "oracle_response", "ts": ..., "turn": 15,
  "responds_to": "<oracle_query event id> | null",  // null = UNSOLICITED intervention
  "severity": 2,                       // 0..5, the assistance severity scale (§3.1)
  "severity_rationale": "...",
  "info_units_revealed": ["requirement:export_csv", "constraint:no_network"],
  "tokens_in": ..., "tokens_out": ..., "cost_usd": ... }

// An acceptance evaluation snapshot (run by the harness, not the oracle persona)
{ "type": "checkpoint", "ts": ..., "turn": 22,
  "criteria_passed": 6, "criteria_total": 9,
  "is_working_version": true }        // ≥1 end-to-end runnable artifact exists
```

### 0.3 AcceptanceResult (the gold check)

Acceptance is **not** a single LLM vibe-judgment. Each task card ships a weighted **acceptance criteria checklist** (§6.1), and each criterion is scored by the cheapest sufficient verifier:

```jsonc
{
  "criteria": [
    { "id": "c1", "desc": "produces a CSV with columns [date,hours]",
      "verifier": "programmatic",        // programmatic | unit_test | oracle_judgment
      "weight": 0.25, "passed": true, "evidence": "stdout hash ..." },
    { "id": "c2", "desc": "handles empty calendar gracefully",
      "verifier": "unit_test", "weight": 0.15, "passed": true },
    { "id": "c5", "desc": "output is readable / well-organized",
      "verifier": "oracle_judgment", "weight": 0.10, "passed": false,
      "judge_score": 0.4, "judge_rationale": "..." }
  ],
  "weighted_score": 0.78,               // Σ weight·passed  (oracle_judgment uses judge_score∈[0,1])
  "accepted": true                      // weighted_score ≥ task.accept_threshold (default 0.80)
}
```

Verifier precedence (to minimize LLM-judge variance, an idea borrowed from **SWE-bench**'s FAIL_TO_PASS/PASS_TO_PASS test gating and **HumanEval**'s programmatic checks): prefer `programmatic` > `unit_test` > `oracle_judgment`. Subjective criteria are capped (default ≤30% of total weight per task) so a task can never be won or lost purely on a judge's mood.

---

## 1. Dimension A — Goal achievement / problem-solving success

The benchmark is open-ended, so "pass/fail" alone wastes signal. We report a continuous partial-credit score **and** a strict binary, plus *when* success arrived.

| id | name | def (operational) | computed from | unit | dir |
|----|------|-------------------|---------------|------|-----|
| **A1** | `success_binary` | `final_acceptance.accepted` | AcceptanceResult.accepted | {0,1} | ↑ |
| **A2** | `criteria_score` | `final_acceptance.weighted_score` ∈ [0,1] — weighted fraction of acceptance criteria met | per-criterion `passed`/`judge_score` × `weight` | ratio | ↑ |
| **A3** | `core_criteria_score` | Same as A2 but restricted to criteria flagged `is_core=true` (functional must-haves), ignoring nice-to-haves. Guards against gaming A2 via easy peripheral criteria. | criteria subset | ratio | ↑ |
| **A4** | `goal_drift` | 1 − (cosine sim between the *delivered artifact's capability summary* and the *gold goal embedding*), judged by oracle on a fixed 0–1 rubric. Detects "built something that passes some checks but isn't what was asked." | oracle_judgment over final artifact vs `task.gold_summary` | ratio | ↓ |
| **A5** | `regression_free` | Fraction of `PASS_TO_PASS`-style invariant checks still passing at the end (did later edits break earlier working features?) | checkpoint test history | ratio | ↑ |

**Notes.** A2 vs A3 separation is the main anti-gaming lever in this dimension: an agent that pads with trivially-satisfiable criteria can lift A2 but not A3. A4 is the open-ended analogue of "did you solve the *right* problem," which closed-form benchmarks get for free from the test suite.

---

## 2. Dimension B — Interaction / intervention load (the novelty)

This is the dimension that distinguishes us from SWE-bench. It counts and times the agent's reliance on the human. **Counts are direction-ambiguous on their own** (zero questions can mean "brilliantly autonomous" *or* "charged ahead and built the wrong thing"), so every B-metric is reported *jointly with A1/A2* and folded into composites in §8 — never read alone.

| id | name | def | computed from | unit | dir |
|----|------|-----|---------------|------|-----|
| **B1** | `n_interventions` | Count of `oracle_response` events (solicited + unsolicited) | events | count | ↓\* |
| **B2** | `n_clarifications` | Count of `oracle_query` with `query_class=clarification` | events | count | ↓\* |
| **B3** | `n_hint_requests` | `oracle_query` with class ∈ {hint_request} | events | count | ↓ |
| **B4** | `n_corrections` | `oracle_response` that are **unsolicited** (`responds_to=null`) — the oracle proactively corrected a wrong path | events | count | ↓ |
| **B5** | `n_handoffs` | `oracle_query` class=handoff_request, or `terminated_reason=oracle_takeover` | events | count | ↓ |
| **B6** | `turns_to_first_working` | `turn` of first checkpoint with `is_working_version=true` (∞ → censored at budget) | checkpoints | turns | ↓ |
| **B7** | `turns_to_acceptance` | `turn` of first checkpoint with weighted_score ≥ threshold | checkpoints | turns | ↓ |
| **B8** | `interventions_to_acceptance` | # `oracle_response` before the accepting checkpoint | events | count | ↓ |
| **B9** | `mean_query_class_entropy` | Shannon entropy over the agent's `query_class` distribution — a *descriptive* texture metric (does it only ever ask one kind of thing?) | events | nats | — |
| **B10** | `time_blocked_fraction` | Σ wall-clock spent in `agent_blocked=true` segments ÷ total wall-clock | timestamps | ratio | ↓ |

\* "lower-is-better" holds **conditional on equal or better A2**. We never reward question-avoidance that costs success — see the `UnderAsk` penalty in §8.3.

---

## 3. Dimension C — Human-assistance AMOUNT & SEVERITY

Counts (Dim B) are blunt: one "here's the whole function" is worth more than ten "did you mean UTC?". This dimension grades *how much help each interaction actually transferred* and aggregates it into a single **assistance cost**.

### 3.1 Severity scale (the graded rubric)

Each `oracle_response` is labeled by the oracle policy on this 0–5 scale. The labels are **constrained, not free**: the oracle is a deterministic-temperature policy whose system prompt forbids it from exceeding the requested severity, and labels are re-graded offline by an independent judge for the variance audit (§7).

| sev | name | meaning | example |
|-----|------|---------|---------|
| **0** | none | acknowledgement only, no task info | "Sounds good." |
| **1** | trivial clarification | restates/disambiguates already-stated info | "Yes, the CSV, not JSON." |
| **2** | substantive spec info | reveals a requirement/constraint not previously derivable | "It must also handle recurring events." |
| **3** | directional hint | points toward the solution approach without giving it | "Look at how `icalendar` parses RRULE." |
| **4** | partial solution | supplies a working fragment (code/config/algorithm) | gives the parsing function |
| **5** | takeover | oracle does the step / completes the task for the agent | writes the module, or `oracle_takeover` |

### 3.2 Metrics

| id | name | def | computed from | unit | dir |
|----|------|-----|---------------|------|-----|
| **C1** | `assistance_cost` (**AC**) | Σ over oracle_responses of `w(sev)`, with default convex weights **w = [0,1,3,6,12,25]** for sev 0..5. Convexity encodes "one sev-5 ≫ many sev-1." | oracle_response.severity | points | ↓ |
| **C2** | `max_severity` | max severity reached in the episode (a sev-5 anywhere is a qualitatively different episode) | oracle_response.severity | 0–5 | ↓ |
| **C3** | `severity_histogram` | vector of counts per severity level (reported, used for stratified analysis) | oracle_response.severity | counts | — |
| **C4** | `spec_info_transferred` | # distinct `info_units_revealed` of class `requirement`/`constraint` the oracle had to give — i.e., spec the agent *failed to elicit or infer*. Normalized by `task.n_hidden_spec_units`. | oracle_response.info_units_revealed | ratio | ↓ |
| **C5** | `solution_leakage` | fraction of the *final accepted solution's* key components whose origin traces to a sev≥4 oracle response (provenance tagging: code spans contributed by the oracle are marked). High → "the human basically built it." | oracle_response provenance × final artifact | ratio | ↓ |
| **C6** | `assistance_efficiency` | A2 gained per unit AC: `Δcriteria_score / (AC + 1)`. How much success each point of help bought. | A2, C1 | ratio | ↑ |

**Why convex weights:** linear weights let an agent trade many tiny clarifications for one takeover at the same cost, which is wrong — a takeover means the agent *couldn't* do it. Defaults are tunable in `metrics.yaml`; §8.4 shows the score is robust to the exact numbers within the convex family.

---

## 4. Dimension D — Autonomy / self-sufficiency

Where Dim C measures help *received*, Dim D measures progress made *without* help and the agent's ability to dig itself out.

| id | name | def | computed from | unit | dir |
|----|------|-----|---------------|------|-----|
| **D1** | `autonomy_ratio` | criteria score attributable to unaided work: `A2 × (1 − C5)` — final success discounted by leaked solution share | A2, C5 | ratio | ↑ |
| **D2** | `unaided_progress_fraction` | Σ positive Δ`criteria_passed` across checkpoint intervals containing **no** intervening sev≥2 oracle_response, ÷ total positive Δ. "Fraction of forward progress made in human-free stretches." | checkpoints + events | ratio | ↑ |
| **D3** | `self_recovery_rate` | Of all *agent-detected* failures (a failed `code_run`/`self_test_passed=false` followed within K turns by a passing one **with no intervening sev≥2 help**), fraction the agent fixed itself. The error-recovery analogue of SWE-agent's "did it iterate." | agent_action self-test fields | ratio | ↑ |
| **D4** | `blocked_resolution_self` | fraction of `agent_blocked` episodes-segments exited **without** a sev≥3 response | events | ratio | ↑ |
| **D5** | `proactive_inference` | # `info_units` the agent correctly satisfied that were *hidden spec* and were **never revealed by the oracle** (inferred from context/reference repo) ÷ `n_hidden_spec_units`. Rewards good inference, the positive complement of C4. | AcceptanceResult vs oracle_response history | ratio | ↑ |

D5 is important: it's the metric that *rewards* an agent for figuring things out instead of asking, which together with the §8.3 under-ask penalty pins down the desired behavior from both sides.

---

## 5. Dimension E — Efficiency

Resource cost per unit of achieved progress. All raw totals are logged; the headline metrics are *cost-per-progress* so that a model isn't punished for spending more to achieve more.

| id | name | def | computed from | unit | dir |
|----|------|-----|---------------|------|-----|
| **E1** | `wall_clock_s` | end − start | timestamps | s | ↓ |
| **E2** | `tokens_total` | Σ tokens_in+out over agent_actions (oracle tokens tracked separately as `oracle_tokens`) | agent_action.tokens_* | tokens | ↓ |
| **E3** | `cost_usd_total` | Σ cost_usd (agent) + oracle cost (reported split) | *.cost_usd | USD | ↓ |
| **E4** | `n_tool_calls` | count agent_action.subtype=tool_call | events | count | ↓ |
| **E5** | `n_edits` / `edit_churn` | count file_edits / Σ edit_loc (churn flags thrashing) | agent_action.edit_loc | count/LOC | ↓ |
| **E6** | `iterations` | # code_run→edit cycles | events | count | ↓ |
| **E7** | `cost_per_progress` | `cost_usd_total / max(A2, ε)` — headline efficiency | E3, A2 | USD/unit | ↓ |
| **E8** | `tokens_per_progress` | `tokens_total / max(A2, ε)` | E2, A2 | tok/unit | ↓ |

Reporting agent vs oracle cost **separately** matters: the oracle is a fixed cost of the harness, and a cheap agent that offloads thinking onto an expensive oracle should not look efficient. `cost_usd_total` for ranking uses agent-side only; oracle cost is a covariate.

---

## 6. Per-task grounding (so the above is computable)

### 6.1 Task card schema (`tasks/<id>.yaml`)
```yaml
id: calendar-analyzer-easy-003
prompt_user: "help me build a tool that analyzes my calendar and tells me where my time goes"
reference_repos: ["https://github.com/<org>/<cal-tool>"]   # grounds gold + criteria
gold_summary: "CLI that ingests .ics, buckets events by category, outputs weekly time report"
difficulty: 2
accept_threshold: 0.80
n_hidden_spec_units: 6          # requirements NOT stated in prompt_user, derivable from refs
hidden_spec:
  - {id: req:recurring, class: requirement, desc: "handle RRULE recurring events"}
  - {id: con:offline,  class: constraint,  desc: "must run with no network"}
acceptance_criteria:
  - {id: c1, is_core: true,  weight: 0.25, verifier: programmatic, ...}
  - {id: c5, is_core: false, weight: 0.10, verifier: oracle_judgment, ...}
oracle_persona: "non-expert user who knows what they want but not how"
```

### 6.2 Difficulty (for normalization)
`difficulty ∈ {1..5}` is set per task from three logged proxies, fit once on a calibration set: (i) `n_hidden_spec_units`, (ii) reference-repo size/complexity (LOC, # modules), (iii) median `turns_to_acceptance` of a reference agent panel. Used only for normalization (§7), never as ground truth.

---

## 7. Normalization, difficulty, and stochasticity

### 7.1 Per-task difficulty normalization
Raw counts (turns, AC) scale with difficulty, so cross-task aggregates use **difficulty-normalized z-scores** computed against a fixed **reference agent panel** P (a frozen set of baseline agents run once per task):

- For metric m on task t: `m̃ = (m − μ_{P,t}) / σ_{P,t}`, where μ,σ are panel mean/std on task t.
- Direction-corrected so higher z = better.
- This makes "AC of 12 on a hard task" comparable to "AC of 4 on an easy task," analogous to **HELM**'s per-scenario normalization and **BigBench**'s normalized-preferred-metric.

For the assistance-cost composite we also report a **difficulty-relative AC**: `AC / E[AC_P,t]` (1.0 = average help for this task).

### 7.2 Stochasticity → Dimension F (robustness)
Each (task, model) is run with **N≥5 seeds** (oracle temperature fixed low, e.g. 0.2; agent at its default). Robustness metrics:

| id | name | def | from | dir |
|----|------|-----|------|-----|
| **F1** | `pass^k` | prob. all k of k random reruns are accepted = `Π` est. via `C(c,k)/C(n,k)` with c successes of n seeds (the **pass^k** estimator from Anthropic/METR reliability work; stricter sibling of HumanEval **pass@k**) | A1 across seeds | ↑ |
| **F2** | `pass@k` | prob. ≥1 of k accepted (unbiased estimator) | A1 across seeds | ↑ |
| **F3** | `success_rate` | mean A1 over seeds | A1 | ↑ |
| **F4** | `score_cv` | coefficient of variation of A2 across seeds (consistency of *partial* credit) | A2 | ↓ |
| **F5** | `assistance_cv` | CV of AC across seeds (is help demand stable or luck-dependent?) | C1 | ↓ |
| **F6** | `usability_iqr` | inter-quartile range of the §8 Usability Score across seeds (reported with every headline number) | §8 | ↓ |

**All headline numbers are reported as median ± IQR over seeds, with N stated.** A single-seed number is never published. `pass^k` is the recommended headline for "can you trust this agent to do it again."

---

## 8. Composite scores

Two composites: one **outcome-centric** Usability Score (headline), one **process-centric** Interaction Efficiency Index. Both are computed *per task* on normalized inputs, then aggregated (median) across tasks.

### 8.1 Inputs (all ∈ [0,1], per task, direction-corrected)
- `S = core_criteria_score` (A3) — capped success signal.
- `H = 1 − min(1, AC_rel)` where `AC_rel = AC / (κ · E[AC_P,t])` — *assistance-light* score (1 = no help, 0 = ≥κ× the panel-average help). Default κ=2.
- `A = autonomy_ratio` (D1).
- `E = 1 / (1 + cost_per_progress / median_cost_per_progress_P,t)` — *efficiency* score.
- `R = pass^k` (F1) at k=2 (robustness).

### 8.2 Usability Score (headline)
A **geometric mean** of success and assistance-lightness (so neither can be faked away to zero), times multiplicative efficiency/robustness discounts:

```
USABILITY = ( S^α · H^β )^(1/(α+β))  ·  E^γ  ·  R^δ
defaults:  α = 0.55 ,  β = 0.45 ,  γ = 0.20 ,  δ = 0.20
```

Rationale for geometric (not arithmetic) coupling of S and H: it enforces that **you must both succeed AND do it with little help** — an agent that scores S=1 by demanding constant takeovers (H→0) gets USABILITY→0, and an agent that asks nothing but fails (S→0) also gets 0. This is the precise behavioral target of the whole benchmark. E and R are exponential discounts so they tune ranking without dominating the success/assistance core.

A **secondary linear variant** is also reported for interpretability and ablation:
`USABILITY_lin = w·[S, H, A, E, R]`, default `w = [.35,.30,.15,.10,.10]`.

### 8.3 Anti-gaming guards (must be enforced, not optional)

The central failure modes and their countermeasures:

1. **Never-ask (charge ahead blindly).** Caught by S: skipping clarification on an *under-specified* task tanks `core_criteria_score`. Reinforced by an explicit **UnderAsk penalty**: if `success_binary=0` AND `n_clarifications=0` AND `goal_drift` (A4) > τ, apply `H ← H · (1−ρ)` (default ρ=0.5). You cannot bank the "no help" reward while having built the wrong thing.

2. **Over-ask (interrogate the oracle).** Caught directly by H via AC, and additionally by E (each query costs turns/tokens). The convex severity weights mean spamming sev-1 questions still accrues cost.

3. **Criteria-stuffing / peripheral wins.** Headline success uses **A3 (core only)**; A2 is reported but not in the composite.

4. **Solution laundering** (let oracle write it, claim success). Caught by `solution_leakage` (C5) feeding `autonomy_ratio` and by the sev-5 weight (25) in AC.

5. **Oracle sycophancy / over-helping** (the *oracle* leaks too much, inflating everyone's S). Caught at harness level: oracle is severity-capped by prompt, severities are independently re-graded (§7), and `spec_info_transferred` (C4) is monitored as a per-task QA metric — a task where every agent extracts the same spec for free is flagged for redesign.

6. **Seed cherry-picking.** Impossible: headline = median over fixed seeds, and `R=pass^k` punishes lucky-single-run success.

### 8.4 Weight sensitivity
We ship a `sensitivity` notebook that perturbs (α,β,γ,δ, severity weights w, κ) over plausible ranges and reports **rank stability** (Kendall's τ of the agent leaderboard). A finding is only reported as a ranking claim if τ ≥ 0.9 across the default neighborhood; otherwise it's reported as "within noise."

---

## 9. Optional Dimension G — UX quality of the agent's OWN communication

Did the agent communicate *well*, independent of whether it needed help? Graded by an LLM judge on a fixed rubric over the agent's `oracle_query` and `message_to_user` events; reported as a side-panel, **not** in the headline composite (to keep the headline judge-light).

| id | name | def | from | dir |
|----|------|-----|------|-----|
| **G1** | `question_quality` | mean judge score (0–1): were questions specific, answerable, non-redundant, batched sensibly? | oracle_query texts | ↑ |
| **G2** | `redundant_query_rate` | fraction of queries whose answer was already available in prior context | query texts vs transcript | ↓ |
| **G3** | `status_transparency` | judge score: did the agent keep the user informed of plan/progress before acting? | message_to_user events | ↑ |
| **G4** | `false_confidence` | rate of agent assertions later contradicted by its own tests / the oracle (mis-calibration) | agent claims vs checkpoints | ↓ |

---

## 10. What ships in the repo

- `schemas/run_log.schema.json` — the §0 event schema (validation gate; an episode failing schema validation is excluded).
- `eval/metrics.py` — pure functions `Episode → {metric_id: value}`, one per §1–§9 row, each unit-tested on hand-built fixture episodes.
- `eval/composite.py` — §8 formulas, weights in `eval/metrics.yaml`.
- `eval/normalize.py` — §7 panel z-scores + difficulty-relative AC.
- `eval/aggregate.py` — median ± IQR over seeds, `pass^k`/`pass@k` estimators, sensitivity sweep.
- `eval/oracle_severity_audit.py` — offline re-grading + inter-rater agreement (Cohen's κ between oracle and independent judge) on the severity labels; reported as a benchmark-validity number.

---

## 11. One-line summary of the design's logic

Success (A) tells you *if* the goal was met; the Interaction/Severity dimensions (B, C) tell you *how much human it cost*; Autonomy (D) credits unaided progress and self-recovery; Efficiency (E) normalizes by resources; Robustness (F) makes it reproducible under LLM stochasticity; and the **geometric Usability Score forces an agent to be simultaneously successful and low-assistance**, with explicit penalties closing the never-ask / over-ask / solution-laundering loopholes that a naive count-based metric would reward.

---

**Relevant deliverable paths (for the orchestrator / sibling agents):**
- This specification is the content for `/Users/davidhuang/Desktop/usabilityBenchMark/docs/metrics.md`.
- It defines the contracts other components must satisfy: the **oracle component** must emit `oracle_response.severity` (§3.1) and `info_units_revealed`; the **task-collection component** must populate `hidden_spec`, `n_hidden_spec_units`, and the weighted `acceptance_criteria` (§6.1); the **harness/infra** must produce the §0 event log and run **N≥5 seeds** per (task, model) for §7.