I have enough grounding. Now I'll write the scoring & evaluation specification document. This is my deliverable — I'll return it as the final message.

I need to be consistent with a "metrics expert's" composite Usability Score formula. Since I don't have their exact document, I'll define a concrete, defensible formula and flag explicitly where it must be reconciled with their design, restating it consistently.

```markdown
# Usability-Benchmark — Scoring & Evaluation Specification

**Component:** `scoring/` (verification, rubric, judge, aggregation, statistics)
**Status:** v0.1 — buildable spec. Sections marked `⚠ RECONCILE` must be kept numerically identical to the metrics expert's `metrics/` module (single source of truth: `bench/spec/usability_score.yaml`).
**Audience:** implementers of the evaluation harness + leaderboard.

---

## 0. Scope & philosophy

This benchmark does **not** ask "did the agent pass the hidden test suite". It asks two coupled questions:

1. **Goal achievement (GA):** how good is the delivered software artifact, relative to the task's hidden specification and a real reference implementation?
2. **Usability cost (UC):** how much human assistance/intervention did the agent consume to get there?

The final number rewards **building the right thing with minimal human babysitting**. A perfect artifact that required the oracle to spoon-feed the entire design is *not* a usable agent. An agent that never asks and confidently ships the wrong thing is also not usable. The scoring must make both failure modes cost something.

The hard part is that GA is over **open-ended deliverables** ("build me a tool that analyzes my calendar"). We grade those with a three-channel verifier:

- **V1 — Functional/sandbox execution** (objective, binary-ish, cheap to trust).
- **V2 — Rubric / acceptance-criteria checklist** (semi-objective, derived from `hidden_spec`, each item independently checkable).
- **V3 — LLM-as-judge** (subjective quality, reference-anchored, bias-controlled).

GA is a weighted combination of V1+V2+V3. GA then composes with the metrics expert's interaction/assistance metrics into the leaderboard **Usability Score**.

Prior art we borrow from explicitly (named inline): τ-bench (simulated-user oracle + `pass^k` reliability), SWE-bench (sandboxed execution + golden tests, as a *component* not the whole grade), ScienceAgentBench (multi-channel rubric: success rate / valid execution rate / similarity / cost), and the LLM-as-judge bias literature (position-swap calibration, panel-of-judges, rubric decomposition).

---

## 1. Task data contract (what scoring consumes)

Scoring depends on a frozen per-task artifact produced by the task-construction pipeline. Scoring code must treat this as read-only and never see it leak to the agent.

```yaml
# tasks/<task_id>/task.yaml
task_id: cal-analyzer-0007
difficulty_tier: medium          # {easy, medium, hard} — used for normalization (§7.3)
prompt_user_facing: |            # the under-specified goal the agent receives
  Help me build a tool that analyzes my calendar and tells me where my time goes.
reference:
  repo: owner/name              # real OSS project grounding the task
  commit: <sha>
  why: "feature set + README define defensible acceptance criteria"
hidden_spec:                     # ORACLE-ONLY. Never shown to agent. Source of V2 + judge anchor.
  intent: "weekly time-allocation breakdown by event category from an ICS file"
  hard_constraints:             # MUST-haves; violation caps GA (§4.4)
    - id: HC1
      text: "ingests a standard .ics file"
      check: { type: functional, ref: checks/ingest_ics.py }
    - id: HC2
      text: "produces per-category time totals"
      check: { type: rubric_auto, ref: checks/category_totals.py }
  soft_prefs:                    # nice-to-haves; weighted rubric items
    - id: SP1
      text: "groups recurring events correctly"
      weight: 2
      check: { type: judge }
    - id: SP2
      text: "outputs both a table and a simple chart"
      weight: 1
      check: { type: functional, ref: checks/has_chart.py }
  hidden_preferences:           # used to reward correct clarification, not auto-graded as GA
    - "user actually wants ISO-week buckets, not calendar-month"
acceptance:
  install: { cmd: "pip install -e .", timeout_s: 600 }
  smoke:   { cmd: "calanalyze sample.ics", timeout_s: 120, expect_exit: 0 }
  provided_fixtures: [fixtures/sample.ics, fixtures/recurring.ics]
oracle_gold:                     # the simulated-user oracle's knowledge bundle
  acceptance_criteria_for_judge: <rendered checklist>
  reference_artifact_ptr: refs/cal-analyzer-0007/   # built reference for pairwise judging
weights:                         # per-task GA channel weights (defaults overridable, §4.3)
  v1: 0.40
  v2: 0.35
  v3: 0.25
```

**Invariant:** every `hard_constraint` and `soft_pref` has exactly one `check.type ∈ {functional, rubric_auto, judge}`. This routes each criterion to exactly one verification channel and makes the GA decomposition auditable.

---

## 2. The delivered artifact (what scoring receives from a run)

A run produces a **submission bundle**:

```
runs/<agent>/<task_id>/<seed>/
  workspace/            # full final repo the agent produced
  transcript.jsonl      # every agent<->oracle + agent<->env event (for interaction metrics)
  manifest.json         # entrypoints, declared run/test commands, env, language
  meta.json             # wall time, tokens, $ cost, tool calls, agent/version, seed
```

`manifest.json` is the agent's **declared** way to install/run/test its own artifact. We use it but never trust it blindly — see degenerate-strategy checks (§8).

---

## 3. Verification channel V1 — functional / sandbox execution

**Goal:** objective evidence the thing installs, runs, and satisfies auto-checkable acceptance criteria. This is the SWE-bench-style backbone, but scoped to "the app the agent chose to build" rather than a fixed test file.

### 3.1 Sandbox
- Every artifact runs in an ephemeral, **network-isolated** container (rootless Podman/Docker; on DAIC, a `--gres=gpu:0` CPU SLURM job or an Apptainer/Singularity image — DAIC compute nodes are network-restricted, which is *desirable* here for hermeticity).
- Pinned base images per language (`python:3.11-slim`, `node:20-slim`, …). Dependencies must resolve from a **pre-warmed local mirror** populated on the login node (which has internet) before the job. Network egress during the run is blocked.
- Resource caps: CPU 2 cores, 4 GB RAM, per-step timeouts from `acceptance.*.timeout_s`, global wall cap 30 min. Timeouts/OOM = that check fails, never a crash of the harness.

### 3.2 What V1 measures (three sub-signals, mirrors ScienceAgentBench)
| Sub-signal | Definition | Range |
|---|---|---|
| `install_ok` | install cmd exits 0 within timeout | {0,1} |
| `valid_exec` | smoke cmd exits 0 (the app *runs at all*) | {0,1} |
| `func_criteria` | fraction of `check.type==functional` criteria passing their scripted check | [0,1] |

Each functional `check.ref` is a **golden checker script** written at task-construction time. It is adversarial and artifact-agnostic: it drives the artifact through its declared entrypoint (CLI/HTTP/import) against the provided fixtures and asserts observable behavior — e.g. parse the printed table, hit `localhost:PORT/api`, import the module and call the function. Checkers assert *behavior*, not source strings, so they generalize across implementations.

When the artifact's interface is genuinely free (the prompt didn't pin a CLI name), the checker first reads `manifest.json` for the entrypoint, with a fallback discovery pass (look for `__main__`, `bin/`, a `Flask`/`FastAPI` app object). Discovery failures count as `valid_exec=0`, not checker error.

### 3.3 V1 score
```
V1 = 0.25*install_ok + 0.25*valid_exec + 0.50*func_criteria
```
`install_ok=0 ⇒ valid_exec=0 ⇒ func_criteria=0` by construction (can't run what won't install). V1 is in [0,1]. V1 is **deterministic** given the artifact (no LLM), so it carries the highest trust weight and is the anchor for catching fake-done (§8).

---

## 4. Verification channel V2 — rubric / acceptance-criteria checklist

**Goal:** a structured, item-by-item checklist derived from `hidden_spec`, so GA is decomposable and explainable rather than a single opaque number.

### 4.1 Construction (offline, at task-build time)
The rubric is **authored from `hidden_spec`, not invented per-run** — this is the key reliability lever from the rubric-generation literature (per-run rubric synthesis is a major variance source). Each `hard_constraint` and `soft_pref` becomes one rubric item:

```yaml
# rubric/<task_id>.yaml  (frozen, versioned)
items:
  - id: HC1
    text: "ingests a standard .ics file"
    kind: hard          # hard | soft
    weight: 1
    channel: functional # which verifier produces the pass/fail
  - id: SP1
    text: "groups recurring events correctly"
    kind: soft
    weight: 2
    channel: judge
```

`rubric_auto` items have a deterministic checker (like V1 but asserting a *criterion*, e.g. "category totals sum to total scheduled time"). `judge` items are scored in V3 but **live in the same rubric** so weighting is unified.

### 4.2 Item scoring
Each item gets `s_i ∈ {0, 0.5, 1}` (0=absent/wrong, 0.5=partial/buggy, 1=met). Functional and `rubric_auto` items return {0,1} only (no partial; scripts are binary). Judge items may return 0.5.

### 4.3 V2 score (soft items only; hard items handled as a gate in §4.4)
```
V2 = ( Σ_{i ∈ soft} w_i * s_i ) / ( Σ_{i ∈ soft} w_i )      ∈ [0,1]
```
Per-task channel weights `weights.{v1,v2,v3}` are set at task-build time and **frozen**; defaults (0.40/0.35/0.25) are used unless a task overrides (e.g. a pure-CLI task with no UI may push more weight to V1).

### 4.4 Hard constraints are a gate, not a weighted term
Hard constraints encode "if this is missing, the deliverable fundamentally isn't the requested tool." Define
```
hard_pass_frac = (Σ_{i∈hard} s_i) / |hard|
```
and apply a **GA cap** (§5.2). This prevents an agent from scoring 0.8 GA on a beautiful tool that *doesn't actually do the requested thing*. This is the analogue of SWE-bench's `FAIL_TO_PASS` gating, but expressed as a graceful cap rather than hard zero (open-ended tasks warrant partial credit).

---

## 5. Goal-achievement score (GA) = how V1+V2+V3 combine

### 5.1 Raw blend
```
GA_raw = weights.v1 * V1 + weights.v2 * V2 + weights.v3 * V3
```
(`weights` sum to 1; V3 defined in §6.)

### 5.2 Hard-constraint cap
```
GA = GA_raw * gate(hard_pass_frac)

gate(h) = 0.30 + 0.70 * h          # h=0 ⇒ GA ≤ 0.30*GA_raw ; h=1 ⇒ no penalty
```
Rationale: missing *all* hard constraints means the artifact is at best a 30%-credit near-miss regardless of polish; meeting all hard constraints removes the cap entirely. The 0.30 floor (vs 0) keeps the metric smooth and avoids the cliff that makes a benchmark brittle to a single mis-specified `HC`.

`GA ∈ [0,1]`. This is the per-(task, seed) goal-achievement value consumed by §9.

---

## 6. Verification channel V3 — LLM-as-judge (bias/variance controlled)

**Goal:** grade qualitative criteria that scripts can't (design sensibility, does the recurring-event grouping "feel right", code clarity, does the output actually answer the user's stated goal). This is the noisiest channel, so it gets the most safeguards and the lowest default weight (0.25).

### 6.1 What the judge sees
- The **user-facing prompt** (what was asked).
- The **rendered acceptance checklist** for `judge`-channel items (from `oracle_gold`, so the judge is **reference-anchored**, not free-floating).
- The **reference artifact** (or a distilled description of it) for pairwise mode.
- The candidate artifact: README + key source files + **captured V1 execution output** (stdout/screenshots of the running app). The judge grades *observed behavior + code*, not source code alone — this blocks "looks plausible but doesn't run" inflation, since V1 already told us if it runs.

The judge **never** sees: the agent's identity, token cost, the transcript, or how many oracle interactions happened (those belong to UC, not GA — keep channels independent).

### 6.2 Mode: pairwise-anchored, scored to a pointwise number
We use **pairwise comparison against the reference artifact** because pairwise judging agrees with humans better and is more calibrated than pointwise scoring (LLM-as-judge literature). But the leaderboard needs a number, so:

- For each `judge` rubric item, ask the judge to compare candidate vs reference and emit a per-item verdict `{worse, comparable, better}` + a justification, then map to `s_i ∈ {0, 0.5, 1}` (worse→0… but see anchoring below) **and** an overall pairwise preference.
- Reference is a strong-but-imperfect anchor (it's a real OSS slice). `comparable→1`, `better→1`, `clearly worse but functional→0.5`, `missing/broken→0`. We deliberately do **not** require beating the reference to get full marks — the reference is the *bar*, not the ceiling.

### 6.3 Bias controls (mandatory)
1. **Position-swap calibration.** Every pairwise judgment is run twice with candidate/reference order swapped. Keep the verdict only if **consistent**; on disagreement, mark the item `0.5` and log a `judge_position_conflict` flag. (Directly from the position-bias mitigation literature: swap-and-average.)
2. **Panel of judges (jury), not a single model.** Use `J=3` heterogeneous judge models (e.g. one Anthropic, one OpenAI, one strong open-weight served on DAIC vLLM). Per item, take the **median** `s_i`; for the overall preference, majority vote. Report inter-judge agreement (Krippendorff's α) per task.
3. **Verbosity/style defense.** Judge prompt explicitly instructs: ignore length, formatting flourish, and confident tone; grade only against the checklist + observed behavior. Candidate and reference are both truncated to equal budgets to neutralize length bias.
4. **Rubric decomposition.** No "give an overall 1–10." The judge only ever scores **named rubric items** with required evidence quotes — decomposed rubric judging is markedly less biased and more reproducible than holistic scoring.
5. **Self-preference guard.** A judge model never judges artifacts produced by the same model family as the agent under test (drop that judge from the panel for that run; backfill from a reserve judge).
6. **Abstention.** Judges may return `INSUFFICIENT_EVIDENCE` for an item; that item falls back to its `rubric_auto` proxy if one exists, else is dropped from V2's denominator (not silently scored 0).

### 6.4 V3 score
```
V3 = ( Σ_{i ∈ judge} w_i * median_j(s_{i,j}) ) / ( Σ_{i ∈ judge} w_i )      ∈ [0,1]
```
V3 inherits the same rubric weights as V2; the V2/V3 split is purely *which verifier produced the item score*. (Implementation note: §4.3's V2 sum and §6.4's V3 sum together cover all soft items partitioned by channel; the GA weights `v2`/`v3` then re-weight the two channels' aggregate quality, which is intentional — it lets us down-weight the noisy judge channel relative to deterministic rubric checks.)

### 6.5 Judge variance budget
Run the **full panel** once per (task, seed) artifact. Judge stochasticity is folded into the run-level reruns (§7), not separately resampled, to keep cost bounded. Track judge α over the whole benchmark; if α < 0.4 on a task, flag the task for human re-annotation in the dev set (it's a bad rubric, not a bad agent).

---

## 7. Reproducibility & statistics

### 7.1 Sources of stochasticity (and how each is pinned)
| Source | Mitigation |
|---|---|
| Agent sampling | `n` independent reruns per task with fixed seeds where the model API supports it; temperature recorded; for non-seedable APIs, `n` reruns still estimate the distribution. |
| Oracle (simulated user) | Oracle runs at **low temperature (≤0.3)**, fixed system prompt, fixed gold bundle, seeded where possible. Oracle model + version pinned per benchmark release. |
| Judge panel | Fixed judge models+versions+prompts per release; temperature ≤0.2; position-swap + median over panel (§6.3). |
| Sandbox | Pinned base images, pinned dep mirror snapshot, network off, fixed resource caps → deterministic V1. |
| Task set | Frozen `task.yaml` + `rubric.yaml` hashes recorded in every result row. |

Everything model/prompt/image/data is pinned by content hash in `release.lock`. A result is only comparable within the same `release.lock`.

### 7.2 Reruns, central estimate, and reliability
Default `n = 5` reruns per (agent, task) for the test split (`n = 3` for dev to save cost).

- **Central estimate:** report **mean GA** and **mean Usability Score** with a **bootstrap 95% CI** (10k resamples over the n×|tasks| result matrix, resampling tasks then seeds — cluster bootstrap to respect within-task correlation).
- **Reliability:** report **`pass^k`** (τ-bench), the probability that the agent succeeds on **all** k independent attempts, estimated unbiasedly from the n reruns:
  ```
  pass^k_task = C(c, k) / C(n, k)        # c = # of "successful" reruns of this task, k ≤ n
  pass^k      = mean over tasks of pass^k_task
  ```
  "Success" for `pass^k` is `GA ≥ τ_GA` (default `τ_GA = 0.8`) **and** `hard_pass_frac = 1`. We report `pass^1` (=mean success rate) and `pass^k` at `k = n`. Reporting both exposes the reliability gap (an agent can have high `pass^1` but low `pass^n` if it succeeds catastrophically-or-not rather than consistently — exactly the distinction the metrics expert wants for "usable").
- We do **not** headline `pass@k` (any-of-k success). It rewards lucky retries, which is the opposite of usability. `pass@k` may appear as a diagnostic only.

### 7.3 Per-task difficulty normalization
Raw GA across tiers isn't comparable (a `hard` task at GA 0.6 ≠ an `easy` task at GA 0.6). Two complementary normalizations, both reported:

1. **Tier-stratified reporting:** always break leaderboard numbers down by `difficulty_tier`. The headline number is the **macro-average over tiers** (mean of per-tier means), so the test set's tier mix can't be gamed.
2. **Reference-relative GA (`GA_norm`):** divide by the **reference artifact's own GA** as scored by the identical pipeline (the reference is a real OSS slice and is *not* perfect on our rubric — typically 0.7–0.95). `GA_norm = GA / GA_ref`, clipped to [0, 1.2]. This expresses "how close to / beyond the human reference," and absorbs rubric-hardness differences between tasks. The leaderboard headlines `GA` (absolute, interpretable) and lists `GA_norm` (difficulty-adjusted) alongside.

### 7.4 Statistical comparison of two agents
To claim agent A > agent B on the leaderboard, require a **paired cluster-bootstrap** test over shared tasks (paired by task, clustered by task to handle the n seeds) with the **Usability Score** as the statistic; report the bootstrap p-value and effect size. The leaderboard shows CIs so overlapping agents are visibly tied rather than spuriously ranked.

---

## 8. Detecting & penalizing degenerate strategies

The benchmark's whole point is *appropriate* interaction, so we must actively penalize three degenerate strategies. Detection feeds both the interaction-metrics module (UC) and a GA-side validity gate.

### 8.1 Fake-done (claims success, artifact doesn't deliver) — GA-side
- Primary defense is structural: GA is grounded in **V1 execution** and **scripted checkers**, so a claimed-but-non-running artifact gets `valid_exec=0 ⇒ low V1` and fails functional `HC`s ⇒ GA capped by §5.2. Confident prose can't move V1.
- **Cross-check flag:** if the agent's final message asserts completion (classifier over `transcript.jsonl`) but `hard_pass_frac < 1`, set `fake_done=1`. This is reported as an integrity stat and, per `⚠ RECONCILE`, feeds a penalty in the composite (§9).

### 8.2 Never-ask (ships on the under-specified prompt without resolving ambiguity) — UC-side, but GA-visible
- Tasks are deliberately ambiguous (`hidden_preferences`, e.g. ISO-week vs month buckets). An agent that never clarifies will, with high probability, guess wrong on at least one hidden preference. We make that *cost GA*: each `hidden_preference` has a `rubric_auto`/judge check; getting it wrong without ever having asked the relevant clarifying question yields the normal GA loss **plus** the never-ask classification in UC.
- Detection: `n_oracle_questions == 0` on a task whose `hidden_preferences` are non-empty ⇒ `never_ask=1`.

### 8.3 Over-ask (offloads the work onto the oracle) — UC-side
- The oracle is instructed to give **graded, bounded help** (answer clarifications; refuse to design/implement). But an agent can still spam low-value questions. Detection combines:
  - `n_interactions` z-score vs the per-task interaction distribution (over-ask = top decile), and
  - an **information-gain filter:** the oracle (or a cheap classifier) labels each agent question as `clarifying / hint-seeking / offloading / redundant`. High `offloading + redundant` fraction ⇒ `over_ask=1`.
- This is exactly the assistance-severity classification the metrics expert owns; scoring's job is to surface the labels and ensure GA isn't inflated by oracle-supplied design. To enforce the latter: the judge (§6) grades the *artifact*, and a separate **attribution check** flags rubric items whose satisfaction is verbatim traceable to oracle messages (`oracle_attributed_credit`), which the composite discounts (`⚠ RECONCILE`).

### 8.4 Summary of integrity flags emitted per (task, seed)
`fake_done`, `never_ask`, `over_ask`, `judge_position_conflict`, `discovery_failed`, `oracle_attributed_credit_frac`. All are logged; the first three carry score consequences via §9.

---

## 9. Composite Usability Score (leaderboard number)

`⚠ RECONCILE` — **this formula is the contract with the metrics expert. `bench/spec/usability_score.yaml` is the single source of truth; the numbers below are the agreed defaults. Do not fork them.**

### 9.1 Interaction/assistance cost (UC) — restated for consistency
The metrics expert defines a per-(task, seed) **assistance cost** `AC ∈ [0,1]`, where 0 = fully autonomous and correct-by-itself, 1 = required maximal human help. Its canonical form (restated here so scoring computes the *same* thing):
```
AC = clip01(  α * norm(n_interactions)
            + β * severity_weighted_help
            + γ * never_ask
            + δ * over_ask ),
with default (α,β,γ,δ) = (0.35, 0.40, 0.15, 0.10)
```
where `severity_weighted_help` sums per-interaction severities (clarification ≈ 0.1, hint ≈ 0.4, partial-handoff ≈ 0.7, full-handoff/oracle-implemented ≈ 1.0), normalized by a per-task cap. `norm(·)` is per-task min-max from the dev-set interaction distribution. (These severity tiers and weights are owned by the metrics module; scoring imports them.)

### 9.2 Per-(task, seed) Usability Score
```
U = GA * (1 - λ * AC) * (1 - fake_done_penalty)

defaults:  λ = 0.5,   fake_done_penalty = 0.25 if fake_done else 0
```
Interpretation:
- **Multiplicative** GA × (assistance discount): you only get usability credit for goal achievement you reached *without* leaning on the human. An agent that nails GA=1.0 but with AC=1.0 lands at `U = 1.0*(1-0.5) = 0.5` — half credit, because a human did half the cognitive work. An agent with GA=0.9, AC=0.1 lands at `U = 0.9*0.95 = 0.855`. This makes "did it well *and* by itself" dominate.
- `λ=0.5` caps the assistance discount at 50% so that interaction is *penalized but not forbidden* — appropriate asking is allowed; the benchmark rewards getting more GA per unit of help.
- `fake_done` applies a flat additional 25% haircut on top, because falsely claiming completion is a trust-critical failure distinct from just doing poorly.

`U ∈ [0,1]`.

### 9.3 Aggregation to the leaderboard
```
U_task   = mean_seed( U )                       # average over n reruns
U_tier   = mean_{task in tier}( U_task )
Usability Score (headline) = mean_tier( U_tier )   # macro-avg over difficulty tiers (§7.3)
```
Report with cluster-bootstrap 95% CI (§7.4).

### 9.4 Why GA and Usability Score are both reported
GA answers "can it build the thing?"; Usability Score answers "can it build the thing *usably*?". A model can top GA and rank lower on Usability if it only got there via heavy oracle help. Both columns ship.

---

## 10. Evaluation protocol (end-to-end)

### 10.1 Splits
| Split | Tasks | Visibility | Purpose |
|---|---|---|---|
| `dev` | ~30, all tiers | **fully public** (prompts, rubrics, checkers, reference) | agent developers iterate; calibrate `norm(·)`, severity caps, judge prompts; tune nothing benchmark-side after freeze |
| `test-public` | ~80 | prompts public, **`hidden_spec`/rubric/checkers/oracle-gold held out** (server-side) | the leaderboard split everyone runs |
| `test-heldout` | ~40 | **entirely private**, rotated each release | anti-overfitting; only run by maintainers; numbers published, tasks not |

Held-out evaluation runs on DAIC behind the harness; submitters send the **submission bundle** (§2) or a containerized agent, never the gold. Rubrics/checkers/oracle prompts for `test-*` live only in the private eval repo.

### 10.2 How a new agent gets evaluated (pipeline)
```
1. Wrap agent to the standard I/O protocol:
     - receives prompt_user_facing
     - may call oracle.ask(question) [logged]  and  env tools (run shell, read/write files)
     - emits final submission bundle
2. For each task in split, for seed in 1..n:
     a. Spin oracle (pinned model+gold).  Run agent episode under wall/turn caps.
     b. Persist workspace/, transcript.jsonl, manifest.json, meta.json.
3. Verification (per task,seed):
     V1: build sandbox image, install, smoke, run functional checkers  -> V1, hard func items
     V2: run rubric_auto checkers                                       -> V2 items
     V3: render judge inputs (+ captured exec output), run J-judge panel
         with position-swap; aggregate                                  -> V3 items
     compute hard_pass_frac, gate, GA                                   (§5)
4. Interaction metrics module consumes transcript.jsonl -> AC, flags    (§8,§9.1)
5. Compose U per (task,seed)                                            (§9.2)
6. Aggregate: U_task, U_tier, Usability Score; GA, GA_norm; pass^1, pass^n;
   bootstrap CIs; integrity-flag rates; judge α; cost.                  (§7,§9.3)
7. Write results/<agent>/<release>/results.parquet + report card.
```
The whole of step 3–6 is **deterministic given the artifacts and `release.lock`** except for judge LLM calls, which are bounded by §6.5 and folded into the reruns.

### 10.3 Reported leaderboard columns (exact set)
| Column | Definition | Primary? |
|---|---|---|
| `Usability Score` | §9.3 headline, macro-avg over tiers, [0,1] | **★ headline** |
| `Usability Score 95% CI` | cluster-bootstrap interval | ★ |
| `GA` | mean goal-achievement, macro-avg over tiers | ★ |
| `GA_norm` | reference-relative GA (§7.3) | yes |
| `pass^1` | mean per-attempt success (GA≥0.8 & all HCs) | ★ |
| `pass^n` | all-of-n reliability (τ-bench `pass^k`, k=n) | ★ |
| `AC` | mean assistance cost [0,1] (lower better) | ★ |
| `n_interactions / task` | mean oracle interactions | yes |
| `help_severity` | mean severity-weighted help | yes |
| `GA by tier` | GA on {easy, medium, hard} | yes |
| `V1 / V2 / V3` | mean per-channel scores (diagnostic) | diagnostic |
| `fake_done %` | rate of completion-claim-but-failed | ★ integrity |
| `never_ask %`, `over_ask %` | degenerate-interaction rates | yes integrity |
| `judge α` | mean inter-judge agreement | diagnostic |
| `$ cost / task`, `tokens / task`, `wall / task` | efficiency (from meta.json) | yes |
| `n_seeds`, `release.lock hash` | reproducibility provenance | required |

Sorting default: `Usability Score` desc, ties broken by lower `AC`. The board renders CIs so statistically-tied agents are grouped, not falsely ranked (§7.4).

### 10.4 Anti-gaming summary
- Held-out split + rotated private tasks → can't overfit rubrics.
- GA grounded in deterministic execution → can't talk its way to GA.
- Multiplicative AC discount + over-ask/attribution flags → can't offload to oracle.
- `pass^n` headline → can't farm lucky retries.
- Macro-tier average + reference-relative norm → can't game the tier mix or cherry-pick easy tasks.
- Self-preference guard + position-swap + jury → judge can't be a single bias.

---

## 11. Open parameters to confirm with metrics expert (`⚠ RECONCILE` list)
1. `λ` (assistance discount cap, default 0.5) and `fake_done_penalty` (0.25) in §9.2.
2. The `U = GA*(1-λ*AC)` **multiplicative** form vs an additive alternative `U = w_g*GA + w_u*(1-AC)`. Multiplicative is recommended (coupling "right thing" with "by itself"); must match `usability_score.yaml`.
3. `AC` term weights `(α,β,γ,δ)` and the severity tier values (§9.1) — owned by metrics, imported by scoring.
4. Success thresholds `τ_GA=0.8` and `pass^k` success definition (§7.2).
5. Default GA channel weights `(0.40, 0.35, 0.25)` and `gate(h)=0.3+0.7h` shape (§4.3, §5.2).

---

## 12. Module layout (for the repo)
```
bench/
  spec/usability_score.yaml        # SINGLE SOURCE OF TRUTH for §9 constants (shared w/ metrics)
  scoring/
    v1_functional.py               # sandbox build/run, checker harness
    v2_rubric.py                   # rubric_auto checkers + V2 aggregation
    v3_judge.py                    # judge panel, position-swap, jury median
    ga.py                          # gate + GA blend (§5)
    compose.py                     # imports metrics.AC -> U (§9)
    stats.py                       # bootstrap CI, pass^k, normalization (§7)
    integrity.py                   # fake_done / never_ask / over_ask / attribution (§8)
    leaderboard.py                 # column assembly (§10.3)
  sandbox/                         # Dockerfiles / Apptainer defs, dep-mirror tooling
  tests/                           # golden tests: known-good & known-bad artifacts per channel
```
Every checker and the V1 pipeline ship with **fixture artifacts** (a known-good, a known-broken, a fake-done, an over-ask transcript) so the scorer itself is regression-tested.

---

### Sources / prior art
- τ-bench (simulated-user oracle, `pass^k` reliability): https://arxiv.org/abs/2406.12045
- Reliability/`pass^k` framing & reliability gap: https://arxiv.org/pdf/2603.29231 , https://arxiv.org/pdf/2601.06112
- LLM-as-judge bias (position/verbosity, swap-calibration, pairwise vs pointwise): https://www.evidentlyai.com/llm-guide/llm-as-a-judge , https://mbrenndoerfer.com/writing/position-bias-in-llm-judges
- Rubric-based judging for open-ended tasks (decompose into criteria, author rubrics offline): https://arxiv.org/html/2602.05125 , https://arxiv.org/pdf/2601.08430
- Multi-channel rubric (success / valid-exec / similarity / cost) — ScienceAgentBench (via survey): https://arxiv.org/pdf/2507.21504
- LLM-simulated-user reliability caveat (why oracle is pinned & low-temp): https://arxiv.org/pdf/2601.17087
```

Below is the complete scoring & evaluation specification (return value). The project directory `/Users/davidhuang/Desktop/usabilityBenchMark` is currently empty; this document is meant to become `/Users/davidhuang/Desktop/usabilityBenchMark/docs/scoring.md`.

---

# Usability-Benchmark — Scoring & Evaluation Specification

**Component:** `scoring/` (verification, rubric, judge, aggregation, statistics)
**Status:** v0.1 — buildable spec. Sections marked `⚠ RECONCILE` must stay numerically identical to the metrics expert's `metrics/` module. Single source of truth: `bench/spec/usability_score.yaml`.

## 0. Scope & philosophy
This benchmark does not ask "did the agent pass a fixed hidden test suite." It asks two coupled questions: (1) **Goal achievement (GA)** — how good is the delivered artifact vs the task's `hidden_spec` and a real OSS reference; (2) **Usability cost (UC/AC)** — how much human assistance the agent consumed. The final score rewards *building the right thing with minimal babysitting*: a perfect artifact spoon-fed by the oracle is not usable, and a never-ask agent that confidently ships the wrong thing is not usable either — both must cost score.

GA over open-ended deliverables is graded by a three-channel verifier: **V1** functional/sandbox execution (deterministic, highest trust), **V2** rubric/acceptance-checklist derived from `hidden_spec` (semi-objective), **V3** LLM-as-judge for qualitative criteria (noisiest, most safeguarded, lowest weight). Named prior art borrowed: τ-bench (simulated-user oracle + `pass^k`), SWE-bench (sandboxed golden checks as a *component*), ScienceAgentBench (multi-channel rubric), and the LLM-as-judge bias literature (position-swap calibration, jury, rubric decomposition).

## 1. Task data contract (read-only, never leaked to agent)
Per task: `task_id`, `difficulty_tier ∈ {easy,medium,hard}`, `prompt_user_facing` (under-specified goal), `reference{repo,commit}`, and the oracle-only `hidden_spec` containing `hard_constraints` (MUST-haves), `soft_prefs` (weighted nice-to-haves), and `hidden_preferences` (ambiguities that reward correct clarification). Each constraint/pref carries exactly one `check.type ∈ {functional, rubric_auto, judge}` routing it to exactly one channel. Plus `acceptance{install,smoke,fixtures}`, `oracle_gold{checklist, reference_artifact_ptr}`, and per-task channel `weights{v1:0.40, v2:0.35, v3:0.25}` (frozen, overridable).

## 2. Submission bundle (per run)
`workspace/` (final repo), `transcript.jsonl` (all agent↔oracle + agent↔env events → interaction metrics), `manifest.json` (declared install/run/test + entrypoints), `meta.json` (wall, tokens, $, tool calls, seed). `manifest` is used but never trusted blindly (§8).

## 3. V1 — functional / sandbox execution
**Sandbox:** ephemeral, **network-isolated** container (rootless Podman; on DAIC an Apptainer image in a CPU SLURM job — compute-node network restriction is *desirable* for hermeticity). Deps resolve from a **pre-warmed local mirror** populated on the internet-capable login node; egress blocked during the run. Caps: 2 CPU, 4 GB, per-step timeouts, 30 min wall. Timeout/OOM ⇒ that check fails, never a harness crash.
**Sub-signals** (mirrors ScienceAgentBench): `install_ok∈{0,1}`, `valid_exec∈{0,1}` (app runs at all), `func_criteria∈[0,1]` (fraction of `functional` checks passing). Golden checker scripts are **behavior-asserting, artifact-agnostic** — they drive the declared entrypoint (CLI/HTTP/import) against fixtures and assert observable output, so they generalize across implementations. Free interfaces: read `manifest` entrypoint, then fallback discovery (`__main__`, `bin/`, Flask/FastAPI app object); discovery failure ⇒ `valid_exec=0`.
**Score:** `V1 = 0.25*install_ok + 0.25*valid_exec + 0.50*func_criteria`, with cascade `install_ok=0 ⇒ valid_exec=0 ⇒ func_criteria=0`. Deterministic (no LLM) → highest trust; anchors fake-done detection.

## 4. V2 — rubric / acceptance-criteria checklist
Rubric is **authored offline from `hidden_spec`, frozen+versioned** — not synthesized per-run (per-run synthesis is the dominant rubric-variance source). Each constraint/pref → one item `{id,text,kind∈{hard,soft},weight,channel}`. Item scores `s_i∈{0,0.5,1}` (functional/`rubric_auto` are binary {0,1}; `judge` may be 0.5).
**Soft items:** `V2 = Σ_{soft} w_i·s_i / Σ_{soft} w_i ∈ [0,1]`.
**Hard constraints are a gate, not a weighted term** (analogue of SWE-bench `FAIL_TO_PASS`, but graceful): `hard_pass_frac = Σ_{hard} s_i / |hard|`.

## 5. Goal-achievement (GA) = how V1+V2+V3 combine
```
GA_raw = w_v1·V1 + w_v2·V2 + w_v3·V3            (weights sum to 1)
GA     = GA_raw · gate(hard_pass_frac)
gate(h)= 0.30 + 0.70·h                          # h=0 ⇒ ≤30% credit; h=1 ⇒ no penalty
GA ∈ [0,1]
```
The 0.30 floor (not 0) keeps the metric smooth and robust to one mis-specified hard constraint.

## 6. V3 — LLM-as-judge (bias/variance controlled)
**Judge sees:** user prompt, rendered acceptance checklist for `judge` items (reference-anchored, not free-floating), the reference artifact (for pairwise), and the candidate's README + key source + **captured V1 execution output**. It grades observed behavior + code, not source alone. It **never** sees agent identity, cost, transcript, or interaction counts (keeps GA independent of UC).
**Mode:** pairwise-against-reference (better human agreement + calibration than pointwise) mapped to per-item `s_i∈{0,0.5,1}`: `comparable/better→1`, `clearly-worse-but-functional→0.5`, `missing/broken→0`. Beating the reference is *not* required for full marks (reference = bar, not ceiling).
**Mandatory bias controls:** (1) **position-swap calibration** — run each pairwise judgment twice with order swapped, keep only if consistent, else `s_i=0.5` + `judge_position_conflict` flag; (2) **jury of J=3 heterogeneous judges** (Anthropic + OpenAI + open-weight vLLM on DAIC), per-item **median**, overall majority vote, report Krippendorff's α; (3) **verbosity/style defense** — instruct judge to ignore length/tone, truncate both sides to equal budget; (4) **rubric decomposition** — score only named items with evidence quotes, never a holistic 1–10; (5) **self-preference guard** — a judge never grades its own model family (backfill from reserve); (6) **abstention** — `INSUFFICIENT_EVIDENCE` falls back to a `rubric_auto` proxy or drops from the denominator (not silently 0).
**Score:** `V3 = Σ_{judge} w_i·median_j(s_{i,j}) / Σ_{judge} w_i ∈ [0,1]`. Panel runs once per (task,seed); judge stochasticity folds into run-level reruns (§7) to bound cost. Task with α<0.4 is flagged for human re-annotation (bad rubric, not bad agent).

## 7. Reproducibility & statistics
**Pinning:** agent reruns with fixed seeds where the API supports it; oracle at temp ≤0.3 with fixed gold+prompt; judges at temp ≤0.2, fixed models/prompts; sandbox images + dep-mirror snapshot pinned. All model/prompt/image/data hashes in `release.lock`; results comparable only within the same lock.
**Reruns:** default `n=5` test / `n=3` dev per (agent,task). Report **mean GA / mean Usability Score with cluster-bootstrap 95% CI** (resample tasks then seeds — respects within-task correlation).
**Reliability — `pass^k` (τ-bench):** probability of succeeding on **all** k attempts, `pass^k_task = C(c,k)/C(n,k)` with `c` = successful reruns; success = `GA≥τ_GA (0.8) AND hard_pass_frac=1`. Headline `pass^1` and `pass^n`; the gap between them exposes consistent-vs-catastrophic agents (the reliability distinction central to "usable"). `pass@k` (any-of-k) is **diagnostic only** — it rewards lucky retries, the opposite of usability.
**Difficulty normalization (both reported):** (1) **tier-stratified** — headline is the **macro-average over tiers** so the tier mix can't be gamed; (2) **reference-relative** `GA_norm = clip(GA / GA_ref, 0, 1.2)`, dividing by the real reference's own GA under the identical pipeline (typically 0.7–0.95), absorbing per-task rubric hardness.
**Comparing agents:** paired cluster-bootstrap over shared tasks on the Usability Score; report p-value + effect size; board shows CIs so tied agents are grouped, not spuriously ranked.

## 8. Degenerate-strategy detection (penalized)
- **Fake-done:** structurally defeated because GA is grounded in V1 execution + scripted checkers (confident prose can't raise V1). Cross-check: completion-claim classifier over transcript AND `hard_pass_frac<1` ⇒ `fake_done=1` → §9 penalty.
- **Never-ask:** ambiguous `hidden_preferences` (e.g. ISO-week vs month buckets) mean a non-asking agent likely guesses wrong → normal GA loss; `n_oracle_questions==0` with non-empty `hidden_preferences` ⇒ `never_ask=1`.
- **Over-ask:** oracle gives graded bounded help (answers clarifications; refuses to design/implement). Detection = interaction-count top-decile z-score **and** an info-gain filter labeling each question `clarifying/hint/offloading/redundant`; high offloading+redundant ⇒ `over_ask=1`. An **attribution check** flags rubric items whose satisfaction is verbatim traceable to oracle messages (`oracle_attributed_credit_frac`), which the composite discounts (`⚠ RECONCILE`).
- Flags emitted per (task,seed): `fake_done, never_ask, over_ask, judge_position_conflict, discovery_failed, oracle_attributed_credit_frac`.

## 9. Composite Usability Score `⚠ RECONCILE` (contract with metrics expert; `usability_score.yaml` is authoritative)
**Assistance cost (restated for consistency):**
```
AC = clip01( α·norm(n_interactions) + β·severity_weighted_help + γ·never_ask + δ·over_ask )
(α,β,γ,δ)=(0.35,0.40,0.15,0.10);  severity: clarification≈0.1, hint≈0.4, partial-handoff≈0.7, full-handoff≈1.0
```
**Per (task,seed):**
```
U = GA · (1 − λ·AC) · (1 − fake_done_penalty)        λ=0.5,  fake_done_penalty=0.25 if fake_done else 0
```
Multiplicative by design: usability credit only for GA reached *without* leaning on the human. GA=1.0 & AC=1.0 → U=0.5 (human did half the work); GA=0.9 & AC=0.1 → U=0.855. `λ=0.5` caps the discount at 50% so appropriate asking is penalized-but-not-forbidden. `fake_done` adds a flat 25% haircut (trust-critical failure).
**Aggregation:** `U_task = mean_seed(U)` → `U_tier = mean_task(U_task)` → **Usability Score = mean_tier(U_tier)** (macro-avg over tiers), with cluster-bootstrap 95% CI. Both `Usability Score` and raw `GA` ship: GA = "can it build it"; Usability = "can it build it usably."

## 10. Protocol
**Splits:** `dev` (~30, fully public incl. rubrics/checkers — for agent-dev iteration + calibrating `norm`/severity/judge prompts), `test-public` (~80, prompts public but `hidden_spec`/rubric/checkers/oracle-gold held server-side), `test-heldout` (~40, fully private, rotated per release for anti-overfitting). Submitters send the bundle/containerized agent; gold lives only in the private eval repo.
**Pipeline:** wrap agent to standard I/O (receives prompt; may call `oracle.ask` [logged] + env tools; emits bundle) → per (task,seed) run oracle+agent under caps, persist artifacts → V1 (sandbox/install/smoke/functional) → V2 (`rubric_auto`) → V3 (jury+swap) → gate→GA → metrics module computes AC+flags from transcript → compose U → aggregate (U_tier, Usability Score, GA, GA_norm, pass^1, pass^n, CIs, flag rates, α, cost). Steps 3–6 deterministic given artifacts + `release.lock` except judge calls (bounded, folded into reruns).
**Leaderboard columns (exact):** `Usability Score` (★ headline) + `95% CI`; `GA` (★); `GA_norm`; `pass^1` (★); `pass^n` (★, τ-bench reliability); `AC` (★, lower better); `n_interactions/task`; `help_severity`; `GA by tier`; `V1/V2/V3` (diagnostic); `fake_done %` (★ integrity); `never_ask % / over_ask %`; `judge α` (diagnostic); `$ cost / tokens / wall per task`; `n_seeds`; `release.lock hash`. Default sort: Usability Score desc, ties → lower AC; CIs rendered so statistically-tied agents group.
**Anti-gaming summary:** held-out + rotated tasks (no rubric overfit); execution-grounded GA (no talking your way up); multiplicative AC + over-ask/attribution flags (no oracle offloading); `pass^n` headline (no lucky-retry farming); macro-tier avg + reference-relative norm (no tier-mix gaming); self-preference guard + swap + jury (no single-judge bias).

## 11. `⚠ RECONCILE` parameters to confirm with metrics expert
`λ`(0.5) and `fake_done_penalty`(0.25); multiplicative `U=GA·(1−λ·AC)` vs additive `w_g·GA+w_u·(1−AC)` (multiplicative recommended); `AC` weights `(α,β,γ,δ)` + severity tiers; `τ_GA=0.8` and `pass^k` success def; GA channel weights `(0.40,0.35,0.25)` and `gate(h)=0.3+0.7h`.

## 12. Repo layout
`bench/spec/usability_score.yaml` (single source of truth for §9 constants, shared with metrics); `bench/scoring/{v1_functional,v2_rubric,v3_judge,ga,compose,stats,integrity,leaderboard}.py`; `bench/sandbox/` (Dockerfiles/Apptainer + dep-mirror tooling); `bench/tests/` (golden known-good / known-broken / fake-done / over-ask fixtures so the scorer is itself regression-tested).

**Prior art:** τ-bench (oracle + pass^k) https://arxiv.org/abs/2406.12045 ; reliability/pass^k gap https://arxiv.org/pdf/2603.29231 ; LLM-judge bias + swap-calibration + pairwise-vs-pointwise https://www.evidentlyai.com/llm-guide/llm-as-a-judge , https://mbrenndoerfer.com/writing/position-bias-in-llm-judges ; offline rubric decomposition https://arxiv.org/html/2602.05125 ; multi-channel rubric (ScienceAgentBench, via survey) https://arxiv.org/pdf/2507.21504 ; LLM-simulated-user reliability caveat (why oracle is pinned/low-temp) https://arxiv.org/pdf/2601.17087