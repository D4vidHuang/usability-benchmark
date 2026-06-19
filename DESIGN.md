# DESIGN.md — `usability-benchmark`

**A benchmark for the *usability* of AI coding agents on open-ended development goals, measured by how much human intervention they require.**

Repo: `github.com/D4vidHuang/usability-benchmark` · Package: `usabench` · Status: design-frozen v0.1 · Lead-architect synthesis of six workstream specs.

> Per-area design docs live in `docs/`: `docs/related-work.md` (positioning), `docs/metrics.md` (metric suite), `docs/tasks.md` (dataset & collection), `docs/protocol.md` (oracle & harness runtime), `docs/scoring.md` (verification & aggregation), `docs/infra.md` (engineering & DAIC). This file is the north star that ties them together; where a per-area doc disagrees with this file, **this file wins** and the disagreement is recorded in §9 / the `inconsistencies` register.

---

## 1. Vision & one-paragraph pitch

Today's coding-agent benchmarks measure *terminal correctness on a closed task*: SWE-bench gives a precise issue and asks "does the diff pass the hidden tests?" That paradigm deliberately removes ambiguity and never measures the thing practitioners actually care about — **how much of my time, attention, and clarification did the agent consume to get there.** `usability-benchmark` inverts the dependent variable. It gives an agent an *open-ended, under-specified, lay-phrased software-development goal* ("build me a tool that analyzes my calendar and tells me where my time goes"), grounds each task in a real open-source project so a defensible gold intent exists, and places an **LLM simulated-user oracle** in the loop that holds the gold knowledge. The agent may ask the oracle, request hints, or hand off — and the harness **counts and severity-grades every interaction**. The headline number rewards *building the right thing with minimal human babysitting*: an agent that succeeds only by extracting constant takeovers scores as poorly as one that never asks and confidently ships the wrong thing.

## 2. Positioning (what is new)

Every close neighbor owns exactly one of three properties; we are the intersection of all three.

- **Open-ended, real-OSS-grounded dev task** — owned by DevAI/Agent-as-a-Judge, ProjDevBench, SWE-Lancer; but they run the agent *autonomously* and score *correctness/value*, never human effort.
- **Simulated-user oracle holding gold intent** — owned by τ-bench / τ²-bench; but the domain is bounded customer-service transactions scored by DB-state, and the user is a *spec to satisfy*, not a *resource whose effort is metered*.
- **Intervention amount + severity as the primary score** — owned by CentaurEval / ProSoftArena / Anthropic's autonomy telemetry; but those depend on *real humans* (not scalable/reproducible) or *production telemetry* (not a controlled benchmark).

We borrow concrete, citable mechanisms: the **simulated-user oracle** and **pass^k reliability** (τ-bench; pass^k unbiased estimator $\binom{c}{k}/\binom{n}{k}$), **interaction-as-measured-budget** (MINT), **anti-spam precision/recall of asking** and **progressive hidden-blocker discovery** (HiL-Bench Ask-F1), **oracle-answered clarification on under-specified tasks** (Ambig-SWE, ClarEval), **hierarchical-requirement agentic grading** (DevAI), **state-based behavior verification allowing many solution paths** (AppWorld, WebArena), **multi-channel rubric** (ScienceAgentBench: success / valid-exec / similarity / cost), **graded human-intervention levels for ablation** (CentaurEval), and the **stop/interrupt taxonomy** (Anthropic). The "LLM-simulated users are imperfect proxies" caveat ("Lost in Simulation") is the reason our oracle is tightly scoped, low-temperature, severity-capped, and human-validated on a calibration subset.

**Positioning statement.** *`usability-benchmark` is the first benchmark to evaluate AI coding agents on open-ended, lay-phrased software-development goals while making human-intervention amount and severity the primary score, using an LLM simulated-user oracle that holds the reference project's gold intent, constraints, and hidden acceptance criteria.*

## 3. Methodology overview — the six areas as one loop

The benchmark is a closed pipeline; each workstream owns one stage and a typed interface to the next.

```
 [COLLECT]            [TASKS]                [RUNTIME: harness+oracle]      [SCORING]            [METRICS]            [REPORT]
 GitHub REST/GraphQL  task.json (+hidden     run_episode(task,agent,        V1 exec / V2 rubric  registry A–G +       leaderboard
 → raw_harvest.jsonl  spec, ambiguity_pts,   oracle,seed) → trace.jsonl     / V3 judge → GA      composite Usability  + trace viewer
 → candidates.jsonl   acceptance_criteria,   (canonical, append-only,       (Goal Achievement);  Score (geom. mean);  + variance
                      verification)          severity-graded interactions)  integrity flags      pass^k over N seeds   report cards
```

1. **Related work / positioning** (`docs/related-work.md`) fixes *what is novel* and *which mechanisms we borrow with attribution*. It is the scientific justification, not runtime code.
2. **Tasks & data collection** (`docs/tasks.md`, `usabench.collect`, `tasks/`) manufactures tasks at scale from public GitHub: harvest real repos (READMEs, feature-request issues, awesome-lists), draft a lay-phrased `user_goal`, author the **gold** (reference implementation, fixtures, `hidden_spec`, weighted `acceptance_criteria`, `ambiguity_points`), and **calibrate** that every shipped task empirically *requires and rewards* assistance (discriminative check). The agent-visible view never leaks gold.
3. **Oracle & harness runtime** (`docs/protocol.md`, `usabench.harness` + `usabench.oracle`) runs one `(task, agent, seed)` episode, mediating **all** agent↔oracle traffic through a single `InteractionBus`, sandboxing the agent's file/exec actions, enforcing budgets, and emitting the **canonical `trace.jsonl`** — the single artifact everything downstream consumes.
4. **Scoring** (`docs/scoring.md`, `usabench.eval` scoring channels) turns the agent's delivered artifact into **Goal Achievement (GA)** via three verifier channels (V1 deterministic execution, V2 frozen rubric, V3 bias-controlled LLM-judge jury), with a hard-constraint gate and degenerate-strategy integrity flags.
5. **Metrics** (`docs/metrics.md`, `usabench.eval` registry) is the scientific core: it defines the canonical metric registry (dimensions A–G), the **0–5 assistance-severity scale**, the **assistance cost** aggregation, and the **headline geometric Usability Score**, plus normalization and the `pass^k` robustness layer.
6. **Infrastructure** (`docs/infra.md`, everything else) makes it reproducible, automated, and DAIC-runnable: one uniform LLM client for API + vLLM models, config hashing + run manifests + lockfiles, SLURM job templates, and a zero-cost `FakeLLM` smoke path.

The architectural commitment that makes all six compose: **the runtime's only job is to produce a complete, replayable, severity-graded `trace.jsonl`; every metric is a pure offline function of that trace plus the frozen task gold.** If a metric cannot be computed from the trace, either the metric or the schema is wrong — we fix the schema, never add hidden runtime state.

## 4. Metric framework summary (canonical)

Metrics are grouped into seven dimensions with stable IDs. The metrics doc owns the full registry; the `metrics` deliverable of this synthesis is the machine-readable distillation. The naming scheme is **one scheme** (dimension-letter + index, e.g. `C1_assistance_cost`); the scoring doc's `GA`, `V1/V2/V3`, and `AC` are **derived names** that map onto it:

- **A — Goal achievement.** `A1_success_binary`, `A2_criteria_score` (weighted fraction of acceptance criteria), `A3_core_criteria_score` (must-haves only — the anti-stuffing signal and the success input to the composite), `A4_goal_drift`, `A5_regression_free`. **`GA` (scoring doc) = the A-dimension score produced by the three verifier channels**; `GA_core` = `A3`.
- **B — Interaction load.** Counts/timings: `B1_n_interventions`, `B2_n_clarifications`, `B3_n_hint_requests`, `B4_n_corrections`, `B5_n_handoffs`, `B6_turns_to_first_working`, `B7_turns_to_acceptance`, `B8_interventions_to_acceptance`. Never read alone — always reported jointly with A.
- **C — Assistance amount & severity (the novelty).** `C1_assistance_cost` (**AC**; convex-weighted sum over the 0–5 severity of each oracle response), `C2_max_severity`, `C3_severity_histogram`, `C4_spec_info_transferred`, `C5_solution_leakage`, `C6_assistance_efficiency`. **The scoring doc's `AC` is `C1`, normalized to [0,1] per task.**
- **D — Autonomy.** `D1_autonomy_ratio`, `D2_unaided_progress_fraction`, `D3_self_recovery_rate`, `D4_blocked_resolution_self`, `D5_proactive_inference` (rewards correctly inferring hidden spec *without* asking — the positive complement of `C4`).
- **E — Efficiency.** `E1_wall_clock_s`, `E2_tokens_total`, `E3_cost_usd_total`, `E7_cost_per_progress`, `E8_tokens_per_progress`. Agent cost and oracle cost are tracked **separately**; ranking uses agent-side only.
- **F — Robustness (stochasticity).** `F1_pass_hat_k` (the headline reliability number), `F2_pass_at_k` (diagnostic only — rewards lucky retries), `F3_success_rate`, `F4_score_cv`, `F5_assistance_cv`, `F6_usability_iqr`. Default **N = 5 seeds** test / 3 dev; **all headline numbers reported as median ± IQR over seeds**, single-seed numbers never published.
- **G — Communication UX (side-panel, not headline).** `G1_question_quality`, `G2_redundant_query_rate`, `G3_status_transparency`, `G4_false_confidence`.

**Severity scale (canonical, one scale).** Each `oracle_response` is graded 0–5: `0` none, `1` trivial clarification, `2` substantive spec info, `3` directional hint, `4` partial solution, `5` takeover. This is the *same* scale the oracle's hint-ladder uses (L0–L5), so the oracle's self-declared level is mechanically checkable against the logged severity. **Canonical convex weights `w = [0, 1, 3, 6, 12, 25]`** (the metrics owner's choice; convexity ensures one sev-5 ≫ many sev-1). *(The harness doc proposed `[0,1,2,4,8,16]`; this synthesis adopts the metrics doc's `[0,1,3,6,12,25]` as the single source of truth in `usability_score.yaml` — see inconsistency #1.)*

**Headline composite — Usability Score (geometric).** Per task, on direction-corrected [0,1] inputs:

```
USABILITY = ( S^α · H^β )^(1/(α+β)) · E^γ · R^δ      α=0.55 β=0.45 γ=0.20 δ=0.20
  S = A3_core_criteria_score
  H = 1 − min(1, AC_rel),  AC_rel = C1 / (κ · E[C1 over reference panel on this task]),  κ=2
  E = 1 / (1 + cost_per_progress / median_cost_per_progress_panel)
  R = F1_pass_hat_k  at k=2
```

The geometric coupling of S and H is the *behavioral target made math*: you must **both** succeed **and** do it with little help — S=1 with H→0 (constant takeovers) yields USABILITY→0, and H=1 with S→0 (never asks, builds wrong thing) also yields 0. A **secondary multiplicative variant** `U = GA·(1−λ·AC)·(1−fake_done_penalty)` (λ=0.5, penalty=0.25) is reported for interpretability and is the form the scoring doc operationalizes; both must reference the *same* `S/GA_core` and `AC` and live in `usability_score.yaml`. *(The two composite forms are reconciled by sharing inputs; the geometric form is headline — see inconsistency #2.)*

**Anti-gaming guards (enforced, not optional).** Never-ask → caught by S (under-specified tasks tank `A3`) plus an explicit `UnderAsk` penalty `H ← H·(1−ρ)` when `success=0 ∧ n_clarifications=0 ∧ goal_drift>τ`. Over-ask → caught by H (convex AC) and E (turns/tokens). Criteria-stuffing → headline uses **core-only** `A3`. Solution-laundering → `C5_solution_leakage` + sev-5 weight 25 + `oracle_attributed_credit` flag. Oracle over-helping → severity-capped prompt + independent offline re-grading (Cohen's κ audit) + `C4` per-task QA. Seed cherry-picking → impossible (median over fixed seeds + `pass^k`). A `sensitivity` notebook perturbs (α,β,γ,δ,w,κ) and reports leaderboard **rank stability (Kendall's τ ≥ 0.9)** before any ranking claim is published.

## 5. Verification: how an open-ended artifact becomes a number (GA)

Because deliverables are open-ended, GA is a weighted blend of three channels with a hard-constraint gate (ScienceAgentBench-style, SWE-bench-style gating expressed gracefully):

```
GA_raw = w_v1·V1 + w_v2·V2 + w_v3·V3            (default 0.40/0.35/0.25, per-task frozen)
GA     = GA_raw · gate(hard_pass_frac),  gate(h)=0.30 + 0.70·h
```

- **V1 — functional/sandbox execution** (deterministic, highest trust): `install_ok`, `valid_exec`, `func_criteria` via *behavior-asserting, artifact-agnostic* golden checkers that drive the declared entrypoint against fixtures. `V1 = 0.25·install_ok + 0.25·valid_exec + 0.50·func_criteria`.
- **V2 — frozen rubric** authored offline from `hidden_spec` (not synthesized per-run — the key variance lever); each criterion routes to exactly one channel (`functional` | `rubric_auto` | `judge`).
- **V3 — LLM-judge jury** for qualitative criteria with **mandatory bias controls**: position-swap calibration, a **panel of J=3 heterogeneous judges** (median), verbosity/style defense, rubric decomposition (no holistic 1–10), self-preference guard, and abstention. Lowest weight because noisiest.

Hard constraints gate rather than zero, so a polished-but-wrong tool is capped, not cliff-edged. GA then composes with `AC` into the Usability Score.

## 6. Architecture overview

```
src/usabench/
  core/      schema (Task, hidden_spec, AcceptanceCriterion, Interaction, trace events), enums (InteractionType, Severity), ids (run_id hashing)
  llm/       ONE uniform client: OpenAI+vLLM (OpenAI-compatible) and Anthropic, normalized; retry/usage/budget/cache
  agent/     agent-under-test wrapper + reference ReAct scaffold (one-action-per-step) + thin framework adapters
  oracle/    simulated-user oracle: persona, reveal/hint-ladder policy, severity classifier, prompt templates
  harness/   runner (the loop), InteractionBus (sole agent↔oracle channel), sandbox (Docker/Apptainer), budget, manifest, batch
  eval/      scoring channels (V1/V2/V3, GA gate), metric registry (A–G), composite, normalize, aggregate (pass^k), integrity, severity audit
  report/    leaderboard + HTML trace viewer
  config/    YAML loader + canonical-JSON config hashing
collect/     GitHub REST/GraphQL ETL → raw_harvest.jsonl → candidates.jsonl (depends only on usabench.core)
tasks/       benchmark content: schema/, curated/*.jsonl, per-task tasks/<id>/{task.json, env/, grader/}
configs/     models/ oracle/ agents/ runs/ daic/  (everything hashed into run_id)
daic/        SLURM sbatch templates + env setup + secrets loader + results sync
schemas/     trace.schema.json (canonical), task.schema.json, run_log views
docs/        the six per-area specs
```

Three cross-cutting invariants make it coherent:

1. **One canonical trace.** `schemas/trace.schema.json` (the harness doc's append-only, totally-ordered, hash-chained JSONL envelope) is *the* artifact. The metrics doc's "Episode/Event" objects and the infra doc's `interactions.jsonl`/`trajectory.jsonl` are **views derived from it**, not separate sources of truth. The scorer reads only `trace.jsonl` + the frozen `hidden_spec`.
2. **One uniform LLM path.** Agent-under-test, oracle, and judges all call `usabench.llm.LLMClient.chat()`. vLLM is reached through the OpenAI client (only `base_url` differs), so there is a *single* code path for OpenAI-shaped backends; the oracle is **always API-based** so its behavior is a constant across the agent grid.
3. **One config → one run_id.** `run_id = sha256(canonical_json(config) + task_id + seed + git_sha)`; everything outcome-affecting (image digests, model ids, decoding, seeds, budgets, oracle prompt hash, lockfile hash) is captured in `run_start` and the run manifest. Results compare only within one `release.lock`.

**DAIC topology.** Data collection + API-agent runs + oracle live on the internet-capable **login/CPU node**; open-weight agents run as **GPU SLURM jobs** serving vLLM (`--gres=gpu:a40:N`). When GPU nodes are firewalled from the model APIs, the harness runs **split-mode**: GPU node serves vLLM and exposes a port; the harness+oracle run on the login node and connect to `http://<gpu-node>:<port>/v1`. Shared state (traces, blobs, hidden_specs, HF cache, conda env) lives on `/tudelft.net/staff-umbrella/CoReFusion/usabench`; the small home dir stays clean. Python **3.11** everywhere; pip + committed lockfiles; conda on DAIC.

## 7. Reproducibility & stochasticity (first-class)

LLM stochasticity is not noise to hide but a quantity to report. Every `(task, agent)` runs **N≥5 seeds**; the oracle runs at low temperature (≤0.3) with a frozen, hashed system prompt and content-addressed **replay cache** for bit-stable re-scoring. Headlines are **median ± IQR** with cluster-bootstrap CIs (resample tasks then seeds). `F1_pass_hat_k` (probability *all* k reruns succeed) is the recommended "can you trust it to do it again" number; `pass@k` is diagnostic only. A `release.lock` content-hashes every model/prompt/image/dataset so two numbers are comparable iff they share a lock. The whole scoring stack is deterministic given the trace + lock, except bounded judge calls folded into the reruns.

## 8. Phased roadmap (summary; full version in the `roadmap` deliverable)

- **P0 Skeleton & contracts** — repo scaffold, `core` schemas, canonical `trace.schema.json`, `FakeLLM` smoke path green end-to-end (zero API cost), CI.
- **P1 Single-episode vertical slice** — harness loop + InteractionBus + sandbox + oracle + one hand-authored task (`ub-cal-0007`), one real API agent, V1 scoring.
- **P2 Metrics & scoring complete** — full A–G registry, three verifier channels, geometric composite, `pass^k`, severity audit, golden-fixture regression tests.
- **P3 Data pipeline at scale** — GitHub collector, drafter, gold-authoring + calibration gate, ~30 dev tasks across 6 domains × 4 tiers.
- **P4 DAIC productionization** — vLLM serving, split-mode, batch fan-out, project-storage layout, results sync.
- **P5 Benchmark v1 release** — ~80 public + ~40 private held-out tasks, baseline agent panel, leaderboard, human-validation subset, paper.

## 9. Risks & known tensions

The top scientific risks are oracle proxy-fidelity, training-data contamination of reference repos, judge variance, the discriminativeness requirement (tasks must *need* assistance), and metric-weight arbitrariness; each has a concrete mitigation (human-validated oracle subset, recency-biased harvest + private held-out + recall probes, jury + position-swap + κ audit, the calibration gate, the sensitivity/rank-stability notebook). The cross-doc contradictions this synthesis resolved — severity weights, composite formula, metric-naming, trace schema, repo layout — are recorded in the `inconsistencies` register with the decision taken. The single most important guard against the benchmark fooling itself: **every shipped task is empirically shown, during calibration, to require and reward assistance**, so the headline metric measures a real, load-bearing quantity rather than an artifact of task design.