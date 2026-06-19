# Leaderboard

Recorded usability-benchmark results. Each row in `data/*.jsonl` is one
`(agent, taskset)` aggregate produced by `usabench score` + `usabench leaderboard`
from the canonical per-episode `trace.jsonl` files (every metric is a pure offline
function of the traces + the frozen task gold).

## `data/smoke_baselines_v0.jsonl` — first DAIC baseline (v0_smoke)

Open-weight models served by **vLLM 0.23** on a TU Delft **DAIC** RTX PRO 6000
(96 GB) GPU. The **same served model plays both the agent-under-test and the
simulated-user oracle** (fully self-contained — no external API keys). Tasks:
the 3 `v0_smoke` tool-building goals; deliverables graded by the deterministic
`FunctionalVerifier`. `n_seeds = 1` (a smoke-scale baseline, not the variance-
controlled `n=5` test protocol).

| Agent | Interface | Usability ★ | GA | pass¹ | V1 | Interactions/task | HelpSev | accepted |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-32B-Instruct** | ReAct text | **1.000** | 1.000 | 1.000 | 1.00 | 2.67 | 0.00 | 3/3 |
| **Qwen2.5-7B-Instruct**  | ReAct text | **0.000** | 0.000 | 0.000 | 0.50 | 2.67 | 2.00 | 0/3 |
| **Qwen2.5-7B-Instruct**  | native tools | **0.000** | 0.000 | 0.000 | 0.33 | 1.33 | 0.33 | 0/3 |

**Reading it.** The benchmark *discriminates*: the 32B builds all three tools and
is accepted essentially autonomously (its 2.67 oracle touches/task are severity-0
clarifications/reviews — `HelpSev 0`), scoring a perfect usability. The 7B solves
nothing under either interface and the verifier correctly fails it — vs. 1.0 for
a reference agent that writes correct files, so this is a genuine
agentic-competence gap, not a grading artifact.

**Native tool-calling does not rescue the 7B (tested).** With vLLM
`--enable-auto-tool-choice --tool-call-parser hermes` and `agent.native_tools:true`,
the 7B *does* call tools — its traces show clarification questions and `run_cmd`
executions — but it never writes the deliverable file before running/declaring,
then hands off. So the easier interface changes the *failure mode* (engaged but
unproductive) without changing the outcome (0.0). The bottleneck is planning, not
format parsing.

**Caveats / next steps.**
- `n_seeds = 1`; the headline `pass^k` reliability metric needs `n ≥ 5` (see
  `configs/runs/full_v1.yaml`).
- These models likely have some exposure to trivial CLI patterns; the real task
  set uses recency + held-out splits to control contamination (`docs/tasks.md`).
- Worth adding a mid-size rung (Qwen2.5-14B, cached) to trace where the
  build-then-verify competence emerges between 7B and 32B.

Reproduce: `sbatch daic/slurm/run_qwen7b_smoke.sbatch` (7B) or with
`--gres=gpu:nvidia_rtx_pro_6000:1 --export=ALL,UB_MODEL=Qwen/Qwen2.5-32B-Instruct,UB_RUN_CONFIG=configs/runs/qwen32b_smoke.yaml,UB_RUNS_SUBDIR=qwen32b_smoke`
(32B). See `daic/README.md`.
