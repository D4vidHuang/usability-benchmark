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

| Agent | Usability ★ | GA | pass¹ | V1 | Interactions/task | HelpSev | accepted |
|---|---|---|---|---|---|---|---|
| **Qwen2.5-32B-Instruct** | **1.000** | 1.000 | 1.000 | 1.00 | 2.67 | 0.00 | 3/3 |
| **Qwen2.5-7B-Instruct**  | **0.000** | 0.000 | 0.000 | 0.50 | 2.67 | 2.00 | 0/3 |

**Reading it.** The benchmark *discriminates*: the 32B builds all three tools
and is accepted essentially autonomously (its 2.67 oracle touches/task are
severity-0 clarifications/reviews — `HelpSev 0`), scoring a perfect usability.
The 7B never emits a parseable file-writing action under the ReAct-text scaffold
— it converses and declares done empty-handed — so it solves nothing and the
verifier correctly fails it (a genuine agentic-competence gap, not a grading
artifact; the verifier gives 1.0 to a reference agent that writes correct files).

**Caveats / next steps.**
- `n_seeds = 1`; the headline `pass^k` reliability metric needs `n ≥ 5` (see
  `configs/runs/full_v1.yaml`).
- These models likely have some exposure to trivial CLI patterns; the real task
  set uses recency + held-out splits to control contamination (`docs/tasks.md`).
- The 7B's failure is partly scaffold-format adherence; enabling vLLM native
  tool-calling (`--enable-auto-tool-choice --tool-call-parser hermes`) would give
  smaller models a fairer shot and is the recommended next iteration.

Reproduce: `sbatch daic/slurm/run_qwen7b_smoke.sbatch` (7B) or with
`--gres=gpu:nvidia_rtx_pro_6000:1 --export=ALL,UB_MODEL=Qwen/Qwen2.5-32B-Instruct,UB_RUN_CONFIG=configs/runs/qwen32b_smoke.yaml,UB_RUNS_SUBDIR=qwen32b_smoke`
(32B). See `daic/README.md`.
