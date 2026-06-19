# Running `usabench` on DAIC

This directory holds everything needed to run the benchmark on the **DAIC**
cluster: environment bootstrap, secrets loading, SLURM job scripts, and a
results-sync helper.

> **DAIC reality (read this first).** DAIC has **no working conda / uv / pixi**
> and only **system Python 3.9**. We therefore bootstrap with **`uv`** (installed
> to `~/.local/bin`) and create a **Python 3.11** virtualenv on **project
> storage**. `daic/env/environment.yml` is a *fallback only* (conda is not
> available here). vLLM runs via **Apptainer 1.5.0** on GPU nodes.

Key facts baked into these scripts:

| Thing | Value |
|---|---|
| Project storage root (`$USABENCH_PROJ`) | `/tudelft.net/staff-umbrella/CoReFusion/usabench` |
| Shared HF cache (`$HF_HOME`) | `/tudelft.net/staff-umbrella/CoReFusion/hf_cache` (reused) |
| Venv (`$VENV`) | `$USABENCH_PROJ/envs/usabench` (uv-managed, Python 3.11) |
| uv / pip / apptainer caches | under `$USABENCH_PROJ/cache` |
| SSH alias | `daic` |
| SLURM | 25.05; partitions `test`, `all` (default), `ewi-insy`, `ewi-me`, `ewi-st` |
| GPUs | `--gres=gpu:a40:1` (types: `a40`, `nvidia_rtx_pro_6000`/`rtx_pro_6000`, `l40`) |
| Containers | Apptainer 1.5.0 (for vLLM) |

---

## 0. Layout of this directory

```
daic/
├── README.md                  # this file
├── env/
│   ├── paths.sh               # the env contract: PROJ/VENV/HF_HOME/caches (source it)
│   ├── setup_env.sh           # idempotent: install uv, uv venv 3.11, uv pip install -e .[extras]
│   └── environment.yml        # conda FALLBACK only (conda unavailable on DAIC)
├── secrets/
│   └── load_secrets.sh        # source ~/.config/usabench/secrets.env (chmod 600)
├── slurm/
│   ├── collect_cpu.sbatch     # CPU, internet: usabench-collect run
│   ├── vllm_serve.sbatch      # GPU, serve-only vLLM (Apptainer) -> writes endpoint.txt  [split mode]
│   ├── run_vllm.sbatch        # GPU, co-located vLLM serve + harness on one node
│   ├── run_api.sbatch         # CPU, internet: harness vs API models + oracle (also split-mode login half)
│   └── score.sbatch           # CPU: offline scoring + leaderboard
└── sync/
    └── pull_results.sh        # run on the Mac: rsync manifests/scores/traces back
```

---

## 1. One-time setup

```bash
# From your Mac:
ssh daic

# Tell the scripts where THIS checkout lives (clone it on project storage so it
# is visible to compute nodes). Adjust if you keep the repo elsewhere.
export USABENCH_REPO=/tudelft.net/staff-umbrella/CoReFusion/usabench/usability-benchmark
cd "$USABENCH_REPO"

# 1a) Load the path contract (idempotent; only exports vars + makes dirs).
source daic/env/paths.sh

# 1b) Bootstrap the environment (installs uv if missing, creates the 3.11 venv,
#     installs the package). Run on the LOGIN node -- it needs internet.
source daic/env/setup_env.sh
#   -> creates $VENV (= $USABENCH_PROJ/envs/usabench) and installs .[api,collect,dev]
#   For a GPU node that will pip-install vLLM too (co-located runs only):
#     USABENCH_GPU=1 source daic/env/setup_env.sh   # adds the .[serve] extra

# 1c) Create your secrets file (NOT in git, chmod 600).
mkdir -p ~/.config/usabench
cat > ~/.config/usabench/secrets.env <<'EOF'
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GITHUB_TOKEN=ghp_...
EOF
chmod 600 ~/.config/usabench/secrets.env
source daic/secrets/load_secrets.sh    # confirms which keys are set (values never printed)
```

**Re-running setup is safe.** `setup_env.sh` checks for `uv`, the venv, and just
re-installs the editable package; nothing is destroyed.

### vLLM image (one-time, GPU runs only)

vLLM runs in **Apptainer**. Build/pull the image **on the login node** (internet)
and store it on project storage:

```bash
source daic/env/paths.sh
apptainer pull "$USABENCH_APPTAINER_SIF" docker://vllm/vllm-openai:latest
# -> /tudelft.net/staff-umbrella/CoReFusion/usabench/images/vllm.sif
```

---

## 2. Collect reference repos (CPU, needs internet)

```bash
cd "$USABENCH_REPO"
sbatch daic/slurm/collect_cpu.sbatch          # writes data/ on project storage
# If compute nodes are firewalled from GitHub, just run it on the login node:
source daic/env/setup_env.sh && source daic/secrets/load_secrets.sh
usabench-collect run --config configs/harvest.yaml --out-dir "$USABENCH_DATA_ROOT"
```

---

## 3. Run the harness

There are two topologies. **Pick based on whether the GPU node can reach the
internet** (the oracle is always an API model and needs internet).

### 3a. API models under test (CPU)

```bash
sbatch daic/slurm/run_api.sbatch              # all API models in the run config
# single model:  USABENCH_ONLY_MODEL=configs/models/claude_opus.yaml sbatch daic/slurm/run_api.sbatch
```

### 3b. Local (vLLM) model under test

**Option A -- co-located (GPU node has internet for the oracle):**

```bash
sbatch daic/slurm/run_vllm.sbatch             # serves vLLM + runs harness on one GPU node
```

**Option B -- SPLIT MODE (GPU node is network-restricted):** the GPU node only
*serves* vLLM; the harness **and the API oracle** run on the login/CPU node and
connect to the GPU node over the cluster network.

```bash
# 1) Serve on a GPU node. It writes a reachable endpoint to endpoint.txt.
JOBID=$(sbatch --parsable daic/slurm/vllm_serve.sbatch)
EPFILE="$USABENCH_RUNS_ROOT/vllm-$JOBID/endpoint.txt"

# 2) Wait for the endpoint file, then run the harness + oracle on CPU/login.
until [ -f "$EPFILE" ]; do sleep 15; done
export USABENCH_VLLM_BASE_URL="$(cat "$EPFILE")"   # e.g. http://<gpu-node>:8000/v1
export USABENCH_VLLM_API_KEY=local-dummy
USABENCH_ONLY_MODEL=configs/models/qwen_vllm.yaml sbatch daic/slurm/run_api.sbatch

# 3) When done, release the GPU:
scancel "$JOBID"
```

**Why split mode exists.** DAIC compute nodes are assumed network-restricted,
but the oracle is an API model that needs the internet. Co-locating the harness
with vLLM on a firewalled GPU node would block the oracle. Splitting keeps the
GPU node serving-only (no internet needed) and runs the harness+oracle where the
internet is. The same `configs/models/qwen_vllm.yaml` works in both modes because
it reads the endpoint from `$USABENCH_VLLM_BASE_URL`.

> **Hermeticity is independent of node connectivity.** Scored runs are hermetic:
> the agent **and** the verification sandbox default to **network-DENY** with a
> per-task allowlist, regardless of whether the node itself has internet. The
> oracle channel is the only network path the agent has, and it is mediated +
> logged by the harness.

---

## 4. Score (CPU, offline)

Scoring is a pure offline function of `trace.jsonl` + frozen task gold (no GPU;
the LLM-judge V3 channel uses the API oracle if enabled).

```bash
sbatch daic/slurm/score.sbatch
# -> writes <run_id>/scores.json under $USABENCH_RUNS_ROOT and a leaderboard JSONL.
```

---

## 5. Pull results back to the Mac

```bash
# Run LOCALLY on your laptop (uses ssh alias `daic`):
bash daic/sync/pull_results.sh ~/usabench-results
# Pulls manifests, scores, trace.jsonl, and the leaderboard.
# EXCLUDES heavy agent workspaces and *.sif images.
```

---

## Cheat sheet

```bash
ssh daic
export USABENCH_REPO=/tudelft.net/staff-umbrella/CoReFusion/usabench/usability-benchmark
cd "$USABENCH_REPO"
source daic/env/paths.sh
source daic/env/setup_env.sh            # one-time / after repo updates
source daic/secrets/load_secrets.sh
sbatch daic/slurm/collect_cpu.sbatch    # collect
sbatch daic/slurm/run_api.sbatch        # run (API)   OR  the split-mode pair (3b)
sbatch daic/slurm/score.sbatch          # score
# then on the Mac:
bash daic/sync/pull_results.sh
```

## Notes / gotchas

- **Submit from the repo root.** The sbatch scripts use `$SLURM_SUBMIT_DIR` as
  the repo root to find `daic/env/setup_env.sh` and the configs.
- **Home dir is tiny.** Everything heavy (venv, HF models, runs, caches) is kept
  under `$USABENCH_PROJ` / `$HF_HOME` by `paths.sh`. Don't override these to home.
- **Partitions / GPUs.** Scripts default to `--partition=all` and
  `--gres=gpu:a40:1`. Edit the `#SBATCH` lines (or use `test` for quick checks,
  `ewi-insy`/`ewi-me`/`ewi-st` for department queues) and swap the GPU type
  (`a40`, `l40`, `rtx_pro_6000`) as availability dictates.
- **The `usabench run|score|leaderboard` subcommands** are wired by the harness /
  eval / report workstreams; the scripts here invoke them with the intended
  flags. `usabench version`, `usabench validate-spec`, and the full
  `usabench-collect` CLI are available today.
```
