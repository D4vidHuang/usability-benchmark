#!/bin/bash
# =============================================================================
# run_qwen7b_smoke.sbatch -- FIRST real baseline on DAIC (co-located serve+run).
#
# One GPU node: serve Qwen2.5-7B-Instruct with vLLM (OpenAI-compatible), wait for
# the endpoint, then run the usabench harness (agent + oracle = the served model)
# on the v0_smoke taskset, score it, and render the leaderboard. Self-contained:
# uses the project uv venv and the model already cached in HF_HOME (offline).
#
# Submit:  cd $USABENCH_PROJ && sbatch daic/slurm/run_qwen7b_smoke.sbatch
# =============================================================================
#SBATCH --job-name=ub-qwen7b-smoke
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=ub-qwen7b-smoke-%j.out

set -uo pipefail

PROJ=/tudelft.net/staff-umbrella/CoReFusion/usabench
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/tudelft.net/staff-umbrella/CoReFusion/hf_cache
export HF_HUB_OFFLINE=1            # use the cached model; never hit the network
export TRANSFORMERS_OFFLINE=1
# Run vLLM KEYLESS: do NOT export VLLM_API_KEY (if set, vLLM *requires* that bearer
# token and the keyless health-check curl below would 401). The harness client
# carries an inline dummy api_key ("EMPTY") which a keyless server ignores.
PORT=8765
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
# Parameterizable so the same script serves any cached model: override via
#   sbatch --export=ALL,UB_MODEL=...,UB_RUN_CONFIG=...,UB_RUNS_SUBDIR=...
MODEL="${UB_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
RUN_CONFIG="${UB_RUN_CONFIG:-configs/runs/qwen7b_smoke.yaml}"
RUNS_ROOT="$PROJ/runs/${UB_RUNS_SUBDIR:-qwen7b_smoke}"
TASKSET="tasks/curated/v0_smoke.jsonl"

mkdir -p "$RUNS_ROOT"
cd "$PROJ" || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate
module load cuda/12.9 2>/dev/null || true
echo "[env] node=$(hostname) python=$(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

echo "[serve] launching vLLM for $MODEL ..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16 \
  > "$RUNS_ROOT/vllm-serve.log" 2>&1 &
VLLM_PID=$!

cleanup() { kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[serve] waiting for endpoint (up to ~5 min) ..."
UP=0
for i in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[serve] endpoint up after $((i*2))s"; UP=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[serve] vLLM process died early; tail of log:"; tail -50 "$RUNS_ROOT/vllm-serve.log"; exit 1
  fi
  sleep 2
done
[ "$UP" -eq 1 ] || { echo "[serve] endpoint never came up; tail of log:"; tail -50 "$RUNS_ROOT/vllm-serve.log"; exit 1; }

usabench serve-check --base-url "$VLLM_BASE_URL" --model "$MODEL" || true

echo "[run] running episodes ..."
usabench run --config "$RUN_CONFIG" --output-root "$RUNS_ROOT/runs" --no-resume

echo "[score] scoring ..."
usabench score --runs "$RUNS_ROOT/runs" --tasks "$TASKSET" --scores "$RUNS_ROOT/scores"

echo "[leaderboard] rendering ..."
usabench leaderboard --scores "$RUNS_ROOT/scores" --out "$RUNS_ROOT/scores" \
  --title "Qwen2.5-7B-Instruct -- smoke baseline (DAIC)"

echo "===== LEADERBOARD (markdown) ====="
cat "$RUNS_ROOT/scores/leaderboard.md" 2>/dev/null || echo "(no leaderboard written)"
echo "[done] artifacts under $RUNS_ROOT"
