#!/usr/bin/env bash
# daic/slurm/run_vllm.sbatch -- CO-LOCATED vLLM serve + harness on ONE GPU node.
#
# Starts vLLM (Apptainer) bound to localhost AND runs the harness against it on
# the same node. Use this when the GPU node CAN reach the API oracle (Anthropic/
# OpenAI) over the network. If it CANNOT, use split mode instead:
# vllm_serve.sbatch (GPU) + run_api.sbatch (login/CPU) -- see README.
#
# The model-under-test is local (vLLM); the ORACLE stays API-based for
# consistency, so this node still needs internet for the oracle. Scored runs are
# HERMETIC: the agent + verification sandbox default to network-DENY with a
# per-task allowlist -- that is enforced by the harness/sandbox, independent of
# this node's connectivity.
#
# Submit:  sbatch daic/slurm/run_vllm.sbatch
#
#SBATCH --job-name=usabench-run-vllm
#SBATCH --partition=all
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/run-vllm-%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/run-vllm-%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
export USABENCH_REPO="$REPO"
export USABENCH_GPU=1

# Need BOTH the venv (to run the harness) and the secrets (oracle API key).
# shellcheck source=daic/env/setup_env.sh
source "$REPO/daic/env/setup_env.sh"
# shellcheck source=daic/secrets/load_secrets.sh
source "$REPO/daic/secrets/load_secrets.sh"

# --- vLLM serving config -----------------------------------------------------
SERVED_MODEL="${USABENCH_VLLM_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
MAX_MODEL_LEN="${USABENCH_VLLM_MAX_LEN:-16384}"
TP_SIZE="${USABENCH_VLLM_TP:-1}"
PORT="${USABENCH_VLLM_PORT:-8000}"
API_KEY="${USABENCH_VLLM_API_KEY:-local-dummy}"
SIF="${USABENCH_APPTAINER_SIF:-$USABENCH_PROJ/images/vllm.sif}"

if [ ! -f "$SIF" ]; then
  echo "[run-vllm] ERROR: Apptainer image not found: $SIF (pull it on the login node first)." >&2
  exit 3
fi

# 1) Start vLLM on localhost (co-located, so 127.0.0.1 is fine).
apptainer exec --nv \
  --bind "$HF_HOME:$HF_HOME" \
  --env "HF_HOME=$HF_HOME" --env "HF_HUB_CACHE=$HF_HUB_CACHE" \
  "$SIF" \
  python -m vllm.entrypoints.openai.api_server \
    --model "$SERVED_MODEL" --served-model-name "$SERVED_MODEL" \
    --tensor-parallel-size "$TP_SIZE" --max-model-len "$MAX_MODEL_LEN" \
    --host 127.0.0.1 --port "$PORT" --api-key "$API_KEY" \
    --download-dir "$HF_HUB_CACHE" &
VLLM_PID=$!
trap 'echo "[run-vllm] stopping vLLM ($VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null || true' EXIT

# 2) Point the harness's model config at the local endpoint.
export USABENCH_VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export USABENCH_VLLM_API_KEY="$API_KEY"

echo "[run-vllm] waiting for vLLM health at $USABENCH_VLLM_BASE_URL ..."
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[run-vllm] vLLM up."
    break
  fi
  sleep 10
  [ "$i" -eq 180 ] && { echo "[run-vllm] ERROR: vLLM failed to start." >&2; exit 1; }
done

# 3) Run the harness against the local vLLM model.
#    NOTE: `usabench run` is wired by the harness workstream; this is the
#    intended invocation (configs/runs/*.yaml owns task set + budgets + seeds).
RUN_CONFIG="${USABENCH_RUN_CONFIG:-$REPO/configs/runs/full_v1.yaml}"
MODEL_CFG="${USABENCH_MODEL_CFG:-$REPO/configs/models/qwen_vllm.yaml}"

echo "[run-vllm] harness: config=$RUN_CONFIG model=$MODEL_CFG runs_root=$USABENCH_RUNS_ROOT"
usabench run \
  --config "$RUN_CONFIG" \
  --only-model "$MODEL_CFG" \
  --output-root "$USABENCH_RUNS_ROOT"

echo "[run-vllm] harness finished; tearing down vLLM."
