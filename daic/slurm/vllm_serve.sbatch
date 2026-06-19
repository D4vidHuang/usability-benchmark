#!/usr/bin/env bash
# daic/slurm/vllm_serve.sbatch -- SPLIT-MODE vLLM server (GPU, serve only).
#
# Starts a vLLM OpenAI-compatible server on a GPU node via Apptainer 1.5.0 and
# writes the node-reachable base_url to endpoint.txt. The HARNESS + ORACLE run
# SEPARATELY on a login/CPU node (see run_api.sbatch / split-mode in README),
# because compute nodes are assumed network-restricted and the API oracle needs
# internet. This job serves and then BLOCKS until its --time wall is hit (or it
# is scancel'd); the harness reads endpoint.txt while it is up.
#
# Submit:  sbatch daic/slurm/vllm_serve.sbatch
# Then on the login node:
#     EP=$(cat $USABENCH_RUNS_ROOT/vllm-<jobid>/endpoint.txt)
#     export USABENCH_VLLM_BASE_URL="$EP"
#     sbatch daic/slurm/run_api.sbatch        # harness+oracle vs the served model
#
#SBATCH --job-name=usabench-vllm-serve
#SBATCH --partition=all
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/vllm-serve-%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/vllm-serve-%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
export USABENCH_REPO="$REPO"
export USABENCH_GPU=1

# Paths only (NOT setup_env): the serving stack runs inside Apptainer, so we do
# not need the uv venv on the GPU node. We just need HF_HOME + the SIF path.
# shellcheck source=daic/env/paths.sh
source "$REPO/daic/env/paths.sh"

# --- Tunables (override via env at submit time) ------------------------------
SERVED_MODEL="${USABENCH_VLLM_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
MAX_MODEL_LEN="${USABENCH_VLLM_MAX_LEN:-16384}"
TP_SIZE="${USABENCH_VLLM_TP:-1}"
PORT="${USABENCH_VLLM_PORT:-8000}"
API_KEY="${USABENCH_VLLM_API_KEY:-local-dummy}"
SIF="${USABENCH_APPTAINER_SIF:-$USABENCH_PROJ/images/vllm.sif}"

# Reachable host for OTHER nodes (split mode): node FQDN/short name, not 127.0.0.1.
HOST_ADDR="$(hostname -f 2>/dev/null || hostname)"

# Where to advertise the endpoint so the login-node harness can find it.
JOB_DIR="${USABENCH_RUNS_ROOT}/vllm-${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_DIR"
ENDPOINT_FILE="$JOB_DIR/endpoint.txt"

echo "[vllm-serve] model=$SERVED_MODEL tp=$TP_SIZE len=$MAX_MODEL_LEN host=$HOST_ADDR port=$PORT"
echo "[vllm-serve] sif=$SIF HF_HOME=$HF_HOME"

if [ ! -f "$SIF" ]; then
  echo "[vllm-serve] ERROR: Apptainer image not found: $SIF" >&2
  echo "[vllm-serve] Build/pull it ONCE on the login node (internet), e.g.:" >&2
  echo "    apptainer pull \"$SIF\" docker://vllm/vllm-openai:latest" >&2
  exit 3
fi

# --- Launch vLLM inside Apptainer, bound to all interfaces (split mode) -------
# --nv exposes the GPUs; --bind shares HF_HOME so the model downloads once.
apptainer exec --nv \
  --bind "$HF_HOME:$HF_HOME" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_HUB_CACHE=$HF_HUB_CACHE" \
  "$SIF" \
  python -m vllm.entrypoints.openai.api_server \
    --model "$SERVED_MODEL" \
    --served-model-name "$SERVED_MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --host 0.0.0.0 --port "$PORT" \
    --api-key "$API_KEY" \
    --download-dir "$HF_HUB_CACHE" &
VLLM_PID=$!
trap 'echo "[vllm-serve] stopping vLLM ($VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null || true; rm -f "$ENDPOINT_FILE"' EXIT

# --- Wait for health, then publish the reachable endpoint --------------------
BASE_URL="http://${HOST_ADDR}:${PORT}/v1"
echo "[vllm-serve] waiting for health at http://127.0.0.1:${PORT}/health ..."
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "$BASE_URL" > "$ENDPOINT_FILE"
    echo "[vllm-serve] UP. endpoint=$BASE_URL  (wrote $ENDPOINT_FILE)"
    break
  fi
  sleep 10
  if [ "$i" -eq 180 ]; then
    echo "[vllm-serve] ERROR: vLLM did not become healthy in 30 min." >&2
    exit 1
  fi
done

echo "[vllm-serve] serving until --time wall or scancel. Run the harness on the login node:"
echo "    export USABENCH_VLLM_BASE_URL=$BASE_URL"
echo "    export USABENCH_VLLM_API_KEY=$API_KEY"
echo "    sbatch daic/slurm/run_api.sbatch   # or: usabench run ... --only-model configs/models/qwen_vllm.yaml"

# Block on the server so the allocation stays alive while the harness uses it.
wait "$VLLM_PID"
