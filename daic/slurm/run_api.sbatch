#!/usr/bin/env bash
# daic/slurm/run_api.sbatch -- harness vs API models + oracle (CPU, needs internet).
#
# Two uses:
#   (a) API models under test (Claude/GPT) graded by the API oracle, and
#   (b) the LOGIN-NODE half of SPLIT MODE: a vLLM model is served on a GPU node
#       (vllm_serve.sbatch) and this CPU job runs the harness + API oracle
#       against it, reaching the GPU node's endpoint over the cluster network.
#
# Needs internet (API oracle + API models), so it targets `all`/`test`, or run
# it directly on the login node.
#
# Split-mode wiring: set USABENCH_VLLM_BASE_URL before submitting, e.g.
#     export USABENCH_VLLM_BASE_URL=$(cat $USABENCH_RUNS_ROOT/vllm-<jobid>/endpoint.txt)
#     sbatch daic/slurm/run_api.sbatch
#
#SBATCH --job-name=usabench-run-api
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/run-api-%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/run-api-%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
export USABENCH_REPO="$REPO"

# CPU/login extras (.[api,collect,dev]); no GPU, no serve extra.
# shellcheck source=daic/env/setup_env.sh
source "$REPO/daic/env/setup_env.sh"
# shellcheck source=daic/secrets/load_secrets.sh
source "$REPO/daic/secrets/load_secrets.sh"

if [ -z "${ANTHROPIC_API_KEY:-}${OPENAI_API_KEY:-}" ]; then
  echo "[run-api] ERROR: no API key set; the oracle (and API models) need one." >&2
  exit 2
fi

# Split-mode convenience: if a vLLM endpoint file path was given, read it.
if [ -n "${USABENCH_VLLM_ENDPOINT_FILE:-}" ] && [ -f "$USABENCH_VLLM_ENDPOINT_FILE" ]; then
  USABENCH_VLLM_BASE_URL="$(cat "$USABENCH_VLLM_ENDPOINT_FILE")"
  export USABENCH_VLLM_BASE_URL
fi
if [ -n "${USABENCH_VLLM_BASE_URL:-}" ]; then
  echo "[run-api] split mode: vLLM endpoint = $USABENCH_VLLM_BASE_URL"
fi

RUN_CONFIG="${USABENCH_RUN_CONFIG:-$REPO/configs/runs/full_v1.yaml}"

# NOTE: `usabench run` is wired by the harness workstream. By default we run all
# API models in the run config and skip the local vLLM model (which is served by
# a GPU job); pass --only-model to target a single config instead.
echo "[run-api] harness: config=$RUN_CONFIG runs_root=$USABENCH_RUNS_ROOT"
usabench run \
  --config "$RUN_CONFIG" \
  ${USABENCH_ONLY_MODEL:+--only-model "$USABENCH_ONLY_MODEL"} \
  ${USABENCH_SKIP_MODEL:+--skip-model "$USABENCH_SKIP_MODEL"} \
  --output-root "$USABENCH_RUNS_ROOT"

echo "[run-api] done -> $USABENCH_RUNS_ROOT"
