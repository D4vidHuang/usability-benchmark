#!/usr/bin/env bash
# daic/slurm/collect_cpu.sbatch -- GitHub data collection (CPU, needs internet).
#
# Runs `usabench-collect run` to harvest reference repos into project storage.
# Collection needs internet (GitHub API), so this targets the `all`/`test`
# partition; if compute nodes are firewalled from GitHub, run it on the LOGIN
# node directly instead (the collector is light + ETag-cached):
#
#     source daic/env/setup_env.sh
#     source daic/secrets/load_secrets.sh
#     usabench-collect run --config configs/harvest.yaml --out-dir "$USABENCH_DATA_ROOT"
#
# Submit:  sbatch daic/slurm/collect_cpu.sbatch
#
#SBATCH --job-name=usabench-collect
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/collect-%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/collect-%j.err

set -euo pipefail

# SLURM_SUBMIT_DIR is the repo root (submit from the repo root). Fall back to cwd.
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
export USABENCH_REPO="$REPO"

# 1) Environment (uv venv) + secrets ($GITHUB_TOKEN).
# shellcheck source=daic/env/setup_env.sh
source "$REPO/daic/env/setup_env.sh"
# shellcheck source=daic/secrets/load_secrets.sh
source "$REPO/daic/secrets/load_secrets.sh"

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "[collect] ERROR: GITHUB_TOKEN unset; collection cannot run." >&2
  exit 2
fi

# 2) Harvest. Output (raw_harvest.jsonl, candidates.jsonl, seen.sqlite, ETag
#    cache) lands on project storage so GPU/score jobs can read it offline.
CONFIG="${USABENCH_HARVEST_CONFIG:-$REPO/configs/harvest.yaml}"
OUT_DIR="${USABENCH_DATA_ROOT:-$REPO/data}"
SOURCES="${USABENCH_COLLECT_SOURCES:-topics,awesome_lists}"

echo "[collect] config=$CONFIG out_dir=$OUT_DIR sources=$SOURCES"
usabench-collect run \
  --config "$CONFIG" \
  --out-dir "$OUT_DIR" \
  --sources "$SOURCES" \
  ${USABENCH_MAX_REPOS:+--max-repos "$USABENCH_MAX_REPOS"}

echo "[collect] done -> $OUT_DIR"
