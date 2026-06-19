#!/usr/bin/env bash
# daic/slurm/score.sbatch -- offline scoring + leaderboard aggregation (CPU).
#
# Scoring is a PURE OFFLINE function of trace.jsonl + the frozen task gold, so
# this needs no GPU and (for deterministic V1/V2 checks) no internet. The V3
# LLM-judge channel DOES call the API oracle, so we source secrets in case the
# scoring config enables it. Verification sandboxes are HERMETIC: network-DENY
# with a per-task allowlist; deps resolve from a pre-warmed mirror.
#
# Submit:  sbatch daic/slurm/score.sbatch
#
#SBATCH --job-name=usabench-score
#SBATCH --partition=all
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/score-%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/score-%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
export USABENCH_REPO="$REPO"

# shellcheck source=daic/env/setup_env.sh
source "$REPO/daic/env/setup_env.sh"
# LLM-judge (V3) checks may need an API key; harmless if unset for V1/V2-only.
# shellcheck source=daic/secrets/load_secrets.sh
source "$REPO/daic/secrets/load_secrets.sh"

RUNS_ROOT="${USABENCH_RUNS_ROOT}"
SCORES_ROOT="${USABENCH_SCORES_ROOT}"
LEADERBOARD_OUT="${USABENCH_LEADERBOARD_OUT:-$REPO/leaderboard/data/v1_results.jsonl}"
mkdir -p "$SCORES_ROOT" "$(dirname "$LEADERBOARD_OUT")"

# NOTE: `usabench score` / `usabench leaderboard` are wired by the eval/report
# workstreams. Each scored run reads <run_id>/trace.jsonl (+ frozen gold) and
# writes <run_id>/scores.json; leaderboard aggregates across repeats/tasks.
echo "[score] runs=$RUNS_ROOT scores=$SCORES_ROOT"
usabench score \
  --runs "$RUNS_ROOT" \
  --out "$SCORES_ROOT"

echo "[score] leaderboard -> $LEADERBOARD_OUT"
usabench leaderboard \
  --scores "$SCORES_ROOT" \
  --out "$LEADERBOARD_OUT"

echo "[score] done. Sync back with: bash daic/sync/pull_results.sh"
