#!/usr/bin/env bash
# daic/sync/pull_results.sh -- rsync lightweight results back from DAIC to the Mac.
#
# Run this LOCALLY (on your laptop), not on the cluster. It pulls the small,
# scientifically-relevant artifacts -- run manifests, scores, traces, and the
# leaderboard -- while EXCLUDING the heavy per-run agent workspaces and any HF
# model blobs. The SSH alias is `daic`.
#
# Usage:
#     bash daic/sync/pull_results.sh [LOCAL_DEST]
#       LOCAL_DEST defaults to ./_daic_results
#
# Override the remote root / ssh alias if needed:
#     USABENCH_SSH=daic \
#     USABENCH_PROJ_REMOTE=/tudelft.net/staff-umbrella/CoReFusion/usabench \
#     bash daic/sync/pull_results.sh ~/usabench-results

set -euo pipefail

SSH_ALIAS="${USABENCH_SSH:-daic}"
PROJ_REMOTE="${USABENCH_PROJ_REMOTE:-/tudelft.net/staff-umbrella/CoReFusion/usabench}"
LOCAL_DEST="${1:-./_daic_results}"

REMOTE="${SSH_ALIAS}:${PROJ_REMOTE}"
mkdir -p "$LOCAL_DEST/runs" "$LOCAL_DEST/scores" "$LOCAL_DEST/logs"

echo "[pull] from $REMOTE -> $LOCAL_DEST"

# 1) Runs: keep manifest/scores/trace + verification artifacts; DROP the heavy
#    agent workspace/ trees and any model/dep blobs.
#    The trailing `--exclude='*'` with `--include` whitelist keeps the pull tiny.
rsync -avz --prune-empty-dirs \
  --include='*/' \
  --include='manifest.json' \
  --include='scores.json' \
  --include='trace.jsonl' \
  --include='run_result.json' \
  --include='verification/**' \
  --exclude='workspace/**' \
  --exclude='*.sif' \
  --exclude='*' \
  "$REMOTE/runs/" "$LOCAL_DEST/runs/"

# 2) Aggregated scores directory (leaderboard inputs).
rsync -avz --prune-empty-dirs \
  "$REMOTE/scores/" "$LOCAL_DEST/scores/" 2>/dev/null || \
  echo "[pull] (no scores/ yet -- run score.sbatch first)"

# 3) SLURM logs (small, useful for debugging).
rsync -avz --prune-empty-dirs \
  --include='*/' --include='*.out' --include='*.err' --exclude='*' \
  "$REMOTE/logs/" "$LOCAL_DEST/logs/" 2>/dev/null || true

echo "[pull] done. Local results under: $LOCAL_DEST"
echo "[pull] (agent workspaces and *.sif images were intentionally excluded.)"
