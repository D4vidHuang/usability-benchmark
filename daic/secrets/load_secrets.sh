# shellcheck shell=bash
# daic/secrets/load_secrets.sh -- load API keys into the environment.
#
# SOURCE this (never execute / never commit keys). It reads a private file that
# lives OUTSIDE the repo:
#
#     ~/.config/usabench/secrets.env        (chmod 600, NOT in git)
#
# Create it ONCE on DAIC:
#
#     mkdir -p ~/.config/usabench
#     cat > ~/.config/usabench/secrets.env <<'EOF'
#     export ANTHROPIC_API_KEY=sk-ant-...
#     export OPENAI_API_KEY=sk-...
#     export GITHUB_TOKEN=ghp_...            # public-read PAT for collection
#     # Split-mode only: where the GPU node's vLLM server is reachable.
#     # export USABENCH_VLLM_BASE_URL=http://<gpu-node>:8000/v1
#     # export USABENCH_VLLM_API_KEY=local-dummy
#     EOF
#     chmod 600 ~/.config/usabench/secrets.env
#
# The sbatch scripts source this AFTER setup_env.sh. Secrets are never echoed:
# logging_setup.py redacts key-shaped tokens, but we still avoid printing them.

_USABENCH_SECRETS_FILE="${USABENCH_SECRETS_FILE:-$HOME/.config/usabench/secrets.env}"

if [ ! -f "$_USABENCH_SECRETS_FILE" ]; then
  echo "[load_secrets] WARNING: $_USABENCH_SECRETS_FILE not found." >&2
  echo "[load_secrets] Create it (chmod 600) with ANTHROPIC_API_KEY / OPENAI_API_KEY / GITHUB_TOKEN." >&2
  echo "[load_secrets] See the header of this script for the template." >&2
else
  # Enforce private permissions; refuse a world/group-readable secrets file.
  _perm="$(stat -c '%a' "$_USABENCH_SECRETS_FILE" 2>/dev/null || stat -f '%Lp' "$_USABENCH_SECRETS_FILE" 2>/dev/null || echo '')"
  case "$_perm" in
    600|400|640|440|"") : ;;  # acceptable / unknown-stat: proceed
    *)
      echo "[load_secrets] WARNING: $_USABENCH_SECRETS_FILE has mode $_perm; expected 600. Run: chmod 600 $_USABENCH_SECRETS_FILE" >&2
      ;;
  esac
  # shellcheck disable=SC1090
  set -a
  source "$_USABENCH_SECRETS_FILE"
  set +a
  unset _perm
fi

# Report which keys are present WITHOUT printing their values.
for _k in ANTHROPIC_API_KEY OPENAI_API_KEY GITHUB_TOKEN USABENCH_VLLM_BASE_URL; do
  if [ -n "${!_k:-}" ]; then
    echo "[load_secrets] $_k: set" >&2
  else
    echo "[load_secrets] $_k: (unset)" >&2
  fi
done
unset _k _USABENCH_SECRETS_FILE
