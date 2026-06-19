# usability-benchmark (import name: usabench) — developer Makefile.
#
# Every target is a REAL command wired to the actual `usabench` CLI (src-layout;
# install editable so `usabench`/`usabench-collect` resolve, or invoke via
# `python -m usabench.cli`). Heavy/optional backends live behind extras + lazy
# imports, so the default `install` target is CPU-only and never pulls vLLM/torch.
#
# The headline target is `make smoke`: the zero-network, zero-API ACCEPTANCE GATE
# that drives the FakeLLM agent + FakeLLM oracle through the WHOLE pipeline
# (run -> score -> leaderboard) on the v0_smoke tasks and MUST exit 0. CI runs it.
#
# Usage: `make <target>`. Override knobs on the command line, e.g.
#   make run CONFIG=configs/runs/smoke.yaml RUNS=_out/runs
#   make score RUNS=_out/runs TASKS=tasks/curated/v0_smoke.jsonl SCORES=_out/scores
#
# Notes:
#  * Targets run through PYTHON, which prefers the project venv if present so they
#    work whether or not the venv is "activated". On DAIC, create it first with
#    `uv venv --python 3.11 .venv` (see docs/infra.md / FROZEN decision #7).
#  * `serve-check` pings a live vLLM/OpenAI endpoint (URL via BASE_URL); it never
#    imports vLLM and is never exercised in CI.

# Prefer the venv interpreter when it exists; else fall back to system python3.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
PIP    ?= $(PYTHON) -m pip
CLI    := $(PYTHON) -m usabench.cli

# Overridable knobs (sane defaults pointing at checked-in artifacts).
CONFIG    ?= configs/runs/smoke.yaml
TASKS     ?= tasks/curated/v0_smoke.jsonl
RUNS      ?= _out/runs
SCORES    ?= _out/scores
SMOKE_DIR ?= _smoke
BASE_URL  ?= http://127.0.0.1:8000/v1

.DEFAULT_GOAL := help

.PHONY: help install lint typecheck test smoke smoke-import run score \
        leaderboard collect serve-check estimate validate-spec fmt clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with CPU extras (no vLLM/torch).
	$(PIP) install -U pip
	$(PIP) install -e ".[api,collect,dev]"

lint: ## Ruff lint (no autofix).
	$(PYTHON) -m ruff check src tests collect drafting qc

fmt: ## Ruff format + import-sort autofix.
	$(PYTHON) -m ruff check --fix src tests collect drafting qc
	$(PYTHON) -m ruff format src tests collect drafting qc

typecheck: ## Mypy (per pyproject).
	$(PYTHON) -m mypy src

test: ## Run the test suite, excluding slow/network/gpu markers.
	$(PYTHON) -m pytest -m "not slow and not network and not gpu"

# --- the acceptance gate ----------------------------------------------------- #
smoke: ## ACCEPTANCE GATE: FakeLLM end-to-end run->score->leaderboard; must exit 0.
	$(CLI) smoke --work-dir $(SMOKE_DIR)

smoke-import: ## Cheap import smoke: load the package + the frozen scoring spec.
	$(PYTHON) -c "import usabench; print('usabench', usabench.__version__)"
	$(CLI) version
	$(CLI) validate-spec

# --- pipeline stages (real CLI flags) ---------------------------------------- #
run: ## Run a batch from a run config into RUNS/.
	$(CLI) run --config $(CONFIG) --output-root $(RUNS)

score: ## Score completed runs (pure offline functions of trace.jsonl + gold).
	$(CLI) score --runs $(RUNS) --tasks $(TASKS) --scores $(SCORES)

leaderboard: ## Aggregate scored runs into leaderboard.{jsonl,md}.
	$(CLI) leaderboard --scores $(SCORES)

estimate: ## Dry token/cost estimate for a run config (no episodes run).
	$(CLI) estimate --config $(CONFIG)

collect: ## Run the GitHub harvest collector (needs GITHUB_TOKEN).
	$(CLI) collect run --config configs/harvest.yaml

serve-check: ## Ping a live vLLM/OpenAI endpoint (BASE_URL); never imports vLLM.
	$(CLI) serve-check --base-url $(BASE_URL)

validate-spec: ## Load + echo the frozen single-source-of-truth scoring spec.
	$(CLI) validate-spec

clean: ## Remove caches, build artifacts, and smoke/out dirs (keeps runs/ + tasks/).
	rm -rf build dist *.egg-info src/*.egg-info \
	  .pytest_cache .mypy_cache .ruff_cache $(SMOKE_DIR) _out
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
