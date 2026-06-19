The directory is writable, but per my instructions I must not write report `.md` files — I return the document as my final message. Here it is.

---

# `usability-benchmark` — Infrastructure & Repo-Engineering Specification

**Audience:** engineers building the repo today. **Scope:** repo layout, model execution (API + vLLM), DAIC execution, GitHub data-collection architecture, and the cross-cutting engineering (reproducibility, logging, CI, cost control, smoke path).
**Pinned baseline:** Python **3.11** everywhere (broadest wheel coverage for harness + vLLM + scientific stack), pip + `pyproject.toml` with a committed lockfile, conda env on DAIC. This document defers the **task schema** and **metric definitions** to the tasks-data and metrics experts; it specifies the *interfaces* (`Task`, `Interaction`, `RunResult`, scorer contract) those experts fill in, and notes every seam where their schema plugs in.

---

## 0. Design constraints this spec must satisfy

| Constraint (from project brief) | Engineering consequence |
|---|---|
| Open-ended, under-specified tasks grounded in real GitHub repos | Data-collection package produces task material as JSONL; harness consumes a `Task` object. Reference repo material is cached, not re-fetched per run. |
| Simulated-user oracle (LLM holds gold knowledge) | Oracle is a *separate process/role* behind the same uniform LLM client as agents. Oracle context (gold spec) is loaded from the task record and never exposed to the agent. |
| Count + classify every interaction; measure assistance amount & severity | The harness mediates **all** agent↔oracle traffic through a single `InteractionBus` that timestamps, types, and persists every message. Nothing reaches the oracle except through the bus. |
| Both API frontier models and open-weight vLLM models, oracle is API | One uniform OpenAI-style client interface; provider chosen by config. vLLM exposes an OpenAI-compatible server so the harness code path is identical. |
| Reproducible, automated, scalable; account for LLM stochasticity | Config hashing, run manifests, lockfiles, seeds, N-repeat runs with variance reporting, deterministic-where-possible sampling params recorded. |
| DAIC: login node has internet, compute nodes restricted; project storage `CoReFusion` | Collection on login/CPU job; model runs on GPU nodes read pre-cached data from project storage; results written to project storage and synced back. |

---

## 1. Repository structure

Repo: `github.com/D4vidHuang/usability-benchmark` (public). Top-level Python distribution name `usabench` (import as `usabench`). Monorepo: one installable package with sub-packages, plus non-code directories.

```
usability-benchmark/
├── README.md                         # What this is, quickstart, links to docs/.
├── LICENSE                           # MIT (permissive; we redistribute only metadata/links from public repos).
├── CITATION.cff                      # How to cite the benchmark.
├── pyproject.toml                    # Build (setuptools), deps, optional-extras [collect,serve,dev,daic], tool config.
├── requirements.lock                # Fully pinned resolved deps (pip-compile output) — the reproducible install.
├── requirements-dev.lock            # Dev/test/CI pins.
├── constraints.txt                  # Hard upper/lower bounds feeding pip-compile (e.g. vllm, torch, cuda).
├── Makefile                          # make install|lint|test|smoke|collect|score — single entry for humans & CI.
├── .env.example                      # Template for secrets (API keys, GH token). Real .env is gitignored.
├── .gitignore                        # Ignores .env, runs/, .cache/, *.lock backups, __pycache__, data/raw blobs.
├── .pre-commit-config.yaml           # ruff + black + mypy + end-of-file/trailing-whitespace hooks.
│
├── src/
│   └── usabench/                     # THE installable package.
│       ├── __init__.py               # Version (__version__), public re-exports.
│       │
│       ├── core/                     # Shared types & contracts. No I/O side effects.
│       │   ├── __init__.py
│       │   ├── schema.py             # Pydantic models: Task, ReferenceRepo, AcceptanceCriterion, Interaction,
│       │   │                         #   InteractionType, RunResult, Manifest. ← tasks-data expert fills field set.
│       │   ├── enums.py              # InteractionType (CLARIFY/HINT/HANDOFF/...), Severity, RunStatus, Provider.
│       │   ├── ids.py                # Deterministic id helpers (task_id, run_id = hash(config+task+seed)).
│       │   └── errors.py             # Typed exceptions (BudgetExceeded, OracleProtocolError, ProviderError).
│       │
│       ├── llm/                      # Uniform model access. EVERYTHING that talks to a model goes through here.
│       │   ├── __init__.py
│       │   ├── client.py             # LLMClient protocol: chat(messages, **params) -> Completion. Sync+async.
│       │   ├── openai_client.py      # OpenAI + vLLM (OpenAI-compatible) impl. base_url switch.
│       │   ├── anthropic_client.py   # Anthropic Messages API impl, normalized to the same Completion shape.
│       │   ├── factory.py            # build_client(model_cfg) -> LLMClient. Reads configs/models/*.yaml.
│       │   ├── retry.py              # tenacity-based retry/backoff; rate-limit (429) + 5xx handling; jitter.
│       │   ├── usage.py              # Token+cost accounting per call; rolls up into BudgetMeter.
│       │   └── cache.py              # Optional on-disk response cache (keyed by hash(model,messages,params)).
│       │
│       ├── agent/                    # The "agent under test" wrapper (system under test).
│       │   ├── __init__.py
│       │   ├── base.py               # Agent protocol: step()/run(task, tools, oracle_channel) -> Trajectory.
│       │   ├── scaffold.py           # Reference ReAct-style scaffold: tool loop in a sandboxed workspace.
│       │   ├── tools.py              # Agent tools: write_file, run_cmd (sandboxed), read_file, ask_user(...).
│       │   └── adapters/             # Optional adapters to external agent frameworks (kept thin, optional dep).
│       │       ├── __init__.py
│       │       └── raw_scaffold.py   # Default; no external agent dep, fully reproducible.
│       │
│       ├── oracle/                   # The simulated-user oracle (holds gold knowledge).
│       │   ├── __init__.py
│       │   ├── oracle.py             # Oracle: answer(interaction, gold_context) -> OracleResponse. LLM-backed.
│       │   ├── policy.py             # When/how much to reveal; refusal of out-of-scope asks; hint laddering.
│       │   ├── prompts/              # Jinja2 templates for oracle persona + gold-spec injection.
│       │   │   ├── system_user.j2    # "You are the non-expert user/maintainer..." persona.
│       │   │   └── grading_user.j2   # Oracle-as-judge prompt for acceptance checks (if LLM-judged criteria).
│       │   └── classifier.py         # Classifies each incoming agent message into InteractionType + severity.
│       │
│       ├── harness/                  # Orchestration: runs one (task, agent, oracle) episode end-to-end.
│       │   ├── __init__.py
│       │   ├── runner.py             # run_episode(task, agent_cfg, oracle_cfg, run_cfg) -> RunResult.
│       │   ├── interaction_bus.py    # The ONLY channel agent↔oracle. Logs/types/timestamps every message.
│       │   ├── sandbox.py            # Per-run isolated workspace (tmp dir / container); resource & wall limits.
│       │   ├── budget.py             # BudgetMeter: token/$/interaction/wallclock caps; raises BudgetExceeded.
│       │   ├── manifest.py           # Builds run manifest (config hash, lockfile hash, git sha, seeds, env).
│       │   └── batch.py              # Fan out N repeats × M tasks × K models; resumable; writes runs/.
│       │
│       ├── eval/                     # Scoring & metrics (consumes RunResult, produces scores). ← metrics expert.
│       │   ├── __init__.py
│       │   ├── acceptance.py         # Run acceptance criteria: programmatic checks + LLM-judged checks.
│       │   ├── checks/               # Pluggable check runners.
│       │   │   ├── __init__.py
│       │   │   ├── pytest_check.py   # Run a provided test suite in the sandbox; parse pass/fail.
│       │   │   ├── cli_check.py      # Invoke built CLI/app with fixtures; assert outputs.
│       │   │   └── rubric_check.py   # LLM-judge rubric scoring against gold acceptance criteria.
│       │   ├── intervention.py       # Aggregate interaction logs -> assistance amount & severity metrics.
│       │   ├── scorer.py             # score(run_result) -> Scorecard. Contract the metrics expert implements.
│       │   └── aggregate.py          # Across repeats: mean/std/CI; across tasks: leaderboard rows.
│       │
│       ├── report/                   # Turning scores into artifacts.
│       │   ├── __init__.py
│       │   ├── leaderboard.py        # Build leaderboard table (jsonl/csv) from aggregated scores.
│       │   └── trace_view.py         # Render a single episode's interaction trace to HTML for inspection.
│       │
│       ├── config/                   # Config loading + validation + hashing.
│       │   ├── __init__.py
│       │   ├── loader.py             # YAML -> typed config (pydantic); env interpolation ${VAR}.
│       │   └── hashing.py            # canonical_json + sha256 -> config_hash used in run_id.
│       │
│       ├── logging_setup.py          # structlog/JSON logging config; one logger, run_id-bound context.
│       └── cli.py                    # `usabench` Typer CLI: collect | run | score | leaderboard | smoke | serve-check.
│
├── collect/                          # DATA-COLLECTION package (separate concern; depends on usabench.core only).
│   ├── __init__.py
│   ├── github_client.py              # REST+GraphQL client; ETag caching, rate-limit (X-RateLimit) handling.
│   ├── sources/                      # One module per source of task material.
│   │   ├── __init__.py
│   │   ├── awesome_lists.py          # Parse "awesome-X" READMEs -> candidate project ideas.
│   │   ├── readmes.py                # Pull README + repo metadata for a project.
│   │   ├── issues.py                 # Feature requests / "good first issue" / enhancement issues.
│   │   └── topics.py                 # Search by topic/stars to seed candidate repos.
│   ├── normalize.py                  # Raw API payloads -> task-material records (tasks-data schema).
│   ├── filters.py                    # License/quality/activity/size filters; dedup; PII/secret scrub.
│   ├── pipeline.py                   # Orchestrates source -> normalize -> filter -> write JSONL.
│   ├── cache.py                      # On-disk HTTP cache (sqlite/diskcache) keyed by URL+ETag.
│   └── cli.py                        # `usabench-collect` CLI (also exposed via usabench cli).
│
├── tasks/                            # TASK DATA DIR (the benchmark content; tasks-data expert owns format).
│   ├── README.md                     # Schema doc + provenance + license notes for collected material.
│   ├── schema/
│   │   └── task.schema.json          # JSON Schema for one task record (validated in CI).
│   ├── raw/                          # Gitignored: raw collected JSONL from GitHub (large; lives on DAIC).
│   ├── curated/                      # Committed: vetted task set(s), small JSONL, references not blobs.
│   │   ├── v0_smoke.jsonl            # 2–3 tiny tasks for the smoke path.
│   │   └── v1.jsonl                  # First real task set.
│   └── references/                   # Cached reference repo snapshots (pointers/manifests; blobs on DAIC).
│
├── configs/                          # All run configuration (YAML). Hashed into run_ids.
│   ├── models/                       # One file per model-under-test.
│   │   ├── claude_opus.yaml
│   │   ├── gpt_frontier.yaml
│   │   └── qwen_vllm.yaml
│   ├── oracle/
│   │   └── oracle_default.yaml       # Oracle model + persona + reveal policy.
│   ├── agents/
│   │   └── scaffold_default.yaml     # Agent scaffold params (max steps, tools enabled, temperature).
│   ├── runs/
│   │   ├── smoke.yaml                # task_set=v0_smoke, repeats=1, tiny budget.
│   │   └── full_v1.yaml             # task_set=v1, repeats=5, full budget.
│   └── daic/
│       └── cluster.yaml              # Paths, partitions, GPU types, module names for DAIC.
│
├── daic/                             # DAIC execution scripts (sbatch + setup).
│   ├── README.md                     # Step-by-step: env setup, secrets, submit, sync back.
│   ├── env/
│   │   ├── environment.yml           # conda env (python=3.11, pip section installs the package + locks).
│   │   └── setup_env.sh              # Idempotent: module load, conda create/activate, pip install -e .[daic].
│   ├── secrets/
│   │   └── load_secrets.sh           # Sources ~/.config/usabench/secrets.env (chmod 600, NOT in repo).
│   ├── slurm/
│   │   ├── collect_cpu.sbatch        # CPU job: GitHub collection on login-adjacent path (internet OK).
│   │   ├── vllm_serve.sbatch         # GPU job: start vLLM OpenAI-compatible server, write endpoint file.
│   │   ├── run_api.sbatch            # CPU job: harness vs API models (login-node internet).
│   │   ├── run_vllm.sbatch           # GPU job: serve vLLM + run harness against local endpoint (one node).
│   │   └── score.sbatch              # CPU job: scoring + leaderboard aggregation.
│   └── sync/
│       └── pull_results.sh           # rsync runs/ + leaderboard back to local from project storage.
│
├── docs/
│   ├── architecture.md               # This doc's prose form; component diagram.
│   ├── data_collection.md            # How tasks are sourced; legal/license stance.
│   ├── running.md                    # Local + DAIC run instructions.
│   ├── metrics.md                    # ← metrics expert: assistance/severity definitions.
│   └── reproducibility.md            # Manifests, hashing, seeds, variance protocol.
│
├── tests/
│   ├── conftest.py                   # Fixtures: fake LLMClient (deterministic), tmp sandbox, sample Task.
│   ├── unit/                         # Pure-unit tests (no network): schema, hashing, bus, budget, retry.
│   │   ├── test_schema.py
│   │   ├── test_config_hashing.py
│   │   ├── test_interaction_bus.py
│   │   ├── test_budget.py
│   │   └── test_intervention_metrics.py
│   ├── integration/                  # Uses FakeLLM + local sandbox; no external APIs.
│   │   ├── test_run_episode_fake.py  # Full episode with deterministic fake agent+oracle.
│   │   └── test_collect_offline.py   # Collection against recorded HTTP fixtures (vcr/cassettes).
│   ├── cassettes/                    # Recorded GitHub API responses for offline collection tests.
│   └── data/                         # Tiny fixture tasks/repos.
│
├── examples/
│   ├── 01_run_smoke_local.md         # Copy-paste local smoke run.
│   ├── 02_add_a_model.md             # How to add configs/models/foo.yaml.
│   ├── 03_collect_tasks.md           # Run the collector with a token.
│   └── notebooks/
│       └── inspect_trace.ipynb       # Load a RunResult, render its interaction trace.
│
├── leaderboard/
│   ├── README.md                     # How numbers are produced + variance caveats.
│   ├── data/
│   │   └── v1_results.jsonl          # Committed aggregated results (one row per model×taskset).
│   └── site/                         # Static leaderboard (optional): index.html reads data/*.jsonl.
│
└── .github/
    ├── workflows/
    │   ├── ci.yml                    # lint + typecheck + unit + offline-integration on push/PR.
    │   ├── schema-validate.yml       # Validate tasks/curated/*.jsonl against task.schema.json.
    │   └── smoke-nightly.yml         # Optional: gated smoke run with FakeLLM (no paid calls) nightly.
    ├── ISSUE_TEMPLATE/
    │   ├── new_task.md
    │   └── bug_report.md
    └── dependabot.yml                # Weekly dep PRs against constraints.txt.
```

**Why this split.** `usabench` is the science (harness/oracle/eval). `collect/` is an independent ETL concern that only depends on `usabench.core` types, so collection can run on the login node with a minimal install (`pip install .[collect]`) without dragging in torch/vllm. `tasks/`, `configs/`, `leaderboard/` are data, not code, and are versioned with the repo so a `run_id` is reproducible from a git sha alone.

---

## 2. Python version & dependency strategy

**Python 3.11.** Reasons: stable wheels for torch/vllm/pydantic-v2/scientific stack on Linux+CUDA; available as an Lmod module and via conda on DAIC; new enough for `tomllib`, exception groups, and good asyncio. We do **not** require uv (absent on the Mac); pip is the universal tool.

**Layered dependencies in `pyproject.toml` via extras:**

- `core` (always): `pydantic>=2`, `pyyaml`, `typer`, `tenacity`, `structlog`, `jinja2`, `httpx`, `diskcache`.
- `[api]`: `openai>=1.40`, `anthropic>=0.40`. (Lets login/CPU nodes drive frontier models + oracle.)
- `[collect]`: `httpx`, `gql[httpx]` (GraphQL), `tenacity`, `python-dateutil`. (No ML deps.)
- `[serve]`: `vllm==<pinned>`, `torch==<pinned-for-cuda12.x>`. **DAIC GPU only** — never installed on the Mac.
- `[dev]`: `ruff`, `black`, `mypy`, `pytest`, `pytest-asyncio`, `pytest-recording`/`vcrpy`, `pip-tools`.
- `[daic]`: `core+api+collect+serve` (the full GPU-node install).

**Lockfiles.** `constraints.txt` holds the human-curated bounds (e.g. `vllm==0.x.y`, the matching `torch`, `cuda` ABI note). `pip-compile pyproject.toml --extra api --extra collect -o requirements.lock` produces the fully-pinned reproducible set the harness installs on CPU/login. A second compile with `--extra serve` produces `requirements-serve.lock` for GPU nodes (kept separate so Mac/CI never resolve torch/vllm). CI installs from the lock, not the ranges.

**Pin note (verify at build time):** vLLM moves fast and is the one fragile pin. Target a **single recent stable vLLM release** and the exact `torch`/CUDA combo its wheels were built against, matching DAIC's `CUDA 12.9` module. Record the chosen versions in `constraints.txt` with a comment linking the vLLM release notes, and let `make serve-check` assert the running server's `/version`. Latest stable vLLM at time of writing is the `0.2x` line and supports Python 3.10–3.12 ([vLLM stable docs](https://docs.vllm.ai/en/stable/), [vLLM releases](https://vllm.ai/releases)); pin the exact patch you validate on DAIC rather than tracking `latest`.

**`pyproject.toml` skeleton:**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "usabench"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
  "pydantic>=2.6", "pyyaml>=6", "typer>=0.12", "tenacity>=8.2",
  "structlog>=24", "jinja2>=3.1", "httpx>=0.27", "diskcache>=5.6",
]

[project.optional-dependencies]
api     = ["openai>=1.40", "anthropic>=0.40"]
collect = ["gql[httpx]>=3.5", "python-dateutil>=2.9"]
serve   = ["vllm==0.0.0", "torch==0.0.0"]   # <- pin exact, set in constraints.txt + CI matrix
dev     = ["ruff>=0.5", "black>=24", "mypy>=1.10", "pytest>=8", "pytest-asyncio>=0.23",
           "vcrpy>=6", "pip-tools>=7.4"]
daic    = ["usabench[api,collect,serve]"]

[project.scripts]
usabench         = "usabench.cli:app"
usabench-collect = "collect.cli:app"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
plugins = ["pydantic.mypy"]

[tool.pytest.ini_options]
addopts = "-q --strict-markers"
markers = ["network: requires external API (skipped in CI)"]
```

---

## 3. How models run (uniform client; API + vLLM)

**Single interface.** Everything that calls a model — agent-under-test, oracle, and LLM-judge checks — goes through `usabench.llm.client.LLMClient`:

```python
class Completion(BaseModel):
    text: str
    tool_calls: list[ToolCall] | None
    usage: Usage          # prompt/completion tokens
    raw: dict             # provider raw for debugging
    model: str

class LLMClient(Protocol):
    async def chat(self, messages: list[Msg], *,
                   temperature: float, max_tokens: int,
                   tools: list[ToolSpec] | None = None,
                   seed: int | None = None,
                   stop: list[str] | None = None) -> Completion: ...
```

Two concrete impls, selected by `factory.build_client(model_cfg)`:

- **`openai_client.py`** serves **both** OpenAI frontier models and **vLLM** open-weight models, because vLLM exposes an OpenAI-compatible `/v1/chat/completions`. The only difference is `base_url` and `api_key` (vLLM uses a dummy/local key). This is the key uniformity decision: the harness has *one* code path for "OpenAI-shaped" backends.
- **`anthropic_client.py`** wraps the Anthropic Messages API and normalizes responses into the same `Completion` shape.

**Reliability (`retry.py`, `usage.py`).** Every call is wrapped with tenacity: exponential backoff + jitter on `429`/`5xx`/timeouts, capped retries, and respect of `Retry-After`. `usage.py` records tokens and computes cost from a per-model price table in the model config; `BudgetMeter` aborts the run with `BudgetExceeded` if the configured `$`/token/wallclock cap is hit. A deterministic on-disk `cache.py` (keyed by `hash(model, messages, params, seed)`) can be enabled for debugging/CI so repeated identical calls don't cost money — disabled for real measurement runs (caching would distort variance).

**Secrets.** Never committed. Loaded from environment, populated locally from `.env` (gitignored) and on DAIC from `~/.config/usabench/secrets.env` (chmod 600). `.env.example`:

```bash
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GITHUB_TOKEN=               # repo+workflow scope, for collection
USABENCH_VLLM_BASE_URL=     # filled in at runtime by vllm_serve.sbatch
USABENCH_VLLM_API_KEY=local-dummy
```

**Example model configs (`configs/models/`):**

```yaml
# claude_opus.yaml — API frontier model under test
id: claude-opus
provider: anthropic
model: claude-opus-4-8
api_key_env: ANTHROPIC_API_KEY
params: { temperature: 0.7, max_tokens: 4096 }
price_per_mtok: { input: 15.0, output: 75.0 }   # used by BudgetMeter/cost reporting
retry: { max_attempts: 6, base_delay_s: 1.5, max_delay_s: 60 }
```

```yaml
# gpt_frontier.yaml — OpenAI API model under test
id: gpt-frontier
provider: openai
model: gpt-frontier-latest
base_url: https://api.openai.com/v1
api_key_env: OPENAI_API_KEY
params: { temperature: 0.7, max_tokens: 4096, seed: 7 }
price_per_mtok: { input: 5.0, output: 15.0 }
retry: { max_attempts: 6, base_delay_s: 1.5, max_delay_s: 60 }
```

```yaml
# qwen_vllm.yaml — open-weight model served by vLLM on DAIC (OpenAI-compatible)
id: qwen-vllm
provider: openai                         # SAME client path as OpenAI; only base_url differs
model: Qwen/Qwen2.5-Coder-32B-Instruct   # served name (must match vLLM --served-model-name)
base_url_env: USABENCH_VLLM_BASE_URL     # written by vllm_serve.sbatch into the run env
api_key_env: USABENCH_VLLM_API_KEY       # dummy local key
params: { temperature: 0.7, max_tokens: 4096, seed: 7 }
price_per_mtok: { input: 0.0, output: 0.0 }   # local; cost tracked as GPU-hours, not $
retry: { max_attempts: 4, base_delay_s: 2.0, max_delay_s: 30 }
serving:                                  # consumed by vllm_serve.sbatch
  hf_model: Qwen/Qwen2.5-Coder-32B-Instruct
  tensor_parallel_size: 2                 # e.g. 2× a40
  max_model_len: 16384
  gpu: a40
  num_gpus: 2
```

```yaml
# configs/oracle/oracle_default.yaml — the simulated user (always an API model for consistency)
id: oracle-default
client:
  provider: anthropic
  model: claude-opus-4-8
  api_key_env: ANTHROPIC_API_KEY
  params: { temperature: 0.3, max_tokens: 1024 }
persona_template: oracle/prompts/system_user.j2
reveal_policy:                            # severity ladder; metrics expert refines the taxonomy
  max_clarifications: 8
  max_direct_hints: 3
  allow_handoff: true
  refuse_out_of_scope: true
```

```yaml
# configs/agents/scaffold_default.yaml
id: scaffold-default
type: raw_scaffold
max_steps: 40
tools: [write_file, read_file, run_cmd, ask_user]
sandbox: { backend: tmpdir, wall_clock_s: 1800, max_cmd_s: 120, network: false }
ask_user_is_interaction: true             # every ask_user routes through InteractionBus -> oracle
```

```yaml
# configs/runs/full_v1.yaml — a run plan
task_set: tasks/curated/v1.jsonl
models: [configs/models/claude_opus.yaml, configs/models/gpt_frontier.yaml, configs/models/qwen_vllm.yaml]
oracle: configs/oracle/oracle_default.yaml
agent:  configs/agents/scaffold_default.yaml
repeats: 5                                # variance: N independent episodes per (task,model)
seeds: [1, 2, 3, 4, 5]
budget: { usd_per_episode: 3.0, tokens_per_episode: 400000, wallclock_s_per_episode: 1800 }
output_root: ${USABENCH_RUNS_ROOT}        # /tudelft.net/staff-umbrella/CoReFusion/runs on DAIC
```

---

## 4. DAIC execution plan

### 4.1 Storage layout (`/tudelft.net/staff-umbrella/CoReFusion`)

Home (`/trinity/home/yongchenghuang`, ~47G) is too small for models/runs; everything heavy lives on project storage. Set once in a sourced `daic/env/paths.sh`:

```
/tudelft.net/staff-umbrella/CoReFusion/usabench/
├── data/            # tasks/raw + references snapshots (collector output; large JSONL + repo tarballs)
├── runs/            # per-run output: <run_id>/{manifest.json, interactions.jsonl, trajectory.jsonl, workspace/, scores.json}
├── cache/           # HTTP cache (collection) + optional LLM response cache + HF model snapshots cache
│   ├── http/
│   ├── llm/
│   └── hf/          # HF_HOME points here so models download once, shared across nodes
├── envs/            # conda env prefix (so the env isn't in the tiny home dir)
│   └── usabench/
├── models/          # optional explicit vLLM model snapshots (if not using hf cache)
└── logs/            # slurm-%j.out collected here
```

Environment contract (sourced by every sbatch):

```bash
export USABENCH_PROJ=/tudelft.net/staff-umbrella/CoReFusion/usabench
export USABENCH_DATA_ROOT=$USABENCH_PROJ/data
export USABENCH_RUNS_ROOT=$USABENCH_PROJ/runs
export HF_HOME=$USABENCH_PROJ/cache/hf
export USABENCH_HTTP_CACHE=$USABENCH_PROJ/cache/http
export CONDA_ENVS_PATH=$USABENCH_PROJ/envs
```

### 4.2 Environment setup (`daic/env/setup_env.sh`)

Idempotent; safe to re-run. Conda+pip (no uv on cluster).

```bash
#!/usr/bin/env bash
set -euo pipefail
source /tudelft.net/staff-umbrella/CoReFusion/usabench/../../usabench/daic/env/paths.sh 2>/dev/null || \
  source "$(dirname "$0")/paths.sh"

module purge
module load miniconda/3            # or the cluster's conda module name
module load cuda/12.9              # matches the pinned torch/vllm CUDA ABI

ENV_PREFIX="$CONDA_ENVS_PATH/usabench"
if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.11
fi
source activate "$ENV_PREFIX"

# Reproducible install from the committed lockfiles.
pip install --upgrade pip
# GPU nodes: serve lock (vllm+torch). CPU/login: api+collect lock.
if [ "${USABENCH_GPU:-0}" = "1" ]; then
  pip install -r requirements-serve.lock
fi
pip install -r requirements.lock
pip install -e .            # editable install of the package (no deps re-resolve; locks already applied)

python -c "import usabench, sys; print('usabench', usabench.__version__, 'py', sys.version)"
```

### 4.3 Secrets on the cluster

Not in the repo, not echoed in logs. Create once:

```bash
mkdir -p ~/.config/usabench
cat > ~/.config/usabench/secrets.env <<'EOF'
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GITHUB_TOKEN=ghp_...
EOF
chmod 600 ~/.config/usabench/secrets.env
```

`daic/secrets/load_secrets.sh` simply `source`s that file. Every sbatch sources it after `setup_env.sh`. API model runs (and collection) need internet, so they go on the **login node or a CPU job**; only login + designated nodes reach the public internet.

### 4.4 SLURM templates

**CPU — data collection (`daic/slurm/collect_cpu.sbatch`).** Runs the GitHub collector where internet is available.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=usabench-collect
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/collect-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/daic/env/setup_env.sh"
source "$SLURM_SUBMIT_DIR/daic/secrets/load_secrets.sh"

usabench-collect run \
  --config "$SLURM_SUBMIT_DIR/configs/runs/full_v1.yaml" \
  --out    "$USABENCH_DATA_ROOT/raw/v1.jsonl" \
  --cache  "$USABENCH_HTTP_CACHE" \
  --max-repos 500 --min-stars 50
```

> Note: if compute nodes are firewalled from GitHub, run this directly on the login node (`bash daic/slurm/collect_cpu.sbatch`-equivalent, no sbatch) — the collector is light and caches, so login-node execution is acceptable.

**GPU — vLLM serve + harness on one node (`daic/slurm/run_vllm.sbatch`).** Co-locates the OpenAI-compatible server and the harness so the harness needs no external network for the model-under-test (oracle still needs API internet — see note).

```bash
#!/usr/bin/env bash
#SBATCH --job-name=usabench-vllm
#SBATCH --partition=ewi-insy
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/vllm-%j.out

set -euo pipefail
export USABENCH_GPU=1
source "$SLURM_SUBMIT_DIR/daic/env/setup_env.sh"
source "$SLURM_SUBMIT_DIR/daic/secrets/load_secrets.sh"

MODEL_CFG="$SLURM_SUBMIT_DIR/configs/models/qwen_vllm.yaml"
PORT=$(( 8000 + RANDOM % 1000 ))
SERVED_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"

# 1) Start vLLM OpenAI-compatible server in the background, bound to localhost.
python -m vllm.entrypoints.openai.api_server \
  --model "$SERVED_NAME" --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 2 --max-model-len 16384 \
  --host 127.0.0.1 --port "$PORT" --api-key local-dummy \
  --download-dir "$HF_HOME" &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# 2) Wait until the endpoint is healthy.
export USABENCH_VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export USABENCH_VLLM_API_KEY=local-dummy
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then echo "vLLM up"; break; fi
  sleep 10
  [ "$i" -eq 120 ] && { echo "vLLM failed to start"; exit 1; }
done

# 3) Run the harness against the local endpoint.
usabench run \
  --config "$SLURM_SUBMIT_DIR/configs/runs/full_v1.yaml" \
  --only-model "$MODEL_CFG" \
  --output-root "$USABENCH_RUNS_ROOT"
```

> **Oracle on a firewalled GPU node:** the oracle is an API model and needs internet. If GPU nodes cannot reach the Anthropic/OpenAI API, run in **split mode**: GPU node serves vLLM and exposes the port; the harness (with the oracle) runs on the login/CPU node and connects to the GPU node's `http://<gpu-node-hostname>:<port>/v1`. `vllm_serve.sbatch` (the serve-only variant) writes the reachable `base_url` to `$USABENCH_RUNS_ROOT/<job>/endpoint.txt`; `run_api.sbatch`/login-node `usabench run` reads it. This is why the model config takes `base_url_env` — the same config works co-located or split.

**CPU — harness vs API models (`daic/slurm/run_api.sbatch`).** No GPU; needs internet (login-adjacent).

```bash
#!/usr/bin/env bash
#SBATCH --job-name=usabench-api
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/api-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/daic/env/setup_env.sh"
source "$SLURM_SUBMIT_DIR/daic/secrets/load_secrets.sh"
usabench run \
  --config "$SLURM_SUBMIT_DIR/configs/runs/full_v1.yaml" \
  --skip-model configs/models/qwen_vllm.yaml \
  --output-root "$USABENCH_RUNS_ROOT"
```

**CPU — scoring (`daic/slurm/score.sbatch`).**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=usabench-score
#SBATCH --partition=all
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/CoReFusion/usabench/logs/score-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/daic/env/setup_env.sh"
source "$SLURM_SUBMIT_DIR/daic/secrets/load_secrets.sh"   # LLM-judge checks may need API
usabench score --runs "$USABENCH_RUNS_ROOT" --out "$USABENCH_RUNS_ROOT/scores"
usabench leaderboard --scores "$USABENCH_RUNS_ROOT/scores" --out "$SLURM_SUBMIT_DIR/leaderboard/data/v1_results.jsonl"
```

### 4.5 Results sync back (`daic/sync/pull_results.sh`, run from the Mac)

```bash
#!/usr/bin/env bash
set -euo pipefail
REMOTE=daic:/tudelft.net/staff-umbrella/CoReFusion/usabench
LOCAL=${1:-./_daic_results}
mkdir -p "$LOCAL"
# Pull lightweight artifacts only (scores, manifests, interaction logs) — not full workspaces.
rsync -avz --prune-empty-dirs \
  --include='*/' \
  --include='manifest.json' --include='scores.json' \
  --include='interactions.jsonl' --include='trajectory.jsonl' \
  --exclude='workspace/**' --exclude='*' \
  "$REMOTE/runs/" "$LOCAL/runs/"
rsync -avz "$REMOTE/runs/scores/" "$LOCAL/scores/"
```

---

## 5. GitHub data-collection architecture

Aligned with the tasks-data expert's schema: the collector's job is to emit **task-material records** (one JSON object per line) that the tasks-data curation step turns into final `Task` records. The collector does **not** invent acceptance criteria; it gathers grounded raw material (README, feature requests, project ideas, license, activity signals) and references.

**Modules** (in `collect/`):

- `github_client.py` — thin wrapper over REST + GraphQL via `httpx`/`gql`. Sends `If-None-Match`/ETag, reads `X-RateLimit-Remaining`/`X-RateLimit-Reset`, sleeps until reset when exhausted, exponential backoff on `403 secondary rate limit`. Honors the `GITHUB_TOKEN` (repo+workflow scope). Conditional requests + on-disk cache mean re-runs are cheap and mostly free against the rate budget.
- `sources/` — one module per material type: `awesome_lists` (parse curated lists into candidate ideas), `readmes` (repo metadata + README), `issues` (enhancement/feature-request/good-first-issue), `topics` (discover candidates by topic/stars). Each yields raw provider payloads.
- `normalize.py` — maps raw payloads → the tasks-data record shape (a thin, schema-validated dict). Records provenance: `repo_full_name`, `commit_sha` pinned, `source_url`, `license`, `fetched_at`.
- `filters.py` — quality/activity/size/license gates; dedup by repo; a secret/PII scrub pass (regex for keys/emails) before anything is written, since output is public.
- `cache.py` — `diskcache`/sqlite keyed by `URL+ETag` under `$USABENCH_HTTP_CACHE`.
- `pipeline.py` — wires source → normalize → filter → JSONL writer; resumable (skips repos already in the output by `repo_full_name@sha`).
- `cli.py` — `usabench-collect run|list-sources|validate`.

**CLI contract:**

```
usabench-collect run \
  --config configs/runs/full_v1.yaml \
  --out    $USABENCH_DATA_ROOT/raw/v1.jsonl \
  --cache  $USABENCH_HTTP_CACHE \
  --sources awesome_lists,readmes,issues \
  --min-stars 50 --max-repos 500 \
  --license-allow MIT,Apache-2.0,BSD-3-Clause
```

**Output** is JSONL; each line validates against `tasks/schema/task.schema.json` (the tasks-data expert owns the exact fields). Minimum provenance fields the infra layer guarantees for every record: `id`, `repo_full_name`, `commit_sha`, `source_url`, `license`, `source_type`, `fetched_at`, plus a `raw` blob for downstream curation. Rate-limit and caching ensure a full v1 collection fits comfortably under the authenticated 5000 req/hr budget; the collector logs requests-used and a cost-free dry-run mode (`--dry-run`) prints the request plan.

---

## 6. Cross-cutting engineering

### 6.1 Reproducibility

- **`run_id = sha256(canonical_json(config) + task_id + seed + git_sha)`** (`config/hashing.py`, `core/ids.py`). Same inputs → same id → idempotent, resumable batches.
- **Run manifest** (`harness/manifest.py`) written to `runs/<run_id>/manifest.json`: git sha (dirty flag), `requirements.lock` hash, resolved config (with secrets redacted), seeds, model + sampling params actually sent, package `__version__`, hostname/SLURM job id, vLLM server `/version` if used, start/end timestamps, total tokens + cost + interaction count.
- **Lockfiles** (`requirements.lock`, `requirements-serve.lock`) are the source of truth for installs; ranges in `pyproject.toml` are for resolution only. CI installs from locks.
- **Variance protocol** (LLM stochasticity): every `(task, model)` is run `repeats` times with distinct seeds; `eval/aggregate.py` reports mean ± std and a bootstrap CI per metric. Sampling params (temperature, seed) are recorded in the manifest so a "deterministic-ish" config (low temp + fixed seed, where the provider honors seeds) is reproducible and a "stochastic" config is properly characterized.
- **Caching is OFF for measurement runs** and only enabled (`USABENCH_LLM_CACHE=1`) for debugging/CI to avoid distorting variance or cost.

### 6.2 Logging

`logging_setup.py` configures `structlog` to emit JSON lines bound to `run_id`. Three artifacts per episode under `runs/<run_id>/`: `trajectory.jsonl` (agent steps + tool calls), `interactions.jsonl` (every agent↔oracle message with `type`, `severity`, `ts`, token cost — the spine of the assistance metrics), and `manifest.json`. Secrets are filtered by a structlog processor (regex redaction of key-shaped tokens). SLURM stdout goes to `logs/<job>-%j.out`.

### 6.3 Testing & CI

- **Unit** (no network): schema validation, config hashing determinism, `InteractionBus` typing/counting, `BudgetMeter` cutoffs, retry/backoff logic, intervention-metric aggregation.
- **Integration with `FakeLLMClient`** (deterministic, scripted responses): a full `run_episode` with a fake agent + fake oracle exercises the whole harness with zero API cost and is fully reproducible. Collection integration uses `vcrpy` cassettes (recorded GitHub responses) so CI never hits the network.
- **`.github/workflows/ci.yml`**: matrix on py3.11; `ruff` + `black --check` + `mypy --strict` + `pytest tests/unit tests/integration` (network-marked tests skipped). Installs from `requirements-dev.lock`. `serve` extra (vllm/torch) is **never** installed in CI — it's GPU-only.
- **`schema-validate.yml`**: validates all `tasks/curated/*.jsonl` against `task.schema.json` on every PR touching tasks.
- **`dependabot.yml`** + weekly recompile keeps locks fresh with human review.

### 6.4 Cost controls / budgets

`BudgetMeter` enforces per-episode caps (`usd`, `tokens`, `wallclock`) from the run config and aborts cleanly with a recorded `BudgetExceeded` status (the episode still scores, marked truncated). `usabench run --estimate` does a dry pass computing expected cost from per-model `price_per_mtok` × projected tokens before spending anything. A global `--max-usd` halts a batch. vLLM-served models have `$0` token cost; their "budget" is GPU wall-time enforced by SLURM `--time`.

### 6.5 Minimal end-to-end smoke path

`make smoke` (and `usabench smoke`) runs the entire pipeline with **zero paid API calls** using `FakeLLMClient` for agent + oracle on `tasks/curated/v0_smoke.jsonl`:

```
usabench run   --config configs/runs/smoke.yaml --fake-llm --output-root ./_smoke/runs
usabench score --runs ./_smoke/runs --out ./_smoke/scores
usabench leaderboard --scores ./_smoke/scores --out ./_smoke/leaderboard.jsonl
```

This validates schema → harness → interaction bus → sandbox → acceptance checks → scoring → aggregation → leaderboard on a laptop in seconds, and is the gate `smoke-nightly.yml` runs. A second, **real** smoke (`make smoke-real`, opt-in, costs a few cents) runs one tiny task against one API model + the API oracle to validate the live client/retry/budget paths and the vLLM `serve-check` (`usabench serve-check --base-url $USABENCH_VLLM_BASE_URL`).

---

## 7. Seams left for the other experts

- **tasks-data expert** owns `core/schema.py` field definitions, `tasks/schema/task.schema.json`, and the `normalize.py` mapping. Infra guarantees the provenance fields and JSONL contract in §5.
- **metrics expert** owns `eval/intervention.py`, `eval/scorer.py`, and the `Severity`/`InteractionType` taxonomy in `core/enums.py`. Infra guarantees that every agent↔oracle message is captured, typed, timestamped, and cost-attributed via the `InteractionBus`, and provides `eval/aggregate.py` variance machinery.

---

### Relevant file paths (all under `/Users/davidhuang/Desktop/usabilityBenchMark/`, currently empty — greenfield)

The repo is to be created at the project root. Top-level entry points to scaffold first: `pyproject.toml`, `requirements.lock`, `Makefile`, `src/usabench/cli.py`, `collect/cli.py`, `daic/env/setup_env.sh`, `daic/slurm/run_vllm.sbatch`, `configs/runs/smoke.yaml`, `tests/integration/test_run_episode_fake.py`.

**Sources:** [vLLM stable docs](https://docs.vllm.ai/en/stable/), [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/), [vLLM releases](https://vllm.ai/releases), [Running vLLM on SLURM clusters](https://veldaio.substack.com/p/running-vllm-on-slurm-clusters-a), [SURF: LLM inference on Snellius with vLLM](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/232851290/LLM+inference+on+Snellius+with+vLLM).