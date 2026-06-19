# usability-benchmark

**How *usable* is an AI coding agent when you hand it an open-ended, vaguely-phrased development goal — and how much hand-holding does it need to get there?**

`usability-benchmark` is a reproducible benchmark for AI coding agents that deliberately departs from SWE-bench-style "fix this bug, pass these hidden tests." Instead it gives an agent a goal the way a non-expert actually phrases it — *"build me a tool that analyzes my calendar and tells me where my time goes"* — grounded in a real open-source project so a defensible gold intent exists. An **LLM simulated-user oracle** holds that gold intent and answers the agent's clarifying questions, hints, and hand-offs. The harness **counts and severity-grades every interaction**, and the headline score rewards *building the right thing with minimal human intervention*: an agent that only succeeds by extracting constant takeovers scores no better than one that never asks and ships the wrong thing.

## Why it's different

| | SWE-bench & friends | usability-benchmark |
|---|---|---|
| Task | closed, fully-specified bug-fix | open-ended, under-specified, lay-phrased dev goal |
| Grounding | one repo + failing tests | real OSS repo → gold intent + acceptance criteria |
| Human in loop | none (single-shot) | **simulated-user oracle** holding gold intent |
| Headline metric | resolved rate | **intervention amount + severity** (+ pass^k for variance) |

We borrow proven mechanisms with attribution: simulated-user oracle and `pass^k` (τ-bench), interaction-as-budget (MINT), anti-spam ask-scoring and progressive hidden blockers (HiL-Bench), oracle-answered clarification (Ambig-SWE/ClarEval), hierarchical-requirement grading (DevAI), behavior-based verification (AppWorld/WebArena), multi-channel rubric (ScienceAgentBench), and the intervention taxonomy (Anthropic autonomy). See `docs/related-work.md`.

## Quickstart

```bash
# 1. Install (CPU / laptop: no torch/vllm)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
pip install -e .[api]                 # add ,collect for the GitHub harvester

# 2. Zero-cost end-to-end smoke (FakeLLM agent + oracle, tiny tasks)
make smoke                            # run → score → leaderboard, no API calls

# 3. Configure secrets for real runs
cp .env.example .env                  # fill ANTHROPIC_API_KEY / OPENAI_API_KEY / GITHUB_TOKEN

# 4. Collect raw task material from public GitHub (login node / laptop)
usabench-collect run --config configs/runs/full_v1.yaml \
  --out tasks/raw/v1.jsonl --min-stars 50 --license-allow MIT,Apache-2.0,BSD-3-Clause

# 5. Run one task against one agent, with the oracle live
usabench run --config configs/runs/smoke.yaml --output-root ./runs

# 6. Score traces and build the leaderboard
usabench score --runs ./runs --out ./runs/scores
usabench leaderboard --scores ./runs/scores --out leaderboard/data/v1_results.jsonl
```

On the TU Delft DAIC cluster, use the SLURM templates in `daic/slurm/` (CPU job for API agents + collection, GPU job for vLLM-served open-weight agents, split-mode when GPU nodes are firewalled from model APIs). See `docs/infra.md` and `daic/README.md`.

## How a run works

1. **A task** (`tasks/<id>/task.json`) carries an agent-visible `user_goal` plus oracle-private gold (`hidden_spec`, weighted `acceptance_criteria`, `ambiguity_points`, `verification`). The agent never sees the gold.
2. **The harness** runs one `(task, agent, seed)` episode, routing *all* agent↔oracle traffic through the `InteractionBus`, sandboxing file/exec actions, enforcing budgets, and writing the canonical **`trace.jsonl`** (`schemas/trace.schema.json`).
3. **Scoring** turns the delivered artifact into **Goal Achievement (GA)** via three channels — V1 deterministic execution, V2 frozen rubric, V3 bias-controlled judge jury — and computes the **assistance cost** from the severity-graded interactions.
4. **The Usability Score** geometrically couples *success* and *assistance-lightness* so you only get credit for goals reached without leaning on the human, reported as **median ± IQR over N≥5 seeds** with `pass^k` reliability.

## Repo layout

```
src/usabench/   harness, oracle, uniform LLM client, scoring + metric registry, reporting
collect/        GitHub REST/GraphQL harvester (depends only on usabench.core)
tasks/          benchmark content: schema + curated task sets + per-task gold/env/grader
configs/        models / oracle / agents / runs / daic  (all hashed into run_id)
daic/           SLURM templates, env setup, secrets loader, results sync
schemas/        trace.schema.json (canonical), task.schema.json
docs/           six per-area design docs + DESIGN.md (north star)
tests/          unit + FakeLLM integration + golden scoring fixtures
```

## Status

Design-frozen v0.1; implementation in phases P0→P5 (see `DESIGN.md` §8). The benchmark is built to be **refreshable** — re-harvested on newer repos each cycle to outrun training-data contamination — and ships a private held-out split for the canonical leaderboard.

## Reproducibility

Python 3.11, committed lockfiles, content-hashed `release.lock`, deterministic `run_id`s, N≥5 seeds per `(task, agent)`, low-temperature oracle with a replay cache. Numbers are comparable only within the same `release.lock`.

## License & citation

Harness ships under **MIT**. We store only links + pinned commit SHAs + short attributed excerpts from public repos — **never vendored upstream code**; all gold implementations and fixtures are authored clean-room. Task data ships CC-BY-style with `ATTRIBUTION.md` listing every referenced repo, license, and commit. See `CITATION.cff`.