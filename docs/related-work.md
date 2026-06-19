I now have everything verified, including the precise pass^k formula (τ-bench uses the unbiased combinatorial estimator $\binom{c}{k}/\binom{n}{k}$ per task, equivalent in expectation to $(c/n)^k$). I have grounded all the named benchmarks plus the closest neighbors on interactivity/clarification/human-effort. Writing the document now.

# Related Work & Positioning: `usability-benchmark`

*Survey, comparative table, and positioning for an open-ended, simulated-user, human-intervention-metric benchmark for AI coding agents. All claims below are grounded in the cited primary sources (verified via web search/fetch June 2026).*

---

## 1. Why a new benchmark? The shape of the gap

The dominant coding-agent benchmarks measure **terminal correctness on a closed task**: given a precise (or precise-enough) specification, does the produced artifact pass hidden tests? This is the SWE-bench paradigm and almost everything downstream of it. Three things are systematically *not* measured by that paradigm, and all three are first-class in `usability-benchmark`:

1. **Open-endedness.** Real users state goals ("build me a CLI that summarizes my git activity"), not issues-with-failing-tests. The space of acceptable solutions is large; the spec is deliberately under-determined.
2. **Interaction as the unit of analysis.** When the goal is ambiguous, a *usable* agent asks, proposes, and confirms. Most benchmarks treat any back-and-forth as either disallowed (single-shot) or as free "feedback" that improves the score, not as a *cost* to be accounted for.
3. **Human effort / intervention as the headline metric.** Usability is not "did it eventually pass" but "how much of *my* time, attention, and clarification did it consume to get there." Almost no coding benchmark reports human-effort as a primary, per-interaction, classified quantity.

The closest neighbors each touch one of these axes — τ-bench on the **simulated user**, MINT on **multi-turn feedback budget**, Ambig-SWE / ClarEval / HiL-Bench on **clarification**, CentaurEval / ProSoftArena / Anthropic's autonomy work on **human-intervention counting** — but none combine *open-ended dev goals* + *simulated-user oracle holding gold intent* + *intervention-amount-and-severity as the primary score*. That combination is the gap.

---

## 2. Survey of prior benchmarks (what each actually measures)

### Code-fix / closed-spec family

- **SWE-bench / SWE-bench Verified** ([Jimenez et al., 2023, arXiv:2310.06770](https://arxiv.org/abs/2310.06770); [OpenAI Verified, 2024](https://openai.com/index/introducing-swe-bench-verified/)). Task = real GitHub issue + full repo; agent emits a diff; success = hidden `FAIL_TO_PASS`/`PASS_TO_PASS` test suite passes. Verified is a 500-instance human-filtered subset removing under-specified issues and weak tests. **Metric: resolved rate (pass@1, pass@3).** No interaction model; no human-effort metric. It *deliberately removes ambiguity* — the opposite of our design.
- **Aider polyglot benchmark** ([Aider, Dec 2024](https://aider.chat/2024/12/21/polyglot.html); [leaderboard](https://aider.chat/docs/leaderboards/)). 225 hardest Exercism problems across C++, Go, Java, JS, Python, Rust. Two attempts; test-error feedback injected after the first failure. Measures **correctness + edit-format adherence** (diff/whole/architect). The "feedback" is a single automated retry, not a user. No human-effort metric.
- **Commit0** ([Zhao et al., 2024, arXiv:2412.01769](https://arxiv.org/abs/2412.01769)). Generate 54 Python *libraries from scratch* against a spec doc + interactive unit tests; static-analysis + execution feedback loop. **Metric: unit-test pass rate** (SOTA low: ~6–29%). Spec is *complete and precise* (it's the gold API); no clarification, no simulated user.
- **BigCodeBench** ([Zhuo et al., 2024, arXiv:2406.15877](https://arxiv.org/abs/2406.15877)). 1,140 tasks invoking 139 libraries; complex docstring instructions; ~5.6 tests/task, ~99% branch coverage. **Metric: pass@1 (Complete/Instruct variants).** Single-shot function generation; no interaction.
- **InterCode** ([Yang et al., 2023, arXiv:2306.14898](https://arxiv.org/abs/2306.14898)). Frames coding as an RL environment: code = actions, *execution feedback* = observations (Bash/SQL/Python/CTF), Dockerized. Multi-turn, but the "turns" are with the **environment**, not a user. **Metric: success rate / reward.** No human-effort accounting.

### Economic / professional realism

- **SWE-Lancer** ([Miserendino et al., 2025, arXiv:2502.12115](https://arxiv.org/abs/2502.12115); [OpenAI](https://openai.com/index/swe-lancer/)). 1,400+ real Upwork freelance tasks worth ~$1M; end-to-end tests triple-verified; plus managerial "pick the best proposal" tasks graded vs the originally-hired engineer. **Metric: % resolved and $ earned.** Tasks come with a *finished, paid spec*; no clarification loop, no simulated client. It quantifies *economic value of correctness*, not *human effort to obtain it*.
- **DevBench** ([Li et al., 2024, arXiv:2403.08604](https://arxiv.org/html/2403.08604v1)). Five waterfall stages (design → env setup → implementation → unit/acceptance testing); each stage fed *reference* inputs from prior stages. Verification mixes LLM-as-Judge (design) + execution (PyTest/JUnit/Jest/GTest). Mentions an optional "Copilot mode" for human refinement but the standard eval is fully automated, no interaction metric.
- **DevAI + Agent-as-a-Judge** ([Zhuge et al., 2024, arXiv:2410.10934](https://arxiv.org/abs/2410.10934)). 55 realistic AI-app-dev tasks with **365 hierarchical, manually-annotated requirements**; an *agentic judge* inspects intermediate steps and scores which requirements were satisfied (~90% alignment with humans). This is the strongest prior art for *requirement-level grading of open-ended dev tasks* — we borrow it. But DevAI's agent runs autonomously; there is no simulated user and no intervention budget.
- **ProjDevBench** ([2026, arXiv:2602.01655](https://arxiv.org/pdf/2602.01655)). End-to-end project development grounded in real OSS repos, execution-based verification, iterative (multi-turn vs env). Closest on *open-ended dev from a spec*, but still spec-in/tests-out with no user oracle and no human-effort metric.

### Tool-use & generalist agents

- **AppWorld** ([Trivedi et al., ACL 2024, arXiv:2407.18901](https://arxiv.org/abs/2407.18901)). 750 day-to-day tasks over 9 simulated apps / 457 APIs / ~100 fictitious users; agent writes interactive code; **state-based unit tests** allow many solution paths while catching collateral damage. Multi-turn vs a *simulated world* (not a user holding intent). No human-effort metric.
- **GAIA** ([Mialon et al., 2023, arXiv:2311.12983](https://arxiv.org/abs/2311.12983)). 466 real-world assistant questions, 3 difficulty levels by #steps/tools; **exact-match accuracy** vs a single gold answer; humans ~92% vs early GPT-4+plugins ~15%. Single-shot Q→A; no interaction, no effort metric.
- **AgentBench** ([Liu et al., 2023, arXiv:2308.03688](https://arxiv.org/abs/2308.03688)). 8 environments (OS, DB, KG, games, web, etc.); **task success rate** across 29 LLMs. Multi-turn vs environments; no simulated user, no effort metric.
- **WebArena / VisualWebArena** ([Zhou et al., 2023, arXiv:2307.13854](https://arxiv.org/abs/2307.13854); [Koh et al., 2024, arXiv:2401.13649](https://arxiv.org/html/2401.13649v2)). 812 / 910 tasks on self-hosted real-stack sites; **functional-correctness reward** on final state (plus VQA/SSIM/fuzzy-match in VWA). Multi-turn vs web; no user oracle, no effort metric.

### Simulated-user & interaction-centric family (closest neighbors)

- **τ-bench** ([Yao et al., 2024, arXiv:2406.12045](https://arxiv.org/abs/2406.12045)). The canonical **LLM-simulated user**: an LLM plays a customer with a goal/persona; the agent has domain APIs + policy docs; success = **final database state == annotated goal state**. Introduces **pass^k** for *reliability over k trials*. This is our nearest neighbor on the *user-simulator mechanism* — but the domain is customer service (retail/airline), the task is *bounded transactions*, not open-ended software construction, and the metric is correctness/reliability, **not human effort**. The user is an obstacle to satisfy, not a resource whose effort we meter.
- **τ²-bench** ([Barres et al., 2025, arXiv:2506.07982](https://arxiv.org/pdf/2506.07982)). Adds a **dual-control** setting (Telecom): both user *and* agent can call tools / mutate state, modeling collaborative troubleshooting; compositional task generation; richer failure-mode taxonomy. Pushes the simulated user from "information source" toward "co-actor" — conceptually relevant to our hand-off/intervention model, still customer-service-shaped, still no effort metric.
- **MINT** ([Wang et al., ICLR 2024, arXiv:2309.10691](https://arxiv.org/abs/2309.10691)). Multi-turn eval where the LLM uses tools (Python execution) **and** receives **GPT-4-simulated natural-language feedback**. Reports performance as a function of **interaction turns / feedback budget** (gains of 1–8% per tool turn, 2–17% with language feedback). This is the prior art for *treating turns/feedback as a measured budget* — we borrow the budget framing. But MINT's feedback *helps* the score; it is not counted as *cost*, the tasks are repurposed reasoning/coding/decision datasets (not open-ended dev), and the user does not hold a defensible gold "intent."
- **Ambig-SWE** ([2025, arXiv:2502.13069](https://arxiv.org/pdf/2502.13069)). Deliberately *removes key info* from SWE-bench issues; an **oracle** answers the agent's clarifying questions; reports #questions, question quality, interaction depth, and resolution lift vs non-interactive. Extremely close in *mechanism* (oracle-answered clarification on under-specified tasks) — but it operates on bug-fix issues with a single gold patch, not open-ended app goals, and clarification is a means to the resolved-rate end rather than the measured outcome.
- **ClarEval** ([2026, arXiv:2603.00187](https://arxiv.org/html/2603.00187v1)). Benchmark of **clarification skill** under typed ambiguity; shows GPT-4o drops from ~89% (clarified) to ~9% (ambiguous), with "ambiguous terminology" hardest. Measures *whether/how well agents clarify*; complements us but is not open-ended dev and not effort-metered.
- **HiL-Bench** ([2026, arXiv:2604.09408](https://arxiv.org/html/2604.09408v3)). Converts static coding tasks into multi-turn collaboration by injecting 3–5 hidden "blockers" discovered *progressively* during execution; a frozen Llama-3.3-70B judge answers questions deterministically; headline metric **Ask-F1** (harmonic mean of question *precision* = relevant/asked and *recall* = blockers resolved), explicitly **penalizing question-spam**. This is the best prior art for *not gaming the score by over-asking* — we adopt the precision/recall-of-asking idea directly.
- **CentaurEval** ([2025, arXiv:2512.04111](https://arxiv.org/abs/2512.04111)). "Collaboration-Necessary" templates solvable by neither human nor AI alone; **4 intervention levels** (human-only, fully-autonomous, minimally-intervened, free collaboration); metrics include pass, partial-pass, **completion time**, **token usage**. Demonstrates collaboration lift (18.9%→31.1%). Uses **real humans**, which is rigorous but *not scalable/reproducible* — precisely why we use a simulated-user oracle instead.
- **ProSoftArena** ([2026, arXiv:2601.02399](https://arxiv.org/pdf/2601.02399)) reports **human-intervention count** and **human-operation-time per task** as explicit metrics — direct precedent that *counting interventions* is a legitimate, citable axis.
- **Anthropic, "Measuring agent autonomy in practice"** ([2026](https://www.anthropic.com/research/measuring-agent-autonomy)). Industrial precedent for our exact metric family: interventions/session (5.4→3.3 over months), turn duration, auto-approve rate, interrupt rate, and a **taxonomy of why agents stop** (35% present choices, 21% gather diagnostics) **vs why humans interrupt** (32% missing context, 17% slowness). We mirror this taxonomy in our intervention classifier.

---

## 3. One-row comparative table

| Benchmark | Task type | Verification | Interaction model | Usability / human-effort metric |
|---|---|---|---|---|
| **SWE-bench / Verified** | Real GitHub bug-fix issue (closed spec) | Hidden test suite (FAIL→PASS) | Single-shot (agent-loop vs env) | None |
| **Aider polyglot** | Exercism coding problems | Unit tests | 1 automated test-error retry | None |
| **Commit0** | Build library from API spec | Unit-test pass rate | Multi-turn vs static analysis/exec | None |
| **BigCodeBench** | Lib-heavy function gen | Unit tests (~99% branch) | Single-shot | None |
| **InterCode** | Bash/SQL/Python/CTF | Reward / success | Multi-turn vs **environment** | None |
| **SWE-Lancer** | Real freelance tasks (closed, paid) | E2E tests (triple-verified) + manager-choice | Single-shot | $ earned (value of correctness, not effort) |
| **DevBench** | 5 waterfall stages | Exec + LLM-judge | Staged, reference-fed | Optional copilot mode (not metered) |
| **DevAI / Agent-as-a-Judge** | Open-ended AI-app dev | **Agentic judge over 365 hierarchical reqs** | Autonomous + intermediate inspection | None (req-satisfaction %) |
| **ProjDevBench** | End-to-end project from spec (real OSS) | Execution tests | Multi-turn vs env | None |
| **AppWorld** | Day-to-day app tasks (457 APIs) | State-based unit tests | Multi-turn vs simulated world | None |
| **GAIA** | Real assistant Q→A | Exact match | Single-shot | None |
| **AgentBench** | 8 agent environments | Success rate | Multi-turn vs env | None |
| **WebArena / VWA** | Web tasks | Functional-correctness reward (+VQA/SSIM) | Multi-turn vs web | None |
| **MINT** | Reasoning/coding/decision (repurposed) | Task success | **Multi-turn + GPT-4 NL feedback** | Turns/feedback **budget** (as benefit) |
| **τ-bench** | Customer-service transactions | **DB state == goal state** | **LLM-simulated user** + tools | **pass^k** (reliability, not effort) |
| **τ²-bench** | CS transactions, dual-control | DB state + telecom env | Simulated user as **co-actor** | pass^k + failure taxonomy |
| **Ambig-SWE** | Under-specified SWE issues | Gold patch / tests | **Oracle-answered clarification** | #questions, question quality |
| **ClarEval** | Typed-ambiguity coding | Tests on clarified spec | Clarification turn | Clarification accuracy |
| **HiL-Bench** | Coding w/ hidden blockers | Pass@3 + blocker resolution | **Frozen-LLM judge answers questions** | **Ask-F1** (penalizes over-asking) |
| **CentaurEval** | Collaboration-necessary coding | Pass / partial-pass | **4 levels, real humans** | Completion time, tokens |
| **ProSoftArena** | Pro software-env tasks | Task success | Human-in-loop | **#interventions, human-op-time** |
| **Anthropic autonomy** | Production Claude Code sessions | n/a (telemetry) | Real users | **Interventions/session + stop/interrupt taxonomy** |
| **`usability-benchmark` (ours)** | **Open-ended dev goal, lay phrasing, grounded in real OSS** | **Agentic/requirement-level acceptance vs reference repo** | **Simulated-user oracle holding gold intent** | **Intervention amount + severity (primary), pass^k for variance** |

---

## 4. Positioning statement

> **`usability-benchmark` is the first benchmark to evaluate AI coding agents on *open-ended, lay-phrased software-development goals* while making *human-intervention amount and severity* the primary score, using an LLM *simulated-user oracle* that holds the reference project's gold intent, constraints, and hidden acceptance criteria.**

Against the **closest neighbors**, the delta is precise:

- **vs τ-bench / τ²-bench (closest on the user simulator).** We adopt τ-bench's central insight — an LLM playing the human with privileged gold knowledge — but invert what is measured and broaden the task. τ-bench's user is a *spec to be satisfied* in bounded customer-service transactions, scored by DB-state correctness/reliability. Our oracle is a *resource whose effort is being metered* across *open-ended app/tool construction*; the agent's job is not just to succeed but to **minimize how much it had to ask, hint-request, or hand off**. Same mechanism, different dependent variable and a development (not transaction) task space.
- **vs MINT (closest on multi-turn feedback budget).** MINT pioneered reporting performance *as a function of interaction budget* with a GPT-4 feedback simulator. But in MINT feedback is a **benefit** that lifts the score and the tasks are repurposed static datasets. We treat interaction as a **cost** with classified severity, on tasks that are *intentionally under-specified open-ended dev goals* grounded in real repos, where some clarification is *necessary and correct* — so we must reward *good, parsimonious* asking, not penalize all asking nor reward all of it.
- **vs Ambig-SWE / ClarEval / HiL-Bench (closest on clarification + oracle answers).** These nail oracle-answered clarification and (HiL-Bench) anti-spam scoring, but on **bug-fix / closed-gold-patch** tasks. We move from "fix this issue with hidden details restored" to "build this thing the way a real user wanted it," where the gold is a *reference project's design intent and acceptance criteria*, not a single patch, and where the score integrates **all** intervention types (clarification, hint, correction, hand-off), not just questions.
- **vs CentaurEval / ProSoftArena / Anthropic (closest on intervention counting).** These establish that counting interventions and human-operation-time is a legitimate, citable metric — but CentaurEval/ProSoftArena depend on **real humans** (not reproducible/scalable) and Anthropic's data is production telemetry (not a controlled benchmark). We make the human a **deterministic-as-possible LLM oracle**, recovering scalability and reproducibility while keeping the intervention taxonomy.
- **vs DevAI / ProjDevBench / SWE-Lancer (closest on open-ended/real dev).** These get the *open-ended, real-repo, requirement-graded* dev task right (we borrow DevAI's hierarchical-requirement agentic grading), but all run the agent **autonomously with no user in the loop** and report **correctness/value, never human effort**.

**In one sentence:** every neighbor owns *one* of {open-ended dev task, simulated-user oracle, intervention-as-primary-metric}; `usability-benchmark` is the intersection of all three.

---

## 5. Proven methodological ideas worth borrowing (with citations)

1. **LLM simulated-user oracle holding gold intent** — from **τ-bench** ([arXiv:2406.12045](https://arxiv.org/abs/2406.12045)) and **τ²-bench** ([arXiv:2506.07982](https://arxiv.org/pdf/2506.07982)). Use it for our oracle; adopt τ²'s *dual-control* framing for hand-off scenarios where the oracle can act, not just answer.
2. **pass^k for reliability under LLM stochasticity** — **τ-bench**. Definition: per task, run *n* trials with *c* successes; the unbiased estimator is $\widehat{\text{pass}^k} = \binom{c}{k}\big/\binom{n}{k}$ (equiv. in expectation to $(c/n)^k$), averaged over tasks — the probability that *all k* runs succeed. Report this *and* a pass^k on the **intervention budget** (probability that k runs stay under an effort threshold) to capture variance in our headline metric. (Contrast with pass@k = $1-\binom{n-c}{k}/\binom{n}{k}$; see [Yao et al.](https://arxiv.org/abs/2406.12045), [explainer](https://www.philschmid.de/agents-pass-at-k-pass-power-k).)
3. **Interaction/feedback as a measured budget** — **MINT** ([arXiv:2309.10691](https://arxiv.org/abs/2309.10691)). Report performance *as a function of* intervention budget (turn/hint caps), and curves of success-vs-effort, not a single number.
4. **Anti-spam scoring of asking (precision/recall of questions)** — **HiL-Bench** Ask-F1 ([arXiv:2604.09408](https://arxiv.org/html/2604.09408v3)). Essential so agents can't game low effort by never asking *or* high success by interrogating the oracle. Score *parsimony*: relevant-asks / total-asks against blockers-resolved / total-blockers.
5. **Progressive, execution-time discovery of hidden blockers** — **HiL-Bench**. Don't surface all ambiguity upfront; let some emerge during the build, mirroring real dev and forcing genuine interaction.
6. **Deterministic oracle answers via registered triggers** — **HiL-Bench** (frozen judge + trigger phrasings) and **Ambig-SWE** ([arXiv:2502.13069](https://arxiv.org/pdf/2502.13069)). Pre-register canonical answers/criteria so oracle responses are reproducible despite LLM stochasticity.
7. **Hierarchical requirement annotations + agentic grading** — **DevAI / Agent-as-a-Judge** ([arXiv:2410.10934](https://arxiv.org/abs/2410.10934)). Grade open-ended dev outputs against a tree of acceptance requirements (~90% human alignment) rather than one pass/fail; gives partial credit and explains *which* requirements were met.
8. **State-based unit tests that allow many solution paths** — **AppWorld** ([arXiv:2407.18901](https://arxiv.org/abs/2407.18901)) and **WebArena** functional-correctness ([arXiv:2307.13854](https://arxiv.org/abs/2307.13854)). For open-ended goals, verify *end-state behavior* and check for collateral damage, not trajectory match.
9. **Real-OSS grounding for defensible acceptance criteria** — **SWE-bench** ([arXiv:2310.06770](https://arxiv.org/abs/2310.06770)), **SWE-Lancer** ([arXiv:2502.12115](https://arxiv.org/abs/2502.12115)), **ProjDevBench** ([arXiv:2602.01655](https://arxiv.org/pdf/2602.01655)). Anchor each task in a real repo/README/feature-request so the "gold intent" is defensible, not invented.
10. **Intervention taxonomy + interventions-per-session** — **Anthropic autonomy** ([link](https://www.anthropic.com/research/measuring-agent-autonomy)) and **ProSoftArena** ([arXiv:2601.02399](https://arxiv.org/pdf/2601.02399)). Classify each intervention by *who initiated* (agent-ask vs oracle-correct) and *why* (missing context, present choices, gather diagnostics, slowness) and *severity*; report count + human-operation-time analogue.
11. **Graded human-intervention levels for ablation** — **CentaurEval** ([arXiv:2512.04111](https://arxiv.org/abs/2512.04111)). Run each agent at L0 (no oracle), L1 (clarification-only), L2 (hints), L3 (free intervention) to produce a *usability curve* per agent, isolating how much each assistance type buys.
12. **Human-validation of the task set** — **SWE-bench Verified** ([OpenAI, 2024](https://openai.com/index/introducing-swe-bench-verified/)). Even with a simulated oracle, human-filter the task set to ensure goals are genuinely under-specified-yet-resolvable and the gold criteria are sound.

---

### Pointers / artifacts on disk
The fetched PDFs of the closest-neighbor papers were cached by the fetch tool under `/Users/davidhuang/.claude/projects/-Users-davidhuang-Desktop-usabilityBenchMark/96aa6afd-8c96-4ae5-a92b-1c7003352996/tool-results/` (τ²-bench, Ambig-SWE, ProjDevBench) if a local copy is useful for the repo's `docs/related-work/` references. Primary canonical sources are the arXiv links above.