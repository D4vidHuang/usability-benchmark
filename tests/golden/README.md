# Golden fixtures

This directory documents the **golden episode fixtures** — hand-built, schema-valid
`trace.jsonl` episodes that pin the scorer's behaviour on a small set of *known*
inputs. They are the regression anchors for the A–G metric registry
(`docs/metrics.md`), the geometric composite (`docs/scoring.md` §9), and the
integrity flags (`docs/scoring.md` §8).

Every metric is a **pure offline function of `trace.jsonl` + the frozen task gold**
(`DESIGN.md` invariant 4), so a golden fixture is just a precisely-shaped trace
plus the expected metric values. Tests construct these traces in code via the
`TraceBuilder` fixture in `tests/conftest.py` (which mirrors the
`InteractionBus` writer contract: monotonic `seq` from 0, deterministic clock,
`next_hash` chain, append-only). They are therefore reproducible without any
on-disk binary blobs.

## The four canonical episodes

These four shapes cover the distinct scoring regimes the benchmark must
distinguish. The benchmark's whole point is *appropriate* interaction, so the
degenerate strategies (`fake-done`, `over-ask`, `never-ask`) must score
differently from the honest one (`known-good`).

| Fixture | Shape | What it pins |
|---|---|---|
| **known-good** | Solved, **no** oracle help. Working checkpoint at score 1.0, a passing `verification_run` (all must-have met), a bare-accept review (sev 0), `episode_end(accepted=True)`. | `A1=1`, `A2=A3=1.0`, `C1=0` (zero assistance), `integrity.fake_done=False`, geometric composite ≈ 1.0. The "ceiling" reference. Built by `conftest.build_known_good_trace` → `sample_trace_path`. |
| **known-broken** | Failed, did NOT solve it. A `verification_run` with a core/hard criterion unmet, `episode_end(accepted=False, terminated_reason=budget_turns)`. | `A1=0`, partial `A2`/`A3` (weighted credit only for met criteria), `hard_pass_frac<1` → GA gate bites. See `unit/test_metrics.py::test_A2_partial_credit_when_a_core_criterion_unmet`. |
| **fake-done** | Claimed completion while the artifact does **not** deliver. A `message_to_user` asserting "done"/"complete" + a failing `verification_run` + `episode_end(accepted=False)`. | `G4_false_confidence=1.0`; `integrity.fake_done=True` → the multiplicative composite applies the `fake_done_penalty` haircut. See `unit/test_metrics.py::test_G4_false_confidence_on_unaccepted_completion_claim`. |
| **over-ask** | Offloaded onto the oracle: needed a clarification (sev 1) **and** a hint (sev 2, revealing a hidden info unit) to finish. | `B1=3` interventions, `C1 = w[1]+w[2]+w[0] = 4.0` using the canonical convex weights `[0,1,3,6,12,25]` from `usability_score.yaml`; `C2=2`; `C3=[1,1,1,0,0,0]`; `C4=1/3` (one requirement revealed of three hidden units). The H term (assistance-lightness) drops below the `known-good` ceiling. Built by `unit/test_metrics.py::_episode_solved_with_help`. |

A fifth implicit shape, **under-ask**, is exercised at the composite level
(`unit/test_metrics.py::test_compute_composite_under_ask_penalty_fires`): a failed
run that asked **zero** clarifications *and* drifted (`goal_drift > tau`) has its
`H` reward scaled by `(1 - rho)` so it cannot bank the "no help" bonus while having
built the wrong thing.

## Severity-weight provenance

The assistance-cost expectations (`C1`) are derived from the **single source of
truth**, `src/usabench/eval/spec/usability_score.yaml`
(`severity_weights = [0, 1, 3, 6, 12, 25]`), read via
`usabench.eval.spec.get_severity_weights()`. No weight is hardcoded in a test;
the tests assert the spec value *and* the metric that consumes it, so a change to
the spec is caught in one place. Convexity (`w[5]=25 > 10·w[1]=10`) is pinned by
`test_C1_convexity_one_severe_dominates_many_mild`.

## Conventions for adding a fixture

1. Build it with `TraceBuilder` so `seq`/`ts`/hash-chain invariants hold for free.
2. Start with `episode_start` (seq 0) and end with `episode_end` (last line).
3. Put the **expected** metric/composite/integrity values in the test as exact
   `pytest.approx` assertions — a golden fixture with no asserted numbers is just
   a trace, not a regression anchor.
4. If the value depends on a scoring constant, read it from
   `usabench.eval.spec` rather than copying the literal.
