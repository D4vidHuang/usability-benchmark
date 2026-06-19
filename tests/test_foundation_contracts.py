"""Foundation contract tests: enums, ids/hashing, schema models, JSON schemas, spec.

These exercise the shared contracts every downstream package imports, so a break
here is caught immediately rather than in a dependent workstream.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

import usabench
from usabench.core import ids
from usabench.core.enums import (
    InteractionType,
    Provider,
    Severity,
)
from usabench.core.schema import (
    SCHEMA_VERSION,
    AcceptanceCriterion,
    AgentMessage,
    HiddenSpec,
    OracleResponse,
    Task,
    TaskEnv,
    TraceEnvelope,
    chain_event,
    parse_event,
)
from usabench.eval.spec import get_severity_weights, load_spec

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"


# --------------------------------------------------------------------------- #
# Package + version                                                            #
# --------------------------------------------------------------------------- #


def test_version() -> None:
    assert usabench.__version__ == "0.1.0"


def test_lightweight_reexports() -> None:
    assert usabench.Severity is Severity
    assert usabench.SCHEMA_VERSION == SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# ids / hashing                                                                #
# --------------------------------------------------------------------------- #


def test_canonical_json_is_stable() -> None:
    a = {"b": 1, "a": [3, 2, 1]}
    b = {"a": [3, 2, 1], "b": 1}
    assert ids.canonical_json(a) == ids.canonical_json(b)


def test_run_id_deterministic() -> None:
    ch = ids.config_hash({"x": 1})
    r1 = ids.run_id(ch, "ub-cal-0007", 7, "abc123")
    r2 = ids.run_id(ch, "ub-cal-0007", 7, "abc123")
    r3 = ids.run_id(ch, "ub-cal-0007", 8, "abc123")
    assert r1 == r2 != r3
    assert len(r1) == 64


def test_hash_chain_links() -> None:
    h0 = ids.next_hash(ids.GENESIS_HASH, ids.canonical_json({"seq": 0}))
    h1 = ids.next_hash(h0, ids.canonical_json({"seq": 1}))
    assert h0 != h1
    assert len(h0) == 64 == len(h1)


# --------------------------------------------------------------------------- #
# Severity scale                                                               #
# --------------------------------------------------------------------------- #


def test_severity_labels_and_assistance() -> None:
    assert Severity.NONE.label == "none"
    assert Severity.TAKEOVER.label == "takeover"
    assert not Severity.is_assistance(0)
    assert Severity.is_assistance(1)
    with pytest.raises(ValueError):
        Severity.from_level(6)


def test_provider_openai_shaped() -> None:
    assert Provider.VLLM.is_openai_shaped
    assert not Provider.ANTHROPIC.is_openai_shaped


# --------------------------------------------------------------------------- #
# Trace envelope: build, chain, round-trip                                     #
# --------------------------------------------------------------------------- #


def _make_event(seq: int, prev: str) -> TraceEnvelope:
    env = TraceEnvelope(
        run_id="r1",
        event_id=f"e{seq}",
        seq=seq,
        ts=1718800000.0 + seq,
        t_turn=seq,
        actor="agent",
        type=InteractionType.AGENT_MESSAGE,
        payload=AgentMessage(text=f"hello {seq}"),
    )
    return chain_event(prev, env)


def test_trace_event_chain_and_parse_roundtrip() -> None:
    e0 = _make_event(0, ids.GENESIS_HASH)
    e1 = _make_event(1, e0.hash or "")
    assert e0.hash is not None and e1.hash is not None
    assert e1.prev_hash == e0.hash
    # Recompute hash matches the stored chain hash.
    assert e0.compute_hash() == e0.hash

    raw = json.loads(e0.model_dump_json())
    parsed = parse_event(raw)
    assert isinstance(parsed.payload, AgentMessage)
    assert parsed.payload.text == "hello 0"


def test_oracle_response_severity_coercion() -> None:
    resp = OracleResponse(severity=2, text="all-day events have no time")  # type: ignore[arg-type]
    assert int(resp.severity) == 2


def test_parse_event_rejects_unknown_type() -> None:
    from usabench.core.errors import SchemaViolation

    with pytest.raises(SchemaViolation):
        parse_event(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": "r",
                "event_id": "e",
                "seq": 0,
                "ts": 1.0,
                "actor": "agent",
                "type": "not_a_real_type",
                "payload": {"type": "not_a_real_type"},
                "prev_hash": ids.GENESIS_HASH,
            }
        )


# --------------------------------------------------------------------------- #
# Task agent_view strips gold                                                  #
# --------------------------------------------------------------------------- #


def _make_task() -> Task:
    return Task(
        id="ub-cal-0007",
        title="Calendar workload summarizer",
        user_goal="build me a tool that analyzes my calendar",
        domain="data-analysis",
        difficulty="T2",  # type: ignore[arg-type]
        deliverable_type="cli-tool",  # type: ignore[arg-type]
        env=TaskEnv(),
        hidden=HiddenSpec(
            summary="weekly time-allocation breakdown from an ICS file",
            acceptance_criteria=[
                AcceptanceCriterion(id="AC1", text="parses ics", is_core=True, is_hard=True, check_kind="func"),  # type: ignore[arg-type]
            ],
        ),
    )


def test_agent_view_has_no_gold_fields() -> None:
    task = _make_task()
    view = task.agent_view()
    dumped = view.model_dump()
    # No gold/hidden surface leaks into the agent view.
    for forbidden in ("hidden", "reference_repos", "acceptance_criteria", "ambiguity_points"):
        assert forbidden not in dumped
    assert dumped["user_goal"] == task.user_goal
    assert task.hidden.acceptance_criteria[0].is_hard is True


# --------------------------------------------------------------------------- #
# JSON schemas parse + are valid draft 2020-12                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["trace.schema.json", "task.schema.json", "raw_harvest.schema.json"])
def test_json_schemas_are_valid(name: str) -> None:
    schema = json.loads((SCHEMAS / name).read_text())
    # Raises if the schema itself is malformed under draft 2020-12.
    jsonschema.Draft202012Validator.check_schema(schema)


def test_trace_schema_accepts_a_real_event() -> None:
    schema = json.loads((SCHEMAS / "trace.schema.json").read_text())
    e0 = _make_event(0, ids.GENESIS_HASH)
    instance = json.loads(e0.model_dump_json())
    jsonschema.validate(instance=instance, schema=schema)


def test_task_schema_accepts_a_real_task() -> None:
    schema = json.loads((SCHEMAS / "task.schema.json").read_text())
    task = _make_task()
    instance = json.loads(task.model_dump_json(by_alias=True))
    jsonschema.validate(instance=instance, schema=schema)


# --------------------------------------------------------------------------- #
# usability_score.yaml is the single source of truth                           #
# --------------------------------------------------------------------------- #


def test_spec_severity_weights_canonical() -> None:
    assert get_severity_weights() == [0, 1, 3, 6, 12, 25]


def test_spec_top_level_keys_present() -> None:
    spec = load_spec()
    for key in (
        "severity_weights",
        "composite",
        "multiplicative",
        "ga_channels",
        "gate",
        "accept_threshold",
        "kappa",
        "epsilon",
        "pass_k",
        "judge",
    ):
        assert key in spec, f"missing spec key: {key}"
    assert spec["composite"] == {"alpha": 0.55, "beta": 0.45, "gamma": 0.20, "delta": 0.20}
    assert spec["multiplicative"]["lambda"] == 0.5
    assert spec["gate"] == {"floor": 0.30, "slope": 0.70}
