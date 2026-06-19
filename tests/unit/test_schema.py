"""Schema contracts: Task + trace-event validation and the visibility partition.

The two-tier visibility guarantee (``docs/tasks.md`` §2.1, ``DESIGN.md`` invariant
on gold leakage) is the most safety-critical schema property: the agent must never
see a gold field. We assert it structurally over the real models AND against the
on-disk JSON schemas.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from usabench.core.enums import InteractionType
from usabench.core.errors import SchemaViolation
from usabench.core.ids import GENESIS_HASH
from usabench.core.schema import (
    SCHEMA_VERSION,
    AgentMessage,
    CodeRun,
    CriterionResult,
    OracleResponse,
    Task,
    TraceEnvelope,
    VerificationRun,
    parse_event,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "schemas"


# --------------------------------------------------------------------------- #
# Task model validation                                                        #
# --------------------------------------------------------------------------- #


def test_sample_task_is_valid(sample_task: Task) -> None:
    """The conftest sample task round-trips through pydantic with its gold intact."""
    assert sample_task.id == "ub-cal-0007"
    assert len(sample_task.hidden.acceptance_criteria) == 3
    assert sample_task.hidden.n_hidden_spec_units == 3
    # is_core / is_hard flags survive round-trip.
    core = [c for c in sample_task.hidden.acceptance_criteria if c.is_core]
    assert {c.id for c in core} == {"AC1", "AC2"}


def test_task_round_trips_through_json_schema(sample_task: Task) -> None:
    """A real task instance validates against ``task.schema.json`` (by alias)."""
    schema = json.loads((SCHEMAS / "task.schema.json").read_text())
    instance = json.loads(sample_task.model_dump_json(by_alias=True))
    jsonschema.validate(instance=instance, schema=schema)


def test_task_rejects_out_of_range_threshold() -> None:
    """``accept_threshold`` is bounded to ``[0, 1]`` by the model."""
    from pydantic import ValidationError

    from usabench.core.schema import HiddenSpec, TaskEnv

    with pytest.raises(ValidationError):
        Task(
            id="bad",
            title="bad",
            user_goal="x",
            domain="d",
            difficulty="T1",  # type: ignore[arg-type]
            deliverable_type="script",  # type: ignore[arg-type]
            env=TaskEnv(),
            accept_threshold=1.5,
            hidden=HiddenSpec(summary="s"),
        )


# --------------------------------------------------------------------------- #
# Visibility partition: agent_view must leak NO gold                            #
# --------------------------------------------------------------------------- #


def test_agent_view_strips_every_gold_field(sample_task: Task) -> None:
    """The agent projection contains none of the oracle-private gold surfaces."""
    view = sample_task.agent_view()
    dumped = view.model_dump()
    forbidden = (
        "hidden",
        "reference_repos",
        "acceptance_criteria",
        "ambiguity_points",
        "info_units",
        "expected_interventions",
        "contamination_label",
        "harvest_provenance_id",
        "accept_threshold",
    )
    for field in forbidden:
        assert field not in dumped, f"gold field leaked into agent_view: {field}"
    # Public goal IS preserved.
    assert dumped["user_goal"] == sample_task.user_goal


def test_agent_view_serialized_does_not_contain_gold_strings(sample_task: Task) -> None:
    """No gold *value* (criterion text, ambiguity gold) appears in the serialized view."""
    blob = sample_task.agent_view().model_dump_json()
    gold_strings = [
        "all-day events count as 0 hours",  # ambiguity gold
        "parses an ICS file",  # criterion text
        "weekly time-allocation breakdown",  # hidden summary
    ]
    for secret in gold_strings:
        assert secret not in blob, f"gold value leaked into serialized agent_view: {secret!r}"


# --------------------------------------------------------------------------- #
# Trace-event validation                                                       #
# --------------------------------------------------------------------------- #


def _agent_message_event(seq: int = 0) -> TraceEnvelope:
    return TraceEnvelope(
        run_id="r1",
        event_id=f"e{seq}",
        seq=seq,
        ts=1.0 + seq,
        t_turn=seq,
        actor="agent",
        type=InteractionType.AGENT_MESSAGE,
        payload=AgentMessage(text="hello"),
        prev_hash=GENESIS_HASH,
    )


def test_trace_envelope_derives_type_from_payload() -> None:
    """A discriminated payload resolves back to its typed model on parse."""
    env = _agent_message_event()
    raw = json.loads(env.model_dump_json())
    parsed = parse_event(raw)
    assert isinstance(parsed.payload, AgentMessage)
    assert parsed.payload.text == "hello"
    assert str(parsed.type) == InteractionType.AGENT_MESSAGE.value


def test_verification_run_payload_round_trips_through_schema() -> None:
    """A verification_run with per-criterion results is schema-valid on disk."""
    schema = json.loads((SCHEMAS / "trace.schema.json").read_text())
    env = TraceEnvelope(
        run_id="r1",
        event_id="e1",
        seq=0,
        ts=1.0,
        t_turn=3,
        actor="harness",
        type=InteractionType.VERIFICATION_RUN,
        payload=VerificationRun(
            trigger="submit",
            entrypoint="python main.py",
            must_have=[CriterionResult(id="AC1", passed=True)],
            should_have=[CriterionResult(id="AC3", score=0.5)],
            all_must_pass=True,
            rubric_score=0.83,
        ),
        prev_hash=GENESIS_HASH,
    )
    instance = json.loads(env.model_dump_json())
    jsonschema.validate(instance=instance, schema=schema)


def test_oracle_response_requires_severity() -> None:
    """Severity is mandatory on an oracle response (the assistance signal)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OracleResponse(text="no severity given")  # type: ignore[call-arg]


def test_oracle_response_severity_coerces_int() -> None:
    """An int severity is coerced onto the 0-5 Severity scale."""
    resp = OracleResponse(severity=4, text="here is most of the solution")  # type: ignore[arg-type]
    assert int(resp.severity) == 4


def test_code_run_self_test_flag_tristate() -> None:
    """``self_test_passed`` is an explicit tri-state (None / True / False)."""
    assert CodeRun(cmd="echo").self_test_passed is None
    assert CodeRun(cmd="pytest", is_test=True, self_test_passed=True).self_test_passed is True
    assert CodeRun(cmd="pytest", is_test=True, self_test_passed=False).self_test_passed is False


def test_parse_event_rejects_unknown_type() -> None:
    """An unknown envelope type is a hard schema violation, not a silent pass."""
    with pytest.raises(SchemaViolation):
        parse_event(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": "r",
                "event_id": "e",
                "seq": 0,
                "ts": 1.0,
                "actor": "agent",
                "type": "does_not_exist",
                "payload": {"type": "does_not_exist"},
                "prev_hash": GENESIS_HASH,
            }
        )
