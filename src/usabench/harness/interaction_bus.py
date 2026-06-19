"""The InteractionBus: the trace writer and the only agent<->oracle channel.

Two responsibilities, deliberately fused so that *every* event passes through one
place and the trace is guaranteed complete:

1. **Trace writer.** The bus owns the monotonic ``seq`` counter, the wall clock,
   the hash chain (via :func:`usabench.core.ids.next_hash`), and the append-only
   ``trace.jsonl`` file handle. ``emit(...)`` is the single funnel that stamps an
   envelope, chains it, writes one JSON line, and returns the chained event. This
   is the ONE canonical artifact (``DESIGN.md`` invariant 4) -- nothing the scorer
   needs may live only in memory.

2. **Oracle gateway.** ``ask_oracle(...)`` is the only path from agent to oracle.
   It logs the ``oracle_query``, routes it to the oracle (any object satisfying the
   structural :class:`OracleLike` protocol), grades/persists the ``oracle_response``
   with its self-declared 0-5 severity, and returns the response. Because it is the
   sole channel, the assistance metrics are computable purely from the trace.

The bus is offline-safe and synchronous; the runner sequences turns around it.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from usabench.core.enums import (
    Actor,
    InteractionType,
    QueryClass,
    Severity,
    Verdict,
)
from usabench.core.ids import GENESIS_HASH, next_hash
from usabench.core.schema import (
    BudgetSnapshot,
    OracleQuery,
    OracleResponse,
    TraceEnvelope,
    TraceEvent,
)
from usabench.logging_setup import get_logger

__all__ = ["OracleLike", "OracleQueryContext", "InteractionBus"]

_log = get_logger(__name__)


@runtime_checkable
class OracleLike(Protocol):
    """Structural protocol the oracle implementation must satisfy.

    Defined here (not imported from a not-yet-built ``usabench.oracle`` package) so
    the harness has no hard dependency on the oracle module. Any object exposing
    ``answer`` works: a real LLM-backed oracle, a replay-cache oracle, or a scripted
    fake for the smoke path.
    """

    def answer(self, ctx: OracleQueryContext) -> OracleResponse:
        """Return a graded :class:`OracleResponse` for the given query context."""
        ...


class OracleQueryContext:
    """Everything the oracle needs to answer one query (read-only view).

    Attributes:
        query: The typed :class:`OracleQuery` payload the agent sent.
        query_event_id: The ``event_id`` of the logged ``oracle_query`` (for
            ``responds_to`` linkage).
        run_id: The current run id.
        seq: The seq of the originating query event.
        t_turn: The agent-turn index of the query.
        history: Optional running list of prior (query, response) pairs for context.
    """

    def __init__(
        self,
        *,
        query: OracleQuery,
        query_event_id: str,
        run_id: str,
        seq: int,
        t_turn: int | None,
        history: list[tuple[OracleQuery, OracleResponse]] | None = None,
    ) -> None:
        self.query = query
        self.query_event_id = query_event_id
        self.run_id = run_id
        self.seq = seq
        self.t_turn = t_turn
        self.history = history if history is not None else []


class InteractionBus:
    """Append-only trace writer + the sole agent<->oracle channel.

    The bus is the single writer of ``trace.jsonl``. It is opened with a target
    path and a run id, then ``emit``s envelopes in total ``seq`` order with a valid
    hash chain. Use it as a context manager so the file handle is always closed::

        with InteractionBus(path, run_id="...") as bus:
            bus.emit(Actor.HARNESS, episode_start_payload, t_turn=None)
            ...

    Thread-unsafe by design: one run is single-threaded for determinism
    (``docs/protocol.md`` §5).
    """

    def __init__(
        self,
        trace_path: str | Path,
        *,
        run_id: str,
        oracle: OracleLike | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialize the bus.

        Args:
            trace_path: Where to write ``trace.jsonl`` (parent dirs are created).
            run_id: The run id stamped on every envelope.
            oracle: Optional oracle backend (required only for :meth:`ask_oracle`).
            clock: Optional callable returning epoch seconds (for deterministic
                tests); defaults to :func:`time.time`.
        """
        self.trace_path = Path(trace_path)
        self.run_id = run_id
        self.oracle = oracle
        if clock is not None:
            self._clock: Callable[[], float] = clock
        else:
            import time as _time

            self._clock = _time.time
        self._seq = 0
        self._prev_hash = GENESIS_HASH
        self._fh: Any | None = None
        self._oracle_history: list[tuple[OracleQuery, OracleResponse]] = []
        self._severity_counts: dict[int, int] = {s: 0 for s in range(6)}

    # --- lifecycle ---------------------------------------------------------- #

    def open(self) -> InteractionBus:
        """Open (truncate) the trace file for writing. Idempotent per instance."""
        if self._fh is None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.trace_path.open("w", encoding="utf-8")
        return self

    def close(self) -> None:
        """Flush and close the trace file handle."""
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> InteractionBus:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- accessors ---------------------------------------------------------- #

    @property
    def seq(self) -> int:
        """The seq that the *next* emitted event will receive."""
        return self._seq

    @property
    def last_hash(self) -> str:
        """The hash of the most recently emitted event (``GENESIS_HASH`` if none)."""
        return self._prev_hash

    @property
    def severity_counts(self) -> dict[int, int]:
        """Running count of oracle-response severities (cache for ``episode_end``)."""
        return dict(self._severity_counts)

    # --- core writer -------------------------------------------------------- #

    def emit(
        self,
        actor: Actor,
        payload: TraceEvent,
        *,
        t_turn: int | None,
        budgets_after: BudgetSnapshot | None = None,
        event_id: str | None = None,
    ) -> TraceEnvelope:
        """Stamp, chain, persist, and return one trace event.

        This is the single funnel for the trace: it assigns the next ``seq``, stamps
        the wall clock, derives the envelope ``type`` from the payload, chains the
        hash from the previous event, writes one canonical JSON line, and advances
        the chain. The written line is byte-stable (sorted keys, compact separators).

        Args:
            actor: Who produced the event.
            payload: One typed member of :data:`~usabench.core.schema.TraceEvent`.
            t_turn: Agent-turn index, or ``None`` for harness-internal events.
            budgets_after: The budget snapshot after this event; defaults to zeros.
            event_id: Optional explicit event id (defaults to a fresh UUID4 hex).

        Returns:
            The chained :class:`TraceEnvelope` (with ``hash`` populated).

        Raises:
            RuntimeError: If the bus file handle is not open.
        """
        if self._fh is None:
            raise RuntimeError("InteractionBus.emit called before open()")
        etype = InteractionType(payload.type)
        envelope = TraceEnvelope(
            run_id=self.run_id,
            event_id=event_id or uuid.uuid4().hex,
            seq=self._seq,
            ts=float(self._clock()),
            t_turn=t_turn,
            actor=actor,
            type=etype,
            payload=payload,
            budgets_after=budgets_after or BudgetSnapshot(),
            prev_hash=self._prev_hash,
        )
        envelope.hash = next_hash(self._prev_hash, envelope.canonical_without_hash())
        line = json.dumps(envelope.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        self._fh.write(line + "\n")
        self._fh.flush()
        self._prev_hash = envelope.hash
        self._seq += 1
        return envelope

    # --- oracle channel ----------------------------------------------------- #

    def ask_oracle(
        self,
        text: str,
        query_class: QueryClass,
        *,
        t_turn: int | None,
        context_refs: list[str] | None = None,
        agent_blocked: bool = False,
        budgets_after: BudgetSnapshot | None = None,
    ) -> tuple[TraceEnvelope, TraceEnvelope]:
        """Route one agent query through the oracle, logging both sides.

        The bus logs the ``oracle_query``, asks the oracle for a graded answer,
        forces the ``responds_to`` linkage to the query's ``event_id``, tallies the
        severity, then logs the ``oracle_response``. This is the only sanctioned
        path to the oracle (``docs/infra.md`` design constraint: nothing reaches the
        oracle except through the bus).

        Args:
            text: The agent's question.
            query_class: The classified query type.
            t_turn: The current agent-turn index.
            context_refs: Optional references (e.g. ``["seq:38", "path:src/cli.py"]``).
            agent_blocked: True if the agent declared itself blocked when asking.
            budgets_after: Budget snapshot to stamp on both events.

        Returns:
            A ``(query_envelope, response_envelope)`` tuple.

        Raises:
            RuntimeError: If no oracle backend was configured.
        """
        if self.oracle is None:
            raise RuntimeError("InteractionBus.ask_oracle called with no oracle configured")

        query_payload = OracleQuery(
            query_class=query_class,
            text=text,
            context_refs=list(context_refs or []),
            agent_blocked=agent_blocked,
        )
        query_env = self.emit(
            Actor.AGENT, query_payload, t_turn=t_turn, budgets_after=budgets_after
        )

        ctx = OracleQueryContext(
            query=query_payload,
            query_event_id=query_env.event_id,
            run_id=self.run_id,
            seq=query_env.seq,
            t_turn=t_turn,
            history=self._oracle_history,
        )
        response = self.oracle.answer(ctx)
        # The bus authoritatively links the response to its query, regardless of
        # what the oracle returned, so the scorer's ref-integrity invariant holds.
        response = response.model_copy(update={"responds_to": query_env.event_id})

        sev = int(response.severity)
        self._severity_counts[sev] = self._severity_counts.get(sev, 0) + 1
        self._oracle_history.append((query_payload, response))

        response_env = self.emit(
            Actor.ORACLE, response, t_turn=t_turn, budgets_after=budgets_after
        )
        return query_env, response_env

    def oracle_review(
        self,
        verdict: Verdict,
        *,
        severity: Severity = Severity.NONE,
        text: str = "",
        cited_criteria: list[str] | None = None,
        t_turn: int | None,
        budgets_after: BudgetSnapshot | None = None,
    ) -> TraceEnvelope:
        """Log an oracle review verdict on a submission as an ``oracle_response``.

        A reject that names a fix is itself assistance, so it carries a severity and
        is tallied like any other oracle response (``docs/protocol.md`` §3.1).

        Args:
            verdict: ``accept`` | ``reject`` | ``na``.
            severity: Assistance severity of the review (0 for a bare accept).
            text: The review text / feedback.
            cited_criteria: Criterion ids the verdict cites.
            t_turn: The current agent-turn index.
            budgets_after: Budget snapshot to stamp.

        Returns:
            The emitted ``oracle_response`` envelope.
        """
        payload = OracleResponse(
            responds_to=None,
            severity=severity,
            text=text,
            verdict=verdict,
            cited_criteria=list(cited_criteria or []),
        )
        sev = int(severity)
        self._severity_counts[sev] = self._severity_counts.get(sev, 0) + 1
        return self.emit(Actor.ORACLE, payload, t_turn=t_turn, budgets_after=budgets_after)
