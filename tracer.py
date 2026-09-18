"""
observability/tracer.py
=======================
Lightweight OpenTelemetry-style trace collection for TrueGuard-MCP.

Every agent run produces a single :class:`Trace` — an ordered tree of
:class:`Span` objects rooted at the user alert:

    User Alert
      └── LLM Routing Decision
            └── MCP Tool Call
                  └── Governance / Evals Check
                        └── Output

Spans are stored in a bounded in-memory ring buffer, keyed by trace ID.
The dashboard polls :func:`get_trace_snapshot` and re-renders the waterfall
on every tick. A :class:`threading.Lock` guards the store so FastAPI worker
threads and asyncio tasks can read and write concurrently.

Design notes
------------
* The tracer deliberately does **not** depend on the ``opentelemetry-*``
  packages. It emits OTel-shaped data so that a real OTel exporter can be
  dropped in later without touching call sites.
* Every mutating helper acquires the store lock only for the duration of a
  dict/list mutation — never across an ``await``.
* The ring buffer is bounded (:data:`MAX_TRACES`), so the harness can run
  for hours during a demo without unbounded memory growth.
* Logging uses the standard library so :func:`logging_utils.configure_logging`
  controls output uniformly across the process.

Public surface
--------------
* :class:`Tracer` — span recorder with sync + async context managers.
* :data:`tracer` — the process-wide singleton.
* :func:`trace_to_timeline` — serializer for the dashboard.
* :data:`_store` — the raw store, re-exported as :data:`_store` for the
  instrumented agent runner in ``app.py`` (uses the live object rather than
  a snapshot to mutate spans in place).
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Final, Iterator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ====================================================================== #
# Constants                                                              #
# ====================================================================== #
MAX_TRACES: Final[int] = 500
MAX_SPANS_PER_TRACE: Final[int] = 500


# ====================================================================== #
# Enums + models                                                         #
# ====================================================================== #
class SpanStatus(str, Enum):
    """Lifecycle outcome of a span."""

    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FALLBACK = "FALLBACK"
    BLOCKED_BY_GOVERNANCE = "BLOCKED_BY_GOVERNANCE"
    ERROR = "ERROR"


class SpanKind(str, Enum):
    """Semantic kind of a span — mirrors OTel span kinds."""

    ALERT = "ALERT"
    LLM = "LLM"
    TOOL = "TOOL"
    GOVERNANCE = "GOVERNANCE"
    OUTPUT = "OUTPUT"


class Span(BaseModel):
    """
    A single unit of work within a trace.

    The shape mirrors an OpenTelemetry span as closely as is useful for a
    demo dashboard: identity, parent linkage, timing, and a flat bag of
    attributes carrying model, tokens, cost, and status.
    """

    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    trace_id: str
    name: str
    kind: SpanKind

    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    ended_at: datetime | None = None
    latency_ms: float | None = None

    model_used: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    status: SpanStatus = SpanStatus.RUNNING
    attributes: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class Trace(BaseModel):
    """An ordered collection of spans belonging to one agent run."""

    trace_id: str
    incident_id: str | None = None
    alert_text: str = ""
    affected_service: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    ended_at: datetime | None = None
    spans: list[Span] = Field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Derived views                                                       #
    # ------------------------------------------------------------------ #
    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def total_tokens_in(self) -> int:
        return sum(s.tokens_in for s in self.spans)

    @property
    def total_tokens_out(self) -> int:
        return sum(s.tokens_out for s in self.spans)

    @property
    def is_complete(self) -> bool:
        return self.ended_at is not None

    @property
    def active_model(self) -> str | None:
        """The most recent model observed in an LLM span."""
        for span in reversed(self.spans):
            if span.kind is SpanKind.LLM and span.model_used:
                return span.model_used
        return None

    @property
    def fallback_executed(self) -> bool:
        """True if any LLM span carries FALLBACK status."""
        return any(
            s.kind is SpanKind.LLM and s.status is SpanStatus.FALLBACK
            for s in self.spans
        )

    @property
    def has_pending_governance(self) -> bool:
        """True if any governance span is still running (awaiting human)."""
        return any(
            s.kind is SpanKind.GOVERNANCE and s.status is SpanStatus.RUNNING
            for s in self.spans
        )

    def span_by_id(self, span_id: str) -> Span | None:
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None


# ====================================================================== #
# Store                                                                  #
# ====================================================================== #
class _TraceStore:
    """
    Bounded, thread-safe in-memory trace registry.

    Traces are stored in an :class:`OrderedDict` so eviction is O(1) when
    the buffer is full. All reads return deep copies to prevent callers
    mutating shared state.
    """

    def __init__(self, max_traces: int = MAX_TRACES) -> None:
        self._traces: OrderedDict[str, Trace] = OrderedDict()
        self._lock: Final[threading.Lock] = threading.Lock()
        self._max = max_traces

    # ------------------------------------------------------------------ #
    # Mutations                                                           #
    # ------------------------------------------------------------------ #
    def put(self, trace: Trace) -> None:
        with self._lock:
            self._traces[trace.trace_id] = trace
            self._traces.move_to_end(trace.trace_id)
            while len(self._traces) > self._max:
                evicted_id, _ = self._traces.popitem(last=False)
                logger.debug("tracer.evicted trace_id=%s", evicted_id)

    def get_live(self, trace_id: str) -> Trace | None:
        """
        Return the *live* trace object (not a copy) for mutation by the
        tracer. Callers must hold no expectation of thread safety across
        the object's lifetime.
        """
        with self._lock:
            return self._traces.get(trace_id)

    def snapshot(self, trace_id: str) -> Trace | None:
        """Return a deep copy of the trace, safe for HTTP serialisation."""
        with self._lock:
            trace = self._traces.get(trace_id)
            return trace.model_copy(deep=True) if trace is not None else None

    def snapshot_all(self, limit: int = 50) -> list[Trace]:
        """Return the most recent ``limit`` traces as deep copies."""
        with self._lock:
            traces = list(self._traces.values())[-limit:]
            return [t.model_copy(deep=True) for t in traces]

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()


#: Module-level singleton used by the tracer, the agent, and the API.
#: Exported (with a leading underscore) so the instrumented runner in
#: ``app.py`` can fetch the live trace object rather than a snapshot.
_store: Final[_TraceStore] = _TraceStore()


# ---------------------------------------------------------------------- #
# Ambient trace propagation                                              #
# ---------------------------------------------------------------------- #
_CURRENT_TRACE: Final[contextvars.ContextVar[Trace | None]] = contextvars.ContextVar(
    "trueguard_current_trace", default=None
)


def set_current_trace(trace: Trace | None) -> contextvars.Token:
    """Install ``trace`` as the ambient trace for the current context."""
    return _CURRENT_TRACE.set(trace)


def get_current_trace() -> Trace | None:
    """Return the ambient trace, if any."""
    return _CURRENT_TRACE.get()


def reset_current_trace(token: contextvars.Token) -> None:
    """Restore the previous ambient trace."""
    _CURRENT_TRACE.reset(token)


# ====================================================================== #
# Tracer                                                                 #
# ====================================================================== #
class Tracer:
    """
    Async-friendly span recorder.

    Typical use::

        tracer = Tracer()
        trace = tracer.start_trace(alert_text="...", incident_id="inc-123")

        with tracer.span(trace, "LLM Routing Decision", SpanKind.LLM) as span:
            span.model_used = "gpt-4o"
            span.tokens_in = 1234
            span.cost_usd = 0.0123
            # span closed with SUCCESS on exit
    """

    # ------------------------------------------------------------------ #
    # Trace lifecycle                                                     #
    # ------------------------------------------------------------------ #
    def start_trace(
        self,
        *,
        alert_text: str,
        incident_id: str | None = None,
        affected_service: str | None = None,
    ) -> Trace:
        """Create and register a new trace, plus its root ALERT span."""
        trace = Trace(
            trace_id=uuid.uuid4().hex,
            incident_id=incident_id,
            alert_text=alert_text,
            affected_service=affected_service,
        )
        _store.put(trace)

        root = Span(
            trace_id=trace.trace_id,
            name="User Alert",
            kind=SpanKind.ALERT,
            attributes={"alert": alert_text, "service": affected_service},
        )
        self._append(trace, root)
        logger.info(
            "tracer.trace_started trace_id=%s incident_id=%s",
            trace.trace_id,
            incident_id,
        )
        return trace

    def end_trace(self, trace: Trace) -> None:
        """Mark the trace as complete and close the root span."""
        live = _store.get_live(trace.trace_id) or trace
        live.ended_at = datetime.now(tz=timezone.utc)
        for span in live.spans:
            if span.status is SpanStatus.RUNNING:
                self.close_span(
                    live,
                    span,
                    status=SpanStatus.ERROR,
                    error="trace ended before span completed",
                )
        logger.info(
            "tracer.trace_ended trace_id=%s span_count=%d cost=$%.4f",
            live.trace_id,
            len(live.spans),
            live.total_cost_usd,
        )

    # ------------------------------------------------------------------ #
    # Span lifecycle                                                      #
    # ------------------------------------------------------------------ #
    @contextmanager
    def span(
        self,
        trace: Trace,
        name: str,
        kind: SpanKind,
        *,
        parent_id: str | None = None,
    ) -> Iterator[Span]:
        """
        Synchronous context manager yielding a running span.

        On exit, if the caller has not explicitly closed the span, it is
        closed with :class:`SpanStatus.SUCCESS` (or :class:`SpanStatus.ERROR`
        if an exception propagated through the block).
        """
        live = _store.get_live(trace.trace_id) or trace
        span = Span(
            trace_id=live.trace_id,
            parent_id=parent_id or self._last_open_span_id(live),
            name=name,
            kind=kind,
        )
        self._append(live, span)
        exc: BaseException | None = None
        try:
            yield span
        except BaseException as e:
            exc = e
            raise
        finally:
            if span.ended_at is None:
                if exc is not None:
                    self.close_span(
                        live, span, status=SpanStatus.ERROR, error=str(exc)
                    )
                elif span.status is SpanStatus.RUNNING:
                    self.close_span(live, span, status=SpanStatus.SUCCESS)

    @asynccontextmanager
    async def aspan(
        self,
        trace: Trace,
        name: str,
        kind: SpanKind,
        *,
        parent_id: str | None = None,
    ) -> AsyncIterator[Span]:
        """Async variant of :meth:`span`."""
        live = _store.get_live(trace.trace_id) or trace
        span = Span(
            trace_id=live.trace_id,
            parent_id=parent_id or self._last_open_span_id(live),
            name=name,
            kind=kind,
        )
        self._append(live, span)
        exc: BaseException | None = None
        try:
            yield span
        except BaseException as e:
            exc = e
            raise
        finally:
            if span.ended_at is None:
                if exc is not None:
                    self.close_span(
                        live, span, status=SpanStatus.ERROR, error=str(exc)
                    )
                elif span.status is SpanStatus.RUNNING:
                    self.close_span(live, span, status=SpanStatus.SUCCESS)

    def close_span(
        self,
        trace: Trace,
        span: Span,
        *,
        status: SpanStatus | None = None,
        error: str | None = None,
        model_used: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Finalise a span, recording latency and any supplied metrics."""
        live = _store.get_live(trace.trace_id) or trace
        target = live.span_by_id(span.span_id) or span

        target.ended_at = datetime.now(tz=timezone.utc)
        target.latency_ms = (
            target.ended_at - target.started_at
        ).total_seconds() * 1000.0

        if status is not None:
            target.status = status
        elif target.status is SpanStatus.RUNNING:
            target.status = SpanStatus.SUCCESS

        if error is not None:
            target.error = error
        if model_used is not None:
            target.model_used = model_used
        if tokens_in is not None:
            target.tokens_in = tokens_in
        if tokens_out is not None:
            target.tokens_out = tokens_out
        if cost_usd is not None:
            target.cost_usd = cost_usd

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #
    def get_trace(self, trace_id: str) -> Trace | None:
        """Return a deep copy of the trace, safe to serialize."""
        return _store.snapshot(trace_id)

    def get_live_trace(self, trace_id: str) -> Trace | None:
        """
        Return the live (mutable) trace object.

        Used by instrumented runners that need to mutate spans in place.
        Callers must not retain a reference across ``await`` points without
        re-fetching, since the store lock is not held across the caller's
        coroutine lifetime.
        """
        return _store.get_live(trace_id)

    def list_traces(self, limit: int = 50) -> list[Trace]:
        """Return the most recent traces as deep copies."""
        return _store.snapshot_all(limit=limit)

    def clear(self) -> None:
        """Clear the store. Used by tests."""
        _store.clear()

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _append(trace: Trace, span: Span) -> None:
        if len(trace.spans) >= MAX_SPANS_PER_TRACE:
            logger.warning(
                "tracer.span_overflow trace_id=%s limit=%d",
                trace.trace_id,
                MAX_SPANS_PER_TRACE,
            )
            return
        trace.spans.append(span)

    @staticmethod
    def _last_open_span_id(trace: Trace) -> str | None:
        """Return the ID of the most recently started still-open span."""
        for span in reversed(trace.spans):
            if span.status is SpanStatus.RUNNING and span.ended_at is None:
                return span.span_id
        return trace.spans[-1].span_id if trace.spans else None


#: Module-level singleton — the canonical import for the rest of the codebase.
tracer: Final[Tracer] = Tracer()


# ====================================================================== #
# Serialisation helpers                                                  #
# ====================================================================== #
def trace_to_timeline(trace: Trace) -> dict[str, Any]:
    """
    Render a trace into the JSON shape the dashboard consumes.

    Returns a dict with:

    * ``trace``  — trace-level metadata + rollups
    * ``spans``  — flat list ordered by start time, ready for waterfall
                   rendering (the dashboard computes nesting from
                   ``parent_id``)
    """
    spans = sorted(trace.spans, key=lambda s: s.started_at)
    return {
        "trace": {
            "trace_id": trace.trace_id,
            "incident_id": trace.incident_id,
            "alert_text": trace.alert_text,
            "affected_service": trace.affected_service,
            "started_at": trace.started_at.isoformat(),
            "ended_at": trace.ended_at.isoformat() if trace.ended_at else None,
            "is_complete": trace.is_complete,
            "active_model": trace.active_model,
            "fallback_executed": trace.fallback_executed,
            "has_pending_governance": trace.has_pending_governance,
            "total_cost_usd": round(trace.total_cost_usd, 6),
            "total_tokens_in": trace.total_tokens_in,
            "total_tokens_out": trace.total_tokens_out,
            "span_count": len(trace.spans),
        },
        "spans": [
            {
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "name": s.name,
                "kind": s.kind.value,
                "status": s.status.value,
                "started_at": s.started_at.isoformat(),
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                "latency_ms": round(s.latency_ms, 2) if s.latency_ms is not None else None,
                "model_used": s.model_used,
                "tokens_in": s.tokens_in,
                "tokens_out": s.tokens_out,
                "cost_usd": round(s.cost_usd, 6),
                "attributes": s.attributes,
                "error": s.error,
            }
            for s in spans
        ],
    }


def get_trace_snapshot(trace_id: str) -> dict[str, Any] | None:
    """
    Convenience helper — returns the timeline dict for ``trace_id`` or
    ``None`` if the trace has been evicted.
    """
    trace = tracer.get_trace(trace_id)
    if trace is None:
        return None
    return trace_to_timeline(trace)
