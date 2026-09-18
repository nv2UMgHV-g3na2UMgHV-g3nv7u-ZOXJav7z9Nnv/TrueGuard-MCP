"""
observability
=============
Trace collection primitives for TrueGuard-MCP.

Exposes:

* :data:`tracer` — the singleton :class:`Tracer` used across the codebase.
* :class:`Trace` / :class:`Span` — the wire-format models the dashboard,
  agent, and eval harness all consume.
* :class:`SpanKind` / :class:`SpanStatus` — the enums used by every span
  producer.
* :func:`trace_to_timeline` — the serializer consumed by
  ``GET /api/v1/traces/{trace_id}``.
"""

from observability.tracer import (
    Span,
    SpanKind,
    SpanStatus,
    Tracer,
    Trace,
    trace_to_timeline,
    tracer,
)

__all__ = [
    "Tracer",
    "Trace",
    "Span",
    "SpanKind",
    "SpanStatus",
    "tracer",
    "trace_to_timeline",
]
