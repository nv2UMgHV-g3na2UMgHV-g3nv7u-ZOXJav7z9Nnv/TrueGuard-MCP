"""Eval harness check tests."""

from __future__ import annotations

from datetime import datetime, timezone

from evals.eval_harness import EvalContext, check_hallucination, run_evals
from observability.tracer import Span, SpanKind, SpanStatus, Trace


def _empty_trace() -> Trace:
    trace = Trace(trace_id="t", incident_id="inc", alert_text="alert")
    trace.spans = [
        Span(
            trace_id="t", name="Output", kind=SpanKind.OUTPUT,
            status=SpanStatus.SUCCESS, ended_at=datetime.now(tz=timezone.utc),
        )
    ]
    return trace


def test_hallucination_flags_unknown_tool() -> None:
    ctx = EvalContext(
        trace=_empty_trace(),
        tool_calls=[{"name": "obliterate_db", "arguments": {}}],
    )
    result = check_hallucination(ctx)
    assert result.passed is False
    assert any("obliterate_db" in e for e in result.evidence)


def test_injection_and_pii_fail_together() -> None:
    ctx = EvalContext(
        trace=_empty_trace(),
        tool_calls=[{"name": "apply_database_migration",
                     "arguments": {"migration_id": "x; DROP TABLE orders; --"}}],
        outgoing_prompts=["contact ops@example.com"],
    )
    scorecard = run_evals(ctx)
    assert scorecard["passed"] is False
    assert scorecard["score"] == "1/3"
