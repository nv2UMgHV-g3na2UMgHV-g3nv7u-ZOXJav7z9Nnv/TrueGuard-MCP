"""Exception hierarchy and serialization tests."""

from __future__ import annotations

from exceptions import (
    BudgetExceededException,
    ErrorContext,
    TrueGuardException,
    create_error_context,
)


def test_budget_exceeded_carries_financials() -> None:
    exc = BudgetExceededException(
        accumulated_usd=0.82,
        halt_usd=0.75,
        spent_on_model="gpt-4o",
    )
    assert exc.accumulated_usd == 0.82
    payload = exc.to_dict()
    assert payload["accumulated_usd"] == 0.82
    assert payload["halt_usd"] == 0.75
    assert "Budget limit exceeded" in payload["message"]


def test_error_context_round_trips() -> None:
    ctx = create_error_context(
        trace_id="t-1", span_id="s-1", operation="op", extra="x"
    )
    payload = ctx.to_dict()
    assert payload["trace_id"] == "t-1"
    assert payload["extra"] == "x"


def test_exception_chains_cause() -> None:
    cause = ValueError("root")
    exc = TrueGuardException("wrapper", cause=cause)
    assert "ValueError" in str(exc)
    assert exc.to_dict()["cause"]["type"] == "ValueError"
