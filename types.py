"""
types.py
========
Shared TypedDicts and enums for TrueGuard-MCP.

These are the wire-format types that cross module boundaries — the ones
mypy checks and the ones the dashboard consumes. Kept in a single module so
downstream code imports them from one place.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


# ====================================================================== #
# Trace / observability                                                  #
# ====================================================================== #
class SpanDict(TypedDict):
    span_id: str
    parent_id: str | None
    name: str
    kind: Literal["ALERT", "LLM", "TOOL", "GOVERNANCE", "OUTPUT"]
    status: Literal["RUNNING", "SUCCESS", "FALLBACK", "BLOCKED_BY_GOVERNANCE", "ERROR"]
    started_at: str
    ended_at: str | None
    latency_ms: float | None
    model_used: str | None
    tokens_in: int
    tokens_out: int
    cost_usd: float
    attributes: dict[str, Any]
    error: str | None


class TraceMeta(TypedDict):
    trace_id: str
    incident_id: str | None
    alert_text: str
    affected_service: str | None
    started_at: str
    ended_at: str | None
    is_complete: bool
    active_model: str | None
    fallback_executed: bool
    has_pending_governance: bool
    total_cost_usd: float
    total_tokens_in: int
    total_tokens_out: int
    span_count: int


class Timeline(TypedDict):
    trace: TraceMeta
    spans: list[SpanDict]


# ====================================================================== #
# Evaluation                                                             #
# ====================================================================== #
class EvalCheckDict(TypedDict):
    name: str
    passed: bool
    detail: str
    evidence: list[str]


class ScorecardDict(TypedDict):
    passed: bool
    score: str
    checks: list[EvalCheckDict]


# ====================================================================== #
# Scenarios                                                              #
# ====================================================================== #
class ScenarioResult(TypedDict, total=False):
    """Return type of every scenario entrypoint."""

    trace: TraceMeta
    scorecard: ScorecardDict
    halted: bool
    report: dict[str, Any] | None


# ====================================================================== #
# MCP tool arguments                                                     #
# ====================================================================== #
class ToolCallDict(TypedDict):
    name: str
    arguments: dict[str, Any]
