"""
orchestrator/steps.py
=====================
The three shared scenario steps: observe, diagnose, plan.

Each function is pure with respect to :class:`ScenarioContext` — it reads
from the context, emits spans into ``ctx.trace``, and returns a typed
result. Nothing here knows about palettes, colours, or CLI concerns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from exceptions import create_error_context
from observability.tracer import SpanKind, SpanStatus
from orchestrator.context import ScenarioContext

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ObservationResult:
    logs: dict[str, Any]
    health: dict[str, Any]


@dataclass(slots=True)
class DiagnosisResult:
    report_json: dict[str, Any]
    model_used: str
    fallback_executed: bool
    cost_usd: float


async def observe(ctx: ScenarioContext) -> ObservationResult:
    """Step 1: fetch logs + health via READ_ONLY MCP tools."""
    with ctx.trace.span if False else _span(ctx, "MCP Tool Call :: observe", SpanKind.TOOL):
        logs = await ctx.mcp_client(
            "fetch_server_logs",
            {"service_name": ctx.affected_service, "lines": 200},
        )
        health = await ctx.mcp_client(
            "check_container_health",
            {"service_name": ctx.affected_service},
        )
    return ObservationResult(logs=logs, health=health)


async def diagnose(
    ctx: ScenarioContext,
    observation: ObservationResult,
    *,
    system_prompt: str,
) -> DiagnosisResult:
    """Step 2: invoke the LLM router to produce the incident report JSON."""
    user_prompt = _format_prompt(ctx, observation)
    with _span(ctx, "LLM Routing Decision", SpanKind.LLM):
        response = await ctx.router.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=ctx.guardrail.default_model,
            temperature=0.0,
            tracker=ctx.guardrail.make_tracker(ctx.incident_id),
            extra_headers={"X-Incident-Id": ctx.incident_id},
        )
    parsed = _parse_json(response.completion.choices[0].message.content or "")
    return DiagnosisResult(
        report_json=parsed,
        model_used=response.model_used,
        fallback_executed=response.fallback_executed,
        cost_usd=response.cost_usd,
    )


async def plan(
    ctx: ScenarioContext,
    diagnosis: DiagnosisResult,
) -> list[dict[str, Any]]:
    """Step 3: extract the remediation plan from the diagnosis JSON."""
    with _span(ctx, "Governance / Evals Check", SpanKind.GOVERNANCE) as span:
        plan = diagnosis.report_json.get("remediation_plan", [])
        span.attributes["high_risk_steps"] = [
            s.get("tool_name") for s in plan if s.get("risk_level") == "HIGH_RISK"
        ]
    return plan


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #
class _span:
    """Tiny adapter so `with _span(ctx, name, kind) as s:` reads naturally."""

    def __init__(self, ctx: ScenarioContext, name: str, kind: SpanKind) -> None:
        self._ctx = ctx
        self._name = name
        self._kind = kind
        self._cm = None

    def __enter__(self):
        from observability.tracer import tracer
        self._cm = tracer.span(self._ctx.trace, self._name, self._kind)
        return self._cm.__enter__()

    def __exit__(self, exc_type, exc, tb):
        assert self._cm is not None
        return self._cm.__exit__(exc_type, exc, tb)


def _format_prompt(ctx: ScenarioContext, obs: ObservationResult) -> str:
    lines = obs.logs.get("entries", [])[-25:]
    log_block = "\n".join(lines) if lines else "(no log entries)"
    return (
        f"INCIDENT ID: {ctx.incident_id}\n"
        f"ALERT: {ctx.alert_text}\n"
        f"AFFECTED SERVICE: {ctx.affected_service}\n\n"
        f"OBSERVED EVIDENCE:\n"
        f"--- fetch_server_logs (last {len(lines)} lines) ---\n{log_block}\n\n"
        f"--- check_container_health ---\n"
        f"status={obs.health.get('status')} "
        f"cpu={obs.health.get('cpu_pct')}% "
        f"mem={obs.health.get('memory_pct')}% "
        f"restarts_24h={obs.health.get('restarts_24h')}\n\n"
        "Produce the IncidentReportResponse JSON now."
    )


def _parse_json(raw: str) -> dict[str, Any]:
    import json
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON object in model response: {raw[:300]!r}")
        return json.loads(text[start : end + 1])
