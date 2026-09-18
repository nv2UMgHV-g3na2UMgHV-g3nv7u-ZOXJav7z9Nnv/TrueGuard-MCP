"""
agents/devops_agent.py
======================
Core reasoning + execution loop for the TrueGuard-MCP DevOps agent.

The agent ingests a production alert and runs a strict four-step protocol:

    1. **Observe**  — call autonomous READ_ONLY MCP tools
                      (``fetch_server_logs``, ``check_container_health``).
    2. **Diagnose** — ask the primary model (gpt-4o via TrueFoundry) for a
                      root-cause hypothesis with citations to the evidence.
    3. **Plan**     — ask the model to emit a structured remediation plan.
                      HIGH_RISK tools are NOT invoked directly; instead they
                      are routed through :mod:`gateways.approval_workflow`.
    4. **Report**   — emit a strictly-typed :class:`IncidentReportResponse`.

Every LLM call flows through :class:`gateways.llm_router.TrueFoundryRouter`
so failover and budget guardrails apply uniformly. Every read-only tool call
flows through the FastMCP client so the governance interceptor sees it.

This revision adopts the project's custom exception hierarchy
(:mod:`exceptions`), structured logging (:mod:`logging_utils`), and shared
types (:mod:`types`). Domain failures are raised as typed exceptions and
each phase is wrapped in a structured span.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Final, Literal

from pydantic import BaseModel, Field, field_validator

from config import settings
from exceptions import (
    ApprovalDeniedException,
    ApprovalTimeoutException,
    BudgetExceededException,
    ModelRouterException,
    TrueGuardException,
    create_error_context,
)
from gateways.approval_workflow import (
    IncidentSummary,
    ToolInvocation,
    register_tool,
    request_approval,
)
from gateways.llm_router import (
    BudgetExceededException as RouterBudgetExceededException,
    CostGuardrail,
    JobBudgetTracker,
    ModelRouterException as RouterModelRouterException,
    RoutedResponse,
    TrueFoundryRouter,
)
from logging_utils import span_context, trace_context
from observability.tracer import SpanKind, SpanStatus, tracer

logger = logging.getLogger(__name__)


# ====================================================================== #
# Output schema                                                          #
# ====================================================================== #
class Severity(str, Enum):
    """Incident severity taxonomy."""

    SEV1 = "SEV1"  # full outage
    SEV2 = "SEV2"  # major degradation
    SEV3 = "SEV3"  # partial degradation
    SEV4 = "SEV4"  # minor / informational


class RemediationStep(BaseModel):
    """A single proposed remediation action."""

    step_id: str = Field(..., description="Stable identifier, e.g. 'step-1'.")
    description: str
    tool_name: str | None = Field(
        default=None,
        description="MCP tool to invoke, or None for a manual/human step.",
    )
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    risk_level: Literal["READ_ONLY", "HIGH_RISK", "MANUAL"]
    rationale: str


class Evidence(BaseModel):
    """A single evidence item cited in the diagnosis."""

    source: str = Field(..., description="e.g. 'fetch_server_logs'.")
    observation: str
    supports_hypothesis: bool = True


class IncidentReportResponse(BaseModel):
    """Strict output contract for every agent run."""

    incident_id: str
    alert_title: str
    affected_service: str
    severity: Severity
    diagnosis: str = Field(..., min_length=20)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    remediation_plan: list[RemediationStep] = Field(default_factory=list)
    high_risk_actions_requiring_approval: list[str] = Field(default_factory=list)
    executed_actions: list[str] = Field(default_factory=list)
    denied_actions: list[str] = Field(default_factory=list)
    total_cost_usd: float = Field(..., ge=0.0)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    run_status: Literal["completed", "budget_halted", "failed"] = "completed"
    error: str | None = None

    @field_validator("high_risk_actions_requiring_approval")
    @classmethod
    def _no_dupes(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in v:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out


# ====================================================================== #
# System prompt                                                          #
# ====================================================================== #
SYSTEM_PROMPT: Final[str] = """\
You are TrueGuard, an autonomous Site Reliability Engineer operating inside a
production DevOps harness. You investigate alerts using read-only MCP tools
and produce a strict JSON remediation plan.

You MUST follow this four-step protocol on every run:

STEP 1 — OBSERVE (read-only, autonomous):
  Call `fetch_server_logs` and `check_container_health` for the affected
  service. Cite the exact observations you rely on.

STEP 2 — DIAGNOSE:
  Produce a concise root-cause hypothesis with a confidence score in [0, 1].
  Every claim must be backed by a specific log line or metric you observed.

STEP 3 — PLAN:
  Formulate a remediation plan as an ordered list of steps. Mark each step
  with a risk level:
    * READ_ONLY — safe to run autonomously.
    * HIGH_RISK — destructive; will be sent to a human approver.
    * MANUAL    — requires a human operator; no tool call.
  HIGH_RISK steps MUST name the exact MCP tool to invoke and its arguments.

STEP 4 — REPORT:
  Emit a single JSON object conforming EXACTLY to the IncidentReportResponse
  schema. No prose, no markdown fences, no commentary.

Available MCP tools:
  READ_ONLY:
    - fetch_server_logs(service_name: str, lines: int)
    - check_container_health(service_name: str)
  HIGH_RISK (approval required):
    - restart_service_container(service_name: str)
    - apply_database_migration(migration_id: str)

You are strictly forbidden from invoking HIGH_RISK tools directly. List them
in the plan; the harness routes them to a human.

Be terse. Be evidence-driven. Never speculate beyond the logs and metrics.
"""


# ====================================================================== #
# MCP client protocol                                                    #
# ====================================================================== #
#: Signature every READ_ONLY MCP dispatcher must satisfy.
McpClient = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


# ====================================================================== #
# Agent                                                                  #
# ====================================================================== #
class DevOpsAgent:
    """
    Orchestrates the four-step incident-response loop for a single alert.

    Parameters
    ----------
    router:
        Optional :class:`TrueFoundryRouter`. Constructed lazily if omitted.
    guardrail:
        Optional :class:`CostGuardrail`. Constructed lazily if omitted.
    mcp_client:
        Async callable ``(tool_name, arguments) -> dict`` used for READ_ONLY
        tool dispatch. Defaults to :func:`_default_mcp_client`, which invokes
        the tools in :mod:`mcp_servers.system_mcp` in-process — appropriate
        for single-process deployments and tests.
    """

    def __init__(
        self,
        *,
        router: TrueFoundryRouter | None = None,
        guardrail: CostGuardrail | None = None,
        mcp_client: McpClient | None = None,
    ) -> None:
        self.guardrail = guardrail or CostGuardrail(
            downgrade_threshold_usd=settings.job_budget_usd,
            halt_threshold_usd=settings.job_budget_usd * 1.5,
        )
        self.router = router or TrueFoundryRouter(guardrail=self.guardrail)
        self.mcp_client: McpClient = mcp_client or _default_mcp_client

        # Register post-approval executors once, so the approval workflow
        # can actually run the tools when a human clicks Approve.
        _register_high_risk_executors()

    # ------------------------------------------------------------------ #
    # Public entrypoint                                                   #
    # ------------------------------------------------------------------ #
    async def handle_alert(
        self,
        alert_text: str,
        *,
        affected_service: str,
        incident_id: str | None = None,
    ) -> IncidentReportResponse:
        """
        Run the full incident-response loop for ``alert_text``.

        Parameters
        ----------
        alert_text:
            Raw alert string, e.g. ``"Alert: High 500 error rate on
            payment-service-v2"``.
        affected_service:
            Logical service identifier extracted from the alert.
        incident_id:
            Optional stable ID; one is generated if omitted.

        Returns
        -------
        IncidentReportResponse
            Strictly-typed incident report. On budget halt, ``run_status`` is
            ``"budget_halted"`` and ``error`` carries the exception message.
            On any other domain failure, ``run_status`` is ``"failed"``.
        """
        incident_id = incident_id or f"inc-{uuid.uuid4().hex[:12]}"
        tracker = self.guardrail.make_tracker(incident_id)

        with trace_context(
            "agent.handle_alert",
            incident_id=incident_id,
            service=affected_service,
        ) as trace:
            logger.info(
                "agent.start incident=%s service=%s alert=%s",
                incident_id,
                affected_service,
                alert_text[:200],
            )

            try:
                report = await self._run_protocol(
                    incident_id=incident_id,
                    alert_text=alert_text,
                    affected_service=affected_service,
                    tracker=tracker,
                    trace=trace,
                )
            except RouterBudgetExceededException as exc:
                logger.error(
                    "agent.budget_halted incident=%s accumulated=%.4f halt=%.4f",
                    incident_id,
                    exc.accumulated_usd,
                    exc.halt_usd,
                )
                return IncidentReportResponse(
                    incident_id=incident_id,
                    alert_title=alert_text,
                    affected_service=affected_service,
                    severity=Severity.SEV2,
                    diagnosis=(
                        "Investigation halted: job budget exhausted before a "
                        "root cause could be confirmed."
                    ),
                    confidence=0.0,
                    total_cost_usd=tracker.accumulated_usd,
                    run_status="budget_halted",
                    error=str(exc),
                )
            except (
                ModelRouterException,
                RouterModelRouterException,
                ApprovalTimeoutException,
                ApprovalDeniedException,
            ) as exc:
                logger.error("agent.domain_failure incident=%s err=%s", incident_id, exc)
                return IncidentReportResponse(
                    incident_id=incident_id,
                    alert_title=alert_text,
                    affected_service=affected_service,
                    severity=Severity.SEV2,
                    diagnosis=f"Agent run failed: {exc}",
                    confidence=0.0,
                    total_cost_usd=tracker.accumulated_usd,
                    run_status="failed",
                    error=str(exc),
                )
            except TrueGuardException as exc:
                logger.exception("agent.trueguard_failure incident=%s", incident_id)
                return IncidentReportResponse(
                    incident_id=incident_id,
                    alert_title=alert_text,
                    affected_service=affected_service,
                    severity=Severity.SEV2,
                    diagnosis=f"Agent run failed: {exc}",
                    confidence=0.0,
                    total_cost_usd=tracker.accumulated_usd,
                    run_status="failed",
                    error=str(exc),
                )
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("agent.unexpected_failure incident=%s", incident_id)
                return IncidentReportResponse(
                    incident_id=incident_id,
                    alert_title=alert_text,
                    affected_service=affected_service,
                    severity=Severity.SEV2,
                    diagnosis=f"Agent run failed: {exc}",
                    confidence=0.0,
                    total_cost_usd=tracker.accumulated_usd,
                    run_status="failed",
                    error=str(exc),
                )

            logger.info(
                "agent.complete incident=%s status=%s cost=$%.4f",
                incident_id,
                report.run_status,
                report.total_cost_usd,
            )
            return report

    # ------------------------------------------------------------------ #
    # Protocol internals                                                  #
    # ------------------------------------------------------------------ #
    async def _run_protocol(
        self,
        *,
        incident_id: str,
        alert_text: str,
        affected_service: str,
        tracker: JobBudgetTracker,
        trace: Any,
    ) -> IncidentReportResponse:
        """Execute the four-step protocol end-to-end."""

        # ---------- STEP 1: OBSERVE ---------- #
        with span_context("observe", trace) as obs_span:
            logs, health = await self._observe(affected_service)
            obs_span.log_metric("log_lines", logs.get("lines_returned", 0))
            obs_span.log_metric("health_status", health.get("status", "unknown"))

        # ---------- STEP 2 + 3: DIAGNOSE + PLAN ---------- #
        with span_context("diagnose_and_plan", trace) as llm_span:
            raw_report, routed = await self._diagnose_and_plan(
                incident_id=incident_id,
                alert_text=alert_text,
                affected_service=affected_service,
                logs=logs,
                health=health,
                tracker=tracker,
                trace=trace,
            )
            llm_span.log_metric("model_used", routed.model_used)
            llm_span.log_metric("fallback_executed", routed.fallback_executed)
            llm_span.log_metric("cost_usd", routed.cost_usd)

        # ---------- STEP 3b: GOVERNANCE for HIGH_RISK steps ---------- #
        remediation_steps = [
            RemediationStep(**step) for step in raw_report.get("remediation_plan", [])
        ]
        high_risk_steps = [
            s for s in remediation_steps if s.risk_level == "HIGH_RISK" and s.tool_name
        ]

        executed: list[str] = []
        denied: list[str] = []
        pending_approval: list[str] = []

        for step in high_risk_steps:
            assert step.tool_name is not None
            pending_approval.append(step.tool_name)

            with span_context(
                f"approve::{step.tool_name}",
                trace,
                tool=step.tool_name,
            ) as approval_span:
                incident_summary = IncidentSummary(
                    incident_id=incident_id,
                    title=alert_text,
                    severity=Severity(raw_report.get("severity", "SEV2")),
                    affected_service=affected_service,
                    root_cause=raw_report.get("diagnosis", "Unspecified"),
                    blast_radius=_summarise_blast_radius(raw_report),
                )
                invocation = ToolInvocation(
                    tool_name=step.tool_name,
                    arguments=step.tool_arguments,
                    risk_rationale=step.rationale,
                )

                try:
                    _, decision = await request_approval(
                        incident=incident_summary,
                        invocation=invocation,
                        estimated_cost_usd=tracker.accumulated_usd,
                        job_budget_usd=tracker.halt_threshold_usd,
                    )
                except TimeoutError as exc:
                    logger.warning(
                        "agent.approval_timeout incident=%s tool=%s err=%s",
                        incident_id,
                        step.tool_name,
                        exc,
                    )
                    denied.append(f"{step.tool_name} (expired)")
                    approval_span.log_metric("decision", "EXPIRED")
                    continue

                if decision.approved:
                    executed.append(
                        f"{step.tool_name} (approved by {decision.decided_by})"
                    )
                    approval_span.log_metric("decision", "APPROVED")
                    approval_span.log_metric("decided_by", decision.decided_by)
                    logger.warning(
                        "agent.approved incident=%s tool=%s by=%s",
                        incident_id,
                        step.tool_name,
                        decision.decided_by,
                    )
                else:
                    denied.append(
                        f"{step.tool_name} (denied by {decision.decided_by})"
                    )
                    approval_span.log_metric("decision", "DENIED")
                    approval_span.log_metric("decided_by", decision.decided_by)
                    logger.warning(
                        "agent.denied incident=%s tool=%s by=%s",
                        incident_id,
                        step.tool_name,
                        decision.decided_by,
                    )

        # ---------- STEP 4: REPORT ---------- #
        with span_context("report", trace) as report_span:
            report = IncidentReportResponse(
                incident_id=incident_id,
                alert_title=alert_text,
                affected_service=affected_service,
                severity=Severity(raw_report.get("severity", "SEV2")),
                diagnosis=raw_report.get(
                    "diagnosis",
                    "Diagnosis unavailable — model did not emit a diagnosis field.",
                ),
                confidence=float(raw_report.get("confidence", 0.5)),
                evidence=[Evidence(**e) for e in raw_report.get("evidence", [])],
                remediation_plan=remediation_steps,
                high_risk_actions_requiring_approval=pending_approval,
                executed_actions=executed,
                denied_actions=denied,
                total_cost_usd=tracker.accumulated_usd,
                run_status="completed",
            )
            report_span.log_metric("severity", report.severity.value)
            report_span.log_metric("cost_usd", report.total_cost_usd)
            report_span.log_metric("executed", len(executed))
            report_span.log_metric("denied", len(denied))

        return report

    # ------------------------------------------------------------------ #
    # Phase implementations                                               #
    # ------------------------------------------------------------------ #
    async def _observe(
        self, affected_service: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Step 1: fetch logs and health via the READ_ONLY MCP client.

        Both calls are issued concurrently; a failure in either is wrapped
        in :class:`TrueGuardException` with structured context.
        """
        try:
            logs, health = await _gather(
                self.mcp_client(
                    "fetch_server_logs",
                    {"service_name": affected_service, "lines": 200},
                ),
                self.mcp_client(
                    "check_container_health",
                    {"service_name": affected_service},
                ),
            )
        except Exception as exc:
            ctx = create_error_context(
                trace_id="pending",
                span_id="observe",
                operation="observe",
                service=affected_service,
            )
            raise TrueGuardException(
                f"MCP observation phase failed for '{affected_service}'",
                context=ctx,
                cause=exc,
            ) from exc

        logger.info(
            "agent.observed service=%s log_lines=%s health=%s",
            affected_service,
            logs.get("lines_returned"),
            health.get("status"),
        )
        return logs, health

    async def _diagnose_and_plan(
        self,
        *,
        incident_id: str,
        alert_text: str,
        affected_service: str,
        logs: dict[str, Any],
        health: dict[str, Any],
        tracker: JobBudgetTracker,
        trace: Any,
    ) -> tuple[dict[str, Any], RoutedResponse]:
        """
        Steps 2 + 3: single LLM call producing the diagnosis and plan.

        Kept as one call so the diagnosis and plan stay coherent and the
        budget guardrail only has to reason about one dispatch per turn.
        """
        evidence_block = _format_evidence(logs, health)
        user_prompt = (
            f"INCIDENT ID: {incident_id}\n"
            f"ALERT: {alert_text}\n"
            f"AFFECTED SERVICE: {affected_service}\n\n"
            f"OBSERVED EVIDENCE:\n{evidence_block}\n\n"
            "Produce the IncidentReportResponse JSON now."
        )

        try:
            routed: RoutedResponse = await self.router.complete(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.guardrail.default_model,
                temperature=0.0,
                tracker=tracker,
                extra_headers={"X-Incident-Id": incident_id},
            )
        except RouterModelRouterException as exc:
            raise ModelRouterException(
                primary_model=self.guardrail.default_model,
                fallback_model=self.guardrail.cheap_model,
                primary_error=exc,
                fallback_error=exc,
            ) from exc

        raw = routed.completion.choices[0].message.content or ""
        parsed = _parse_report_json(raw)

        logger.info(
            "agent.diagnosed incident=%s model=%s fallback=%s high_risk=%d",
            incident_id,
            routed.model_used,
            routed.fallback_executed,
            len(
                [
                    s
                    for s in parsed.get("remediation_plan", [])
                    if s.get("risk_level") == "HIGH_RISK"
                ]
            ),
        )
        return parsed, routed


# ====================================================================== #
# Module-level helpers                                                   #
# ====================================================================== #
async def _gather(*aws: Awaitable[Any]) -> tuple[Any, ...]:
    """Small alias for :func:`asyncio.gather` that preserves order."""
    import asyncio

    return tuple(await asyncio.gather(*aws))


def _format_evidence(logs: dict[str, Any], health: dict[str, Any]) -> str:
    """Render the observed logs + health into a compact prompt block."""
    log_entries = logs.get("entries", [])[-25:]  # cap prompt size
    log_text = "\n".join(log_entries) if log_entries else "(no log entries)"
    return (
        f"--- fetch_server_logs (last {len(log_entries)} lines) ---\n"
        f"{log_text}\n\n"
        f"--- check_container_health ---\n"
        f"status={health.get('status')} "
        f"cpu={health.get('cpu_pct')}% "
        f"mem={health.get('memory_pct')}% "
        f"restarts_24h={health.get('restarts_24h')}"
    )


def _summarise_blast_radius(report: dict[str, Any]) -> str:
    """Derive a short blast-radius string from the model's report."""
    affected = report.get("affected_service", "unknown")
    severity = report.get("severity", "SEV2")
    return f"{severity} impact on {affected} and upstream dependents."


def _parse_report_json(raw: str) -> dict[str, Any]:
    """
    Extract the JSON object from the model's response.

    Tolerates accidental markdown fences and leading prose. Raises
    :class:`ValueError` if no parseable object can be recovered.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model response contained no JSON object: {raw[:300]!r}")
        return json.loads(text[start : end + 1])


# ====================================================================== #
# Default in-process MCP client                                          #
# ====================================================================== #
async def _default_mcp_client(
    tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """
    Dispatch a READ_ONLY tool against the in-process FastMCP server.

    Imports are local so that importing :mod:`agents.devops_agent` does not
    force the MCP server module to be importable in every deployment
    topology (e.g. when the agent runs as a separate process from the MCP
    server). Replace with an HTTP client to a remote FastMCP instance in
    distributed deployments.
    """
    from mcp_servers.system_mcp import (  # local import — see docstring
        check_container_health,
        fetch_server_logs,
    )

    dispatch: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
        "fetch_server_logs": fetch_server_logs,
        "check_container_health": check_container_health,
    }
    if tool_name not in dispatch:
        raise ValueError(f"Tool '{tool_name}' is not a READ_ONLY MCP tool.")
    fn = dispatch[tool_name]
    return await fn(**arguments)


# ====================================================================== #
# Post-approval executors                                                #
# ====================================================================== #
def _register_high_risk_executors() -> None:
    """
    Register HIGH_RISK executors with the approval workflow.

    When a human approves a HIGH_RISK tool, the approval workflow's
    background task calls the executor with ``(arguments, approved_by)``.
    The executor here dispatches the real tool from
    :mod:`mcp_servers.system_mcp`, passing ``approved_by`` so the tool's
    in-body guard is satisfied.
    """

    async def _restart(
        arguments: dict[str, Any], approved_by: str
    ) -> dict[str, Any]:
        from mcp_servers.system_mcp import restart_service_container

        return await restart_service_container(
            service_name=str(arguments["service_name"]),
            approved_by=approved_by,
        )

    async def _migrate(
        arguments: dict[str, Any], approved_by: str
    ) -> dict[str, Any]:
        from mcp_servers.system_mcp import apply_database_migration

        return await apply_database_migration(
            migration_id=str(arguments["migration_id"]),
            approved_by=approved_by,
        )

    register_tool("restart_service_container", _restart)
    register_tool("apply_database_migration", _migrate)
