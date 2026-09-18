"""
gateways/approval_workflow.py
=============================
Human-in-the-loop approval workflow for TrueGuard-MCP.

Exposes a FastAPI router that:

1. Receives approval requests from the MCP Governance interceptor
   (:mod:`gateways.mcp_governance`) and renders a rich Slack Block-Kit card
   containing the incident summary, proposed tool + args, and estimated cost.

2. Receives interactive button callbacks at ``/api/v1/approval/callback``
   from Slack (or any HITL system), records the decision in the shared
   :class:`ApprovalStore`, and — on approval — invokes the pending MCP
   tool with the operator's identity.

This revision raises the project's typed :class:`ApprovalTimeoutException`
and :class:`ApprovalDeniedException` where applicable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Final, Literal

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from pydantic import BaseModel, Field

from config import settings
from exceptions import (
    ApprovalDeniedException,
    ApprovalTimeoutException,
    create_error_context,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/approval", tags=["approval"])


# ====================================================================== #
# Domain models                                                          #
# ====================================================================== #
class ToolInvocation(BaseModel):
    """A concrete MCP tool call awaiting human approval."""

    tool_name: str = Field(..., description="Registered MCP tool name.")
    arguments: dict[str, Any] = Field(default_factory=dict)
    risk_rationale: str = Field(..., description="Justification for the action.")


class IncidentSummary(BaseModel):
    """Condensed incident context surfaced to the approver."""

    incident_id: str
    title: str
    severity: Literal["SEV1", "SEV2", "SEV3", "SEV4"]
    affected_service: str
    detected_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    root_cause: str
    blast_radius: str


class ApprovalCard(BaseModel):
    """
    Canonical payload describing an approval request.

    Rendered into a Slack Block-Kit message by :func:`build_slack_payload`,
    and returned verbatim by the ``/pending`` endpoint for any UI client.
    """

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    incident: IncidentSummary
    invocation: ToolInvocation
    estimated_cost_usd: float = Field(..., ge=0.0)
    job_budget_usd: float = Field(..., gt=0.0)
    requested_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    expires_at: datetime


class ApprovalDecision(BaseModel):
    """Decision recorded by an approver."""

    request_id: str
    approved: bool
    decided_by: str
    reason: str | None = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class ApprovalResponse(BaseModel):
    """Response body returned to the calling UI."""

    request_id: str
    status: Literal["APPROVED", "DENIED", "EXPIRED", "UNKNOWN"]
    decided_by: str | None = None
    tool_result: dict[str, Any] | None = None
    error: str | None = None


# ====================================================================== #
# Approval store                                                         #
# ====================================================================== #
class _ApprovalStore:
    """
    In-memory registry of pending approval cards.

    Thread-safe and async-safe via a single :class:`asyncio.Lock`. This is
    deliberately process-local — production deployments should swap this
    for Redis.
    """

    def __init__(self) -> None:
        self._cards: dict[str, ApprovalCard] = {}
        self._decisions: dict[str, ApprovalDecision] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._lock: Final[asyncio.Lock] = asyncio.Lock()

    async def register(self, card: ApprovalCard) -> asyncio.Event:
        """Store a card and return the event awaiting its decision."""
        async with self._lock:
            self._cards[card.request_id] = card
            event = asyncio.Event()
            self._events[card.request_id] = event
            return event

    async def wait_for_decision(
        self, request_id: str, timeout_s: float
    ) -> ApprovalDecision:
        """
        Block until a decision arrives or the request expires.

        Raises
        ------
        ApprovalTimeoutException
            If no decision is recorded within ``timeout_s`` seconds.
        KeyError
            If ``request_id`` was never registered.
        """
        async with self._lock:
            if request_id not in self._cards:
                raise KeyError(f"Unknown approval request: {request_id}")
            event = self._events[request_id]

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            ctx = create_error_context(
                trace_id=request_id,
                span_id=request_id,
                operation="approval.wait_for_decision",
                timeout_s=timeout_s,
            )
            raise ApprovalTimeoutException(
                timeout_seconds=timeout_s,
                tool_name=self._cards[request_id].invocation.tool_name,
                context=ctx,
            ) from exc

        async with self._lock:
            decision = self._decisions.get(request_id)
        if decision is None:  # pragma: no cover — invariant
            raise RuntimeError(f"Event fired but no decision stored for {request_id}")
        return decision

    async def record(self, decision: ApprovalDecision) -> None:
        """Store a decision and wake any awaiting coroutine."""
        async with self._lock:
            if decision.request_id not in self._cards:
                raise KeyError(f"Unknown approval request: {decision.request_id}")
            if decision.request_id in self._decisions:
                prev = self._decisions[decision.request_id]
                raise KeyError(
                    f"Request {decision.request_id} already decided by {prev.decided_by}"
                )
            self._decisions[decision.request_id] = decision
            event = self._events.get(decision.request_id)
            if event is not None:
                event.set()

    async def get_card(self, request_id: str) -> ApprovalCard | None:
        async with self._lock:
            return self._cards.get(request_id)

    async def get_decision(self, request_id: str) -> ApprovalDecision | None:
        async with self._lock:
            return self._decisions.get(request_id)

    async def pending(self) -> list[ApprovalCard]:
        async with self._lock:
            return [
                card
                for rid, card in self._cards.items()
                if rid not in self._decisions
            ]


#: Module-level singleton shared by the agent loop and the HTTP handlers.
approval_store: Final[_ApprovalStore] = _ApprovalStore()


# ====================================================================== #
# Tool dispatch registry                                                 #
# ====================================================================== #
#: Signature every MCP tool executor must satisfy.
ToolExecutor = Callable[[dict[str, Any], str], Awaitable[dict[str, Any]]]

_tool_registry: Final[dict[str, ToolExecutor]] = {}


def register_tool(name: str, executor: ToolExecutor) -> None:
    """
    Register an MCP tool executor under ``name``.

    ``executor`` receives ``(arguments, approved_by)`` and must return the
    tool's JSON-serialisable result.
    """
    _tool_registry[name] = executor
    logger.info("approval.tool_registered tool=%s", name)


def get_tool(name: str) -> ToolExecutor:
    """Return the executor for ``name`` or raise :class:`KeyError`."""
    if name not in _tool_registry:
        raise KeyError(f"No executor registered for tool '{name}'")
    return _tool_registry[name]


# ====================================================================== #
# Slack Block-Kit rendering                                              #
# ====================================================================== #
def _pretty_json(value: Any) -> str:
    """Compact, deterministic JSON rendering for Slack code blocks."""
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def build_slack_payload(card: ApprovalCard) -> dict[str, Any]:
    """
    Render an :class:`ApprovalCard` into a Slack Block-Kit message.

    Approve / Deny buttons carry direct ``url`` fields pointing at the
    callback so the harness works with an incoming-webhook-only Slack app.
    """
    callback = str(settings.approval_callback_url).rstrip("/")
    approve_url = f"{callback}?request_id={card.request_id}&approved=true"
    deny_url = f"{callback}?request_id={card.request_id}&approved=false"

    usage_pct = min(100.0, (card.estimated_cost_usd / card.job_budget_usd) * 100.0)

    return {
        "text": (
            f":rotating_light: *Approval required* — "
            f"{card.incident.severity} on `{card.incident.affected_service}`"
        ),
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"TrueGuard :: {card.incident.severity} Approval",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Incident:*\n{card.incident.title}"},
                    {
                        "type": "mrkdwn",
                        "text": f"*Service:*\n`{card.incident.affected_service}`",
                    },
                    {"type": "mrkdwn", "text": f"*Root cause:*\n{card.incident.root_cause}"},
                    {"type": "mrkdwn", "text": f"*Blast radius:*\n{card.incident.blast_radius}"},
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Proposed tool:*\n`{card.invocation.tool_name}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Arguments:*\n```{_pretty_json(card.invocation.arguments)}```",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Rationale:*\n_{card.invocation.risk_rationale}_",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"*Estimated cost so far:*\n"
                            f"`${card.estimated_cost_usd:.4f}` / "
                            f"`${card.job_budget_usd:.2f}` ({usage_pct:.1f}%)"
                        ),
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Expires:*\n`{card.expires_at.isoformat()}`",
                    },
                ],
            },
            {
                "type": "actions",
                "block_id": f"trueguard_actions_{card.request_id}",
                "elements": [
                    {
                        "type": "button",
                        "style": "primary",
                        "text": {"type": "plain_text", "text": "✅ Approve & Execute"},
                        "url": approve_url,
                        "value": f"approve::{card.request_id}",
                        "action_id": "trueguard_approve",
                    },
                    {
                        "type": "button",
                        "style": "danger",
                        "text": {"type": "plain_text", "text": "🛑 Deny"},
                        "url": deny_url,
                        "value": f"deny::{card.request_id}",
                        "action_id": "trueguard_deny",
                    },
                ],
            },
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"Request ID: `{card.request_id}`"}],
            },
        ],
    }


# ====================================================================== #
# Slack delivery                                                         #
# ====================================================================== #
async def post_approval_card(card: ApprovalCard) -> bool:
    """
    POST the approval card to the configured Slack webhook.

    Returns ``True`` on successful delivery, ``False`` otherwise. Never
    raises — Slack delivery must not fail the approval workflow.
    """
    webhook = settings.get_slack_webhook()
    if webhook is None:
        logger.info("approval.slack.skipped request_id=%s reason=no-webhook", card.request_id)
        return False

    payload = build_slack_payload(card)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(webhook, json=payload)
            response.raise_for_status()
        logger.info("approval.slack.sent request_id=%s", card.request_id)
        return True
    except httpx.HTTPError as exc:
        logger.warning("approval.slack.failed request_id=%s error=%s", card.request_id, exc)
        return False


# ====================================================================== #
# Public API — used by the agent loop                                    #
# ====================================================================== #
async def request_approval(
    *,
    incident: IncidentSummary,
    invocation: ToolInvocation,
    estimated_cost_usd: float,
    job_budget_usd: float,
    timeout_s: float = 900.0,
) -> tuple[ApprovalCard, ApprovalDecision]:
    """
    Register an approval card, notify Slack, and block until decided.

    Raises
    ------
    ApprovalTimeoutException
        If no decision is recorded within ``timeout_s`` seconds.
    """
    card = ApprovalCard(
        incident=incident,
        invocation=invocation,
        estimated_cost_usd=estimated_cost_usd,
        job_budget_usd=job_budget_usd,
        expires_at=datetime.now(tz=timezone.utc).replace(microsecond=0)
        + timedelta(seconds=timeout_s),
    )
    await approval_store.register(card)
    await post_approval_card(card)

    logger.warning(
        "approval.awaiting_decision request_id=%s tool=%s",
        card.request_id,
        invocation.tool_name,
    )

    try:
        decision = await approval_store.wait_for_decision(card.request_id, timeout_s)
    except ApprovalTimeoutException:
        # Already typed by the store; re-raise as-is so callers see it.
        raise

    return card, decision


# ====================================================================== #
# FastAPI routes                                                         #
# ====================================================================== #
@router.get("/pending", response_model=list[ApprovalCard])
async def list_pending() -> list[ApprovalCard]:
    """Return every approval card awaiting a decision."""
    return await approval_store.pending()


@router.get("/{request_id}", response_model=ApprovalCard)
async def get_approval(request_id: str) -> ApprovalCard:
    """Return a specific approval card by ID."""
    card = await approval_store.get_card(request_id)
    if card is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown approval request: {request_id}",
        )
    return card


@router.get("/callback", response_model=ApprovalResponse)
async def approval_callback(
    background_tasks: BackgroundTasks,
    request_id: str = Query(..., description="Approval request ID."),
    approved: bool = Query(..., description="True to approve, False to deny."),
    operator: str = Query("slack-user", description="Operator identifier."),
    reason: str | None = Query(None, description="Optional denial reason."),
) -> ApprovalResponse:
    """
    Record a human decision and, on approval, dispatch the pending MCP tool.

    Idempotency-safe: a second call for the same ``request_id`` returns the
    recorded decision without re-executing.
    """
    card = await approval_store.get_card(request_id)
    if card is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown approval request: {request_id}",
        )

    existing = await approval_store.get_decision(request_id)
    if existing is not None:
        logger.info(
            "approval.callback_idempotent request_id=%s status=%s",
            request_id,
            "APPROVED" if existing.approved else "DENIED",
        )
        return ApprovalResponse(
            request_id=request_id,
            status="APPROVED" if existing.approved else "DENIED",
            decided_by=existing.decided_by,
        )

    decision = ApprovalDecision(
        request_id=request_id,
        approved=approved,
        decided_by=operator,
        reason=reason,
    )

    try:
        await approval_store.record(decision)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    logger.warning(
        "approval.decision_recorded request_id=%s approved=%s decided_by=%s",
        request_id,
        approved,
        operator,
    )

    if approved:
        background_tasks.add_task(_execute_tool_after_approval, card, operator)
        return ApprovalResponse(
            request_id=request_id,
            status="APPROVED",
            decided_by=operator,
        )

    return ApprovalResponse(
        request_id=request_id,
        status="DENIED",
        decided_by=operator,
    )


# ====================================================================== #
# Post-approval tool execution                                           #
# ====================================================================== #
async def _execute_tool_after_approval(card: ApprovalCard, operator: str) -> None:
    """
    Dispatch the approved tool via the shared executor registry.

    Errors are logged but not raised — the background task runs after the
    HTTP response has been returned.
    """
    try:
        executor = get_tool(card.invocation.tool_name)
    except KeyError as exc:
        logger.error(
            "approval.executor_missing request_id=%s tool=%s error=%s",
            card.request_id,
            card.invocation.tool_name,
            exc,
        )
        return

    try:
        result = await executor(card.invocation.arguments, operator)
        logger.warning(
            "approval.tool_executed request_id=%s tool=%s approved_by=%s result=%s",
            card.request_id,
            card.invocation.tool_name,
            operator,
            result,
        )
    except ApprovalDeniedException as exc:
        logger.warning(
            "approval.tool_denied request_id=%s tool=%s error=%s",
            card.request_id,
            card.invocation.tool_name,
            exc,
        )
    except Exception as exc:
        logger.exception(
            "approval.tool_execution_failed request_id=%s tool=%s error=%s",
            card.request_id,
            card.invocation.tool_name,
            exc,
        )
