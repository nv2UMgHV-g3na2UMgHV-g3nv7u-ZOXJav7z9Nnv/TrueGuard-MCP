"""
gateways/mcp_governance.py
==========================
Runtime permission enforcement for MCP tool calls, simulating the
TrueFoundry MCP Gateway's policy engine.

Design
------
The ``@require_permission`` decorator wraps an async MCP tool. It inspects
the declared permission level and behaves as follows:

* ``READ_ONLY``   → invokes the tool immediately (autonomous).
* ``HIGH_RISK``   → builds an :class:`ApprovalRequest`, pushes it onto an
                    in-memory approval queue AND (if configured) posts it to
                    Slack. Execution is paused awaiting a decision from the
                    approval callback endpoint. On approval, the wrapped
                    tool is invoked with ``approved_by=<operator>`` injected.

This revision raises the project's typed exceptions from :mod:`exceptions`
and logs via the standard library so :func:`logging_utils.configure_logging`
controls output uniformly.

The approval registry is a process-local singleton. In production this
would be backed by Redis / a durable queue; here it is intentionally
lightweight so the harness stays portable.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Final, TypeVar, cast

import httpx
from pydantic import BaseModel, Field

from config import settings
from exceptions import (
    ApprovalDeniedException,
    ApprovalTimeoutException,
    ErrorContext,
    create_error_context,
)

logger = logging.getLogger(__name__)


# ====================================================================== #
# Types                                                                   #
# ====================================================================== #
ToolCallable = Callable[..., Awaitable[dict[str, Any]]]
F = TypeVar("F", bound=ToolCallable)


class PermissionLevel(str, Enum):
    """Permission tiers recognised by the governance interceptor."""

    READ_ONLY = "READ_ONLY"
    HIGH_RISK = "HIGH_RISK"


class ApprovalStatus(str, Enum):
    """Lifecycle states of a pending approval request."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    EXPIRED = "EXPIRED"


# ====================================================================== #
# Models                                                                  #
# ====================================================================== #
class ApprovalRequest(BaseModel):
    """Payload emitted to the human-in-the-loop approval queue."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    permission_level: PermissionLevel
    arguments: dict[str, Any]
    requested_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc) + timedelta(minutes=15)
    )
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_by: str | None = None
    decided_at: datetime | None = None
    reason: str | None = None


class ApprovalDecision(BaseModel):
    """Decision body accepted by the approval registry."""

    request_id: str
    approved: bool
    decided_by: str
    reason: str | None = None


# Re-exported aliases so downstream imports from this module keep working.
ApprovalTimeoutError = ApprovalTimeoutException
__all__ = [
    "require_permission",
    "PermissionLevel",
    "ApprovalRequest",
    "ApprovalDecision",
    "ApprovalStatus",
    "ApprovalTimeoutError",
    "ApprovalDeniedException",
    "approval_registry",
]


# ====================================================================== #
# Approval registry                                                       #
# ====================================================================== #
class _ApprovalRegistry:
    """
    Process-local registry of pending approval requests.

    Each pending request owns an :class:`asyncio.Event`; the interceptor
    awaits that event, and the decision endpoint sets it. This keeps the
    interceptor purely async with no polling.
    """

    def __init__(self) -> None:
        self._requests: dict[str, ApprovalRequest] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._lock: Final[asyncio.Lock] = asyncio.Lock()

    async def register(self, request: ApprovalRequest) -> asyncio.Event:
        """Register a new request and return the event the caller should await."""
        async with self._lock:
            self._requests[request.request_id] = request
            event = asyncio.Event()
            self._events[request.request_id] = event
            return event

    async def decide(self, decision: ApprovalDecision) -> ApprovalRequest:
        """
        Record a decision and wake the awaiting interceptor.

        Raises
        ------
        KeyError
            If ``request_id`` is unknown or already resolved.
        """
        async with self._lock:
            request = self._requests.get(decision.request_id)
            if request is None:
                raise KeyError(f"Unknown approval request: {decision.request_id}")
            if request.status is not ApprovalStatus.PENDING:
                raise KeyError(
                    f"Request {decision.request_id} already resolved "
                    f"as {request.status.value}"
                )
            request.status = (
                ApprovalStatus.APPROVED if decision.approved else ApprovalStatus.DENIED
            )
            request.decided_by = decision.decided_by
            request.decided_at = datetime.now(tz=timezone.utc)
            request.reason = decision.reason
            event = self._events.get(request.request_id)
            if event is not None:
                event.set()
            return request

    async def expire(self, request_id: str) -> None:
        """Mark a request as expired and wake the awaiting interceptor."""
        async with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.status is not ApprovalStatus.PENDING:
                return
            request.status = ApprovalStatus.EXPIRED
            request.decided_at = datetime.now(tz=timezone.utc)
            event = self._events.get(request_id)
            if event is not None:
                event.set()

    async def get(self, request_id: str) -> ApprovalRequest | None:
        """Return a request by ID, or ``None`` if unknown."""
        async with self._lock:
            return self._requests.get(request_id)

    async def pending(self) -> list[ApprovalRequest]:
        """Return all requests still awaiting a decision."""
        async with self._lock:
            return [
                r for r in self._requests.values() if r.status is ApprovalStatus.PENDING
            ]


#: Singleton registry — imported by the FastAPI approval router as well.
approval_registry: Final[_ApprovalRegistry] = _ApprovalRegistry()


# ====================================================================== #
# Slack notification                                                      #
# ====================================================================== #
async def _notify_slack(request: ApprovalRequest) -> None:
    """
    POST a Block Kit payload to the configured Slack webhook.

    Failures are logged and swallowed — Slack delivery must never block
    or fail a governed tool call.
    """
    if not settings.is_slack_enabled:
        logger.info(
            "governance.slack.skipped request_id=%s reason=no-webhook",
            request.request_id,
        )
        return

    webhook = settings.get_slack_webhook()
    if webhook is None:  # pragma: no cover — invariant
        return

    payload = {
        "text": (
            f":rotating_light: *HIGH_RISK tool call pending approval*\n"
            f">*Tool:* `{request.tool_name}`\n"
            f">*Args:* `{request.arguments}`\n"
            f">*Request ID:* `{request.request_id}`"
        ),
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "TrueGuard-MCP :: Approval Required",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Tool:*\n`{request.tool_name}`"},
                    {
                        "type": "mrkdwn",
                        "text": f"*Level:*\n`{request.permission_level.value}`",
                    },
                    {"type": "mrkdwn", "text": f"*Args:*\n`{request.arguments}`"},
                    {
                        "type": "mrkdwn",
                        "text": f"*Expires:*\n{request.expires_at.isoformat()}",
                    },
                ],
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "style": "primary",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "url": (
                            f"{str(settings.approval_callback_url).rstrip('/')}"
                            f"?request_id={request.request_id}&approved=true"
                        ),
                    },
                    {
                        "type": "button",
                        "style": "danger",
                        "text": {"type": "plain_text", "text": "Deny"},
                        "url": (
                            f"{str(settings.approval_callback_url).rstrip('/')}"
                            f"?request_id={request.request_id}&approved=false"
                        ),
                    },
                ],
            },
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(webhook, json=payload)
            response.raise_for_status()
        logger.info("governance.slack.sent request_id=%s", request.request_id)
    except httpx.HTTPError as exc:
        logger.warning(
            "governance.slack.failed request_id=%s error=%s",
            request.request_id,
            exc,
        )


# ====================================================================== #
# Decorator                                                               #
# ====================================================================== #
def require_permission(level: str | PermissionLevel) -> Callable[[F], F]:
    """
    Wrap an async MCP tool with the TrueFoundry MCP Gateway permission model.

    Parameters
    ----------
    level:
        Either ``"READ_ONLY"`` or ``"HIGH_RISK"``.

    Returns
    -------
    Callable[[F], F]
        A decorator that returns a drop-in replacement for the wrapped
        coroutine function, preserving metadata via :func:`functools.wraps`.

    Raises
    ------
    ValueError
        At decoration time if ``level`` is not a recognised tier.
    TypeError
        At decoration time if the wrapped object is not an async function.
    """
    try:
        resolved = PermissionLevel(level)
    except ValueError as exc:
        raise ValueError(
            f"Unknown permission level {level!r}; "
            f"expected one of {[p.value for p in PermissionLevel]}"
        ) from exc

    def decorator(func: F) -> F:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(
                f"@require_permission can only wrap async functions; "
                f"{func.__qualname__} is sync."
            )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            tool_name = func.__name__

            # ---- READ_ONLY: pass straight through ---- #
            if resolved is PermissionLevel.READ_ONLY:
                logger.debug("governance.allow tool=%s level=%s", tool_name, resolved.value)
                return await func(*args, **kwargs)

            # ---- HIGH_RISK: build + enqueue approval request ---- #
            arguments = _bind_arguments(func, args, kwargs)
            request = ApprovalRequest(
                tool_name=tool_name,
                permission_level=resolved,
                arguments=arguments,
            )
            logger.warning(
                "governance.paused tool=%s request_id=%s arguments=%s",
                tool_name,
                request.request_id,
                arguments,
            )

            event = await approval_registry.register(request)
            await _notify_slack(request)

            timeout_s = (request.expires_at - request.requested_at).total_seconds()
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout_s)
            except asyncio.TimeoutError as exc:
                await approval_registry.expire(request.request_id)
                ctx = create_error_context(
                    trace_id=request.request_id,
                    span_id=request.request_id,
                    operation="governance.approval",
                    tool_name=tool_name,
                )
                logger.error(
                    "governance.timeout tool=%s request_id=%s",
                    tool_name,
                    request.request_id,
                )
                raise ApprovalTimeoutException(
                    timeout_seconds=timeout_s,
                    tool_name=tool_name,
                    context=ctx,
                ) from exc

            resolved_request = await approval_registry.get(request.request_id)
            if resolved_request is None:  # pragma: no cover — registry invariant
                raise RuntimeError(
                    f"Approval request {request.request_id} vanished from registry."
                )

            if resolved_request.status is ApprovalStatus.DENIED:
                ctx = create_error_context(
                    trace_id=request.request_id,
                    span_id=request.request_id,
                    operation="governance.approval",
                    tool_name=tool_name,
                    decided_by=resolved_request.decided_by or "unknown",
                )
                logger.warning(
                    "governance.denied tool=%s request_id=%s decided_by=%s",
                    tool_name,
                    request.request_id,
                    resolved_request.decided_by,
                )
                raise ApprovalDeniedException(
                    tool_name=tool_name,
                    reason=resolved_request.reason,
                    context=ctx,
                )

            if resolved_request.status is not ApprovalStatus.APPROVED:
                ctx = create_error_context(
                    trace_id=request.request_id,
                    span_id=request.request_id,
                    operation="governance.approval",
                    tool_name=tool_name,
                    final_status=resolved_request.status.value,
                )
                raise ApprovalTimeoutException(
                    timeout_seconds=timeout_s,
                    tool_name=tool_name,
                    context=ctx,
                )

            logger.info(
                "governance.approved tool=%s request_id=%s decided_by=%s",
                tool_name,
                request.request_id,
                resolved_request.decided_by,
            )

            # Inject `approved_by` only if the target signature accepts it.
            sig = inspect.signature(func)
            if "approved_by" in sig.parameters and "approved_by" not in kwargs:
                kwargs["approved_by"] = resolved_request.decided_by

            return await func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


# ====================================================================== #
# Helpers                                                                 #
# ====================================================================== #
def _bind_arguments(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Best-effort extraction of call arguments into a JSON-serialisable dict.

    Non-serialisable objects are coerced via :func:`repr`, so the approval
    payload never crashes on exotic values.
    """
    import json

    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        raw = dict(bound.arguments)
    except TypeError:
        raw = {**{f"arg_{i}": v for i, v in enumerate(args)}, **kwargs}

    safe: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "approved_by":
            continue
        try:
            json.dumps(value)
            safe[key] = value
        except (TypeError, ValueError):
            safe[key] = repr(value)
    return safe
