"""
mcp_servers/system_mcp.py
=========================
FastMCP server exposing four system-operations tools to the agent runtime.

Permission tiers
----------------
* READ_ONLY   → ``fetch_server_logs``, ``check_container_health``
                (autonomous; no human approval)
* HIGH_RISK   → ``restart_service_container``, ``apply_database_migration``
                (destructive; routed through the MCP Governance interceptor)

The permission tier of each tool is enforced *outside* this module by the
``@require_permission`` decorator in :mod:`gateways.mcp_governance`. This
file deliberately contains only business logic, so the tools remain
unit-testable without the governance runtime attached.

For every invocation the tool:

1. Validates inputs against :class:`TrueGuardException`-derived errors so
   callers get structured, serialisable failures.
2. Wraps its backend call and re-raises any transport error as
   :class:`TrueGuardException` with a populated :class:`ErrorContext`.
3. Logs via the standard library so :func:`logging_utils.configure_logging`
   controls output uniformly.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any, Final, Literal

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from exceptions import (
    ErrorContext,
    TrueGuardException,
    create_error_context,
)
from gateways.mcp_governance import require_permission

logger = logging.getLogger(__name__)


# ====================================================================== #
# FastMCP server instance                                                #
# ====================================================================== #
mcp: Final[FastMCP] = FastMCP(
    name="trueguard-system-mcp",
    instructions=(
        "System operations MCP for the TrueGuard DevOps harness. "
        "Read-only tools may be called autonomously; destructive tools are "
        "gated by the TrueFoundry MCP Gateway and require human approval."
    ),
)


# ====================================================================== #
# Typed response models                                                  #
# ====================================================================== #
class LogFetchResult(BaseModel):
    """Structured result returned by :func:`fetch_server_logs`."""

    service_name: str
    lines_requested: int
    lines_returned: int
    entries: list[str] = Field(default_factory=list)
    fetched_at: datetime


class ContainerHealth(BaseModel):
    """Structured result returned by :func:`check_container_health`."""

    service_name: str
    status: Literal["healthy", "degraded", "unhealthy", "unknown"]
    cpu_pct: float
    memory_pct: float
    restarts_24h: int
    checked_at: datetime


class RestartResult(BaseModel):
    """Structured result returned by :func:`restart_service_container`."""

    service_name: str
    previous_status: str
    new_status: str
    restarted_at: datetime
    approved_by: str | None = None


class MigrationResult(BaseModel):
    """Structured result returned by :func:`apply_database_migration`."""

    migration_id: str
    applied: bool
    duration_ms: int
    applied_at: datetime
    approved_by: str | None = None


# ====================================================================== #
# Simulated backends (replace with real infra calls in production)        #
# ====================================================================== #
_LOG_LEVELS: Final[tuple[str, ...]] = ("INFO", "WARN", "ERROR", "DEBUG")


async def _simulate_latency(base_ms: int = 120, jitter_ms: int = 80) -> None:
    """Await a small randomised latency to mimic real network I/O."""
    await asyncio.sleep((base_ms + random.randint(0, jitter_ms)) / 1000)


def _now() -> datetime:
    """UTC timestamp helper."""
    return datetime.now(tz=timezone.utc)


def _error_context(
    *,
    tool_name: str,
    service_name: str | None = None,
    migration_id: str | None = None,
    **extra: Any,
) -> ErrorContext:
    """
    Build a populated :class:`ErrorContext` for tool-level failures.

    The correlation IDs are synthesised because the governance interceptor
    owns the *trace* / *span* identifiers in the real system. In tests, the
    tool's own name is sufficient for structured triage.
    """
    metadata: dict[str, Any] = {
        "tool_name": tool_name,
        **extra,
    }
    if service_name is not None:
        metadata["service_name"] = service_name
    if migration_id is not None:
        metadata["migration_id"] = migration_id
    return create_error_context(
        trace_id=tool_name,
        span_id=tool_name,
        operation=tool_name,
        **metadata,
    )


# ====================================================================== #
# Tool 1 — fetch_server_logs (READ_ONLY)                                  #
# ====================================================================== #
@mcp.tool(
    name="fetch_server_logs",
    description="Fetch the most recent log lines for a named service. Read-only.",
)
@require_permission(level="READ_ONLY")
async def fetch_server_logs(service_name: str, lines: int = 100) -> dict[str, Any]:
    """
    Fetch the tail of a service's log stream.

    Parameters
    ----------
    service_name:
        Logical service identifier, e.g. ``"payments-api"``.
    lines:
        Number of trailing log lines to return. Must be ``1 <= lines <= 5000``.

    Returns
    -------
    dict
        Serialised :class:`LogFetchResult`.

    Raises
    ------
    TrueGuardException
        If ``service_name`` is blank, ``lines`` is out of range, or the
        simulated log backend fails.
    """
    if not service_name or not service_name.strip():
        raise TrueGuardException(
            "service_name must be a non-empty string.",
            context=_error_context(tool_name="fetch_server_logs"),
        )
    if not 1 <= lines <= 5000:
        raise TrueGuardException(
            f"lines must be between 1 and 5000, got {lines}.",
            context=_error_context(
                tool_name="fetch_server_logs",
                service_name=service_name,
                requested_lines=lines,
            ),
        )

    logger.info("tool.fetch_server_logs.start service=%s lines=%d", service_name, lines)
    try:
        await _simulate_latency()
        entries: list[str] = [
            f"{_now().isoformat()} {random.choice(_LOG_LEVELS)} "
            f"[{service_name}] simulated log line {i}"
            for i in range(lines)
        ]
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("tool.fetch_server_logs.error service=%s", service_name)
        raise TrueGuardException(
            f"Failed to fetch logs for '{service_name}'",
            context=_error_context(
                tool_name="fetch_server_logs",
                service_name=service_name,
            ),
            cause=exc,
        ) from exc

    result = LogFetchResult(
        service_name=service_name,
        lines_requested=lines,
        lines_returned=len(entries),
        entries=entries,
        fetched_at=_now(),
    )
    logger.info(
        "tool.fetch_server_logs.ok service=%s count=%d",
        service_name,
        len(entries),
    )
    return result.model_dump(mode="json")


# ====================================================================== #
# Tool 2 — check_container_health (READ_ONLY)                             #
# ====================================================================== #
@mcp.tool(
    name="check_container_health",
    description="Inspect health, CPU, memory and restart count for a service. Read-only.",
)
@require_permission(level="READ_ONLY")
async def check_container_health(service_name: str) -> dict[str, Any]:
    """
    Inspect the runtime health of a service container.

    Parameters
    ----------
    service_name:
        Logical service identifier.

    Returns
    -------
    dict
        Serialised :class:`ContainerHealth`.

    Raises
    ------
    TrueGuardException
        If ``service_name`` is blank or the simulated health backend fails.
    """
    if not service_name or not service_name.strip():
        raise TrueGuardException(
            "service_name must be a non-empty string.",
            context=_error_context(tool_name="check_container_health"),
        )

    logger.info("tool.check_container_health.start service=%s", service_name)
    try:
        await _simulate_latency(base_ms=80)
        cpu = round(random.uniform(5.0, 95.0), 2)
        mem = round(random.uniform(10.0, 95.0), 2)
        status: Literal["healthy", "degraded", "unhealthy", "unknown"]
        if cpu > 85.0 or mem > 90.0:
            status = "unhealthy"
        elif cpu > 70.0 or mem > 75.0:
            status = "degraded"
        else:
            status = "healthy"
        payload = ContainerHealth(
            service_name=service_name,
            status=status,
            cpu_pct=cpu,
            memory_pct=mem,
            restarts_24h=random.randint(0, 7),
            checked_at=_now(),
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("tool.check_container_health.error service=%s", service_name)
        raise TrueGuardException(
            f"Failed to check health for '{service_name}'",
            context=_error_context(
                tool_name="check_container_health",
                service_name=service_name,
            ),
            cause=exc,
        ) from exc

    logger.info("tool.check_container_health.ok service=%s status=%s", service_name, status)
    return payload.model_dump(mode="json")


# ====================================================================== #
# Tool 3 — restart_service_container (HIGH_RISK)                          #
# ====================================================================== #
@mcp.tool(
    name="restart_service_container",
    description=(
        "Restart a service container. DESTRUCTIVE — brief downtime expected. "
        "Requires human approval via the MCP Governance Gateway."
    ),
)
@require_permission(level="HIGH_RISK")
async def restart_service_container(
    service_name: str,
    *,
    approved_by: str | None = None,
) -> dict[str, Any]:
    """
    Restart a service container.

    The ``approved_by`` parameter is injected by the governance interceptor
    once a human has approved the request. Direct callers that bypass the
    interceptor will hit the guard below and be rejected.

    Parameters
    ----------
    service_name:
        Logical service identifier.
    approved_by:
        Identifier (email / Slack handle) of the approving operator.

    Returns
    -------
    dict
        Serialised :class:`RestartResult`.

    Raises
    ------
    TrueGuardException
        If ``service_name`` is blank, the call bypassed the governance
        interceptor (no ``approved_by``), or the simulated restart fails.
    """
    if not service_name or not service_name.strip():
        raise TrueGuardException(
            "service_name must be a non-empty string.",
            context=_error_context(tool_name="restart_service_container"),
        )
    if not approved_by:
        raise TrueGuardException(
            "restart_service_container requires human approval; "
            "invoke it through the MCP Governance interceptor.",
            context=_error_context(
                tool_name="restart_service_container",
                service_name=service_name,
                missing="approved_by",
            ),
        )

    logger.warning(
        "tool.restart_service_container.start service=%s approved_by=%s",
        service_name,
        approved_by,
    )
    try:
        await _simulate_latency(base_ms=350)
        result = RestartResult(
            service_name=service_name,
            previous_status="running",
            new_status="running",
            restarted_at=_now(),
            approved_by=approved_by,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("tool.restart_service_container.error service=%s", service_name)
        raise TrueGuardException(
            f"Failed to restart '{service_name}'",
            context=_error_context(
                tool_name="restart_service_container",
                service_name=service_name,
                approved_by=approved_by,
            ),
            cause=exc,
        ) from exc

    logger.warning("tool.restart_service_container.ok service=%s", service_name)
    return result.model_dump(mode="json")


# ====================================================================== #
# Tool 4 — apply_database_migration (HIGH_RISK)                           #
# ====================================================================== #
@mcp.tool(
    name="apply_database_migration",
    description=(
        "Apply a database migration by ID. DESTRUCTIVE — may mutate schema/data. "
        "Requires human approval via the MCP Governance Gateway."
    ),
)
@require_permission(level="HIGH_RISK")
async def apply_database_migration(
    migration_id: str,
    *,
    approved_by: str | None = None,
) -> dict[str, Any]:
    """
    Apply a database migration.

    Parameters
    ----------
    migration_id:
        Identifier of the migration, e.g. ``"2024_11_add_orders_index"``.
    approved_by:
        Injected by the governance interceptor after human approval.

    Returns
    -------
    dict
        Serialised :class:`MigrationResult`.

    Raises
    ------
    TrueGuardException
        If ``migration_id`` is blank, the call bypassed the governance
        interceptor, or the simulated migration fails.
    """
    if not migration_id or not migration_id.strip():
        raise TrueGuardException(
            "migration_id must be a non-empty string.",
            context=_error_context(tool_name="apply_database_migration"),
        )
    if not approved_by:
        raise TrueGuardException(
            "apply_database_migration requires human approval; "
            "invoke it through the MCP Governance interceptor.",
            context=_error_context(
                tool_name="apply_database_migration",
                migration_id=migration_id,
                missing="approved_by",
            ),
        )

    logger.warning(
        "tool.apply_database_migration.start migration_id=%s approved_by=%s",
        migration_id,
        approved_by,
    )
    try:
        await _simulate_latency(base_ms=500, jitter_ms=250)
        result = MigrationResult(
            migration_id=migration_id,
            applied=True,
            duration_ms=random.randint(250, 2500),
            applied_at=_now(),
            approved_by=approved_by,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception(
            "tool.apply_database_migration.error migration_id=%s", migration_id
        )
        raise TrueGuardException(
            f"Failed to apply migration '{migration_id}'",
            context=_error_context(
                tool_name="apply_database_migration",
                migration_id=migration_id,
                approved_by=approved_by,
            ),
            cause=exc,
        ) from exc

    logger.warning("tool.apply_database_migration.ok migration_id=%s", migration_id)
    return result.model_dump(mode="json")


# ====================================================================== #
# Standalone entrypoint                                                  #
# ====================================================================== #
if __name__ == "__main__":
    # Canonical invocation: `fastmcp run mcp_servers/system_mcp.py:mcp`.
    # This block allows `python -m mcp_servers.system_mcp` during development.
    mcp.run()
