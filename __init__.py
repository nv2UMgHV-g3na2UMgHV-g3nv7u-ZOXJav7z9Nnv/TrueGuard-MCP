"""
agents
======
Agent implementations for TrueGuard-MCP.

Currently exposes a single agent — :class:`DevOpsAgent` — which orchestrates
the four-step incident-response protocol (observe → diagnose → plan → report).
"""

from agents.devops_agent import (
    SYSTEM_PROMPT,
    DevOpsAgent,
    Evidence,
    IncidentReportResponse,
    RemediationStep,
    Severity,
)

__all__ = [
    "DevOpsAgent",
    "IncidentReportResponse",
    "RemediationStep",
    "Evidence",
    "Severity",
    "SYSTEM_PROMPT",
]
