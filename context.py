"""
orchestrator/context.py
=======================
Shared context object passed through every scenario step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import Settings
from gateways.llm_router import CostGuardrail, TrueFoundryRouter
from observability.tracer import Trace


@dataclass(slots=True)
class ScenarioContext:
    """
    Bundle of dependencies every scenario needs.

    Built once per scenario run and threaded through :func:`observe`,
    :func:`diagnose`, and :func:`plan` so individual step functions stay
    small and testable.
    """

    name: str
    alert_text: str
    affected_service: str
    incident_id: str
    trace: Trace
    settings: Settings
    router: TrueFoundryRouter
    guardrail: CostGuardrail
    mcp_client: Any
    metadata: dict[str, Any] = field(default_factory=dict)
