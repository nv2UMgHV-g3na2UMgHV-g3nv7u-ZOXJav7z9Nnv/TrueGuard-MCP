"""
gateways
========
Gateway interceptors and runtime enforcement for TrueGuard-MCP.

Exposes:

* :mod:`gateways.mcp_governance`   — ``@require_permission`` decorator that
  enforces TrueFoundry MCP Gateway policy at call time.
* :mod:`gateways.llm_router`       — :class:`TrueFoundryRouter` and
  :class:`CostGuardrail` for model failover + budget enforcement.
* :mod:`gateways.approval_workflow` — human-in-the-loop approval store and
  FastAPI router.
"""

from gateways.llm_router import (
    BudgetExceededException,
    CostGuardrail,
    JobBudgetTracker,
    LedgerEntry,
    ModelPricing,
    RoutedResponse,
    TrueFoundryRouter,
    get_current_tracker,
    reset_current_tracker,
    set_current_tracker,
)
from gateways.mcp_governance import (
    ApprovalDeniedException,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalTimeoutError,
    PermissionLevel,
    approval_registry,
    require_permission,
)

__all__ = [
    # mcp_governance
    "require_permission",
    "PermissionLevel",
    "ApprovalRequest",
    "ApprovalStatus",
    "approval_registry",
    "ApprovalTimeoutError",
    "ApprovalDeniedException",
    # llm_router
    "TrueFoundryRouter",
    "CostGuardrail",
    "JobBudgetTracker",
    "LedgerEntry",
    "ModelPricing",
    "RoutedResponse",
    "BudgetExceededException",
    "set_current_tracker",
    "get_current_tracker",
    "reset_current_tracker",
]
