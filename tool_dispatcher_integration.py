"""
tool_dispatcher_integration.py
==============================
Real-world example: How to integrate TrueForge agent registry into your
MCP tool dispatcher, with all three governance layers working together.

Shows:
- Hallucination detection
- Fallback routing on failure
- Cost tracking across agent invocations
- HITL approval for HIGH_RISK agents
- Structured observability
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from agent_registry import (
    AgentGovernanceLayer,
    AgentMetadata,
    TrueForgeAgentClient,
)
from exceptions import (
    ApprovalDeniedException,
    ApprovalTimeoutException,
    BudgetExceededException,
    HallucinationDetectedException,
    ErrorContext,
    create_error_context,
)
from logging_utils import trace_context, span_context, TraceLogger

logger = logging.getLogger(__name__)


# ====================================================================== #
# Cost Tracking (from your improved_config.py pattern)                  #
# ====================================================================== #


@dataclass
class AgentInvocationCost:
    """Track cost of a single agent invocation."""
    agent_id: str
    agent_name: str
    invocation_tokens: int  # proxy for computation
    cost_usd: float
    success: bool
    fallback_executed: bool = False


class CostTracker:
    """
    Simple ledger for tracking agent invocation costs.
    
    Integrates with your BudgetExceededException to enforce hard limits.
    """
    
    def __init__(self, job_budget_usd: float, halt_multiplier: float = 1.5):
        self.job_budget_usd = job_budget_usd
        self.halt_threshold_usd = job_budget_usd * halt_multiplier
        self.ledger: list[AgentInvocationCost] = []
    
    def record(self, entry: AgentInvocationCost) -> None:
        """Record an agent invocation cost."""
        self.ledger.append(entry)
        logger.info(
            f"Cost recorded: {entry.agent_name} → ${entry.cost_usd:.4f}",
            extra={
                "agent_id": entry.agent_id,
                "cost_usd": entry.cost_usd,
                "accumulated_usd": self.accumulated_usd,
            },
        )
    
    @property
    def accumulated_usd(self) -> float:
        """Total cost accumulated so far."""
        return sum(e.cost_usd for e in self.ledger)
    
    @property
    def remaining_usd(self) -> float:
        """Remaining budget before hard halt."""
        return self.halt_threshold_usd - self.accumulated_usd
    
    def check_halt_threshold(self) -> None:
        """Raise if hard threshold exceeded."""
        if self.accumulated_usd >= self.halt_threshold_usd:
            raise BudgetExceededException(
                accumulated_usd=self.accumulated_usd,
                halt_usd=self.halt_threshold_usd,
            )


# ====================================================================== #
# Approval Decision Store                                               #
# ====================================================================== #


@dataclass
class ApprovalDecision:
    """Record of a human approval decision."""
    tool_name: str
    trace_id: str
    approved: bool
    decision_time: datetime
    operator_id: Optional[str] = None
    reason: Optional[str] = None


class ApprovalDecisionStore:
    """
    Simple in-memory store for approval decisions.
    In production, this would be a persistent database.
    """
    
    def __init__(self):
        self.decisions: dict[str, ApprovalDecision] = {}
    
    async def request_approval(
        self,
        tool_name: str,
        agent_id: str,
        agent_metadata: AgentMetadata,
        trace_id: str,
        timeout_seconds: float = 300,
    ) -> ApprovalDecision:
        """
        Request human approval via Slack (or other channel).
        
        Returns when approval is granted/denied or timeout expires.
        """
        decision_key = f"{trace_id}:{agent_id}"
        
        # In production: send Slack card, wait for response
        # For demo: auto-approve after 1 second
        logger.info(
            f"HITL approval requested for {tool_name} (agent: {agent_id})",
            extra={
                "trace_id": trace_id,
                "agent_id": agent_id,
                "risk_level": agent_metadata.risk_level,
                "cost_usd": agent_metadata.cost_per_invocation_usd,
            },
        )
        
        # Simulate wait for operator decision
        await asyncio.sleep(0.5)
        
        # Mock: auto-approve (in demo)
        decision = ApprovalDecision(
            tool_name=tool_name,
            trace_id=trace_id,
            approved=True,
            decision_time=datetime.now(tz=timezone.utc),
            operator_id="demo-operator",
            reason="Auto-approved in demo",
        )
        
        self.decisions[decision_key] = decision
        
        logger.info(
            f"HITL approval granted: {tool_name}",
            extra={"trace_id": trace_id, "operator_id": decision.operator_id},
        )
        
        return decision


# ====================================================================== #
# Integrated Tool Dispatcher                                            #
# ====================================================================== #


class AgentToolDispatcher:
    """
    Dispatch agent invocations with full governance:
    - Hallucination detection
    - Cost checking
    - HITL approval routing
    - Fallback on failure
    """
    
    def __init__(
        self,
        agent_governance: AgentGovernanceLayer,
        approval_store: ApprovalDecisionStore,
        cost_tracker: CostTracker,
    ):
        self.governance = agent_governance
        self.approvals = approval_store
        self.costs = cost_tracker
    
    async def invoke_agent(
        self,
        agent_id: str,
        operation: str,
        args: dict[str, Any],
        trace_id: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> dict[str, Any]:
        """
        Invoke an agent with full governance gates.
        
        Steps:
        1. Validate agent exists (hallucination guard)
        2. Check budget availability
        3. Route HITL approval if needed
        4. Invoke agent
        5. On failure, retry with fallback (if allowed)
        
        Returns:
            {"success": bool, "result": Any, "agent_used": str, ...}
        """
        trace_id = trace_id or uuid4().hex[:8]
        
        with trace_context("agent_invocation", trace_id=trace_id) as trace_logger:
            trace_logger.start_operation(
                f"invoke_agent {operation}",
                agent_id=agent_id,
                operation=operation,
                allow_fallback=allow_fallback,
            )
            
            # ---- Step 1: Validate agent exists ---- #
            try:
                with span_context("validate_agent", trace_logger) as span:
                    agent = await self.governance.validate_agent_exists(
                        agent_id,
                        trace_id=trace_id,
                    )
                    span.log_metric("agent.status", agent.status)
                    span.log_metric("agent.risk_level", agent.risk_level)
                    span.log_metric("agent.cost_usd", agent.cost_per_invocation_usd)
                
            except HallucinationDetectedException as exc:
                logger.error(
                    f"Hallucination detected: {exc}",
                    extra={"trace_id": trace_id},
                )
                return {
                    "success": False,
                    "error": "agent_not_found",
                    "agent_id": agent_id,
                    "message": str(exc),
                    "trace_id": trace_id,
                }
            
            # ---- Step 2: Check budget ---- #
            try:
                with span_context("check_budget", trace_logger) as span:
                    self.costs.check_halt_threshold()
                    
                    if agent.cost_per_invocation_usd > self.costs.remaining_usd:
                        raise BudgetExceededException(
                            accumulated_usd=self.costs.accumulated_usd,
                            halt_usd=self.costs.halt_threshold_usd,
                            spent_on_model=agent.name,
                        )
                    
                    span.log_metric("remaining_budget_usd", self.costs.remaining_usd)
                
            except BudgetExceededException as exc:
                logger.error(
                    f"Budget exceeded: {exc}",
                    extra={"trace_id": trace_id},
                )
                return {
                    "success": False,
                    "error": "budget_exceeded",
                    "accumulated_usd": exc.accumulated_usd,
                    "halt_usd": exc.halt_usd,
                    "trace_id": trace_id,
                }
            
            # ---- Step 3: HITL approval if needed ---- #
            if agent.requires_approval:
                try:
                    with span_context("request_approval", trace_logger) as span:
                        decision = await self.approvals.request_approval(
                            tool_name=agent.name,
                            agent_id=agent_id,
                            agent_metadata=agent,
                            trace_id=trace_id,
                            timeout_seconds=300,
                        )
                        span.log_metric("approval.granted", decision.approved)
                        span.log_metric("approval.operator", decision.operator_id)
                
                except ApprovalTimeoutException as exc:
                    logger.error(
                        f"Approval timeout: {exc}",
                        extra={"trace_id": trace_id},
                    )
                    return {
                        "success": False,
                        "error": "approval_timeout",
                        "tool_name": agent.name,
                        "timeout_seconds": exc.timeout_seconds,
                        "trace_id": trace_id,
                    }
                
                if not decision.approved:
                    raise ApprovalDeniedException(
                        tool_name=agent.name,
                        reason=decision.reason,
                    )
            
            # ---- Step 4: Invoke agent ---- #
            try:
                result = await self._do_invoke_agent(
                    agent=agent,
                    operation=operation,
                    args=args,
                    trace_id=trace_id,
                    trace_logger=trace_logger,
                )
                
                # Record successful invocation
                self.costs.record(
                    AgentInvocationCost(
                        agent_id=agent_id,
                        agent_name=agent.name,
                        invocation_tokens=200,  # Mock
                        cost_usd=agent.cost_per_invocation_usd,
                        success=True,
                        fallback_executed=False,
                    )
                )
                
                return {
                    "success": True,
                    "result": result,
                    "agent_used": agent_id,
                    "cost_usd": agent.cost_per_invocation_usd,
                    "trace_id": trace_id,
                }
            
            except Exception as exc:
                logger.warning(
                    f"Agent invocation failed: {exc}",
                    extra={
                        "trace_id": trace_id,
                        "agent_id": agent_id,
                        "error": str(exc),
                    },
                )
                
                # ---- Step 5: Fallback retry ---- #
                if allow_fallback:
                    logger.info(
                        f"Attempting fallback for {agent_id}",
                        extra={"trace_id": trace_id},
                    )
                    
                    with span_context("fallback_selection", trace_logger) as span:
                        fallback = await self.governance.get_fallback_agent(
                            primary_agent_id=agent_id,
                            remaining_budget_usd=self.costs.remaining_usd,
                            trace_id=trace_id,
                        )
                        
                        if fallback:
                            span.log_metric("fallback_agent", fallback.agent_id)
                            
                            # Recursive call (but only fallback once to prevent infinite loop)
                            return await self.invoke_agent(
                                agent_id=fallback.agent_id,
                                operation=operation,
                                args=args,
                                trace_id=trace_id,
                                allow_fallback=False,  # Don't fallback again
                            )
                
                # If no fallback available or fallback disabled
                return {
                    "success": False,
                    "error": "agent_invocation_failed",
                    "agent_id": agent_id,
                    "message": str(exc),
                    "fallback_available": fallback is not None if allow_fallback else False,
                    "trace_id": trace_id,
                }
    
    async def _do_invoke_agent(
        self,
        agent: AgentMetadata,
        operation: str,
        args: dict[str, Any],
        trace_id: str,
        trace_logger: TraceLogger,
    ) -> dict[str, Any]:
        """
        Actually invoke the agent (HTTP call to agent endpoint).
        
        This is where you'd make the real gRPC/HTTP call to the agent runtime.
        """
        with span_context("agent_execution", trace_logger) as span:
            span.log_metric("agent_id", agent.agent_id)
            span.log_metric("operation", operation)
            
            # Mock: simulate agent execution
            logger.info(
                f"Executing agent {agent.name} :: {operation}",
                extra={
                    "trace_id": trace_id,
                    "agent_id": agent.agent_id,
                    "operation": operation,
                },
            )
            
            # In production, this would be:
            # async with httpx.AsyncClient() as client:
            #     response = await client.post(
            #         f"{agent_endpoint}/invoke",
            #         json={"operation": operation, "args": args},
            #     )
            #     return response.json()
            
            # Mock result
            await asyncio.sleep(0.1)
            return {
                "status": "success",
                "agent_id": agent.agent_id,
                "operation": operation,
                "result": {"message": f"{operation} completed successfully"},
            }


# ====================================================================== #
# Example Usage                                                          #
# ====================================================================== #


async def demo_scenario() -> None:
    """
    Example: LLM proposes agent invocation → full governance flow.
    """
    from improved_config import settings
    
    # Initialize
    agent_client = TrueForgeAgentClient(
        truefoundry_mcp_gateway_url=settings.truefoundry_mcp_gateway_url,
        api_key=settings.truefoundry_api_key.get_secret_value(),
        workspace_id="ws-demo-123",
        cache_ttl_seconds=300,
    )
    
    governance = AgentGovernanceLayer(
        agent_client=agent_client,
        allow_high_risk_without_approval=False,
    )
    
    approval_store = ApprovalDecisionStore()
    cost_tracker = CostTracker(
        job_budget_usd=0.50,
        halt_multiplier=1.5,  # Halt at $0.75
    )
    
    dispatcher = AgentToolDispatcher(
        agent_governance=governance,
        approval_store=approval_store,
        cost_tracker=cost_tracker,
    )
    
    # Example 1: Hallucination detection
    print("\n=== Example 1: Hallucination Detection ===")
    result = await dispatcher.invoke_agent(
        agent_id="agent-nonexistent-xyz",
        operation="check_logs",
        args={"service": "payment"},
    )
    print(f"Result: {result}")
    # → "error": "agent_not_found"
    
    # Example 2: Valid invocation
    print("\n=== Example 2: Valid Invocation ===")
    result = await dispatcher.invoke_agent(
        agent_id="agent-diag-v1",
        operation="check_logs",
        args={"service": "payment"},
    )
    print(f"Result: {result}")
    # → "success": True, "cost_usd": 0.02
    
    # Example 3: Budget exceeded
    print("\n=== Example 3: Budget Exceeded ===")
    # Artificially spike cost tracker to near halt threshold
    cost_tracker.ledger.append(
        AgentInvocationCost(
            agent_id="dummy",
            agent_name="dummy",
            invocation_tokens=100,
            cost_usd=0.74,  # Pushes past $0.75 halt
            success=True,
        )
    )
    
    result = await dispatcher.invoke_agent(
        agent_id="agent-diag-v1",
        operation="check_logs",
        args={"service": "payment"},
    )
    print(f"Result: {result}")
    # → "error": "budget_exceeded"
    
    # Cleanup
    await agent_client.close()


if __name__ == "__main__":
    # For actual demo, mock the settings
    # asyncio.run(demo_scenario())
    print("See demo_scenario() function for usage example.")
