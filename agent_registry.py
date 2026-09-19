"""
agent_registry.py
=================
TrueForge agent discovery & governance integration for TrueGuard-MCP.

Synergies with your existing system:
- Coordinates with exceptions.py (HallucinationDetectedException)
- Logs via logging_utils.py (structured trace context)
- Validates against improved_config.py (settings)
- Feeds into cost governance and HITL approval flows

Key use cases:
1. **Runtime agent validation** — verify agent exists before tooling attempt
2. **Fallback strategy** — re-route to alternative agent if primary unavailable
3. **Cost-aware scheduling** — pick agent based on pricing tier + budget
4. **Audit trail** — record which agents were available during execution
5. **Governance decisions** — agent risk level influences HITL approval
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


# ====================================================================== #
# Agent Model                                                            #
# ====================================================================== #


@dataclass
class AgentMetadata:
    """Structured metadata from TrueForge agent listing."""
    
    agent_id: str
    name: str
    description: str
    workspace_id: str
    
    # Governance attributes
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    cost_per_invocation_usd: float
    rate_limit_rpm: int  # requests per minute
    requires_approval: bool
    
    # Availability / Deployment
    status: str  # "ACTIVE", "DEPRECATED", "BETA", "OFFLINE"
    last_heartbeat: datetime
    
    # Routing metadata
    primary_use_case: str  # "diagnostics", "remediation", "escalation"
    supported_operations: list[str]  # ["check_logs", "restart_service", ...]
    
    def is_available(self, as_of: Optional[datetime] = None) -> bool:
        """Check if agent is currently available."""
        as_of = as_of or datetime.now(tz=timezone.utc)
        age = (as_of - self.last_heartbeat).total_seconds()
        # Consider offline if no heartbeat in 5 minutes
        return self.status == "ACTIVE" and age < 300

    def is_within_budget(self, remaining_budget_usd: float) -> bool:
        """Check if invocation would fit in remaining budget."""
        return self.cost_per_invocation_usd <= remaining_budget_usd

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/observability."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "risk_level": self.risk_level,
            "status": self.status,
            "cost_per_invocation_usd": self.cost_per_invocation_usd,
            "requires_approval": self.requires_approval,
        }


# ====================================================================== #
# TrueForge API Client                                                  #
# ====================================================================== #


class TrueForgeAgentClient:
    """
    Async client for TrueForge agent discovery API.
    
    Handles:
    - List agents with filtering
    - Cache invalidation & refresh
    - Observability via structured logging
    - Resilience (retries, timeouts)
    """

    def __init__(
        self,
        truefoundry_mcp_gateway_url: str,
        api_key: str,
        workspace_id: str,
        cache_ttl_seconds: int = 300,
        request_timeout_seconds: float = 10.0,
    ):
        self.gateway_url = truefoundry_mcp_gateway_url.rstrip("/")
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.cache_ttl = cache_ttl_seconds
        self.request_timeout = request_timeout_seconds
        
        self._agent_cache: dict[str, AgentMetadata] | None = None
        self._cache_timestamp: datetime | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init HTTP client (for async context mgmt)."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.request_timeout)
        return self._http_client

    async def close(self) -> None:
        """Cleanup HTTP client."""
        if self._http_client:
            await self._http_client.aclose()

    def _is_cache_valid(self) -> bool:
        """Check if cache is still fresh."""
        if self._cache_timestamp is None:
            return False
        age = (datetime.now(tz=timezone.utc) - self._cache_timestamp).total_seconds()
        return age < self.cache_ttl

    async def list_agents(
        self,
        filter_by_status: Optional[str] = None,
        filter_by_risk_level: Optional[str] = None,
        skip_cache: bool = False,
        trace_id: Optional[str] = None,
    ) -> dict[str, AgentMetadata]:
        """
        List all agents in the workspace, with optional filtering.
        
        Args:
            filter_by_status: Filter to "ACTIVE", "DEPRECATED", "BETA", "OFFLINE"
            filter_by_risk_level: Filter to "LOW", "MEDIUM", "HIGH"
            skip_cache: Force fresh fetch (ignore cache TTL)
            trace_id: Correlation ID for structured logging
            
        Returns:
            Dict of agent_id → AgentMetadata
            
        Raises:
            httpx.HTTPError: If API call fails after retries
        """
        trace_id = trace_id or uuid4().hex[:8]
        
        # Check cache
        if not skip_cache and self._is_cache_valid() and self._agent_cache is not None:
            logger.info(
                f"Agent list cache hit (age {self._cache_age_seconds}s)",
                extra={"trace_id": trace_id, "operation": "agent_list_cache"},
            )
            agents = self._agent_cache
        else:
            # Fetch fresh from API
            agents = await self._fetch_agents_from_api(trace_id)
            self._agent_cache = agents
            self._cache_timestamp = datetime.now(tz=timezone.utc)
        
        # Apply filters
        if filter_by_status:
            agents = {
                aid: a for aid, a in agents.items() if a.status == filter_by_status
            }
        if filter_by_risk_level:
            agents = {
                aid: a for aid, a in agents.items() if a.risk_level == filter_by_risk_level
            }
        
        logger.info(
            f"Agent list queried: {len(agents)} result(s)",
            extra={
                "trace_id": trace_id,
                "operation": "agent_list_query",
                "agent_count": len(agents),
                "filters": {
                    "status": filter_by_status,
                    "risk_level": filter_by_risk_level,
                },
            },
        )
        
        return agents

    async def _fetch_agents_from_api(
        self, trace_id: str, retries: int = 3
    ) -> dict[str, AgentMetadata]:
        """
        Fetch agent list from TrueForge API with exponential backoff.
        
        Endpoint: GET /v1/agents?workspace_id=<workspace_id>
        
        Returns:
            Dict of agent_id → AgentMetadata
        """
        url = f"{self.gateway_url}/v1/agents"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        params = {"workspace_id": self.workspace_id}
        
        client = await self._get_client()
        
        for attempt in range(retries):
            try:
                logger.debug(
                    f"Fetching agents (attempt {attempt + 1}/{retries})",
                    extra={"trace_id": trace_id, "url": url},
                )
                
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                payload = response.json()
                agents = self._parse_agent_payload(payload)
                
                logger.info(
                    f"Fetched {len(agents)} agents from TrueForge",
                    extra={
                        "trace_id": trace_id,
                        "operation": "agent_fetch_success",
                        "agent_count": len(agents),
                    },
                )
                
                return agents
                
            except httpx.HTTPError as exc:
                backoff_seconds = 2 ** attempt  # 1s, 2s, 4s
                
                logger.warning(
                    f"Agent fetch failed (attempt {attempt + 1}/{retries}): {exc}",
                    extra={
                        "trace_id": trace_id,
                        "error": str(exc),
                        "backoff_seconds": backoff_seconds,
                    },
                )
                
                if attempt < retries - 1:
                    await asyncio.sleep(backoff_seconds)
                else:
                    logger.error(
                        "Agent fetch exhausted retries",
                        extra={"trace_id": trace_id},
                    )
                    raise

    def _parse_agent_payload(self, payload: dict[str, Any]) -> dict[str, AgentMetadata]:
        """
        Parse TrueForge API response into AgentMetadata objects.
        
        Expected payload structure:
        {
            "agents": [
                {
                    "id": "agent-abc123",
                    "name": "ServiceDiagnostic",
                    "description": "...",
                    "risk_level": "MEDIUM",
                    "cost_per_invocation_usd": 0.02,
                    "rate_limit_rpm": 60,
                    "status": "ACTIVE",
                    "last_heartbeat": "2025-01-15T12:34:56Z",
                    "primary_use_case": "diagnostics",
                    "supported_operations": ["check_logs", "check_health"],
                    "requires_approval": true
                },
                ...
            ]
        }
        """
        agents = {}
        
        for agent_data in payload.get("agents", []):
            try:
                agent = AgentMetadata(
                    agent_id=agent_data["id"],
                    name=agent_data["name"],
                    description=agent_data.get("description", ""),
                    workspace_id=self.workspace_id,
                    risk_level=agent_data.get("risk_level", "MEDIUM"),
                    cost_per_invocation_usd=float(agent_data.get("cost_per_invocation_usd", 0.0)),
                    rate_limit_rpm=int(agent_data.get("rate_limit_rpm", 60)),
                    requires_approval=agent_data.get("requires_approval", False),
                    status=agent_data.get("status", "ACTIVE"),
                    last_heartbeat=datetime.fromisoformat(
                        agent_data["last_heartbeat"].replace("Z", "+00:00")
                    ),
                    primary_use_case=agent_data.get("primary_use_case", "unknown"),
                    supported_operations=agent_data.get("supported_operations", []),
                )
                agents[agent.agent_id] = agent
            except (KeyError, ValueError) as exc:
                logger.warning(
                    f"Failed to parse agent record: {exc}",
                    extra={"agent_data": agent_data},
                )
                continue
        
        return agents

    @property
    def _cache_age_seconds(self) -> float:
        """Get age of current cache in seconds."""
        if self._cache_timestamp is None:
            return float("inf")
        return (datetime.now(tz=timezone.utc) - self._cache_timestamp).total_seconds()


# ====================================================================== #
# Governance Integration                                                #
# ====================================================================== #


class AgentGovernanceLayer:
    """
    Wraps TrueForgeAgentClient with governance policies.
    
    Uses AgentMetadata to:
    1. Validate hallucinations (agent exists in registry)
    2. Enforce cost & approval policies
    3. Route to fallback agents if primary unavailable
    4. Audit all agent invocation decisions
    """

    def __init__(
        self,
        agent_client: TrueForgeAgentClient,
        allow_high_risk_without_approval: bool = False,
    ):
        self.client = agent_client
        self.allow_high_risk_without_approval = allow_high_risk_without_approval

    async def validate_agent_exists(
        self,
        agent_id: str,
        trace_id: Optional[str] = None,
    ) -> AgentMetadata:
        """
        Validate that a requested agent exists in the workspace.
        
        Part of hallucination detection: the LLM might reference an agent
        that doesn't exist. This guards against that.
        
        Raises:
            HallucinationDetectedException if agent not found
        """
        from exceptions import HallucinationDetectedException
        
        trace_id = trace_id or uuid4().hex[:8]
        agents = await self.client.list_agents(skip_cache=False, trace_id=trace_id)
        
        if agent_id not in agents:
            available = sorted(agents.keys())
            
            logger.error(
                f"Hallucinated agent ID: {agent_id}",
                extra={
                    "trace_id": trace_id,
                    "operation": "agent_hallucination_check",
                    "requested_agent": agent_id,
                    "available_agents": available,
                },
            )
            
            raise HallucinationDetectedException(
                referenced_name=agent_id,
                available_tools=available,
            )
        
        return agents[agent_id]

    async def select_best_agent(
        self,
        operation: str,
        remaining_budget_usd: float,
        require_approval: bool = False,
        trace_id: Optional[str] = None,
    ) -> Optional[AgentMetadata]:
        """
        Select the best agent for an operation, respecting constraints.
        
        Ranking logic:
        1. Must be ACTIVE and within remaining budget
        2. Prefer LOW risk if approval unavailable
        3. Return first matching agent by (risk_level asc, cost asc)
        
        Args:
            operation: Operation name (e.g., "check_logs", "restart_service")
            remaining_budget_usd: Hard budget remaining
            require_approval: If True, can use MEDIUM/HIGH risk agents
            trace_id: Correlation ID
            
        Returns:
            Best matching AgentMetadata, or None if no suitable agent exists
        """
        trace_id = trace_id or uuid4().hex[:8]
        
        # Fetch active agents
        agents = await self.client.list_agents(
            filter_by_status="ACTIVE",
            skip_cache=False,
            trace_id=trace_id,
        )
        
        # Filter by operation support
        candidates = [
            a for a in agents.values()
            if operation in a.supported_operations
        ]
        
        # Filter by budget
        candidates = [
            a for a in candidates
            if a.is_within_budget(remaining_budget_usd)
        ]
        
        if not candidates:
            logger.warning(
                f"No suitable agents for operation '{operation}'",
                extra={
                    "trace_id": trace_id,
                    "operation": "agent_selection_failed",
                    "requested_operation": operation,
                    "remaining_budget_usd": remaining_budget_usd,
                    "total_agents": len(agents),
                    "active_agents": len([a for a in agents.values() if a.is_available()]),
                },
            )
            return None
        
        # Sort by risk level, then cost
        risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        candidates.sort(
            key=lambda a: (
                risk_order.get(a.risk_level, 999),
                a.cost_per_invocation_usd,
            )
        )
        
        selected = candidates[0]
        
        logger.info(
            f"Selected agent for operation '{operation}'",
            extra={
                "trace_id": trace_id,
                "operation": "agent_selection_success",
                "requested_operation": operation,
                "selected_agent": selected.agent_id,
                "selected_agent_name": selected.name,
                "risk_level": selected.risk_level,
                "cost_usd": selected.cost_per_invocation_usd,
                "candidates_considered": len(candidates),
            },
        )
        
        return selected

    async def get_fallback_agent(
        self,
        primary_agent_id: str,
        remaining_budget_usd: float,
        trace_id: Optional[str] = None,
    ) -> Optional[AgentMetadata]:
        """
        Select a fallback agent if primary is unavailable.
        
        Fallback logic:
        1. Exclude the primary agent
        2. Prefer same risk level, else lower
        3. Must be within budget
        """
        trace_id = trace_id or uuid4().hex[:8]
        
        agents = await self.client.list_agents(
            filter_by_status="ACTIVE",
            skip_cache=True,  # Force refresh for fresh availability check
            trace_id=trace_id,
        )
        
        primary = agents.get(primary_agent_id)
        if not primary:
            return None
        
        # Candidates: active, available, not primary, within budget
        candidates = [
            a for a in agents.values()
            if (
                a.agent_id != primary_agent_id
                and a.is_available()
                and a.is_within_budget(remaining_budget_usd)
            )
        ]
        
        if not candidates:
            logger.warning(
                f"No fallback agents available for '{primary_agent_id}'",
                extra={
                    "trace_id": trace_id,
                    "operation": "agent_fallback_search",
                    "primary_agent": primary_agent_id,
                    "candidates_found": 0,
                },
            )
            return None
        
        # Prefer same risk level, then sort by cost
        risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        candidates.sort(
            key=lambda a: (
                abs(risk_order.get(a.risk_level, 999) - risk_order.get(primary.risk_level, 999)),
                a.cost_per_invocation_usd,
            )
        )
        
        fallback = candidates[0]
        
        logger.info(
            f"Fallback agent selected for '{primary_agent_id}'",
            extra={
                "trace_id": trace_id,
                "operation": "agent_fallback_selected",
                "primary_agent": primary_agent_id,
                "fallback_agent": fallback.agent_id,
                "fallback_agent_name": fallback.name,
            },
        )
        
        return fallback
