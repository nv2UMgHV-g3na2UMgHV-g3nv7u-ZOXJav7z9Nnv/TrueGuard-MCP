"""
mcp_servers
===========
FastMCP server packages for TrueGuard-MCP.

Exposes :data:`mcp` — the single FastMCP server instance hosting the four
system-operations tools that the agent may invoke.
"""

from mcp_servers.system_mcp import (
    ContainerHealth,
    LogFetchResult,
    MigrationResult,
    RestartResult,
    mcp,
)

__all__ = [
    "mcp",
    "LogFetchResult",
    "ContainerHealth",
    "RestartResult",
    "MigrationResult",
]
