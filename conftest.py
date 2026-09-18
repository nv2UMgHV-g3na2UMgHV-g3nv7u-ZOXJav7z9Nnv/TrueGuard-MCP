"""Shared pytest fixtures."""

from __future__ import annotations

import os
from typing import Iterator

import pytest

# Env must be set before any module imports `settings`.
os.environ.setdefault("TRUEFOUNDRY_GATEWAY_URL", "https://gateway.truefoundry.ai")
os.environ.setdefault("TRUEFOUNDRY_API_KEY", "tfy-test-key")
os.environ.setdefault("TRUEFOUNDRY_MCP_GATEWAY_URL", "https://mcp-gateway.truefoundry.ai")
os.environ.setdefault("APPROVAL_CALLBACK_URL", "http://localhost:8000/api/v1/approval/callback")


@pytest.fixture
def fresh_settings() -> Iterator[None]:
    """Reset the settings cache around each test."""
    from config import SettingsCache

    SettingsCache.reset()
    yield
    SettingsCache.reset()
