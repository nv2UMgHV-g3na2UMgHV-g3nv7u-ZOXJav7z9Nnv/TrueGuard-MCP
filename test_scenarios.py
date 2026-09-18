"""End-to-end scenario tests (stub-backed, no network)."""

from __future__ import annotations

import pytest

from scenarios import scenario_a
from scenarios._common import Palette


@pytest.mark.asyncio
async def test_scenario_a_passes() -> None:
    result = await scenario_a(Palette(enabled=False))
    assert result["scorecard"]["passed"] is True
    assert result["scorecard"]["score"] == "3/3"
    assert result["halted"] is False
