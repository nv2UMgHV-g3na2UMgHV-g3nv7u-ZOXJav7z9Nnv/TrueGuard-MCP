"""Configuration validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import Settings


def test_settings_reject_identical_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIMARY_MODEL", "gpt-4o")
    monkeypatch.setenv("FALLBACK_MODEL", "gpt-4o")
    with pytest.raises(ValidationError, match="must differ"):
        Settings()  # type: ignore[call-arg]


def test_settings_reject_out_of_range_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_BUDGET_USD", "9999")
    with pytest.raises(ValidationError, match="job_budget_usd must be between"):
        Settings()  # type: ignore[call-arg]


def test_settings_openai_base_url() -> None:
    s = Settings()  # type: ignore[call-arg]
    assert s.openai_base_url.endswith("/v1")
    assert s.is_slack_enabled is False
    assert s.get_slack_webhook() is None
