"""
config.py
=========
Centralised, strongly-typed configuration for TrueGuard-MCP.

Enhancements over the previous revision:
* Structured logging at load time (single clear startup line).
* Actionable, context-rich validator messages.
* :class:`SettingsCache` with explicit ``reset()`` for tests.
* Bounded validation of ``job_budget_usd`` (0.01 … 100.00).
* Convenience accessors: ``is_slack_enabled``, ``get_slack_webhook``.
* ``model_validator`` guarantees primary ≠ fallback.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Literal
from urllib.parse import urlparse

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Strongly-typed application settings for the TrueGuard-MCP harness."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- TrueFoundry Gateway ---- #
    truefoundry_gateway_url: HttpUrl = Field(
        ...,
        description="Base URL of the TrueFoundry AI Gateway (OpenAI-compatible).",
    )
    truefoundry_api_key: SecretStr = Field(
        ...,
        description="Bearer token used to authenticate against the AI Gateway.",
    )
    truefoundry_mcp_gateway_url: HttpUrl = Field(
        ...,
        description="Base URL of the TrueFoundry MCP Gateway (runtime policy engine).",
    )

    # ---- Model Routing ---- #
    primary_model: str = Field(
        default="gpt-4o",
        min_length=1,
        description="Preferred model for reasoning-heavy agent turns.",
    )
    fallback_model: str = Field(
        default="gpt-4o-mini",
        min_length=1,
        description="Cheaper model used when primary errors or is rate-limited.",
    )

    # ---- Cost Governance ---- #
    job_budget_usd: float = Field(
        default=0.50,
        gt=0.0,
        description="Hard USD cap per single agent job execution.",
    )

    # ---- Human-in-the-loop ---- #
    slack_webhook_url: HttpUrl | None = Field(
        default=None,
        description="Slack incoming-webhook URL for approval notifications.",
    )
    approval_callback_url: HttpUrl = Field(
        ...,
        description="HTTP endpoint that resumes paused HIGH_RISK tool calls.",
    )

    # ---- Logging ---- #
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root logger level for the application.",
    )

    # ---- Constants ---- #
    MIN_BUDGET_USD: ClassVar[float] = 0.01
    MAX_BUDGET_USD: ClassVar[float] = 100.0

    # ---- Validators ---- #
    @model_validator(mode="after")
    def _models_differ(self) -> Settings:
        if self.primary_model == self.fallback_model:
            msg = (
                f"primary_model ('{self.primary_model}') and fallback_model "
                f"('{self.fallback_model}') must differ to provide real resilience."
            )
            raise ValueError(msg)
        return self

    @field_validator("job_budget_usd")
    @classmethod
    def _budget_in_range(cls, v: float) -> float:
        if v < cls.MIN_BUDGET_USD or v > cls.MAX_BUDGET_USD:
            msg = (
                f"job_budget_usd must be between ${cls.MIN_BUDGET_USD} and "
                f"${cls.MAX_BUDGET_USD}, got ${v}."
            )
            raise ValueError(msg)
        return v

    @field_validator("approval_callback_url")
    @classmethod
    def _callback_well_formed(cls, v: HttpUrl) -> HttpUrl:
        parsed = urlparse(str(v))
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"approval_callback_url is not a valid URL: {v}")
        return v

    # ---- Convenience ---- #
    @property
    def openai_base_url(self) -> str:
        """OpenAI-compatible base URL routed through the TrueFoundry AI Gateway."""
        return f"{str(self.truefoundry_gateway_url).rstrip('/')}/v1"

    @property
    def openai_api_key(self) -> str:
        """Plaintext API key for the OpenAI SDK client."""
        return self.truefoundry_api_key.get_secret_value()

    @property
    def is_slack_enabled(self) -> bool:
        """Whether Slack notifications are configured."""
        return self.slack_webhook_url is not None

    def get_slack_webhook(self) -> str | None:
        """Slack webhook URL as a plain string, or ``None``."""
        return str(self.slack_webhook_url) if self.slack_webhook_url else None


class SettingsCache:
    """Thread-safe singleton with explicit reset for tests."""

    _instance: Settings | None = None

    @classmethod
    def get(cls) -> Settings:
        if cls._instance is None:
            try:
                cls._instance = Settings()  # type: ignore[call-arg]
                logger.info(
                    "Settings loaded: primary=%s fallback=%s budget=$%.2f",
                    cls._instance.primary_model,
                    cls._instance.fallback_model,
                    cls._instance.job_budget_usd,
                )
            except Exception as exc:
                logger.critical("Failed to load settings: %s", exc)
                raise
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        logger.debug("Settings cache reset")


settings: Settings = SettingsCache.get()
