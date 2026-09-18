"""
exceptions.py (Improved)
========================
Custom exception hierarchy with context and traceability.

Key features:
- Structured exception types for better error handling
- Trace correlation IDs for debugging
- Actionable error messages
- Proper exception chaining
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class ErrorContext:
    """Structured error context for better debugging."""
    trace_id: str
    span_id: str
    timestamp: datetime
    operation: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            **self.metadata,
        }


class TrueGuardException(Exception):
    """
    Base exception for all TrueGuard errors.
    
    Provides structured context and correlation tracking.
    """

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        self.message = message
        self.context = context
        self.cause = cause
        super().__init__(message)

    def __str__(self) -> str:
        """Format exception with context."""
        parts = [self.message]
        if self.context:
            parts.append(f"[trace: {self.context.trace_id}]")
            parts.append(f"[span: {self.context.span_id}]")
        if self.cause:
            parts.append(f"caused by: {type(self.cause).__name__}")
        return " — ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception for logging/observability."""
        data = {
            "exception_type": type(self).__name__,
            "message": self.message,
        }
        if self.context:
            data["context"] = self.context.to_dict()
        if self.cause:
            data["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
            }
        return data


class BudgetExceededException(TrueGuardException):
    """
    Raised when accumulated cost exceeds hard threshold.
    
    This is a governance-level exception that stops agent execution.
    """

    def __init__(
        self,
        accumulated_usd: float,
        halt_usd: float,
        spent_on_model: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        self.accumulated_usd = accumulated_usd
        self.halt_usd = halt_usd
        self.spent_on_model = spent_on_model

        message = (
            f"Budget limit exceeded: accumulated ${accumulated_usd:.4f} "
            f"> halt threshold ${halt_usd:.4f}"
        )
        if spent_on_model:
            message += f" (last call: {spent_on_model})"

        super().__init__(message, context, cause)

    def to_dict(self) -> dict[str, Any]:
        """Include financial details in serialization."""
        data = super().to_dict()
        data.update({
            "accumulated_usd": self.accumulated_usd,
            "halt_usd": self.halt_usd,
            "spent_on_model": self.spent_on_model,
        })
        return data


class ModelRouterException(TrueGuardException):
    """
    Raised when all model routing attempts fail.
    
    This includes primary model failure + fallback model failure.
    """

    def __init__(
        self,
        primary_model: str,
        fallback_model: str,
        primary_error: Exception,
        fallback_error: Exception,
        context: Optional[ErrorContext] = None,
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.primary_error = primary_error
        self.fallback_error = fallback_error

        message = (
            f"Model routing failed: {primary_model} → {str(primary_error)[:50]}..., "
            f"{fallback_model} → {str(fallback_error)[:50]}..."
        )
        super().__init__(message, context, cause=fallback_error)

    def to_dict(self) -> dict[str, Any]:
        """Include routing details in serialization."""
        data = super().to_dict()
        data.update({
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "primary_error": str(self.primary_error),
            "fallback_error": str(self.fallback_error),
        })
        return data


class ApprovalTimeoutException(TrueGuardException):
    """
    Raised when human approval request times out.
    
    Human-in-the-loop (HITL) escalations must complete within
    a defined timeout window.
    """

    def __init__(
        self,
        timeout_seconds: float,
        tool_name: str,
        context: Optional[ErrorContext] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.tool_name = tool_name

        message = (
            f"Approval timeout after {timeout_seconds}s "
            f"for high-risk tool '{tool_name}'"
        )
        super().__init__(message, context)

    def to_dict(self) -> dict[str, Any]:
        """Include timeout details in serialization."""
        data = super().to_dict()
        data.update({
            "timeout_seconds": self.timeout_seconds,
            "tool_name": self.tool_name,
        })
        return data


class ApprovalDeniedException(TrueGuardException):
    """Raised when a human explicitly denies a high-risk tool invocation."""

    def __init__(
        self,
        tool_name: str,
        reason: Optional[str] = None,
        context: Optional[ErrorContext] = None,
    ):
        self.tool_name = tool_name
        self.reason = reason

        message = f"High-risk tool '{tool_name}' denied by human operator"
        if reason:
            message += f": {reason}"

        super().__init__(message, context)

    def to_dict(self) -> dict[str, Any]:
        """Include denial details in serialization."""
        data = super().to_dict()
        data.update({
            "tool_name": self.tool_name,
            "reason": self.reason,
        })
        return data


class InjectionDetectedException(TrueGuardException):
    """
    Raised when injection attacks are detected in tool arguments.
    
    Part of the security eval framework.
    """

    def __init__(
        self,
        injection_type: str,
        pattern: str,
        argument_value: str,
        context: Optional[ErrorContext] = None,
    ):
        self.injection_type = injection_type  # "sql", "shell", etc.
        self.pattern = pattern
        self.argument_value = argument_value

        message = (
            f"Injection attack detected: {injection_type} pattern matched. "
            f"Argument value matched pattern: {pattern[:50]}"
        )
        super().__init__(message, context)

    def to_dict(self) -> dict[str, Any]:
        """Include injection details (safely masked)."""
        data = super().to_dict()
        data.update({
            "injection_type": self.injection_type,
            "pattern": self.pattern,
            "argument_masked": self.argument_value[:10] + "***",  # Mask sensitive data
        })
        return data


class PiiDetectedException(TrueGuardException):
    """
    Raised when PII/secrets are detected in prompts or outputs.
    
    Part of the security eval framework.
    """

    def __init__(
        self,
        pii_type: str,
        detected_at: str,  # "prompt", "output"
        sample_masked: str,  # First 4 chars of detected secret
        context: Optional[ErrorContext] = None,
    ):
        self.pii_type = pii_type  # "openai_key", "aws_secret", etc.
        self.detected_at = detected_at
        self.sample_masked = sample_masked

        message = (
            f"PII/{pii_type} detected in {detected_at}: "
            f"masked value {sample_masked}"
        )
        super().__init__(message, context)

    def to_dict(self) -> dict[str, Any]:
        """Include PII detection details (safely masked)."""
        data = super().to_dict()
        data.update({
            "pii_type": self.pii_type,
            "detected_at": self.detected_at,
            "sample_masked": self.sample_masked,
        })
        return data


class HallucinationDetectedException(TrueGuardException):
    """
    Raised when model references non-existent tools or methods.
    
    Hallucinations must be caught before tool invocation.
    """

    def __init__(
        self,
        referenced_name: str,
        available_tools: list[str],
        context: Optional[ErrorContext] = None,
    ):
        self.referenced_name = referenced_name
        self.available_tools = available_tools

        message = (
            f"Model hallucinated tool name: '{referenced_name}'. "
            f"Available tools: {', '.join(available_tools)}"
        )
        super().__init__(message, context)

    def to_dict(self) -> dict[str, Any]:
        """Include hallucination details."""
        data = super().to_dict()
        data.update({
            "referenced_name": self.referenced_name,
            "available_tools": self.available_tools,
        })
        return data


def create_error_context(
    trace_id: str,
    span_id: str,
    operation: str,
    **metadata: Any,
) -> ErrorContext:
    """Factory function to create error context with current timestamp."""
    return ErrorContext(
        trace_id=trace_id,
        span_id=span_id,
        timestamp=datetime.now(tz=timezone.utc),
        operation=operation,
        metadata=metadata,
    )
