"""
logging_utils.py (Improved)
============================
Structured logging utilities for TrueGuard-MCP.

Features:
- Correlation IDs for tracing across services
- Context managers for automatic span lifecycle
- JSON and plaintext formatters
- Configurable log levels per module
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Generator, Optional
from uuid import uuid4

import structlog


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs JSON-structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include exception info if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Include extra fields (correlation IDs, spans, etc.)
        if hasattr(record, "trace_id"):
            log_obj["trace_id"] = record.trace_id
        if hasattr(record, "span_id"):
            log_obj["span_id"] = record.span_id
        if hasattr(record, "operation"):
            log_obj["operation"] = record.operation

        return json.dumps(log_obj)


class ColoredFormatter(logging.Formatter):
    """Formatter with ANSI colors for console output."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        parts = [
            f"{timestamp}",
            f"{color}{record.levelname:8}{self.RESET}",
            f"{record.name}",
        ]

        # Add correlation IDs if present
        if hasattr(record, "trace_id"):
            parts.append(f"[{record.trace_id}]")
        if hasattr(record, "span_id"):
            parts.append(f"[{record.span_id}]")

        parts.append(record.getMessage())

        # Include exception details
        if record.exc_info:
            parts.append(self.formatException(record.exc_info))

        return " | ".join(parts)


def configure_logging(
    level: str = "INFO",
    structured: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Use JSON structured logging (for production)
        log_file: Optional file path to log to (in addition to stderr)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = ColoredFormatter()

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
        if structured
        else [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


class TraceLogger:
    """Helper for managing trace/span lifecycle with structured logging."""

    def __init__(self, trace_id: Optional[str] = None, span_id: Optional[str] = None):
        self.trace_id = trace_id or uuid4().hex[:8]
        self.span_id = span_id or uuid4().hex[:8]
        self.logger = logging.getLogger(__name__)

    def start_operation(self, operation: str, **metadata: Any) -> None:
        """Log operation start."""
        self.logger.info(
            f"Operation started: {operation}",
            extra={
                "trace_id": self.trace_id,
                "span_id": self.span_id,
                "operation": operation,
                **metadata,
            },
        )

    def end_operation(
        self,
        operation: str,
        success: bool = True,
        duration_seconds: Optional[float] = None,
        **metadata: Any,
    ) -> None:
        """Log operation completion."""
        level = logging.INFO if success else logging.WARNING
        status = "succeeded" if success else "failed"

        self.logger.log(
            level,
            f"Operation {status}: {operation}",
            extra={
                "trace_id": self.trace_id,
                "span_id": self.span_id,
                "operation": operation,
                "duration_seconds": duration_seconds,
                **metadata,
            },
        )

    def log_metric(self, name: str, value: Any, **tags: Any) -> None:
        """Log a metric (cost, tokens, latency, etc.)."""
        self.logger.info(
            f"Metric: {name}={value}",
            extra={
                "trace_id": self.trace_id,
                "span_id": self.span_id,
                "metric_name": name,
                "metric_value": value,
                **tags,
            },
        )

    def log_error(
        self,
        message: str,
        error: Exception,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log an error with full context."""
        self.logger.error(
            f"{message}: {error}",
            extra={
                "trace_id": self.trace_id,
                "span_id": self.span_id,
                "error_type": type(error).__name__,
                **(context or {}),
            },
            exc_info=True,
        )


@contextlib.contextmanager
def trace_context(
    operation_name: str,
    trace_id: Optional[str] = None,
    **initial_metadata: Any,
) -> Generator[TraceLogger, None, None]:
    """
    Context manager for structured tracing across operations.
    
    Automatically logs operation start/end and tracks duration.
    
    Usage:
        with trace_context("deploy_service", service="payment") as trace:
            trace.log_metric("deployment.latency", 2.34)
    """
    trace_logger = TraceLogger(trace_id=trace_id)
    start_time = time.time()

    trace_logger.start_operation(operation_name, **initial_metadata)

    try:
        yield trace_logger
        success = True
    except Exception as exc:
        success = False
        trace_logger.log_error(operation_name, exc)
        raise
    finally:
        duration = time.time() - start_time
        trace_logger.end_operation(
            operation_name,
            success=success,
            duration_seconds=duration,
        )


@contextlib.contextmanager
def span_context(
    span_name: str,
    trace_logger: TraceLogger,
    **metadata: Any,
) -> Generator[TraceLogger, None, None]:
    """
    Context manager for nested spans within a trace.
    
    Usage:
        with trace_context("deploy_service") as trace:
            with span_context("validate_config", trace) as span:
                # Work within this span
                span.log_metric("config_size_bytes", 1024)
    """
    start_time = time.time()
    span_id = uuid4().hex[:8]

    child_logger = TraceLogger(
        trace_id=trace_logger.trace_id,
        span_id=span_id,
    )

    child_logger.start_operation(span_name, **metadata)

    try:
        yield child_logger
        success = True
    except Exception as exc:
        success = False
        child_logger.log_error(span_name, exc)
        raise
    finally:
        duration = time.time() - start_time
        child_logger.end_operation(
            span_name,
            success=success,
            duration_seconds=duration,
        )


class LogBuffer:
    """
    Buffer for capturing logs during scenario execution.
    
    Useful for replaying demo transcripts or storing logs for analysis.
    """

    def __init__(self, max_entries: int = 1000):
        self.entries: list[dict[str, Any]] = []
        self.max_entries = max_entries
        self.handler = logging.StreamHandler()
        self.handler.emit = self._capture_entry

    def _capture_entry(self, record: logging.LogRecord) -> None:
        """Capture a log entry."""
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        if hasattr(record, "trace_id"):
            entry["trace_id"] = record.trace_id
        if hasattr(record, "span_id"):
            entry["span_id"] = record.span_id

        self.entries.append(entry)

        # Respect max size
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def get_entries(self, trace_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Retrieve buffered entries, optionally filtered by trace_id."""
        if trace_id:
            return [e for e in self.entries if e.get("trace_id") == trace_id]
        return self.entries

    def clear(self) -> None:
        """Clear all buffered entries."""
        self.entries.clear()

    def to_json(self) -> str:
        """Export buffered entries as JSON."""
        return json.dumps(self.entries, indent=2)


# Module-level logger
logger = logging.getLogger(__name__)
