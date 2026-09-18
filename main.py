"""
main.py
=======
FastAPI composition root for TrueGuard-MCP.

Wires up logging, config, exception handlers, and routers. This is the
canonical process entrypoint — run it with::

    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app import router as dashboard_router
from config import settings
from exceptions import (
    ApprovalDeniedException,
    ApprovalTimeoutException,
    BudgetExceededException,
    ModelRouterException,
    TrueGuardException,
)
from gateways.approval_workflow import router as approval_router
from logging_utils import configure_logging

# ---- Logging first, before anything else reads settings ---- #
configure_logging(level=settings.log_level, structured=False)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "TrueGuard-MCP starting: gateway=%s primary=%s fallback=%s budget=$%.2f",
        settings.openai_base_url,
        settings.primary_model,
        settings.fallback_model,
        settings.job_budget_usd,
    )
    yield
    logger.info("TrueGuard-MCP shutting down")


app = FastAPI(
    title="TrueGuard-MCP",
    version="0.2.0",
    description="Agentic DevOps & Incident Escalation Harness",
    lifespan=lifespan,
)

app.include_router(approval_router)
app.include_router(dashboard_router)


# ====================================================================== #
# Exception handlers — map domain exceptions to HTTP responses           #
# ====================================================================== #
@app.exception_handler(BudgetExceededException)
async def _budget_exceeded(_: Request, exc: BudgetExceededException) -> JSONResponse:
    logger.warning("HTTP 402: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        content={"error": "budget_exceeded", **exc.to_dict()},
    )


@app.exception_handler(ApprovalTimeoutException)
async def _approval_timeout(_: Request, exc: ApprovalTimeoutException) -> JSONResponse:
    logger.warning("HTTP 408: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_408_REQUEST_TIMEOUT,
        content={"error": "approval_timeout", **exc.to_dict()},
    )


@app.exception_handler(ApprovalDeniedException)
async def _approval_denied(_: Request, exc: ApprovalDeniedException) -> JSONResponse:
    logger.warning("HTTP 403: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"error": "approval_denied", **exc.to_dict()},
    )


@app.exception_handler(ModelRouterException)
async def _router_failure(_: Request, exc: ModelRouterException) -> JSONResponse:
    logger.error("HTTP 502: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"error": "model_router_failure", **exc.to_dict()},
    )


@app.exception_handler(TrueGuardException)
async def _generic(_: Request, exc: TrueGuardException) -> JSONResponse:
    logger.error("HTTP 500: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "trueguard_error", **exc.to_dict()},
    )
