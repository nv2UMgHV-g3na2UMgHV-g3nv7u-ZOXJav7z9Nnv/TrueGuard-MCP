"""
gateways/llm_router.py
======================
Model fallback routing + hard job budget enforcement for TrueGuard-MCP.

This module is the single choke-point through which every LLM call in the
harness flows. It provides:

* :class:`TrueFoundryRouter` — an OpenAI-SDK client wrapper pointed at the
  TrueFoundry AI Gateway, with automatic failover from the primary model to
  a cheaper fallback on 429 / 504 / connection errors, and propagation of
  an ``X-Fallback-Executed`` telemetry header on every response.

* :class:`JobBudgetTracker` — a thread-safe, async-safe accumulator of
  cumulative USD cost per job execution. It supports two enforcement tiers:

  - ``DOWNGRADE_THRESHOLD`` (default ``$0.50``) — the router silently forces
    all subsequent calls onto the cheap fallback model.
  - ``HALT_THRESHOLD`` (default ``$0.75``) — the router raises
    :class:`BudgetExceededException` and refuses further calls until a human
    re-authorises the job.

* :class:`CostGuardrail` — a thin facade that owns the tracker and exposes
  the pricing table, the model-downgrade decision, and the halt check.

This revision raises the project's typed :class:`ModelRouterException` when
all attempts fail, logging through the standard library.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Final, Mapping

import httpx
import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from config import settings
from exceptions import (
    BudgetExceededException,
    ErrorContext,
    ModelRouterException,
    create_error_context,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TrueFoundryRouter",
    "CostGuardrail",
    "JobBudgetTracker",
    "LedgerEntry",
    "ModelPricing",
    "RoutedResponse",
    "BudgetExceededException",
    "ModelRouterException",
    "set_current_tracker",
    "get_current_tracker",
    "reset_current_tracker",
]


# ====================================================================== #
# Pricing table                                                          #
# ====================================================================== #
@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-million-token USD pricing for a single model."""

    prompt_usd_per_million: float
    completion_usd_per_million: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute the USD cost of a single completion."""
        return (
            (prompt_tokens / 1_000_000.0) * self.prompt_usd_per_million
            + (completion_tokens / 1_000_000.0) * self.completion_usd_per_million
        )


# Public list prices, correct as of writing. Override in tests as needed.
_DEFAULT_PRICING: Final[Mapping[str, ModelPricing]] = {
    "gpt-4o": ModelPricing(prompt_usd_per_million=2.50, completion_usd_per_million=10.00),
    "gpt-4o-mini": ModelPricing(prompt_usd_per_million=0.15, completion_usd_per_million=0.60),
    "o3-mini": ModelPricing(prompt_usd_per_million=1.10, completion_usd_per_million=4.40),
}


# ====================================================================== #
# Budget tracker                                                         #
# ====================================================================== #
@dataclass(slots=True)
class LedgerEntry:
    """A single LLM call's contribution to the job ledger."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    fallback_executed: bool
    timestamp: float = field(default_factory=time.time)


class JobBudgetTracker:
    """
    Thread-safe + async-safe per-job USD spend accumulator.

    All mutating operations take :attr:`_lock`, which is a regular
    :class:`threading.Lock`. Async callers still acquire it non-blockingly
    because the critical sections are pure-Python arithmetic with no I/O —
    the lock is never held across an ``await``.
    """

    def __init__(
        self,
        *,
        job_id: str,
        downgrade_threshold_usd: float,
        halt_threshold_usd: float,
    ) -> None:
        if halt_threshold_usd < downgrade_threshold_usd:
            raise ValueError(
                "halt_threshold_usd must be >= downgrade_threshold_usd "
                f"(got halt={halt_threshold_usd}, downgrade={downgrade_threshold_usd})"
            )
        self.job_id = job_id
        self.downgrade_threshold_usd = downgrade_threshold_usd
        self.halt_threshold_usd = halt_threshold_usd
        self._lock = threading.Lock()
        self._ledger: list[LedgerEntry] = []
        self._accumulated_usd: float = 0.0
        self._halted: bool = False
        self._downgraded: bool = False

    # ------------------------------------------------------------------ #
    # Read-only properties                                                #
    # ------------------------------------------------------------------ #
    @property
    def accumulated_usd(self) -> float:
        with self._lock:
            return self._accumulated_usd

    @property
    def is_halted(self) -> bool:
        with self._lock:
            return self._halted

    @property
    def is_downgraded(self) -> bool:
        with self._lock:
            return self._downgraded

    @property
    def ledger(self) -> list[LedgerEntry]:
        with self._lock:
            return list(self._ledger)

    # ------------------------------------------------------------------ #
    # Mutating operations                                                 #
    # ------------------------------------------------------------------ #
    def record(self, entry: LedgerEntry) -> None:
        """
        Record a completed LLM call and re-evaluate the enforcement tiers.

        Sets :attr:`_downgraded` once the soft threshold is crossed, and
        :attr:`_halted` once the hard threshold is crossed. Does **not**
        raise — the router decides how to react to the halt flag.
        """
        with self._lock:
            self._ledger.append(entry)
            self._accumulated_usd += entry.cost_usd

            if not self._downgraded and self._accumulated_usd >= self.downgrade_threshold_usd:
                self._downgraded = True
                logger.warning(
                    "budget.downgrade job_id=%s accumulated=$%.4f threshold=$%.2f",
                    self.job_id,
                    self._accumulated_usd,
                    self.downgrade_threshold_usd,
                )

            if not self._halted and self._accumulated_usd >= self.halt_threshold_usd:
                self._halted = True
                logger.error(
                    "budget.halt job_id=%s accumulated=$%.4f threshold=$%.2f",
                    self.job_id,
                    self._accumulated_usd,
                    self.halt_threshold_usd,
                )

    def reset(self) -> None:
        """Clear the ledger and enforcement flags. Used after human re-auth."""
        with self._lock:
            self._ledger.clear()
            self._accumulated_usd = 0.0
            self._halted = False
            self._downgraded = False
        logger.info("budget.reset job_id=%s", self.job_id)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the tracker state."""
        with self._lock:
            return {
                "job_id": self.job_id,
                "accumulated_usd": round(self._accumulated_usd, 6),
                "downgrade_threshold_usd": self.downgrade_threshold_usd,
                "halt_threshold_usd": self.halt_threshold_usd,
                "is_downgraded": self._downgraded,
                "is_halted": self._halted,
                "call_count": len(self._ledger),
            }


# ---------------------------------------------------------------------- #
# Per-job context propagation                                            #
# ---------------------------------------------------------------------- #
_CURRENT_TRACKER: Final[contextvars.ContextVar[JobBudgetTracker | None]] = (
    contextvars.ContextVar("trueguard_current_budget_tracker", default=None)
)


def set_current_tracker(tracker: JobBudgetTracker | None) -> contextvars.Token:
    """Install ``tracker`` as the ambient job tracker for the current context."""
    return _CURRENT_TRACKER.set(tracker)


def get_current_tracker() -> JobBudgetTracker | None:
    """Return the ambient job tracker, if any."""
    return _CURRENT_TRACKER.get()


def reset_current_tracker(token: contextvars.Token) -> None:
    """Restore the previous ambient tracker (context-manager style)."""
    _CURRENT_TRACKER.reset(token)


# ====================================================================== #
# Cost guardrail facade                                                  #
# ====================================================================== #
class CostGuardrail:
    """
    Pricing + enforcement facade consulted by :class:`TrueFoundryRouter`.

    Responsibilities
    ----------------
    * Own the model pricing table.
    * Compute per-call USD cost from usage metrics.
    * Decide, per request, whether to force a downgrade.
    * Raise :class:`BudgetExceededException` when a job is halted.
    """

    #: Mutable class-level pricing table — override in tests or at boot.
    PRICING: dict[str, ModelPricing] = dict(_DEFAULT_PRICING)

    def __init__(
        self,
        *,
        downgrade_threshold_usd: float = 0.50,
        halt_threshold_usd: float = 0.75,
        default_model: str | None = None,
        cheap_model: str | None = None,
    ) -> None:
        if halt_threshold_usd < downgrade_threshold_usd:
            raise ValueError("halt_threshold_usd must be >= downgrade_threshold_usd")
        self.downgrade_threshold_usd = downgrade_threshold_usd
        self.halt_threshold_usd = halt_threshold_usd
        self.default_model = default_model or settings.primary_model
        self.cheap_model = cheap_model or settings.fallback_model

    def price_for(self, model: str) -> ModelPricing:
        """
        Return pricing for ``model``.

        Unknown models fall back to the pricing of :attr:`default_model` so
        that a novel gateway alias cannot silently zero out accounting.
        """
        if model in self.PRICING:
            return self.PRICING[model]
        logger.warning("pricing.unknown_model model=%s using=%s", model, self.default_model)
        return self.PRICING.get(
            self.default_model,
            ModelPricing(prompt_usd_per_million=2.50, completion_usd_per_million=10.00),
        )

    def cost_of(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Compute the USD cost of a single call."""
        return self.price_for(model).cost(prompt_tokens, completion_tokens)

    def make_tracker(self, job_id: str) -> JobBudgetTracker:
        """Construct a fresh tracker bound to this guardrail's thresholds."""
        return JobBudgetTracker(
            job_id=job_id,
            downgrade_threshold_usd=self.downgrade_threshold_usd,
            halt_threshold_usd=self.halt_threshold_usd,
        )

    def enforce(
        self,
        tracker: JobBudgetTracker,
        *,
        context: ErrorContext | None = None,
    ) -> None:
        """
        Raise :class:`BudgetExceededException` if the job has been halted.

        Called by the router *before* dispatching a request so we never
        burn tokens on a job that has already exhausted its budget.
        """
        if tracker.is_halted:
            ctx = context or create_error_context(
                trace_id=tracker.job_id,
                span_id="preflight",
                operation="budget.enforce",
                accumulated_usd=tracker.accumulated_usd,
                halt_usd=tracker.halt_threshold_usd,
            )
            raise BudgetExceededException(
                accumulated_usd=tracker.accumulated_usd,
                halt_usd=tracker.halt_threshold_usd,
                context=ctx,
            )

    def choose_model(self, tracker: JobBudgetTracker, requested_model: str) -> str:
        """
        Return the model to actually dispatch.

        If the tracker has crossed the soft threshold, all traffic is forced
        onto :attr:`cheap_model` regardless of what the caller asked for.
        """
        if tracker.is_downgraded and requested_model != self.cheap_model:
            logger.info(
                "router.force_downgrade requested=%s downgraded_to=%s accumulated=$%.4f",
                requested_model,
                self.cheap_model,
                tracker.accumulated_usd,
            )
            return self.cheap_model
        return requested_model


# ====================================================================== #
# Routed response wrapper                                                #
# ====================================================================== #
@dataclass(slots=True)
class RoutedResponse:
    """
    Envelope around an OpenAI :class:`ChatCompletion` carrying routing metadata.

    ``headers`` contains the ``X-Fallback-Executed`` flag plus any diagnostic
    headers the caller wants to forward to downstream services.
    """

    completion: ChatCompletion
    model_used: str
    fallback_executed: bool
    attempts: list[str]
    cost_usd: float
    headers: dict[str, str] = field(default_factory=dict)


# ====================================================================== #
# Router                                                                 #
# ====================================================================== #
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({429, 504})
_PRIMARY_TIMEOUT_S: Final[float] = 5.0


class _RetryableRouterError(Exception):
    """Internal marker for failures that should trigger failover."""

    def __init__(
        self,
        *,
        reason: str,
        status_code: int | None,
        detail: str,
    ) -> None:
        super().__init__(f"{reason} ({status_code}): {detail}")
        self.reason = reason
        self.status_code = status_code
        self.detail = detail


class TrueFoundryRouter:
    """
    LLM router that dispatches through the TrueFoundry AI Gateway with
    automatic failover and hard budget enforcement.

    Parameters
    ----------
    guardrail:
        The :class:`CostGuardrail` used for pricing and enforcement. If
        ``None``, one is constructed with the project defaults from
        :mod:`config`.
    fallback_chain:
        Ordered list of models to try after the primary fails. Defaults to
        ``[settings.fallback_model, "o3-mini"]``.
    primary_timeout_s:
        Per-attempt timeout. A timeout is treated as a retryable failure and
        triggers failover to the next model in the chain.
    client:
        Optional injected :class:`openai.AsyncOpenAI` client. Used by tests
        to substitute a stub.
    """

    def __init__(
        self,
        *,
        guardrail: CostGuardrail | None = None,
        fallback_chain: list[str] | None = None,
        primary_timeout_s: float = _PRIMARY_TIMEOUT_S,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.guardrail = guardrail or CostGuardrail(
            downgrade_threshold_usd=settings.job_budget_usd,
            halt_threshold_usd=settings.job_budget_usd * 1.5,
        )
        self.primary_model = settings.primary_model
        self.fallback_chain: list[str] = fallback_chain or [
            settings.fallback_model,
            "o3-mini",
        ]
        self.primary_timeout_s = primary_timeout_s
        self._client = client or AsyncOpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout=httpx.Timeout(30.0, connect=10.0),
            max_retries=0,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.0,
        tracker: JobBudgetTracker | None = None,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> RoutedResponse:
        """
        Dispatch a chat completion with failover and budget enforcement.

        Raises
        ------
        BudgetExceededException
            If the tracker has crossed the hard halt threshold.
        ModelRouterException
            If every model in the chain failed.
        """
        requested_model = model or self.primary_model

        # -------- 1. Pre-flight budget enforcement -------- #
        if tracker is not None:
            self.guardrail.enforce(tracker)
            dispatch_model = self.guardrail.choose_model(tracker, requested_model)
        else:
            dispatch_model = requested_model

        # -------- 2. Build the failover chain -------- #
        chain = self._build_chain(dispatch_model)

        # -------- 3. Dispatch with failover -------- #
        attempts: list[str] = []
        last_exc: Exception | None = None
        primary_exc: Exception | None = None

        for index, candidate in enumerate(chain):
            attempts.append(candidate)
            is_fallback = index > 0
            try:
                completion, elapsed = await self._attempt(
                    model=candidate,
                    messages=messages,
                    temperature=temperature,
                    timeout=self.primary_timeout_s,
                    extra_headers=extra_headers,
                )
            except _RetryableRouterError as exc:
                last_exc = exc
                if index == 0:
                    primary_exc = exc
                logger.warning(
                    "router.attempt_failed model=%s attempt=%d reason=%s status=%s",
                    candidate,
                    index,
                    exc.reason,
                    exc.status_code,
                )
                continue
            except Exception as exc:
                logger.exception("router.attempt_fatal model=%s attempt=%d", candidate, index)
                raise

            # -------- 4. Account for spend -------- #
            usage = completion.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            cost = self.guardrail.cost_of(
                model=candidate,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            if tracker is not None:
                tracker.record(
                    LedgerEntry(
                        model=candidate,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost_usd=cost,
                        fallback_executed=is_fallback,
                    )
                )

            headers: dict[str, str] = {
                "X-Fallback-Executed": "true" if is_fallback else "false",
                "X-Model-Used": candidate,
                "X-Latency-Ms": f"{elapsed * 1000:.1f}",
                "X-Cost-USD": f"{cost:.6f}",
            }

            logger.info(
                "router.success model=%s fallback=%s cost=$%.6f elapsed_ms=%.1f",
                candidate,
                is_fallback,
                cost,
                elapsed * 1000,
            )
            return RoutedResponse(
                completion=completion,
                model_used=candidate,
                fallback_executed=is_fallback,
                attempts=attempts,
                cost_usd=cost,
                headers=headers,
            )

        # -------- 5. All providers failed -------- #
        ctx = create_error_context(
            trace_id=tracker.job_id if tracker else "no-tracker",
            span_id="router.complete",
            operation="model_routing",
            attempts=attempts,
        )
        raise ModelRouterException(
            primary_model=chain[0],
            fallback_model=chain[-1],
            primary_error=primary_exc or last_exc or RuntimeError("primary failed"),
            fallback_error=last_exc or RuntimeError("all attempts failed"),
            context=ctx,
        )

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #
    def _build_chain(self, dispatch_model: str) -> list[str]:
        """
        Return an ordered, de-duplicated failover chain beginning with the
        dispatch model, then the configured fallbacks.
        """
        seen: set[str] = set()
        chain: list[str] = []
        for candidate in [dispatch_model, *self.fallback_chain]:
            if candidate not in seen:
                seen.add(candidate)
                chain.append(candidate)
        return chain

    async def _attempt(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        timeout: float,
        extra_headers: dict[str, str] | None,
    ) -> tuple[ChatCompletion, float]:
        """
        Make a single call to the gateway, mapping failures to
        :class:`_RetryableRouterError` or letting fatal errors propagate.
        """
        headers: dict[str, str] = {
            "X-TrueGuard-Model": model,
            "X-Request-Id": uuid.uuid4().hex[:16],
        }
        if extra_headers:
            headers.update(extra_headers)

        start = time.perf_counter()
        try:
            completion = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    extra_headers=headers,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise _RetryableRouterError(
                reason="timeout",
                status_code=504,
                detail=f"attempt exceeded {timeout:.1f}s",
            ) from exc
        except openai.APIStatusError as exc:
            if exc.status_code in _RETRYABLE_STATUS:
                raise _RetryableRouterError(
                    reason=f"http_{exc.status_code}",
                    status_code=exc.status_code,
                    detail=str(exc),
                ) from exc
            raise
        except (openai.APIConnectionError, httpx.TransportError) as exc:
            raise _RetryableRouterError(
                reason="connection_error",
                status_code=None,
                detail=str(exc),
            ) from exc

        elapsed = time.perf_counter() - start
        return completion, elapsed
