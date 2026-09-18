"""
fixtures/stubs.py
=================
Deterministic, in-memory stubs for the demo runner.

These replace the real OpenAI SDK client and the real FastMCP client so the
demo runs with zero external dependencies. The production code paths (router,
governance, approval store) are exercised unchanged.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
import openai


class _StubUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class _StubCompletion:
    def __init__(self, model: str, content: str, prompt: int, completion: int) -> None:
        self.id = f"chatcmpl-demo-{uuid.uuid4().hex[:8]}"
        self.model = model
        self.usage = _StubUsage(prompt, completion)
        self.choices = [
            type("Choice", (), {
                "index": 0,
                "message": type("Msg", (), {"role": "assistant", "content": content})(),
                "finish_reason": "stop",
            })()
        ]


class _StubChatCompletions:
    def __init__(self, script: list[tuple[str, Any]]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _StubCompletion:
        self.calls.append(kwargs)
        model = kwargs["model"]
        if not self._script:
            return _StubCompletion(model, "{}", 100, 50)
        expected, outcome = self._script.pop(0)
        if model != expected:
            raise AssertionError(f"Dispatcher mismatch: expected {expected}, got {model}")
        if isinstance(outcome, int):
            request = httpx.Request("POST", "https://gateway.truefoundry.ai/v1/chat/completions")
            response = httpx.Response(outcome, request=request)
            raise openai.APIStatusError(
                f"simulated status {outcome}", response=response, body=None
            )
        if outcome == "timeout":
            await asyncio.sleep(10)
            raise AssertionError("unreachable")
        return _StubCompletion(model, str(outcome), 800, 220)


class _StubChat:
    def __init__(self, completions: _StubChatCompletions) -> None:
        self.completions = completions


class StubClient:
    """Duck-typed replacement for :class:`openai.AsyncOpenAI`."""

    def __init__(self, script: list[tuple[str, Any]]) -> None:
        self._completions = _StubChatCompletions(script)
        self.chat = _StubChat(self._completions)

    @property
    def dispatched(self) -> list[str]:
        return [c["model"] for c in self._completions.calls]


async def stub_mcp_client(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Deterministic stand-in for the FastMCP client."""
    await asyncio.sleep(0.02)
    now = datetime.now(tz=timezone.utc)

    if tool_name == "fetch_server_logs":
        service = arguments.get("service_name", "unknown")
        entries = [
            f"{now.isoformat()} ERROR [{service}] upstream timeout after 5000ms",
            f"{now.isoformat()} WARN  [{service}] retry 3/3 exhausted",
            f"{now.isoformat()} ERROR [{service}] HTTP 500 returned to caller",
        ]
        return {
            "service_name": service,
            "lines_requested": int(arguments.get("lines", 100)),
            "lines_returned": len(entries),
            "entries": entries,
            "fetched_at": now.isoformat(),
        }

    if tool_name == "check_container_health":
        return {
            "service_name": arguments.get("service_name", "unknown"),
            "status": "degraded",
            "cpu_pct": 78.5,
            "memory_pct": 82.1,
            "restarts_24h": 4,
            "checked_at": now.isoformat(),
        }

    raise ValueError(f"Stub MCP client does not implement '{tool_name}'.")
