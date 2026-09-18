"""
scenarios/scenario_a.py
=======================
Scenario A — Autonomous remediation (READ_ONLY only).
"""

from __future__ import annotations

from typing import Any, Final

from agents.devops_agent import SYSTEM_PROMPT, DevOpsAgent
from config import settings
from evals.eval_harness import EvalContext, format_scorecard, run_evals
from fixtures.stubs import StubClient, stub_mcp_client
from gateways.llm_router import CostGuardrail, TrueFoundryRouter
from observability.tracer import Tracer, trace_to_timeline
from scenarios._common import Palette, banner, err, kv, ok, step
from types import ScenarioResult


_PLAN: Final[list[dict[str, Any]]] = [
    {
        "step_id": "step-1",
        "description": "Warm up the upstream connection pool.",
        "tool_name": None,
        "tool_arguments": {},
        "risk_level": "MANUAL",
        "rationale": "Connection pool re-warm is a config change owned by SRE.",
    },
]


async def scenario_a(pal: Palette) -> ScenarioResult:
    banner(pal, "SCENARIO A — Autonomous Remediation (READ_ONLY only)")

    import json

    script = [
        ("gpt-4o", json.dumps({
            "severity": "SEV3",
            "diagnosis": (
                "Upstream dependency is timing out at 5s; the payment-service-v2 "
                "worker pool is degraded but not down."
            ),
            "confidence": 0.82,
            "evidence": [
                {"source": "fetch_server_logs",
                 "observation": "upstream timeout after 5000ms",
                 "supports_hypothesis": True},
            ],
            "remediation_plan": _PLAN,
        })),
    ]

    client = StubClient(script)
    guardrail = CostGuardrail(
        downgrade_threshold_usd=0.50,
        halt_threshold_usd=0.75,
        default_model=settings.primary_model,
        cheap_model=settings.fallback_model,
    )
    router = TrueFoundryRouter(guardrail=guardrail, client=client)  # type: ignore[arg-type]
    tracer = Tracer()
    agent = DevOpsAgent(router=router, guardrail=guardrail, mcp_client=stub_mcp_client)

    step(pal, "Step 1 — OBSERVE", "fetch_server_logs + check_container_health")
    step(pal, "Step 2 — DIAGNOSE", f"{settings.primary_model} via TrueFoundry AI Gateway")
    step(pal, "Step 3 — PLAN", "no HIGH_RISK tools → autonomous path")

    trace = tracer.start_trace(
        alert_text="Alert: High 500 error rate on payment-service-v2",
        incident_id="inc-demo-A",
        affected_service="payment-service-v2",
    )

    report = await agent.handle_alert(
        "Alert: High 500 error rate on payment-service-v2",
        affected_service="payment-service-v2",
        incident_id="inc-demo-A",
    )
    tracer.end_trace(trace)

    ok(pal, f"Diagnosis: {report.diagnosis[:80]}…")
    ok(pal, f"Confidence: {report.confidence:.2f}")
    ok(pal, f"High-risk actions proposed: {len(report.high_risk_actions_requiring_approval)}")
    ok(pal, f"Total cost: ${report.total_cost_usd:.4f}")
    ok(pal, f"Run status: {report.run_status}")

    scorecard = run_evals(EvalContext(
        trace=trace,
        tool_calls=[
            {"name": "fetch_server_logs",
             "arguments": {"service_name": "payment-service-v2", "lines": 200}},
            {"name": "check_container_health",
             "arguments": {"service_name": "payment-service-v2"}},
        ],
        outgoing_prompts=["ALERT: High 500 error rate on payment-service-v2"],
    ))

    print()
    print(pal.bold("  Eval Scorecard"))
    print(format_scorecard(scorecard))
    print()
    if scorecard["passed"]:
        ok(pal, pal.bold(pal.green(f"SCENARIO A PASSED — {scorecard['score']}")))
    else:
        err(pal, pal.bold(pal.red(f"SCENARIO A FAILED — {scorecard['score']}")))

    return ScenarioResult(
        trace=trace_to_timeline(trace)["trace"],
        scorecard=scorecard,  # type: ignore[typeddict-item]
        halted=False,
    )
