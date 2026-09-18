"""
app.py
======
Dashboard + incident routes for TrueGuard-MCP.

This module exposes an :data:`APIRouter` (mounted by :mod:`main`) rather than
a standalone FastAPI app, so logging setup, exception handlers, and lifespan
concerns are owned by the composition root.

Routes
------
* ``POST /api/v1/incident/trigger``      — ingest a demo incident, start an
                                            agent run in the background.
* ``GET  /api/v1/traces``                — recent traces (summaries).
* ``GET  /api/v1/traces/{trace_id}``     — full waterfall timeline for one
                                            trace.
* ``GET  /api/v1/incidents/{id}/report`` — final ``IncidentReportResponse``.
* ``GET  /``                             — single-file Tailwind dashboard.

The dashboard is a single HTML string served with ``text/html`` — no build
step, no separate static directory. It polls the trace endpoint every second
and re-renders the waterfall, model indicator, cost meter, and HITL modal.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Final

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agents.devops_agent import DevOpsAgent, IncidentReportResponse
from observability.tracer import (
    SpanKind,
    SpanStatus,
    Trace,
    trace_to_timeline,
    tracer,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ====================================================================== #
# In-process incident registry                                           #
# ====================================================================== #
class _IncidentRegistry:
    """Maps ``incident_id`` → ``(trace_id, report | None)``. Process-local."""

    def __init__(self) -> None:
        self._traces: dict[str, str] = {}
        self._reports: dict[str, IncidentReportResponse] = {}
        self._lock = asyncio.Lock()

    async def bind_trace(self, incident_id: str, trace_id: str) -> None:
        async with self._lock:
            self._traces[incident_id] = trace_id

    async def set_report(
        self, incident_id: str, report: IncidentReportResponse
    ) -> None:
        async with self._lock:
            self._reports[incident_id] = report

    async def trace_id(self, incident_id: str) -> str | None:
        async with self._lock:
            return self._traces.get(incident_id)

    async def report(
        self, incident_id: str
    ) -> IncidentReportResponse | None:
        async with self._lock:
            return self._reports.get(incident_id)


incidents: Final[_IncidentRegistry] = _IncidentRegistry()


# ====================================================================== #
# Request / response models                                              #
# ====================================================================== #
class TriggerIncidentRequest(BaseModel):
    """Request body for ``POST /api/v1/incident/trigger``."""

    alert_text: str = Field(
        ...,
        min_length=5,
        description=(
            "Raw alert string, e.g. 'Alert: High 500 error rate on "
            "payment-service-v2'."
        ),
    )
    affected_service: str = Field(
        ...,
        min_length=1,
        description="Logical service identifier.",
    )
    incident_id: str | None = Field(
        default=None,
        description="Optional stable ID; generated if omitted.",
    )


class TriggerIncidentResponse(BaseModel):
    """Immediate response acknowledging an incident trigger."""

    incident_id: str
    trace_id: str
    status: str = "accepted"
    message: str = "Investigation started. Poll /api/v1/traces/{trace_id}."


class TraceSummary(BaseModel):
    """Compact trace row for the ``GET /api/v1/traces`` list."""

    trace_id: str
    incident_id: str | None
    alert_text: str
    started_at: str
    is_complete: bool
    active_model: str | None
    fallback_executed: bool
    total_cost_usd: float
    span_count: int


# ====================================================================== #
# Incident orchestration                                                 #
# ====================================================================== #
async def _run_agent(
    *,
    incident_id: str,
    alert_text: str,
    affected_service: str,
    trace_id: str,
) -> None:
    """
    Background task: run the agent and persist the report.

    The agent emits its own instrumentation into the trace via its
    ``trace_context`` / ``span_context`` calls. This runner only needs to
    bookend the run with a final ``end_trace`` call so the dashboard knows
    the trace is complete.
    """
    live_trace = tracer.get_live_trace(trace_id)
    if live_trace is None:  # pragma: no cover — invariant
        logger.error("app.trace_missing trace_id=%s", trace_id)
        return

    agent = DevOpsAgent()

    try:
        report = await agent.handle_alert(
            alert_text,
            affected_service=affected_service,
            incident_id=incident_id,
        )
        await incidents.set_report(incident_id, report)
        logger.info(
            "app.agent_complete incident_id=%s run_status=%s cost=$%.4f",
            incident_id,
            report.run_status,
            report.total_cost_usd,
        )
    except Exception:  # pragma: no cover — defensive
        logger.exception("app.agent_failed incident_id=%s", incident_id)
    finally:
        tracer.end_trace(live_trace)


# ====================================================================== #
# Routes — incident                                                      #
# ====================================================================== #
@router.post(
    "/api/v1/incident/trigger",
    response_model=TriggerIncidentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_incident(
    payload: TriggerIncidentRequest,
    background_tasks: BackgroundTasks,
) -> TriggerIncidentResponse:
    """
    Ingest a demo incident and start an agent run.

    Returns immediately with a ``trace_id`` the client polls for updates.
    The actual agent execution runs in a FastAPI background task.
    """
    incident_id = payload.incident_id or f"inc-{uuid.uuid4().hex[:12]}"
    trace = tracer.start_trace(
        alert_text=payload.alert_text,
        incident_id=incident_id,
        affected_service=payload.affected_service,
    )
    await incidents.bind_trace(incident_id, trace.trace_id)

    background_tasks.add_task(
        _run_agent,
        incident_id=incident_id,
        alert_text=payload.alert_text,
        affected_service=payload.affected_service,
        trace_id=trace.trace_id,
    )

    logger.info(
        "api.incident_triggered incident_id=%s trace_id=%s service=%s",
        incident_id,
        trace.trace_id,
        payload.affected_service,
    )
    return TriggerIncidentResponse(
        incident_id=incident_id,
        trace_id=trace.trace_id,
    )


# ====================================================================== #
# Routes — traces                                                        #
# ====================================================================== #
@router.get("/api/v1/traces", response_model=list[TraceSummary])
async def list_traces(limit: int = 25) -> list[TraceSummary]:
    """Return compact summaries of the most recent traces."""
    traces = tracer.list_traces(limit=limit)
    return [
        TraceSummary(
            trace_id=t.trace_id,
            incident_id=t.incident_id,
            alert_text=t.alert_text,
            started_at=t.started_at.isoformat(),
            is_complete=t.is_complete,
            active_model=t.active_model,
            fallback_executed=t.fallback_executed,
            total_cost_usd=round(t.total_cost_usd, 6),
            span_count=len(t.spans),
        )
        for t in traces
    ]


@router.get("/api/v1/traces/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    """
    Return the full waterfall timeline for a trace.

    The dashboard polls this every second. A 404 means the trace has been
    evicted from the ring buffer.
    """
    trace: Trace | None = tracer.get_trace(trace_id)
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown trace: {trace_id}",
        )
    return trace_to_timeline(trace)


# ====================================================================== #
# Routes — incident report                                               #
# ====================================================================== #
@router.get(
    "/api/v1/incidents/{incident_id}/report",
    response_model=IncidentReportResponse,
)
async def get_incident_report(incident_id: str) -> IncidentReportResponse:
    """Return the final report for an incident, once the agent has finished."""
    report = await incidents.report(incident_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No report yet for incident {incident_id}",
        )
    return report


# ====================================================================== #
# Dashboard                                                              #
# ====================================================================== #
DASHBOARD_HTML: Final[str] = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>TrueGuard-MCP :: Live Incident Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
<style>
  :root { color-scheme: dark; }
  body { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  .span-bar { transition: width 0.3s ease, background-color 0.3s ease; }
  .pulse-dot { animation: pulse 1.4s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.3 } }
  .glow-green { box-shadow: 0 0 12px rgba(34,197,94,0.6); }
  .glow-red   { box-shadow: 0 0 12px rgba(239,68,68,0.6); }
  .glow-amber { box-shadow: 0 0 12px rgba(245,158,11,0.6); }
  [x-cloak] { display: none !important; }
</style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">

<div x-data="dashboard()" x-init="init()" class="max-w-7xl mx-auto p-6 space-y-6">

  <!-- Header -->
  <header class="flex items-center justify-between border-b border-slate-800 pb-4">
    <div class="flex items-center gap-3">
      <div class="w-3 h-3 rounded-full bg-emerald-400 pulse-dot"></div>
      <h1 class="text-2xl font-bold tracking-tight">TrueGuard<span class="text-emerald-400">-MCP</span></h1>
      <span class="text-xs text-slate-500 uppercase tracking-widest">Live Incident Dashboard</span>
    </div>
    <div class="flex items-center gap-3 text-xs">
      <span class="px-3 py-1 rounded-full bg-slate-800 border border-slate-700"
            :class="activeModel === 'gpt-4o' ? 'text-emerald-300 glow-green' : (activeModel ? 'text-amber-300 glow-amber' : 'text-slate-400')"
            x-text="activeModel ? `MODEL · ${activeModel}` : 'MODEL · idle'"></span>
      <span x-show="fallbackExecuted" x-cloak
            class="px-3 py-1 rounded-full bg-amber-500/20 border border-amber-500/50 text-amber-300 glow-amber">
        ⚡ FALLBACK EXECUTED
      </span>
    </div>
  </header>

  <!-- Trigger row -->
  <section class="grid grid-cols-1 lg:grid-cols-3 gap-4">
    <div class="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-4">
      <label class="text-xs uppercase tracking-widest text-slate-500">Trigger Demo Incident</label>
      <div class="flex gap-3 mt-2">
        <input x-model="alertText" type="text" placeholder="Alert: High 500 error rate on payment-service-v2"
               class="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50" />
        <input x-model="service" type="text" placeholder="payment-service-v2"
               class="w-56 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50" />
        <button @click="trigger()" :disabled="triggering"
                class="px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 text-slate-950 text-sm font-semibold">
          <span x-show="!triggering">▶ Trigger</span><span x-show="triggering">⏳ Running…</span>
        </button>
      </div>
      <p class="text-xs text-slate-500 mt-2" x-show="lastIncident">
        Last incident: <span class="font-mono" x-text="lastIncident"></span>
      </p>
    </div>

    <!-- Cost meter -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
      <div class="flex items-center justify-between">
        <label class="text-xs uppercase tracking-widest text-slate-500">Cost Meter</label>
        <span x-show="costUsd >= 0.50 && costUsd < 0.75" x-cloak
              class="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/20 border border-amber-500/50 text-amber-300">
          ⚠ BUDGET WARNING
        </span>
        <span x-show="costUsd >= 0.75" x-cloak
              class="text-[10px] px-2 py-0.5 rounded-full bg-red-500/20 border border-red-500/50 text-red-300 glow-red">
          🛑 HALTED
        </span>
      </div>
      <div class="mt-3 flex items-baseline gap-2">
        <span class="text-3xl font-bold" :class="costUsd >= 0.75 ? 'text-red-400' : (costUsd >= 0.50 ? 'text-amber-400' : 'text-emerald-400')"
              x-text="'$' + costUsd.toFixed(4)"></span>
        <span class="text-slate-500 text-sm">/ $0.50 soft · $0.75 halt</span>
      </div>
      <div class="mt-2 h-2 rounded-full bg-slate-800 overflow-hidden">
        <div class="h-full span-bar"
             :class="costUsd >= 0.75 ? 'bg-red-500' : (costUsd >= 0.50 ? 'bg-amber-500' : 'bg-emerald-500')"
             :style="`width: ${Math.min(100, (costUsd / 0.75) * 100)}%`"></div>
      </div>
      <div class="mt-2 flex justify-between text-[10px] text-slate-500">
        <span>tokens in: <span x-text="tokensIn"></span></span>
        <span>tokens out: <span x-text="tokensOut"></span></span>
      </div>
    </div>
  </section>

  <!-- Waterfall -->
  <section class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-sm uppercase tracking-widest text-slate-400">Trace Timeline</h2>
      <div class="flex items-center gap-3 text-[11px] text-slate-500">
        <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-emerald-500"></span>SUCCESS</span>
        <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-amber-500"></span>FALLBACK</span>
        <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-red-500"></span>BLOCKED</span>
        <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-sky-500"></span>RUNNING</span>
      </div>
    </div>

    <template x-if="!trace">
      <p class="text-slate-500 text-sm py-8 text-center">No trace loaded. Trigger an incident above.</p>
    </template>

    <template x-if="trace">
      <div>
        <div class="flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-500 mb-4">
          <span>trace: <span class="font-mono text-slate-300" x-text="trace.trace.trace_id.slice(0,16)+'…'"></span></span>
          <span>incident: <span class="font-mono text-slate-300" x-text="trace.trace.incident_id || '—'"></span></span>
          <span>service: <span class="font-mono text-slate-300" x-text="trace.trace.affected_service || '—'"></span></span>
          <span>spans: <span class="font-mono text-slate-300" x-text="trace.trace.span_count"></span></span>
          <span x-show="trace.trace.is_complete" class="text-emerald-400">✓ complete</span>
          <span x-show="!trace.trace.is_complete" class="text-sky-400">● running</span>
        </div>

        <div class="space-y-1">
          <template x-for="row in rows" :key="row.span.span_id">
            <div class="group grid grid-cols-12 gap-2 items-center text-xs hover:bg-slate-800/40 rounded px-2 py-1">
              <div class="col-span-4 flex items-center gap-2 min-w-0">
                <span class="w-1 h-4 rounded" :class="statusBg(row.span.status)"></span>
                <span class="truncate" :style="`padding-left: ${row.depth * 14}px`">
                  <span class="text-slate-200" x-text="row.span.name"></span>
                  <span class="text-slate-600 ml-1">· <span x-text="row.span.kind"></span></span>
                </span>
              </div>
              <div class="col-span-6 relative h-5 bg-slate-950/60 rounded overflow-hidden">
                <div class="absolute inset-y-0 span-bar rounded"
                     :class="statusBg(row.span.status)"
                     :style="`left:${row.offsetPct}%; width:${row.widthPct}%`"></div>
                <span class="absolute inset-y-0 right-2 flex items-center text-[10px] text-slate-400"
                      x-text="row.span.latency_ms !== null ? row.span.latency_ms.toFixed(0)+'ms' : '…'"></span>
              </div>
              <div class="col-span-2 flex items-center justify-end gap-2 text-[11px]">
                <span x-show="row.span.model_used" class="text-slate-400 font-mono truncate" x-text="row.span.model_used"></span>
                <span x-show="row.span.cost_usd > 0" class="text-emerald-400 font-mono" x-text="'$'+row.span.cost_usd.toFixed(4)"></span>
                <span x-show="row.span.error" class="text-red-400" :title="row.span.error">⚠</span>
              </div>
            </div>
          </template>
        </div>
      </div>
    </template>
  </section>

  <!-- HITL pending approvals -->
  <section x-show="pending.length > 0" x-cloak
           class="bg-amber-950/40 border border-amber-600/60 rounded-2xl p-4 space-y-3">
    <div class="flex items-center gap-2">
      <span class="w-2 h-2 rounded-full bg-amber-400 pulse-dot"></span>
      <h2 class="text-sm uppercase tracking-widest text-amber-300">Human Approval Required</h2>
    </div>

    <template x-for="card in pending" :key="card.request_id">
      <div class="bg-slate-900 border border-amber-700/40 rounded-xl p-4 space-y-3">
        <div class="flex items-start justify-between gap-4">
          <div>
            <p class="text-sm font-semibold" x-text="card.incident.title"></p>
            <p class="text-xs text-slate-400 mt-1">
              <span class="px-2 py-0.5 rounded bg-red-500/20 border border-red-500/40 text-red-300 mr-2"
                    x-text="card.incident.severity"></span>
              <span x-text="card.incident.affected_service"></span>
            </p>
          </div>
          <div class="text-right text-xs text-slate-400">
            <div>Cost so far</div>
            <div class="text-emerald-400 font-mono text-lg" x-text="'$'+card.estimated_cost_usd.toFixed(4)"></div>
            <div class="text-slate-500">cap <span x-text="'$'+card.job_budget_usd.toFixed(2)"></span></div>
          </div>
        </div>

        <div class="bg-slate-950 rounded-lg p-3 text-xs space-y-1">
          <div class="text-slate-500 uppercase tracking-widest text-[10px]">Proposed Tool</div>
          <div class="font-mono text-amber-300" x-text="card.invocation.tool_name"></div>
          <pre class="text-slate-400 mt-1 overflow-x-auto" x-text="JSON.stringify(card.invocation.arguments, null, 2)"></pre>
          <p class="text-slate-400 italic mt-2" x-text="card.invocation.risk_rationale"></p>
        </div>

        <div class="flex gap-3">
          <button @click="decide(card.request_id, true)"
                  class="flex-1 px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-sm">
            ✅ Approve &amp; Execute
          </button>
          <button @click="decide(card.request_id, false)"
                  class="flex-1 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white font-semibold text-sm">
            🛑 Deny
          </button>
        </div>
        <p class="text-[10px] text-slate-500 font-mono">request: <span x-text="card.request_id"></span></p>
      </div>
    </template>
  </section>

  <!-- Raw JSON viewer -->
  <section class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
    <details>
      <summary class="text-xs uppercase tracking-widest text-slate-500 cursor-pointer">Raw trace JSON</summary>
      <pre class="mt-3 text-[11px] text-slate-400 overflow-x-auto max-h-72" x-text="trace ? JSON.stringify(trace, null, 2) : '—'"></pre>
    </details>
  </section>

</div>

<script>
function dashboard() {
  return {
    // form
    alertText: "Alert: High 500 error rate on payment-service-v2",
    service: "payment-service-v2",
    triggering: false,

    // state
    trace: null,
    traceId: null,
    lastIncident: null,
    pending: [],
    pollHandle: null,

    init() {
      this.pollHandle = setInterval(() => this.poll(), 1000);
      this.poll();
    },

    async trigger() {
      this.triggering = true;
      try {
        const res = await fetch("/api/v1/incident/trigger", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            alert_text: this.alertText,
            affected_service: this.service,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        this.traceId = data.trace_id;
        this.lastIncident = data.incident_id;
        this.trace = null;
        this.poll();
      } catch (e) {
        console.error("trigger failed", e);
        alert("Trigger failed: " + e.message);
      } finally {
        this.triggering = false;
      }
    },

    async poll() {
      try {
        const r = await fetch("/api/v1/approval/pending");
        if (r.ok) this.pending = await r.json();
      } catch (_) {}

      if (!this.traceId) return;
      try {
        const res = await fetch(`/api/v1/traces/${this.traceId}`);
        if (res.ok) this.trace = await res.json();
      } catch (_) {}
    },

    async decide(requestId, approved) {
      try {
        const url = `/api/v1/approval/callback?request_id=${encodeURIComponent(requestId)}` +
                    `&approved=${approved}&operator=dashboard-operator`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(await res.text());
        this.pending = this.pending.filter(c => c.request_id !== requestId);
        this.poll();
      } catch (e) {
        console.error("decision failed", e);
        alert("Decision failed: " + e.message);
      }
    },

    // ------- derived ------- //
    get activeModel() {
      return this.trace?.trace?.active_model || null;
    },
    get fallbackExecuted() {
      return !!this.trace?.trace?.fallback_executed;
    },
    get costUsd() {
      return this.trace?.trace?.total_cost_usd ?? 0;
    },
    get tokensIn() {
      return this.trace?.trace?.total_tokens_in ?? 0;
    },
    get tokensOut() {
      return this.trace?.trace?.total_tokens_out ?? 0;
    },

    // waterfall layout: compute depth + horizontal offset per span
    get rows() {
      if (!this.trace) return [];
      const spans = this.trace.spans;
      if (spans.length === 0) return [];

      const startTimes = spans.map(s => new Date(s.started_at).getTime());
      const endTimes = spans.map(s => {
        const e = s.ended_at ? new Date(s.ended_at).getTime() : Date.now();
        return e;
      });
      const t0 = Math.min(...startTimes);
      const t1 = Math.max(...endTimes);
      const total = Math.max(1, t1 - t0);

      const byId = Object.fromEntries(spans.map(s => [s.span_id, s]));
      const depthOf = (s) => {
        let d = 0, cur = s;
        const seen = new Set();
        while (cur && cur.parent_id && byId[cur.parent_id] && !seen.has(cur.parent_id)) {
          seen.add(cur.parent_id);
          d += 1;
          cur = byId[cur.parent_id];
        }
        return d;
      };

      return spans.map(s => {
        const st = new Date(s.started_at).getTime();
        const en = s.ended_at ? new Date(s.ended_at).getTime() : Date.now();
        const offsetPct = ((st - t0) / total) * 100;
        const widthPct = Math.max(0.8, ((en - st) / total) * 100);
        return { span: s, depth: depthOf(s), offsetPct, widthPct };
      });
    },

    statusBg(status) {
      switch (status) {
        case "SUCCESS": return "bg-emerald-500";
        case "FALLBACK": return "bg-amber-500";
        case "BLOCKED_BY_GOVERNANCE": return "bg-red-500";
        case "ERROR": return "bg-red-600";
        case "RUNNING": return "bg-sky-500";
        default: return "bg-slate-500";
      }
    },
  };
}
</script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the single-file Tailwind dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)
