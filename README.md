# TrueGuard-MCP

**Agentic DevOps & Incident Escalation Harness** — a production-grade
reference implementation of an autonomous SRE agent that investigates
production alerts, diagnoses root causes, proposes remediation plans, and
routes destructive actions through a human-in-the-loop approval gateway.

Built on Python, FastAPI, FastMCP, and the OpenAI SDK through
**TrueFoundry's AI, Agent, and MCP Gateways**.

---

## Features

| Capability | Where |
|---|---|
| **Model failover routing** — automatic failover from `gpt-4o` to `gpt-4o-mini` on 429 / 504 / connection errors, with `X-Fallback-Executed` telemetry | `gateways/llm_router.py` |
| **Hard job budget enforcement** — soft downgrade at `$0.50`, hard halt at `$0.75` per job execution | `gateways/llm_router.py` |
| **MCP permission governance** — `@require_permission(level="READ_ONLY" \| "HIGH_RISK")` decorator that pauses HIGH_RISK tool calls pending human approval | `gateways/mcp_governance.py` |
| **Human-in-the-loop approval workflow** — Slack Block-Kit cards with `[Approve & Execute]` / `[Deny]` buttons, idempotent callbacks | `gateways/approval_workflow.py` |
| **Four FastMCP tools** — log fetch, health check, container restart, DB migration | `mcp_servers/system_mcp.py` |
| **OpenTelemetry-style tracing** — span tree with `span_id`, `parent_id`, `latency_ms`, `model_used`, `tokens_in/out`, `cost_usd`, `status` | `observability/tracer.py` |
| **Live dashboard** — real-time waterfall, cost meter, active-model indicator, HITL modal (Tailwind + Alpine.js, single file) | `app.py` |
| **Inline eval suite** — hallucination, SQL/shell injection, PII masking checks on every run | `evals/eval_harness.py` |
| **Typed exceptions** — every domain failure carries an `ErrorContext` with `trace_id` / `span_id` / operation | `exceptions.py` |
| **Structured logging** — context-manager-based tracing across every phase | `logging_utils.py` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  User Alert  →  POST /api/v1/incident/trigger                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  DevOpsAgent.handle_alert  (agents/devops_agent.py)             │
│                                                                 │
│   1. OBSERVE   ──►  fetch_server_logs, check_container_health   │
│                     (READ_ONLY, autonomous)                     │
│                                                                 │
│   2. DIAGNOSE  ──►  TrueFoundryRouter.complete()                │
│   3. PLAN           gpt-4o via AI Gateway                       │
│                     (failover → gpt-4o-mini on 429/504)         │
│                                                                 │
│   4. GOVERN    ──►  HIGH_RISK tools:                            │
│                     request_approval() ──► Slack card           │
│                     ──► await human decision                    │
│                     ──► execute via registered tool executor    │
│                                                                 │
│   5. REPORT    ──►  IncidentReportResponse (strict Pydantic)    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tracer  →  waterfall spans  →  Live Dashboard  (/)             │
│  Evals   →  scorecard        →  presentation summary            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# edit .env — set TRUEFOUNDRY_API_KEY, APPROVAL_CALLBACK_URL, optionally SLACK_WEBHOOK_URL
```

### 3. Run the harness

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/` — the live dashboard.

### 4. Trigger a demo incident

Click **▶ Trigger** in the dashboard, or:

```bash
curl -X POST http://localhost:8000/api/v1/incident/trigger \
  -H 'Content-Type: application/json' \
  -d '{
    "alert_text": "Alert: High 500 error rate on payment-service-v2",
    "affected_service": "payment-service-v2"
  }'
```

---

## Demo Runner (Zero-Friction)

The demo runner executes three scripted scenarios end-to-end against
in-memory stubs — no OpenAI key, no Slack webhook, no TrueFoundry credentials
required. The real router, governance, approval store, and eval harness
execute the same code paths as production.

```bash
python demo_runner.py                  # all three scenarios
python demo_runner.py --scenario B     # just the HITL one
python demo_runner.py --no-color       # CI-friendly
```

### Scenarios

| Scenario | What it demonstrates |
|---|---|
| **A — Autonomous remediation** | READ_ONLY-only path; no HIGH_RISK tools; eval scorecard `3/3` |
| **B — High-risk migration (HITL)** | Governance pauses run → Slack card emitted → operator approves → tool executes with `approved_by` |
| **C — Fallback + budget halt** | Primary `gpt-4o` returns 504 → failover to `gpt-4o-mini` → `$0.50` downgrade → `$0.75` halt → `BudgetExceededException` |

---

## Eval Harness

Every agent run produces a scorecard with three deterministic checks:

| Check | Purpose |
|---|---|
| **Hallucination** | Every tool name referenced by the LLM exists in the MCP schema |
| **Injection guard** | Tool arguments contain no destructive SQL / shell syntax |
| **PII masking** | No API keys, bearer tokens, or emails in outgoing LLM prompts |

Self-test the harness in isolation:

```bash
python -m evals.eval_harness
```

---

## Project Layout

```
TrueGuard-MCP/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── main.py                    # composition root (logging + routers + handlers)
├── app.py                     # dashboard + incident routes
├── config.py                  # Pydantic BaseSettings
├── exceptions.py              # typed exception hierarchy
├── logging_utils.py           # structured logging + trace_context
├── types.py                   # shared TypedDicts
├── demo_runner.py             # presentation CLI
│
├── agents/
│   └── devops_agent.py        # 4-step incident-response loop
│
├── gateways/
│   ├── llm_router.py          # failover + budget enforcement
│   ├── mcp_governance.py      # @require_permission interceptor
│   └── approval_workflow.py   # Slack cards + approval store
│
├── mcp_servers/
│   └── system_mcp.py          # 4 FastMCP tools
│
├── observability/
│   └── tracer.py              # OTel-style span recorder
│
├── evals/
│   ├── patterns.py            # pre-compiled regex
│   └── eval_harness.py        # 3 deterministic checks
│
├── orchestrator/
│   ├── context.py
│   └── steps.py               # shared observe/diagnose/plan
│
├── fixtures/
│   └── stubs.py               # in-memory stubs for demo
│
├── scenarios/
│   ├── _common.py             # Palette + print helpers
│   ├── scenario_a.py
│   ├── scenario_b.py
│   └── scenario_c.py
│
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_exceptions.py
    ├── test_eval_harness.py
    └── test_scenarios.py
```

---

## API Reference

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/incident/trigger` | Start an agent run for an alert |
| `GET`  | `/api/v1/traces` | List recent traces (summaries) |
| `GET`  | `/api/v1/traces/{trace_id}` | Full waterfall timeline for one trace |
| `GET`  | `/api/v1/incidents/{id}/report` | Final `IncidentReportResponse` |
| `GET`  | `/api/v1/approval/pending` | Pending HITL approval cards |
| `GET`  | `/api/v1/approval/{request_id}` | One approval card by ID |
| `GET`  | `/api/v1/approval/callback` | Slack button receiver (idempotent) |
| `GET`  | `/` | Live dashboard |

---

## Configuration

All settings are loaded via Pydantic `BaseSettings` from `.env`. See
`.env.example` for the full list of variables and their semantics.

### Budget thresholds

* `JOB_BUDGET_USD` (default `0.50`) — soft downgrade threshold. When
  accumulated cost crosses this, the router forces all subsequent calls onto
  the cheap fallback model.
* Hard halt threshold = `JOB_BUDGET_USD * 1.5` (default `0.75`). When
  crossed, `BudgetExceededException` is raised on the next pre-flight check.

### Model routing

* `PRIMARY_MODEL` (default `gpt-4o`) — must differ from `FALLBACK_MODEL`.
* `FALLBACK_MODEL` (default `gpt-4o-mini`) — cheap resilient target.

The router also tries `o3-mini` if both fail.

---

## Testing

```bash
pytest tests/ -q
python -m evals.eval_harness
python demo_runner.py --no-color
```

Type-check:

```bash
mypy --strict .
ruff check .
```

---

## Human-in-the-Loop Flow

```
Agent proposes HIGH_RISK tool
    │
    ▼
gateways/mcp_governance.require_permission(HIGH_RISK)
    │
    ├── build ApprovalRequest
    ├── register with approval_registry (asyncio.Event)
    ├── POST Block-Kit card to SLACK_WEBHOOK_URL
    └── await event.wait()
             │
             ▼
GET /api/v1/approval/callback?request_id=…&approved=true&operator=alice@co
    │
    ├── approval_store.record(decision)   ← wakes the awaiting coroutine
    └── BackgroundTasks: _execute_tool_after_approval(card, "alice@co")
             │
             ▼
    get_tool("restart_service_container")(arguments, "alice@co")
             │
             ▼
    mcp_servers/system_mcp.restart_service_container(…, approved_by="alice@co")
```

The `Approval & Execute` button in the dashboard hits the exact same endpoint
as Slack — the workflow works identically against a webhook-only Slack app,
a custom UI, or a CLI loop.

---

## License

see `LICENSE`.
