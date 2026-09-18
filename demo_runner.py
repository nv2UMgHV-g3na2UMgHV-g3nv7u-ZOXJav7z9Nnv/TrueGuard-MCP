"""
demo_runner.py
==============
Zero-friction presentation runner.

Usage::

    python demo_runner.py
    python demo_runner.py --scenario B
    python demo_runner.py --no-color
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

# --- Bootstrap env before importing anything that reads settings --- #
os.environ.setdefault("TRUEFOUNDRY_GATEWAY_URL", "https://gateway.truefoundry.ai")
os.environ.setdefault("TRUEFOUNDRY_API_KEY", "tfy-demo-key-not-used-in-stub-mode")
os.environ.setdefault("TRUEFOUNDRY_MCP_GATEWAY_URL", "https://mcp-gateway.truefoundry.ai")
os.environ.setdefault("PRIMARY_MODEL", "gpt-4o")
os.environ.setdefault("FALLBACK_MODEL", "gpt-4o-mini")
os.environ.setdefault("JOB_BUDGET_USD", "0.50")
os.environ.setdefault("APPROVAL_CALLBACK_URL", "http://localhost:8000/api/v1/approval/callback")

from config import settings  # noqa: E402
from logging_utils import configure_logging  # noqa: E402
from scenarios import scenario_a, scenario_b, scenario_c  # noqa: E402
from scenarios._common import Palette, banner  # noqa: E402
from types import ScenarioResult  # noqa: E402


_SCENARIOS: dict[str, object] = {
    "A": scenario_a,
    "B": scenario_b,
    "C": scenario_c,
}


def _print_summary(pal: Palette, results: dict[str, ScenarioResult]) -> int:
    banner(pal, "DEMO SUMMARY")
    all_passed = True
    for name, result in results.items():
        sc = result["scorecard"]
        passed = sc["passed"]
        all_passed = all_passed and passed
        marker = pal.green("✓ PASS") if passed else pal.red("✗ FAIL")
        detail = sc["checks"][0]["detail"][:60] if sc["checks"] else ""
        print(f"  {marker}  {name:<12} {sc['score']}  ({detail}…)")
    print()
    if all_passed:
        print(pal.bold(pal.green("  ALL SCENARIOS PASSED — demo ready.")))
        return 0
    print(pal.bold(pal.red("  DEMO FAILURES PRESENT — do not present.")))
    return 1


async def _main() -> int:
    parser = argparse.ArgumentParser(description="TrueGuard-MCP demo runner")
    parser.add_argument("--scenario", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    # Wire logging first.
    configure_logging(level="WARNING", structured=False)  # quiet during demo

    pal = Palette((not args.no_color) and sys.stdout.isatty())

    print()
    print(pal.bold(pal.magenta("  TrueGuard-MCP :: Live Demo Runner")))
    print(pal.dim(f"  {datetime.now(tz=timezone.utc).isoformat()}"))
    print(pal.dim(
        f"  primary={settings.primary_model}  fallback={settings.fallback_model}  "
        f"budget=${settings.job_budget_usd:.2f}"
    ))

    results: dict[str, ScenarioResult] = {}
    selected = _SCENARIOS.keys() if args.scenario == "all" else [args.scenario]
    for key in selected:
        fn = _SCENARIOS[key]
        results[f"Scenario {key}"] = await fn(pal)  # type: ignore[operator]

    return _print_summary(pal, results)


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
