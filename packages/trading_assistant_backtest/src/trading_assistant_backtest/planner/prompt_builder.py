"""Planner prompt builder seam."""

from __future__ import annotations

from trading_assistant_backtest.contract_models import MonthlyRunManifest


def build_prompt(manifest: MonthlyRunManifest) -> str:
    return (
        f"Build a monthly optimizer plan for {manifest.bot_id}/{manifest.strategy_id} "
        f"for run {manifest.run_id}. Use only evidence listed in the manifest."
    )
