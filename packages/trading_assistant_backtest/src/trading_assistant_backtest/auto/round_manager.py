"""Round lineage helpers."""

from __future__ import annotations

from trading_assistant_backtest.contract_models import MonthlyRunManifest


def current_round_id(manifest: MonthlyRunManifest) -> str:
    return manifest.round_id or f"{manifest.run_month}-round-0"


def next_round_id(manifest: MonthlyRunManifest) -> str:
    return manifest.next_round_id or f"{manifest.run_month}-round-1"
