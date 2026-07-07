"""Approval-readiness helpers shared by replay plugins."""

from __future__ import annotations

import json
from pathlib import Path

from trading_assistant_backtest.contract_models import MonthlyRunManifest
from trading_assistant_backtest.strategies.contracts import load_strategy_plugin_contract
from trading_assistant_backtest.strategies.deployment import deployment_metadata_errors
from trading_assistant_backtest.validation.deployment_metadata_contract import (
    live_deployment_metadata_errors,
)


def adoption_enabled_for_manifest(manifest: MonthlyRunManifest) -> bool:
    """Return true only when this bridge is approval-ready for optimizer adoption."""

    if not manifest.strategy_plugin_contract_path or not manifest.deployment_metadata_path:
        return False
    contract, errors = load_strategy_plugin_contract(manifest.strategy_plugin_contract_path)
    if errors or contract is None or not contract.eligible_for_approval:
        return False
    if deployment_metadata_errors(manifest, contract):
        return False
    try:
        metadata = json.loads(Path(manifest.deployment_metadata_path).read_text(encoding="utf-8"))
    except Exception:
        return False
    return not live_deployment_metadata_errors(metadata if isinstance(metadata, dict) else {})
