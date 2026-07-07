"""Manifest loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trading_assistant_backtest.contract_models import MonthlyRunManifest
from trading_assistant_backtest.paths import monorepo_root, normalize_workspace_path

_PATH_FIELDS = (
    "market_data_manifest_path",
    "telemetry_manifest_path",
    "backtest_repo_path",
    "deployment_metadata_path",
    "artifact_root",
    "strategy_plugin_contract_path",
    "round_n_strategy_config_path",
    "round_n_portfolio_config_path",
    "data_bundle_manifest_path",
    "fold_manifest_path",
    "rounds_manifest_path",
    "end_of_round_diagnostics_path",
    "candidate_workspace_root",
    "candidate_workspace_manifest_path",
    "checkpoint_path",
    "cache_path",
    "outcome_prior_snapshot_path",
    "workflow_contract_path",
)
_PATH_MAP_FIELDS = (
    "deployment_metadata_paths",
    "bridge_deployment_metadata_paths",
    "strategy_plugin_contract_paths",
    "bridge_contract_paths",
)


def load_manifest(path: str | Path) -> MonthlyRunManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = _normalize_manifest_paths(payload)
    return MonthlyRunManifest.model_validate(payload)


def _normalize_manifest_paths(payload: dict[str, Any]) -> dict[str, Any]:
    root = monorepo_root()
    normalized = dict(payload)
    for key in _PATH_FIELDS:
        value = normalized.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = str(normalize_workspace_path(root, value))
    for key in _PATH_MAP_FIELDS:
        value = normalized.get(key)
        if not isinstance(value, dict):
            continue
        normalized[key] = {
            str(item_key): str(normalize_workspace_path(root, item_value))
            for item_key, item_value in value.items()
            if str(item_key).strip() and str(item_value).strip()
        }
    return normalized
