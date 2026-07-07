"""Strategy plugin contract checks owned by the runner boundary."""

from __future__ import annotations

import json
from pathlib import Path

from trading_contracts.models import (
    DataBundleManifest,
    MonthlyRunManifest,
    StrategyPluginContract,
    StrategyPluginMaturity,
)
from trading_assistant_backtest.paths import monorepo_root
from trading_assistant_backtest.strategies.deployment import deployment_metadata_errors
from trading_assistant_backtest.strategies.live_clone import validate_clean_checkout

MATURE_STATES = {
    StrategyPluginMaturity.SHADOW_VALIDATED,
    StrategyPluginMaturity.APPROVAL_READY,
}
FAMILY_PLUGIN_IDS = {
    "k-stock-olr-kalcb",
    "trading-stock-family",
    "trading-momentum-family",
    "trading-swing-family",
}


def strategy_plugin_errors(
    manifest: MonthlyRunManifest,
    bundle: DataBundleManifest | None,
) -> list[str]:
    path_text = manifest.strategy_plugin_contract_path.strip()
    if not path_text:
        return deployment_metadata_errors(manifest)
    path = Path(path_text)
    if not path.exists():
        return [f"strategy plugin contract path is missing: {path}"]
    contract, load_errors = load_strategy_plugin_contract(path)
    if load_errors:
        return load_errors
    assert contract is not None

    errors = deployment_metadata_errors(manifest, contract)
    if manifest.strategy_plugin_id and contract.plugin_id != manifest.strategy_plugin_id:
        errors.append("strategy plugin contract plugin_id does not match run manifest")
    if manifest.trading_repo_commit_sha and contract.live_repo_commit_sha:
        if contract.live_repo_commit_sha != manifest.trading_repo_commit_sha:
            errors.append("strategy plugin contract live repo SHA does not match run manifest")
    if manifest.backtest_repo_commit_sha and contract.backtest_adapter_commit_sha:
        if contract.backtest_adapter_commit_sha != manifest.backtest_repo_commit_sha:
            errors.append(
                "strategy plugin contract backtest adapter SHA does not match run manifest"
            )

    if contract.maturity in MATURE_STATES:
        if not str(getattr(manifest, "deployment_metadata_path", "") or "").strip():
            errors.append(
                "deployment metadata path is required for mature strategy plugin contract"
            )
        if contract.maturity == StrategyPluginMaturity.APPROVAL_READY:
            errors.extend(_clean_checkout_errors(contract))
        if bundle is not None and not _family_or_portfolio_scope(manifest, contract):
            errors.extend(_support_errors(contract, bundle))
    return errors


def load_strategy_plugin_contract(
    path: str | Path,
) -> tuple[StrategyPluginContract | None, list[str]]:
    contract_path = Path(path)
    if not contract_path.exists():
        return None, [f"strategy plugin contract path is missing: {contract_path}"]
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"strategy plugin contract is malformed: {exc}"]
    if not isinstance(payload, dict):
        return None, ["strategy plugin contract must be a JSON object"]
    try:
        contract = StrategyPluginContract.model_validate(payload)
        _resolve_contract_paths(contract, contract_path.parent)
        return contract, []
    except Exception as exc:
        return None, [f"strategy plugin contract is invalid: {exc}"]


def _support_errors(
    contract: StrategyPluginContract, bundle: DataBundleManifest
) -> list[str]:
    supported_symbols = set(contract.supported_symbols)
    supported_timeframes = set(contract.supported_timeframes)
    errors: list[str] = []
    for item in bundle.slice_manifests:
        if supported_symbols and item.symbol.upper() not in supported_symbols:
            errors.append(f"strategy plugin does not support symbol: {item.symbol}")
        if supported_timeframes and item.timeframe not in supported_timeframes:
            errors.append(f"strategy plugin does not support timeframe: {item.timeframe}")
    return errors


def _family_or_portfolio_scope(
    manifest: MonthlyRunManifest,
    contract: StrategyPluginContract,
) -> bool:
    if contract.plugin_id in FAMILY_PLUGIN_IDS:
        return True
    bridge_maps = (
        getattr(manifest, "bridge_contract_paths", {}),
        getattr(manifest, "strategy_plugin_contract_paths", {}),
    )
    return any(isinstance(value, dict) and len(value) > 1 for value in bridge_maps)


def _clean_checkout_errors(contract: StrategyPluginContract) -> list[str]:
    if not contract.live_repo_path or not contract.live_repo_commit_sha:
        return []
    try:
        return validate_clean_checkout(
            Path(contract.live_repo_path),
            contract.live_repo_commit_sha,
        )
    except Exception as exc:
        return [f"live repo checkout validation failed: {exc}"]


def _resolve_contract_paths(contract: StrategyPluginContract, base_dir: Path) -> None:
    if contract.live_repo_path:
        live_repo_path = Path(contract.live_repo_path)
        if not live_repo_path.is_absolute():
            contract.live_repo_path = str(_resolve_contract_relative(base_dir, live_repo_path))
    resolved_fixtures: list[str] = []
    for fixture_path in contract.parity_fixture_set:
        path = Path(fixture_path)
        resolved_fixtures.append(
            str(path if path.is_absolute() else _resolve_contract_relative(base_dir, path))
        )
    contract.parity_fixture_set = resolved_fixtures


def _resolve_contract_relative(base_dir: Path, path: Path) -> Path:
    resolved = (base_dir / path).resolve()
    if resolved.exists():
        return resolved
    root_resolved = (monorepo_root() / path).resolve()
    return root_resolved if root_resolved.exists() else resolved
