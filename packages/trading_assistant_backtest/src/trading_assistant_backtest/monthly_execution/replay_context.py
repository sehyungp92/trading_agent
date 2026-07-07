"""Native monthly runner CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trading_assistant_backtest.contract_models import (
    DataBundleManifest,
    MonthlyRunManifest,
)
from trading_assistant_backtest.monthly_execution.structural_registry import (
    CRYPTO_PLUGIN_IDS,
    K_STOCK_PLUGIN_ID,
    TRADING_MOMENTUM_PLUGIN_ID,
    TRADING_STOCK_PLUGIN_ID,
    TRADING_SWING_PLUGIN_ID,
)
from trading_assistant_backtest.replay.types import ReplayResult
from trading_assistant_backtest.replay.windows import (
    resolve_selection_oos_window,
)


@dataclass
class ReplayEvaluationContext:
    plugin: Any | None = None
    baseline: Any | None = None
    incumbent: ReplayResult | None = None
    selection_oos_incumbent: ReplayResult | None = None
    diagnostics: dict[str, Any] | None = None
    baseline_score: float = 0.0
    replay_engine_version: str = ""
    replay_backed: bool = False
    reason: str = ""


def build_replay_context(
    manifest: MonthlyRunManifest,
    bundle: DataBundleManifest | None,
    data_errors: list[str],
) -> ReplayEvaluationContext:
    if data_errors:
        return ReplayEvaluationContext(reason="data or plugin validation failed")
    if bundle is None:
        return ReplayEvaluationContext(reason="data bundle is unavailable")
    plugin = _replay_plugin_for_manifest(manifest)
    if plugin is None:
        return ReplayEvaluationContext(reason="strategy plugin has no replay-backed evaluator")
    selection_oos = resolve_selection_oos_window(manifest)
    try:
        baseline = plugin.load_baseline(manifest, bundle)
        incumbent = plugin.run_incumbent(selection_oos, baseline)
        diagnostics = plugin.run_diagnostics(incumbent)
    except Exception as exc:
        return ReplayEvaluationContext(reason=f"replay-backed evaluator failed: {exc}")
    return ReplayEvaluationContext(
        plugin=plugin,
        baseline=baseline,
        incumbent=incumbent,
        diagnostics=diagnostics,
        baseline_score=incumbent.objective_score,
        replay_engine_version=str(
            getattr(plugin, "replay_engine_version", "")
            or diagnostics.get("replay_engine_version", "")
        ),
        replay_backed=True,
        reason="replay-backed evaluator completed",
    )


def _replay_plugin_for_manifest(manifest: MonthlyRunManifest) -> Any | None:
    if manifest.strategy_plugin_id not in CRYPTO_PLUGIN_IDS:
        if manifest.strategy_plugin_id == K_STOCK_PLUGIN_ID:
            from trading_assistant_backtest.strategies.krx.replay_evaluator import (
                KStockReplayPlugin,
            )

            return KStockReplayPlugin(
                plugin_id=manifest.strategy_plugin_id,
                strategy_id=manifest.strategy_id,
            )
        if manifest.strategy_plugin_id == TRADING_STOCK_PLUGIN_ID:
            from trading_assistant_backtest.strategies.trading.equity_replay_evaluator import (
                TradingStockReplayPlugin,
            )

            return TradingStockReplayPlugin(
                plugin_id=manifest.strategy_plugin_id,
                strategy_id=manifest.strategy_id,
            )
        if manifest.strategy_plugin_id == TRADING_MOMENTUM_PLUGIN_ID:
            from trading_assistant_backtest.strategies.trading.momentum_replay_evaluator import (
                TradingMomentumReplayPlugin,
            )

            return TradingMomentumReplayPlugin(
                plugin_id=manifest.strategy_plugin_id,
                strategy_id=manifest.strategy_id,
            )
        if manifest.strategy_plugin_id == TRADING_SWING_PLUGIN_ID:
            from trading_assistant_backtest.strategies.trading.equity_replay_evaluator import (
                TradingSwingReplayPlugin,
            )

            return TradingSwingReplayPlugin(
                plugin_id=manifest.strategy_plugin_id,
                strategy_id=manifest.strategy_id,
            )
        return None
    from trading_assistant_backtest.strategies.crypto.replay_evaluator import CryptoReplayPlugin

    return CryptoReplayPlugin(
        plugin_id=manifest.strategy_plugin_id,
        strategy_id=manifest.strategy_id,
    )
