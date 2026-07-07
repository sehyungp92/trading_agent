"""Replay-backed evaluators for trading stock and swing US-equity shadow slices."""

from __future__ import annotations

from trading_assistant_backtest.strategies.bar_replay import (
    BarReplayConfig,
    BarReplayPluginBase,
)

STOCK_REPLAY_ENGINE_VERSION = "trading_stock_us_equity_bar_replay_v1"
SWING_REPLAY_ENGINE_VERSION = "trading_swing_us_equity_bar_replay_v1"


class TradingStockReplayPlugin(BarReplayPluginBase):
    CONFIG = BarReplayConfig(
        family="trading_stock_family",
        market="us_equity",
        source="ibkr",
        replay_engine_version=STOCK_REPLAY_ENGINE_VERSION,
        diagnostics_schema_version="trading_stock_replay_diagnostics_v1",
        result_schema_version="trading_stock_bar_replay_result_v1",
        supported_symbols=("MSFT",),
        supported_timeframes=("5m",),
        threshold_bps=4.0,
        position_weight=1.0,
        max_positions=1,
        quantity=100.0,
        adoption_enabled=False,
    )


class TradingSwingReplayPlugin(BarReplayPluginBase):
    CONFIG = BarReplayConfig(
        family="trading_swing_family",
        market="us_equity",
        source="ibkr",
        replay_engine_version=SWING_REPLAY_ENGINE_VERSION,
        diagnostics_schema_version="trading_swing_replay_diagnostics_v1",
        result_schema_version="trading_swing_bar_replay_result_v1",
        supported_symbols=("QQQ",),
        supported_timeframes=("1h",),
        threshold_bps=6.0,
        position_weight=1.0,
        max_positions=1,
        quantity=100.0,
        adoption_enabled=False,
    )
