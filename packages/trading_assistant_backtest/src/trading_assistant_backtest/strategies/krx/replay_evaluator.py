"""Replay-backed evaluator for the KRX/KIS OLR/KALCB shadow slice."""

from __future__ import annotations

from trading_assistant_backtest.strategies.bar_replay import (
    BarReplayConfig,
    BarReplayPluginBase,
)

K_STOCK_REPLAY_ENGINE_VERSION = "k_stock_krx_bar_replay_v1"


class KStockReplayPlugin(BarReplayPluginBase):
    CONFIG = BarReplayConfig(
        family="k_stock_olr_kalcb",
        market="krx_equity",
        source="kis",
        replay_engine_version=K_STOCK_REPLAY_ENGINE_VERSION,
        diagnostics_schema_version="k_stock_replay_diagnostics_v1",
        result_schema_version="k_stock_bar_replay_result_v1",
        supported_symbols=("005930",),
        supported_timeframes=("5m",),
        threshold_bps=5.0,
        position_weight=1.0,
        max_positions=1,
        quantity=10.0,
        adoption_enabled=False,
    )
