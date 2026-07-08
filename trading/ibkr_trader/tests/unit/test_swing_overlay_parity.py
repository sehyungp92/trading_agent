from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from backtests.swing.config import SlippageConfig
from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.data.preprocessing import NumpyBars
from backtests.swing.engine.unified_portfolio_engine import _overlay_ema, _overlay_transaction_cost, _rebalance_overlay
from strategies.swing.overlay.config import OverlayConfig
from strategies.swing.overlay.engine import _compute_ema
from strategies.swing.overlay.shared import allocate_weighted_targets


def _daily_bars(closes: list[float], *, open_offset: float = 0.5) -> tuple[NumpyBars, list[date]]:
    dates = [date(2026, 4, 1) + timedelta(days=offset) for offset in range(len(closes))]
    opens = np.array([close + open_offset for close in closes], dtype=float)
    closes_arr = np.array(closes, dtype=float)
    return (
        NumpyBars(
            opens=opens,
            highs=np.maximum(opens, closes_arr) + 1.0,
            lows=np.minimum(opens, closes_arr) - 1.0,
            closes=closes_arr,
            volumes=np.full(len(closes_arr), 1_000_000.0),
            times=np.array([np.datetime64(day.isoformat()) for day in dates], dtype="datetime64[ns]"),
        ),
        dates,
    )


def _overlay_targets_for_date(
    config: UnifiedBacktestConfig,
    daily: dict[str, NumpyBars],
    current_date: date,
    portfolio_equity: float,
) -> dict[str, float]:
    signals: dict[str, bool] = {}
    prices: dict[str, float] = {}

    for sym in config.overlay_symbols:
        bars = daily.get(sym)
        if bars is None:
            signals[sym] = False
            continue
        dates = [date.fromisoformat(str(ts)[:10]) for ts in bars.times]
        date_index = {day: idx for idx, day in enumerate(dates)}
        d_idx = date_index.get(current_date)
        fast, slow = config.overlay_ema_overrides.get(sym, (config.overlay_ema_fast, config.overlay_ema_slow))
        if d_idx is None or d_idx < slow:
            signals[sym] = False
            continue
        ema_fast = _compute_ema(bars.closes, fast)
        ema_slow = _compute_ema(bars.closes, slow)
        signals[sym] = bool(ema_fast[d_idx] > ema_slow[d_idx])
        prices[sym] = float(bars.opens[d_idx + 1] if d_idx + 1 < len(bars.opens) else bars.closes[d_idx])

    targets = allocate_weighted_targets(
        config.overlay_symbols,
        signals=signals,
        prices=prices,
        portfolio_equity=portfolio_equity,
        max_equity_pct=config.overlay_max_pct,
        weights=config.overlay_weights,
    )
    return {sym: float(targets.get(sym, 0)) for sym in config.overlay_symbols}


def test_overlay_config_defaults_match_optimized_live_posture() -> None:
    live = OverlayConfig()
    backtest = UnifiedBacktestConfig()

    assert live.enabled is False
    assert live.symbols == backtest.overlay_symbols
    assert live.max_equity_pct == backtest.overlay_max_pct
    assert live.ema_fast == backtest.overlay_ema_fast
    assert live.ema_slow == backtest.overlay_ema_slow
    assert live.ema_overrides == backtest.overlay_ema_overrides
    assert live.weights == backtest.overlay_weights


def test_overlay_ema_matches_backtest_reference() -> None:
    closes = np.array([100.0, 101.0, 100.5, 102.5, 103.0, 104.0, 103.5, 105.0])

    live = _compute_ema(closes, 3)
    backtest = _overlay_ema(closes, 3)

    assert np.allclose(live, backtest, equal_nan=True)


def test_overlay_ema_crossover_signal_semantics() -> None:
    """Fast EMA > slow EMA = bullish (buy); both live and backtest agree."""
    # Uptrend series: fast EMA rises above slow EMA
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0])
    fast_period = 3
    slow_period = 8

    live_fast = _compute_ema(closes, fast_period)
    live_slow = _compute_ema(closes, slow_period)
    bt_fast = _overlay_ema(closes, fast_period)
    bt_slow = _overlay_ema(closes, slow_period)

    # At the last bar, fast > slow in uptrend (bullish)
    assert live_fast[-1] > live_slow[-1]
    assert bt_fast[-1] > bt_slow[-1]
    # Live and backtest agree on signal direction
    live_bullish = live_fast[-1] > live_slow[-1]
    bt_bullish = bt_fast[-1] > bt_slow[-1]
    assert live_bullish == bt_bullish


def test_overlay_per_symbol_ema_overrides_consistent() -> None:
    """Per-symbol EMA overrides produce same periods in live and backtest configs."""
    live = OverlayConfig()
    backtest = UnifiedBacktestConfig()

    for sym in live.symbols:
        live_override = live.ema_overrides.get(sym)
        bt_override = backtest.overlay_ema_overrides.get(sym)
        assert live_override == bt_override, f"EMA override mismatch for {sym}"

        # Verify override periods are applied correctly
        if live_override:
            fast, slow = live_override
            assert fast < slow, f"Fast EMA must be < slow EMA for {sym}"


def test_overlay_rebalance_targets_match_live_weighted_allocation_semantics() -> None:
    qqq, qqq_dates = _daily_bars([100, 101, 102, 103, 104, 105, 106, 107], open_offset=1.0)
    gld, _ = _daily_bars([50, 50.5, 51, 51.5, 52, 52.5, 53, 53.5], open_offset=0.5)
    current_date = qqq_dates[-2]
    config = UnifiedBacktestConfig(
        overlay_mode="ema",
        overlay_symbols=["QQQ", "GLD"],
        overlay_ema_fast=3,
        overlay_ema_slow=5,
        overlay_ema_overrides={"QQQ": (2, 4), "GLD": (3, 5)},
        overlay_max_pct=0.80,
        overlay_weights={"QQQ": 0.75, "GLD": 0.25},
    )
    daily = {"QQQ": qqq, "GLD": gld}
    daily_date_idx = {
        "QQQ": {day: idx for idx, day in enumerate(qqq_dates)},
        "GLD": {day: idx for idx, day in enumerate(qqq_dates)},
    }
    overlay_emas = {
        "QQQ": (_overlay_ema(qqq.closes, 2), _overlay_ema(qqq.closes, 4)),
        "GLD": (_overlay_ema(gld.closes, 3), _overlay_ema(gld.closes, 5)),
    }
    overlay_shares = {"QQQ": 0.0, "GLD": 0.0}

    _rebalance_overlay(
        config,
        daily,
        overlay_emas,
        daily_date_idx,
        current_date,
        10_000.0,
        overlay_shares,
    )

    assert overlay_shares == _overlay_targets_for_date(config, daily, current_date, 10_000.0)


def test_overlay_rebalance_matches_live_last_day_fallback_and_bearish_flattening() -> None:
    qqq, qqq_dates = _daily_bars([100, 101, 102, 101, 100, 99, 98], open_offset=0.25)
    gld, _ = _daily_bars([50, 52, 55, 59, 64, 70, 77], open_offset=0.75)
    current_date = qqq_dates[-1]
    config = UnifiedBacktestConfig(
        overlay_mode="ema",
        overlay_symbols=["QQQ", "GLD"],
        overlay_ema_fast=2,
        overlay_ema_slow=3,
        overlay_ema_overrides={},
        overlay_max_pct=0.60,
    )
    daily = {"QQQ": qqq, "GLD": gld}
    daily_date_idx = {
        "QQQ": {day: idx for idx, day in enumerate(qqq_dates)},
        "GLD": {day: idx for idx, day in enumerate(qqq_dates)},
    }
    overlay_emas = {
        "QQQ": (_overlay_ema(qqq.closes, 2), _overlay_ema(qqq.closes, 3)),
        "GLD": (_overlay_ema(gld.closes, 2), _overlay_ema(gld.closes, 3)),
    }
    overlay_shares = {"QQQ": 25.0, "GLD": 10.0}

    _rebalance_overlay(
        config,
        daily,
        overlay_emas,
        daily_date_idx,
        current_date,
        12_000.0,
        overlay_shares,
    )

    assert overlay_shares == _overlay_targets_for_date(config, daily, current_date, 12_000.0)
    assert overlay_shares["QQQ"] == 0.0
    assert overlay_shares["GLD"] > 0.0




def test_overlay_transaction_cost_includes_min_commission_and_slippage() -> None:
    slippage = SlippageConfig(
        commission_per_share_etf=0.0035,
        commission_min_etf_order=0.35,
        overlay_slip_bps=5.0,
    )

    assert _overlay_transaction_cost(10, 100.0, slippage) == pytest.approx(0.85)
    assert _overlay_transaction_cost(200, 100.0, slippage) == pytest.approx(10.70)


def test_overlay_shared_allocator_handles_missing_prices_and_bearish_symbols() -> None:
    targets = allocate_weighted_targets(
        ["QQQ", "GLD", "TLT"],
        signals={"QQQ": True, "GLD": False, "TLT": True},
        prices={"QQQ": 100.0, "GLD": 50.0},
        portfolio_equity=10_000.0,
        max_equity_pct=0.50,
        weights={"QQQ": 0.75, "GLD": 0.10, "TLT": 0.25},
    )

    assert targets["QQQ"] == 37
    assert targets["GLD"] == 0
    assert targets["TLT"] == 0


def test_overlay_is_live_only_exception() -> None:
    """Overlay has no core/logic.py -- it is explicitly a live-only engine."""
    from pathlib import Path
    core_logic = Path(__file__).resolve().parents[2] / "strategies" / "swing" / "overlay" / "core" / "logic.py"
    assert not core_logic.exists(), "Overlay should NOT have core/logic.py (live-only exception)"
