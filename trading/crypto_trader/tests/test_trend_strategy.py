"""Tests for TrendStrategy orchestrator."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, patch

from crypto_trader.core.engine import StrategyContext, MultiTimeFrameBars
from crypto_trader.core.events import EventBus, PositionClosedEvent
from crypto_trader.core.models import (
    Bar, Fill, Order, OrderType, Position, SetupGrade, Side, TimeFrame, Trade,
)
from crypto_trader.strategy.trend.config import TrendConfig
from crypto_trader.strategy.trend.confirmation import TriggerResult
from crypto_trader.strategy.trend.setup import TrendSetupResult
from crypto_trader.strategy.trend.sizing import SizingResult
from crypto_trader.strategy.trend.strategy import TrendStrategy, WARMUP_BARS, _PendingTrendSetup


def _make_bar(symbol, tf, close, high=None, low=None, hour=10, day=15):
    return Bar(
        timestamp=datetime(2026, 3, day, hour, 0, tzinfo=timezone.utc),
        symbol=symbol, open=close - 10,
        high=high or close + 50, low=low or close - 50,
        close=close, volume=100.0, timeframe=tf,
    )


def _make_ctx():
    broker = MagicMock()
    broker.equity = 10000.0
    broker.get_position.return_value = None
    broker.open_orders = []
    events = EventBus()
    bars = MultiTimeFrameBars()
    clock = MagicMock()
    return StrategyContext(broker=broker, clock=clock, bars=bars, events=events)


def _mock_setup(direction=Side.LONG):
    return TrendSetupResult(
        grade=SetupGrade.B,
        direction=direction,
        impulse_start=49000,
        impulse_end=50500,
        impulse_atr_move=2.0,
        pullback_depth=0.3,
        confluences=("h1_ema_zone", "rsi_pullback"),
        zone_price=50000,
        room_r=2.5,
        stop_level=49500,
        setup_score=2.0,
    )


def _mock_sizing():
    return SizingResult(
        qty=0.1,
        leverage=5.0,
        liquidation_price=40000,
        risk_pct_actual=0.005,
        notional=5000,
        was_reduced=False,
        reduction_reason=None,
    )


class TestTrendStrategy:
    def test_properties(self):
        s = TrendStrategy()
        assert s.name == "trend_anchor"
        assert TimeFrame.H1 in s.timeframes
        assert TimeFrame.D1 in s.timeframes
        assert len(s.symbols) == 3

    def test_on_init_creates_state(self):
        s = TrendStrategy()
        ctx = _make_ctx()
        s.on_init(ctx)
        assert "BTC" in s._h1_bar_count
        assert "BTC" in s._h1_inc
        assert "BTC" in s._d1_inc

    def test_d1_bar_updates_regime(self):
        cfg = TrendConfig(symbols=["BTC"])
        s = TrendStrategy(cfg)
        ctx = _make_ctx()
        s.on_init(ctx)

        # Feed enough D1 bars for indicators to warm up
        for i in range(250):
            bar = Bar(
                timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
                symbol="BTC", open=50000 + i, high=50100 + i,
                low=49900 + i, close=50050 + i, volume=1000.0,
                timeframe=TimeFrame.D1,
            )
            s.on_bar(bar, ctx)

        # After warmup, d1_indicators should be populated
        assert s._d1_indicators["BTC"] is not None

    def test_h1_bar_increments_count(self):
        cfg = TrendConfig(symbols=["BTC"])
        s = TrendStrategy(cfg)
        ctx = _make_ctx()
        s.on_init(ctx)

        bar = _make_bar("BTC", TimeFrame.H1, 50000)
        s.on_bar(bar, ctx)
        assert s._h1_bar_count["BTC"] == 1

    def test_warmup_skips_entry(self):
        """No entries during warmup period."""
        cfg = TrendConfig(symbols=["BTC"])
        s = TrendStrategy(cfg)
        ctx = _make_ctx()
        s.on_init(ctx)

        for i in range(WARMUP_BARS - 1):
            bar = _make_bar("BTC", TimeFrame.H1, 50000 + i)
            s.on_bar(bar, ctx)

        # Broker should not have received any orders
        ctx.broker.submit_order.assert_not_called()

    def test_unknown_symbol_ignored(self):
        cfg = TrendConfig(symbols=["BTC"])
        s = TrendStrategy(cfg)
        ctx = _make_ctx()
        s.on_init(ctx)

        # Bar for symbol not in config
        bar = _make_bar("DOGE", TimeFrame.H1, 1000)
        s.on_bar(bar, ctx)  # Should not crash

    def test_on_fill_entry_creates_stop(self):
        cfg = TrendConfig(symbols=["BTC"])
        s = TrendStrategy(cfg)
        ctx = _make_ctx()
        s.on_init(ctx)

        # Pre-populate position meta
        from crypto_trader.strategy.trend.strategy import _PositionMeta
        s._position_meta["BTC"] = _PositionMeta(
            setup_grade=SetupGrade.B,
            entry_price=50000,
            stop_level=49500,
            stop_distance=500,
            original_qty=0.2,
        )

        fill = Fill(
            order_id="test_entry_1",
            symbol="BTC", side=Side.LONG,
            qty=0.1, fill_price=50000,
            commission=5.0,
            timestamp=datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
            tag="entry",
        )
        s.on_fill(fill, ctx)

        # Should have submitted a protective stop
        ctx.broker.submit_order.assert_called()
        stop_order = ctx.broker.submit_order.call_args[0][0]
        assert stop_order.tag == "protective_stop"
        assert stop_order.order_type == OrderType.STOP
        assert stop_order.qty == pytest.approx(fill.qty)
        assert stop_order.stop_price == 49500
        assert s._position_meta["BTC"].original_qty == pytest.approx(fill.qty)

    def test_position_closed_enriches_trade(self):
        cfg = TrendConfig(symbols=["BTC"])
        s = TrendStrategy(cfg)
        ctx = _make_ctx()
        s.on_init(ctx)

        # Set up meta
        from crypto_trader.strategy.trend.strategy import _PositionMeta
        s._position_meta["BTC"] = _PositionMeta(
            setup_grade=SetupGrade.B,
            confluences=("h1_ema_zone", "rsi_pullback"),
            confirmation_type="engulfing",
            entry_method="aggressive",
            entry_price=50000,
            stop_level=49500,
            stop_distance=500,
            original_qty=0.1,
        )

        # Initialize exit state
        s._exit_manager.init_position("BTC", 50000, 500, 0.1, Side.LONG)

        trade = Trade(
            trade_id="t1", symbol="BTC", direction=Side.LONG,
            entry_price=50000, exit_price=50800, qty=0.1,
            entry_time=datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
            exit_time=datetime(2026, 3, 15, 18, 0, tzinfo=timezone.utc),
            pnl=80, r_multiple=None, commission=10,
            bars_held=8, setup_grade=None, exit_reason="tp1",
            confluences_used=None, confirmation_type=None,
            entry_method=None, funding_paid=0, mae_r=None, mfe_r=None,
        )

        event = PositionClosedEvent(
            timestamp=datetime(2026, 3, 15, 18, 0, tzinfo=timezone.utc),
            trade=trade,
        )
        s._on_position_closed(event)

        # Trade should be enriched
        assert trade.setup_grade == SetupGrade.B
        assert trade.confluences_used == ["h1_ema_zone", "rsi_pullback"]
        assert trade.confirmation_type == "engulfing"
        assert trade.entry_method == "aggressive"

    def test_position_closed_records_net_loss_reentry_state_from_realized_r(self):
        cfg = TrendConfig(symbols=["BTC"])
        s = TrendStrategy(cfg)
        ctx = _make_ctx()
        s.on_init(ctx)

        from crypto_trader.strategy.trend.strategy import _PositionMeta
        s._position_meta["BTC"] = _PositionMeta(
            setup_grade=SetupGrade.B,
            confirmation_type="engulfing",
            entry_method="aggressive",
            entry_price=50000,
            stop_level=49500,
            stop_distance=500,
            original_qty=0.2,
        )
        exit_state = MagicMock()
        exit_state.mae_r = -0.3
        exit_state.mfe_r = 0.6
        s._exit_manager.remove_position = MagicMock(return_value=exit_state)
        s._risk_manager.record_trade = MagicMock()

        trade = Trade(
            trade_id="t_net_loss", symbol="BTC", direction=Side.LONG,
            entry_price=50000, exit_price=50100, qty=0.1,
            entry_time=datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
            exit_time=datetime(2026, 3, 15, 18, 0, tzinfo=timezone.utc),
            pnl=10.0, r_multiple=None, commission=15.0,
            bars_held=8, setup_grade=None, exit_reason="protective_stop",
            confluences_used=None, confirmation_type=None,
            entry_method=None, funding_paid=0, mae_r=None, mfe_r=None,
        )

        s._on_position_closed(PositionClosedEvent(timestamp=trade.exit_time, trade=trade))

        assert trade.r_multiple == pytest.approx(0.2)
        assert trade.realized_r_multiple == pytest.approx(-0.1)
        assert s._recent_exits["BTC"]["loss_r"] == pytest.approx(0.1)
        s._risk_manager.record_trade.assert_called_once_with(trade.net_pnl, trade.exit_time)

    def test_on_shutdown_does_not_crash(self):
        s = TrendStrategy()
        ctx = _make_ctx()
        s.on_init(ctx)
        s.on_shutdown(ctx)

    def test_skip_entry_when_position_exists(self):
        """No new entry when position already open for symbol."""
        cfg = TrendConfig(symbols=["BTC"])
        s = TrendStrategy(cfg)
        ctx = _make_ctx()
        s.on_init(ctx)

        # Broker has an existing position
        ctx.broker.get_position.return_value = Position("BTC", Side.LONG, 0.1, 50000)

        for i in range(WARMUP_BARS + 5):
            bar = _make_bar("BTC", TimeFrame.H1, 50000 + i)
            s.on_bar(bar, ctx)

        # No entry orders should be submitted (only position mgmt)
        entry_calls = [c for c in ctx.broker.submit_order.call_args_list
                      if c[0][0].tag == "entry"]
        assert len(entry_calls) == 0

    def test_risk_check_stops_entry(self):
        """Risk manager stopping session prevents new entries."""
        cfg = TrendConfig(symbols=["BTC"])
        s = TrendStrategy(cfg)
        ctx = _make_ctx()
        s.on_init(ctx)

        # Manually stop risk manager
        s._risk_manager._consecutive_losses = 5
        s._risk_manager._cfg.max_consecutive_losses = 3

        for i in range(WARMUP_BARS + 5):
            bar = _make_bar("BTC", TimeFrame.H1, 50000 + i)
            s.on_bar(bar, ctx)

        ctx.broker.submit_order.assert_not_called()

    def test_journal_property(self):
        s = TrendStrategy()
        assert s.journal is not None

    def test_reentry_uses_scaled_risk_after_scratch_exit(self):
        from crypto_trader.strategy.trend.config import TrendReentryParams

        cfg = TrendConfig(
            symbols=["BTC"],
            reentry=TrendReentryParams(
                enabled=True,
                cooldown_bars=1,
                max_loss_r=0.25,
                max_reentries=1,
                min_confluences_override=1,
                max_wait_bars=4,
                require_same_direction=True,
                only_after_scratch_exit=True,
                risk_scale=0.5,
            ),
        )
        ctx = _make_ctx()
        ctx.broker.get_equity.return_value = 10000.0
        ctx.bars.get = MagicMock(return_value=[_make_bar("BTC", TimeFrame.H1, 50000)] * 50)

        s = TestTrendSymbolFilter()._make_strategy_with_regime(cfg, ctx, direction=Side.LONG)
        s._recent_exits["BTC"] = {
            "bar_idx": WARMUP_BARS,
            "side": Side.LONG,
            "loss_r": 0.10,
            "exit_reason": "scratch_exit",
        }
        s._setup_detector.detect = MagicMock(return_value=_mock_setup(Side.LONG))
        s._stop_placer.compute = MagicMock(return_value=49500)
        s._sizer.compute = MagicMock(return_value=(_mock_sizing(), ""))

        s._handle_h1(_make_bar("BTC", TimeFrame.H1, 50000), "BTC", ctx)

        assert ctx.broker.submit_order.called
        assert s._setup_detector.detect.call_args.kwargs["min_confluences_override"] == 1
        assert s._sizer.compute.call_args.kwargs["risk_scale"] == 0.5
        assert s._reentry_count["BTC"] == 1

    def test_reentry_same_direction_mismatch_falls_back_to_normal_entry(self):
        from crypto_trader.strategy.trend.config import TrendReentryParams

        cfg = TrendConfig(
            symbols=["BTC"],
            reentry=TrendReentryParams(
                enabled=True,
                cooldown_bars=1,
                max_loss_r=0.25,
                max_reentries=1,
                min_confluences_override=1,
                require_same_direction=True,
                only_after_scratch_exit=True,
            ),
        )
        ctx = _make_ctx()
        ctx.broker.get_equity.return_value = 10000.0
        ctx.bars.get = MagicMock(return_value=[_make_bar("BTC", TimeFrame.H1, 50000)] * 50)

        s = TestTrendSymbolFilter()._make_strategy_with_regime(cfg, ctx, direction=Side.LONG)
        s._recent_exits["BTC"] = {
            "bar_idx": WARMUP_BARS,
            "side": Side.SHORT,
            "loss_r": 0.10,
            "exit_reason": "scratch_exit",
        }
        s._setup_detector.detect = MagicMock(return_value=_mock_setup(Side.LONG))
        s._stop_placer.compute = MagicMock(return_value=49500)
        s._sizer.compute = MagicMock(return_value=(_mock_sizing(), ""))

        s._handle_h1(_make_bar("BTC", TimeFrame.H1, 50000), "BTC", ctx)

        assert ctx.broker.submit_order.called
        assert s._setup_detector.detect.call_args.kwargs["min_confluences_override"] is None
        assert s._sizer.compute.call_args.kwargs["risk_scale"] == 1.0
        assert s._recent_exits["BTC"] == {}

    def test_non_scratch_loss_falls_back_to_normal_entry_when_required(self):
        from crypto_trader.strategy.trend.config import TrendReentryParams

        cfg = TrendConfig(
            symbols=["BTC"],
            reentry=TrendReentryParams(
                enabled=True,
                cooldown_bars=1,
                max_loss_r=0.25,
                max_reentries=1,
                min_confluences_override=1,
                only_after_scratch_exit=True,
                risk_scale=0.5,
            ),
        )
        ctx = _make_ctx()
        ctx.broker.get_equity.return_value = 10000.0
        ctx.bars.get = MagicMock(return_value=[_make_bar("BTC", TimeFrame.H1, 50000)] * 50)

        s = TestTrendSymbolFilter()._make_strategy_with_regime(cfg, ctx, direction=Side.LONG)
        s._recent_exits["BTC"] = {
            "bar_idx": WARMUP_BARS,
            "side": Side.LONG,
            "loss_r": 0.10,
            "exit_reason": "protective_stop",
        }
        s._setup_detector.detect = MagicMock(return_value=_mock_setup(Side.LONG))
        s._stop_placer.compute = MagicMock(return_value=49500)
        s._sizer.compute = MagicMock(return_value=(_mock_sizing(), ""))

        s._handle_h1(_make_bar("BTC", TimeFrame.H1, 50000), "BTC", ctx)

        assert ctx.broker.submit_order.called
        assert s._setup_detector.detect.call_args.kwargs["min_confluences_override"] is None
        assert s._sizer.compute.call_args.kwargs["risk_scale"] == 1.0
        assert s._recent_exits["BTC"] == {}

    def test_required_confirmation_stores_pending_setup(self):
        cfg = TrendConfig(symbols=["BTC"])
        cfg.confirmation.require_confirmation_for_b = True
        cfg.confirmation.max_bars_after_setup = 2
        ctx = _make_ctx()
        ctx.broker.get_equity.return_value = 10000.0
        ctx.bars.get = MagicMock(return_value=[_make_bar("BTC", TimeFrame.H1, 50000)] * 50)

        s = TestTrendSymbolFilter()._make_strategy_with_regime(cfg, ctx, direction=Side.LONG)
        setup = _mock_setup(Side.LONG)
        s._setup_detector.detect = MagicMock(return_value=setup)
        s._trigger_detector.check = MagicMock(return_value=None)
        s._stop_placer.compute = MagicMock(return_value=49500)

        s._handle_h1(_make_bar("BTC", TimeFrame.H1, 50000), "BTC", ctx)

        ctx.broker.submit_order.assert_not_called()
        s._stop_placer.compute.assert_not_called()
        assert s._pending_setups["BTC"].setup is setup

    def test_pending_setup_enters_on_later_confirmation(self):
        cfg = TrendConfig(symbols=["BTC"])
        cfg.confirmation.require_confirmation_for_b = True
        cfg.confirmation.max_bars_after_setup = 2
        ctx = _make_ctx()
        ctx.broker.get_equity.return_value = 10000.0
        ctx.bars.get = MagicMock(return_value=[_make_bar("BTC", TimeFrame.H1, 50000)] * 50)

        s = TestTrendSymbolFilter()._make_strategy_with_regime(cfg, ctx, direction=Side.LONG)
        setup = _mock_setup(Side.LONG)
        s._pending_setups["BTC"] = _PendingTrendSetup(
            setup=setup,
            created_h1_bar_index=s._h1_bar_count["BTC"],
            regime_tier="A",
        )
        s._setup_detector.detect = MagicMock(return_value=None)
        s._trigger_detector.check = MagicMock(return_value=TriggerResult(
            pattern="structure_break",
            trigger_price=50100,
            bar_index=0,
            volume_confirmed=True,
        ))
        s._stop_placer.compute = MagicMock(return_value=49500)
        s._sizer.compute = MagicMock(return_value=(_mock_sizing(), ""))

        s._handle_h1(_make_bar("BTC", TimeFrame.H1, 50000), "BTC", ctx)

        ctx.broker.submit_order.assert_called()
        assert "BTC" not in s._pending_setups
        assert s._position_meta["BTC"].confirmation_type == "structure_break"


class TestTrendSymbolFilter:
    """Tests for per-symbol direction filtering."""

    def _make_strategy_with_regime(self, cfg, ctx, sym="BTC", direction=Side.LONG):
        """Set up strategy past warmup with a forced regime."""
        from crypto_trader.strategy.trend.regime import RegimeResult
        s = TrendStrategy(cfg)
        s.on_init(ctx)
        # Skip warmup
        s._h1_bar_count[sym] = WARMUP_BARS + 1
        # Force an indicator snapshot
        from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
        s._h1_indicators[sym] = IndicatorSnapshot(
            ema_fast=49000, ema_fast_arr=None, ema_mid=48000, ema_mid_arr=None,
            ema_slow=0, ema_slow_arr=None, atr=200.0, atr_avg=200.0,
            rsi=50.0, adx=25.0, di_plus=20.0, di_minus=15.0,
            adx_rising=False, volume_ma=100.0,
        )
        # Force regime
        s._current_regime[sym] = RegimeResult("A", direction, 25.0, 49000, 48000, ("test",))
        return s

    def test_both_allows_all_directions(self):
        """Default 'both' allows long and short."""
        from crypto_trader.strategy.trend.config import TrendSymbolFilterParams
        cfg = TrendConfig(
            symbols=["BTC"],
            symbol_filter=TrendSymbolFilterParams(btc_direction="both"),
        )
        ctx = _make_ctx()
        ctx.broker.get_equity.return_value = 10000.0
        ctx.bars.get = MagicMock(return_value=[_make_bar("BTC", TimeFrame.H1, 50000)] * 50)

        s = self._make_strategy_with_regime(cfg, ctx, direction=Side.LONG)
        bar = _make_bar("BTC", TimeFrame.H1, 49000)
        # Should NOT return early at direction filter — may or may not find a setup
        # but the code should proceed past the filter step
        s._handle_h1(bar, "BTC", ctx)
        # No assertion on order — just confirming no crash and filter didn't block

    def test_long_only_blocks_short(self):
        """long_only should block short direction entries."""
        from crypto_trader.strategy.trend.config import TrendSymbolFilterParams
        cfg = TrendConfig(
            symbols=["BTC"],
            symbol_filter=TrendSymbolFilterParams(btc_direction="long_only"),
        )
        ctx = _make_ctx()
        ctx.broker.get_equity.return_value = 10000.0
        ctx.bars.get = MagicMock(return_value=[_make_bar("BTC", TimeFrame.H1, 50000)] * 50)

        s = self._make_strategy_with_regime(cfg, ctx, direction=Side.SHORT)
        bar = _make_bar("BTC", TimeFrame.H1, 49000)
        s._handle_h1(bar, "BTC", ctx)
        ctx.broker.submit_order.assert_not_called()

    def test_short_only_blocks_long(self):
        """short_only should block long direction entries."""
        from crypto_trader.strategy.trend.config import TrendSymbolFilterParams
        cfg = TrendConfig(
            symbols=["BTC"],
            symbol_filter=TrendSymbolFilterParams(btc_direction="short_only"),
        )
        ctx = _make_ctx()
        ctx.broker.get_equity.return_value = 10000.0
        ctx.bars.get = MagicMock(return_value=[_make_bar("BTC", TimeFrame.H1, 50000)] * 50)

        s = self._make_strategy_with_regime(cfg, ctx, direction=Side.LONG)
        bar = _make_bar("BTC", TimeFrame.H1, 49000)
        s._handle_h1(bar, "BTC", ctx)
        ctx.broker.submit_order.assert_not_called()

    def test_disabled_blocks_all(self):
        """disabled should block all entries for that symbol."""
        from crypto_trader.strategy.trend.config import TrendSymbolFilterParams
        cfg = TrendConfig(
            symbols=["BTC"],
            symbol_filter=TrendSymbolFilterParams(btc_direction="disabled"),
        )
        ctx = _make_ctx()
        ctx.broker.get_equity.return_value = 10000.0
        ctx.bars.get = MagicMock(return_value=[_make_bar("BTC", TimeFrame.H1, 50000)] * 50)

        s = self._make_strategy_with_regime(cfg, ctx, direction=Side.LONG)
        bar = _make_bar("BTC", TimeFrame.H1, 49000)
        s._handle_h1(bar, "BTC", ctx)
        ctx.broker.submit_order.assert_not_called()

    def test_h1_regime_fallback_proceeds_past_gate(self):
        """When D1 regime is 'none' but H1 conditions met, entry path continues."""
        from crypto_trader.strategy.trend.config import TrendSymbolFilterParams
        from crypto_trader.strategy.trend.regime import RegimeResult
        cfg = TrendConfig(symbols=["BTC"])
        ctx = _make_ctx()
        ctx.broker.get_equity.return_value = 10000.0
        ctx.bars.get = MagicMock(return_value=[_make_bar("BTC", TimeFrame.H1, 50000)] * 50)

        s = TrendStrategy(cfg)
        s.on_init(ctx)
        # Skip warmup
        s._h1_bar_count["BTC"] = WARMUP_BARS + 1
        # Set H1 indicators that satisfy h1 regime (close > emas, emas ordered, adx >= 20)
        from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
        s._h1_indicators["BTC"] = IndicatorSnapshot(
            ema_fast=49000, ema_fast_arr=None, ema_mid=48000, ema_mid_arr=None,
            ema_slow=0, ema_slow_arr=None, atr=200.0, atr_avg=200.0,
            rsi=50.0, adx=25.0, di_plus=20.0, di_minus=15.0,
            adx_rising=False, volume_ma=100.0,
        )
        # Set D1 regime to "none" — triggers H1 fallback
        s._current_regime["BTC"] = RegimeResult("none", None, 8.0, 49000, 48000, ("adx_below_no_trade",))

        bar = _make_bar("BTC", TimeFrame.H1, 50000)
        # Spy on evaluate_h1 to confirm it was called
        with patch.object(s._regime_classifier, "evaluate_h1", wraps=s._regime_classifier.evaluate_h1) as spy:
            s._handle_h1(bar, "BTC", ctx)
            spy.assert_called_once_with(bar.close, s._h1_indicators["BTC"])

    def test_unknown_symbol_defaults_both(self):
        """Symbols not in filter params default to 'both' via getattr."""
        from crypto_trader.strategy.trend.config import TrendSymbolFilterParams
        cfg = TrendConfig(
            symbols=["DOGE"],
            symbol_filter=TrendSymbolFilterParams(),
        )
        ctx = _make_ctx()
        ctx.broker.get_equity.return_value = 10000.0
        ctx.bars.get = MagicMock(return_value=[_make_bar("DOGE", TimeFrame.H1, 0.15)] * 50)

        s = self._make_strategy_with_regime(cfg, ctx, sym="DOGE", direction=Side.LONG)
        bar = _make_bar("DOGE", TimeFrame.H1, 0.15)
        # getattr(sf, "doge_direction", "both") returns "both" — should not block
        s._handle_h1(bar, "DOGE", ctx)
        # No crash, no early return at filter — may or may not submit depending on setup
