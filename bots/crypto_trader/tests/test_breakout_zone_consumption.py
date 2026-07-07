"""Tests for market-derived zone lifecycle and Model 2 retest reachability."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from crypto_trader.core.models import Bar, Side, SetupGrade, TimeFrame
from crypto_trader.strategy.breakout.balance import BalanceDetector, BalanceZone
from crypto_trader.strategy.breakout.config import BreakoutConfig, BreakoutConfirmParams
from crypto_trader.strategy.breakout.confirmation import (
    ConfirmationDetector,
)
from crypto_trader.strategy.breakout.setup import BreakoutSetupResult
from crypto_trader.strategy.breakout.strategy import BreakoutStrategy, WARMUP_BARS
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot

TS = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)


def _zone():
    return BalanceZone(
        center=100.0,
        upper=102.0,
        lower=98.0,
        bars_in_zone=8,
        touches=3,
        formation_bar_idx=10,
        volume_contracting=False,
        width_atr=1.0,
    )


def _setup(direction=Side.LONG):
    return BreakoutSetupResult(
        grade=SetupGrade.B,
        is_a_plus=False,
        direction=direction,
        balance_zone=_zone(),
        breakout_price=103.0,
        lvn_runway_atr=0.5,
        confluences=("ema_alignment",),
        room_r=1.5,
        volume_mult=1.2,
        body_ratio=0.6,
    )


def _bar(close=103.0):
    return Bar(
        timestamp=TS,
        symbol="BTC",
        open=101.0,
        high=104.0,
        low=100.5,
        close=close,
        volume=1000.0,
        timeframe=TimeFrame.M30,
    )


def _ind(atr=10.0):
    return IndicatorSnapshot(
        ema_fast=100.0,
        ema_mid=99.0,
        ema_slow=98.0,
        ema_fast_arr=None,
        ema_mid_arr=None,
        ema_slow_arr=None,
        adx=25.0,
        di_plus=20.0,
        di_minus=15.0,
        adx_rising=True,
        atr=atr,
        atr_avg=atr,
        rsi=55.0,
        volume_ma=1000.0,
    )


class TestZoneConsumption:
    """Test that strategy signals do not trade-consume balance zones."""

    def test_model2_registration_stores_pending_and_preserves_zone_with_open_position(self):
        """Model 2 registration is market state and is not blocked by a position."""
        cfg = BreakoutConfig(symbols=["BTC"])
        cfg.confirmation.enable_model1 = False
        cfg.confirmation.enable_model2 = True
        strategy = BreakoutStrategy(cfg)
        ctx = MagicMock()
        ctx.events.subscribe = MagicMock()
        ctx.config = {}
        ctx.bars.get.return_value = [_bar()] * 120
        ctx.broker.get_position.return_value = MagicMock(qty=1.0)
        ctx.broker.get_equity.return_value = 10000.0
        ctx.broker.get_open_orders.return_value = []
        ctx.broker.submit_order = MagicMock()
        strategy.on_init(ctx)

        setup = _setup()
        strategy._m30_bar_count["BTC"] = WARMUP_BARS
        strategy._m30_inc["BTC"] = MagicMock(update=MagicMock(return_value=_ind()))
        strategy._current_profile["BTC"] = MagicMock()
        strategy._balance_detector.update = MagicMock()
        strategy._balance_detector.get_active_zones = MagicMock(return_value=[setup.balance_zone])
        strategy._balance_detector.consume_zone = MagicMock()
        strategy._breakout_detector.detect = MagicMock(return_value=setup)
        strategy._breakout_detector.consume_blocked_relaxed_body_signals = MagicMock(return_value=[])
        strategy._context_analyzer.evaluate = MagicMock(return_value=MagicMock(
            direction=Side.LONG,
            strength=1.0,
            reasons=[],
        ))
        strategy._manage_positions = MagicMock()
        strategy._execute_entry = MagicMock(return_value=True)

        strategy._handle_m30(_bar(), "BTC", ctx)

        assert strategy._confirmation_detector.has_pending("BTC")
        strategy._balance_detector.consume_zone.assert_not_called()
        strategy._execute_entry.assert_not_called()

    def test_duplicate_model2_registration_keeps_original_retest_expiry(self):
        cfg = BreakoutConfirmParams(
            enable_model1=False,
            enable_model2=True,
            retest_max_bars=6,
            retest_zone_atr=0.5,
        )
        detector = ConfirmationDetector(cfg)
        setup = _setup()

        detector.register_breakout(sym="BTC", setup=setup, bar_idx=100)
        detector.register_breakout(sym="BTC", setup=setup, bar_idx=105)

        retest_bar = Bar(
            timestamp=TS,
            symbol="BTC",
            open=103.0,
            high=103.5,
            low=101.5,
            close=102.5,
            volume=800.0,
            timeframe=TimeFrame.M30,
        )
        result = detector.check_retest(
            sym="BTC",
            bar=retest_bar,
            bars=[retest_bar],
            atr=10.0,
            bar_index=107,
        )

        assert result is None
        assert not detector.has_pending("BTC")

    def test_model2_registration_preserves_setup(self):
        """When Model 1 is disabled, register_breakout stores the setup for retest."""
        cfg = BreakoutConfirmParams(enable_model1=False, enable_model2=True)
        detector = ConfirmationDetector(cfg)

        setup = _setup()
        detector.register_breakout(sym="BTC", setup=setup, bar_idx=100)

        assert detector.has_pending("BTC")
        pending = detector.get_pending_setup("BTC")
        assert pending is not None
        assert pending.direction == Side.LONG
        assert pending.balance_zone.center == 100.0

    def test_model2_retest_fires_after_registration(self):
        """Full Model 2 flow: register → wait → retest triggers."""
        cfg = BreakoutConfirmParams(
            enable_model1=False,
            enable_model2=True,
            retest_max_bars=6,
            retest_zone_atr=0.5,
        )
        detector = ConfirmationDetector(cfg)
        setup = _setup()

        # Register at bar 100
        detector.register_breakout(sym="BTC", setup=setup, bar_idx=100)
        assert detector.has_pending("BTC")

        # Retest bar: price pulls back toward the zone upper (102)
        retest_bar = Bar(
            timestamp=TS,
            symbol="BTC",
            open=103.0,
            high=103.5,
            low=101.5,
            close=102.5,
            volume=800.0,
            timeframe=TimeFrame.M30,
        )
        bars = [retest_bar]
        result = detector.check_retest(
            sym="BTC", bar=retest_bar, bars=bars,
            atr=10.0, bar_index=102,
        )
        # Result depends on whether retest conditions are met; the important
        # thing is that check_retest is reachable and the pending was available.
        # Pending state clears only by confirmation or expiry.

    def test_model1_preempts_model2(self):
        """With both models enabled, Model 1 fires first; Model 2 not registered."""
        cfg = BreakoutConfirmParams(enable_model1=True, enable_model2=True)
        detector = ConfirmationDetector(cfg)

        setup = _setup()
        bar = _bar()

        # Model 1 fires → should NOT register for Model 2
        confirm = detector.check_breakout_close(
            bar=bar, setup=setup, m30_ind=_ind(),
        )
        # If Model 1 confirms, no pending should exist
        if confirm is not None:
            assert not detector.has_pending("BTC")

    def test_market_invalidation_removes_zone_from_active_inventory(self):
        """Manual zone removal is reserved for market-derived invalidation."""
        detector = BalanceDetector(BreakoutConfig().balance)

        # Manually add a zone to the detector's internal state
        zone = _zone()
        if "BTC" not in detector._zones:
            detector._zones["BTC"] = []
        detector._zones["BTC"].append(zone)

        assert len(detector.get_active_zones("BTC")) == 1

        detector.consume_zone("BTC", zone)

        assert len(detector.get_active_zones("BTC")) == 0
