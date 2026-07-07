"""Tests for BreakoutStrategy — properties and basic behavior."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.core.events import PositionClosedEvent
from crypto_trader.core.models import Bar, Fill, Order, OrderType, Position, SetupGrade, Side, TimeFrame, Trade
from crypto_trader.core.runtime_types import OrderIntent
from crypto_trader.strategy.breakout.balance import BalanceZone
from crypto_trader.strategy.breakout.config import BreakoutConfig
from crypto_trader.strategy.breakout.confirmation import BreakoutConfirmation
from crypto_trader.strategy.breakout.setup import BreakoutSetupResult
from crypto_trader.strategy.breakout.strategy import BreakoutStrategy, WARMUP_BARS, _PositionMeta
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
from crypto_trader.strategy.momentum.journal import TradeJournal


# ---------------------------------------------------------------------------
# Minimal mock context
# ---------------------------------------------------------------------------

class _MockBroker:
    def get_position(self, sym):
        return None

    def get_equity(self):
        return 10000.0

    def get_open_orders(self, sym):
        return []

    def submit_order(self, order):
        pass

    def cancel_order(self, oid):
        pass


class _MockBars:
    def get(self, sym, tf, count=100):
        return []


class _MockEvents:
    def subscribe(self, event_type, handler):
        pass


class _MockClock:
    pass


class _MockCtx:
    def __init__(self):
        self.broker = _MockBroker()
        self.bars = _MockBars()
        self.events = _MockEvents()
        self.clock = _MockClock()
        self.config = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_bar(
    sym: str = "BTC",
    tf: TimeFrame = TimeFrame.M30,
    ts: datetime = _TS,
) -> Bar:
    return Bar(
        timestamp=ts,
        symbol=sym,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0,
        timeframe=tf,
    )


def _make_fill(tag: str = "entry", sym: str = "BTC") -> Fill:
    return Fill(
        order_id="test-oid",
        symbol=sym,
        side=Side.LONG,
        qty=0.1,
        fill_price=100.0,
        commission=0.01,
        timestamp=_TS,
        tag=tag,
    )


def _make_indicator() -> IndicatorSnapshot:
    return IndicatorSnapshot(
        ema_fast=101.0,
        ema_mid=100.0,
        ema_slow=99.0,
        ema_fast_arr=None,
        ema_mid_arr=None,
        ema_slow_arr=None,
        adx=25.0,
        di_plus=20.0,
        di_minus=10.0,
        adx_rising=True,
        atr=1.0,
        atr_avg=1.0,
        rsi=55.0,
        volume_ma=1000.0,
    )


def _make_zone() -> BalanceZone:
    return BalanceZone(
        center=100.0,
        upper=101.0,
        lower=99.0,
        bars_in_zone=5,
        touches=2,
        formation_bar_idx=10,
        volume_contracting=True,
        width_atr=0.8,
    )


def _make_setup() -> BreakoutSetupResult:
    return BreakoutSetupResult(
        grade=SetupGrade.B,
        is_a_plus=False,
        direction=Side.LONG,
        balance_zone=_make_zone(),
        breakout_price=101.0,
        lvn_runway_atr=1.5,
        confluences=("ema_alignment",),
        room_r=1.8,
        volume_mult=1.2,
        body_ratio=0.6,
    )


def _make_confirmation() -> BreakoutConfirmation:
    return BreakoutConfirmation(
        model="model1_close",
        trigger_price=100.5,
        bar_index=0,
        volume_confirmed=True,
    )


def _prime_model1_signal_path(
    strategy: BreakoutStrategy,
    ctx: _MockCtx,
    setup: BreakoutSetupResult,
    *,
    m30_ind: IndicatorSnapshot | None = None,
) -> None:
    m30_ind = m30_ind or _make_indicator()
    strategy._m30_bar_count["BTC"] = WARMUP_BARS
    strategy._m30_inc["BTC"] = MagicMock(update=MagicMock(return_value=m30_ind))
    strategy._current_profile["BTC"] = MagicMock()
    strategy._profile_bar_count["BTC"] = 0
    ctx.bars.get = MagicMock(return_value=[_make_bar()] * 120)
    ctx.broker.get_position = MagicMock(return_value=None)
    ctx.broker.get_equity = MagicMock(return_value=10000.0)
    ctx.broker.get_open_orders = MagicMock(return_value=[])
    ctx.broker.submit_order = MagicMock()
    strategy._manage_positions = MagicMock()
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
    strategy._confirmation_detector.check_breakout_close = MagicMock(
        return_value=_make_confirmation()
    )


# ---------------------------------------------------------------------------
# Tests — properties
# ---------------------------------------------------------------------------

class TestBreakoutStrategyProperties:
    """Test strategy name, timeframes, symbols, journal."""

    def test_name(self):
        """strategy.name == 'volume_profile_breakout'."""
        s = BreakoutStrategy()
        assert s.name == "volume_profile_breakout"

    def test_timeframes(self):
        """strategy.timeframes == [TimeFrame.M30, TimeFrame.H4]."""
        s = BreakoutStrategy()
        assert s.timeframes == [TimeFrame.M30, TimeFrame.H4]

    def test_symbols(self):
        """strategy.symbols matches config symbols."""
        cfg = BreakoutConfig(symbols=["BTC", "SOL"])
        s = BreakoutStrategy(config=cfg)
        assert s.symbols == ["BTC", "SOL"]

    def test_default_config(self):
        """Default config is BreakoutConfig() when none provided."""
        s = BreakoutStrategy()
        assert s.symbols == BreakoutConfig().symbols

    def test_journal_exists(self):
        """strategy.journal is a TradeJournal."""
        s = BreakoutStrategy()
        assert isinstance(s.journal, TradeJournal)

    def test_custom_config(self):
        """Custom config overrides defaults."""
        cfg = BreakoutConfig(symbols=["ETH"])
        s = BreakoutStrategy(config=cfg)
        assert s.symbols == ["ETH"]
        assert len(s.symbols) == 1


# ---------------------------------------------------------------------------
# Tests — on_bar / on_fill robustness
# ---------------------------------------------------------------------------

class TestBreakoutStrategyBehavior:
    """Test on_bar and on_fill edge cases."""

    def _init_strategy(self, cfg: BreakoutConfig | None = None) -> BreakoutStrategy:
        s = BreakoutStrategy(config=cfg)
        ctx = _MockCtx()
        s.on_init(ctx)
        return s

    def test_on_bar_ignores_unknown_symbol(self):
        """on_bar with unknown symbol doesn't crash."""
        s = self._init_strategy()
        bar = _make_bar(sym="DOGE", tf=TimeFrame.M30)
        ctx = _MockCtx()
        # Should silently return — DOGE not in config symbols
        s.on_bar(bar, ctx)

    def test_on_bar_ignores_wrong_timeframe(self):
        """on_bar with TimeFrame.D1 does nothing (no crash)."""
        s = self._init_strategy()
        bar = _make_bar(sym="BTC", tf=TimeFrame.D1)
        ctx = _MockCtx()
        s.on_bar(bar, ctx)

    def test_exit_manager_blank_orders_get_deterministic_ids_before_submit(self):
        s = self._init_strategy()
        ctx = _MockCtx()
        submitted: list[Order] = []
        ctx.broker = MagicMock()
        ctx.broker.get_position.return_value = Position("BTC", Side.LONG, 0.1, 100.0)
        ctx.broker.get_open_orders.return_value = []
        ctx.broker.submit_order.side_effect = lambda order: submitted.append(order) or order.order_id
        ctx.broker.cancel_order.return_value = True
        s._position_meta["BTC"] = _PositionMeta(
            entry_price=100.0,
            stop_level=98.0,
            stop_distance=2.0,
            original_qty=0.1,
            entry_bar_index=1,
        )
        state = SimpleNamespace(
            direction=Side.LONG,
            entry_price=100.0,
            stop_distance=2.0,
            remaining_qty=0.05,
            bars_since_entry=3,
            current_r=1.2,
            mfe_r=1.5,
            mae_r=-0.1,
            peak_r=1.5,
            tp1_hit=True,
            tp2_hit=False,
            be_moved=False,
            early_lock_applied=False,
        )
        s._exit_manager.process_bar = MagicMock(return_value=[
            Order("", "BTC", Side.SHORT, OrderType.MARKET, 0.03, tag="tp1"),
            Order("", "BTC", Side.SHORT, OrderType.MARKET, 0.02, tag="time_stop"),
            Order("", "BTC", Side.SHORT, OrderType.MARKET, 0.05, tag="invalidation"),
        ])
        s._exit_manager.get_state = MagicMock(return_value=state)
        s._trail_manager.update = MagicMock(return_value=None)
        s._m30_bar_count["BTC"] = 4
        bar = _make_bar(ts=datetime(2026, 5, 31, 12, 30, tzinfo=timezone.utc))

        s._manage_positions(bar, "BTC", ctx, [bar], _make_indicator())

        assert [order.order_id.split("_", 2)[:2] for order in submitted] == [
            ["brk", "tp1"],
            ["brk", "time"],
            ["brk", "invalidation"],
        ]
        assert len({order.order_id for order in submitted}) == 3
        for order in submitted:
            intent = OrderIntent.from_order(order)
            assert intent.client_order_id == order.order_id
            assert intent.intent_id == order.order_id
            assert "manual:intent:1" not in order.order_id

    def test_warmup_gate(self):
        """M30 bars before WARMUP_BARS don't trigger entries."""
        s = self._init_strategy()
        ctx = _MockCtx()
        # Feed fewer than WARMUP_BARS (101) M30 bars — no orders submitted
        for i in range(50):
            day = 1 + i // 48
            hour = (i // 2) % 24
            minute = (i % 2) * 30
            ts = datetime(2026, 1, day, hour, minute, tzinfo=timezone.utc)
            bar = _make_bar(sym="BTC", tf=TimeFrame.M30, ts=ts)
            s.on_bar(bar, ctx)
        # No crash and no orders — broker.submit_order was never called
        # (mock doesn't track, but no exception = pass)

    def test_on_fill_unknown_tag(self):
        """on_fill with unknown tag doesn't crash."""
        s = self._init_strategy()
        ctx = _MockCtx()
        fill = _make_fill(tag="unknown_tag_xyz")
        s.on_fill(fill, ctx)

    def test_on_fill_entry_uses_actual_fill_qty_for_stop_management(self):
        s = BreakoutStrategy(BreakoutConfig(symbols=["BTC"]))
        ctx = _MockCtx()
        ctx.broker = MagicMock()
        s.on_init(ctx)

        s._position_meta["BTC"] = _PositionMeta(
            setup_grade=SetupGrade.B,
            entry_price=100.0,
            stop_level=95.0,
            stop_distance=5.0,
            original_qty=0.2,
            balance_zone=BalanceZone(
                center=100.0,
                upper=101.0,
                lower=99.0,
                bars_in_zone=5,
                touches=2,
                formation_bar_idx=10,
                volume_contracting=True,
                width_atr=0.8,
            ),
        )

        fill = _make_fill(tag="entry")
        s.on_fill(fill, ctx)

        stop_order = ctx.broker.submit_order.call_args[0][0]
        assert stop_order.tag == "protective_stop"
        assert stop_order.qty == pytest.approx(fill.qty)
        assert s._position_meta["BTC"].original_qty == pytest.approx(fill.qty)

    def test_position_closed_records_recent_net_loss_from_realized_r(self):
        s = BreakoutStrategy(BreakoutConfig(symbols=["BTC"]))
        ctx = _MockCtx()
        ctx.broker = MagicMock()
        ctx.broker.get_open_orders.return_value = []
        s.on_init(ctx)

        exit_state = MagicMock()
        exit_state.mae_r = -0.2
        exit_state.mfe_r = 0.4
        s._exit_manager.remove = MagicMock(return_value=exit_state)
        s._trail_manager.remove = MagicMock()
        s._confirmation_detector.clear_pending = MagicMock()
        s._risk_manager.record_trade_exit = MagicMock()
        s._position_meta["BTC"] = _PositionMeta(
            setup_grade=SetupGrade.B,
            confirmation_type="model1",
            entry_method="model1",
            entry_price=100.0,
            stop_level=95.0,
            stop_distance=5.0,
            original_qty=2.0,
        )

        trade = Trade(
            trade_id="t1",
            symbol="BTC",
            direction=Side.LONG,
            entry_price=100.0,
            exit_price=102.0,
            qty=1.0,
            entry_time=_TS,
            exit_time=_TS,
            pnl=2.0,
            r_multiple=None,
            commission=3.0,
            bars_held=4,
            setup_grade=None,
            exit_reason="protective_stop",
            confluences_used=None,
            confirmation_type=None,
            entry_method=None,
            funding_paid=0.0,
            mae_r=None,
            mfe_r=None,
        )

        s._on_position_closed(PositionClosedEvent(timestamp=_TS, trade=trade))

        assert trade.r_multiple == pytest.approx(0.4)
        assert trade.realized_r_multiple == pytest.approx(-0.2)
        assert s._recent_exits["BTC"]["loss_r"] == pytest.approx(0.2)
        s._confirmation_detector.clear_pending.assert_not_called()
        s._risk_manager.record_trade_exit.assert_called_once_with(
            trade.net_pnl,
            trade.exit_time,
        )


class TestBreakoutPathIndependentSignalState:
    """Signal-state updates should not depend on execution-state gates."""

    def _init_strategy_and_ctx(
        self,
        cfg: BreakoutConfig | None = None,
    ) -> tuple[BreakoutStrategy, _MockCtx]:
        cfg = cfg or BreakoutConfig(symbols=["BTC"])
        s = BreakoutStrategy(config=cfg)
        ctx = _MockCtx()
        s.on_init(ctx)
        return s, ctx

    def test_model1_preserves_zone_before_measurement_without_order(self):
        s, ctx = self._init_strategy_and_ctx()
        ctx.config = BacktestConfig(
            symbols=["BTC"],
            start_date=date(2026, 1, 5),
            initial_equity=10000.0,
        )
        setup = _make_setup()
        _prime_model1_signal_path(s, ctx, setup)
        s._execute_entry = MagicMock(return_value=True)

        s._handle_m30(
            _make_bar(ts=datetime(2026, 1, 4, 23, 30, tzinfo=timezone.utc)),
            "BTC",
            ctx,
        )

        s._balance_detector.consume_zone.assert_not_called()
        s._execute_entry.assert_not_called()
        ctx.broker.submit_order.assert_not_called()

    def test_model1_preserves_zone_with_open_position_without_order(self):
        s, ctx = self._init_strategy_and_ctx()
        setup = _make_setup()
        _prime_model1_signal_path(s, ctx, setup)
        ctx.broker.get_position.return_value = MagicMock(qty=1.0)
        s._execute_entry = MagicMock(return_value=True)

        s._handle_m30(_make_bar(), "BTC", ctx)

        s._balance_detector.consume_zone.assert_not_called()
        s._execute_entry.assert_not_called()
        ctx.broker.submit_order.assert_not_called()

    def test_model1_preserves_zone_when_risk_stopped_without_order(self):
        s, ctx = self._init_strategy_and_ctx()
        setup = _make_setup()
        _prime_model1_signal_path(s, ctx, setup)
        s._risk_manager.is_session_stopped = MagicMock(return_value=(True, "daily_loss_limit"))
        s._execute_entry = MagicMock(return_value=True)

        s._handle_m30(_make_bar(), "BTC", ctx)

        s._balance_detector.consume_zone.assert_not_called()
        s._execute_entry.assert_not_called()
        ctx.broker.submit_order.assert_not_called()

    def test_reentry_cooldown_blocks_order_not_signal_state(self):
        s, ctx = self._init_strategy_and_ctx()
        setup = _make_setup()
        _prime_model1_signal_path(s, ctx, setup)
        s._recent_exits["BTC"] = {
            "bar_idx": WARMUP_BARS,
            "side": Side.LONG,
            "loss_r": 0.4,
        }
        s._execute_entry = MagicMock(return_value=True)

        s._handle_m30(_make_bar(), "BTC", ctx)

        s._balance_detector.consume_zone.assert_not_called()
        s._execute_entry.assert_not_called()
        assert s._recent_exits["BTC"]["loss_r"] == pytest.approx(0.4)

    def test_stale_recent_exit_is_cleared_after_max_wait_bars(self):
        s, _ = self._init_strategy_and_ctx()
        s._m30_bar_count["BTC"] = 100
        s._recent_exits["BTC"] = {
            "bar_idx": 80,
            "side": Side.LONG,
            "loss_r": 0.4,
        }
        s._reentry_count["BTC"] = 1

        is_reentry, block_reason = s._evaluate_reentry_for_execution("BTC", Side.LONG)

        assert is_reentry is False
        assert block_reason == ""
        assert s._recent_exits["BTC"] == {}
        assert s._reentry_count["BTC"] == 0

    def test_max_reentries_reached_clears_after_cooldown_for_fresh_signals(self):
        s, _ = self._init_strategy_and_ctx()
        s._m30_bar_count["BTC"] = 100
        s._recent_exits["BTC"] = {
            "bar_idx": 96,
            "side": Side.LONG,
            "loss_r": 0.4,
        }
        s._reentry_count["BTC"] = s._cfg.reentry.max_reentries

        is_reentry, block_reason = s._evaluate_reentry_for_execution("BTC", Side.LONG)

        assert is_reentry is False
        assert block_reason == ""
        assert s._recent_exits["BTC"] == {}
        assert s._reentry_count["BTC"] == 0

    def test_nonpositive_max_wait_preserves_unbounded_reentry_wait(self):
        cfg = BreakoutConfig(symbols=["BTC"])
        cfg.reentry.max_wait_bars = 0
        s, _ = self._init_strategy_and_ctx(cfg)
        s._m30_bar_count["BTC"] = 100
        s._recent_exits["BTC"] = {
            "bar_idx": 10,
            "side": Side.LONG,
            "loss_r": 0.4,
        }
        s._reentry_count["BTC"] = 0

        is_reentry, block_reason = s._evaluate_reentry_for_execution("BTC", Side.LONG)

        assert is_reentry is True
        assert block_reason == ""
        assert s._recent_exits["BTC"]["bar_idx"] == 10

    def test_opposite_direction_after_loss_is_normal_signal(self):
        s, _ = self._init_strategy_and_ctx()
        s._m30_bar_count["BTC"] = 104
        s._recent_exits["BTC"] = {
            "bar_idx": 100,
            "side": Side.LONG,
            "loss_r": 0.4,
        }

        is_reentry, block_reason = s._evaluate_reentry_for_execution("BTC", Side.SHORT)

        assert is_reentry is False
        assert block_reason == ""
        assert s._recent_exits["BTC"] == {}

    def test_reentry_disabled_clears_recent_loss_without_cooldown_block(self):
        cfg = BreakoutConfig(symbols=["BTC"])
        cfg.reentry.enabled = False
        s, _ = self._init_strategy_and_ctx(cfg)
        s._m30_bar_count["BTC"] = 101
        s._recent_exits["BTC"] = {
            "bar_idx": 100,
            "side": Side.LONG,
            "loss_r": 0.4,
        }

        is_reentry, block_reason = s._evaluate_reentry_for_execution("BTC", Side.LONG)

        assert is_reentry is False
        assert block_reason == ""
        assert s._recent_exits["BTC"] == {}

    def test_reentry_risk_scale_is_execution_overlay(self):
        cfg = BreakoutConfig(symbols=["BTC"])
        cfg.reentry.cooldown_bars = 0
        cfg.reentry.risk_scale = 0.5
        s, ctx = self._init_strategy_and_ctx(cfg)
        setup = _make_setup()
        _prime_model1_signal_path(s, ctx, setup)
        s._recent_exits["BTC"] = {
            "bar_idx": WARMUP_BARS,
            "side": Side.LONG,
            "loss_r": 0.4,
        }
        s._execute_entry = MagicMock(return_value=True)

        s._handle_m30(_make_bar(), "BTC", ctx)

        execution_setup = s._execute_entry.call_args.args[3]
        assert execution_setup.risk_scale == pytest.approx(0.5)
        assert setup.risk_scale == pytest.approx(1.0)
