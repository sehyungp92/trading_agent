"""Targeted tests for MomentumStrategy close-path accounting."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crypto_trader.core.events import PositionClosedEvent
from crypto_trader.core.models import Bar, Fill, Order, OrderType, Position, SetupGrade, Side, TimeFrame, Trade
from crypto_trader.core.runtime_types import OrderIntent
from crypto_trader.strategy.momentum.config import MomentumConfig
from crypto_trader.strategy.momentum.strategy import MomentumStrategy, _PositionMeta


def _make_ctx():
    broker = MagicMock()
    broker.get_open_orders.return_value = []
    events = MagicMock()
    clock = MagicMock()
    bars = MagicMock()
    return SimpleNamespace(broker=broker, events=events, clock=clock, bars=bars, config={})


class TestMomentumStrategyClosePath:
    def test_position_closed_computes_realized_r_and_records_net_pnl(self):
        strategy = MomentumStrategy(MomentumConfig(symbols=["BTC"]))
        ctx = _make_ctx()
        strategy.on_init(ctx)

        exit_state = SimpleNamespace(mae_r=-0.2, mfe_r=0.6, partial_exits=[])
        strategy._exit_manager.remove_position = MagicMock(return_value=exit_state)
        strategy._trail_manager.remove = MagicMock()
        strategy._risk_manager.record_trade = MagicMock()
        strategy._position_meta["BTC"] = _PositionMeta(
            setup_grade=SetupGrade.B,
            confluences=("m15_ema20",),
            confirmation_type="inside_bar_break",
            entry_method="close",
            entry_price=50_000.0,
            stop_level=49_500.0,
            stop_distance=500.0,
            original_qty=0.2,
        )

        trade = Trade(
            trade_id="t1",
            symbol="BTC",
            direction=Side.LONG,
            entry_price=50_000.0,
            exit_price=50_100.0,
            qty=0.1,
            entry_time=datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
            exit_time=datetime(2026, 3, 15, 18, 0, tzinfo=timezone.utc),
            pnl=10.0,
            r_multiple=None,
            commission=15.0,
            bars_held=8,
            setup_grade=None,
            exit_reason="tp1",
            confluences_used=None,
            confirmation_type=None,
            entry_method=None,
            funding_paid=0.0,
            mae_r=None,
            mfe_r=None,
        )

        strategy._on_position_closed(
            PositionClosedEvent(timestamp=trade.exit_time, trade=trade)
        )

        assert trade.r_multiple == pytest.approx(0.2)
        assert trade.realized_r_multiple == pytest.approx(-0.1)
        strategy._risk_manager.record_trade.assert_called_once_with(
            trade.net_pnl,
            trade.exit_time,
        )


class TestMomentumManagementOrderIds:
    def test_protective_stop_uses_deterministic_nonblank_id_without_decision_context(self):
        strategy = MomentumStrategy(MomentumConfig(symbols=["BTC"]))
        ctx = _make_ctx()
        ctx.broker.submit_order.side_effect = lambda order: order.order_id
        strategy.on_init(ctx)
        strategy._position_meta["BTC"] = _PositionMeta(stop_level=49_500.0)
        fill = Fill(
            order_id="entry_1",
            exchange_order_id="ex_entry_1",
            exchange_fill_id="fill_1",
            symbol="BTC",
            side=Side.LONG,
            qty=0.2,
            fill_price=50_000.0,
            commission=0.0,
            timestamp=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
            tag="entry",
        )

        strategy.on_fill(fill, ctx)
        submitted = ctx.broker.submit_order.call_args.args[0]
        intent = OrderIntent.from_order(submitted)

        assert submitted.order_id.startswith("mom_stop_BTC_")
        assert submitted.order_id != "momentum:BTC:manual:intent:1"
        assert intent.client_order_id == submitted.order_id
        assert intent.intent_id == submitted.order_id
        assert strategy._management_order_id("stop", "BTC", {
            "fill_id": "fill_1",
            "order_id": "entry_1",
            "timestamp": "2026-05-31T12:00:00+00:00",
            "side": "LONG",
            "qty": 0.2,
            "stop_price": 49_500.0,
        }) == submitted.order_id

    def test_tp_replacement_and_trailing_stops_use_distinct_deterministic_ids(self):
        strategy = MomentumStrategy(MomentumConfig(symbols=["BTC"]))
        ctx = _make_ctx()
        submitted: list[Order] = []
        ctx.broker.submit_order.side_effect = lambda order: submitted.append(order) or order.order_id
        ctx.broker.cancel_order.return_value = True
        strategy.on_init(ctx)
        strategy._position_meta["BTC"] = _PositionMeta(
            confirmation_type="engulfing",
            entry_price=50_000.0,
            stop_level=49_500.0,
            stop_distance=500.0,
            entry_bar_index=1,
        )

        state = SimpleNamespace(
            current_stop_order_id="old_stop",
            current_stop_price=49_500.0,
            current_stop_tag="protective_stop",
            remaining_qty=0.1,
            be_moved=False,
            mfe_r=1.2,
        )
        strategy._exit_manager.get_state = MagicMock(return_value=state)
        fill = Fill(
            order_id="tp_1",
            exchange_order_id="ex_tp_1",
            exchange_fill_id="fill_tp_1",
            symbol="BTC",
            side=Side.SHORT,
            qty=0.1,
            fill_price=51_000.0,
            commission=0.0,
            timestamp=datetime(2026, 5, 31, 13, 0, tzinfo=timezone.utc),
            tag="tp1",
        )

        strategy.on_fill(fill, ctx)
        tp_stop_id = submitted[-1].order_id

        strategy._exit_manager.manage = MagicMock(return_value=[])
        strategy._trail_manager.update = MagicMock(return_value=50_500.0)
        strategy._m15_bar_count["BTC"] = 4
        ctx.broker.get_positions.return_value = [Position("BTC", Side.LONG, 0.1, 50_000.0)]
        ctx.broker.get_open_orders.return_value = [
            Order(tp_stop_id, "BTC", Side.SHORT, OrderType.STOP, 0.1, stop_price=49_500.0)
        ]
        bar = Bar(
            timestamp=datetime(2026, 5, 31, 13, 15, tzinfo=timezone.utc),
            symbol="BTC",
            open=50_900.0,
            high=51_100.0,
            low=50_800.0,
            close=51_000.0,
            volume=10.0,
            timeframe=TimeFrame.M15,
        )

        strategy._manage_positions(bar, ctx, [bar], MagicMock())
        trailing_stop_id = submitted[-1].order_id

        assert tp_stop_id.startswith("mom_stop_BTC_")
        assert trailing_stop_id.startswith("mom_trail_BTC_")
        assert tp_stop_id != trailing_stop_id
        assert "manual:intent:1" not in tp_stop_id
        assert "manual:intent:1" not in trailing_stop_id

    def test_exit_manager_blank_orders_get_deterministic_ids_before_submit(self):
        strategy = MomentumStrategy(MomentumConfig(symbols=["BTC"]))
        ctx = _make_ctx()
        submitted: list[Order] = []
        ctx.broker.submit_order.side_effect = lambda order: submitted.append(order) or order.order_id
        ctx.broker.get_positions.return_value = [Position("BTC", Side.LONG, 0.1, 50_000.0)]
        ctx.broker.get_open_orders.return_value = []
        strategy.on_init(ctx)
        strategy._position_meta["BTC"] = _PositionMeta(
            confirmation_type="engulfing",
            entry_price=50_000.0,
            stop_level=49_500.0,
            stop_distance=500.0,
            entry_bar_index=1,
        )
        state = SimpleNamespace(
            entry_price=50_000.0,
            stop_distance=500.0,
            remaining_qty=0.1,
            bars_since_entry=4,
            mfe_r=1.5,
            mae_r=-0.2,
            current_stop_order_id="old_stop",
            current_stop_price=49_500.0,
            current_stop_tag="protective_stop",
            tp1_hit=False,
            tp2_hit=False,
            be_moved=False,
            proof_lock_moved=False,
        )
        strategy._exit_manager.manage = MagicMock(return_value=[
            Order("", "BTC", Side.SHORT, OrderType.MARKET, 0.03, tag="tp1"),
            Order("", "BTC", Side.SHORT, OrderType.STOP, 0.07, stop_price=50_000.0, tag="breakeven_stop"),
        ])
        strategy._exit_manager.get_state = MagicMock(return_value=state)
        strategy._trail_manager.update = MagicMock(return_value=None)
        strategy._m15_bar_count["BTC"] = 5
        bar = Bar(
            timestamp=datetime(2026, 5, 31, 14, 0, tzinfo=timezone.utc),
            symbol="BTC",
            open=50_900.0,
            high=51_100.0,
            low=50_800.0,
            close=51_000.0,
            volume=10.0,
            timeframe=TimeFrame.M15,
        )

        strategy._manage_positions(bar, ctx, [bar], MagicMock())

        assert [order.order_id.split("_", 2)[:2] for order in submitted] == [
            ["mom", "tp1"],
            ["mom", "breakeven"],
        ]
        assert len({order.order_id for order in submitted}) == 2
        for order in submitted:
            intent = OrderIntent.from_order(order)
            assert intent.client_order_id == order.order_id
            assert intent.intent_id == order.order_id
            assert "manual:intent:1" not in order.order_id
