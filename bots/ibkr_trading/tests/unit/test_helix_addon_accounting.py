from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from backtests.swing.config_helix import HelixBacktestConfig
from backtests.swing.engine.helix_engine import HelixEngine, _ActivePosition
from backtests.swing.engine.sim_broker import OrderSide, OrderType, SimBroker, SimOrder
from strategies.swing.akc_helix.config import SYMBOL_CONFIGS
from strategies.swing.akc_helix.allocator import (
    apply_initial_risk_basis,
    compute_initial_risk_basis,
)
from strategies.swing.akc_helix.models import Direction


def _engine() -> HelixEngine:
    return HelixEngine(
        "QQQ",
        SYMBOL_CONFIGS["QQQ"],
        HelixBacktestConfig(initial_equity=100_000.0),
        point_value=1.0,
    )


def _setup(direction: Direction = Direction.LONG):
    return SimpleNamespace(
        direction=direction,
        unit1_risk_dollars=1_000.0,
        base_unit1_risk_dollars=1_000.0,
        target_initial_risk_dollars=1_000.0,
        actual_initial_risk_dollars=1_000.0,
        risk_utilization=1.0,
        setup_class=SimpleNamespace(value="D"),
        origin_tf="1H",
        qty_planned=100,
        bos_level=100.0,
        stop0=98.0,
        setup_id="setup-test",
        setup_size_mult=1.0,
        adx_at_entry=0.0,
        div_mag_norm=0.0,
        regime_4h_at_entry="",
    )


def test_add_fill_updates_average_cost_basis_and_mtm() -> None:
    engine = _engine()
    pos = _ActivePosition(
        setup=_setup(Direction.LONG),
        fill_price=100.0,
        avg_entry_price=100.0,
        qty_open=100,
    )
    engine.active_position = pos
    engine.equity = 100_000.0

    engine._apply_add_fill_cost_basis(pos, fill_price=110.0, qty=100)

    assert pos.qty_open == 200
    assert pos.avg_entry_price == pytest.approx(105.0)
    assert engine._mtm_equity(112.0) == pytest.approx(101_400.0)


def test_add_quantity_respects_position_cap() -> None:
    engine = _engine()
    pos = _ActivePosition(setup=_setup(), fill_price=100.0, qty_open=480)

    assert engine._cap_add_qty(pos, 100) == 20

    pos.qty_open = 500
    assert engine._cap_add_qty(pos, 100) == 0


def test_flatten_uses_average_cost_basis_and_per_trade_commission() -> None:
    engine = _engine()
    pos = _ActivePosition(
        setup=_setup(Direction.LONG),
        fill_price=100.0,
        avg_entry_price=105.0,
        qty_open=200,
        entry_time=datetime(2024, 1, 2, 15),
        commission=5.0,
    )
    engine.active_position = pos
    engine.equity = 100_000.0

    engine._flatten_position(
        pos,
        exit_price=110.0,
        bar_time=datetime(2024, 1, 3, 15),
        reason="TEST",
        exit_commission=3.0,
    )

    trade = engine.trades[0]
    assert trade.pnl_dollars == pytest.approx(1_000.0)
    assert trade.commission == pytest.approx(8.0)
    assert trade.net_pnl_dollars == pytest.approx(992.0)
    assert trade.net_r_multiple == pytest.approx(0.992)
    assert trade.avg_entry_price == pytest.approx(105.0)
    assert engine.equity == pytest.approx(100_997.0)


def test_initial_risk_basis_records_cap_utilization() -> None:
    setup = _setup(Direction.LONG)
    setup.stop0 = 98.0

    basis = apply_initial_risk_basis(
        setup,
        entry_price=100.0,
        qty=250,
        point_value=1.0,
        target_risk_dollars=800.0,
    )

    assert basis == compute_initial_risk_basis(100.0, 98.0, 250, 1.0, 800.0)
    assert setup.actual_initial_risk_dollars == pytest.approx(500.0)
    assert setup.target_initial_risk_dollars == pytest.approx(800.0)
    assert setup.risk_utilization == pytest.approx(0.625)
    assert setup.unit1_risk_dollars == pytest.approx(500.0)


def test_flatten_r_multiple_uses_actual_initial_risk_after_caps() -> None:
    engine = _engine()
    setup = _setup(Direction.LONG)
    setup.unit1_risk_dollars = 500.0
    setup.target_initial_risk_dollars = 800.0
    setup.actual_initial_risk_dollars = 500.0
    setup.risk_utilization = 0.625
    pos = _ActivePosition(
        setup=setup,
        fill_price=100.0,
        avg_entry_price=100.0,
        qty_open=100,
        entry_time=datetime(2024, 1, 2, 15),
    )

    engine._flatten_position(
        pos,
        exit_price=105.0,
        bar_time=datetime(2024, 1, 3, 15),
        reason="TEST",
        exit_commission=0.0,
    )

    trade = engine.trades[0]
    assert trade.pnl_dollars == pytest.approx(500.0)
    assert trade.r_multiple == pytest.approx(1.0)
    assert trade.net_pnl_dollars == pytest.approx(500.0)
    assert trade.net_r_multiple == pytest.approx(1.0)
    assert trade.target_initial_risk_dollars == pytest.approx(800.0)
    assert trade.actual_initial_risk_dollars == pytest.approx(500.0)
    assert trade.risk_utilization == pytest.approx(0.625)


def test_entry_quantity_caps_to_actual_gap_fill_risk() -> None:
    engine = _engine()
    setup = _setup(Direction.LONG)
    setup.stop0 = 100.0
    setup.target_initial_risk_dollars = 1_000.0

    assert engine._cap_entry_qty_to_initial_risk(setup, fill_price=130.0, qty=100) == 33
    assert engine._cap_entry_qty_to_initial_risk(setup, fill_price=2_000.0, qty=100) == 0


def test_end_of_data_flatten_uses_public_broker_market_fill() -> None:
    engine = _engine()
    pos = _ActivePosition(
        setup=_setup(Direction.LONG),
        fill_price=100.0,
        avg_entry_price=100.0,
        qty_open=100,
        entry_time=datetime(2024, 1, 2, 15),
        commission=5.0,
    )
    engine.active_position = pos
    engine.equity = 100_000.0
    calls: list[tuple[SimOrder, float]] = []
    original_fill_market_order = engine.broker.fill_market_order

    def _spy_fill_market_order(order, bar_time, price, tick_size, extra_slip=False):
        calls.append((order, price))
        return original_fill_market_order(order, bar_time, price, tick_size, extra_slip)

    engine.broker.fill_market_order = _spy_fill_market_order  # type: ignore[method-assign]

    engine._flatten_at_end_of_data(110.0, datetime(2024, 1, 3, 15))

    assert len(calls) == 1
    order, price = calls[0]
    assert price == pytest.approx(110.0)
    assert order.order_type is OrderType.MARKET
    assert order.side is OrderSide.SELL
    trade = engine.trades[0]
    assert trade.exit_reason == "END_OF_DATA"
    assert trade.exit_price <= 110.0
    assert trade.commission > 5.0
    assert trade.net_pnl_dollars == pytest.approx(trade.pnl_dollars - trade.commission)


def test_broker_does_not_retro_fill_orders_submitted_after_bar_processing() -> None:
    broker = SimBroker()
    bar_time = datetime(2024, 1, 2, 15)

    assert broker.process_bar("QQQ", bar_time, 100.0, 110.0, 99.0, 109.0, 0.01) == []

    order = SimOrder(
        order_id="ENTRY-1",
        symbol="QQQ",
        side=OrderSide.BUY,
        order_type=OrderType.STOP_LIMIT,
        qty=1,
        stop_price=105.0,
        limit_price=106.0,
        tick_size=0.01,
        submit_time=bar_time,
        tag="entry",
    )
    broker.submit_order(order)

    assert len(broker.pending_orders) == 1
    fills = broker.process_bar("QQQ", datetime(2024, 1, 2, 16), 105.0, 106.0, 104.0, 105.5, 0.01)
    assert len(fills) == 1
    assert fills[0].order.order_id == "ENTRY-1"
