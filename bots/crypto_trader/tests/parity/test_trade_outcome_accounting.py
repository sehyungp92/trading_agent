"""TradeOutcome and equity reconciliation parity tests."""

from datetime import datetime, timezone

import pytest

from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.core.models import Bar, Fill, Order, OrderType, Side, TerminalMark, TimeFrame, Trade
from crypto_trader.core.runtime_types import TradeOutcome
from crypto_trader.live.lifecycle import PositionLifecycleLedger
from crypto_trader.parity.accounting import reconcile_equity


def test_reconcile_equity_uses_net_trades_and_terminal_marks() -> None:
    trade = Trade(
        trade_id="t1",
        symbol="BTC",
        direction=Side.LONG,
        entry_price=100.0,
        exit_price=110.0,
        qty=1.0,
        entry_time=datetime(2026, 5, 24, 10, tzinfo=timezone.utc),
        exit_time=datetime(2026, 5, 24, 11, tzinfo=timezone.utc),
        pnl=8.0,
        r_multiple=None,
        commission=0.5,
        bars_held=4,
        setup_grade=None,
        exit_reason="tp",
        confluences_used=None,
        confirmation_type=None,
        entry_method=None,
        funding_paid=2.0,
        mae_r=None,
        mfe_r=None,
    )
    mark = TerminalMark(
        symbol="ETH",
        direction=Side.SHORT,
        qty=1.0,
        timestamp=datetime(2026, 5, 24, 12, tzinfo=timezone.utc),
        entry_price=100.0,
        mark_price_raw=95.0,
        mark_price_net_liquidation=95.0,
        unrealized_pnl_net=5.0,
        unrealized_r_at_mark=None,
    )

    result = reconcile_equity(
        initial_equity=10_000.0,
        final_equity=10_012.5,
        trades=[trade],
        terminal_marks=[mark],
    )

    assert result.passed is True
    assert result.realized_net == pytest.approx(7.5)
    assert result.terminal_unrealized_net == pytest.approx(5.0)


def _bar(price: float, minute: int) -> Bar:
    return Bar(
        timestamp=datetime(2026, 5, 24, 12, minute, tzinfo=timezone.utc),
        symbol="BTC",
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1.0,
        timeframe=TimeFrame.M15,
    )


def _fill(side: Side, qty: float, price: float, tag: str, minute: int) -> Fill:
    return Fill(
        order_id=f"{tag}_{minute}",
        symbol="BTC",
        side=side,
        qty=qty,
        fill_price=price,
        commission=0.0,
        timestamp=datetime(2026, 5, 24, 12, minute, tzinfo=timezone.utc),
        tag=tag,
    )


def _submit_market(broker: SimBroker, side: Side, qty: float, price: float, minute: int, tag: str) -> None:
    broker.submit_order(
        Order(
            order_id="",
            symbol="BTC",
            side=side,
            order_type=OrderType.MARKET,
            qty=qty,
            tag=tag,
        )
    )
    broker.process_bar(_bar(price, minute))


def test_sim_and_live_partial_exit_economics_share_contract() -> None:
    broker = SimBroker(
        initial_equity=10_000.0,
        taker_fee_bps=0.0,
        maker_fee_bps=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
    )
    broker.process_bar(_bar(100.0, 0))
    _submit_market(broker, Side.LONG, 1.0, 100.0, 15, "entry")
    _submit_market(broker, Side.SHORT, 0.4, 110.0, 30, "tp1")
    _submit_market(broker, Side.SHORT, 0.6, 105.0, 45, "stop")

    ledger = PositionLifecycleLedger()
    assert ledger.apply_fill("momentum", _fill(Side.LONG, 1.0, 100.0, "entry", 15)) is None
    assert ledger.apply_fill("momentum", _fill(Side.SHORT, 0.4, 110.0, "tp1", 30)) is None
    live_trade = ledger.apply_fill("momentum", _fill(Side.SHORT, 0.6, 105.0, "stop", 45))

    assert live_trade is not None
    sim_trade = broker._closed_trades[0]
    sim_outcome = TradeOutcome.from_trade(sim_trade)
    live_outcome = TradeOutcome.from_trade(live_trade)

    assert sim_trade.qty == pytest.approx(live_trade.qty)
    assert sim_trade.exit_price == pytest.approx(live_trade.exit_price)
    assert sim_outcome.price_pnl_gross == pytest.approx(live_outcome.price_pnl_gross)
    assert sim_outcome.total_fees == pytest.approx(live_outcome.total_fees)
    assert sim_outcome.funding_paid == pytest.approx(live_outcome.funding_paid)
    assert sim_outcome.realized_pnl_net == pytest.approx(live_outcome.realized_pnl_net)
