"""Tests for live lifecycle ledger economics."""

from datetime import datetime, timezone

import pytest

from crypto_trader.core.models import Fill, Side
from crypto_trader.core.runtime_types import TradeOutcome
from crypto_trader.live.lifecycle import PositionLifecycleLedger


def _fill(side: Side, qty: float, price: float, tag: str, minute: int) -> Fill:
    return Fill(
        order_id=f"{tag}_{minute}",
        symbol="BTC",
        side=side,
        qty=qty,
        fill_price=price,
        commission=0.1,
        timestamp=datetime(2026, 5, 24, 12, minute, tzinfo=timezone.utc),
        tag=tag,
    )


def test_lifecycle_ledger_accumulates_partial_exits_before_trade() -> None:
    ledger = PositionLifecycleLedger()

    assert ledger.apply_fill("momentum", _fill(Side.LONG, 1.0, 100.0, "entry", 0)) is None
    assert ledger.apply_fill("momentum", _fill(Side.SHORT, 0.4, 110.0, "tp1", 15)) is None
    trade = ledger.apply_fill("momentum", _fill(Side.SHORT, 0.6, 105.0, "stop", 30))

    assert trade is not None
    assert trade.symbol == "BTC"
    assert trade.direction == Side.LONG
    assert trade.qty == pytest.approx(1.0)
    assert trade.exit_price == pytest.approx(107.0)
    assert trade.pnl == pytest.approx(7.0)
    assert trade.commission == pytest.approx(0.3)
    assert trade.net_pnl == pytest.approx(6.7)
    outcome = TradeOutcome.from_trade(trade)
    assert outcome.price_pnl_gross == pytest.approx(7.0)
    assert outcome.total_fees == pytest.approx(0.3)
    assert outcome.funding_paid == pytest.approx(0.0)
    assert outcome.realized_pnl_net == pytest.approx(6.7)
    assert ledger.open_positions() == []
