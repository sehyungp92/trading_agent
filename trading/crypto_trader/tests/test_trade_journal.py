from __future__ import annotations

from datetime import datetime, timezone

from crypto_trader.core.models import SetupGrade, Side, Trade
from crypto_trader.strategy.momentum.journal import TradeJournal


def test_trade_journal_records_economic_r_and_net_pnl():
    journal = TradeJournal()
    trade = Trade(
        trade_id="t1",
        symbol="BTC",
        direction=Side.LONG,
        entry_price=100.0,
        exit_price=101.0,
        qty=1.0,
        entry_time=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        exit_time=datetime(2026, 3, 1, 13, 0, tzinfo=timezone.utc),
        pnl=12.0,
        r_multiple=1.0,
        commission=2.0,
        bars_held=3,
        setup_grade=SetupGrade.A,
        exit_reason="tp1",
        confluences_used=["m15_ema20"],
        confirmation_type="inside_bar_break",
        entry_method="close",
        funding_paid=0.0,
        mae_r=-0.3,
        mfe_r=1.5,
        realized_r_multiple=0.4,
    )

    entry = journal.record(trade, {})

    assert entry.final_r == 0.4
    assert entry.pnl_usd == 10.0
