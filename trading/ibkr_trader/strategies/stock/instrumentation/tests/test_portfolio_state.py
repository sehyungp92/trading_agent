from strategies.stock.instrumentation.src.trade_logger import TradeEvent


def test_trade_event_has_portfolio_state_field():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={})
    assert te.portfolio_state_at_entry is None


def test_portfolio_state_captured_in_dict():
    state = {
        "total_exposure_r": 3.5,
        "daily_realized_pnl": 1200.0,
        "weekly_realized_pnl": 5600.0,
        "open_risk_r": 2.0,
        "portfolio_heat_r": 5.5,
    }
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={},
                    portfolio_state_at_entry=state)
    d = te.to_dict()
    assert d["portfolio_state_at_entry"]["total_exposure_r"] == 3.5
