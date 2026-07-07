from strategies.momentum.instrumentation.src.trade_logger import TradeEvent


def test_trade_event_has_fill_details_field():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={})
    assert te.entry_fill_details is None
    assert te.exit_fill_details is None


def test_fill_details_structure():
    fill = {
        "num_fills": 3,
        "fill_prices": [21000.0, 21000.25, 21000.50],
        "fill_sizes": [2, 2, 1],
        "first_fill_to_last_fill_ms": 1200,
        "vwap_fill_price": 21000.15,
    }
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={},
                    entry_fill_details=fill)
    d = te.to_dict()
    assert d["entry_fill_details"]["num_fills"] == 3
