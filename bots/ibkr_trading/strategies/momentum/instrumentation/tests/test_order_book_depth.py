from strategies.momentum.instrumentation.src.trade_logger import TradeEvent


def test_trade_event_has_order_book_depth_field():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={})
    assert te.order_book_depth_at_entry is None


def test_order_book_depth_structure():
    depth = {
        "bid_volume_5_levels": [100, 85, 72, 60, 55],
        "ask_volume_5_levels": [90, 80, 70, 65, 50],
        "bid_total_5": 372,
        "ask_total_5": 355,
        "imbalance_ratio": 1.048,
    }
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={},
                    order_book_depth_at_entry=depth)
    d = te.to_dict()
    assert d["order_book_depth_at_entry"]["imbalance_ratio"] == 1.048
