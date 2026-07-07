from strategies.momentum.instrumentation.src.trade_logger import TradeEvent
from strategies.momentum.instrumentation.src.missed_opportunity import MissedOpportunityEvent


def test_trade_event_has_market_conditions():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={})
    assert te.market_conditions_at_entry is None


def test_missed_event_has_market_conditions():
    me = MissedOpportunityEvent(
        event_metadata={}, market_snapshot={},
        pair="NQ", side="LONG", signal="Class_M", signal_id="s1",
        signal_strength=0.5, blocked_by="heat_cap", block_reason="heat > cap")
    assert me.market_conditions_at_entry is None


def test_market_conditions_structure():
    conditions = {
        "vix_level": 18.5,
        "vix_term_structure": "contango",
        "market_breadth_adv_dec": 1.8,
        "put_call_ratio": 0.85,
    }
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={},
                    market_conditions_at_entry=conditions)
    d = te.to_dict()
    assert d["market_conditions_at_entry"]["vix_level"] == 18.5
