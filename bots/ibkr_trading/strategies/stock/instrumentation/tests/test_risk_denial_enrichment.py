from unittest.mock import MagicMock, patch
from strategies.stock.instrumentation.src.bootstrap import InstrumentationManager


def test_handle_risk_denial_uses_enriched_payload():
    """RISK_DENIAL with enriched payload should use symbol/side/signal from payload."""
    oms = MagicMock()
    mgr = InstrumentationManager(oms, "test_strat", "helix")
    # Replace the missed_logger with a mock to capture calls
    mgr.missed_logger = MagicMock()

    event = MagicMock()
    event.payload = {
        "reason": "Heat cap breach: 3.2R > 3.0R",
        "symbol": "NQ",
        "side": "LONG",
        "signal_name": "Class_M",
        "signal_strength": 0.667,
        "strategy_id": "helix",
    }
    event.oms_order_id = "ord_123"
    event.timestamp = None

    mgr._handle_risk_denial(event)

    assert mgr.missed_logger.log_missed.called
    kwargs = mgr.missed_logger.log_missed.call_args[1]
    assert kwargs["pair"] == "NQ"
    assert kwargs["side"] == "LONG"
    assert kwargs["signal"] == "Class_M"
    assert kwargs["signal_strength"] == 0.667
    assert kwargs["blocked_by"] == "risk_gateway"
    assert "Heat cap" in kwargs["block_reason"]


def test_handle_risk_denial_fallback_without_enrichment():
    """RISK_DENIAL without enriched payload should use defaults."""
    oms = MagicMock()
    mgr = InstrumentationManager(oms, "test_strat", "strategy_alcb")
    mgr.missed_logger = MagicMock()

    event = MagicMock()
    event.payload = {"reason": "Global stand-down active"}
    event.oms_order_id = "ord_456"
    event.timestamp = None

    mgr._handle_risk_denial(event)

    kwargs = mgr.missed_logger.log_missed.call_args[1]
    assert kwargs["pair"] == "SPY"  # falls back to stock_trader proxy defaults
    assert kwargs["side"] == "UNKNOWN"  # no enrichment
    assert kwargs["signal"] == "risk_denial_stock_trader"  # generated name
    assert kwargs["signal_strength"] == 0.0
