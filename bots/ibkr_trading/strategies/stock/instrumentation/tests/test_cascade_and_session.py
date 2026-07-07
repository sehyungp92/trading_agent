"""Tests for execution cascade timestamps (#16) and session transitions (#17)."""
import pytest
from unittest.mock import MagicMock
from strategies.stock.instrumentation.src.trade_logger import TradeEvent
from strategies.stock.instrumentation.src.facade import InstrumentationKit


class TestTradeEventSchemaAdditions:
    def test_execution_timestamps_default_none(self):
        t = TradeEvent(trade_id="x", event_metadata={}, entry_snapshot={})
        assert t.execution_timestamps is None

    def test_session_transitions_default_none(self):
        t = TradeEvent(trade_id="x", event_metadata={}, entry_snapshot={})
        assert t.session_transitions is None

    def test_to_dict_includes_new_fields(self):
        t = TradeEvent(trade_id="x", event_metadata={}, entry_snapshot={})
        d = t.to_dict()
        assert "execution_timestamps" in d
        assert "session_transitions" in d
        assert d["execution_timestamps"] is None
        assert d["session_transitions"] is None

    def test_execution_timestamps_with_data(self):
        ts = {
            "signal_detected_at": "2026-03-01T10:00:00Z",
            "fill_received_at": "2026-03-01T10:00:01Z",
            "cascade_duration_ms": 1000,
        }
        t = TradeEvent(trade_id="x", event_metadata={}, entry_snapshot={},
                       execution_timestamps=ts)
        assert t.execution_timestamps["cascade_duration_ms"] == 1000

    def test_session_transitions_with_data(self):
        transitions = [
            {
                "from_session": "ETH_PRIME",
                "to_session": "RTH_PRIME1",
                "transition_time": "2026-03-01T09:30:00Z",
                "unrealized_pnl_r": 0.5,
                "bars_held": 3,
                "price_at_transition": 21050.0,
            },
        ]
        t = TradeEvent(trade_id="x", event_metadata={}, entry_snapshot={},
                       session_transitions=transitions)
        d = t.to_dict()
        assert len(d["session_transitions"]) == 1
        assert d["session_transitions"][0]["from_session"] == "ETH_PRIME"


class TestFacadeExecutionTimestamps:
    @pytest.fixture
    def mock_manager(self):
        mgr = MagicMock()
        mgr.trade_logger = MagicMock()
        mgr.trade_logger._open_trades = {}
        mgr.missed_logger = MagicMock()
        mgr.regime_classifier = MagicMock()
        mgr.regime_classifier.current_regime.return_value = "trending_up"
        return mgr

    def test_log_entry_passes_execution_timestamps(self, mock_manager):
        kit = InstrumentationKit(mock_manager, strategy_type="helix")
        ts = {"signal_detected_at": "2026-03-01T10:00:00Z"}

        kit.log_entry(
            trade_id="t1",
            pair="NQ",
            side="LONG",
            entry_price=21000.0,
            position_size=5,
            position_size_quote=2100000.0,
            entry_signal="Class_M",
            entry_signal_id="s1",
            entry_signal_strength=0.667,
            strategy_params={},
            execution_timestamps=ts,
        )
        kwargs = mock_manager.trade_logger.log_entry.call_args[1]
        assert kwargs["execution_timestamps"] == ts

    def test_log_exit_passes_session_transitions(self, mock_manager):
        kit = InstrumentationKit(mock_manager, strategy_type="helix")
        transitions = [{"from_session": "ETH", "to_session": "RTH"}]

        kit.log_exit(
            trade_id="t1",
            exit_price=21050.0,
            exit_reason="TRAILING_STOP",
            session_transitions=transitions,
        )
        kwargs = mock_manager.trade_logger.log_exit.call_args[1]
        assert kwargs["session_transitions"] == transitions

    def test_log_exit_without_session_transitions(self, mock_manager):
        kit = InstrumentationKit(mock_manager, strategy_type="helix")
        kit.log_exit(trade_id="t1", exit_price=21050.0, exit_reason="STOP")
        kwargs = mock_manager.trade_logger.log_exit.call_args[1]
        assert kwargs["session_transitions"] is None


class TestFacadeGracefulDegradation:
    def test_none_manager_with_execution_timestamps(self):
        kit = InstrumentationKit(None, strategy_type="helix")
        # Should not raise
        kit.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=21000.0, position_size=1,
            position_size_quote=21000.0,
            entry_signal="test", entry_signal_id="s1",
            entry_signal_strength=0.5,
            strategy_params={},
            execution_timestamps={"signal_detected_at": "2026-03-01T10:00:00Z"},
        )

    def test_none_manager_with_session_transitions(self):
        kit = InstrumentationKit(None, strategy_type="helix")
        # Should not raise
        kit.log_exit(
            trade_id="t1", exit_price=21050.0, exit_reason="STOP",
            session_transitions=[{"from_session": "ETH", "to_session": "RTH"}],
        )
