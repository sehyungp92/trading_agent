import pytest
from unittest.mock import MagicMock
from strategies.stock.instrumentation.src.facade import InstrumentationKit


@pytest.fixture
def mock_manager():
    mgr = MagicMock()
    mgr.trade_logger = MagicMock()
    mgr.trade_logger._open_trades = {}
    mgr.missed_logger = MagicMock()
    mgr.regime_classifier = MagicMock()
    mgr.regime_classifier.current_regime.return_value = "trending_up"
    return mgr


@pytest.fixture
def kit(mock_manager):
    return InstrumentationKit(mock_manager, strategy_type="helix")


def test_log_entry_delegates_to_trade_logger(kit, mock_manager):
    kit.log_entry(
        trade_id="t1",
        pair="NQ",
        side="LONG",
        entry_price=21000.0,
        position_size=5,
        position_size_quote=2100000.0,
        entry_signal="Class_M",
        entry_signal_id="setup_001",
        entry_signal_strength=0.667,
        expected_entry_price=20999.0,
        strategy_params={"stop0": 20950.0},
        signal_factors=[{"factor_name": "alignment", "factor_value": 2, "threshold": 1, "contribution": 0.667}],
        filter_decisions=[{"filter_name": "heat_cap", "threshold": 3.0, "actual_value": 2.1, "passed": True, "margin_pct": 30.0}],
        sizing_inputs={"unit_risk_usd": 500, "dd_mult": 1.0},
        session_type="RTH_PRIME1",
        contract_month="2026-03",
        concurrent_positions=2,
        drawdown_pct=0.05,
        drawdown_tier="full",
        drawdown_size_mult=1.0,
    )
    assert mock_manager.trade_logger.log_entry.called
    kwargs = mock_manager.trade_logger.log_entry.call_args[1]
    assert kwargs["active_filters"] == ["heat_cap"]
    assert kwargs["passed_filters"] == ["heat_cap"]
    assert kwargs["market_regime"] == "trending_up"
    assert kwargs["signal_factors"] == [{"factor_name": "alignment", "factor_value": 2, "threshold": 1, "contribution": 0.667}]
    assert kwargs["contract_month"] == "2026-03"


def test_log_entry_populates_active_and_passed_filters(kit, mock_manager):
    kit.log_entry(
        trade_id="t2",
        pair="NQ",
        side="SHORT",
        entry_price=21000.0,
        position_size=3,
        position_size_quote=1260000.0,
        entry_signal="Class_M",
        entry_signal_id="s2",
        entry_signal_strength=0.5,
        strategy_params={},
        filter_decisions=[
            {"filter_name": "heat_cap", "threshold": 3.0, "actual_value": 2.1, "passed": True},
            {"filter_name": "spread", "threshold": 0.5, "actual_value": 0.8, "passed": False},
        ],
    )
    kwargs = mock_manager.trade_logger.log_entry.call_args[1]
    assert kwargs["active_filters"] == ["heat_cap", "spread"]
    assert kwargs["passed_filters"] == ["heat_cap"]  # spread failed


def test_log_exit_delegates(kit, mock_manager):
    kit.log_exit(trade_id="t1", exit_price=21050.0, exit_reason="TRAILING_STOP")
    assert mock_manager.trade_logger.log_exit.called
    kwargs = mock_manager.trade_logger.log_exit.call_args[1]
    assert kwargs["trade_id"] == "t1"
    assert kwargs["exit_price"] == 21050.0
    assert kwargs["exit_reason"] == "TRAILING_STOP"


def test_log_entry_and_exit_forward_runtime_refs(kit, mock_manager):
    kit.log_entry(
        trade_id="t_runtime",
        pair="QQQ",
        side="LONG",
        entry_price=500.0,
        position_size=2,
        position_size_quote=1000.0,
        entry_signal="setup",
        entry_signal_id="decision_1",
        entry_signal_strength=0.9,
        strategy_params={},
        fill_order_id="oms-entry",
        fill_id="exec-entry",
        fill_qty=2,
        intent_id="intent_1",
        portfolio_decision_ref="portfolio_rule_1",
    )
    entry_kwargs = mock_manager.trade_logger.log_entry.call_args[1]
    assert entry_kwargs["fill_order_id"] == "oms-entry"
    assert entry_kwargs["fill_id"] == "exec-entry"
    assert entry_kwargs["fill_qty"] == 2
    assert entry_kwargs["intent_id"] == "intent_1"
    assert entry_kwargs["portfolio_decision_ref"] == "portfolio_rule_1"

    kit.log_exit(
        trade_id="t_runtime",
        exit_price=502.0,
        exit_reason="TARGET",
        fill_order_id="oms-exit",
        exit_fill_id="exec-exit",
        fill_qty=2,
    )
    exit_kwargs = mock_manager.trade_logger.log_exit.call_args[1]
    assert exit_kwargs["fill_order_id"] == "oms-exit"
    assert exit_kwargs["exit_fill_id"] == "exec-exit"
    assert exit_kwargs["fill_qty"] == 2


def test_log_missed_delegates_with_enriched_params(kit, mock_manager):
    kit.log_missed(
        pair="NQ",
        side="LONG",
        signal="Class_M",
        signal_id="setup_001",
        signal_strength=0.667,
        blocked_by="heat_cap",
        block_reason="heat 3.2 > cap 3.0",
        strategy_params={"score": 2},
        filter_decisions=[{"filter_name": "heat_cap", "passed": False}],
        session_type="RTH_PRIME1",
        concurrent_positions=3,
        drawdown_pct=0.05,
        drawdown_tier="full",
    )
    assert mock_manager.missed_logger.log_missed.called
    kwargs = mock_manager.missed_logger.log_missed.call_args[1]
    assert kwargs["strategy_params"]["_concurrent_positions"] == 3
    assert kwargs["strategy_params"]["_session_type"] == "RTH_PRIME1"
    assert kwargs["strategy_params"]["_drawdown_pct"] == 0.05
    assert kwargs["strategy_params"]["_filter_decisions"] == [{"filter_name": "heat_cap", "passed": False}]
    assert kwargs["filter_decisions"] == [{"filter_name": "heat_cap", "passed": False}]
    assert kwargs["concurrent_positions"] == 3
    assert kwargs["session_type"] == "RTH_PRIME1"
    assert kwargs["drawdown_pct"] == 0.05
    assert kwargs["drawdown_tier"] == "full"


def test_kit_graceful_on_none_manager():
    kit = InstrumentationKit(None, strategy_type="helix")
    # Should not raise
    kit.log_entry(trade_id="t1", pair="NQ", side="LONG", entry_price=21000.0,
                  position_size=1, position_size_quote=21000.0,
                  entry_signal="test", entry_signal_id="s1", entry_signal_strength=0.5,
                  expected_entry_price=21000.0, strategy_params={})
    kit.log_exit(trade_id="t1", exit_price=21050.0, exit_reason="STOP")
    kit.log_missed(pair="NQ", side="LONG", signal="test", signal_id="s1",
                   signal_strength=0.5, blocked_by="test", block_reason="test",
                   strategy_params={})


def test_kit_active_property():
    kit_active = InstrumentationKit(MagicMock(), strategy_type="helix")
    assert kit_active.active is True
    kit_none = InstrumentationKit(None)
    assert kit_none.active is False
