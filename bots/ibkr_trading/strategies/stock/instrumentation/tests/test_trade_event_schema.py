"""Tests for MFE/MAE and exit_efficiency fields in TradeEvent schema."""
from strategies.stock.instrumentation.src.trade_logger import TradeEvent


def test_trade_event_has_mfe_mae_fields():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={})
    assert te.mfe_r is None
    assert te.mae_r is None
    assert te.mfe_price is None
    assert te.mae_price is None


def test_trade_event_has_exit_efficiency():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={},
                    mfe_r=2.0, pnl_pct=1.5, entry_price=21000.0)
    assert te.exit_efficiency is None  # computed at exit, not set at init


def test_trade_event_mfe_mae_in_dict():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={},
                    mfe_r=1.5, mae_r=0.3)
    d = te.to_dict()
    assert d["mfe_r"] == 1.5
    assert d["mae_r"] == 0.3


def test_trade_event_exit_efficiency_in_dict():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={},
                    exit_efficiency=0.75)
    d = te.to_dict()
    assert d["exit_efficiency"] == 0.75


def test_trade_event_mfe_mae_defaults_none_in_dict():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={})
    d = te.to_dict()
    assert d["mfe_r"] is None
    assert d["mae_r"] is None
    assert d["mfe_price"] is None
    assert d["mae_price"] is None
    assert d["exit_efficiency"] is None


def test_trade_event_has_strategy_type_field():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={})
    assert te.strategy_type == ""
    te2 = TradeEvent(trade_id="t2", event_metadata={}, entry_snapshot={}, strategy_type="helix")
    assert te2.strategy_type == "helix"


def test_trade_event_has_param_set_id_field():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={})
    assert te.param_set_id is None
    te2 = TradeEvent(trade_id="t2", event_metadata={}, entry_snapshot={}, param_set_id="abc123")
    assert te2.param_set_id == "abc123"


def test_trade_event_strategy_type_in_dict():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={}, strategy_type="nqdtc")
    d = te.to_dict()
    assert d["strategy_type"] == "nqdtc"


def test_trade_event_param_set_id_in_dict():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={}, param_set_id="deadbeef01234567")
    d = te.to_dict()
    assert d["param_set_id"] == "deadbeef01234567"
