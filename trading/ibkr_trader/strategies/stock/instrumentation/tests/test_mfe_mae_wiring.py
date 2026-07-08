"""Tests for MFE/MAE wiring from engines through facade to TradeEvent."""
from unittest.mock import MagicMock
from strategies.stock.instrumentation.src.facade import InstrumentationKit


def test_log_exit_passes_mfe_mae_to_trade_logger():
    mgr = MagicMock()
    mgr.trade_logger = MagicMock()
    mgr.trade_logger._open_trades = {}
    kit = InstrumentationKit(mgr, strategy_type="helix")

    kit.log_exit(
        trade_id="t1",
        exit_price=21050.0,
        exit_reason="TRAILING_STOP",
        mfe_r=2.1,
        mae_r=0.4,
        mfe_price=21100.0,
        mae_price=20980.0,
    )

    kwargs = mgr.trade_logger.log_exit.call_args[1]
    assert kwargs["mfe_r"] == 2.1
    assert kwargs["mae_r"] == 0.4
    assert kwargs["mfe_price"] == 21100.0
    assert kwargs["mae_price"] == 20980.0


def test_log_exit_passes_none_when_mfe_mae_not_provided():
    mgr = MagicMock()
    mgr.trade_logger = MagicMock()
    mgr.trade_logger._open_trades = {}
    kit = InstrumentationKit(mgr, strategy_type="helix")

    kit.log_exit(
        trade_id="t2",
        exit_price=21050.0,
        exit_reason="STALE",
    )

    kwargs = mgr.trade_logger.log_exit.call_args[1]
    assert kwargs["mfe_r"] is None
    assert kwargs["mae_r"] is None
    assert kwargs["mfe_price"] is None
    assert kwargs["mae_price"] is None


def test_log_exit_inactive_kit_does_not_raise():
    kit = InstrumentationKit(None, strategy_type="helix")
    # Should not raise
    kit.log_exit(
        trade_id="t3",
        exit_price=21050.0,
        exit_reason="STOP",
        mfe_r=1.0,
        mae_r=0.5,
    )


def test_facade_log_exit_exception_is_caught():
    mgr = MagicMock()
    mgr.trade_logger.log_exit.side_effect = Exception("boom")
    kit = InstrumentationKit(mgr, strategy_type="helix")

    # Should not propagate
    kit.log_exit(
        trade_id="t4",
        exit_price=21050.0,
        exit_reason="FLATTEN",
        mfe_r=1.5,
        mae_r=0.3,
    )
