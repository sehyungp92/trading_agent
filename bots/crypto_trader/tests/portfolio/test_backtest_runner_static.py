"""Portfolio backtest wiring regressions."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_backtest_trade_close_callback_passes_trade_context_to_coordinator():
    source = (ROOT / "src/crypto_trader/portfolio/backtest_runner.py").read_text(encoding="utf-8")

    assert "coordinator.on_trade_closed(strategy_id, trade.symbol, pnl_R, trade=trade)" in source
