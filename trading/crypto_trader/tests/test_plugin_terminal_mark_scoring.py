from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.core.models import Side, TerminalMark
from crypto_trader.optimize.types import Experiment
from crypto_trader.optimize.types import GreedyResult
from crypto_trader.strategy.breakout.config import BreakoutConfig
from crypto_trader.strategy.momentum.config import MomentumConfig
from crypto_trader.strategy.trend.config import TrendConfig


@pytest.mark.parametrize(
    ("module_path", "plugin_cls_name", "config_factory"),
    [
        ("crypto_trader.optimize.momentum_plugin", "MomentumPlugin", MomentumConfig),
        ("crypto_trader.optimize.trend_plugin", "TrendPlugin", TrendConfig),
        ("crypto_trader.optimize.breakout_plugin", "BreakoutPlugin", BreakoutConfig),
    ],
)
def test_plugins_do_not_pass_backtest_end_filters_to_parallel_eval(
    module_path: str,
    plugin_cls_name: str,
    config_factory,
):
    module = __import__(module_path, fromlist=[plugin_cls_name])
    plugin_cls = getattr(module, plugin_cls_name)
    plugin = plugin_cls(
        backtest_config=BacktestConfig(
            start_date=date(2026, 3, 1),
            end_date=date(2026, 4, 1),
            symbols=["BTC"],
            initial_equity=10_000.0,
        ),
        base_config=config_factory(),
        data_dir=Path("data"),
    )

    with patch(f"{module_path}.evaluate_parallel", return_value=[]) as mock_eval:
        evaluate_fn = plugin.create_evaluate_batch(
            phase=1,
            cumulative_mutations={},
            scoring_weights={"returns": 1.0},
            hard_rejects={},
        )
        result = evaluate_fn([Experiment("noop", {})], {})

    assert result == []
    assert "exclude_exit_reasons" not in mock_eval.call_args.kwargs


@pytest.mark.parametrize(
    ("module_path", "plugin_cls_name", "config_factory"),
    [
        ("crypto_trader.optimize.momentum_plugin", "MomentumPlugin", MomentumConfig),
        ("crypto_trader.optimize.trend_plugin", "TrendPlugin", TrendConfig),
        ("crypto_trader.optimize.breakout_plugin", "BreakoutPlugin", BreakoutConfig),
    ],
)
def test_phase_diagnostics_handle_terminal_marks_without_realized_trades(
    module_path: str,
    plugin_cls_name: str,
    config_factory,
):
    module = __import__(module_path, fromlist=[plugin_cls_name])
    plugin_cls = getattr(module, plugin_cls_name)
    plugin = plugin_cls(
        backtest_config=BacktestConfig(
            start_date=date(2026, 3, 1),
            end_date=date(2026, 4, 1),
            symbols=["BTC"],
            initial_equity=10_000.0,
        ),
        base_config=config_factory(),
        data_dir=Path("data"),
    )
    plugin._last_result = SimpleNamespace(
        trades=[],
        terminal_marks=[
            TerminalMark(
                symbol="BTC",
                direction=Side.LONG,
                qty=1.0,
                timestamp=datetime(2026, 3, 31, tzinfo=timezone.utc),
                entry_price=100.0,
                mark_price_raw=105.0,
                mark_price_net_liquidation=104.8,
                unrealized_pnl_net=4.8,
                unrealized_r_at_mark=0.48,
            )
        ],
    )

    greedy = GreedyResult(
        accepted_experiments=[],
        rejected_experiments=[],
        final_mutations={},
        final_score=0.0,
        base_score=0.0,
        accepted_count=0,
        rounds=[],
    )
    text = plugin.run_phase_diagnostics(
        phase=1,
        state=None,
        metrics={"terminal_mark_count": 1.0},
        greedy_result=greedy,
    )

    assert "terminal mark" in text.lower()
    assert "net liq" in text.lower() or "net liquidation" in text.lower()
