from __future__ import annotations

import numpy as np

from backtests.swing.auto.atrss.scoring import ATRSSMetrics, composite_score
from backtests.swing.engine.backtest_engine import BacktestEngine


def test_atrss_datetime64_item_keeps_historical_time() -> None:
    ts = np.datetime64("2026-04-22T13:00:00.000000000")

    assert BacktestEngine._to_datetime(ts).isoformat() == "2026-04-22T13:00:00+00:00"
    assert BacktestEngine._to_datetime(ts.item()).isoformat() == "2026-04-22T13:00:00+00:00"


def test_atrss_composite_score_has_seven_active_components() -> None:
    score = composite_score(
        ATRSSMetrics(
            total_trades=300,
            win_rate=0.80,
            profit_factor=5.0,
            max_dd_pct=0.02,
            calmar_r=50.0,
            net_return_pct=50.0,
            total_r=250.0,
            avg_r=0.80,
            mfe_capture=0.70,
            trades_per_month=6.0,
        ),
        hard_rejects={"min_trades": 1, "max_dd_pct": 1.0, "min_pf": 0.0, "min_wr": 0.0},
        profile="r9_synchronized",
    )

    active_components = [
        name for name in score.__dataclass_fields__
        if name.endswith("_component")
    ]

    assert active_components == [
        "return_component",
        "pf_component",
        "risk_component",
        "frequency_component",
        "mfe_capture_component",
        "avg_r_component",
        "win_rate_component",
    ]
    assert 0.0 < score.total <= 1.0
