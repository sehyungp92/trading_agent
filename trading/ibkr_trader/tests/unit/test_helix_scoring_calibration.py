from __future__ import annotations

import pytest

from backtests.swing.auto.helix.plugin import HelixPlugin, PHASE_HARD_REJECTS
from backtests.swing.auto.helix.scoring import HelixMetrics, composite_score


def _round1_seed_metrics() -> dict[str, float]:
    return {
        "total_trades": 384,
        "profit_factor": 1.2231767730244474,
        "net_return_pct": 28.900379999997423,
        "max_r_dd": 16.026263932467984,
        "exit_efficiency": 0.10219564225623623,
        "waste_ratio": 0.5489347229011103,
        "tail_pct": 0.7159399505185957,
        "bull_pf": 1.4187924592914267,
        "bear_pf": 0.7739852431606772,
        "min_regime_pf": 0.7739852431606772,
        "long_pf": 1.4187924592914267,
        "short_pf": 0.7739852431606772,
        "min_side_pf": 0.7739852431606772,
        "total_r": 45.89510754652965,
        "gross_win_r": 244.65353783440406,
        "gross_loss_r": 198.7584302878744,
        "stale_r": 15.68405503355316,
        "short_hold_r": 94.67066080294597,
        "big_winner_r": 175.15724177136264,
        "sharpe": 0.39158110059062173,
        "calmar_r": 2.863743398955865,
        "win_rate": 31.770833333333332,
        "avg_win_r": 2.005356867495115,
        "avg_loss_r": -0.7586199629308183,
    }


def test_helix_resolved_hard_rejects_keep_corrected_seed_in_play(tmp_path) -> None:
    plugin = HelixPlugin(tmp_path, initial_equity=10_000, max_workers=1)
    baseline = _round1_seed_metrics()

    resolved = plugin._resolve_phase_hard_rejects(1, baseline, PHASE_HARD_REJECTS[1])
    score = composite_score(HelixMetrics(**baseline), hard_rejects=resolved)

    assert resolved["min_pf"] <= baseline["profit_factor"]
    assert resolved["min_regime_pf"] <= baseline["min_regime_pf"]
    assert resolved["max_r_dd"] >= baseline["max_r_dd"]
    assert not score.rejected
    assert score.total > 0.0


def test_helix_score_components_are_scaled_to_frontier_targets() -> None:
    target_metrics = HelixMetrics(
        total_trades=420,
        profit_factor=3.5,
        net_return_pct=250.0,
        total_r=170.0,
        max_r_dd=4.0,
        exit_efficiency=0.65,
        waste_ratio=0.85,
        tail_pct=0.85,
        big_winner_r=230.0,
        min_regime_pf=1.0,
        long_pf=3.5,
        short_pf=3.5,
        min_side_pf=3.5,
        win_rate=55.0,
        avg_win_r=2.5,
    )

    score = composite_score(target_metrics, hard_rejects={"min_trades": 1, "min_pf": 0.0, "max_r_dd": 25.0})

    assert not score.rejected
    assert score.net_profit_component == pytest.approx(1.0)
    assert score.win_rate_component == pytest.approx(1.0)
    assert score.frequency_component == pytest.approx(1.0)
    assert score.pf_component == pytest.approx(1.0)
    assert score.side_quality_component == pytest.approx(1.0)
    assert score.exit_quality_component == pytest.approx(1.0)
    assert score.inv_dd_component == pytest.approx(1.0)
    assert score.total == pytest.approx(1.0)


def test_helix_repaired_baseline_has_real_score_headroom() -> None:
    baseline = HelixMetrics(
        total_trades=270,
        profit_factor=2.370576050456337,
        net_return_pct=114.58087600000246,
        max_r_dd=6.773643264725349,
        exit_efficiency=0.45403976548917024,
        waste_ratio=0.7156612123608579,
        tail_pct=0.7976600147716284,
        min_regime_pf=2.2841560553593254,
        long_pf=2.370576050456337,
        short_pf=2.2841560553593254,
        min_side_pf=2.2841560553593254,
        total_r=164.80,
        big_winner_r=205.0,
        win_rate=38.148148148148145,
        avg_win_r=2.60,
    )

    score = composite_score(baseline, hard_rejects={"min_trades": 1, "min_pf": 0.0, "max_r_dd": 25.0})

    assert not score.rejected
    assert score.win_rate_component < 0.20
    assert score.frequency_component < 0.10
    assert score.winning_trades_component < 0.10
    assert 0.15 <= score.total <= 0.26
