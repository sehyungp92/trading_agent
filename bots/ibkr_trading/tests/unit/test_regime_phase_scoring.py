from __future__ import annotations

import pytest
import pandas as pd

from backtests.regime.analysis.metrics import PortfolioMetrics
from backtests.regime.analysis.assessment_validation import compute_spy_allocation_range_bp
from backtests.regime.auto import greedy_optimize
from backtests.regime.auto.phase_candidates import get_phase_candidates
from backtests.regime.auto.presets import get_research_preset
from backtests.regime.auto.phase_scoring import (
    compute_regime_stats,
    phase_1_score,
    phase_2_score,
    phase_3_score,
    phase_4_score,
)


def _make_regime_signals() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2008-10-03",
            "2008-10-10",
            "2020-03-06",
            "2020-03-13",
            "2022-01-07",
            "2022-01-14",
            "2022-02-04",
            "2023-01-06",
        ]
    )
    return pd.DataFrame(
        {
            "P_G": [0.05, 0.10, 0.15, 0.10, 0.10, 0.10, 0.15, 0.80],
            "P_R": [0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
            "P_S": [0.10, 0.10, 0.30, 0.35, 0.70, 0.75, 0.65, 0.05],
            "P_D": [0.80, 0.70, 0.45, 0.45, 0.10, 0.05, 0.10, 0.05],
            "Conf": [0.8, 0.75, 0.65, 0.60, 0.55, 0.50, 0.58, 0.85],
            "pi_SPY": [0.10, 0.12, 0.18, 0.20, 0.02, 0.01, 0.03, 0.28],
            "pi_EFA": [0.05, 0.06, 0.10, 0.09, 0.01, 0.01, 0.02, 0.15],
            "pi_TLT": [0.35, 0.33, 0.25, 0.24, 0.32, 0.35, 0.30, 0.10],
            "pi_GLD": [0.25, 0.26, 0.20, 0.22, 0.28, 0.30, 0.26, 0.08],
            "pi_CASH": [0.25, 0.23, 0.27, 0.25, 0.27, 0.23, 0.24, 0.39],
        },
        index=idx,
    )


def _soften_posteriors(signals: pd.DataFrame, dominant_prob: float = 0.55) -> pd.DataFrame:
    out = signals.copy()
    regime_cols = ["P_G", "P_R", "P_S", "P_D"]
    floor = (1.0 - dominant_prob) / 3.0
    softened = []
    for _, row in signals[regime_cols].iterrows():
        probs = [floor, floor, floor, floor]
        probs[int(row.values.argmax())] = dominant_prob
        softened.append(probs)
    out[regime_cols] = softened
    return out


def _signals_from_dominants(dominants: list[str]) -> pd.DataFrame:
    idx = _make_regime_signals().index
    alloc_map = {
        "G": {"pi_SPY": 0.28, "pi_EFA": 0.15, "pi_TLT": 0.10, "pi_GLD": 0.08, "pi_CASH": 0.39},
        "R": {"pi_SPY": 0.22, "pi_EFA": 0.12, "pi_TLT": 0.05, "pi_GLD": 0.20, "pi_CASH": 0.41},
        "S": {"pi_SPY": 0.03, "pi_EFA": 0.02, "pi_TLT": 0.30, "pi_GLD": 0.26, "pi_CASH": 0.39},
        "D": {"pi_SPY": 0.10, "pi_EFA": 0.05, "pi_TLT": 0.35, "pi_GLD": 0.25, "pi_CASH": 0.25},
    }
    rows = []
    for regime in dominants:
        row = {"P_G": 0.01, "P_R": 0.01, "P_S": 0.01, "P_D": 0.01, "Conf": 0.75}
        row[f"P_{regime}"] = 0.97
        row.update(alloc_map[regime])
        rows.append(row)
    return pd.DataFrame(rows, index=idx)


def test_regime_stats_include_new_step5_metrics():
    stats = compute_regime_stats(_make_regime_signals())

    assert stats["n_active_regimes"] >= 3
    assert stats["alloc_differentiation"] > 0.0
    assert stats["crisis_accuracy"] > 0.4
    assert 0.0 <= stats["posterior_penalty"] <= 1.0
    assert 0.0 <= stats["transition_penalty"] <= 1.0


def test_phase3_scoring_applies_financial_floor_and_new_components():
    metrics = PortfolioMetrics(
        total_return=1.0,
        cagr=0.12,
        sharpe=1.35,
        sortino=1.8,
        calmar=1.4,
        max_drawdown_pct=0.10,
        max_drawdown_duration=50,
        avg_annual_turnover=5.0,
        n_rebalances=20,
    )
    score = phase_3_score(metrics, compute_regime_stats(_make_regime_signals()))
    assert not score.rejected
    assert score.total > 0.0

    rejected = phase_3_score(
        PortfolioMetrics(
            total_return=1.0,
            cagr=0.12,
            sharpe=1.10,
            sortino=1.8,
            calmar=1.4,
            max_drawdown_pct=0.10,
            max_drawdown_duration=50,
            avg_annual_turnover=5.0,
            n_rebalances=20,
        ),
        compute_regime_stats(_make_regime_signals()),
    )
    assert rejected.rejected
    assert "Sharpe" in rejected.reject_reason


def test_phase1_and_phase2_penalize_soft_posteriors_instead_of_rewarding_them():
    metrics = PortfolioMetrics(
        total_return=1.0,
        cagr=0.12,
        sharpe=1.35,
        sortino=1.8,
        calmar=1.3,
        max_drawdown_pct=0.10,
        max_drawdown_duration=50,
        avg_annual_turnover=5.0,
        n_rebalances=20,
    )
    hard = _make_regime_signals()
    soft = _soften_posteriors(hard)

    phase1_hard = phase_1_score(metrics, compute_regime_stats(hard)).total
    phase1_soft = phase_1_score(metrics, compute_regime_stats(soft)).total
    phase2_hard = phase_2_score(metrics, compute_regime_stats(hard)).total
    phase2_soft = phase_2_score(metrics, compute_regime_stats(soft)).total

    assert phase1_hard > phase1_soft
    assert phase2_hard > phase2_soft


def test_phase1_and_phase2_penalize_noisy_transitions():
    metrics = PortfolioMetrics(
        total_return=1.0,
        cagr=0.12,
        sharpe=1.35,
        sortino=1.8,
        calmar=1.3,
        max_drawdown_pct=0.10,
        max_drawdown_duration=50,
        avg_annual_turnover=5.0,
        n_rebalances=20,
    )
    stable = _signals_from_dominants(["D", "D", "D", "S", "S", "S", "S", "G"])
    noisy = _signals_from_dominants(["D", "D", "S", "D", "S", "D", "S", "G"])

    stable_stats = compute_regime_stats(stable)
    noisy_stats = compute_regime_stats(noisy)

    assert noisy_stats["transition_rate"] > stable_stats["transition_rate"]
    assert noisy_stats["transition_penalty"] >= stable_stats["transition_penalty"]
    assert phase_1_score(metrics, stable_stats).total > phase_1_score(metrics, noisy_stats).total
    assert phase_2_score(metrics, stable_stats).total >= phase_2_score(metrics, noisy_stats).total


def test_phase1_and_phase2_reward_crisis_accuracy_and_allocation_differentiation():
    metrics = PortfolioMetrics(
        total_return=1.0,
        cagr=0.12,
        sharpe=1.35,
        sortino=1.8,
        calmar=1.3,
        max_drawdown_pct=0.10,
        max_drawdown_duration=50,
        avg_annual_turnover=5.0,
        n_rebalances=20,
    )
    good = _make_regime_signals()
    weak = good.copy()
    weak[["pi_SPY", "pi_EFA", "pi_TLT", "pi_GLD", "pi_CASH"]] = 0.20
    weak.loc["2022-01-07":"2022-02-04", ["P_G", "P_R", "P_S", "P_D"]] = [0.70, 0.10, 0.10, 0.10]

    good_stats = compute_regime_stats(good)
    weak_stats = compute_regime_stats(weak)

    assert good_stats["alloc_differentiation"] > weak_stats["alloc_differentiation"]
    assert good_stats["crisis_accuracy"] > weak_stats["crisis_accuracy"]
    assert phase_1_score(metrics, good_stats).total > phase_1_score(metrics, weak_stats).total
    assert phase_2_score(metrics, good_stats).total > phase_2_score(metrics, weak_stats).total


def test_all_phase_scorers_enforce_allocation_differentiation_floor():
    metrics = PortfolioMetrics(
        total_return=1.0,
        cagr=0.12,
        sharpe=1.35,
        sortino=1.8,
        calmar=1.3,
        max_drawdown_pct=0.10,
        max_drawdown_duration=50,
        avg_annual_turnover=5.0,
        n_rebalances=20,
    )
    weak = _make_regime_signals()
    weak[["pi_SPY", "pi_EFA", "pi_TLT", "pi_GLD", "pi_CASH"]] = 0.20
    weak_stats = compute_regime_stats(weak)

    assert weak_stats["alloc_differentiation"] < 0.015

    for scorer in (phase_1_score, phase_2_score, phase_3_score, phase_4_score):
        score = scorer(metrics, weak_stats)
        assert score.rejected
        assert "Alloc differentiation" in score.reject_reason


def test_phase2_candidates_use_paired_posterior_calibration_only():
    names = [name for name, _ in get_phase_candidates(2)]

    assert "posterior_pair_1p2_0p8" in names
    assert "posterior_pair_1p5_0p7" in names
    assert "posterior_pair_1p5_0p8" in names
    assert not any(name.startswith("posterior_temp_") for name in names)
    assert not any(name.startswith("posterior_ema_") for name in names)
    assert "posterior_full_cal" not in names
    assert "add_vix" not in names
    assert "add_realized_vol" not in names
    assert "add_trend_div" not in names
    assert "vix_replace_eqbond" not in names
    assert "vix_replace_breadth" not in names
    assert "vol_regime_combo" not in names


def test_phase3_fast_combo_uses_short_crisis_window():
    candidates = dict(get_phase_candidates(3))

    assert candidates["crisis_fast_combo"]["crisis_z_window"] == 21
    assert candidates["crisis_fast_combo"]["posterior_ema_risk_off_alpha"] == 1.0


def test_step9_preset_matches_assessment_r6_start():
    step9 = get_research_preset("step9_r6")

    assert step9["sticky_diag"] == 15.0
    assert step9["use_warm_start"] is False
    assert step9["use_expanding_window"] is False
    assert step9["rolling_window_years"] == 7
    assert step9["refit_ll_tolerance"] == 5.0
    assert step9["refit_freq"] == "QE"
    assert step9["z_minp"] == 30
    assert step9["posterior_temperature"] == 1.5
    assert step9["posterior_smoothing_eps"] == 0.01
    assert step9["posterior_ema_alpha"] == 0.7
    assert step9["stability_weight"] == 0.8
    assert step9["per_strat_max"] == 0.5
    assert step9["base_target_vol_annual"] == 0.08
    assert step9["kappa_totalvol_cap"] == 1.5
    assert step9["crisis_logit_a"] == 3.0
    assert step9["weight_smoothing_alpha"] == 0.5
    assert step9["L_max"] == 1.6
    assert step9["delta_rho_exempt"] == 0.0
    assert step9["sigma_floor_annual"] == 0.09
    assert step9["delta_rho_threshold"] == 0.15
    assert tuple(step9["crisis_weights"]) == (0.25, 0.50, 0.10, 0.15)
    assert step9["scanner_enabled"] is True
    assert step9["n_ensemble_models"] == 5


def test_step9_phase2_candidates_use_joint_moderate_pairs_only():
    names = [name for name, _ in get_phase_candidates(2, profile="step9_r6")]

    assert "posterior_pair_1p2_0p6" in names
    assert "posterior_pair_1p2_0p65" in names
    assert "posterior_pair_1p5_0p6" in names
    assert "posterior_pair_1p5_0p65" in names
    assert "posterior_pair_1p2_0p8" not in names
    assert "posterior_pair_1p5_0p7" not in names
    assert "posterior_pair_1p5_0p8" not in names
    assert not any(name.startswith("posterior_temp_") for name in names)
    assert not any(name.startswith("posterior_ema_") for name in names)


def test_step9_phase3_candidates_are_narrow_targeted_set():
    candidates = dict(get_phase_candidates(3, profile="step9_r6"))

    assert set(candidates) == {
        "crisis_weights_spread_focused",
        "crisis_z_window_21",
        "crisis_z_window_42",
        "cross_asset_corr_weight_0.2",
        "cross_asset_corr_weight_0.3",
    }
    assert candidates["crisis_weights_spread_focused"]["crisis_weights"] == (
        0.15,
        0.55,
        0.10,
        0.20,
    )
    assert candidates["cross_asset_corr_weight_0.2"]["crisis_weights"] == (
        0.20,
        0.50,
        0.10,
        0.20,
    )
    assert candidates["cross_asset_corr_weight_0.3"]["crisis_weights"] == (
        0.10,
        0.50,
        0.10,
        0.30,
    )


def test_step9_phase4_has_no_generic_fine_tuning_candidates():
    assert get_phase_candidates(4, profile="step9_r6") == []


def test_assessment_helpers_prefer_allocations_over_levered_positions():
    idx = pd.to_datetime(["2022-01-07", "2022-01-14", "2022-01-21", "2022-01-28"])
    signals = pd.DataFrame(
        {
            "P_G": [0.9, 0.8, 0.1, 0.1],
            "P_R": [0.05, 0.1, 0.1, 0.1],
            "P_S": [0.03, 0.05, 0.7, 0.75],
            "P_D": [0.02, 0.05, 0.1, 0.05],
            "w_SPY": [0.40, 0.38, 0.12, 0.10],
            "w_TLT": [0.10, 0.12, 0.30, 0.32],
            "pi_SPY": [0.18, 0.17, 0.14, 0.13],
            "pi_TLT": [0.05, 0.06, 0.35, 0.36],
        },
        index=idx,
    )

    stats = compute_regime_stats(signals)

    assert compute_spy_allocation_range_bp(signals) == pytest.approx(2800.0)
    assert stats["alloc_differentiation"] > 0.20


def test_research_presets_compose_recommended_stack_from_r3_reference():
    r3 = get_research_preset("r3_reference")
    full = get_research_preset("recommended_full_stack")

    assert r3["posterior_ema_alpha"] == 0.8
    assert r3["scanner_enabled"] is False
    assert r3["n_ensemble_models"] == 1
    assert tuple(r3["crisis_weights"]) == (0.3, 0.6, 0.1)

    assert full["posterior_ema_alpha"] == r3["posterior_ema_alpha"]
    assert full["scanner_enabled"] is True
    assert full["n_ensemble_models"] == 5
    assert full["crisis_z_window"] == 21
    assert tuple(full["crisis_weights"]) == (0.3, 0.6, 0.1, 0.1)
    assert full["posterior_ema_risk_off_alpha"] == 1.0
    assert full["posterior_temperature"] == 1.0
    assert full["posterior_smoothing_eps"] == 0.0


def test_step9_preset_and_candidate_profile_match_assessment_deltas():
    step9 = get_research_preset("step9_r6")
    phase1_names = [name for name, _ in get_phase_candidates(1, profile="step9_r6")]
    phase2 = dict(get_phase_candidates(2, profile="step9_r6"))
    phase3 = dict(get_phase_candidates(3, profile="step9_r6"))
    phase4 = get_phase_candidates(4, profile="step9_r6")

    assert step9["scanner_enabled"] is True
    assert step9["n_ensemble_models"] == 5
    assert step9["posterior_temperature"] == 1.5
    assert step9["posterior_ema_alpha"] == 0.7
    assert tuple(step9["crisis_weights"]) == (0.25, 0.50, 0.10, 0.15)

    assert "ensemble_5" not in phase1_names
    assert "ensemble_10" not in phase1_names
    assert set(name for name in phase2 if name.startswith("posterior_pair_")) == {
        "posterior_pair_1p2_0p6",
        "posterior_pair_1p2_0p65",
        "posterior_pair_1p5_0p6",
        "posterior_pair_1p5_0p65",
    }
    assert "crisis_weights_spread_focused" in phase3
    assert set(phase3) == {
        "crisis_weights_spread_focused",
        "crisis_z_window_21",
        "crisis_z_window_42",
        "cross_asset_corr_weight_0.2",
        "cross_asset_corr_weight_0.3",
    }
    assert phase4 == []
    assert phase3["crisis_weights_spread_focused"]["crisis_weights"] == (
        0.15,
        0.55,
        0.10,
        0.20,
    )
    assert phase3["cross_asset_corr_weight_0.2"]["crisis_weights"] == (
        0.20,
        0.50,
        0.10,
        0.20,
    )
    assert phase3["cross_asset_corr_weight_0.3"]["crisis_weights"] == (
        0.10,
        0.50,
        0.10,
        0.30,
    )


def test_rollback_helper_removes_harmful_recent_mutation(monkeypatch):
    score_map = {
        frozenset({"a", "b"}): 1.05,
        frozenset({"a", "c"}): 0.98,
        frozenset({"a"}): 1.00,
        frozenset({"b"}): 0.90,
    }

    def fake_score_mutation_set(pool, mutations, timeout):
        return score_map[frozenset(mutations.keys())], False, ""

    monkeypatch.setattr(greedy_optimize, "_score_mutation_set", fake_score_mutation_set)

    accepted_sequence = [
        ("a", {"a": 1}),
        ("b", {"b": 1}),
        ("c", {"c": 1}),
    ]
    new_sequence, new_score, rollback_rounds = greedy_optimize._rollback_last_mutations(
        pool=object(),
        base_muts={},
        accepted_sequence=accepted_sequence,
        current_score=1.0,
        min_delta=0.01,
        timeout=1.0,
        verbose=False,
    )

    assert [name for name, _ in new_sequence] == ["a", "b"]
    assert new_score == 1.05
    assert rollback_rounds[0].candidate_id == "rollback_remove_c"
