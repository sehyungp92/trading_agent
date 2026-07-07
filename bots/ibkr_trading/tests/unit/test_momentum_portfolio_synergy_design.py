from pathlib import Path

from backtests.momentum.auto.portfolio_synergy.phase_candidates import (
    ROUND_TARGETS,
    SCORE_WEIGHTS,
    SEED_PORTFOLIO_CONFIG,
    STRATEGY_ORDER,
    get_phase_candidates,
    phase_summary,
)
from backtests.momentum.auto.portfolio_synergy.round_design import build_round_design


def test_portfolio_synergy_design_uses_all_four_momentum_strategies() -> None:
    design = build_round_design(Path("."))

    assert design["initial_equity"] == 50_000.0
    assert design["risk_stance"] == "aggressive_controlled"
    assert [item["strategy_id"] for item in design["diagnostic_assessments"]] == list(STRATEGY_ORDER)
    assert len(design["seed_portfolio_config"]["strategy_allocations"]) == 4
    assert len(design["scoring_weights"]) <= 7
    assert abs(sum(design["scoring_weights"].values()) - 1.0) < 1e-9


def test_portfolio_synergy_seed_overweights_clean_frequency() -> None:
    allocations = SEED_PORTFOLIO_CONFIG["strategy_allocations"]

    assert allocations["NQ_REGIME"]["priority"] == 0
    assert allocations["NQ_REGIME"]["base_risk_pct"] > allocations["DownturnDominator_v1"]["base_risk_pct"]
    assert allocations["VdubusNQ_v4"]["base_risk_pct"] > allocations["NQDTC_v2.1"]["base_risk_pct"]
    assert SEED_PORTFOLIO_CONFIG["portfolio_rules"]["max_total_positions"] == 4
    assert SEED_PORTFOLIO_CONFIG["portfolio_rules"]["drawdown_tiers"][-1] == (0.20, 0.00)


def test_portfolio_synergy_phase_space_is_frequency_alpha_first_but_dd_guarded() -> None:
    summaries = phase_summary()

    assert len(summaries) == 7
    assert SCORE_WEIGHTS["alpha_return"] > SCORE_WEIGHTS["drawdown_control"]
    assert SCORE_WEIGHTS["trade_frequency"] > SCORE_WEIGHTS["profit_factor_quality"]
    assert ROUND_TARGETS["max_drawdown_pct"] == 0.18
    assert any(candidate["name"] == "heat_cap_4_5_probe" for candidate in get_phase_candidates(2))
    assert any(candidate["name"] == "drawdown_tiers_controlled_aggressive" for candidate in get_phase_candidates(6))
