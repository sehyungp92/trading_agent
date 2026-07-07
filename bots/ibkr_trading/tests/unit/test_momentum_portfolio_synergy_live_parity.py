from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


ACTIVE_PORTFOLIO_ROUND = 2


def _optimized_config() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "backtests"
        / "output"
        / "momentum"
        / "portfolio_synergy"
        / f"round_{ACTIVE_PORTFOLIO_ROUND}"
        / "optimized_portfolio_config.json"
    )
    return json.loads(path.read_text())


def test_momentum_live_configs_match_active_portfolio_synergy() -> None:
    from strategies.momentum import coordinator
    from strategies.momentum.downturn import config as downturn_config
    from strategies.momentum.nq_regime import config as nq_regime_config
    from strategies.momentum.nqdtc import config as nqdtc_config
    from strategies.momentum.vdub import config as vdub_config

    optimized = _optimized_config()
    optimized_rules = optimized["rules"]
    optimized_dynamic = optimized["dynamic_risk"]
    optimized_allocs = {
        row["strategy_id"]: row for row in optimized["strategy_allocations"]
    }

    assert coordinator._PORTFOLIO_HEAT_CAP_R == optimized["heat_cap_R"]
    assert coordinator._PORTFOLIO_DAILY_STOP_R == optimized["portfolio_daily_stop_R"]
    assert coordinator._PORTFOLIO_WEEKLY_STOP_R == optimized["portfolio_weekly_stop_R"]
    assert coordinator._OPTIMIZED_INITIAL_EQUITY == optimized["initial_equity"]
    assert (
        coordinator._OPTIMIZED_REFERENCE_UNIT_RISK_DOLLARS
        == optimized["reference_unit_risk_dollars"]
    )
    assert coordinator._REFERENCE_UNIT_RISK_PCT == pytest.approx(
        optimized["reference_unit_risk_dollars"] / optimized["initial_equity"]
    )
    assert coordinator._MAX_TOTAL_POSITIONS == optimized["max_total_positions"]
    assert (
        coordinator._MAX_FAMILY_CONTRACTS_MNQ_EQ
        == optimized_rules["max_family_contracts_mnq_eq"]
    )
    assert coordinator._DIRECTIONAL_CAP_R == optimized_rules["directional_cap_R"]
    assert (
        coordinator._DIRECTIONAL_CAP_LONG_R
        == optimized_rules["directional_cap_long_R"]
    )
    assert (
        coordinator._DIRECTIONAL_CAP_SHORT_R
        == optimized_rules["directional_cap_short_R"]
    )
    assert (
        coordinator._NQDTC_DIRECTION_FILTER_ENABLED
        == optimized_rules["nqdtc_direction_filter_enabled"]
    )
    assert coordinator._NQDTC_AGREE_SIZE_MULT == optimized_rules["nqdtc_agree_size_mult"]
    assert (
        coordinator._NQDTC_OPPOSE_SIZE_MULT
        == optimized_rules["nqdtc_oppose_size_mult"]
    )
    assert coordinator._DD_TIERS == tuple(
        tuple(row) for row in optimized_rules["dd_tiers"]
    )
    assert coordinator._STRATEGY_PRIORITIES == tuple(
        tuple(row) for row in optimized_rules["strategy_priorities"]
    )
    assert coordinator._STRATEGY_SIZE_MULTIPLIERS == tuple(
        tuple(row) for row in optimized_dynamic["strategy_multipliers"]
    )
    assert (
        coordinator._DYNAMIC_EXISTING_POSITION_MULT
        == optimized_dynamic["existing_position_mult"]
    )
    assert (
        coordinator._DYNAMIC_HEAT_PRESSURE_THRESHOLD
        == optimized_dynamic["heat_pressure_threshold"]
    )
    assert coordinator._DYNAMIC_HEAT_PRESSURE_MULT == optimized_dynamic["heat_pressure_mult"]
    assert (
        coordinator._DYNAMIC_SAME_DIRECTION_PRESSURE_THRESHOLD
        == optimized_dynamic["same_direction_pressure_threshold"]
    )
    assert (
        coordinator._DYNAMIC_SAME_DIRECTION_PRESSURE_MULT
        == optimized_dynamic["same_direction_pressure_mult"]
    )
    assert coordinator._DYNAMIC_MAX_TRADE_RISK_R == optimized_dynamic["max_trade_risk_R"]
    assert coordinator._DYNAMIC_MIN_QTY == optimized_dynamic["min_qty"]
    assert (
        coordinator._DYNAMIC_FIT_TO_REMAINING_HEAT
        == optimized_dynamic["fit_to_remaining_heat"]
    )
    assert (
        coordinator._DYNAMIC_FIT_TO_REMAINING_DIRECTIONAL_CAP
        == optimized_dynamic["fit_to_remaining_directional_cap"]
    )
    assert (
        coordinator._DYNAMIC_FIT_TO_REMAINING_FAMILY_CAP
        == optimized_dynamic["fit_to_remaining_family_cap"]
    )

    live_base_risk = {
        nq_regime_config.STRATEGY_ID: nq_regime_config.BASE_RISK_PCT,
        vdub_config.STRATEGY_ID: vdub_config.BASE_RISK_PCT,
        nqdtc_config.STRATEGY_ID: nqdtc_config.BASE_RISK_PCT,
        downturn_config.STRATEGY_ID: downturn_config.BASE_RISK_PCT,
    }
    live_daily_stops = dict(coordinator._STRATEGY_DAILY_STOPS_R)
    live_max_concurrent = dict(coordinator._MAX_STRATEGY_ACTIVE_POSITIONS)
    live_priorities = dict(coordinator._STRATEGY_PRIORITIES)

    assert set(live_base_risk) == set(optimized_allocs)
    for strategy_id, expected in optimized_allocs.items():
        assert live_base_risk[strategy_id] == expected["base_risk_pct"], strategy_id
        assert live_daily_stops[strategy_id] == expected["daily_stop_R"], strategy_id
        assert live_max_concurrent[strategy_id] == expected["max_concurrent"], strategy_id
        assert live_priorities[strategy_id] == expected["priority"], strategy_id

    assert (
        abs(nqdtc_config.DAILY_STOP_R)
        == optimized_allocs["NQDTC_v2.1"]["daily_stop_R"]
    )
