from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backtests.momentum.analysis.regime_diagnostics import generate_regime_diagnostics
from backtests.momentum.auto.nq_regime.phase_candidates import BASE_MUTATIONS
from backtests.momentum.auto.nq_regime.phase_candidates import build_round5_seed_from_configs
from backtests.momentum.auto.nq_regime.phase_candidates import get_phase_candidates
from backtests.momentum.auto.nq_regime.plugin import NqRegimePlugin
from backtests.momentum.auto.nq_regime.scoring import PHASE_HARD_REJECTS
from backtests.momentum.auto.nq_regime.scoring import PHASE_WEIGHTS
from backtests.momentum.auto.nq_regime.scoring import composite_score
from backtests.shared.auto.phase_state import PhaseState


def test_nq_regime_plugin_exposes_phased_optimizer_specs(tmp_path: Path) -> None:
    plugin = NqRegimePlugin(tmp_path, initial_equity=10_000, max_workers=1)
    state = PhaseState()

    assert plugin.name == "nq_regime"
    assert plugin.num_phases == 7
    for phase in range(1, plugin.num_phases + 1):
        spec = plugin.get_phase_spec(phase, state)
        assert spec.focus
        assert spec.candidates
        assert spec.scoring_weights
        assert len(spec.scoring_weights) <= 7
        assert callable(spec.gate_criteria_fn)
    phase7 = plugin.get_phase_spec(7, state)
    phase7_names = {candidate.name for candidate in phase7.candidates}
    assert "Synergy" in phase7.focus
    assert "r5_final_frequency_stack" in phase7_names
    assert len(PHASE_WEIGHTS[7]) <= 7


def test_nq_regime_plugin_handles_empty_replay_bundle_without_data(tmp_path: Path) -> None:
    plugin = NqRegimePlugin(tmp_path, initial_equity=10_000, max_workers=1)

    metrics = plugin.compute_final_metrics({})

    assert metrics["total_trades"] == 0.0
    assert "profit_factor" in metrics
    assert "max_drawdown_pct" in metrics
    assert "module_coverage" in metrics


def test_nq_regime_end_of_round_artifacts_include_candidate_funnel(tmp_path: Path) -> None:
    plugin = NqRegimePlugin(tmp_path, initial_equity=10_000, max_workers=1)

    artifacts = plugin.build_end_of_round_artifacts(PhaseState())

    assert "candidate_funnel" in artifacts.dimension_reports
    assert "second_wind" in artifacts.dimension_reports["candidate_funnel"]


def test_nq_regime_diagnostics_show_all_component_edges() -> None:
    trades = [
        SimpleNamespace(
            module="second_wind",
            side="BUY",
            pnl_dollars=100.0,
            r_multiple=1.0,
            mfe_r=1.4,
            mae_r=-0.2,
            grade="A",
            setup_score=9,
            setup_type="pm_squeeze_fire",
            entry_model="breakout_close_retest",
            squeeze_duration=6,
            squeeze_range=24.0,
            volume_multiple=1.4,
            target_room_r=2.0,
            exit_reason="target_2",
            entry_time=None,
        )
    ]
    text = generate_regime_diagnostics(
        trades,
        {
            "total_trades": 1.0,
            "net_profit": 100.0,
            "profit_factor": 10.0,
            "win_rate": 1.0,
            "avg_r": 1.0,
            "total_r": 1.0,
            "module_coverage": 1 / 3,
        },
    )

    assert "nq_1" in text
    assert "nq_2" in text
    assert "nq_3" in text
    assert "Second-Wind Continuation" in text
    assert "Structural Expansion" in text
    assert "Liquidity Reversion" in text


def test_nq_regime_diagnostics_separates_commission_only_excursion_gaps() -> None:
    trade = SimpleNamespace(
        module="liquidity_reversion",
        symbol="MNQ",
        side="BUY",
        qty=1,
        entry_price=100.0,
        initial_stop=90.0,
        pnl_dollars=-2.20,
        commission=2.0,
        r_multiple=-0.11,
        mfe_r=0.0,
        mae_r=0.0,
        grade="A",
        setup_score=9,
        setup_type="sweep_reclaim",
        entry_model="swept_level_retest",
        target_room_r=2.0,
        exit_reason="reversion_vwap_touch",
        entry_time=None,
    )

    text = generate_regime_diagnostics(
        [trade],
        {
            "total_trades": 1.0,
            "net_profit": -2.20,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "avg_r": -0.11,
            "total_r": -0.11,
            "module_coverage": 1 / 3,
        },
    )

    assert "Tol/Comm" in text
    assert "Excursion bounds are internally consistent within tolerance." in text


def test_nq_regime_plugin_exposes_reproducibility_fingerprint_parts(tmp_path: Path) -> None:
    plugin = NqRegimePlugin(tmp_path, initial_equity=10_000, max_workers=1)

    parts = plugin.source_fingerprint_parts()

    assert plugin.source_fingerprint()
    assert "round5_auto_package" in parts
    assert "shared_auto_runner" in parts


def test_nq_regime_exposes_synergy_optimizer_specs(tmp_path: Path) -> None:
    plugin = NqRegimePlugin(tmp_path, initial_equity=10_000, max_workers=1)
    state = PhaseState()

    assert plugin.name == "nq_regime"
    assert plugin.num_phases == 7
    assert BASE_MUTATIONS["flags.enable_structural_expansion"] is True
    assert BASE_MUTATIONS["flags.enable_liquidity_reversion"] is True
    assert BASE_MUTATIONS["flags.enable_second_wind"] is True

    for phase in range(1, plugin.num_phases + 1):
        spec = plugin.get_phase_spec(phase, state)
        assert spec.focus
        assert spec.candidates
        assert spec.scoring_weights
        assert len(spec.scoring_weights) <= 7
        assert callable(spec.gate_criteria_fn)

    assert "Synergy" in plugin.get_phase_spec(7, state).focus
    assert PHASE_WEIGHTS[7]["alpha_return"] >= PHASE_WEIGHTS[1]["alpha_return"]
    assert PHASE_HARD_REJECTS[7]["min_module_coverage"] == 1.0


def test_nq_regime_candidates_do_not_disable_components() -> None:
    for phase in range(1, 8):
        for _, mutations in get_phase_candidates(phase, {}):
            assert mutations.get("flags.enable_structural_expansion", True) is True
            assert mutations.get("flags.enable_liquidity_reversion", True) is True
            assert mutations.get("flags.enable_second_wind", True) is True


def test_nq_regime_seed_blends_round4_configs_without_starving_modules() -> None:
    round4a = {
        "flags.enable_structural_expansion": True,
        "flags.enable_liquidity_reversion": True,
        "flags.enable_second_wind": True,
        "param_overrides.STRUCTURAL_MIN_SCORE": 9,
        "param_overrides.REVERSION_MIN_SCORE": 7,
        "param_overrides.SECOND_WIND_MIN_SCORE": 8,
        "param_overrides.SECOND_WIND_STOP_CAP": 30.0,
        "param_overrides.TARGET_ROOM_MIN_R": 0.75,
        "param_overrides.MAX_FULL_RISK_TRADES": 4,
        "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": False,
    }
    round4b = {
        "param_overrides.STRUCTURAL_MIN_SCORE": 8,
        "param_overrides.STRUCTURAL_STOP_MODEL": "recent_5m",
        "param_overrides.REVERSION_MIN_SCORE": 6,
        "param_overrides.SECOND_WIND_MIN_SCORE": 6,
    }
    round4c = {
        "flags.enable_structural_expansion": False,
        "flags.enable_liquidity_reversion": True,
        "flags.enable_second_wind": False,
        "param_overrides.STRUCTURAL_MIN_SCORE": 10,
        "param_overrides.REVERSION_MIN_SCORE": 8,
        "param_overrides.REVERSION_RETEST_OFFSET_TICKS": 0,
        "param_overrides.SECOND_WIND_MIN_SCORE": 7,
    }

    seed = build_round5_seed_from_configs(round4a, round4b, round4c)

    assert seed["flags.enable_structural_expansion"] is True
    assert seed["flags.enable_liquidity_reversion"] is True
    assert seed["flags.enable_second_wind"] is True
    assert seed["param_overrides.STRUCTURAL_MIN_SCORE"] == 8
    assert seed["param_overrides.STRUCTURAL_STOP_MODEL"] == "recent_5m"
    assert seed["param_overrides.REVERSION_MIN_SCORE"] == 8
    assert seed["param_overrides.REVERSION_RETEST_OFFSET_TICKS"] == 0
    assert seed["param_overrides.SECOND_WIND_MIN_SCORE"] == 8
    assert seed["param_overrides.SECOND_WIND_STOP_CAP"] == 30.0
    assert seed["param_overrides.TARGET_ROOM_MIN_R"] == 0.50
    assert seed["param_overrides.MAX_FULL_RISK_TRADES"] == 3
    assert seed["param_overrides.ROUTE_CANDIDATE_LED_ENABLED"] is True


def test_nq_regime_rejects_module_starvation() -> None:
    liquidity_only_metrics = {
        "total_trades": 500.0,
        "trades_per_month": 10.0,
        "total_r_per_month": 12.0,
        "profit_factor": 13.25,
        "max_drawdown_pct": 0.009,
        "avg_r": 1.09,
        "module_coverage": 1 / 3,
        "min_module_trades": 0.0,
        "module_second_wind_trades": 0.0,
        "module_structural_expansion_trades": 0.0,
        "module_liquidity_reversion_trades": 500.0,
        "module_liquidity_reversion_avg_r": 1.09,
        "module_liquidity_reversion_profit_factor": 13.25,
    }

    score = composite_score(liquidity_only_metrics, PHASE_WEIGHTS[7], PHASE_HARD_REJECTS[7])

    assert score.rejected is True
    assert score.reject_reason in {"min_module_coverage", "min_module_trades", "min_nq1_trades"}
