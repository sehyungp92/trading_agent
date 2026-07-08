from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import backtests.shared.auto.phase_runner as phase_runner_module
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.types import EndOfRoundArtifacts, GateCriterion, GateResult, GreedyResult, PhaseAnalysis
from backtests.stock.analysis.iaric_pullback_diagnostics import (
    _threshold_sweep_rows,
    compute_pullback_diagnostic_snapshot,
    pullback_full_diagnostic,
)
from backtests.stock.analysis.metrics import PerformanceMetrics
from backtests.stock.auto.iaric.phase_candidates import (
    BASE_MUTATIONS,
    PHASE_CANDIDATES,
    PHASE_FOCUS,
    V4R1_BASE_MUTATIONS,
    V4R1_PHASE_CANDIDATES,
    V5R1_BASE_MUTATIONS,
    V5R1_PHASE_CANDIDATES,
    V5R2_BASE_MUTATIONS,
    V5R2_PHASE_CANDIDATES,
    get_phase_candidate_lookup,
    get_phase_candidates,
)
from backtests.stock.auto.config_mutator import mutate_iaric_config
from backtests.stock.config_iaric import IARICBacktestConfig
from backtests.stock.auto.iaric.phase_scoring import (
    V5R1_PHASE_HARD_REJECTS,
    V5R2_PHASE_SCORING_WEIGHTS,
    score_v5r1_pullback_phase,
)

from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin, select_pullback_branch
from backtests.stock.auto.runners.run_live_aligned_ablation_perturbation import _perturbation_candidates
from backtests.stock.auto.scoring import IARIC_NORM, composite_score
from backtests.stock.engine.iaric_pullback_engine import (
    _build_selection_attribution,
    _shared_pullback_state,
    _passes_rank_gate,
    _rank_gate_reason,
)
from backtests.stock.models import Direction, TradeRecord
from backtests.shared.auto.phase_runner import PhaseRunner
from backtests.shared.auto.phase_state import PhaseState
from strategies.stock.iaric.config import StrategySettings


def test_select_pullback_branch_obeys_round3_selection_rule():
    aggressive_win = select_pullback_branch(
        {"avg_r": 0.25, "total_trades": 100.0, "profit_factor": 2.30, "max_drawdown_pct": 0.05},
        {"avg_r": 0.28, "total_trades": 100.0, "profit_factor": 2.05, "max_drawdown_pct": 0.06},
    )
    mainline_win = select_pullback_branch(
        {"avg_r": 0.25, "total_trades": 100.0, "profit_factor": 2.30, "max_drawdown_pct": 0.05},
        {"avg_r": 0.26, "total_trades": 100.0, "profit_factor": 1.95, "max_drawdown_pct": 0.08},
    )

    assert aggressive_win["selected_profile"] == "aggressive"
    assert mainline_win["selected_profile"] == "mainline"


def test_live_aligned_perturbations_skip_zero_partial_trigger():
    candidates = dict(
        _perturbation_candidates(
            {
                "param_overrides.pb_v2_partial_profit_trigger_r": 0.1,
                "param_overrides.pb_v2_partial_profit_remainder_stop_r": 0.7,
            }
        )
    )

    assert "perturb_v2_partial_profit_trigger_r_down" not in candidates
    assert candidates["perturb_v2_partial_profit_trigger_r_up"]["param_overrides.pb_v2_partial_profit_trigger_r"] == pytest.approx(0.2)


def test_plugin_suggests_structural_experiments_for_weak_signal_and_route_metrics():
    plugin = IARICPullbackPlugin(data_dir=Path("."))
    suggested = plugin.suggest_experiments(
        1,
        {
            "signal_score_edge": 0.01,
            "crowded_day_discrimination": -0.01,
            "crowded_day_missed_alpha_inverse": 0.0,
            "route_score_monotonicity": 0.0,
            "open_scored_inverse": 0.0,
            "route_diversity": 0.0,
            "managed_exit_share": 0.0,
            "eod_flatten_share": 1.0,
            "carry_avg_r": 0.0,
            "stop_hit_total_r": 0.0,
        },
        weaknesses=["weak signal model"],
        state=PhaseState(),
    )

    assert suggested
    assert all(experiment.name in {name for name, _ in PHASE_CANDIDATES[1]} for experiment in suggested)


def test_open_scored_threshold_sweep_can_reclass_entered_open_routes_without_ready_bar():
    rows = _threshold_sweep_rows(
        [
            {
                "disposition": "entered",
                "entry_route_family": "OPEN_SCORED_ENTRY",
                "daily_signal_score": 70.0,
                "daily_signal_rank_pct": 20.0,
                "actual_r": 0.40,
                "shadow_r": -0.20,
                "ready_bar_index": None,
            }
        ],
        param_key="pb_open_scored_min_score",
        active_value=65.0,
        values=[65.0, 75.0],
    )

    assert rows[0]["accepted"] == 1
    assert rows[1]["accepted"] == 0
    assert rows[1]["rejected_avg_shadow_r"] == pytest.approx(-0.20)


def test_percentile_rank_gate_respects_absolute_and_percentile_bounds():
    settings = StrategySettings(
        pb_entry_rank_min=1,
        pb_entry_rank_max=999,
        pb_entry_rank_pct_min=20.0,
        pb_entry_rank_pct_max=50.0,
    )

    assert not _passes_rank_gate(3, 20, settings)
    assert _passes_rank_gate(4, 20, settings)
    assert _passes_rank_gate(10, 20, settings)
    assert not _passes_rank_gate(11, 20, settings)


def test_iaric_immutable_score_uses_150_trade_frequency_ceiling():
    metrics = PerformanceMetrics(
        total_trades=120,
        win_rate=0.50,
        net_profit=1_000.0,
        profit_factor=2.0,
        calmar=1.0,
        max_drawdown_pct=0.02,
    )

    score = composite_score(metrics, initial_equity=10_000.0, norm=IARIC_NORM)

    assert IARIC_NORM.freq_ceiling == 150.0
    assert score.freq_component == pytest.approx(120.0 / 150.0)


def test_v5r1_guardrails_score_conservative_open_scored_baseline():
    metrics = {
        "total_trades": 669.0,
        "avg_r": 0.0570,
        "expected_total_r": 38.14,
        "profit_factor": 1.36,
        "sharpe": 1.85,
        "max_drawdown_pct": 0.0575,
        "gap_selectivity_edge": 0.0688,
        "crowded_day_discrimination": 0.0400,
        "eod_flatten_share": 0.0374,
        "stop_hit_share": 0.592,
        "protection_candidate_share": 0.085,
    }

    assert IARICPullbackPlugin._phase_reject_reason(1, metrics, V5R1_PHASE_HARD_REJECTS[1]) == ""
    assert score_v5r1_pullback_phase(1, metrics) > 0.0


def test_v5r2_score_stays_normalized_and_limited_to_seven_components():
    for weights in V5R2_PHASE_SCORING_WEIGHTS.values():
        assert len(weights) <= 7
        assert sum(weights.values()) == pytest.approx(1.0)


def test_iaric_current_round_candidates_target_strategy_settings_surface():
    settings_fields = set(StrategySettings.__dataclass_fields__)
    stale_top_level_keys = {"max_per_sector", "max_positions_tier_b"}

    for base_mutations, phase_candidates in (
        (V4R1_BASE_MUTATIONS, V4R1_PHASE_CANDIDATES),
        (V5R1_BASE_MUTATIONS, V5R1_PHASE_CANDIDATES),
        (V5R2_BASE_MUTATIONS, V5R2_PHASE_CANDIDATES),
    ):
        mutation_sets = [base_mutations]
        mutation_sets.extend(mutations for candidates in phase_candidates.values() for _, mutations in candidates)
        for mutations in mutation_sets:
            assert not stale_top_level_keys.intersection(mutations)
            for key in mutations:
                if key.startswith("param_overrides."):
                    assert key.split(".", 1)[1] in settings_fields


def test_iaric_legacy_capacity_mutations_map_to_strategy_settings():
    config = mutate_iaric_config(
        IARICBacktestConfig(),
        {
            "max_per_sector": 3,
            "max_positions_tier_b": 6,
        },
    )

    assert config.param_overrides["max_positions_per_sector"] == 3
    assert config.param_overrides["max_positions_tier_b"] == 6


def test_v5r1_gate_allows_avg_r_tradeoff_when_expected_total_r_improves():
    reference = {
        "total_trades": 669.0,
        "avg_r": 0.05701584937885738,
        "expected_total_r": 38.14360323445559,
        "profit_factor": 1.3611101158429302,
        "sharpe": 1.8534233918119702,
        "max_drawdown_pct": 0.05749607031371148,
    }
    metrics = {
        **reference,
        "total_trades": 862.0,
        "avg_r": 0.051314058115277984,
        "expected_total_r": 44.23271809536962,
        "profit_factor": 1.3123883855257508,
        "sharpe": 1.8080579094946547,
        "max_drawdown_pct": 0.04469789104481218,
    }
    plugin = IARICPullbackPlugin.__new__(IARICPullbackPlugin)
    plugin._phase_hard_rejects = V5R1_PHASE_HARD_REJECTS
    plugin._baseline_metrics = lambda: reference

    criteria = plugin._v5r1_gate_criteria(1, metrics, PhaseState())

    assert all(item.passed for item in criteria)


def test_phase_runner_keeps_plugin_initial_mutations_for_phase_one(tmp_path):
    class _DummyPlugin:
        initial_mutations = {"foo": 1, "bar": 2}

    runner = PhaseRunner(plugin=_DummyPlugin(), output_dir=Path(tmp_path))
    state = runner.load_state()

    prepared = runner._prepare_state_for_phase(state, 1)

    assert prepared.cumulative_mutations == {"foo": 1, "bar": 2}


def test_phase_runner_keeps_prior_phase_mutations_for_later_phases(tmp_path):
    class _DummyPlugin:
        initial_mutations = {"foo": 1}

    runner = PhaseRunner(plugin=_DummyPlugin(), output_dir=Path(tmp_path))
    state = PhaseState(
        completed_phases=[1],
        cumulative_mutations={"foo": 1, "bar": 2},
        phase_results={
            1: {
                "final_mutations": {"foo": 1, "bar": 2},
                "new_mutations": {"bar": 2},
            }
        },
    )

    prepared = runner._prepare_state_for_phase(state, 2)

    assert prepared.cumulative_mutations == {"foo": 1, "bar": 2}


def test_rank_gate_reason_distinguishes_abs_and_percentile_rejects():
    settings = StrategySettings(
        pb_entry_rank_min=4,
        pb_entry_rank_max=12,
        pb_entry_rank_pct_min=20.0,
        pb_entry_rank_pct_max=50.0,
    )

    assert _rank_gate_reason(2, 20, settings) == "rank_abs_reject"
    assert _rank_gate_reason(13, 20, settings) == "rank_abs_reject"
    assert _rank_gate_reason(4, 30, settings) == "rank_pct_reject"
    assert _rank_gate_reason(9, 20, settings) is None


def test_selection_attribution_tracks_best_skipped_and_marginal_loss():
    ledger = {
        datetime(2026, 3, 31, tzinfo=timezone.utc).date(): [
            {"symbol": "AAA", "disposition": "entered", "actual_r": 0.10},
            {"symbol": "BBB", "disposition": "rank_pct_reject", "shadow_r": 0.35},
            {"symbol": "CCC", "disposition": "sector_cap_reject", "shadow_r": -0.20},
        ]
    }

    attribution = _build_selection_attribution(ledger)
    day = next(iter(attribution.values()))

    assert day["best_skipped_symbol"] == "BBB"
    assert day["skipped_beating_worst_entered"] == 1
    assert pytest.approx(day["skipped_avg_shadow_r"], rel=1e-6) == 0.075


def test_pullback_full_diagnostic_uses_new_sections_and_downgrades_old_ones():
    trades = [
        TradeRecord(
            strategy="IARIC_PB",
            symbol="AAA",
            direction=Direction.LONG,
            entry_time=datetime(2026, 1, 5, tzinfo=timezone.utc),
            exit_time=datetime(2026, 1, 5, tzinfo=timezone.utc),
            entry_price=10.0,
            exit_price=10.5,
            quantity=100,
            pnl=50.0,
            r_multiple=0.5,
            risk_per_share=1.0,
            commission=0.0,
            slippage=0.0,
            exit_reason="EOD_FLATTEN",
            sector="Tech",
            regime_tier="A",
            metadata={"entry_rsi": 4.0, "entry_rank": 5, "entry_rank_pct": 25.0, "entry_sma_dist_pct": 4.0, "entry_cdd": 3, "entry_gap_pct": -0.5, "close_r": 0.5, "close_pct": 0.8, "mfe_r": 1.2, "mae_r": 0.2},
        ),
        TradeRecord(
            strategy="IARIC_PB",
            symbol="BBB",
            direction=Direction.LONG,
            entry_time=datetime(2026, 1, 6, tzinfo=timezone.utc),
            exit_time=datetime(2026, 1, 6, tzinfo=timezone.utc),
            entry_price=20.0,
            exit_price=19.0,
            quantity=100,
            pnl=-100.0,
            r_multiple=-1.0,
            risk_per_share=1.0,
            commission=0.0,
            slippage=0.0,
            exit_reason="STOP_HIT",
            sector="Health",
            regime_tier="B",
            metadata={"entry_rsi": 8.0, "entry_rank": 9, "entry_rank_pct": 45.0, "entry_sma_dist_pct": 9.0, "entry_cdd": 5, "entry_gap_pct": 0.2, "close_r": -1.0, "close_pct": 0.2, "mfe_r": 0.4, "mae_r": 1.1},
        ),
    ]
    report = pullback_full_diagnostic(
        trades,
        metrics={"sharpe": 1.23, "max_drawdown_pct": 0.03, "managed_exit_share": 0.2, "eod_flatten_share": 0.5},
        candidate_ledger={},
        funnel_counters={"universe_seen": 10, "triggered": 4, "candidate_pool": 3, "entered": 2},
        rejection_log=[{"gate": "rank_pct_reject", "shadow_r": -0.2}],
        selection_attribution={datetime(2026, 1, 5, tzinfo=timezone.utc).date(): {"candidate_count": 3, "entered_count": 1, "entered_avg_r": 0.5, "skipped_avg_shadow_r": -0.1, "best_skipped_symbol": "ZZZ", "best_skipped_shadow_r": 0.1, "skipped_beating_worst_entered": 0}},
    )

    assert "Executive Verdicts" in report
    assert "Signal Funnel & Gate Attribution" in report
    assert "Selector Frontier & Filter Attribution" in report
    assert "Capacity & Replacement Analysis" in report
    assert "Economic Exit Frontier" in report
    assert "Entry Timing Counterfactuals" in report
    assert "Entry Quality Deep Dive" in report
    assert "Trade Management Deep Dive" in report
    assert "Exit Replacement & Carry Audit" in report
    assert "Positive Contribution / Alpha Source" in report
    assert "RSI Exit Threshold Sensitivity" not in report
    assert "Hold Duration Analysis" not in report


def test_snapshot_exposes_shadow_and_selection_quality_fields():
    trade = TradeRecord(
        strategy="IARIC_PB",
        symbol="AAA",
        direction=Direction.LONG,
        entry_time=datetime(2026, 1, 5, tzinfo=timezone.utc),
        exit_time=datetime(2026, 1, 5, tzinfo=timezone.utc),
        entry_price=10.0,
        exit_price=10.6,
        quantity=100,
        pnl=60.0,
        r_multiple=0.6,
        risk_per_share=1.0,
        commission=0.0,
        slippage=0.0,
        exit_reason="EOD_FLATTEN",
        sector="Tech",
        regime_tier="A",
        metadata={"entry_rsi": 3.0, "entry_rank": 4, "entry_rank_pct": 20.0, "entry_sma_dist_pct": 3.0, "entry_cdd": 2, "entry_gap_pct": -0.3, "close_r": 0.6, "close_pct": 0.9, "mfe_r": 1.1},
    )
    snapshot = compute_pullback_diagnostic_snapshot(
        [trade],
        metrics={"sharpe": 1.5, "max_drawdown_pct": 0.02, "managed_exit_share": 0.2, "eod_flatten_share": 0.8},
        funnel_counters={"candidate_pool": 4, "entered": 1},
        rejection_log=[{"gate": "rank_pct_reject", "shadow_r": -0.1}],
        selection_attribution={datetime(2026, 1, 5, tzinfo=timezone.utc).date(): {"candidate_count": 4, "entered_count": 1, "entered_avg_r": 0.6, "skipped_avg_shadow_r": -0.1, "best_skipped_symbol": "BBB", "best_skipped_shadow_r": 0.0, "skipped_beating_worst_entered": 0}},
    )

    assert snapshot["shadow"]["delta_avg_r"] > 0
    assert snapshot["selection"]["crowded_days"] == 1
    assert snapshot["funnel"]["accept_rate"] == 0.25


def test_snapshot_hides_intraday_summary_when_run_has_no_intraday_data():
    trade = TradeRecord(
        strategy="IARIC_PB",
        symbol="AAA",
        direction=Direction.LONG,
        entry_time=datetime(2026, 1, 5, tzinfo=timezone.utc),
        exit_time=datetime(2026, 1, 5, tzinfo=timezone.utc),
        entry_price=10.0,
        exit_price=10.6,
        quantity=100,
        pnl=60.0,
        r_multiple=0.6,
        risk_per_share=1.0,
        commission=0.0,
        slippage=0.0,
        exit_reason="EOD_FLATTEN",
        sector="Tech",
        regime_tier="A",
        metadata={"entry_rsi": 3.0, "entry_rank": 4, "entry_rank_pct": 20.0, "entry_sma_dist_pct": 3.0, "entry_cdd": 2, "entry_gap_pct": -0.3},
    )

    snapshot = compute_pullback_diagnostic_snapshot([trade], candidate_ledger={datetime(2026, 1, 5, tzinfo=timezone.utc).date(): [{"symbol": "AAA", "disposition": "entered"}]})

    assert snapshot["intraday"]["stage_counts"]["watchlist"] == 0


def test_snapshot_exposes_selector_frontier_and_new_diagnostic_sections():
    trade_date = datetime(2026, 1, 5, tzinfo=timezone.utc).date()
    trade = TradeRecord(
        strategy="IARIC_PB",
        symbol="AAA",
        direction=Direction.LONG,
        entry_time=datetime(2026, 1, 5, 14, 55, tzinfo=timezone.utc),
        exit_time=datetime(2026, 1, 5, 15, 30, tzinfo=timezone.utc),
        entry_price=10.0,
        exit_price=10.7,
        quantity=100,
        pnl=70.0,
        r_multiple=0.7,
        risk_per_share=1.0,
        commission=0.0,
        slippage=0.0,
        exit_reason="EOD_FLATTEN",
        sector="Tech",
        regime_tier="A",
        metadata={
            "entry_trigger": "OPENING_RECLAIM",
            "entry_route_family": "OPENING_RECLAIM",
            "route_score": 61.0,
            "daily_signal_score": 58.0,
            "intraday_score": 61.0,
            "reclaim_bars": 3,
            "micropressure_signal": "ACCUMULATE",
            "mfe_r": 1.1,
            "mfe_before_negative_exit_r": 0.0,
            "bars_to_exit": 7,
            "entry_score_component_daily_signal": 20.0,
            "entry_score_component_reclaim": 16.0,
            "entry_score_component_volume": 12.0,
            "entry_score_component_vwap_hold": 8.0,
            "entry_score_component_cpr": 7.0,
            "entry_score_component_speed": 4.0,
            "entry_score_component_context_adjust": 2.5,
            "entry_score_component_micro_penalty": -3.0,
            "entry_score_component_weak_vwap_penalty": -2.0,
            "entry_score_component_rescue_penalty": 0.0,
            "carry_decision_path": "flatten",
        },
    )
    snapshot = compute_pullback_diagnostic_snapshot(
        [trade],
        metrics={"sharpe": 1.4, "max_drawdown_pct": 0.02, "managed_exit_share": 0.5, "eod_flatten_share": 0.5},
        candidate_ledger={
            trade_date: [
                {
                    "trade_date": trade_date,
                    "symbol": "AAA",
                    "disposition": "entered",
                    "intraday_data_available": True,
                    "live_intraday_candidate": True,
                    "actual_r": 0.7,
                    "daily_signal_score": 58.0,
                    "route_score": 61.0,
                    "intraday_score": 61.0,
                    "ready_bar_index": 5,
                    "ready_cpr": 0.72,
                    "ready_volume_ratio": 1.25,
                    "entry_score_threshold": 50.0,
                    "delayed_confirm_score_threshold": 40.0,
                    "ready_min_cpr_threshold": 0.6,
                    "ready_min_volume_ratio_threshold": 0.8,
                    "delayed_confirm_after_bar_threshold": 4,
                    "refinement_route": "OPENING_RECLAIM",
                    "entry_trigger": "OPENING_RECLAIM",
                    "entry_route_family": "OPENING_RECLAIM",
                    "shadow_r": 0.5,
                },
                {
                    "trade_date": trade_date,
                    "symbol": "BBB",
                    "disposition": "flush_stale",
                    "intraday_data_available": True,
                    "live_intraday_candidate": True,
                    "shadow_r": 0.45,
                    "daily_signal_score": 52.0,
                    "intraday_score": 47.0,
                    "ready_bar_index": 3,
                    "ready_cpr": 0.68,
                    "ready_volume_ratio": 1.10,
                    "entry_score_threshold": 50.0,
                    "delayed_confirm_score_threshold": 40.0,
                    "ready_min_cpr_threshold": 0.6,
                    "ready_min_volume_ratio_threshold": 0.8,
                    "delayed_confirm_after_bar_threshold": 4,
                    "refinement_route": "OPENING_RECLAIM",
                    "entry_route_family": "OPENING_RECLAIM",
                    "blocked_by_capacity_reason": "slot_cap",
                    "ready_timestamp": datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc),
                },
            ]
        },
    )

    assert snapshot["selector_frontier"]["gate_rows"]
    flush_row = next(row for row in snapshot["selector_frontier"]["invalidated_rows"] if row["gate"] == "flush_stale")
    assert flush_row["worked_later"] == 1
    assert snapshot["capacity_opportunity"]["blocked_rows"]
    assert snapshot["entry_quality"]["component_rows"]
    assert "route_monotonicity_rows" in snapshot["entry_quality"]
    assert snapshot["management_forensics"]["hold_rows"]
    assert snapshot["exit_replacement"]["carry_audit"]["actually_carried"] == 0
    assert snapshot["alpha_sources"]["route_rows"]


def test_selector_frontier_uses_entered_daily_fallback_only_and_route_scoped_threshold_sweeps():
    trade_date = datetime(2026, 1, 5, tzinfo=timezone.utc).date()
    trades = [
        TradeRecord(
            strategy="IARIC_PB",
            symbol="AAA",
            direction=Direction.LONG,
            entry_time=datetime(2026, 1, 5, 13, 0, tzinfo=timezone.utc),
            exit_time=datetime(2026, 1, 5, 19, 55, tzinfo=timezone.utc),
            entry_price=10.0,
            exit_price=10.4,
            quantity=100,
            pnl=40.0,
            r_multiple=0.4,
            risk_per_share=1.0,
            commission=0.0,
            slippage=0.0,
            exit_reason="EOD_FLATTEN",
            sector="Tech",
            regime_tier="A",
            metadata={"entry_trigger": "OPEN_SCORED_ENTRY", "entry_route_family": "OPEN_SCORED_ENTRY", "mfe_r": 0.6},
        ),
        TradeRecord(
            strategy="IARIC_PB",
            symbol="CCC",
            direction=Direction.LONG,
            entry_time=datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc),
            exit_time=datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc),
            entry_price=20.0,
            exit_price=20.5,
            quantity=100,
            pnl=50.0,
            r_multiple=0.5,
            risk_per_share=1.0,
            commission=0.0,
            slippage=0.0,
            exit_reason="FLOW_REVERSAL",
            sector="Tech",
            regime_tier="A",
            metadata={"entry_trigger": "DELAYED_CONFIRM", "entry_route_family": "DELAYED_CONFIRM", "mfe_r": 0.8},
        ),
        TradeRecord(
            strategy="IARIC_PB",
            symbol="DDD",
            direction=Direction.LONG,
            entry_time=datetime(2026, 1, 5, 15, 5, tzinfo=timezone.utc),
            exit_time=datetime(2026, 1, 5, 18, 5, tzinfo=timezone.utc),
            entry_price=30.0,
            exit_price=30.7,
            quantity=100,
            pnl=70.0,
            r_multiple=0.7,
            risk_per_share=1.0,
            commission=0.0,
            slippage=0.0,
            exit_reason="FLOW_REVERSAL",
            sector="Tech",
            regime_tier="A",
            metadata={"entry_trigger": "OPENING_RECLAIM", "entry_route_family": "OPENING_RECLAIM", "mfe_r": 1.0},
        ),
    ]
    snapshot = compute_pullback_diagnostic_snapshot(
        trades,
        metrics={"sharpe": 1.4, "max_drawdown_pct": 0.02, "managed_exit_share": 0.5, "eod_flatten_share": 0.5},
        candidate_ledger={
            trade_date: [
                {
                    "trade_date": trade_date,
                    "symbol": "AAA",
                    "disposition": "entered",
                        "intraday_data_available": False,
                        "actual_r": 0.4,
                        "shadow_r": 0.2,
                        "daily_signal_score": 64.0,
                        "daily_signal_rank_pct": 18.0,
                        "refinement_route": "OPEN_SCORED_ENTRY",
                        "entry_trigger": "OPEN_SCORED_ENTRY",
                        "entry_route_family": "OPEN_SCORED_ENTRY",
                    },
                    {
                        "trade_date": trade_date,
                        "symbol": "BBB",
                        "disposition": "sector_cap_reject",
                        "intraday_data_available": False,
                        "shadow_r": 0.3,
                        "daily_signal_score": 58.0,
                        "daily_signal_rank_pct": 36.0,
                        "refinement_route": "OPEN_SCORED_ENTRY",
                        "entry_trigger": "OPEN_SCORED_ENTRY",
                        "entry_route_family": "OPEN_SCORED_ENTRY",
                    },
                    {
                        "trade_date": trade_date,
                        "symbol": "CCC",
                        "disposition": "entered",
                        "intraday_data_available": True,
                        "actual_r": 0.5,
                        "shadow_r": 0.4,
                        "daily_signal_score": 59.0,
                        "route_score": 53.0,
                        "intraday_score": 53.0,
                        "ready_bar_index": 5,
                        "ready_cpr": 0.72,
                    "ready_volume_ratio": 1.25,
                    "entry_score_threshold": 55.0,
                    "delayed_confirm_score_threshold": 52.0,
                        "ready_min_cpr_threshold": 0.6,
                        "ready_min_volume_ratio_threshold": 0.8,
                        "delayed_confirm_after_bar_threshold": 4,
                        "refinement_route": "DELAYED_CONFIRM",
                        "entry_trigger": "DELAYED_CONFIRM",
                        "entry_route_family": "DELAYED_CONFIRM",
                    },
                    {
                        "trade_date": trade_date,
                        "symbol": "DDD",
                        "disposition": "entered",
                        "intraday_data_available": True,
                        "actual_r": 0.7,
                        "shadow_r": 0.6,
                        "daily_signal_score": 63.0,
                        "route_score": 61.0,
                        "intraday_score": 61.0,
                        "ready_bar_index": 5,
                        "ready_cpr": 0.75,
                    "ready_volume_ratio": 1.35,
                    "entry_score_threshold": 55.0,
                    "delayed_confirm_score_threshold": 52.0,
                        "ready_min_cpr_threshold": 0.6,
                        "ready_min_volume_ratio_threshold": 0.8,
                        "delayed_confirm_after_bar_threshold": 4,
                        "refinement_route": "OPENING_RECLAIM",
                        "entry_trigger": "OPENING_RECLAIM",
                        "entry_route_family": "OPENING_RECLAIM",
                    },
                    {
                        "trade_date": trade_date,
                        "symbol": "EEE",
                        "disposition": "flush_stale",
                        "intraday_data_available": True,
                        "shadow_r": 0.1,
                        "daily_signal_score": 55.0,
                        "route_score": 47.0,
                        "intraday_score": 47.0,
                        "ready_bar_index": 5,
                        "ready_cpr": 0.65,
                    "ready_volume_ratio": 1.05,
                    "entry_score_threshold": 55.0,
                    "delayed_confirm_score_threshold": 52.0,
                        "ready_min_cpr_threshold": 0.6,
                        "ready_min_volume_ratio_threshold": 0.8,
                        "delayed_confirm_after_bar_threshold": 4,
                        "refinement_route": "OPENING_RECLAIM",
                        "entry_trigger": "OPENING_RECLAIM",
                        "entry_route_family": "OPENING_RECLAIM",
                    },
                ]
            },
        )

    open_scored_missing = next(row for row in snapshot["selector_frontier"]["cohorts"] if row["label"] == "open-scored missing 5m")
    routed_live = next(row for row in snapshot["selector_frontier"]["cohorts"] if row["label"] == "routed 5m accepted")
    assert open_scored_missing["count"] == 1
    assert routed_live["count"] == 2
    entry_sweep = snapshot["selector_frontier"]["threshold_sweeps"]["pb_entry_score_min"]
    delayed_sweep = snapshot["selector_frontier"]["threshold_sweeps"]["pb_delayed_confirm_score_min"]
    active_entry = next(row for row in entry_sweep if row["active"])
    active_delayed = next(row for row in delayed_sweep if row["active"])
    assert active_entry["accepted"] == 1
    assert active_delayed["accepted"] == 1


def test_pullback_precompute_cache_reuses_downstream_settings_only():
    class _Replay:
        pass

    replay = _Replay()
    closes = np.linspace(10.0, 20.0, 80)
    replay._universe = [("AAA", "Tech", "NASDAQ")]
    replay._daily_arrs = {
        "AAA": {
            "close": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "volume": np.full(len(closes), 1_000_000.0),
        }
    }
    replay._daily_didx = {"AAA": ([date(2026, 1, 5)] * len(closes), list(range(len(closes))))}
    replay._daily_flow = {"AAA": np.linspace(-1.0, 1.0, len(closes))}

    base = StrategySettings()
    downstream = replace(base, pb_partial_r=1.5, pb_trail_activate_r=1.75)
    upstream = replace(base, pb_rsi_period=3)

    ind_base, iloc_base, flow_base = _shared_pullback_state(replay, base)
    ind_down, iloc_down, flow_down = _shared_pullback_state(replay, downstream)
    ind_up, iloc_up, flow_up = _shared_pullback_state(replay, upstream)

    assert ind_base is ind_down
    assert iloc_base is iloc_down
    assert flow_base is flow_down
    assert ind_up is not ind_base
    assert iloc_up is iloc_base
    assert flow_up is flow_base


def test_phase_runner_emits_end_of_round_when_final_phase_completes(tmp_path, monkeypatch):
    class _DummyPlugin:
        name = "dummy"
        num_phases = 1
        ultimate_targets = {}
        initial_mutations = {}

        def get_phase_spec(self, phase, state):
            return PhaseSpec(
                focus="test",
                candidates=[],
                gate_criteria_fn=lambda metrics: [GateCriterion("avg_r", 0.0, metrics["avg_r"], True)],
                scoring_weights={},
                hard_rejects={},
                analysis_policy=PhaseAnalysisPolicy(focus_metrics=[]),
                max_rounds=1,
                prune_threshold=0.0,
            )

        def create_evaluate_batch(self, phase, cumulative_mutations, *, scoring_weights=None, hard_rejects=None):
            return lambda candidates, current_mutations: []

        def compute_final_metrics(self, mutations):
            return {"avg_r": 0.1}

        def run_phase_diagnostics(self, phase, state, metrics, greedy_result):
            return "diag"

        def run_enhanced_diagnostics(self, phase, state, metrics, greedy_result):
            return "diag"

        def build_end_of_round_artifacts(self, state):
            return EndOfRoundArtifacts(
                final_diagnostics_text="final diag",
                dimension_reports={"signal_extraction": "ok"},
                overall_verdict="ok",
            )

    monkeypatch.setattr(
        phase_runner_module,
        "run_greedy",
        lambda *args, **kwargs: GreedyResult(
            base_score=1.0,
            final_score=1.0,
            final_mutations={},
            kept_features=[],
            rounds=[],
            final_metrics={"avg_r": 0.1},
            total_candidates=0,
            accepted_count=0,
            elapsed_seconds=0.0,
        ),
    )
    monkeypatch.setattr(
        phase_runner_module,
        "evaluate_gate",
        lambda criteria, greedy_result=None: GateResult(passed=True, criteria=tuple(criteria)),
    )
    monkeypatch.setattr(
        phase_runner_module,
        "analyze_phase",
        lambda *args, **kwargs: PhaseAnalysis(phase=1, recommendation="advance", recommendation_reason="done", report="done"),
    )

    runner = PhaseRunner(plugin=_DummyPlugin(), output_dir=Path(tmp_path))
    state = runner.run_phase(1, runner.load_state())

    assert state.completed_phases == [1]
    assert (Path(tmp_path) / "round_evaluation.txt").exists()
    assert (Path(tmp_path) / "round_final_diagnostics.txt").exists()
