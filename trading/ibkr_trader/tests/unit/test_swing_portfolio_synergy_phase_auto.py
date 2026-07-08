from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backtests.shared.auto.cache_keys import build_cache_key
from backtests.shared.auto.replay_bundle import ReplayBundle
from backtests.swing.analysis.metrics import PerformanceMetrics
from backtests.swing.auto.greedy_optimize import (
    _backtest_months,
    _portfolio_synergy_score,
)
from backtests.swing.auto.portfolio_synergy import run_phase_auto_from_latest as phase_auto
from backtests.swing.auto.portfolio_synergy import run_phase_auto_round2 as phase_auto_round2
from backtests.swing.config_unified import UnifiedBacktestConfig


def test_phase_auto_design_uses_controlled_aggressive_latest_swing_artifacts() -> None:
    design = phase_auto._round_design_record("latest_strategy_mutations")

    assert design["initial_equity"] == 50_000.0
    assert design["risk_stance"] == "controlled_aggressive"
    assert design["round"] == "round_1_phase_auto"
    assert design["base_config"] == "latest_strategy_mutations"
    assert "round_3" in design["source_strategy_configs"]["ATRSS"]
    assert "round_5" in design["source_strategy_configs"]["AKC_HELIX"]
    assert set(design["source_strategy_artifacts"]) == {"ATRSS", "AKC_HELIX", "TPC"}
    assert len(design["source_strategy_artifacts"]["ATRSS"]["optimized_config_sha256"]) == 64
    assert len(design["source_strategy_artifacts"]["AKC_HELIX"]["diagnostic_summary_sha256"]) == 64
    helix_summary = json.loads(phase_auto.PHASE_SOURCE_DIAGNOSTICS["AKC_HELIX"].read_text())
    assert (
        design["source_strategy_artifacts"]["AKC_HELIX"]["diagnostic_metrics"]["total_trades"]
        == helix_summary["total_trades"]
    )
    assert "sizing/admission evidence" in design["known_execution_assumptions"][1]
    assert design["thin_layer_assessment"]["portfolio_risk_layer"]["status"] == "thin_over_live_risk_and_coordination"
    assert design["thin_layer_assessment"]["ATRSS"]["status"] == "mixed_not_optimal_thin_layer"
    assert design["thin_layer_assessment"]["AKC_HELIX"]["status"] == "mixed_not_optimal_thin_layer"
    assert design["thin_layer_assessment"]["TPC"]["status"] == "thin_source_replay"
    assert design["replay_architecture"] == phase_auto.REPLAY_ARCHITECTURE
    assert (
        design["live_parity_contract"]["swing_heat_adapter"]
        == "libs.oms.risk.swing_portfolio_adapter.evaluate_swing_entry"
    )
    assert (
        design["live_parity_contract"]["portfolio_rule_checker"]
        == "libs.oms.risk.portfolio_rules.PortfolioRuleChecker"
    )
    assert (
        design["live_parity_contract"]["strategy_coordinator"]
        == "libs.oms.coordination.coordinator.StrategyCoordinator"
    )
    assert [item["strategy_id"] for item in design["diagnostic_assessments"]] == [
        "ATRSS",
        "AKC_HELIX",
        "TPC",
    ]
    assert design["diagnostic_assessments"][1]["latest_round"] == "round_5"
    assert len(design["phases"]) == 5
    assert len(phase_auto.PHASE_SCORING_KWARGS["score_weights"]) <= 7
    assert phase_auto.PHASE_SCORING_KWARGS["max_drawdown_hard_pct"] == 0.16
    assert phase_auto.PHASE_SCORING_KWARGS["min_trades"] == 520
    assert phase_auto.PHASE_SCORING_KWARGS["trade_count_target"] == 620
    assert phase_auto.PHASE_SCORING_KWARGS["min_required_strategy_trades"] == 25
    assert phase_auto.PHASE_SCORING_KWARGS["max_single_strategy_static_pnl_share"] == 0.80
    assert (
        phase_auto.PHASE_SCORING_KWARGS["score_weights"]["alpha_quality"]
        > phase_auto.PHASE_SCORING_KWARGS["score_weights"]["drawdown_quality"]
    )
    assert any(
        "tpc" in name
        for phase in design["phases"]
        for name in phase["candidate_names"]
    )


def test_round2_phase_auto_design_continues_round1_with_guarded_alpha_frequency_search() -> None:
    assert phase_auto_round2.ROUND2_BASE_CONFIG.name == "optimized_config.json"
    assert "round_1" in str(phase_auto_round2.ROUND2_BASE_CONFIG)
    assert len(phase_auto_round2.ROUND2_PHASES) == 5
    assert len(phase_auto_round2.ROUND2_SCORING_KWARGS["score_weights"]) <= 7
    assert phase_auto_round2.ROUND2_SCORING_KWARGS["max_drawdown_hard_pct"] == 0.16
    assert phase_auto_round2.ROUND2_SCORING_KWARGS["min_trades"] == 620
    assert phase_auto_round2.ROUND2_SCORING_KWARGS["trade_count_target"] == 710
    assert phase_auto_round2.ROUND2_SCORING_KWARGS["max_single_strategy_static_pnl_share"] == 0.78
    assert any(
        "dynamic" in phase_name or "risk" in phase_name
        for phase_name, _ in phase_auto_round2.ROUND2_PHASES
    )
    assert any(
        "tpc" in candidate_name
        for _, candidates in phase_auto_round2.ROUND2_PHASES
        for candidate_name, _ in candidates
    )


def test_backtest_months_handles_object_datetime_arrays() -> None:
    timestamps = np.array(
        [np.datetime64("2021-01-01T00:00:00"), np.datetime64("2021-04-02T00:00:00")],
        dtype=object,
    )

    assert _backtest_months(timestamps) > 2.9


def test_unified_atrss_defaults_match_latest_diagnostic_baseline() -> None:
    atrss_config = UnifiedBacktestConfig().build_atrss_config()

    assert atrss_config.flags.stall_exit is False
    assert atrss_config.slippage.commission_per_contract == 1.00


def test_portfolio_synergy_score_uses_custom_alpha_frequency_targets() -> None:
    config = UnifiedBacktestConfig(initial_equity=50_000.0)
    result = SimpleNamespace(
        combined_timestamps=np.array(["2021-01-01", "2026-01-01"], dtype="datetime64[ns]"),
        strategy_results={
            "ATRSS": SimpleNamespace(
                total_trades=260,
                total_r=220.0,
                entry_signals_fired=500,
                entries_accepted_by_portfolio=260,
            ),
            "AKC_HELIX": SimpleNamespace(
                total_trades=400,
                total_r=430.0,
                entry_signals_fired=500,
                entries_accepted_by_portfolio=400,
            ),
            "TPC": SimpleNamespace(
                total_trades=80,
                total_r=40.0,
                entry_signals_fired=130,
                entries_accepted_by_portfolio=80,
            ),
        },
    )
    metrics = PerformanceMetrics(
        total_trades=740,
        profit_factor=3.2,
        max_drawdown_pct=0.16,
        sharpe=2.1,
        net_profit=0.0,
    )
    base_kwargs = {
        "max_drawdown_hard_pct": 0.18,
        "drawdown_comfort_pct": 0.145,
        "min_profit_factor": 2.40,
        "min_trades": 650,
        "required_strategies": ("ATRSS", "AKC_HELIX", "TPC"),
        "min_required_strategy_trades": 50,
    }

    loose = _portfolio_synergy_score(
        metrics,
        config,
        result,
        50_000.0,
        net_profit_override=300_000.0,
        scoring_kwargs={**base_kwargs, "alpha_return_target_pct": 350.0, "trades_per_month_target": 8.0},
    )
    demanding = _portfolio_synergy_score(
        metrics,
        config,
        result,
        50_000.0,
        net_profit_override=300_000.0,
        scoring_kwargs={**base_kwargs, "alpha_return_target_pct": 750.0, "trades_per_month_target": 14.0},
    )

    assert not loose.rejected
    assert not demanding.rejected
    assert demanding.total < loose.total


def test_phase_auto_evaluation_cache_key_is_source_fingerprinted(monkeypatch) -> None:
    plugin = phase_auto.PortfolioSynergyPhasePlugin(
        Path("backtest/data/raw"),
        max_workers=1,
        initial_mutations={"heat_cap_R": 4.25},
        base_source="unit-test-base",
    )
    monkeypatch.setattr(
        plugin,
        "_ensure_bundle",
        lambda: ReplayBundle(
            data=object(),
            cache_key="bundle-key",
            cache_source_fingerprint="swing-source-fingerprint",
        ),
    )

    evaluator = plugin.create_evaluate_batch(
        2,
        {},
        scoring_weights={"alpha_quality": 0.3},
        hard_rejects={"min_trades": 650},
    )

    expected_key = build_cache_key(
        "swing.portfolio_synergy.evaluation",
        source_fingerprint="swing-source-fingerprint",
        extra={
            "phase": 2,
            "score_profile": phase_auto.SCORE_PROFILE,
            "score_components": ["alpha_quality"],
            "scoring_weights": {"alpha_quality": 0.3},
            "hard_rejects": {"min_trades": 650},
            "return_basis": phase_auto.OPTIMIZATION_RETURN_BASIS,
            "initial_equity": phase_auto.STARTING_EQUITY,
            "risk_stance": phase_auto.RISK_STANCE,
            "data_dir": str(Path("backtest/data/raw").resolve()),
        },
    )
    assert evaluator._signature_prefix == f"{plugin.name}:local:{expected_key}"
