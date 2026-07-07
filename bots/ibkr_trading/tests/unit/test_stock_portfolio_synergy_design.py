import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtests.shared.auto.cache_keys import build_cache_key
from backtests.shared.auto.plugin_utils import mutation_signature
from backtests.shared.auto.phase_state import PhaseState
from backtests.stock.auto.portfolio_synergy.core.logic import run_portfolio_replay
from backtests.stock.auto.portfolio_synergy.core.serializers import (
    hydrate_portfolio_state,
    snapshot_portfolio_state,
)
from backtests.stock.auto.portfolio_synergy.core.state import PortfolioActionType
from backtests.stock.auto.portfolio_synergy.evaluator import (
    _stock_portfolio_mtm_metrics,
    _latest_optimized_config_path,
    _latest_strategy_mutation_paths,
    build_effective_portfolio_config,
    replay_trade_streams,
)
from backtests.stock.auto.portfolio_synergy.phase_candidates import (
    BLOCKED_ALPHA_ROUND3_PROFILE,
    ROUND_TARGETS,
    SCORE_WEIGHTS,
    SEED_PORTFOLIO_CONFIG,
    STRATEGY_ORDER,
    get_phase_candidates,
    get_phase_gates,
    get_score_weights,
    phase_summary,
)
from backtests.stock.auto.portfolio_synergy.plugin import StockPortfolioSynergyPlugin
from backtests.stock.auto.portfolio_synergy.round_design import build_round_design
from backtests.stock.auto.portfolio_synergy.scoring import SCORE_COMPONENTS, score_portfolio_metrics
from backtests.stock.models import Direction, TradeRecord


def test_stock_portfolio_synergy_design_uses_latest_two_stock_rounds() -> None:
    design = build_round_design(Path("."))

    assert design["initial_equity"] == 25_000.0
    assert design["risk_stance"] == "aggressive_controlled"
    assert [item["strategy_id"] for item in design["diagnostic_assessments"]] == list(STRATEGY_ORDER)
    assert len(design["seed_portfolio_config"]["strategy_allocations"]) == 2
    assert len(design["scoring_weights"]) == 7
    assert abs(sum(design["scoring_weights"].values()) - 1.0) < 1e-9


def test_stock_seed_is_aggressive_but_drawdown_guarded() -> None:
    rules = SEED_PORTFOLIO_CONFIG["portfolio_rules"]
    allocations = SEED_PORTFOLIO_CONFIG["strategy_allocations"]

    assert rules["heat_cap_R"] >= 6.0
    assert rules["drawdown_tiers"][-1] == (0.13, 0.00)
    assert allocations["IARIC_V5R1"]["priority"] == 0
    assert allocations["IARIC_V5R1"]["unit_risk_pct"] > allocations["ALCB_R3"]["unit_risk_pct"]
    assert ROUND_TARGETS["hard_max_drawdown_pct"] == 0.10


def test_stock_phase_space_prioritizes_alpha_frequency_and_blocked_alpha() -> None:
    summaries = phase_summary()

    assert len(summaries) == 7
    assert SCORE_WEIGHTS["alpha_return"] > SCORE_WEIGHTS["drawdown_control"]
    assert SCORE_WEIGHTS["trade_frequency"] > SCORE_WEIGHTS["profit_factor_quality"]
    assert any(candidate["name"] == "heat_cap_7_0_probe" for candidate in get_phase_candidates(2))
    assert any(candidate["name"] == "blocked_alpha_ranker" for candidate in get_phase_candidates(5))
    assert any(candidate["name"] == "drawdown_tiers_aggressive_controlled" for candidate in get_phase_candidates(6))


def test_stock_blocked_alpha_round3_profile_targets_capacity_under_six_pct_dd() -> None:
    summaries = phase_summary(BLOCKED_ALPHA_ROUND3_PROFILE)
    weights = get_score_weights(BLOCKED_ALPHA_ROUND3_PROFILE)

    assert len(summaries) == 7
    assert len(weights) == 7
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["synergy_capture"] > SCORE_WEIGHTS["synergy_capture"]
    assert get_phase_gates(7, BLOCKED_ALPHA_ROUND3_PROFILE)["hard_max_drawdown_pct"] == 0.06
    assert any(
        candidate["name"] == "strategy_heat_5_7_4_2"
        for candidate in get_phase_candidates(2, BLOCKED_ALPHA_ROUND3_PROFILE)
    )
    assert any(
        candidate["name"] == "drawdown_tiers_5_6_guard"
        for candidate in get_phase_candidates(6, BLOCKED_ALPHA_ROUND3_PROFILE)
    )


def test_stock_portfolio_synergy_score_is_capped_at_seven_components() -> None:
    assert len(SCORE_COMPONENTS) == 7
    assert tuple(SCORE_WEIGHTS) == SCORE_COMPONENTS

    score = score_portfolio_metrics(
        {
            "net_return_pct": 1.5,
            "total_r_per_month": 14.0,
            "active_trades_per_month": 55.0,
            "max_drawdown_pct": 0.07,
            "profit_factor": 2.8,
            "trade_capture_ratio": 0.88,
            "positive_alpha_block_rate": 0.12,
            "candidate_discrimination": 0.65,
            "active_strategy_count": 2.0,
            "max_strategy_trade_share": 0.64,
            "max_strategy_risk_share": 0.66,
            "sharpe": 4.0,
            "positive_slices": 4.0,
        }
    )

    assert not score.rejected
    assert set(score.components) == set(SCORE_COMPONENTS)


def test_stock_portfolio_hard_rejects_do_not_use_aspirational_objectives() -> None:
    plugin = StockPortfolioSynergyPlugin(Path("backtests/stock/data/raw"), max_workers=2)

    phase_2_rejects = plugin._hard_rejects_for_phase(2)
    phase_5_rejects = plugin._hard_rejects_for_phase(5)

    assert "min_active_trades_per_month" not in phase_2_rejects
    assert "max_positive_alpha_block_rate" not in phase_5_rejects
    assert len(SCORE_COMPONENTS) == 7


def test_stock_portfolio_worker_count_is_capped_at_two() -> None:
    plugin = StockPortfolioSynergyPlugin(Path("backtests/stock/data/raw"), max_workers=8)

    assert plugin.max_workers == 2


def test_stock_portfolio_plugin_namespaces_fingerprinted_evaluation_cache(monkeypatch) -> None:
    plugin = StockPortfolioSynergyPlugin(Path("backtests/stock/data/raw"), max_workers=2)
    monkeypatch.setattr(
        plugin,
        "_ensure_bundle",
        lambda: type("Bundle", (), {"cache_source_fingerprint": "stock-portfolio-fp"})(),
    )

    spec = plugin.get_phase_spec(1, state=PhaseState())
    evaluator = plugin.create_evaluate_batch(2, {})

    assert len(spec.scoring_weights or {}) == 7
    assert evaluator._signature_prefix == build_cache_key(
        "stock.portfolio_synergy.evaluation",
        source_fingerprint="stock-portfolio-fp",
        extra={
            "phase": 2,
            "score_components": list(SCORE_COMPONENTS),
            "scoring_weights": {},
            "hard_rejects": {},
            "initial_equity": 25_000.0,
            "start_date": "2024-01-01",
            "end_date": "2026-03-01",
        },
    )


def test_stock_effective_config_applies_nested_mutations() -> None:
    effective = build_effective_portfolio_config(
        {
            "portfolio_rules.heat_cap_R": 7.0,
            "strategy_allocations.ALCB_R3.unit_risk_pct": 0.007,
            "cross_strategy_rules.candidate_rank_mode": "expected_alpha_per_heat",
        }
    )

    assert effective["portfolio_rules"]["heat_cap_R"] == 7.0
    assert effective["strategy_allocations"]["ALCB_R3"]["unit_risk_pct"] == 0.007
    assert effective["cross_strategy_rules"]["candidate_rank_mode"] == "expected_alpha_per_heat"


def test_stock_portfolio_uses_latest_individual_strategy_configs(tmp_path: Path) -> None:
    for strategy in ("alcb", "iaric"):
        for round_num in (1, 3):
            path = tmp_path / "backtests" / "output" / "stock" / strategy / f"round_{round_num}" / "optimized_config.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
    newer_iaric = tmp_path / "backtests" / "output" / "stock" / "iaric" / "round_4" / "optimized_config.json"
    newer_iaric.parent.mkdir(parents=True, exist_ok=True)
    newer_iaric.write_text("{}", encoding="utf-8")

    alcb_path, iaric_path = _latest_strategy_mutation_paths(tmp_path)

    assert alcb_path.as_posix().endswith("stock/alcb/round_3/optimized_config.json")
    assert iaric_path.as_posix().endswith("stock/iaric/round_4/optimized_config.json")


def test_stock_portfolio_prefers_active_manifest_round_over_archived_directory(tmp_path: Path) -> None:
    strategy_root = tmp_path / "backtests" / "output" / "stock" / "alcb"
    for round_num in (3, 4):
        path = strategy_root / f"round_{round_num}" / "optimized_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (strategy_root / "rounds_manifest.json").write_text(
        json.dumps(
            {
                "family": "stock",
                "strategy": "alcb",
                "rounds": [{"round": 3}, {"round": 4, "archived": True}],
            }
        ),
        encoding="utf-8",
    )

    path = _latest_optimized_config_path(tmp_path, "alcb")

    assert path.as_posix().endswith("stock/alcb/round_3/optimized_config.json")


def test_stock_portfolio_synergy_final_metrics_add_total_score_from_cache() -> None:
    plugin = StockPortfolioSynergyPlugin(Path("backtests/stock/data/raw"), max_workers=2)
    mutations = {"risk_stance": "aggressive_controlled"}
    cached_metrics = {
        "net_return_pct": 1.5,
        "total_r_per_month": 14.0,
        "active_trades_per_month": 55.0,
        "max_drawdown_pct": 0.07,
        "profit_factor": 2.8,
        "trade_capture_ratio": 0.88,
        "positive_alpha_block_rate": 0.12,
        "candidate_discrimination": 0.65,
        "active_strategy_count": 2.0,
        "max_strategy_trade_share": 0.64,
        "max_strategy_risk_share": 0.66,
        "sharpe": 4.0,
        "positive_slices": 4.0,
    }
    plugin._metrics_cache[mutation_signature(mutations)] = dict(cached_metrics)

    metrics = plugin.compute_final_metrics(mutations)

    expected = score_portfolio_metrics(cached_metrics, scoring_weights=SCORE_WEIGHTS)
    assert metrics["score_total"] == expected.total
    assert set(metrics) >= {f"score_{key}" for key in SCORE_COMPONENTS}


def _synthetic_stock_portfolio_trades() -> tuple[list[TradeRecord], list[TradeRecord]]:
    start = datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)
    alcb = TradeRecord(
        strategy="ALCB",
        symbol="AAPL",
        direction=Direction.LONG,
        entry_time=start,
        exit_time=start + timedelta(hours=2),
        entry_price=100.0,
        exit_price=101.0,
        quantity=10,
        pnl=100.0,
        r_multiple=0.8,
        risk_per_share=1.0,
        commission=0.0,
        slippage=0.0,
        entry_type="OR_BREAKOUT",
        sector="Technology",
        metadata={"momentum_score": 6, "rvol": 3.2},
    )
    iaric = TradeRecord(
        strategy="IARIC",
        symbol="MSFT",
        direction=Direction.LONG,
        entry_time=start,
        exit_time=start + timedelta(hours=3),
        entry_price=200.0,
        exit_price=203.0,
        quantity=5,
        pnl=150.0,
        r_multiple=1.1,
        risk_per_share=1.0,
        commission=0.0,
        slippage=0.0,
        entry_type="OPEN_SCORED_ENTRY",
        sector="Technology",
        metadata={"daily_signal_score": 82.0, "entry_gap_pct": -1.0},
    )
    return [alcb], [iaric]


def test_stock_portfolio_replay_reports_blocked_positive_alpha() -> None:
    alcb, iaric = _synthetic_stock_portfolio_trades()
    effective = build_effective_portfolio_config({"portfolio_rules.max_total_active_positions": 1})

    metrics = replay_trade_streams(alcb, iaric, effective)

    assert metrics["entries_accepted_by_portfolio"] == 1.0
    assert metrics["entries_blocked_by_portfolio"] == 1.0
    assert metrics["positive_alpha_block_rate"] > 0.0
    assert metrics["blocked_positive_fraction"] == 1.0


def test_stock_portfolio_core_emits_neutral_actions_and_trade_outcomes() -> None:
    alcb, iaric = _synthetic_stock_portfolio_trades()
    effective = build_effective_portfolio_config({"portfolio_rules.max_total_active_positions": 1})

    result = run_portfolio_replay(alcb, iaric, effective)
    action_types = [action.action_type for action in result.actions]

    assert result.metrics["entries_accepted_by_portfolio"] == 1.0
    assert result.metrics["entries_blocked_by_portfolio"] == 1.0
    assert PortfolioActionType.SUBMIT_ENTRY in action_types
    assert PortfolioActionType.BLOCK_ENTRY in action_types
    assert PortfolioActionType.SUBMIT_EXIT in action_types
    assert {event.decision_code for event in result.decisions} == {"ACCEPT_ENTRY", "BLOCK_ENTRY"}
    assert len(result.trade_outcomes) == 1
    assert result.trade_outcomes[0].net_pnl == result.state.accepted_positions[0].pnl
    assert result.replay_architecture == "stock_portfolio_core_live_rule_adapter"


def test_stock_portfolio_mtm_drawdown_marks_open_loser_before_exit() -> None:
    start = datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)
    trade = TradeRecord(
        strategy="ALCB",
        symbol="AAPL",
        direction=Direction.LONG,
        entry_time=start,
        exit_time=start + timedelta(hours=2),
        entry_price=100.0,
        exit_price=99.8,
        quantity=10,
        pnl=-2.0,
        r_multiple=-0.2,
        risk_per_share=1.0,
        commission=0.0,
        slippage=0.0,
        entry_type="OR_BREAKOUT",
        sector="Technology",
    )
    effective = build_effective_portfolio_config(
        {"strategy_allocations.ALCB_R3.unit_risk_pct": 0.01},
        initial_equity=1_000.0,
    )
    result = run_portfolio_replay([trade], [], effective)
    bars = pd.DataFrame(
        {"close": [100.0, 90.0, 99.8]},
        index=pd.DatetimeIndex([start, start + timedelta(hours=1), start + timedelta(hours=2)]),
    )

    mtm = _stock_portfolio_mtm_metrics(
        result.state.accepted_positions,
        initial_equity=1_000.0,
        price_bars_by_symbol={"AAPL": bars},
    )

    assert mtm["risk_basis"] == "bar_close_mark_to_market"
    assert mtm["max_drawdown_pct"] > result.metrics["max_drawdown_pct"]


def test_stock_portfolio_mtm_marks_partial_price_coverage() -> None:
    alcb, iaric = _synthetic_stock_portfolio_trades()
    alcb[0].exit_time = alcb[0].entry_time + timedelta(hours=2)
    iaric[0].exit_time = iaric[0].entry_time + timedelta(hours=2)
    effective = build_effective_portfolio_config({"portfolio_rules.max_total_active_positions": 2})
    result = run_portfolio_replay(alcb, iaric, effective)
    ts = pd.date_range(alcb[0].entry_time, alcb[0].exit_time, freq="5min", tz="UTC")
    bars = pd.DataFrame({"close": [alcb[0].entry_price] * len(ts)}, index=ts)

    mtm = _stock_portfolio_mtm_metrics(
        result.state.accepted_positions,
        initial_equity=effective["initial_equity"],
        price_bars_by_symbol={alcb[0].symbol: bars},
    )

    assert mtm["risk_basis"] == "partial_bar_close_mark_to_market"
    assert mtm["priced_symbol_count"] == 1
    assert iaric[0].symbol in mtm["missing_price_symbols"]


def test_stock_portfolio_commissions_visible_without_double_deduction() -> None:
    alcb, _ = _synthetic_stock_portfolio_trades()
    alcb[0].commission = 1.0
    effective = build_effective_portfolio_config({"portfolio_rules.max_total_active_positions": 1})

    result = run_portfolio_replay(alcb, [], effective)

    outcome = result.trade_outcomes[0]
    assert outcome.commission > 0.0
    assert outcome.net_pnl == result.state.accepted_positions[0].pnl
    assert outcome.gross_pnl == outcome.net_pnl + outcome.commission
    assert result.state.equity == effective["initial_equity"] + outcome.net_pnl


def test_stock_portfolio_core_snapshot_hydrates_typed_state() -> None:
    alcb, iaric = _synthetic_stock_portfolio_trades()
    effective = build_effective_portfolio_config({"portfolio_rules.max_total_active_positions": 1})
    result = run_portfolio_replay(alcb, iaric, effective)

    hydrated = hydrate_portfolio_state(snapshot_portfolio_state(result.state))

    assert hydrated.equity == result.state.equity
    assert len(hydrated.accepted_positions) == len(result.state.accepted_positions)
    assert len(hydrated.blocked_candidates) == len(result.state.blocked_candidates)
    assert hydrated.accepted_positions[0].direction is Direction.LONG
    assert set(hydrated.strategy_recent) == set(STRATEGY_ORDER)
    assert all(recent.maxlen == 60 for recent in hydrated.strategy_recent.values())


def test_stock_portfolio_evaluator_wrapper_uses_core_replay_metrics() -> None:
    alcb, iaric = _synthetic_stock_portfolio_trades()
    effective = build_effective_portfolio_config({"portfolio_rules.max_total_active_positions": 1})

    assert replay_trade_streams(alcb, iaric, effective) == run_portfolio_replay(alcb, iaric, effective).metrics
