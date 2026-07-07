"""Tests for sim-driven StrategyEngine detectors.

These detectors consume the dumped output of the weekly handler's
simulation skills (FilterSensitivityAnalyzer, CounterfactualSimulator,
ExitStrategySimulator) and emit StrategySuggestion when an edge is detected.
The fixtures here mirror the *real* schema dumps so a regression in shape
is caught here rather than silently producing zero suggestions in prod.
"""
from __future__ import annotations

from trading_assistant.analysis.strategy_engine import StrategyEngine
from trading_assistant.schemas.counterfactual import (
    CounterfactualResult,
    CounterfactualScenario,
    ScenarioType,
)
from trading_assistant.schemas.filter_sensitivity import (
    FilterSensitivityCurve,
    FilterSensitivityReport,
)


def _engine() -> StrategyEngine:
    return StrategyEngine(week_start="2026-05-01", week_end="2026-05-07")


# ---------------------------- Exit sweep detector ----------------------------

def _exit_sweep_dump(
    baseline: float, simulated: float, total_trades: int, name: str = "trailing_stop",
) -> dict:
    """Mimic ExitSweepResult.model_dump(mode='json') shape from skills/exit_strategy_simulator.py."""
    return {
        "bot_id": "bot_a",
        "configs_tested": 1,
        "baseline_pnl": baseline,
        "results": [
            {
                "strategy": {"strategy_type": name, "params": {"trail_pct": 1.5}},
                "total_trades": total_trades,
                "trades_with_data": total_trades,
                "baseline_pnl": baseline,
                "simulated_pnl": simulated,
                "comparisons": [],
            }
        ],
        "best_strategy": {"strategy_type": name, "params": {"trail_pct": 1.5}},
        "best_improvement": simulated - baseline,
    }


def test_exit_sweep_emits_suggestion_when_edge_above_threshold() -> None:
    sweep = _exit_sweep_dump(baseline=100.0, simulated=125.0, total_trades=50)
    suggestions = _engine().detect_better_exit_strategies("bot_a", sweep)
    assert len(suggestions) == 1
    s = suggestions[0]
    assert s.bot_id == "bot_a"
    assert "trailing_stop" in s.title
    assert s.detection_context is not None
    assert s.detection_context.sample_size == 50
    assert s.detection_context.observed_value > 0.10


def test_exit_sweep_silent_when_below_threshold() -> None:
    sweep = _exit_sweep_dump(baseline=100.0, simulated=105.0, total_trades=50)  # only +5%
    assert _engine().detect_better_exit_strategies("bot_a", sweep) == []


def test_exit_sweep_silent_when_sample_size_below_min() -> None:
    sweep = _exit_sweep_dump(baseline=100.0, simulated=200.0, total_trades=5)
    assert _engine().detect_better_exit_strategies("bot_a", sweep) == []


def test_exit_sweep_silent_on_empty_payload() -> None:
    assert _engine().detect_better_exit_strategies("bot_a", {}) == []
    assert _engine().detect_better_exit_strategies("bot_a", {"results": []}) == []


# ---------------------- Filter sensitivity detector --------------------------

def test_filter_sensitivity_emits_when_filter_blocks_more_value_than_saves() -> None:
    """A filter with negative net impact AND blocked winners should fire."""
    report = FilterSensitivityReport(
        bot_id="bot_a",
        date="2026-05-07",
        curves=[
            FilterSensitivityCurve(
                filter_name="rsi_oversold",
                bot_id="bot_a",
                current_block_count=20,
                current_net_impact=-150.0,
                blocked_winners=8,
                blocked_losers=12,
            )
        ],
    )
    suggestions = _engine().detect_filter_sensitivity_findings(
        "bot_a", report.model_dump(mode="json"),
    )
    assert len(suggestions) == 1
    assert "rsi_oversold" in suggestions[0].title
    assert suggestions[0].detection_context.observed_value == -150.0
    assert suggestions[0].detection_context.sample_size == 20


def test_filter_sensitivity_silent_when_net_positive() -> None:
    """Filter that saves more than it costs shouldn't fire."""
    report = FilterSensitivityReport(
        bot_id="bot_a", date="2026-05-07",
        curves=[FilterSensitivityCurve(
            filter_name="vol_spike",
            bot_id="bot_a",
            current_block_count=20,
            current_net_impact=+200.0,
            blocked_winners=2,
            blocked_losers=18,
        )],
    )
    assert _engine().detect_filter_sensitivity_findings(
        "bot_a", report.model_dump(mode="json"),
    ) == []


def test_filter_sensitivity_silent_when_no_winners_blocked() -> None:
    """Pure-safety filter (only blocks losers) should not be flagged."""
    report = FilterSensitivityReport(
        bot_id="bot_a", date="2026-05-07",
        curves=[FilterSensitivityCurve(
            filter_name="hard_stop",
            bot_id="bot_a",
            current_block_count=10,
            current_net_impact=-50.0,  # negative but only because of opportunity cost
            blocked_winners=0,
            blocked_losers=10,
        )],
    )
    assert _engine().detect_filter_sensitivity_findings(
        "bot_a", report.model_dump(mode="json"),
    ) == []


def test_filter_sensitivity_silent_when_below_min_blocks() -> None:
    report = FilterSensitivityReport(
        bot_id="bot_a", date="2026-05-07",
        curves=[FilterSensitivityCurve(
            filter_name="rsi_oversold",
            bot_id="bot_a",
            current_block_count=2,
            current_net_impact=-100.0,
            blocked_winners=2,
            blocked_losers=0,
        )],
    )
    assert _engine().detect_filter_sensitivity_findings(
        "bot_a", report.model_dump(mode="json"),
    ) == []


# ----------------------- Counterfactual gaps detector ------------------------

def _cf_dump(baseline: float, modified: float, base_count: int) -> dict:
    return CounterfactualResult(
        scenario=CounterfactualScenario(
            scenario_type=ScenarioType.ADD_REGIME_GATE,
            description="add bull-only gate",
        ),
        baseline_pnl=baseline,
        modified_pnl=modified,
        baseline_trade_count=base_count,
        modified_trade_count=base_count - 5,
        baseline_win_rate=0.45,
        modified_win_rate=0.55,
    ).model_dump(mode="json")


def test_counterfactual_emits_when_modified_outperforms_baseline() -> None:
    cf = _cf_dump(baseline=200.0, modified=260.0, base_count=40)  # +30%
    suggestions = _engine().detect_counterfactual_gaps("bot_a", cf)
    assert len(suggestions) == 1
    s = suggestions[0]
    assert "add bull-only gate" in s.title or "add_regime_gate" in s.title
    assert s.detection_context.sample_size == 40
    assert s.detection_context.observed_value > 0.10


def test_counterfactual_silent_when_below_threshold() -> None:
    cf = _cf_dump(baseline=200.0, modified=210.0, base_count=40)  # only +5%
    assert _engine().detect_counterfactual_gaps("bot_a", cf) == []


def test_counterfactual_silent_when_below_min_trades() -> None:
    cf = _cf_dump(baseline=200.0, modified=400.0, base_count=5)
    assert _engine().detect_counterfactual_gaps("bot_a", cf) == []


def test_counterfactual_accepts_list_of_results() -> None:
    """Forward-compat: a future caller may pass a list."""
    cf_list = [
        _cf_dump(baseline=200.0, modified=260.0, base_count=40),
        _cf_dump(baseline=100.0, modified=105.0, base_count=40),  # below threshold
    ]
    suggestions = _engine().detect_counterfactual_gaps("bot_a", cf_list)
    assert len(suggestions) == 1


# ------------------------ build_report integration ---------------------------

def test_build_report_threads_sim_outputs_to_detectors() -> None:
    """End-to-end: sims fed through build_report kwargs surface as suggestions."""
    from trading_assistant.schemas.weekly_metrics import BotWeeklySummary

    summary = BotWeeklySummary(
        bot_id="bot_a", week_start="2026-05-01", week_end="2026-05-07",
        total_trades=50, win_count=25, loss_count=25, net_pnl=100.0,
    )
    engine = _engine()
    report = engine.build_report(
        {"bot_a": summary},
        exit_sweep={"bot_a": _exit_sweep_dump(100.0, 125.0, 50)},
    )
    titles = [s.title for s in report.suggestions]
    assert any("Better exit candidate" in t for t in titles)
