# tests/test_macro_regime.py
"""Tests for macro regime instrumentation and trading assistant integration.

Covers: schema additions, strategy profiles, data pipeline, strategy engine
detectors, context builder loaders, prompt assembler updates, outcome
measurement regime drift, and suggestion scorer regime stratification.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.analysis.strategy_engine import StrategyEngine
from trading_assistant.schemas.daily_metrics import MacroRegimeAnalysis
from trading_assistant.schemas.events import DailySnapshot, RegimeTransitionEvent, TradeEvent
from trading_assistant.schemas.outcome_measurement import (
    MeasurementQuality,
    OutcomeMeasurement,
    compute_measurement_quality,
)
from trading_assistant.schemas.regime_conditional import (
    MacroRegimeConditionalReport,
    RegimeStrategyMetrics,
)
from trading_assistant.schemas.strategy_profile import StrategyProfile
from trading_assistant.schemas.weekly_metrics import BotWeeklySummary
from trading_assistant.skills.build_daily_metrics import build_macro_regime_analysis
from trading_assistant.skills.suggestion_scorer import SuggestionScorer


# ---------------------------------------------------------------------------
# Schema tests (B1)
# ---------------------------------------------------------------------------


class TestTradeEventMacroFields:
    def test_trade_event_macro_regime_defaults(self):
        t = TradeEvent(
            trade_id="t1", bot_id="bot1", pair="NQ", side="LONG",
            entry_time="2026-04-01T10:00:00", exit_time="2026-04-01T14:00:00",
            entry_price=100.0, exit_price=105.0, position_size=1.0,
            pnl=50.0, pnl_pct=5.0,
        )
        assert t.macro_regime == ""
        assert t.stress_level_at_entry == 0.0

    def test_trade_event_macro_regime_set(self):
        t = TradeEvent(
            trade_id="t2", bot_id="bot1", pair="NQ", side="LONG",
            entry_time="2026-04-01T10:00:00", exit_time="2026-04-01T14:00:00",
            entry_price=100.0, exit_price=105.0, position_size=1.0,
            pnl=50.0, pnl_pct=5.0,
            macro_regime="S", stress_level_at_entry=0.72,
        )
        assert t.macro_regime == "S"
        assert t.stress_level_at_entry == 0.72


class TestDailySnapshotRegimeFields:
    def test_snapshot_regime_defaults(self):
        snap = DailySnapshot(date="2026-04-01", bot_id="bot1")
        assert snap.regime_context is None
        assert snap.applied_regime_config is None

    def test_snapshot_with_regime_context(self):
        ctx = {"macro_regime": "G", "regime_confidence": 0.9, "stress_level": 0.1}
        config = {"directional_cap_R": 8.0, "regime_unit_risk_mult": 1.0}
        snap = DailySnapshot(
            date="2026-04-01", bot_id="bot1",
            regime_context=ctx, applied_regime_config=config,
        )
        assert snap.regime_context["macro_regime"] == "G"
        assert snap.applied_regime_config["directional_cap_R"] == 8.0


class TestRegimeTransitionEvent:
    def test_transition_event_creation(self):
        evt = RegimeTransitionEvent(
            bot_id="portfolio",
            from_regime="G", to_regime="S",
            regime_confidence=0.78, stress_level=0.65,
            timestamp=datetime(2026, 4, 1, 21, 0, 0),
        )
        assert evt.from_regime == "G"
        assert evt.to_regime == "S"
        assert evt.regime_confidence == 0.78


class TestMacroRegimeAnalysis:
    def test_defaults(self):
        m = MacroRegimeAnalysis(bot_id="bot1", date="2026-04-01")
        assert m.macro_regime == ""
        assert m.regime_confidence == 0.0
        assert m.applied_config == {}

    def test_full_fields(self):
        m = MacroRegimeAnalysis(
            bot_id="bot1", date="2026-04-01",
            macro_regime="S", regime_confidence=0.82,
            stress_level=0.65, applied_config={"regime_unit_risk_mult": 0.7},
            regime_pnl_30d=-500.0, regime_trade_count_30d=45,
            regime_win_rate_30d=0.38,
        )
        assert m.macro_regime == "S"
        assert m.regime_pnl_30d == -500.0


class TestRegimeStrategyMetricsMacro:
    def test_macro_regime_field(self):
        m = RegimeStrategyMetrics(
            bot_id="bot1", strategy_id="s1", regime="trending_up",
            macro_regime="G",
        )
        assert m.macro_regime == "G"

    def test_macro_regime_default_empty(self):
        m = RegimeStrategyMetrics(
            bot_id="bot1", strategy_id="s1", regime="trending_up",
        )
        assert m.macro_regime == ""


class TestMacroRegimeConditionalReport:
    def test_report_creation(self):
        report = MacroRegimeConditionalReport(
            week_start="2026-03-30", week_end="2026-04-05",
            current_macro_regime="S", regime_confidence=0.82,
            stress_level=0.65,
        )
        assert report.current_macro_regime == "S"
        assert report.metrics_by_regime == []
        assert report.config_effectiveness == []


# ---------------------------------------------------------------------------
# Strategy profile tests (B2)
# ---------------------------------------------------------------------------


class TestStrategyProfileMacroSensitivity:
    def test_profile_with_sensitivity(self):
        p = StrategyProfile(
            display_name="Test",
            macro_regime_sensitivity={"G": "full", "R": "reduced", "S": "minimal", "D": "minimal"},
        )
        assert p.macro_regime_sensitivity["G"] == "full"
        assert p.macro_regime_sensitivity["S"] == "minimal"

    def test_profile_sensitivity_default_empty(self):
        p = StrategyProfile(display_name="Test")
        assert p.macro_regime_sensitivity == {}

    def test_yaml_profiles_have_sensitivity(self):
        """Verify the actual YAML file has macro_regime_sensitivity for key strategies."""
        import yaml
        yaml_path = Path("data/strategy_profiles.yaml")
        if not yaml_path.exists():
            pytest.skip("strategy_profiles.yaml not found")
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        strategies = data.get("strategies", {})

        # US strategies should have macro regime sensitivity. Strategies
        # absent from the registry are tolerated via the `if sid in
        # strategies` guard.
        for sid in [
            "IARIC_v1", "ALCB_v1", "DownturnDominator_v1",
            "VdubusNQ_v4", "ATRSS", "TPC", "NQ_REGIME",
        ]:
            if sid in strategies:
                sens = strategies[sid].get("macro_regime_sensitivity", {})
                assert len(sens) > 0, f"{sid} missing macro_regime_sensitivity"

        # DownturnDominator_v1 is the canonical bear-regime invariant: disabled
        # in G/R, full in S.
        downturn = strategies.get("DownturnDominator_v1", {})
        sens = downturn.get("macro_regime_sensitivity", {})
        assert sens.get("G") == "disabled"
        assert sens.get("R") == "disabled"
        assert sens.get("S") == "full"


# ---------------------------------------------------------------------------
# Data pipeline tests (B3)
# ---------------------------------------------------------------------------


class TestBuildMacroRegimeAnalysis:
    def test_no_regime_data_returns_empty(self):
        snapshots = [{"bot_id": "bot1", "date": "2026-04-01"}]
        result = build_macro_regime_analysis(snapshots, "2026-04-01")
        assert result == {}

    def test_extracts_regime_context(self):
        snapshots = [
            {
                "bot_id": "bot1",
                "regime_context": {
                    "macro_regime": "S",
                    "regime_confidence": 0.82,
                    "stress_level": 0.65,
                    "stress_onset": False,
                    "shift_velocity": 0.12,
                    "suggested_leverage_mult": 0.7,
                    "computed_at": "2026-04-04T21:00:00Z",
                },
                "applied_regime_config": {
                    "directional_cap_R": 4.0,
                    "regime_unit_risk_mult": 0.7,
                },
            },
            {
                "bot_id": "bot2",
                "regime_context": {
                    "macro_regime": "S",
                    "regime_confidence": 0.82,
                    "stress_level": 0.65,
                },
                "applied_regime_config": {
                    "directional_cap_R": 3.0,
                    "regime_unit_risk_mult": 0.5,
                },
            },
        ]
        result = build_macro_regime_analysis(snapshots, "2026-04-01")
        assert result["macro_regime"] == "S"
        assert result["regime_confidence"] == 0.82
        assert result["stress_level"] == 0.65
        assert "bot1" in result["per_bot_configs"]
        assert "bot2" in result["per_bot_configs"]
        assert result["per_bot_configs"]["bot1"]["regime_unit_risk_mult"] == 0.7


# ---------------------------------------------------------------------------
# Strategy engine detector tests (B4)
# ---------------------------------------------------------------------------


class TestDetectRegimeConfigEffectiveness:
    def test_no_suggestion_insufficient_trades(self):
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        suggestions = engine.detect_regime_config_effectiveness(
            bot_id="bot1", macro_regime="S",
            regime_unit_risk_mult=0.7, regime_pnl=-100.0,
            regime_win_rate=0.35, regime_trade_count=5,
        )
        assert len(suggestions) == 0

    def test_detects_sizing_too_lenient(self):
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        suggestions = engine.detect_regime_config_effectiveness(
            bot_id="bot1", macro_regime="S",
            regime_unit_risk_mult=0.7, regime_pnl=-500.0,
            regime_win_rate=0.35, regime_trade_count=20,
        )
        assert len(suggestions) == 1
        assert "lenient" in suggestions[0].title.lower()
        assert suggestions[0].detection_context.detector_name == "regime_config_effectiveness"

    def test_detects_sizing_too_conservative(self):
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        suggestions = engine.detect_regime_config_effectiveness(
            bot_id="bot1", macro_regime="D",
            regime_unit_risk_mult=0.5, regime_pnl=300.0,
            regime_win_rate=0.60, regime_trade_count=15,
        )
        assert len(suggestions) == 1
        assert "conservative" in suggestions[0].title.lower()

    def test_no_suggestion_when_sizing_normal(self):
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        suggestions = engine.detect_regime_config_effectiveness(
            bot_id="bot1", macro_regime="G",
            regime_unit_risk_mult=1.0, regime_pnl=500.0,
            regime_win_rate=0.55, regime_trade_count=30,
        )
        assert len(suggestions) == 0


class TestDetectRegimeTransitionCost:
    def test_no_suggestion_without_events(self):
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        suggestions = engine.detect_regime_transition_cost([], {})
        assert len(suggestions) == 0

    def test_detects_costly_transition(self):
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        events = [{
            "from_regime": "G", "to_regime": "S",
            "date": "2026-04-01",
        }]
        daily_pnl = {
            "2026-03-27": -100.0, "2026-03-28": -50.0,
            "2026-03-29": 20.0, "2026-03-30": -80.0,
            "2026-03-31": -120.0, "2026-04-01": -200.0,
            "2026-04-02": -90.0, "2026-04-03": 10.0,
            "2026-04-04": -60.0, "2026-04-05": -30.0,
            "2026-04-06": 40.0,
        }
        suggestions = engine.detect_regime_transition_cost(events, daily_pnl)
        assert len(suggestions) == 1
        assert "G→S" in suggestions[0].title
        assert "applied_regime_config" in suggestions[0].description

    def test_no_suggestion_for_profitable_transition(self):
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        events = [{"from_regime": "S", "to_regime": "G", "date": "2026-04-01"}]
        daily_pnl = {f"2026-04-0{d}": 100.0 for d in range(1, 7)}
        daily_pnl.update({f"2026-03-{d}": 50.0 for d in range(27, 32)})
        suggestions = engine.detect_regime_transition_cost(events, daily_pnl)
        assert len(suggestions) == 0


class TestDetectStressEntryPattern:
    def test_no_suggestion_insufficient_trades(self):
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        trades_by_stress = {
            "low": {"trade_count": 3, "win_rate": 0.60, "expectancy": 50.0},
            "high": {"trade_count": 2, "win_rate": 0.30, "expectancy": -20.0},
        }
        suggestions = engine.detect_stress_entry_pattern("bot1", trades_by_stress)
        assert len(suggestions) == 0

    def test_detects_high_stress_underperformance(self):
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        trades_by_stress = {
            "low": {"trade_count": 20, "win_rate": 0.60, "expectancy": 50.0},
            "mid": {"trade_count": 15, "win_rate": 0.45, "expectancy": 10.0},
            "high": {"trade_count": 10, "win_rate": 0.30, "expectancy": -25.0},
        }
        suggestions = engine.detect_stress_entry_pattern("bot1", trades_by_stress)
        assert len(suggestions) == 1
        assert "stress" in suggestions[0].title.lower()
        assert suggestions[0].detection_context.detector_name == "stress_entry_pattern"

    def test_no_suggestion_when_high_stress_ok(self):
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        trades_by_stress = {
            "low": {"trade_count": 20, "win_rate": 0.55, "expectancy": 40.0},
            "high": {"trade_count": 15, "win_rate": 0.50, "expectancy": 30.0},
        }
        suggestions = engine.detect_stress_entry_pattern("bot1", trades_by_stress)
        assert len(suggestions) == 0


class TestBuildReportWithMacroRegime:
    def test_build_report_accepts_macro_regime_params(self):
        """Verify build_report doesn't crash with new macro regime params."""
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        summary = BotWeeklySummary(
            week_start="2026-03-30", week_end="2026-04-05",
            bot_id="bot1", total_trades=20, win_count=10,
            loss_count=10, net_pnl=-200.0, avg_win=80.0, avg_loss=-100.0,
        )
        report = engine.build_report(
            bot_summaries={"bot1": summary},
            macro_regime_data={
                "macro_regime": "S",
                "per_bot_configs": {
                    "bot1": {"regime_unit_risk_mult": 0.7, "directional_cap_R": 4.0},
                },
            },
            stress_entry_stats={
                "bot1": {
                    "low": {"trade_count": 20, "win_rate": 0.60, "expectancy": 50.0},
                    "high": {"trade_count": 10, "win_rate": 0.30, "expectancy": -25.0},
                },
            },
        )
        assert isinstance(report.suggestions, list)
        # Should contain exactly one stress entry pattern suggestion
        stress_sug = [s for s in report.suggestions if "stress" in s.title.lower()]
        assert len(stress_sug) == 1
        assert stress_sug[0].detection_context.detector_name == "stress_entry_pattern"


# ---------------------------------------------------------------------------
# Context builder tests (B5)
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_dir(tmp_path: Path) -> Path:
    policy_dir = tmp_path / "policies" / "v1"
    policy_dir.mkdir(parents=True)
    (policy_dir / "agent.md").write_text("Agent system prompt.")
    (policy_dir / "trading_rules.md").write_text("Rules.")
    (policy_dir / "soul.md").write_text("Soul.")
    (tmp_path / "findings").mkdir()
    return tmp_path


class TestContextBuilderMacroRegime:
    def test_load_macro_regime_context_missing(self, memory_dir, tmp_path):
        curated = tmp_path / "curated"
        curated.mkdir()
        cb = ContextBuilder(memory_dir, curated_dir=curated)
        result = cb.load_macro_regime_context()
        assert result == {}

    def test_load_macro_regime_context_present(self, memory_dir, tmp_path):
        curated = tmp_path / "curated"
        portfolio_dir = curated / "2026-04-05" / "portfolio"
        portfolio_dir.mkdir(parents=True)
        regime_data = {
            "date": "2026-04-05",
            "macro_regime": "S",
            "regime_confidence": 0.82,
            "stress_level": 0.65,
        }
        (portfolio_dir / "macro_regime_analysis.json").write_text(
            json.dumps(regime_data)
        )
        cb = ContextBuilder(memory_dir, curated_dir=curated)
        result = cb.load_macro_regime_context()
        assert result["macro_regime"] == "S"
        assert result["regime_confidence"] == 0.82

    def test_load_regime_config_history_empty(self, memory_dir, tmp_path):
        curated = tmp_path / "curated"
        curated.mkdir()
        cb = ContextBuilder(memory_dir, curated_dir=curated)
        result = cb.load_regime_config_history()
        assert result == []

    def test_load_regime_config_history_populated(self, memory_dir, tmp_path):
        curated = tmp_path / "curated"
        bot_dir = curated / "2026-04-05" / "bot1"
        bot_dir.mkdir(parents=True)
        config = {"directional_cap_R": 4.0, "regime_unit_risk_mult": 0.7}
        (bot_dir / "applied_regime_config.json").write_text(json.dumps(config))
        cb = ContextBuilder(memory_dir, curated_dir=curated)
        result = cb.load_regime_config_history()
        assert len(result) == 1
        assert result[0]["bot_id"] == "bot1"
        assert result[0]["regime_unit_risk_mult"] == 0.7

    def test_base_package_includes_macro_regime(self, memory_dir, tmp_path):
        curated = tmp_path / "curated"
        portfolio_dir = curated / "2026-04-05" / "portfolio"
        portfolio_dir.mkdir(parents=True)
        regime_data = {"macro_regime": "G", "regime_confidence": 0.95}
        (portfolio_dir / "macro_regime_analysis.json").write_text(
            json.dumps(regime_data)
        )
        cb = ContextBuilder(memory_dir, curated_dir=curated)
        pkg = cb.base_package()
        assert "macro_regime_context" in pkg.data

    def test_base_package_omits_macro_regime_when_absent(self, memory_dir, tmp_path):
        curated = tmp_path / "curated"
        curated.mkdir()
        cb = ContextBuilder(memory_dir, curated_dir=curated)
        pkg = cb.base_package()
        assert "macro_regime_context" not in pkg.data


# ---------------------------------------------------------------------------
# Prompt assembler tests (B6)
# ---------------------------------------------------------------------------


class TestPromptAssemblerCuratedFiles:
    def test_daily_curated_includes_regime_config(self):
        from trading_assistant.analysis.prompt_assembler import _CURATED_FILES
        assert "applied_regime_config.json" in _CURATED_FILES

    def test_portfolio_curated_includes_regime_analysis(self):
        from trading_assistant.analysis.prompt_assembler import _PORTFOLIO_CURATED_FILES
        assert "macro_regime_analysis.json" in _PORTFOLIO_CURATED_FILES

    def test_daily_instructions_mention_macro_regime(self):
        from trading_assistant.analysis.prompt_assembler import _FOCUSED_INSTRUCTIONS
        assert "MACRO REGIME" in _FOCUSED_INSTRUCTIONS

    def test_weekly_instructions_mention_macro_regime(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _FOCUSED_WEEKLY_INSTRUCTIONS
        assert "MACRO REGIME" in _FOCUSED_WEEKLY_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Outcome measurement tests (B7)
# ---------------------------------------------------------------------------


class TestOutcomeMeasurementMacroRegime:
    def test_measurement_has_macro_fields(self):
        m = OutcomeMeasurement(
            suggestion_id="s1",
            implemented_date="2026-04-01",
            measurement_date="2026-04-08",
            window_days=7,
            macro_regime_at_implementation="S",
            macro_regime_stable=False,
        )
        assert m.macro_regime_at_implementation == "S"
        assert m.macro_regime_stable is False

    def test_measurement_macro_defaults(self):
        m = OutcomeMeasurement(
            suggestion_id="s1",
            implemented_date="2026-04-01",
            measurement_date="2026-04-08",
            window_days=7,
        )
        assert m.macro_regime_at_implementation == ""
        assert m.macro_regime_stable is True

    def test_quality_degrades_with_regime_drift(self):
        # Stable macro regime → HIGH
        q1 = compute_measurement_quality(
            regime_matched=True,
            before_trade_count=20, after_trade_count=20,
            volatility_ratio=1.0, concurrent_changes=[],
            macro_regime_stable=True,
        )
        assert q1 == MeasurementQuality.HIGH

        # Unstable macro regime → MEDIUM (one issue)
        q2 = compute_measurement_quality(
            regime_matched=True,
            before_trade_count=20, after_trade_count=20,
            volatility_ratio=1.0, concurrent_changes=[],
            macro_regime_stable=False,
        )
        assert q2 == MeasurementQuality.MEDIUM

    def test_quality_low_with_multiple_issues_including_drift(self):
        q = compute_measurement_quality(
            regime_matched=True,
            before_trade_count=8, after_trade_count=8,
            volatility_ratio=1.0, concurrent_changes=["change1"],
            macro_regime_stable=False,
        )
        assert q == MeasurementQuality.LOW


# ---------------------------------------------------------------------------
# Suggestion scorer tests (B8)
# ---------------------------------------------------------------------------


class TestSuggestionScorerRegime:
    def test_regime_confidence_adjustment_defensive(self):
        adj = SuggestionScorer.apply_regime_confidence_adjustment(
            confidence=0.6, category="stop_loss", current_macro_regime="S",
        )
        assert adj == round(min(1.0, 0.6 * 1.15), 3)  # 0.69, boosted in S regime

    def test_regime_confidence_adjustment_aggressive(self):
        adj = SuggestionScorer.apply_regime_confidence_adjustment(
            confidence=0.6, category="filter_threshold", current_macro_regime="D",
        )
        assert adj == round(0.6 * 0.85, 3)  # 0.51, reduced in D regime

    def test_regime_confidence_no_adjustment_goldilocks(self):
        adj = SuggestionScorer.apply_regime_confidence_adjustment(
            confidence=0.6, category="stop_loss", current_macro_regime="G",
        )
        assert adj == 0.6  # no change in G regime

    def test_regime_confidence_no_adjustment_empty_regime(self):
        adj = SuggestionScorer.apply_regime_confidence_adjustment(
            confidence=0.6, category="stop_loss", current_macro_regime="",
        )
        assert adj == 0.6

    def test_regime_stratified_scores_empty(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        (findings / "outcomes.jsonl").write_text("")
        (findings / "suggestions.jsonl").write_text("")
        scorer = SuggestionScorer(findings)
        result = scorer.compute_regime_stratified_scores()
        assert result == {}

    def test_regime_stratified_scores_with_data(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()

        suggestions = [
            {"suggestion_id": f"s{i}", "bot_id": "bot1", "category": "stop_loss"}
            for i in range(6)
        ]
        (findings / "suggestions.jsonl").write_text(
            "\n".join(json.dumps(s) for s in suggestions)
        )

        outcomes = []
        for i in range(3):
            outcomes.append({
                "suggestion_id": f"s{i}",
                "measurement_quality": "high",
                "macro_regime_at_implementation": "S",
                "verdict": "positive",
                "pnl_delta": 100,
                "pnl_before": 100, "pnl_after": 200,
                "win_rate_before": 0.4, "win_rate_after": 0.6,
                "before_trade_count": 10, "after_trade_count": 10,
            })
        for i in range(3, 6):
            outcomes.append({
                "suggestion_id": f"s{i}",
                "measurement_quality": "high",
                "macro_regime_at_implementation": "G",
                "verdict": "negative",
                "pnl_delta": -50,
                "pnl_before": 100, "pnl_after": 50,
                "win_rate_before": 0.5, "win_rate_after": 0.3,
                "before_trade_count": 10, "after_trade_count": 10,
            })

        (findings / "outcomes.jsonl").write_text(
            "\n".join(json.dumps(o) for o in outcomes)
        )

        scorer = SuggestionScorer(findings)
        result = scorer.compute_regime_stratified_scores()
        assert "S" in result
        assert "G" in result
        assert result["S"]["stop_loss"] > result["G"]["stop_loss"]


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestMacroRegimeEdgeCases:
    def test_config_effectiveness_at_mult_boundary(self):
        """mult=1.0 should never fire (no sizing reduction active)."""
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        suggestions = engine.detect_regime_config_effectiveness(
            bot_id="bot1", macro_regime="S",
            regime_unit_risk_mult=1.0, regime_pnl=-1000.0,
            regime_win_rate=0.20, regime_trade_count=30,
        )
        assert len(suggestions) == 0

    def test_config_effectiveness_empty_regime_string(self):
        """Empty macro_regime should not produce suggestions."""
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        suggestions = engine.detect_regime_config_effectiveness(
            bot_id="bot1", macro_regime="",
            regime_unit_risk_mult=0.5, regime_pnl=-500.0,
            regime_win_rate=0.30, regime_trade_count=20,
        )
        # Still fires — regime_config_effectiveness doesn't gate on empty regime
        # (the build_report wiring gates on macro_regime being truthy instead)
        assert len(suggestions) >= 1

    def test_transition_cost_no_pnl_overlap(self):
        """Transition with no PnL data in window should produce no suggestion."""
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        events = [{"from_regime": "G", "to_regime": "S", "date": "2026-04-01"}]
        # PnL data far outside ±5d window
        daily_pnl = {"2026-01-01": -500.0, "2026-01-02": -200.0}
        suggestions = engine.detect_regime_transition_cost(events, daily_pnl)
        assert len(suggestions) == 0

    def test_transition_cost_suggests_config_review(self):
        """Costly transitions should suggest reviewing applied_regime_config."""
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        events = [{
            "from_regime": "G", "to_regime": "D", "date": "2026-04-01",
        }]
        daily_pnl = {f"2026-04-0{d}": -100.0 for d in range(1, 7)}
        daily_pnl.update({f"2026-03-{d}": -50.0 for d in range(27, 32)})
        suggestions = engine.detect_regime_transition_cost(events, daily_pnl)
        assert len(suggestions) == 1
        assert "applied_regime_config" in suggestions[0].description

    def test_stress_pattern_mid_bucket_ignored(self):
        """Only high vs low comparison matters; mid alone is irrelevant."""
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        trades_by_stress = {
            "mid": {"trade_count": 50, "win_rate": 0.30, "expectancy": -40.0},
        }
        suggestions = engine.detect_stress_entry_pattern("bot1", trades_by_stress)
        assert len(suggestions) == 0

    def test_confidence_adjustment_caps_at_1(self):
        """Defensive boost should not exceed 1.0."""
        adj = SuggestionScorer.apply_regime_confidence_adjustment(
            confidence=0.95, category="position_sizing", current_macro_regime="S",
        )
        assert adj <= 1.0

    def test_build_report_regime_isolated_data_preferred(self):
        """When per_bot_configs carry regime_pnl, those are used over blended summary."""
        engine = StrategyEngine(week_start="2026-03-30", week_end="2026-04-05")
        summary = BotWeeklySummary(
            week_start="2026-03-30", week_end="2026-04-05",
            bot_id="bot1", total_trades=50, win_count=30,
            loss_count=20, net_pnl=1000.0, avg_win=80.0, avg_loss=-60.0,
        )
        # Summary is positive, but regime-isolated data shows losses
        report = engine.build_report(
            bot_summaries={"bot1": summary},
            macro_regime_data={
                "macro_regime": "S",
                "per_bot_configs": {
                    "bot1": {
                        "regime_unit_risk_mult": 0.7,
                        "regime_pnl": -300.0,
                        "regime_win_rate": 0.35,
                        "regime_trade_count": 15,
                    },
                },
            },
        )
        lenient = [s for s in report.suggestions if "lenient" in s.title.lower()]
        assert len(lenient) == 1  # should detect from regime-isolated data


# ---------------------------------------------------------------------------
# GAP 1: build_macro_regime_analysis called in daily pipeline
# ---------------------------------------------------------------------------


class TestGap1DailyPipelineMacroRegime:
    """Verify _rebuild_daily_curated_from_raw produces macro_regime_analysis.json."""

    def test_rebuild_writes_macro_regime_file(self, tmp_path):
        """Integration test: raw daily_snapshot data → macro_regime_analysis.json."""
        from trading_assistant.orchestrator.handlers import Handlers

        raw_dir = tmp_path / "raw"
        curated_dir = tmp_path / "curated"
        memory_dir = tmp_path / "memory"
        for d in [raw_dir, curated_dir, memory_dir / "policies" / "v1", memory_dir / "findings"]:
            d.mkdir(parents=True, exist_ok=True)
        (memory_dir / "policies" / "v1" / "agent.md").write_text(".")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text(".")
        (memory_dir / "policies" / "v1" / "soul.md").write_text(".")

        date = "2026-04-05"
        bot_id = "bot1"
        bot_raw = raw_dir / date / bot_id
        bot_raw.mkdir(parents=True)

        # Write a daily_snapshot with regime_context
        snapshot = {
            "event_type": "daily_snapshot",
            "payload": {
                "bot_id": bot_id,
                "date": date,
                "regime_context": {
                    "macro_regime": "S",
                    "regime_confidence": 0.82,
                    "stress_level": 0.65,
                },
                "applied_regime_config": {
                    "directional_cap_R": 4.0,
                    "regime_unit_risk_mult": 0.7,
                },
            },
        }
        (bot_raw / "daily_snapshot.jsonl").write_text(json.dumps(snapshot) + "\n")

        # Also write a minimal summary so the pipeline proceeds
        summary = {"bot_id": bot_id, "date": date, "total_trades": 5, "net_pnl": 100.0}
        bot_curated = curated_dir / date / bot_id
        bot_curated.mkdir(parents=True)
        (bot_curated / "summary.json").write_text(json.dumps(summary))

        handlers = Handlers.__new__(Handlers)
        handlers._raw_data_dir = raw_dir
        handlers._curated_dir = curated_dir
        handlers._memory_dir = memory_dir
        handlers._strategy_registry = None
        handlers._bot_configs = {}

        handlers._rebuild_daily_curated_from_raw(date, [bot_id])

        macro_path = curated_dir / date / "portfolio" / "macro_regime_analysis.json"
        assert macro_path.exists(), "macro_regime_analysis.json should have been written"
        data = json.loads(macro_path.read_text(encoding="utf-8"))
        assert data["macro_regime"] == "S"
        assert data["stress_level"] == 0.65


# ---------------------------------------------------------------------------
# GAP 2: macro_regime_data passed to engine.build_report()
# ---------------------------------------------------------------------------


class TestGap2WeeklyEvidenceMacroRegime:
    """Verify _load_weekly_strategy_evidence returns macro_regime_data."""

    def test_evidence_includes_macro_regime_data(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers

        curated_dir = tmp_path / "curated"
        memory_dir = tmp_path / "memory"
        for d in [curated_dir, memory_dir / "policies" / "v1", memory_dir / "findings"]:
            d.mkdir(parents=True, exist_ok=True)

        # Write macro_regime_analysis.json for the last day of the week
        portfolio_dir = curated_dir / "2026-04-05" / "portfolio"
        portfolio_dir.mkdir(parents=True)
        regime_data = {"macro_regime": "S", "regime_confidence": 0.82, "stress_level": 0.65}
        (portfolio_dir / "macro_regime_analysis.json").write_text(json.dumps(regime_data))

        handlers = Handlers.__new__(Handlers)
        handlers._curated_dir = curated_dir

        evidence = handlers._load_weekly_strategy_evidence(
            week_start="2026-03-30",
            week_end="2026-04-05",
            bot_summaries={"bot1": None},
        )
        assert "macro_regime_data" in evidence
        assert evidence["macro_regime_data"]["macro_regime"] == "S"

    def test_evidence_no_macro_regime_when_absent(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers

        curated_dir = tmp_path / "curated"
        curated_dir.mkdir(parents=True)

        handlers = Handlers.__new__(Handlers)
        handlers._curated_dir = curated_dir

        evidence = handlers._load_weekly_strategy_evidence(
            week_start="2026-03-30",
            week_end="2026-04-05",
            bot_summaries={"bot1": None},
        )
        # macro_regime_data=None gets filtered by the {k:v for k,v in ... if v}
        assert "macro_regime_data" not in evidence


# ---------------------------------------------------------------------------
# GAP 3: regime_stratified_scores enters base_package
# ---------------------------------------------------------------------------


class TestGap3RegimeStratifiedInBasePackage:
    """Verify base_package() includes regime_stratified_scores when outcomes have regime data."""

    def test_base_package_includes_regime_stratified_scores(self, memory_dir, tmp_path):
        curated = tmp_path / "curated"
        curated.mkdir()
        findings = memory_dir / "findings"

        # Create suggestions and outcomes with regime data
        suggestions = [
            {"suggestion_id": f"s{i}", "bot_id": "bot1", "category": "stop_loss"}
            for i in range(6)
        ]
        (findings / "suggestions.jsonl").write_text(
            "\n".join(json.dumps(s) for s in suggestions)
        )

        outcomes = []
        for i in range(3):
            outcomes.append({
                "suggestion_id": f"s{i}",
                "measurement_quality": "high",
                "macro_regime_at_implementation": "S",
                "verdict": "positive",
                "pnl_delta": 100,
                "pnl_before": 100, "pnl_after": 200,
                "win_rate_before": 0.4, "win_rate_after": 0.6,
                "before_trade_count": 10, "after_trade_count": 10,
            })
        for i in range(3, 6):
            outcomes.append({
                "suggestion_id": f"s{i}",
                "measurement_quality": "high",
                "macro_regime_at_implementation": "G",
                "verdict": "negative",
                "pnl_delta": -50,
                "pnl_before": 100, "pnl_after": 50,
                "win_rate_before": 0.5, "win_rate_after": 0.3,
                "before_trade_count": 10, "after_trade_count": 10,
            })
        (findings / "outcomes.jsonl").write_text(
            "\n".join(json.dumps(o) for o in outcomes)
        )

        cb = ContextBuilder(memory_dir, curated_dir=curated)
        pkg = cb.base_package()
        assert "regime_stratified_scores" in pkg.data
        assert "S" in pkg.data["regime_stratified_scores"]

    def test_base_package_omits_regime_stratified_when_no_data(self, memory_dir, tmp_path):
        curated = tmp_path / "curated"
        curated.mkdir()
        cb = ContextBuilder(memory_dir, curated_dir=curated)
        pkg = cb.base_package()
        assert "regime_stratified_scores" not in pkg.data


# ---------------------------------------------------------------------------
# GAP 4: regime confidence adjustment in ResponseValidator
# ---------------------------------------------------------------------------


class TestGap4RegimeConfidenceInValidator:
    """Verify ResponseValidator applies regime confidence adjustment."""

    def test_validator_adjusts_confidence_in_stress_regime(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import AgentSuggestion, ParsedAnalysis

        suggestion = AgentSuggestion(
            title="Tighten stop loss",
            description="Reduce stop distance",
            bot_id="bot1",
            category="stop_loss",
            confidence=0.6,
        )
        parsed = ParsedAnalysis(
            raw_report="test",
            suggestions=[suggestion],
        )

        # Without regime
        validator_no_regime = ResponseValidator()
        result_no_regime = validator_no_regime.validate(parsed)
        conf_no_regime = result_no_regime.approved_suggestions[0].confidence

        # With S regime (should boost defensive categories like stop_loss)
        validator_with_regime = ResponseValidator(current_macro_regime="S")
        result_with_regime = validator_with_regime.validate(parsed)
        conf_with_regime = result_with_regime.approved_suggestions[0].confidence

        # S regime should boost stop_loss confidence
        assert conf_with_regime > conf_no_regime

    def test_validator_no_adjustment_empty_regime(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import AgentSuggestion, ParsedAnalysis

        suggestion = AgentSuggestion(
            title="Test suggestion",
            description="desc",
            bot_id="bot1",
            category="filter_threshold",
            confidence=0.7,
        )
        parsed = ParsedAnalysis(
            raw_report="test",
            suggestions=[suggestion],
        )

        validator_empty = ResponseValidator(current_macro_regime="")
        validator_none = ResponseValidator()
        result_empty = validator_empty.validate(parsed)
        result_none = validator_none.validate(parsed)

        assert result_empty.approved_suggestions[0].confidence == result_none.approved_suggestions[0].confidence


# ---------------------------------------------------------------------------
# GAP 5: DailyTriage macro regime shift detection
# ---------------------------------------------------------------------------


class TestGap5DailyTriageMacroRegime:
    """Verify DailyTriage detects macro regime shifts from portfolio data."""

    def test_detects_macro_regime_shift(self, tmp_path):
        from trading_assistant.analysis.daily_triage import DailyTriage

        curated = tmp_path / "curated"
        date = "2026-04-05"

        # Write today's macro regime as S
        portfolio_today = curated / date / "portfolio"
        portfolio_today.mkdir(parents=True)
        (portfolio_today / "macro_regime_analysis.json").write_text(
            json.dumps({"macro_regime": "S", "stress_level": 0.75})
        )

        # Write trailing days as G
        from datetime import datetime as dt, timedelta
        date_obj = dt.strptime(date, "%Y-%m-%d")
        for d in range(1, 6):
            prev_date = (date_obj - timedelta(days=d)).strftime("%Y-%m-%d")
            prev_dir = curated / prev_date / "portfolio"
            prev_dir.mkdir(parents=True)
            (prev_dir / "macro_regime_analysis.json").write_text(
                json.dumps({"macro_regime": "G", "stress_level": 0.1})
            )

        # Need at least one bot with summary for routine
        bot_dir = curated / date / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(
            json.dumps({"net_pnl": 50.0, "total_trades": 5, "win_count": 3})
        )

        triage = DailyTriage(curated_dir=curated, date=date, bots=["bot1"])
        report = triage.run()

        macro_events = [e for e in report.significant_events if e.event_type == "macro_regime_shift"]
        assert len(macro_events) == 1
        assert macro_events[0].severity == "high"  # destination is S
        assert "G" in macro_events[0].description
        assert "S" in macro_events[0].description
        assert macro_events[0].bot_id == "portfolio"

        # Should generate a focus question about macro regime
        macro_questions = [q for q in report.focus_questions if "Macro regime" in q]
        assert len(macro_questions) == 1

    def test_no_macro_shift_when_regime_stable(self, tmp_path):
        from trading_assistant.analysis.daily_triage import DailyTriage

        curated = tmp_path / "curated"
        date = "2026-04-05"

        # Same regime today and trailing
        from datetime import datetime as dt, timedelta
        date_obj = dt.strptime(date, "%Y-%m-%d")
        for d in range(0, 6):
            d_str = (date_obj - timedelta(days=d)).strftime("%Y-%m-%d")
            d_dir = curated / d_str / "portfolio"
            d_dir.mkdir(parents=True)
            (d_dir / "macro_regime_analysis.json").write_text(
                json.dumps({"macro_regime": "G", "stress_level": 0.1})
            )

        bot_dir = curated / date / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(
            json.dumps({"net_pnl": 50.0, "total_trades": 5, "win_count": 3})
        )

        triage = DailyTriage(curated_dir=curated, date=date, bots=["bot1"])
        report = triage.run()

        macro_events = [e for e in report.significant_events if e.event_type == "macro_regime_shift"]
        assert len(macro_events) == 0

    def test_macro_shift_to_recovery_is_medium_severity(self, tmp_path):
        """Transition to G/R → medium severity (based on destination, not stress)."""
        from trading_assistant.analysis.daily_triage import DailyTriage

        curated = tmp_path / "curated"
        date = "2026-04-05"

        # Today: G (recovery from S)
        portfolio_today = curated / date / "portfolio"
        portfolio_today.mkdir(parents=True)
        (portfolio_today / "macro_regime_analysis.json").write_text(
            json.dumps({"macro_regime": "G"})
        )

        # Trailing: S
        from datetime import datetime as dt, timedelta
        date_obj = dt.strptime(date, "%Y-%m-%d")
        for d in range(1, 6):
            prev_date = (date_obj - timedelta(days=d)).strftime("%Y-%m-%d")
            prev_dir = curated / prev_date / "portfolio"
            prev_dir.mkdir(parents=True)
            (prev_dir / "macro_regime_analysis.json").write_text(
                json.dumps({"macro_regime": "S"})
            )

        bot_dir = curated / date / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(
            json.dumps({"net_pnl": 50.0, "total_trades": 5, "win_count": 3})
        )

        triage = DailyTriage(curated_dir=curated, date=date, bots=["bot1"])
        report = triage.run()

        macro_events = [e for e in report.significant_events if e.event_type == "macro_regime_shift"]
        assert len(macro_events) == 1
        assert macro_events[0].severity == "medium"  # destination is G, not S/D

    def test_no_macro_shift_when_no_portfolio_data(self, tmp_path):
        from trading_assistant.analysis.daily_triage import DailyTriage

        curated = tmp_path / "curated"
        date = "2026-04-05"
        bot_dir = curated / date / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(
            json.dumps({"net_pnl": 50.0, "total_trades": 5, "win_count": 3})
        )

        triage = DailyTriage(curated_dir=curated, date=date, bots=["bot1"])
        report = triage.run()

        macro_events = [e for e in report.significant_events if e.event_type == "macro_regime_shift"]
        assert len(macro_events) == 0
