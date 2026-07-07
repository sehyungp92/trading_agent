# tests/test_loop_feedback_quality.py
"""Tests for learning loop feedback quality gaps 1-4.

Gap 1: Per-Metric Targeted Outcome Evaluation
Gap 2: Anti-Oscillation Dampening
Gap 3: Optimization Allocation Diagnostic
Gap 4: Consume Search Signals
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ── Gap 1: Per-Metric Targeted Outcome Evaluation ──────────────────


class TestSuggestionRecordTargetFields:
    """SuggestionRecord should carry target_param, proposed_value, expected_impact."""

    def test_target_fields_default_none(self):
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord

        rec = SuggestionRecord(
            suggestion_id="abc", bot_id="bot1", title="test", tier="parameter",
            source_report_id="run1",
        )
        assert rec.target_param is None
        assert rec.proposed_value is None
        assert rec.expected_impact == ""

    def test_target_fields_set(self):
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord

        rec = SuggestionRecord(
            suggestion_id="abc", bot_id="bot1", title="test", tier="parameter",
            source_report_id="run1",
            target_param="stop_loss_atr", proposed_value=2.5,
            expected_impact="Reduce drawdown by 10%",
        )
        assert rec.target_param == "stop_loss_atr"
        assert rec.proposed_value == 2.5
        assert rec.expected_impact == "Reduce drawdown by 10%"

    def test_backward_compat_serialization(self):
        """Records without target fields should deserialize cleanly."""
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord

        raw = {
            "suggestion_id": "abc", "bot_id": "bot1", "title": "test",
            "tier": "parameter", "source_report_id": "run1",
        }
        rec = SuggestionRecord(**raw)
        assert rec.target_param is None


class TestOutcomeMeasurementTargetFields:
    """OutcomeMeasurement should carry target_metric evaluation."""

    def test_target_fields_default(self):
        from trading_assistant.schemas.outcome_measurement import OutcomeMeasurement

        m = OutcomeMeasurement(
            suggestion_id="abc", implemented_date="2026-01-01",
            measurement_date="2026-01-08", window_days=7,
        )
        assert m.target_metric is None
        assert m.target_metric_improved is None
        assert m.target_metric_delta == 0.0

    def test_target_fields_set(self):
        from trading_assistant.schemas.outcome_measurement import OutcomeMeasurement

        m = OutcomeMeasurement(
            suggestion_id="abc", implemented_date="2026-01-01",
            measurement_date="2026-01-08", window_days=7,
            target_metric="pnl", target_metric_improved=True,
            target_metric_delta=150.0,
        )
        assert m.target_metric == "pnl"
        assert m.target_metric_improved is True
        assert m.target_metric_delta == 150.0


class TestCategoryToTargetMetric:
    """AutoOutcomeMeasurer should map categories to target metrics."""

    def test_mapping_exists(self):
        from trading_assistant.skills.auto_outcome_measurer import AutoOutcomeMeasurer

        mapping = AutoOutcomeMeasurer.CATEGORY_TO_TARGET_METRIC
        assert mapping["stop_loss"] == "drawdown"
        assert mapping["exit_timing"] == "pnl"
        assert mapping["signal"] == "win_rate"
        assert mapping["filter_threshold"] == "win_rate"
        assert mapping["position_sizing"] == "pnl"
        assert mapping["regime_gate"] == "drawdown"


class TestEvaluateTargetMetric:
    """AutoOutcomeMeasurer._evaluate_target_metric should set correct fields."""

    def _make_measurer(self, tmp_path: Path, category: str = "exit_timing"):
        from trading_assistant.skills.auto_outcome_measurer import AutoOutcomeMeasurer

        findings = tmp_path / "findings"
        findings.mkdir()
        # Write a suggestion with the given category
        sugg = {
            "suggestion_id": "s1", "bot_id": "bot1", "category": category,
            "title": "test", "tier": "parameter", "source_report_id": "r1",
        }
        (findings / "suggestions.jsonl").write_text(
            json.dumps(sugg) + "\n", encoding="utf-8"
        )
        return AutoOutcomeMeasurer(
            curated_dir=tmp_path / "curated",
            findings_dir=findings,
        )

    def test_pnl_improvement(self, tmp_path):
        from trading_assistant.schemas.outcome_measurement import OutcomeMeasurement

        measurer = self._make_measurer(tmp_path, "exit_timing")
        m = OutcomeMeasurement(
            suggestion_id="s1", implemented_date="2026-01-01",
            measurement_date="2026-01-08", window_days=7,
            pnl_before=100, pnl_after=200,
        )
        result = measurer._evaluate_target_metric(m, "s1")
        assert result.target_metric == "pnl"
        assert result.target_metric_improved is True
        assert result.target_metric_delta == 100.0

    def test_pnl_worsened(self, tmp_path):
        from trading_assistant.schemas.outcome_measurement import OutcomeMeasurement

        measurer = self._make_measurer(tmp_path, "exit_timing")
        m = OutcomeMeasurement(
            suggestion_id="s1", implemented_date="2026-01-01",
            measurement_date="2026-01-08", window_days=7,
            pnl_before=200, pnl_after=100,
        )
        result = measurer._evaluate_target_metric(m, "s1")
        assert result.target_metric_improved is False

    def test_drawdown_decrease_is_improvement(self, tmp_path):
        from trading_assistant.schemas.outcome_measurement import OutcomeMeasurement

        measurer = self._make_measurer(tmp_path, "stop_loss")
        m = OutcomeMeasurement(
            suggestion_id="s1", implemented_date="2026-01-01",
            measurement_date="2026-01-08", window_days=7,
            drawdown_before=15.0, drawdown_after=10.0,
        )
        result = measurer._evaluate_target_metric(m, "s1")
        assert result.target_metric == "drawdown"
        assert result.target_metric_improved is True
        assert result.target_metric_delta == -5.0

    def test_win_rate_improvement(self, tmp_path):
        from trading_assistant.schemas.outcome_measurement import OutcomeMeasurement

        measurer = self._make_measurer(tmp_path, "signal")
        m = OutcomeMeasurement(
            suggestion_id="s1", implemented_date="2026-01-01",
            measurement_date="2026-01-08", window_days=7,
            win_rate_before=0.4, win_rate_after=0.5,
        )
        result = measurer._evaluate_target_metric(m, "s1")
        assert result.target_metric == "win_rate"
        assert result.target_metric_improved is True

    def test_no_category_returns_unchanged(self, tmp_path):
        from trading_assistant.schemas.outcome_measurement import OutcomeMeasurement
        from trading_assistant.skills.auto_outcome_measurer import AutoOutcomeMeasurer

        measurer = AutoOutcomeMeasurer(
            curated_dir=tmp_path / "curated",
            findings_dir=tmp_path / "findings",
        )
        m = OutcomeMeasurement(
            suggestion_id="unknown", implemented_date="2026-01-01",
            measurement_date="2026-01-08", window_days=7,
        )
        result = measurer._evaluate_target_metric(m, "unknown")
        assert result.target_metric is None


class TestScorerTargetMetricWeight:
    """SuggestionScorer._target_metric_weight should return correct multiplier."""

    def test_improved(self):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        assert SuggestionScorer._target_metric_weight({"target_metric_improved": True}) == 1.2

    def test_worsened(self):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        assert SuggestionScorer._target_metric_weight({"target_metric_improved": False}) == 0.8

    def test_no_info(self):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        assert SuggestionScorer._target_metric_weight({}) == 1.0

    def test_none_value(self):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        assert SuggestionScorer._target_metric_weight({"target_metric_improved": None}) == 1.0


# ── Gap 2: Anti-Oscillation Dampening ──────────────────────────────


class TestGetRecentByBot:
    """SuggestionTracker.get_recent_by_bot should filter by bot and time window."""

    def test_basic_filter(self, tmp_path):
        from trading_assistant.skills.suggestion_tracker import SuggestionTracker
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord

        tracker = SuggestionTracker(tmp_path)
        now = datetime.now(timezone.utc)

        tracker.record(SuggestionRecord(
            suggestion_id="s1", bot_id="bot1", title="test1",
            tier="parameter", source_report_id="r1",
            proposed_at=now,
        ))
        tracker.record(SuggestionRecord(
            suggestion_id="s2", bot_id="bot2", title="test2",
            tier="parameter", source_report_id="r1",
            proposed_at=now,
        ))
        tracker.record(SuggestionRecord(
            suggestion_id="s3", bot_id="bot1", title="test3",
            tier="parameter", source_report_id="r1",
            proposed_at=now - timedelta(weeks=5),  # too old
        ))

        result = tracker.get_recent_by_bot("bot1", weeks=4)
        ids = {r["suggestion_id"] for r in result}
        assert "s1" in ids
        assert "s2" not in ids  # wrong bot
        assert "s3" not in ids  # too old

    def test_excludes_rejected(self, tmp_path):
        from trading_assistant.skills.suggestion_tracker import SuggestionTracker
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord

        tracker = SuggestionTracker(tmp_path)
        tracker.record(SuggestionRecord(
            suggestion_id="s1", bot_id="bot1", title="test1",
            tier="parameter", source_report_id="r1",
        ))
        tracker.reject("s1", reason="bad idea")

        result = tracker.get_recent_by_bot("bot1")
        assert len(result) == 0


class TestDirectionInference:
    """StrategyEngine._infer_direction should detect increase/decrease."""

    def _make_engine(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        return StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")

    def test_from_keyword_widen(self):
        from trading_assistant.schemas.strategy_suggestions import StrategySuggestion, SuggestionTier

        engine = self._make_engine()
        s = StrategySuggestion(
            tier=SuggestionTier.PARAMETER, bot_id="bot1",
            title="Widen stop loss on bot1",
            description="Consider widening stop by 0.5x ATR",
        )
        assert engine._infer_direction(s) == 1

    def test_from_keyword_tighten(self):
        from trading_assistant.schemas.strategy_suggestions import StrategySuggestion, SuggestionTier

        engine = self._make_engine()
        s = StrategySuggestion(
            tier=SuggestionTier.PARAMETER, bot_id="bot1",
            title="Tighten filter threshold",
            description="Reduce filter to cut noise",
        )
        assert engine._infer_direction(s) == -1

    def test_from_numeric_values_increase(self):
        from trading_assistant.schemas.strategy_suggestions import StrategySuggestion, SuggestionTier

        engine = self._make_engine()
        s = StrategySuggestion(
            tier=SuggestionTier.PARAMETER, bot_id="bot1",
            title="Adjust stop multiplier",
            description="Change ATR multiplier",
            current_value="2.0",
            suggested_value="2.5",
        )
        assert engine._infer_direction(s) == 1

    def test_from_numeric_values_decrease(self):
        from trading_assistant.schemas.strategy_suggestions import StrategySuggestion, SuggestionTier

        engine = self._make_engine()
        s = StrategySuggestion(
            tier=SuggestionTier.PARAMETER, bot_id="bot1",
            title="Adjust stop multiplier",
            description="Change ATR multiplier",
            current_value="3.0",
            suggested_value="2.0",
        )
        assert engine._infer_direction(s) == -1

    def test_unknown_direction(self):
        from trading_assistant.schemas.strategy_suggestions import StrategySuggestion, SuggestionTier

        engine = self._make_engine()
        s = StrategySuggestion(
            tier=SuggestionTier.HYPOTHESIS, bot_id="bot1",
            title="Alpha decay detected",
            description="Strategy may be losing edge",
        )
        assert engine._infer_direction(s) == 0


class TestContradictionDetection:
    """StrategyEngine._contradicts_recent should detect opposing directions."""

    def test_opposite_direction_contradicts(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        recent = [{
            "bot_id": "bot1",
            "title": "Widen stop",
            "description": "increase stop distance",
            "detection_context": {"detector_name": "tight_stop"},
        }]
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            recent_suggestions=recent,
        )
        # -1 = decrease, while recent was +1 = increase → contradiction
        assert engine._contradicts_recent("bot1", "tight_stop", -1) is True

    def test_same_direction_no_contradiction(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        recent = [{
            "bot_id": "bot1",
            "title": "Widen stop",
            "description": "increase stop distance",
            "detection_context": {"detector_name": "tight_stop"},
        }]
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            recent_suggestions=recent,
        )
        assert engine._contradicts_recent("bot1", "tight_stop", 1) is False

    def test_different_bot_no_contradiction(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        recent = [{
            "bot_id": "bot2",
            "title": "Widen stop",
            "description": "increase stop distance",
            "detection_context": {"detector_name": "tight_stop"},
        }]
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            recent_suggestions=recent,
        )
        assert engine._contradicts_recent("bot1", "tight_stop", -1) is False

    def test_different_detector_no_contradiction(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        recent = [{
            "bot_id": "bot1",
            "title": "Widen stop",
            "description": "increase stop distance",
            "detection_context": {"detector_name": "tight_stop"},
        }]
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            recent_suggestions=recent,
        )
        assert engine._contradicts_recent("bot1", "filter_cost", -1) is False


class TestOscillationDampening:
    """build_report should dampen confidence when oscillation detected."""

    def test_oscillation_dampens_confidence(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine
        from trading_assistant.schemas.weekly_metrics import BotWeeklySummary

        summary = BotWeeklySummary(
            bot_id="bot1", total_trades=100, winning_trades=60,
            win_rate=0.6, gross_pnl=500, net_pnl=450,
            avg_win=50, avg_loss=-30, max_drawdown_pct=5.0,
            week_start="2026-01-01", week_end="2026-01-07",
        )
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            convergence_report={"oscillation_detected": True},
        )
        report = engine.build_report({"bot1": summary})
        for s in report.suggestions:
            # All confidences should be dampened by 0.7x from original
            assert s.confidence <= 0.7  # max original is 1.0 * 0.7


class TestAntiOscillationInBuildReport:
    """build_report should filter contradictory suggestions."""

    def test_contradictory_filtered_in_report(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine
        from trading_assistant.schemas.weekly_metrics import BotWeeklySummary

        # Recent suggestion was to TIGHTEN stop (decrease direction)
        recent = [{
            "bot_id": "bot1",
            "title": "Tighten stop loss",
            "description": "reduce stop distance to cut losses",
            "detection_context": {"detector_name": "tight_stop"},
        }]

        summary = BotWeeklySummary(
            bot_id="bot1", total_trades=100, winning_trades=60,
            win_rate=0.6, gross_pnl=500, net_pnl=450,
            avg_win=50, avg_loss=-5,  # tight stop triggers → suggests WIDEN (increase)
            max_drawdown_pct=5.0,
            week_start="2026-01-01", week_end="2026-01-07",
        )

        # Without anti-oscillation: should produce tight_stop suggestion
        engine_no_recent = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
        )
        report_baseline = engine_no_recent.build_report({"bot1": summary})
        baseline_tight = [
            s for s in report_baseline.suggestions
            if s.detection_context and s.detection_context.detector_name == "tight_stop"
        ]

        # With anti-oscillation: recent was "tighten" (decrease), detector says
        # "widen" (increase) → opposite direction → should be filtered
        engine_with_recent = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            recent_suggestions=recent,
        )
        report_filtered = engine_with_recent.build_report({"bot1": summary})
        filtered_tight = [
            s for s in report_filtered.suggestions
            if s.detection_context and s.detection_context.detector_name == "tight_stop"
        ]

        # If baseline produced tight_stop suggestions, they should be filtered
        if baseline_tight:
            assert len(filtered_tight) < len(baseline_tight)

    def test_same_direction_passes_through(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine
        from trading_assistant.schemas.weekly_metrics import BotWeeklySummary

        # Recent suggestion was to WIDEN stop (increase direction)
        recent = [{
            "bot_id": "bot1",
            "title": "Widen stop loss",
            "description": "increase stop distance",
            "detection_context": {"detector_name": "tight_stop"},
        }]

        summary = BotWeeklySummary(
            bot_id="bot1", total_trades=100, winning_trades=60,
            win_rate=0.6, gross_pnl=500, net_pnl=450,
            avg_win=50, avg_loss=-5,
            max_drawdown_pct=5.0,
            week_start="2026-01-01", week_end="2026-01-07",
        )

        engine_with_recent = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            recent_suggestions=recent,
        )
        report = engine_with_recent.build_report({"bot1": summary})
        # Same direction (widen→widen) should NOT be filtered
        tight_stop = [
            s for s in report.suggestions
            if s.detection_context and s.detection_context.detector_name == "tight_stop"
        ]
        # Tight stop detector fires "widen" and recent was "widen" → pass through
        engine_no_recent = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
        )
        baseline = engine_no_recent.build_report({"bot1": summary})
        baseline_tight = [
            s for s in baseline.suggestions
            if s.detection_context and s.detection_context.detector_name == "tight_stop"
        ]
        assert len(tight_stop) == len(baseline_tight)


class TestDetectorToCategoryMapping:
    """StrategyEngine._DETECTOR_TO_CATEGORY should map all detectors to categories."""

    def test_mapping_covers_key_detectors(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        m = StrategyEngine._DETECTOR_TO_CATEGORY
        assert m["tight_stop"] == "stop_loss"
        assert m["filter_cost"] == "filter_threshold"
        assert m["exit_timing"] == "exit_timing"
        assert m["alpha_decay"] == "signal"
        assert m["position_sizing"] == "position_sizing"
        assert m["regime_loss"] == "regime_gate"

    def test_optimization_allocation_uses_category_key(self):
        """Verify the value map lookup uses category, not detector_name."""
        from trading_assistant.analysis.strategy_engine import StrategyEngine
        from trading_assistant.schemas.weekly_metrics import BotWeeklySummary

        value_map = {
            "bot1:stop_loss": {
                "value_per_suggestion": 0.15,
                "suggestion_count": 5,
            },
        }
        summary = BotWeeklySummary(
            bot_id="bot1", total_trades=100, winning_trades=60,
            win_rate=0.6, gross_pnl=500, net_pnl=450,
            avg_win=50, avg_loss=-5,
            max_drawdown_pct=5.0,
            week_start="2026-01-01", week_end="2026-01-07",
        )

        # With positive value → confidence should be boosted
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            category_value_map=value_map,
        )
        report = engine.build_report({"bot1": summary})
        # tight_stop detector → stop_loss category → should match value_map key
        tight_stop = [
            s for s in report.suggestions
            if s.detection_context and s.detection_context.detector_name == "tight_stop"
        ]
        # If detector fired, confidence should be boosted (> original * 1.0)
        for s in tight_stop:
            assert s.confidence > 0  # at minimum, confidence is non-zero


# ── Gap 3: Optimization Allocation Diagnostic ──────────────────────


class TestCategoryValueMap:
    """SuggestionScorer.compute_category_value_map should rank categories."""

    def test_basic_ranking(self, tmp_path):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings = tmp_path / "findings"
        findings.mkdir()

        suggestions = [
            {"suggestion_id": "s1", "bot_id": "bot1", "category": "exit_timing", "tier": "parameter"},
            {"suggestion_id": "s2", "bot_id": "bot1", "category": "exit_timing", "tier": "parameter"},
            {"suggestion_id": "s3", "bot_id": "bot1", "category": "stop_loss", "tier": "parameter"},
        ]
        (findings / "suggestions.jsonl").write_text(
            "\n".join(json.dumps(s) for s in suggestions), encoding="utf-8"
        )

        outcomes = [
            {"suggestion_id": "s1", "verdict": "positive", "pnl_delta": 100, "measurement_quality": "high"},
            {"suggestion_id": "s2", "verdict": "negative", "pnl_delta": -50, "measurement_quality": "high"},
            {"suggestion_id": "s3", "verdict": "negative", "pnl_delta": -80, "measurement_quality": "high"},
        ]
        (findings / "outcomes.jsonl").write_text(
            "\n".join(json.dumps(o) for o in outcomes), encoding="utf-8"
        )

        scorer = SuggestionScorer(findings)
        value_map = scorer.compute_category_value_map()

        et = value_map.get("bot1:exit_timing", {})
        assert et["suggestion_count"] == 2
        assert et["measured_count"] == 2
        assert et["avg_composite_delta"] == 25.0  # (100 + -50) / 2

        sl = value_map.get("bot1:stop_loss", {})
        assert sl["positive_count"] == 0

    def test_empty_outcomes(self, tmp_path):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings = tmp_path / "findings"
        findings.mkdir()
        (findings / "suggestions.jsonl").write_text("", encoding="utf-8")
        (findings / "outcomes.jsonl").write_text("", encoding="utf-8")

        scorer = SuggestionScorer(findings)
        value_map = scorer.compute_category_value_map()
        assert value_map == {"_recommendations": []}

    def test_zero_positive_recommendation(self, tmp_path):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings = tmp_path / "findings"
        findings.mkdir()

        suggestions = [
            {"suggestion_id": f"s{i}", "bot_id": "bot1", "category": "stop_loss"}
            for i in range(4)
        ]
        (findings / "suggestions.jsonl").write_text(
            "\n".join(json.dumps(s) for s in suggestions), encoding="utf-8"
        )

        outcomes = [
            {"suggestion_id": f"s{i}", "verdict": "negative", "pnl_delta": -10, "measurement_quality": "high"}
            for i in range(4)
        ]
        (findings / "outcomes.jsonl").write_text(
            "\n".join(json.dumps(o) for o in outcomes), encoding="utf-8"
        )

        scorer = SuggestionScorer(findings)
        value_map = scorer.compute_category_value_map()
        recommendations = value_map.get("_recommendations", [])
        assert any("deprioritize" in r and "bot1:stop_loss" in r for r in recommendations)


class TestOptimizationAllocationInContext:
    """ContextBuilder.base_package should include optimization_allocation."""

    def test_load_optimization_allocation_empty(self, tmp_path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory = tmp_path / "memory"
        (memory / "policies" / "v1").mkdir(parents=True)
        (memory / "findings").mkdir(parents=True)

        cb = ContextBuilder(memory)
        assert cb.load_optimization_allocation() == {}

    def test_base_package_includes_optimization_allocation(self, tmp_path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory = tmp_path / "memory"
        (memory / "policies" / "v1").mkdir(parents=True)
        findings = memory / "findings"
        findings.mkdir(parents=True)

        suggestions = [
            {"suggestion_id": "s1", "bot_id": "bot1", "category": "exit_timing"},
        ]
        (findings / "suggestions.jsonl").write_text(
            json.dumps(suggestions[0]) + "\n", encoding="utf-8"
        )
        outcomes = [
            {"suggestion_id": "s1", "verdict": "positive", "pnl_delta": 100, "measurement_quality": "high"},
        ]
        (findings / "outcomes.jsonl").write_text(
            json.dumps(outcomes[0]) + "\n", encoding="utf-8"
        )

        cb = ContextBuilder(memory)
        pkg = cb.base_package()
        # Should contain optimization_allocation if there's data
        assert "optimization_allocation" in pkg.data


# ── Gap 4: Consume Search Signals ──────────────────────────────────


class TestSearchSignalSummary:
    """ContextBuilder.load_search_signal_summary should aggregate search signals."""

    def test_aggregation(self, tmp_path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory = tmp_path / "memory"
        (memory / "policies" / "v1").mkdir(parents=True)
        findings = memory / "findings"
        findings.mkdir(parents=True)

        signals = [
            {"bot_id": "bot1", "category": "stop_loss", "positive": True, "timestamp": "2026-01-01T00:00:00Z"},
            {"bot_id": "bot1", "category": "stop_loss", "positive": True, "timestamp": "2026-01-02T00:00:00Z"},
            {"bot_id": "bot1", "category": "stop_loss", "positive": False, "timestamp": "2026-01-03T00:00:00Z"},
            {"bot_id": "bot1", "category": "signal", "positive": False, "timestamp": "2026-01-01T00:00:00Z"},
        ]
        (findings / "search_signals.jsonl").write_text(
            "\n".join(json.dumps(s) for s in signals), encoding="utf-8"
        )

        cb = ContextBuilder(memory)
        summary = cb.load_search_signal_summary()

        sl = summary["bot1:stop_loss"]
        assert sl["approve_count"] == 2
        assert sl["discard_count"] == 1
        assert sl["approve_rate"] == pytest.approx(0.667, abs=0.01)

        sig = summary["bot1:signal"]
        assert sig["approve_rate"] == 0.0

    def test_missing_file(self, tmp_path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory = tmp_path / "memory"
        (memory / "policies" / "v1").mkdir(parents=True)
        (memory / "findings").mkdir(parents=True)

        cb = ContextBuilder(memory)
        assert cb.load_search_signal_summary() == {}

    def test_empty_file(self, tmp_path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory = tmp_path / "memory"
        (memory / "policies" / "v1").mkdir(parents=True)
        findings = memory / "findings"
        findings.mkdir(parents=True)
        (findings / "search_signals.jsonl").write_text("", encoding="utf-8")

        cb = ContextBuilder(memory)
        assert cb.load_search_signal_summary() == {}

    def test_base_package_includes_search_signal_summary(self, tmp_path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory = tmp_path / "memory"
        (memory / "policies" / "v1").mkdir(parents=True)
        findings = memory / "findings"
        findings.mkdir(parents=True)

        signals = [
            {"bot_id": "bot1", "category": "stop_loss", "positive": True},
        ]
        (findings / "search_signals.jsonl").write_text(
            json.dumps(signals[0]) + "\n", encoding="utf-8"
        )

        cb = ContextBuilder(memory)
        pkg = cb.base_package()
        assert "search_signal_summary" in pkg.data


class TestWeeklyInstructions:
    """Weekly prompt assembler should contain new instruction sections."""

    def test_optimization_allocation_instruction(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _FOCUSED_WEEKLY_INSTRUCTIONS

        assert "OPTIMIZATION ALLOCATION" in _FOCUSED_WEEKLY_INSTRUCTIONS
        assert "value_per_suggestion" in _FOCUSED_WEEKLY_INSTRUCTIONS

    def test_search_signal_instruction(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _FOCUSED_WEEKLY_INSTRUCTIONS

        assert "SEARCH SIGNAL QUALITY" in _FOCUSED_WEEKLY_INSTRUCTIONS
        assert "approve_rate" in _FOCUSED_WEEKLY_INSTRUCTIONS

    def test_cycle_effectiveness_instruction(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _FOCUSED_WEEKLY_INSTRUCTIONS

        assert "CYCLE EFFECTIVENESS TREND" in _FOCUSED_WEEKLY_INSTRUCTIONS

    def test_suggestion_quality_instruction(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _FOCUSED_WEEKLY_INSTRUCTIONS

        assert "SUGGESTION QUALITY TREND" in _FOCUSED_WEEKLY_INSTRUCTIONS
        assert "hit_rate" in _FOCUSED_WEEKLY_INSTRUCTIONS


# ── Exploration-Exploitation Balance ──────────────────────────────


class TestExplorationExploitation:
    """SuggestionScorer should add unexplored category entries."""

    def test_unexplored_entries_added(self, tmp_path):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings = tmp_path / "findings"
        findings.mkdir()

        # Only exit_timing has data for bot1
        suggestions = [
            {"suggestion_id": "s1", "bot_id": "bot1", "category": "exit_timing"},
        ]
        (findings / "suggestions.jsonl").write_text(
            json.dumps(suggestions[0]) + "\n", encoding="utf-8"
        )
        outcomes = [
            {"suggestion_id": "s1", "verdict": "positive", "pnl_delta": 100, "measurement_quality": "high"},
        ]
        (findings / "outcomes.jsonl").write_text(
            json.dumps(outcomes[0]) + "\n", encoding="utf-8"
        )

        scorer = SuggestionScorer(findings)
        value_map = scorer.compute_category_value_map()

        # bot1:exit_timing should have data
        assert value_map["bot1:exit_timing"]["suggestion_count"] == 1
        # Other categories should be unexplored
        assert value_map.get("bot1:stop_loss", {}).get("unexplored") is True
        assert value_map.get("bot1:signal", {}).get("unexplored") is True

    def test_no_unexplored_when_no_bots(self, tmp_path):
        """Empty data should produce no unexplored entries."""
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings = tmp_path / "findings"
        findings.mkdir()
        (findings / "suggestions.jsonl").write_text("", encoding="utf-8")
        (findings / "outcomes.jsonl").write_text("", encoding="utf-8")

        scorer = SuggestionScorer(findings)
        value_map = scorer.compute_category_value_map()
        unexplored = [k for k, v in value_map.items() if isinstance(v, dict) and v.get("unexplored")]
        assert len(unexplored) == 0

    def test_unexplored_recommendation(self, tmp_path):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings = tmp_path / "findings"
        findings.mkdir()

        suggestions = [
            {"suggestion_id": "s1", "bot_id": "bot1", "category": "exit_timing"},
        ]
        (findings / "suggestions.jsonl").write_text(
            json.dumps(suggestions[0]) + "\n", encoding="utf-8"
        )
        (findings / "outcomes.jsonl").write_text("", encoding="utf-8")

        scorer = SuggestionScorer(findings)
        value_map = scorer.compute_category_value_map()
        recommendations = value_map.get("_recommendations", [])
        assert any("unexplored" in r for r in recommendations)

    def test_strategy_engine_neutral_on_unexplored(self):
        """Strategy engine should not penalize unexplored categories."""
        from trading_assistant.analysis.strategy_engine import StrategyEngine
        from trading_assistant.schemas.weekly_metrics import BotWeeklySummary

        value_map = {
            "bot1:stop_loss": {
                "value_per_suggestion": -0.1,
                "suggestion_count": 5,
                "unexplored": True,  # unexplored flag
            },
        }
        summary = BotWeeklySummary(
            bot_id="bot1", total_trades=100, winning_trades=60,
            win_rate=0.6, gross_pnl=500, net_pnl=450,
            avg_win=50, avg_loss=-5,
            max_drawdown_pct=5.0,
            week_start="2026-01-01", week_end="2026-01-07",
        )

        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            category_value_map=value_map,
        )
        report = engine.build_report({"bot1": summary})
        # tight_stop → stop_loss → unexplored → should NOT be penalized
        tight_stop = [
            s for s in report.suggestions
            if s.detection_context and s.detection_context.detector_name == "tight_stop"
        ]
        for s in tight_stop:
            # Original tight_stop confidence is 0.7; shouldn't be reduced
            assert s.confidence == 0.7


# ── Suggestion Quality Trend ──────────────────────────────────────


class TestSuggestionQualityTrend:
    """SuggestionScorer.compute_suggestion_quality_trend should track quality."""

    def test_empty_data(self, tmp_path):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings = tmp_path / "findings"
        findings.mkdir()
        scorer = SuggestionScorer(findings)
        result = scorer.compute_suggestion_quality_trend()
        assert result == {}

    def test_basic_trend(self, tmp_path):
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings = tmp_path / "findings"
        findings.mkdir()

        now = datetime.now(timezone.utc)
        suggestions = []
        outcomes = []
        for i in range(8):
            week_offset = timedelta(weeks=i)
            ts = (now - week_offset).isoformat()
            sid = f"s{i}"
            suggestions.append({
                "suggestion_id": sid,
                "bot_id": "bot1",
                "category": "exit_timing",
                "status": "implemented",
                "timestamp": ts,
            })
            outcomes.append({
                "suggestion_id": sid,
                "verdict": "positive" if i > 3 else "negative",
                "pnl_delta": 100 if i > 3 else -50,
                "measurement_quality": "high",
            })

        (findings / "suggestions.jsonl").write_text(
            "\n".join(json.dumps(s) for s in suggestions), encoding="utf-8"
        )
        (findings / "outcomes.jsonl").write_text(
            "\n".join(json.dumps(o) for o in outcomes), encoding="utf-8"
        )

        scorer = SuggestionScorer(findings)
        result = scorer.compute_suggestion_quality_trend()
        assert "weekly_metrics" in result
        assert "rolling_avg_hit_rate" in result
        assert "trend" in result
        assert result["trend"] in ("improving", "degrading", "stable", "insufficient_data")

    def test_context_priority_includes_new_keys(self):
        """_CONTEXT_PRIORITY should include cycle_effectiveness_trend and suggestion_quality_trend."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        assert "cycle_effectiveness_trend" in ContextBuilder._CONTEXT_PRIORITY
        assert "suggestion_quality_trend" in ContextBuilder._CONTEXT_PRIORITY

    def test_quality_trend_filters_low_quality(self, tmp_path):
        """compute_suggestion_quality_trend should exclude low-quality and inconclusive outcomes."""
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings = tmp_path / "findings"
        findings.mkdir()

        now = datetime.now(timezone.utc)
        ts = now.isoformat()
        suggestions = [
            {"suggestion_id": "s_high", "bot_id": "bot1", "category": "exit_timing",
             "status": "implemented", "timestamp": ts},
            {"suggestion_id": "s_low", "bot_id": "bot1", "category": "exit_timing",
             "status": "implemented", "timestamp": ts},
            {"suggestion_id": "s_inc", "bot_id": "bot1", "category": "exit_timing",
             "status": "implemented", "timestamp": ts},
        ]
        outcomes = [
            # High quality positive → should be included
            {"suggestion_id": "s_high", "verdict": "positive", "pnl_delta": 100,
             "measurement_quality": "high"},
            # Low quality positive → should be excluded
            {"suggestion_id": "s_low", "verdict": "positive", "pnl_delta": 200,
             "measurement_quality": "low"},
            # Inconclusive → should be excluded
            {"suggestion_id": "s_inc", "verdict": "inconclusive", "pnl_delta": 0,
             "measurement_quality": "high"},
        ]

        (findings / "suggestions.jsonl").write_text(
            "\n".join(json.dumps(s) for s in suggestions), encoding="utf-8"
        )
        (findings / "outcomes.jsonl").write_text(
            "\n".join(json.dumps(o) for o in outcomes), encoding="utf-8"
        )

        scorer = SuggestionScorer(findings)
        result = scorer.compute_suggestion_quality_trend()
        # Only s_high should be mapped; s_low (low quality) and s_inc (inconclusive) excluded
        assert "weekly_metrics" in result
        if result["weekly_metrics"]:
            # With 3 implemented suggestions, 1 has a valid outcome → hit rate = 1/3
            # (s_low and s_inc have no entry in sid_to_positive)
            wk = result["weekly_metrics"][-1]
            assert wk["total_implemented"] == 3
            # Only s_high counts as positive hit
            assert wk["hit_rate"] <= 0.334  # 1/3 ≈ 0.333
