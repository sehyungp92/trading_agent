# tests/test_learning_loop_gaps.py
"""Tests for learning loop gap closures (Gaps 1-6)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


from trading_assistant.analysis.response_validator import ResponseValidator
from trading_assistant.schemas.agent_response import AgentPrediction, AgentSuggestion, ParsedAnalysis
from trading_assistant.schemas.convergence import ConvergenceDimension, ConvergenceReport, DimensionStatus
from trading_assistant.schemas.suggestion_scoring import CategoryScore, CategoryScorecard
from trading_assistant.skills.convergence_tracker import ConvergenceTracker
from trading_assistant.skills.suggestion_scorer import SuggestionScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_suggestion(**kwargs):
    defaults = {
        "bot_id": "bot1",
        "title": "Test suggestion",
        "category": "exit_timing",
        "confidence": 0.8,
    }
    defaults.update(kwargs)
    return AgentSuggestion(**defaults)


def _make_prediction(**kwargs):
    defaults = {
        "bot_id": "bot1",
        "metric": "pnl",
        "direction": "improve",
        "confidence": 0.8,
    }
    defaults.update(kwargs)
    return AgentPrediction(**defaults)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _ts_weeks_ago(weeks: int) -> str:
    """Return ISO timestamp N weeks ago."""
    return (datetime.now(timezone.utc) - timedelta(weeks=weeks)).isoformat()


# ===========================================================================
# Gap 2: Temporal Decay on CategoryScorecard
# ===========================================================================


class TestTemporalDecay:
    def test_recent_outcomes_weigh_more(self, tmp_path):
        """Recent positive outcomes should produce higher win_rate than old ones."""
        findings = tmp_path / "findings"
        findings.mkdir()

        # 5 old negative outcomes + 2 recent positive outcomes
        outcomes = []
        for i in range(5):
            outcomes.append({
                "suggestion_id": f"old_{i}",
                "verdict": "negative",
                "pnl_delta": -100,
                "measured_at": _ts_weeks_ago(20 + i),
            })
        for i in range(2):
            outcomes.append({
                "suggestion_id": f"new_{i}",
                "verdict": "positive",
                "pnl_delta": 200,
                "measured_at": _ts_weeks_ago(1),
            })

        suggestions = []
        for i in range(5):
            suggestions.append({
                "suggestion_id": f"old_{i}",
                "bot_id": "bot1",
                "category": "exit_timing",
            })
        for i in range(2):
            suggestions.append({
                "suggestion_id": f"new_{i}",
                "bot_id": "bot1",
                "category": "exit_timing",
            })

        _write_jsonl(findings / "outcomes.jsonl", outcomes)
        _write_jsonl(findings / "suggestions.jsonl", suggestions)

        scorer = SuggestionScorer(findings)
        scorecard = scorer.compute_scorecard()
        score = scorecard.get_score("bot1", "exit_timing")
        assert score is not None
        # With decay, recent positives should push win_rate above raw 2/7 ≈ 0.286
        assert score.win_rate > 0.35, f"Expected win_rate > 0.35, got {score.win_rate}"

    def test_category_recovery_over_time(self, tmp_path):
        """A category with only old negative outcomes should recover (high multiplier)."""
        findings = tmp_path / "findings"
        findings.mkdir()

        # 3 very old negative outcomes (52 weeks ago)
        outcomes = [
            {
                "suggestion_id": f"old_{i}",
                "verdict": "negative",
                "pnl_delta": -100,
                "measured_at": _ts_weeks_ago(52),
            }
            for i in range(3)
        ]
        suggestions = [
            {"suggestion_id": f"old_{i}", "bot_id": "bot1", "category": "signal"}
            for i in range(3)
        ]

        _write_jsonl(findings / "outcomes.jsonl", outcomes)
        _write_jsonl(findings / "suggestions.jsonl", suggestions)

        scorer = SuggestionScorer(findings)
        scorecard = scorer.compute_scorecard()
        score = scorecard.get_score("bot1", "signal")
        assert score is not None
        # Very old negatives should have decayed weights, making multiplier
        # closer to prior (1.0) than to the 0.3 floor
        assert score.confidence_multiplier > 0.6, (
            f"Expected recovery (multiplier > 0.6), got {score.confidence_multiplier}"
        )

    def test_unknown_timestamp_gets_half_weight(self, tmp_path):
        """Outcomes without timestamps should get 0.5 weight."""
        findings = tmp_path / "findings"
        findings.mkdir()

        outcomes = [
            {"suggestion_id": "no_ts", "verdict": "positive", "pnl_delta": 100},
        ]
        suggestions = [
            {"suggestion_id": "no_ts", "bot_id": "bot1", "category": "signal"},
        ]

        _write_jsonl(findings / "outcomes.jsonl", outcomes)
        _write_jsonl(findings / "suggestions.jsonl", suggestions)

        scorer = SuggestionScorer(findings)
        weight = scorer._compute_age_weight(outcomes[0])
        assert weight == 0.5

    def test_decay_weight_recent_is_near_one(self):
        """An outcome from today should have weight close to 1.0."""
        outcome = {"measured_at": datetime.now(timezone.utc).isoformat()}
        weight = SuggestionScorer._compute_age_weight(outcome)
        assert weight > 0.99


# ===========================================================================
# Gap 3: Wire directional_bias into ResponseValidator
# ===========================================================================


class TestDirectionalBias:
    def test_optimistic_bias_reduces_improve_confidence(self):
        """Optimistic bias on 'pnl' should reduce confidence for 'improve' predictions."""
        forecast_meta = {
            "directional_bias": {
                "pnl": {"bias": "optimistic", "gap_pct": 15},
            },
        }
        v = ResponseValidator(forecast_meta=forecast_meta)
        pred = _make_prediction(metric="pnl", direction="improve", confidence=0.8)
        parsed = ParsedAnalysis(predictions=[pred])
        result = v.validate(parsed)
        adjusted = result.approved_predictions[0]
        assert adjusted.confidence < 0.8, (
            f"Expected confidence < 0.8, got {adjusted.confidence}"
        )

    def test_pessimistic_bias_reduces_decline_confidence(self):
        """Pessimistic bias should reduce confidence for 'decline' predictions."""
        forecast_meta = {
            "directional_bias": {
                "pnl": {"bias": "pessimistic", "gap_pct": 10},
            },
        }
        v = ResponseValidator(forecast_meta=forecast_meta)
        pred = _make_prediction(metric="pnl", direction="decline", confidence=0.8)
        parsed = ParsedAnalysis(predictions=[pred])
        result = v.validate(parsed)
        adjusted = result.approved_predictions[0]
        assert adjusted.confidence < 0.8

    def test_balanced_bias_no_effect(self):
        """Balanced bias should not change confidence."""
        forecast_meta = {
            "directional_bias": {
                "pnl": {"bias": "balanced", "gap_pct": 5},
            },
        }
        v = ResponseValidator(forecast_meta=forecast_meta)
        pred = _make_prediction(metric="pnl", direction="improve", confidence=0.8)
        parsed = ParsedAnalysis(predictions=[pred])
        result = v.validate(parsed)
        adjusted = result.approved_predictions[0]
        assert adjusted.confidence == 0.8

    def test_optimistic_bias_no_effect_on_decline(self):
        """Optimistic bias should NOT reduce decline predictions."""
        forecast_meta = {
            "directional_bias": {
                "pnl": {"bias": "optimistic", "gap_pct": 15},
            },
        }
        v = ResponseValidator(forecast_meta=forecast_meta)
        pred = _make_prediction(metric="pnl", direction="decline", confidence=0.8)
        parsed = ParsedAnalysis(predictions=[pred])
        result = v.validate(parsed)
        adjusted = result.approved_predictions[0]
        assert adjusted.confidence == 0.8

    def test_bias_penalty_capped_at_20_percent(self):
        """Gap_pct of 50 should still cap penalty at 0.2 (20%)."""
        forecast_meta = {
            "directional_bias": {
                "pnl": {"bias": "optimistic", "gap_pct": 50},
            },
        }
        v = ResponseValidator(forecast_meta=forecast_meta)
        pred = _make_prediction(metric="pnl", direction="improve", confidence=1.0)
        parsed = ParsedAnalysis(predictions=[pred])
        result = v.validate(parsed)
        adjusted = result.approved_predictions[0]
        assert adjusted.confidence >= 0.8, (
            f"Penalty should be capped at 20%, got conf={adjusted.confidence}"
        )


# ===========================================================================
# Gap 4: Discovery Context in Prompt Instructions
# ===========================================================================


class TestDiscoveryInstructions:
    def test_daily_instructions_contain_discovery_section(self):
        from trading_assistant.analysis.prompt_assembler import _FOCUSED_INSTRUCTIONS
        assert "DISCOVERIES" in _FOCUSED_INSTRUCTIONS
        assert "discovery agent" in _FOCUSED_INSTRUCTIONS.lower()

    def test_weekly_instructions_contain_discovery_section(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _FOCUSED_WEEKLY_INSTRUCTIONS
        assert "DISCOVERIES AND STRATEGY IDEAS" in _FOCUSED_WEEKLY_INSTRUCTIONS
        assert "strategy_ideas" in _FOCUSED_WEEKLY_INSTRUCTIONS

    def test_daily_instructions_contain_convergence_section(self):
        from trading_assistant.analysis.prompt_assembler import _FOCUSED_INSTRUCTIONS
        assert "CONVERGENCE STATUS" in _FOCUSED_INSTRUCTIONS

    def test_weekly_instructions_contain_convergence_section(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _FOCUSED_WEEKLY_INSTRUCTIONS
        assert "CONVERGENCE STATUS" in _FOCUSED_WEEKLY_INSTRUCTIONS


# ===========================================================================
# Gap 5: Strategy Suggestion Pre-Validation
# ===========================================================================


class TestStrategyPreValidation:
    def test_record_suggestions_skips_poor_category(self, tmp_path):
        """_record_suggestions should skip suggestions in categories with poor track record."""
        from unittest.mock import MagicMock

        from trading_assistant.schemas.strategy_suggestions import StrategySuggestion, SuggestionTier

        scorecard = CategoryScorecard(scores=[
            CategoryScore(
                bot_id="bot1",
                category="exit_timing",
                win_rate=0.2,
                sample_size=10,
                confidence_multiplier=0.5,
            ),
        ])

        suggestion = StrategySuggestion(
            tier=SuggestionTier.PARAMETER,
            bot_id="bot1",
            title="Bad suggestion",
            description="Should be skipped",
        )

        # Create a minimal Handlers-like object to test _record_suggestions
        from trading_assistant.orchestrator.handlers import Handlers

        tracker = MagicMock()
        tracker.record.return_value = True
        stream = MagicMock()

        handlers = Handlers(
            agent_runner=MagicMock(),
            event_stream=stream,
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
            suggestion_tracker=tracker,
        )

        result = handlers._record_suggestions(
            [suggestion], "run-1", category_scorecard=scorecard,
        )
        # Should be empty because category has poor track record
        assert len(result) == 0
        tracker.record.assert_not_called()

    def test_record_suggestions_allows_good_category(self, tmp_path):
        """_record_suggestions should allow suggestions in good categories."""
        from unittest.mock import MagicMock

        from trading_assistant.schemas.strategy_suggestions import StrategySuggestion, SuggestionTier

        scorecard = CategoryScorecard(scores=[
            CategoryScore(
                bot_id="bot1",
                category="exit_timing",
                win_rate=0.6,
                sample_size=10,
                confidence_multiplier=0.9,
            ),
        ])

        suggestion = StrategySuggestion(
            tier=SuggestionTier.PARAMETER,
            bot_id="bot1",
            title="Good suggestion",
            description="Should pass",
        )

        from trading_assistant.orchestrator.handlers import Handlers

        tracker = MagicMock()
        tracker.record.return_value = True
        stream = MagicMock()

        handlers = Handlers(
            agent_runner=MagicMock(),
            event_stream=stream,
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
            suggestion_tracker=tracker,
        )

        result = handlers._record_suggestions(
            [suggestion], "run-1", category_scorecard=scorecard,
        )
        assert len(result) == 1
        tracker.record.assert_called_once()


# ===========================================================================
# Gap 6: Per-Detector Confidence Calibration
# ===========================================================================


class TestDetectorConfidence:
    def test_compute_detector_confidence(self, tmp_path):
        """Compute per-detector confidence from outcome data."""
        findings = tmp_path / "findings"
        findings.mkdir()

        suggestions = [
            {
                "suggestion_id": f"s{i}",
                "bot_id": "bot1",
                "category": "exit_timing",
                "detection_context": {"detector_name": "tight_stop"},
            }
            for i in range(6)
        ]
        # 4 positive, 2 negative → should give multiplier > 0.6
        outcomes = []
        for i in range(4):
            outcomes.append({
                "suggestion_id": f"s{i}",
                "verdict": "positive",
                "pnl_delta": 100,
                "measured_at": _ts_weeks_ago(1),
            })
        for i in range(4, 6):
            outcomes.append({
                "suggestion_id": f"s{i}",
                "verdict": "negative",
                "pnl_delta": -50,
                "measured_at": _ts_weeks_ago(1),
            })

        _write_jsonl(findings / "suggestions.jsonl", suggestions)
        _write_jsonl(findings / "outcomes.jsonl", outcomes)

        scorer = SuggestionScorer(findings)
        confidence = scorer.compute_detector_confidence()
        assert "tight_stop" in confidence
        assert confidence["tight_stop"] > 0.6

    def test_detector_confidence_empty_data(self, tmp_path):
        """Empty data should return empty dict."""
        findings = tmp_path / "findings"
        findings.mkdir()
        scorer = SuggestionScorer(findings)
        assert scorer.compute_detector_confidence() == {}

    def test_detector_confidence_applied_in_strategy_engine(self):
        """Detector confidence should adjust suggestion confidence in build_report."""
        from trading_assistant.schemas.weekly_metrics import BotWeeklySummary
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        engine = StrategyEngine(
            week_start="2026-03-01",
            week_end="2026-03-07",
            detector_confidence={"tight_stop": 0.5},
        )

        summary = BotWeeklySummary(
            bot_id="bot1",
            week_start="2026-03-01",
            week_end="2026-03-07",
            total_trades=20,
            win_rate=0.45,
            avg_win=100.0,
            avg_loss=-10.0,  # loss/win ratio = 0.1, triggers tight_stop
            pnl=500,
        )

        report = engine.build_report({"bot1": summary})
        # Find the tight_stop suggestion
        tight_stop_sug = [
            s for s in report.suggestions
            if s.detection_context and s.detection_context.detector_name == "tight_stop"
        ]
        assert tight_stop_sug, "Expected tight_stop detector to fire"
        # Confidence should be reduced by the 0.5 multiplier
        assert tight_stop_sug[0].confidence < 0.7, (
            f"Expected reduced confidence, got {tight_stop_sug[0].confidence}"
        )

    def test_detector_confidence_no_multiplier_preserves_confidence(self):
        """Unknown detectors should keep original confidence (multiplier=1.0)."""
        from trading_assistant.schemas.weekly_metrics import BotWeeklySummary
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        engine = StrategyEngine(
            week_start="2026-03-01",
            week_end="2026-03-07",
            detector_confidence={"other_detector": 0.5},  # tight_stop not in map
        )

        summary = BotWeeklySummary(
            bot_id="bot1",
            week_start="2026-03-01",
            week_end="2026-03-07",
            total_trades=20,
            win_rate=0.45,
            avg_win=100.0,
            avg_loss=-10.0,
            pnl=500,
        )

        report = engine.build_report({"bot1": summary})
        tight_stop_sug = [
            s for s in report.suggestions
            if s.detection_context and s.detection_context.detector_name == "tight_stop"
        ]
        assert tight_stop_sug, "Expected tight_stop detector to fire"
        # Should keep original confidence of 0.7
        assert tight_stop_sug[0].confidence == 0.7


# ===========================================================================
# Gap 1: Loop Convergence Tracker
# ===========================================================================


class TestConvergenceTracker:
    def test_improving_trend(self, tmp_path):
        """Steady positive and increasing deltas should report IMPROVING."""
        findings = tmp_path / "findings"
        findings.mkdir()

        # Use dict form (production shape: {bot_id: delta})
        ledger = [
            {"composite_delta": {"bot1": 0.05 + i * 0.02}, "week": f"2026-W{i:02d}"}
            for i in range(8)
        ]
        _write_jsonl(findings / "learning_ledger.jsonl", ledger)
        forecasts = [
            {"accuracy": 0.4 + i * 0.05, "week": f"2026-W{i:02d}"}
            for i in range(8)
        ]
        _write_jsonl(findings / "forecast_history.jsonl", forecasts)

        tracker = ConvergenceTracker(findings)
        report = tracker.compute_report(weeks=12)
        composite_dim = next(d for d in report.dimensions if d.name == "composite_scores")
        assert composite_dim.status == DimensionStatus.IMPROVING

    def test_degrading_trend(self, tmp_path):
        """Steady negative and decreasing deltas should report DEGRADING."""
        findings = tmp_path / "findings"
        findings.mkdir()

        ledger = [
            {"composite_delta": -0.05 - i * 0.02, "week": f"2026-W{i:02d}"}
            for i in range(8)
        ]
        _write_jsonl(findings / "learning_ledger.jsonl", ledger)

        tracker = ConvergenceTracker(findings)
        report = tracker.compute_report(weeks=12)
        composite_dim = next(d for d in report.dimensions if d.name == "composite_scores")
        assert composite_dim.status == DimensionStatus.DEGRADING

    def test_oscillating_trend(self, tmp_path):
        """Alternating positive/negative deltas should report OSCILLATING."""
        findings = tmp_path / "findings"
        findings.mkdir()

        ledger = [
            {"composite_delta": 0.05 if i % 2 == 0 else -0.05}
            for i in range(8)
        ]
        _write_jsonl(findings / "learning_ledger.jsonl", ledger)

        tracker = ConvergenceTracker(findings)
        report = tracker.compute_report(weeks=12)
        composite_dim = next(d for d in report.dimensions if d.name == "composite_scores")
        assert composite_dim.status == DimensionStatus.OSCILLATING

    def test_insufficient_data(self, tmp_path):
        """Too few weeks should report INSUFFICIENT_DATA."""
        findings = tmp_path / "findings"
        findings.mkdir()

        ledger = [{"composite_delta": 0.02}]
        _write_jsonl(findings / "learning_ledger.jsonl", ledger)

        tracker = ConvergenceTracker(findings)
        report = tracker.compute_report(weeks=12)
        composite_dim = next(d for d in report.dimensions if d.name == "composite_scores")
        assert composite_dim.status == DimensionStatus.INSUFFICIENT_DATA

    def test_overall_synthesis(self, tmp_path):
        """Overall status should be synthesized from dimensions."""
        findings = tmp_path / "findings"
        findings.mkdir()

        # Improving composites + improving predictions → overall improving
        ledger = [
            {"composite_delta": 0.02 + i * 0.005}
            for i in range(8)
        ]
        forecasts = [
            {"accuracy": 0.5 + i * 0.02}
            for i in range(8)
        ]
        outcomes = [
            {
                "suggestion_id": f"s{i}",
                "verdict": "positive",
                "pnl_delta": 100,
                "measured_at": _ts_weeks_ago(i),
            }
            for i in range(8)
        ]

        _write_jsonl(findings / "learning_ledger.jsonl", ledger)
        _write_jsonl(findings / "forecast_history.jsonl", forecasts)
        _write_jsonl(findings / "outcomes.jsonl", outcomes)

        tracker = ConvergenceTracker(findings)
        report = tracker.compute_report(weeks=12)
        # prediction_accuracy has slope=0.02 (IMPROVING), composites slope=0.005 (STABLE)
        # outcome_ratio/scorecard may be INSUFFICIENT or STABLE
        # Overall: improving > degrading → IMPROVING
        assert report.overall_status == DimensionStatus.IMPROVING
        assert report.recommendation != ""

    def test_empty_findings(self, tmp_path):
        """Empty findings dir should return all INSUFFICIENT_DATA."""
        findings = tmp_path / "findings"
        findings.mkdir()

        tracker = ConvergenceTracker(findings)
        report = tracker.compute_report(weeks=12)
        assert report.overall_status == DimensionStatus.INSUFFICIENT_DATA
        assert all(
            d.status == DimensionStatus.INSUFFICIENT_DATA
            for d in report.dimensions
        )

    def test_oscillation_detected_flag(self, tmp_path):
        """oscillation_detected should be True when any dimension oscillates."""
        findings = tmp_path / "findings"
        findings.mkdir()

        ledger = [
            {"composite_delta": 0.05 if i % 2 == 0 else -0.05}
            for i in range(8)
        ]
        _write_jsonl(findings / "learning_ledger.jsonl", ledger)

        tracker = ConvergenceTracker(findings)
        report = tracker.compute_report(weeks=12)
        assert report.oscillation_detected is True

    def test_convergence_schema(self):
        """ConvergenceReport schema should serialize correctly."""
        report = ConvergenceReport(
            overall_status=DimensionStatus.IMPROVING,
            dimensions=[
                ConvergenceDimension(
                    name="test",
                    status=DimensionStatus.IMPROVING,
                    trend_value=0.02,
                    window_weeks=8,
                    detail="test detail",
                ),
            ],
            oscillation_detected=False,
            weeks_analyzed=12,
            recommendation="System converging",
        )
        data = report.model_dump(mode="json")
        assert data["overall_status"] == "improving"
        assert data["dimensions"][0]["status"] == "improving"

    def test_context_builder_priority_includes_convergence(self):
        """convergence_report should be in _CONTEXT_PRIORITY."""
        from trading_assistant.analysis.context_builder import ContextBuilder
        assert "convergence_report" in ContextBuilder._CONTEXT_PRIORITY

    def test_recommendation_for_degrading(self, tmp_path):
        """Degrading status should generate review recommendation."""
        findings = tmp_path / "findings"
        findings.mkdir()

        ledger = [{"composite_delta": -0.03 - i * 0.01} for i in range(6)]
        forecasts = [{"accuracy": 0.5 - i * 0.03} for i in range(6)]
        _write_jsonl(findings / "learning_ledger.jsonl", ledger)
        _write_jsonl(findings / "forecast_history.jsonl", forecasts)

        tracker = ConvergenceTracker(findings)
        report = tracker.compute_report(weeks=12)
        assert "review" in report.recommendation.lower() or "degrad" in report.recommendation.lower()

    def test_loop_balance_dimension_present(self, tmp_path):
        """compute_report should return 5 dimensions including loop_balance."""
        findings = tmp_path / "findings"
        findings.mkdir()
        _write_jsonl(findings / "learning_ledger.jsonl", [])

        tracker = ConvergenceTracker(findings)
        report = tracker.compute_report(weeks=12)
        dim_names = [d.name for d in report.dimensions]
        assert "loop_balance" in dim_names
        assert len(report.dimensions) == 5

    def test_loop_balance_improving(self, tmp_path):
        """Loop balance with converging inner/outer rates should be IMPROVING."""
        findings = tmp_path / "findings"
        findings.mkdir()

        # Inner and outer rates start divergent and converge over time
        ledger = [
            {
                "inner_total_outcomes": 10, "inner_positive_outcomes": 8,
                "outer_total_outcomes": 10, "outer_positive_outcomes": 2 + i,
            }
            for i in range(6)
        ]
        _write_jsonl(findings / "learning_ledger.jsonl", ledger)

        tracker = ConvergenceTracker(findings)
        report = tracker.compute_report(weeks=12)
        loop_dim = next(d for d in report.dimensions if d.name == "loop_balance")
        assert loop_dim.status == DimensionStatus.IMPROVING

    def test_loop_balance_detects_imbalance(self, tmp_path):
        """100% inner / 0% outer should show low balance score, not 50%."""
        findings = tmp_path / "findings"
        findings.mkdir()

        # Inner perfect, outer zero — consistently imbalanced
        ledger = [
            {
                "inner_total_outcomes": 5, "inner_positive_outcomes": 5,
                "outer_total_outcomes": 5, "outer_positive_outcomes": 0,
            }
            for _ in range(6)
        ]
        _write_jsonl(findings / "learning_ledger.jsonl", ledger)

        tracker = ConvergenceTracker(findings)
        report = tracker.compute_report(weeks=12)
        loop_dim = next(d for d in report.dimensions if d.name == "loop_balance")
        # balance_score = 1.0 - abs(1.0 - 0.0) = 0.0 — should be STABLE at 0
        assert loop_dim.status == DimensionStatus.STABLE
        assert loop_dim.trend_value == 0.0  # flat at 0

    def test_loop_balance_single_loop_skipped(self, tmp_path):
        """Weeks with only one loop should be skipped (INSUFFICIENT_DATA)."""
        findings = tmp_path / "findings"
        findings.mkdir()

        ledger = [
            {
                "inner_total_outcomes": 5, "inner_positive_outcomes": 3,
                "outer_total_outcomes": 0, "outer_positive_outcomes": 0,
            }
            for _ in range(6)
        ]
        _write_jsonl(findings / "learning_ledger.jsonl", ledger)

        tracker = ConvergenceTracker(findings)
        report = tracker.compute_report(weeks=12)
        loop_dim = next(d for d in report.dimensions if d.name == "loop_balance")
        assert loop_dim.status == DimensionStatus.INSUFFICIENT_DATA


# ===========================================================================
# Cycle Effectiveness Score
# ===========================================================================


class TestCycleEffectiveness:
    def test_backward_compat_default_zero(self):
        """New fields should default to 0.0."""
        from trading_assistant.schemas.learning_ledger import LearningLedgerEntry

        entry = LearningLedgerEntry(week_start="2026-01-01", week_end="2026-01-07")
        assert entry.cycle_effectiveness == 0.0
        assert entry.inner_suggestions_proposed == 0
        assert entry.outer_suggestions_proposed == 0

    def test_compute_cycle_effectiveness_basic(self, tmp_path):
        """compute_cycle_effectiveness should return value in [0, 1]."""
        from trading_assistant.skills.learning_ledger import LearningLedger

        findings = tmp_path / "findings"
        findings.mkdir()

        ledger = LearningLedger(findings)
        score = ledger.compute_cycle_effectiveness(
            composite_delta={"bot1": 0.05, "bot2": -0.02},
            suggestions_proposed=10,
            suggestions_implemented=5,
            lessons=["lesson1", "lesson2", "lesson3"],
            week_start="2026-01-01",
            week_end="2026-01-07",
        )
        assert 0.0 <= score <= 1.0

    def test_cycle_effectiveness_zero_inputs(self, tmp_path):
        """Should handle zero inputs gracefully."""
        from trading_assistant.skills.learning_ledger import LearningLedger

        findings = tmp_path / "findings"
        findings.mkdir()

        ledger = LearningLedger(findings)
        score = ledger.compute_cycle_effectiveness(
            composite_delta={},
            suggestions_proposed=0,
            suggestions_implemented=0,
            lessons=[],
            week_start="2026-01-01",
            week_end="2026-01-07",
        )
        # sigmoid(0)=0.5 for improvement, 0 for conversion, 0 for quality, 0 for yield
        assert 0.0 <= score <= 1.0
        # Should be approximately 0.125 (0.5/4)
        assert score < 0.2

    def test_cycle_effectiveness_high_performance(self, tmp_path):
        """High performance should give high effectiveness."""
        from trading_assistant.skills.learning_ledger import LearningLedger

        findings = tmp_path / "findings"
        findings.mkdir()

        # Write positive outcomes for the week
        _write_jsonl(findings / "outcomes.jsonl", [
            {"suggestion_id": f"s{i}", "verdict": "positive", "measured_at": "2026-01-05T00:00:00Z"}
            for i in range(5)
        ])

        ledger = LearningLedger(findings)
        score = ledger.compute_cycle_effectiveness(
            composite_delta={"bot1": 0.1, "bot2": 0.08},
            suggestions_proposed=10,
            suggestions_implemented=8,
            lessons=["a", "b", "c", "d", "e"],
            week_start="2026-01-01",
            week_end="2026-01-07",
        )
        assert score > 0.6


# ===========================================================================
# Loop Source Classification
# ===========================================================================


class TestLoopSourceClassification:
    def test_inner_loop_has_detector(self):
        """Suggestion with detection_context.detector_name → inner."""
        from trading_assistant.skills.learning_cycle import LearningCycle

        s = {"detection_context": {"detector_name": "tight_stop"}}
        assert LearningCycle.classify_suggestion_source(s) == "inner"

    def test_outer_loop_no_detector(self):
        """Suggestion without detection_context → outer."""
        from trading_assistant.skills.learning_cycle import LearningCycle

        assert LearningCycle.classify_suggestion_source({}) == "outer"
        assert LearningCycle.classify_suggestion_source({"detection_context": {}}) == "outer"
        assert LearningCycle.classify_suggestion_source({"detection_context": None}) == "outer"

    def test_ledger_loop_fields_persist(self, tmp_path):
        """Loop source fields should persist in JSONL."""
        from trading_assistant.skills.learning_ledger import LearningLedger

        findings = tmp_path / "findings"
        findings.mkdir()

        ledger = LearningLedger(findings)
        entry = ledger.record_week(
            week_start="2026-01-01",
            week_end="2026-01-07",
            inner_suggestions_proposed=3,
            outer_suggestions_proposed=5,
            inner_positive_outcomes=2,
            outer_positive_outcomes=1,
            inner_total_outcomes=4,
            outer_total_outcomes=3,
        )
        assert entry.inner_suggestions_proposed == 3
        assert entry.outer_suggestions_proposed == 5
        assert entry.inner_positive_outcomes == 2

        # Verify persistence
        latest = ledger.get_latest()
        assert latest is not None
        assert latest.inner_suggestions_proposed == 3


# ===========================================================================
# Experiment Failure Blacklist
# ===========================================================================


class TestCrossWeekOutcomeAttribution:
    def test_cross_week_outcome_attribution(self, tmp_path):
        """Outcomes for older suggestions should use correct source, not default to outer."""
        from unittest.mock import MagicMock
        from trading_assistant.skills.learning_cycle import LearningCycle

        # Suggestion from 2 weeks ago (inner loop)
        old_suggestion = {
            "suggestion_id": "old_s1",
            "bot_id": "bot1",
            "timestamp": "2026-01-01T00:00:00Z",
            "detection_context": {"detector_name": "tight_stop"},
        }
        # Suggestion from current week (outer loop)
        new_suggestion = {
            "suggestion_id": "new_s1",
            "bot_id": "bot1",
            "timestamp": "2026-01-15T00:00:00Z",
        }

        tracker = MagicMock()
        tracker.load_all.return_value = [old_suggestion, new_suggestion]

        # Outcome measured this week for the OLD suggestion
        findings = tmp_path / "findings"
        findings.mkdir(parents=True)
        _write_jsonl(findings / "outcomes.jsonl", [
            {
                "suggestion_id": "old_s1",
                "verdict": "positive",
                "measured_at": "2026-01-16T00:00:00Z",
            },
        ])

        cycle = LearningCycle(
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            bots=["bot1"],
            suggestion_tracker=tracker,
        )

        result = cycle._classify_loop_sources("2026-01-14", "2026-01-20", findings)
        inner_proposed, outer_proposed, inner_pos, outer_pos, inner_total, outer_total = result

        # old_s1 was an inner-loop suggestion (has detector) — outcome should be inner
        assert inner_total == 1, f"Expected inner_total=1, got {inner_total}"
        assert inner_pos == 1, f"Expected inner_pos=1, got {inner_pos}"
        # outer should have 0 outcomes (new_s1 has no outcome)
        assert outer_total == 0, f"Expected outer_total=0, got {outer_total}"

        # Proposal counts: only new_s1 is in current week, and it's outer
        assert inner_proposed == 0
        assert outer_proposed == 1

    def test_inconclusive_outcomes_excluded_from_loop_counts(self, tmp_path):
        """INCONCLUSIVE outcomes should not inflate inner/outer totals in ledger."""
        from unittest.mock import MagicMock
        from trading_assistant.skills.learning_cycle import LearningCycle

        suggestion = {
            "suggestion_id": "s1",
            "bot_id": "bot1",
            "timestamp": "2026-01-15T00:00:00Z",
            "detection_context": {"detector_name": "tight_stop"},
        }

        tracker = MagicMock()
        tracker.load_all.return_value = [suggestion]

        findings = tmp_path / "findings"
        findings.mkdir(parents=True)
        _write_jsonl(findings / "outcomes.jsonl", [
            {
                "suggestion_id": "s1",
                "verdict": "positive",
                "measured_at": "2026-01-16T00:00:00Z",
            },
            {
                "suggestion_id": "s1",
                "verdict": "inconclusive",
                "measured_at": "2026-01-17T00:00:00Z",
            },
        ])

        cycle = LearningCycle(
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            bots=["bot1"],
            suggestion_tracker=tracker,
        )

        result = cycle._classify_loop_sources("2026-01-14", "2026-01-20", findings)
        _, _, inner_pos, _, inner_total, _ = result

        # Only the positive outcome should be counted; inconclusive excluded
        assert inner_total == 1, f"Expected inner_total=1 (inconclusive excluded), got {inner_total}"
        assert inner_pos == 1


class TestExperimentBlacklist:
    def test_get_failed_experiments(self, tmp_path):
        """get_failed_experiments should return FAILED and ABANDONED records."""
        from trading_assistant.skills.structural_experiment_tracker import StructuralExperimentTracker
        from trading_assistant.schemas.structural_experiment import ExperimentRecord, ExperimentStatus

        tracker = StructuralExperimentTracker(tmp_path)
        tracker.record_experiment(ExperimentRecord(
            experiment_id="exp1", bot_id="bot1", title="test1",
            hypothesis_id="hyp1", status=ExperimentStatus.FAILED,
        ))
        tracker.record_experiment(ExperimentRecord(
            experiment_id="exp2", bot_id="bot1", title="test2",
            hypothesis_id="hyp2", status=ExperimentStatus.ACTIVE,
        ))
        tracker.record_experiment(ExperimentRecord(
            experiment_id="exp3", bot_id="bot2", title="test3",
            hypothesis_id="hyp3", status=ExperimentStatus.ABANDONED,
        ))

        failed = tracker.get_failed_experiments()
        ids = {e.experiment_id for e in failed}
        assert ids == {"exp1", "exp3"}

    def test_blacklist_skips_failed_combo(self, tmp_path):
        """_select_next_experiments should skip hypothesis+bot combos that already failed."""
        from unittest.mock import MagicMock
        from trading_assistant.skills.learning_cycle import LearningCycle
        from trading_assistant.schemas.structural_experiment import ExperimentRecord, ExperimentStatus

        # Create a mock hypothesis library
        hyp = MagicMock()
        hyp.hypothesis_id = "hyp1"
        hyp.category = "stop_loss"
        hyp.effectiveness = 0.8

        hyp_lib = MagicMock()
        hyp_lib.get_active.return_value = [hyp]

        # Create experiment tracker with a failed combo
        exp_tracker = MagicMock()
        exp_tracker.get_active_experiments.return_value = []
        exp_tracker.get_failed_experiments.return_value = [
            ExperimentRecord(
                experiment_id="exp1", bot_id="bot1", title="failed test",
                hypothesis_id="hyp1", status=ExperimentStatus.FAILED,
            ),
        ]

        cycle = LearningCycle(
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            bots=["bot1"],
            hypothesis_library=hyp_lib,
            experiment_tracker=exp_tracker,
        )

        selected = cycle._select_next_experiments()
        # hyp1+bot1 failed before → should be skipped
        assert len(selected) == 0

    def test_expired_blacklist_allows_retry(self, tmp_path):
        """Failed experiments older than 12 weeks should expire from blacklist."""
        from unittest.mock import MagicMock
        from trading_assistant.skills.learning_cycle import LearningCycle
        from trading_assistant.schemas.structural_experiment import ExperimentRecord, ExperimentStatus

        hyp = MagicMock()
        hyp.hypothesis_id = "hyp1"
        hyp.category = "stop_loss"
        hyp.effectiveness = 0.8

        hyp_lib = MagicMock()
        hyp_lib.get_active.return_value = [hyp]

        # Failed experiment resolved 20 weeks ago → should be expired
        old_resolved = (datetime.now(timezone.utc) - timedelta(weeks=20)).isoformat()
        exp_tracker = MagicMock()
        exp_tracker.get_active_experiments.return_value = []
        failed_exp = ExperimentRecord(
            experiment_id="exp1", bot_id="bot1", title="old failure",
            hypothesis_id="hyp1", status=ExperimentStatus.FAILED,
        )
        failed_exp.resolved_at = old_resolved
        exp_tracker.get_failed_experiments.return_value = [failed_exp]

        cycle = LearningCycle(
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            bots=["bot1"],
            hypothesis_library=hyp_lib,
            experiment_tracker=exp_tracker,
        )

        selected = cycle._select_next_experiments()
        # Expired blacklist → hyp1+bot1 should be allowed
        assert len(selected) == 1
        assert selected[0]["hypothesis_id"] == "hyp1"

    def test_recent_blacklist_still_blocks(self, tmp_path):
        """Failed experiments within 12 weeks should still be blocked."""
        from unittest.mock import MagicMock
        from trading_assistant.skills.learning_cycle import LearningCycle
        from trading_assistant.schemas.structural_experiment import ExperimentRecord, ExperimentStatus

        hyp = MagicMock()
        hyp.hypothesis_id = "hyp1"
        hyp.category = "stop_loss"
        hyp.effectiveness = 0.8

        hyp_lib = MagicMock()
        hyp_lib.get_active.return_value = [hyp]

        # Failed experiment resolved 4 weeks ago → still active
        recent_resolved = (datetime.now(timezone.utc) - timedelta(weeks=4)).isoformat()
        exp_tracker = MagicMock()
        exp_tracker.get_active_experiments.return_value = []
        failed_exp = ExperimentRecord(
            experiment_id="exp1", bot_id="bot1", title="recent failure",
            hypothesis_id="hyp1", status=ExperimentStatus.FAILED,
        )
        failed_exp.resolved_at = recent_resolved
        exp_tracker.get_failed_experiments.return_value = [failed_exp]

        cycle = LearningCycle(
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            bots=["bot1"],
            hypothesis_library=hyp_lib,
            experiment_tracker=exp_tracker,
        )

        selected = cycle._select_next_experiments()
        assert len(selected) == 0
