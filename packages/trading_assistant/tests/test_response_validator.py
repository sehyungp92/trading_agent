# tests/test_response_validator.py
"""Tests for response validation and suggestion scoring."""
from __future__ import annotations

import json

import pytest

from trading_assistant.analysis.response_validator import ResponseValidator, _jaccard_similarity
from trading_assistant.schemas.agent_response import AgentPrediction, AgentSuggestion, ParsedAnalysis, StructuralProposal
from trading_assistant.schemas.suggestion_scoring import CategoryScore, CategoryScorecard
from trading_assistant.skills.suggestion_scorer import SuggestionScorer


class TestJaccardSimilarity:
    def test_identical(self):
        assert _jaccard_similarity("foo bar", "foo bar") == 1.0

    def test_no_overlap(self):
        assert _jaccard_similarity("foo bar", "baz qux") == 0.0

    def test_partial_overlap(self):
        sim = _jaccard_similarity("widen stop loss bot1", "widen stop loss")
        assert sim > 0.6

    def test_empty(self):
        assert _jaccard_similarity("", "foo") == 0.0


class TestResponseValidator:
    def _make_suggestion(self, **kwargs):
        defaults = {
            "bot_id": "bot1",
            "title": "Test suggestion",
            "category": "exit_timing",
            "confidence": 0.8,
        }
        defaults.update(kwargs)
        return AgentSuggestion(**defaults)

    def _make_prediction(self, **kwargs):
        defaults = {
            "bot_id": "bot1",
            "metric": "pnl",
            "direction": "improve",
            "confidence": 0.8,
        }
        defaults.update(kwargs)
        return AgentPrediction(**defaults)

    def test_clean_suggestions_pass_through(self):
        s = self._make_suggestion()
        parsed = ParsedAnalysis(suggestions=[s])
        v = ResponseValidator()
        result = v.validate(parsed)
        assert len(result.approved_suggestions) == 1
        assert len(result.blocked_suggestions) == 0

    def test_rejected_suggestion_blocked(self):
        s = self._make_suggestion(title="Widen stop loss on bot1")
        rejected = [{"bot_id": "bot1", "title": "Widen stop loss on bot1", "tier": "parameter"}]
        parsed = ParsedAnalysis(suggestions=[s])
        v = ResponseValidator(rejected_suggestions=rejected)
        result = v.validate(parsed)
        assert len(result.blocked_suggestions) == 1
        assert "rejected" in result.blocked_suggestions[0].reason.lower()
        assert len(result.approved_suggestions) == 0

    def test_fuzzy_rejection_match(self):
        s = self._make_suggestion(title="Widen stop loss")
        rejected = [{"bot_id": "bot1", "title": "Widen stop loss on bot1"}]
        parsed = ParsedAnalysis(suggestions=[s])
        v = ResponseValidator(rejected_suggestions=rejected)
        result = v.validate(parsed)
        assert len(result.blocked_suggestions) == 1

    def test_different_bot_not_blocked(self):
        s = self._make_suggestion(bot_id="bot2", title="Widen stop loss")
        rejected = [{"bot_id": "bot1", "title": "Widen stop loss"}]
        parsed = ParsedAnalysis(suggestions=[s])
        v = ResponseValidator(rejected_suggestions=rejected)
        result = v.validate(parsed)
        assert len(result.approved_suggestions) == 1

    def test_poor_calibration_caps_confidence(self):
        s = self._make_suggestion(confidence=0.9)
        parsed = ParsedAnalysis(suggestions=[s])
        meta = {"rolling_accuracy_4w": 0.4, "calibration_adjustment": -0.3}
        v = ResponseValidator(forecast_meta=meta)
        result = v.validate(parsed)
        assert result.approved_suggestions[0].confidence <= 0.6

    def test_poor_category_track_record_blocks(self):
        s = self._make_suggestion(category="exit_timing")
        scorecard = CategoryScorecard(scores=[
            CategoryScore(bot_id="bot1", category="exit_timing", win_rate=0.2, sample_size=5),
        ])
        parsed = ParsedAnalysis(suggestions=[s])
        v = ResponseValidator(category_scorecard=scorecard)
        result = v.validate(parsed)
        assert len(result.blocked_suggestions) == 1
        assert "track record" in result.blocked_suggestions[0].reason.lower()

    def test_insufficient_data_not_penalized(self):
        s = self._make_suggestion(category="exit_timing")
        scorecard = CategoryScorecard(scores=[
            CategoryScore(bot_id="bot1", category="exit_timing", win_rate=0.1, sample_size=2),
        ])
        parsed = ParsedAnalysis(suggestions=[s])
        v = ResponseValidator(category_scorecard=scorecard)
        result = v.validate(parsed)
        assert len(result.approved_suggestions) == 1

    def test_prediction_confidence_adjusted(self):
        p = self._make_prediction(confidence=0.9)
        parsed = ParsedAnalysis(predictions=[p])
        meta = {"rolling_accuracy_4w": 0.4, "calibration_adjustment": -0.3}
        v = ResponseValidator(forecast_meta=meta)
        result = v.validate(parsed)
        assert result.approved_predictions[0].confidence < 0.9

    def test_validator_notes_formatted(self):
        s = self._make_suggestion(title="Widen stop loss")
        rejected = [{"bot_id": "bot1", "title": "Widen stop loss"}]
        parsed = ParsedAnalysis(suggestions=[s])
        v = ResponseValidator(rejected_suggestions=rejected)
        result = v.validate(parsed)
        assert "blocked" in result.validator_notes.lower()
        assert "Widen stop loss" in result.validator_notes

    def test_calibration_does_not_compound(self):
        """Confidence should use min(rolling_factor, category_factor), not multiply both."""
        s = self._make_suggestion(confidence=0.8, category="exit_timing")
        scorecard = CategoryScorecard(scores=[
            CategoryScore(bot_id="bot1", category="exit_timing", win_rate=0.5, sample_size=10,
                          confidence_multiplier=0.5),
        ])
        meta = {"rolling_accuracy_4w": 0.4}
        parsed = ParsedAnalysis(suggestions=[s])
        v = ResponseValidator(forecast_meta=meta, category_scorecard=scorecard)
        result = v.validate(parsed)
        # min(0.4, 0.5) = 0.4 → 0.8 * 0.4 = 0.32
        # NOT 0.8 * 0.4 * 0.5 = 0.16 (compounded)
        assert result.approved_suggestions[0].confidence == pytest.approx(0.32, abs=0.01)

    def test_calibration_warning_in_notes(self):
        parsed = ParsedAnalysis(suggestions=[])
        meta = {"rolling_accuracy_4w": 0.3}
        v = ResponseValidator(forecast_meta=meta)
        result = v.validate(parsed)
        assert "calibration" in result.validator_notes.lower()

    def test_negative_monthly_prior_blocks_low_confidence_suggestion(self):
        s = self._make_suggestion(
            category="filter_threshold",
            confidence=0.8,
            strategy_id="strat1",
        )
        parsed = ParsedAnalysis(suggestions=[s])
        v = ResponseValidator(outcome_priors=[{
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "category": "filter_threshold",
            "negative_count": 1,
            "gate_strictness": "stricter",
        }])
        result = v.validate(parsed)
        assert len(result.blocked_suggestions) == 1
        assert "negative monthly prior" in result.blocked_suggestions[0].reason.lower()

    def test_negative_monthly_prior_blocks_structural_proposal(self):
        proposal = StructuralProposal(
            bot_id="bot1",
            affected_strategy_id="strat1",
            title="Tighten threshold routing",
            confidence=0.8,
            acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
        )
        parsed = ParsedAnalysis(structural_proposals=[proposal])
        v = ResponseValidator(outcome_priors=[{
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "category": "signal",
            "negative_count": 1,
            "gate_strictness": "stricter",
        }])
        result = v.validate(parsed)
        assert len(result.blocked_structural_proposals) == 1
        assert "negative monthly prior" in result.blocked_structural_proposals[0].reason.lower()

    def test_end_to_end_parse_validate_annotate(self):
        """Parse → validate → annotated report."""
        from trading_assistant.analysis.response_parser import parse_response

        response = """# Daily Report
Analysis text.

<!-- STRUCTURED_OUTPUT
{
  "predictions": [{"bot_id": "bot1", "metric": "pnl", "direction": "improve", "confidence": 0.8}],
  "suggestions": [
    {"bot_id": "bot1", "title": "Widen stop loss", "category": "exit_timing", "confidence": 0.7},
    {"bot_id": "bot1", "title": "New filter idea", "category": "filter_threshold", "confidence": 0.6}
  ]
}
-->
"""
        parsed = parse_response(response)
        assert parsed.parse_success

        rejected = [{"bot_id": "bot1", "title": "Widen stop loss on bot1"}]
        v = ResponseValidator(rejected_suggestions=rejected)
        result = v.validate(parsed)

        assert len(result.blocked_suggestions) == 1
        assert len(result.approved_suggestions) == 1
        assert result.approved_suggestions[0].title == "New filter idea"

        # Annotate report
        final = parsed.raw_report
        if result.validator_notes:
            final += "\n\n---\n## Validator Notes\n" + result.validator_notes
        assert "Validator Notes" in final
        assert "blocked" in final.lower()


class TestSuggestionScorer:
    def test_empty_data(self, tmp_path):
        scorer = SuggestionScorer(tmp_path)
        scorecard = scorer.compute_scorecard()
        assert scorecard.scores == []

    def test_compute_scorecard(self, tmp_path):
        # Write suggestions
        suggestions = [
            {"suggestion_id": "s1", "bot_id": "bot1", "tier": "parameter", "title": "A"},
            {"suggestion_id": "s2", "bot_id": "bot1", "tier": "parameter", "title": "B"},
            {"suggestion_id": "s3", "bot_id": "bot1", "tier": "parameter", "title": "C"},
        ]
        with open(tmp_path / "suggestions.jsonl", "w") as f:
            for s in suggestions:
                f.write(json.dumps(s) + "\n")

        # Write outcomes
        outcomes = [
            {"suggestion_id": "s1", "pnl_delta_7d": 100.0},
            {"suggestion_id": "s2", "pnl_delta_7d": -50.0},
            {"suggestion_id": "s3", "pnl_delta_7d": 200.0},
        ]
        with open(tmp_path / "outcomes.jsonl", "w") as f:
            for o in outcomes:
                f.write(json.dumps(o) + "\n")

        scorer = SuggestionScorer(tmp_path)
        scorecard = scorer.compute_scorecard()
        assert len(scorecard.scores) == 1
        score = scorecard.scores[0]
        assert score.bot_id == "bot1"
        assert score.category == "filter_threshold"  # "parameter" maps to "filter_threshold"
        assert score.sample_size == 3
        assert score.win_rate == pytest.approx(2 / 3, abs=0.01)
        assert score.confidence_multiplier == 1.0  # 3 samples below threshold of 5

    def test_insufficient_sample_no_penalty(self, tmp_path):
        suggestions = [
            {"suggestion_id": "s1", "bot_id": "bot1", "tier": "exit_timing", "title": "X"},
        ]
        outcomes = [{"suggestion_id": "s1", "pnl_delta_7d": -100.0}]
        with open(tmp_path / "suggestions.jsonl", "w") as f:
            f.write(json.dumps(suggestions[0]) + "\n")
        with open(tmp_path / "outcomes.jsonl", "w") as f:
            f.write(json.dumps(outcomes[0]) + "\n")

        scorer = SuggestionScorer(tmp_path)
        scorecard = scorer.compute_scorecard()
        assert len(scorecard.scores) == 1
        # Bayesian posterior with temporal decay: no timestamp → weight=0.5
        # posterior = (0 + 1) / (0.5 + 2) * 2.0 = 0.8
        assert scorecard.scores[0].confidence_multiplier == pytest.approx(0.8, abs=0.01)

    def test_get_score(self, tmp_path):
        scorecard = CategoryScorecard(scores=[
            CategoryScore(bot_id="bot1", category="exit_timing", win_rate=0.5, sample_size=10),
        ])
        assert scorecard.get_score("bot1", "exit_timing") is not None
        assert scorecard.get_score("bot2", "exit_timing") is None
