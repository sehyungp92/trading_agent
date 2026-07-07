"""Tests for ThresholdLearner — percentile-based threshold optimization."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading_assistant.skills.threshold_learner import ThresholdLearner


def _write_suggestions(path: Path, suggestions: list[dict]) -> None:
    """Write suggestion records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in suggestions:
            f.write(json.dumps(s) + "\n")


def _write_outcomes(path: Path, outcomes: list[dict]) -> None:
    """Write outcome records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for o in outcomes:
            f.write(json.dumps(o) + "\n")


def _make_suggestion(
    suggestion_id: str,
    detector_name: str,
    threshold_name: str,
    threshold_value: float,
    observed_value: float,
    bot_id: str = "bot_a",
) -> dict:
    return {
        "suggestion_id": suggestion_id,
        "bot_id": bot_id,
        "title": "Test suggestion",
        "tier": "parameter",
        "detection_context": {
            "detector_name": detector_name,
            "bot_id": bot_id,
            "threshold_name": threshold_name,
            "threshold_value": threshold_value,
            "observed_value": observed_value,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def _make_outcome(suggestion_id: str, pnl_delta_7d: float) -> dict:
    return {
        "suggestion_id": suggestion_id,
        "implemented_date": "2026-01-01",
        "pnl_delta_7d": pnl_delta_7d,
    }


class TestThresholdLearnerNoData:
    """Tests for edge cases with missing or insufficient data."""

    def test_no_data_returns_empty(self, tmp_path: Path) -> None:
        """Empty findings dir -> learn_thresholds returns empty dict."""
        learner = ThresholdLearner(tmp_path)
        result = learner.learn_thresholds()
        assert result == {}

    def test_below_min_samples_returns_empty(self, tmp_path: Path) -> None:
        """Only 5 suggestions + outcomes -> no learned thresholds (min_samples=10)."""
        suggestions = [
            _make_suggestion(f"s{i}", "alpha_decay", "decay_threshold", 0.3, 0.1 * i)
            for i in range(5)
        ]
        outcomes = [_make_outcome(f"s{i}", 10.0 if i >= 3 else -5.0) for i in range(5)]

        _write_suggestions(tmp_path / "suggestions.jsonl", suggestions)
        _write_outcomes(tmp_path / "outcomes.jsonl", outcomes)

        learner = ThresholdLearner(tmp_path, min_samples=10)
        result = learner.learn_thresholds()
        assert result == {}

    def test_empty_outcomes_but_suggestions_exist(self, tmp_path: Path) -> None:
        """Suggestions exist but no outcomes -> empty result."""
        suggestions = [
            _make_suggestion(f"s{i}", "alpha_decay", "decay_threshold", 0.3, 0.1 * i)
            for i in range(15)
        ]
        _write_suggestions(tmp_path / "suggestions.jsonl", suggestions)
        _write_outcomes(tmp_path / "outcomes.jsonl", [])

        learner = ThresholdLearner(tmp_path)
        result = learner.learn_thresholds()
        assert result == {}

    def test_suggestions_without_detection_context_skipped(self, tmp_path: Path) -> None:
        """Suggestions lacking detection_context are ignored."""
        # 15 suggestions without detection_context
        suggestions_no_ctx = [
            {"suggestion_id": f"s{i}", "bot_id": "bot_a", "title": "No context", "tier": "parameter"}
            for i in range(15)
        ]
        # 5 with context (below min_samples of 10)
        suggestions_with_ctx = [
            _make_suggestion(f"ctx{i}", "alpha_decay", "decay_threshold", 0.3, 0.1 * i)
            for i in range(5)
        ]
        all_suggestions = suggestions_no_ctx + suggestions_with_ctx
        outcomes = (
            [_make_outcome(f"s{i}", 10.0) for i in range(15)]
            + [_make_outcome(f"ctx{i}", 10.0) for i in range(5)]
        )

        _write_suggestions(tmp_path / "suggestions.jsonl", all_suggestions)
        _write_outcomes(tmp_path / "outcomes.jsonl", outcomes)

        learner = ThresholdLearner(tmp_path, min_samples=10)
        result = learner.learn_thresholds()
        # Only 5 with context, below min_samples=10
        assert result == {}


class TestThresholdLearnerOptimization:
    """Tests for threshold optimization logic."""

    def test_simple_separation(self, tmp_path: Path) -> None:
        """15 positive outcomes with observed > 0.3, 5 negative with observed < 0.3
        -> threshold near 0.3."""
        suggestions = []
        outcomes = []
        # 15 positive: observed values from 0.31 to 0.45
        for i in range(15):
            sid = f"pos{i}"
            suggestions.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.31 + i * 0.01)
            )
            outcomes.append(_make_outcome(sid, 10.0))
        # 5 negative: observed values from 0.05 to 0.25
        for i in range(5):
            sid = f"neg{i}"
            suggestions.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.05 + i * 0.05)
            )
            outcomes.append(_make_outcome(sid, -5.0))

        _write_suggestions(tmp_path / "suggestions.jsonl", suggestions)
        _write_outcomes(tmp_path / "outcomes.jsonl", outcomes)

        learner = ThresholdLearner(tmp_path, min_samples=10)
        result = learner.learn_thresholds()

        assert "bot_a" in result
        profile = result["bot_a"]
        record = profile.thresholds["alpha_decay:decay_threshold"]
        # The learned threshold should separate positives from negatives
        # All positives are >= 0.31, all negatives are <= 0.25
        # Optimal should be 0.31 (first value that captures all positives, no negatives)
        assert record.learned_value is not None
        assert record.learned_value >= 0.25
        assert record.learned_value <= 0.45
        assert record.sample_count == 20

    def test_all_positive_outcomes(self, tmp_path: Path) -> None:
        """All outcomes positive -> threshold set to minimum observed value."""
        suggestions = []
        outcomes = []
        for i in range(15):
            sid = f"s{i}"
            suggestions.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.1 + i * 0.05)
            )
            outcomes.append(_make_outcome(sid, 10.0))

        _write_suggestions(tmp_path / "suggestions.jsonl", suggestions)
        _write_outcomes(tmp_path / "outcomes.jsonl", outcomes)

        learner = ThresholdLearner(tmp_path, min_samples=10)
        result = learner.learn_thresholds()

        assert "bot_a" in result
        record = result["bot_a"].thresholds["alpha_decay:decay_threshold"]
        # All positive: best F1 at lowest threshold (captures all true positives)
        assert record.learned_value is not None
        assert record.learned_value == pytest.approx(0.1, abs=0.01)

    def test_all_negative_outcomes(self, tmp_path: Path) -> None:
        """All outcomes negative -> no good F1 achievable (all candidates yield F1=0)."""
        suggestions = []
        outcomes = []
        for i in range(15):
            sid = f"s{i}"
            suggestions.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.1 + i * 0.05)
            )
            outcomes.append(_make_outcome(sid, -5.0))

        _write_suggestions(tmp_path / "suggestions.jsonl", suggestions)
        _write_outcomes(tmp_path / "outcomes.jsonl", outcomes)

        learner = ThresholdLearner(tmp_path, min_samples=10)
        result = learner.learn_thresholds()

        assert "bot_a" in result
        record = result["bot_a"].thresholds["alpha_decay:decay_threshold"]
        # All negative: F1 is 0 for every candidate (no true positives)
        # Ties resolved by preferring closest to default (0.3)
        assert record.learned_value is not None
        assert record.confidence > 0

    def test_f1_ties_prefer_conservative(self, tmp_path: Path) -> None:
        """When F1 is tied, prefer threshold closer to default."""
        # Create a scenario where multiple thresholds yield the same F1
        suggestions = []
        outcomes = []
        # Two groups with identical distributions but different observed values
        # All at the same observed value -> only 1 unique candidate -> returns default
        # Use values symmetric around default (0.3): 0.2 and 0.4
        # Both yield the same F1 when all are positive -> closest to default wins
        for i in range(8):
            sid = f"close{i}"
            suggestions.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.29)
            )
            outcomes.append(_make_outcome(sid, 10.0))
        for i in range(8):
            sid = f"far{i}"
            suggestions.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.5)
            )
            outcomes.append(_make_outcome(sid, 10.0))

        _write_suggestions(tmp_path / "suggestions.jsonl", suggestions)
        _write_outcomes(tmp_path / "outcomes.jsonl", outcomes)

        learner = ThresholdLearner(tmp_path, min_samples=10)
        result = learner.learn_thresholds()

        assert "bot_a" in result
        record = result["bot_a"].thresholds["alpha_decay:decay_threshold"]
        # Both 0.29 and 0.5 give F1=1.0 (all positive). 0.29 is closer to default 0.3
        assert record.learned_value == pytest.approx(0.29, abs=0.01)


class TestThresholdLearnerPerBot:
    """Tests for per-bot specialization."""

    def test_per_bot_specialization(self, tmp_path: Path) -> None:
        """bot_a and bot_b have different outcome distributions -> different learned thresholds."""
        suggestions = []
        outcomes = []

        # bot_a: positive outcomes above 0.4
        for i in range(10):
            sid = f"a_pos{i}"
            suggestions.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.4 + i * 0.01, bot_id="bot_a")
            )
            outcomes.append(_make_outcome(sid, 10.0))
        for i in range(5):
            sid = f"a_neg{i}"
            suggestions.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.1 + i * 0.05, bot_id="bot_a")
            )
            outcomes.append(_make_outcome(sid, -5.0))

        # bot_b: positive outcomes above 0.2
        for i in range(10):
            sid = f"b_pos{i}"
            suggestions.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.2 + i * 0.01, bot_id="bot_b")
            )
            outcomes.append(_make_outcome(sid, 10.0))
        for i in range(5):
            sid = f"b_neg{i}"
            suggestions.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.01 + i * 0.02, bot_id="bot_b")
            )
            outcomes.append(_make_outcome(sid, -5.0))

        _write_suggestions(tmp_path / "suggestions.jsonl", suggestions)
        _write_outcomes(tmp_path / "outcomes.jsonl", outcomes)

        learner = ThresholdLearner(tmp_path, min_samples=10)
        result = learner.learn_thresholds()

        assert "bot_a" in result
        assert "bot_b" in result
        threshold_a = result["bot_a"].thresholds["alpha_decay:decay_threshold"].learned_value
        threshold_b = result["bot_b"].thresholds["alpha_decay:decay_threshold"].learned_value
        assert threshold_a is not None
        assert threshold_b is not None
        # bot_a's threshold should be higher than bot_b's
        assert threshold_a > threshold_b

    def test_multiple_detectors_in_profile(self, tmp_path: Path) -> None:
        """Multiple detector/threshold combos in single profile."""
        suggestions = []
        outcomes = []

        # Detector 1: alpha_decay
        for i in range(12):
            sid = f"alpha{i}"
            suggestions.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.1 + i * 0.05)
            )
            outcomes.append(_make_outcome(sid, 10.0 if i >= 6 else -5.0))

        # Detector 2: exit_timing — shifted so optimal threshold differs from alpha_decay
        for i in range(12):
            sid = f"exit{i}"
            suggestions.append(
                _make_suggestion(sid, "exit_timing", "efficiency_threshold", 0.5, 0.5 + i * 0.05)
            )
            outcomes.append(_make_outcome(sid, 10.0 if i >= 6 else -5.0))

        _write_suggestions(tmp_path / "suggestions.jsonl", suggestions)
        _write_outcomes(tmp_path / "outcomes.jsonl", outcomes)

        learner = ThresholdLearner(tmp_path, min_samples=10)
        result = learner.learn_thresholds()

        assert "bot_a" in result
        profile = result["bot_a"]
        assert "alpha_decay:decay_threshold" in profile.thresholds
        assert "exit_timing:efficiency_threshold" in profile.thresholds
        # Both should have different learned values
        alpha_val = profile.thresholds["alpha_decay:decay_threshold"].learned_value
        exit_val = profile.thresholds["exit_timing:efficiency_threshold"].learned_value
        assert alpha_val is not None
        assert exit_val is not None
        assert alpha_val != exit_val


class TestThresholdLearnerConfidence:
    """Tests for confidence computation."""

    def test_confidence_computation(self, tmp_path: Path) -> None:
        """15 samples -> confidence = 15/30 = 0.5; 30 samples -> confidence = 1.0."""
        # Test with 15 samples
        suggestions_15 = []
        outcomes_15 = []
        for i in range(15):
            sid = f"s{i}"
            suggestions_15.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.1 + i * 0.05)
            )
            outcomes_15.append(_make_outcome(sid, 10.0 if i >= 5 else -5.0))

        _write_suggestions(tmp_path / "suggestions.jsonl", suggestions_15)
        _write_outcomes(tmp_path / "outcomes.jsonl", outcomes_15)

        learner = ThresholdLearner(tmp_path, min_samples=10)
        result_15 = learner.learn_thresholds()

        assert "bot_a" in result_15
        confidence_15 = result_15["bot_a"].thresholds["alpha_decay:decay_threshold"].confidence
        assert confidence_15 == pytest.approx(15 / 30, abs=0.01)

        # Now test with 30 samples
        suggestions_30 = []
        outcomes_30 = []
        for i in range(30):
            sid = f"s30_{i}"
            suggestions_30.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.1 + i * 0.02)
            )
            outcomes_30.append(_make_outcome(sid, 10.0 if i >= 10 else -5.0))

        tmp_path2 = tmp_path / "dir2"
        _write_suggestions(tmp_path2 / "suggestions.jsonl", suggestions_30)
        _write_outcomes(tmp_path2 / "outcomes.jsonl", outcomes_30)

        learner2 = ThresholdLearner(tmp_path2, min_samples=10)
        result_30 = learner2.learn_thresholds()

        assert "bot_a" in result_30
        confidence_30 = result_30["bot_a"].thresholds["alpha_decay:decay_threshold"].confidence
        assert confidence_30 == pytest.approx(1.0, abs=0.01)


class TestThresholdLearnerPersistence:
    """Tests for JSONL roundtrip persistence."""

    def test_jsonl_roundtrip_persistence(self, tmp_path: Path) -> None:
        """learn, save, load -> same thresholds."""
        suggestions = []
        outcomes = []
        for i in range(15):
            sid = f"s{i}"
            suggestions.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.1 + i * 0.05)
            )
            outcomes.append(_make_outcome(sid, 10.0 if i >= 5 else -5.0))

        _write_suggestions(tmp_path / "suggestions.jsonl", suggestions)
        _write_outcomes(tmp_path / "outcomes.jsonl", outcomes)

        learner = ThresholdLearner(tmp_path, min_samples=10)
        original_profiles = learner.learn_thresholds()

        # The profiles are saved by learn_thresholds.
        # Now load them via get_threshold to verify persistence.
        learner2 = ThresholdLearner(tmp_path, min_samples=10)
        loaded_profile = learner2._load_profile("bot_a")

        assert loaded_profile is not None
        assert loaded_profile.bot_id == "bot_a"
        original_record = original_profiles["bot_a"].thresholds["alpha_decay:decay_threshold"]
        loaded_record = loaded_profile.thresholds["alpha_decay:decay_threshold"]
        assert loaded_record.learned_value == original_record.learned_value
        assert loaded_record.sample_count == original_record.sample_count
        assert loaded_record.confidence == pytest.approx(original_record.confidence, abs=0.001)

    def test_get_threshold_returns_default_no_data(self, tmp_path: Path) -> None:
        """No learned data -> returns default."""
        learner = ThresholdLearner(tmp_path)
        result = learner.get_threshold("alpha_decay", "decay_threshold", "bot_a", 0.3)
        assert result == 0.3

    def test_get_threshold_returns_learned_value(self, tmp_path: Path) -> None:
        """Learned data -> returns learned value."""
        suggestions = []
        outcomes = []
        for i in range(15):
            sid = f"s{i}"
            suggestions.append(
                _make_suggestion(sid, "alpha_decay", "decay_threshold", 0.3, 0.1 + i * 0.05)
            )
            outcomes.append(_make_outcome(sid, 10.0 if i >= 5 else -5.0))

        _write_suggestions(tmp_path / "suggestions.jsonl", suggestions)
        _write_outcomes(tmp_path / "outcomes.jsonl", outcomes)

        learner = ThresholdLearner(tmp_path, min_samples=10)
        profiles = learner.learn_thresholds()

        # Fresh learner should load from disk
        learner2 = ThresholdLearner(tmp_path, min_samples=10)
        learned = learner2.get_threshold("alpha_decay", "decay_threshold", "bot_a", 0.3)
        expected = profiles["bot_a"].thresholds["alpha_decay:decay_threshold"].learned_value
        assert learned == expected
        assert learned != 0.3  # Should differ from default
