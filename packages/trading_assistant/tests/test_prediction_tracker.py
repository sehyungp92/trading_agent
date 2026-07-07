# tests/test_prediction_tracker.py
"""Tests for prediction tracking and evaluation."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from trading_assistant.schemas.agent_response import AgentPrediction
from trading_assistant.skills.prediction_tracker import PredictionTracker


def _write_summary(curated, date: str, bot_id: str, **summary) -> None:
    bot_dir = curated / date / bot_id
    bot_dir.mkdir(parents=True, exist_ok=True)
    (bot_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


class TestPredictionTracker:
    def test_record_and_load(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        predictions = [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.8),
            AgentPrediction(bot_id="bot2", metric="win_rate", direction="decline", confidence=0.6),
        ]
        tracker.record_predictions("2026-03-01", predictions)

        loaded = tracker.load_predictions("2026-03-01")
        assert len(loaded) == 2
        assert loaded[0].bot_id == "bot1"
        assert loaded[1].metric == "win_rate"

    def test_load_filtered_by_week(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.8),
        ])
        tracker.record_predictions("2026-03-08", [
            AgentPrediction(bot_id="bot2", metric="pnl", direction="decline", confidence=0.7),
        ])

        week1 = tracker.load_predictions("2026-03-01")
        assert len(week1) == 1
        assert week1[0].bot_id == "bot1"

        all_predictions = tracker.load_predictions()
        assert len(all_predictions) == 2

    def test_evaluate_correct_prediction(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.9),
        ])

        curated = tmp_path / "curated"
        _write_summary(curated, "2026-03-01", "bot1", total_pnl=100.0)
        _write_summary(curated, "2026-03-08", "bot1", total_pnl=500.0)

        evaluation = tracker.evaluate_predictions("2026-03-01", curated)
        assert evaluation.total == 1
        assert evaluation.correct == 1
        assert evaluation.accuracy == 1.0
        assert evaluation.verdicts[0].status == "correct"

    def test_evaluate_incorrect_prediction(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.9),
        ])

        curated = tmp_path / "curated"
        _write_summary(curated, "2026-03-01", "bot1", total_pnl=100.0)
        _write_summary(curated, "2026-03-08", "bot1", total_pnl=-200.0)

        evaluation = tracker.evaluate_predictions("2026-03-01", curated)
        assert evaluation.total == 1
        assert evaluation.correct == 0
        assert evaluation.accuracy == 0.0
        assert evaluation.verdicts[0].status == "incorrect"

    def test_evaluate_missing_curated_data(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.8),
        ])

        curated = tmp_path / "curated"
        curated.mkdir()

        evaluation = tracker.evaluate_predictions("2026-03-01", curated)
        assert evaluation.total == 0  # insufficient data excluded from total
        assert evaluation.verdicts[0].status == "insufficient_data"

    def test_confidence_weighted_accuracy(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.9),
            AgentPrediction(bot_id="bot2", metric="pnl", direction="decline", confidence=0.3),
        ])

        curated = tmp_path / "curated"
        for bot_id in ["bot1", "bot2"]:
            _write_summary(curated, "2026-03-01", bot_id, total_pnl=100.0)
            _write_summary(curated, "2026-03-08", bot_id, total_pnl=150.0)

        evaluation = tracker.evaluate_predictions("2026-03-01", curated)
        assert evaluation.total == 2
        assert evaluation.correct == 1  # bot1 correct, bot2 wrong
        # CW accuracy: (0.9 * 1 + 0.3 * 0) / (0.9 + 0.3) = 0.75
        assert evaluation.confidence_weighted_accuracy == pytest.approx(0.75, abs=0.01)

    def test_accuracy_by_metric(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.8),
            AgentPrediction(bot_id="bot1", metric="win_rate", direction="improve", confidence=0.7),
        ])

        curated = tmp_path / "curated"
        _write_summary(curated, "2026-03-01", "bot1", total_pnl=100.0, win_rate=0.60)
        _write_summary(curated, "2026-03-08", "bot1", total_pnl=150.0, win_rate=0.55)

        evaluation = tracker.evaluate_predictions("2026-03-01", curated)
        assert evaluation.accuracy_by_metric.get("pnl") == 1.0
        assert evaluation.accuracy_by_metric.get("win_rate") == 0.0

    def test_evaluation_uses_metric_change_not_absolute_level(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="decline", confidence=0.8),
        ])

        curated = tmp_path / "curated"
        _write_summary(curated, "2026-03-01", "bot1", total_pnl=500.0)
        _write_summary(curated, "2026-03-08", "bot1", total_pnl=300.0)

        evaluation = tracker.evaluate_predictions("2026-03-01", curated)
        assert evaluation.total == 1
        assert evaluation.correct == 1

    def test_empty_predictions(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        evaluation = tracker.evaluate_predictions("2026-03-01", tmp_path)
        assert evaluation.total == 0
        assert evaluation.accuracy == 0.0

    def test_two_tracker_instances_serialize_appends(self, tmp_path):
        tracker_a = PredictionTracker(tmp_path)
        tracker_b = PredictionTracker(tmp_path)
        prediction = AgentPrediction(
            bot_id="bot1",
            metric="pnl",
            direction="improve",
            confidence=0.8,
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(tracker_a.record_predictions, "2026-03-01", [prediction]),
                pool.submit(tracker_b.record_predictions, "2026-03-02", [prediction]),
            ]
            for future in futures:
                future.result()

        lines = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["bot_id"] == "bot1" for line in lines)
