# skills/backtest_calibration_tracker.py
"""Tracks backtest prediction accuracy to calibrate routing trust.

Closes the meta-learning loop: does our backtest actually predict live outcomes?
Records what the backtest predicted (after APPROVE routing) and later what
actually happened (from AutoOutcomeMeasurer). Over time, builds per-(bot, category)
reliability ratios that modify routing decisions.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from trading_assistant.schemas.parameter_search import BacktestCalibrationRecord, SearchRouting

logger = logging.getLogger(__name__)


class BacktestCalibrationTracker:
    """Tracks backtest prediction accuracy to calibrate routing trust."""

    _MIN_SAMPLES = 5

    def __init__(self, store_dir: Path) -> None:
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._store_dir / "backtest_calibration.jsonl"

    def record_prediction(
        self,
        suggestion_id: str,
        bot_id: str,
        param_category: str,
        predicted_improvement: float,
        predicted_routing: SearchRouting,
    ) -> None:
        """Record what the backtest predicted. Called after APPROVE routing."""
        record = BacktestCalibrationRecord(
            suggestion_id=suggestion_id,
            bot_id=bot_id,
            param_category=param_category,
            predicted_improvement=predicted_improvement,
            predicted_routing=predicted_routing,
        )
        self._append(record)
        logger.debug(
            "Recorded calibration prediction for %s: improvement=%.2f, routing=%s",
            suggestion_id, predicted_improvement, predicted_routing.value,
        )

    def record_outcome(
        self,
        suggestion_id: str,
        actual_composite_delta: float,
    ) -> None:
        """Record what actually happened. Called from AutoOutcomeMeasurer."""
        records = self._load_all()
        updated = False
        for record in records:
            if record.suggestion_id == suggestion_id and record.actual_composite_delta is None:
                record.actual_composite_delta = actual_composite_delta
                record.measured_at = datetime.now(timezone.utc)
                record.prediction_correct = (
                    (record.predicted_improvement > 1.0) == (actual_composite_delta > 0)
                )
                updated = True
                break

        if updated:
            self._write_all(records)
            logger.debug("Recorded calibration outcome for %s: delta=%.4f", suggestion_id, actual_composite_delta)
        else:
            logger.debug("No pending calibration record for suggestion %s", suggestion_id)

    def get_reliability(
        self,
        bot_id: str,
        param_category: str,
        lookback_n: int = 10,
    ) -> tuple[float, int]:
        """Returns (reliability_ratio, sample_count) for measured records."""
        records = self._load_all()
        measured = [
            r for r in records
            if r.bot_id == bot_id
            and r.param_category == param_category
            and r.prediction_correct is not None
        ]
        # Take most recent N
        measured = measured[-lookback_n:]
        if not measured:
            return 0.0, 0

        correct = sum(1 for r in measured if r.prediction_correct)
        return correct / len(measured), len(measured)

    def get_approval_modifier(
        self,
        bot_id: str,
        param_category: str,
    ) -> Literal["fast_track", "normal", "require_experiment"]:
        """Route modifier based on historical reliability."""
        reliability, n = self.get_reliability(bot_id, param_category)

        if n < self._MIN_SAMPLES:
            return "normal"

        if reliability >= 0.70:
            return "fast_track"

        if reliability < 0.50:
            return "require_experiment"

        return "normal"

    def _load_all(self) -> list[BacktestCalibrationRecord]:
        if not self._path.exists():
            return []
        records: list[BacktestCalibrationRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(BacktestCalibrationRecord(**json.loads(line)))
                except Exception:
                    logger.warning("Skipping malformed calibration record")
        return records

    def _write_all(self, records: list[BacktestCalibrationRecord]) -> None:
        with self._path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")

    def _append(self, record: BacktestCalibrationRecord) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
