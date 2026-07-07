"""Threshold learner — auto-tunes strategy engine thresholds from outcome data.

Uses percentile-based optimization: finds the threshold value that maximizes
F1 score between positive and negative outcome observations.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from trading_assistant.schemas.detection_context import DetectionContext, ThresholdProfile, ThresholdRecord
from trading_assistant.skills._outcome_utils import is_positive_outcome

logger = logging.getLogger(__name__)


class ThresholdLearner:
    """Learns optimal thresholds from suggestion outcome data."""

    def __init__(self, findings_dir: Path, min_samples: int = 10) -> None:
        self._findings_dir = findings_dir
        self._min_samples = min_samples
        self._thresholds_path = findings_dir / "learned_thresholds.jsonl"
        self._profile: ThresholdProfile | None = None
        self._cached_bot_id: str | None = None

    def learn_thresholds(self) -> dict[str, ThresholdProfile]:
        """Compute optimal thresholds from suggestions + outcomes.

        Returns a dict of bot_id -> ThresholdProfile.
        """
        outcomes = self._load_detection_outcomes()
        if not outcomes:
            self._profile = None
            self._cached_bot_id = None
            return {}

        # Group by (detector_name, bot_id, threshold_name)
        groups: dict[tuple[str, str, str], list[tuple[float, bool]]] = defaultdict(list)
        for ctx, is_positive in outcomes:
            key = (ctx.detector_name, ctx.bot_id, ctx.threshold_name)
            groups[key].append((ctx.observed_value, is_positive))

        # Compute per-group optimal thresholds
        profiles: dict[str, ThresholdProfile] = {}
        for (detector_name, bot_id, threshold_name), observations in groups.items():
            if len(observations) < self._min_samples:
                continue

            # Get default from existing threshold records or use 0.0
            default = self._get_default_threshold(detector_name, threshold_name)
            learned, confidence = self._compute_optimal_threshold(observations, default)

            if bot_id not in profiles:
                profiles[bot_id] = ThresholdProfile(
                    bot_id=bot_id,
                    thresholds={},
                    total_outcomes_used=0,
                )

            key = f"{detector_name}:{threshold_name}"
            profiles[bot_id].thresholds[key] = ThresholdRecord(
                detector_name=detector_name,
                bot_id=bot_id,
                threshold_name=threshold_name,
                default_value=default,
                learned_value=learned,
                sample_count=len(observations),
                confidence=confidence,
            )
            profiles[bot_id].total_outcomes_used += len(observations)

        # Persist and invalidate cache
        self._save_profiles(profiles)
        self._profile = None
        self._cached_bot_id = None
        return profiles

    def get_threshold(
        self,
        detector_name: str,
        threshold_name: str,
        bot_id: str,
        default: float,
    ) -> float:
        """Return learned threshold if available and confident, else default."""
        if self._profile is None or self._cached_bot_id != bot_id:
            self._profile = self._load_profile(bot_id)
            self._cached_bot_id = bot_id
        if self._profile is None:
            return default

        key = f"{detector_name}:{threshold_name}"
        record = self._profile.thresholds.get(key)
        if record is None:
            return default
        if record.learned_value is not None and record.confidence > 0:
            return record.learned_value
        return default

    def _load_detection_outcomes(self) -> list[tuple[DetectionContext, bool]]:
        """Join suggestions (with detection_context) to outcomes (positive/negative)."""
        suggestions_path = self._findings_dir / "suggestions.jsonl"
        outcomes_path = self._findings_dir / "outcomes.jsonl"

        if not suggestions_path.exists() or not outcomes_path.exists():
            return []

        # Load suggestions with detection_context
        suggestions_by_id: dict[str, DetectionContext] = {}
        for line in suggestions_path.read_text().strip().splitlines():
            try:
                data = json.loads(line)
                ctx_data = data.get("detection_context")
                if ctx_data is None:
                    continue
                suggestion_id = data.get("suggestion_id", "")
                if not suggestion_id:
                    continue
                suggestions_by_id[suggestion_id] = DetectionContext.model_validate(ctx_data)
            except (json.JSONDecodeError, Exception):
                continue

        if not suggestions_by_id:
            return []

        # Load outcomes and join
        results: list[tuple[DetectionContext, bool]] = []
        for line in outcomes_path.read_text().strip().splitlines():
            try:
                data = json.loads(line)
                suggestion_id = data.get("suggestion_id", "")
                if suggestion_id not in suggestions_by_id:
                    continue
                is_positive = is_positive_outcome(data)
                results.append((suggestions_by_id[suggestion_id], is_positive))
            except (json.JSONDecodeError, Exception):
                continue

        return results

    def _compute_optimal_threshold(
        self,
        observations: list[tuple[float, bool]],
        default: float,
    ) -> tuple[float, float]:
        """Find threshold maximizing F1 between positive/negative outcomes.

        Returns (optimal_threshold, confidence).
        """
        if not observations:
            return default, 0.0

        # Get unique observed values as candidate thresholds
        candidates = sorted(set(obs for obs, _ in observations))
        if len(candidates) < 2:
            return default, 0.0

        best_f1 = -1.0
        best_threshold = default

        for candidate in candidates:
            # Predict positive if observed >= candidate threshold
            tp = sum(1 for obs, pos in observations if obs >= candidate and pos)
            fp = sum(1 for obs, pos in observations if obs >= candidate and not pos)
            fn = sum(1 for obs, pos in observations if obs < candidate and pos)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            if f1 > best_f1 or (
                f1 == best_f1
                and abs(candidate - default) < abs(best_threshold - default)
            ):
                best_f1 = f1
                best_threshold = candidate

        confidence = min(1.0, len(observations) / 30)
        return best_threshold, confidence

    def _get_default_threshold(self, detector_name: str, threshold_name: str) -> float:
        """Return known default thresholds for each detector."""
        defaults = {
            ("tight_stop", "tight_stop_ratio"): 0.3,
            ("filter_cost", "filter_cost_threshold"): 0.0,
            ("regime_loss", "regime_min_weeks"): 3.0,
            ("alpha_decay", "decay_threshold"): 0.3,
            ("signal_decay", "decay_threshold"): 0.2,
            ("exit_timing", "efficiency_threshold"): 0.5,
            ("correlation", "threshold"): 0.7,
            ("time_of_day", "loss_threshold"): 0.35,
            ("drawdown_concentration", "concentration_threshold"): 3.0,
            ("position_sizing", "loss_win_ratio_threshold"): 1.5,
            ("component_signal_decay", "stability_threshold"): 0.3,
            ("filter_interactions", "redundancy_threshold"): 0.5,
            ("factor_decay", "below_threshold"): 1.0,
        }
        return defaults.get((detector_name, threshold_name), 0.0)

    def _save_profiles(self, profiles: dict[str, ThresholdProfile]) -> None:
        """Persist learned thresholds to JSONL (atomic write)."""
        from trading_assistant.skills._atomic_write import atomic_rewrite_jsonl

        atomic_rewrite_jsonl(self._thresholds_path, list(profiles.values()))

    def _load_profile(self, bot_id: str) -> ThresholdProfile | None:
        """Load learned threshold profile for a specific bot."""
        if not self._thresholds_path.exists():
            return None
        for line in self._thresholds_path.read_text().strip().splitlines():
            try:
                data = json.loads(line)
                if data.get("bot_id") == bot_id:
                    return ThresholdProfile.model_validate(data)
            except (json.JSONDecodeError, Exception):
                continue
        return None
