"""Tests for ThresholdLearner cache and async fixes (Task 4)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_assistant.schemas.detection_context import ThresholdProfile, ThresholdRecord
from trading_assistant.skills.threshold_learner import ThresholdLearner


def _write_profile(path: Path, bot_id: str, detector: str, threshold: str, value: float):
    """Write a threshold profile directly to JSONL."""
    profile = ThresholdProfile(
        bot_id=bot_id,
        thresholds={
            f"{detector}:{threshold}": ThresholdRecord(
                detector_name=detector,
                bot_id=bot_id,
                threshold_name=threshold,
                default_value=0.0,
                learned_value=value,
                sample_count=20,
                confidence=0.8,
            ),
        },
        total_outcomes_used=20,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(profile.model_dump(), default=str) + "\n")


@pytest.fixture
def learner(tmp_path) -> ThresholdLearner:
    findings = tmp_path / "findings"
    findings.mkdir()
    return ThresholdLearner(findings_dir=findings)


class TestCrossBotCache:
    """4a: Fix cross-bot cache bug — different bot_id returns wrong profile."""

    def test_different_bot_ids_return_different_thresholds(self, learner, tmp_path):
        path = tmp_path / "findings" / "learned_thresholds.jsonl"
        _write_profile(path, "bot1", "tight_stop", "tight_stop_ratio", 0.25)
        _write_profile(path, "bot2", "tight_stop", "tight_stop_ratio", 0.40)

        val1 = learner.get_threshold("tight_stop", "tight_stop_ratio", "bot1", default=0.3)
        val2 = learner.get_threshold("tight_stop", "tight_stop_ratio", "bot2", default=0.3)

        assert val1 == 0.25
        assert val2 == 0.40

    def test_same_bot_id_uses_cache(self, learner, tmp_path):
        path = tmp_path / "findings" / "learned_thresholds.jsonl"
        _write_profile(path, "bot1", "tight_stop", "tight_stop_ratio", 0.25)

        val1 = learner.get_threshold("tight_stop", "tight_stop_ratio", "bot1", default=0.3)
        assert val1 == 0.25
        assert learner._cached_bot_id == "bot1"

        # Second call should use cache
        val2 = learner.get_threshold("tight_stop", "tight_stop_ratio", "bot1", default=0.3)
        assert val2 == 0.25


class TestCacheInvalidation:
    """4b: Invalidate cache after learning."""

    def test_cache_invalidated_after_learn(self, learner):
        # Pre-populate cache
        learner._profile = ThresholdProfile(
            bot_id="old", thresholds={}, total_outcomes_used=0,
        )
        learner._cached_bot_id = "old"

        # learn_thresholds with no data returns empty
        learner.learn_thresholds()

        assert learner._profile is None
        assert learner._cached_bot_id is None


class TestAtomicSave:
    """4d: Atomic file save uses temp file + os.replace."""

    def test_save_profiles_creates_file(self, learner, tmp_path):
        from trading_assistant.schemas.detection_context import ThresholdProfile, ThresholdRecord

        profiles = {
            "bot1": ThresholdProfile(
                bot_id="bot1",
                thresholds={
                    "test:thresh": ThresholdRecord(
                        detector_name="test",
                        bot_id="bot1",
                        threshold_name="thresh",
                        default_value=0.5,
                        learned_value=0.3,
                        sample_count=15,
                        confidence=0.7,
                    ),
                },
                total_outcomes_used=15,
            ),
        }
        learner._save_profiles(profiles)
        assert learner._thresholds_path.exists()
        content = learner._thresholds_path.read_text(encoding="utf-8")
        assert "bot1" in content


class TestAsyncWrapper:
    """4c: Async wrapper for sync learn_thresholds."""

    async def test_async_to_thread_wrapper(self, learner):
        import asyncio
        result = await asyncio.to_thread(learner.learn_thresholds)
        assert isinstance(result, dict)
