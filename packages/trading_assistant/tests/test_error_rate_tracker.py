# tests/test_error_rate_tracker.py
"""Tests for error rate tracker — sliding window frequency detection."""
from datetime import datetime, timezone, timedelta

from trading_assistant.schemas.bug_triage import ErrorEvent
from trading_assistant.skills.error_rate_tracker import ErrorRateTracker


class TestErrorRateTracker:
    def test_single_error_below_threshold(self):
        tracker = ErrorRateTracker(window_seconds=3600, threshold=3)
        e = ErrorEvent(bot_id="bot1", error_type="E", message="m", stack_trace="s")
        tracker.record(e)
        assert tracker.is_repeated("bot1") is False

    def test_three_errors_hits_threshold(self):
        tracker = ErrorRateTracker(window_seconds=3600, threshold=3)
        for _ in range(3):
            e = ErrorEvent(bot_id="bot1", error_type="E", message="m", stack_trace="s")
            tracker.record(e)
        assert tracker.is_repeated("bot1") is True

    def test_errors_from_different_bots_tracked_separately(self):
        tracker = ErrorRateTracker(window_seconds=3600, threshold=3)
        for _ in range(3):
            tracker.record(ErrorEvent(bot_id="bot1", error_type="E", message="m", stack_trace="s"))
        tracker.record(ErrorEvent(bot_id="bot2", error_type="E", message="m", stack_trace="s"))
        assert tracker.is_repeated("bot1") is True
        assert tracker.is_repeated("bot2") is False

    def test_old_errors_expire(self):
        tracker = ErrorRateTracker(window_seconds=3600, threshold=3)
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        for _ in range(3):
            e = ErrorEvent(
                bot_id="bot1", error_type="E", message="m", stack_trace="s",
                timestamp=old_time,
            )
            tracker.record(e)
        assert tracker.is_repeated("bot1") is False

    def test_get_rate_returns_count(self):
        tracker = ErrorRateTracker(window_seconds=3600, threshold=3)
        for _ in range(5):
            tracker.record(ErrorEvent(bot_id="bot1", error_type="E", message="m", stack_trace="s"))
        assert tracker.get_rate("bot1") == 5

    def test_get_rate_unknown_bot_returns_zero(self):
        tracker = ErrorRateTracker(window_seconds=3600, threshold=3)
        assert tracker.get_rate("unknown") == 0

    def test_clear_bot(self):
        tracker = ErrorRateTracker(window_seconds=3600, threshold=3)
        for _ in range(5):
            tracker.record(ErrorEvent(bot_id="bot1", error_type="E", message="m", stack_trace="s"))
        tracker.clear("bot1")
        assert tracker.get_rate("bot1") == 0
        assert tracker.is_repeated("bot1") is False
