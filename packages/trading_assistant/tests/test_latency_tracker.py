"""Tests for LatencyTracker (#22)."""
import time
from unittest.mock import patch

import pytest

from trading_assistant.orchestrator.latency_tracker import LatencyTracker


class TestLatencyTracker:
    def test_single_event(self):
        tracker = LatencyTracker()
        tracker.record("bot1", "2026-03-01T12:00:00+00:00", "2026-03-01T12:00:10+00:00")
        stats = tracker.get_stats("bot1")
        assert stats.sample_count == 1
        assert stats.p50 == pytest.approx(10.0)
        assert stats.max == pytest.approx(10.0)

    def test_multiple_bots(self):
        tracker = LatencyTracker()
        tracker.record("bot1", "2026-03-01T12:00:00+00:00", "2026-03-01T12:00:05+00:00")
        tracker.record("bot2", "2026-03-01T12:00:00+00:00", "2026-03-01T12:00:20+00:00")
        all_stats = tracker.get_all_stats()
        assert "bot1" in all_stats
        assert "bot2" in all_stats
        assert all_stats["bot1"].p50 == pytest.approx(5.0)
        assert all_stats["bot2"].p50 == pytest.approx(20.0)

    def test_correct_delta(self):
        tracker = LatencyTracker()
        tracker.record("bot1", "2026-03-01T12:00:00+00:00", "2026-03-01T12:05:30+00:00")
        stats = tracker.get_stats("bot1")
        assert stats.p50 == pytest.approx(330.0)

    def test_percentiles(self):
        tracker = LatencyTracker()
        # Add 20 observations with increasing latency (1s, 2s, ... 20s)
        for i in range(1, 21):
            ex = "2026-03-01T12:00:00+00:00"
            rx = f"2026-03-01T12:00:{i:02d}+00:00"
            tracker.record("bot1", ex, rx)
        # Add one high outlier (1 hour)
        tracker.record("bot1", "2026-03-01T12:00:00+00:00", "2026-03-01T13:00:00+00:00")
        stats = tracker.get_stats("bot1")
        assert stats.sample_count == 21
        assert stats.max == pytest.approx(3600.0)
        assert stats.p95 > stats.p50

    def test_aggregate_stats(self):
        tracker = LatencyTracker()
        tracker.record("bot1", "2026-03-01T12:00:00+00:00", "2026-03-01T12:00:05+00:00")
        tracker.record("bot2", "2026-03-01T12:00:00+00:00", "2026-03-01T12:00:20+00:00")
        agg = tracker.get_aggregate_stats()
        assert agg.sample_count == 2
        assert agg.max == pytest.approx(20.0)

    def test_window_pruning(self):
        tracker = LatencyTracker(window_seconds=1.0)
        tracker.record("bot1", "2026-03-01T12:00:00+00:00", "2026-03-01T12:00:10+00:00")

        # Patch monotonic time to simulate expiry
        current = time.monotonic()
        with patch("trading_assistant.orchestrator.latency_tracker.time") as mock_time:
            mock_time.monotonic.return_value = current + 2.0
            stats = tracker.get_stats("bot1")
        assert stats.sample_count == 0

    def test_negative_clamped(self):
        """Negative latency (clock skew) clamped to 0."""
        tracker = LatencyTracker()
        # received_at is before exchange_timestamp
        tracker.record("bot1", "2026-03-01T12:00:10+00:00", "2026-03-01T12:00:00+00:00")
        stats = tracker.get_stats("bot1")
        assert stats.sample_count == 1
        assert stats.p50 == 0.0

    def test_bad_timestamps_skipped(self):
        """Unparseable timestamps are silently skipped."""
        tracker = LatencyTracker()
        tracker.record("bot1", "not-a-date", "also-not-a-date")
        tracker.record("bot1", "", "")
        stats = tracker.get_stats("bot1")
        assert stats.sample_count == 0

    def test_get_all_stats_empty(self):
        tracker = LatencyTracker()
        assert tracker.get_all_stats() == {}
