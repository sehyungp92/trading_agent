"""Tests for HTF alignment: verify no look-ahead bias after label='right' fix."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_5m_bars(start: str, periods: int) -> pd.DataFrame:
    """Create a simple 5-minute OHLCV DataFrame."""
    idx = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    n = len(idx)
    return pd.DataFrame(
        {
            "open": np.arange(n, dtype=float),
            "high": np.arange(n, dtype=float) + 0.5,
            "low": np.arange(n, dtype=float) - 0.5,
            "close": np.arange(n, dtype=float) + 0.1,
            "volume": np.ones(n),
        },
        index=idx,
    )


class TestIntraday30mAlignment:
    """Verify that 5m bars map to COMPLETED 30m bars, not in-progress ones."""

    def test_5m_at_1335_maps_to_completed_30m_bar(self):
        from backtests.momentum.data.preprocessing import (
            align_higher_tf_to_5m,
            resample_5m_to_30m,
        )

        # Create 5m bars covering 13:00 - 14:25 (18 bars)
        m_df = _make_5m_bars("2024-01-02 13:00", periods=18)
        m30_df = resample_5m_to_30m(m_df)

        idx_map = align_higher_tf_to_5m(m_df, m30_df)

        # With label="right", 30m bars are labeled at window END:
        # Window [13:00,13:30) -> label 13:30
        # Window [13:30,14:00) -> label 14:00
        # 5m bar at 13:35 (index 7) should map to the 30m bar at 13:30
        # (the completed window [13:00,13:30)), NOT the in-progress 14:00 bar

        bar_1335_idx = 7  # 13:35 is the 8th bar (0-indexed)
        mapped_30m_time = m30_df.index[idx_map[bar_1335_idx]]

        # The mapped 30m bar should be 13:30 (completed window ending at 13:30)
        assert mapped_30m_time.hour == 13
        assert mapped_30m_time.minute == 30

    def test_5m_at_1405_maps_to_just_completed_30m_bar(self):
        from backtests.momentum.data.preprocessing import (
            align_higher_tf_to_5m,
            resample_5m_to_30m,
        )

        m_df = _make_5m_bars("2024-01-02 13:00", periods=18)
        m30_df = resample_5m_to_30m(m_df)

        idx_map = align_higher_tf_to_5m(m_df, m30_df)

        # 5m bar at 14:05 (index 13) should map to 30m bar at 14:00
        # (the just-completed window [13:30,14:00))
        # Note: the 5m bar exactly AT 14:00 still maps to 13:30 because
        # searchsorted(side='left') uses strict-less-than, which is
        # conservative (no look-ahead).
        bar_1405_idx = 13
        mapped_30m_time = m30_df.index[idx_map[bar_1405_idx]]
        assert mapped_30m_time.hour == 14
        assert mapped_30m_time.minute == 0


class TestDailyAlignment:
    """Verify daily alignment uses date-normalised logic (no intraday leak)."""

    def test_daily_alignment_marks_unavailable_first_session(self):
        from backtests.swing.data.preprocessing import align_daily_to_hourly

        hourly_df = pd.DataFrame(
            {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02 14:00", tz="UTC")]),
        )
        daily_df = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC")]),
        )

        idx_map = align_daily_to_hourly(hourly_df, daily_df)

        assert idx_map.tolist() == [-1]

    def test_daily_alignment_uses_previous_day(self):
        from backtests.momentum.data.preprocessing import align_daily_to_5m

        # Create 5m bars for 2 trading days
        day1 = _make_5m_bars("2024-01-02 14:00", periods=24)  # Jan 2
        day2 = _make_5m_bars("2024-01-03 14:00", periods=24)  # Jan 3
        m_df = pd.concat([day1, day2])

        # Daily bars for Jan 2 and Jan 3
        daily_idx = pd.DatetimeIndex(
            [pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")]
        )
        daily_df = pd.DataFrame(
            {"open": [100, 101], "high": [102, 103], "low": [99, 100], "close": [101, 102]},
            index=daily_idx,
        )

        idx_map = align_daily_to_5m(m_df, daily_df)

        # All 5m bars on Jan 3 should map to Jan 2's daily bar (index 0)
        # because Jan 3's daily bar is not complete until end of Jan 3
        jan3_mask = m_df.index.normalize() == pd.Timestamp("2024-01-03", tz="UTC")
        jan3_mapped = idx_map[jan3_mask]
        assert all(i == 0 for i in jan3_mapped), (
            "Jan 3 intraday bars should see Jan 2's daily bar, not Jan 3's"
        )


class TestSwing4hAlignment:
    """Verify swing 1h->4h resample uses right-edge labels."""

    def test_completed_htf_alignment_does_not_expose_first_bar_at_its_close(self):
        from libs.config.completed_bar_policy import align_completed_higher_timeframe_indices

        lower_times = np.array(
            [
                "2024-01-02T03:00:00",
                "2024-01-02T04:00:00",
                "2024-01-02T04:01:00",
            ],
            dtype="datetime64[ns]",
        )
        higher_times = np.array(["2024-01-02T04:00:00"], dtype="datetime64[ns]")

        idx_map = align_completed_higher_timeframe_indices(
            lower_times,
            higher_times,
            unavailable_index=-1,
        )

        assert idx_map.tolist() == [-1, -1, 0]

    def test_shared_completed_alignment_supports_explicit_legacy_pre_history_clip(self):
        from libs.config.completed_bar_policy import align_completed_higher_timeframe_indices

        lower_times = np.array(
            [
                "2024-01-02T03:00:00",
                "2024-01-02T04:00:00",
                "2024-01-02T04:01:00",
            ],
            dtype="datetime64[ns]",
        )
        higher_times = np.array(["2024-01-02T04:00:00"], dtype="datetime64[ns]")

        idx_map = align_completed_higher_timeframe_indices(
            lower_times,
            higher_times,
            unavailable_index=0,
        )

        assert idx_map.tolist() == [0, 0, 0]

    def test_1h_to_4h_drops_incomplete_trailing_window(self):
        from backtests.swing.data.preprocessing import resample_1h_to_4h

        idx = pd.date_range("2024-01-02 00:00", periods=6, freq="1h", tz="UTC")
        h_df = pd.DataFrame(
            {
                "open": range(6),
                "high": range(1, 7),
                "low": range(6),
                "close": range(6),
                "volume": [100] * 6,
            },
            index=idx,
        )

        h4_df = resample_1h_to_4h(h_df)

        assert list(h4_df.index) == [pd.Timestamp("2024-01-02 04:00", tz="UTC")]

    def test_1h_to_4h_right_label(self):
        from backtests.swing.data.preprocessing import (
            align_4h_to_hourly,
            resample_1h_to_4h,
        )

        # Create 1h bars 00:00 - 11:00 (12 bars)
        idx = pd.date_range("2024-01-02 00:00", periods=12, freq="1h", tz="UTC")
        h_df = pd.DataFrame(
            {
                "open": range(12),
                "high": range(1, 13),
                "low": range(12),
                "close": range(12),
                "volume": [100] * 12,
            },
            index=idx,
        )

        h4_df = resample_1h_to_4h(h_df)
        idx_map = align_4h_to_hourly(h_df, h4_df)

        # 1h bar at 05:00 should map to the 4h bar at 04:00
        # (completed window [00:00,04:00)), NOT in-progress [04:00,08:00)
        bar_0500_pos = 5  # 05:00 is the 6th bar
        mapped_4h_time = h4_df.index[idx_map[bar_0500_pos]]
        assert mapped_4h_time.hour == 4
        assert mapped_4h_time.minute == 0
