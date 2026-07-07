from __future__ import annotations

import numpy as np

from backtests.scalp.data.preprocessing import NumpyBars
from backtests.scalp.engine.po3_reversal_engine import _slice_completed
from libs.config.completed_bar_policy import align_completed_higher_timeframe_indices
from datetime import datetime, timezone


def test_scalp_higher_timeframe_available_only_after_close() -> None:
    lower = np.array(["2026-04-29T13:59:00", "2026-04-29T14:00:00", "2026-04-29T14:01:00"], dtype="datetime64[ns]")
    higher = np.array(["2026-04-29T10:00:00", "2026-04-29T14:00:00"], dtype="datetime64[ns]")

    idx = align_completed_higher_timeframe_indices(lower, higher)

    assert idx.tolist() == [0, 0, 1]


def test_scalp_daily_context_excludes_same_session_daily_bar() -> None:
    daily = NumpyBars(
        times=np.array(["2026-04-29T00:00:00"], dtype="datetime64[ns]"),
        opens=np.array([100.0]),
        highs=np.array([110.0]),
        lows=np.array([90.0]),
        closes=np.array([105.0]),
        volumes=np.array([1.0]),
    )
    primary_time = datetime(2026, 4, 29, 14, 30, tzinfo=timezone.utc)

    visible = _slice_completed(daily, np.array([0]), 0, primary_time, "daily")

    assert visible == []
