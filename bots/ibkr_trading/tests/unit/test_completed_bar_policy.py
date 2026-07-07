from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from libs.config.completed_bar_policy import (
    align_completed_daily_session_indices,
    align_completed_higher_timeframe_indices,
    filter_completed_live_bars,
)

UTC = timezone.utc


def _bar(ts: datetime | str) -> SimpleNamespace:
    return SimpleNamespace(
        date=ts,
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=100.0,
    )


@pytest.mark.parametrize(
    ("bar_size", "bars", "as_of"),
    [
        (
            "5 mins",
            [_bar(datetime(2024, 1, 5, 10, 0, tzinfo=UTC)), _bar(datetime(2024, 1, 5, 10, 5, tzinfo=UTC))],
            datetime(2024, 1, 5, 10, 9, tzinfo=UTC),
        ),
        (
            "15 mins",
            [_bar(datetime(2024, 1, 5, 10, 0, tzinfo=UTC)), _bar(datetime(2024, 1, 5, 10, 15, tzinfo=UTC))],
            datetime(2024, 1, 5, 10, 20, tzinfo=UTC),
        ),
        (
            "30 mins",
            [_bar(datetime(2024, 1, 5, 10, 0, tzinfo=UTC)), _bar(datetime(2024, 1, 5, 10, 30, tzinfo=UTC))],
            datetime(2024, 1, 5, 10, 45, tzinfo=UTC),
        ),
        (
            "1 hour",
            [_bar(datetime(2024, 1, 5, 10, 0, tzinfo=UTC)), _bar(datetime(2024, 1, 5, 11, 0, tzinfo=UTC))],
            datetime(2024, 1, 5, 11, 30, tzinfo=UTC),
        ),
        (
            "4 hours",
            [_bar(datetime(2024, 1, 5, 8, 0, tzinfo=UTC)), _bar(datetime(2024, 1, 5, 12, 0, tzinfo=UTC))],
            datetime(2024, 1, 5, 14, 0, tzinfo=UTC),
        ),
    ],
)
def test_filter_completed_live_bars_drops_in_progress_intraday_tail(
    bar_size: str,
    bars: list[SimpleNamespace],
    as_of: datetime,
) -> None:
    filtered = filter_completed_live_bars(
        bars,
        bar_size_setting=bar_size,
        use_rth=False,
        as_of=as_of,
    )

    assert len(filtered) == 1
    assert filtered[0].date == bars[0].date


def test_filter_completed_live_bars_drops_same_day_rth_daily_bar_before_close() -> None:
    bars = [_bar("20240104"), _bar("20240105")]

    filtered = filter_completed_live_bars(
        bars,
        bar_size_setting="1 day",
        use_rth=True,
        as_of=datetime(2024, 1, 5, 15, 30, tzinfo=UTC),
    )

    assert [bar.date for bar in filtered] == ["20240104"]


def test_filter_completed_live_bars_keeps_rth_daily_bar_after_close() -> None:
    bars = [_bar("20240104"), _bar("20240105")]

    filtered = filter_completed_live_bars(
        bars,
        bar_size_setting="1 day",
        use_rth=True,
        as_of=datetime(2024, 1, 5, 21, 5, tzinfo=UTC),
    )

    assert [bar.date for bar in filtered] == ["20240104", "20240105"]


def test_filter_completed_live_bars_keeps_extended_daily_bar_after_session_close() -> None:
    bars = [_bar("20240104"), _bar("20240105")]

    filtered = filter_completed_live_bars(
        bars,
        bar_size_setting="1 day",
        use_rth=False,
        as_of=datetime(2024, 1, 5, 23, 5, tzinfo=UTC),
    )

    assert [bar.date for bar in filtered] == ["20240104", "20240105"]


def test_filter_completed_live_bars_skips_filtering_for_explicit_end_datetime() -> None:
    bars = [_bar(datetime(2024, 1, 5, 10, 0, tzinfo=UTC)), _bar(datetime(2024, 1, 5, 10, 5, tzinfo=UTC))]

    filtered = filter_completed_live_bars(
        bars,
        bar_size_setting="5 mins",
        use_rth=False,
        end_datetime="20240105 10:09:00",
        as_of=datetime(2024, 1, 5, 10, 9, tzinfo=UTC),
    )

    assert [bar.date for bar in filtered] == [bar.date for bar in bars]


def test_align_completed_higher_timeframe_indices_uses_strictly_completed_bars() -> None:
    lower = np.array(
        [
            np.datetime64("2024-01-05T13:35:00"),
            np.datetime64("2024-01-05T14:05:00"),
            np.datetime64("2024-01-05T14:35:00"),
        ]
    )
    higher = np.array(
        [
            np.datetime64("2024-01-05T13:30:00"),
            np.datetime64("2024-01-05T14:00:00"),
            np.datetime64("2024-01-05T14:30:00"),
        ]
    )

    idx = align_completed_higher_timeframe_indices(lower, higher)

    assert idx.tolist() == [0, 1, 2]


def test_align_completed_daily_session_indices_uses_previous_session() -> None:
    lower = np.array(
        [
            np.datetime64("2024-01-02"),
            np.datetime64("2024-01-03"),
            np.datetime64("2024-01-04"),
        ]
    )
    daily = np.array(
        [
            np.datetime64("2024-01-02"),
            np.datetime64("2024-01-03"),
            np.datetime64("2024-01-04"),
        ]
    )

    idx = align_completed_daily_session_indices(lower, daily)

    assert idx.tolist() == [-1, 0, 1]
