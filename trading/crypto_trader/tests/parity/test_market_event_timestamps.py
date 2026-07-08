"""Market timestamp normalization tests."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from crypto_trader.core.models import Bar
from crypto_trader.core.market_time import completes_higher_timeframe, higher_timeframe_open
from crypto_trader.data.historical_feed import HistoricalFeed
from crypto_trader.core.models import TimeFrame
from crypto_trader.live.feed import LiveFeed, _candle_open_time


def test_live_candle_prefers_open_time_field() -> None:
    open_ms = int(datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    close_ms = int(datetime(2026, 5, 24, 12, 15, tzinfo=timezone.utc).timestamp() * 1000)

    assert _candle_open_time({"t": open_ms, "T": close_ms}, TimeFrame.M15) == (
        datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    )


def test_live_candle_falls_back_from_close_time_field() -> None:
    close_time = datetime(2026, 5, 24, 12, 15, tzinfo=timezone.utc)
    close_ms = int(close_time.timestamp() * 1000)

    assert _candle_open_time({"T": close_ms}, TimeFrame.M15) == close_time - timedelta(minutes=15)


def test_canonical_higher_timeframe_boundary_math() -> None:
    assert completes_higher_timeframe(
        datetime(2026, 5, 24, 12, 45, tzinfo=timezone.utc),
        TimeFrame.M15,
        TimeFrame.H1,
    )
    assert completes_higher_timeframe(
        datetime(2026, 5, 24, 15, 45, tzinfo=timezone.utc),
        TimeFrame.M15,
        TimeFrame.H4,
    )
    assert completes_higher_timeframe(
        datetime(2026, 5, 24, 23, 45, tzinfo=timezone.utc),
        TimeFrame.M15,
        TimeFrame.D1,
    )
    assert completes_higher_timeframe(
        datetime(2026, 5, 24, 15, 30, tzinfo=timezone.utc),
        TimeFrame.M30,
        TimeFrame.H4,
    )
    assert not completes_higher_timeframe(
        datetime(2026, 5, 24, 12, 30, tzinfo=timezone.utc),
        TimeFrame.M15,
        TimeFrame.H1,
    )
    assert higher_timeframe_open(
        datetime(2026, 5, 24, 15, 45, tzinfo=timezone.utc),
        TimeFrame.H4,
    ) == datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _bar(ts: datetime, tf: TimeFrame = TimeFrame.M15) -> Bar:
    return Bar(
        timestamp=ts,
        symbol="BTC",
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1.0,
        timeframe=tf,
    )


def test_higher_timeframe_bars_are_visible_only_on_completed_boundaries() -> None:
    feed = HistoricalFeed.__new__(HistoricalFeed)
    feed.primary_timeframe = TimeFrame.M15
    feed._tf_bars = {
        ("BTC", TimeFrame.H1): {
            datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc): _bar(
                datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
                TimeFrame.H1,
            )
        },
        ("BTC", TimeFrame.H4): {
            datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc): _bar(
                datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
                TimeFrame.H4,
            )
        },
        ("BTC", TimeFrame.D1): {
            datetime(2026, 5, 24, 0, 0, tzinfo=timezone.utc): _bar(
                datetime(2026, 5, 24, 0, 0, tzinfo=timezone.utc),
                TimeFrame.D1,
            )
        },
    }

    assert feed._check_boundary("BTC", TimeFrame.H1, _bar(datetime(2026, 5, 24, 12, 30, tzinfo=timezone.utc))) is None
    assert feed._check_boundary("BTC", TimeFrame.H1, _bar(datetime(2026, 5, 24, 12, 45, tzinfo=timezone.utc))).timeframe == TimeFrame.H1
    assert feed._check_boundary("BTC", TimeFrame.H4, _bar(datetime(2026, 5, 24, 15, 45, tzinfo=timezone.utc))).timeframe == TimeFrame.H4
    assert feed._check_boundary("BTC", TimeFrame.D1, _bar(datetime(2026, 5, 24, 23, 45, tzinfo=timezone.utc))).timeframe == TimeFrame.D1


def test_historical_and_live_market_events_share_candle_times() -> None:
    open_time = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    open_ms = int(open_time.timestamp() * 1000)
    close_ms = int((open_time + timedelta(minutes=15)).timestamp() * 1000)

    historical = HistoricalFeed(
        symbols=["BTC"],
        timeframes=[TimeFrame.M15],
        store=_Store(open_ms),
        start_date=open_time,
        end_date=open_time,
        primary_timeframe=TimeFrame.M15,
    )
    historical_event = next(historical.iter_market_events())

    live = LiveFeed(
        _Info(open_ms, close_ms),
        ["BTC"],
        {"momentum": [TimeFrame.M15]},
    )
    live_event = live.poll_market_events()[0]

    assert live_event.open_time == historical_event.open_time == open_time
    assert live_event.close_time == historical_event.close_time
    assert live_event.available_at == historical_event.available_at
    assert live_event.timestamp_policy == historical_event.timestamp_policy
    assert live_event.to_bar() == historical_event.to_bar()


class _Store:
    def __init__(self, open_ms: int) -> None:
        self.open_ms = open_ms

    def load_candles(self, _symbol: str, _tf: str):
        return pd.DataFrame([{
            "ts": self.open_ms,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1_000.0,
        }])


class _Info:
    def __init__(self, open_ms: int, close_ms: int) -> None:
        self.open_ms = open_ms
        self.close_ms = close_ms

    def candles_snapshot(self, *_args):
        return [
            _candle(self.open_ms - 900_000, self.close_ms - 900_000),
            _candle(self.open_ms, self.close_ms),
            _candle(self.open_ms + 900_000, self.close_ms + 900_000),
        ]


def _candle(open_ms: int, close_ms: int) -> dict:
    return {
        "t": open_ms,
        "T": close_ms,
        "o": "100.0",
        "h": "105.0",
        "l": "95.0",
        "c": "102.0",
        "v": "1000.0",
    }
