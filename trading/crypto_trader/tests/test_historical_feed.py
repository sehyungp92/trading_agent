from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from crypto_trader.core.models import TimeFrame
from crypto_trader.data.historical_feed import HistoricalFeed


class DictStore:
    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame]) -> None:
        self.frames = frames

    def load_candles(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        return self.frames.get((symbol, timeframe))


def _frame(*timestamps: datetime) -> pd.DataFrame:
    rows = []
    for idx, ts in enumerate(timestamps):
        rows.append(
            {
                "ts": int(ts.timestamp() * 1000),
                "open": 100.0 + idx,
                "high": 101.0 + idx,
                "low": 99.0 + idx,
                "close": 100.5 + idx,
                "volume": 1_000.0,
            }
        )
    return pd.DataFrame(rows)


def test_multi_symbol_same_timestamp_order_is_suffix_stable() -> None:
    symbols = ["BTC", "ETH", "SOL"]
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    store = DictStore(
        {
            ("BTC", "30m"): _frame(t0, t1, t2),
            ("ETH", "30m"): _frame(t1, t2),
            ("SOL", "30m"): _frame(t1, t2),
        }
    )

    full_feed = HistoricalFeed(
        symbols=symbols,
        timeframes=[TimeFrame.M30],
        store=store,  # type: ignore[arg-type]
        start_date=t0,
        end_date=t2,
        primary_timeframe=TimeFrame.M30,
    )
    suffix_feed = HistoricalFeed(
        symbols=symbols,
        timeframes=[TimeFrame.M30],
        store=store,  # type: ignore[arg-type]
        start_date=t1,
        end_date=t2,
        primary_timeframe=TimeFrame.M30,
    )

    full_suffix = [(bar.timestamp, bar.symbol) for bar in full_feed if bar.timestamp >= t1]
    suffix = [(bar.timestamp, bar.symbol) for bar in suffix_feed]

    assert full_suffix == suffix == [
        (t1, "BTC"),
        (t1, "ETH"),
        (t1, "SOL"),
        (t2, "BTC"),
        (t2, "ETH"),
        (t2, "SOL"),
    ]
