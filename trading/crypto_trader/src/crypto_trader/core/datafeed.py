"""Data feed protocol for bar iteration."""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from crypto_trader.core.models import Bar, TimeFrame


@runtime_checkable
class DataFeed(Protocol):
    """Sync data feed that yields bars in chronological order."""

    def subscribe(self, symbol: str, timeframes: list[TimeFrame]) -> None:
        """Register interest in specific symbol/timeframe combinations.

        For HistoricalFeed this is validation-only (data must be pre-loaded).
        For a live feed this would start streaming.
        """
        ...

    def __iter__(self) -> Iterator[Bar]:
        """Iterate over bars in chronological order.

        Multi-TF feeds emit higher-TF bars before the corresponding
        primary-TF bar (D1 -> H4 -> H1 -> M15).
        """
        ...

    def get_history(self, symbol: str, timeframe: TimeFrame, count: int) -> list[Bar]:
        """Get the last `count` bars for a symbol/timeframe."""
        ...
