"""Funding rate utilities for perpetual futures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_trader.core.models import Side


class FundingHelper:
    """Manages funding rate lookups and cost computation.

    Expects a DataFrame with columns:
      - ts: int64 millisecond epoch timestamps
      - rate: float hourly funding rate
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if df.empty:
            self._ts = np.array([], dtype=np.int64)
            self._rates = np.array([], dtype=np.float64)
        else:
            sorted_df = df.sort_values("ts").reset_index(drop=True)
            self._ts = sorted_df["ts"].values.astype(np.int64)
            self._rates = sorted_df["rate"].values.astype(np.float64)

    def get_rate_at(self, timestamp_ms: int) -> float:
        """Get the funding rate effective at the given timestamp.

        Uses binary search to find the most recent rate <= timestamp_ms.
        Returns 0.0 if no rate data is available before the timestamp.
        """
        if len(self._ts) == 0:
            return 0.0
        idx = int(np.searchsorted(self._ts, timestamp_ms, side="right")) - 1
        if idx < 0:
            return 0.0
        return float(self._rates[idx])

    def compute_funding_cost(
        self,
        direction: Side,
        notional: float,
        start_ms: int,
        end_ms: int,
    ) -> float:
        """Compute total funding cost over a time range.

        Positive return means the position paid funding.
        Longs pay positive rates; shorts pay negative rates.
        """
        if len(self._ts) == 0 or start_ms >= end_ms:
            return 0.0

        # Find rates within the range [start_ms, end_ms)
        start_idx = int(np.searchsorted(self._ts, start_ms, side="left"))
        end_idx = int(np.searchsorted(self._ts, end_ms, side="left"))

        sign = 1.0 if direction == Side.LONG else -1.0
        total = 0.0

        for i in range(start_idx, end_idx):
            total += notional * float(self._rates[i]) * sign

        return total

    @staticmethod
    def next_funding_time(current_ms: int) -> int:
        """Get the next hourly funding boundary in milliseconds."""
        hour_ms = 3_600_000
        return ((current_ms // hour_ms) + 1) * hour_ms
