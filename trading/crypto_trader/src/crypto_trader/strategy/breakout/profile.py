"""Volume profile computation from OHLCV bars.

Distributes each bar's volume uniformly across its price range into discrete
bins, then identifies Point of Control (POC), Value Area (VAH/VAL), and
High/Low Volume Nodes (HVN/LVN).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crypto_trader.core.models import Bar, Side

from .config import ProfileParams


@dataclass(frozen=True)
class VolumeProfileResult:
    """Computed volume profile for a price range."""
    poc: float                       # Point of Control (highest volume bin midpoint)
    vah: float                       # Value Area High
    val: float                       # Value Area Low
    hvn_levels: tuple[float, ...]    # High Volume Node midpoints
    lvn_levels: tuple[float, ...]    # Low Volume Node midpoints
    bin_edges: tuple[float, ...]
    bin_volumes: tuple[float, ...]
    total_volume: float
    price_low: float
    price_high: float


class VolumeProfiler:
    """Builds volume profiles from OHLCV bars."""

    def __init__(self, cfg: ProfileParams) -> None:
        self._p = cfg

    def build(self, bars: list[Bar]) -> VolumeProfileResult | None:
        """Build a volume profile from a list of bars.

        Returns None if insufficient bars.
        """
        if len(bars) < self._p.min_bars:
            return None

        price_low = min(b.low for b in bars)
        price_high = max(b.high for b in bars)

        # Guard: all bars at same price (extremely unlikely)
        if price_high <= price_low:
            price_high = price_low + 1.0

        num_bins = self._p.num_bins
        bin_edges = np.linspace(price_low, price_high, num_bins + 1)
        bin_volumes = np.zeros(num_bins, dtype=np.float64)
        bin_width = bin_edges[1] - bin_edges[0]

        # Distribute each bar's volume across overlapping bins
        for bar in bars:
            bar_range = bar.high - bar.low
            if bar_range <= 0:
                # Doji: assign all volume to close bin
                idx = int((bar.close - price_low) / bin_width)
                idx = min(max(idx, 0), num_bins - 1)
                bin_volumes[idx] += bar.volume
                continue

            lo_idx = max(0, int(np.floor((bar.low - price_low) / bin_width)))
            hi_idx = min(num_bins - 1, int(np.floor((bar.high - price_low) / bin_width)))
            # Clamp hi_idx for bar.high == price_high edge case
            hi_idx = min(hi_idx, num_bins - 1)

            for i in range(lo_idx, hi_idx + 1):
                overlap = max(0.0, min(bar.high, bin_edges[i + 1]) - max(bar.low, bin_edges[i]))
                bin_volumes[i] += bar.volume * (overlap / bar_range)

        total_volume = float(np.sum(bin_volumes))

        # POC: midpoint of highest-volume bin
        poc_idx = int(np.argmax(bin_volumes))
        poc = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0)

        # Value Area: expand from POC alternating up/down until 70% cumulative
        vah_idx, val_idx = self._compute_value_area(bin_volumes, poc_idx, total_volume)
        vah = float(bin_edges[vah_idx + 1])  # Upper edge of highest included bin
        val = float(bin_edges[val_idx])       # Lower edge of lowest included bin

        # HVN/LVN detection
        mean_volume = float(np.mean(bin_volumes[bin_volumes > 0])) if np.any(bin_volumes > 0) else 0.0
        hvn_levels: list[float] = []
        lvn_levels: list[float] = []

        for i in range(num_bins):
            mid = float((bin_edges[i] + bin_edges[i + 1]) / 2.0)
            if mean_volume > 0:
                if bin_volumes[i] >= self._p.hvn_threshold_pct * mean_volume:
                    hvn_levels.append(mid)
                elif 0 < bin_volumes[i] <= self._p.lvn_threshold_pct * mean_volume:
                    lvn_levels.append(mid)

        return VolumeProfileResult(
            poc=poc,
            vah=vah,
            val=val,
            hvn_levels=tuple(hvn_levels),
            lvn_levels=tuple(lvn_levels),
            bin_edges=tuple(float(e) for e in bin_edges),
            bin_volumes=tuple(float(v) for v in bin_volumes),
            total_volume=total_volume,
            price_low=price_low,
            price_high=price_high,
        )

    def _compute_value_area(
        self, bin_volumes: np.ndarray, poc_idx: int, total_volume: float
    ) -> tuple[int, int]:
        """Expand from POC bin alternating up/down until value_area_pct reached."""
        target = self._p.value_area_pct * total_volume
        cumulative = float(bin_volumes[poc_idx])
        upper = poc_idx
        lower = poc_idx
        n = len(bin_volumes)

        while cumulative < target:
            up_vol = float(bin_volumes[upper + 1]) if upper + 1 < n else -1.0
            dn_vol = float(bin_volumes[lower - 1]) if lower - 1 >= 0 else -1.0

            if up_vol < 0 and dn_vol < 0:
                break  # No more bins to expand into

            if up_vol >= dn_vol:
                upper += 1
                cumulative += up_vol
            else:
                lower -= 1
                cumulative += dn_vol

        return upper, lower

    def find_lvn_runway(
        self, profile: VolumeProfileResult, price: float, direction: Side, atr: float
    ) -> float:
        """Measure LVN space ahead of price in ATR units.

        Scans bins from the breakout price outward in the breakout direction.
        Returns the distance (in ATR) of continuous low-volume space.
        """
        if atr <= 0:
            return 0.0

        bin_edges = np.array(profile.bin_edges)
        bin_volumes = np.array(profile.bin_volumes)
        num_bins = len(bin_volumes)
        bin_width = bin_edges[1] - bin_edges[0] if num_bins > 0 else 0.0

        if bin_width <= 0:
            return 0.0

        mean_volume = float(np.mean(bin_volumes[bin_volumes > 0])) if np.any(bin_volumes > 0) else 0.0
        if mean_volume <= 0:
            return 0.0

        lvn_threshold = self._p.lvn_threshold_pct * mean_volume

        # Find starting bin
        start_idx = int((price - profile.price_low) / bin_width)
        start_idx = min(max(start_idx, 0), num_bins - 1)

        # Scan outward in breakout direction
        runway_bins = 0
        if direction == Side.LONG:
            for i in range(start_idx, num_bins):
                if bin_volumes[i] <= lvn_threshold:
                    runway_bins += 1
                else:
                    break
        else:
            for i in range(start_idx, -1, -1):
                if bin_volumes[i] <= lvn_threshold:
                    runway_bins += 1
                else:
                    break

        return (runway_bins * bin_width) / atr
