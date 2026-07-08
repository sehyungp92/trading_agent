"""Balance zone detection — identifies HVN consolidation areas.

A balance zone forms when price rotates around a High Volume Node for
a sustained period, creating a clear auction range with defined boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from crypto_trader.core.models import Bar

from .config import BalanceParams
from .profile import VolumeProfileResult


@dataclass(frozen=True)
class BalanceZone:
    """An identified consolidation zone around an HVN."""
    center: float            # HVN price level
    upper: float             # Upper boundary
    lower: float             # Lower boundary
    bars_in_zone: int        # Bars with close inside zone
    touches: int             # Returns to center from outside
    formation_bar_idx: int   # Global bar index when detected
    volume_contracting: bool
    width_atr: float         # Actual width / ATR


class BalanceDetector:
    """Detects and manages balance zones from volume profile HVN levels."""

    def __init__(self, cfg: BalanceParams) -> None:
        self._p = cfg
        # Active zones per symbol: list of (zone, formation_bar_idx)
        self._zones: dict[str, list[BalanceZone]] = {}

    def update(
        self,
        sym: str,
        bars: list[Bar],
        profile: VolumeProfileResult,
        atr: float,
        bar_index: int,
    ) -> None:
        """Update balance zones for a symbol based on current profile and bars.

        Detects new zones from HVN levels, expires old zones, de-duplicates.
        """
        if atr <= 0 or not profile.hvn_levels:
            return

        if sym not in self._zones:
            self._zones[sym] = []

        # Expire old zones
        self._zones[sym] = [
            z for z in self._zones[sym]
            if (bar_index - z.formation_bar_idx) <= self._p.max_zone_age_bars
        ]

        half_width = self._p.zone_width_atr * atr / 2.0

        for hvn in profile.hvn_levels:
            upper = hvn + half_width
            lower = hvn - half_width

            # De-duplicate: skip if existing zone is within dedup_atr_frac * ATR
            if self._has_nearby_zone(sym, hvn, atr):
                continue

            # Count bars with close inside zone
            bars_in = 0
            touches = 0
            was_outside = True

            for bar in bars:
                inside = lower <= bar.close <= upper
                if inside:
                    bars_in += 1
                    if was_outside:
                        touches += 1
                    was_outside = False
                else:
                    was_outside = True

            if bars_in < self._p.min_bars_in_zone:
                continue
            if touches < self._p.min_touches:
                continue

            # Optional volume contraction check
            vol_contracting = False
            if self._p.require_volume_contraction and len(bars) >= 10:
                vol_contracting = self._check_volume_contraction(bars)
                if not vol_contracting:
                    continue
            elif len(bars) >= 10:
                vol_contracting = self._check_volume_contraction(bars)

            width_atr = (upper - lower) / atr if atr > 0 else 0.0

            zone = BalanceZone(
                center=hvn,
                upper=upper,
                lower=lower,
                bars_in_zone=bars_in,
                touches=touches,
                formation_bar_idx=bar_index,
                volume_contracting=vol_contracting,
                width_atr=width_atr,
            )
            self._zones[sym].append(zone)

    def get_active_zones(self, sym: str) -> list[BalanceZone]:
        """Get currently active balance zones for a symbol."""
        return list(self._zones.get(sym, []))

    def consume_zone(self, sym: str, zone: BalanceZone) -> None:
        """Remove a zone from active inventory for market-derived invalidation."""
        zones = self._zones.get(sym, [])
        self._zones[sym] = [z for z in zones if z is not zone]

    def clear(self, sym: str) -> None:
        """Clear all zones for a symbol."""
        self._zones[sym] = []

    def _has_nearby_zone(self, sym: str, center: float, atr: float) -> bool:
        """Check if an existing zone is within dedup distance."""
        threshold = self._p.dedup_atr_frac * atr
        for z in self._zones.get(sym, []):
            if abs(z.center - center) < threshold:
                return True
        return False

    def _check_volume_contraction(self, bars: list[Bar]) -> bool:
        """Check if volume is contracting (recent < threshold * earlier)."""
        half = len(bars) // 2
        if half < 3:
            return False
        early_vol = sum(b.volume for b in bars[:half]) / half
        late_vol = sum(b.volume for b in bars[half:]) / (len(bars) - half)
        if early_vol <= 0:
            return False
        return late_vol < self._p.contraction_threshold * early_vol
