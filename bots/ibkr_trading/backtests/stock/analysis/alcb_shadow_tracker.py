"""ALCB shadow tracker — bar-by-bar simulation of rejected setups.

Tracks what would have happened if rejected entries had been taken.
Shadows are updated bar-by-bar using 5m bars.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from strategies.stock.alcb.models import Direction


@dataclass
class ShadowSetup:
    """A rejected entry that we simulate forward."""

    symbol: str
    trade_date: date
    rejection_gate: str
    direction: Direction
    entry_price: float
    stop_price: float
    risk_per_share: float = 0.0
    # Momentum-specific
    momentum_score: int = 0
    rvol_at_rejection: float = 0.0
    entry_type: str = ""
    # Updated bar-by-bar
    active: bool = True
    bars_held: int = 0
    max_price: float = 0.0
    min_price: float = 0.0
    # Final result
    simulated_r: float = 0.0
    simulated_exit: str = ""
    mfe_r: float = 0.0
    mae_r: float = 0.0


_MAX_SHADOW_BARS = 78  # ~1 day of 5m bars


class ALCBShadowTracker:
    """Tracks rejected setups and simulates their outcomes bar-by-bar."""

    def __init__(self) -> None:
        self._active_shadows: list[ShadowSetup] = []
        self._completed: list[ShadowSetup] = []
        self._funnel: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_funnel(self, stage: str) -> None:
        """Increment funnel counter."""
        self._funnel[stage] = self._funnel.get(stage, 0) + 1

    def record_rejection(self, setup: ShadowSetup) -> None:
        """Record a rejected entry for shadow simulation."""
        rps = abs(setup.entry_price - setup.stop_price)
        setup.risk_per_share = rps if rps > 0 else 1.0
        setup.max_price = setup.entry_price
        setup.min_price = setup.entry_price
        self._active_shadows.append(setup)

    # ------------------------------------------------------------------
    # Bar-by-bar update
    # ------------------------------------------------------------------

    def update_bar(self, symbol: str, bar_high: float, bar_low: float, bar_close: float) -> None:
        """Process one 5m bar for all active shadows on this symbol.

        Simulates stop hit, partial target, stale exit (>= MAX_SHADOW_BARS bars).
        """
        for s in self._active_shadows:
            if s.symbol != symbol or not s.active:
                continue

            s.bars_held += 1
            rps = s.risk_per_share

            if s.direction == Direction.LONG:
                s.max_price = max(s.max_price, bar_high)
                s.min_price = min(s.min_price, bar_low)

                # Stop hit
                if bar_low <= s.stop_price:
                    s.simulated_r = (s.stop_price - s.entry_price) / rps
                    s.simulated_exit = "STOP_HIT"
                    s.active = False
                # Simple 1.5R target
                elif bar_high >= s.entry_price + 1.5 * rps:
                    s.simulated_r = 1.5
                    s.simulated_exit = "TARGET_HIT"
                    s.active = False

                s.mfe_r = (s.max_price - s.entry_price) / rps
                s.mae_r = (s.entry_price - s.min_price) / rps

            else:  # SHORT
                s.min_price = min(s.min_price, bar_low)
                s.max_price = max(s.max_price, bar_high)

                if bar_high >= s.stop_price:
                    s.simulated_r = (s.entry_price - s.stop_price) / rps
                    s.simulated_exit = "STOP_HIT"
                    s.active = False
                elif bar_low <= s.entry_price - 1.5 * rps:
                    s.simulated_r = 1.5
                    s.simulated_exit = "TARGET_HIT"
                    s.active = False

                s.mfe_r = (s.entry_price - s.min_price) / rps
                s.mae_r = (s.max_price - s.entry_price) / rps

            # Stale exit
            if s.active and s.bars_held >= _MAX_SHADOW_BARS:
                if s.direction == Direction.LONG:
                    s.simulated_r = (bar_close - s.entry_price) / rps
                else:
                    s.simulated_r = (s.entry_price - bar_close) / rps
                s.simulated_exit = "STALE_EXIT"
                s.active = False

        # Move completed shadows
        still_active = []
        for s in self._active_shadows:
            if s.active:
                still_active.append(s)
            else:
                self._completed.append(s)
        self._active_shadows = still_active

    # ------------------------------------------------------------------
    # End-of-day cleanup
    # ------------------------------------------------------------------

    def flush_stale(self) -> None:
        """Force-close any remaining active shadows (e.g. end of backtest)."""
        for s in self._active_shadows:
            if s.active:
                s.simulated_exit = "EXPIRED"
                s.simulated_r = 0.0
                s.active = False
                self._completed.append(s)
        self._active_shadows = []

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def completed(self) -> list[ShadowSetup]:
        return self._completed

    @property
    def funnel(self) -> dict[str, int]:
        return dict(self._funnel)

    def get_filter_summary(self) -> dict[str, list[ShadowSetup]]:
        """Group completed shadows by rejection gate."""
        by_gate: dict[str, list[ShadowSetup]] = {}
        for s in self._completed:
            by_gate.setdefault(s.rejection_gate, []).append(s)
        return by_gate

    def funnel_report(self) -> str:
        """Format the funnel as monotonic pass counts plus rejection counts."""
        if not self._funnel:
            return "  No funnel data recorded."
        stage_keys = ["evaluated", "entry_signal", "entered"]
        reject_keys = sorted(k for k in self._funnel if k not in stage_keys)
        lines = ["  Signal Funnel:", "  " + "-" * 40, "  Cumulative Pass Counts:"]
        for stage in stage_keys:
            count = self._funnel.get(stage, 0)
            if count > 0 or stage == "evaluated":
                lines.append(f"  {stage:<20s} {count:>6}")
        if reject_keys:
            lines.extend(["", "  Rejection Counts By Gate:"])
            for gate in reject_keys:
                lines.append(f"  {gate:<20s} {self._funnel.get(gate, 0):>6}")
        return "\n".join(lines)
