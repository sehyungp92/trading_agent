"""Drawdown throttle and daily loss cap for position sizing.

Standalone utility — no dependencies on any strategy module.
Each backtest engine instantiates its own DrawdownThrottle.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DrawdownThrottleConfig:
    """Configuration for drawdown-based sizing throttle.

    dd_tiers: sorted list of (dd_pct_threshold, size_multiplier).
    The multiplier applies when drawdown is BELOW the threshold but
    above the previous tier.  Must be sorted ascending by dd_pct.

    Example default:
        0-8% DD  → 1.0x (full size)
        8-12% DD → 0.5x
        12-15% DD → 0.25x
        >15% DD  → 0.25x (floor — prevents deadlock)
    """
    dd_tiers: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.08, 1.00),
        (0.12, 0.50),
        (0.15, 0.25),
        (1.00, 0.25),
    ])
    daily_loss_cap_r: float | None = -2.0  # None disables daily cap


class DrawdownThrottle:
    """Track equity HWM and provide sizing multiplier + daily loss halt."""

    def __init__(
        self,
        initial_equity: float,
        config: DrawdownThrottleConfig | None = None,
    ) -> None:
        self._cfg = config or DrawdownThrottleConfig()
        self._equity = initial_equity
        self.hwm = initial_equity
        self.dd_pct: float = 0.0

        # Daily loss tracking
        self.daily_realized_r: float = 0.0
        self._daily_halted: bool = False

        # Diagnostic counters
        self.entries_blocked_daily: int = 0
        self.entries_blocked_dd: int = 0
        self.dd_mult_history: list[float] = []

    # ── Equity updates ────────────────────────────────────────────

    def update_equity(self, equity: float) -> None:
        """Call whenever equity changes (fill, close, partial, commission)."""
        self._equity = equity
        if equity > self.hwm:
            self.hwm = equity
        self.dd_pct = (self.hwm - equity) / self.hwm if self.hwm > 0 else 0.0

    # ── Trade close ───────────────────────────────────────────────

    def record_trade_close(self, r_multiple: float) -> None:
        """Accumulate daily realized R; trigger halt if cap breached."""
        self.daily_realized_r += r_multiple
        if (self._cfg.daily_loss_cap_r is not None
                and self.daily_realized_r <= self._cfg.daily_loss_cap_r):
            self._daily_halted = True

    # ── Daily reset ───────────────────────────────────────────────

    def daily_reset(self) -> None:
        """Zero daily counters.  Called at each engine's session boundary."""
        self.daily_realized_r = 0.0
        self._daily_halted = False

    # ── Sizing multiplier ─────────────────────────────────────────

    @property
    def dd_size_mult(self) -> float:
        """Return [0.0, 1.0] multiplier based on current drawdown tier."""
        for dd_threshold, mult in self._cfg.dd_tiers:
            if self.dd_pct < dd_threshold:
                return mult
        # Past all tiers — return the last multiplier (should be 0.0)
        return self._cfg.dd_tiers[-1][1] if self._cfg.dd_tiers else 1.0

    @property
    def daily_halted(self) -> bool:
        return self._daily_halted

    # ── Snapshot for diagnostics ──────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "hwm": self.hwm,
            "equity": self._equity,
            "dd_pct": self.dd_pct,
            "dd_size_mult": self.dd_size_mult,
            "daily_realized_r": self.daily_realized_r,
            "daily_halted": self._daily_halted,
            "entries_blocked_daily": self.entries_blocked_daily,
            "entries_blocked_dd": self.entries_blocked_dd,
        }
