"""Downturn Dominator v1 -- live-engine position & order models.

Re-exports all backtest model types so the engine imports from one place.
Defines live-specific ActivePosition and WorkingEntry that carry OMS state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ── Re-exports from bt_models ────────────────────────────────────────────
from .bt_models import (  # noqa: F401
    CompositeRegime,
    CorrectionWindow,
    DownturnRegimeCtx,
    DownturnTradeRecord,
    EngineCounters,
    EngineTag,
    FadeSignal,
    FadeState,
    Regime4H,
    ReversalSignal,
    ReversalState,
    VolState,
)


# ── Live-specific models ─────────────────────────────────────────────────


@dataclass
class ActivePosition:
    """Tracks an open short position with OMS order references."""

    # Identity
    engine_tag: EngineTag = EngineTag.FADE
    signal_class: str = ""
    trade_id: str = ""

    # Prices & sizing
    entry_price: float = 0.0
    stop0: float = 0.0
    qty: int = 0
    remaining_qty: int = 0

    # OMS references
    entry_oms_order_id: str = ""
    stop_oms_order_id: str = ""

    # Timing
    entry_time: Optional[datetime] = None
    hold_bars_5m: int = 0
    hold_bars_1h: int = 0
    hold_bars_30m: int = 0
    hold_bars_4h: int = 0

    # R-state tracking
    mfe_price: float = 0.0       # lowest price reached (short = profit)
    mae_price: float = 0.0       # highest price reached (short = loss)
    r_at_peak: float = 0.0       # best R-multiple reached

    # Stop management
    chandelier_stop: float = 0.0
    be_triggered: bool = False
    exit_trigger: str = ""       # last stop-update reason

    # Tiered exits
    tp_schedule: list[tuple[float, float]] = field(default_factory=list)
    tp_idx: int = 0
    scaled_out: bool = False

    # Context at entry
    composite_regime: CompositeRegime = CompositeRegime.NEUTRAL
    vol_state: VolState = VolState.NORMAL
    in_correction: bool = False
    predator: bool = False

    # Commission
    commission: float = 0.0

    @property
    def risk_per_unit(self) -> float:
        return abs(self.stop0 - self.entry_price)

    def r_state(self, current_price: float) -> float:
        rpu = self.risk_per_unit
        if rpu <= 0:
            return 0.0
        return (self.entry_price - current_price) / rpu


@dataclass
class WorkingEntry:
    """Tracks a pending entry order before fill."""

    oms_order_id: str = ""
    engine_tag: EngineTag = EngineTag.FADE
    signal_class: str = ""
    entry_price: float = 0.0
    stop0: float = 0.0
    qty: int = 0
    submitted_bar_idx: int = 0
    ttl_bars: int = 6

    # Context for position creation on fill
    composite_regime: CompositeRegime = CompositeRegime.NEUTRAL
    vol_state: VolState = VolState.NORMAL
    in_correction: bool = False
    predator: bool = False
    tp_schedule: list[tuple[float, float]] = field(default_factory=list)
    signal_strength: float = 0.5    # for instrumentation
