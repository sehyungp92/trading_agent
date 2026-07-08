"""Downturn Dominator signal detection -- per-engine signal logic."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .indicators import compute_divergence_magnitude
from .config import DownturnAblationFlags
from .bt_models import (
    BreakdownBoxState,
    BreakdownSignal,
    CompositeRegime,
    EngineTag,
    FadeSignal,
    FadeState,
    ReversalSignal,
    ReversalState,
    VolState,
)


# ---------------------------------------------------------------------------
# Reversal engine signal detection  (Spec S4)
# ---------------------------------------------------------------------------

def detect_reversal_short(
    reversal_state: ReversalState,
    trend_strength: float,
    trend_strength_3d_ago: float,
    close_d: float,
    ema_fast_d: float,
    atr_d: float,
    atr_4h_fast: float,
    atr_4h_slow: float,
    flags: DownturnAblationFlags,
    param_overrides: dict[str, float] | None = None,
) -> Optional[ReversalSignal]:
    """Detect reversal short signal from 4H pivot divergence.

    Gate: 2-of-3 weakening trend / extended above mean / vol coil  (Spec S4.1)
    Setup: H2>H1 with bearish MACD divergence, magnitude threshold
    """
    po = param_overrides or {}
    rs = reversal_state

    if rs.disabled:
        return None

    # Need valid H1/H2 pair
    if rs.h1_price <= 0 or rs.h2_price <= 0:
        return None

    # H2 must be higher than H1 (higher high in price)
    if rs.h2_price <= rs.h1_price:
        return None

    # Bearish MACD divergence: MACD at H2 < MACD at H1
    if rs.macd_at_h2 >= rs.macd_at_h1:
        return None

    # Divergence magnitude check
    div_threshold = po.get("divergence_mag_threshold", 0.15)
    div_mag = compute_divergence_magnitude(rs.macd_at_h1, rs.macd_at_h2, atr_d)
    if flags.reversal_divergence_gate and div_mag < div_threshold:
        return None

    # Extension gate: reject if price not extended above mean
    if flags.reversal_extension_gate and not getattr(flags, "reversal_no_extension_gate", False):
        if atr_d > 0 and close_d <= ema_fast_d + 1.0 * atr_d:
            return None

    # N-of-3 gate: weakening trend / extended above mean / vol coil
    if flags.reversal_trend_weakness_gate:
        gate_count = 0
        # (1) Weakening trend: trend_strength decreased
        if trend_strength < trend_strength_3d_ago:
            gate_count += 1
        # (2) Extended above mean
        if atr_d > 0 and close_d > ema_fast_d + 1.5 * atr_d:
            gate_count += 1
        # (3) Vol coil: ATR contraction (ratio threshold from param_overrides)
        vol_coil_threshold = po.get("vol_coil_ratio_threshold", 0.75)
        if atr_4h_slow > 0 and atr_4h_fast / atr_4h_slow < vol_coil_threshold:
            gate_count += 1
        min_gates = getattr(flags, "reversal_min_gate_count", 2)
        if gate_count < min_gates:
            return None

    # Corridor cap: reject if price too far from pivot
    if flags.reversal_corridor_cap:
        corridor_mult = po.get("corridor_cap_mult", 2.0)
        wider = getattr(flags, "reversal_wider_corridor", 0.0)
        if wider > 0:
            corridor_mult = wider
        if atr_d > 0 and abs(close_d - rs.h2_price) > corridor_mult * atr_d:
            return None

    # Check for Predator overlay (1H lower highs + momentum divergence)
    # Simplified: if divergence magnitude is large, consider predator present
    predator = div_mag > div_threshold * 1.5
    class_mult = 0.65 if predator else 0.40

    return ReversalSignal(
        h1_price=rs.h1_price,
        h2_price=rs.h2_price,
        divergence_mag=div_mag,
        class_mult=class_mult,
        predator_present=predator,
    )


# ---------------------------------------------------------------------------
# Breakdown engine signal detection  (Spec S6)
# ---------------------------------------------------------------------------

def detect_breakdown_short(
    box_state: BreakdownBoxState,
    close_30m: float,
    disp_metric: float,
    disp_history: list[float],
    chop_score: int,
    bar_range_30m: float,
    body_ratio: float,
    rvol: float,
    atr_30m: float,
    flags: DownturnAblationFlags,
    param_overrides: dict[str, float] | None = None,
) -> Optional[BreakdownSignal]:
    """Detect breakdown short signal from 30m box breach.

    Structural: close < box_low  (Spec S6.2)
    Displacement: metric >= rolling quantile  (Spec S6.2)
    Spike reject: excessive range with small body  (Spec S6.2)
    """
    po = param_overrides or {}

    if not box_state.active:
        return None

    # Structural: close below box low
    if close_30m >= box_state.range_low:
        return None

    # Containment gate
    if flags.breakdown_containment_gate:
        min_containment = po.get("box_containment_min", 0.80)
        if box_state.containment_ratio < min_containment:
            return None

    # Chop filter
    if flags.breakdown_chop_filter and chop_score >= 3:
        return None

    # Displacement gate
    if flags.breakdown_displacement_gate:
        q_disp = po.get("displacement_quantile", 0.70)
        if len(disp_history) >= 5:
            threshold = float(np.quantile(disp_history, q_disp))
            if disp_metric < threshold:
                return None

    # Spike reject: range > 1.8xATR AND body_ratio < 0.30 AND RVOL > 2.0
    if flags.breakdown_spike_reject:
        if atr_30m > 0 and bar_range_30m > 1.8 * atr_30m and body_ratio < 0.30 and rvol > 2.0:
            return None

    return BreakdownSignal(
        box_high=box_state.range_high,
        box_low=box_state.range_low,
        displacement_metric=disp_metric,
        box_age=box_state.age,
    )


# ---------------------------------------------------------------------------
# Fade engine signal detection  (Spec S7)
# ---------------------------------------------------------------------------

def detect_fade_short(
    fade_state: FadeState,
    close_15m: float,
    high_15m_recent: np.ndarray,
    regime: CompositeRegime,
    mom_slope_ok: bool,
    extension_short: bool,
    atr_15m: float,
    session_type: str,
    flags: DownturnAblationFlags,
    param_overrides: dict[str, float] | None = None,
) -> Optional[FadeSignal]:
    """Detect fade short signal from VWAP rejection.

    Touch: high >= VWAP within last 8x15m bars  (Spec S7.2)
    Rejection: close < VWAP  (Spec S7.2)
    Cap gate: close >= VWAP - b_cap x ATR15  (Spec S7.2)
    Momentum: SlopeOK_short(t)  (Spec S7.4)
    """
    po = param_overrides or {}
    vwap = fade_state.vwap_used

    if vwap <= 0:
        return None

    # Bear regime required
    if flags.fade_bear_regime_required:
        if regime not in (CompositeRegime.ALIGNED_BEAR, CompositeRegime.EMERGING_BEAR):
            return None

    # Touch: high >= VWAP within last 8 bars
    if flags.fade_vwap_rejection:
        lookback = int(po.get("rejection_lookback_bars", 8))
        recent = high_15m_recent[-lookback:] if len(high_15m_recent) >= lookback else high_15m_recent
        if len(recent) == 0 or not np.any(recent >= vwap):
            return None

    # Rejection: close < VWAP
    if close_15m >= vwap:
        return None

    # Cap gate: close >= VWAP - b_cap x ATR15
    if flags.fade_cap_gate and atr_15m > 0:
        if session_type == "core":
            b_cap = po.get("vwap_cap_core", 0.30)
        else:
            b_cap = po.get("vwap_cap_extended", 0.50)
        if close_15m < vwap - b_cap * atr_15m:
            return None

    # Momentum confirmation
    if flags.fade_momentum_confirm and not mom_slope_ok:
        return None

    # Predator overlay check (simplified)
    predator = extension_short
    class_mult = 1.0 if predator else 0.70

    return FadeSignal(
        vwap_used=vwap,
        rejection_close=close_15m,
        class_mult=class_mult,
        predator_present=predator,
    )


# ---------------------------------------------------------------------------
# Box state management  (Spec S6.1)
# ---------------------------------------------------------------------------

def update_box_state(
    box: BreakdownBoxState,
    high_30m: float,
    low_30m: float,
    close_30m: float,
    atr_30m: float,
    adaptive_L: int,
    param_overrides: dict[str, float] | None = None,
) -> BreakdownBoxState:
    """Update breakdown box state on 30m boundary.

    Tracks containment, violations, and activates/freezes bounds.
    """
    po = param_overrides or {}

    # If no active box, try to form one
    if not box.active:
        if box.age == 0:
            # Start tracking a new potential box
            return BreakdownBoxState(
                range_high=high_30m,
                range_low=low_30m,
                age=1,
                containment_ratio=1.0,
                violations=0,
                active=False,
                vwap_box=close_30m,  # placeholder
                displacement_history=box.displacement_history,
                expiry_countdown=adaptive_L,
                adaptive_L=adaptive_L,
            )
        else:
            # Growing the box
            new_high = max(box.range_high, high_30m)
            new_low = min(box.range_low, low_30m)
            new_age = box.age + 1

            # Check width
            box_width = new_high - new_low
            min_width_mult = po.get("box_width_min_mult", 0.50)
            if atr_30m > 0 and box_width < min_width_mult * atr_30m:
                # Too narrow, keep growing
                return BreakdownBoxState(
                    range_high=new_high,
                    range_low=new_low,
                    age=new_age,
                    containment_ratio=box.containment_ratio,
                    violations=box.violations,
                    active=False,
                    vwap_box=box.vwap_box,
                    displacement_history=box.displacement_history,
                    expiry_countdown=box.expiry_countdown - 1,
                    adaptive_L=adaptive_L,
                )

            # Check if enough bars for activation
            min_bars = int(po.get("box_l_low", 20))
            if new_age >= min_bars:
                # Activate (freeze bounds)
                return BreakdownBoxState(
                    range_high=new_high,
                    range_low=new_low,
                    age=new_age,
                    containment_ratio=1.0,
                    violations=0,
                    active=True,
                    vwap_box=box.vwap_box,
                    displacement_history=box.displacement_history,
                    expiry_countdown=adaptive_L - new_age,
                    adaptive_L=adaptive_L,
                )

            return BreakdownBoxState(
                range_high=new_high,
                range_low=new_low,
                age=new_age,
                containment_ratio=box.containment_ratio,
                violations=box.violations,
                active=False,
                vwap_box=box.vwap_box,
                displacement_history=box.displacement_history,
                expiry_countdown=box.expiry_countdown - 1,
                adaptive_L=adaptive_L,
            )

    # Active box: track containment and violations
    inside = box.range_low <= close_30m <= box.range_high
    new_violations = box.violations + (0 if inside else 1)
    new_age = box.age + 1
    new_containment = (new_age - new_violations) / new_age if new_age > 0 else 0.0

    # Expiry
    new_countdown = box.expiry_countdown - 1
    if new_countdown <= 0:
        # Box expired, reset
        return BreakdownBoxState(
            displacement_history=box.displacement_history,
            adaptive_L=adaptive_L,
        )

    return BreakdownBoxState(
        range_high=box.range_high,  # frozen
        range_low=box.range_low,    # frozen
        age=new_age,
        containment_ratio=new_containment,
        violations=new_violations,
        active=True,
        vwap_box=box.vwap_box,
        displacement_history=box.displacement_history,
        expiry_countdown=new_countdown,
        adaptive_L=adaptive_L,
    )


# ---------------------------------------------------------------------------
# Entry subtype + stop computation  (Spec S10)
# ---------------------------------------------------------------------------

def compute_entry_subtype_stop(
    engine_tag: EngineTag,
    signal: ReversalSignal | BreakdownSignal | FadeSignal,
    close: float,
    atr: float,
    low_recent: float,
    tick_size: float = 0.50,
    param_overrides: dict[str, float] | None = None,
) -> tuple[float, float, str]:
    """Compute entry price, stop0 price, and entry order type.

    Returns (entry_price, stop0_price, entry_type).
    """
    import math
    po = param_overrides or {}

    def _round_tick(price: float, direction: str = "nearest") -> float:
        if direction == "down":
            return math.floor(price / tick_size) * tick_size
        elif direction == "up":
            return math.ceil(price / tick_size) * tick_size
        return round(price / tick_size) * tick_size

    if engine_tag == EngineTag.REVERSAL:
        # Stop-market at L_last_4h - buffer
        buffer = po.get("entry_buffer_ticks", 2.0) * tick_size
        entry_price = _round_tick(low_recent - buffer, "down")
        stop_mult = po.get("reversal_stop_atr_mult", 0.75)
        stop0 = _round_tick(signal.h2_price + stop_mult * atr, "up")
        return entry_price, stop0, "stop_market"

    elif engine_tag == EngineTag.BREAKDOWN:
        # A_latch: stop at breakout_low - 2 ticks
        buffer = po.get("entry_buffer_ticks", 2.0) * tick_size
        entry_price = _round_tick(signal.box_low - buffer, "down")
        box_mid = (signal.box_high + signal.box_low) / 2.0
        stop_mult = po.get("breakdown_stop_atr_mult", 0.10)
        stop0 = _round_tick(box_mid + stop_mult * atr, "up")
        return entry_price, stop0, "stop_limit"

    elif engine_tag == EngineTag.FADE:
        # Stop-limit at Low(trigger) - buffer
        buffer = po.get("entry_buffer_ticks", 2.0) * tick_size
        entry_price = _round_tick(low_recent - buffer, "down")
        stop_mult = po.get("fade_stop_atr_mult", 0.50)
        stop0 = _round_tick(signal.vwap_used + stop_mult * atr, "up")
        return entry_price, stop0, "stop_limit"

    return close, close + atr, "market"


# ---------------------------------------------------------------------------
# Momentum impulse signal (R6 -- alternative fade trigger without VWAP rejection)
# ---------------------------------------------------------------------------

def detect_momentum_impulse(
    close_15m: float,
    ema_fast_15m: float,
    roc_5bar: float,
    regime: CompositeRegime,
    param_overrides: dict[str, float] | None = None,
) -> bool:
    """Momentum-based short signal -- fires during corrections without VWAP rejection.

    Requires bear regime + price below fast EMA + negative 5-bar ROC.
    """
    if regime not in (CompositeRegime.ALIGNED_BEAR, CompositeRegime.EMERGING_BEAR):
        return False
    po = param_overrides or {}
    roc_threshold = po.get("momentum_roc_threshold", -0.005)
    return close_15m < ema_fast_15m and roc_5bar < roc_threshold
