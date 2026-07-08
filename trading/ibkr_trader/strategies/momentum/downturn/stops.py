"""Downturn Dominator exits -- stops, tiered TPs, chandelier trail, exit checks."""
from __future__ import annotations

import math

from .bt_models import CompositeRegime, EngineTag


# ---------------------------------------------------------------------------
# Tiered take-profit schedule  (Spec S11.4)
# ---------------------------------------------------------------------------

def compute_tiered_tp_schedule(
    engine_tag: EngineTag,
    composite_regime: CompositeRegime,
    param_overrides: dict[str, float] | None = None,
) -> list[tuple[float, float]]:
    """Compute tiered TP schedule as [(r_level, pct_to_close), ...].

    Aligned Bear: TP1=+1.5R/25%, TP2=+3R/20%, TP3=+5R/10%, runner=45%
    Neutral/Emerging: TP1=+1R/35%, TP2=+2R/30%, runner=35%
    Range: TP1=+1R/50%, TP2=+1.8R/30%, runner=20%
    """
    po = param_overrides or {}

    if composite_regime == CompositeRegime.ALIGNED_BEAR:
        return [
            (po.get("tp1_r_aligned", 1.5), 0.25),
            (po.get("tp2_r_aligned", 3.0), 0.20),
            (po.get("tp3_r_aligned", 5.0), 0.10),
            # runner = remaining 45%
        ]
    elif composite_regime in (CompositeRegime.EMERGING_BEAR, CompositeRegime.NEUTRAL):
        return [
            (po.get("tp1_r_emerging", 1.0), po.get("tp1_pct_emerging", 0.35)),
            (po.get("tp2_r_emerging", 2.0), po.get("tp2_pct_emerging", 0.30)),
            # runner = remaining %
        ]
    else:  # RANGE, COUNTER
        return [
            (po.get("tp1_r_range", 1.0), 0.50),
            (po.get("tp2_r_range", 1.8), 0.30),
            # runner = remaining 20%
        ]


# ---------------------------------------------------------------------------
# Chandelier trailing stop  (Spec S12.1)
# ---------------------------------------------------------------------------

def update_chandelier_trail(
    lowest_low: float,
    atr_1h: float,
    r_state: float,
    strong_bear: bool,
    current_stop: float,
    tick_size: float = 0.50,
    param_overrides: dict[str, float] | None = None,
    tp1_hit: bool = False,
    regime_mult: float | None = None,
) -> float:
    """Update chandelier trailing stop for short positions.

    mult = max(2.0, 4.0 - r_state/5.0)
    if strong_bear: mult = max(2.5, mult)
    if tp1_hit: mult = max(post_tp1_chandelier_mult, mult)
    Short: lowest_low + mult x ATR1H, stop never loosens.
    """
    po = param_overrides or {}
    base_lookback = int(po.get("chandelier_lookback", 14))
    _ = base_lookback  # lookback applied externally when computing lowest_low

    base_floor = po.get("chandelier_mult_floor", 2.0)
    base_ceiling = po.get("chandelier_mult_ceiling", 4.0)
    mult = max(base_floor, base_ceiling - r_state / 5.0)
    if strong_bear:
        bear_floor = po.get("chandelier_bear_floor", 2.5)
        mult = max(bear_floor, mult)

    # Post-TP1 widening: give remaining position more room for TP2/TP3
    if tp1_hit:
        post_tp1_mult = po.get("post_tp1_chandelier_mult", 3.5)
        mult = max(mult, post_tp1_mult)

    # Regime-adaptive multiplier (applied after tp1 widening)
    if regime_mult is not None:
        mult *= regime_mult

    new_stop = lowest_low + mult * atr_1h
    new_stop = math.ceil(new_stop / tick_size) * tick_size

    # Stop never loosens (moves down only for shorts)
    if current_stop > 0:
        return min(new_stop, current_stop)
    return new_stop


# ---------------------------------------------------------------------------
# Stale exit check  (Spec S11.9)
# ---------------------------------------------------------------------------

def check_stale_exit(
    engine_tag: EngineTag,
    bars_held: int,
    r_state: float,
    param_overrides: dict[str, float] | None = None,
) -> bool:
    """Check if position should be exited due to staleness.

    Fade: 28x1H bars if r_state < +0.3
    Breakdown: 12x30m bars if r_state < +0.5
    Reversal: 12x4H bars if r_state < +0.3
    """
    po = param_overrides or {}

    if engine_tag == EngineTag.FADE:
        max_bars = int(po.get("stale_bars_fade", 28))
        r_threshold = 0.3
    elif engine_tag == EngineTag.BREAKDOWN:
        max_bars = int(po.get("stale_bars_breakdown", 12))
        r_threshold = 0.5
    elif engine_tag == EngineTag.REVERSAL:
        max_bars = int(po.get("stale_bars_reversal", 12))
        r_threshold = 0.3
    else:
        return False

    return bars_held >= max_bars and r_state < r_threshold


# ---------------------------------------------------------------------------
# Climax exit  (Spec S11.6)
# ---------------------------------------------------------------------------

def check_climax_exit(
    close_1h: float,
    ema20_1h: float,
    atr_1h: float,
    r_state: float,
    param_overrides: dict[str, float] | None = None,
) -> bool:
    """Check for climax exit condition.

    Short: close_1h < ema20_1h - 2.5xATR1H AND r_state > +2.0
    """
    po = param_overrides or {}
    climax_mult = po.get("climax_mult", 2.5)
    return close_1h < ema20_1h - climax_mult * atr_1h and r_state > 2.0


# ---------------------------------------------------------------------------
# VWAP failure exit (Fade only)  (Spec S11.8)
# ---------------------------------------------------------------------------

def check_vwap_failure_exit(
    consecutive_above_vwap: int,
    r_state: float,
) -> bool:
    """Check for VWAP failure exit (Fade only).

    2 consecutive 15m closes > VWAP, disabled after +1R.
    """
    if r_state >= 1.0:
        return False  # disabled after +1R
    return consecutive_above_vwap >= 2


# ---------------------------------------------------------------------------
# Catastrophic exit  (Spec S11.2)
# ---------------------------------------------------------------------------

def check_catastrophic_exit(r_state: float) -> bool:
    """Check for catastrophic exit. r_state < -2.0 -> flatten."""
    return r_state < -2.0


# ---------------------------------------------------------------------------
# Profit floor trail  (locks fraction of profit once threshold R is reached)
# ---------------------------------------------------------------------------

def compute_profit_floor_stop(
    entry_price: float,
    r_state: float,
    risk_per_unit: float,
    tick_size: float = 0.50,
    param_overrides: dict[str, float] | None = None,
) -> float | None:
    """Profit floor: once r_state >= threshold, trail stop at entry - lock_pct * profit.

    For shorts: stop = entry - (r_state * lock_pct * risk_per_unit)
    Returns None if threshold not reached.
    """
    po = param_overrides or {}
    threshold = po.get("profit_floor_r_threshold", 0.5)
    lock_pct = po.get("profit_floor_lock_pct", 0.40)

    if r_state < threshold:
        return None

    # For shorts: lower price = profit, stop above entry
    # profit in price terms = r_state * risk_per_unit
    # lock that fraction: stop = entry - (1 - lock_pct) * profit
    # i.e., we're willing to give back (1 - lock_pct) of the profit
    locked_r = r_state * lock_pct
    stop = entry_price - locked_r * risk_per_unit
    return math.ceil(stop / tick_size) * tick_size


# ---------------------------------------------------------------------------
# Breakeven stop  (Spec S11.3)
# ---------------------------------------------------------------------------

def compute_breakeven_stop(
    avg_entry: float,
    atr_1h: float,
    tick_size: float = 0.50,
    param_overrides: dict[str, float] | None = None,
) -> float:
    """After +1R: entry + buffer x ATR1H for shorts (tighter than initial stop)."""
    po = param_overrides or {}
    buffer = po.get("be_stop_buffer_mult", 0.20)
    be_stop = avg_entry + buffer * atr_1h
    return math.ceil(be_stop / tick_size) * tick_size


# ---------------------------------------------------------------------------
# Multi-tier profit floor (5-tier ratchet)
# ---------------------------------------------------------------------------

# Default tiers: (MFE threshold in R, locked profit in R)
_DEFAULT_PROFIT_TIERS = [
    (1.0, 0.25),
    (1.5, 0.50),
    (2.0, 1.00),
    (3.0, 1.75),
    (5.0, 3.00),
]


def compute_multi_tier_profit_floor(
    entry_price: float,
    mfe_r: float,
    risk_per_unit: float,
    tick_size: float = 0.50,
    param_overrides: dict[str, float] | None = None,
) -> float | None:
    """Multi-tier profit floor based on MFE (peak R-multiple).

    Uses highest qualifying tier. Once MFE crosses a tier threshold,
    that lock level is permanent (ratchet -- never decreases).

    For shorts: stop = entry - locked_r * risk_per_unit
    Returns None if MFE hasn't reached the first tier.
    """
    po = param_overrides or {}
    scale = po.get("profit_floor_scale", 1.0)

    locked_r = None
    for threshold, lock in _DEFAULT_PROFIT_TIERS:
        if mfe_r >= threshold:
            locked_r = lock * scale
        else:
            break

    if locked_r is None:
        return None

    stop = entry_price - locked_r * risk_per_unit
    return math.ceil(stop / tick_size) * tick_size


# ---------------------------------------------------------------------------
# Regime-adaptive chandelier multiplier
# ---------------------------------------------------------------------------

def compute_chandelier_regime_mult(
    composite_regime: "CompositeRegime",
    param_overrides: dict[str, float] | None = None,
) -> float:
    """Regime-based multiplier for chandelier trailing width.

    ALIGNED_BEAR:  wider (let winners run in strong trend)
    EMERGING_BEAR: baseline
    All others:    tighter (protect profits in weaker regimes)
    """
    po = param_overrides or {}

    if composite_regime == CompositeRegime.ALIGNED_BEAR:
        return po.get("chandelier_regime_mult_aligned", 1.15)
    elif composite_regime == CompositeRegime.EMERGING_BEAR:
        return po.get("chandelier_regime_mult_emerging", 1.0)
    else:
        return po.get("chandelier_regime_mult_other", 0.80)


# ---------------------------------------------------------------------------
# Adaptive profit floor lock (R6 -- MFE-tiered lock_pct)
# ---------------------------------------------------------------------------

def compute_adaptive_lock_pct(
    mfe_r: float,
    base_lock: float,
    param_overrides: dict[str, float] | None = None,
) -> float:
    """Increase lock_pct as MFE grows -- capture more from big winners.

    Tiers (defaults):
      MFE < 2R:   base_lock (unchanged)
      MFE 2-5R:   base_lock + 0.10, cap 0.75
      MFE 5-10R:  base_lock + 0.15, cap 0.80
      MFE >= 10R: base_lock + 0.25, cap 0.85
    """
    po = param_overrides or {}
    t1 = po.get("adaptive_lock_t1", 2.0)
    t2 = po.get("adaptive_lock_t2", 5.0)
    t3 = po.get("adaptive_lock_t3", 10.0)
    b1 = po.get("adaptive_lock_bonus_1", 0.10)
    b2 = po.get("adaptive_lock_bonus_2", 0.15)
    b3 = po.get("adaptive_lock_bonus_3", 0.25)
    if mfe_r >= t3:
        return min(0.85, base_lock + b3)
    elif mfe_r >= t2:
        return min(0.80, base_lock + b2)
    elif mfe_r >= t1:
        return min(0.75, base_lock + b1)
    return base_lock
