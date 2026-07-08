"""Pullback exit logic for IARIC V2 (hybrid intraday engine).

Exit priority chain (matching research engine order):
  1. Stop hit (low <= stop)
  2. Quick exit (early cut for losers, 2-bar window)
  2.5. Stale exit (force-close stale positions, distinct from stale tighten)
  3. EMA reversion (profitable mean-reversion exit)
  4. RSI exit (route-specific thresholds)
  5. Time stop (max hold exceeded)
  6. VWAP fail (structural: descending highs + CPR gate)
  7. EOD flatten (15:55 forced close)

Also: V2 partial profit (on mfe_r), MFE stage management (3->2->1),
stale tighten (tightens stop, does NOT exit), overnight carry.
"""
from __future__ import annotations

from datetime import datetime

from .config import ET, StrategySettings
from .models import Bar, PBSymbolState


# ---------------------------------------------------------------------------
# Exit chain checks (pure functions, return (should_exit, exit_reason))
# ---------------------------------------------------------------------------

def check_stop_hit(bar_low: float, current_stop: float) -> tuple[bool, str]:
    """Exit if bar low breached the stop."""
    if bar_low <= current_stop:
        return True, "STOP_HIT"
    return False, ""


def check_quick_exit(
    hold_bars: int,
    unrealized_r: float,
    close_below_vwap: bool,
    threshold_bars: int = 2,
    loss_r: float = 0.0,
) -> tuple[bool, str]:
    """Early cut for positions losing quickly (2-bar window, route-specific loss_r)."""
    if loss_r <= 0:
        return False, ""
    if hold_bars <= threshold_bars and unrealized_r <= -loss_r and close_below_vwap:
        return True, "QUICK_EXIT"
    return False, ""


def check_ema_reversion(
    bar_close: float,
    ema10_value: float | None,
    unrealized_r: float,
    min_r: float = 0.03,
) -> tuple[bool, str]:
    """Profitable mean-reversion exit: close above EMA10 with positive R."""
    if ema10_value is None:
        return False, ""
    if bar_close >= ema10_value and unrealized_r > min_r:
        return True, "EMA_REVERSION"
    return False, ""


def check_rsi_exit(
    rsi_value: float | None,
    route_family: str,
    config: StrategySettings,
) -> tuple[bool, str]:
    """Route-specific RSI exit thresholds."""
    if rsi_value is None:
        return False, ""
    thresholds = {
        "OPEN_SCORED_ENTRY": config.pb_v2_rsi_exit_open_scored,
        "DELAYED_CONFIRM": config.pb_v2_rsi_exit_delayed,
        "VWAP_BOUNCE": config.pb_v2_rsi_exit_vwap_bounce,
        "AFTERNOON_RETEST": config.pb_v2_rsi_exit_afternoon,
        "OPENING_RECLAIM": 58.0,
    }
    threshold = thresholds.get(route_family, 60.0)
    if rsi_value > threshold:
        return True, "RSI_EXIT"
    return False, ""


def check_time_stop(hold_days: int, max_hold: int) -> tuple[bool, str]:
    """Exit if max hold duration exceeded."""
    if max_hold > 0 and hold_days >= max_hold:
        return True, "TIME_STOP"
    return False, ""


def check_vwap_fail(
    recent_bars: list[Bar],
    session_vwap: float | None,
    lookback: int = 3,
    cpr_max: float = -1.0,
) -> tuple[bool, str]:
    """Structural VWAP failure: descending highs + close below VWAP + CPR gate.

    Research parity: requires N bars with descending highs, last bar close < VWAP,
    and last bar CPR <= cpr_max. cpr_max < 0 disables the exit.
    """
    if lookback <= 1 or cpr_max < 0:
        return False, ""
    lookback = max(2, lookback)
    if len(recent_bars) < lookback:
        return False, ""
    window = recent_bars[-lookback:]
    if session_vwap is None or window[-1].close >= session_vwap:
        return False, ""
    if window[-1].cpr > cpr_max:
        return False, ""
    # Check descending highs (lower highs pattern)
    highs = [b.high for b in window]
    if not all(highs[i] <= highs[i - 1] + 1e-9 for i in range(1, len(highs))):
        return False, ""
    return True, "VWAP_FAIL"


def check_eod_flatten(now: datetime, config: StrategySettings) -> tuple[bool, str]:
    """Forced flatten at EOD (15:55 ET)."""
    et_time = now.astimezone(ET).time()
    if et_time >= config.pb_intraday_force_exit:
        return True, "EOD_FLATTEN"
    return False, ""


# ---------------------------------------------------------------------------
# Per-route parameter lookup (general, covers all per-route settings)
# ---------------------------------------------------------------------------

def _route_param(route_family: str, suffix: str, config: StrategySettings, default: float = 0.0) -> float:
    """Look up per-route parameter, fall back to global pb_ prefix."""
    prefix_map = {
        "OPEN_SCORED_ENTRY": "pb_open_scored",
        "DELAYED_CONFIRM": "pb_delayed_confirm",
        "OPENING_RECLAIM": "pb_opening_reclaim",
    }
    prefix = prefix_map.get(route_family, "pb")
    val = getattr(config, f"{prefix}_{suffix}", None)
    if val is not None:
        return float(val)
    return float(getattr(config, f"pb_{suffix}", default))


# ---------------------------------------------------------------------------
# Stale exit (force-close, distinct from stale tighten)
# ---------------------------------------------------------------------------

def check_stale_exit(
    hold_bars: int,
    max_mfe_r: float,
    stale_exit_bars: int,
    stale_exit_min_r: float,
) -> tuple[bool, str]:
    """Force-close stale positions (research parity: backtest STALE_EXIT)."""
    if stale_exit_bars <= 0:
        return False, ""
    if hold_bars >= stale_exit_bars and max_mfe_r < stale_exit_min_r:
        return True, "STALE_EXIT"
    return False, ""


# ---------------------------------------------------------------------------
# Stale position tighten (NOT an exit -- tightens stop only)
# ---------------------------------------------------------------------------

def compute_stale_tighten(
    hold_bars: int,
    max_mfe_r: float,
    entry_price: float,
    risk_per_share: float,
    current_stop: float,
    stale_bars: int,
    stale_mfe_thresh: float,
    stale_tighten_pct: float,
) -> float | None:
    """Tighten stop for stale positions (research parity: does NOT exit).

    Returns new stop level if tightened, None otherwise.
    """
    if stale_bars <= 0 or hold_bars < stale_bars:
        return None
    if max_mfe_r >= stale_mfe_thresh:
        return None
    tighten_stop = entry_price - (1.0 - stale_tighten_pct) * risk_per_share
    if tighten_stop > current_stop:
        return tighten_stop
    return None


# ---------------------------------------------------------------------------
# V2 partial profit (triggers on MFE, not unrealized)
# ---------------------------------------------------------------------------

def check_v2_partial(
    mfe_r: float,
    already_taken: bool,
    trigger_r: float = 0.50,
) -> bool:
    """Check if partial profit trigger is met (based on MFE, not current price)."""
    if already_taken:
        return False
    if trigger_r <= 0:
        return False
    return mfe_r >= trigger_r


# ---------------------------------------------------------------------------
# MFE stage management (check 3->2->1 descending, use entry_atr for trail)
# ---------------------------------------------------------------------------

def update_mfe_stages(
    state: PBSymbolState,
    bar_high: float,
    entry_price: float,
    risk_per_share: float,
    entry_atr: float,
    config: StrategySettings,
) -> float:
    """Advance MFE protection stages (3->2->1 order), return updated stop level.

    Research parity: checks stage 3 first so large single-bar moves jump
    directly to the correct stage. Uses entry_atr (session ATR at entry time)
    for trailing stop, not daily_atr.
    """
    if risk_per_share <= 0 or state.position is None:
        return state.stop_level

    mfe_price = max(state.position.max_favorable_price, bar_high)
    mfe_r = (mfe_price - entry_price) / risk_per_share
    current_stop = state.stop_level

    # Stage 3: Trailing (checked first -- research parity)
    if mfe_r >= config.pb_v2_mfe_stage3_trigger and state.mfe_stage < 3:
        state.mfe_stage = 3
        state.trail_active = True
        trail_stop = bar_high - config.pb_v2_mfe_stage3_trail_atr * max(entry_atr, 0.01)
        current_stop = max(current_stop, trail_stop)

    # Stage 2: Breakeven
    elif mfe_r >= config.pb_v2_mfe_stage2_trigger and state.mfe_stage < 2:
        state.mfe_stage = 2
        state.breakeven_activated = True
        current_stop = max(current_stop, entry_price)

    # Stage 1: Tighten stop
    elif mfe_r >= config.pb_v2_mfe_stage1_trigger and state.mfe_stage < 1:
        state.mfe_stage = 1
        stage1_stop = entry_price + config.pb_v2_mfe_stage1_stop_r * risk_per_share
        current_stop = max(current_stop, stage1_stop)

    # Stage 3 trailing update each bar (always runs when trail active)
    if state.mfe_stage >= 3:
        trail_stop = bar_high - config.pb_v2_mfe_stage3_trail_atr * max(entry_atr, 0.01)
        current_stop = max(current_stop, trail_stop)

    # Stage 1 stop protection each bar
    if state.mfe_stage >= 1:
        protect_stop = entry_price + config.pb_v2_mfe_stage1_stop_r * risk_per_share
        current_stop = max(current_stop, protect_stop)

    return current_stop


# ---------------------------------------------------------------------------
# Overnight carry decision
# ---------------------------------------------------------------------------

def should_carry_overnight(
    state: PBSymbolState,
    unrealized_r: float,
    close_in_range_pct: float,
    regime_tier: str,
    flow_history: list[float] | None,
    hold_days: int,
    config: StrategySettings,
) -> tuple[bool, str]:
    """Inverted carry logic: default = carry, flatten only when conditions met.

    Returns (should_carry, decision_path).
    """
    # Hard flatten: deep loss
    if unrealized_r < config.pb_v2_flatten_loss_r:
        return False, "flatten_loss"

    # Regime C with insufficient profit
    if regime_tier == "C" and unrealized_r < config.pb_v2_flatten_regime_c_min_r:
        return False, "flatten_regime_c"

    # Flow reversal (past grace period)
    if hold_days > config.pb_v2_flow_grace_days:
        if flow_history is not None and len(flow_history) >= 2:
            if all(v < 0 for v in flow_history[-2:]):
                return False, "flatten_flow_reversal"

    # Time stop (research parity: backtest _should_flatten_v2 checks this)
    if hold_days >= config.pb_max_hold_days:
        return False, "flatten_time_stop"

    return True, "carry"


def _route_carry_param(route_family: str, suffix: str, config: StrategySettings) -> float:
    """Look up per-route carry parameter, fall back to global."""
    return _route_param(route_family, f"carry_{suffix}", config)


def carry_quality_gate(
    route_family: str,
    close_in_range_pct: float,
    max_mfe_r: float,
    config: StrategySettings,
) -> bool:
    """Per-route carry quality gate (research parity)."""
    close_min = _route_carry_param(route_family, "close_pct_min", config)
    mfe_min = _route_carry_param(route_family, "mfe_gate_r", config)
    return close_in_range_pct >= close_min and max_mfe_r >= mfe_min


def compute_overnight_stop(
    entry_price: float,
    current_stop: float,
    risk_per_share: float,
    unrealized_r: float,
    config: StrategySettings,
) -> float:
    """Compute profit-lock overnight stop (research parity).

    Locks in profit above close_r - profit_lock_r threshold, ratcheting
    the stop up for profitable positions.
    """
    profit_lock_r = config.pb_v2_carry_profit_lock_r
    overnight_stop = entry_price + max(0.0, unrealized_r - profit_lock_r) * risk_per_share
    return max(current_stop, overnight_stop)


# ---------------------------------------------------------------------------
# Full exit chain runner
# ---------------------------------------------------------------------------

def run_exit_chain(
    state: PBSymbolState,
    bar: Bar,
    now: datetime,
    unrealized_r: float,
    max_mfe_r: float,
    ema10_value: float | None,
    rsi_value: float | None,
    session_vwap: float | None,
    hold_days: int,
    flow_history: list[float] | None,
    recent_5m_bars: list[Bar],
    quick_exit_loss_r: float,
    config: StrategySettings,
    stale_exit_bars: int = 0,
    stale_exit_min_r: float = 0.0,
) -> tuple[bool, str]:
    """Run the complete exit priority chain (research engine order).

    Returns (should_exit, reason).
    """
    # 1. Stop hit
    hit, reason = check_stop_hit(bar.low, state.stop_level)
    if hit:
        return True, reason

    # 2. Quick exit (2-bar window, route-specific loss_r)
    close_below_vwap = session_vwap is not None and bar.close < session_vwap
    hit, reason = check_quick_exit(
        state.hold_bars, unrealized_r, close_below_vwap,
        threshold_bars=2, loss_r=quick_exit_loss_r,
    )
    if hit:
        return True, reason

    # 2.5. Stale exit (force-close, distinct from stale tighten)
    hit, reason = check_stale_exit(
        state.hold_bars, max_mfe_r, stale_exit_bars, stale_exit_min_r,
    )
    if hit:
        return True, reason

    # 3. EMA reversion
    if config.pb_v2_ema_reversion_exit:
        hit, reason = check_ema_reversion(
            bar.close, ema10_value, unrealized_r,
            min_r=config.pb_v2_ema_reversion_min_r,
        )
        if hit:
            return True, reason

    # 4. RSI exit
    hit, reason = check_rsi_exit(rsi_value, state.route_family, config)
    if hit:
        return True, reason

    # 5. Time stop (per-route max_hold_days)
    max_hold = int(_route_param(state.route_family, "max_hold_days", config))
    hit, reason = check_time_stop(hold_days, max_hold)
    if hit:
        return True, reason

    # 6. VWAP fail (structural: descending highs + CPR gate, per-route)
    hit, reason = check_vwap_fail(
        recent_5m_bars,
        session_vwap,
        lookback=int(_route_param(state.route_family, "vwap_fail_lookback_bars", config)),
        cpr_max=_route_param(state.route_family, "vwap_fail_cpr_max", config, default=-1.0),
    )
    if hit:
        return True, reason

    # 7. EOD flatten
    hit, reason = check_eod_flatten(now, config)
    if hit:
        return True, reason

    return False, ""
