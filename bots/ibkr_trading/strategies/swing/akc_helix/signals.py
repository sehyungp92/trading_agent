"""AKC-Helix Swing — setup detection (stateless pure functions).

Four setup classes:
  - Class A: 4H hidden divergence continuation (spec s10.2)
  - Class B: 1H hidden divergence continuation (spec s10.3)
  - Class C: 4H classic divergence reversal, gated (spec s10.4)
  - Class D: 1H no-div momentum continuation, trend-only (spec s10.5)
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from .config import (
    CLASS_A_SIZE_TREND,
    CLASS_A_SIZE_CHOP,
    CLASS_A_SIZE_COUNTER,
    CLASS_B_SIZE_TREND,
    CLASS_B_SIZE_CHOP,
    CLASS_B_SIZE_COUNTER,
    CLASS_C_SIZE_CHOP,
    CLASS_C_SIZE_COUNTER,
    CLASS_C_SIZE_TREND,
    CLASS_D_SIZE_TREND,
    CLASS_D_FRESH_BREAK_ATR,
    CLASS_D_HIST_SLOPE_LOOKBACK,
    CLASS_D_MOM_LOOKBACK,
    CLASS_D_MAX_ARM_OVEREXT_ATR,
    CLASS_D_MAX_DAILY_EXTENSION_ATR,
    CLASS_D_MAX_ENTRY_STOP_ATR,
    CLASS_D_MAX_PULLBACK_ATR,
    CLASS_D_MAX_PIVOT2_AGE_BARS,
    CLASS_D_MIN_HIST_DELTA_ATR,
    CLASS_D_MIN_MACD_DELTA_ATR,
    CLASS_D_MIN_PIVOT_SEP_BARS,
    CLASS_D_MIN_PULLBACK_ATR,
    DIV_MAG_MIN_HISTORY,
    DIV_MAG_DEFAULT_THRESHOLD,
    DIV_MAG_FLOOR,
    DIV_MAG_PERCENTILE,
    STOP_1H_HIGHVOL,
    STOP_1H_STD,
    STOP_4H_MULT,
    HIGH_VOL_PCT,
    SymbolConfig,
)
from .indicators import calc_buffer
from .models import (
    DailyState,
    Direction,
    Pivot,
    PivotKind,
    PivotStore,
    Regime,
    SetupClass,
    SetupInstance,
    SetupState,
    TFState,
)


# ---------------------------------------------------------------------------
# Direction / regime alignment
# ---------------------------------------------------------------------------

def is_trend_aligned(direction: Direction, regime: Regime) -> bool:
    """LONG+BULL or SHORT+BEAR."""
    if direction == Direction.LONG and regime == Regime.BULL:
        return True
    if direction == Direction.SHORT and regime == Regime.BEAR:
        return True
    return False


# ---------------------------------------------------------------------------
# Divergence magnitude helpers (spec s9)
# ---------------------------------------------------------------------------

def compute_div_magnitude(p1_macd: float, p2_macd: float, p2_atr: float) -> float:
    """Normalised divergence magnitude = |MACD(P1) - MACD(P2)| / ATR_TF(P2)."""
    if p2_atr <= 0:
        return 0.0
    return abs(p1_macd - p2_macd) / p2_atr


def div_mag_passes(
    norm: float,
    history: list[float],
    min_history: int = DIV_MAG_MIN_HISTORY,
    default_threshold: float = DIV_MAG_DEFAULT_THRESHOLD,
    floor: float = DIV_MAG_FLOOR,
    percentile: int = DIV_MAG_PERCENTILE,
) -> bool:
    """Return True if divergence magnitude passes the adaptive threshold.

    If history < min_history, use default_threshold.
    Otherwise threshold = max(floor, percentile(history, pct)).
    """
    if len(history) < min_history:
        threshold = default_threshold
    else:
        sorted_h = sorted(history)
        idx = max(0, int(len(sorted_h) * percentile / 100) - 1)
        threshold = max(floor, sorted_h[idx])
    return norm >= threshold


# ---------------------------------------------------------------------------
# Class A: 4H Hidden Divergence Continuation (spec s10.2)
# ---------------------------------------------------------------------------

def detect_class_a(
    symbol: str,
    pivots_4h: PivotStore,
    daily: DailyState,
    tf4h: TFState,
    cfg: SymbolConfig,
    div_mag_history: list[float],
    now: Optional[datetime] = None,
) -> Optional[SetupInstance]:
    """Detect a Class A (4H hidden divergence continuation) setup.

    Requirements:
      - >= 2 pivots on 4H bars
      - Higher low (long) / lower high (short) on 4H
      - Hidden divergence: MACD diverges from price at P1 vs P2
      - Divergence magnitude passes adaptive threshold
      - BOS between P1 and P2
      - No ADX gate, no 4H regime gate

    Stop: P2 - 0.75 * ATR4H (long) or P2 + 0.75 * ATR4H (short)
    Size: regime-dependent (1.0 trend, 0.65 chop, 0.50 counter)
    """
    setup = _try_class_a_long(symbol, pivots_4h, daily, tf4h, cfg, div_mag_history, now)
    if setup is not None:
        return setup
    return _try_class_a_short(symbol, pivots_4h, daily, tf4h, cfg, div_mag_history, now)


def _try_class_a_long(
    symbol: str,
    pivots_4h: PivotStore,
    daily: DailyState,
    tf4h: TFState,
    cfg: SymbolConfig,
    div_mag_history: list[float],
    now: Optional[datetime],
) -> Optional[SetupInstance]:
    if len(pivots_4h.lows) < 2:
        return None

    L1 = pivots_4h.lows[-2]
    L2 = pivots_4h.lows[-1]

    # Higher low (price)
    if L2.price <= L1.price:
        return None

    # Hidden bullish divergence: MACD makes lower low while price makes higher low
    if L2.macd_line >= L1.macd_line:
        return None

    # Divergence magnitude filter
    dmn = compute_div_magnitude(L1.macd_line, L2.macd_line, L2.atr_tf)
    if not div_mag_passes(dmn, div_mag_history):
        return None

    # BOS: most recent high between L1 and L2
    highs_between = [p for p in pivots_4h.highs if L1.ts < p.ts < L2.ts]
    if not highs_between:
        return None
    bos = max(highs_between, key=lambda p: p.price)

    # Stop and buffer
    buffer = calc_buffer(cfg.tick_size, L2.atr_tf, cfg.is_etf)
    stop0 = L2.price - STOP_4H_MULT * L2.atr_tf

    # Corridor cap
    entry_to_stop = bos.price + buffer - stop0
    cap_mult = _corridor_cap_mult(daily, Direction.LONG)
    corridor_cap = cap_mult * daily.atr_d
    if corridor_cap > 0 and entry_to_stop > corridor_cap:
        return None

    # Regime-based sizing
    size_mult = _class_a_size_mult(Direction.LONG, daily.regime)

    return SetupInstance(
        symbol=symbol,
        setup_class=SetupClass.CLASS_A,
        direction=Direction.LONG,
        origin_tf="4H",
        state=SetupState.NEW,
        created_ts=now,
        pivot_1=L1,
        pivot_2=L2,
        bos_pivot=bos,
        bos_level=bos.price + buffer,
        stop0=stop0,
        buffer=buffer,
        adx_at_entry=daily.adx,
        div_mag_norm=dmn,
        setup_size_mult=size_mult,
    )


def _try_class_a_short(
    symbol: str,
    pivots_4h: PivotStore,
    daily: DailyState,
    tf4h: TFState,
    cfg: SymbolConfig,
    div_mag_history: list[float],
    now: Optional[datetime],
) -> Optional[SetupInstance]:
    if len(pivots_4h.highs) < 2:
        return None

    H1 = pivots_4h.highs[-2]
    H2 = pivots_4h.highs[-1]

    # Lower high (price)
    if H2.price >= H1.price:
        return None

    # Hidden bearish divergence: MACD makes higher high while price makes lower high
    if H2.macd_line <= H1.macd_line:
        return None

    # Divergence magnitude filter
    dmn = compute_div_magnitude(H1.macd_line, H2.macd_line, H2.atr_tf)
    if not div_mag_passes(dmn, div_mag_history):
        return None

    # BOS: most recent low between H1 and H2
    lows_between = [p for p in pivots_4h.lows if H1.ts < p.ts < H2.ts]
    if not lows_between:
        return None
    bos = min(lows_between, key=lambda p: p.price)

    buffer = calc_buffer(cfg.tick_size, H2.atr_tf, cfg.is_etf)
    stop0 = H2.price + STOP_4H_MULT * H2.atr_tf

    entry_to_stop = stop0 - (bos.price - buffer)
    cap_mult = _corridor_cap_mult(daily, Direction.SHORT)
    corridor_cap = cap_mult * daily.atr_d
    if corridor_cap > 0 and entry_to_stop > corridor_cap:
        return None

    size_mult = _class_a_size_mult(Direction.SHORT, daily.regime)

    return SetupInstance(
        symbol=symbol,
        setup_class=SetupClass.CLASS_A,
        direction=Direction.SHORT,
        origin_tf="4H",
        state=SetupState.NEW,
        created_ts=now,
        pivot_1=H1,
        pivot_2=H2,
        bos_pivot=bos,
        bos_level=bos.price - buffer,
        stop0=stop0,
        buffer=buffer,
        adx_at_entry=daily.adx,
        div_mag_norm=dmn,
        setup_size_mult=size_mult,
    )


def _class_a_size_mult(direction: Direction, regime: Regime) -> float:
    """Regime-dependent sizing for Class A (spec s10.1)."""
    if is_trend_aligned(direction, regime):
        return CLASS_A_SIZE_TREND
    if regime == Regime.CHOP:
        return CLASS_A_SIZE_CHOP
    return CLASS_A_SIZE_COUNTER


# ---------------------------------------------------------------------------
# Class B: 1H Hidden Divergence Continuation (spec s10.3)
# ---------------------------------------------------------------------------

def detect_class_b(
    symbol: str,
    pivots_1h: PivotStore,
    daily: DailyState,
    tf1h: TFState,
    cfg: SymbolConfig,
    div_mag_history: list[float],
    now: datetime | None = None,
) -> SetupInstance | None:
    """Detect a Class B (1H hidden divergence continuation) setup.

    Requirements:
      - >= 2 pivots on 1H bars
      - Higher low (long) / lower high (short) on 1H
      - Hidden divergence: MACD diverges from price at P1 vs P2
      - Divergence magnitude passes adaptive threshold
      - BOS between P1 and P2

    Stop: P2 - 0.50 * ATR1H (std) or 0.75 * ATR1H (high vol)
    Size: regime-dependent (1.0 trend, 0.65 chop, 0.50 counter)
    """
    setup = _try_class_b_long(symbol, pivots_1h, daily, tf1h, cfg, div_mag_history, now)
    if setup is not None:
        return setup
    return _try_class_b_short(symbol, pivots_1h, daily, tf1h, cfg, div_mag_history, now)


def _try_class_b_long(
    symbol: str,
    pivots_1h: PivotStore,
    daily: DailyState,
    tf1h: TFState,
    cfg: SymbolConfig,
    div_mag_history: list[float],
    now: datetime | None,
) -> SetupInstance | None:
    if len(pivots_1h.lows) < 2:
        return None

    L1 = pivots_1h.lows[-2]
    L2 = pivots_1h.lows[-1]

    # Higher low (price)
    if L2.price <= L1.price:
        return None

    # Hidden bullish divergence: MACD makes lower low while price makes higher low
    if L2.macd_line >= L1.macd_line:
        return None

    # Divergence magnitude filter
    dmn = compute_div_magnitude(L1.macd_line, L2.macd_line, L2.atr_tf)
    if not div_mag_passes(dmn, div_mag_history):
        return None

    # BOS: most recent high between L1 and L2
    highs_between = [p for p in pivots_1h.highs if L1.ts < p.ts < L2.ts]
    if not highs_between:
        return None
    bos = max(highs_between, key=lambda p: p.price)

    # Stop
    vol_pct = daily.vol_pct
    mult = STOP_1H_HIGHVOL if vol_pct > HIGH_VOL_PCT else STOP_1H_STD
    buffer = calc_buffer(cfg.tick_size, L2.atr_tf, cfg.is_etf)
    stop0 = L2.price - mult * L2.atr_tf

    # Corridor cap
    entry_to_stop = bos.price + buffer - stop0
    cap_mult = _corridor_cap_mult(daily, Direction.LONG)
    corridor_cap = cap_mult * daily.atr_d
    if corridor_cap > 0 and entry_to_stop > corridor_cap:
        return None

    size_mult = _class_b_size_mult(Direction.LONG, daily.regime)

    return SetupInstance(
        symbol=symbol,
        setup_class=SetupClass.CLASS_B,
        direction=Direction.LONG,
        origin_tf="1H",
        state=SetupState.NEW,
        created_ts=now,
        pivot_1=L1,
        pivot_2=L2,
        bos_pivot=bos,
        bos_level=bos.price + buffer,
        stop0=stop0,
        buffer=buffer,
        adx_at_entry=daily.adx,
        div_mag_norm=dmn,
        setup_size_mult=size_mult,
    )


def _try_class_b_short(
    symbol: str,
    pivots_1h: PivotStore,
    daily: DailyState,
    tf1h: TFState,
    cfg: SymbolConfig,
    div_mag_history: list[float],
    now: datetime | None,
) -> SetupInstance | None:
    if len(pivots_1h.highs) < 2:
        return None

    H1 = pivots_1h.highs[-2]
    H2 = pivots_1h.highs[-1]

    # Lower high (price)
    if H2.price >= H1.price:
        return None

    # Hidden bearish divergence: MACD makes higher high while price makes lower high
    if H2.macd_line <= H1.macd_line:
        return None

    # Divergence magnitude filter
    dmn = compute_div_magnitude(H1.macd_line, H2.macd_line, H2.atr_tf)
    if not div_mag_passes(dmn, div_mag_history):
        return None

    # BOS: most recent low between H1 and H2
    lows_between = [p for p in pivots_1h.lows if H1.ts < p.ts < H2.ts]
    if not lows_between:
        return None
    bos = min(lows_between, key=lambda p: p.price)

    vol_pct = daily.vol_pct
    mult = STOP_1H_HIGHVOL if vol_pct > HIGH_VOL_PCT else STOP_1H_STD
    buffer = calc_buffer(cfg.tick_size, H2.atr_tf, cfg.is_etf)
    stop0 = H2.price + mult * H2.atr_tf

    entry_to_stop = stop0 - (bos.price - buffer)
    cap_mult = _corridor_cap_mult(daily, Direction.SHORT)
    corridor_cap = cap_mult * daily.atr_d
    if corridor_cap > 0 and entry_to_stop > corridor_cap:
        return None

    size_mult = _class_b_size_mult(Direction.SHORT, daily.regime)

    return SetupInstance(
        symbol=symbol,
        setup_class=SetupClass.CLASS_B,
        direction=Direction.SHORT,
        origin_tf="1H",
        state=SetupState.NEW,
        created_ts=now,
        pivot_1=H1,
        pivot_2=H2,
        bos_pivot=bos,
        bos_level=bos.price - buffer,
        stop0=stop0,
        buffer=buffer,
        adx_at_entry=daily.adx,
        div_mag_norm=dmn,
        setup_size_mult=size_mult,
    )


def _class_b_size_mult(direction: Direction, regime: Regime) -> float:
    """Regime-dependent sizing for Class B (spec s10.1)."""
    if is_trend_aligned(direction, regime):
        return CLASS_B_SIZE_TREND
    if regime == Regime.CHOP:
        return CLASS_B_SIZE_CHOP
    return CLASS_B_SIZE_COUNTER


# ---------------------------------------------------------------------------
# Class C: 4H Classic Divergence Reversal, Gated (spec s10.4)
# ---------------------------------------------------------------------------

def detect_class_c(
    symbol: str,
    pivots_4h: PivotStore,
    daily: DailyState,
    tf4h: TFState,
    cfg: SymbolConfig,
    div_mag_history: list[float],
    now: datetime | None = None,
) -> SetupInstance | None:
    """Detect a Class C (4H classic divergence reversal) setup.

    Gating: allowed only if any of:
      - regime == chop
      - trend_strength today < trend_strength 3 days ago
      - abs(close - EMA_fast) > 1.5 * ATRd  (extension)

    Requirements:
      - >= 2 pivots on 4H bars
      - Classic divergence (price higher high but MACD lower high → short)
      - Divergence magnitude passes adaptive threshold
      - BOS between P1 and P2

    Stop: P2 + 0.75 * ATR4H (short) or P2 - 0.75 * ATR4H (long)
    """
    # Gating check (spec s10.4)
    if not _class_c_gate_ok(daily):
        return None

    setup = _try_class_c_short(symbol, pivots_4h, daily, tf4h, cfg, div_mag_history, now)
    if setup is not None:
        return setup
    return _try_class_c_long(symbol, pivots_4h, daily, tf4h, cfg, div_mag_history, now)


def _class_c_gate_ok(daily: DailyState) -> bool:
    """Check gating conditions for Class C (tighter: extension OR chop only)."""
    # regime == chop
    if daily.regime == Regime.CHOP:
        return True
    # extension: price far from EMA_fast
    if daily.atr_d > 0 and abs(daily.close - daily.ema_fast) > 1.5 * daily.atr_d:
        return True
    return False


def _try_class_c_short(
    symbol: str,
    pivots_4h: PivotStore,
    daily: DailyState,
    tf4h: TFState,
    cfg: SymbolConfig,
    div_mag_history: list[float],
    now: datetime | None,
) -> SetupInstance | None:
    """Classic bearish divergence: price higher high, MACD lower high → short."""
    if len(pivots_4h.highs) < 2:
        return None

    H1 = pivots_4h.highs[-2]
    H2 = pivots_4h.highs[-1]

    # Higher high (price)
    if H2.price <= H1.price:
        return None

    # Classic bearish divergence: MACD makes lower high while price makes higher high
    if H2.macd_line >= H1.macd_line:
        return None

    # Divergence magnitude filter
    dmn = compute_div_magnitude(H1.macd_line, H2.macd_line, H2.atr_tf)
    if not div_mag_passes(dmn, div_mag_history):
        return None

    # BOS: most recent low between H1 and H2
    lows_between = [p for p in pivots_4h.lows if H1.ts < p.ts < H2.ts]
    if not lows_between:
        return None
    bos = min(lows_between, key=lambda p: p.price)

    buffer = calc_buffer(cfg.tick_size, H2.atr_tf, cfg.is_etf)
    stop0 = H2.price + STOP_4H_MULT * H2.atr_tf

    entry_to_stop = stop0 - (bos.price - buffer)
    cap_mult = _corridor_cap_mult(daily, Direction.SHORT)
    corridor_cap = cap_mult * daily.atr_d
    if corridor_cap > 0 and entry_to_stop > corridor_cap:
        return None

    size_mult = _class_c_size_mult(Direction.SHORT, daily.regime)

    return SetupInstance(
        symbol=symbol,
        setup_class=SetupClass.CLASS_C,
        direction=Direction.SHORT,
        origin_tf="4H",
        state=SetupState.NEW,
        created_ts=now,
        pivot_1=H1,
        pivot_2=H2,
        bos_pivot=bos,
        bos_level=bos.price - buffer,
        stop0=stop0,
        buffer=buffer,
        adx_at_entry=daily.adx,
        div_mag_norm=dmn,
        setup_size_mult=size_mult,
    )


def _try_class_c_long(
    symbol: str,
    pivots_4h: PivotStore,
    daily: DailyState,
    tf4h: TFState,
    cfg: SymbolConfig,
    div_mag_history: list[float],
    now: datetime | None,
) -> SetupInstance | None:
    """Classic bullish divergence: price lower low, MACD higher low → long."""
    if len(pivots_4h.lows) < 2:
        return None

    L1 = pivots_4h.lows[-2]
    L2 = pivots_4h.lows[-1]

    # Lower low (price)
    if L2.price >= L1.price:
        return None

    # Classic bullish divergence: MACD makes higher low while price makes lower low
    if L2.macd_line <= L1.macd_line:
        return None

    # Divergence magnitude filter
    dmn = compute_div_magnitude(L1.macd_line, L2.macd_line, L2.atr_tf)
    if not div_mag_passes(dmn, div_mag_history):
        return None

    # BOS: most recent high between L1 and L2
    highs_between = [p for p in pivots_4h.highs if L1.ts < p.ts < L2.ts]
    if not highs_between:
        return None
    bos = max(highs_between, key=lambda p: p.price)

    buffer = calc_buffer(cfg.tick_size, L2.atr_tf, cfg.is_etf)
    stop0 = L2.price - STOP_4H_MULT * L2.atr_tf

    entry_to_stop = bos.price + buffer - stop0
    cap_mult = _corridor_cap_mult(daily, Direction.LONG)
    corridor_cap = cap_mult * daily.atr_d
    if corridor_cap > 0 and entry_to_stop > corridor_cap:
        return None

    size_mult = _class_c_size_mult(Direction.LONG, daily.regime)

    return SetupInstance(
        symbol=symbol,
        setup_class=SetupClass.CLASS_C,
        direction=Direction.LONG,
        origin_tf="4H",
        state=SetupState.NEW,
        created_ts=now,
        pivot_1=L1,
        pivot_2=L2,
        bos_pivot=bos,
        bos_level=bos.price + buffer,
        stop0=stop0,
        buffer=buffer,
        adx_at_entry=daily.adx,
        div_mag_norm=dmn,
        setup_size_mult=size_mult,
    )


def _class_c_size_mult(direction: Direction, regime: Regime) -> float:
    """Regime-dependent sizing for Class C (spec s10.1).

    Classic divergence reversal:
      chop: 1.00
      countertrend (reversing into trend): 0.85
      trend-aligned (fading trend): 0.40
    """
    if regime == Regime.CHOP:
        return CLASS_C_SIZE_CHOP
    # "countertrend" for Class C means the reversal aligns with the trend
    # (i.e., reversing INTO the trend direction)
    if is_trend_aligned(direction, regime):
        return CLASS_C_SIZE_TREND  # fading the trend → smallest size
    return CLASS_C_SIZE_COUNTER  # reversing into trend → larger size


# ---------------------------------------------------------------------------
# Class D: 1H No-Div Momentum Continuation (spec s10.5)
# ---------------------------------------------------------------------------

def _hours_between(start: Optional[datetime], end: Optional[datetime]) -> float:
    if start is None or end is None:
        return 0.0
    try:
        return abs((end - start).total_seconds()) / 3600.0
    except TypeError:
        start_naive = start.replace(tzinfo=None)
        end_naive = end.replace(tzinfo=None)
        return abs((end_naive - start_naive).total_seconds()) / 3600.0


def _atr_norm(value: float, atr_value: float) -> float:
    if atr_value <= 0:
        return 0.0
    return float(value) / float(atr_value)


def _class_d_hist_slope_passes(direction: Direction, tf1h: TFState) -> bool:
    lookback = int(CLASS_D_HIST_SLOPE_LOOKBACK or 0)
    if lookback <= 0:
        return True

    hist = tf1h.macd_hist_history
    if len(hist) < lookback + 1 or tf1h.atr <= 0:
        return False

    raw_delta = float(tf1h.macd_hist) - float(hist[-1 - lookback])
    directional_delta = raw_delta if direction == Direction.LONG else -raw_delta
    return _atr_norm(directional_delta, tf1h.atr) >= float(CLASS_D_MIN_HIST_DELTA_ATR or 0.0)


def _class_d_quality_passes(
    direction: Direction,
    pivot_1: Pivot,
    pivot_2: Pivot,
    bos: Pivot,
    bos_level: float,
    entry_to_stop: float,
    daily: DailyState,
    tf1h: TFState,
    now: Optional[datetime],
) -> bool:
    atr_ref = pivot_2.atr_tf if pivot_2.atr_tf > 0 else tf1h.atr
    if atr_ref <= 0:
        return False

    min_sep = int(CLASS_D_MIN_PIVOT_SEP_BARS or 0)
    if min_sep > 0 and _hours_between(pivot_1.ts, pivot_2.ts) < min_sep:
        return False

    max_age = int(CLASS_D_MAX_PIVOT2_AGE_BARS or 0)
    if max_age > 0 and now is not None and _hours_between(pivot_2.ts, now) > max_age:
        return False

    pullback_atr = _atr_norm(abs(float(bos.price) - float(pivot_2.price)), atr_ref)
    if CLASS_D_MIN_PULLBACK_ATR > 0 and pullback_atr < CLASS_D_MIN_PULLBACK_ATR:
        return False
    if CLASS_D_MAX_PULLBACK_ATR > 0 and pullback_atr > CLASS_D_MAX_PULLBACK_ATR:
        return False

    if CLASS_D_MAX_ENTRY_STOP_ATR > 0 and _atr_norm(entry_to_stop, atr_ref) > CLASS_D_MAX_ENTRY_STOP_ATR:
        return False

    if CLASS_D_MAX_ARM_OVEREXT_ATR < 999.0 and tf1h.atr > 0:
        overext = (tf1h.close - bos_level) if direction == Direction.LONG else (bos_level - tf1h.close)
        if _atr_norm(max(0.0, overext), tf1h.atr) > CLASS_D_MAX_ARM_OVEREXT_ATR:
            return False

    macd_delta = (
        float(tf1h.macd_line) - float(pivot_2.macd_line)
        if direction == Direction.LONG
        else float(pivot_2.macd_line) - float(tf1h.macd_line)
    )
    if CLASS_D_MIN_MACD_DELTA_ATR > 0 and _atr_norm(macd_delta, atr_ref) < CLASS_D_MIN_MACD_DELTA_ATR:
        return False

    if not _class_d_hist_slope_passes(direction, tf1h):
        return False

    if CLASS_D_MAX_DAILY_EXTENSION_ATR > 0 and daily.atr_d > 0:
        extension = abs(float(daily.close) - float(daily.ema_fast)) / float(daily.atr_d)
        if extension > CLASS_D_MAX_DAILY_EXTENSION_ATR:
            return False

    return True


def detect_class_d(
    symbol: str,
    pivots_1h: PivotStore,
    daily: DailyState,
    tf1h: TFState,
    cfg: SymbolConfig,
    now: Optional[datetime] = None,
) -> Optional[SetupInstance]:
    """Detect a Class D (1H momentum continuation) setup.

    Requirements:
      - Trend-aligned only (daily.regime BULL for long, BEAR for short)
      - >= 2 pivots on 1H bars
      - Higher low (long) / lower high (short) on 1H
      - MACD line momentum: current > P2 AND current > lookback(3)
      - BOS between P1 and P2

    Stop: P2 - 0.50 * ATR1H (std) or 0.75 * ATR1H (high vol)
    Size: CLASS_D_SIZE_TREND (0.80)
    """
    setup = _try_class_d_long(symbol, pivots_1h, daily, tf1h, cfg, now)
    if setup is not None:
        return setup
    return _try_class_d_short(symbol, pivots_1h, daily, tf1h, cfg, now)


def _try_class_d_long(
    symbol: str,
    pivots_1h: PivotStore,
    daily: DailyState,
    tf1h: TFState,
    cfg: SymbolConfig,
    now: Optional[datetime],
) -> Optional[SetupInstance]:
    # Trend-aligned only: daily regime must be BULL for long
    if daily.regime != Regime.BULL:
        return None

    if len(pivots_1h.lows) < 2:
        return None

    L1 = pivots_1h.lows[-2]
    L2 = pivots_1h.lows[-1]

    # Higher low
    if L2.price <= L1.price:
        return None

    # Skip if hidden divergence exists — Class B would have taken priority
    # Hidden bullish div: price higher low BUT MACD lower low
    if L2.macd_line < L1.macd_line:
        return None

    # Momentum filter on MACD line:
    # 1) current MACD line > MACD line at L2
    if tf1h.macd_line <= L2.macd_line:
        return None
    # 2) current MACD line > MACD line lookback bars ago
    ml_hist = tf1h.macd_line_history
    if len(ml_hist) >= CLASS_D_MOM_LOOKBACK + 1:
        if tf1h.macd_line <= ml_hist[-1 - CLASS_D_MOM_LOOKBACK]:
            return None
    else:
        return None

    # BOS: most recent high between L1 and L2
    highs_between = [p for p in pivots_1h.highs if L1.ts < p.ts < L2.ts]
    if not highs_between:
        return None
    bos = max(highs_between, key=lambda p: p.price)

    # Stop
    vol_pct = daily.vol_pct
    mult = STOP_1H_HIGHVOL if vol_pct > HIGH_VOL_PCT else STOP_1H_STD
    buffer = calc_buffer(cfg.tick_size, L2.atr_tf, cfg.is_etf)
    stop0 = L2.price - mult * L2.atr_tf
    bos_level = bos.price + buffer
    if CLASS_D_FRESH_BREAK_ATR > 0 and tf1h.atr > 0:
        bos_level = max(bos_level, tf1h.close + CLASS_D_FRESH_BREAK_ATR * tf1h.atr)

    # Corridor cap
    entry_to_stop = bos_level - stop0
    cap_mult = _corridor_cap_mult(daily, Direction.LONG)
    corridor_cap = cap_mult * daily.atr_d
    if corridor_cap > 0 and entry_to_stop > corridor_cap:
        return None

    if not _class_d_quality_passes(
        Direction.LONG, L1, L2, bos, bos_level, entry_to_stop, daily, tf1h, now,
    ):
        return None

    return SetupInstance(
        symbol=symbol,
        setup_class=SetupClass.CLASS_D,
        direction=Direction.LONG,
        origin_tf="1H",
        state=SetupState.NEW,
        created_ts=now,
        pivot_1=L1,
        pivot_2=L2,
        bos_pivot=bos,
        bos_level=bos_level,
        stop0=stop0,
        buffer=buffer,
        adx_at_entry=daily.adx,
        setup_size_mult=CLASS_D_SIZE_TREND,
    )


def _try_class_d_short(
    symbol: str,
    pivots_1h: PivotStore,
    daily: DailyState,
    tf1h: TFState,
    cfg: SymbolConfig,
    now: Optional[datetime],
) -> Optional[SetupInstance]:
    # Trend-aligned only: daily regime must be BEAR for short
    if daily.regime != Regime.BEAR:
        return None

    if len(pivots_1h.highs) < 2:
        return None

    H1 = pivots_1h.highs[-2]
    H2 = pivots_1h.highs[-1]

    # Lower high
    if H2.price >= H1.price:
        return None

    # Skip if hidden divergence exists — Class B would have taken priority
    # Hidden bearish div: price lower high BUT MACD higher high
    if H2.macd_line > H1.macd_line:
        return None

    # Momentum filter on MACD line:
    # 1) current MACD line < MACD line at H2 (bearish momentum increasing)
    if tf1h.macd_line >= H2.macd_line:
        return None
    # 2) current MACD line < MACD line lookback bars ago
    ml_hist = tf1h.macd_line_history
    if len(ml_hist) >= CLASS_D_MOM_LOOKBACK + 1:
        if tf1h.macd_line >= ml_hist[-1 - CLASS_D_MOM_LOOKBACK]:
            return None
    else:
        return None

    # BOS: most recent low between H1 and H2
    lows_between = [p for p in pivots_1h.lows if H1.ts < p.ts < H2.ts]
    if not lows_between:
        return None
    bos = min(lows_between, key=lambda p: p.price)

    vol_pct = daily.vol_pct
    mult = STOP_1H_HIGHVOL if vol_pct > HIGH_VOL_PCT else STOP_1H_STD
    buffer = calc_buffer(cfg.tick_size, H2.atr_tf, cfg.is_etf)
    stop0 = H2.price + mult * H2.atr_tf
    bos_level = bos.price - buffer
    if CLASS_D_FRESH_BREAK_ATR > 0 and tf1h.atr > 0:
        bos_level = min(bos_level, tf1h.close - CLASS_D_FRESH_BREAK_ATR * tf1h.atr)

    entry_to_stop = stop0 - bos_level
    cap_mult = _corridor_cap_mult(daily, Direction.SHORT)
    corridor_cap = cap_mult * daily.atr_d
    if corridor_cap > 0 and entry_to_stop > corridor_cap:
        return None

    if not _class_d_quality_passes(
        Direction.SHORT, H1, H2, bos, bos_level, entry_to_stop, daily, tf1h, now,
    ):
        return None

    return SetupInstance(
        symbol=symbol,
        setup_class=SetupClass.CLASS_D,
        direction=Direction.SHORT,
        origin_tf="1H",
        state=SetupState.NEW,
        created_ts=now,
        pivot_1=H1,
        pivot_2=H2,
        bos_pivot=bos,
        bos_level=bos_level,
        stop0=stop0,
        buffer=buffer,
        adx_at_entry=daily.adx,
        setup_size_mult=CLASS_D_SIZE_TREND,
    )


# ---------------------------------------------------------------------------
# Add-on detection (spec s15.2)
# ---------------------------------------------------------------------------

def detect_add_setup(
    symbol: str,
    direction: Direction,
    pivots_1h: PivotStore,
    tf1h: TFState,
    parent_bos_level: float,
    cfg: SymbolConfig,
    daily: DailyState,
    now: Optional[datetime] = None,
) -> Optional[SetupInstance]:
    """Detect an add-on entry on the 1H timeframe.

    Uses structural confirmation: new BOS above parent entry for long,
    below for short.  Requires MACD momentum confirmation per spec s15.2:
      macd[t] > macd[t-3] AND macd[t] > macd(L2)
    """
    if direction == Direction.LONG:
        if len(pivots_1h.lows) < 2:
            return None
        L1 = pivots_1h.lows[-2]
        L2 = pivots_1h.lows[-1]
        if L2.price <= L1.price:
            return None

        # Momentum confirmation (spec s15.2)
        # macd[t] > macd(L2)
        if tf1h.macd_line <= L2.macd_line:
            return None
        # macd[t] > macd[t-3]
        ml_hist = tf1h.macd_line_history
        if len(ml_hist) >= CLASS_D_MOM_LOOKBACK + 1:
            if tf1h.macd_line <= ml_hist[-1 - CLASS_D_MOM_LOOKBACK]:
                return None
        else:
            return None

        highs_between = [p for p in pivots_1h.highs if L1.ts < p.ts < L2.ts]
        if not highs_between:
            return None
        bos = max(highs_between, key=lambda p: p.price)

        # Add BOS must be above the parent entry
        if bos.price <= parent_bos_level:
            return None

        buffer = calc_buffer(cfg.tick_size, L2.atr_tf, cfg.is_etf)
        stop0 = L2.price - STOP_1H_STD * L2.atr_tf

        return SetupInstance(
            symbol=symbol,
            setup_class=SetupClass.CLASS_D,
            direction=Direction.LONG,
            origin_tf="1H",
            state=SetupState.NEW,
            created_ts=now,
            pivot_1=L1,
            pivot_2=L2,
            bos_pivot=bos,
            bos_level=bos.price + buffer,
            stop0=stop0,
            buffer=buffer,
        )

    else:  # SHORT
        if len(pivots_1h.highs) < 2:
            return None
        H1 = pivots_1h.highs[-2]
        H2 = pivots_1h.highs[-1]
        if H2.price >= H1.price:
            return None

        # Momentum confirmation (spec s15.2)
        # macd[t] < macd(H2) (bearish momentum increasing)
        if tf1h.macd_line >= H2.macd_line:
            return None
        # macd[t] < macd[t-3]
        ml_hist = tf1h.macd_line_history
        if len(ml_hist) >= CLASS_D_MOM_LOOKBACK + 1:
            if tf1h.macd_line >= ml_hist[-1 - CLASS_D_MOM_LOOKBACK]:
                return None
        else:
            return None

        lows_between = [p for p in pivots_1h.lows if H1.ts < p.ts < H2.ts]
        if not lows_between:
            return None
        bos = min(lows_between, key=lambda p: p.price)

        if bos.price >= parent_bos_level:
            return None

        buffer = calc_buffer(cfg.tick_size, H2.atr_tf, cfg.is_etf)
        stop0 = H2.price + STOP_1H_STD * H2.atr_tf

        return SetupInstance(
            symbol=symbol,
            setup_class=SetupClass.CLASS_D,
            direction=Direction.SHORT,
            origin_tf="1H",
            state=SetupState.NEW,
            created_ts=now,
            pivot_1=H1,
            pivot_2=H2,
            bos_pivot=bos,
            bos_level=bos.price - buffer,
            stop0=stop0,
            buffer=buffer,
        )


# ---------------------------------------------------------------------------
# Structure invalidation (spec s12.2)
# ---------------------------------------------------------------------------

def is_structure_invalidated(
    setup: SetupInstance,
    pivots: PivotStore,
) -> bool:
    """Check if new pivot invalidates the setup structure (spec s12.2).

    Long: new pivot low <= L2 price, OR BoS superseded (new H_last redefines entry).
    Short: new pivot high >= H2 price, OR BoS superseded.
    """
    if setup.pivot_2 is None:
        return False

    if setup.direction == Direction.LONG:
        # Structure invalidation: new pivot low <= L2
        for p in pivots.lows:
            if p.ts > setup.pivot_2.ts and p.price <= setup.pivot_2.price:
                return True
        # BoS supersession: new pivot high after P2 materially above original BoS
        if setup.bos_pivot is not None:
            for p in pivots.highs:
                if p.ts > setup.pivot_2.ts and p.price > setup.bos_pivot.price:
                    return True
    else:
        for p in pivots.highs:
            if p.ts > setup.pivot_2.ts and p.price >= setup.pivot_2.price:
                return True
        if setup.bos_pivot is not None:
            for p in pivots.lows:
                if p.ts > setup.pivot_2.ts and p.price < setup.bos_pivot.price:
                    return True

    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _corridor_cap_mult(daily: DailyState, direction: Direction = Direction.FLAT) -> float:
    """Return corridor cap multiplier based on regime and trend alignment (spec s10.2).

    trend_aligned = setup direction matches regime (LONG+BULL or SHORT+BEAR).
    """
    if daily.regime == Regime.CHOP:
        return 1.3
    if direction != Direction.FLAT and is_trend_aligned(direction, daily.regime):
        return 1.6
    return 1.4
