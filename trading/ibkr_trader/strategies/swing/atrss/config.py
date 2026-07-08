"""ETRS vFinal strategy constants and per-symbol configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry

# ---------------------------------------------------------------------------
# Strategy identity
# ---------------------------------------------------------------------------
STRATEGY_ID = "ATRSS"

# ---------------------------------------------------------------------------
# Per-symbol indicator / execution parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolConfig:
    symbol: str
    # Daily indicator periods
    daily_ema_fast: int = 20
    daily_ema_slow: int = 55
    adx_period: int = 14
    atr_daily_period: int = 20
    # Hourly indicator periods
    atr_hourly_period: int = 48
    ema_mom_period: int = 20
    ema_pull_strong: int = 34
    ema_pull_normal: int = 50
    donchian_period: int = 26
    # ATR stop multipliers
    daily_mult: float = 2.2
    hourly_mult: float = 3.0
    # Chandelier trailing multiplier
    chand_mult: float = 3.0
    # Per-symbol ADX thresholds (spec Section 2.2)
    adx_on: int = 14
    adx_off: int = 18
    # Contract spec (for Instrument construction)
    tick_size: float = 0.25
    multiplier: float = 2.0
    exchange: str = "CME"
    trading_class: str = ""
    contract_expiry: str = ""
    sec_type: str = "FUT"
    primary_exchange: str = ""
    # Per-symbol limit band parameters (spec Section 6)
    limit_ticks: int = 2
    limit_pct: float = 0.0010
    # Per-symbol bad-fill slippage (spec Section 6)
    max_entry_slip_pct: float = 0.0015
    # Per-symbol base risk (spec Section 9)
    base_risk_pct: float = 0.016
    # Per-symbol short trade control
    shorts_enabled: bool = True
    # Per-symbol time/day filters (hours in ET, weekdays 0=Mon..6=Sun)
    blocked_hours_et: tuple[int, ...] = ()
    blocked_weekdays: tuple[int, ...] = ()
    # Per-symbol size reduction by month (month_number -> fraction, e.g. {12: 0.5})
    size_reduction_months: tuple[tuple[int, float], ...] = ()
    # Experiment A/B tracking
    experiment_id: str = ""
    experiment_variant: str = ""


_MICRO_CONFIGS: dict[str, SymbolConfig] = {
    "MNQ": SymbolConfig(
        symbol="MNQ",
        adx_on=18, adx_off=16,
        tick_size=0.25, multiplier=2.0,
        exchange="CME", trading_class="MNQ",
    ),
    "MCL": SymbolConfig(
        symbol="MCL",
        daily_mult=2.5, hourly_mult=3.2, chand_mult=3.2,
        adx_on=20, adx_off=18,
        tick_size=0.01, multiplier=100.0,
        exchange="NYMEX", trading_class="MCL",
        limit_pct=0.0015, max_entry_slip_pct=0.0025,
    ),
    "MGC": SymbolConfig(
        symbol="MGC",
        daily_mult=2.0, hourly_mult=2.6, chand_mult=3.0,
        adx_on=20, adx_off=18,
        tick_size=0.10, multiplier=10.0,
        exchange="COMEX", trading_class="MGC",
    ),
    "MBT": SymbolConfig(
        symbol="MBT",
        daily_mult=2.5, hourly_mult=3.0, chand_mult=3.2,
        adx_on=18, adx_off=16,
        tick_size=5.0, multiplier=0.1,
        exchange="CME", trading_class="MBT",
        base_risk_pct=0.0075,
        limit_pct=0.0015, max_entry_slip_pct=0.0025,
    ),
}

_FULL_CONFIGS: dict[str, SymbolConfig] = {
    "NQ": SymbolConfig(
        symbol="NQ",
        adx_on=18, adx_off=16,
        tick_size=0.25, multiplier=20.0,
        exchange="CME", trading_class="NQ",
    ),
    "CL": SymbolConfig(
        symbol="CL",
        daily_mult=2.5, hourly_mult=3.2, chand_mult=3.2,
        adx_on=20, adx_off=18,
        tick_size=0.01, multiplier=1000.0,
        exchange="NYMEX", trading_class="CL",
        limit_pct=0.0015, max_entry_slip_pct=0.0025,
    ),
    "GC": SymbolConfig(
        symbol="GC",
        daily_mult=2.0, hourly_mult=2.6, chand_mult=3.0,
        adx_on=20, adx_off=18,
        tick_size=0.10, multiplier=100.0,
        exchange="COMEX", trading_class="GC",
    ),
    "BRR": SymbolConfig(
        symbol="BRR",
        daily_mult=2.5, hourly_mult=3.0, chand_mult=3.2,
        adx_on=18, adx_off=16,
        tick_size=5.0, multiplier=5.0,
        exchange="CME", trading_class="BRR",
        base_risk_pct=0.0075,
        limit_pct=0.0015, max_entry_slip_pct=0.0025,
    ),
}

_ETF_CONFIGS: dict[str, SymbolConfig] = {
    "QQQ": SymbolConfig(
        symbol="QQQ",
        daily_mult=2.1, hourly_mult=2.7, chand_mult=2.2,
        adx_on=14, adx_off=16,
        ema_pull_normal=40, donchian_period=20,
        tick_size=0.01, multiplier=1.0,
        exchange="SMART", sec_type="STK", primary_exchange="NASDAQ",
        limit_pct=0.0015,
        base_risk_pct=0.016,
        shorts_enabled=False,
        size_reduction_months=((12, 0.5),),
    ),
    "GLD": SymbolConfig(
        symbol="GLD",
        daily_mult=2.0, hourly_mult=2.6, chand_mult=3.0,
        adx_on=14, adx_off=18,
        tick_size=0.01, multiplier=1.0,
        exchange="SMART", sec_type="STK", primary_exchange="ARCA",
        base_risk_pct=0.016,
        shorts_enabled=False,
    ),
    "USO": SymbolConfig(
        symbol="USO",
        daily_mult=2.3, hourly_mult=3.0, chand_mult=3.0,
        adx_on=18, adx_off=16,
        ema_pull_normal=45, donchian_period=20,
        tick_size=0.01, multiplier=1.0,
        exchange="SMART", sec_type="STK", primary_exchange="ARCA",
        limit_pct=0.0015,
        base_risk_pct=0.01,
        shorts_enabled=False,
    ),
}

_ALL_CONFIGS: dict[str, SymbolConfig] = {**_MICRO_CONFIGS, **_FULL_CONFIGS, **_ETF_CONFIGS}
ALL_SYMBOL_CONFIGS: dict[str, SymbolConfig] = _ALL_CONFIGS

_SETS: dict[str, list[str]] = {
    "micro": list(_MICRO_CONFIGS),
    "full": list(_FULL_CONFIGS),
    "etf": ["QQQ", "GLD"],
    "all": list(_ALL_CONFIGS),
}


def _resolve_symbols() -> tuple[list[str], dict[str, SymbolConfig]]:
    raw = os.environ.get("ATRSS_SYMBOL_SET", "etf")
    if raw in _SETS:
        syms = _SETS[raw]
    else:
        syms = [s.strip() for s in raw.split(",") if s.strip()]
    cfgs = {s: _ALL_CONFIGS[s] for s in syms if s in _ALL_CONFIGS}
    return list(cfgs), cfgs


SYMBOLS, SYMBOL_CONFIGS = _resolve_symbols()

# ---------------------------------------------------------------------------
# Portfolio-level risk limits
# ---------------------------------------------------------------------------
MAX_PORTFOLIO_HEAT: float = 0.06       # 6% of equity

# ---------------------------------------------------------------------------
# Fixed-quantity risk overlays (optimizer opt-in; default preserves behavior)
# ---------------------------------------------------------------------------
FIXED_QTY_REGIME_SCALING_ENABLED: bool = False
FIXED_QTY_STRONG_TREND_MULT: float = 1.25
FIXED_QTY_WEAK_TREND_MULT: float = 0.75

# Dynamic risk sizing overlays (optimizer opt-in; defaults match prior behavior)
DYNAMIC_RISK_STRONG_TREND_MULT: float = 1.15
DYNAMIC_RISK_WEAK_TREND_MULT: float = 0.8

# ---------------------------------------------------------------------------
# Regime / entry thresholds
# ---------------------------------------------------------------------------
ADX_STRONG: int = 30
ADX_STRONG_SLOPE_FLOOR: float = -2.0
CONFIRM_DAYS_NORMAL: int = 1              # hold_count needed when score < FAST_CONFIRM
PULLBACK_LOOKBACK: int = 8                # bars to look back for EMA_pull touch
PULLBACK_TOUCH_TOLERANCE_ATR: float = 0.55  # fraction of ATR_hourly for near-touch
PULLBACK_TOUCH_TOLERANCE_PCT: float = 0.005  # min tolerance as fraction of price (50bps)
RECOVERY_TOLERANCE_ATR: float = 0.40       # EMA pull recovery tolerance (fraction of ATRh)
RECOVERY_TOLERANCE_ATR_TREND: float = 0.55
RECOVERY_TOLERANCE_ATR_STRONG: float = 0.85    # regime-adaptive: STRONG_TREND
MOMENTUM_TOLERANCE_ATR: float = 0.10      # Momentum filter tolerance (fraction of ATRh)
PULLBACK_MOMENTUM_FILTER_ENABLED: bool = False  # Optional stricter pullback discrimination
SCORE_REVERSE_MIN: int = 60
FAST_CONFIRM_SCORE: int = 55     # 1-day confirm when score >= 55
FAST_CONFIRM_ADX: int = 22       # ADX floor for fast 1-day confirm (spec S2.6)
QUALITY_GATE_THRESHOLD: float = 4.0  # Minimum entry quality score (0-7)

# Path C confirmation constants (spec Section 2.6)
DI_MIN: int = 10
SEP_MIN: float = 0.20
ADX_MIN_STRUCT: int = 20

# ---------------------------------------------------------------------------
# Order management
# ---------------------------------------------------------------------------
ORDER_EXPIRY_HOURS: int = 18
VOUCHER_VALID_HOURS: int = 24         # Voucher window (spec Section 4)

# ---------------------------------------------------------------------------
# Bad-fill slippage guard (spec Section 6)
# ---------------------------------------------------------------------------
MAX_ENTRY_SLIP_ATR: float = 0.25      # 0.25 * ATRh

# ---------------------------------------------------------------------------
# Cooldown hours by regime
# ---------------------------------------------------------------------------
COOLDOWN_HOURS: dict[str, int] = {
    "RANGE": 4,
    "TREND": 2,
    "STRONG_TREND": 1,
}

# ---------------------------------------------------------------------------
# Pyramiding
# ---------------------------------------------------------------------------
ADDON_A_R: float = 1.5      # MFE threshold for add-on A (raised per rescaled optimizer)
ADDON_B_R: float = 2.0      # MFE threshold for add-on B
ADDON_A_SIZE_MULT: float = 0.5
ADDON_B_SIZE_MULT: float = 0.5  # Add-on B qty cap = 0.5 * base qty
ADDON_B_ENABLED: bool = False
FIXED_QTY_ADDON_B_ENABLED: bool = False

# ---------------------------------------------------------------------------
# BE offset
# ---------------------------------------------------------------------------
BE_ATR_OFFSET: float = 0.1  # BE + 0.1 * daily_ATR20
BE_TRIGGER_R: float = 0.75  # Move to BE at +0.75R MFE (tightened per rescaled optimizer)

# ---------------------------------------------------------------------------
# Chandelier trailing activation
# ---------------------------------------------------------------------------
CHANDELIER_TRIGGER_R: float = 1.25  # Activate chandelier at +1.25R MFE

# ---------------------------------------------------------------------------
# Partial profit-taking
# ---------------------------------------------------------------------------
TP1_R: float = 1.0       # Take partial profit at +1.0R
TP1_FRAC: float = 0.33   # Close 33% of position at TP1
TP2_R: float = 1.5       # Take partial profit at +1.5R
TP2_FRAC: float = 0.33   # Close 33% of remaining at TP2

# ---------------------------------------------------------------------------
# Time decay
# ---------------------------------------------------------------------------
MAX_HOLD_HOURS: int = 88
STALL_EXIT_ENABLED: bool = True   # enable full stall flatten (rescaled optimizer: flatten at 36h if MFE<0.4R)
STALL_CHECK_HOURS: int = 36     # Check for stall after this many bars
STALL_MFE_THRESHOLD: float = 0.4  # MFE_R below this = stall
EARLY_STALL_ENABLED: bool = False
EARLY_STALL_CHECK_HOURS: int = 12
EARLY_STALL_MFE_THRESHOLD: float = 0.2
EARLY_STALL_PARTIAL_FRAC: float = 0.5   # Exit 50% of base position

# ---------------------------------------------------------------------------
# Profit floor (spec Section 10.4): MFE_R → min_stop_R
# ---------------------------------------------------------------------------
PROFIT_FLOOR: dict[float, float] = {1.0: 0.2, 1.5: 0.5, 2.0: 1.0, 3.0: 1.8, 4.0: 2.8}
PROFIT_FLOOR_SHORT: dict[float, float] = {0.75: 0.10, 1.0: 0.50, 1.5: 1.0, 2.0: 1.25}

# ---------------------------------------------------------------------------
# Breakout arm window (spec Section 7.2)
# ---------------------------------------------------------------------------
ARM_WINDOW_HOURS: int = 24
BREAKOUT_RETRACE_ENTRY_FRAC: float = 0.30
BREAKOUT_RETRACE_LIMIT_FRAC: float = 0.50
BREAKOUT_REQUIRE_DIRECTIONAL_CANDLE: bool = True
BREAKOUT_DIRECT_ENTRY: bool = False

# ---------------------------------------------------------------------------
# Stop tightening for non-STRONG_TREND regimes (spec Section 5.6)
# ---------------------------------------------------------------------------
TREND_STOP_TIGHTENING: float = 0.60   # Scale ATR stop mults by this in TREND regime

# ---------------------------------------------------------------------------
# Portfolio candidate ranking
# ---------------------------------------------------------------------------
CANDIDATE_RANK_MODE: str = "score"  # score, stop_first, score_per_risk, gld_first, qqq_first

# ---------------------------------------------------------------------------
# Builder helpers — create Instrument / template / route dicts
# ---------------------------------------------------------------------------

def build_instruments() -> dict[str, Instrument]:
    """Create Instrument objects for every symbol and register them."""
    instruments: dict[str, Instrument] = {}
    for sym, cfg in SYMBOL_CONFIGS.items():
        inst = Instrument(
            symbol=sym,
            root=sym,
            venue=cfg.exchange,
            tick_size=cfg.tick_size,
            tick_value=cfg.tick_size * cfg.multiplier,
            multiplier=cfg.multiplier,
            contract_expiry=cfg.contract_expiry,
        )
        InstrumentRegistry.register(inst)
        instruments[sym] = inst
    return instruments
