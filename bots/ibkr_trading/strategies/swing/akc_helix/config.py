"""AKC-Helix Swing v2.0 — constants and per-symbol configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from libs.broker_ibkr.config.schemas import ContractTemplate, ExchangeRoute
from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry

# ---------------------------------------------------------------------------
# Strategy identity
# ---------------------------------------------------------------------------
STRATEGY_ID = "AKC_HELIX"

# ---------------------------------------------------------------------------
# Per-symbol configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolConfig:
    symbol: str
    # Instrument classification
    is_etf: bool = False
    sec_type: str = "FUT"          # "FUT" or "STK"
    tick_size: float = 0.25
    multiplier: float = 2.0
    exchange: str = "CME"
    trading_class: str = ""
    contract_expiry: str = ""
    futures_pair: str = ""         # ETF→futures link for s17 basket rule
    # Entry windows (ET)
    entry_window_start_et: str = "03:00"
    entry_window_end_et: str = "16:00"
    # Spread gate (spec s7.1)
    max_spread_ticks: int = 2          # futures: ticks-based
    max_spread_dollars: float = 0.0    # ETF: dollar spread cap (§7.1)
    max_spread_bps: float = 0.0        # ETF: bps spread cap (§7.1)
    # Min stop floor
    min_stop_floor_dollars: float = 100.0
    # Slippage guard (spec s11.2, ETF-only)
    slip_max_dollars: float = 0.0      # max acceptable slippage in $
    slip_max_bps: float = 0.0          # max acceptable slippage in bps
    # Corridor cap ATRd multipliers
    corridor_cap_chop: float = 1.3
    corridor_cap_trend: float = 1.6
    corridor_cap_other: float = 1.4
    # Chandelier
    chandelier_lookback: int = 30
    # Position limits
    max_contracts: int = 10
    # Offset tiers
    offset_tight_ticks: int = 2
    offset_wide_ticks: int = 4
    # Rescue
    teleport_offset_mult: float = 2.0
    # Risk
    base_risk_pct: float = 0.009
    # Experiment A/B tracking
    experiment_id: str = ""
    experiment_variant: str = ""


# ---------------------------------------------------------------------------
# Futures configs
# ---------------------------------------------------------------------------

_FUTURES_CONFIGS: dict[str, SymbolConfig] = {
    "MNQ": SymbolConfig(
        symbol="MNQ",
        tick_size=0.25, multiplier=2.0,
        exchange="CME", trading_class="MNQ",
        max_spread_ticks=2,
        min_stop_floor_dollars=100.0,
        teleport_offset_mult=2.0,
        offset_tight_ticks=6, offset_wide_ticks=10,
    ),
    "MCL": SymbolConfig(
        symbol="MCL",
        tick_size=0.01, multiplier=100.0,
        exchange="NYMEX", trading_class="MCL",
        max_spread_ticks=3,
        min_stop_floor_dollars=80.0,
        chandelier_lookback=24,
        teleport_offset_mult=3.0,
        offset_tight_ticks=6, offset_wide_ticks=10,
    ),
    "MGC": SymbolConfig(
        symbol="MGC",
        tick_size=0.10, multiplier=10.0,
        exchange="COMEX", trading_class="MGC",
        max_spread_ticks=2,
        min_stop_floor_dollars=60.0,
        teleport_offset_mult=2.0,
        offset_tight_ticks=6, offset_wide_ticks=10,
    ),
    "MBT": SymbolConfig(
        symbol="MBT",
        tick_size=5.0, multiplier=0.1,
        exchange="CME", trading_class="MBT",
        max_spread_ticks=4,
        min_stop_floor_dollars=150.0,
        chandelier_lookback=20,
        teleport_offset_mult=3.0,
        offset_tight_ticks=4, offset_wide_ticks=8,
    ),
    "NQ": SymbolConfig(
        symbol="NQ",
        tick_size=0.25, multiplier=20.0,
        exchange="CME", trading_class="NQ",
        max_spread_ticks=2,
        min_stop_floor_dollars=100.0,
        teleport_offset_mult=2.0,
        offset_tight_ticks=6, offset_wide_ticks=10,
    ),
    "CL": SymbolConfig(
        symbol="CL",
        tick_size=0.01, multiplier=1000.0,
        exchange="NYMEX", trading_class="CL",
        max_spread_ticks=3,
        min_stop_floor_dollars=80.0,
        chandelier_lookback=24,
        teleport_offset_mult=3.0,
        offset_tight_ticks=6, offset_wide_ticks=10,
    ),
    "GC": SymbolConfig(
        symbol="GC",
        tick_size=0.10, multiplier=100.0,
        exchange="COMEX", trading_class="GC",
        max_spread_ticks=2,
        min_stop_floor_dollars=60.0,
        teleport_offset_mult=2.0,
        offset_tight_ticks=6, offset_wide_ticks=10,
    ),
    "BT": SymbolConfig(
        symbol="BT",
        tick_size=5.0, multiplier=5.0,
        exchange="CME", trading_class="BT",
        max_spread_ticks=4,
        min_stop_floor_dollars=150.0,
        chandelier_lookback=20,
        teleport_offset_mult=3.0,
        offset_tight_ticks=4, offset_wide_ticks=8,
    ),
}

# ---------------------------------------------------------------------------
# ETF configs
# ---------------------------------------------------------------------------

_ETF_CONFIGS: dict[str, SymbolConfig] = {
    "QQQ": SymbolConfig(
        symbol="QQQ",
        is_etf=True, sec_type="STK",
        tick_size=0.01, multiplier=1.0,
        exchange="SMART", trading_class="",
        futures_pair="NQ",
        entry_window_start_et="09:35",
        entry_window_end_et="15:45",
        max_spread_dollars=0.02,
        max_spread_bps=2,
        min_stop_floor_dollars=0.10,
        max_contracts=500,
        slip_max_dollars=0.05,
        slip_max_bps=5,
    ),
    "USO": SymbolConfig(
        symbol="USO",
        is_etf=True, sec_type="STK",
        tick_size=0.01, multiplier=1.0,
        exchange="SMART", trading_class="",
        futures_pair="CL",
        entry_window_start_et="09:35",
        entry_window_end_et="15:45",
        max_spread_dollars=0.05,
        max_spread_bps=5,
        min_stop_floor_dollars=0.10,
        max_contracts=1000,
        chandelier_lookback=24,
        slip_max_dollars=0.08,
        slip_max_bps=8,
    ),
    "GLD": SymbolConfig(
        symbol="GLD",
        is_etf=True, sec_type="STK",
        tick_size=0.01, multiplier=1.0,
        exchange="SMART", trading_class="",
        futures_pair="GC",
        entry_window_start_et="09:35",
        entry_window_end_et="15:45",
        max_spread_dollars=0.02,
        max_spread_bps=2,
        min_stop_floor_dollars=0.10,
        max_contracts=500,
        slip_max_dollars=0.05,
        slip_max_bps=5,
    ),
}

# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------

_ALL_CONFIGS: dict[str, SymbolConfig] = {**_FUTURES_CONFIGS, **_ETF_CONFIGS}

_SETS: dict[str, list[str]] = {
    "micro_futures": ["MNQ", "MCL", "MGC", "MBT"],
    "full_futures": ["NQ", "CL", "GC", "BT"],
    "etf": ["QQQ", "GLD"],
    "all": list(_ALL_CONFIGS),
}


def _resolve_symbols() -> tuple[list[str], dict[str, SymbolConfig]]:
    raw = os.environ.get("AKCHELIX_SYMBOL_SET", "etf")
    if raw in _SETS:
        syms = _SETS[raw]
    else:
        syms = [s.strip() for s in raw.split(",") if s.strip()]
    cfgs = {s: _ALL_CONFIGS[s] for s in syms if s in _ALL_CONFIGS}
    return list(cfgs), cfgs


SYMBOLS, SYMBOL_CONFIGS = _resolve_symbols()

# ---------------------------------------------------------------------------
# MACD parameters (spec s3)
# ---------------------------------------------------------------------------
MACD_FAST: int = 8
MACD_SLOW: int = 21
MACD_SIGNAL: int = 5

# ---------------------------------------------------------------------------
# Daily indicators (spec s5)
# ---------------------------------------------------------------------------
DAILY_EMA_FAST: int = 20
DAILY_EMA_SLOW: int = 50
ATR_DAILY_PERIOD: int = 14
VOLFACTOR_BASE_PERIOD: int = 60

# ---------------------------------------------------------------------------
# VolFactor (spec s6)
# ---------------------------------------------------------------------------
VOLFACTOR_MIN: float = 0.4
VOLFACTOR_MAX: float = 1.5
EXTREME_VOL_PCT: float = 999.0
LOW_VOL_PCT: float = 20.0
HIGH_VOL_PCT: float = 68.0

# ---------------------------------------------------------------------------
# Heat caps (spec s8.3)
# ---------------------------------------------------------------------------
PORTFOLIO_CAP_R: float = 1.50
INSTRUMENT_CAP_R: float = 0.85
EXTREME_VOL_CAP_R: float = 1.25

# ---------------------------------------------------------------------------
# Circuit breakers (spec s8.4)
# ---------------------------------------------------------------------------
WEEKLY_STOP_R: float = -5.0
DAILY_STOP_R: float = -2.5
CONSEC_STOPS_HALVE: int = 3

# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------
ADX_PERIOD: int = 14
ADX_UPPER_GATE: float = 60.5

# ---------------------------------------------------------------------------
# 4H regime EMAs (kept for regime computation)
# ---------------------------------------------------------------------------
EMA_4H_FAST: int = 20
EMA_4H_SLOW: int = 50

# ---------------------------------------------------------------------------
# MACD momentum lookback for Class D (spec s10.5: macd[t] > macd[t-3])
# ---------------------------------------------------------------------------
CLASS_D_MOM_LOOKBACK: int = 3

# ---------------------------------------------------------------------------
# Class B quality-filter parameters
# ---------------------------------------------------------------------------
CLASS_B_MOM_LOOKBACK: int = 5
CLASS_B_MIN_ADX: float = 28.0
CLASS_B_MIN_PIVOT_SEP_BARS: int = 8   # reject micro-divergence (pivots < 8 bars apart)
CLASS_B_BAIL_BARS: int = 9            # R1: bail trigger: exit if R < threshold after N bars
CLASS_B_BAIL_R_THRESH: float = -0.35  # bail trigger: minimum R to avoid early exit

# ---------------------------------------------------------------------------
# Class C min hold (spec s10.4)
# ---------------------------------------------------------------------------
CLASS_C_MIN_HOLD_BARS: int = 12       # min bars before stale can trigger on Class C

# ---------------------------------------------------------------------------
# Class D bail (disabled by default; 0 = off)
# ---------------------------------------------------------------------------
CLASS_D_BAIL_BARS: int = 0            # set >0 to enable early bail for momentum trades
CLASS_D_BAIL_R_THRESH: float = -0.5   # bail if R < threshold after N bars

# Class D quality gates (disabled by default)
CLASS_D_MIN_ADX: float = 0.0           # all Class D entries require daily ADX >= threshold
CLASS_D_SHORT_MIN_ADX: float = 0.0     # short-side overlay for weak bear/short alpha
CLASS_D_HIST_SIGN_GATE: bool = False   # require 1H histogram sign to agree with direction
CLASS_D_REGIME_STREAK_MIN: int = 2     # require N consecutive daily regime bars before D entry

# Class D pre-entry discriminator.  Defaults are disabled to preserve the
# baseline unless an optimizer candidate explicitly opts in.
CLASS_D_MIN_PIVOT_SEP_BARS: int = 4       # reject compressed higher-low/lower-high patterns
CLASS_D_MAX_PIVOT2_AGE_BARS: int = 0      # reject stale P2 pivots after N 1H bars; 0 disables
CLASS_D_MIN_PULLBACK_ATR: float = 0.0     # require pullback depth from BOS to P2, normalized by ATR1H
CLASS_D_MAX_PULLBACK_ATR: float = 0.0     # reject oversized pullbacks; 0 disables
CLASS_D_MAX_ENTRY_STOP_ATR: float = 0.0   # reject wide entry-to-stop structures; 0 disables
CLASS_D_MAX_ARM_OVEREXT_ATR: float = 999.0  # reject late/overextended arms past BOS
CLASS_D_MIN_MACD_DELTA_ATR: float = 0.0   # require current MACD impulse vs P2, normalized by ATR1H
CLASS_D_HIST_SLOPE_LOOKBACK: int = 0      # require histogram improving over N bars; 0 disables
CLASS_D_MIN_HIST_DELTA_ATR: float = 0.0   # minimum normalized histogram improvement
CLASS_D_MAX_DAILY_EXTENSION_ATR: float = 3.5  # reject daily EMA overextension; 0 disables
CLASS_D_FRESH_BREAK_ATR: float = 0.0      # require fresh continuation beyond close before entry; 0 disables

# ---------------------------------------------------------------------------
# Size multipliers — Class A (spec s10.1) and Class D (spec s10.5)
# ---------------------------------------------------------------------------
CLASS_A_SIZE_TREND: float = 1.00   # 4H hidden div continuation, trend-aligned
CLASS_A_SIZE_CHOP: float = 0.65    # 4H hidden div continuation, chop
CLASS_A_SIZE_COUNTER: float = 0.50 # 4H hidden div continuation, countertrend
CLASS_B_SIZE_TREND: float = 1.00   # 1H hidden div continuation, trend-aligned (spec s10.1)
CLASS_B_SIZE_CHOP: float = 0.65    # 1H hidden div continuation, chop
CLASS_B_SIZE_COUNTER: float = 0.50 # 1H hidden div continuation, countertrend
CLASS_C_SIZE_CHOP: float = 1.00    # 4H classic div reversal, chop (spec s10.1)
CLASS_C_SIZE_COUNTER: float = 0.85 # 4H classic div reversal, countertrend (reversing into trend)
CLASS_C_SIZE_TREND: float = 0.40   # 4H classic div reversal, trend-aligned (fading trend)
CLASS_D_SIZE_TREND: float = 0.80   # 1H no-div momentum, trend-only

# R1: Class A (4H hidden div) disabled -- zero value in 8-phase optimization
DISABLE_CLASS_A: bool = True
DISABLE_CLASS_C: bool = True
DISABLE_CIRCUIT_BREAKER: bool = True

# ---------------------------------------------------------------------------
# Divergence magnitude filter (spec s9)
# ---------------------------------------------------------------------------
DIV_MAG_MIN_HISTORY: int = 20
DIV_MAG_DEFAULT_THRESHOLD: float = 0.05
DIV_MAG_FLOOR: float = 0.04
DIV_MAG_PERCENTILE: int = 25

# ---------------------------------------------------------------------------
# Stop ATR multipliers (spec s10)
# ---------------------------------------------------------------------------
STOP_4H_MULT: float = 0.75      # spec s10.2: L2 - 0.75*ATR4H
STOP_1H_STD: float = 0.50      # spec s10.3/10.5: L2 - 0.50*ATR1H
STOP_1H_HIGHVOL: float = 0.75  # spec s10.3: high-vol L2 - 0.75*ATR1H
EMERGENCY_STOP_R: float = -2.0  # catastrophic stop at -2R
# ---------------------------------------------------------------------------
# Execution (spec s11)
# ---------------------------------------------------------------------------
CATCHUP_OVERSHOOT_FRAC: float = 0.15
CATCHUP_OVERSHOOT_OPEN_FRAC: float = 0.20  # spec s11.3: at 09:35 re-arming
CATCHUP_TTL_MIN: int = 5
RESCUE_TTL_MIN: int = 2
RESCUE_SLIP_FRAC: float = 0.5

# ---------------------------------------------------------------------------
# TTL (spec s12)
# ---------------------------------------------------------------------------
TTL_1H_HOURS: int = 6
TTL_4H_HOURS: int = 12
TTL_ADD_HOURS: int = 6

# ---------------------------------------------------------------------------
# R milestones (v2.0 — wider stops need more breathing room)
# ---------------------------------------------------------------------------
R_BE: float = 1.0               # spec s13.2: +1R transition (4H origin)
R_BE_1H: float = 0.756
R_PARTIAL_2P5: float = 1.8
R_PARTIAL_5: float = 6.6
BE_ATR1H_OFFSET: float = 0.08
PARTIAL_2P5_FRAC: float = 0.968
PARTIAL_5_FRAC: float = 0.25    # spec s13.4: exit 25% at +5R
PARTIAL_5_TRAIL_BONUS: float = 0.5

# ---------------------------------------------------------------------------
# Stale exit (spec s13)
# ---------------------------------------------------------------------------
EARLY_STALE_BARS: int = 24
STALE_1H_BARS: int = 24           # reduced stale timeout for 1H-origin
STALE_4H_BARS: int = 15           # spec s13.5: 4H-origin stale after 15 4H bars
STALE_R_THRESH: float = 0.3       # spec s13.5: stale if R_state < +0.5
STALE_TIGHTEN_ATR_MULT: float = 0.25
STALE_FLATTEN_R_FLOOR: float = -0.25
STALE_PROFIT_MULT_OVERRIDE: float = 1.5
STALE_LOSS_CAP_FRAC: float = 0.3

# ---------------------------------------------------------------------------
# Right-then-stopped leakage guard (disabled by default)
# ---------------------------------------------------------------------------
RTS_GUARD_MFE_R: float = 0.25            # min peak MFE before guard can arm; <=0 disables
RTS_GUARD_MIN_GIVEBACK_R: float = 0.05   # required giveback from peak MFE
RTS_GUARD_FLOOR_R: float = 0.05          # protective stop level in R after decay is detected
RTS_GUARD_MIN_BARS: int = 10             # ignore first bars to avoid same-bar noise
RTS_GUARD_FADE_BARS: int = 1             # consecutive adverse MACD histogram bars required
RTS_GUARD_MAX_MFE_R: float = 1.75        # keep guard focused on small/mid MFE leaks
RTS_FAIL_FLATTEN_R: float = -999.0       # optional late decay flatten; <=-900 disables

# ---------------------------------------------------------------------------
# Trailing (spec s14)
# ---------------------------------------------------------------------------
TRAIL_MIN: float = 2.0
TRAIL_MAX: float = 4.0
TRAIL_BASE: float = 4.0            # spec s14.1: mult_base = max(2.0, 4.0 - R/5)
TRAIL_R_DIV: float = 8.4
TRAIL_MOM_BONUS: float = 0.5
TRAIL_CHOP_PENALTY: float = 0.25
TRAIL_FLIP_PENALTY: float = 0.50
TRAIL_PROFIT_DELAY_BARS: int = 6

# ---------------------------------------------------------------------------
# Inline trailing tightening (configurable versions of hardcoded engine values)
# ---------------------------------------------------------------------------
# Momentum fade layer
TRAIL_FADE_ONSET_BARS: int = 2        # negative MACD hist bars before fade triggers
TRAIL_FADE_PENALTY: float = 1.32      # ATR mult penalty on momentum fade
TRAIL_FADE_FLOOR: float = 1.5         # floor after momentum fade
TRAIL_FADE_MIN_R: float = 0.5         # min R before fade can trigger
# Time-decay layer
TRAIL_TIMEDECAY_ONSET: int = 25       # bars at R>=1 before time decay starts
TRAIL_TIMEDECAY_RATE: float = 0.05    # decay rate per bar beyond onset
TRAIL_TIMEDECAY_FLOOR: float = 2.5    # floor for time-decay trail mult
# Stall layer
TRAIL_STALL_ONSET: int = 5
TRAIL_STALL_RATE: float = 0.1584      # stall decay rate per bar
TRAIL_STALL_FLOOR: float = 1.5        # floor for stall-decay trail mult

# ---------------------------------------------------------------------------
# R-band trailing profiles (all 0 = disabled, uses global values)
# ---------------------------------------------------------------------------
R_BAND_MID: float = 0.0           # R threshold for mid band (e.g., 2.0)
R_BAND_HIGH: float = 0.0          # R threshold for high band (e.g., 5.0)
TRAIL_BASE_LOW_R: float = 0.0     # base mult for R < R_BAND_MID
TRAIL_R_DIV_LOW_R: float = 0.0    # R divisor for low band
TRAIL_BASE_MID_R: float = 0.0     # base mult for R_BAND_MID <= R < R_BAND_HIGH
TRAIL_R_DIV_MID_R: float = 0.0    # R divisor for mid band
TRAIL_BASE_HIGH_R: float = 0.0    # base mult for R >= R_BAND_HIGH
TRAIL_R_DIV_HIGH_R: float = 0.0   # R divisor for high band

# ---------------------------------------------------------------------------
# Class-specific trailing (all 0 = disabled, uses global values)
# ---------------------------------------------------------------------------
TRAIL_BASE_CLASS_D: float = 0.0
TRAIL_R_DIV_CLASS_D: float = 0.0
TRAIL_STALL_ONSET_CLASS_D: int = 0
TRAIL_FADE_PENALTY_CLASS_D: float = 0.0
TRAIL_FADE_MIN_R_CLASS_D: float = 0.0
TRAIL_BASE_CLASS_B: float = 0.0
TRAIL_R_DIV_CLASS_B: float = 0.0
TRAIL_STALL_ONSET_CLASS_B: int = 0

# ---------------------------------------------------------------------------
# Adds (spec s15)
# ---------------------------------------------------------------------------
ADD_4H_R: float = 0.25
ADD_1H_R: float = 0.7776          # R1: 1H-origin add after +0.9R
ADD_RISK_FRAC: float = 2.0592
ADD_MIN_BARS: int = 2
ADD_MAX_BARS: int = 18             # extended window (avg hold ~30 bars)
ADD_OVERNIGHT_R: float = 2.0
ADD_PRICE_GATE_ATR_MULT: float = 0.5  # price-based add: price > BoS + 0.5×ATR

# ---------------------------------------------------------------------------
# News windows (spec s4)
# ---------------------------------------------------------------------------
NEWS_WINDOWS: dict[str, tuple[int, int]] = {
    "CPI": (-60, 30),
    "NFP": (-60, 30),
    "FOMC": (-60, 60),
    "FED_SPEECH": (-30, 30),
    "CL_INVENTORY": (-20, 20),
    "ECB": (-60, 60),
    "BOJ": (-60, 60),
    "CRYPTO_EVENT": (-30, 30),
}

# ---------------------------------------------------------------------------
# Basket rule (spec s8.3)
# ---------------------------------------------------------------------------
BASKET_SYMBOLS: set[str] = {"NQ", "MNQ", "QQQ", "BT", "MBT"}
BASKET_4H_SECOND_MULT: float = 0.60


# ---------------------------------------------------------------------------
# Builder helpers
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


def build_contract_templates() -> dict[str, ContractTemplate]:
    """Build ContractTemplate dict matching config/contracts.yaml."""
    templates: dict[str, ContractTemplate] = {}
    for sym, cfg in SYMBOL_CONFIGS.items():
        templates[sym] = ContractTemplate(
            symbol=sym,
            sec_type=cfg.sec_type,
            exchange=cfg.exchange,
            currency="USD",
            multiplier=cfg.multiplier,
            tick_size=cfg.tick_size,
            tick_value=cfg.tick_size * cfg.multiplier,
            trading_class=cfg.trading_class or None,
            primary_exchange=cfg.exchange if cfg.is_etf else None,
        )
    return templates


def build_exchange_routes() -> dict[str, ExchangeRoute]:
    """Build ExchangeRoute dict matching config/routing.yaml."""
    routes: dict[str, ExchangeRoute] = {}
    for sym, cfg in SYMBOL_CONFIGS.items():
        routes[sym] = ExchangeRoute(
            root_symbol=sym,
            exchange=cfg.exchange,
            trading_class=cfg.trading_class or None,
            primary_exchange=cfg.exchange if cfg.is_etf else None,
        )
    return routes
