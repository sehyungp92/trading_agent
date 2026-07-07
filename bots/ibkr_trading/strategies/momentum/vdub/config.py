"""Vdubus NQ Dominant-Trend Swing Protocol v4.0 — configuration."""
from __future__ import annotations

import os

from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry

STRATEGY_ID = "VdubusNQ_v4"
DEFAULT_SYMBOL = os.environ.get("TRADING_SYMBOL", "NQ")

# ── Contract spec ────────────────────────────────────────────────────
NQ_SPECS = {
    "NQ":  {"tick": 0.25, "tick_value": 5.00, "point_value": 20.00},
    "MNQ": {"tick": 0.25, "tick_value": 0.50, "point_value":  2.00},
}
NQ_SPEC = NQ_SPECS[DEFAULT_SYMBOL]
ES_SPEC = {"tick": 0.25, "tick_value": 12.50, "point_value": 50.00}

# ── Sessions & Windows (ET) ─────────────────────────────────────────
RTH_ENTRY_START = (9, 40)
RTH_ENTRY_END = (15, 50)
EVENING_START = (19, 0)
EVENING_END = (21, 0)
HARD_BLOCKS = [((9, 30), (9, 40)), ((15, 50), (16, 0)), ((22, 0), (9, 40))]
MIDDAY_DEAD_START = (10, 45)      # midday dead zone start (ET) — block before 11:00 entries
MIDDAY_DEAD_END = (15, 0)         # midday dead zone end (ET) — reopen at 15:00
# Shoulder periods (not used — dead zone relaxation tested and reverted)
MIDDAY_SHOULDER_EARLY = ((10, 45), (11, 45))
MIDDAY_SHOULDER_LATE = ((14, 0), (15, 0))
SHOULDER_REQUIRE_HOURLY_ALIGNMENT = True
SHOULDER_REQUIRE_LOW_CHOP = True

# ── Late Shoulder ──────────────────────────────────────────────────
USE_LATE_SHOULDER = False
LATE_SHOULDER_RANGE = ((14, 0), (15, 0))
LATE_SHOULDER_REQUIRE_1H_ALIGN = True
LATE_SHOULDER_REQUIRE_LOW_CHOP = True
LATE_SHOULDER_VWAP_CAP = 0.60
LATE_SHOULDER_CLASS_MULT = 0.50

# Sub-windows
OPEN_RANGE = ((9, 40), (10, 30))
CORE_RANGE = ((10, 30), (15, 30))
CLOSE_RANGE = ((15, 30), (15, 50))
EVENING_RANGE = ((19, 0), (21, 0))

# ── Risk ─────────────────────────────────────────────────────────────
BASE_RISK_PCT = 0.0065
RISK_PCT = BASE_RISK_PCT
VOL_FACTOR = {"Normal": 1.0, "High": 0.65, "Shock": 0.0}
HEAT_CAP_MULT = 3.50   # portfolio v4 shared heat cap
DAILY_BREAKER_MULT = -2.0
CLASS_MULT_PREDATOR = 1.00
CLASS_MULT_NOPRED = 0.70
CLASS_MULT_FLIP = 0.50
SESSION_MULT = {"RTH": 1.00, "EVENING": 0.50}  # OPT4: reduce EVENING from 0.70 to 0.50
PYRAMID_ADD_RISK_MULT = 0.50

# Stops (points)
MIN_STOP_POINTS = 20
MAX_STOP_POINTS = 120
STRUCTURE_STOP_ATR_MULT = 0.10
ATR_STOP_MULT = 1.4

# Entry caps
MAX_LONGS_PER_DAY = 2
MAX_SHORTS_PER_DAY = 2

# ── Type A ───────────────────────────────────────────────────────────
TOUCH_LOOKBACK_15M = 8
VWAP_CAP_CORE = 0.85
VWAP_CAP_OPEN_EVE = 0.935
VWAP_CAP_EVENING = 0.35

USE_TYPE_B = False                # disable Type B (breakout retest) entries
TYPE_B_ALLOWED_WINDOWS = ("OPEN", "CLOSE")  # restrict to high-conviction windows (if enabled)
TYPE_B_CLASS_MULT = 0.50          # half size until validated (if enabled)
TYPE_B_REQUIRE_1H_ALIGN = True    # 1H trend must match direction (if enabled)

# Type C continuation reclaim (v4.5 research, disabled by default). Converts
# the shadow-positive "no_signal" bucket into an explicit completed-bar signal.
USE_TYPE_C = False
TYPE_C_ALLOWED_WINDOWS = ("OPEN", "CORE", "CLOSE")
TYPE_C_CLASS_MULT = 0.50
TYPE_C_LOOKBACK_15M = 12
TYPE_C_BREAK_BUFFER_ATR = 0.00
TYPE_C_MIN_CLOSE_FRAC = 0.62
TYPE_C_MAX_BAR_ATR = 1.60
TYPE_C_MAX_VWAP_DIST_ATR = 1.80
TYPE_C_REQUIRE_VWAP_SIDE = True

# ── Type B ───────────────────────────────────────────────────────────
BREAK_LOOKBACK_1H = 20
BREAKOUT_LOOKBACK_15M = 20
RETEST_LOOKBACK_15M = 8
RETEST_TOL_ATR = 0.35
EXTENSION_SKIP_ATR = 1.2

# ── Momentum ─────────────────────────────────────────────────────────
MOM_N = 50              # default: 12.5h momentum lookback
FLOOR_PCT = 0.25
SLOPE_LB = 3

# ── Pivots ───────────────────────────────────────────────────────────
NCONFIRM_1H = 4
NCONFIRM_D = 2

# ── Execution ────────────────────────────────────────────────────────
BUFFER_TICKS = 2
TTL_BARS = 4
TELEPORT_TICKS = 12
OFFSET_TICKS_MIN = 1
OFFSET_TICKS_MAX = 12
OFFSET_TICKS_ATR_FRAC = 0.15

# Fallback
FALLBACK_WAIT_BARS = 1
FALLBACK_SPREAD_MAX_TICKS = 2
FALLBACK_SLIP_MAX_TICKS = 2
FALLBACK_ATR_TICKS_CAP = 120

# ── Viability ────────────────────────────────────────────────────────
COST_RISK_MAX = 0.12
SANITY_SPREAD_MAX = 8
SANITY_SLIP_MAX = 6
RT_COMM_FEES = 4.12
SLIP_TICKS_BY_WINDOW = {"OPEN": 2, "EVENING": 2, "CORE": 1, "CLOSE": 1}

# ── +1R Partial ──────────────────────────────────────────────────────
PLUS_1R_PARTIAL_ENABLED = False
PARTIAL_PCT = 0.33

# ── ACTIVE_FREE exit management ─────────────────────────────────────
FREE_STALE_BARS_15M = 24          # 6 hrs: flatten stale free-ride positions
FREE_STALE_R_THRESHOLD = 0.30     # unrealized R below this = stale
FREE_PROFIT_LOCK_R = 0.25         # lock at +0.25R once peak >= 0.50R

# ── Max position duration ────────────────────────────────────────────
MAX_POSITION_BARS_15M = 51.2
MAX_POSITION_BARS_FREE = 96       # 24 hrs: hard cap for ACTIVE_FREE


# ── Decision Gate (15:50) ────────────────────────────────────────────
HOLD_WEEKDAY_R = 1.0
HOLD_WEEKDAY_BORDER_R = 0.5
HOLD_FRIDAY_R = 1.5
WEEKEND_LOCK_R = 0.5

# ── Exits ────────────────────────────────────────────────────────────
VWAP_FAIL_CONSEC = 2
VWAP_FAIL_EVENING = True
STALE_BARS_15M = 8  # FINAL: reduce from 16 to 8 (2H instead of 4H wait time)
STALE_BARS_BY_WINDOW = {      # v4.2: adaptive stale timer per sub-window
    "OPEN": 8,                # keep current (validated)
    "CORE": 8,                # keep current (validated)
    "CLOSE": 12,              # 3H — give CLOSE entries more room (avgR=+0.578, EdgeR=2.78)
    "EVENING": 5,             # 1.25H — cut faster (evening stales are -$5,755 drag)
}
STALE_R = 0.30
STALE_MFE_EXEMPT_R = 0.50             # v4.3: min peak MFE to exempt from stale exit

# -- Late Trail (v4.4) -------------------------------------------------------
# Independent trail: no partial close, late activation, wide multiplier.
# Operates on full position in ACTIVE_RISK. Does NOT touch plus_1r_partial.
LATE_TRAIL_ACTIVATE_R = 1.5       # peak MFE threshold to activate trailing
LATE_TRAIL_BE_R = 1.0             # peak MFE threshold to move stop to BE (0 = never)
LATE_TRAIL_MULT = 4.0             # ATR multiplier for trail distance
LATE_TRAIL_MULT_MIN = 2.0         # minimum trail mult at high R
LATE_TRAIL_LOOKBACK = 20          # bars lookback for swing high/low
LATE_TRAIL_TIGHTEN_R = 2.5        # R above which trail starts tightening toward MULT_MIN
LATE_TRAIL_TIGHTEN_DIVISOR = 8.0  # divisor for progressive tightening
LATE_TRAIL_WINDOW_MULT = {"OPEN": 1.0, "CORE": 1.0, "CLOSE": 1.0, "EVENING": 1.0}

TRAIL_LOOKBACK_15M = 12
TRAIL_MULT_MIN = 1.5
TRAIL_MULT_BASE = 2.5
TRAIL_MULT_R_DIV = 6.0
TRAIL_MULT_POST_PARTIAL = 2.0    # ACTIVE_FREE also needs room
TRAIL_CORE_TRANSITION_REDUCTION = 0.80    # reduce trail mult by 20% in CORE for OPEN entries
TRAIL_WINDOW_MULT = {"OPEN": 1.0, "CORE": 0.60, "CLOSE": 1.0, "EVENING": 0.70}  # OPT5: tighten EVENING trail

# ── VWAP-A failure ───────────────────────────────────────────────────
VWAP_A_FAIL_MIN_SESSIONS = 1                  # allow first-session exits
VWAP_A_FAIL_FIRST_SESSION_MARGIN_ATR = 0.10   # v4.1: tightened from 0.25 (+0.008 Sharpe, +$698)

# ── Overnight ────────────────────────────────────────────────────────
OVERNIGHT_1H_LOOKBACK = 4
OVERNIGHT_ATR_MULT = 0.5

# ── Event Safety ─────────────────────────────────────────────────────
EVENT_PRE_MINUTES = 60
EVENT_POST_MINUTES = 15
COOLDOWN_BARS = {"CPI": 3, "NFP": 3, "FOMC": 6}
ATR_NORM_MULT = 1.3
BAR_CALM_RANGE_ATR = 1.8
MAX_POST_EVENT_MINUTES = 60

# ── Hourly Alignment Sizing ─────────────────────────────────────────
HOURLY_ALIGNED_MULT = 1.0
HOURLY_NEUTRAL_MULT = 0.60

# v4.5 conditional gate bypasses. Disabled by default; when enabled they only
# let shadow-positive blocked cohorts through when quality, session, and chop agree.
HOURLY_BYPASS_ALLOWED_WINDOWS = ("OPEN", "CLOSE")
HOURLY_BYPASS_EQS_MIN = 2
HOURLY_BYPASS_MAX_CHOP = 35.2
HOURLY_BYPASS_SIZE_MULT = 0.5
SLOPE_BYPASS_ALLOWED_WINDOWS = ("OPEN", "CORE", "CLOSE")
SLOPE_BYPASS_EQS_MIN = 4
SLOPE_BYPASS_MAX_CHOP = 36.0
SLOPE_BYPASS_MOM_ABS_MIN = 0.0
SLOPE_BYPASS_SIZE_MULT = 1.0

# ── Regime ───────────────────────────────────────────────────────────
DAILY_SMA_PERIOD = 200
TREND_PERSIST_BARS = 2
HOURLY_EMA_PERIOD = 50
HOURLY_PERSIST_BARS = 2

# ── Vol State ────────────────────────────────────────────────────────
VOL_ATR_PERIOD = 14
VOL_LOOKBACK = 252
SHOCK_PCTL = 90
SHOCK_MED_MULT = 1.2
HIGH_PCTL = 85

# ── Choppiness ──────────────────────────────────────────────────────
CHOP_PERIOD = 20        # 1H bars for choppiness index
CHOP_THRESHOLD = 32.0
CHOP_MAX_LONGS = 1      # max longs/day when choppy (vs normal 2)
CHOP_MAX_SHORTS = 1     # max shorts/day when choppy

# ── Early Kill ─────────────────────────────────────────────────────
EARLY_KILL_BARS = 4               # first 4 bars (60 minutes)
EARLY_KILL_R = -0.25              # default: exit if down 0.25R in first 4 bars
EARLY_KILL_MFE_FLOOR = 0.25      # AND peak MFE never reached this

# ── v4.1 Improvements ──────────────────────────────────────────────
DOW_SIZE_MULT = {0: 0.5, 3: 0.65}
VWAP_CAP_EVENING = 0.35

# ── v4.2 Improvements ──────────────────────────────────────────────
EVENING_20_BLOCK = True              # block entries during the 20:00 ET hour (40% WR, avgR=-0.141)
CLOSE_SKIP_PARTIAL = True            # CLOSE entries skip +1R partial — let runners run (avgR=+0.578, EdgeR=2.78)
VWAP_CAP_CLOSE_STRICT = 0.50         # stricter VWAP proximity for CLOSE skip-partial entries (vs 0.72 default)
# CLOSE-specific MFE ratchet: applied in ACTIVE_FREE when close_skip_partial is on
# Ultra-tight single-tier: locks +0.85R at +1R MFE — clips "reach +1R then fall back"
# variance while keeping full position.  Sharpe +0.03 over no-skip, +$2.2k PnL, MDD -0.3%.
CLOSE_MFE_RATCHET_TIERS = [
    (1.00, 0.85),   # MFE >= 1.00R → stop floor +0.85R (85% capture, only 0.15R room)
]
# Bar quality gate (v4.2) — filter incomplete pullbacks on trigger bar
BAR_QUALITY_SPIKE_ATR = 1.5          # reject if trigger bar range > N * ATR15 (spike bar)
BAR_QUALITY_CLOSE_FRAC = 0.30        # for longs, reject if close in bottom N% of bar range (weak close)
BAR_QUALITY_PERSIST = True           # require previous bar also closed above VWAP (1-bar persistence)
# MFE ratchet: (mfe_threshold_R, lock_floor_R) — progressive profit floor
MFE_RATCHET_TIERS = [
    (0.75, 0.15),   # MFE >= 0.75R → stop locked at minimum +0.15R
    (1.25, 0.40),   # MFE >= 1.25R → stop locked at minimum +0.40R
    (2.00, 0.80),   # MFE >= 2.00R → stop locked at minimum +0.80R
    (3.00, 1.50),   # MFE >= 3.00R → stop locked at minimum +1.50R
]
MFE_RESCUE_MIN_R = 0.50              # peak MFE required before stall protection
MFE_RESCUE_AFTER_BARS = 5            # avoid protecting first-hour noise
MFE_RESCUE_TRIGGER_R = 0.1
MFE_RESCUE_LOCK_R = 0.1
VWAP_CAP_CLOSE = 0.85                # CLOSE VWAP cap (same as CORE by default)
EQS_MIN_RTH = 0                      # entry quality score min for RTH (0 = disabled)
EQS_MIN_EVENING = 0                  # entry quality score min for evening (0 = disabled)
EQS_APPROACH_BARS = 3
EQS_BAR_RANGE_ATR_MAX = 1.5

# ── Optional 5m Micro Trigger ────────────────────────────────────────
USE_MICRO_TRIGGER = False
MICRO_WINDOW_BARS = 3


def build_instruments() -> list[Instrument]:
    # Always use MNQ contract specs (traded contract), even though
    # DEFAULT_SYMBOL="NQ" is used for price data routing.
    mnq = NQ_SPECS["MNQ"]
    instruments = [
        Instrument(
            symbol=DEFAULT_SYMBOL, root=DEFAULT_SYMBOL, venue="CME",
            tick_size=mnq["tick"], tick_value=mnq["tick_value"],
            multiplier=mnq["point_value"], currency="USD",
            point_value=mnq["point_value"], contract_expiry="",
            sec_type="FUT", trading_class=DEFAULT_SYMBOL,
        ),
    ]
    for inst in instruments:
        InstrumentRegistry.register(inst)
    return instruments
