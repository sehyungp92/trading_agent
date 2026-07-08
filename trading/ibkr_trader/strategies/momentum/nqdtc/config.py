"""NQ Dominant Trend Capture v2.1 — strategy constants and instrument setup."""
from __future__ import annotations

import os

from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry

# ---------------------------------------------------------------------------
# Strategy identity
# ---------------------------------------------------------------------------
STRATEGY_ID = "NQDTC_v2.1"

# ---------------------------------------------------------------------------
# Contract specifications (Section 2)
# ---------------------------------------------------------------------------
NQ_SPECS = {
    "NQ":  {"tick": 0.25, "tick_value": 5.00, "point_value": 20.00},
    "MNQ": {"tick": 0.25, "tick_value": 0.50, "point_value":  2.00},
}
DEFAULT_SYMBOL = os.environ.get("TRADING_SYMBOL", "NQ")

# ---------------------------------------------------------------------------
# Sessions (Section 1.3)
# ---------------------------------------------------------------------------
# ETH: 18:00–09:30 NY  |  RTH: 09:30–16:15 NY
ETH_START_H, ETH_START_M = 18, 0
RTH_START_H, RTH_START_M = 9, 30
RTH_END_H, RTH_END_M = 16, 15

# Entry windows
ETH_ENTRY_START_H, ETH_ENTRY_START_M = 4, 30
ETH_ENTRY_END_H, ETH_ENTRY_END_M = 9, 15
RTH_ENTRY_START_H, RTH_ENTRY_START_M = 9, 45
RTH_ENTRY_END_H, RTH_ENTRY_END_M = 12, 0

# RTH entries: enabled to increase frequency
RTH_ENTRIES_ENABLED = True
BLOCK_RTH_DEGRADED = True      # block DEGRADED-mode entries during RTH

# Hour-level entry blocks (ported from NQDTCAblationFlags — optimisation pass)
BLOCK_04_ET = True   # block all entries during the 04:00 ET hour (thin pre-dawn liquidity)
BLOCK_05_ET = False
BLOCK_06_ET = True   # P5: block all entries during the 06:00 ET hour (pre-European-open, WR=39%)
BLOCK_09_ET = False
BLOCK_12_ET = True   # block all entries during the 12:00 ET hour (17% WR, outlier-dependent)
BLOCK_THURSDAY = False  # tested: blocks 54 trades, -$11.6k PnL, Sharpe -0.03
BLOCK_ETH_SHORTS = True

# ---------------------------------------------------------------------------
# Risk (Section 3)
# ---------------------------------------------------------------------------
RISK_PCT = 0.0045
BASE_RISK_PCT = 0.0045

# Costs (Section 3.2)
COMMISSION_RT = {"NQ": 4.12, "MNQ": 1.64}
SLIPPAGE_TICKS = {"NQ": 1, "MNQ": 1}
COST_BUFFER_USD = {"NQ": 2.00, "MNQ": 1.00}
FRICTION_CAP = 0.10         # 10% of 1R

# ---------------------------------------------------------------------------
# News blackout (Section 4)
# ---------------------------------------------------------------------------
NEWS_BLACKOUT_MINUTES = 30
NEWS_PROFIT_FUNDED_BUFFER_MIN = 15

# ---------------------------------------------------------------------------
# Box compression (Section 8)
# ---------------------------------------------------------------------------
VIOL_MAX = 4
M_BREAK = 8                 # DIRTY window bars
CONTAINMENT_MIN = 0.85
BOX_HEIGHT_MIN_ATR_SHORT = 0.35  # L=20
BOX_HEIGHT_MIN_ATR_LONG = 0.30   # L=32,48
BOX_BUCKET_HYSTERESIS = 2        # consecutive bars to change L

# Adaptive L values (session-scaled for session-filtered bars — fix #1 alignment)
# ETH ≈ 31 bars/session vs 45 total/day → 0.69 session fraction
# Original unfiltered values were {LOW: 20, MID: 32, HIGH: 48}
# Session-scaled to match equivalent calendar coverage:
ADAPTIVE_L = {"LOW": 16, "MID": 18, "HIGH": 40}

# DIRTY (Section G)
DIRTY_TIMEOUT_MULT = 2.0
DIRTY_DEPTH_FRAC = 0.55
DIRTY_RESET_SHIFT_ATR = 0.25
DIRTY_RESET_DURATION_FRAC = 0.30

# ---------------------------------------------------------------------------
# Regime (Section 9)
# ---------------------------------------------------------------------------
K_SLOPE = 0.10
ADX_TRENDING = 25
ADX_RANGE = 20

# nqdtc_v4 step 8: Range outperforms (+0.867R) but was under-sized (0.40); Aligned worst-per-trade (0.439R) was over-sized
REGIME_MULT = {
    "Aligned": 0.80,  # was 1.00 — worst per-trade performance (+0.439R)
    "Neutral": 0.60,  # unchanged
    "Caution": 0.35,  # unchanged
    "Range":   0.65,  # was 0.40 — best performance (+0.867R avg R)
    "Counter": 0.00,  # unchanged
}

# ---------------------------------------------------------------------------
# CHOP (Section 10)
# ---------------------------------------------------------------------------
CHOP_ATR_PCTL_1 = 75
CHOP_ATR_PCTL_2 = 90
CHOP_VWAP_CROSS_1 = 5
CHOP_VWAP_CROSS_2 = 8
CHOP_VWAP_CROSS_LB = 40     # bars lookback for cross count
CHOP_SIZE_MULT = 0.70

# ---------------------------------------------------------------------------
# Breakout qualification (Section 11)
# ---------------------------------------------------------------------------
Q_DISP = 0.10               # nqdtc_v4 step 4: was 0.30 — lowest-disp breakouts outperform
Q_DISP_TIGHT_BOX = 0.05    # nqdtc_v4 step 4: was 0.15
Q_DISP_ALIGNED = 0.10      # nqdtc_v4 step 4: was 0.20
BREAKOUT_REJECT_RANGE_MULT = 1.8
BREAKOUT_REJECT_BODY_RATIO = 0.30
BREAKOUT_REJECT_WICK_RATIO = 0.50
BREAKOUT_REJECT_RVOL = 2.0

# ---------------------------------------------------------------------------
# Breakout state (Section 12)
# ---------------------------------------------------------------------------
HARD_EXPIRY_EXTENSION = 12
CONTINUATION_R_PROXY = 2.0
MM_WIDTH_MULT = 2.0
INVALIDATION_CONSEC_INSIDE = 3

# ---------------------------------------------------------------------------
# Evidence scorecard (Section 13)
# ---------------------------------------------------------------------------
SCORE_NORMAL = 1.5
SCORE_DEGRADED = 2.5
RVOL_SCORE_THRESH = 1.5

# Contextual score filters.
# Score and displacement are weak alone; these gates only act on score bands
# that diagnostics showed need box/RVOL context rather than a blunt threshold.
WEAK_SCORE_BAND_FILTER_ENABLED = False
WEAK_SCORE_BAND_LOW = 2.5
WEAK_SCORE_BAND_HIGH = 3.0
WEAK_SCORE_BAND_MAX_BOX_WIDTH = 225.0
WEAK_SCORE_BAND_MIN_RVOL = 1.75
WIDE_BOX_SCORE_FILTER_ENABLED = False
WIDE_BOX_MIN_WIDTH = 275.0
WIDE_BOX_MIN_SCORE = 3.0
WIDE_BOX_MIN_RVOL = 1.75

# ---------------------------------------------------------------------------
# Sizing (Section 15)
# ---------------------------------------------------------------------------
QUALITY_MULT_MIN = 0.50
QUALITY_MULT_MAX = 1.00
RISK_FLOOR_FRAC = 0.15      # risk floor = 0.15 * base_risk_pct

# Post-audit composite regime blocking (Step 0a)
# When True, entries in that composite regime are hard-blocked (not just sized down)
BLOCK_NEUTRAL_REGIME = False
BLOCK_ALIGNED_REGIME = True
BLOCK_CAUTION_REGIME = True
SCORE_NON_RANGE_MULT = 2.25    # score threshold multiplier for non-Range regimes
DISPLACEMENT_THRESHOLD_ENABLED = False

# ---------------------------------------------------------------------------
# Entries (Section 16)
# ---------------------------------------------------------------------------
A_ENTRY_ENABLED = True
A_ENTRY_RETEST_ENABLED = True
A_ENTRY_LATCH_ENABLED = False
C_CONT_ENTRY_ENABLED = False # nqdtc_v4 step 1: 5 trades, 40% WR, -0.341 avg R

# Entry A
A_STOP_ATR_MULT = 0.40
A1_OFFSET_TICKS = 2
A2_BUFFER_TICKS = 8
A_TTL_5M_BARS = 12
A_CANCEL_DEPTH_ATR = 0.15
A_MAX_BOX_WIDTH = 225.0
A_MIN_SCORE = 0.0      # <=0 disables the subtype-specific A score floor
A_BLOCK_WEAK_SCORE_BAND = False
A_WEAK_SCORE_BAND_LOW = 2.5
A_WEAK_SCORE_BAND_HIGH = 3.0

# Entry B
B_STOP_ATR_MULT = 0.40
B_SWEEP_DEPTH_ATR = 0.20
B_ALLOW_ALIGNED = True
B_ALLOW_RANGE = False
B_ALLOW_NEUTRAL = False
B_ALLOW_CAUTION = False
B_MIN_DISP_Q = 0.90

# Regime x subtype blocks (Phase 4)
BLOCK_CONT_ALIGNED = True    # block C_continuation when composite = Aligned
BLOCK_STD_NEUTRAL_LOW_DISP = True  # block C_standard in Neutral when disp_norm < 0.50

# C_continuation fixes (Improvement 1)
C_CONT_STOP_USE_BOX_EDGE = True     # use structural box-edge stop instead of hold_ref
C_CONT_MFE_GATE_R = 0.50            # require prior trade in breakout to have reached this MFE

# Entry C
C_HOLD_BARS = 1
C_ENTRY_OFFSET_ATR = 0.10
C_ENTRY_OFFSET_ATR_STANDARD = 0.248    # wider offset for high-value C_standard entries
C_ENTRY_OFFSET_ATR_CONTINUATION = 0.08 # tighter offset for C_continuation
C_CONT_PAUSE_ATR_MULT = 0.40

# Maker offset (Section 16.6)
OFFSET_BASE_ATR = 0.015
OFFSET_MULT_HIGH = 1.25
OFFSET_MULT_LOW = 0.85
OFFSET_ATR_MIN = 0.010
OFFSET_ATR_MAX = 0.030
RESCUE_MAX_SLIP_ATR = 0.03

# ---------------------------------------------------------------------------
# Trade management (Section 17)
# ---------------------------------------------------------------------------
# Measured move
MM_DURATION_MIN = 0.8
MM_DURATION_MAX = 1.4

# Exit tiers (Phase 2.3: widened Caution/Neutral, reduced TP1 fractions)
# nqdtc_v4 step 6: collapse TP2/TP3 — chandelier runner captures post-TP1 at +1.928 avg R
# Caution (TP1+runner) was +0.847R; Neutral/Aligned had TP2 diluting runner gains
TP1_R = 1.6
EXIT_TIERS = {
    "Aligned": [(TP1_R, 0.45), (2.25, 0.20)],
    "Neutral":  [(TP1_R, 0.45), (2.25, 0.20)],
    "Caution":  [(TP1_R, 0.45), (2.25, 0.20)],
}

# Profit-funded
BE_BUFFER_ATR_5M = 0.05

# Early breakeven: move stop to entry when MFE reaches threshold (pre-TP1)
# TESTED: 0.8R → Sharpe 2.23, 1.0R → Sharpe 2.16 (both worse than baseline 2.25)
EARLY_BE_MFE_R = 0.8             # disabled via flag — clips winners that retrace before running

# Post-TP1 ratchet floor (Section 17.4b)
RATCHET_LOCK_PCT = 0.35          # exit-opt: lock 35% of peak R (was 25%)
RATCHET_THRESHOLD_R = 0.5

# TP1-only cap mode. "range_degraded" is the historical behavior.
# Other supported values: "degraded_only", "range_only", "off".
TP1_ONLY_CAP_MODE = "degraded_only"

# Optional MFE-tier stop ratchet. Disabled by default; optimization probes this
# because diagnostics show strong MFE followed by STOP exits.
MFE_RATCHET_TIERS_ENABLED = False
MFE_RATCHET_T1_R = 2.0
MFE_RATCHET_T1_LOCK_R = 0.80
MFE_RATCHET_T2_R = 3.0
MFE_RATCHET_T2_LOCK_R = 1.35
MFE_RATCHET_T3_R = 4.0
MFE_RATCHET_T3_LOCK_R = 2.00

# Post-TP1 chandelier tightening (Improvement 4 — REVERTED: cut winners too aggressively)
CHANDELIER_POST_TP1_MULT_DECAY = 0.0    # disabled
CHANDELIER_POST_TP1_FLOOR_MULT = 1.2    # minimum chandelier mult after decay

# Runner chandelier (Section 17.5, Phase 3.3: tightened early tiers)
CHANDELIER_TIERS = [
    # (min_R, max_R, mm_reached, lookback, mult)
    (0.0, 1.5, False, 10, 3.0),    # exit-opt: wider early trail (was 2.5)
    (1.5, 3.0, False,  8, 2.2),    # exit-opt: slightly wider mid (was 2.0)
    (3.0, 4.0, False,  8, 1.6),    # exit-opt: tighter high-R (was 1.8)
    (4.0, 999, False,  8, 1.3),    # exit-opt: tighter 4R+ (was 1.8)
    (4.0, 999, True,   6, 1.0),    # exit-opt: very tight MM capture (was 1.2)
]

# Per-tier chandelier multiplier overrides (adaptive exit)
# 0 = use flat CHANDELIER_ATR_MULT or tier default
CHANDELIER_TIER0_MULT = 0.0   # R < 1.5 (default 3.0)
CHANDELIER_TIER1_MULT = 0.0   # 1.5-3R (default 2.2)
CHANDELIER_TIER2_MULT = 0.0   # 3-4R (default 1.6)
CHANDELIER_TIER3_MULT = 0.0   # 4R+ (default 1.3)
CHANDELIER_TIER4_MULT = 0.0   # 4R+ MM (default 1.0)
CHANDELIER_GRACE_BARS_30M = 0  # min bars before chandelier activates (0=immediate)

# Stale exit (Section 17.6)
STALE_BARS_NORMAL = 20       # 30m bars (10h)
STALE_BARS_DEGRADED = 14     # 30m bars (7h)
STALE_R_THRESHOLD = 0.3

# Re-entry (Section 18.1)
REENTRY_MIN_LOSS_R = -0.5
REENTRY_COOLDOWN_MIN = 30

# ---------------------------------------------------------------------------
# Squeeze (for scorecard)
# ---------------------------------------------------------------------------
SQUEEZE_GOOD_QUANTILE = 0.20
SQUEEZE_LOOSE_QUANTILE = 0.60

# ---------------------------------------------------------------------------
# News blackout events (Section 4)
# ---------------------------------------------------------------------------
NEWS_EVENTS = ["FOMC", "CPI", "NFP", "PCE", "GDP", "ISM", "FED_SPEECH", "JACKSON_HOLE"]
NEWS_BLACKOUT_WINDOW_BEFORE_MIN = 30
NEWS_BLACKOUT_WINDOW_AFTER_MIN = 30
NEWS_FLATTEN_LEAD_MIN = 15

# ---------------------------------------------------------------------------
# Overnight bridge (Section 17.6)
# ---------------------------------------------------------------------------
OVERNIGHT_BRIDGE_EXTRA_BARS = 4  # 30m bars added to stale timer at next RTH open

# ---------------------------------------------------------------------------
# Daily / weekly / monthly risk controls (Section 15 / 6)
# ---------------------------------------------------------------------------
DAILY_STOP_R = -2.5       # halt if daily realized <= this
WEEKLY_STOP_R = -6.0      # throttle threshold
MONTHLY_STOP_R = -10.0    # halt threshold

# ---------------------------------------------------------------------------
# 15m Slope Filter (Phase 1.1)
# ---------------------------------------------------------------------------
SLOPE_FILTER_ENABLED = True
MACD_FAST = 8
MACD_SLOW = 21
MACD_SIGNAL = 5
SLOPE_LOOKBACK = 3
CONT_SIZE_MULT = 0.50       # sizing multiplier for continuation breakouts
REVERSAL_SIZE_MULT = 1.15   # sizing bonus for reversal breakouts (slope opposes direction)
CONTINUATION_BREAKOUT_SIZE_MULT = 0.70  # v7: portfolio-level continuation sizing (box continuation_mode)

# ---------------------------------------------------------------------------
# Stop distance cap (Phase 3.1)
# ---------------------------------------------------------------------------
MAX_STOP_ATR_MULT = 0.65    # cap stop distance at 0.65 * ATR14_30m (tightened from 0.80)
MAX_STOP_WIDTH_PTS = 200.0  # v7: reject entries with stop > 200 points (absolute cap)
MAX_LOSS_CAP_R = -3.0       # v7: force exit if unrealized loss exceeds -3R (initial risk basis)

# ---------------------------------------------------------------------------
# ES SMA200 regime — cross-strategy directional sizing (Idea 2)
# ---------------------------------------------------------------------------
ES_DAILY_SMA_PERIOD = 200
ES_TREND_PERSIST_BARS = 2           # 2-bar confirmation (same as Vdubus)
ES_OPPOSING_SIZE_MULT = 0.60        # 40% size reduction when opposing ES trend

# ---------------------------------------------------------------------------
# ATR / indicator periods
# ---------------------------------------------------------------------------
ATR14_PERIOD = 14
ATR50_PERIOD = 50
EMA50_PERIOD = 50
ADX_PERIOD = 14
ATR14_1H_PERIOD = 14
ATR14_5M_PERIOD = 14

# ---------------------------------------------------------------------------
# Auto-optimization param_overrides defaults (Prereq 4)
# ---------------------------------------------------------------------------
MIN_INTER_TRADE_GAP_MINUTES = 30
ETH_SHORT_SIZE_MULT = 1.0            # no reduction by default (ETH short sizing)
MIN_BOX_WIDTH = 100
MAX_BOX_WIDTH = 99999                # no filter by default (box width gate)
LOSS_STREAK_THRESHOLD = 2
LOSS_STREAK_SKIP_BARS = 6            # 5m bars to skip after streak (6 = 30 min)
PROFIT_BE_R = 0.0                    # BE trigger R (0 = on TP1 fill, current behavior)

# ---------------------------------------------------------------------------
# Persistence (Section 20)
# ---------------------------------------------------------------------------
STATE_FILE = "nqdtc_state.json"


# ---------------------------------------------------------------------------
# Instrument builders
# ---------------------------------------------------------------------------

def build_instruments() -> dict[str, Instrument]:
    """Register NQ instrument with the OMS."""
    instruments: dict[str, Instrument] = {}
    for sym, spec in NQ_SPECS.items():
        inst = Instrument(
            symbol=sym, root=sym, venue="CME",
            tick_size=spec["tick"],
            tick_value=spec["tick_value"],
            multiplier=spec["point_value"],
            contract_expiry="",
            sec_type="FUT",
            trading_class=sym,
        )
        InstrumentRegistry.register(inst)
        instruments[sym] = inst
    return instruments
