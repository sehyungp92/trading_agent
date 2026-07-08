"""Crisis detector thresholds -- auto-optimized via 4-phase greedy search.

5-channel primary architecture:
  VIX, Credit Spread, Yield Curve, SPY-TLT Correlation, SPY Drawdown
  + hybrid conjunction (single CRISIS channel can trigger WARNING)

Calibrated against 2003-2026 daily data covering 9 labeled periods:
  7 crises (D/S): GFC, Euro Crisis, 2015 China, 2018 Q4, COVID,
                  2022 Inflation, 2025 Tariff
  2 corrections (C): 2023 SVB, 2026 Iran

Round 1 results (4-channel, conjunction=2):
  7/7 crises, avg latency 31.6d, WARNING FP 5.57%, CRISIS FP 1.90%

Round 2 results (5-channel + hybrid conjunction):
  7/7 crises, avg latency 18.1d, WARNING FP 4.45%, CRISIS FP 1.27%

Round 3: Recovery architecture + robustness validation

Auto-opt output: backtests/output/regime/crisis/
"""
from __future__ import annotations


# ── VIX Level (FRED: VIXCLS) ─────────────────────────────────────────
VIX_WATCH = 29.0       # R3: loosened from 27.0 to reduce persistent WATCH noise
VIX_WARNING = 41.5584  # R9: tightened after hard credit-impulse bridge
VIX_CRISIS = 42.0      # R3: loosened from 38.0 to cut CRISIS false positives

# ── Credit Spread (FRED: BAMLH0A0HYM2, basis points) ─────────────────
SPREAD_WATCH_BPS = 250.0   # was 400 -- aggressive lowering for earlier detection
SPREAD_WARNING_BPS = 336.0  # was 500 -- fine-tuned from Phase 4
SPREAD_CRISIS_BPS = 500.0   # R8: balanced lower advisory saturation vs latency

# ── Yield Curve (FRED: T10Y2Y, percentage points) ────────────────────
# Watch: persistent inversion
SLOPE_WATCH_THRESHOLD = -0.50

# Warning/Crisis: rapid un-inversion (curve steepening after inversion)
# Historically precedes recessions by 3-12 months.
SLOPE_STEEPEN_WARNING = 0.50   # was 0.40 -- raised to reduce warning FP
SLOPE_STEEPEN_CRISIS = 0.60    # 20d change >+0.60% AND was inverted within 90d
SLOPE_INVERSION_LOOKBACK = 90  # days to check for prior inversion

# ── SPY-TLT Correlation ──────────────────────────────────────────────
# Positive correlation = stocks and bonds falling together (risk-off failure)
CORR_WINDOW = 20               # was 10 -- wider window for stability
CORR_WATCH = 0.40              # was 0.30 -- raised to reduce watch noise
CORR_WARNING = 0.55            # was 0.50 -- raised to reduce warning FP
CORR_CRISIS = 0.55             # was 0.50 -- raised to match warning
CORR_CRISIS_SPY_DD = -0.07     # was -0.05 -- deeper drawdown required for crisis

# ── Confirming: VIX Term Structure ────────────────────────────────────
# VIX/VIX3M > 1.0 = backwardation (near-term fear exceeds longer-term)
VIX_TERM_STRUCTURE_THRESHOLD = 1.0  # ratio VIX/VIX3M

# ── SPY Drawdown (5th Primary Channel) ────────────────────────────────
SPY_DD_WATCH = -0.05    # 10d cumulative return thresholds (negative)
SPY_DD_WARNING = -0.055296  # R8: fine-tuned for faster warning confirmation
SPY_DD_CRISIS = -0.10

# ── Conjunction Requirements ──────────────────────────────────────────
# No single indicator can trigger WARNING or CRISIS.
WATCH_MIN_PRIMARY = 1     # 1+ primary at Watch -> WATCH
WARNING_MIN_PRIMARY = 2   # 2+ primary at Warning -> WARNING
CRISIS_MIN_PRIMARY = 2    # 2+ primary at Crisis -> CRISIS
CRISIS_ALT_WARNING = 3    # OR 3+ primary at Warning -> CRISIS

# External advisory split. Internal WATCH is intentionally sensitive because it
# warms hysteresis; external WATCH is stricter so dashboards/users do not see
# the 93% always-on internal buffer as a real early alert.
ADVISORY_WATCH_MIN_PRIMARY = 4    # 4+ primary channels at WATCH+
ADVISORY_WATCH_MIN_WARNING = 2    # OR 2+ primary channels at WARNING+
ADVISORY_WATCH_MIN_CRISIS = 1     # OR any one primary at CRISIS

# Stress formation advisory. These shock/grind signals do not alter internal
# WARNING/CRISIS thresholds by themselves; they surface earlier stress for
# advisory and economic action-layer testing.
STRESS_FORMATION_MIN_SCORE = 2

# Shock path: fast equity/vol/correlation breaks.
# R8 early-action optimization: more sensitive equity/vol trigger improved
# portfolio-action latency while keeping pre-action FP below target.
SHOCK_SPY_3D_RETURN = -0.025
SHOCK_SPY_5D_RETURN = -0.0432
SHOCK_VIX_3D_CHANGE = 5.0
SHOCK_MIN_VIX = 22.0
SHOCK_CORR_MIN = 0.45
SHOCK_CORR_SPY_5D_RETURN = -0.02

# Grind path: slower credit/vol/drawdown deterioration.
GRIND_SPREAD_20D_CHANGE_BPS = 100.0
GRIND_SPY_20D_RETURN = -0.08
GRIND_VIX_MIN = 28.0
GRIND_VIX_PERSIST_DAYS = 7
GRIND_SPREAD_CONFIRM_BPS = 300.0
GRIND_SPY_CONFIRM_20D_RETURN = -0.05

# Credit-impulse bridge: R8 accepted conservative 500 bps spread gate.
# This targets confirmation-limited events where credit is already stressed and
# a short equity impulse appears before the slower SPY drawdown channel confirms.
CREDIT_IMPULSE_SPREAD_BPS = 540.0
CREDIT_IMPULSE_SPY_3D_RETURN = -0.015
CREDIT_IMPULSE_MIN_VIX = 18.0

# Hard confirmation bridge.
# When credit stress is already severe and the credit-impulse path persists,
# this can promote raw WARNING before the slower 10d SPY drawdown channel
# confirms. Persistence keeps the downstream hard regime label conservative.
HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS = 3
HARD_CREDIT_IMPULSE_WARNING_MIN_PRIMARY = 1

# ── Hybrid Conjunction ──────────────────────────────────────────────
# When any primary is at CRISIS, allow WARNING with fewer channels.
HYBRID_WARNING_MIN_CRISIS = 1   # channels at CRISIS to activate hybrid mode
HYBRID_WARNING_MIN_PRIMARY = 2  # R2: raised from 1 -- requires 2 at WARNING+ in hybrid mode

# ── Hysteresis (sticky de-escalation) ─────────────────────────────────
# Immediate escalation, but require consecutive days below to de-escalate.
DEESCALATE_CRISIS_DAYS = 2   # R8: slightly stickier crisis state improved stability
DEESCALATE_WARNING_DAYS = 1  # R3: fast recovery reduces opportunity cost
DEESCALATE_WATCH_DAYS = 1    # R3: WATCH is an internal buffer, not a risk cut

# ── Accelerated De-escalation ────────────────────────────────────────
# When raw alert is NORMAL for N consecutive days, jump directly
# to NORMAL instead of stepping down one level at a time.
# Set to 0 to disable (standard step-down only).
ACCEL_DEESCALATE_NORMAL_DAYS = 3  # R3: jump to NORMAL after 3 raw-normal days

# ── Recovery Ramp (post-de-escalation) ───────────────────────────────
# After de-escalating from WARNING/CRISIS, gradually ramp risk back up.
RECOVERY_RAMP_DAYS = 5            # days to ramp from floor to full risk
RECOVERY_RAMP_FLOOR = 0.75        # starting multiplier during ramp

# ── Risk Multipliers ─────────────────────────────────────────────────
RISK_MULT_NORMAL = 1.0
RISK_MULT_WATCH = 1.0       # Watch = internal hysteresis buffer, no portfolio impact
RISK_MULT_WARNING = 0.65    # Action-layer economic opt: 35% risk reduction
RISK_MULT_CRISIS = 0.30     # Action-layer economic opt: 70% risk reduction
STRESS_FORMATION_RISK_MULT_SHOCK = 0.75  # R8 economic opt: fast crash pre-action
STRESS_FORMATION_RISK_MULT_GRIND = 0.90  # Slow deterioration pre-action
STRESS_FORMATION_RISK_MULT_CREDIT_IMPULSE = 0.75  # Credit stress + equity impulse

# Action-layer regime/provenance modifiers.
# HMM stress/defensive regimes already reduce baseline risk, so WARNING should
# add a smaller incremental cut there.  Credit-impulse bridge WARNING keeps the
# hard early label, but uses pre-action-like sizing until broader confirmation.
ACTION_WARNING_RISK_MULT_STRESS_REGIME = 0.80
ACTION_WARNING_RISK_MULT_DEFENSIVE_REGIME = 0.85
ACTION_WARNING_DD_MULT_STRESS_REGIME = 0.95
ACTION_WARNING_DD_MULT_DEFENSIVE_REGIME = 1.00
ACTION_WARNING_CAP_MULT_STRESS_REGIME = 0.90
ACTION_WARNING_CAP_MULT_DEFENSIVE_REGIME = 0.95
ACTION_CREDIT_BRIDGE_WARNING_RISK_MULT = 0.75
ACTION_CREDIT_BRIDGE_WARNING_DD_MULT = 0.95
ACTION_CREDIT_BRIDGE_WARNING_CAP_MULT = 0.90
ACTION_CREDIT_BRIDGE_WARNING_LONG_MULT = 0.80
ACTION_CREDIT_BRIDGE_WARNING_CONTRACTS_MULT = 0.85

DD_TIER_MULT_NORMAL = 1.0
DD_TIER_MULT_WATCH = 1.0
DD_TIER_MULT_WARNING = 0.90  # 10% DD tightening
DD_TIER_MULT_CRISIS = 0.75   # 25% DD tightening

# ── Staleness ─────────────────────────────────────────────────────────
STALENESS_THRESHOLD_DAYS = 3  # more aggressive than regime's 7 days (daily cadence)

# ── Alert Level Constants ─────────────────────────────────────────────
ALERT_NORMAL = "NORMAL"
ALERT_WATCH = "WATCH"
ALERT_WARNING = "WARNING"
ALERT_CRISIS = "CRISIS"

ALERT_LEVELS = (ALERT_NORMAL, ALERT_WATCH, ALERT_WARNING, ALERT_CRISIS)
ALERT_LEVEL_INT = {ALERT_NORMAL: 0, ALERT_WATCH: 1, ALERT_WARNING: 2, ALERT_CRISIS: 3}
