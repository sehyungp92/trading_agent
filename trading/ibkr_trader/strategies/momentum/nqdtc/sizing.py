"""NQ Dominant Trend Capture v2.0 — risk sizing, quality multiplier, friction gate."""
from __future__ import annotations

from . import config as C
from .models import ChopMode, CompositeRegime


# ---------------------------------------------------------------------------
# Friction gate (Section 3.3)
# ---------------------------------------------------------------------------

def friction_ok(symbol: str, R_dollars: float) -> bool:
    """Block entry if friction > FRICTION_CAP of 1R."""
    spec = C.NQ_SPECS.get(symbol, C.NQ_SPECS["NQ"])
    tv = spec["tick_value"]
    slip_cost = 2 * C.SLIPPAGE_TICKS.get(symbol, 1) * tv
    total = slip_cost + C.COMMISSION_RT.get(symbol, 4.12) + C.COST_BUFFER_USD.get(symbol, 2.00)
    return total <= C.FRICTION_CAP * R_dollars


def fee_R_estimate(symbol: str, R_dollars: float) -> float:
    """Fee-in-R for logging."""
    spec = C.NQ_SPECS.get(symbol, C.NQ_SPECS["NQ"])
    tv = spec["tick_value"]
    slip_cost = 2 * C.SLIPPAGE_TICKS.get(symbol, 1) * tv
    total = slip_cost + C.COMMISSION_RT.get(symbol, 4.12) + C.COST_BUFFER_USD.get(symbol, 2.00)
    return total / R_dollars if R_dollars > 0 else 999.0


def tp1_viable(symbol: str, R_dollars: float) -> bool:
    """Check that minimum TP1 profit exceeds round-trip fees."""
    min_tp1_r = min(sched[0][0] for sched in C.EXIT_TIERS.values())
    fee_r = fee_R_estimate(symbol, R_dollars)
    return min_tp1_r > fee_r


# ---------------------------------------------------------------------------
# Quality multiplier (Section 15.2)
# ---------------------------------------------------------------------------

def compute_quality_mult(
    composite: CompositeRegime,
    mode: ChopMode,
    disp_norm: float,
    es_opposing: bool = False,
) -> float:
    """quality_mult = clamp(regime_mult * chop_mult * disp_mult * es_mult, min, max).

    Restored per spec §15.2: multiplicative regime/chop/displacement scaling.
    es_opposing: True when trade direction opposes ES SMA200 daily trend.
    """
    regime_mult = C.REGIME_MULT.get(composite.value, 1.0)
    chop_mult = C.CHOP_SIZE_MULT if mode == ChopMode.DEGRADED else 1.0
    disp_mult = 1.0   # nqdtc_v4 step 5: was 0.70+0.30*disp_norm — low-disp entries outperform but were penalized
    es_mult = C.ES_OPPOSING_SIZE_MULT if es_opposing else 1.0
    raw = regime_mult * chop_mult * disp_mult * es_mult
    return max(C.QUALITY_MULT_MIN, min(C.QUALITY_MULT_MAX, raw))


def compute_disp_norm(disp_metric: float, t70: float, t90: float) -> float:
    """Normalize displacement to [0,1] range."""
    denom = max(1e-9, t90 - t70)
    return max(0.0, min(1.0, (disp_metric - t70) / denom))


# ---------------------------------------------------------------------------
# Final risk (Section 15.3)
# ---------------------------------------------------------------------------

def compute_final_risk_pct(
    quality_mult: float,
) -> tuple[float, bool]:
    """Returns (final_risk_pct, floored_to_risk_floor).

    Phase 1.3: expiry_mult removed (was always 1.0).
    """
    raw = C.BASE_RISK_PCT * quality_mult
    risk_floor = C.RISK_FLOOR_FRAC * C.BASE_RISK_PCT
    floored = raw < risk_floor
    final = max(risk_floor, min(C.BASE_RISK_PCT, raw))
    return final, floored


# ---------------------------------------------------------------------------
# Position sizing (Section 15.5)
# ---------------------------------------------------------------------------

def compute_contracts(
    symbol: str,
    entry: float,
    stop: float,
    equity: float,
    final_risk_pct: float,
) -> int:
    """Compute number of contracts."""
    spec = C.NQ_SPECS.get(symbol, C.NQ_SPECS["NQ"])
    tick = spec["tick"]
    tv = spec["tick_value"]

    stop_ticks = abs(entry - stop) / tick
    risk_per_contract = (
        stop_ticks * tv +
        C.COMMISSION_RT.get(symbol, 4.12) / 2 +
        C.COST_BUFFER_USD.get(symbol, 2.00) / 2
    )
    if risk_per_contract <= 0:
        return 0
    qty = int((equity * final_risk_pct) // risk_per_contract)
    return max(0, qty)
