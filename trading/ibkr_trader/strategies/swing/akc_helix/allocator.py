"""AKC-Helix Swing — priority ranking, position sizing, heat-cap allocation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime as _dt

from libs.broker_ibkr.risk_support.tick_rules import round_qty
from libs.oms.models.instrument import Instrument

from .config import (
    INSTRUMENT_CAP_R,
    PORTFOLIO_CAP_R,
    EXTREME_VOL_CAP_R,
    SYMBOL_CONFIGS,
)
from .models import (
    CircuitBreakerState,
    DailyState,
    Direction,
    SetupClass,
    SetupInstance,
    SetupState,
)

logger = logging.getLogger(__name__)

# Priority ordering: A > C > B > D (spec s10.6)
_CLASS_PRIORITY: dict[SetupClass, int] = {
    SetupClass.CLASS_A: 0,
    SetupClass.CLASS_C: 1,
    SetupClass.CLASS_B: 2,
    SetupClass.CLASS_D: 3,
}

_EPOCH = _dt.min


@dataclass(frozen=True)
class InitialRiskBasis:
    """Target versus actual initial risk after fills, rounding, and caps."""

    target_risk_dollars: float
    actual_risk_dollars: float
    utilization: float


# ---------------------------------------------------------------------------
# Ranking (spec s10.6)
# ---------------------------------------------------------------------------

def rank_setups(setups: list[SetupInstance]) -> list[SetupInstance]:
    """Sort by priority: 4H > 1H, then created_ts asc, then symbol asc."""
    return sorted(
        setups,
        key=lambda s: (
            _CLASS_PRIORITY.get(s.setup_class, 99),
            s.created_ts or _EPOCH,
            s.symbol,
        ),
    )


# ---------------------------------------------------------------------------
# Position sizing (spec s8.1, s8.2)
# ---------------------------------------------------------------------------

def compute_unit1_risk(
    equity: float,
    base_risk_pct: float,
    vol_factor: float,
) -> float:
    """Unit1 risk = equity * base_risk_pct * VolFactor (spec s8.1)."""
    return equity * base_risk_pct * vol_factor


def compute_position_size(
    fill_price: float,
    stop0: float,
    unit1_risk: float,
    setup_size_mult: float,
    point_value: float,
    max_contracts: int,
    mid_price: float = 0.0,
) -> int:
    """Risk-based position sizing.

    qty = (unit1_risk * setup_size_mult) / (|entry - stop| * point_value)
    Capped at max_contracts and MaxNotional (spec s8.2).
    """
    risk_per_contract = abs(fill_price - stop0) * point_value
    if risk_per_contract <= 0:
        return 0
    raw = (unit1_risk * setup_size_mult) / risk_per_contract
    qty = round_qty(raw)
    qty = min(qty, max_contracts)
    # MaxNotional check (spec s8.2): MaxNotional = MaxShares × mid_price
    if mid_price > 0 and max_contracts > 0:
        max_notional = max_contracts * mid_price
        if qty * mid_price > max_notional:
            qty = int(max_notional / mid_price)
    return qty


def compute_initial_risk_basis(
    entry_price: float,
    stop0: float,
    qty: int,
    point_value: float,
    target_risk_dollars: float,
) -> InitialRiskBasis:
    """Return the actual filled initial risk and target utilization."""
    target = max(float(target_risk_dollars or 0.0), 0.0)
    actual = abs(float(entry_price) - float(stop0)) * max(int(qty), 0) * float(point_value)
    utilization = actual / target if target > 0 else 0.0
    return InitialRiskBasis(
        target_risk_dollars=target,
        actual_risk_dollars=actual,
        utilization=utilization,
    )


def apply_initial_risk_basis(
    setup: SetupInstance,
    entry_price: float,
    qty: int,
    point_value: float,
    target_risk_dollars: float | None = None,
) -> InitialRiskBasis:
    """Persist actual initial risk as the R-accounting denominator."""
    target = (
        float(target_risk_dollars)
        if target_risk_dollars is not None
        else float(setup.target_initial_risk_dollars or setup.unit1_risk_dollars or 0.0)
    )
    basis = compute_initial_risk_basis(entry_price, setup.stop0, qty, point_value, target)
    setup.target_initial_risk_dollars = basis.target_risk_dollars
    setup.actual_initial_risk_dollars = basis.actual_risk_dollars
    setup.risk_utilization = basis.utilization
    if basis.actual_risk_dollars > 0:
        setup.unit1_risk_dollars = basis.actual_risk_dollars
    elif basis.target_risk_dollars > 0:
        setup.unit1_risk_dollars = basis.target_risk_dollars
    return basis


def compute_risk_r(
    entry: float,
    stop0: float,
    qty: int,
    point_value: float,
    unit1_risk: float,
) -> float:
    """Compute risk in R terms: (|entry-stop| * qty * pv) / unit1_risk."""
    if unit1_risk <= 0:
        return float("inf")
    return abs(entry - stop0) * qty * point_value / unit1_risk


# ---------------------------------------------------------------------------
# Portfolio allocation (spec s8.3, s10.6, s16)
# ---------------------------------------------------------------------------

def allocate(
    candidates: list[SetupInstance],
    active_setups: dict[str, SetupInstance],
    daily_states: dict[str, DailyState],
    equity: float,
    instruments: dict[str, Instrument],
    circuit_breakers: dict[str, CircuitBreakerState],
) -> list[SetupInstance]:
    """Rank candidates, apply basket rule & heat caps, return accepted with qty set."""
    if equity <= 0:
        return []

    ranked = rank_setups(list(candidates))
    accepted: list[SetupInstance] = []

    # Running tallies (in R terms)
    portfolio_r = 0.0
    instrument_r: dict[str, float] = {}

    # Existing position heat
    for sid, setup in active_setups.items():
        if setup.state not in (SetupState.FILLED, SetupState.ACTIVE):
            continue
        inst = instruments.get(setup.symbol)
        if inst is None:
            continue
        daily = daily_states.get(setup.symbol)
        vf = daily.vol_factor if daily else 1.0
        sym_cfg = SYMBOL_CONFIGS.get(setup.symbol)
        u1 = compute_unit1_risk(equity, sym_cfg.base_risk_pct if sym_cfg else 0.005, vf)
        if u1 > 0:
            r = compute_risk_r(
                setup.fill_price or setup.bos_level,
                setup.current_stop or setup.stop0,
                setup.qty_open or setup.qty_planned,
                inst.point_value,
                u1,
            )
            portfolio_r += r
            instrument_r[setup.symbol] = instrument_r.get(setup.symbol, 0.0) + r

    for s in ranked:
        sym = s.symbol
        inst = instruments.get(sym)
        if inst is None:
            continue

        daily = daily_states.get(sym)
        vf = daily.vol_factor if daily else 1.0
        u1 = compute_unit1_risk(equity, s.unit1_risk_dollars / equity if s.unit1_risk_dollars > 0 else 0.005, vf)
        if u1 <= 0:
            u1 = compute_unit1_risk(equity, 0.005, vf)

        # Compute qty if not already set
        if s.qty_planned <= 0:
            sym_cfg = SYMBOL_CONFIGS.get(sym)
            max_qty = sym_cfg.max_contracts if sym_cfg else 10
            s.qty_planned = compute_position_size(
                s.bos_level, s.stop0, u1, s.setup_size_mult,
                inst.point_value, max_qty,
            )
        if s.qty_planned <= 0:
            continue

        # Compute candidate R
        cand_r = compute_risk_r(
            s.bos_level, s.stop0, s.qty_planned, inst.point_value, u1,
        )

        # Portfolio heat cap
        extreme = daily.extreme_vol if daily else False
        port_cap = EXTREME_VOL_CAP_R if extreme else PORTFOLIO_CAP_R
        if portfolio_r + cand_r > port_cap:
            logger.debug("Skipping %s %s: portfolio heat exceeded", sym, s.setup_class.value)
            continue

        # Instrument heat cap
        sym_r = instrument_r.get(sym, 0.0) + cand_r
        if sym_r > INSTRUMENT_CAP_R:
            logger.debug("Skipping %s %s: instrument heat exceeded", sym, s.setup_class.value)
            continue

        # Halved sizing from circuit breaker
        cb = circuit_breakers.get(sym, CircuitBreakerState())
        if cb.halved_until and _dt.now() < cb.halved_until:
            s.qty_planned = max(1, s.qty_planned // 2)

        # Accept
        target_risk = u1 * s.setup_size_mult
        s.base_unit1_risk_dollars = u1
        s.target_initial_risk_dollars = target_risk
        s.unit1_risk_dollars = target_risk if target_risk > 0 else u1
        portfolio_r += cand_r
        instrument_r[sym] = instrument_r.get(sym, 0.0) + cand_r
        accepted.append(s)

    return accepted
