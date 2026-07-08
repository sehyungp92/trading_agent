"""ETRS vFinal — portfolio-level candidate ranking and allocation."""
from __future__ import annotations

import logging
from typing import Sequence

from libs.oms.models.instrument import Instrument

from .config import (
    CANDIDATE_RANK_MODE,
    MAX_PORTFOLIO_HEAT,
)
from .models import (
    Candidate,
    CandidateType,
    DailyState,
    Direction,
    HourlyState,
    PositionBook,
)

logger = logging.getLogger(__name__)


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Sort candidates per spec Section 13.2 — deterministic tie-break.

    Primary: daily_score desc (higher conviction wins)
    Tie-break 1: stop distance asc (tighter stop = higher priority)
    Tie-break 2: symbol asc (stable ordering)
    """
    def stop_dist(c: Candidate) -> float:
        return abs(c.trigger_price - c.initial_stop)

    mode = CANDIDATE_RANK_MODE
    if mode == "stop_first":
        key_fn = lambda c: (stop_dist(c), -c.rank_score, c.symbol)
    elif mode == "score_per_risk":
        key_fn = lambda c: (-(c.rank_score / max(stop_dist(c), 1e-9)), stop_dist(c), c.symbol)
    elif mode == "gld_first":
        key_fn = lambda c: (0 if c.symbol == "GLD" else 1, -c.rank_score, stop_dist(c), c.symbol)
    elif mode == "qqq_first":
        key_fn = lambda c: (0 if c.symbol == "QQQ" else 1, -c.rank_score, stop_dist(c), c.symbol)
    else:
        key_fn = lambda c: (-c.rank_score, stop_dist(c), c.symbol)

    return sorted(candidates, key=key_fn)


def _heat_of_candidate(c: Candidate, instrument: Instrument) -> float:
    """Dollar heat for one candidate = qty * |entry - stop| * point_value."""
    risk_per_contract = abs(c.trigger_price - c.initial_stop) * instrument.point_value
    return risk_per_contract * c.qty


def allocate(
    candidates: list[Candidate],
    positions: dict[str, PositionBook],
    daily_states: dict[str, DailyState],
    equity: float,
    instruments: dict[str, Instrument],
    hourly_states: dict[str, HourlyState] | None = None,
) -> list[Candidate]:
    """Accept candidates in rank order subject to portfolio constraints.

    Returns the sub-list of accepted candidates with their ``qty`` set.
    """
    if equity <= 0:
        return []

    ranked = rank_candidates(list(candidates))
    accepted: list[Candidate] = []

    # Running tallies
    portfolio_heat_dollars = 0.0
    accepted_symbols: set[str] = set()

    # Existing position heat — use mark price (hourly close) per spec 11.2
    for sym, pos in positions.items():
        if pos.direction == Direction.FLAT:
            continue
        inst = instruments.get(sym)
        if inst is None:
            continue
        mark_price = pos.avg_entry  # fallback
        if hourly_states and sym in hourly_states:
            mark_price = hourly_states[sym].close
        pos_heat = abs(mark_price - pos.current_stop) * inst.point_value * pos.total_qty
        portfolio_heat_dollars += pos_heat

    for c in ranked:
        sym = c.symbol
        inst = instruments.get(sym)
        if inst is None:
            continue

        # Skip if no qty (shouldn't happen, but guard)
        if c.qty <= 0:
            continue

        cand_heat = _heat_of_candidate(c, inst)

        # --- Portfolio heat cap ---
        if (portfolio_heat_dollars + cand_heat) / equity > MAX_PORTFOLIO_HEAT:
            logger.debug("Skipping %s %s: portfolio heat exceeded", sym, c.type.value)
            continue

        # --- One base entry per symbol ---
        is_addon = c.type in (CandidateType.ADDON_A, CandidateType.ADDON_B)
        if not is_addon and sym in accepted_symbols:
            continue

        # Accepted
        portfolio_heat_dollars += cand_heat
        if not is_addon:
            accepted_symbols.add(sym)
        accepted.append(c)

    return accepted


def compute_position_size(
    entry_price: float,
    stop_price: float,
    equity: float,
    risk_pct: float,
    point_value: float,
) -> int:
    """Risk-based position sizing (spec Section 13.1 — floor).

    qty = floor((equity * risk_pct) / (|entry - stop| * point_value))
    """
    risk_per_contract = abs(entry_price - stop_price) * point_value
    if risk_per_contract <= 0:
        return 0
    raw = (equity * risk_pct) / risk_per_contract
    return int(raw)
