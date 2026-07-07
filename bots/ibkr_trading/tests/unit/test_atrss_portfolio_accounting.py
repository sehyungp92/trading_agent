from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from backtests.swing.config import BacktestConfig
from backtests.swing.engine.portfolio_engine import (
    _candidate_qty_for_equity,
    _portfolio_mtm_equity,
    _timestamp_key,
)
import strategies.swing.atrss.config as atrss_config
from strategies.swing.atrss.models import (
    Candidate,
    CandidateType,
    DailyState,
    Direction,
    LegType,
    PositionBook,
    PositionLeg,
    Regime,
)


def test_atrss_timestamp_key_is_unit_stable_for_synchronized_join() -> None:
    ts_ns = np.datetime64("2026-05-07T13:00:00", "ns")
    ts_us = np.datetime64("2026-05-07T13:00:00", "us")

    mapping = {_timestamp_key(ts_ns): 7}

    assert mapping[_timestamp_key(ts_us)] == 7


def test_atrss_synchronized_portfolio_equity_includes_open_mtm() -> None:
    engines = {
        "QQQ": SimpleNamespace(equity=105_000.0, equity_curve=[106_250.0]),
        "GLD": SimpleNamespace(equity=98_000.0, equity_curve=[97_400.0]),
    }

    assert _portfolio_mtm_equity(engines, 103_000.0) == pytest.approx(103_650.0)


def test_atrss_deferred_candidate_qty_uses_current_shared_equity() -> None:
    candidate = Candidate(
        symbol="QQQ",
        type=CandidateType.PULLBACK,
        direction=Direction.LONG,
        trigger_price=110.0,
        initial_stop=100.0,
        qty=1,
    )
    daily_state = DailyState(regime=Regime.TREND, score=50.0)
    cfg = SimpleNamespace(base_risk_pct=0.02, size_reduction_months=[])

    qty = _candidate_qty_for_equity(
        candidate,
        daily_state,
        cfg,
        BacktestConfig(fixed_qty=None),
        equity=150_000.0,
        point_value=1.0,
    )

    assert qty == 300


def test_atrss_deferred_addon_b_qty_uses_current_shared_equity_and_base_cap() -> None:
    candidate = Candidate(
        symbol="QQQ",
        type=CandidateType.ADDON_B,
        direction=Direction.LONG,
        trigger_price=110.0,
        initial_stop=100.0,
        qty=1,
    )
    position = PositionBook(
        symbol="QQQ",
        direction=Direction.LONG,
        legs=[PositionLeg(leg_type=LegType.BASE, qty=10, entry_price=105.0)],
    )
    cfg = SimpleNamespace(base_risk_pct=0.02, size_reduction_months=[])

    qty = _candidate_qty_for_equity(
        candidate,
        DailyState(regime=Regime.TREND, score=50.0),
        cfg,
        BacktestConfig(fixed_qty=None),
        equity=1_000_000.0,
        point_value=1.0,
        position=position,
    )

    assert qty == max(1, int(10 * atrss_config.ADDON_B_SIZE_MULT))
