"""Stock-trader position-state tests for MFE/MAE-compatible telemetry."""

from datetime import datetime, timezone

import pytest

from strategies.stock.iaric.models import PositionState as IARICPositionState
from strategies.stock.alcb.models import Direction as ALCBDirection
from strategies.stock.alcb.models import PositionState as ALCBPositionState


def _now():
    return datetime.now(timezone.utc)


def test_iaric_position_exposes_fields_needed_for_mfe_mae_tracking():
    pos = IARICPositionState(
        entry_price=100.0,
        qty_entry=10,
        qty_open=10,
        final_stop=98.0,
        current_stop=98.0,
        entry_time=_now(),
        initial_risk_per_share=2.0,
        max_favorable_price=100.0,
        max_adverse_price=100.0,
    )

    assert pos.total_initial_risk_usd == 20.0
    assert pos.max_favorable_price == 100.0
    assert pos.max_adverse_price == 100.0


def test_alcb_position_exposes_fields_needed_for_mfe_mae_tracking():
    pos = ALCBPositionState(
        direction=ALCBDirection.LONG,
        entry_price=50.0,
        qty_entry=20,
        qty_open=20,
        final_stop=49.0,
        current_stop=49.0,
        entry_time=_now(),
        initial_risk_per_share=1.0,
        max_favorable_price=50.0,
        max_adverse_price=50.0,
        tp1_price=51.0,
        tp2_price=52.0,
    )

    assert pos.total_initial_risk_usd == 20.0
    assert pos.max_favorable_price == 50.0
    assert pos.max_adverse_price == 50.0


def test_iaric_max_favorable_price_supports_mfe_r_calculation():
    pos = IARICPositionState(
        entry_price=100.0,
        qty_entry=10,
        qty_open=10,
        final_stop=98.0,
        current_stop=98.0,
        entry_time=_now(),
        initial_risk_per_share=2.0,
        max_favorable_price=103.0,
        max_adverse_price=99.0,
    )

    mfe_r = (pos.max_favorable_price - pos.entry_price) / pos.initial_risk_per_share
    assert mfe_r == pytest.approx(1.5)


def test_alcb_max_favorable_price_supports_mfe_r_calculation():
    pos = ALCBPositionState(
        direction=ALCBDirection.LONG,
        entry_price=50.0,
        qty_entry=20,
        qty_open=20,
        final_stop=49.0,
        current_stop=49.0,
        entry_time=_now(),
        initial_risk_per_share=1.0,
        max_favorable_price=52.5,
        max_adverse_price=49.4,
        tp1_price=51.0,
        tp2_price=52.0,
    )

    mfe_r = (pos.max_favorable_price - pos.entry_price) / pos.initial_risk_per_share
    assert mfe_r == pytest.approx(2.5)


def test_position_max_adverse_price_supports_mae_r_calculation():
    pos = IARICPositionState(
        entry_price=100.0,
        qty_entry=10,
        qty_open=10,
        final_stop=98.0,
        current_stop=98.0,
        entry_time=_now(),
        initial_risk_per_share=2.0,
        max_favorable_price=103.0,
        max_adverse_price=97.5,
    )

    mae_r = (pos.max_adverse_price - pos.entry_price) / pos.initial_risk_per_share
    assert mae_r == pytest.approx(-1.25)
