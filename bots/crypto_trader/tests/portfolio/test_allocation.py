from datetime import datetime, timezone

import pytest

from crypto_trader.core.models import Position, Side
from crypto_trader.portfolio.allocation import (
    admin_correct_unknown_allocation,
    allocation_residuals,
    derive_strategy_position_allocations,
    exchange_net_positions,
)
from crypto_trader.portfolio.state import OpenRisk


def test_derives_exact_allocation_from_lifecycle_entry() -> None:
    entry_time = datetime(2026, 6, 4, tzinfo=timezone.utc)

    allocations = derive_strategy_position_allocations([{
        "position_instance_id": "pos_1",
        "strategy_id": "momentum",
        "symbol": "BTC",
        "direction": "LONG",
        "qty": 0.2,
        "avg_entry": 90_000.0,
        "entry_time": entry_time.isoformat(),
        "metadata": {
            "risk_R": 0.5,
            "entry_order_ids": ["entry_1"],
            "entry_fill_ids": ["fill_1"],
        },
    }])

    assert len(allocations) == 1
    allocation = allocations[0]
    assert allocation.position_instance_id == "pos_1"
    assert allocation.confidence == "exact"
    assert allocation.source == "lifecycle"
    assert allocation.entry_order_ids == ["entry_1"]
    assert allocation.entry_fill_ids == ["fill_1"]


def test_falls_back_to_portfolio_state_with_recovered_confidence() -> None:
    risk = OpenRisk(
        strategy_id="trend",
        symbol="ETH",
        direction=Side.SHORT,
        risk_R=0.7,
        risk_id="risk_1",
        filled_qty=1.5,
        applied_fill_ids=["fill_a"],
    )

    allocations = derive_strategy_position_allocations([], [risk])

    assert allocations[0].source == "portfolio_state"
    assert allocations[0].confidence == "recovered"
    assert allocations[0].allocated_qty == pytest.approx(1.5)


def test_exchange_residual_is_explicit_unknown_allocation() -> None:
    observed = datetime(2026, 6, 4, tzinfo=timezone.utc)
    exchange = exchange_net_positions([
        Position("BTC", Side.LONG, 0.3, 90_000.0),
    ], observed_at=observed)
    allocations = derive_strategy_position_allocations([{
        "position_instance_id": "pos_1",
        "strategy_id": "momentum",
        "symbol": "BTC",
        "direction": "LONG",
        "qty": 0.1,
        "avg_entry": 90_000.0,
        "entry_time": observed.isoformat(),
        "metadata": {},
    }])

    residuals = allocation_residuals(exchange, allocations)

    assert len(residuals) == 1
    assert residuals[0].unallocated_qty == pytest.approx(0.2)
    assert residuals[0].unknown_allocation is True


def test_allocation_without_exchange_position_is_explicit_drift_not_unknown_exchange() -> None:
    observed = datetime(2026, 6, 4, tzinfo=timezone.utc)
    allocations = derive_strategy_position_allocations([{
        "position_instance_id": "pos_1",
        "strategy_id": "momentum",
        "symbol": "BTC",
        "direction": "LONG",
        "qty": 0.1,
        "avg_entry": 90_000.0,
        "entry_time": observed.isoformat(),
        "metadata": {},
    }])

    residuals = allocation_residuals([], allocations)

    assert len(residuals) == 1
    assert residuals[0].net_exchange_qty == 0.0
    assert residuals[0].allocated_qty == pytest.approx(0.1)
    assert residuals[0].unallocated_qty == pytest.approx(-0.1)
    assert residuals[0].unknown_allocation is False


def test_opposite_direction_allocation_and_exchange_position_reports_both_drifts() -> None:
    observed = datetime(2026, 6, 4, tzinfo=timezone.utc)
    exchange = exchange_net_positions([
        Position("BTC", Side.SHORT, 0.3, 90_000.0),
    ], observed_at=observed)
    allocations = derive_strategy_position_allocations([{
        "position_instance_id": "pos_1",
        "strategy_id": "momentum",
        "symbol": "BTC",
        "direction": "LONG",
        "qty": 0.1,
        "avg_entry": 90_000.0,
        "entry_time": observed.isoformat(),
        "metadata": {},
    }])

    residuals = allocation_residuals(exchange, allocations)

    assert {(item.direction, item.unallocated_qty, item.unknown_allocation) for item in residuals} == {
        (Side.SHORT, 0.3, True),
        (Side.LONG, -0.1, False),
    }


def test_admin_correction_assigns_unknown_exposure_with_audit_keys() -> None:
    residual = allocation_residuals(
        exchange_net_positions([Position("SOL", Side.LONG, 2.0, 150.0)]),
        [],
    )[0]

    corrected = admin_correct_unknown_allocation(
        residual,
        strategy_id="breakout",
        position_instance_id="admin_pos_1",
        avg_entry=150.0,
        reason="operator matched exchange fill",
    )

    assert corrected.source == "admin_correction"
    assert corrected.confidence == "recovered"
    assert corrected.strategy_id == "breakout"
    assert corrected.position_instance_id == "admin_pos_1"
