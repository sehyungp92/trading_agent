from __future__ import annotations

from datetime import datetime, timezone

from libs.oms.models.events import OMSEvent, OMSEventType
from libs.oms.models.instrument import Instrument
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderStatus, OrderType
import pytest

from tests.integration.parity.normalizers import (
    normalize_oms_events,
    normalize_order_intents,
    normalize_state_snapshot,
    normalize_trade_ledger,
)


def test_order_normalizer_drops_broker_ids_but_keeps_role_and_economics() -> None:
    order = OMSOrder(
        oms_order_id="volatile-oms-id",
        client_order_id="CLIENT-1",
        strategy_id="TPC",
        instrument=Instrument("QQQ", "QQQ", "SMART", 0.01, 0.01, 1.0),
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        limit_price=101.234,
        role=OrderRole.ENTRY,
        status=OrderStatus.ROUTED,
        broker_order_id=12345,
        perm_id=67890,
    )

    assert normalize_order_intents(
        [order],
        family_for_strategy=lambda _sid: "swing",
        instrument_ticks={"QQQ": 0.01},
    ) == [
        {
            "strategy_id": "TPC",
            "family": "swing",
            "symbol": "QQQ",
            "side": "BUY",
            "qty": 10,
            "order_type": "LIMIT",
            "tif": "DAY",
            "limit_price": 101.23,
            "stop_price": None,
            "parent_order_id": "",
            "client_tag": "CLIENT-1",
            "order_role": "ENTRY",
        }
    ]


def test_event_normalizer_buckets_risk_denial_reason_and_keeps_fill_economics() -> None:
    events = [
        OMSEvent(
            event_type=OMSEventType.RISK_DENIAL,
            timestamp=datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc),
            strategy_id="VdubusNQ_v4",
            payload={"reason": "Portfolio rule: directional_cap: LONG risk too high"},
        ),
        OMSEvent(
            event_type=OMSEventType.FILL,
            timestamp=datetime(2026, 5, 20, 14, 31, tzinfo=timezone.utc),
            strategy_id="NQDTC_v2.1",
            payload={
                "symbol": "MNQ",
                "side": "BUY",
                "qty": 5,
                "price": 20000.12,
                "timestamp": "2026-05-20T14:31:00+00:00",
                "role": "ENTRY",
            },
        ),
    ]

    assert normalize_oms_events(
        events,
        family_for_strategy=lambda _sid: "momentum",
        instrument_ticks={"MNQ": 0.25},
    ) == [
        {
            "event_type": "RISK_DENIAL",
            "strategy_id": "VdubusNQ_v4",
            "family": "momentum",
            "symbol": "",
            "side": "",
            "qty": 0,
            "price": None,
            "status": "",
            "reason": "portfolio_rule:directional_cap",
            "order_role": "",
            "event_time": "2026-05-20T14:30:00+00:00",
        },
        {
            "event_type": "FILL",
            "strategy_id": "NQDTC_v2.1",
            "family": "momentum",
            "symbol": "MNQ",
            "side": "BUY",
            "qty": 5,
            "price": 20000.0,
            "status": "",
            "reason": "",
            "order_role": "ENTRY",
            "event_time": "2026-05-20T14:31:00+00:00",
        },
    ]


def test_order_normalizer_buckets_generated_client_tags() -> None:
    order = OMSOrder(
        client_order_id="MSFT-entry-deadbeef1234",
        strategy_id="IARIC_v1",
        instrument=Instrument("MSFT", "MSFT", "SMART", 0.01, 0.01, 1.0),
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        limit_price=412.0,
        role=OrderRole.ENTRY,
    )

    [normalized] = normalize_order_intents(
        [order],
        family_for_strategy=lambda _sid: "stock",
        instrument_ticks={"MSFT": 0.01},
    )

    assert normalized["client_tag"] == "MSFT:ENTRY"


def test_state_normalizer_rejects_unmodeled_values_instead_of_stringifying() -> None:
    with pytest.raises(TypeError, match="unsupported parity state value"):
        normalize_state_snapshot({"bad": object()})


def test_order_normalizer_rejects_unknown_object_shape() -> None:
    with pytest.raises(TypeError, match="unsupported parity order value"):
        normalize_order_intents([object()])


def test_event_normalizer_rejects_unknown_object_shape() -> None:
    with pytest.raises(TypeError, match="unsupported parity event value"):
        normalize_oms_events([object()])


def test_event_normalizer_rejects_unknown_event_type() -> None:
    with pytest.raises(TypeError, match="unsupported parity event type"):
        normalize_oms_events([{"event_type": "NOT_AN_OMS_EVENT"}])


def test_event_normalizer_rejects_malformed_payload() -> None:
    with pytest.raises(TypeError, match="event payload"):
        normalize_oms_events([{"event_type": "FILL", "payload": []}])


def test_trade_ledger_normalizer_rejects_unknown_object_shape() -> None:
    with pytest.raises(TypeError, match="unsupported parity trade value"):
        normalize_trade_ledger([object()])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_state_normalizer_rejects_non_finite_float(value: float) -> None:
    with pytest.raises(TypeError, match="non-finite"):
        normalize_state_snapshot({"bad": value})
