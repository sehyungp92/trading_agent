from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from strategies.swing.akc_helix.engine import HelixEngine
from strategies.swing.akc_helix.models import Direction as HelixDirection
from strategies.swing.akc_helix.models import SetupClass, SetupInstance as HelixSetup, SetupState as HelixSetupState

UTC = timezone.utc


def _helix_setup() -> HelixSetup:
    return HelixSetup(
        setup_id="HELIX-1",
        symbol="QQQ",
        setup_class=SetupClass.CLASS_A,
        direction=HelixDirection.LONG,
        origin_tf="4H",
        state=HelixSetupState.NEW,
        created_ts=datetime(2026, 4, 27, 13, 0, tzinfo=UTC),
        bos_level=505.5,
        stop0=499.0,
        current_stop=499.0,
        qty_planned=3,
        oca_group="HELIX-OCA",
    )


def test_akc_helix_live_wrapper_routes_entry_and_exit_requests_through_core() -> None:
    engine = HelixEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    setup = _helix_setup()

    routed_setup = engine._route_core_entry_request(
        bar_ts=datetime(2026, 4, 27, 13, 0, tzinfo=UTC),
        setup=setup,
        client_order_id="OMS-HELIX-ENTRY",
        order_type="STOP_LIMIT",
        limit_price=505.75,
    )

    assert routed_setup.state is HelixSetupState.ARMED
    assert engine.pending_setups[setup.setup_id].state is HelixSetupState.ARMED
    assert engine.health_status()["last_decision_code"] == "ENTRY_REQUESTED"

    routed_setup.state = HelixSetupState.ACTIVE
    routed_setup.fill_price = 505.75
    routed_setup.fill_qty = 3
    routed_setup.qty_open = 3
    routed_setup.stop_order_id = "STOP-1"
    engine.pending_setups.pop(setup.setup_id, None)
    engine.active_setups[setup.setup_id] = routed_setup

    engine._route_core_stop_update(
        bar_ts=datetime(2026, 4, 27, 14, 0, tzinfo=UTC),
        setup_id=setup.setup_id,
        symbol=setup.symbol,
        stop_price=500.5,
        qty=3,
        reason="trailing",
    )
    assert engine.active_setups[setup.setup_id].current_stop == 500.5
    assert engine.health_status()["last_decision_code"] == "STOP_REPLACEMENT_REQUESTED"

    engine._route_core_partial_exit_request(
        bar_ts=datetime(2026, 4, 27, 14, 5, tzinfo=UTC),
        setup_id=setup.setup_id,
        symbol=setup.symbol,
        client_order_id="OMS-HELIX-PARTIAL",
        qty=1,
        reason="partial",
    )
    assert engine.health_status()["last_decision_code"] == "PARTIAL_EXIT_REQUESTED"

    engine._route_core_flatten_request(
        bar_ts=datetime(2026, 4, 27, 14, 10, tzinfo=UTC),
        setup_id=setup.setup_id,
        symbol=setup.symbol,
        reason="bias_flip",
    )
    assert engine.active_setups[setup.setup_id].state is HelixSetupState.CLOSING
    assert engine.health_status()["last_decision_code"] == "FLATTEN_REQUESTED"
