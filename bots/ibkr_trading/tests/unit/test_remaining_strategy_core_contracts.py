from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from backtests.shared.parity.decision_capture import normalize_decision_stream
from strategies.core.actions import SubmitEntry, SubmitExit, SubmitProtectiveStop
from strategies.momentum.vdub.core import logic as vdub_logic
from strategies.momentum.vdub.core.state import (
    VdubCoreState,
    VdubEntryFillContext,
    VdubEntrySubmitted,
    VdubFill,
    VdubOrderUpdate,
)
from strategies.momentum.vdub.models import Direction, EntryType, SessionWindow, WorkingEntry
from strategies.stock.iaric.core import logic as iaric_logic
from strategies.stock.iaric.core.state import IARICBarInput, IARICFill, IARICOrderUpdate
from strategies.stock.iaric.models import IntradayStateSnapshot
from strategies.swing.akc_helix.core import logic as akc_helix_logic
from strategies.swing.akc_helix.core.state import (
    AKCHelixBarInput,
    AKCHelixCoreState,
    AKCHelixEntryRequest,
    AKCHelixFill,
    AKCHelixOrderUpdate,
)
from strategies.swing.akc_helix.models import (
    Direction as AKCHelixDirection,
    SetupClass as AKCHelixSetupClass,
    SetupInstance as AKCHelixSetup,
    SetupState as AKCHelixSetupState,
)
from strategies.swing.atrss.core import logic as atrss_logic
from strategies.swing.atrss.core.state import ATRSSBarInput, ATRSSCoreState, ATRSSEntryRequest, ATRSSFill, ATRSSOrderUpdate
from strategies.swing.atrss.models import Candidate, CandidateType, Direction as ATRSSDirection, HourlyState

UTC = timezone.utc


def _iaric_state() -> IntradayStateSnapshot:
    return IntradayStateSnapshot(
        trade_date=date(2026, 4, 25),
        saved_at=datetime(2026, 4, 25, 15, 30, tzinfo=UTC),
        symbols=[],
        last_decision_code="IDLE",
    )


def _last_bar_marker(state) -> str | datetime | None:
    if hasattr(state, "last_bar_ts"):
        return state.last_bar_ts
    if hasattr(state, "meta") and isinstance(state.meta, dict):
        return state.meta.get("last_bar_ts")
    return None


def _atrss_candidate() -> Candidate:
    return Candidate(
        symbol="QQQ",
        type=CandidateType.PULLBACK,
        direction=ATRSSDirection.LONG,
        trigger_price=510.25,
        initial_stop=503.5,
        qty=3,
        signal_bar=HourlyState(time=datetime(2026, 4, 25, 13, 0, tzinfo=UTC)),
    )


def _akc_helix_setup() -> AKCHelixSetup:
    return AKCHelixSetup(
        setup_id="HELIX-1",
        symbol="QQQ",
        setup_class=AKCHelixSetupClass.CLASS_A,
        direction=AKCHelixDirection.LONG,
        origin_tf="4H",
        state=AKCHelixSetupState.NEW,
        created_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
        bos_level=505.5,
        stop0=499.0,
        current_stop=499.0,
        qty_planned=3,
        oca_group="HELIX-OCA",
    )


def test_atrss_core_realistic_flow_preserves_shared_lifecycle_invariants() -> None:
    state, actions, events = atrss_logic.on_bar(
        ATRSSCoreState(),
        bar_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
        entry_request=ATRSSEntryRequest(
            client_order_id="ENTRY-1",
            symbol="QQQ",
            candidate=_atrss_candidate(),
            limit_price=510.75,
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitEntry)
    assert events[0].code == "ENTRY_REQUESTED"
    bar_marker = _last_bar_marker(state)

    state.pending_orders["ENTRY-1"] = {
        "symbol": "QQQ",
        "type": CandidateType.PULLBACK,
        "direction": ATRSSDirection.LONG,
        "trigger_price": 510.25,
        "initial_stop": 503.5,
        "qty": 3,
    }
    state, actions, events = atrss_logic.on_fill(
        state,
        ATRSSFill(
            oms_order_id="ENTRY-1",
            fill_price=510.5,
            fill_qty=3,
            fill_time=datetime(2026, 4, 25, 13, 5, tzinfo=UTC),
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitProtectiveStop)
    assert events[0].code == "ENTRY_FILLED"
    assert _last_bar_marker(state) == bar_marker


def test_akc_helix_core_realistic_flow_preserves_shared_lifecycle_invariants() -> None:
    state, actions, events = akc_helix_logic.on_bar(
        AKCHelixCoreState(),
        bar_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
        entry_request=AKCHelixEntryRequest(
            client_order_id="ENTRY-1",
            setup=_akc_helix_setup(),
            order_type="STOP_LIMIT",
            limit_price=505.75,
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitEntry)
    assert events[0].code == "ENTRY_REQUESTED"
    bar_marker = _last_bar_marker(state)

    state.order_to_setup["ENTRY-1"] = "HELIX-1"
    state, actions, events = akc_helix_logic.on_fill(
        state,
        AKCHelixFill(
            oms_order_id="ENTRY-1",
            fill_price=505.75,
            fill_qty=3,
            fill_time=datetime(2026, 4, 25, 13, 5, tzinfo=UTC),
            order_role="entry",
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitProtectiveStop)
    assert events[0].code == "ENTRY_FILLED"
    assert _last_bar_marker(state) == bar_marker


@pytest.mark.parametrize(
    ("state", "logic", "bar_payload", "order_payload", "fill_payload"),
    [
        (
            ATRSSCoreState(),
            atrss_logic,
            ATRSSBarInput(
                symbol="QQQ",
                timeframe="1h",
                bar_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
                decision_code="ATRSS_BAR",
                decision_details={"symbol": "QQQ"},
            ),
            ATRSSOrderUpdate(
                oms_order_id="OMS-1",
                symbol="QQQ",
                timeframe="1h",
                timestamp=datetime(2026, 4, 25, 13, 1, tzinfo=UTC),
                decision_code="ATRSS_ORDER",
            ),
            ATRSSFill(
                oms_order_id="OMS-1",
                symbol="QQQ",
                timeframe="1h",
                fill_time=datetime(2026, 4, 25, 13, 2, tzinfo=UTC),
                decision_code="ATRSS_FILL",
            ),
        ),
        (
            AKCHelixCoreState(),
            akc_helix_logic,
            AKCHelixBarInput(
                symbol="CL",
                timeframe="1h",
                bar_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
                decision_code="HELIX_BAR",
            ),
            AKCHelixOrderUpdate(
                oms_order_id="OMS-1",
                symbol="CL",
                timeframe="1h",
                timestamp=datetime(2026, 4, 25, 13, 1, tzinfo=UTC),
                decision_code="HELIX_ORDER",
            ),
            AKCHelixFill(
                oms_order_id="OMS-1",
                symbol="CL",
                timeframe="1h",
                fill_time=datetime(2026, 4, 25, 13, 2, tzinfo=UTC),
                decision_code="HELIX_FILL",
            ),
        ),
        # These strategies now have real shared cores; keep a generic lifecycle
        # contract smoke test here alongside their dedicated core suites.
        (
            _iaric_state(),
            iaric_logic,
            IARICBarInput(
                symbol="MSFT",
                timeframe="5m",
                bar_ts=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
                decision_code="IARIC_BAR",
            ),
            IARICOrderUpdate(
                oms_order_id="OMS-1",
                symbol="MSFT",
                timeframe="5m",
                timestamp=datetime(2026, 4, 25, 13, 1, tzinfo=UTC),
                decision_code="IARIC_ORDER",
            ),
            IARICFill(
                oms_order_id="OMS-1",
                symbol="MSFT",
                timeframe="5m",
                fill_time=datetime(2026, 4, 25, 13, 2, tzinfo=UTC),
                decision_code="IARIC_FILL",
            ),
        ),
    ],
)
def test_remaining_strategy_cores_preserve_shared_lifecycle_contract(
    state,
    logic,
    bar_payload,
    order_payload,
    fill_payload,
) -> None:
    state, actions, events = logic.on_bar(state, bar_payload)
    assert actions == []
    assert normalize_decision_stream(events)[0]["code"].endswith("_BAR")
    assert state.last_decision_code.endswith("_BAR")
    bar_marker = _last_bar_marker(state)
    assert bar_marker is not None

    state, actions, events = logic.on_order_update(state, order_payload)
    assert actions == []
    assert normalize_decision_stream(events)[0]["code"].endswith("_ORDER")
    assert state.last_decision_code.endswith("_ORDER")
    assert _last_bar_marker(state) == bar_marker

    state, actions, events = logic.on_fill(state, fill_payload)
    assert actions == []
    assert normalize_decision_stream(events)[0]["code"].endswith("_FILL")
    assert state.last_decision_code.endswith("_FILL")
    assert _last_bar_marker(state) == bar_marker



# ── Vdub real core tests ─────────────────────────────────────────


def _make_working_entry(**overrides) -> WorkingEntry:
    """Minimal WorkingEntry for testing Vdub core logic."""
    defaults = dict(
        oms_order_id="WE-001",
        entry_type=EntryType.TYPE_A,
        direction=Direction.LONG,
        stop_entry=20010.0,
        limit_entry=0.0,
        qty=2,
        initial_stop=19980.0,
        session=SessionWindow.RTH,
    )
    defaults.update(overrides)
    return WorkingEntry(**defaults)


def test_vdub_on_bar_entry_submitted_registers_working_entry():
    """Entry submitted via on_bar registers in working_entries."""
    state = VdubCoreState()
    bar_ts = datetime(2026, 4, 25, 14, 0, tzinfo=UTC)
    we = _make_working_entry()

    submitted = VdubEntrySubmitted(
        working_entry=we, oms_order_id="OMS-V1", bar_idx=42,
    )

    next_state, actions, events = vdub_logic.on_bar(
        state, bar_ts=bar_ts, entry_submitted=submitted,
    )

    assert "OMS-V1" in next_state.working_entries
    assert next_state.working_entries["OMS-V1"].direction == Direction.LONG
    assert len(events) == 1
    assert events[0].code == "ENTRY_SUBMITTED"
    assert next_state.last_bar_ts == bar_ts


def test_vdub_on_fill_entry_creates_position_and_emits_stop():
    """Entry fill creates position and requests protective stop."""
    we = _make_working_entry(oms_order_id="OMS-V1")
    state = VdubCoreState(
        working_entries={"OMS-V1": we},
    )

    fill = VdubFill(
        oms_order_id="OMS-V1",
        fill_price=20010.0,
        fill_qty=2,
        fill_time=datetime(2026, 4, 25, 14, 1, tzinfo=UTC),
        point_value=2.0,
        entry_context=VdubEntryFillContext(working_entry=we),
    )

    next_state, actions, events = vdub_logic.on_fill(state, fill)

    # Position created
    assert len(next_state.positions) == 1
    assert next_state.positions[0].entry_price == 20010.0
    # Working entry consumed
    assert "OMS-V1" not in next_state.working_entries
    # SubmitExit for protective stop
    assert len(actions) == 1
    assert isinstance(actions[0], SubmitExit)
    assert actions[0].order_type == "STOP"
    # Event
    assert events[0].code == "ENTRY_FILLED"


def test_vdub_on_order_update_terminal_removes_working_entry():
    """Terminal order update for entry removes working entry."""
    we = _make_working_entry(oms_order_id="OMS-V2")
    state = VdubCoreState(
        working_entries={"OMS-V2": we},
    )

    update = VdubOrderUpdate(
        oms_order_id="OMS-V2",
        status="cancelled",
        timestamp=datetime(2026, 4, 25, 14, 5, tzinfo=UTC),
    )

    next_state, actions, events = vdub_logic.on_order_update(state, update)

    assert "OMS-V2" not in next_state.working_entries
    assert events[0].code == "ENTRY_CANCELLED"
