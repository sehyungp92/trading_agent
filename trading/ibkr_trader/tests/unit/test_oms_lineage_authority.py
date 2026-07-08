from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

from libs.instrumentation.lineage import LineageContext
from libs.oms.events.bus import EventBus
from libs.oms.intent.handler import IntentHandler
from libs.oms.models.events import OMSEventType
from libs.oms.models.instrument import Instrument
from libs.oms.models.intent import Intent, IntentResult, IntentType
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderType, RiskContext
from libs.oms.services.factory import (
    build_multi_strategy_oms,
    _make_fill_lifecycle_writer,
    _make_reconciliation_lifecycle_writer,
)


def _lineage(**overrides) -> LineageContext:
    values = {
        "bot_id": "stock_oms",
        "strategy_id": "IARIC_v1",
        "family_id": "stock",
        "portfolio_id": "paper_default",
        "account_alias": "oms_alias",
        "strategy_version": "IARIC_v1.0.0",
        "config_version": "cfg_oms",
        "portfolio_config_version": "pcfg_oms",
        "risk_config_version": "risk_oms",
        "allocation_version": "alloc_oms",
        "strategy_registry_version": "registry_oms",
        "deployment_id": "dep_oms",
        "parameter_set_id": "param_oms",
        "code_sha": "abc123",
        "trace_id": "trace_oms",
    }
    values.update(overrides)
    return LineageContext(**values)


def _instrument() -> Instrument:
    return Instrument(
        symbol="AAPL",
        root="AAPL",
        venue="SMART",
        tick_size=0.01,
        tick_value=0.01,
        multiplier=1.0,
        point_value=1.0,
    )


def _order() -> OMSOrder:
    return OMSOrder(
        strategy_id="IARIC_v1",
        client_order_id="client_1",
        instrument=_instrument(),
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        role=OrderRole.ENTRY,
        risk_context=RiskContext(
            planned_entry_price=100.0,
            stop_for_risk=95.0,
            risk_dollars=50.0,
            unit_risk_dollars=100.0,
        ),
    )


class _Repo:
    async def get_positions(self, strategy_id: str, symbol: str | None = None):
        return []

    async def get_order_id_by_client_order_id(self, strategy_id: str, client_order_id: str):
        return None

    async def save_order_and_event(self, order: OMSOrder, event_type: str, payload: dict, conn=None):
        return None

    def transaction(self):
        return _Transaction()


class _Transaction:
    async def __aenter__(self):
        return SimpleNamespace()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Risk:
    _family_id = "stock"

    def __init__(self, lineage: LineageContext):
        self._current_oms_lineage = lambda: lineage
        self.seen_trace_id = ""
        self.seen_lineage_context: dict = {}

    async def check_entry(self, order: OMSOrder, *, skip_account_gate: bool = False):
        risk_ctx = order.risk_context
        self.seen_trace_id = risk_ctx.trace_id
        self.seen_lineage_context = dict(risk_ctx.lineage_context)
        risk_ctx.gateway_decision_context = {
            "gateway_gate": "entry",
            "daily_stop_usage": {"strategy_stop_R": 2.0, "strategy_realized_R": -0.5},
            "weekly_stop_usage": {"portfolio_stop_R": 12.0, "portfolio_realized_R": -2.0},
            "strategy_heat": {"max_heat_R": 3.0, "open_risk_R": 0.25},
            "portfolio_heat": {"heat_cap_R": 5.0, "open_risk_R": 0.25},
            "account_gate": {"result": "skipped" if skip_account_gate else "allow"},
            "session_gate": {"result": "allow"},
            "portfolio_rule_refs": ["portfolio_rule_1"],
        }
        return None

    async def check_account_gate(self, order: OMSOrder, conn=None):
        order.risk_context.gateway_decision_context["gateway_gate"] = "account_gate"
        order.risk_context.gateway_decision_context["account_gate"] = {"result": "allow"}
        return None


class _Router:
    async def route(self, order: OMSOrder):
        return None


class _Adapter:
    is_congested = False

    def __init__(self) -> None:
        self.cache = SimpleNamespace(contracts={})

    async def request_open_orders(self):
        return []

    async def request_positions(self):
        return []

    async def request_executions(self):
        return []

    async def submit_order(self, **kwargs):
        return SimpleNamespace(broker_order_id=1, perm_id=1001)


def _read_latest_payload(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])


async def test_risk_decision_uses_current_oms_lineage_and_persisted_correlation() -> None:
    oms_lineage = _lineage(account_alias="oms_alias", risk_config_version="risk_authoritative")
    risk = _Risk(oms_lineage)
    bus = EventBus()
    handler = IntentHandler(risk, _Router(), _Repo(), bus)
    queue = bus.subscribe_all()
    order = _order()

    receipt = await handler.submit(Intent(IntentType.NEW_ORDER, "IARIC_v1", order=order))

    assert receipt.result == IntentResult.ACCEPTED
    payload = queue.get_nowait().payload
    assert payload["event_type"] == "risk_decision"
    assert payload["account_alias"] == "oms_alias"
    assert payload["risk_config_version"] == "risk_authoritative"
    assert payload["lineage"]["account_alias"] == "oms_alias"
    assert payload["lineage"]["risk_config_version"] == "risk_authoritative"
    assert "lineage_gaps" not in payload
    assert order.risk_context.trace_id
    assert risk.seen_trace_id == order.risk_context.trace_id
    assert risk.seen_lineage_context["account_alias"] == "oms_alias"


async def test_shared_swing_multi_strategy_risk_decision_has_per_strategy_lineage(tmp_path) -> None:
    oms, _coordinator = await build_multi_strategy_oms(
        adapter=_Adapter(),
        strategies=[
            {"id": "ATRSS", "unit_risk_dollars": 100.0, "daily_stop_R": 2.0, "priority": 1, "max_heat_R": 5.0},
            {"id": "TPC", "unit_risk_dollars": 100.0, "daily_stop_R": 2.0, "priority": 2, "max_heat_R": 5.0},
        ],
        family_id="swing",
        heat_cap_R=5.0,
        instrumentation_data_dir=str(tmp_path),
        paper_initial_equity=100_000.0,
        strategy_manifests={
            "ATRSS": {"family": "swing", "artifact_config": {"version": "ATRSS.test"}},
            "TPC": {"family": "swing", "artifact_config": {"version": "TPC.test"}},
        },
    )
    await oms.start()
    queue = oms.stream_all_events()
    order = OMSOrder(
        strategy_id="TPC",
        client_order_id="TPC-entry-1",
        instrument=_instrument(),
        side=OrderSide.BUY,
        qty=1,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        role=OrderRole.ENTRY,
        risk_context=RiskContext(
            planned_entry_price=100.0,
            stop_for_risk=99.0,
            signal_id="TPC-setup-1",
            bar_id="QQQ:2026-06-04T14:30:00+00:00",
        ),
    )
    try:
        receipt = await oms.submit_intent(Intent(IntentType.NEW_ORDER, "TPC", order=order))
    finally:
        await oms.stop()

    assert receipt.result == IntentResult.ACCEPTED
    while True:
        event = queue.get_nowait()
        if event.event_type == OMSEventType.RISK_DECISION:
            payload = event.payload
            break
    assert payload["strategy_id"] == "TPC"
    assert payload["strategy_version"] == "TPC.test"
    assert payload["parameter_set_id"]
    assert payload["signal_id"] == "TPC-setup-1"
    assert payload["bar_id"] == "QQQ:2026-06-04T14:30:00+00:00"
    assert payload["lineage"]["strategy_id"] == "TPC"
    assert payload["lineage"]["strategy_version"] == "TPC.test"
    assert "lineage_gaps" not in payload


def test_portfolio_risk_halt_does_not_inherit_strategy_from_oms_lineage() -> None:
    bus = EventBus()
    bus._current_oms_lineage = lambda: _lineage(strategy_id="IARIC_v1")
    queue = bus.subscribe_all()

    bus.emit_risk_halt("", "portfolio_daily_stop")

    event = queue.get_nowait()
    assert event.event_type == OMSEventType.RISK_HALT
    assert event.strategy_id == ""
    assert event.payload["halt_scope"] == "portfolio"
    assert event.payload["strategy_id"] == ""
    assert event.payload["lineage"]["strategy_id"] == ""
    assert event.payload["account_alias"] == "oms_alias"
    assert "lineage_gaps" not in event.payload


def test_lifecycle_writers_resolve_current_oms_lineage_at_write_time(tmp_path) -> None:
    current = {"lineage": _lineage(risk_config_version="risk_start", config_version="cfg_start")}
    fill_writer = _make_fill_lifecycle_writer(str(tmp_path), lambda: current["lineage"])
    recon_writer = _make_reconciliation_lifecycle_writer(str(tmp_path), lambda: current["lineage"])
    current["lineage"] = replace(
        current["lineage"],
        risk_config_version="risk_after",
        config_version="cfg_after",
    )

    position = {
        "portfolio_id": "paper_default",
        "account_alias": "oms_alias",
        "family_id": "stock",
        "strategy_id": "IARIC_v1",
        "symbol": "AAPL",
        "qty": 1,
        "avg_price": 100.0,
        "mark_price": 101.0,
    }
    fill_writer(
        {
            "position": position,
            "positions": [position],
            "fill": {"exec_id": "fill_1", "price": 101.0},
            "order": {"strategy_id": "IARIC_v1", "symbol": "AAPL"},
            "portfolio_risk": {"open_risk_R": 0.25},
            "account_state": {"account_alias": "oms_alias", "equity": 50_000.0},
            "allocation_targets": {},
        }
    )
    current["lineage"] = replace(
        current["lineage"],
        risk_config_version="risk_reconciled",
        config_version="cfg_reconciled",
    )
    recon_writer(
        {
            "lifecycle_action": "allocation_freeze",
            "status": "active",
            "details": {"family_id": "stock"},
        }
    )

    position_payload = _read_latest_payload(next((tmp_path / "positions").glob("positions_*.jsonl")))
    recon_payload = _read_latest_payload(next((tmp_path / "allocation_drift").glob("allocation_drift_*.jsonl")))
    assert position_payload["risk_config_version"] == "risk_after"
    assert position_payload["config_version"] == "cfg_after"
    assert position_payload["account_alias"] == "oms_alias"
    assert recon_payload["risk_config_version"] == "risk_reconciled"
    assert recon_payload["config_version"] == "cfg_reconciled"
    assert recon_payload["account_alias"] == "oms_alias"
