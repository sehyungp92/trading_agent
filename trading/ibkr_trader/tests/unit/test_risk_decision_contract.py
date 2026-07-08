from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

from libs.instrumentation.lineage import LineageContext, lineage_to_payload
from libs.oms.events.bus import EventBus
from libs.oms.models.events import OMSEvent, OMSEventType
from libs.oms.config.risk_config import RiskConfig, StrategyRiskConfig
from libs.oms.intent.handler import IntentHandler
from libs.oms.models.instrument import Instrument
from libs.oms.models.intent import Intent, IntentResult, IntentType
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderType, RiskContext
from libs.oms.models.risk_state import PortfolioRiskState, StrategyRiskState
from libs.oms.risk.calendar import EventCalendar
from libs.oms.risk.gateway import RiskGateway
from strategies.swing.instrumentation.src.context import InstrumentationContext
from strategies.momentum.instrumentation.src.bootstrap import InstrumentationManager as MomentumInstrumentationManager
from strategies.momentum.instrumentation.src.sidecar import Sidecar as MomentumSidecar
from strategies.stock.instrumentation.src.bootstrap import InstrumentationManager
from strategies.stock.instrumentation.src.sidecar import Sidecar


def _lineage() -> LineageContext:
    return LineageContext(
        bot_id="stock_trader",
        strategy_id="IARIC_v1",
        family_id="stock",
        portfolio_id="paper_default",
        strategy_version="IARIC_v1.0.0",
        config_version="cfg_runtime",
        portfolio_config_version="pcfg_runtime",
        risk_config_version="risk_runtime",
        allocation_version="alloc_runtime",
        strategy_registry_version="registry_runtime",
        deployment_id="dep_runtime",
        parameter_set_id="param_runtime",
        code_sha="abc123",
        trace_id="trace_runtime",
    )


def test_event_bus_emits_risk_decision() -> None:
    bus = EventBus()
    queue = bus.subscribe_all()

    bus.emit_risk_decision("IARIC_v1", "oms_1", {"decision": "approve"})

    event = queue.get_nowait()
    assert event.event_type == OMSEventType.RISK_DECISION
    assert event.strategy_id == "IARIC_v1"
    assert event.oms_order_id == "oms_1"
    assert event.payload == {"decision": "approve"}


def test_stock_instrumentation_persists_risk_decision_for_sidecar(tmp_path) -> None:
    manager = object.__new__(InstrumentationManager)
    manager._config = {"data_dir": str(tmp_path)}
    manager.lineage = replace(_lineage(), account_alias="strategy_alias", risk_config_version="risk_strategy")
    oms_lineage = replace(_lineage(), account_alias="oms_alias", risk_config_version="risk_oms")

    manager._handle_risk_decision(
        OMSEvent(
            event_type=OMSEventType.RISK_DECISION,
            timestamp=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
            strategy_id="IARIC_v1",
            oms_order_id="oms_1",
            payload={
                "intent_id": "intent_1",
                    "decision": "scale",
                    "requested_qty": 10,
                    "approved_qty": 5,
                    "lineage": lineage_to_payload(oms_lineage),
                },
            )
        )

    path = next((tmp_path / "risk_decisions").glob("risk_decisions_*.jsonl"))
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["schema_version"] == "risk_decision_v1"
    assert payload["event_type"] == "risk_decision"
    assert payload["scope"] == "oms"
    assert payload["decision"] == "scale"
    assert payload["approved_qty"] == 5
    assert payload["risk_config_version"] == "risk_oms"
    assert payload["account_alias"] == "oms_alias"
    assert "lineage_gaps" not in payload

    files = Sidecar(
        {
            "bot_id": "stock_trader",
            "data_dir": str(tmp_path),
            "sidecar": {"relay_url": "http://relay.local/events"},
        }
    )._get_event_files()
    assert (path, "risk_decision") in files


def test_stock_instrumentation_persists_risk_halt_for_sidecar(tmp_path) -> None:
    manager = object.__new__(InstrumentationManager)
    manager._config = {"data_dir": str(tmp_path)}
    manager.lineage = _lineage()

    manager._handle_risk_halt(
        OMSEvent(
            event_type=OMSEventType.RISK_HALT,
            timestamp=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
            strategy_id="IARIC_v1",
            payload={"reason": "portfolio_daily_stop", "source": "risk_gateway"},
        )
    )

    path = next((tmp_path / "risk_halts").glob("risk_halts_*.jsonl"))
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["schema_version"] == "risk_halt_v1"
    assert payload["event_type"] == "risk_halt"
    assert payload["scope"] == "oms"
    assert payload["reason"] == "portfolio_daily_stop"
    assert payload["halt_scope"] == "strategy"
    assert payload["source"] == "risk_gateway"
    assert payload["risk_config_version"] == "risk_runtime"

    files = Sidecar(
        {
            "bot_id": "stock_trader",
            "data_dir": str(tmp_path),
            "sidecar": {"relay_url": "http://relay.local/events"},
        }
    )._get_event_files()
    assert (path, "risk_halt") in files


def test_momentum_instrumentation_persists_risk_halt_for_sidecar(tmp_path) -> None:
    manager = object.__new__(MomentumInstrumentationManager)
    manager._config = {"data_dir": str(tmp_path)}
    manager.lineage = _lineage()

    manager._handle_risk_halt(
        OMSEvent(
            event_type=OMSEventType.RISK_HALT,
            timestamp=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
            strategy_id="NQ_REGIME",
            payload={"reason": "weekly_stop"},
        )
    )

    path = next((tmp_path / "risk_halts").glob("risk_halts_*.jsonl"))
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["event_type"] == "risk_halt"
    assert payload["reason"] == "weekly_stop"
    assert payload["halt_scope"] == "strategy"
    assert payload["source"] == "oms"

    files = MomentumSidecar(
        {
            "bot_id": "momentum_nq_01",
            "data_dir": str(tmp_path),
            "sidecar": {"relay_url": "http://relay.local/events"},
        }
    )._get_event_files()
    assert (path, "risk_halt") in files


def _order(qty: int = 10) -> OMSOrder:
    return OMSOrder(
        strategy_id="IARIC_v1",
        instrument=Instrument(
            symbol="AAPL",
            root="AAPL",
            venue="SMART",
            tick_size=0.01,
            tick_value=0.01,
            multiplier=1.0,
            point_value=1.0,
        ),
        side=OrderSide.BUY,
        qty=qty,
        order_type=OrderType.LIMIT,
        role=OrderRole.ENTRY,
        risk_context=RiskContext(planned_entry_price=100.0, stop_for_risk=95.0),
    )


async def _strategy_risk() -> StrategyRiskState:
    return StrategyRiskState(
        strategy_id="IARIC_v1",
        trade_date=datetime(2026, 5, 31, tzinfo=timezone.utc).date(),
        daily_realized_R=-0.5,
        weekly_realized_R=-1.0,
        open_risk_R=0.25,
    )


async def _portfolio_risk() -> PortfolioRiskState:
    return PortfolioRiskState(
        trade_date=datetime(2026, 5, 31, tzinfo=timezone.utc).date(),
        daily_realized_R=-1.0,
        weekly_realized_R=-2.0,
        open_risk_R=0.25,
        pending_entry_risk_R=0.1,
    )


async def test_risk_gateway_records_full_decision_context() -> None:
    order = _order()
    order.risk_context.intent_id = "intent_1"
    gateway = RiskGateway(
        config=RiskConfig(
            heat_cap_R=5.0,
            portfolio_daily_stop_R=3.0,
            portfolio_weekly_stop_R=12.0,
            portfolio_urd=100.0,
            strategy_configs={
                "IARIC_v1": StrategyRiskConfig(
                    strategy_id="IARIC_v1",
                    daily_stop_R=2.0,
                    unit_risk_dollars=100.0,
                    max_heat_R=3.0,
                )
            },
        ),
        calendar=EventCalendar(),
        get_strategy_risk=lambda strategy_id: _strategy_risk(),
        get_portfolio_risk=_portfolio_risk,
        family_id="stock",
    )

    denial = await gateway.check_entry(order)

    assert denial is None
    ctx = order.risk_context.gateway_decision_context
    assert ctx["decision"] == "approve"
    assert ctx["daily_stop_usage"]["strategy_stop_R"] == 2.0
    assert ctx["weekly_stop_usage"]["portfolio_stop_R"] == 12.0
    assert ctx["strategy_heat"]["max_heat_R"] == 3.0
    assert ctx["portfolio_heat"]["heat_cap_R"] == 5.0
    assert ctx["requested_qty"] == 10
    assert ctx["approved_qty"] == 10


async def test_risk_gateway_records_portfolio_stop_denial_context() -> None:
    async def _stopped_portfolio_risk() -> PortfolioRiskState:
        return PortfolioRiskState(
            trade_date=datetime(2026, 5, 31, tzinfo=timezone.utc).date(),
            daily_realized_R=-4.0,
            weekly_realized_R=-5.0,
            open_risk_R=0.25,
            pending_entry_risk_R=0.1,
        )

    order = _order()
    order.risk_context.intent_id = "intent_stop"
    gateway = RiskGateway(
        config=RiskConfig(
            heat_cap_R=5.0,
            portfolio_daily_stop_R=3.0,
            portfolio_weekly_stop_R=12.0,
            portfolio_urd=100.0,
            strategy_configs={
                "IARIC_v1": StrategyRiskConfig(
                    strategy_id="IARIC_v1",
                    daily_stop_R=2.0,
                    unit_risk_dollars=100.0,
                    max_heat_R=3.0,
                )
            },
        ),
        calendar=EventCalendar(),
        get_strategy_risk=lambda strategy_id: _strategy_risk(),
        get_portfolio_risk=_stopped_portfolio_risk,
        family_id="stock",
    )

    denial = await gateway.check_entry(order)

    assert denial == "Portfolio daily stop: -4.00R"
    ctx = order.risk_context.gateway_decision_context
    assert ctx["decision"] == "deny"
    assert ctx["gateway_gate"] == "portfolio_daily_stop"
    assert ctx["daily_stop_usage"]["portfolio_realized_R"] == -4.0
    assert ctx["weekly_stop_usage"]["portfolio_realized_R"] == -5.0


async def test_preapproved_gateway_records_full_denial_context() -> None:
    async def _stopped_portfolio_risk() -> PortfolioRiskState:
        return PortfolioRiskState(
            trade_date=datetime(2026, 5, 31, tzinfo=timezone.utc).date(),
            daily_realized_R=-4.0,
            weekly_realized_R=-5.0,
            open_risk_R=0.25,
            pending_entry_risk_R=0.1,
        )

    order = _order()
    order.risk_context.intent_id = "intent_preapproved"
    gateway = RiskGateway(
        config=RiskConfig(
            heat_cap_R=5.0,
            portfolio_daily_stop_R=3.0,
            portfolio_weekly_stop_R=12.0,
            portfolio_urd=100.0,
            strategy_configs={
                "IARIC_v1": StrategyRiskConfig(
                    strategy_id="IARIC_v1",
                    daily_stop_R=2.0,
                    unit_risk_dollars=100.0,
                    max_heat_R=3.0,
                )
            },
        ),
        calendar=EventCalendar(),
        get_strategy_risk=lambda strategy_id: _strategy_risk(),
        get_portfolio_risk=_stopped_portfolio_risk,
        family_id="stock",
    )

    denial = await gateway.check_preapproved_entry(order)

    assert denial == "Portfolio daily stop: -4.00R"
    ctx = order.risk_context.gateway_decision_context
    assert ctx["decision"] == "deny"
    assert ctx["gateway_gate"] == "preapproved_portfolio_daily_stop"
    assert ctx["daily_stop_usage"]["portfolio_stop_R"] == 3.0
    assert ctx["strategy_heat"]["max_heat_R"] == 3.0
    assert ctx["requested_risk_dollars"] == 50.0


async def test_intent_handler_emits_risk_decision_for_pre_gateway_denial() -> None:
    bus = EventBus()
    queue = bus.subscribe_all()
    handler = IntentHandler(
        risk=SimpleNamespace(_family_id="stock"),
        router=SimpleNamespace(),
        repo=SimpleNamespace(),
        bus=bus,
    )
    order = _order(qty=0)

    receipt = await handler.submit(Intent(IntentType.NEW_ORDER, strategy_id="IARIC_v1", order=order))

    assert receipt.result == IntentResult.DENIED
    event = queue.get_nowait()
    assert event.event_type == OMSEventType.RISK_DECISION
    assert event.payload["decision"] == "deny"
    assert event.payload["reason"] == "Order qty must be > 0"
    assert event.payload["risk_decision_ref"].startswith("risk_decision_")


async def test_swing_context_persists_risk_decision_for_sidecar(tmp_path) -> None:
    bus = EventBus()
    ctx = InstrumentationContext(
        data_dir=str(tmp_path),
        lineage=_lineage(),
        oms=SimpleNamespace(stream_all_events=bus.subscribe_all),
    )

    ctx.start()
    bus.emit_risk_decision(
        "TPC",
        "oms_1",
        {"intent_id": "intent_1", "decision": "deny", "reason": "daily_stop"},
    )
    import asyncio

    await asyncio.sleep(0.05)
    await ctx.stop_async()

    path = next((tmp_path / "risk_decisions").glob("risk_decisions_*.jsonl"))
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["event_type"] == "risk_decision"
    assert payload["decision"] == "deny"
    assert payload["reason"] == "daily_stop"


async def test_swing_context_persists_risk_halt_for_sidecar(tmp_path) -> None:
    bus = EventBus()
    ctx = InstrumentationContext(
        data_dir=str(tmp_path),
        lineage=_lineage(),
        oms=SimpleNamespace(stream_all_events=bus.subscribe_all),
    )

    ctx.start()
    bus.emit_risk_halt("TPC", "portfolio_halted")
    import asyncio

    await asyncio.sleep(0.05)
    await ctx.stop_async()

    path = next((tmp_path / "risk_halts").glob("risk_halts_*.jsonl"))
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["event_type"] == "risk_halt"
    assert payload["reason"] == "portfolio_halted"
    assert payload["halt_scope"] == "strategy"
    assert payload["source"] == "oms"
