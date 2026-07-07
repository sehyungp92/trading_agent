from __future__ import annotations

from types import SimpleNamespace

from libs.oms.events.bus import EventBus
from libs.oms.intent.handler import IntentHandler
from libs.oms.models.events import OMSEventType
from libs.oms.models.instrument import Instrument
from libs.oms.models.intent import Intent, IntentResult, IntentType
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderType, RiskContext


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


def _order(*, qty: int = 10, role: OrderRole = OrderRole.ENTRY) -> OMSOrder:
    return OMSOrder(
        strategy_id="IARIC_v1",
        instrument=_instrument(),
        side=OrderSide.SELL if role in {OrderRole.EXIT, OrderRole.STOP} else OrderSide.BUY,
        qty=qty,
        order_type=OrderType.LIMIT,
        role=role,
        risk_context=RiskContext(
            planned_entry_price=100.0,
            stop_for_risk=95.0,
            risk_dollars=50.0,
            unit_risk_dollars=100.0,
        ),
    )


class _Repo:
    def __init__(self, *, open_qty: int = 0):
        self.open_qty = open_qty
        self.saved: list[tuple[OMSOrder, str, dict]] = []

    async def get_positions(self, strategy_id: str, symbol: str | None = None):
        return [SimpleNamespace(net_qty=self.open_qty)] if self.open_qty else []

    async def get_order_id_by_client_order_id(self, strategy_id: str, client_order_id: str):
        return None

    async def save_order_and_event(self, order: OMSOrder, event_type: str, payload: dict, conn=None):
        self.saved.append((order, event_type, payload))

    def transaction(self):
        return _Transaction()


class _Transaction:
    async def __aenter__(self):
        return SimpleNamespace()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Risk:
    _family_id = "stock"

    def __init__(self, *, entry_denial: str | None = None, account_denial: str | None = None):
        self.entry_denial = entry_denial
        self.account_denial = account_denial

    async def check_entry(self, order: OMSOrder, *, skip_account_gate: bool = False):
        order.risk_context.gateway_decision_context = {
            "gateway_gate": "entry",
            "daily_stop_usage": {"strategy_stop_R": 2.0, "strategy_realized_R": -0.5},
            "weekly_stop_usage": {"portfolio_stop_R": 12.0, "portfolio_realized_R": -2.0},
            "strategy_heat": {"max_heat_R": 3.0, "open_risk_R": 0.25},
            "portfolio_heat": {"heat_cap_R": 5.0, "open_risk_R": 0.25},
            "account_gate": {"result": "skipped" if skip_account_gate else "allow"},
            "session_gate": {"result": "allow"},
            "portfolio_rule_refs": ["portfolio_rule_1"],
        }
        return self.entry_denial

    async def check_preapproved_entry(self, order: OMSOrder):
        return await self.check_entry(order)

    async def check_account_gate(self, order: OMSOrder, conn=None):
        order.risk_context.gateway_decision_context["gateway_gate"] = "account_gate"
        order.risk_context.gateway_decision_context["account_gate"] = {
            "result": "deny" if self.account_denial else "allow",
            "reason": self.account_denial or "",
        }
        return self.account_denial


class _Router:
    def __init__(self):
        self.routed: list[OMSOrder] = []

    async def route(self, order: OMSOrder):
        self.routed.append(order)


def _handler(*, risk=None, repo=None, router=None, bus=None) -> tuple[IntentHandler, EventBus]:
    event_bus = bus or EventBus()
    handler = IntentHandler(
        risk=risk or _Risk(),
        router=router or _Router(),
        repo=repo or _Repo(),
        bus=event_bus,
    )
    return handler, event_bus


def _next_risk_decision(queue):
    event = queue.get_nowait()
    assert event.event_type == OMSEventType.RISK_DECISION
    return event.payload


async def test_unknown_intent_type_emits_risk_decision() -> None:
    handler, bus = _handler(risk=SimpleNamespace(_family_id="stock"))
    queue = bus.subscribe_all()
    intent = SimpleNamespace(intent_type="BOGUS", strategy_id="IARIC_v1", order=None)

    receipt = await handler.submit(intent)

    assert receipt.result == IntentResult.DENIED
    payload = _next_risk_decision(queue)
    assert payload["decision"] == "deny"
    assert payload["reason"] == "Unknown intent type"
    assert payload["family_id"] == "stock"


async def test_missing_order_emits_risk_decision() -> None:
    handler, bus = _handler(risk=SimpleNamespace(_family_id="stock"))
    queue = bus.subscribe_all()

    receipt = await handler.submit(Intent(IntentType.NEW_ORDER, strategy_id="IARIC_v1"))

    assert receipt.result == IntentResult.DENIED
    payload = _next_risk_decision(queue)
    assert payload["decision"] == "deny"
    assert payload["reason"] == "No order in intent"
    assert payload["requested_qty"] == 0


async def test_exit_qty_exceeds_position_emits_risk_decision_before_gateway() -> None:
    handler, bus = _handler(repo=_Repo(open_qty=5))
    queue = bus.subscribe_all()
    order = _order(qty=10, role=OrderRole.EXIT)

    receipt = await handler.submit(Intent(IntentType.NEW_ORDER, strategy_id="IARIC_v1", order=order))

    assert receipt.result == IntentResult.DENIED
    payload = _next_risk_decision(queue)
    assert payload["decision"] == "deny"
    assert payload["reason"] == "Exit qty 10 exceeds open position 5"
    assert payload["requested_qty"] == 10
    assert payload["approved_qty"] == 0


async def test_account_gate_denial_preserves_gateway_context_in_risk_decision() -> None:
    repo = _Repo()
    risk = _Risk(account_denial="Account gate: gross cap")
    handler, bus = _handler(risk=risk, repo=repo)
    queue = bus.subscribe_all()
    order = _order()

    receipt = await handler.submit(Intent(IntentType.NEW_ORDER, strategy_id="IARIC_v1", order=order))

    assert receipt.result == IntentResult.DENIED
    payload = _next_risk_decision(queue)
    assert payload["decision"] == "deny"
    assert payload["reason"] == "Account gate: gross cap"
    assert payload["gateway_gate"] == "account_gate"
    assert payload["account_gate"]["result"] == "deny"
    assert payload["session_gate"]["result"] == "allow"
    assert payload["daily_stop_usage"]["strategy_stop_R"] == 2.0
    assert payload["weekly_stop_usage"]["portfolio_stop_R"] == 12.0
    assert payload["portfolio_rule_refs"] == ["portfolio_rule_1"]
    assert repo.saved[0][1] == "RISK_DENIED"


async def test_non_entry_approval_emits_risk_decision_and_routes() -> None:
    repo = _Repo(open_qty=5)
    router = _Router()
    handler, bus = _handler(repo=repo, router=router)
    queue = bus.subscribe_all()
    order = _order(qty=5, role=OrderRole.EXIT)

    receipt = await handler.submit(Intent(IntentType.NEW_ORDER, strategy_id="IARIC_v1", order=order))

    assert receipt.result == IntentResult.ACCEPTED
    payload = _next_risk_decision(queue)
    assert payload["decision"] == "approve"
    assert payload["role"] == "EXIT"
    assert payload["approved_qty"] == 5
    assert payload["strategy_heat"]["max_heat_R"] == 3.0
    assert len(router.routed) == 1
