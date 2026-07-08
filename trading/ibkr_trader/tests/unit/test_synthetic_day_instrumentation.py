from __future__ import annotations

import json
import sys
import gzip
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from libs.broker_ibkr.models.types import PositionSnapshot
from libs.instrumentation.event_contract import (
    append_jsonl_event,
    enrich_payload,
    write_risk_halt_event,
    write_startup_events,
)
from libs.instrumentation.lineage import LineageContext
from libs.oms.instrumentation.portfolio_rule_event import build_portfolio_rule_event
from libs.oms.instrumentation.runtime_refs import fill_runtime_refs
from libs.oms.models.events import OMSEventType
from libs.oms.models.instrument import Instrument
from libs.oms.models.intent import Intent, IntentResult, IntentType
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderType, RiskContext
from libs.oms.persistence.in_memory import InMemoryRepository
from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
from libs.oms.services.factory import build_multi_strategy_oms, build_oms_service
from strategies.core.actions import SubmitEntry
from strategies.momentum.nq_regime.engine import NQRegimeEngine
from strategies.swing.tpc.engine import TPCEngine
from strategies.stock.instrumentation.src import sidecar as stock_sidecar_module
from strategies.stock.instrumentation.src.bootstrap import InstrumentationManager
from strategies.stock.instrumentation.src.facade import InstrumentationKit
from strategies.stock.instrumentation.src.sidecar import Sidecar
from strategies.stock.coordinator import StockFamilyCoordinator


def _lineage() -> LineageContext:
    return LineageContext(
        bot_id="stock_trader",
        strategy_id="IARIC_v1",
        family_id="stock",
        portfolio_id="paper_default",
        account_alias="paper",
        strategy_version="IARIC_v1.0.0",
        config_version="cfg_synth",
        portfolio_config_version="pcfg_synth",
        risk_config_version="risk_synth",
        allocation_version="alloc_synth",
        strategy_registry_version="registry_synth",
        deployment_id="dep_synth",
        parameter_set_id="param_synth",
        code_sha="abc123",
        trace_id="trace_synth",
    )


def _payload(event: dict) -> dict:
    return json.loads(event["payload"])


def _family_lineage(family: str, strategy_id: str) -> LineageContext:
    return LineageContext(
        bot_id=f"{family}_trader" if family != "momentum" else "momentum_nq_01",
        strategy_id=strategy_id,
        family_id=family,
        portfolio_id="paper_default",
        account_alias="paper",
        strategy_version=f"{strategy_id}.synthetic",
        config_version=f"cfg_{family}_synth",
        portfolio_config_version=f"pcfg_{family}_synth",
        risk_config_version=f"risk_{family}_synth",
        allocation_version=f"alloc_{family}_synth",
        strategy_registry_version=f"registry_{family}_synth",
        deployment_id=f"dep_{family}_synth",
        parameter_set_id=f"param_{family}_synth",
        code_sha="abc123",
        trace_id=f"trace_{family}_synth",
    )


def _instrument(symbol: str = "QQQ") -> Instrument:
    return Instrument(
        symbol=symbol,
        root=symbol,
        venue="SMART",
        tick_size=0.01,
        tick_value=0.01,
        multiplier=1.0,
        point_value=1.0,
        currency="USD",
        primary_exchange="NASDAQ",
        sec_type="STK",
    )


async def _next_event(queue, event_type: OMSEventType):
    while True:
        event = await queue.get()
        if event.event_type == event_type:
            return event


class FakeAdapter:
    is_congested = False

    def __init__(self) -> None:
        self.cache = SimpleNamespace(contracts={101: SimpleNamespace(symbol="QQQ")})
        self.positions: list[PositionSnapshot] = []
        self.submitted: list[dict] = []

    async def request_open_orders(self) -> list:
        return []

    async def request_positions(self) -> list[PositionSnapshot]:
        return list(self.positions)

    async def request_executions(self) -> list:
        return []

    async def submit_order(self, **kwargs):
        self.submitted.append(dict(kwargs))
        idx = len(self.submitted)
        return SimpleNamespace(broker_order_id=idx, perm_id=10_000 + idx)

    async def cancel_order(self, *_args, **_kwargs) -> None:
        return None

    async def replace_order(self, *_args, **_kwargs) -> None:
        return None


class SyntheticStrategy:
    def __init__(self, oms, kit: InstrumentationKit) -> None:
        self.oms = oms
        self.kit = kit
        self.queue = oms.stream_events("IARIC_v1")
        self.trade_id = "trade_1"
        self.entry_decision_ref = "decision_1"

    async def submit_entry(self):
        order = OMSOrder(
            oms_order_id="order_entry",
            client_order_id="client_entry",
            strategy_id="IARIC_v1",
            account_id="paper",
            instrument=_instrument("QQQ"),
            side=OrderSide.BUY,
            qty=10,
            order_type=OrderType.LIMIT,
            limit_price=500.0,
            role=OrderRole.ENTRY,
            risk_context=RiskContext(
                stop_for_risk=450.0,
                planned_entry_price=500.0,
                risk_dollars=500.0,
                unit_risk_dollars=500.0,
            ),
        )
        return await self.oms.submit_intent(Intent(IntentType.NEW_ORDER, "IARIC_v1", order=order))

    async def submit_exit(self):
        order = OMSOrder(
            oms_order_id="order_exit",
            client_order_id="client_exit",
            strategy_id="IARIC_v1",
            account_id="paper",
            instrument=_instrument("QQQ"),
            side=OrderSide.SELL,
            qty=5,
            order_type=OrderType.MARKET,
            role=OrderRole.EXIT,
        )
        return await self.oms.submit_intent(Intent(IntentType.NEW_ORDER, "IARIC_v1", order=order))

    async def submit_denied_entry(self):
        order = OMSOrder(
            oms_order_id="order_denied",
            client_order_id="client_denied",
            strategy_id="IARIC_v1",
            account_id="paper",
            instrument=_instrument("QQQ"),
            side=OrderSide.BUY,
            qty=0,
            order_type=OrderType.LIMIT,
            limit_price=500.0,
            role=OrderRole.ENTRY,
            risk_context=RiskContext(
                stop_for_risk=450.0,
                planned_entry_price=500.0,
                risk_dollars=50.0,
                unit_risk_dollars=500.0,
            ),
        )
        return await self.oms.submit_intent(Intent(IntentType.NEW_ORDER, "IARIC_v1", order=order))

    async def log_next_fill(self, *, is_exit: bool = False):
        fill_event = await _next_event(self.queue, OMSEventType.FILL)
        payload = fill_event.payload or {}
        if is_exit:
            self.kit.log_exit(
                trade_id=self.trade_id,
                exit_price=float(payload["price"]),
                exit_reason="TARGET",
                fees_paid=float(payload.get("commission", 0.0) or 0.0),
                **fill_runtime_refs(
                    fill_event.oms_order_id or "",
                    payload,
                    fill_qty=payload.get("qty"),
                    is_exit=True,
                ),
            )
        else:
            self.kit.log_entry(
                trade_id=self.trade_id,
                pair="QQQ",
                side="LONG",
                entry_price=float(payload["price"]),
                position_size=float(payload["qty"]),
                position_size_quote=float(payload["price"]) * float(payload["qty"]),
                entry_signal="synthetic_breakout",
                entry_signal_id=self.entry_decision_ref,
                entry_signal_strength=0.9,
                strategy_params={"lookback": 20},
                sizing_inputs={"risk_R": 0.5},
                portfolio_state={"open_risk_R": 0.5},
                decision_ref=self.entry_decision_ref,
                **fill_runtime_refs(fill_event.oms_order_id or "", payload, fill_qty=payload.get("qty")),
            )
        return fill_event


async def run_synthetic_day_instrumentation_chain(tmp_path, monkeypatch) -> list[dict]:
    data_dir = tmp_path
    monkeypatch.setenv("INSTRUMENTATION_DATA_DIR", str(data_dir))
    lineage = _lineage()
    allocation_targets = {
        "families": {"stock": 1.0},
        "strategies": {"IARIC_v1": 1.0},
        "source": "synthetic",
    }

    adapter = FakeAdapter()
    repo = InMemoryRepository()
    oms = await build_oms_service(
        adapter=adapter,
        strategy_id="IARIC_v1",
        unit_risk_dollars=500.0,
        family_id="stock",
        family_strategy_ids=["IARIC_v1"],
        instrumentation_data_dir=str(data_dir),
        repository=repo,
        allocation_targets=allocation_targets,
        portfolio_rules_config=PortfolioRulesConfig(
            family_strategy_ids=("IARIC_v1",),
            strategy_size_multipliers=(("IARIC_v1", 0.5),),
            nqdtc_direction_filter_enabled=False,
            initial_equity=100_000.0,
        ),
        get_current_equity=lambda: 100_000.0,
        paper_initial_equity=100_000.0,
    )
    manager = InstrumentationManager(
        oms,
        "IARIC_v1",
        strategy_type="iaric",
        family_strategy_ids=["IARIC_v1"],
        write_daily_closeout_on_stop=False,
        stop_sidecar_on_stop=False,
    )
    manager.lineage = lineage
    manager.trade_logger._lineage = lineage
    manager.regime_classifier = SimpleNamespace(current_regime=lambda _pair: "synthetic")
    manager.sidecar.start = lambda: None
    manager.sidecar.stop = lambda: None

    await oms.start()
    await manager.start()
    kit = InstrumentationKit(manager, strategy_type="iaric")
    strategy = SyntheticStrategy(oms, kit)

    try:
        entry_receipt = await strategy.submit_entry()
        assert entry_receipt.result == IntentResult.ACCEPTED
        assert entry_receipt.oms_order_id
        entry_result = adapter.on_fill(
            entry_receipt.oms_order_id,
            "fill_exec_1",
            500.0,
            5,
            datetime.now(timezone.utc),
            1.25,
        )
        if entry_result is not None and hasattr(entry_result, "__await__"):
            assert await entry_result is True
        await strategy.log_next_fill()

        exit_receipt = await strategy.submit_exit()
        assert exit_receipt.result == IntentResult.ACCEPTED
        assert exit_receipt.oms_order_id
        exit_result = adapter.on_fill(
            exit_receipt.oms_order_id,
            "fill_exec_2",
            512.5,
            5,
            datetime.now(timezone.utc),
            1.25,
        )
        if exit_result is not None and hasattr(exit_result, "__await__"):
            assert await exit_result is True
        await strategy.log_next_fill(is_exit=True)

        denied = await strategy.submit_denied_entry()
        assert denied.result == IntentResult.DENIED

        oms.event_bus.emit_risk_halt("IARIC_v1", "synthetic_halt")

        base_rules = PortfolioRulesConfig(
            family_strategy_ids=("IARIC_v1",),
            directional_cap_R=3.0,
        )
        updated_rules = PortfolioRulesConfig(
            family_strategy_ids=("IARIC_v1",),
            directional_cap_R=4.0,
        )
        coordinator = object.__new__(StockFamilyCoordinator)
        coordinator._instrumentations = [manager]
        coordinator._portfolio_checkers = [SimpleNamespace(_cfg=updated_rules)]
        coordinator._base_portfolio_rules = base_rules
        coordinator._regime_adjusted_rules = updated_rules
        coordinator._write_coordination_event(
            "synthetic_regime_update",
            {
                "family": "stock",
                "regime": "SYNTHETIC",
                "rules_applied": {"directional_cap_R": updated_rules.directional_cap_R},
            },
        )
        import asyncio

        await asyncio.sleep(0.05)

        adapter.positions = [
            PositionSnapshot(account="paper", con_id=101, symbol="QQQ", qty=1, avg_cost=512.5)
        ]
        await oms.request_reconciliation()
        await manager.write_daily_closeout(
            oms_services=[oms],
            strategy_ids=["IARIC_v1"],
            family_id="stock",
        )

        kit.log_missed(
            pair="QQQ",
            side="LONG",
            signal="synthetic_block",
            signal_id="miss_1",
            signal_strength=0.2,
            blocked_by="heat_cap",
            block_reason="synthetic heat cap",
            filter_decisions=[{
                "filter_name": "spread",
                "threshold": 1.0,
                "actual_value": 3.0,
                "passed": False,
            }],
        )
        kit.emit_heartbeat(active_positions=1, open_orders=0, uptime_s=60.0, error_count_1h=0)
        kit.log_error(
            error_type="synthetic_error",
            message="synthetic error",
            severity="warning",
            category="test",
        )

        captured: list[dict] = []

        class FakeResponse:
            status_code = 200

        class FakeRequests:
            def post(self, _url, data, headers, timeout):
                captured.append(json.loads(data.decode("utf-8")))
                return FakeResponse()

        fake_requests = FakeRequests()
        monkeypatch.setattr(stock_sidecar_module, "requests", fake_requests)
        alias = sys.modules.get("instrumentation.src.sidecar")
        if alias is not None:
            monkeypatch.setattr(alias, "requests", fake_requests, raising=False)
        sidecar = Sidecar(
            {
                "bot_id": "stock_trader",
                "data_dir": str(data_dir),
                "sidecar": {
                    "relay_url": "http://relay.local",
                    "batch_size": 100,
                    "retry_max": 1,
                    "retry_backoff_base_seconds": 0,
                },
            }
        )
        sidecar.run_once()
        sent = [event for envelope in captured for event in envelope["events"]]
        assert sidecar.get_diagnostics()["total_forwarded"] == len(sent)
        return sent
    finally:
        await manager.stop()
        await oms.stop()


async def _run_family_oms_submit(
    *,
    family: str,
    data_dir,
) -> dict:
    if family == "momentum":
        strategy_id = "NQ_REGIME"
        adapter = FakeAdapter()
        oms = await build_oms_service(
            adapter=adapter,
            strategy_id=strategy_id,
            unit_risk_dollars=500.0,
            family_id=family,
            instrumentation_data_dir=str(data_dir),
            repository=InMemoryRepository(),
            paper_initial_equity=100_000.0,
            portfolio_id="paper_default",
            account_alias="paper",
            strategy_manifest={"family": family, "artifact_config": {"version": "NQ_REGIME.synthetic"}},
        )
        engine = NQRegimeEngine(oms_service=oms, state_dir=data_dir)
        action = SubmitEntry(
            client_order_id="NQ_REGIME-synth-entry-1",
            symbol="MNQ",
            side="BUY",
            qty=1,
            order_type="LIMIT",
            limit_price=20_000.0,
            stop_price=19_980.0,
            risk_context={"stop_for_risk": 19_980.0, "planned_entry_price": 20_000.0},
            metadata={
                "module": "synthetic",
                "candidate_id": "nqreg-synth-202606041430-BUY",
                "signal_ts": "2026-06-04T14:30:00+00:00",
            },
        )
        order = engine._order_from_entry(action)
    else:
        strategy_id = "TPC"
        adapter = FakeAdapter()
        oms, _coord = await build_multi_strategy_oms(
            adapter=adapter,
            strategies=[
                {"id": "ATRSS", "unit_risk_dollars": 500.0, "daily_stop_R": 2.0, "priority": 1, "max_heat_R": 5.0},
                {"id": strategy_id, "unit_risk_dollars": 500.0, "daily_stop_R": 2.0, "priority": 2, "max_heat_R": 5.0},
            ],
            family_id=family,
            heat_cap_R=5.0,
            instrumentation_data_dir=str(data_dir),
            repository=InMemoryRepository(),
            paper_initial_equity=100_000.0,
            portfolio_id="paper_default",
            account_alias="paper",
            strategy_manifests={
                "ATRSS": {"family": family, "artifact_config": {"version": "ATRSS.synthetic"}},
                strategy_id: {"family": family, "artifact_config": {"version": "TPC.synthetic"}},
            },
        )
        tpc = TPCEngine(
            ib_session=object(),
            oms_service=oms,
            instruments={"QQQ": _instrument("QQQ")},
            config={},
            kit=SimpleNamespace(active=False),
            equity=100_000.0,
        )
        action = SubmitEntry(
            client_order_id="TPC-synth-entry-1",
            symbol="QQQ",
            side="BUY",
            qty=1,
            order_type="LIMIT",
            limit_price=500.0,
            stop_price=495.0,
            risk_context={
                "stop_for_risk": 495.0,
                "planned_entry_price": 500.0,
                "signal_id": "TPC-synth-setup-1",
                "bar_id": "QQQ:2026-06-04T14:30:00+00:00",
                "exchange_timestamp": "2026-06-04T14:30:00+00:00",
            },
            metadata={"setup_id": "TPC-synth-setup-1"},
        )
        order = OMSOrder(
            oms_order_id="swing_synth_order",
            client_order_id=action.client_order_id,
            strategy_id=strategy_id,
            instrument=_instrument("QQQ"),
            side=OrderSide.BUY,
            qty=1,
            order_type=OrderType.LIMIT,
            limit_price=500.0,
            role=OrderRole.ENTRY,
            risk_context=tpc._build_risk_context(action, _instrument("QQQ")),
        )

    await oms.start()
    queue = oms.stream_all_events()
    try:
        receipt = await oms.submit_intent(Intent(IntentType.NEW_ORDER, strategy_id, order=order))
        assert receipt.result == IntentResult.ACCEPTED
    finally:
        await oms.stop()

    while True:
        event = queue.get_nowait()
        if event.event_type == OMSEventType.RISK_DECISION:
            return dict(event.payload or {})


async def run_family_synthetic_instrumentation_chain(tmp_path, monkeypatch, family: str) -> list[dict]:
    if family == "stock":
        return await run_synthetic_day_instrumentation_chain(tmp_path, monkeypatch)
    data_dir = tmp_path
    monkeypatch.setenv("INSTRUMENTATION_DATA_DIR", str(data_dir))
    strategy_id = "NQ_REGIME" if family == "momentum" else "TPC"
    lineage = _family_lineage(family, strategy_id)

    write_startup_events(
        data_dir,
        lineage,
        effective_config={"family": family, "synthetic": True},
        allocation_state={"targets": {"strategies": {strategy_id: 1.0}}, "raw_nav": 100_000.0},
        portfolio_state={"equity": 100_000.0, "account_alias": "paper", "positions": []},
        positions=[],
        portfolio_rules_config=PortfolioRulesConfig(family_strategy_ids=(strategy_id,)),
    )
    risk_decision = await _run_family_oms_submit(family=family, data_dir=data_dir)
    append_jsonl_event(data_dir, "risk_decisions", "risk_decisions", risk_decision)
    write_risk_halt_event(
        data_dir,
        risk_decision.get("lineage") or lineage,
        reason=f"{family}_synthetic_halt",
        strategy_id=strategy_id,
        source="synthetic_day",
    )
    append_jsonl_event(
        data_dir,
        "portfolio_rules",
        "rules",
        build_portfolio_rule_event(
            {
                "rule": "synthetic_scale",
                "strategy_id": strategy_id,
                "direction": "LONG",
                "symbol": "MNQ" if family == "momentum" else "QQQ",
                "approved": True,
                "size_multiplier": 0.5,
                "trace_id": risk_decision.get("trace_id", ""),
                "signal_id": risk_decision.get("signal_id", ""),
                "bar_id": risk_decision.get("bar_id", ""),
                "exchange_timestamp": risk_decision.get("exchange_timestamp"),
            },
            portfolio_rules_config=PortfolioRulesConfig(family_strategy_ids=(strategy_id,)),
            request_context={
                "requested_sizing": {"risk_R": 1.0, "qty": 2, "risk_dollars": 200.0},
                "current_size_multiplier": 1.0,
            },
            lineage=risk_decision.get("lineage") or lineage,
        ),
    )
    heartbeat_subdir = "heartbeats" if family == "momentum" else "heartbeat"
    for subdir, prefix, event_type, payload in (
        ("missed", "missed", "missed_opportunity", {"signal_id": "miss_1", "blocked_by": "synthetic"}),
        ("filter_decisions", "filter_decisions", "filter_decision", {"filter_name": "synthetic", "passed": False}),
        (heartbeat_subdir, "heartbeats", "heartbeat", {"active_positions": 0, "open_orders": 0}),
        ("errors", "instrumentation_errors", "error", {"error_type": "synthetic", "message": "synthetic"}),
        ("coordination_events", "coordination_events", "coordinator_action", {"action_type": "synthetic_update"}),
    ):
        append_jsonl_event(
            data_dir,
            subdir,
            prefix,
            enrich_payload(payload, lineage=lineage, event_type=event_type),
        )

    captured: list[dict] = []

    class FakeResponse:
        status_code = 200

    class FakeRequests:
        def post(self, _url, data, headers, timeout):
            body = gzip.decompress(data) if headers.get("Content-Encoding") == "gzip" else data
            captured.append(json.loads(body.decode("utf-8")))
            return FakeResponse()

    if family == "momentum":
        from strategies.momentum.instrumentation.src import sidecar as sidecar_module
        from strategies.momentum.instrumentation.src.sidecar import Sidecar as FamilySidecar
    else:
        from strategies.swing.instrumentation.src import sidecar as sidecar_module
        from strategies.swing.instrumentation.src.sidecar import Sidecar as FamilySidecar

    fake_requests = FakeRequests()
    monkeypatch.setattr(sidecar_module, "requests", fake_requests)
    alias = sys.modules.get("instrumentation.src.sidecar")
    if alias is not None:
        monkeypatch.setattr(alias, "requests", fake_requests, raising=False)
    sidecar = FamilySidecar(
        {
            "bot_id": lineage.bot_id,
            "data_dir": str(data_dir),
            "sidecar": {
                "relay_url": "http://relay.local",
                "batch_size": 100,
                "retry_max": 1,
                "retry_backoff_base_seconds": 0,
            },
        }
    )
    sidecar.run_once()
    sent = [event for envelope in captured for event in envelope["events"]]
    if hasattr(sidecar, "get_diagnostics"):
        assert sidecar.get_diagnostics()["total_forwarded"] == len(sent)
    return sent


@pytest.mark.asyncio
async def test_full_synthetic_day_instrumentation_chain(tmp_path, monkeypatch) -> None:
    sent = await run_synthetic_day_instrumentation_chain(tmp_path, monkeypatch)
    event_types = {event["event_type"] for event in sent}

    assert {
        "deployment",
        "config_snapshot",
        "portfolio_rule_check",
        "risk_decision",
        "position_snapshot",
        "portfolio_snapshot",
        "allocation_snapshot",
        "allocation_drift",
        "allocation_unfreeze",
        "family_daily_snapshot",
        "trade_entry",
        "trade",
        "missed_opportunity",
        "filter_decision",
        "risk_denial",
        "risk_halt",
        "coordinator_action",
        "heartbeat",
        "error",
    }.issubset(event_types)

    rule_payload = next(
        _payload(event)
        for event in sent
        if event["event_type"] == "portfolio_rule_check"
        and _payload(event).get("result") == "scale"
    )
    assert rule_payload["approved_qty"] == 5

    allocation_payload = next(
        _payload(event)
        for event in sent
        if event["event_type"] == "allocation_snapshot" and _payload(event).get("source") == "fill"
    )
    assert allocation_payload["family_target_weights"] == {"stock": 1.0}
    assert allocation_payload["strategy_target_weights"] == {"IARIC_v1": 1.0}
    assert allocation_payload["raw_nav"] == 100_000.0
    assert allocation_payload["allocated_nav"] == 100_000.0

    portfolio_payload = next(
        _payload(event)
        for event in sent
        if event["event_type"] == "portfolio_snapshot" and _payload(event).get("source") == "fill"
    )
    assert portfolio_payload["equity"] == 100_000.0

    position_payload = next(
        _payload(event)
        for event in sent
        if event["event_type"] == "position_snapshot" and _payload(event).get("source") == "fill"
    )
    assert position_payload["portfolio_id"]
    assert position_payload["portfolio_id"] != "stock"

    trade_payload = next(_payload(event) for event in sent if event["event_type"] == "trade")
    assert trade_payload["decision_ref"] == "decision_1"
    assert trade_payload["intent_id"]
    assert trade_payload["portfolio_decision_ref"]
    assert trade_payload["order_ids"] == ["order_entry", "order_exit"]
    assert trade_payload["fill_ids"] == ["fill_exec_1", "fill_exec_2"]
    assert trade_payload["artifact_hash"].startswith("trade_artifact_")
    assert trade_payload["resource_plan_hash"].startswith("resource_plan_")

    halt_payload = next(_payload(event) for event in sent if event["event_type"] == "risk_halt")
    assert halt_payload["schema_version"] == "risk_halt_v1"
    assert halt_payload["reason"] == "synthetic_halt"
    assert halt_payload["halt_scope"] == "strategy"

    coordination_payload = next(_payload(event) for event in sent if event["event_type"] == "coordinator_action")
    assert coordination_payload["schema_version"] == "coordinator_action_v1"
    assert coordination_payload["action_type"] == "synthetic_regime_update"
    assert coordination_payload["risk_config_version_before"] != coordination_payload["risk_config_version_after"]
    assert coordination_payload["effective_config_evidence"]["portfolio_rules_config_after"]["directional_cap_R"] == 4.0

    for event in sent:
        payload = _payload(event)
        assert event["bot_id"] == "stock_trader"
        assert payload.get("lineage")
