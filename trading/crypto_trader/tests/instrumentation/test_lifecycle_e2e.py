from __future__ import annotations

import asyncio
import gzip
import hashlib
import hmac
import json
import urllib.request
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from crypto_trader.core.clock import SimClock
from crypto_trader.core.engine import MultiTimeFrameBars, StrategyContext
from crypto_trader.core.events import CanonicalRuntimeEvent, EventBus, PositionClosedEvent
from crypto_trader.core.execution_gateway import ExecutionGateway
from crypto_trader.core.models import (
    Bar,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TimeFrame,
    Trade,
)
from crypto_trader.core.runtime_types import ExecutionReport, ExecutionReportKind, OrderIntent
from crypto_trader.instrumentation.collector import InstrumentationCollector
from crypto_trader.instrumentation.daily_aggregator import DailyAggregator
from crypto_trader.instrumentation.emitter import EventEmitter
from crypto_trader.instrumentation.lineage import LineageContext
from crypto_trader.instrumentation.sidecar import SidecarForwarder
from crypto_trader.instrumentation.sinks import InMemorySink, JsonlSink
from crypto_trader.instrumentation.types import (
    DailySnapshot,
    EventMetadata,
    FilterDecision,
    HealthReportSnapshot,
    MarketContext,
    MissedOpportunityEvent,
    PipelineFunnelSnapshot,
)
from crypto_trader.live import engine as live_engine_module
from crypto_trader.live.config import LiveConfig
from crypto_trader.live.engine import LiveEngine, _DEFAULT_BRIDGE_CONTRACT_ROOT, _StrategySlot
from crypto_trader.live.lifecycle import LivePositionLedgerEntry, PositionLifecycleLedger
from crypto_trader.live.oms_store import OmsStore
from crypto_trader.portfolio.config import PortfolioConfig, StrategyAllocation
from crypto_trader.portfolio.coordinator import BrokerProxy, StrategyCoordinator
from crypto_trader.portfolio.manager import PortfolioManager
from crypto_trader.portfolio.state import OpenRisk, PortfolioState
from crypto_trader.relay.store import RelayStore


class _FakeBroker:
    def __init__(self) -> None:
        self.positions: list[Position] = []
        self.open_orders: list[Order] = []
        self.submitted: list[Order] = []
        self.equity = 10_000.0

    def submit_order(self, order: Order) -> str:
        order.status = OrderStatus.WORKING
        self.submitted.append(order)
        self.open_orders.append(order)
        return order.order_id

    def get_positions(self) -> list[Position]:
        return list(self.positions)

    def get_open_orders(self, symbol: str = "") -> list[Order]:
        if symbol:
            return [order for order in self.open_orders if order.symbol == symbol]
        return list(self.open_orders)

    def get_equity(self) -> float:
        return self.equity

    def get_order_owner(self, _order_id: str) -> str | None:
        return None


class _RuntimeStrategy:
    name = "momentum"
    symbols = ["BTC"]
    timeframes = [TimeFrame.M15]

    def __init__(self, collector: InstrumentationCollector) -> None:
        self._collector = collector
        self.fills: list[Fill] = []

    def on_fill(self, fill: Fill, _ctx) -> None:
        self.fills.append(fill)


class _StartSubmittingStrategy(_RuntimeStrategy):
    def __init__(self, collector: InstrumentationCollector) -> None:
        super().__init__(collector)
        self.orders_submitted = 0
        self.initialized = False
        self.shutdown = False

    def on_init(self, _ctx) -> None:
        self.initialized = True

    def on_bar(self, bar: Bar, ctx) -> None:
        if self.orders_submitted:
            return
        ctx.broker.submit_order(Order(
            order_id="start_entry_o",
            symbol=bar.symbol,
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="entry",
            metadata={"risk_R": 0.5},
        ))
        self.orders_submitted += 1

    def on_shutdown(self, _ctx) -> None:
        self.shutdown = True


class _StartedFakeBroker:
    def __init__(self, *args, **kwargs) -> None:
        self.equity = 10_000.0
        self.positions: list[Position] = []
        self.open_orders: list[Order] = []
        self.submitted: list[Order] = []
        self.fills: list[Fill] = []
        self._orders: dict[str, Order] = {}
        self._local_to_oid: dict[str, str] = {}
        self._oid_map: dict[str, str] = {}

    def submit_order(self, order: Order) -> str:
        exchange_id = f"ex_{order.order_id}"
        order.status = OrderStatus.FILLED
        order.metadata.setdefault("exchange_order_id", exchange_id)
        self.submitted.append(order)
        self._orders[order.order_id] = order
        self._local_to_oid[order.order_id] = exchange_id
        self._oid_map[exchange_id] = order.order_id
        ts = datetime.now(timezone.utc)
        self.positions = [
            Position(order.symbol, order.side, order.qty, 100.0, open_time=ts),
        ]
        self.fills.append(Fill(
            order_id=exchange_id,
            exchange_order_id=exchange_id,
            exchange_fill_id=f"fill_{order.order_id}",
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            fill_price=100.0,
            commission=0.01,
            timestamp=ts,
            tag="",
        ))
        return order.order_id

    def get_positions(self) -> list[Position]:
        return list(self.positions)

    def get_open_orders(self, symbol: str = "") -> list[Order]:
        return []

    def get_equity(self) -> float:
        return self.equity

    def get_fills_since(self, since: datetime) -> list[Fill]:
        return [fill for fill in self.fills if fill.timestamp >= since]


class _SingleFillReportAdapter:
    def __init__(self, *, exchange_order_id: str, role: str, timestamp: datetime) -> None:
        self.exchange_order_id = exchange_order_id
        self.role = role
        self.timestamp = timestamp

    def submit(self, intent: OrderIntent) -> list[ExecutionReport]:
        metadata = {
            **dict(intent.metadata),
            "strategy_id": intent.strategy_id,
            "order_type": intent.order_type.value,
            "role": self.role,
            "submitted_at": self.timestamp.isoformat(),
        }
        return [
            ExecutionReport(
                report_id=f"report_{intent.client_order_id}",
                kind=ExecutionReportKind.FILL,
                timestamp=self.timestamp,
                symbol=intent.symbol,
                side=intent.side,
                client_order_id=intent.client_order_id,
                exchange_order_id=self.exchange_order_id,
                order_status=OrderStatus.FILLED,
                qty=intent.qty,
                filled_qty=intent.qty,
                fill_price=100.0 if self.role == "entry" else 110.0,
                commission=0.01 if self.role == "entry" else 0.02,
                metadata=metadata,
            )
        ]

    def cancel(self, _client_order_id: str) -> list[ExecutionReport]:
        return []


def _portfolio_config(*, max_total_positions: int = 9) -> PortfolioConfig:
    return PortfolioConfig(
        strategies=(
            StrategyAllocation(strategy_id="momentum", priority=0),
            StrategyAllocation(strategy_id="trend", priority=1),
            StrategyAllocation(strategy_id="breakout", priority=2),
        ),
        max_total_positions=max_total_positions,
        dd_tiers=((0.05, 0.50), (1.00, 0.00)),
    )


def _make_engine(tmp_path: Path) -> tuple[LiveEngine, InMemorySink, _FakeBroker]:
    config = LiveConfig(
        state_dir=tmp_path,
        data_dir=tmp_path / "data",
        bot_id="synthetic_bot",
        family_id="crypto_perps",
        portfolio_id="paper_portfolio",
        account_alias="paper",
        symbols=["BTC", "ETH", "SOL"],
    )
    lineage = LineageContext(
        bot_id="synthetic_bot",
        family_id="crypto_perps",
        portfolio_id="paper_portfolio",
        account_alias="paper",
        code_sha="test-sha",
        deployment_id="deploy1",
        config_version="cfg1",
        portfolio_config_version="pcfg1",
        risk_config_version="risk1",
        allocation_version="alloc1",
        strategy_config_versions={"momentum": "strat1"},
        symbol_universe=["BTC", "ETH", "SOL"],
        deployment_manifest_version="manifest1",
    )
    manager = PortfolioManager(
        _portfolio_config(),
        PortfolioState(equity=10_000.0, peak_equity=10_000.0),
    )
    broker = _FakeBroker()
    emitter = EventEmitter()
    memory = InMemorySink()
    emitter.add_sink(memory)
    emitter.add_sink(JsonlSink(tmp_path))

    engine = object.__new__(LiveEngine)
    engine._config = config
    engine._running = False
    engine._slots = []
    engine._broker = broker
    engine._coordinator = None
    engine._manager = manager
    engine._feed = None
    engine._oms = OmsStore(tmp_path)
    engine._lifecycle = None
    engine._tracked_positions = {}
    engine._strategy_dispatched_fill_ids = set()
    engine._coordinator_applied_fill_ids = set()
    engine._lifecycle_applied_fill_ids = set()
    engine._lifecycle_closed_trades_by_fill_id = {}
    engine._tracked_fill_ids = set()
    engine._emitted_lifecycle_trade_ids = set()
    engine._finalized_fill_ids = set()
    engine._pending_missed = {}
    engine._last_funnels = {}
    engine._lineage = lineage
    engine._last_assistant_event_at = {}
    engine._emitter = emitter
    engine._daily_aggregator = None
    engine._sidecar = None
    engine._pg_sink = None
    return engine, memory, broker


def _wire_fill_runtime(engine: LiveEngine, broker: _FakeBroker) -> InstrumentationCollector:
    collector = InstrumentationCollector("momentum", "synthetic_bot", engine._lineage)
    strategy = _RuntimeStrategy(collector)
    events = EventBus()
    events.subscribe(
        PositionClosedEvent,
        lambda event: engine._emitter.emit_trade(
            collector.on_trade_closed(event.trade.symbol, event.trade)
        ),
    )
    ctx = StrategyContext(
        broker=SimpleNamespace(clear_ttl_for_fill=lambda _fill: None),
        clock=SimClock(),
        bars=MultiTimeFrameBars(),
        events=events,
    )
    slot = _StrategySlot(
        strategy_id="momentum",
        strategy=strategy,
        ctx=ctx,
        bars=MultiTimeFrameBars(),
        subscribed_tfs={TimeFrame.M15},
        primary_tf=TimeFrame.M15,
    )
    engine._slots = [slot]
    engine._coordinator = StrategyCoordinator(
        broker,
        engine._manager,
        event_callback=engine._emit_assistant_payload,
    )
    engine._lifecycle = PositionLifecycleLedger()
    return collector


def _payloads(memory: InMemorySink, event_type: str) -> list[dict]:
    return [event.to_dict()["payload"] for event in memory.events_by_type.get(event_type, [])]


_BRIDGE_IDS = ("crypto_trend_v1", "crypto_momentum_v1", "crypto_breakout_v1")


def _write_bridge_contracts(contract_root: Path) -> None:
    for bridge_id in _BRIDGE_IDS:
        path = contract_root / bridge_id / "strategy_plugin_contract.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "bridge_id": bridge_id,
                "contract_schema_version": "strategy_plugin_contract_v1",
            }, sort_keys=True),
            encoding="utf-8",
        )


def test_bridge_contract_root_defaults_to_assistant_bridge_contracts(monkeypatch) -> None:
    monkeypatch.delenv("CRYPTO_TRADER_BRIDGE_CONTRACT_ROOT", raising=False)
    assert LiveEngine._bridge_contract_root() == _DEFAULT_BRIDGE_CONTRACT_ROOT
    assert _DEFAULT_BRIDGE_CONTRACT_ROOT == Path("contracts") / "assistant_bridges"


def _submit_through_gateway(
    engine: LiveEngine,
    broker: _FakeBroker,
    order: Order,
    *,
    exchange_order_id: str,
    role: str,
    timestamp: datetime,
) -> None:
    events = EventBus()
    events.subscribe(CanonicalRuntimeEvent, engine._record_canonical_event)
    gateway = ExecutionGateway(
        adapter=_SingleFillReportAdapter(
            exchange_order_id=exchange_order_id,
            role=role,
            timestamp=timestamp,
        ),
        broker=broker,
        events=events,
        oms_store=engine._oms,
    )
    gateway.submit_order(order)


def test_startup_snapshots_emit_typed_config_and_deployment_metadata(tmp_path, monkeypatch) -> None:
    engine, memory, _broker = _make_engine(tmp_path)
    try:
        contract_root = tmp_path / "contracts"
        _write_bridge_contracts(contract_root)
        monkeypatch.setenv("CRYPTO_TRADER_BRIDGE_CONTRACT_ROOT", str(contract_root))
        engine._config.portfolio_config_path = tmp_path / "portfolio.json"
        engine._config.strategy_configs = {"momentum": tmp_path / "momentum.json"}
        engine._config.deployment_manifest_path = tmp_path / "deployment.json"
        deployment_manifest = {
            "candidate": "candidate_1",
            "portfolio_round": 3,
            "required_strategy_ids": ["momentum", "trend", "breakout"],
            "portfolio_rounds_manifest_path": "output/portfolio/rounds_manifest.json",
            "parity_alignment_path": "output/portfolio/round_3/parity_alignment.json",
        }

        engine._emit_startup_snapshots(
            _portfolio_config(),
            {"momentum": {"risk_pct": 0.01}},
            deployment_manifest,
        )

        deployment = _payloads(memory, "deployment")[-1]
        deployment_event = memory.events_by_type["deployment"][-1].to_dict()
        assert deployment_event["event_id"] == engine._lineage.deployment_id
        assert deployment_event["logical_event_id"] == engine._lineage.deployment_id
        assert deployment["candidate"] == "candidate_1"
        assert deployment["portfolio_round"] == 3
        assert set(deployment["deployment_metadata_artifacts"]) == set(_BRIDGE_IDS)

        config_payloads = _payloads(memory, "config_snapshot")
        config_kinds = {payload["config_kind"] for payload in config_payloads}
        assert {
            "live",
            "portfolio",
            "risk",
            "allocation",
            "strategy",
            "deployment_manifest",
            "bundle",
        }.issubset(config_kinds)
        strategy_snapshot = next(
            payload for payload in config_payloads
            if payload["config_kind"] == "strategy"
        )
        assert strategy_snapshot["strategy_id"] == "momentum"
        assert strategy_snapshot["redacted_config"] == {"risk_pct": 0.01}

        for bridge_id in _BRIDGE_IDS:
            contract_path = contract_root / bridge_id / "deployment_metadata.json"
            state_path = tmp_path / "deployment_metadata" / bridge_id / "deployment_metadata.json"
            assert contract_path.exists()
            assert state_path.exists()
            assert deployment["deployment_metadata_artifacts"][bridge_id] == str(contract_path)
            metadata = json.loads(contract_path.read_text(encoding="utf-8"))
            assert metadata["metadata_source"] == "live_bot_runtime_deployment_metadata_v1"
            assert metadata["strategy_id"] == bridge_id
            assert metadata["telemetry_schema_version"] == "trade_event_v1"
            assert metadata["assistant_event_schema_version"] == "assistant_event_v1"
            assert len(metadata["strategy_plugin_contract_hash"]) == 64
            assert metadata["deployment_metadata_path"] == contract_path.as_posix()
            assert metadata["state_deployment_metadata_path"] == state_path.as_posix()
            assert metadata["deployed_commit_sha"] == engine._lineage.code_sha
            assert metadata["source_control_commit_sha"] == engine._lineage.code_sha
            assert metadata["runtime_instance_id"]
            assert isinstance(metadata["approval_ready"], bool)
            assert isinstance(metadata["approval_blockers"], list)

        typed_snapshots = [
            payload for payload in config_payloads
            if payload["config_kind"] != "bundle"
        ]
        event_ids = [
            event.to_dict()["event_id"]
            for event in memory.events_by_type["config_snapshot"]
            if event.to_dict()["payload"].get("config_kind") != "bundle"
        ]
        assert len(event_ids) == len(set(event_ids)) == len(typed_snapshots)
        startup_heartbeat = memory.events_by_type["heartbeat"][-1].to_dict()
        assert startup_heartbeat["event_type"] == "heartbeat"
        assert startup_heartbeat["report"]["startup"] is True
    finally:
        engine._oms.close()


def test_config_snapshot_identity_uses_kind_and_version_only(tmp_path) -> None:
    engine, memory, _broker = _make_engine(tmp_path)
    try:
        engine._emit_config_snapshot(
            config_kind="strategy",
            config_path="configs/momentum.json",
            config_version="strategy-v1",
            redacted_config={"risk_pct": 0.01},
            hash_inputs={"risk_pct": 0.01},
            loaded_at="2026-05-31T00:00:00+00:00",
            strategy_id="momentum",
        )
        first = memory.events_by_type["config_snapshot"][-1].to_dict()

        engine._lineage = replace(engine._lineage, deployment_id="deploy-after-restart")
        engine._runtime_instance_id = "runtime-after-restart"
        engine._emit_config_snapshot(
            config_kind="strategy",
            config_path="configs/trend.json",
            config_version="strategy-v1",
            redacted_config={"risk_pct": 0.02},
            hash_inputs={"risk_pct": 0.02},
            loaded_at="2026-06-01T00:00:00+00:00",
            strategy_id="trend",
        )
        second = memory.events_by_type["config_snapshot"][-1].to_dict()

        engine._emit_config_snapshot(
            config_kind="strategy",
            config_path="configs/trend.json",
            config_version="strategy-v2",
            redacted_config={"risk_pct": 0.03},
            hash_inputs={"risk_pct": 0.03},
            loaded_at="2026-06-01T00:01:00+00:00",
            strategy_id="trend",
        )
        third = memory.events_by_type["config_snapshot"][-1].to_dict()

        assert first["event_id"] == second["event_id"]
        assert first["logical_event_id"] == second["logical_event_id"]
        assert first["payload"]["config_snapshot_event_id"] == first["event_id"]
        assert second["payload"]["config_snapshot_event_id"] == second["event_id"]
        assert third["event_id"] != first["event_id"]
        assert third["logical_event_id"] == third["event_id"]
    finally:
        engine._oms.close()


def test_deployment_metadata_can_be_approval_ready_with_bridge_contracts(tmp_path, monkeypatch) -> None:
    engine, _memory, _broker = _make_engine(tmp_path)
    contract_root = tmp_path / "contracts"
    _write_bridge_contracts(contract_root)
    monkeypatch.setenv("CRYPTO_TRADER_BRIDGE_CONTRACT_ROOT", str(contract_root))
    monkeypatch.setattr(engine, "_git_remote_url", lambda: "https://github.com/example/crypto_trader")
    monkeypatch.setattr(engine, "_git_worktree_clean", lambda: True)

    try:
        artifacts = engine._emit_deployment_metadata_artifacts(
            {
                "momentum": {"risk_pct": 0.01},
                "trend": {"risk_pct": 0.01},
                "breakout": {"risk_pct": 0.01},
            },
            _portfolio_config(),
            started_at=datetime(2026, 5, 31, tzinfo=timezone.utc).isoformat(),
        )

        assert set(artifacts) == {
            "crypto_trend_v1",
            "crypto_momentum_v1",
            "crypto_breakout_v1",
        }
        for bridge_id, path in artifacts.items():
            assert Path(path) == contract_root / bridge_id / "deployment_metadata.json"
            assert (tmp_path / "deployment_metadata" / bridge_id / "deployment_metadata.json").exists()
            metadata = json.loads(Path(path).read_text(encoding="utf-8"))
            assert metadata["strategy_id"] == bridge_id
            assert metadata["approval_ready"] is True
            assert metadata["approval_blockers"] == []
            assert metadata["repo_url"] == "https://github.com/example/crypto_trader"
            assert metadata["source_control_worktree_clean"] is True
            assert len(metadata["strategy_plugin_contract_hash"]) == 64
    finally:
        engine._oms.close()


def test_decision_and_market_ids_are_promoted_to_top_level_identity(tmp_path) -> None:
    engine, memory, _broker = _make_engine(tmp_path)
    ts = datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)

    try:
        engine._record_canonical_event(CanonicalRuntimeEvent(
            timestamp=ts,
            stream="decision",
            payload={
                "decision_id": "decision_1",
                "strategy_id": "momentum",
                "symbol": "BTC",
                "timeframe": "15m",
                "decision_time": ts.isoformat(),
                "decision_key": "decision_1",
                "action": "order",
                "bar_id": "bar_1",
            },
        ))
        engine._record_canonical_event(CanonicalRuntimeEvent(
            timestamp=ts,
            stream="market",
            payload={
                "strategy_id": "momentum",
                "symbol": "BTC",
                "timeframe": "15m",
                "available_at": ts.isoformat(),
                "bar_id": "bar_1",
            },
        ))

        decision = memory.events_by_type["decision_event"][-1].to_dict()
        market = memory.events_by_type["market_snapshot"][-1].to_dict()
        assert decision["event_id"] == "decision_1"
        assert decision["logical_event_id"] == "decision_1"
        assert market["event_id"] == "bar_1"
        assert market["logical_event_id"] == "bar_1"
    finally:
        engine._oms.close()


def test_startup_portfolio_snapshot_counts_synced_open_orders(tmp_path, monkeypatch) -> None:
    engine, memory, broker = _make_engine(tmp_path)

    try:
        contract_root = tmp_path / "contracts"
        _write_bridge_contracts(contract_root)
        monkeypatch.setenv("CRYPTO_TRADER_BRIDGE_CONTRACT_ROOT", str(contract_root))
        broker.open_orders = [
            Order(
                order_id="resting_entry",
                symbol="BTC",
                side=Side.LONG,
                order_type=OrderType.LIMIT,
                qty=0.1,
                limit_price=100.0,
                status=OrderStatus.WORKING,
                tag="entry",
                metadata={"strategy_id": "momentum"},
            ),
        ]

        engine._sync_open_orders_to_oms()
        engine._emit_startup_snapshots(
            engine._manager.config,
            {"momentum": {"risk_pct": 0.01}},
            {"deployment": "synthetic"},
        )

        portfolio = _payloads(memory, "portfolio_snapshot")[-1]
        assert portfolio["source"] == "startup"
        assert portfolio["pending_orders_count"] == 1
    finally:
        engine._oms.close()


def _metadata(
    lineage: LineageContext,
    *,
    event_type: str,
    payload_key: str,
    strategy_id: str = "momentum",
    timestamp: datetime,
) -> EventMetadata:
    return EventMetadata.create(
        "synthetic_bot",
        strategy_id,
        timestamp,
        event_type,
        payload_key,
        lineage=lineage.for_strategy(strategy_id) if strategy_id != "portfolio" else lineage.for_portfolio(),
        family_id=lineage.family_id,
        portfolio_id=lineage.portfolio_id,
        account_alias=lineage.account_alias,
        config_version=lineage.config_version,
        deployment_id=lineage.deployment_id,
        code_sha=lineage.code_sha,
    )


def _trade(entry_ts: datetime, exit_ts: datetime) -> Trade:
    return Trade(
        trade_id="trade_btc_1",
        symbol="BTC",
        direction=Side.LONG,
        entry_price=100.0,
        exit_price=110.0,
        qty=0.1,
        entry_time=entry_ts,
        exit_time=exit_ts,
        pnl=1.0,
        r_multiple=1.2,
        commission=0.03,
        bars_held=8,
        setup_grade=None,
        exit_reason="tp",
        confluences_used=["trend"],
        confirmation_type="momentum",
        entry_method="market",
        funding_paid=0.0,
        mae_r=-0.2,
        mfe_r=1.4,
    )


def _position_instance_id(ts: datetime) -> str:
    return f"momentum:BTC:LONG:{int(ts.timestamp() * 1000)}"


def _entry_fill(ts: datetime) -> Fill:
    return Fill(
        order_id="entry_o",
        exchange_order_id="ex_entry",
        exchange_fill_id="entry_fill",
        symbol="BTC",
        side=Side.LONG,
        qty=0.1,
        fill_price=100.0,
        commission=0.01,
        timestamp=ts,
        tag="entry",
    )


def _exit_fill(ts: datetime) -> Fill:
    return Fill(
        order_id="exit_o",
        exchange_order_id="ex_exit",
        exchange_fill_id="exit_fill",
        symbol="BTC",
        side=Side.SHORT,
        qty=0.1,
        fill_price=110.0,
        commission=0.02,
        timestamp=ts,
        tag="tp",
    )


def _seed_trade_join(engine: LiveEngine, entry_ts: datetime, exit_ts: datetime) -> None:
    engine._oms.upsert_order(
        client_order_id="entry_o",
        exchange_order_id="ex_entry",
        strategy_id="momentum",
        symbol="BTC",
        side=Side.LONG.value,
        status="FILLED",
        order_type="MARKET",
        role="entry",
        decision_id="decision_1",
        metadata={
            "decision_id": "decision_1",
            "bar_id": "bar_1",
            "decision_time": entry_ts.isoformat(),
            "intent_id": "intent_1",
            "portfolio_rule_event_id": "portfolio_rule_1",
            "risk_decision_id": "risk_decision_1",
        },
    )
    engine._oms.upsert_order(
        client_order_id="exit_o",
        exchange_order_id="ex_exit",
        strategy_id="momentum",
        symbol="BTC",
        side=Side.SHORT.value,
        status="FILLED",
        order_type="MARKET",
        role="exit",
        decision_id="exit_decision_1",
        metadata={
            "decision_id": "exit_decision_1",
            "bar_id": "bar_exit_1",
            "decision_time": exit_ts.isoformat(),
        },
    )
    engine._oms.record_received_fill(
        fill_id="entry_fill",
        client_order_id="entry_o",
        exchange_order_id="ex_entry",
        strategy_id="momentum",
        symbol="BTC",
        side=Side.LONG.value,
        qty=0.1,
        price=100.0,
        commission=0.01,
        timestamp=entry_ts,
        exchange_fill_id="entry_fill",
        raw={},
    )
    engine._oms.record_received_fill(
        fill_id="exit_fill",
        client_order_id="exit_o",
        exchange_order_id="ex_exit",
        strategy_id="momentum",
        symbol="BTC",
        side=Side.SHORT.value,
        qty=0.1,
        price=110.0,
        commission=0.02,
        timestamp=exit_ts,
        exchange_fill_id="exit_fill",
        raw={},
    )


def _seed_stale_trade_join(engine: LiveEngine, entry_ts: datetime, exit_ts: datetime) -> None:
    engine._oms.upsert_order(
        client_order_id="stale_entry_o",
        exchange_order_id="stale_ex_entry",
        strategy_id="momentum",
        symbol="BTC",
        side=Side.LONG.value,
        status="FILLED",
        order_type="MARKET",
        role="entry",
        decision_id="stale_decision",
        metadata={
            "decision_id": "stale_decision",
            "bar_id": "stale_bar",
            "decision_time": entry_ts.isoformat(),
            "intent_id": "stale_intent",
            "portfolio_rule_event_id": "stale_portfolio_rule",
            "risk_decision_id": "stale_risk_decision",
        },
    )
    engine._oms.upsert_order(
        client_order_id="stale_exit_o",
        exchange_order_id="stale_ex_exit",
        strategy_id="momentum",
        symbol="BTC",
        side=Side.SHORT.value,
        status="FILLED",
        order_type="MARKET",
        role="exit",
        decision_id="stale_exit_decision",
        metadata={
            "decision_id": "stale_exit_decision",
            "bar_id": "stale_exit_bar",
            "decision_time": exit_ts.isoformat(),
        },
    )
    engine._oms.record_received_fill(
        fill_id="stale_entry_fill",
        client_order_id="stale_entry_o",
        exchange_order_id="stale_ex_entry",
        strategy_id="momentum",
        symbol="BTC",
        side=Side.LONG.value,
        qty=0.1,
        price=90.0,
        commission=0.01,
        timestamp=entry_ts,
        exchange_fill_id="stale_entry_fill",
        raw={"tag": "entry"},
    )
    engine._oms.record_received_fill(
        fill_id="stale_exit_fill",
        client_order_id="stale_exit_o",
        exchange_order_id="stale_ex_exit",
        strategy_id="momentum",
        symbol="BTC",
        side=Side.SHORT.value,
        qty=0.1,
        price=95.0,
        commission=0.02,
        timestamp=exit_ts,
        exchange_fill_id="stale_exit_fill",
        raw={"tag": "tp"},
    )


def test_fill_triggers_position_allocation_and_portfolio_snapshots(tmp_path) -> None:
    engine, memory, broker = _make_engine(tmp_path)
    ts = datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)
    position_instance_id = _position_instance_id(ts)

    try:
        broker.positions = [
            Position("BTC", Side.LONG, 0.1, 100.0, open_time=ts),
        ]
        engine._manager.state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 0.5, ts))

        engine._track_entry_fill_once("entry_fill", "momentum", _entry_fill(ts))

        portfolio = _payloads(memory, "portfolio_snapshot")[-1]
        position = _payloads(memory, "position_snapshot")[-1]
        allocation = _payloads(memory, "allocation_snapshot")[-1]

        assert portfolio["source"] == "entry_fill"
        assert portfolio["fill_id"] == "entry_fill"
        assert position["source"] == "entry_fill"
        assert position["position_instance_id"] == position_instance_id
        assert position["open_order_ids"] == []
        assert allocation["source"] == "entry_fill"
        assert allocation["allocation_version"] == "alloc1"
        assert allocation["fill_id"] == "entry_fill"
    finally:
        engine._oms.close()


def test_fill_snapshot_preserves_unrelated_position_identity(tmp_path) -> None:
    engine, memory, broker = _make_engine(tmp_path)
    ts = datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)

    try:
        broker.positions = [
            Position("BTC", Side.LONG, 0.1, 100.0, open_time=ts),
            Position("ETH", Side.SHORT, 0.5, 2500.0, open_time=ts),
        ]
        engine._manager.state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 0.5, ts))

        engine._track_entry_fill_once("entry_fill", "momentum", _entry_fill(ts))

        positions = [
            payload for payload in _payloads(memory, "position_snapshot")
            if payload["source"] == "entry_fill"
        ]
        eth = next(payload for payload in positions if payload["symbol"] == "ETH")
        btc = next(payload for payload in positions if payload["symbol"] == "BTC")

        assert btc["strategy_id"] == "momentum"
        assert btc["fill_symbol"] == "BTC"
        assert eth["strategy_id"] == "unknown"
        assert eth["fill_symbol"] == "BTC"
        assert eth["fill_id"] == "entry_fill"
    finally:
        engine._oms.close()


def test_exit_fill_emits_flat_position_snapshot_after_full_close(tmp_path) -> None:
    engine, memory, broker = _make_engine(tmp_path)
    entry_ts = datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)
    exit_ts = entry_ts + timedelta(hours=2)
    position_instance_id = _position_instance_id(entry_ts)

    try:
        broker.positions = []
        engine._tracked_positions["BTC"] = {
            "strategy_id": "momentum",
            "direction": Side.LONG,
            "position_instance_id": position_instance_id,
            "entry_price": 100.0,
            "entry_time": entry_ts,
            "qty": 0.1,
            "entry_commission": 0.01,
        }

        engine._emit_fill_lifecycle_snapshots(
            "exit_fill",
            "momentum",
            _exit_fill(exit_ts),
            source="exit_fill",
            closed_trade=_trade(entry_ts, exit_ts),
        )

        flat = _payloads(memory, "position_snapshot")[-1]
        assert flat["source"] == "exit_fill"
        assert flat["fill_id"] == "exit_fill"
        assert flat["strategy_id"] == "momentum"
        assert flat["symbol"] == "BTC"
        assert flat["qty"] == 0.0
        assert flat["risk_R"] == 0.0
        assert flat["position_instance_id"] == position_instance_id
        assert flat["position_status"] == "closed"
    finally:
        engine._oms.close()


def test_closed_oms_position_row_reuses_lifecycle_position_instance_id(tmp_path) -> None:
    engine, _memory, _broker = _make_engine(tmp_path)
    entry_ts = datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)
    exit_ts = entry_ts + timedelta(hours=2)
    position_instance_id = _position_instance_id(entry_ts)
    trade = _trade(entry_ts, exit_ts)
    trade.trade_id = f"live_{position_instance_id}:{int(exit_ts.timestamp() * 1000)}"

    try:
        engine._record_lifecycle_trade_position("momentum", trade)
        row = engine._oms._conn.execute(
            "SELECT * FROM positions WHERE position_instance_id=?",
            (position_instance_id,),
        ).fetchone()

        assert row is not None
        assert row["position_instance_id"] == position_instance_id
        assert row["status"] == "CLOSED"
    finally:
        engine._oms.close()


def test_oms_role_propagates_to_untagged_entry_fill_before_snapshots(tmp_path) -> None:
    engine, memory, broker = _make_engine(tmp_path)
    ts = datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)

    try:
        _wire_fill_runtime(engine, broker)
        _seed_trade_join(engine, ts, ts + timedelta(hours=2))
        broker.positions = [
            Position("BTC", Side.LONG, 0.1, 100.0, open_time=ts),
        ]
        incoming = Fill(
            order_id="ex_entry",
            exchange_order_id="ex_entry",
            exchange_fill_id="entry_fill",
            symbol="BTC",
            side=Side.LONG,
            qty=0.1,
            fill_price=100.0,
            commission=0.01,
            timestamp=ts,
            tag="",
        )

        result = engine._process_fills([incoming])

        assert [fill.exchange_fill_id for fill in result.processed] == ["entry_fill"]
        assert engine._slots[0].strategy.fills[0].tag == "entry"
        assert engine._manager.state.total_positions() == 1
        assert _payloads(memory, "portfolio_snapshot")[-1]["source"] == "entry_fill"
        assert _payloads(memory, "fill")[-1]["tag"] == "entry"
    finally:
        engine._oms.close()


def test_reconciliation_drift_event_records_unknown_allocation_and_snapshots(tmp_path) -> None:
    engine, memory, broker = _make_engine(tmp_path)
    ts = datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)

    try:
        broker.positions = [
            Position("ETH", Side.SHORT, 0.5, 2500.0, open_time=ts),
        ]
        fill = Fill(
            order_id="mystery_order",
            exchange_order_id="ex_unknown",
            exchange_fill_id="unknown_fill",
            symbol="ETH",
            side=Side.SHORT,
            qty=0.5,
            fill_price=2500.0,
            commission=0.5,
            timestamp=ts,
            tag="",
        )

        result = engine._process_fills([fill])

        event = _payloads(memory, "reconciliation_event")[-1]
        discrepancy = engine._oms.list_unresolved_discrepancies()[0]
        portfolio = _payloads(memory, "portfolio_snapshot")[-1]
        position = _payloads(memory, "position_snapshot")[-1]
        allocation = _payloads(memory, "allocation_snapshot")[-1]

        assert [item.exchange_fill_id for item in result.unresolved] == ["unknown_fill"]
        assert event["lifecycle_event_kind"] == "drift_assignment"
        assert event["action"] == "admin_correction_required"
        assert event["strategy_id"] == "UNKNOWN"
        assert event["unknown_allocation"] is True
        assert event["requires_admin_correction"] is True
        assert discrepancy["kind"] == "unattributed_fill"
        assert discrepancy["metadata"]["fill_id"] == "unknown_fill"
        assert portfolio["source"] == "unattributed_fill"
        assert portfolio["fill_id"] == "unknown_fill"
        assert portfolio["unknown_allocation"] is True
        assert position["source"] == "unattributed_fill"
        assert position["strategy_id"] == "UNKNOWN"
        assert position["symbol"] == "ETH"
        assert allocation["source"] == "unattributed_fill"
        assert allocation["unknown_allocation"] is True
    finally:
        engine._oms.close()


def test_unresolved_flat_fill_emits_zero_qty_position_snapshot(tmp_path) -> None:
    engine, memory, broker = _make_engine(tmp_path)
    ts = datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)

    try:
        broker.positions = []
        fill = Fill(
            order_id="mystery_order",
            exchange_order_id="ex_unknown",
            exchange_fill_id="flat_unknown_fill",
            symbol="ETH",
            side=Side.SHORT,
            qty=0.5,
            fill_price=2500.0,
            commission=0.5,
            timestamp=ts,
            tag="",
        )

        result = engine._process_fills([fill])

        position = _payloads(memory, "position_snapshot")[-1]
        assert [item.exchange_fill_id for item in result.unresolved] == ["flat_unknown_fill"]
        assert position["source"] == "unattributed_fill"
        assert position["position_status"] == "unresolved_flat"
        assert position["strategy_id"] == "UNKNOWN"
        assert position["symbol"] == "ETH"
        assert position["qty"] == 0.0
        assert position["fill_id"] == "flat_unknown_fill"
        assert position["reconciliation_status"] == "unattributed_fill"
    finally:
        engine._oms.close()


def test_unresolved_fill_retry_does_not_reemit_snapshots(tmp_path) -> None:
    engine, memory, broker = _make_engine(tmp_path)
    ts = datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)

    try:
        broker.positions = []
        fill = Fill(
            order_id="mystery_order",
            exchange_order_id="ex_unknown",
            exchange_fill_id="retry_unknown_fill",
            symbol="ETH",
            side=Side.SHORT,
            qty=0.5,
            fill_price=2500.0,
            commission=0.5,
            timestamp=ts,
            tag="",
        )

        first = engine._process_fills([fill])
        counts = {
            event_type: len(_payloads(memory, event_type))
            for event_type in (
                "reconciliation_event",
                "portfolio_snapshot",
                "position_snapshot",
                "allocation_snapshot",
            )
        }
        second = engine._process_fills([fill])

        assert [item.exchange_fill_id for item in first.unresolved] == ["retry_unknown_fill"]
        assert [item.exchange_fill_id for item in second.unresolved] == ["retry_unknown_fill"]
        assert len(engine._oms.list_unresolved_discrepancies()) == 1
        assert {
            event_type: len(_payloads(memory, event_type))
            for event_type in counts
        } == counts
    finally:
        engine._oms.close()


def test_startup_reconciliation_refreezes_from_persisted_unresolved_discrepancies(tmp_path) -> None:
    engine, memory, _broker = _make_engine(tmp_path)

    try:
        engine._oms.record_discrepancy(
            kind="missing_position",
            description="Persisted missing BTC position.",
            symbol="BTC",
            strategy_id="UNKNOWN",
        )

        engine._handle_startup_reconciliation([])

        freeze = _payloads(memory, "reconciliation_event")[-1]
        assert engine._manager.entries_blocked_reason == "live OMS/exchange reconciliation unresolved"
        assert freeze["lifecycle_event_kind"] == "freeze"
        assert freeze["action"] == "freeze_entries"
        assert freeze["status"] == "open"
        assert freeze["discrepancies"][0]["kind"] == "missing_position"
        assert "prior run" in freeze["description"]
        assert _payloads(memory, "allocation_snapshot")[-1]["source"] == "reconciliation_discrepancy"
        blocked = engine._manager.check_entry("momentum", "BTC", Side.LONG, 0.1)
        assert blocked.approved is False
        assert blocked.blocking_rule == "reconciliation_block"
    finally:
        engine._oms.close()


def test_startup_reconciliation_dedupes_repeated_fresh_discrepancies(tmp_path) -> None:
    engine, memory, _broker = _make_engine(tmp_path)
    discrepancy = SimpleNamespace(
        kind="phantom",
        symbol="BTC",
        expected="flat",
        actual="LONG 0.1",
    )

    try:
        engine._handle_startup_reconciliation([discrepancy])
        engine._handle_startup_reconciliation([discrepancy])

        unresolved = engine._oms.list_unresolved_discrepancies()
        assert len(unresolved) == 1
        assert unresolved[0]["kind"] == "phantom"
        freeze_events = [
            payload for payload in _payloads(memory, "reconciliation_event")
            if payload["lifecycle_event_kind"] == "freeze"
        ]
        assert len(freeze_events) == 2
        assert all(
            len(event["discrepancies"]) == 1
            for event in freeze_events
        )
    finally:
        engine._oms.close()


def test_clean_startup_reconciliation_clears_owned_entry_freeze(tmp_path) -> None:
    engine, memory, _broker = _make_engine(tmp_path)

    try:
        engine._manager.entries_blocked_reason = "live OMS/exchange reconciliation unresolved"

        engine._handle_startup_reconciliation([])

        unfreeze = _payloads(memory, "reconciliation_event")[-1]
        assert engine._manager.entries_blocked_reason == ""
        assert unfreeze["lifecycle_event_kind"] == "unfreeze"
        assert unfreeze["action"] == "allow_entries"
        assert unfreeze["status"] == "clear"
    finally:
        engine._oms.close()


def test_admin_correction_resolves_discrepancy_and_emits_lifecycle_event(tmp_path) -> None:
    engine, memory, _broker = _make_engine(tmp_path)

    try:
        discrepancy_id = engine._oms.record_discrepancy(
            kind="missing_position",
            description="Synthetic missing BTC position.",
            symbol="BTC",
            strategy_id="UNKNOWN",
            metadata={"fill_id": "unknown_fill"},
        )

        resolved = engine.record_admin_correction(
            discrepancy_id,
            resolution="Assigned fill to manual correction ledger.",
            resolved_by="operator_1",
            metadata={"ticket": "INC-1"},
        )

        correction = _payloads(memory, "reconciliation_event")[-1]
        assert resolved is True
        assert engine._oms.list_unresolved_discrepancies() == []
        assert engine._oms.get_discrepancy(discrepancy_id)["status"] == "RESOLVED"
        assert correction["lifecycle_event_kind"] == "admin_correction"
        assert correction["status"] == "resolved"
        assert correction["correction_applied"] is True
        assert correction["requires_admin_correction"] is False
        assert correction["metadata"]["resolved_by"] == "operator_1"
        assert _payloads(memory, "allocation_snapshot")[-1]["source"] == "admin_correction"
    finally:
        engine._oms.close()


def test_admin_correction_unfreezes_entries_after_last_discrepancy(tmp_path) -> None:
    engine, memory, _broker = _make_engine(tmp_path)

    try:
        engine._manager.entries_blocked_reason = "live OMS/exchange reconciliation unresolved"
        first_id = engine._oms.record_discrepancy(
            kind="missing_position",
            description="Synthetic missing BTC position.",
            symbol="BTC",
            strategy_id="UNKNOWN",
        )
        second_id = engine._oms.record_discrepancy(
            kind="missing_position",
            description="Synthetic missing ETH position.",
            symbol="ETH",
            strategy_id="UNKNOWN",
        )

        blocked = engine._manager.check_entry("momentum", "BTC", Side.LONG, 0.1)
        assert blocked.approved is False
        assert blocked.blocking_rule == "reconciliation_block"

        assert engine.record_admin_correction(first_id, resolution="Resolved BTC drift.") is True
        assert engine._manager.entries_blocked_reason == "live OMS/exchange reconciliation unresolved"
        assert not any(
            event["lifecycle_event_kind"] == "unfreeze"
            for event in _payloads(memory, "reconciliation_event")
        )

        assert engine.record_admin_correction(second_id, resolution="Resolved ETH drift.") is True
        assert engine._manager.entries_blocked_reason == ""
        unfreeze = _payloads(memory, "reconciliation_event")[-1]
        assert unfreeze["lifecycle_event_kind"] == "unfreeze"
        assert unfreeze["action"] == "allow_entries"
        assert unfreeze["status"] == "clear"
        assert _payloads(memory, "allocation_snapshot")[-1]["source"] == "reconciliation_unfreeze"
        allowed = engine._manager.check_entry("momentum", "BTC", Side.LONG, 0.1)
        assert allowed.approved is True
    finally:
        engine._oms.close()


def test_trade_completion_event_is_enriched_from_order_fill_runtime_join(tmp_path) -> None:
    engine, _memory, _broker = _make_engine(tmp_path)
    entry_ts = datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)
    exit_ts = entry_ts + timedelta(hours=2)

    try:
        _seed_stale_trade_join(
            engine,
            entry_ts - timedelta(days=1),
            exit_ts - timedelta(days=1),
        )
        _seed_trade_join(engine, entry_ts, exit_ts)
        trade = _trade(entry_ts, exit_ts)

        engine._attach_trade_completion_context("momentum", trade, fill_id="exit_fill")
        collector = InstrumentationCollector("momentum", "synthetic_bot", engine._lineage)
        event = collector.on_trade_closed("BTC", trade)

        assert event.intent_id == "intent_1"
        assert event.entry_decision_id == "decision_1"
        assert event.exit_decision_id == "exit_decision_1"
        assert event.exit_bar_id == "bar_exit_1"
        assert event.entry_order_ids == ["entry_o"]
        assert event.exit_order_ids == ["exit_o"]
        assert event.entry_fill_ids == ["entry_fill"]
        assert event.exit_fill_ids == ["exit_fill"]
        assert event.client_order_ids == ["entry_o", "exit_o"]
        assert event.exchange_order_ids == ["ex_entry", "ex_exit"]
        assert "stale_entry_o" not in event.client_order_ids
        assert "stale_entry_fill" not in event.entry_fill_ids
        assert event.decision_ref["config_version"] == "cfg1"
        assert event.portfolio_decision_ref["allocation_version"] == "alloc1"
        assert event.artifact_hash
        assert event.resource_plan_hash
        assert event.runtime_join["fill_id"] == "exit_fill"
        assert event.portfolio_rule_event_id == "portfolio_rule_1"
        assert event.risk_decision_id == "risk_decision_1"
    finally:
        engine._oms.close()


def test_lifecycle_trade_join_uses_matched_fill_ids_from_process_fills(tmp_path) -> None:
    engine, memory, broker = _make_engine(tmp_path)
    entry_ts = datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)
    exit_ts = entry_ts + timedelta(hours=2)

    try:
        _wire_fill_runtime(engine, broker)
        _seed_stale_trade_join(engine, entry_ts, exit_ts)

        entry_order = Order(
            order_id="entry_o",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="entry",
            metadata={
                "strategy_id": "momentum",
                "client_order_id": "entry_o",
                "risk_R": 0.5,
                "decision_id": "decision_1",
                "bar_id": "bar_1",
                "decision_time": entry_ts.isoformat(),
                "intent_id": "intent_1",
                "portfolio_rule_event_id": "portfolio_rule_1",
                "risk_decision_id": "risk_decision_1",
            },
        )
        _submit_through_gateway(
            engine,
            broker,
            entry_order,
            exchange_order_id="ex_entry",
            role="entry",
            timestamp=entry_ts,
        )
        broker.positions = [
            Position("BTC", Side.LONG, 0.1, 100.0, open_time=entry_ts),
        ]
        engine._process_fills([Fill(
            order_id="ex_entry",
            exchange_order_id="ex_entry",
            exchange_fill_id="entry_fill",
            symbol="BTC",
            side=Side.LONG,
            qty=0.1,
            fill_price=100.0,
            commission=0.01,
            timestamp=entry_ts,
            tag="",
        )])

        exit_order = Order(
            order_id="exit_o",
            symbol="BTC",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="tp",
            metadata={
                "strategy_id": "momentum",
                "decision_id": "exit_decision_1",
                "bar_id": "bar_exit_1",
                "decision_time": exit_ts.isoformat(),
            },
        )
        _submit_through_gateway(
            engine,
            broker,
            exit_order,
            exchange_order_id="ex_exit",
            role="exit",
            timestamp=exit_ts,
        )
        broker.positions = []
        engine._process_fills([Fill(
            order_id="ex_exit",
            exchange_order_id="ex_exit",
            exchange_fill_id="exit_fill",
            symbol="BTC",
            side=Side.SHORT,
            qty=0.1,
            fill_price=110.0,
            commission=0.02,
            timestamp=exit_ts,
            tag="tp",
        )])

        event = memory.trades[-1]
        assert event.entry_order_ids == ["entry_o"]
        assert event.exit_order_ids == ["exit_o"]
        assert event.entry_fill_ids == ["entry_fill"]
        assert event.exit_fill_ids == ["exit_fill"]
        assert event.client_order_ids == ["entry_o", "exit_o"]
        assert {fill["fill_id"] for fill in event.runtime_join["fills"]} == {
            "entry_fill",
            "exit_fill",
        }
        assert "stale_entry_o" not in event.client_order_ids
        assert "stale_entry_fill" not in {
            fill["fill_id"] for fill in event.runtime_join["fills"]
        }
    finally:
        engine._oms.close()


@pytest.mark.asyncio
async def test_live_start_dispatches_strategy_decision_order_and_fill_path(
    tmp_path,
    monkeypatch,
) -> None:
    config = LiveConfig(
        state_dir=tmp_path,
        data_dir=tmp_path / "data",
        bot_id="synthetic_bot",
        family_id="crypto_perps",
        portfolio_id="paper_portfolio",
        account_alias="paper",
        symbols=["BTC"],
        strategy_configs={"momentum": tmp_path / "momentum.json"},
        deployment_manifest_path=tmp_path / "deployment.json",
    )
    config.strategy_configs["momentum"].write_text("{}", encoding="utf-8")
    config.deployment_manifest_path.write_text("{}", encoding="utf-8")
    contract_root = tmp_path / "contracts"
    _write_bridge_contracts(contract_root)
    monkeypatch.setenv("CRYPTO_TRADER_BRIDGE_CONTRACT_ROOT", str(contract_root))

    collector = InstrumentationCollector("momentum", "synthetic_bot")
    strategy = _StartSubmittingStrategy(collector)

    class _NoWarmupFeed:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def load_warmup_bars(self, *_args, **_kwargs) -> list[Bar]:
            return []

    monkeypatch.setattr(live_engine_module, "HyperliquidBroker", _StartedFakeBroker)
    monkeypatch.setattr(live_engine_module, "LiveFeed", _NoWarmupFeed)
    monkeypatch.setattr(
        live_engine_module,
        "_create_strategy",
        lambda *_args, **_kwargs: (strategy, [TimeFrame.M15], TimeFrame.M15),
    )
    monkeypatch.setattr(
        LiveEngine,
        "_load_strategy_config",
        lambda self, _strategy_id, _config_path: SimpleNamespace(symbols=[]),
    )
    monkeypatch.setattr(
        LiveEngine,
        "_load_portfolio_config",
        lambda self: _portfolio_config(),
    )

    import hyperliquid.info as hl_info
    monkeypatch.setattr(hl_info, "Info", lambda *_args, **_kwargs: object())

    engine = LiveEngine(config)
    memory = InMemorySink()
    engine._emitter.add_sink(memory)

    try:
        await engine.start()
        engine._dispatch_bar(Bar(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10.0,
            timeframe=TimeFrame.M15,
        ))

        assert strategy.initialized is True
        assert strategy.orders_submitted == 1
        assert [fill.exchange_fill_id for fill in strategy.fills] == ["fill_start_entry_o"]
        assert _payloads(memory, "deployment")
        assert _payloads(memory, "decision_event")[-1]["action"] == "order"
        assert _payloads(memory, "order")
        assert _payloads(memory, "fill")[-1]["fill_id"] == "fill_start_entry_o"
        assert _payloads(memory, "portfolio_snapshot")[-1]["source"] == "entry_fill"
        assert engine._oms.get_order("start_entry_o") is not None
        assert engine._oms.get_fill("fill_start_entry_o")["status"] == "PROCESSED"
    finally:
        await engine.shutdown()
        engine._oms.close()


@pytest.mark.asyncio
async def test_daily_reset_loop_waits_for_midnight_before_reconciling_and_resetting(
    tmp_path,
    monkeypatch,
) -> None:
    engine, memory, _broker = _make_engine(tmp_path)
    ts = datetime(2026, 5, 31, 23, 59, tzinfo=timezone.utc)

    class _Aggregator:
        def compute_snapshot(self, date_str: str) -> DailySnapshot:
            return DailySnapshot(
                metadata=_metadata(
                    engine._lineage,
                    event_type="daily_snapshot",
                    payload_key=f"daily_{date_str}",
                    strategy_id="portfolio",
                    timestamp=ts,
                ),
                lineage=engine._lineage.for_portfolio(),
                date=date_str,
                total_trades=1,
                win_count=1,
                net_pnl=12.0,
                per_strategy_summary={
                    "momentum": {"pnl": 12.0, "net_pnl": 12.0, "realized_R": 2.5}
                },
            )

    seen_pnl_before_reset: list[float] = []
    original_reconcile = engine._emit_daily_reconciliation
    real_datetime = datetime
    clock_ticks = [
        real_datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        real_datetime(2026, 5, 31, 13, 0, tzinfo=timezone.utc),
        real_datetime(2026, 5, 31, 23, 59, 59, tzinfo=timezone.utc),
        real_datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
    ]

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            value = clock_ticks.pop(0) if clock_ticks else real_datetime(2026, 6, 1, tzinfo=timezone.utc)
            if tz is None:
                return value.replace(tzinfo=None)
            return value.astimezone(tz)

        @classmethod
        def fromisoformat(cls, value: str):
            return real_datetime.fromisoformat(value)

    def capture_reconciliation(date_str: str, snapshot: DailySnapshot) -> None:
        seen_pnl_before_reset.append(engine._manager.state.portfolio_daily_pnl_R)
        original_reconcile(date_str, snapshot)
        engine._running = False

    sleep_calls: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    try:
        engine._daily_aggregator = _Aggregator()
        engine._manager.state.current_day = date(2000, 1, 1)
        engine._manager.state.daily_pnl_R["momentum"] = 2.5
        engine._manager.state.portfolio_daily_pnl_R = 2.5
        engine._emit_daily_reconciliation = capture_reconciliation
        engine._running = True
        monkeypatch.setattr(asyncio, "sleep", record_sleep)
        monkeypatch.setattr(live_engine_module, "datetime", _Clock)

        await engine._daily_reset_loop()

        reconciliation = _payloads(memory, "daily_reconciliation")[-1]
        assert sleep_calls == [3600, 1.0]
        assert seen_pnl_before_reset == [2.5]
        assert reconciliation["date"] == "2026-05-31"
        assert reconciliation["risk_table"]["portfolio_daily_pnl_R"] == 2.5
        assert reconciliation["trade_summary"]["realized_R"] == {"momentum": 2.5}
        assert reconciliation["trade_summary"]["net_pnl_by_strategy"] == {"momentum": 12.0}
        assert engine._manager.state.portfolio_daily_pnl_R == 0.0
    finally:
        engine._oms.close()


def test_daily_reconciliation_includes_overnight_order_referenced_by_report_date_fill(tmp_path) -> None:
    engine, memory, _broker = _make_engine(tmp_path)
    submitted = datetime(2026, 5, 31, 23, 55, tzinfo=timezone.utc)
    filled = datetime(2026, 6, 1, 0, 5, tzinfo=timezone.utc)

    try:
        engine._oms.upsert_order(
            client_order_id="overnight_entry",
            exchange_order_id="ex_overnight_entry",
            strategy_id="momentum",
            symbol="BTC",
            side=Side.LONG.value,
            status="FILLED",
            order_type="MARKET",
            role="entry",
            decision_id="overnight_decision",
            metadata={
                "decision_id": "overnight_decision",
                "decision_time": submitted.isoformat(),
                "submitted_at": submitted.isoformat(),
                "intent_id": "overnight_intent",
            },
        )
        engine._oms.upsert_order(
            client_order_id="previous_day_only",
            exchange_order_id="ex_previous_day_only",
            strategy_id="momentum",
            symbol="BTC",
            side=Side.SHORT.value,
            status="FILLED",
            order_type="MARKET",
            role="exit",
            decision_id="old_decision",
            metadata={
                "decision_id": "old_decision",
                "decision_time": submitted.isoformat(),
                "submitted_at": submitted.isoformat(),
            },
        )
        engine._oms.record_fill(
            fill_id="overnight_fill",
            client_order_id="overnight_entry",
            exchange_order_id="ex_overnight_entry",
            strategy_id="momentum",
            symbol="BTC",
            side=Side.LONG.value,
            qty=0.1,
            price=100.0,
            commission=0.01,
            timestamp=filled,
            exchange_fill_id="overnight_fill",
            raw={"tag": "entry"},
        )

        engine._emit_daily_reconciliation(
            "2026-06-01",
            DailySnapshot(
                metadata=_metadata(
                    engine._lineage,
                    event_type="daily_snapshot",
                    payload_key="daily_2026-06-01",
                    strategy_id="portfolio",
                    timestamp=filled,
                ),
                lineage=engine._lineage.for_portfolio(),
                date="2026-06-01",
            ),
        )

        reconciliation = _payloads(memory, "daily_reconciliation")[-1]
        assert {order["client_order_id"] for order in reconciliation["orders"]} == {
            "overnight_entry"
        }
    finally:
        engine._oms.close()


@pytest.mark.asyncio
async def test_synthetic_day_writes_complete_lifecycle_chain_and_relay_can_ingest(
    tmp_path,
    monkeypatch,
) -> None:
    engine, memory, broker = _make_engine(tmp_path)
    ts = datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)
    entry_ts = ts + timedelta(minutes=1)
    exit_ts = ts + timedelta(hours=2)

    try:
        contract_root = tmp_path / "contracts"
        _write_bridge_contracts(contract_root)
        monkeypatch.setenv("CRYPTO_TRADER_BRIDGE_CONTRACT_ROOT", str(contract_root))
        engine._daily_aggregator = DailyAggregator(bot_id="synthetic_bot")
        engine._emitter.add_sink(engine._daily_aggregator)
        engine._emit_startup_snapshots(
            engine._manager.config,
            {"momentum": {"risk_pct": 0.01}},
            {"deployment": "synthetic"},
        )
        engine._emit_assistant_payload("decision_event", {
            "decision_event_id": "decision_event_1",
            "decision_id": "decision_1",
            "bar_id": "bar_1",
            "strategy_id": "momentum",
            "symbol": "BTC",
            "action": "submit_order",
            "intent_id": "intent_1",
            "timestamp": ts.isoformat(),
        })

        order = Order(
            order_id="entry_o",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="entry",
            metadata={
                "risk_R": 0.5,
                "decision_id": "decision_1",
                "bar_id": "bar_1",
                "decision_time": ts.isoformat(),
                "intent_id": "intent_1",
            },
        )
        proxy = BrokerProxy(
            broker,
            engine._manager,
            "momentum",
            event_callback=engine._emit_assistant_payload,
        )
        assert proxy.submit_order(order) == "entry_o"
        _submit_through_gateway(
            engine,
            broker,
            order,
            exchange_order_id="ex_entry",
            role="entry",
            timestamp=entry_ts,
        )

        blocking_state = PortfolioState(equity=10_000.0, peak_equity=10_000.0)
        blocking_proxy = BrokerProxy(
            _FakeBroker(),
            PortfolioManager(_portfolio_config(max_total_positions=0), blocking_state),
            "momentum",
            event_callback=engine._emit_assistant_payload,
        )
        blocking_proxy.submit_order(Order(
            order_id="blocked_o",
            symbol="ETH",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="entry",
            metadata={"risk_R": 0.5, "decision_id": "blocked_decision"},
        ))

        scaled_state = PortfolioState(equity=9_000.0, peak_equity=10_000.0)
        scaled_proxy = BrokerProxy(
            _FakeBroker(),
            PortfolioManager(_portfolio_config(), scaled_state),
            "momentum",
            event_callback=engine._emit_assistant_payload,
        )
        scaled_order = Order(
            order_id="scaled_o",
            symbol="SOL",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=1.0,
            tag="entry",
            metadata={"risk_R": 1.0, "decision_id": "scaled_decision"},
        )
        scaled_proxy.submit_order(scaled_order)

        _wire_fill_runtime(engine, broker)
        _seed_stale_trade_join(
            engine,
            entry_ts - timedelta(days=1),
            exit_ts - timedelta(days=1),
        )
        broker.positions = [
            Position("BTC", Side.LONG, 0.1, 100.0, open_time=entry_ts),
        ]
        entry_result = engine._process_fills([
            Fill(
                order_id="ex_entry",
                exchange_order_id="ex_entry",
                exchange_fill_id="entry_fill",
                symbol="BTC",
                side=Side.LONG,
                qty=0.1,
                fill_price=100.0,
                commission=0.01,
                timestamp=entry_ts,
                tag="",
            )
        ])

        exit_order = Order(
            order_id="exit_o",
            symbol="BTC",
            side=Side.SHORT,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="tp",
            metadata={
                "strategy_id": "momentum",
                "decision_id": "exit_decision_1",
                "bar_id": "bar_exit_1",
                "decision_time": exit_ts.isoformat(),
                "client_order_id": "exit_o",
            },
        )
        _submit_through_gateway(
            engine,
            broker,
            exit_order,
            exchange_order_id="ex_exit",
            role="exit",
            timestamp=exit_ts,
        )

        broker.positions = []
        exit_result = engine._process_fills([
            Fill(
                order_id="ex_exit",
                exchange_order_id="ex_exit",
                exchange_fill_id="exit_fill",
                symbol="BTC",
                side=Side.SHORT,
                qty=0.1,
                fill_price=110.0,
                commission=0.02,
                timestamp=exit_ts,
                tag="tp",
            )
        ])

        trade_payload = memory.trades[-1].to_dict()
        assert [fill.exchange_fill_id for fill in entry_result.processed] == ["entry_fill"]
        assert [fill.exchange_fill_id for fill in exit_result.processed] == ["exit_fill"]
        assert engine._manager.state.total_positions() == 0
        assert trade_payload["entry_order_ids"] == ["entry_o"]
        assert trade_payload["exit_order_ids"] == ["exit_o"]
        assert trade_payload["entry_fill_ids"] == ["entry_fill"]
        assert trade_payload["exit_fill_ids"] == ["exit_fill"]
        assert "stale_entry_o" not in trade_payload["client_order_ids"]
        joined_fill_ids = {
            fill["fill_id"] for fill in trade_payload["runtime_join"]["fills"]
        }
        assert "stale_entry_fill" not in joined_fill_ids

        missed = MissedOpportunityEvent(
            metadata=_metadata(
                engine._lineage,
                event_type="missed_opportunity",
                payload_key="missed_1:revision:0",
                timestamp=ts,
            ),
            lineage=engine._lineage.for_strategy("momentum"),
            opportunity_id="missed_1",
            logical_event_id="missed_1",
            pair="ETH",
            symbol="ETH",
            timeframe="15m",
            bar_id="bar_2",
            decision_id="decision_2",
            signal_id="signal_2",
            signal="momentum_B",
            signal_strength=1.3,
            blocked_by="portfolio_rule",
            block_reason="heat cap",
            blocking_rule_type="portfolio_rule",
            hypothetical_entry=2500.0,
            simulation_policy={"entry_price_source": "bar_close"},
            market_context=MarketContext(atr=10.0, adx=25.0),
            filter_decisions=[
                FilterDecision("setup", True),
                FilterDecision("portfolio_rule", False, reason="heat cap"),
            ],
            portfolio_rule_event_id="portfolio_rule_blocked",
        )
        engine._emitter.emit_missed(missed)
        missed.outcome_1h = 0.8
        missed.outcome_4h = 1.2
        missed.outcome_24h = 0.3
        missed.backfill_status = "complete"
        missed.bump_revision()
        engine._emitter.emit_missed(missed)

        engine._emitter.emit_health_report(HealthReportSnapshot(
            timestamp=ts.isoformat(),
            report={"status": "ok", "sidecar": {"enabled": True}},
            metadata=_metadata(
                engine._lineage,
                event_type="heartbeat",
                payload_key="heartbeat_1",
                strategy_id="portfolio",
                timestamp=ts,
            ),
            lineage=engine._lineage.for_portfolio(),
        ))
        engine._emitter.emit_funnel(PipelineFunnelSnapshot(
            strategy_id="momentum",
            timestamp=ts.isoformat(),
            period_start=ts.isoformat(),
            period_end=(ts + timedelta(hours=1)).isoformat(),
            funnel={"bars": 4, "decisions": 2, "orders": 1, "fills": 2},
            assessment="normal",
            metadata=_metadata(
                engine._lineage,
                event_type="pipeline_funnel",
                payload_key="funnel_1",
                timestamp=ts,
            ),
            lineage=engine._lineage.for_strategy("momentum"),
        ))

        engine._record_fill_discrepancy(
            Fill(
                order_id="unknown_o",
                exchange_order_id="ex_unknown",
                exchange_fill_id="unknown_fill",
                symbol="SOL",
                side=Side.SHORT,
                qty=1.0,
                fill_price=50.0,
                commission=0.01,
                timestamp=ts,
                tag="",
            ),
            fill_id="unknown_fill",
            kind="unattributed_fill",
            description="Synthetic unknown fill for drift coverage.",
        )
        engine._oms.upsert_lifecycle_entry(LivePositionLedgerEntry(
            strategy_id="momentum",
            symbol="BTC",
            direction=Side.LONG,
            position_instance_id="current_lifecycle_position",
            qty=0.1,
            avg_entry=100.0,
            entry_time=entry_ts,
        ))
        engine._oms.upsert_lifecycle_entry(LivePositionLedgerEntry(
            strategy_id="momentum",
            symbol="BTC",
            direction=Side.LONG,
            position_instance_id="carried_lifecycle_position",
            qty=0.1,
            avg_entry=90.0,
            entry_time=entry_ts - timedelta(days=1),
        ))
        engine._oms.upsert_lifecycle_entry(LivePositionLedgerEntry(
            strategy_id="momentum",
            symbol="BTC",
            direction=Side.LONG,
            position_instance_id="future_lifecycle_position",
            qty=0.1,
            avg_entry=120.0,
            entry_time=entry_ts + timedelta(days=1),
        ))

        original_reconcile = engine._emit_daily_reconciliation
        real_datetime = datetime
        closeout_ticks = [
            real_datetime(2026, 5, 31, 23, 59, 59, tzinfo=timezone.utc),
            real_datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
        ]

        class _CloseoutClock(datetime):
            @classmethod
            def now(cls, tz=None):
                value = (
                    closeout_ticks.pop(0)
                    if closeout_ticks
                    else real_datetime(2026, 6, 1, 0, 0, 1, tzinfo=timezone.utc)
                )
                if tz is None:
                    return value.replace(tzinfo=None)
                return value.astimezone(tz)

            @classmethod
            def fromisoformat(cls, value: str):
                return real_datetime.fromisoformat(value)

        sleep_calls: list[float] = []

        async def record_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        def capture_reconciliation(date_str: str, snapshot: DailySnapshot) -> None:
            original_reconcile(date_str, snapshot)
            engine._running = False

        engine._manager.state.current_day = date(2026, 5, 31)
        engine._manager.state.daily_pnl_R["momentum"] = 0.0
        engine._manager.state.portfolio_daily_pnl_R = 0.0
        engine._emit_daily_reconciliation = capture_reconciliation
        engine._running = True
        monkeypatch.setattr(asyncio, "sleep", record_sleep)
        monkeypatch.setattr(live_engine_module, "datetime", _CloseoutClock)

        await engine._daily_reset_loop()
        daily_reconciliation = _payloads(memory, "daily_reconciliation")[-1]

        assert sleep_calls == [1.0]
        assert {fill["fill_id"] for fill in daily_reconciliation["oms_fills"]} == {
            "entry_fill",
            "exit_fill",
        }
        assert {order["client_order_id"] for order in daily_reconciliation["orders"]} == {
            "entry_o",
            "exit_o",
        }
        assert {
            entry["position_instance_id"]
            for entry in daily_reconciliation["lifecycle_entries"]
        } == {"current_lifecycle_position", "carried_lifecycle_position"}
        assert daily_reconciliation["trade_summary"]["net_pnl_by_strategy"] == {"momentum": 0.97}
        assert daily_reconciliation["trade_summary"]["realized_R"] == {"momentum": 0.0}
        assert daily_reconciliation["trade_summary"]["fees"] == pytest.approx(0.03)
        assert daily_reconciliation["trade_summary"]["funding"] == pytest.approx(0.0)
        assert daily_reconciliation["trade_summary"]["family_realized_R"] == pytest.approx(0.0)
        family_daily = _payloads(memory, "family_daily_snapshot")[-1]
        assert family_daily["fees"] == pytest.approx(0.03)
        assert family_daily["funding"] == pytest.approx(0.0)
        assert family_daily["realized_R"] == pytest.approx(0.0)
        assert family_daily["family_summary"]["fees"] == pytest.approx(0.03)

        risk_actions = [payload["action"] for payload in _payloads(memory, "risk_decision")]
        assert {"allow", "block", "scale"}.issubset(set(risk_actions))

        rows = _canonical_rows(tmp_path)
        event_types = {row["event_type"] for row in rows}
        expected_event_types = {
            "deployment",
            "config_snapshot",
            "allocation_snapshot",
            "portfolio_snapshot",
            "position_snapshot",
            "decision_event",
            "portfolio_rule",
            "risk_decision",
            "order",
            "fill",
            "trade",
            "missed_opportunity",
            "heartbeat",
            "pipeline_funnel",
            "daily_snapshot",
            "family_daily_snapshot",
            "daily_reconciliation",
            "reconciliation_event",
        }
        assert expected_event_types.issubset(event_types)
        for row in rows:
            encoded = json.dumps(row, sort_keys=True, default=str)
            assert row["schema_version"] == "assistant_event_v1"
            assert row["event_type"]
            assert isinstance(row["lineage"], dict)
            payload = row["payload"]
            for key in (
                "event_id",
                "logical_event_id",
                "event_type",
                "bot_id",
                "family_id",
                "portfolio_id",
                "account_alias",
                "strategy_id",
                "assistant_strategy_id",
                "deployment_id",
                "config_version",
                "code_sha",
            ):
                assert payload[key]
            for key in (
                "event_id",
                "logical_event_id",
                "event_type",
                "bot_id",
                "family_id",
                "portfolio_id",
                "account_alias",
                "strategy_id",
                "assistant_strategy_id",
                "deployment_id",
                "code_sha",
            ):
                assert payload[key] == row[key]
            if row["event_type"] == "config_snapshot":
                assert payload["metadata"]["config_version"] == row["config_version"]
            else:
                assert payload["config_version"] == row["config_version"]
            if row["event_type"] in {"portfolio_rule", "risk_decision"}:
                for key in (
                    "portfolio_rule_event_id",
                    "risk_decision_id",
                    "intent_id",
                    "client_order_id",
                    "order_id",
                ):
                    assert payload[key]
            if row["event_type"] == "order" and payload.get("client_order_id"):
                assert payload["order_id"]
            if row["event_type"] == "fill":
                assert payload["fill_id"]
                assert payload["client_order_id"]
                assert payload["order_id"]
            assert "private_key" not in encoded
            assert "relay_secret" not in encoded
            assert "postgres_dsn" not in encoded
            assert "0x" not in encoded

        forwarder = SidecarForwarder(tmp_path, "http://localhost:8000", "synthetic_bot", "secret")
        relay = RelayStore(tmp_path / "relay.sqlite3")
        try:
            inserted = 0
            sent_batches: list[tuple[str, int, bool]] = []

            class _Response:
                status = 202

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            def fake_urlopen(request, timeout: float = 0):
                nonlocal inserted
                headers = {key.lower(): value for key, value in request.header_items()}
                body = request.data or b""
                compressed = headers.get("content-encoding") == "gzip"
                canonical = gzip.decompress(body) if compressed else body
                expected_signature = hmac.new(
                    b"secret",
                    canonical,
                    hashlib.sha256,
                ).hexdigest()
                assert timeout == 10
                assert request.full_url == "http://localhost:8000/events"
                assert headers["x-bot-id"] == "synthetic_bot"
                assert headers["x-signature"] == expected_signature

                payload = json.loads(canonical.decode("utf-8"))
                assert payload["bot_id"] == "synthetic_bot"
                assert payload["event_type"]
                assert payload["events"]
                assert all(
                    event["schema_version"] == "assistant_event_v1"
                    for event in payload["events"]
                )
                sent_batches.append((payload["event_type"], len(payload["events"]), compressed))
                inserted += relay.insert_events(
                    payload["bot_id"],
                    payload["event_type"],
                    payload["events"],
                )
                return _Response()

            monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
            forwarder._poll_once()
            health = relay.get_health()
        finally:
            relay.close()

        assert sent_batches
        assert any(compressed for _, _, compressed in sent_batches)
        assert forwarder._watermarks
        assert inserted >= len(expected_event_types)
        assert expected_event_types.issubset(set(health["event_type_counts"]))
    finally:
        engine._oms.close()


def _canonical_rows(state_dir: Path) -> list[dict]:
    root = state_dir / "instrumentation" / "events"
    rows: list[dict] = []
    for path in sorted(root.glob("*/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows
