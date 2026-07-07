"""Runtime replay parity between backtest-style and mocked-live dispatch."""

from datetime import datetime, timezone

from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.core.clock import SimClock
from crypto_trader.core.engine import MultiTimeFrameBars, StrategyContext, StrategyEngine
from crypto_trader.core.events import CanonicalRuntimeEvent, EventBus
from crypto_trader.core.execution_adapter import ExecutionCapabilities
from crypto_trader.core.execution_gateway import ExecutionGateway
from crypto_trader.core.models import Bar, Order, OrderStatus, OrderType, Side, TimeFrame
from crypto_trader.core.runtime_types import ExecutionReport, ExecutionReportKind, MarketEvent, TimestampPolicy
from crypto_trader.core.strategy_runtime import StrategySlotRuntime


class ReplayStrategy:
    name = "replay"
    symbols = ["BTC"]
    timeframes = [TimeFrame.M15]

    def on_init(self, ctx) -> None:
        pass

    def on_bar(self, bar, ctx) -> None:
        if bar.close > 100:
            ctx.broker.submit_order(Order(
                order_id="",
                symbol=bar.symbol,
                side=Side.LONG,
                order_type=OrderType.LIMIT,
                qty=0.1,
                limit_price=bar.close,
                tag="entry",
                metadata={"strategy_id": "replay"},
            ))

    def on_fill(self, fill, ctx) -> None:
        pass

    def on_shutdown(self, ctx) -> None:
        pass


class MockLiveAdapter:
    capabilities = ExecutionCapabilities()

    def submit(self, intent):
        client_id = intent.client_order_id or intent.intent_id
        return [ExecutionReport(
            report_id=f"live_accept_{client_id}",
            kind=ExecutionReportKind.ACCEPTED,
            timestamp=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
            symbol=intent.symbol,
            side=intent.side,
            client_order_id=client_id,
            order_status=OrderStatus.WORKING,
            qty=intent.qty,
            metadata=intent.metadata,
        )]

    def cancel(self, client_order_id):
        return []

    def sync_open_orders(self):
        return []

    def sync_positions(self):
        return []

    def sync_fills(self, watermark):
        return []


def _event(close: float, minute: int) -> MarketEvent:
    bar = Bar(
        timestamp=datetime(2026, 5, 24, 12, minute, tzinfo=timezone.utc),
        symbol="BTC",
        open=99.0,
        high=101.0,
        low=98.0,
        close=close,
        volume=1_000.0,
        timeframe=TimeFrame.M15,
    )
    return MarketEvent.from_bar(bar, source="fixture", timestamp_policy=TimestampPolicy.OPEN_TIME)


def _run_backtest_style(events: list[MarketEvent]) -> list[dict]:
    bus = EventBus()
    canonical = []
    bus.subscribe(CanonicalRuntimeEvent, canonical.append)
    broker = SimBroker(initial_equity=10_000.0)
    engine = StrategyEngine(
        strategy=ReplayStrategy(),
        broker=broker,
        feed=[event.to_bar() for event in events],
        clock=SimClock(),
        events=bus,
        primary_timeframe=TimeFrame.M15,
    )
    engine.run()
    return _decision_and_order_payloads(canonical)


def _run_live_style(events: list[MarketEvent]) -> list[dict]:
    bus = EventBus()
    canonical = []
    bus.subscribe(CanonicalRuntimeEvent, canonical.append)
    broker = object()
    gateway = ExecutionGateway(
        adapter=MockLiveAdapter(),
        broker=broker,
        events=bus,
    )
    bars = MultiTimeFrameBars()
    ctx = StrategyContext(
        broker=gateway,
        clock=SimClock(),
        bars=bars,
        events=bus,
    )
    runtime = StrategySlotRuntime(
        strategy=ReplayStrategy(),
        ctx=ctx,
        broker=broker,
        bars=bars,
        events=bus,
        primary_timeframe=TimeFrame.M15,
        strategy_id="replay",
    )
    for event in events:
        runtime.process_bar(event, process_broker=False)
    return _decision_and_order_payloads(canonical)


def _decision_and_order_payloads(canonical) -> list[dict]:
    return [
        _normalize_payload(event.payload)
        for event in canonical
        if event.stream in {"decision", "order_intent"}
    ]


def _normalize_payload(payload: dict) -> dict:
    if "decision_id" in payload and "intent_id" not in payload:
        return {
            "stream": "decision",
            "decision_id": payload["decision_id"],
            "strategy_id": payload["strategy_id"],
            "symbol": payload["symbol"],
            "timeframe": payload["timeframe"],
            "action": payload["action"],
        }
    return {
        "stream": "order_intent",
        "intent_id": payload["intent_id"],
        "decision_id": payload["decision_id"],
        "strategy_id": payload["strategy_id"],
        "symbol": payload["symbol"],
        "side": payload["side"],
        "order_type": payload["order_type"],
        "qty": payload["qty"],
        "tag": payload.get("metadata", {}).get("tag"),
    }


def test_backtest_and_mocked_live_replay_emit_same_decisions_and_intents() -> None:
    events = [_event(99.0, 0), _event(101.0, 15)]

    assert _run_backtest_style(events) == _run_live_style(events)
