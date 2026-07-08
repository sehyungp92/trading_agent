"""Shared strategy runtime parity contract tests."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from crypto_trader.core.clock import SimClock
from crypto_trader.core.engine import MultiTimeFrameBars, StrategyContext
from crypto_trader.core.events import CanonicalRuntimeEvent, EventBus
from crypto_trader.core.models import Bar, Fill, Side, TimeFrame
from crypto_trader.core.runtime_types import MarketEvent, TimestampPolicy
from crypto_trader.core.strategy_runtime import StrategySlotRuntime


def _bar(tf: TimeFrame = TimeFrame.M15) -> Bar:
    return Bar(
        timestamp=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
        symbol="BTC",
        open=100.0,
        high=110.0,
        low=95.0,
        close=105.0,
        volume=1_000.0,
        timeframe=tf,
    )


@dataclass
class _Strategy:
    calls: list[str]

    @property
    def name(self) -> str:
        return "test"

    @property
    def symbols(self) -> list[str]:
        return ["BTC"]

    @property
    def timeframes(self) -> list[TimeFrame]:
        return [TimeFrame.M15]

    def on_bar(self, bar, ctx) -> None:
        self.calls.append(f"bar:{bar.timeframe.value}")

    def on_fill(self, fill, ctx) -> None:
        self.calls.append(f"fill:{fill.order_id}")


class _Broker:
    def process_bar(self, bar):
        return [Fill(
            order_id="o1",
            symbol="BTC",
            side=Side.LONG,
            qty=0.1,
            fill_price=100.0,
            commission=0.01,
            timestamp=bar.timestamp,
            tag="entry",
        )]


def test_strategy_runtime_dispatches_fills_before_bar_and_emits_canonical_events() -> None:
    events = EventBus()
    canonical = []
    events.subscribe(CanonicalRuntimeEvent, canonical.append)
    strategy = _Strategy(calls=[])
    bars = MultiTimeFrameBars()
    ctx = StrategyContext(
        broker=_Broker(),
        clock=SimClock(),
        bars=bars,
        events=events,
    )
    runtime = StrategySlotRuntime(
        strategy=strategy,
        ctx=ctx,
        broker=ctx.broker,
        bars=bars,
        events=events,
        primary_timeframe=TimeFrame.M15,
    )

    runtime.process_bar(_bar())

    assert strategy.calls == ["fill:o1", "bar:15m"]
    assert ctx.clock.now() == _bar().timestamp + timedelta(minutes=15)
    assert [event.stream for event in canonical] == ["execution", "market", "decision"]
    assert canonical[1].timestamp == _bar().timestamp + timedelta(minutes=15)
    assert canonical[1].payload["bar_id"] == canonical[-1].payload["bar_id"]
    assert canonical[1].payload["decision_id"] == canonical[-1].payload["decision_id"]
    assert canonical[-1].payload["action"] == "no_order"


def test_strategy_runtime_accepts_market_events_without_changing_callbacks() -> None:
    canonical = []
    strategy = _Strategy(calls=[])
    bars = MultiTimeFrameBars()
    events = EventBus()
    events.subscribe(CanonicalRuntimeEvent, canonical.append)
    ctx = StrategyContext(
        broker=object(),
        clock=SimClock(),
        bars=bars,
        events=events,
    )
    runtime = StrategySlotRuntime(
        strategy=strategy,
        ctx=ctx,
        broker=ctx.broker,
        bars=bars,
        events=events,
        primary_timeframe=TimeFrame.M15,
    )

    event = MarketEvent.from_bar(
        _bar(),
        source="historical",
        timestamp_policy=TimestampPolicy.OPEN_TIME,
    )
    runtime.process_bar(event, process_broker=False)

    assert strategy.calls == ["bar:15m"]
    assert ctx.clock.now() == event.available_at
    assert bars.latest("BTC", TimeFrame.M15).timestamp == event.open_time
    assert canonical[0].payload["source"] == "historical"
    assert canonical[0].payload["bar_id"] == canonical[1].payload["bar_id"]


def test_higher_timeframe_deferral_stops_when_strategy_raises() -> None:
    class RaisingStrategy(_Strategy):
        def on_bar(self, bar, ctx) -> None:
            raise RuntimeError("boom")

    class DeferringBroker:
        def __init__(self) -> None:
            self.starts = 0
            self.stops = 0

        def start_deferring(self) -> None:
            self.starts += 1

        def stop_deferring(self) -> None:
            self.stops += 1

    broker = DeferringBroker()
    strategy = RaisingStrategy(calls=[])
    bars = MultiTimeFrameBars()
    ctx = StrategyContext(
        broker=broker,
        clock=SimClock(),
        bars=bars,
        events=EventBus(),
    )
    runtime = StrategySlotRuntime(
        strategy=strategy,
        ctx=ctx,
        broker=broker,
        bars=bars,
        events=ctx.events,
        primary_timeframe=TimeFrame.M15,
    )

    with pytest.raises(RuntimeError):
        runtime.process_bar(_bar(TimeFrame.H1), process_broker=False)

    assert broker.starts == 1
    assert broker.stops == 1
