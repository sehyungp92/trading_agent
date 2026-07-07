"""Strategy engine: unified run loop for backtesting and live trading."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import structlog

from crypto_trader.core.broker import BrokerAdapter
from crypto_trader.core.clock import Clock
from crypto_trader.core.events import EventBus
from crypto_trader.core.execution_gateway import ExecutionGateway
from crypto_trader.core.models import Bar, Fill, TerminalMark, TimeFrame
from crypto_trader.broker.sim_execution_adapter import SimExecutionAdapter
from crypto_trader.core.strategy_runtime import StrategySlotRuntime

log = structlog.get_logger()


@runtime_checkable
class Strategy(Protocol):
    """Protocol that all strategies must implement."""

    @property
    def name(self) -> str: ...

    @property
    def symbols(self) -> list[str]: ...

    @property
    def timeframes(self) -> list[TimeFrame]: ...

    def on_init(self, ctx: StrategyContext) -> None: ...
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None: ...
    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None: ...
    def on_shutdown(self, ctx: StrategyContext) -> None: ...


class MultiTimeFrameBars:
    """Rolling window of bars per (symbol, timeframe) pair."""

    def __init__(self, max_bars: int = 500) -> None:
        self._max_bars = max_bars
        self._bars: dict[tuple[str, TimeFrame], deque[Bar]] = defaultdict(
            lambda: deque(maxlen=max_bars)
        )

    def append(self, bar: Bar) -> None:
        self._bars[(bar.symbol, bar.timeframe)].append(bar)

    def get(
        self,
        symbol: str,
        tf: TimeFrame,
        count: int | None = None,
    ) -> list[Bar]:
        buf = self._bars.get((symbol, tf))
        if buf is None:
            return []
        if count is None:
            return list(buf)
        return list(buf)[-count:]

    def latest(self, symbol: str, tf: TimeFrame) -> Bar | None:
        buf = self._bars.get((symbol, tf))
        if not buf:
            return None
        return buf[-1]

    def snapshot_state(self) -> dict[str, Any]:
        """Return an in-memory checkpoint of rolling bar buffers."""
        return {
            "max_bars": self._max_bars,
            "bars": {
                key: list(buffer)
                for key, buffer in self._bars.items()
            },
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        """Restore bar buffers captured by :meth:`snapshot_state`."""
        self._max_bars = int(snapshot.get("max_bars", self._max_bars))
        restored: dict[tuple[str, TimeFrame], deque[Bar]] = defaultdict(
            lambda: deque(maxlen=self._max_bars)
        )
        for key, bars in snapshot.get("bars", {}).items():
            restored[key] = deque(deepcopy(bars), maxlen=self._max_bars)
        self._bars = restored


@dataclass
class StrategyContext:
    """Injected into strategy callbacks and runtime services."""

    broker: BrokerAdapter
    clock: Clock
    bars: MultiTimeFrameBars
    events: EventBus
    config: Any = None


class StrategyEngine:
    """Drive a single strategy over a feed through the shared slot runtime."""

    def __init__(
        self,
        strategy: Strategy,
        broker: BrokerAdapter,
        feed: Any,
        clock: Clock,
        events: EventBus | None = None,
        config: Any = None,
        primary_timeframe: TimeFrame = TimeFrame.M15,
    ) -> None:
        self.strategy = strategy
        self.broker = broker
        self.feed = feed
        self.clock = clock
        self.events = events or EventBus()
        self.config = config
        self.primary_timeframe = primary_timeframe
        self.execution_gateway = ExecutionGateway(
            adapter=SimExecutionAdapter(self.broker),
            broker=self.broker,
            events=self.events,
        )

        self._bars = MultiTimeFrameBars()
        self._ctx = StrategyContext(
            broker=self.execution_gateway,
            clock=self.clock,
            bars=self._bars,
            events=self.events,
            config=self.config,
        )
        self._runtime = StrategySlotRuntime(
            strategy=self.strategy,
            ctx=self._ctx,
            broker=self.broker,
            bars=self._bars,
            events=self.events,
            primary_timeframe=self.primary_timeframe,
            strategy_id=getattr(self.strategy, "name", "strategy"),
        )
        self._bar_count = 0
        self._fill_count = 0

    def run(self) -> None:
        """Execute the full strategy run loop."""
        log.info(
            "engine.start",
            strategy=self.strategy.name,
            symbols=self.strategy.symbols,
            primary_tf=self.primary_timeframe.value,
        )

        self.strategy.on_init(self._ctx)

        for bar in self.feed:
            self._process_single_bar(bar)

        self.strategy.on_shutdown(self._ctx)

        log.info(
            "engine.complete",
            bars_processed=self._bar_count,
            fills=self._fill_count,
        )

    def _process_single_bar(self, bar: Bar) -> None:
        """Process a single bar through the shared slot runtime."""
        self._runtime.process_bar(bar)
        self._bar_count = self._runtime.bar_count
        self._fill_count = self._runtime.fill_count

    def close_open_positions(self) -> list[Fill]:
        """Force-close positions while dispatching fills and close events."""
        fills = self._runtime.close_open_positions()
        self._fill_count = self._runtime.fill_count
        return fills

    def mark_open_positions(self) -> list[TerminalMark]:
        """Create explicit terminal marks and let the strategy enrich them."""
        return self._runtime.mark_open_positions()

    def snapshot_state(self) -> dict[str, Any]:
        """Return an in-memory checkpoint suitable for split-run continuation."""
        strategy_snapshot = None
        snapshot_strategy = getattr(self.strategy, "snapshot_state", None)
        if callable(snapshot_strategy):
            strategy_snapshot = snapshot_strategy()

        return {
            "broker": self.broker.snapshot_state()
            if hasattr(self.broker, "snapshot_state")
            else None,
            "clock": self.clock.snapshot_state()
            if hasattr(self.clock, "snapshot_state")
            else None,
            "bars": self._bars.snapshot_state(),
            "runtime": {
                "bar_count": self._runtime.bar_count,
                "fill_count": self._runtime.fill_count,
                "engine_bar_count": self._bar_count,
                "engine_fill_count": self._fill_count,
            },
            "strategy": strategy_snapshot,
            "execution_gateway": {
                "last_reports": deepcopy(getattr(self.execution_gateway, "_last_reports", [])),
                "pending_immediate_fill_syncs": deepcopy(
                    getattr(self.execution_gateway, "_pending_immediate_fill_syncs", [])
                ),
            },
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        """Restore an engine checkpoint before continuing with a new feed."""
        broker_snapshot = snapshot.get("broker")
        if broker_snapshot is not None and hasattr(self.broker, "restore_state"):
            self.broker.restore_state(broker_snapshot)

        clock_snapshot = snapshot.get("clock")
        if clock_snapshot is not None and hasattr(self.clock, "restore_state"):
            self.clock.restore_state(clock_snapshot)

        bars_snapshot = snapshot.get("bars")
        if bars_snapshot is not None:
            self._bars.restore_state(bars_snapshot)

        strategy_snapshot = snapshot.get("strategy")
        restore_strategy = getattr(self.strategy, "restore_state", None)
        if strategy_snapshot is not None and callable(restore_strategy):
            restore_strategy(strategy_snapshot)

        runtime = snapshot.get("runtime", {})
        self._runtime.bar_count = int(runtime.get("bar_count", self._runtime.bar_count))
        self._runtime.fill_count = int(runtime.get("fill_count", self._runtime.fill_count))
        self._bar_count = int(runtime.get("engine_bar_count", self._runtime.bar_count))
        self._fill_count = int(runtime.get("engine_fill_count", self._runtime.fill_count))

        gateway_snapshot = snapshot.get("execution_gateway", {})
        if gateway_snapshot:
            self.execution_gateway._last_reports = deepcopy(
                gateway_snapshot.get("last_reports", [])
            )
            self.execution_gateway._pending_immediate_fill_syncs = deepcopy(
                gateway_snapshot.get("pending_immediate_fill_syncs", [])
            )

        sync_open_orders = getattr(self.execution_gateway.adapter, "sync_open_orders", None)
        if callable(sync_open_orders):
            sync_open_orders()
