"""Shared strategy slot runtime for backtest, portfolio, and live loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from crypto_trader.core.events import (
    BarEvent,
    CanonicalRuntimeEvent,
    EventBus,
    FillEvent,
    PositionClosedEvent,
)
from crypto_trader.core.models import Bar, Fill, OrderStatus, TerminalMark, TimeFrame, Trade
from crypto_trader.core.runtime_types import (
    DecisionContext,
    ExecutionReport,
    ExecutionReportKind,
    MarketEvent,
    TimestampPolicy,
    TradeOutcome,
)


@dataclass(slots=True)
class StrategyRuntimeCallbacks:
    """Optional integration hooks around the canonical strategy lifecycle."""

    on_fill: Callable[[Fill], None] | None = None
    on_trade_closed: Callable[[Trade], None] | None = None
    before_strategy_bar: Callable[[Bar], None] | None = None
    after_strategy_bar: Callable[[Bar], None] | None = None


class StrategySlotRuntime:
    """Drive one strategy slot through the shared bar/fill lifecycle."""

    def __init__(
        self,
        *,
        strategy: Any,
        ctx: Any,
        broker: Any,
        bars: Any,
        events: EventBus,
        primary_timeframe: TimeFrame,
        strategy_id: str | None = None,
        callbacks: StrategyRuntimeCallbacks | None = None,
    ) -> None:
        self.strategy = strategy
        self.ctx = ctx
        self.broker = broker
        self.bars = bars
        self.events = events
        self.primary_timeframe = primary_timeframe
        self.strategy_id = strategy_id or getattr(strategy, "name", "unknown")
        self.callbacks = callbacks or StrategyRuntimeCallbacks()
        self.bar_count = 0
        self.fill_count = 0

    def process_bar(
        self,
        bar: Bar | MarketEvent,
        *,
        process_broker: bool = True,
        advance_clock: bool = True,
    ) -> None:
        """Process one normalized bar without changing strategy callback APIs."""
        market_event = bar if isinstance(bar, MarketEvent) else None
        if isinstance(bar, MarketEvent):
            bar = bar.to_bar()
        else:
            market_event = MarketEvent.from_bar(
                bar,
                source="runtime",
                timestamp_policy=TimestampPolicy.OPEN_TIME,
            )

        if advance_clock and hasattr(self.ctx.clock, "advance"):
            self.ctx.clock.advance(market_event.available_at)

        if bar.timeframe == self.primary_timeframe:
            self._process_primary_bar(
                bar,
                process_broker=process_broker,
                market_event=market_event,
            )
        else:
            self._process_higher_timeframe_bar(bar, market_event=market_event)

    def dispatch_fill(self, fill: Fill, *, notify_callback: bool = True) -> None:
        """Dispatch a fill through integration hooks, strategy, and events."""
        if notify_callback and self.callbacks.on_fill is not None:
            self.callbacks.on_fill(fill)
        self.fill_count += 1
        self.strategy.on_fill(fill, self.ctx)
        self.events.emit(FillEvent(timestamp=fill.timestamp, fill=fill))
        self.events.emit(CanonicalRuntimeEvent(
            timestamp=fill.timestamp,
            stream="execution",
            payload=ExecutionReport(
                report_id=f"runtime_fill_{fill.order_id}_{int(fill.timestamp.timestamp() * 1000)}",
                kind=ExecutionReportKind.FILL,
                timestamp=fill.timestamp,
                symbol=fill.symbol,
                side=fill.side,
                client_order_id=fill.order_id,
                exchange_order_id=fill.order_id,
                fill_id=f"{fill.order_id}:{int(fill.timestamp.timestamp() * 1000)}",
                order_status=OrderStatus.FILLED,
                filled_qty=fill.qty,
                fill_price=fill.fill_price,
                commission=fill.commission,
                metadata={"tag": fill.tag},
            ).to_dict(),
        ))

    def close_open_positions(self) -> list[Fill]:
        """Force-close open positions while preserving fill/close callbacks."""
        close_fn = getattr(self.broker, "close_open_positions", None)
        if close_fn is None:
            return []

        closed_before = len(getattr(self.broker, "_closed_trades", []))
        fills = close_fn()
        for fill in fills:
            self.dispatch_fill(fill)
        self._emit_new_closed_trades(closed_before)
        return fills

    def mark_open_positions(self) -> list[TerminalMark]:
        """Create terminal marks and let strategies enrich them when supported."""
        mark_fn = getattr(self.broker, "mark_open_positions", None)
        if mark_fn is None:
            return []

        terminal_marks = mark_fn()
        enrich_fn = getattr(self.strategy, "enrich_terminal_marks", None)
        if enrich_fn is not None and terminal_marks:
            enrich_fn(terminal_marks)
        return terminal_marks

    def _process_primary_bar(
        self,
        bar: Bar,
        *,
        process_broker: bool,
        market_event: MarketEvent | None = None,
    ) -> None:
        closed_before = len(getattr(self.broker, "_closed_trades", []))

        fills = self._try_process_bar(bar) if process_broker else []
        for fill in fills:
            self.dispatch_fill(fill)

        if fills:
            self._check_entry_bar_stops(bar)

        self._emit_new_closed_trades(closed_before)
        self._activate_deferred_orders()
        self._dispatch_bar_to_strategy(bar, market_event=market_event)

    def _process_higher_timeframe_bar(
        self,
        bar: Bar,
        *,
        market_event: MarketEvent | None = None,
    ) -> None:
        self.bars.append(bar)
        event = self._emit_market_event(bar, market_event=market_event)
        decision_context = self._decision_context(bar, event)

        start_fn = getattr(self.broker, "start_deferring", None)
        stop_fn = getattr(self.broker, "stop_deferring", None)
        if start_fn is not None:
            start_fn()

        try:
            self._begin_decision_context(decision_context)
            self.strategy.on_bar(bar, self.ctx)
            self.bar_count += 1
            self._drain_immediate_fill_syncs()
        finally:
            self._end_decision_context(decision_context)
            if stop_fn is not None:
                stop_fn()

        self._emit_decision_event(decision_context)
        self.events.emit(BarEvent(timestamp=bar.timestamp, bar=bar))

    def _check_entry_bar_stops(self, bar: Bar) -> None:
        check_fn = getattr(self.broker, "check_entry_bar_stops", None)
        if check_fn is None:
            return

        recheck_fills = check_fn(bar)
        for fill in recheck_fills:
            self.dispatch_fill(fill)
        if recheck_fills:
            refresh_fn = getattr(self.broker, "refresh_current_bar_equity", None)
            if refresh_fn is not None:
                refresh_fn(bar.timestamp)

    def _emit_new_closed_trades(self, closed_before: int) -> None:
        closed_trades = getattr(self.broker, "_closed_trades", [])
        for trade in closed_trades[closed_before:]:
            if self.callbacks.on_trade_closed is not None:
                self.callbacks.on_trade_closed(trade)
            self.events.emit(PositionClosedEvent(timestamp=trade.exit_time, trade=trade))
            self.events.emit(CanonicalRuntimeEvent(
                timestamp=trade.exit_time,
                stream="trade",
                payload=TradeOutcome.from_trade(trade).to_dict(),
            ))

    def _activate_deferred_orders(self) -> None:
        activate_fn = getattr(self.broker, "activate_deferred", None)
        if activate_fn is not None:
            activate_fn()

    def _dispatch_bar_to_strategy(
        self,
        bar: Bar,
        *,
        market_event: MarketEvent | None = None,
    ) -> None:
        if self.callbacks.before_strategy_bar is not None:
            self.callbacks.before_strategy_bar(bar)

        self.bars.append(bar)
        event = self._emit_market_event(bar, market_event=market_event)
        decision_context = self._decision_context(bar, event)
        try:
            self._begin_decision_context(decision_context)
            self.strategy.on_bar(bar, self.ctx)
            self.bar_count += 1
            self._drain_immediate_fill_syncs()
        finally:
            self._end_decision_context(decision_context)
        self._emit_decision_event(decision_context)
        self.events.emit(BarEvent(timestamp=bar.timestamp, bar=bar))

        if self.callbacks.after_strategy_bar is not None:
            self.callbacks.after_strategy_bar(bar)

    def _try_process_bar(self, bar: Bar) -> list[Fill]:
        process_bar = getattr(self.broker, "process_bar", None)
        if process_bar is None:
            return []
        return process_bar(bar)

    def _drain_immediate_fill_syncs(self) -> None:
        drain = getattr(self.ctx.broker, "drain_immediate_fill_syncs", None)
        if callable(drain):
            drain()

    def _emit_market_event(
        self,
        bar: Bar,
        *,
        market_event: MarketEvent | None = None,
    ) -> MarketEvent:
        event = market_event or MarketEvent.from_bar(
            bar,
            source="runtime",
            timestamp_policy=TimestampPolicy.OPEN_TIME,
        )
        bar_id = self._bar_id(bar, event)
        decision_id = self._decision_key(bar, event)
        payload = event.to_dict()
        payload.update({
            "strategy_id": self.strategy_id,
            "decision_id": decision_id,
            "bar_id": bar_id,
            "metadata": {
                **payload.get("metadata", {}),
                "strategy_id": self.strategy_id,
                "decision_id": decision_id,
                "bar_id": bar_id,
            },
        })
        self.events.emit(CanonicalRuntimeEvent(
            timestamp=event.available_at,
            stream="market",
            payload=payload,
        ))
        return event

    def _decision_context(self, bar: Bar, event: MarketEvent) -> DecisionContext:
        key = self._decision_key(bar, event)
        bar_id = self._bar_id(bar, event)
        return DecisionContext(
            decision_id=key,
            strategy_id=self.strategy_id,
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            decision_time=event.available_at,
            decision_key=key,
            metadata={
                "source": event.source,
                "bar_id": bar_id,
                "market_available_at": event.available_at.isoformat(),
            },
        )

    def _decision_key(self, bar: Bar, event: MarketEvent) -> str:
        return (
            f"{self.strategy_id}|{bar.symbol}|{bar.timeframe.value}|"
            f"{event.available_at.isoformat()}"
        )

    def _bar_id(self, bar: Bar, event: MarketEvent) -> str:
        return (
            f"{self.strategy_id}:{bar.symbol}:{bar.timeframe.value}:"
            f"{event.available_at.isoformat()}"
        )

    def _begin_decision_context(self, context: DecisionContext) -> None:
        begin_fn = getattr(self.ctx.broker, "begin_decision_context", None)
        if begin_fn is not None:
            begin_fn(context)
        collector = getattr(self.strategy, "_collector", None)
        begin_collector = getattr(collector, "begin_decision_context", None)
        if callable(begin_collector):
            begin_collector(context)

    def _end_decision_context(self, context: DecisionContext) -> None:
        end_fn = getattr(self.ctx.broker, "end_decision_context", None)
        if end_fn is not None:
            end_fn(context)
        collector = getattr(self.strategy, "_collector", None)
        end_collector = getattr(collector, "end_decision_context", None)
        if callable(end_collector):
            end_collector(context)

    def _emit_decision_event(self, context: DecisionContext) -> None:
        event = context.to_decision_event()
        self.events.emit(CanonicalRuntimeEvent(
            timestamp=event.decision_time,
            stream="decision",
            payload=event.to_dict(),
        ))
