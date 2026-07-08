"""Portfolio backtest runner — multi-strategy backtest with per-strategy brokers and portfolio rules.

Each strategy gets its own SimBroker (isolated positions/margin), matching individual
backtest behavior. The portfolio manager provides cross-strategy coordination rules.
The only trade-count difference from individual runs should come from portfolio blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import heapq

import structlog

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.metrics import PerformanceMetrics, compute_metrics
from crypto_trader.backtest.runner import _create_strategy
from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.broker.sim_execution_adapter import SimExecutionAdapter
from crypto_trader.core.clock import SimClock
from crypto_trader.core.engine import MultiTimeFrameBars, StrategyContext
from crypto_trader.core.events import EventBus
from crypto_trader.core.execution_gateway import ExecutionGateway
from crypto_trader.core.models import Bar, TerminalMark, TimeFrame, Trade
from crypto_trader.core.runtime_types import MarketEvent
from crypto_trader.core.strategy_runtime import (
    StrategyRuntimeCallbacks,
    StrategySlotRuntime,
)
from crypto_trader.data.historical_feed import HistoricalFeed, _TF_PRIORITY
from crypto_trader.data.store import ParquetStore
from crypto_trader.exchange.funding import FundingHelper
from crypto_trader.exchange.meta import AssetMeta
from crypto_trader.instrumentation.lineage import (
    ALLOCATION_CONFIG_KEYS,
    RISK_CONFIG_KEYS,
    stable_hash,
    subset_keys,
)
from crypto_trader.portfolio.config import PortfolioConfig
from crypto_trader.portfolio.coordinator import StrategyCoordinator
from crypto_trader.portfolio.manager import PortfolioManager
from crypto_trader.portfolio.state import PortfolioState

log = structlog.get_logger()

# Strategy-specific warmup requirements (in days)
_STRATEGY_WARMUP = {
    "momentum": 0,    # 200 M15 bars ≈ 2 days, handled by bar count
    "trend": 60,      # D1 EMA50 needs 51 bars
    "breakout": 0,    # 101 M30 bars ≈ 2 days, handled by bar count
}


@dataclass
class RuleEvent:
    """Backtest portfolio rule event using the live assistant schema."""

    timestamp: datetime
    event_type: str = "portfolio_rule"
    strategy_id: str = ""
    symbol: str = ""
    direction: str = ""
    risk_R: float = 0.0
    approved: bool = False
    denial_reason: str | None = None
    size_multiplier: float = 1.0
    portfolio_rule_event_id: str = ""
    rule_event_id: str = ""
    risk_decision_id: str = ""
    rule_evaluation_id: str = ""
    decision_id: str = ""
    bar_id: str = ""
    intent_id: str = ""
    client_order_id: str = ""
    requested_risk_R: float = 0.0
    adjusted_risk_R: float = 0.0
    action: str = "block"
    blocking_rule: str = ""
    rule_evaluations: list[dict] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    state_before: dict = field(default_factory=dict)
    state_after_preview: dict = field(default_factory=dict)
    allocation: dict = field(default_factory=dict)
    request: dict = field(default_factory=dict)
    portfolio_config: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)
    portfolio_config_version: str = ""
    risk_config_version: str = ""
    allocation_version: str = ""
    payload: dict = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        timestamp: datetime,
        portfolio_config: dict[str, Any],
        portfolio_config_version: str,
        risk_config_version: str,
        allocation_version: str,
    ) -> "RuleEvent":
        strategy_id = str(payload.get("strategy_id") or "")
        requested_risk = float(payload.get("requested_risk_R") or 0.0)
        lineage = {
            "source": "portfolio_backtest",
            "strategy_id": strategy_id,
            "portfolio_config_version": portfolio_config_version,
            "risk_config_version": risk_config_version,
            "allocation_version": allocation_version,
        }
        portfolio_rule_event_id = str(payload.get("portfolio_rule_event_id") or payload.get("rule_event_id") or "")
        return cls(
            timestamp=timestamp,
            event_type=str(payload.get("event_type") or "portfolio_rule"),
            strategy_id=strategy_id,
            symbol=str(payload.get("symbol") or ""),
            direction=str(payload.get("direction") or payload.get("side") or ""),
            risk_R=requested_risk,
            approved=bool(payload.get("approved")),
            denial_reason=payload.get("denial_reason"),
            size_multiplier=float(payload.get("size_multiplier") or 1.0),
            portfolio_rule_event_id=portfolio_rule_event_id,
            rule_event_id=portfolio_rule_event_id,
            risk_decision_id=str(payload.get("risk_decision_id") or ""),
            rule_evaluation_id=str(payload.get("rule_evaluation_id") or ""),
            decision_id=str(payload.get("decision_id") or ""),
            bar_id=str(payload.get("bar_id") or ""),
            intent_id=str(payload.get("intent_id") or ""),
            client_order_id=str(payload.get("client_order_id") or ""),
            requested_risk_R=requested_risk,
            adjusted_risk_R=float(payload.get("adjusted_risk_R") or 0.0),
            action=str(payload.get("action") or ("allow" if payload.get("approved") else "block")),
            blocking_rule=str(payload.get("blocking_rule") or ""),
            rule_evaluations=list(payload.get("rule_evaluations") or payload.get("evaluations") or []),
            evaluations=list(payload.get("evaluations") or payload.get("rule_evaluations") or []),
            state_before=dict(payload.get("state_before") or {}),
            state_after_preview=dict(payload.get("state_after_preview") or {}),
            allocation=dict(payload.get("allocation") or {}),
            request=dict(payload.get("request") or {}),
            portfolio_config=dict(portfolio_config),
            lineage=lineage,
            portfolio_config_version=portfolio_config_version,
            risk_config_version=risk_config_version,
            allocation_version=allocation_version,
            payload={
                **payload,
                "portfolio_config": portfolio_config,
                "lineage": lineage,
                "portfolio_config_version": portfolio_config_version,
                "risk_config_version": risk_config_version,
                "allocation_version": allocation_version,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "portfolio_rule_event_id": self.portfolio_rule_event_id,
            "rule_event_id": self.rule_event_id,
            "risk_decision_id": self.risk_decision_id,
            "rule_evaluation_id": self.rule_evaluation_id,
            "decision_id": self.decision_id,
            "bar_id": self.bar_id,
            "intent_id": self.intent_id,
            "client_order_id": self.client_order_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "requested_risk_R": self.requested_risk_R,
            "risk_R": self.risk_R,
            "approved": self.approved,
            "action": self.action,
            "denial_reason": self.denial_reason,
            "blocking_rule": self.blocking_rule,
            "size_multiplier": self.size_multiplier,
            "adjusted_risk_R": self.adjusted_risk_R,
            "state_before": dict(self.state_before),
            "state_after_preview": dict(self.state_after_preview),
            "rule_evaluations": list(self.rule_evaluations),
            "evaluations": list(self.evaluations),
            "allocation": dict(self.allocation),
            "request": dict(self.request),
            "portfolio_config": dict(self.portfolio_config),
            "lineage": dict(self.lineage),
            "portfolio_config_version": self.portfolio_config_version,
            "risk_config_version": self.risk_config_version,
            "allocation_version": self.allocation_version,
        }


@dataclass
class BacktestEvidenceEvent:
    """First-class backtest evidence event using the live assistant payload shape."""

    event_type: str
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        event_type: str,
        payload: dict[str, Any],
        *,
        timestamp: datetime,
        portfolio_config: dict[str, Any],
        portfolio_config_version: str,
        risk_config_version: str,
        allocation_version: str,
    ) -> "BacktestEvidenceEvent":
        lineage = {
            "source": "portfolio_backtest",
            "strategy_id": str(payload.get("strategy_id") or ""),
            "portfolio_config_version": portfolio_config_version,
            "risk_config_version": risk_config_version,
            "allocation_version": allocation_version,
        }
        enriched = {
            **payload,
            "event_type": event_type,
            "timestamp": payload.get("timestamp") or timestamp.isoformat(),
            "portfolio_config": portfolio_config,
            "lineage": lineage,
            "portfolio_config_version": portfolio_config_version,
            "risk_config_version": risk_config_version,
            "allocation_version": allocation_version,
        }
        return cls(event_type=event_type, timestamp=timestamp, payload=enriched)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass
class PortfolioBacktestResult:
    """Results from a portfolio backtest."""

    per_strategy_trades: dict[str, list[Trade]]
    all_trades: list[Trade]
    equity_curve: list[tuple[datetime, float]]
    metrics: PerformanceMetrics
    rule_events: list[RuleEvent]
    risk_decision_events: list[BacktestEvidenceEvent] = field(default_factory=list)
    order_events: list[BacktestEvidenceEvent] = field(default_factory=list)
    per_strategy_metrics: dict[str, PerformanceMetrics | None] = field(default_factory=dict)
    config: BacktestConfig | None = None
    portfolio_config: PortfolioConfig | None = None
    execution_mode: str = "shared_capital"
    terminal_marks: dict[str, list[TerminalMark]] = field(default_factory=dict)


@dataclass
class _StrategySlot:
    """Internal: wraps one strategy within the portfolio backtest."""

    strategy_id: str
    strategy: Any  # Strategy protocol
    broker: SimBroker  # per-strategy broker (isolated positions)
    ctx: StrategyContext
    bars: MultiTimeFrameBars
    subscribed_tfs: set[TimeFrame]
    primary_tf: TimeFrame
    feed_tfs: list[TimeFrame]
    feed: HistoricalFeed | None = None
    runtime: StrategySlotRuntime | None = None


def run_portfolio_backtest(
    portfolio_config: PortfolioConfig,
    strategy_configs: dict[str, Any],
    backtest_config: BacktestConfig,
    data_dir: Path = Path("data"),
    meta_path: Path | None = None,
    store: Any | None = None,
    execution_mode: str = "shared_capital",
    terminal_accounting_mode: str | None = None,
) -> PortfolioBacktestResult:
    """Run a multi-strategy portfolio backtest.

    Architecture: each strategy gets its own SimBroker for position isolation.
    Entry orders are intercepted by BrokerProxy → PortfolioManager for
    cross-strategy rule checks. The only trade-count difference from
    individual runs comes from portfolio blocks.

    Args:
        portfolio_config: Portfolio-level risk rules
        strategy_configs: {strategy_id: config_object} for each strategy
        backtest_config: Shared backtest parameters
        data_dir: Path to data directory
        meta_path: Path to asset metadata cache

    Returns:
        PortfolioBacktestResult with per-strategy and combined results
    """
    if execution_mode != "shared_capital":
        raise ValueError(
            f"Unsupported portfolio execution_mode={execution_mode!r}; "
            "only 'shared_capital' is official"
        )
    accounting_mode = terminal_accounting_mode or portfolio_config.terminal_accounting_mode
    if accounting_mode not in {"terminal_mark", "force_close"}:
        raise ValueError(
            f"Unsupported terminal_accounting_mode={accounting_mode!r}; "
            "expected 'terminal_mark' or 'force_close'"
        )

    symbols = backtest_config.symbols or ["BTC", "ETH", "SOL"]

    if store is None:
        store = ParquetStore(base_dir=data_dir)

    # Load asset meta
    asset_meta = None
    if meta_path and meta_path.exists():
        asset_meta = AssetMeta.from_cache(meta_path)

    # Load funding
    funding_helpers: dict[str, FundingHelper] = {}
    if backtest_config.apply_funding:
        for sym in symbols:
            df = store.load_funding(sym)
            if df is not None and not df.empty:
                funding_helpers[sym] = FundingHelper(df)

    # Compute warmup: max across all strategies
    max_warmup = max(
        _STRATEGY_WARMUP.get(sid, 0)
        for sid in strategy_configs
    )
    max_warmup = max(max_warmup, backtest_config.warmup_days)

    actual_start = backtest_config.start_date
    if max_warmup > 0 and actual_start is not None:
        warmup_start = actual_start - timedelta(days=max_warmup)
    else:
        warmup_start = actual_start

    clock = SimClock()

    # Create portfolio management (coordinator uses a dummy broker — not used for orders)
    state = PortfolioState(
        equity=portfolio_config.initial_equity,
        peak_equity=portfolio_config.initial_equity,
    )
    manager = PortfolioManager(config=portfolio_config, state=state)
    rule_events: list[RuleEvent] = []
    risk_decision_events: list[BacktestEvidenceEvent] = []
    order_events: list[BacktestEvidenceEvent] = []
    portfolio_config_payload = portfolio_config.to_dict()
    portfolio_config_version = stable_hash(portfolio_config_payload)
    risk_config_version = stable_hash(subset_keys(portfolio_config_payload, RISK_CONFIG_KEYS))
    allocation_version = stable_hash(subset_keys(portfolio_config_payload, ALLOCATION_CONFIG_KEYS))

    def _record_portfolio_event(event_type: str, payload: dict) -> None:
        timestamp = clock.now()
        if event_type == "portfolio_rule":
            rule_events.append(RuleEvent.from_payload(
                payload,
                timestamp=timestamp,
                portfolio_config=portfolio_config_payload,
                portfolio_config_version=portfolio_config_version,
                risk_config_version=risk_config_version,
                allocation_version=allocation_version,
            ))
            return
        if event_type == "risk_decision":
            risk_decision_events.append(BacktestEvidenceEvent.from_payload(
                event_type,
                payload,
                timestamp=timestamp,
                portfolio_config=portfolio_config_payload,
                portfolio_config_version=portfolio_config_version,
                risk_config_version=risk_config_version,
                allocation_version=allocation_version,
            ))
            return
        if event_type == "order":
            order_events.append(BacktestEvidenceEvent.from_payload(
                event_type,
                payload,
                timestamp=timestamp,
                portfolio_config=portfolio_config_payload,
                portfolio_config_version=portfolio_config_version,
                risk_config_version=risk_config_version,
                allocation_version=allocation_version,
            ))

    # Create a minimal broker reference for coordinator (only used for order owner lookup)
    # Each strategy has its own actual broker
    _coordinator_broker = SimBroker(initial_equity=0)
    coordinator = StrategyCoordinator(
        broker=_coordinator_broker,
        manager=manager,
        event_callback=_record_portfolio_event,
    )

    # Create strategy slots — each with its own SimBroker
    slots: list[_StrategySlot] = []

    for strategy_id, strategy_config in strategy_configs.items():
        alloc = portfolio_config.get_strategy(strategy_id)
        if alloc is None or not alloc.enabled:
            continue

        # Ensure symbols are set on strategy config
        strategy_config.symbols = symbols

        strategy, feed_tfs, primary_tf = _create_strategy(strategy_id, strategy_config)
        subscribed_tfs = set(feed_tfs)

        # Per-strategy broker (isolated positions, same initial equity)
        strategy_broker = SimBroker(
            initial_equity=portfolio_config.initial_equity,
            taker_fee_bps=backtest_config.taker_fee_bps,
            maker_fee_bps=backtest_config.maker_fee_bps,
            slippage_bps=backtest_config.slippage_bps,
            spread_bps=backtest_config.spread_bps,
            asset_meta=asset_meta,
            funding_helpers=funding_helpers if funding_helpers else None,
        )

        events = EventBus()
        bars = MultiTimeFrameBars()
        execution_gateway = ExecutionGateway(
            adapter=SimExecutionAdapter(strategy_broker),
            broker=strategy_broker,
            events=events,
        )

        proxy = coordinator.get_proxy(strategy_id, use_manager_equity=True)
        # Point proxy at this strategy's gateway (not the coordinator's dummy)
        proxy._broker = execution_gateway

        ctx = StrategyContext(
            broker=proxy,
            clock=clock,
            bars=bars,
            events=events,
            config=backtest_config,
        )

        slots.append(_StrategySlot(
            strategy_id=strategy_id,
            strategy=strategy,
            broker=strategy_broker,
            ctx=ctx,
            bars=bars,
            subscribed_tfs=subscribed_tfs,
            primary_tf=primary_tf,
            feed_tfs=feed_tfs,
        ))

    # Create per-strategy feeds (each with correct primary_timeframe)
    # Guarantees each strategy sees exactly the same bars as in individual mode
    for slot in slots:
        slot.feed = HistoricalFeed(
            store=store,
            symbols=symbols,
            timeframes=sorted(slot.feed_tfs, key=lambda tf: tf.minutes),
            start_date=warmup_start,
            end_date=backtest_config.end_date,
            primary_timeframe=slot.primary_tf,
        )

    # Set module-level reference for shared-capital equity callbacks.
    global _all_slots_ref
    _all_slots_ref = slots

    for slot in slots:
        def _on_trade_closed(trade: Trade, strategy_id: str = slot.strategy_id) -> None:
            pnl_R = trade.r_multiple if trade.r_multiple is not None else 0.0
            coordinator.on_trade_closed(strategy_id, trade.symbol, pnl_R, trade=trade)

        def _before_strategy_bar(_bar: Bar) -> None:
            total_equity = _portfolio_equity_from_slots(
                _all_slots_ref,
                manager.config.initial_equity,
            )
            manager.update_equity(total_equity)

        slot.runtime = StrategySlotRuntime(
            strategy=slot.strategy,
            ctx=slot.ctx,
            broker=slot.broker,
            bars=slot.bars,
            events=slot.ctx.events,
            primary_timeframe=slot.primary_tf,
            strategy_id=slot.strategy_id,
            callbacks=StrategyRuntimeCallbacks(
                on_fill=coordinator.on_fill,
                on_trade_closed=_on_trade_closed,
                before_strategy_bar=_before_strategy_bar,
            ),
        )

    # Init strategies
    for slot in slots:
        slot.strategy.on_init(slot.ctx)

    log.info(
        "portfolio_backtest.start",
        strategies=[s.strategy_id for s in slots],
        symbols=symbols,
        start=str(backtest_config.start_date),
        end=str(backtest_config.end_date),
    )

    # Determine measurement start for warmup filtering
    measurement_start = None
    if max_warmup > 0 and actual_start is not None:
        measurement_start = datetime.combine(
            actual_start, datetime.min.time(), tzinfo=timezone.utc
        )

    # Main loop — merged iteration from per-strategy feeds
    # Each strategy sees exactly the same bars as in individual mode
    slot_map = {s.strategy_id: s for s in slots}
    feed_iters: dict[str, object] = {}
    _bar_store: dict[int, Bar | MarketEvent] = {}
    _heap: list[tuple] = []
    _seq = 0

    for slot in slots:
        it = slot.feed.iter_market_events()
        feed_iters[slot.strategy_id] = it
        try:
            bar = next(it)
            _bar_store[_seq] = bar
            heapq.heappush(_heap, (*_market_sort_key(bar), _seq, slot.strategy_id))
            _seq += 1
        except StopIteration:
            pass

    while _heap:
        _, _, seq_id, sid = heapq.heappop(_heap)
        bar = _bar_store.pop(seq_id)
        slot = slot_map[sid]

        event_time = bar.available_at if isinstance(bar, MarketEvent) else (
            bar.timestamp + timedelta(minutes=bar.timeframe.minutes)
        )
        if hasattr(clock, "advance"):
            clock.advance(event_time)

        today = event_time.date()
        manager.maybe_reset_daily(today)

        assert slot.runtime is not None
        slot.runtime.process_bar(bar, advance_clock=False)

        # Push next bar from this feed
        it = feed_iters[sid]
        try:
            next_bar = next(it)
            _bar_store[_seq] = next_bar
            heapq.heappush(_heap, (*_market_sort_key(next_bar), _seq, sid))
            _seq += 1
        except StopIteration:
            pass

    terminal_marks: dict[str, list[TerminalMark]] = {}
    for slot in slots:
        assert slot.runtime is not None
        if accounting_mode == "force_close":
            slot.runtime.close_open_positions()
        else:
            marks = slot.runtime.mark_open_positions()
            if marks:
                terminal_marks[slot.strategy_id] = marks

    # Trim warmup and collect results
    per_strategy_trades: dict[str, list[Trade]] = {}
    per_strategy_metrics: dict[str, PerformanceMetrics | None] = {}
    all_trades: list[Trade] = []

    for slot in slots:
        broker = slot.broker

        # Trim warmup from this broker's equity
        if measurement_start is not None:
            broker._equity_history = [
                (ts, eq) for ts, eq in broker._equity_history
                if ts >= measurement_start
            ]
            liq_hist = getattr(broker, '_liquidation_equity_history', [])
            if liq_hist:
                broker._liquidation_equity_history = [
                    (ts, eq) for ts, eq in liq_hist
                    if ts >= measurement_start
                ]
            initial_curve = broker._liquidation_equity_history or broker._equity_history
            if initial_curve:
                broker._initial_equity = initial_curve[0][1]

        # Collect trades (filter warmup)
        strategy_trades = []
        for trade in broker._closed_trades:
            if measurement_start and trade.entry_time < measurement_start:
                continue
            strategy_trades.append(trade)

        per_strategy_trades[slot.strategy_id] = strategy_trades
        all_trades.extend(strategy_trades)

        # Per-strategy metrics
        sm = compute_metrics(broker)
        per_strategy_metrics[slot.strategy_id] = sm

    all_trades.sort(key=lambda t: t.entry_time)

    # Shutdown strategies
    for slot in slots:
        slot.strategy.on_shutdown(slot.ctx)

    # Build a synthetic broker for combined metrics
    combined_broker = SimBroker(initial_equity=portfolio_config.initial_equity)
    combined_broker._closed_trades = all_trades
    combined_broker._terminal_marks = [
        mark for marks in terminal_marks.values() for mark in marks
    ]
    combined_broker._initial_equity = portfolio_config.initial_equity

    # Build combined equity: sum equity deltas across all strategies
    _build_combined_equity(slots, combined_broker, portfolio_config.initial_equity, measurement_start)

    metrics = compute_metrics(combined_broker)

    return PortfolioBacktestResult(
        per_strategy_trades=per_strategy_trades,
        all_trades=all_trades,
        equity_curve=combined_broker._liquidation_equity_history or combined_broker._equity_history,
        metrics=metrics,
        rule_events=rule_events,
        risk_decision_events=risk_decision_events,
        order_events=order_events,
        per_strategy_metrics=per_strategy_metrics,
        config=backtest_config,
        portfolio_config=portfolio_config,
        execution_mode=execution_mode,
        terminal_marks=terminal_marks,
    )


def _build_combined_equity(
    slots: list[_StrategySlot],
    combined_broker: SimBroker,
    initial_equity: float,
    measurement_start: datetime | None,
) -> None:
    """Build combined equity curve from per-strategy equity histories.

    Portfolio equity = initial + sum of per-strategy P&L at each timestamp.
    """
    # Collect all equity snapshots with strategy identity
    all_snapshots: list[tuple[datetime, str, float]] = []
    strategy_initial: dict[str, float] = {}

    for slot in slots:
        broker = slot.broker
        eq_curve = broker._liquidation_equity_history or broker._equity_history
        strategy_initial[slot.strategy_id] = broker._initial_equity

        for ts, eq in eq_curve:
            all_snapshots.append((ts, slot.strategy_id, eq))

    if not all_snapshots:
        return

    # Sort by timestamp
    all_snapshots.sort(key=lambda x: x[0])

    # Track latest equity per strategy, compute combined
    latest_equity: dict[str, float] = dict(strategy_initial)
    combined_history: list[tuple[datetime, float]] = []

    for ts, sid, eq in all_snapshots:
        latest_equity[sid] = eq
        # Combined = initial + sum(strategy_equity - strategy_initial)
        combined = initial_equity + sum(
            latest_equity[s] - strategy_initial[s]
            for s in latest_equity
        )
        combined_history.append((ts, combined))

    combined_broker._equity_history = combined_history
    if combined_history:
        combined_broker._initial_equity = combined_history[0][1]


def _market_sort_key(bar: Bar | MarketEvent) -> tuple[datetime, int]:
    if isinstance(bar, MarketEvent):
        return bar.available_at, _TF_PRIORITY.get(bar.timeframe, 99)
    return (
        bar.timestamp + timedelta(minutes=bar.timeframe.minutes),
        _TF_PRIORITY.get(bar.timeframe, 99),
    )


# Module-level reference to all slots (set during run_portfolio_backtest)
_all_slots_ref: list[_StrategySlot] = []


def _portfolio_equity_from_slots(slots: list[_StrategySlot], initial_equity: float) -> float:
    """Compute shared portfolio equity from strategy-local broker deltas."""
    unique_brokers: list[SimBroker] = []
    seen: set[int] = set()
    for slot in slots:
        ident = id(slot.broker)
        if ident in seen:
            continue
        seen.add(ident)
        unique_brokers.append(slot.broker)

    return initial_equity + sum(
        broker.get_equity() - broker.initial_equity
        for broker in unique_brokers
    )
