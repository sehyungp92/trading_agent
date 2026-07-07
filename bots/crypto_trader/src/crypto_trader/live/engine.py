"""Live trading engine — async polling loop for paper/live trading."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.backtest.runner import _create_strategy
from crypto_trader.core.clock import WallClock
from crypto_trader.core.engine import MultiTimeFrameBars, StrategyContext
from crypto_trader.core.events import CanonicalRuntimeEvent, EventBus, PositionClosedEvent
from crypto_trader.core.execution_gateway import ExecutionGateway
from crypto_trader.core.models import Bar, Fill, Order, OrderStatus, OrderType, Position, SetupGrade, Side, TimeFrame, Trade
from crypto_trader.core.order_semantics import (
    EXIT_OCA_POLICY,
    NATIVE_OCA_POLICY,
    entry_position_instance_id,
    is_exit_order,
    validate_strategy_scoped_oca_group,
)
from crypto_trader.core.runtime_types import MarketEvent
from crypto_trader.core.runtime_types import ExecutionReport, ExecutionReportKind
from crypto_trader.core.strategy_runtime import StrategySlotRuntime
from crypto_trader.exchange.meta import AssetMeta
from crypto_trader.live.broker import HyperliquidBroker
from crypto_trader.live.config import LiveConfig
from crypto_trader.live.execution_adapter import HyperliquidExecutionAdapter
from crypto_trader.live.feed import LiveFeed
from crypto_trader.live.health import HealthMonitor
from crypto_trader.live.lifecycle import PositionLifecycleLedger
from crypto_trader.live.oms_store import (
    FILL_COORDINATOR_APPLIED_STATUSES,
    FILL_FINALIZED_STATUSES,
    FILL_LIFECYCLE_APPLIED_STATUSES,
    FILL_STRATEGY_DISPATCHED_STATUSES,
    OmsStore,
    fill_identity,
)
from crypto_trader.live.reconciler import Discrepancy, PositionReconciler
from crypto_trader.live.state import PersistentState
from crypto_trader.portfolio.config import PortfolioConfig
from crypto_trader.portfolio.allocation import (
    allocation_residuals,
    derive_strategy_position_allocations,
    exchange_net_positions,
)
from crypto_trader.portfolio.coordinator import StrategyCoordinator
from crypto_trader.portfolio.manager import PortfolioManager
from crypto_trader.portfolio.state import PortfolioState
from crypto_trader.instrumentation.backfill import MissedOpportunityBackfiller
from crypto_trader.instrumentation.emitter import EventEmitter
from crypto_trader.instrumentation.lineage import (
    ALLOCATION_CONFIG_KEYS,
    LineageContext,
    RISK_CONFIG_KEYS,
    from_live_engine_inputs,
    read_json_file,
    stable_hash,
    strip_secret_fields,
    subset_keys,
)
from crypto_trader.instrumentation.sinks import JsonlSink
from crypto_trader.instrumentation.sidecar import SidecarForwarder
from crypto_trader.instrumentation.daily_aggregator import DailyAggregator
from crypto_trader.instrumentation.types import (
    ErrorEvent,
    EventMetadata,
    GenericInstrumentationEvent,
    HealthReportSnapshot,
    PipelineFunnelSnapshot,
)
from crypto_trader.instrumentation.pipeline_tracker import PipelineTracker
from crypto_trader.live.health_report import HealthReportBuilder

log = structlog.get_logger()

# Expected bar intervals by timeframe (seconds)
_TF_INTERVALS: dict[str, float] = {
    "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}

_RECONCILIATION_BLOCK_REASON = "live OMS/exchange reconciliation unresolved"

_STRATEGY_BRIDGE_IDS = {
    "trend": "crypto_trend_v1",
    "momentum": "crypto_momentum_v1",
    "breakout": "crypto_breakout_v1",
}
_DEFAULT_BRIDGE_CONTRACT_ROOT = Path("contracts") / "assistant_bridges"
_TEXT_ARTIFACT_SUFFIXES = frozenset(
    {
        ".json",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)

# Warmup bar counts per timeframe
_WARMUP_COUNTS = {
    TimeFrame.M15: 200,
    TimeFrame.M30: 101,
    TimeFrame.H1: 50,
    TimeFrame.H4: 50,
    TimeFrame.D1: 60,
}


@dataclass(slots=True)
class _FillProcessingResult:
    processed: list[Fill]
    duplicates: list[Fill]
    unresolved: list[Fill]
    safe_watermark_fills: list[Fill]


def _enum_or_default(enum_cls, value, default):
    try:
        if isinstance(value, enum_cls):
            return value
        return enum_cls(value)
    except (TypeError, ValueError):
        return default


def _asset_meta_broker_kwargs(asset_meta_path: Path | None) -> dict[str, dict[str, float]]:
    if asset_meta_path is None:
        return {}
    asset_meta = AssetMeta.from_cache(asset_meta_path)
    log.info(
        "engine.asset_meta_loaded",
        path=str(asset_meta_path),
        symbols=len(asset_meta.asset_index),
    )
    return {
        "lot_sizes": dict(asset_meta.lot_sizes),
        "tick_sizes": dict(asset_meta.tick_sizes),
    }


class _WarmupBrokerProxy:
    """Null broker that silently rejects orders during warmup.

    Prevents strategies from placing real orders while processing
    historical warmup bars with stale data.
    """

    def submit_order(self, order):
        order.status = OrderStatus.REJECTED
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        return True

    def cancel_all(self, symbol: str = "") -> int:
        return 0

    def get_position(self, symbol: str):
        return None

    def get_positions(self) -> list:
        return []

    def get_open_orders(self, symbol: str = "") -> list:
        return []

    def get_equity(self) -> float:
        return 0.0

    def get_fills_since(self, since) -> list:
        return []

    def get_portfolio_snapshot(self, symbol: str, direction: Side) -> None:
        return None


class _StrategySlot:
    """Internal: holds one strategy's runtime state."""

    def __init__(
        self,
        strategy_id: str,
        strategy: Any,
        ctx: StrategyContext,
        bars: MultiTimeFrameBars,
        subscribed_tfs: set[TimeFrame],
        primary_tf: TimeFrame,
    ) -> None:
        self.strategy_id = strategy_id
        self.strategy = strategy
        self.ctx = ctx
        self.bars = bars
        self.subscribed_tfs = subscribed_tfs
        self.primary_tf = primary_tf
        self.runtime = StrategySlotRuntime(
            strategy=strategy,
            ctx=ctx,
            broker=ctx.broker,
            bars=bars,
            events=ctx.events,
            primary_timeframe=primary_tf,
            strategy_id=strategy_id,
        )


class LiveEngine:
    """Async polling engine for live/paper trading.

    Concurrent tasks:
    - _poll_candles_loop: poll for new bars, dispatch to strategies
    - _poll_fills_loop: poll for new fills, route to strategies
    - _equity_snapshot_loop: periodic equity recording
    - _daily_reset_loop: reset daily P&L at UTC midnight
    - _health_check_loop: heartbeat + stale data detection
    """

    def __init__(self, config: LiveConfig) -> None:
        self._config = config
        self._running = False
        self._slots: list[_StrategySlot] = []
        self._broker: HyperliquidBroker | None = None
        self._execution_adapter: HyperliquidExecutionAdapter | None = None
        self._coordinator: StrategyCoordinator | None = None
        self._manager: PortfolioManager | None = None
        self._feed: LiveFeed | None = None
        self._health = HealthMonitor()
        self._persistent = PersistentState(config.state_dir)
        self._oms = OmsStore(config.state_dir)
        self._lifecycle = PositionLifecycleLedger()
        self._last_fill_check = self._load_fill_watermark() or datetime.now(timezone.utc)
        self._tracked_positions: dict[str, dict] = {}  # sym → tracked entry data
        self._strategy_dispatched_fill_ids: set[str] = set()
        self._coordinator_applied_fill_ids: set[str] = set()
        self._lifecycle_applied_fill_ids: set[str] = set()
        self._lifecycle_closed_trades_by_fill_id: dict[str, Trade | None] = {}
        self._tracked_fill_ids: set[str] = set()
        self._emitted_lifecycle_trade_ids: set[str] = set()
        self._finalized_fill_ids: set[str] = set()
        self._pending_missed: dict[str, Any] = {}
        self._last_funnels: dict[str, dict] = {}  # strategy_id → last funnel dict
        self._report_builder = HealthReportBuilder()
        self._lineage: LineageContext = LineageContext(
            bot_id=getattr(config, "bot_id", ""),
            family_id=getattr(config, "family_id", "crypto_perps"),
            portfolio_id=getattr(config, "portfolio_id", "default"),
            account_alias=getattr(config, "account_alias", "default"),
            venue_environment="testnet" if config.is_testnet else "mainnet",
            symbol_universe=list(config.symbols),
        )
        self._last_assistant_event_at: dict[str, str] = {}
        self._runtime_started_at_utc = datetime.now(timezone.utc).isoformat()
        self._runtime_instance_id = stable_hash({
            "bot_id": getattr(config, "bot_id", ""),
            "portfolio_id": getattr(config, "portfolio_id", "default"),
            "started_at": self._runtime_started_at_utc,
        })

        # Instrumentation
        self._emitter = EventEmitter()
        self._emitter.add_sink(JsonlSink(config.state_dir))
        self._daily_aggregator = DailyAggregator(bot_id=getattr(config, "bot_id", ""))
        self._emitter.add_sink(self._daily_aggregator)  # aggregator receives all events
        self._sidecar: SidecarForwarder | None = None

        # PostgreSQL sink (optional — wired as additional Sink for trades/daily/health)
        self._pg_sink = None
        if config.postgres_dsn:
            try:
                if getattr(config, "postgres_async_enabled", True):
                    from crypto_trader.instrumentation.async_postgres_sink import AsyncPostgresSink
                    self._pg_sink = AsyncPostgresSink(
                        config.postgres_dsn,
                        queue_capacity=getattr(config, "postgres_queue_capacity", 5000),
                        error_callback=self._emit_postgres_error_event,
                    )
                else:
                    from crypto_trader.instrumentation.postgres_sink import PostgresSink
                    self._pg_sink = PostgresSink(
                        config.postgres_dsn,
                        error_callback=self._emit_postgres_error_event,
                    )
                self._emitter.add_sink(self._pg_sink)
                log.info("engine.postgres_sink_enabled")
            except Exception as exc:
                log.exception("engine.postgres_sink_init_failed")
                self._emit_error_event(
                    "postgres_sink",
                    exc,
                    severity="medium",
                    recovery_action="disable_postgres_sink",
                    error_type=type(exc).__name__,
                )

    async def start(self) -> None:
        """Initialize all components."""
        log.info("engine.starting", testnet=self._config.is_testnet)

        asset_meta_kwargs = _asset_meta_broker_kwargs(self._config.asset_meta_path)

        # Create broker
        self._broker = HyperliquidBroker(
            wallet_address=self._config.wallet_address,
            private_key=self._config.private_key,
            is_testnet=self._config.is_testnet,
            max_slippage_pct=self._config.max_slippage_pct,
            rate_limit_per_sec=self._config.rate_limit_per_sec,
            **asset_meta_kwargs,
        )
        self._execution_adapter = HyperliquidExecutionAdapter(self._broker)

        # Load portfolio config
        portfolio_config = self._load_portfolio_config()
        strategy_config_payloads = {
            strategy_id: read_json_file(config_path)
            for strategy_id, config_path in self._config.strategy_configs.items()
        }
        deployment_manifest = read_json_file(self._config.deployment_manifest_path)
        self._lineage = from_live_engine_inputs(
            config=self._config,
            portfolio_config=portfolio_config,
            strategy_configs=strategy_config_payloads,
            deployment_manifest=deployment_manifest,
            cwd=Path.cwd(),
        )

        # Create portfolio management
        state = PortfolioState(
            equity=self._broker.get_equity(),
            peak_equity=self._broker.get_equity(),
        )

        # Try to restore from persistent state
        saved_state = self._persistent.load_portfolio_state()
        if saved_state:
            restored = PortfolioState.from_dict(saved_state)
            state.open_risks = restored.open_risks
            state.daily_pnl_R = restored.daily_pnl_R
            state.portfolio_daily_pnl_R = restored.portfolio_daily_pnl_R
            state.current_day = restored.current_day
            state.peak_equity = max(state.equity, restored.peak_equity or state.equity)
            today = datetime.now(timezone.utc).date()
            if state.current_day != today:
                state.reset_daily(today)
            log.info(
                "engine.state_restored",
                peak_equity=state.peak_equity,
                open_risks=len(state.open_risks),
                current_day=str(state.current_day) if state.current_day else None,
            )

        self._manager = PortfolioManager(config=portfolio_config, state=state)
        self._coordinator = StrategyCoordinator(
            broker=self._broker,
            manager=self._manager,
            event_callback=self._emit_assistant_payload,
        )

        # Create strategies
        strategy_tfs: dict[str, list[TimeFrame]] = {}

        for strategy_id, config_path in self._config.strategy_configs.items():
            alloc = portfolio_config.get_strategy(strategy_id)
            if alloc is None or not alloc.enabled:
                log.info("engine.strategy_skipped", strategy=strategy_id)
                continue

            strategy_config = self._load_strategy_config(strategy_id, config_path)
            strategy_config.symbols = self._config.symbols

            bot_id = getattr(self._config, "bot_id", "")
            strategy, feed_tfs, primary_tf = _create_strategy(strategy_id, strategy_config, bot_id=bot_id)
            collector = getattr(strategy, "_collector", None)
            if collector is not None:
                collector.set_lineage(self._lineage)
            strategy_tfs[strategy_id] = feed_tfs

            clock = WallClock()
            events = EventBus()
            events.subscribe(CanonicalRuntimeEvent, self._record_canonical_event)
            bars = MultiTimeFrameBars()
            execution_gateway = ExecutionGateway(
                adapter=HyperliquidExecutionAdapter(self._broker, strategy_id=strategy_id),
                broker=self._broker,
                events=events,
                oms_store=self._oms,
                immediate_fill_sync=self._sync_fills_after_submit,
            )
            proxy = self._coordinator.get_proxy(strategy_id)
            proxy._broker = execution_gateway

            ctx = StrategyContext(
                broker=proxy,
                clock=clock,
                bars=bars,
                events=events,
                config=strategy_config,
            )

            self._slots.append(_StrategySlot(
                strategy_id=strategy_id,
                strategy=strategy,
                ctx=ctx,
                bars=bars,
                subscribed_tfs=set(feed_tfs),
                primary_tf=primary_tf,
            ))

        self._rehydrate_oms_orders()

        # Create feed
        from hyperliquid.info import Info
        info = Info(self._config.base_url, skip_ws=True)
        self._feed = LiveFeed(info, self._config.symbols, strategy_tfs)

        # Load warmup bars
        warmup_bars = self._feed.load_warmup_bars(info, _WARMUP_COUNTS)

        # Init strategies with real broker (strategies may check initial state)
        for slot in self._slots:
            slot.strategy.on_init(slot.ctx)

        # Swap to warmup proxy — silently rejects all orders during warmup
        warmup_proxy = _WarmupBrokerProxy()
        real_brokers: list[Any] = []
        for slot in self._slots:
            real_brokers.append(slot.ctx.broker)
            slot.ctx.broker = warmup_proxy

        warmup_measurement_start = None
        if warmup_bars:
            warmup_measurement_start = max(
                bar.timestamp for bar in warmup_bars
            ) + timedelta(microseconds=1)

        original_start_dates: list[tuple[bool, Any]] = []
        for slot in self._slots:
            had_start_date = hasattr(slot.ctx.config, "start_date")
            original_start_dates.append((had_start_date, getattr(slot.ctx.config, "start_date", None)))
            if warmup_measurement_start is not None:
                setattr(slot.ctx.config, "start_date", warmup_measurement_start)

        # Feed warmup bars (orders silently rejected, no emitter wired)
        log.info("engine.warmup_start", bars=len(warmup_bars))
        for bar in warmup_bars:
            for slot in self._slots:
                if bar.timeframe in slot.subscribed_tfs and bar.symbol in slot.strategy.symbols:
                    slot.bars.append(bar)
                    slot.strategy.on_bar(bar, slot.ctx)
        log.info("engine.warmup_complete")

        # Restore real brokers after warmup
        for slot, real_broker, (had_start_date, original_start_date) in zip(
            self._slots,
            real_brokers,
            original_start_dates,
        ):
            slot.ctx.broker = real_broker
            if had_start_date:
                setattr(slot.ctx.config, "start_date", original_start_date)
            elif hasattr(slot.ctx.config, "start_date"):
                delattr(slot.ctx.config, "start_date")

        # Discard warmup-only instrumentation before wiring the live emitter.
        for slot in self._slots:
            collector = getattr(slot.strategy, "_collector", None)
            if collector is None:
                continue
            collector.flush_missed()
            collector.pipeline.snapshot_and_reset()

        self._restore_strategy_snapshots()
        self._restore_lifecycle()

        # Wire instrumentation AFTER warmup (no stale telemetry)
        for slot in self._slots:
            collector = getattr(slot.strategy, "_collector", None)
            if collector is not None:
                collector.emitter = self._emitter

        open_orders = self._sync_open_orders_to_oms()

        self._emit_startup_snapshots(
            portfolio_config,
            strategy_config_payloads,
            deployment_manifest,
        )

        # Initial reconciliation — compare portfolio state expectations with exchange
        reconciler = PositionReconciler()
        actual = self._broker.get_positions()
        # On fresh start, no positions expected; on restart, portfolio state has open_risks
        expected = self._expected_positions_from_portfolio_state()
        # Also mark symbols with no expected position
        for sym in self._config.symbols:
            if sym not in expected:
                expected[sym] = None
        startup_discrepancies = reconciler.reconcile(expected, actual)
        startup_discrepancies.extend(self._allocation_drift_discrepancies(actual))
        startup_discrepancies.extend(self._cleanup_flat_symbol_exit_orders(open_orders, actual))
        startup_discrepancies.extend(self._reconcile_open_oca_groups(open_orders))
        self._handle_startup_reconciliation(startup_discrepancies)
        self._seed_ttl_trackers_from_open_orders(open_orders)

        # Start sidecar forwarder if relay is configured
        relay_url = getattr(self._config, "relay_url", "")
        relay_secret = getattr(self._config, "relay_secret", "")
        bot_id = getattr(self._config, "bot_id", "")
        if relay_url and relay_secret and bot_id:
            self._sidecar = SidecarForwarder(
                state_dir=self._config.state_dir,
                relay_url=relay_url,
                bot_id=bot_id,
                shared_secret=relay_secret,
                error_callback=self._emit_sidecar_error_event,
            )
            self._sidecar.start()

        self._running = True
        log.info("engine.started", strategies=[s.strategy_id for s in self._slots])

    async def run(self) -> None:
        """Run the engine loop with concurrent tasks."""
        if not self._running:
            await self.start()

        tasks = [
            asyncio.create_task(self._poll_candles_loop()),
            asyncio.create_task(self._poll_fills_loop()),
            asyncio.create_task(self._equity_snapshot_loop()),
            asyncio.create_task(self._daily_reset_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._funnel_report_loop()),
            asyncio.create_task(self._health_report_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            log.info("engine.cancelled")
        except Exception:
            log.exception("engine.fatal_error")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        log.info("engine.shutting_down")

        for slot in self._slots:
            try:
                slot.strategy.on_shutdown(slot.ctx)
            except Exception:
                log.exception("engine.shutdown_error", strategy=slot.strategy_id)

        # Stop sidecar forwarder
        if self._sidecar is not None:
            self._sidecar.stop()

        # Close PostgreSQL connection pool
        if self._pg_sink is not None:
            try:
                self._pg_sink.close(
                    flush_timeout_sec=getattr(self._config, "postgres_flush_timeout_sec", 5.0),
                )
            except TypeError:
                self._pg_sink.close()

        # Persist final state before closing the durable store.
        try:
            if self._manager:
                self._persistent.save_portfolio_state(self._manager.state.to_dict())
            self._persist_strategy_snapshots()
            self._persist_lifecycle()
        finally:
            self._oms.close()

        log.info("engine.shutdown_complete")

    # -------------------------------------------------------------------
    # Polling loops
    # -------------------------------------------------------------------

    async def _poll_candles_loop(self) -> None:
        """Poll for new bars at configured interval."""
        while self._running:
            try:
                self._health.on_poll()
                poll_events = getattr(self._feed, "poll_market_events", None)
                bars = poll_events() if callable(poll_events) else self._feed.poll()

                for bar in bars:
                    self._dispatch_bar(bar)

            except Exception as exc:
                self._health.on_error("candle_poll")
                self._emit_error_event("candle_poll", exc, severity="medium", recovery_action="backoff")
                delay = self._health.get_backoff_delay()
                await asyncio.sleep(delay)
                continue

            await asyncio.sleep(self._config.poll_interval_sec)

    async def _poll_fills_loop(self) -> None:
        """Poll for new fills at configured interval."""
        while self._running:
            try:
                self._poll_and_process_fills()

            except Exception as exc:
                self._health.on_error("fill_poll")
                self._emit_error_event("fill_poll", exc, severity="medium", recovery_action="continue")

            await asyncio.sleep(self._config.fill_poll_interval_sec)

    def _sync_fills_after_submit(self, _order_id: str) -> None:
        """Immediately ingest fills that may have landed during submission."""
        try:
            self._poll_and_process_fills()
        except Exception as exc:
            log.exception("engine.immediate_fill_sync_failed")
            self._health.on_error("immediate_fill_sync")
            self._emit_error_event("immediate_fill_sync", exc, severity="medium", recovery_action="continue")

    def _poll_and_process_fills(self) -> list[Fill]:
        """Poll exchange fills with overlap and process them idempotently."""
        if self._broker is None:
            return []
        since = self._last_fill_check - timedelta(seconds=self._fill_query_overlap_sec())
        fills, reports = self._poll_fill_reports(since)
        result = self._process_fills(fills)
        self._record_fill_poll_reports(reports)

        latest_ts = max((fill.timestamp for fill in result.safe_watermark_fills), default=None)
        if latest_ts is not None:
            self._last_fill_check = max(self._last_fill_check, latest_ts)
        oms = getattr(self, "_oms", None)
        if oms is not None:
            oms.set_watermark("fills_since", self._last_fill_check.isoformat())
            oms.set_watermark("fills_last_poll_at", datetime.now(timezone.utc).isoformat())
        return result.processed

    def _poll_fill_reports(self, since: datetime) -> tuple[list[Fill], list[ExecutionReport]]:
        adapter = getattr(self, "_execution_adapter", None)
        sync_fills = getattr(adapter, "sync_fills", None)
        if callable(sync_fills):
            reports = list(sync_fills(since))
            fills = [
                fill for report in reports
                if (fill := self._fill_from_execution_report(report)) is not None
            ]
            non_fill_reports = [
                report for report in reports
                if report.kind not in {ExecutionReportKind.FILL, ExecutionReportKind.PARTIAL_FILL}
            ]
            return fills, non_fill_reports
        broker = getattr(self, "_broker", None)
        if broker is None:
            return [], []
        return list(broker.get_fills_since(since)), []

    def _fill_from_execution_report(self, report: ExecutionReport) -> Fill | None:
        if report.kind not in {ExecutionReportKind.FILL, ExecutionReportKind.PARTIAL_FILL}:
            return None
        if report.side is None:
            return None
        filled_qty = report.filled_qty if report.filled_qty else report.qty
        if filled_qty <= 0:
            return None
        metadata = dict(report.metadata or {})
        return Fill(
            order_id=report.client_order_id,
            exchange_order_id=report.exchange_order_id,
            exchange_fill_id=report.fill_id or "",
            symbol=report.symbol,
            side=report.side,
            qty=filled_qty,
            fill_price=report.fill_price or 0.0,
            commission=report.commission,
            timestamp=report.timestamp,
            tag=str(metadata.get("tag") or ""),
            raw=metadata,
        )

    def _record_fill_poll_reports(self, reports: list[ExecutionReport]) -> None:
        for report in reports:
            self._record_execution_report(report)
            self._record_canonical_event(CanonicalRuntimeEvent(
                timestamp=report.timestamp,
                stream="execution",
                payload=report.to_dict(),
            ))

    def _record_execution_report(self, report: ExecutionReport) -> None:
        oms = getattr(self, "_oms", None)
        if oms is None:
            return
        record_fn = getattr(oms, "record_execution_report", None)
        if callable(record_fn):
            record_fn(report)
        upsert_fn = getattr(oms, "upsert_order", None)
        if not callable(upsert_fn) or not report.client_order_id:
            return
        metadata = dict(report.metadata or {})
        order_metadata = {**report.to_dict(), **metadata}
        upsert_fn(
            client_order_id=report.client_order_id,
            exchange_order_id=report.exchange_order_id,
            strategy_id=str(metadata.get("strategy_id") or ""),
            symbol=report.symbol,
            side=report.side.value if report.side is not None else "",
            order_type=str(metadata.get("order_type") or ""),
            status=report.order_status.value if report.order_status is not None else report.kind.value.upper(),
            role=str(metadata.get("role") or metadata.get("tag") or ""),
            decision_id=str(metadata.get("decision_id") or ""),
            position_instance_id=str(metadata.get("position_instance_id") or ""),
            reduce_only=bool(metadata.get("reduce_only", False)),
            oca_group=metadata.get("oca_group"),
            bracket_group=metadata.get("bracket_group"),
            metadata=order_metadata,
        )

    def _process_fills(self, fills: list[Fill]) -> _FillProcessingResult:
        processed_fills: list[Fill] = []
        duplicate_fills: list[Fill] = []
        unresolved_fills: list[Fill] = []
        safe_watermark_fills: list[Fill] = []
        ledger_closed_symbols: set[str] = set()

        for fill in fills:
            fill_id = fill_identity(fill)
            if self._is_processed_oms_fill_id(fill_id):
                duplicate_fills.append(fill)
                safe_watermark_fills.append(fill)
                continue

            self._record_oms_fill_received(fill_id, fill, "")
            strategy_id, fill = self._resolve_fill_owner(fill)

            if self._is_processed_oms_fill_id(fill_id):
                duplicate_fills.append(fill)
                safe_watermark_fills.append(fill)
                continue

            self._record_oms_fill_received(fill_id, fill, strategy_id or "")
            if not strategy_id:
                log.warning("engine.unattributed_fill", order_id=fill.order_id)
                unresolved_fills.append(fill)
                self._mark_oms_fill_unresolved(
                    fill_id,
                    strategy_id="",
                    reason="unattributed_fill",
                )
                should_emit_snapshots = self._record_fill_discrepancy(
                    fill,
                    fill_id=fill_id,
                    kind="unattributed_fill",
                    description="Exchange fill could not be matched to a strategy owner.",
                )
                if should_emit_snapshots:
                    self._emit_unresolved_fill_snapshots(
                        fill_id,
                        fill,
                        strategy_id="UNKNOWN",
                        reason="unattributed_fill",
                    )
                continue

            self._register_resolved_fill_order_ids(fill, strategy_id)
            slot = self._find_slot(strategy_id)
            if slot is None:
                unresolved_fills.append(fill)
                self._mark_oms_fill_unresolved(
                    fill_id,
                    strategy_id=strategy_id,
                    reason="missing_strategy_slot_fill",
                )
                should_emit_snapshots = self._record_fill_discrepancy(
                    fill,
                    fill_id=fill_id,
                    kind="missing_strategy_slot_fill",
                    description="Exchange fill owner was resolved but no live strategy slot exists.",
                    strategy_id=strategy_id,
                )
                if should_emit_snapshots:
                    self._emit_unresolved_fill_snapshots(
                        fill_id,
                        fill,
                        strategy_id=strategy_id or "UNKNOWN",
                        reason="missing_strategy_slot_fill",
                    )
                continue

            self._clear_ttl_tracking_for_fill(slot, fill)
            try:
                status = self._oms_fill_status(fill_id)
                if status not in FILL_STRATEGY_DISPATCHED_STATUSES:
                    self._apply_strategy_fill_phase(fill_id, slot, fill, strategy_id)
                    status = self._oms_fill_status(fill_id)

                if status not in FILL_COORDINATOR_APPLIED_STATUSES:
                    strategy_id = self._apply_coordinator_fill_phase(
                        fill_id,
                        fill,
                        strategy_id,
                    )
                    status = self._oms_fill_status(fill_id)

                closed_trade = self._closed_trade_for_fill(fill_id)
                if status not in FILL_LIFECYCLE_APPLIED_STATUSES:
                    closed_trade = self._apply_lifecycle_fill_phase(
                        fill_id,
                        fill,
                        strategy_id,
                    )
                    status = self._oms_fill_status(fill_id)

                if status not in FILL_FINALIZED_STATUSES:
                    self._apply_finalization_fill_phase(
                        fill_id,
                        fill,
                        strategy_id,
                        closed_trade,
                        ledger_closed_symbols,
                    )

                self._mark_oms_fill_processed(fill_id, strategy_id=strategy_id)
                self._record_fill_telemetry(fill_id, slot, fill)
            except Exception as exc:
                unresolved_fills.append(fill)
                self._handle_fill_processing_exception(
                    fill_id,
                    fill,
                    strategy_id=strategy_id,
                    error=exc,
                )
                continue

            processed_fills.append(fill)
            safe_watermark_fills.append(fill)
            log.info(
                "engine.fill",
                strategy=strategy_id,
                symbol=fill.symbol,
                side=fill.side.value,
                qty=fill.qty,
                price=fill.fill_price,
            )

        fallback_fills = [
            fill for fill in processed_fills
            if fill.symbol not in ledger_closed_symbols
        ]
        if fallback_fills:
            self._detect_position_closures(fallback_fills)
        return _FillProcessingResult(
            processed=processed_fills,
            duplicates=duplicate_fills,
            unresolved=unresolved_fills,
            safe_watermark_fills=safe_watermark_fills,
        )

    def _apply_strategy_fill_phase(
        self,
        fill_id: str,
        slot: _StrategySlot,
        fill: Fill,
        strategy_id: str,
    ) -> None:
        dispatched = self._phase_fill_ids("_strategy_dispatched_fill_ids")
        if fill_id not in dispatched:
            slot.runtime.dispatch_fill(fill, notify_callback=False)
            dispatched.add(fill_id)
        self._mark_oms_fill_strategy_dispatched(fill_id, strategy_id=strategy_id)

    def _apply_coordinator_fill_phase(
        self,
        fill_id: str,
        fill: Fill,
        strategy_id: str,
    ) -> str:
        applied = self._phase_fill_ids("_coordinator_applied_fill_ids")
        applied_strategy_id = strategy_id
        if fill_id not in applied:
            if self._coordinator is not None:
                resolved_id = self._coordinator.on_fill(fill)
                if isinstance(resolved_id, str) and resolved_id:
                    applied_strategy_id = resolved_id
                elif isinstance(self._coordinator, StrategyCoordinator) and fill.tag == "entry":
                    self._record_fill_discrepancy(
                        fill,
                        fill_id=fill_id,
                        kind="coordinator_fill_unapplied",
                        description="Owned entry fill could not be applied by the strategy coordinator.",
                        strategy_id=strategy_id,
                    )
                    raise RuntimeError("coordinator could not apply owned entry fill")
            applied.add(fill_id)
        self._mark_oms_fill_coordinator_applied(fill_id, strategy_id=applied_strategy_id)
        return applied_strategy_id

    def _apply_lifecycle_fill_phase(
        self,
        fill_id: str,
        fill: Fill,
        strategy_id: str,
    ) -> Trade | None:
        applied = self._phase_fill_ids("_lifecycle_applied_fill_ids")
        closed_trades = self._lifecycle_closed_trades()
        if fill_id not in applied:
            lifecycle = getattr(self, "_lifecycle", None)
            closed_trades[fill_id] = (
                lifecycle.apply_fill(strategy_id, fill)
                if lifecycle is not None
                else None
            )
            applied.add(fill_id)
        closed_trade = closed_trades.get(fill_id)
        persist_phase = getattr(getattr(self, "_oms", None), "persist_lifecycle_phase", None)
        if callable(persist_phase):
            persist_phase(
                fill_id,
                self._lifecycle_snapshot(),
                strategy_id=strategy_id,
                closed_trade_event=self._closed_trade_event(fill_id, closed_trade),
            )
        else:
            self._persist_lifecycle()
            self._persist_lifecycle_closed_trade(fill_id, closed_trade)
            self._mark_oms_fill_lifecycle_applied(fill_id, strategy_id=strategy_id)
        return closed_trade

    def _apply_finalization_fill_phase(
        self,
        fill_id: str,
        fill: Fill,
        strategy_id: str,
        closed_trade: Trade | None,
        ledger_closed_symbols: set[str],
    ) -> None:
        finalized = self._phase_fill_ids("_finalized_fill_ids")
        if fill_id not in finalized:
            self._track_entry_fill_once(fill_id, strategy_id, fill)
            if closed_trade is not None:
                self._emit_lifecycle_trade_once(fill_id, strategy_id, closed_trade)
                ledger_closed_symbols.add(fill.symbol)
            if fill.tag != "entry":
                self._emit_fill_lifecycle_snapshots(
                    fill_id,
                    strategy_id,
                    fill,
                    source="exit_fill",
                    closed_trade=closed_trade,
                )
            self._emit_fill_event(fill_id, strategy_id, fill, closed_trade=closed_trade)
            finalized.add(fill_id)
        elif closed_trade is not None and fill_id in self._phase_fill_ids("_emitted_lifecycle_trade_ids"):
            ledger_closed_symbols.add(fill.symbol)
        self._mark_oms_fill_finalized(fill_id, strategy_id=strategy_id)

    def _track_entry_fill_once(self, fill_id: str, strategy_id: str, fill: Fill) -> None:
        if fill.tag != "entry":
            return
        tracked = self._phase_fill_ids("_tracked_fill_ids")
        if fill_id in tracked:
            return
        self._track_entry_fill(strategy_id, fill)
        self._emit_fill_lifecycle_snapshots(fill_id, strategy_id, fill, source="entry_fill")
        tracked.add(fill_id)

    def _emit_lifecycle_trade_once(
        self,
        fill_id: str,
        strategy_id: str,
        trade: Trade,
    ) -> None:
        emitted = self._phase_fill_ids("_emitted_lifecycle_trade_ids")
        if fill_id not in emitted:
            self._emit_lifecycle_trade(strategy_id, trade, fill_id=fill_id)
            emitted.add(fill_id)
        self._record_lifecycle_trade_position(strategy_id, trade)

    def _record_fill_telemetry(self, fill_id: str, slot: _StrategySlot, fill: Fill) -> None:
        collector = getattr(slot.strategy, "_collector", None)
        if collector is None:
            return
        try:
            collector.pipeline.record_fill(fill.symbol)
        except Exception:
            log.exception("engine.fill_telemetry_failed", fill_id=fill_id)

    def _handle_fill_processing_exception(
        self,
        fill_id: str,
        fill: Fill,
        *,
        strategy_id: str,
        error: Exception,
    ) -> None:
        if self._fill_phase_started(fill_id):
            self._record_oms_fill_processing_error(
                fill_id,
                strategy_id=strategy_id,
                error=str(error),
            )
        else:
            self._mark_oms_fill_processing_failed(
                fill_id,
                strategy_id=strategy_id,
                error=str(error),
            )
        self._record_fill_discrepancy(
            fill,
            fill_id=fill_id,
            kind="fill_processing_failed",
            description="Owned exchange fill processing failed before it was safely consumed.",
            strategy_id=strategy_id,
            metadata={"error": str(error)},
        )
        health = getattr(self, "_health", None)
        if health is not None:
            health.on_error("fill_processing")
        self._emit_error_event(
            "fill_processing",
            error,
            strategy_id=strategy_id,
            symbol=fill.symbol,
            severity="high",
            recovery_action="mark_fill_unresolved",
        )
        log.exception(
            "engine.fill_processing_failed",
            strategy=strategy_id,
            order_id=fill.order_id,
            exchange_order_id=fill.exchange_order_id,
            fill_id=fill_id,
        )

    def _fill_phase_started(self, fill_id: str) -> bool:
        status = self._oms_fill_status(fill_id)
        if status in FILL_STRATEGY_DISPATCHED_STATUSES:
            return True
        phase_attrs = (
            "_strategy_dispatched_fill_ids",
            "_coordinator_applied_fill_ids",
            "_lifecycle_applied_fill_ids",
            "_tracked_fill_ids",
            "_emitted_lifecycle_trade_ids",
            "_finalized_fill_ids",
        )
        return any(fill_id in self._phase_fill_ids(attr) for attr in phase_attrs)

    def _phase_fill_ids(self, attr: str) -> set[str]:
        values = getattr(self, attr, None)
        if not isinstance(values, set):
            values = set()
            setattr(self, attr, values)
        return values

    def _lifecycle_closed_trades(self) -> dict[str, Trade | None]:
        values = getattr(self, "_lifecycle_closed_trades_by_fill_id", None)
        if not isinstance(values, dict):
            values = {}
            setattr(self, "_lifecycle_closed_trades_by_fill_id", values)
        return values

    def _closed_trade_for_fill(self, fill_id: str) -> Trade | None:
        closed_trades = self._lifecycle_closed_trades()
        if fill_id not in closed_trades:
            closed_trades[fill_id] = self._load_lifecycle_closed_trade(fill_id)
        return closed_trades.get(fill_id)

    def _persist_lifecycle_closed_trade(self, fill_id: str, trade: Trade | None) -> None:
        event = self._closed_trade_event(fill_id, trade)
        if event is None:
            return
        append_fn = getattr(getattr(self, "_oms", None), "append_event", None)
        if callable(append_fn):
            append_fn("fill_lifecycle_closed_trade", event[0], event[1])

    def _closed_trade_event(
        self,
        fill_id: str,
        trade: Trade | None,
    ) -> tuple[datetime, dict[str, Any]] | None:
        if trade is None:
            return None
        return trade.exit_time, {"fill_id": fill_id, "trade": self._trade_payload(trade)}

    def _load_lifecycle_closed_trade(self, fill_id: str) -> Trade | None:
        list_fn = getattr(getattr(self, "_oms", None), "list_events", None)
        if not callable(list_fn):
            return None
        for event in reversed(list_fn("fill_lifecycle_closed_trade")):
            payload = event.get("payload") or {}
            if payload.get("fill_id") == fill_id:
                return self._trade_from_payload(payload.get("trade") or {})
        return None

    def _trade_payload(self, trade: Trade) -> dict[str, Any]:
        return {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "direction": trade.direction.value,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "qty": trade.qty,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat(),
            "pnl": trade.pnl,
            "r_multiple": trade.r_multiple,
            "commission": trade.commission,
            "bars_held": trade.bars_held,
            "setup_grade": trade.setup_grade.value if trade.setup_grade is not None else None,
            "exit_reason": trade.exit_reason,
            "confluences_used": trade.confluences_used,
            "confirmation_type": trade.confirmation_type,
            "entry_method": trade.entry_method,
            "funding_paid": trade.funding_paid,
            "mae_r": trade.mae_r,
            "mfe_r": trade.mfe_r,
            "realized_r_multiple": trade.realized_r_multiple,
            "signal_variant": trade.signal_variant,
        }

    def _trade_from_payload(self, data: dict[str, Any]) -> Trade:
        setup_grade = data.get("setup_grade")
        return Trade(
            trade_id=str(data["trade_id"]),
            symbol=str(data["symbol"]),
            direction=Side(data["direction"]),
            entry_price=float(data["entry_price"]),
            exit_price=float(data["exit_price"]),
            qty=float(data["qty"]),
            entry_time=datetime.fromisoformat(data["entry_time"]),
            exit_time=datetime.fromisoformat(data["exit_time"]),
            pnl=float(data["pnl"]),
            r_multiple=data.get("r_multiple"),
            commission=float(data["commission"]),
            bars_held=int(data.get("bars_held", 0)),
            setup_grade=SetupGrade(setup_grade) if setup_grade else None,
            exit_reason=str(data.get("exit_reason") or "exchange_fill"),
            confluences_used=data.get("confluences_used"),
            confirmation_type=data.get("confirmation_type"),
            entry_method=data.get("entry_method"),
            funding_paid=float(data.get("funding_paid", 0.0)),
            mae_r=data.get("mae_r"),
            mfe_r=data.get("mfe_r"),
            realized_r_multiple=data.get("realized_r_multiple"),
            signal_variant=data.get("signal_variant"),
        )

    def _attach_trade_completion_context(
        self,
        strategy_id: str,
        trade: Trade,
        *,
        fill_id: str = "",
    ) -> None:
        context = self._trade_completion_context(strategy_id, trade, fill_id=fill_id)
        setattr(trade, "instrumentation_context", context)

    def _trade_completion_context(
        self,
        strategy_id: str,
        trade: Trade,
        *,
        fill_id: str = "",
    ) -> dict[str, Any]:
        lineage = getattr(self, "_lineage", LineageContext())
        fills = self._trade_fills(strategy_id, trade, fill_id=fill_id)
        orders = self._trade_orders(
            strategy_id,
            trade.symbol,
            trade=trade,
            fills=fills,
        )

        def _order_role(order: dict[str, Any]) -> str:
            metadata = self._order_metadata(order)
            return str(order.get("role") or metadata.get("tag") or "").lower()

        order_metadata = [self._order_metadata(order) for order in orders]
        entry_orders = [order for order in orders if _order_role(order) == "entry"]
        exit_orders = [order for order in orders if order not in entry_orders]
        entry_order_ids = {
            str(order.get("client_order_id") or "")
            for order in entry_orders
            if order.get("client_order_id")
        }
        exit_order_ids = {
            str(order.get("client_order_id") or "")
            for order in exit_orders
            if order.get("client_order_id")
        }
        entry_exchange_ids = {
            str(order.get("exchange_order_id") or "")
            for order in entry_orders
            if order.get("exchange_order_id")
        }
        exit_exchange_ids = {
            str(order.get("exchange_order_id") or "")
            for order in exit_orders
            if order.get("exchange_order_id")
        }

        def _fill_role(fill: dict[str, Any]) -> str:
            raw = fill.get("raw") if isinstance(fill.get("raw"), dict) else {}
            tag = str(raw.get("tag") or "").lower()
            if tag:
                return tag
            client_id = str(fill.get("client_order_id") or "")
            exchange_id = str(fill.get("exchange_order_id") or "")
            if client_id in entry_order_ids or exchange_id in entry_exchange_ids:
                return "entry"
            if client_id in exit_order_ids or exchange_id in exit_exchange_ids:
                return "exit"
            return ""

        entry_fills = [fill for fill in fills if _fill_role(fill) == "entry"]
        exit_fills = [fill for fill in fills if fill not in entry_fills]

        def _first_metadata_key(key: str) -> str:
            for metadata in order_metadata:
                value = metadata.get(key)
                if value:
                    return str(value)
            return ""

        def _first_order_metadata_key(source_orders: list[dict[str, Any]], key: str) -> str:
            for order in source_orders:
                value = self._order_metadata(order).get(key)
                if value:
                    return str(value)
            return ""

        decision_id = _first_order_metadata_key(entry_orders, "decision_id") or _first_metadata_key("decision_id")
        exit_decision_id = _first_order_metadata_key(exit_orders, "decision_id")
        intent_id = _first_order_metadata_key(entry_orders, "intent_id") or _first_metadata_key("intent_id")
        portfolio_rule_event_id = (
            _first_order_metadata_key(entry_orders, "portfolio_rule_event_id")
            or _first_metadata_key("portfolio_rule_event_id")
        )
        risk_decision_id = (
            _first_order_metadata_key(entry_orders, "risk_decision_id")
            or _first_metadata_key("risk_decision_id")
        )
        client_order_ids = list(dict.fromkeys(str(order.get("client_order_id") or "") for order in orders if order.get("client_order_id")))
        exchange_order_ids = list(dict.fromkeys(str(order.get("exchange_order_id") or "") for order in orders if order.get("exchange_order_id")))
        entry_fill_ids = [str(fill.get("fill_id")) for fill in entry_fills if fill.get("fill_id")]
        exit_fill_ids = [str(fill.get("fill_id")) for fill in exit_fills if fill.get("fill_id")]
        position_instance_id = self._resolve_trade_position_instance_id(
            strategy_id,
            trade,
            order_metadata=order_metadata,
        )
        artifact_inputs = {
            "trade": self._trade_payload(trade),
            "orders": client_order_ids,
            "fills": [fill.get("fill_id") for fill in fills],
            "deployment_id": lineage.deployment_id,
        }
        resource_inputs = {
            "decision_id": decision_id,
            "intent_id": intent_id,
            "portfolio_rule_event_id": portfolio_rule_event_id,
            "risk_decision_id": risk_decision_id,
            "config_version": lineage.config_version,
            "allocation_version": lineage.allocation_version,
            "order_metadata": order_metadata,
        }
        return {
            "entry_decision_id": decision_id,
            "exit_decision_id": exit_decision_id,
            "exit_bar_id": _first_order_metadata_key(exit_orders, "bar_id"),
            "intent_id": intent_id,
            "portfolio_rule_event_id": portfolio_rule_event_id,
            "risk_decision_id": risk_decision_id,
            "entry_order_ids": sorted(entry_order_ids),
            "exit_order_ids": sorted(exit_order_ids),
            "entry_fill_ids": entry_fill_ids,
            "exit_fill_ids": exit_fill_ids,
            "position_instance_id": position_instance_id,
            "client_order_ids": client_order_ids,
            "exchange_order_ids": exchange_order_ids,
            "decision_ref": {
                "decision_id": decision_id,
                "bar_id": _first_metadata_key("bar_id"),
                "decision_time": _first_metadata_key("decision_time"),
                "config_version": lineage.config_version,
            },
            "action_ref": {
                "intent_id": intent_id,
                "client_order_ids": client_order_ids,
                "exchange_order_ids": exchange_order_ids,
            },
            "portfolio_decision_ref": {
                "portfolio_rule_event_id": portfolio_rule_event_id,
                "risk_decision_id": risk_decision_id,
                "allocation_version": lineage.allocation_version,
                "risk_config_version": lineage.risk_config_version,
            },
            "artifact_hash": stable_hash(artifact_inputs, length=32),
            "resource_plan_hash": stable_hash(resource_inputs, length=32),
            "runtime_join": {
                "orders": orders,
                "fills": fills,
                "fill_id": fill_id,
                "config_version": lineage.config_version,
                "deployment_id": lineage.deployment_id,
            },
        }

    def _trade_orders(
        self,
        strategy_id: str,
        symbol: str,
        *,
        trade: Trade | None = None,
        fills: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        oms = getattr(self, "_oms", None)
        list_orders = getattr(oms, "list_orders", None)
        if not callable(list_orders):
            return []
        orders = [
            order for order in list_orders()
            if order.get("strategy_id") == strategy_id and order.get("symbol") == symbol
        ]
        if trade is None:
            return orders

        context_order_ids = self._trade_context_order_ids(trade)
        if context_order_ids:
            matched = [
                order for order in orders
                if self._order_id_set(order) & context_order_ids
            ]
            if matched:
                return matched

        fill_order_ids = self._fill_order_id_set(fills or [])
        if fill_order_ids:
            return [
                order for order in orders
                if self._order_id_set(order) & fill_order_ids
            ]

        position_instance_id = self._trade_position_instance_id(trade)
        if position_instance_id:
            matched = [
                order for order in orders
                if str(order.get("position_instance_id") or "") == position_instance_id
            ]
            if matched:
                return matched

        existing_context = getattr(trade, "instrumentation_context", {})
        decision_id = ""
        if isinstance(existing_context, dict):
            decision_id = str(existing_context.get("entry_decision_id") or "")
        if decision_id:
            matched = [
                order for order in orders
                if self._order_metadata(order).get("decision_id") == decision_id
            ]
            if matched:
                return matched

        return [
            order for order in orders
            if self._order_in_trade_window(order, trade)
        ]

    @staticmethod
    def _order_metadata(order: dict[str, Any]) -> dict[str, Any]:
        metadata = order.get("metadata")
        return dict(metadata) if isinstance(metadata, dict) else {}

    @staticmethod
    def _order_id_set(order: dict[str, Any]) -> set[str]:
        return {
            str(value)
            for value in (
                order.get("client_order_id"),
                order.get("exchange_order_id"),
            )
            if value
        }

    @staticmethod
    def _fill_order_id_set(fills: list[dict[str, Any]]) -> set[str]:
        ids: set[str] = set()
        for fill in fills:
            for key in ("client_order_id", "exchange_order_id"):
                value = fill.get(key)
                if value:
                    ids.add(str(value))
        return ids

    @classmethod
    def _trade_context_order_ids(cls, trade: Trade) -> set[str]:
        context = cls._trade_existing_context(trade)
        return cls._context_string_set(context, "entry_order_ids", "exit_order_ids")

    @classmethod
    def _trade_context_fill_ids(cls, trade: Trade) -> set[str]:
        context = cls._trade_existing_context(trade)
        return cls._context_string_set(context, "entry_fill_ids", "exit_fill_ids")

    @staticmethod
    def _trade_existing_context(trade: Trade) -> dict[str, Any]:
        context = getattr(trade, "instrumentation_context", {})
        return context if isinstance(context, dict) else {}

    @staticmethod
    def _context_string_set(context: dict[str, Any], *keys: str) -> set[str]:
        values: set[str] = set()
        for key in keys:
            raw = context.get(key)
            if isinstance(raw, (list, tuple, set)):
                values.update(str(item) for item in raw if item)
            elif raw:
                values.add(str(raw))
        return values

    @staticmethod
    def _trade_position_instance_id(trade: Trade) -> str:
        context = LiveEngine._trade_existing_context(trade)
        context_position_id = str(context.get("position_instance_id") or "")
        if context_position_id:
            return context_position_id
        prefix = "live_"
        if not trade.trade_id.startswith(prefix):
            return ""
        body = trade.trade_id[len(prefix):]
        if ":" not in body:
            return ""
        return body.rsplit(":", 1)[0]

    def _resolve_trade_position_instance_id(
        self,
        strategy_id: str,
        trade: Trade,
        *,
        order_metadata: list[dict[str, Any]] | None = None,
    ) -> str:
        position_instance_id = self._trade_position_instance_id(trade)
        if position_instance_id:
            return position_instance_id

        for metadata in order_metadata or []:
            position_instance_id = str(metadata.get("position_instance_id") or "")
            if position_instance_id:
                return position_instance_id

        tracked = self._tracked_positions.get(trade.symbol, {})
        position_instance_id = str(tracked.get("position_instance_id") or "")
        if position_instance_id:
            return position_instance_id

        position_instance_id = self._open_lifecycle_position_instance_id(
            strategy_id,
            trade.symbol,
            trade.direction,
        )
        if position_instance_id:
            return position_instance_id

        manager = getattr(self, "_manager", None)
        open_risks = getattr(getattr(manager, "state", None), "open_risks", [])
        for risk in open_risks:
            if risk.strategy_id != strategy_id or risk.symbol != trade.symbol:
                continue
            if risk.direction != trade.direction:
                continue
            position_instance_id = str(risk.position_instance_id or "")
            if position_instance_id:
                return position_instance_id
        return ""

    def _order_in_trade_window(self, order: dict[str, Any], trade: Trade) -> bool:
        metadata = self._order_metadata(order)
        timestamp = self._parse_optional_datetime(
            metadata.get("submitted_at")
            or metadata.get("decision_time")
            or order.get("updated_at")
        )
        if timestamp is None:
            return False
        return trade.entry_time <= timestamp <= trade.exit_time

    def _trade_fills(
        self,
        strategy_id: str,
        trade: Trade,
        *,
        fill_id: str = "",
    ) -> list[dict[str, Any]]:
        oms = getattr(self, "_oms", None)
        list_fills = getattr(oms, "list_fills", None)
        if not callable(list_fills):
            return []
        all_fills = [
            fill for fill in list_fills()
            if fill.get("strategy_id") == strategy_id and fill.get("symbol") == trade.symbol
        ]
        context_fill_ids = self._trade_context_fill_ids(trade)
        if fill_id:
            context_fill_ids.add(fill_id)
        context_order_ids = self._trade_context_order_ids(trade)
        if context_fill_ids or context_order_ids:
            matched = [
                fill for fill in all_fills
                if (
                    str(fill.get("fill_id") or "") in context_fill_ids
                    or self._fill_order_id_set([fill]) & context_order_ids
                )
            ]
            if matched:
                if fill_id and not context_order_ids and context_fill_ids == {fill_id}:
                    return self._unique_fill_rows(
                        [*self._trade_entry_time_fills(all_fills, trade), *matched]
                    )
                return matched

        fills = []
        entry_time = trade.entry_time if trade.entry_time.tzinfo else trade.entry_time.replace(tzinfo=timezone.utc)
        exit_time = trade.exit_time if trade.exit_time.tzinfo else trade.exit_time.replace(tzinfo=timezone.utc)
        for fill in all_fills:
            fill_ts = self._parse_optional_datetime(fill.get("timestamp"))
            in_trade_window = (
                fill_ts is not None
                and entry_time <= fill_ts <= exit_time
            )
            if (
                str(fill.get("fill_id") or "") == fill_id
                or self._fill_matches_trade_entry_time(fill, trade)
                or (not fill_id and in_trade_window)
            ):
                fills.append(fill)
        return fills

    def _trade_entry_time_fills(
        self,
        fills: list[dict[str, Any]],
        trade: Trade,
    ) -> list[dict[str, Any]]:
        return [
            fill for fill in fills
            if self._fill_matches_trade_entry_time(fill, trade)
        ]

    @staticmethod
    def _unique_fill_rows(fills: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for fill in fills:
            key = str(fill.get("fill_id") or stable_hash(fill))
            if key in seen:
                continue
            seen.add(key)
            unique.append(fill)
        return unique

    def _fill_matches_trade_entry_time(self, fill: dict[str, Any], trade: Trade) -> bool:
        role = str((fill.get("raw") or {}).get("tag") or "").lower()
        if role and role != "entry":
            return False
        timestamp = self._parse_optional_datetime(fill.get("timestamp"))
        if timestamp is None:
            return False
        entry_time = trade.entry_time if trade.entry_time.tzinfo else trade.entry_time.replace(tzinfo=timezone.utc)
        return abs((timestamp - entry_time).total_seconds()) <= 1e-3

    def _register_resolved_fill_order_ids(self, fill: Fill, strategy_id: str) -> None:
        if self._coordinator is None:
            return
        register = getattr(self._coordinator, "register_order", None)
        if not callable(register):
            return
        order_ids = self._fill_order_ids(fill)
        broker = getattr(self, "_broker", None)
        local_to_oid = getattr(broker, "_local_to_oid", None)
        if isinstance(local_to_oid, dict):
            for order_id in list(order_ids):
                exchange_id = local_to_oid.get(order_id)
                if exchange_id:
                    order_ids.append(str(exchange_id))
        for order_id in dict.fromkeys(order_ids):
            register(order_id, strategy_id)

    def _resolve_fill_owner(self, fill: Fill) -> tuple[str | None, Fill]:
        """Resolve strategy ownership while keeping OMS client IDs canonical."""
        coordinator_owner = self._coordinator_fill_owner(fill)
        oms_owner, canonical_fill = self._oms_canonical_fill(fill, coordinator_owner)
        if oms_owner:
            return oms_owner, canonical_fill
        if coordinator_owner:
            return coordinator_owner, fill
        broker_owner = self._broker_fill_owner(fill)
        if broker_owner:
            return broker_owner, fill
        return None, fill

    def _fill_order_ids(self, fill: Fill) -> list[str]:
        """Return stable candidate order IDs from a fill without duplicates."""
        return list(dict.fromkeys(
            str(order_id)
            for order_id in (fill.order_id, fill.exchange_order_id)
            if order_id
        ))

    def _coordinator_fill_owner(self, fill: Fill) -> str | None:
        if self._coordinator is None:
            return None
        owner_fn = getattr(self._coordinator, "get_strategy_for_order", None)
        if owner_fn is None:
            return None
        for order_id in self._fill_order_ids(fill):
            owner = owner_fn(order_id)
            if isinstance(owner, str) and owner:
                return owner
        return None

    def _oms_canonical_fill(
        self,
        fill: Fill,
        coordinator_owner: str | None,
    ) -> tuple[str | None, Fill]:
        oms = getattr(self, "_oms", None)
        if oms is None:
            return None, fill

        for order_id in self._fill_order_ids(fill):
            record = oms.get_order(order_id)
            if not record:
                continue
            strategy_id = str(record.get("strategy_id") or "")
            if not strategy_id:
                continue
            client_id = str(record.get("client_order_id") or fill.order_id or "")
            exchange_id = str(record.get("exchange_order_id") or fill.exchange_order_id or "")
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            role = str(fill.tag or record.get("role") or metadata.get("tag") or "")

            if coordinator_owner and coordinator_owner != strategy_id:
                log.warning(
                    "engine.fill_owner_mismatch",
                    order_id=fill.order_id,
                    exchange_order_id=fill.exchange_order_id,
                    coordinator_owner=coordinator_owner,
                    oms_owner=strategy_id,
                )
                self._record_fill_discrepancy(
                    fill,
                    kind="fill_owner_mismatch",
                    description="Coordinator and durable OMS disagree on fill ownership; OMS owner was used.",
                    strategy_id=strategy_id,
                    metadata={"coordinator_owner": coordinator_owner, "oms_owner": strategy_id},
                )
            if client_id and (
                client_id != fill.order_id
                or (exchange_id and not fill.exchange_order_id)
            ):
                self._emit_reconciliation_event(
                    lifecycle_event_kind="inferred_fill",
                    action="map_exchange_fill_to_oms_order",
                    status="applied",
                    description="Exchange fill was mapped to the durable OMS client order.",
                    strategy_id=strategy_id,
                    fill=fill,
                    metadata={
                        "oms_client_order_id": client_id,
                        "oms_exchange_order_id": exchange_id,
                        "incoming_order_id": fill.order_id,
                        "incoming_exchange_order_id": fill.exchange_order_id,
                    },
                    severity="info",
                )

            if self._coordinator is not None:
                if client_id:
                    self._coordinator.register_order(client_id, strategy_id)
                if exchange_id:
                    self._coordinator.register_order(exchange_id, strategy_id)

            if client_id and (
                client_id != fill.order_id
                or (exchange_id and not fill.exchange_order_id)
                or (role and role != fill.tag)
            ):
                raw = dict(fill.raw)
                if role and not raw.get("tag"):
                    raw["tag"] = role
                fill = replace(
                    fill,
                    order_id=client_id,
                    exchange_order_id=fill.exchange_order_id or exchange_id,
                    tag=role or fill.tag,
                    raw=raw,
                )
            return strategy_id, fill

        return None, fill

    def _broker_fill_owner(self, fill: Fill) -> str | None:
        if self._broker is None:
            return None
        for order_id in self._fill_order_ids(fill):
            owner = self._broker.get_order_owner(order_id)
            if owner:
                return owner
        return None

    def _is_processed_oms_fill(self, fill: Fill) -> bool:
        return self._is_processed_oms_fill_id(fill_identity(fill))

    def _is_processed_oms_fill_id(self, fill_id: str) -> bool:
        oms = getattr(self, "_oms", None)
        is_processed = getattr(oms, "is_fill_processed", None)
        return bool(callable(is_processed) and is_processed(fill_id))

    def _oms_fill_status(self, fill_id: str) -> str | None:
        oms = getattr(self, "_oms", None)
        get_status = getattr(oms, "get_fill_status", None)
        if not callable(get_status):
            return None
        return get_status(fill_id)

    def _record_fill_discrepancy(
        self,
        fill: Fill,
        *,
        fill_id: str | None = None,
        kind: str,
        description: str,
        strategy_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        oms = getattr(self, "_oms", None)
        if oms is None:
            return True
        fill_id = fill_id or fill_identity(fill)
        list_fn = getattr(oms, "list_unresolved_discrepancies", None)
        if callable(list_fn):
            for discrepancy in list_fn():
                if (
                    discrepancy.get("kind") == kind
                    and (discrepancy.get("metadata") or {}).get("fill_id") == fill_id
                ):
                    return False
        payload = {
            "fill_id": fill_id,
            "order_id": fill.order_id,
            "exchange_order_id": fill.exchange_order_id,
            "exchange_fill_id": fill.exchange_fill_id,
            "timestamp": fill.timestamp.isoformat(),
            "tag": fill.tag,
            **(metadata or {}),
        }
        record_fn = getattr(oms, "record_discrepancy", None)
        if callable(record_fn):
            record_fn(
                kind=kind,
                description=description,
                symbol=fill.symbol,
                strategy_id=strategy_id,
                metadata=payload,
            )
        lifecycle_kind = (
            "fill_processing_failed"
            if kind == "fill_processing_failed"
            else "drift_assignment"
        )
        action = (
            "admin_correction_required"
            if kind in {"unattributed_fill", "missing_strategy_slot_fill", "fill_processing_failed"}
            else "assign_drift"
        )
        self._emit_reconciliation_event(
            lifecycle_event_kind=lifecycle_kind,
            action=action,
            status="open",
            description=description,
            symbol=fill.symbol,
            strategy_id=strategy_id or "UNKNOWN",
            fill=fill,
            fill_id=fill_id,
            severity="error" if kind == "fill_processing_failed" else "warning",
            metadata={"discrepancy_kind": kind, **payload},
        )
        return True

    def _handle_startup_reconciliation(self, discrepancies: list[Any]) -> None:
        if discrepancies:
            log.warning("engine.init_discrepancies", count=len(discrepancies))
            for discrepancy in discrepancies:
                self._oms.record_discrepancy(
                    kind=getattr(discrepancy, "kind", type(discrepancy).__name__),
                    description=str(discrepancy),
                    symbol=getattr(discrepancy, "symbol", ""),
                    metadata={
                        key: str(value)
                        for key, value in getattr(discrepancy, "__dict__", {}).items()
                    },
                )
            self._freeze_entries_for_reconciliation(
                discrepancies=discrepancies,
                description="Startup reconciliation found broker/OMS drift; new entries are blocked.",
                reconciliation_status="drift_detected",
            )
            return

        unresolved = self._unresolved_reconciliation_discrepancies()
        if unresolved:
            self._freeze_entries_for_reconciliation(
                discrepancies=unresolved,
                description=(
                    "Startup found unresolved OMS reconciliation discrepancies from a prior run; "
                    "new entries are blocked."
                ),
                reconciliation_status="persisted_drift_detected",
            )
            return

        self._clear_reconciliation_entry_block()
        self._emit_reconciliation_event(
            lifecycle_event_kind="unfreeze",
            action="allow_entries",
            status="clear",
            description="Startup reconciliation matched broker truth.",
            discrepancies=[],
            severity="info",
        )

    def _freeze_entries_for_reconciliation(
        self,
        *,
        discrepancies: list[Any],
        description: str,
        reconciliation_status: str,
    ) -> None:
        manager = getattr(self, "_manager", None)
        if manager is not None:
            manager.entries_blocked_reason = _RECONCILIATION_BLOCK_REASON
        self._emit_reconciliation_event(
            lifecycle_event_kind="freeze",
            action="freeze_entries",
            status="open",
            description=description,
            discrepancies=self._reconciliation_discrepancy_payloads(discrepancies),
            severity="error",
        )
        self._emit_runtime_snapshots(
            source="reconciliation_discrepancy",
            context={"reconciliation_status": reconciliation_status},
            include_allocation=True,
        )

    def _clear_reconciliation_entry_block(self) -> None:
        manager = getattr(self, "_manager", None)
        if (
            manager is not None
            and getattr(manager, "entries_blocked_reason", "") == _RECONCILIATION_BLOCK_REASON
        ):
            manager.entries_blocked_reason = ""

    def _unresolved_reconciliation_discrepancies(self) -> list[dict[str, Any]]:
        oms = getattr(self, "_oms", None)
        list_fn = getattr(oms, "list_unresolved_discrepancies", None)
        if not callable(list_fn):
            return []
        return list(list_fn())

    def _reconciliation_discrepancy_payloads(self, discrepancies: list[Any]) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for discrepancy in discrepancies:
            payloads.append(
                dict(discrepancy)
                if isinstance(discrepancy, dict)
                else self._discrepancy_payload(discrepancy)
            )
        return payloads

    def load_instrumentation_context_from_config(self) -> None:
        """Load lineage/config evidence without starting broker connectivity."""
        portfolio_config = self._load_portfolio_config()
        strategy_config_payloads = {
            strategy_id: read_json_file(config_path)
            for strategy_id, config_path in self._config.strategy_configs.items()
        }
        deployment_manifest = read_json_file(self._config.deployment_manifest_path)
        self._lineage = from_live_engine_inputs(
            config=self._config,
            portfolio_config=portfolio_config,
            strategy_configs=strategy_config_payloads,
            deployment_manifest=deployment_manifest,
            cwd=Path.cwd(),
        )

    def record_admin_correction(
        self,
        discrepancy_id: int,
        *,
        resolution: str,
        resolved_by: str = "admin",
        action: str = "resolve_discrepancy",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Resolve a durable reconciliation discrepancy and emit correction evidence."""
        return self._record_admin_correction(
            discrepancy_id,
            resolution=resolution,
            resolved_by=resolved_by,
            action=action,
            description=description,
            metadata=metadata,
        )

    def _record_admin_correction(
        self,
        discrepancy_id: int,
        *,
        resolution: str,
        resolved_by: str = "admin",
        action: str = "resolve_discrepancy",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        oms = getattr(self, "_oms", None)
        if oms is None:
            return False
        get_fn = getattr(oms, "get_discrepancy", None)
        resolve_fn = getattr(oms, "resolve_discrepancy", None)
        discrepancy = get_fn(discrepancy_id) if callable(get_fn) else None
        if not callable(resolve_fn):
            return False
        correction_metadata = {
            "discrepancy_id": discrepancy_id,
            "resolution": resolution,
            "resolved_by": resolved_by,
            **(metadata or {}),
        }
        resolved = resolve_fn(
            discrepancy_id,
            resolution=resolution,
            resolved_by=resolved_by,
            metadata=metadata,
        )
        if resolved and callable(get_fn):
            discrepancy = get_fn(discrepancy_id) or discrepancy
        self._emit_reconciliation_event(
            lifecycle_event_kind="admin_correction",
            action=action,
            status="resolved" if resolved else "missing",
            description=description or resolution,
            discrepancies=[discrepancy] if isinstance(discrepancy, dict) else [],
            symbol=str((discrepancy or {}).get("symbol") or ""),
            strategy_id=str((discrepancy or {}).get("strategy_id") or "UNKNOWN"),
            severity="info" if resolved else "warning",
            metadata=correction_metadata,
        )
        unblocked = self._maybe_unfreeze_after_admin_correction(discrepancy_id) if resolved else False
        if resolved:
            self._emit_runtime_snapshots(
                source="reconciliation_unfreeze" if unblocked else "admin_correction",
                context={"discrepancy_id": discrepancy_id},
                include_allocation=True,
            )
        return bool(resolved)

    def _maybe_unfreeze_after_admin_correction(self, discrepancy_id: int) -> bool:
        manager = getattr(self, "_manager", None)
        if manager is None:
            return False
        if getattr(manager, "entries_blocked_reason", "") != _RECONCILIATION_BLOCK_REASON:
            return False
        oms = getattr(self, "_oms", None)
        list_fn = getattr(oms, "list_unresolved_discrepancies", None)
        unresolved = list_fn() if callable(list_fn) else []
        if unresolved:
            return False

        manager.entries_blocked_reason = ""
        self._emit_reconciliation_event(
            lifecycle_event_kind="unfreeze",
            action="allow_entries",
            status="clear",
            description="All OMS/exchange reconciliation discrepancies have been resolved; entries are unblocked.",
            discrepancies=[],
            severity="info",
            metadata={"resolved_discrepancy_id": discrepancy_id},
        )
        return True

    def _track_entry_fill(self, strategy_id: str, fill: Fill) -> None:
        if fill.tag != "entry":
            return
        position_instance_id = self._entry_fill_position_instance_id(strategy_id, fill)
        tracked = self._tracked_positions.get(fill.symbol)
        if (
            tracked is not None
            and tracked.get("strategy_id") == strategy_id
            and tracked.get("direction") == fill.side
        ):
            prev_qty = float(tracked.get("qty", 0.0))
            total_qty = prev_qty + fill.qty
            if total_qty > 0:
                tracked["entry_price"] = (
                    (tracked.get("entry_price", 0.0) * prev_qty)
                    + (fill.fill_price * fill.qty)
                ) / total_qty
            tracked["qty"] = total_qty
            tracked["entry_time"] = min(tracked["entry_time"], fill.timestamp)
            tracked["entry_commission"] = tracked.get("entry_commission", 0.0) + fill.commission
            if position_instance_id:
                tracked.setdefault("position_instance_id", position_instance_id)
            self._append_tracked_fill_refs(tracked, fill, prefix="entry")
            return

        self._tracked_positions[fill.symbol] = {
            "strategy_id": strategy_id,
            "direction": fill.side,
            "position_instance_id": position_instance_id,
            "entry_price": fill.fill_price,
            "entry_time": fill.timestamp,
            "qty": fill.qty,
            "entry_commission": fill.commission,
            "entry_fill_ids": [fill_identity(fill)],
            "entry_order_ids": self._fill_order_ids(fill),
        }

    def _entry_fill_position_instance_id(self, strategy_id: str, fill: Fill) -> str:
        raw = dict(fill.raw or {})
        explicit = str(raw.get("position_instance_id") or "")
        if explicit:
            return explicit
        lifecycle_id = self._open_lifecycle_position_instance_id(
            strategy_id,
            fill.symbol,
            fill.side,
        )
        if lifecycle_id:
            return lifecycle_id
        return entry_position_instance_id(strategy_id, fill.symbol, fill.side, fill.timestamp)

    def _open_lifecycle_position_instance_id(
        self,
        strategy_id: str,
        symbol: str,
        direction: Side | str | None,
    ) -> str:
        direction_value = direction.value if isinstance(direction, Side) else str(direction or "")
        for entry in self._lifecycle_snapshot():
            entry_strategy = str(self._entry_value(entry, "strategy_id") or "")
            entry_symbol = str(self._entry_value(entry, "symbol") or "")
            entry_direction = self._entry_value(entry, "direction")
            entry_direction_value = (
                entry_direction.value
                if isinstance(entry_direction, Side)
                else str(entry_direction or "")
            )
            try:
                qty = float(self._entry_value(entry, "qty") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if (
                entry_strategy == strategy_id
                and entry_symbol == symbol
                and entry_direction_value == direction_value
                and abs(qty) > 1e-12
            ):
                position_instance_id = str(self._entry_value(entry, "position_instance_id") or "")
                if position_instance_id:
                    return position_instance_id
        return ""

    @staticmethod
    def _entry_value(entry: Any, key: str) -> Any:
        if isinstance(entry, dict):
            return entry.get(key)
        return getattr(entry, key, None)

    def _append_tracked_fill_refs(self, tracked: dict[str, Any], fill: Fill, *, prefix: str) -> None:
        fill_ids = tracked.setdefault(f"{prefix}_fill_ids", [])
        fill_id = fill_identity(fill)
        if fill_id and fill_id not in fill_ids:
            fill_ids.append(fill_id)
        order_ids = tracked.setdefault(f"{prefix}_order_ids", [])
        for order_id in self._fill_order_ids(fill):
            if order_id not in order_ids:
                order_ids.append(order_id)

    async def _equity_snapshot_loop(self) -> None:
        """Record equity snapshots periodically."""
        while self._running:
            await asyncio.sleep(self._config.equity_snapshot_interval_sec)
            try:
                equity = self._broker.get_equity()
                self._manager.update_equity(equity)
                self._persistent.append_equity_snapshot(equity)
                self._daily_aggregator.record_equity(datetime.now(timezone.utc), equity)
                self._persistent.save_portfolio_state(self._manager.state.to_dict())
                self._emit_runtime_snapshots(source="equity_interval")

                # Write equity + positions to PostgreSQL
                if self._pg_sink is not None:
                    self._pg_sink.write_equity(equity, datetime.now(timezone.utc))
                    self._pg_sink.upsert_positions(self._build_positions_snapshot())
                    ownership = self._allocation_ownership_payload()
                    upsert_allocations = getattr(self._pg_sink, "upsert_strategy_position_allocations", None)
                    if callable(upsert_allocations):
                        upsert_allocations(ownership["strategy_allocations"])
                    upsert_exchange_positions = getattr(self._pg_sink, "upsert_exchange_positions", None)
                    if callable(upsert_exchange_positions):
                        upsert_exchange_positions(ownership["exchange_positions"])
            except Exception as exc:
                self._health.on_error("equity_snapshot")
                self._emit_error_event("equity_snapshot", exc, severity="low", recovery_action="continue")

    async def _daily_reset_loop(self) -> None:
        """Reset daily P&L counters at UTC midnight."""
        while self._running:
            now = datetime.now(timezone.utc)
            # Calculate seconds until next midnight
            midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if midnight <= now:
                midnight += timedelta(days=1)
            wait_secs = (midnight - now).total_seconds()
            await asyncio.sleep(min(wait_secs, 3600))  # check at least hourly
            if not self._running:
                break

            closeout_at = datetime.now(timezone.utc)
            if closeout_at < midnight:
                continue

            today = closeout_at.date()
            try:
                # Close out the previous UTC day before resetting counters for
                # the new day, so reconciliation sees the final prior-day state.
                yesterday = (closeout_at - timedelta(days=1)).strftime("%Y-%m-%d")
                snapshot = self._daily_aggregator.compute_snapshot(yesterday)
                snapshot.lineage = self._lineage.for_portfolio()
                snapshot.metadata.family_id = self._lineage.family_id
                snapshot.metadata.portfolio_id = self._lineage.portfolio_id
                snapshot.metadata.account_alias = self._lineage.account_alias
                snapshot.metadata.config_version = self._lineage.config_version
                snapshot.metadata.deployment_id = self._lineage.deployment_id
                snapshot.metadata.code_sha = self._lineage.code_sha
                snapshot.metadata.lineage = self._lineage.for_portfolio()
                self._emitter.emit_daily(snapshot)
                self._emit_daily_reconciliation(yesterday, snapshot)
            except Exception as exc:
                log.exception("engine.daily_snapshot_error")
                self._emit_error_event("daily_snapshot", exc, severity="low", recovery_action="continue")
            finally:
                self._manager.maybe_reset_daily(today)

    async def _health_check_loop(self) -> None:
        """Periodic health check, heartbeat, stale feed detection, reconnect check."""
        while self._running:
            await asyncio.sleep(self._config.health_check_interval_sec)
            self._health.heartbeat()

            if self._health.is_stale():
                log.warning("engine.stale_data")

            # Per-(sym, tf) stale feed detection
            stale_feeds = self._health.get_stale_feeds(_TF_INTERVALS)
            for sym, tf, elapsed in stale_feeds:
                log.error("engine.stale_feed", symbol=sym, tf=tf, elapsed_sec=round(elapsed))

            # Reconnect check
            if self._health.should_reconnect():
                status = self._health.get_status()
                log.error(
                    "engine.reconnect_needed",
                    consecutive_errors=status["consecutive_errors"],
                )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _emit_startup_snapshots(
        self,
        portfolio_config: PortfolioConfig,
        strategy_configs: dict[str, dict],
        deployment_manifest: dict,
    ) -> None:
        """Emit fail-open startup evidence snapshots for assistant ingestion."""
        try:
            started_at = self._runtime_started_at()
            deployment_metadata_paths = self._emit_deployment_metadata_artifacts(
                strategy_configs,
                portfolio_config,
                started_at=started_at,
            )
            strategy_registry = {
                slot.strategy_id: {
                    "assistant_strategy_id": getattr(slot.strategy, "name", slot.strategy_id),
                    "primary_timeframe": slot.primary_tf.value,
                    "subscribed_timeframes": sorted(tf.value for tf in slot.subscribed_tfs),
                    "symbols": list(getattr(slot.strategy, "symbols", [])),
                    "strategy_config_version": self._lineage.strategy_config_versions.get(slot.strategy_id, ""),
                }
                for slot in self._slots
            }
            self._emit_assistant_payload("deployment", {
                "deployment_id": self._lineage.deployment_id,
                "deployment_manifest_path": str(self._config.deployment_manifest_path or ""),
                "deployment_manifest_version": self._lineage.deployment_manifest_version,
                "candidate": deployment_manifest.get("candidate", ""),
                "portfolio_round": deployment_manifest.get("portfolio_round"),
                "required_strategy_ids": list(
                    deployment_manifest.get("required_strategy_ids")
                    or self._config.strategy_configs.keys()
                ),
                "strategy_config_paths": {
                    strategy_id: str(path)
                    for strategy_id, path in self._config.strategy_configs.items()
                },
                "portfolio_config_path": str(self._config.portfolio_config_path or ""),
                "portfolio_rounds_manifest_path": deployment_manifest.get("portfolio_rounds_manifest_path", ""),
                "parity_alignment_path": deployment_manifest.get("parity_alignment_path", ""),
                "deployment_manifest": strip_secret_fields(deployment_manifest),
                "code_sha": self._lineage.code_sha,
                "started_at": started_at,
                "is_testnet": self._config.is_testnet,
                "config_version": self._lineage.config_version,
                "portfolio_config_version": self._lineage.portfolio_config_version,
                "risk_config_version": self._lineage.risk_config_version,
                "allocation_version": self._lineage.allocation_version,
                "strategy_registry": strategy_registry,
                "symbol_universe": list(self._config.symbols),
                "symbols": list(self._config.symbols),
                "bot_id": getattr(self._config, "bot_id", ""),
                "family_id": self._lineage.family_id,
                "portfolio_id": self._lineage.portfolio_id,
                "account_alias": self._lineage.account_alias,
                "venue_environment": "testnet" if self._config.is_testnet else "mainnet",
                "exchange": "hyperliquid",
                "deployment_metadata_artifacts": deployment_metadata_paths,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            self._emit_startup_config_snapshots(
                portfolio_config,
                strategy_configs,
                deployment_manifest,
            )
            self._emit_assistant_payload("config_snapshot", {
                "snapshot_kind": "startup",
                "config_kind": "bundle",
                "config_version": self._lineage.config_version,
                "live_config": strip_secret_fields(self._config.to_dict(redacted=True)),
                "portfolio_config": strip_secret_fields(portfolio_config.to_dict()),
                "strategy_configs": strip_secret_fields(strategy_configs),
                "strategy_config_versions": dict(self._lineage.strategy_config_versions),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            self._emit_allocation_snapshot(source="startup")
            self._emit_runtime_snapshots(source="startup")
            self._emit_family_daily_snapshot()
            self._emit_startup_heartbeat()
        except Exception:
            log.exception("engine.startup_snapshot_failed")

    def _emit_startup_heartbeat(self) -> None:
        try:
            now = datetime.now(timezone.utc)
            sidecar_status = self._relay_health_status()
            report_payload = {
                "assessment": "startup",
                "status": "starting",
                "startup": True,
                "timestamp": now.isoformat(),
                "runtime_instance_id": self._runtime_instance_id_value(),
                "relay": sidecar_status,
                "sidecar": sidecar_status,
                "emitter_sink_failures": self._emitter.sink_failures,
            }
            if getattr(self, "_manager", None) is not None:
                report_payload["portfolio_state"] = {
                    "heat_R": self._manager.state.total_heat_R(),
                    "heat_cap_R": self._manager.config.heat_cap_R,
                    "daily_pnl_R": self._manager.state.portfolio_daily_pnl_R,
                    "open_risk_count": self._manager.state.total_positions(),
                }
            oms = getattr(self, "_oms", None)
            if oms is not None and hasattr(oms, "list_orders"):
                report_payload["oms_summary"] = {
                    "open_orders": len(oms.list_orders()),
                    "last_market_event_at": self._last_assistant_event_at.get("market_snapshot"),
                    "last_order_event_at": self._last_assistant_event_at.get("order"),
                    "last_trade_event_at": self._last_assistant_event_at.get("trade"),
                }
            self._emitter.emit_health_report(HealthReportSnapshot(
                timestamp=now.isoformat(),
                report=report_payload,
                metadata=EventMetadata.create(
                    bot_id=getattr(self._config, "bot_id", ""),
                    strategy_id="system",
                    exchange_ts=now,
                    event_type="heartbeat",
                    payload_key=f"startup:{self._runtime_instance_id_value()}",
                    lineage=self._lineage.for_portfolio(),
                    family_id=self._lineage.family_id,
                    portfolio_id=self._lineage.portfolio_id,
                    account_alias=self._lineage.account_alias,
                    config_version=self._lineage.config_version,
                    deployment_id=self._lineage.deployment_id,
                    code_sha=self._lineage.code_sha,
                ),
                lineage=self._lineage.for_portfolio(),
            ))
        except Exception:
            log.exception("engine.startup_heartbeat_failed")

    def _emit_startup_config_snapshots(
        self,
        portfolio_config: PortfolioConfig,
        strategy_configs: dict[str, dict],
        deployment_manifest: dict,
    ) -> None:
        loaded_at = datetime.now(timezone.utc).isoformat()
        live_config = strip_secret_fields(self._config.to_dict(redacted=True))
        portfolio_payload = strip_secret_fields(portfolio_config.to_dict())
        risk_payload = subset_keys(portfolio_payload, RISK_CONFIG_KEYS)
        allocation_payload = subset_keys(portfolio_payload, ALLOCATION_CONFIG_KEYS)

        self._emit_config_snapshot(
            config_kind="live",
            config_path="",
            config_version=self._lineage.config_version,
            redacted_config=live_config,
            hash_inputs={"live_config": live_config},
            loaded_at=loaded_at,
        )
        self._emit_config_snapshot(
            config_kind="portfolio",
            config_path=str(self._config.portfolio_config_path or ""),
            config_version=self._lineage.portfolio_config_version,
            redacted_config=portfolio_payload,
            hash_inputs={"portfolio_config": portfolio_payload},
            loaded_at=loaded_at,
        )
        self._emit_config_snapshot(
            config_kind="risk",
            config_path=str(self._config.portfolio_config_path or ""),
            config_version=self._lineage.risk_config_version,
            redacted_config=risk_payload,
            hash_inputs={"risk_config": risk_payload},
            loaded_at=loaded_at,
        )
        self._emit_config_snapshot(
            config_kind="allocation",
            config_path=str(self._config.portfolio_config_path or ""),
            config_version=self._lineage.allocation_version,
            redacted_config=allocation_payload,
            hash_inputs={"allocation_config": allocation_payload},
            loaded_at=loaded_at,
        )
        for strategy_id, payload in sorted(strategy_configs.items()):
            redacted_payload = strip_secret_fields(payload)
            self._emit_config_snapshot(
                config_kind="strategy",
                config_path=str(self._config.strategy_configs.get(strategy_id, "")),
                config_version=self._lineage.strategy_config_versions.get(strategy_id, ""),
                redacted_config=redacted_payload,
                hash_inputs={"strategy_config": redacted_payload},
                loaded_at=loaded_at,
                strategy_id=strategy_id,
            )
        if deployment_manifest:
            redacted_manifest = strip_secret_fields(deployment_manifest)
            self._emit_config_snapshot(
                config_kind="deployment_manifest",
                config_path=str(self._config.deployment_manifest_path or ""),
                config_version=self._lineage.deployment_manifest_version,
                redacted_config=redacted_manifest,
                hash_inputs={"deployment_manifest": redacted_manifest},
                loaded_at=loaded_at,
            )
        if self._config.asset_meta_path is not None:
            asset_meta = self._file_metadata(self._config.asset_meta_path)
            self._emit_config_snapshot(
                config_kind="asset_meta",
                config_path=str(self._config.asset_meta_path),
                config_version=stable_hash(asset_meta),
                redacted_config=asset_meta,
                hash_inputs={"asset_meta": asset_meta},
                loaded_at=loaded_at,
            )

    def _emit_config_snapshot(
        self,
        *,
        config_kind: str,
        config_path: str,
        config_version: str,
        redacted_config: dict,
        hash_inputs: dict,
        loaded_at: str,
        strategy_id: str = "",
    ) -> None:
        identity_seed = {
            "config_kind": config_kind,
            "config_version": config_version,
        }
        snapshot_id = stable_hash(identity_seed)
        self._emit_assistant_payload("config_snapshot", {
            "config_snapshot_event_id": snapshot_id,
            "logical_event_id": snapshot_id,
            "snapshot_kind": "startup",
            "config_kind": config_kind,
            "config_path": config_path,
            "config_version": config_version,
            "config_identity_seed": identity_seed,
            "redacted_config": strip_secret_fields(redacted_config),
            "hash_inputs": strip_secret_fields(hash_inputs),
            "loaded_at": loaded_at,
            "timestamp": loaded_at,
            "strategy_id": strategy_id,
        })

    def _emit_deployment_metadata_artifacts(
        self,
        strategy_configs: dict[str, dict],
        portfolio_config: PortfolioConfig,
        *,
        started_at: str,
    ) -> dict[str, str]:
        """Write runtime deployment metadata artifacts for assistant bridge import."""
        artifacts: dict[str, str] = {}
        try:
            repo_url = self._git_remote_url()
            worktree_clean = self._git_worktree_clean()
            strategy_version = self._package_version()
            host_fingerprint = self._runtime_host_fingerprint()
            emitted_at = datetime.now(timezone.utc).isoformat()
            portfolio_payload = strip_secret_fields(portfolio_config.to_dict())
            contract_root = self._bridge_contract_root()
            for strategy_id, bridge_id in _STRATEGY_BRIDGE_IDS.items():
                strategy_payload = strip_secret_fields(strategy_configs.get(strategy_id, {}))
                strategy_config_present = bool(strategy_payload)
                contract_dir = contract_root / bridge_id
                contract_path = contract_dir / "strategy_plugin_contract.json"
                contract_metadata_path = contract_dir / "deployment_metadata.json"
                state_metadata_path = (
                    self._config.state_dir
                    / "deployment_metadata"
                    / bridge_id
                    / "deployment_metadata.json"
                )
                telemetry_schema_versions = (
                    self._required_telemetry_schema_versions(contract_path)
                    or ["trade_event_v1"]
                )
                config_hash = stable_hash({
                    "strategy_config": strategy_payload,
                    "portfolio_config": portfolio_payload,
                }, length=64)
                contract_hash = self._file_sha256(contract_path)
                approval_ready = bool(
                    repo_url
                    and self._lineage.code_sha
                    and self._lineage.code_sha != "unknown"
                    and worktree_clean
                    and contract_hash
                    and strategy_config_present
                )
                metadata = {
                    "metadata_source": "live_bot_runtime_deployment_metadata_v1",
                    "emission_environment": self._emission_environment(),
                    "repo_url": repo_url,
                    "source_control_origin": repo_url,
                    "deployed_commit_sha": self._lineage.code_sha,
                    "source_control_commit_sha": self._lineage.code_sha,
                    "source_control_worktree_clean": worktree_clean,
                    "bot_id": getattr(self._config, "bot_id", ""),
                    "portfolio_id": self._lineage.portfolio_id,
                    "strategy_id": bridge_id,
                    "source_strategy_id": strategy_id,
                    "config_hash": config_hash,
                    "strategy_version": strategy_version,
                    "config_version": self._lineage.strategy_config_versions.get(strategy_id, ""),
                    "deployment_id": self._lineage.deployment_id,
                    "telemetry_schema_version": telemetry_schema_versions[0],
                    "telemetry_schema_versions": telemetry_schema_versions,
                    "assistant_event_schema_version": "assistant_event_v1",
                    "strategy_plugin_contract_path": contract_path.as_posix(),
                    "strategy_plugin_contract_hash": contract_hash,
                    "deployment_metadata_path": contract_metadata_path.as_posix(),
                    "state_deployment_metadata_path": state_metadata_path.as_posix(),
                    "emitted_at_utc": emitted_at,
                    "live_runtime_started_at_utc": started_at,
                    "runtime_entrypoint": "crypto_trader.cli:live",
                    "runtime_instance_id": self._runtime_instance_id_value(),
                    "runtime_host_fingerprint": host_fingerprint,
                    "dry_run": False,
                    "portfolio_config_version": self._lineage.portfolio_config_version,
                    "risk_config_version": self._lineage.risk_config_version,
                    "allocation_version": self._lineage.allocation_version,
                    "strategy_config_version": self._lineage.strategy_config_versions.get(strategy_id, ""),
                    "symbol_universe": list(self._lineage.symbol_universe),
                    "portfolio_config_hash": stable_hash(portfolio_payload, length=64),
                    "deployment_manifest_version": self._lineage.deployment_manifest_version,
                    "strategy_config_present": strategy_config_present,
                    "approval_ready": approval_ready,
                    "approval_blockers": self._deployment_metadata_blockers(
                        repo_url=repo_url,
                        worktree_clean=worktree_clean,
                        contract_hash=contract_hash,
                        strategy_config_present=strategy_config_present,
                    ),
                }
                encoded = json.dumps(metadata, sort_keys=True, indent=2, default=str) + "\n"
                for path in (contract_metadata_path, state_metadata_path):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(encoded, encoding="utf-8")
                artifacts[bridge_id] = str(contract_metadata_path)
        except Exception:
            log.exception("engine.deployment_metadata_artifacts_failed")
        return artifacts

    @staticmethod
    def _bridge_contract_root() -> Path:
        configured = os.environ.get("CRYPTO_TRADER_BRIDGE_CONTRACT_ROOT", "")
        return Path(configured).expanduser() if configured else _DEFAULT_BRIDGE_CONTRACT_ROOT

    def _emission_environment(self) -> str:
        allowed = {"live_bot", "vps", "paper_vps", "production_vps"}
        configured = os.environ.get("CRYPTO_TRADER_EMISSION_ENVIRONMENT", "")
        if configured in allowed:
            return configured
        return "paper_vps" if self._config.is_testnet else "production_vps"

    def _deployment_metadata_blockers(
        self,
        *,
        repo_url: str,
        worktree_clean: bool,
        contract_hash: str,
        strategy_config_present: bool,
    ) -> list[str]:
        blockers: list[str] = []
        if not repo_url:
            blockers.append("missing_git_remote_origin")
        if not self._lineage.code_sha or self._lineage.code_sha == "unknown":
            blockers.append("missing_deployed_commit_sha")
        if not worktree_clean:
            blockers.append("source_control_worktree_dirty")
        if not contract_hash:
            blockers.append("missing_strategy_plugin_contract_hash")
        if not strategy_config_present:
            blockers.append("missing_strategy_config")
        return blockers

    @staticmethod
    def _file_sha256(path: Path) -> str:
        try:
            data = path.read_bytes()
            if LiveEngine._is_text_artifact(path, data):
                data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            return hashlib.sha256(data).hexdigest()
        except Exception:
            return ""

    @staticmethod
    def _is_text_artifact(path: Path, data: bytes) -> bool:
        if path.suffix.lower() in _TEXT_ARTIFACT_SUFFIXES:
            return True
        if b"\x00" in data:
            return False
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True

    @staticmethod
    def _required_telemetry_schema_versions(contract_path: Path) -> list[str]:
        try:
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []
        values = payload.get("required_telemetry_schemas") or []
        if not isinstance(values, list):
            values = [values]
        return [str(value).strip() for value in values if str(value or "").strip()]

    @staticmethod
    def _file_metadata(path: Path) -> dict[str, Any]:
        try:
            stat = path.stat()
            return {
                "path": str(path),
                "exists": True,
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": LiveEngine._file_sha256(path) if path.is_file() else "",
            }
        except Exception:
            return {"path": str(path), "exists": False}

    @staticmethod
    def _git_remote_url() -> str:
        try:
            completed = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            return completed.stdout.strip()
        except Exception:
            return ""

    @staticmethod
    def _git_worktree_clean() -> bool:
        try:
            completed = subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            return completed.stdout.strip() == ""
        except Exception:
            return False

    @staticmethod
    def _package_version() -> str:
        try:
            return version("crypto-trader")
        except PackageNotFoundError:
            return "0.1.0"
        except Exception:
            return "unknown"

    @staticmethod
    def _runtime_host_fingerprint() -> str:
        host_payload = {
            "node": socket.gethostname(),
            "platform": platform.system(),
            "machine": platform.machine(),
        }
        return stable_hash(host_payload, length=32)

    def _runtime_started_at(self) -> str:
        value = getattr(self, "_runtime_started_at_utc", "")
        if value:
            return str(value)
        value = datetime.now(timezone.utc).isoformat()
        self._runtime_started_at_utc = value
        return value

    def _runtime_instance_id_value(self) -> str:
        value = getattr(self, "_runtime_instance_id", "")
        if value:
            return str(value)
        value = stable_hash({
            "bot_id": getattr(self._config, "bot_id", ""),
            "portfolio_id": getattr(self._config, "portfolio_id", "default"),
            "started_at": self._runtime_started_at(),
        })
        self._runtime_instance_id = value
        return value

    def _emit_assistant_payload(self, event_type: str, payload: dict) -> None:
        """Emit a generic assistant event through the fail-open emitter."""
        try:
            if not hasattr(self, "_emitter") or not hasattr(self, "_lineage"):
                return
            timestamp = self._payload_timestamp(payload)
            nested_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            strategy_id = str(payload.get("strategy_id") or nested_metadata.get("strategy_id") or "")
            lineage = (
                self._lineage.for_strategy(strategy_id)
                if strategy_id
                else self._lineage.for_portfolio()
            )
            explicit_event_id = payload.get(f"{event_type}_event_id") or payload.get("event_id")
            if not explicit_event_id and event_type == "order":
                explicit_event_id = payload.get("order_event_id")
            if not explicit_event_id and event_type == "portfolio_rule":
                explicit_event_id = payload.get("portfolio_rule_event_id")
            if not explicit_event_id and event_type == "risk_decision":
                explicit_event_id = payload.get("risk_decision_id")
            if not explicit_event_id and event_type == "decision_event":
                explicit_event_id = payload.get("decision_event_id") or payload.get("decision_id")
            if not explicit_event_id and event_type == "market_snapshot":
                explicit_event_id = payload.get("market_snapshot_id") or payload.get("bar_id")
            if not explicit_event_id and event_type == "fill":
                explicit_event_id = payload.get("fill_event_id")
            if not explicit_event_id and event_type == "reconciliation_event":
                explicit_event_id = payload.get("reconciliation_event_id")
            if not explicit_event_id and event_type == "daily_reconciliation":
                explicit_event_id = payload.get("daily_reconciliation_id")
            if not explicit_event_id and event_type == "deployment":
                explicit_event_id = payload.get("deployment_id")
            payload_key = str(explicit_event_id or stable_hash({
                "event_type": event_type,
                "payload": payload,
            }))
            metadata = EventMetadata.create(
                bot_id=getattr(self._config, "bot_id", ""),
                strategy_id=strategy_id or "portfolio",
                exchange_ts=timestamp,
                event_type=event_type,
                payload_key=payload_key,
                bar_id=str(payload.get("bar_id") or ""),
                lineage=lineage,
                family_id=self._lineage.family_id,
                portfolio_id=self._lineage.portfolio_id,
                account_alias=self._lineage.account_alias,
                config_version=self._lineage.config_version,
                deployment_id=self._lineage.deployment_id,
                code_sha=self._lineage.code_sha,
            )
            if explicit_event_id:
                metadata.event_id = str(explicit_event_id)
            event = GenericInstrumentationEvent(
                metadata=metadata,
                payload={
                    **payload,
                    "strategy_id": strategy_id or payload.get("strategy_id", ""),
                    "event_type": event_type,
                    "lineage": lineage,
                },
                lineage=lineage,
                logical_event_id=str(payload.get("logical_event_id") or explicit_event_id or metadata.event_id),
            )
            self._emitter.emit(event_type, event)
            self._last_assistant_event_at[event_type] = datetime.now(timezone.utc).isoformat()
        except Exception:
            log.exception("engine.assistant_event_emit_failed", event_type=event_type)

    @staticmethod
    def _payload_timestamp(payload: dict) -> datetime:
        raw = payload.get("timestamp") or payload.get("exchange_timestamp") or payload.get("decision_time")
        if isinstance(raw, datetime):
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        if isinstance(raw, str) and raw:
            try:
                parsed = datetime.fromisoformat(raw)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    @staticmethod
    def _parse_optional_datetime(raw: Any) -> datetime | None:
        if isinstance(raw, datetime):
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        if isinstance(raw, str) and raw:
            try:
                parsed = datetime.fromisoformat(raw)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return None

    def _emit_runtime_snapshots(
        self,
        *,
        source: str = "interval",
        context: dict[str, Any] | None = None,
        include_allocation: bool = False,
        extra_positions: list[dict[str, Any]] | None = None,
    ) -> None:
        try:
            if not hasattr(self, "_emitter") or not hasattr(self, "_lineage"):
                return
            now = datetime.now(timezone.utc).isoformat()
            context = dict(context or {})
            portfolio_snapshot = {
                **self._portfolio_snapshot_payload(source=source, timestamp=now),
                **context,
            }
            self._emit_assistant_payload("portfolio_snapshot", portfolio_snapshot)
            positions = self._build_positions_snapshot()
            positions.extend(extra_positions or [])
            for position in positions:
                self._emit_assistant_payload(
                    "position_snapshot",
                    self._position_snapshot_payload(
                        position,
                        source=source,
                        timestamp=now,
                        context=context,
                    ),
                )
            self._emit_position_ownership_snapshots(source=source, timestamp=now, context=context)
            self._emit_assistant_payload("correlation_exposure", {
                "timestamp": now,
                "source": source,
                "exposures": self._correlation_exposure_payload(),
                **context,
            })
            if include_allocation:
                self._emit_allocation_snapshot(source=source, timestamp=now, context=context)
        except Exception:
            log.exception("engine.runtime_snapshot_emit_failed", source=source)

    def _position_snapshot_payload(
        self,
        position: dict[str, Any],
        *,
        source: str,
        timestamp: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            **context,
            **position,
            "timestamp": timestamp,
            "source": source,
        }
        fill_symbol = context.get("symbol")
        fill_strategy_id = str(context.get("strategy_id") or "")
        if fill_symbol:
            payload.setdefault("fill_symbol", fill_symbol)
        if context.get("side"):
            payload.setdefault("fill_side", context.get("side"))
        if context.get("tag"):
            payload.setdefault("fill_tag", context.get("tag"))
        if (
            fill_symbol
            and position.get("symbol") == fill_symbol
            and fill_strategy_id
            and self._unknown_strategy_id(position.get("strategy_id"))
        ):
            payload["strategy_id"] = fill_strategy_id
        unknown_allocation = self._unknown_strategy_id(payload.get("strategy_id"))
        payload.setdefault("position_instance_id", "")
        payload.setdefault("allocated_qty", 0.0 if unknown_allocation else payload.get("qty", 0.0))
        payload.setdefault("net_exchange_qty", payload.get("qty", 0.0))
        payload.setdefault("allocation_confidence", "unknown" if unknown_allocation else "inferred")
        payload.setdefault("allocation_source", payload.get("source", "broker"))
        payload.setdefault("unknown_allocation", unknown_allocation)
        payload.setdefault("unallocated_qty", payload.get("qty", 0.0) if unknown_allocation else 0.0)
        payload.setdefault("entry_order_ids", [])
        payload.setdefault("entry_fill_ids", [])
        return payload

    @staticmethod
    def _unknown_strategy_id(strategy_id: Any) -> bool:
        return str(strategy_id or "").strip().lower() in {"", "unknown"}

    def _emit_allocation_snapshot(
        self,
        *,
        source: str,
        timestamp: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        manager = getattr(self, "_manager", None)
        config = manager.config if manager else None
        strategies = list(getattr(config, "strategies", ()) if config else ())
        payload = {
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "source": source,
            "allocation_version": self._lineage.allocation_version,
            "portfolio_config_version": self._lineage.portfolio_config_version,
            "risk_config_version": self._lineage.risk_config_version,
            "strategy_allocations": [
                {
                    "strategy_id": alloc.strategy_id,
                    "enabled": alloc.enabled,
                    "base_risk_pct": alloc.base_risk_pct,
                    "max_concurrent": alloc.max_concurrent,
                    "daily_stop_R": alloc.daily_stop_R,
                    "priority": alloc.priority,
                }
                for alloc in strategies
            ],
            "enabled_strategies": [alloc.strategy_id for alloc in strategies if alloc.enabled],
            "disabled_strategies": [alloc.strategy_id for alloc in strategies if not alloc.enabled],
            "heat_cap_R": getattr(config, "heat_cap_R", None),
            "directional_cap_R": getattr(config, "directional_cap_R", None),
            "symbol_collision": getattr(config, "symbol_collision", None),
            "symbol_exposure_cap_R": getattr(config, "symbol_exposure_cap_R", None),
            "drawdown_tiers": list(getattr(config, "dd_tiers", ()) or ()),
            "priority_headroom_R": getattr(config, "priority_headroom_R", None),
            "priority_reserve_threshold": getattr(config, "priority_reserve_threshold", None),
        }
        payload.update(context or {})
        self._emit_assistant_payload("allocation_snapshot", payload)

    def _emit_position_ownership_snapshots(
        self,
        *,
        source: str,
        timestamp: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        ownership = self._allocation_ownership_payload(timestamp=timestamp)
        base = {
            "timestamp": timestamp,
            "source": source,
            "snapshot_kind": "position_ownership",
            "schema_version": 1,
            **dict(context or {}),
        }
        for allocation in ownership["strategy_allocations"]:
            self._emit_assistant_payload("position_allocation_snapshot", {
                **base,
                **allocation,
                "unknown_allocation": False,
                "unallocated_qty": 0.0,
                "allocation_confidence": allocation.get("confidence"),
                "allocation_source": allocation.get("source"),
            })
        for residual in ownership["residuals"]:
            self._emit_assistant_payload("position_allocation_snapshot", {
                **base,
                **residual,
                "position_instance_id": "",
                "strategy_id": "",
                "allocation_confidence": "unknown",
                "allocation_source": "exchange_residual",
            })

    def _allocation_ownership_payload(self, *, timestamp: str | None = None) -> dict[str, Any]:
        observed_at = _parse_datetime_or_now(timestamp)
        lifecycle_entries = self._lifecycle_snapshot()
        manager = getattr(self, "_manager", None)
        open_risks = list(manager.state.open_risks) if manager is not None else []
        allocations = derive_strategy_position_allocations(
            lifecycle_entries,
            open_risks,
            observed_at=observed_at,
        )
        broker_positions = self._broker.get_positions() if getattr(self, "_broker", None) else []
        exchange_positions = exchange_net_positions(broker_positions, observed_at=observed_at)
        residuals = allocation_residuals(exchange_positions, allocations)
        return {
            "timestamp": observed_at.isoformat(),
            "exchange_positions": [position.to_dict() for position in exchange_positions],
            "strategy_allocations": [allocation.to_dict() for allocation in allocations],
            "residuals": [residual.to_dict() for residual in residuals],
            "allocation_count": len(allocations),
            "unallocated_exposure_count": len(residuals),
            "max_allocation_net_residual": max(
                (abs(residual.unallocated_qty) for residual in residuals),
                default=0.0,
            ),
            "position_ownership_drift": bool(residuals),
        }

    def _allocation_drift_discrepancies(
        self,
        actual_positions: list[Position] | None = None,
    ) -> list[Discrepancy]:
        observed_at = datetime.now(timezone.utc)
        manager = getattr(self, "_manager", None)
        open_risks = list(manager.state.open_risks) if manager is not None else []
        allocations = derive_strategy_position_allocations(
            self._lifecycle_snapshot(),
            open_risks,
            observed_at=observed_at,
        )
        exchange_positions = exchange_net_positions(
            list(actual_positions or []),
            observed_at=observed_at,
        )
        residuals = allocation_residuals(exchange_positions, allocations)
        return [
            Discrepancy(
                symbol=residual.symbol,
                kind="position_ownership_drift",
                expected=(
                    f"{residual.direction.value} allocated_qty="
                    f"{residual.allocated_qty}"
                ),
                actual=(
                    f"exchange_qty={residual.net_exchange_qty} "
                    f"residual={residual.unallocated_qty} "
                    f"unknown_allocation={residual.unknown_allocation}"
                ),
            )
            for residual in residuals
        ]

    def _portfolio_snapshot_payload(self, *, source: str, timestamp: str) -> dict:
        manager = getattr(self, "_manager", None)
        state = manager.state if manager else None
        config = manager.config if manager else None
        positions = self._build_positions_snapshot()
        open_orders = []
        oms = getattr(self, "_oms", None)
        list_open_orders = getattr(oms, "list_open_orders", None)
        if callable(list_open_orders):
            try:
                open_orders = list_open_orders()
            except Exception:
                open_orders = []
        ownership = self._allocation_ownership_payload(timestamp=timestamp)
        return {
            "timestamp": timestamp,
            "source": source,
            "portfolio_id": self._lineage.portfolio_id,
            "account_alias": self._lineage.account_alias,
            "equity": state.equity if state else None,
            "peak_equity": state.peak_equity if state else None,
            "drawdown_pct": state.dd_pct() if state else None,
            "heat_R": state.total_heat_R() if state else None,
            "total_heat_R": state.total_heat_R() if state else None,
            "heat_cap_R": config.heat_cap_R if config else None,
            "directional_cap_R": config.directional_cap_R if config else None,
            "directional_risk_R": {
                "long": state.directional_risk_R(Side.LONG),
                "short": state.directional_risk_R(Side.SHORT),
            } if state else {},
            "symbol_risk_R": self._symbol_risk_payload() if state else {},
            "max_total_positions": config.max_total_positions if config else None,
            "portfolio_daily_pnl_R": state.portfolio_daily_pnl_R if state else None,
            "portfolio_daily_stop_R": config.portfolio_daily_stop_R if config else None,
            "strategy_daily_pnl_R": dict(state.daily_pnl_R) if state else {},
            "open_risk_count": state.total_positions() if state else 0,
            "open_risks": state.to_dict().get("open_risks", []) if state else [],
            "positions_count": len(positions),
            "pending_orders_count": len(open_orders),
            "allocation_count": ownership["allocation_count"],
            "unallocated_exposure_count": ownership["unallocated_exposure_count"],
            "max_allocation_net_residual": ownership["max_allocation_net_residual"],
            "position_ownership_drift": ownership["position_ownership_drift"],
            "risk_config_version": self._lineage.risk_config_version,
            "allocation_version": self._lineage.allocation_version,
        }

    def _symbol_risk_payload(self) -> dict[str, dict[str, float]]:
        manager = getattr(self, "_manager", None)
        if manager is None:
            return {}
        return {
            symbol: {
                "long_R": manager.state.symbol_risk_R(symbol, Side.LONG),
                "short_R": manager.state.symbol_risk_R(symbol, Side.SHORT),
            }
            for symbol in getattr(self._config, "symbols", [])
        }

    def _emit_fill_lifecycle_snapshots(
        self,
        fill_id: str,
        strategy_id: str,
        fill: Fill,
        *,
        source: str,
        closed_trade: Trade | None = None,
    ) -> None:
        """Emit exposure snapshots at the moment a fill changes real exposure."""
        context = self._fill_snapshot_context(fill_id, strategy_id, fill)
        extra_positions = (
            [self._flat_position_snapshot(strategy_id, fill, closed_trade=closed_trade)]
            if source == "exit_fill" and self._position_is_flat_after_fill(fill, closed_trade)
            else []
        )
        self._emit_runtime_snapshots(
            source=source,
            context=context,
            include_allocation=True,
            extra_positions=extra_positions,
        )

    def _position_is_flat_after_fill(self, fill: Fill, closed_trade: Trade | None) -> bool:
        broker = getattr(self, "_broker", None)
        get_positions = getattr(broker, "get_positions", None)
        if not callable(get_positions):
            return closed_trade is not None
        try:
            return not any(
                position.symbol == fill.symbol and abs(float(position.qty)) > 0.0
                for position in get_positions()
            )
        except Exception:
            log.exception("engine.position_flat_check_failed", symbol=fill.symbol)
            return closed_trade is not None

    def _flat_position_snapshot(
        self,
        strategy_id: str,
        fill: Fill,
        *,
        closed_trade: Trade | None = None,
    ) -> dict[str, Any]:
        tracked = self._tracked_positions.get(fill.symbol, {})
        entry_time = tracked.get("entry_time") or getattr(closed_trade, "entry_time", None)
        direction = tracked.get("direction") or getattr(closed_trade, "direction", None)
        direction_value = direction.value if isinstance(direction, Side) else str(direction or "unknown")
        avg_entry = float(
            tracked.get("entry_price")
            or getattr(closed_trade, "entry_price", 0.0)
            or fill.fill_price
        )
        entry_commission = float(tracked.get("entry_commission", 0.0))
        realized_pnl = float(getattr(closed_trade, "pnl", 0.0) or 0.0)
        position_instance_id = self._closed_position_instance_id(
            strategy_id,
            fill,
            tracked=tracked,
            closed_trade=closed_trade,
            entry_time=entry_time,
        )
        return {
            "position_instance_id": position_instance_id,
            "strategy_id": strategy_id,
            "symbol": fill.symbol,
            "direction": direction_value,
            "qty": 0.0,
            "avg_entry": avg_entry,
            "mark_price": fill.fill_price,
            "notional_usd": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": realized_pnl,
            "risk_r": 0.0,
            "risk_R": 0.0,
            "stop_price": None,
            "liquidation_price": None,
            "liquidation_distance_pct": None,
            "fees_paid": entry_commission + fill.commission,
            "funding_paid": float(
                tracked.get("funding_paid", getattr(closed_trade, "funding_paid", 0.0) or 0.0)
            ),
            "mfe_r": tracked.get("mfe_r", getattr(closed_trade, "mfe_r", None)),
            "mae_r": tracked.get("mae_r", getattr(closed_trade, "mae_r", None)),
            "open_order_ids": [],
            "entry_time": entry_time,
            "exit_time": fill.timestamp,
            "position_status": "closed",
        }

    def _closed_position_instance_id(
        self,
        strategy_id: str,
        fill: Fill,
        *,
        tracked: dict[str, Any],
        closed_trade: Trade | None,
        entry_time: Any,
    ) -> str:
        if closed_trade is not None:
            position_instance_id = self._trade_position_instance_id(closed_trade)
            if position_instance_id:
                return position_instance_id
        position_instance_id = str(tracked.get("position_instance_id") or "")
        if position_instance_id:
            return position_instance_id
        raw = dict(fill.raw or {})
        position_instance_id = str(raw.get("position_instance_id") or "")
        if position_instance_id:
            return position_instance_id
        position_instance_id = self._position_instance_id_from_fill_order(fill)
        if position_instance_id:
            return position_instance_id
        if closed_trade is not None:
            position_instance_id = self._resolve_trade_position_instance_id(
                strategy_id,
                closed_trade,
            )
            if position_instance_id:
                return position_instance_id
        if hasattr(entry_time, "timestamp"):
            direction = getattr(closed_trade, "direction", None) or tracked.get("direction") or fill.side
            return entry_position_instance_id(strategy_id, fill.symbol, direction, entry_time)
        return ""

    def _position_instance_id_from_fill_order(self, fill: Fill) -> str:
        oms = getattr(self, "_oms", None)
        get_order = getattr(oms, "get_order", None)
        if not callable(get_order):
            return ""
        for order_id in (fill.order_id, fill.exchange_order_id):
            if not order_id:
                continue
            try:
                order = get_order(order_id)
            except Exception:
                log.exception("engine.fill_order_position_id_lookup_failed", order_id=order_id)
                continue
            if not order:
                continue
            metadata = dict(order.get("metadata") or {})
            position_instance_id = str(
                order.get("position_instance_id")
                or metadata.get("position_instance_id")
                or ""
            )
            if position_instance_id:
                return position_instance_id
        return ""

    def _emit_unresolved_fill_snapshots(
        self,
        fill_id: str,
        fill: Fill,
        *,
        strategy_id: str,
        reason: str,
    ) -> None:
        context = {
            **self._fill_snapshot_context(fill_id, strategy_id, fill),
            "reconciliation_status": reason,
            "unknown_allocation": strategy_id in {"", "UNKNOWN", "unknown"},
        }
        extra_positions = (
            [self._unresolved_flat_position_snapshot(fill_id, strategy_id, fill, reason=reason)]
            if self._broker_position_is_flat_for_symbol(fill.symbol)
            else []
        )
        self._emit_runtime_snapshots(
            source=reason,
            context=context,
            include_allocation=True,
            extra_positions=extra_positions,
        )

    def _broker_position_is_flat_for_symbol(self, symbol: str) -> bool:
        broker = getattr(self, "_broker", None)
        get_positions = getattr(broker, "get_positions", None)
        if not callable(get_positions):
            return True
        try:
            return not any(
                position.symbol == symbol and abs(float(position.qty)) > 0.0
                for position in get_positions()
            )
        except Exception:
            log.exception("engine.unresolved_fill_position_check_failed", symbol=symbol)
            return True

    @staticmethod
    def _unresolved_flat_position_snapshot(
        fill_id: str,
        strategy_id: str,
        fill: Fill,
        *,
        reason: str,
    ) -> dict[str, Any]:
        snapshot_strategy_id = strategy_id if strategy_id and strategy_id.lower() != "unknown" else "UNKNOWN"
        return {
            "position_instance_id": f"unresolved:{snapshot_strategy_id}:{fill.symbol}:{fill_id}",
            "strategy_id": snapshot_strategy_id,
            "symbol": fill.symbol,
            "direction": fill.side.value if fill.side else "unknown",
            "qty": 0.0,
            "avg_entry": fill.fill_price,
            "mark_price": fill.fill_price,
            "notional_usd": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "risk_r": 0.0,
            "risk_R": 0.0,
            "stop_price": None,
            "liquidation_price": None,
            "liquidation_distance_pct": None,
            "fees_paid": fill.commission,
            "funding_paid": 0.0,
            "mfe_r": None,
            "mae_r": None,
            "open_order_ids": [],
            "entry_time": None,
            "exit_time": fill.timestamp,
            "position_status": "unresolved_flat",
            "reconciliation_status": reason,
        }

    def _fill_snapshot_context(self, fill_id: str, strategy_id: str, fill: Fill) -> dict[str, Any]:
        return {
            "fill_id": fill_id,
            "client_order_id": fill.order_id,
            "exchange_order_id": fill.exchange_order_id,
            "exchange_fill_id": fill.exchange_fill_id,
            "strategy_id": strategy_id,
            "symbol": fill.symbol,
            "side": fill.side.value,
            "tag": fill.tag,
            "fill_strategy_id": strategy_id,
            "fill_symbol": fill.symbol,
            "fill_side": fill.side.value,
            "fill_tag": fill.tag,
            "fill_qty": fill.qty,
            "fill_price": fill.fill_price,
            "fill_commission": fill.commission,
            "fill_timestamp": fill.timestamp.isoformat(),
        }

    def _emit_fill_event(
        self,
        fill_id: str,
        strategy_id: str,
        fill: Fill,
        *,
        closed_trade: Trade | None = None,
    ) -> None:
        payload = {
            **self._fill_snapshot_context(fill_id, strategy_id, fill),
            "fill_event_id": fill_id,
            "event_kind": "fill",
            "trade_id": closed_trade.trade_id if closed_trade is not None else "",
            "timestamp": fill.timestamp.isoformat(),
        }
        self._emit_assistant_payload("fill", payload)

    def _emit_reconciliation_event(
        self,
        *,
        lifecycle_event_kind: str,
        action: str,
        status: str,
        description: str,
        discrepancies: list[dict[str, Any]] | None = None,
        symbol: str = "",
        strategy_id: str = "",
        fill: Fill | None = None,
        fill_id: str = "",
        severity: str = "warning",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        discrepancies = list(discrepancies or [])
        metadata = dict(metadata or {})
        if fill is not None:
            fill_id = fill_id or fill_identity(fill)
            metadata.update(self._fill_snapshot_context(fill_id, strategy_id, fill))
            symbol = symbol or fill.symbol
        event_seed = {
            "kind": lifecycle_event_kind,
            "action": action,
            "status": status,
            "symbol": symbol,
            "strategy_id": strategy_id,
            "fill_id": fill_id,
            "discrepancies": discrepancies,
            "metadata": metadata,
        }
        self._emit_assistant_payload("reconciliation_event", {
            "reconciliation_event_id": stable_hash({"reconciliation_event": event_seed}),
            "lifecycle_event_kind": lifecycle_event_kind,
            "action": action,
            "status": status,
            "description": description,
            "severity": severity,
            "symbol": symbol,
            "strategy_id": strategy_id,
            "fill_id": fill_id,
            "discrepancies": discrepancies,
            "metadata": metadata,
            "unknown_allocation": strategy_id in {"", "unknown", "UNKNOWN"},
            "requires_admin_correction": (
                status not in {"applied", "clear", "resolved"}
                and lifecycle_event_kind in {
                    "admin_correction",
                    "drift_assignment",
                    "fill_processing_failed",
                    "freeze",
                }
            ),
            "correction_applied": (
                lifecycle_event_kind == "admin_correction"
                and status in {"applied", "resolved"}
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    @staticmethod
    def _discrepancy_payload(discrepancy: Any) -> dict[str, Any]:
        return {
            key: str(value)
            for key, value in getattr(discrepancy, "__dict__", {}).items()
        }

    def _correlation_exposure_payload(self) -> dict:
        manager = getattr(self, "_manager", None)
        if manager is None:
            return {}
        exposures: dict[str, dict[str, float]] = {}
        for symbol in getattr(self._config, "symbols", []):
            exposures[symbol] = {
                "long_R": manager.state.symbol_risk_R(symbol, Side.LONG),
                "short_R": manager.state.symbol_risk_R(symbol, Side.SHORT),
            }
        exposures["directional_totals"] = {
            "long_R": manager.state.directional_risk_R(Side.LONG),
            "short_R": manager.state.directional_risk_R(Side.SHORT),
        }
        return exposures

    def _emit_daily_reconciliation(self, date_str: str, snapshot) -> None:
        fills = self._oms_fills_for_date(date_str)
        orders = self._oms_orders_for_date(date_str)
        lifecycle_entries = self._oms_lifecycle_entries_for_date(date_str)
        oms = getattr(self, "_oms", None)
        discrepancies = (
            oms.list_unresolved_discrepancies()
            if oms is not None and hasattr(oms, "list_unresolved_discrepancies")
            else []
        )
        payload = {
            "daily_reconciliation_id": stable_hash({
                "date": date_str,
                "deployment_id": self._lineage.deployment_id,
                "fills": [fill.get("fill_id") for fill in fills],
                "orders": [order.get("client_order_id") for order in orders],
                "lifecycle_entries": [
                    entry.get("position_instance_id")
                    for entry in lifecycle_entries
                ],
                "discrepancies": [item.get("id") for item in discrepancies],
            }),
            "date": date_str,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "daily_closeout",
            "status": "open_discrepancies" if discrepancies else "reconciled",
            "trade_summary": {
                "total_trades": snapshot.total_trades,
                "win_count": snapshot.win_count,
                "loss_count": snapshot.loss_count,
                "gross_pnl": snapshot.gross_pnl,
                "net_pnl": snapshot.net_pnl,
                "fees": self._snapshot_family_value(snapshot, "fees"),
                "funding": self._snapshot_family_value(snapshot, "funding"),
                "family_realized_R": self._snapshot_family_value(snapshot, "realized_R"),
                "net_pnl_by_strategy": {
                    sid: self._strategy_summary_net_pnl(summary)
                    for sid, summary in snapshot.per_strategy_summary.items()
                },
                "realized_R": {
                    sid: self._strategy_summary_realized_r(summary)
                    for sid, summary in snapshot.per_strategy_summary.items()
                },
            },
            "oms_fill_count": len(fills),
            "oms_fills": fills,
            "order_count": len(orders),
            "orders": orders,
            "allocation_version": self._lineage.allocation_version,
            "portfolio_config_version": self._lineage.portfolio_config_version,
            "risk_config_version": self._lineage.risk_config_version,
            "risk_table": self._portfolio_snapshot_payload(
                source="daily_reconciliation",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            "lifecycle_entries": lifecycle_entries,
            "unresolved_discrepancies": discrepancies,
            "session_closeout": {
                "timezone": "UTC",
                "date": date_str,
                "assumption": "24/7 crypto perpetual daily boundary",
            },
        }
        self._emit_assistant_payload("daily_reconciliation", payload)
        self._emit_family_daily_snapshot(
            date_str=date_str,
            source="daily_reconciliation",
            daily_snapshot=snapshot,
            reconciliation=payload,
        )
        self._emit_runtime_snapshots(
            source="daily_reconciliation",
            context={"daily_reconciliation_id": payload["daily_reconciliation_id"], "date": date_str},
            include_allocation=True,
        )

    def _oms_fills_for_date(self, date_str: str) -> list[dict[str, Any]]:
        oms = getattr(self, "_oms", None)
        list_fills = getattr(oms, "list_fills", None)
        if not callable(list_fills):
            return []
        fills = []
        for fill in list_fills():
            if self._value_matches_utc_date(fill.get("timestamp"), date_str):
                fills.append(fill)
        return fills

    def _oms_orders_for_date(self, date_str: str) -> list[dict[str, Any]]:
        oms = getattr(self, "_oms", None)
        list_orders = getattr(oms, "list_orders", None)
        if not callable(list_orders):
            return []
        semantic_fields = ("submitted_at", "decision_time", "timestamp")
        fill_order_ids = self._fill_order_id_set(self._oms_fills_for_date(date_str))
        return [
            order for order in list_orders()
            if (
                self._order_id_set(order) & fill_order_ids
                or self._order_matches_daily_reconciliation_date(
                    order,
                    date_str,
                    semantic_fields=semantic_fields,
                )
            )
        ]

    def _oms_lifecycle_entries_for_date(self, date_str: str) -> list[dict[str, Any]]:
        oms = getattr(self, "_oms", None)
        list_entries = getattr(oms, "list_lifecycle_entries", None)
        if not callable(list_entries):
            return []
        return [
            entry for entry in list_entries()
            if self._lifecycle_entry_is_open_by_utc_date(entry, date_str)
        ]

    def _lifecycle_entry_is_open_by_utc_date(self, entry: dict[str, Any], date_str: str) -> bool:
        metadata = entry.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        for raw in (entry.get("entry_time"), metadata.get("entry_time"), metadata.get("timestamp")):
            if self._value_on_or_before_utc_date(raw, date_str):
                return True
        return False

    def _record_matches_utc_date(
        self,
        record: dict[str, Any],
        date_str: str,
        *,
        fields: tuple[str, ...],
        metadata_fields: tuple[str, ...] = (),
    ) -> bool:
        for field in fields:
            if self._value_matches_utc_date(record.get(field), date_str):
                return True
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return False
        return any(
            self._value_matches_utc_date(metadata.get(field), date_str)
            for field in metadata_fields
        )

    def _order_matches_daily_reconciliation_date(
        self,
        order: dict[str, Any],
        date_str: str,
        *,
        semantic_fields: tuple[str, ...],
    ) -> bool:
        if self._record_matches_utc_date(
            order,
            date_str,
            fields=(),
            metadata_fields=semantic_fields,
        ):
            return True
        metadata = order.get("metadata")
        has_semantic_timestamp = isinstance(metadata, dict) and any(
            metadata.get(field) for field in semantic_fields
        )
        if has_semantic_timestamp:
            return False
        return self._record_matches_utc_date(
            order,
            date_str,
            fields=("updated_at",),
        )

    @classmethod
    def _value_matches_utc_date(cls, raw: Any, date_str: str) -> bool:
        timestamp = cls._parse_optional_datetime(raw)
        if timestamp is not None:
            return timestamp.astimezone(timezone.utc).date().isoformat() == date_str
        return isinstance(raw, str) and raw.startswith(date_str)

    @classmethod
    def _value_on_or_before_utc_date(cls, raw: Any, date_str: str) -> bool:
        timestamp = cls._parse_optional_datetime(raw)
        if timestamp is not None:
            return timestamp.astimezone(timezone.utc).date().isoformat() <= date_str
        return isinstance(raw, str) and len(raw) >= 10 and raw[:10] <= date_str

    @staticmethod
    def _strategy_summary_net_pnl(summary: dict[str, Any]) -> float:
        for key in ("net_pnl", "pnl", "realized_pnl_net"):
            value = summary.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    @staticmethod
    def _strategy_summary_realized_r(summary: dict[str, Any]) -> float:
        for key in ("realized_R", "realized_r", "pnl_R", "r_multiple"):
            value = summary.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def _snapshot_family_value(self, snapshot: Any, key: str) -> float:
        family_summary = getattr(snapshot, "family_summary", {})
        if isinstance(family_summary, dict):
            value = family_summary.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return self._sum_strategy_summary_key(
            getattr(snapshot, "per_strategy_summary", {}),
            key,
        )

    @staticmethod
    def _sum_strategy_summary_key(per_strategy_summary: dict[str, dict], key: str) -> float:
        total = 0.0
        for summary in per_strategy_summary.values():
            if not isinstance(summary, dict):
                continue
            value = summary.get(key)
            if isinstance(value, (int, float)):
                total += float(value)
        return total

    def _emit_family_daily_snapshot(
        self,
        *,
        date_str: str | None = None,
        source: str = "interval",
        daily_snapshot: Any | None = None,
        reconciliation: dict[str, Any] | None = None,
    ) -> None:
        if self._manager is None:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "date": date_str,
            "family_id": self._lineage.family_id,
            "portfolio_id": self._lineage.portfolio_id,
            "strategies": [slot.strategy_id for slot in getattr(self, "_slots", [])],
            "symbols": list(getattr(self._config, "symbols", [])),
            "portfolio_daily_pnl_R": self._manager.state.portfolio_daily_pnl_R,
            "strategy_daily_pnl_R": dict(self._manager.state.daily_pnl_R),
            "open_risk_count": self._manager.state.total_positions(),
            "heat_R": self._manager.state.total_heat_R(),
            "portfolio_heat_summary": self._portfolio_snapshot_payload(
                source=source,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            "reconciliation_status": (reconciliation or {}).get("status"),
        }
        if daily_snapshot is not None:
            family_summary = (
                dict(daily_snapshot.family_summary)
                if isinstance(getattr(daily_snapshot, "family_summary", None), dict)
                else {}
            )
            payload.update({
                "family_level_trades": daily_snapshot.total_trades,
                "net_pnl": daily_snapshot.net_pnl,
                "gross_pnl": daily_snapshot.gross_pnl,
                "fees": self._snapshot_family_value(daily_snapshot, "fees"),
                "funding": self._snapshot_family_value(daily_snapshot, "funding"),
                "realized_R": self._snapshot_family_value(daily_snapshot, "realized_R"),
                "win_count": daily_snapshot.win_count,
                "loss_count": daily_snapshot.loss_count,
                "max_drawdown": daily_snapshot.max_drawdown_pct,
                "exposure": daily_snapshot.exposure_pct,
                "missed_opportunities": daily_snapshot.missed_count,
                "missed_opportunities_would_have_won": daily_snapshot.missed_would_have_won,
                "process_quality_avg": daily_snapshot.avg_process_quality,
                "root_cause_distribution": daily_snapshot.root_cause_distribution,
                "per_strategy_summary": daily_snapshot.per_strategy_summary,
                "family_summary": family_summary,
                "portfolio_summary": daily_snapshot.portfolio_summary,
            })
        self._emit_assistant_payload("family_daily_snapshot", payload)

    def _emit_error_event(
        self,
        component: str,
        error: Exception | str,
        *,
        strategy_id: str = "",
        symbol: str = "",
        severity: str = "low",
        recovery_action: str = "",
        error_type: str | None = None,
        stack_trace: str = "",
        skip_postgres_sink: bool = False,
    ) -> None:
        try:
            if not hasattr(self, "_emitter"):
                return
            message = str(error)
            lineage_context = getattr(self, "_lineage", None)
            if lineage_context is None:
                config = getattr(self, "_config", None)
                lineage_context = LineageContext(
                    bot_id=getattr(config, "bot_id", ""),
                    family_id=getattr(config, "family_id", "crypto_perps"),
                    portfolio_id=getattr(config, "portfolio_id", "default"),
                    account_alias=getattr(config, "account_alias", "default"),
                    venue_environment=(
                        "testnet"
                        if bool(getattr(config, "is_testnet", True))
                        else "mainnet"
                    ),
                    symbol_universe=list(getattr(config, "symbols", [])),
                )
            lineage = (
                lineage_context.for_strategy(strategy_id)
                if strategy_id
                else lineage_context.for_portfolio()
            )
            metadata = EventMetadata.create(
                bot_id=getattr(getattr(self, "_config", None), "bot_id", ""),
                strategy_id=strategy_id or "system",
                exchange_ts=datetime.now(timezone.utc),
                event_type="error",
                payload_key=stable_hash({"component": component, "message": message}),
                lineage=lineage,
                family_id=lineage_context.family_id,
                portfolio_id=lineage_context.portfolio_id,
                account_alias=lineage_context.account_alias,
                config_version=lineage_context.config_version,
                deployment_id=lineage_context.deployment_id,
                code_sha=lineage_context.code_sha,
            )
            event = ErrorEvent(
                metadata=metadata,
                lineage=lineage,
                error_type=error_type or (
                    type(error).__name__ if isinstance(error, Exception) else "runtime_error"
                ),
                message=message,
                stack_trace=stack_trace,
                severity=severity,
                component=component,
                symbol=symbol,
                recovery_action=recovery_action,
            )
            if skip_postgres_sink:
                self._emit_error_without_postgres_sink(event)
            else:
                self._emitter.emit_error(event)
        except Exception:
            log.exception("engine.error_event_emit_failed", component=component)

    def _emit_error_without_postgres_sink(self, event: ErrorEvent) -> None:
        emitter = getattr(self, "_emitter", None)
        pg_sink = getattr(self, "_pg_sink", None)
        sinks = list(getattr(emitter, "_sinks", []))
        failures = getattr(emitter, "_sink_failures", None)
        for sink in sinks:
            if sink is pg_sink:
                continue
            sink_name = type(sink).__name__
            try:
                sink.write_error(event)
            except Exception:
                if isinstance(failures, dict):
                    failures[sink_name] = failures.get(sink_name, 0) + 1
                log.exception("emitter.error_failed", sink=sink_name)

    def _emit_sidecar_error_event(self, payload: dict[str, Any]) -> None:
        self._emit_error_event(
            str(payload.get("component") or "sidecar"),
            str(payload.get("message") or payload.get("error") or ""),
            severity=str(payload.get("severity") or "medium"),
            recovery_action=str(payload.get("recovery_action") or "retry_next_poll"),
            error_type=str(payload.get("error_type") or "runtime_error"),
        )

    def _emit_postgres_error_event(self, payload: dict[str, Any]) -> None:
        self._emit_error_event(
            str(payload.get("component") or "postgres_sink"),
            str(payload.get("message") or payload.get("error") or ""),
            severity=str(payload.get("severity") or "medium"),
            recovery_action=str(payload.get("recovery_action") or "continue_without_postgres"),
            error_type=str(payload.get("error_type") or "runtime_error"),
            skip_postgres_sink=True,
        )

    def _dispatch_bar(self, bar: Bar | MarketEvent) -> None:
        """Route a bar to subscribing strategies."""
        visible_bar = bar.to_bar() if isinstance(bar, MarketEvent) else bar
        self._health.on_bar_received(visible_bar.symbol, visible_bar.timeframe.value)
        for slot in self._slots:
            if (
                visible_bar.timeframe in slot.subscribed_tfs
                and visible_bar.symbol in slot.strategy.symbols
            ):
                if visible_bar.timeframe == slot.primary_tf:
                    expire_fn = getattr(slot.ctx.broker, "expire_ttl_orders_for_bar", None)
                    if callable(expire_fn):
                        expire_fn(visible_bar)
                slot.runtime.process_bar(bar, process_broker=False, advance_clock=False)
        self._drain_and_backfill_missed()

    def _drain_and_backfill_missed(self) -> None:
        pending = getattr(self, "_pending_missed", {})
        for slot in self._slots:
            collector = getattr(slot.strategy, "_collector", None)
            if collector is None:
                continue
            for event in collector.flush_missed():
                pending[event.logical_event_id or event.metadata.event_id] = event

        if not pending:
            self._pending_missed = pending
            return

        bars_by_symbol = self._bars_by_symbol_for_backfill()
        if not bars_by_symbol:
            self._pending_missed = pending
            return

        for event_id, event in list(pending.items()):
            before = (
                event.outcome_1h,
                event.outcome_4h,
                event.outcome_24h,
                event.backfill_status,
            )
            MissedOpportunityBackfiller.backfill_from_bars([event], bars_by_symbol)
            after = (
                event.outcome_1h,
                event.outcome_4h,
                event.outcome_24h,
                event.backfill_status,
            )
            if after != before:
                event.bump_revision()
                self._emitter.emit_missed(event)
            if event.backfill_status == "complete":
                pending.pop(event_id, None)

        self._pending_missed = pending

    def _bars_by_symbol_for_backfill(self) -> dict[str, list[Bar]]:
        bars_by_symbol: dict[str, tuple[int, list[Bar]]] = {}
        for slot in self._slots:
            for tf in slot.subscribed_tfs:
                for sym in slot.strategy.symbols:
                    bars = slot.bars.get(sym, tf)
                    if not bars:
                        continue
                    current = bars_by_symbol.get(sym)
                    if current is None or tf.minutes < current[0]:
                        bars_by_symbol[sym] = (tf.minutes, bars)
        return {sym: bars for sym, (_, bars) in bars_by_symbol.items()}

    def _derive_bars_held(
        self,
        strategy_id: str,
        entry_time: datetime,
        exit_time: datetime,
    ) -> int:
        slot = self._find_slot(strategy_id)
        if slot is None:
            return 0
        primary_tf = slot.primary_tf if isinstance(slot.primary_tf, TimeFrame) else TimeFrame.M15
        interval_sec = _TF_INTERVALS.get(primary_tf.value)
        if not interval_sec or exit_time <= entry_time:
            return 0
        elapsed_sec = (exit_time - entry_time).total_seconds()
        return max(1, int((elapsed_sec + interval_sec - 1) // interval_sec))

    def _emit_lifecycle_trade(
        self,
        strategy_id: str,
        trade: Trade,
        *,
        fill_id: str = "",
    ) -> None:
        """Emit a ledger-built live trade through the same close-event path."""
        slot = self._find_slot(strategy_id)
        if slot is None:
            return

        trade.bars_held = self._derive_bars_held(
            strategy_id,
            trade.entry_time,
            trade.exit_time,
        )
        self._attach_trade_completion_context(strategy_id, trade, fill_id=fill_id)
        slot.ctx.events.emit(PositionClosedEvent(
            timestamp=trade.exit_time,
            trade=trade,
        ))
        pnl_R = trade.r_multiple if trade.r_multiple is not None else 0.0
        if self._coordinator is not None:
            self._record_coordinator_trade_closed(strategy_id, trade.symbol, pnl_R, trade)
        self._cancel_strategy_open_orders_for_closed_symbol(slot, trade.symbol)
        self._tracked_positions.pop(trade.symbol, None)
        self._emit_runtime_snapshots(source="trade_closed")

    def _record_lifecycle_trade_position(self, strategy_id: str, trade: Trade) -> None:
        oms = getattr(self, "_oms", None)
        if oms is not None:
            position_instance_id = (
                self._resolve_trade_position_instance_id(strategy_id, trade)
                or entry_position_instance_id(
                    strategy_id,
                    trade.symbol,
                    trade.direction,
                    trade.entry_time,
                )
            )
            oms.upsert_position(
                position_instance_id=position_instance_id,
                strategy_id=strategy_id,
                symbol=trade.symbol,
                direction=trade.direction.value,
                qty=0.0,
                avg_entry=trade.entry_price,
                status="CLOSED",
                metadata={"trade_id": trade.trade_id},
            )

    def _detect_position_closures(self, recent_fills: list[Fill]) -> None:
        """Check tracked positions against exchange; emit PositionClosedEvent for closures."""
        current_positions = {p.symbol: p for p in self._broker.get_positions()}

        for sym, tracked in list(self._tracked_positions.items()):
            if sym in current_positions and current_positions[sym].qty != 0:
                continue  # Still open

            # Position closed — find the exit fill
            exit_fill = None
            for fill in reversed(recent_fills):
                if fill.symbol == sym and fill.tag != "entry":
                    exit_fill = fill
                    break
            if exit_fill is None:
                continue

            # Build Trade from tracked data + strategy _position_meta
            slot = self._find_slot(tracked["strategy_id"])
            if slot is None:
                continue

            meta = getattr(slot.strategy, "_position_meta", {}).get(sym)
            entry_price = meta.entry_price if meta and hasattr(meta, "entry_price") else tracked["entry_price"]
            direction = tracked["direction"]
            qty = tracked["qty"]
            stop_distance = meta.stop_distance if meta and hasattr(meta, "stop_distance") else 0.0

            if direction == Side.LONG:
                pnl = (exit_fill.fill_price - entry_price) * qty
            else:
                pnl = (entry_price - exit_fill.fill_price) * qty

            commission = tracked.get("entry_commission", 0.0) + exit_fill.commission
            bars_held = self._derive_bars_held(
                tracked["strategy_id"],
                tracked["entry_time"],
                exit_fill.timestamp,
            )

            trade = Trade(
                trade_id=f"live_{sym}_{exit_fill.timestamp.strftime('%Y%m%d_%H%M%S')}",
                symbol=sym,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_fill.fill_price,
                qty=qty,
                entry_time=tracked["entry_time"],
                exit_time=exit_fill.timestamp,
                pnl=pnl,
                r_multiple=None,
                commission=commission,
                bars_held=bars_held,
                setup_grade=None,
                exit_reason=exit_fill.tag or "exchange_fill",
                confluences_used=None,
                confirmation_type=None,
                entry_method=None,
                funding_paid=0.0,
                mae_r=None,
                mfe_r=None,
            )
            setattr(trade, "instrumentation_context", {
                "position_instance_id": str(tracked.get("position_instance_id") or ""),
                "entry_fill_ids": list(tracked.get("entry_fill_ids", [])),
                "exit_fill_ids": [fill_identity(exit_fill)],
                "entry_order_ids": list(tracked.get("entry_order_ids", [])),
                "exit_order_ids": self._fill_order_ids(exit_fill),
            })
            self._attach_trade_completion_context(
                tracked["strategy_id"],
                trade,
                fill_id=fill_identity(exit_fill),
            )

            # Emit — synchronously fires strategy's _on_position_closed
            slot.ctx.events.emit(PositionClosedEvent(
                timestamp=exit_fill.timestamp, trade=trade,
            ))

            # Fire coordinator for portfolio heat release (AFTER strategy enrichment)
            pnl_R = trade.r_multiple if trade.r_multiple is not None else 0.0
            self._record_coordinator_trade_closed(tracked["strategy_id"], sym, pnl_R, trade)
            self._cancel_strategy_open_orders_for_closed_symbol(slot, sym)

            del self._tracked_positions[sym]
            log.info(
                "engine.position_closed",
                symbol=sym,
                strategy=tracked["strategy_id"],
                pnl=f"{trade.pnl:.2f}",
                r=f"{trade.r_multiple:.2f}" if trade.r_multiple else "N/A",
            )

    def _record_coordinator_trade_closed(
        self,
        strategy_id: str,
        symbol: str,
        pnl_R: float,
        trade: Trade,
    ) -> None:
        if isinstance(self._coordinator, StrategyCoordinator):
            self._coordinator.on_trade_closed(strategy_id, symbol, pnl_R, trade=trade)
        elif self._coordinator is not None:
            self._coordinator.on_trade_closed(strategy_id, symbol, pnl_R)

    def _cancel_strategy_open_orders_for_closed_symbol(self, slot: _StrategySlot, symbol: str) -> int:
        """Cancel remaining strategy-owned orders after a terminal close."""
        broker = getattr(slot.ctx, "broker", None)
        if broker is None:
            return 0
        cancelled = 0
        failures: list[Discrepancy] = []
        try:
            open_orders = list(broker.get_open_orders(symbol))
        except Exception:
            log.exception(
                "engine.closed_symbol_open_order_query_failed",
                strategy=slot.strategy_id,
                symbol=symbol,
            )
            return 0

        for order in open_orders:
            try:
                if broker.cancel_order(order.order_id):
                    cancelled += 1
                    continue
            except Exception:
                log.exception(
                    "engine.closed_symbol_order_cancel_failed",
                    strategy=slot.strategy_id,
                    symbol=symbol,
                    order_id=order.order_id,
                )
            failures.append(Discrepancy(
                symbol=symbol,
                kind="stale_exit_order",
                expected="no strategy-owned open orders after terminal close",
                actual=f"{order.order_id} {order.tag or order.order_type.value}",
            ))

        if failures:
            self._freeze_entries_for_reconciliation(
                discrepancies=failures,
                description=(
                    "A terminal close left strategy-owned open orders that could not be "
                    "cancelled; new entries are blocked."
                ),
                reconciliation_status="stale_exit_cleanup_failed",
            )
        if cancelled:
            log.info(
                "engine.closed_symbol_orders_cancelled",
                strategy=slot.strategy_id,
                symbol=symbol,
                count=cancelled,
            )
        return cancelled

    async def _funnel_report_loop(self) -> None:
        """Periodic pipeline funnel snapshots."""
        while self._running:
            await asyncio.sleep(self._funnel_report_interval())
            try:
                for slot in self._slots:
                    collector = getattr(slot.strategy, "_collector", None)
                    if collector is None:
                        continue
                    funnel = collector.pipeline.snapshot_and_reset()
                    assessment = PipelineTracker.assess(funnel)
                    funnel_dict = funnel.to_dict()

                    # Cache for health report (avoids double-reset)
                    self._last_funnels[slot.strategy_id] = funnel_dict

                    snapshot = PipelineFunnelSnapshot(
                        strategy_id=slot.strategy_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        period_start=funnel.period_start.isoformat(),
                        period_end=funnel.period_end.isoformat(),
                        funnel=funnel_dict,
                        assessment=assessment,
                        metadata=EventMetadata.create(
                            bot_id=getattr(self._config, "bot_id", ""),
                            strategy_id=slot.strategy_id,
                            exchange_ts=datetime.now(timezone.utc),
                            event_type="pipeline_funnel",
                            payload_key=f"{slot.strategy_id}:{funnel.period_end.isoformat()}",
                            lineage=self._lineage.for_strategy(slot.strategy_id),
                            family_id=self._lineage.family_id,
                            portfolio_id=self._lineage.portfolio_id,
                            account_alias=self._lineage.account_alias,
                            config_version=self._lineage.config_version,
                            deployment_id=self._lineage.deployment_id,
                            code_sha=self._lineage.code_sha,
                        ),
                        lineage=self._lineage.for_strategy(slot.strategy_id),
                    )
                    self._emitter.emit_funnel(snapshot)

                    if assessment in ("pipeline_broken", "stalled"):
                        log.error(
                            "engine.funnel_alert",
                            strategy=slot.strategy_id,
                            assessment=assessment,
                        )
            except Exception as exc:
                log.exception("engine.funnel_report_error")
                self._health.on_error("funnel_report")
                self._emit_error_event("funnel_report", exc, severity="low", recovery_action="continue")

    async def _health_report_loop(self) -> None:
        """Periodic health report."""
        while self._running:
            await asyncio.sleep(self._health_report_interval())
            try:
                status = self._health.get_status()

                # Read last emitted funnel data (no reset — funnel_report_loop owns the reset)
                funnels: dict[str, dict] = {}
                for slot in self._slots:
                    collector = getattr(slot.strategy, "_collector", None)
                    if collector is not None:
                        funnels[slot.strategy_id] = self._last_funnels.get(
                            slot.strategy_id, {},
                        )

                # Collect positions
                positions = []
                if self._broker:
                    for p in self._broker.get_positions():
                        positions.append({
                            "symbol": p.symbol,
                            "direction": p.direction.value if p.direction else "unknown",
                            "qty": p.qty,
                        })

                # Portfolio state
                portfolio_state = {}
                if self._manager:
                    portfolio_state = {
                        "heat_R": sum(r.risk_R for r in self._manager.state.open_risks),
                        "heat_cap_R": self._manager.config.heat_cap_R,
                        "daily_pnl_R": self._manager.state.portfolio_daily_pnl_R,
                        "open_risk_count": len(self._manager.state.open_risks),
                    }

                stale_feeds = self._health.get_stale_feeds(_TF_INTERVALS)

                report = self._report_builder.build(
                    uptime_sec=status["uptime_sec"],
                    health_status=status,
                    stale_feeds=stale_feeds,
                    funnels=funnels,
                    positions=positions,
                    portfolio_state=portfolio_state,
                    tf_last_bar=self._health.get_tf_last_bar(),
                    now_mono=time.monotonic(),
                )

                report_payload = report.to_dict()
                sidecar_status = self._relay_health_status()
                report_payload["relay"] = sidecar_status
                report_payload["sidecar"] = sidecar_status
                report_payload["oms_summary"] = {
                    "open_orders": len(self._oms.list_orders()),
                    "last_market_event_at": self._last_assistant_event_at.get("market_snapshot"),
                    "last_order_event_at": self._last_assistant_event_at.get("order"),
                    "last_trade_event_at": self._last_assistant_event_at.get("trade"),
                }
                report_payload["emitter_sink_failures"] = self._emitter.sink_failures
                report_payload["postgres_sink"] = self._postgres_sink_health_payload()

                self._emitter.emit_health_report(HealthReportSnapshot(
                    timestamp=report.timestamp,
                    report=report_payload,
                    metadata=EventMetadata.create(
                        bot_id=getattr(self._config, "bot_id", ""),
                        strategy_id="system",
                        exchange_ts=datetime.now(timezone.utc),
                        event_type="heartbeat",
                        payload_key=report.timestamp,
                        lineage=self._lineage.for_portfolio(),
                        family_id=self._lineage.family_id,
                        portfolio_id=self._lineage.portfolio_id,
                        account_alias=self._lineage.account_alias,
                        config_version=self._lineage.config_version,
                        deployment_id=self._lineage.deployment_id,
                        code_sha=self._lineage.code_sha,
                    ),
                    lineage=self._lineage.for_portfolio(),
                ))

                if report.assessment == "critical":
                    log.error("engine.health_critical", alerts=len(report.alerts))

            except Exception as exc:
                log.exception("engine.health_report_error")
                self._health.on_error("health_report")
                self._emit_error_event("health_report", exc, severity="low", recovery_action="continue")

    def _relay_health_status(self) -> dict:
        """Return relay/sidecar state for health report enrichment."""
        if self._sidecar is None:
            return {
                "enabled": False,
                "sidecar_running": False,
                "event_files": [],
                "canonical_event_files": [],
                "event_file_map": {},
            }

        try:
            status = self._sidecar.status()
        except Exception as exc:
            return {
                "enabled": True,
                "sidecar_running": False,
                "event_files": [],
                "canonical_event_files": [],
                "event_file_map": {},
                "status_error": str(exc),
            }

        return {
            "enabled": True,
            "sidecar_running": status.get("running", False),
            "event_files": status.get("event_files", []),
            "canonical_event_files": status.get("canonical_event_files", []),
            "event_file_map": status.get("event_file_map", {}),
            "watermarks": status.get("watermarks", {}),
            "watermark_file": status.get("watermark_file"),
            "last_successful_send_at": status.get("last_successful_send_at"),
            "consecutive_send_failures": status.get("consecutive_send_failures", 0),
            "last_error": status.get("last_error"),
        }

    def _postgres_sink_health_payload(self) -> dict:
        sink = getattr(self, "_pg_sink", None)
        if sink is None:
            return {"enabled": False}
        metrics = getattr(sink, "metrics", None)
        if callable(metrics):
            payload = metrics()
        else:
            payload = {"enabled": True, "worker_alive": True, "mode": "sync"}
        queue_depth = float(payload.get("queue_depth", 0.0) or 0.0)
        queue_capacity = float(payload.get("queue_capacity", 0.0) or 0.0)
        if queue_capacity > 0 and queue_depth / queue_capacity >= 0.8:
            self._emit_postgres_error_event({
                "component": "postgres_sink",
                "event_type": "postgres_sink",
                "error_type": "QueueHighWatermark",
                "message": "postgres sink queue is above 80 percent capacity",
                "severity": "medium",
                "recovery_action": "monitor_and_backfill_from_jsonl_if_needed",
            })
        return payload

    def _funnel_report_interval(self) -> float:
        return max(60.0, self._config.funnel_report_interval_sec)

    def _health_report_interval(self) -> float:
        return max(60.0, self._config.health_report_interval_sec)

    def _fill_query_overlap_sec(self) -> float:
        return max(0.0, float(getattr(self._config, "fill_query_overlap_sec", 300.0)))

    def _load_fill_watermark(self) -> datetime | None:
        raw = self._oms.get_watermark("fills_since")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            log.warning("engine.fill_watermark_invalid", value=raw)
            return None

    def _record_canonical_event(self, event: CanonicalRuntimeEvent) -> None:
        payload = {
            "timestamp": event.timestamp.isoformat(),
            "stream": event.stream,
            "payload": event.payload,
        }
        path = self._config.state_dir / "parity_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        append_fn = getattr(self._oms, "append_event", None)
        if append_fn is not None:
            append_fn(event.stream, event.timestamp, event.payload)
        self._emit_assistant_from_canonical_event(event)

    def _emit_assistant_from_canonical_event(self, event: CanonicalRuntimeEvent) -> None:
        """Promote parity streams to assistant telemetry while parity stays deterministic."""
        try:
            payload = dict(event.payload or {})
            payload["timestamp"] = event.timestamp.isoformat()
            if event.stream == "order_intent":
                payload["event_kind"] = "intent"
                payload["order_event_id"] = stable_hash({
                    "stream": event.stream,
                    "intent_id": payload.get("intent_id"),
                    "timestamp": event.timestamp.isoformat(),
                })
                self._emit_assistant_payload("order", payload)
            elif event.stream == "execution":
                payload["event_kind"] = payload.get("kind", "execution")
                payload["order_event_id"] = stable_hash({
                    "stream": event.stream,
                    "report_id": payload.get("report_id"),
                    "timestamp": event.timestamp.isoformat(),
                })
                self._emit_assistant_payload("order", payload)
            elif event.stream == "decision":
                metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                payload["bar_id"] = payload.get("bar_id") or metadata.get("bar_id", "")
                payload["decision_event_id"] = str(payload.get("decision_id") or stable_hash({
                    "action": payload.get("action"),
                    "bar_id": payload.get("bar_id"),
                }))
                self._emit_assistant_payload("decision_event", payload)
            elif event.stream == "market":
                payload["bar_id"] = payload.get("bar_id") or stable_hash({
                    "symbol": payload.get("symbol"),
                    "timeframe": payload.get("timeframe"),
                    "available_at": payload.get("available_at"),
                })
                payload["market_snapshot_id"] = str(payload.get("bar_id") or stable_hash({
                    "symbol": payload.get("symbol"),
                    "timeframe": payload.get("timeframe"),
                    "available_at": payload.get("available_at"),
                }))
                self._emit_assistant_payload("market_snapshot", payload)
        except Exception:
            log.exception("engine.assistant_canonical_conversion_failed", stream=event.stream)

    def _restore_strategy_snapshots(self) -> None:
        for slot in self._slots:
            snapshot = self._oms.get_strategy_snapshot(slot.strategy_id)
            restore_fn = getattr(slot.strategy, "restore_state", None)
            if snapshot is not None and restore_fn is not None:
                restore_fn(snapshot)

    def _persist_strategy_snapshots(self) -> None:
        for slot in self._slots:
            snapshot_fn = getattr(slot.strategy, "snapshot_state", None)
            if snapshot_fn is not None:
                self._oms.upsert_strategy_snapshot(slot.strategy_id, snapshot_fn())

    def _restore_lifecycle(self) -> None:
        restore_fn = getattr(self._lifecycle, "restore", None)
        if restore_fn is not None:
            restore_fn(self._oms.list_lifecycle_entries())

    def _persist_lifecycle(self) -> None:
        oms = getattr(self, "_oms", None)
        if oms is None:
            return
        entries = self._lifecycle_snapshot()
        replace_fn = getattr(oms, "replace_lifecycle_entries", None)
        if replace_fn is not None:
            replace_fn(entries)
        else:
            for entry in entries:
                oms.upsert_lifecycle_entry(entry)

    def _lifecycle_snapshot(self) -> list[Any]:
        lifecycle = getattr(self, "_lifecycle", None)
        if lifecycle is None:
            return []
        snapshot_fn = getattr(lifecycle, "snapshot", None)
        if snapshot_fn is None:
            return []
        return list(snapshot_fn())

    def _rehydrate_oms_orders(self) -> None:
        """Restore in-memory order ownership from durable OMS records."""
        broker_orders = getattr(self._broker, "_orders", None)
        local_to_oid = getattr(self._broker, "_local_to_oid", None)
        oid_map = getattr(self._broker, "_oid_map", None)
        for record in self._oms.list_orders():
            strategy_id = record.get("strategy_id") or ""
            if not strategy_id:
                continue
            client_id = record.get("client_order_id") or ""
            exchange_id = record.get("exchange_order_id") or ""

            if client_id and exchange_id:
                if isinstance(local_to_oid, dict):
                    local_to_oid[str(client_id)] = str(exchange_id)
                if isinstance(oid_map, dict):
                    oid_map[str(exchange_id)] = str(client_id)

            if isinstance(broker_orders, dict) and client_id and client_id not in broker_orders:
                raw_metadata = dict(record.get("metadata") or {})
                nested_metadata = raw_metadata.get("metadata")
                metadata = {
                    **raw_metadata,
                    **(nested_metadata if isinstance(nested_metadata, dict) else {}),
                }
                metadata.setdefault("strategy_id", strategy_id)
                metadata.setdefault("client_order_id", client_id)
                ttl_bars = metadata.get("ttl_bars")
                if ttl_bars is not None:
                    try:
                        ttl_bars = int(ttl_bars)
                    except (TypeError, ValueError):
                        ttl_bars = None
                ttl_bars_alive = metadata.get("ttl_bars_alive", 0)
                try:
                    ttl_bars_alive = int(ttl_bars_alive)
                except (TypeError, ValueError):
                    ttl_bars_alive = 0
                broker_orders[str(client_id)] = Order(
                    order_id=str(client_id),
                    symbol=str(record.get("symbol") or ""),
                    side=_enum_or_default(Side, record.get("side"), Side.LONG),
                    order_type=_enum_or_default(
                        OrderType,
                        record.get("order_type") or metadata.get("order_type"),
                        OrderType.LIMIT,
                    ),
                    qty=0.0,
                    status=_enum_or_default(OrderStatus, record.get("status"), OrderStatus.WORKING),
                    tag=str(record.get("role") or metadata.get("tag") or ""),
                    ttl_bars=ttl_bars,
                    metadata=metadata,
                    _bars_alive=ttl_bars_alive,
                )

            if self._coordinator is not None and client_id:
                self._coordinator.register_order(client_id, strategy_id)
            if self._coordinator is not None and exchange_id:
                self._coordinator.register_order(exchange_id, strategy_id)

    def _sync_open_orders_to_oms(self) -> list[Order]:
        """Persist currently visible exchange orders into the OMS store."""
        if self._broker is None:
            return []
        local_to_oid = getattr(self._broker, "_local_to_oid", {})
        open_orders = self._broker.get_open_orders()
        for order in open_orders:
            exchange_oid = str(local_to_oid.get(order.order_id, ""))
            existing = self._oms.get_order(order.order_id)
            if existing is None and exchange_oid:
                existing = self._oms.get_order(exchange_oid)
            if existing is not None:
                exchange_oid = exchange_oid or str(existing.get("exchange_order_id") or "")
                existing_metadata = dict(existing.get("metadata") or {})
                metadata = {**existing_metadata, **dict(order.metadata)}
                for key in ("strategy_id", "position_instance_id", "oca_group"):
                    value = existing.get(key)
                    if value and not metadata.get(key):
                        metadata[key] = value
                if metadata.get("oca_group") and not order.oca_group:
                    order.oca_group = str(metadata["oca_group"])
                if existing.get("reduce_only") and not metadata.get("reduce_only"):
                    metadata["reduce_only"] = True
                order.metadata = metadata

            strategy_id = ""
            if self._coordinator is not None:
                strategy_id = self._coordinator.get_strategy_for_order(order.order_id) or ""
            owner = ""
            owner_fn = getattr(self._broker, "get_order_owner", None)
            if callable(owner_fn):
                owner = str(owner_fn(order.order_id) or "")
            strategy_id = (
                strategy_id
                or str(order.metadata.get("strategy_id") or "")
                or owner
                or (str(existing.get("strategy_id") or "") if existing is not None else "")
                or "unknown"
            )
            client_order_id = (
                str(existing.get("client_order_id") or "")
                if existing is not None
                else ""
            ) or order.order_id
            order.metadata.setdefault("client_order_id", client_order_id)
            order.metadata.setdefault("strategy_id", strategy_id)
            self._oms.upsert_order(
                client_order_id=client_order_id,
                exchange_order_id=exchange_oid,
                strategy_id=strategy_id,
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                status=order.status.value,
                role=order.tag,
                position_instance_id=str(order.metadata.get("position_instance_id") or ""),
                reduce_only=bool(order.metadata.get("reduce_only", False)),
                oca_group=order.oca_group or order.metadata.get("oca_group"),
                metadata=dict(order.metadata),
            )
            if self._coordinator is not None and strategy_id != "unknown":
                self._coordinator.register_order(order.order_id, strategy_id, order)
                if client_order_id != order.order_id:
                    self._coordinator.register_order(client_order_id, strategy_id, order)
                if exchange_oid:
                    self._coordinator.register_order(exchange_oid, strategy_id, order)
        return open_orders

    def _expected_positions_from_portfolio_state(self) -> dict[str, Position]:
        """Aggregate restored portfolio risks into exchange-net expected positions."""
        expected: dict[str, Position] = {}
        manager = getattr(self, "_manager", None)
        if manager is None:
            return expected

        for risk in manager.state.open_risks:
            existing = expected.get(risk.symbol)
            risk_qty_known = risk.filled_qty > 0
            if existing is None:
                expected[risk.symbol] = Position(
                    symbol=risk.symbol,
                    direction=risk.direction,
                    qty=risk.filled_qty if risk_qty_known else 0.0,
                    avg_entry=0.0,
                    metadata={"qty_known": risk_qty_known},
                )
                continue

            existing.metadata["qty_known"] = bool(existing.metadata.get("qty_known", True)) and risk_qty_known
            if existing.metadata["qty_known"]:
                existing.qty += risk.filled_qty
            else:
                existing.qty = 0.0
            if existing.direction != risk.direction:
                existing.metadata["direction_conflict"] = True
        return expected

    def _cleanup_flat_symbol_exit_orders(
        self,
        open_orders: list[Order],
        actual_positions: list[Position],
    ) -> list[Discrepancy]:
        """Cancel exit-only open orders for symbols currently flat on exchange."""
        if self._broker is None:
            return []
        live_symbols = {
            pos.symbol
            for pos in actual_positions
            if abs(pos.qty) > 1e-8
        }
        discrepancies: list[Discrepancy] = []
        remaining_orders: list[Order] = []
        for order in open_orders:
            if order.symbol in live_symbols:
                remaining_orders.append(order)
                continue
            if not (is_exit_order(order) or bool(order.metadata.get("reduce_only", False))):
                remaining_orders.append(order)
                continue
            try:
                cancelled = self._broker.cancel_order(order.order_id)
            except Exception:
                log.exception(
                    "engine.startup_flat_exit_cancel_failed",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    tag=order.tag,
                )
                cancelled = False
            if cancelled:
                self._mark_oms_order_cancelled(order)
                log.warning(
                    "engine.startup_cancelled_flat_exit_order",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    tag=order.tag,
                )
                continue
            remaining_orders.append(order)
            discrepancies.append(Discrepancy(
                symbol=order.symbol,
                kind="stale_exit_order",
                expected="no open exit orders while flat",
                actual=f"{order.order_id} {order.tag or order.order_type.value}",
            ))
        open_orders[:] = remaining_orders
        return discrepancies

    def _mark_oms_order_cancelled(self, order: Order) -> None:
        update_fn = getattr(getattr(self, "_oms", None), "update_order_metadata", None)
        if not callable(update_fn):
            return
        updates = {
            "startup_flat_exit_cancelled": True,
            "startup_flat_exit_cancelled_at": datetime.now(timezone.utc).isoformat(),
        }
        if order.metadata.get("cancel_reason"):
            updates["cancel_reason"] = order.metadata.get("cancel_reason")
        if update_fn(order.order_id, metadata_updates=updates, status=OrderStatus.CANCELLED.value):
            return
        exchange_order_id = str(order.metadata.get("exchange_order_id") or "")
        if exchange_order_id:
            update_fn(exchange_order_id, metadata_updates=updates, status=OrderStatus.CANCELLED.value)

    def _reconcile_open_oca_groups(self, open_orders: list[Order]) -> list[Discrepancy]:
        """Recover broker-managed OCA groups from OMS plus exchange open orders."""
        oms = getattr(self, "_oms", None)
        list_orders = getattr(oms, "list_orders", None)
        if not callable(list_orders):
            return []

        try:
            oms_orders = list_orders()
        except Exception:
            log.exception("engine.oca_reconcile_oms_query_failed")
            return []

        terminal_by_group: dict[str, list[dict[str, Any]]] = {}
        for record in oms_orders:
            group = str(record.get("oca_group") or record.get("metadata", {}).get("oca_group") or "")
            if not group:
                continue
            status = str(record.get("status") or "")
            if status in {OrderStatus.FILLED.value, OrderStatus.CANCELLED.value, OrderStatus.REJECTED.value, OrderStatus.EXPIRED.value}:
                terminal_by_group.setdefault(group, []).append(record)

        discrepancies: list[Discrepancy] = []
        remaining: list[Order] = []
        for order in open_orders:
            group = str(order.oca_group or order.metadata.get("oca_group") or "")
            if not group:
                remaining.append(order)
                continue

            strategy_id = str(order.metadata.get("strategy_id") or "")
            oca_root = self._oca_root_for_open_order(order, group, strategy_id)
            if not strategy_id or not oca_root:
                discrepancies.append(Discrepancy(
                    symbol=order.symbol,
                    kind="oca_group_inconsistent",
                    expected="open OCA member mapped to strategy and stable position/entry root",
                    actual=f"{order.order_id} group={group}",
                ))
                self._append_oca_event("oca_group_inconsistent", order, group, {
                    "reason": "missing_strategy_or_valid_stable_root",
                })
                remaining.append(order)
                continue

            filled_members = [
                record for record in terminal_by_group.get(group, [])
                if str(record.get("status") or "") == OrderStatus.FILLED.value
            ]
            if not filled_members:
                self._append_oca_event("oca_member_accepted", order, group)
                remaining.append(order)
                continue

            should_cancel, cancel_block_reason = self._startup_oca_sibling_cancel_decision(order)
            if not should_cancel:
                if cancel_block_reason == "residual_position_open":
                    self._append_oca_event("oca_member_accepted", order, group, {
                        "reason": "filled_member_but_residual_position_open",
                    })
                else:
                    discrepancies.append(Discrepancy(
                        symbol=order.symbol,
                        kind="oca_group_inconsistent",
                        expected="broker position state known before cancelling OCA sibling",
                        actual=f"{order.order_id} group={group} reason={cancel_block_reason}",
                    ))
                    self._append_oca_event("oca_group_inconsistent", order, group, {
                        "reason": cancel_block_reason,
                    })
                remaining.append(order)
                continue

            cancelled = False
            try:
                cancelled = bool(self._broker.cancel_order(order.order_id)) if self._broker is not None else False
            except Exception:
                log.exception(
                    "engine.oca_sibling_cancel_failed",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    oca_group=group,
                )
            if cancelled:
                order.metadata["cancel_reason"] = "oca_sibling_filled"
                self._mark_oms_order_cancelled(order)
                self._append_oca_event("oca_member_cancelled", order, group, {
                    "cancel_reason": "oca_sibling_filled",
                })
                continue

            discrepancies.append(Discrepancy(
                symbol=order.symbol,
                kind="oca_group_inconsistent",
                expected="open sibling cancelled after OCA member filled",
                actual=f"{order.order_id} group={group}",
            ))
            remaining.append(order)

        open_orders[:] = remaining
        completed_groups = {
            str(record.get("oca_group") or record.get("metadata", {}).get("oca_group") or "")
            for members in terminal_by_group.values()
            for record in members
            if str(record.get("status") or "") == OrderStatus.CANCELLED.value
        }
        for group in completed_groups:
            if group:
                self._append_oca_event("oca_group_completed", None, group)
        return discrepancies

    def _startup_oca_sibling_cancel_decision(self, order: Order) -> tuple[bool, str]:
        if not self._uses_terminal_close_oca_policy(order):
            return True, "cancel_siblings_on_any_fill"
        flat = self._startup_symbol_position_flat(order.symbol)
        if flat is True:
            return True, "terminal_close"
        if flat is False:
            return False, "residual_position_open"
        return False, "unknown_position_state"

    @staticmethod
    def _uses_terminal_close_oca_policy(order: Order) -> bool:
        metadata = dict(order.metadata or {})
        policy = str(metadata.get("oca_policy") or "")
        if policy in {EXIT_OCA_POLICY, NATIVE_OCA_POLICY}:
            return True
        if _metadata_truthy(metadata.get("reduce_only")) or _metadata_truthy(metadata.get("exit_only")):
            return True
        return is_exit_order(order)

    def _startup_symbol_position_flat(self, symbol: str) -> bool | None:
        broker = getattr(self, "_broker", None)
        get_positions = getattr(broker, "get_positions", None)
        if not callable(get_positions):
            return None
        try:
            positions = list(get_positions())
        except Exception:
            log.exception("engine.oca_reconcile_position_query_failed", symbol=symbol)
            return None
        for position in positions:
            if getattr(position, "symbol", None) != symbol:
                continue
            try:
                qty = float(getattr(position, "qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                return None
            if abs(qty) > 1e-12:
                return False
        return True

    def _oca_root_for_open_order(self, order: Order, group: str, strategy_id: str) -> str:
        if not strategy_id:
            return ""
        invalid_reason = validate_strategy_scoped_oca_group(
            group,
            strategy_id=strategy_id,
            symbol=order.symbol,
        )
        if invalid_reason:
            return ""
        metadata = dict(order.metadata or {})
        explicit_root = str(
            metadata.get("position_instance_id")
            or metadata.get("oca_root")
            or metadata.get("entry_root_id")
            or metadata.get("entry_intent_id")
            or ""
        )
        if explicit_root:
            return explicit_root
        prefix = f"{strategy_id}:{order.symbol}:"
        if group.startswith(prefix) and group.endswith(":exit_oca"):
            return group.removeprefix(prefix).removesuffix(":exit_oca")
        return ""

    def _append_oca_event(
        self,
        event_kind: str,
        order: Order | None,
        oca_group: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        append_fn = getattr(getattr(self, "_oms", None), "append_event", None)
        if not callable(append_fn):
            return
        payload = {
            "event_kind": event_kind,
            "oca_group": oca_group,
            "symbol": order.symbol if order is not None else "",
            "order_id": order.order_id if order is not None else "",
            "strategy_id": str(order.metadata.get("strategy_id") or "") if order is not None else "",
            "metadata": dict(metadata or {}),
        }
        append_fn(event_kind, datetime.now(timezone.utc), payload)

    def _seed_ttl_trackers_from_open_orders(self, open_orders: list[Order] | None = None) -> None:
        """Seed per-strategy live TTL adapters after durable and exchange state sync."""
        for slot in self._slots:
            if open_orders is not None:
                seed_fn = getattr(slot.ctx.broker, "seed_ttl_orders", None)
                if callable(seed_fn):
                    seed_fn(open_orders)
                    continue
            seed_fn = getattr(slot.ctx.broker, "seed_ttl_orders_from_open_orders", None)
            if callable(seed_fn):
                seed_fn()

    def _clear_ttl_tracking_for_fill(self, slot: _StrategySlot, fill: Fill) -> None:
        clear_fn = getattr(slot.ctx.broker, "clear_ttl_for_fill", None)
        if callable(clear_fn):
            clear_fn(fill)

    def _record_oms_fill_received(self, fill_id: str, fill: Fill, strategy_id: str) -> None:
        """Persist a seen fill without treating it as safely consumed."""
        oms = getattr(self, "_oms", None)
        if oms is None:
            return
        broker = getattr(self, "_broker", None)
        exchange_oid = str(getattr(broker, "_local_to_oid", {}).get(fill.order_id, ""))
        exchange_oid = fill.exchange_order_id or exchange_oid
        record_fn = getattr(oms, "record_received_fill", None)
        if not callable(record_fn):
            return
        raw = dict(fill.raw)
        if fill.tag and not raw.get("tag"):
            raw["tag"] = fill.tag
        record_fn(
            fill_id=fill_id,
            client_order_id=fill.order_id,
            exchange_order_id=exchange_oid,
            strategy_id=strategy_id,
            symbol=fill.symbol,
            side=fill.side.value,
            qty=fill.qty,
            price=fill.fill_price,
            commission=fill.commission,
            timestamp=fill.timestamp,
            exchange_fill_id=fill.exchange_fill_id,
            raw=raw,
        )

    def _mark_oms_fill_strategy_dispatched(self, fill_id: str, *, strategy_id: str) -> None:
        self._mark_oms_fill(fill_id, "mark_fill_strategy_dispatched", strategy_id=strategy_id)

    def _mark_oms_fill_coordinator_applied(self, fill_id: str, *, strategy_id: str) -> None:
        self._mark_oms_fill(fill_id, "mark_fill_coordinator_applied", strategy_id=strategy_id)

    def _mark_oms_fill_lifecycle_applied(self, fill_id: str, *, strategy_id: str) -> None:
        self._mark_oms_fill(fill_id, "mark_fill_lifecycle_applied", strategy_id=strategy_id)

    def _mark_oms_fill_finalized(self, fill_id: str, *, strategy_id: str) -> None:
        self._mark_oms_fill(fill_id, "mark_fill_finalized", strategy_id=strategy_id)

    def _mark_oms_fill_processed(self, fill_id: str, *, strategy_id: str) -> None:
        self._mark_oms_fill(fill_id, "mark_fill_processed", strategy_id=strategy_id)

    def _mark_oms_fill_unresolved(
        self,
        fill_id: str,
        *,
        strategy_id: str = "",
        reason: str = "",
    ) -> None:
        self._mark_oms_fill(
            fill_id,
            "mark_fill_unresolved",
            strategy_id=strategy_id,
            reason=reason,
        )

    def _mark_oms_fill_processing_failed(
        self,
        fill_id: str,
        *,
        strategy_id: str,
        error: str,
    ) -> None:
        self._mark_oms_fill(
            fill_id,
            "mark_fill_processing_failed",
            strategy_id=strategy_id,
            error=error,
        )

    def _record_oms_fill_processing_error(
        self,
        fill_id: str,
        *,
        strategy_id: str,
        error: str,
    ) -> None:
        self._mark_oms_fill(
            fill_id,
            "record_fill_processing_error",
            strategy_id=strategy_id,
            error=error,
        )

    def _mark_oms_fill(self, fill_id: str, method_name: str, **kwargs: Any) -> None:
        oms = getattr(self, "_oms", None)
        mark_fn = getattr(oms, method_name, None)
        if callable(mark_fn):
            mark_fn(fill_id, **kwargs)

    def _find_slot(self, strategy_id: str) -> _StrategySlot | None:
        for slot in self._slots:
            if slot.strategy_id == strategy_id:
                return slot
        return None

    def _load_portfolio_config(self) -> PortfolioConfig:
        """Load portfolio config from file or create default."""
        if self._config.portfolio_config_path and self._config.portfolio_config_path.exists():
            with open(self._config.portfolio_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return PortfolioConfig.from_dict(data)

        # Default: all strategies enabled
        from crypto_trader.portfolio.config import StrategyAllocation
        return PortfolioConfig(
            initial_equity=self._broker.get_equity() if self._broker else 10_000.0,
            strategies=tuple(
                StrategyAllocation(strategy_id=sid)
                for sid in self._config.strategy_configs
            ),
        )

    def _load_strategy_config(self, strategy_id: str, config_path: Path) -> Any:
        """Load strategy-specific config from JSON file."""
        if not config_path.exists():
            log.warning("engine.config_not_found", strategy=strategy_id, path=str(config_path))
            return self._default_strategy_config(strategy_id)

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Unwrap "strategy" key if present (from optimization output)
        if "strategy" in data:
            data = data["strategy"]

        if strategy_id == "momentum":
            from crypto_trader.strategy.momentum.config import MomentumConfig
            return MomentumConfig.from_dict(data)
        elif strategy_id == "trend":
            from crypto_trader.strategy.trend.config import TrendConfig
            return TrendConfig.from_dict(data)
        elif strategy_id == "breakout":
            from crypto_trader.strategy.breakout.config import BreakoutConfig
            return BreakoutConfig.from_dict(data)
        else:
            raise ValueError(f"Unknown strategy: {strategy_id}")

    def _build_positions_snapshot(self) -> list[dict]:
        """Build position snapshot for PG upsert."""
        result = []
        if not self._broker:
            return result
        open_orders_by_symbol: dict[str, list[str]] = {}
        if getattr(self, "_oms", None) is not None and hasattr(self._oms, "list_open_orders"):
            try:
                for order in self._oms.list_open_orders():
                    symbol = str(order.get("symbol") or "")
                    if not symbol:
                        continue
                    order_id = str(order.get("client_order_id") or "")
                    if order_id:
                        open_orders_by_symbol.setdefault(symbol, []).append(order_id)
            except Exception:
                open_orders_by_symbol = {}
        manager = getattr(self, "_manager", None)
        open_risks = list(manager.state.open_risks) if manager is not None else []
        allocations = derive_strategy_position_allocations(
            self._lifecycle_snapshot(),
            open_risks,
            observed_at=datetime.now(timezone.utc),
        )
        for pos in self._broker.get_positions():
            if pos.qty == 0:
                continue
            tracked = self._tracked_positions.get(pos.symbol, {})
            strategy_id = tracked.get("strategy_id", "unknown")
            allocation = self._allocation_for_broker_position(pos, allocations, strategy_id)
            if allocation is not None and self._unknown_strategy_id(strategy_id):
                strategy_id = allocation.strategy_id
            risk_r = 0.0
            stop_price = None
            entry_time = tracked.get("entry_time") or getattr(pos, "open_time", None)
            position_instance_id = str(tracked.get("position_instance_id") or "")
            if allocation is not None:
                position_instance_id = position_instance_id or allocation.position_instance_id
                risk_r = allocation.open_risk_R
                entry_time = allocation.entry_time or entry_time
            if manager:
                for risk in manager.state.open_risks:
                    if risk.symbol == pos.symbol:
                        risk_r = risk_r or risk.risk_R
                        stop_price = getattr(risk, "stop_price", None)
                        entry_time = entry_time or risk.entry_time
                        position_instance_id = position_instance_id or str(risk.position_instance_id or "")
                        break
            position_instance_id = (
                position_instance_id
                or self._open_lifecycle_position_instance_id(strategy_id, pos.symbol, pos.direction)
                or (
                    entry_position_instance_id(strategy_id, pos.symbol, pos.direction, entry_time)
                    if hasattr(entry_time, "timestamp") and not self._unknown_strategy_id(strategy_id)
                    else ""
                )
            )
            mark_price = pos.avg_entry
            notional_usd = abs(pos.qty * mark_price)
            liquidation_price = getattr(pos, "liquidation_price", None)
            liquidation_distance_pct = None
            if liquidation_price and mark_price:
                liquidation_distance_pct = abs(mark_price - liquidation_price) / mark_price * 100
            result.append({
                "position_instance_id": position_instance_id,
                "strategy_id": strategy_id,
                "symbol": pos.symbol,
                "direction": pos.direction.value if pos.direction else "unknown",
                "qty": pos.qty,
                "avg_entry": pos.avg_entry,
                "mark_price": mark_price,
                "notional_usd": notional_usd,
                "unrealized_pnl": pos.unrealized_pnl,
                "realized_pnl": pos.realized_pnl,
                "risk_r": risk_r,
                "risk_R": risk_r,
                "stop_price": stop_price,
                "liquidation_price": liquidation_price,
                "liquidation_distance_pct": liquidation_distance_pct,
                "fees_paid": getattr(pos, "partial_exit_commission", 0.0),
                "funding_paid": tracked.get("funding_paid", 0.0),
                "mfe_r": tracked.get("mfe_r"),
                "mae_r": tracked.get("mae_r"),
                "open_order_ids": open_orders_by_symbol.get(pos.symbol, []),
                "entry_time": entry_time,
                "source": "broker",
            })
        return result

    def _allocation_for_broker_position(
        self,
        position: Position,
        allocations: list[Any],
        strategy_id: str,
    ) -> Any | None:
        direction = position.direction
        matches = [
            allocation for allocation in allocations
            if allocation.symbol == position.symbol and allocation.direction == direction
        ]
        if not matches:
            return None
        if not self._unknown_strategy_id(strategy_id):
            for allocation in matches:
                if allocation.strategy_id == strategy_id:
                    return allocation
            return None
        return matches[0] if len(matches) == 1 else None

    def _default_strategy_config(self, strategy_id: str) -> Any:
        """Create default config for a strategy."""
        if strategy_id == "momentum":
            from crypto_trader.strategy.momentum.config import MomentumConfig
            return MomentumConfig()
        elif strategy_id == "trend":
            from crypto_trader.strategy.trend.config import TrendConfig
            return TrendConfig()
        elif strategy_id == "breakout":
            from crypto_trader.strategy.breakout.config import BreakoutConfig
            return BreakoutConfig()
        else:
            raise ValueError(f"Unknown strategy: {strategy_id}")


def _parse_datetime_or_now(value: str | None) -> datetime:
    if value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _metadata_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
