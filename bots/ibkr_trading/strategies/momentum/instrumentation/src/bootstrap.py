"""Instrumentation bootstrap — wires all components and integrates with OMS EventBus."""
from __future__ import annotations

import asyncio
import logging
import os
import weakref
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

if TYPE_CHECKING:
    from libs.oms.services.oms_service import OMSService
from libs.instrumentation.event_contract import (
    append_jsonl_event,
    enrich_payload,
    write_risk_halt_event,
    write_startup_events,
)
from libs.instrumentation.lineage import lineage_from_config
from libs.instrumentation.startup_state import collect_startup_snapshot_state

from .market_snapshot import MarketSnapshotService
from .trade_logger import TradeLogger
from .missed_opportunity import MissedOpportunityLogger
from .process_scorer import ProcessScorer
from .daily_snapshot import DailySnapshotBuilder
from .regime_classifier import RegimeClassifier
from .sidecar import Sidecar
from .experiment import ExperimentRegistry
from .order_logger import OrderLogger
from .config_watcher import ConfigWatcher

logger = logging.getLogger("instrumentation.bootstrap")

_BOT_ID = "momentum_nq_01"


def _resolve_applied_portfolio_rules_config(get_applied_config) -> object | None:
    if not callable(get_applied_config):
        return None
    try:
        return get_applied_config()
    except Exception as exc:
        logger.warning("Failed to read applied portfolio rules config for instrumentation lineage: %s", exc)
        return None


def _load_config(strategy_id: str, strategy_type: str) -> dict:
    """Load instrumentation_config.yaml with family bot_id and per-strategy strategy_id."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "instrumentation_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    _default_data_dir = str(Path(__file__).resolve().parent.parent / "data")
    config["bot_id"] = _BOT_ID
    config["strategy_id"] = strategy_id
    config["strategy_type"] = strategy_type
    config.setdefault("data_dir", _default_data_dir)
    config.setdefault("data_source_id", "ibkr_cme_nq")
    config["portfolio_id"] = config.get("portfolio_id") or os.environ.get("PORTFOLIO_ID") or "paper_default"
    config["account_alias"] = (
        config.get("account_alias")
        or os.environ.get("ACCOUNT_ALIAS")
        or os.environ.get("TRADING_ACCOUNT_ALIAS")
        or os.environ.get("BROKER_ACCOUNT_ALIAS")
        or "paper_ibkr_1"
    )
    config.setdefault("sidecar", {})
    config["sidecar"].setdefault("relay_url", "http://127.0.0.1:8000/events")

    # Experiment tracking from config or environment
    config["experiment_id"] = config.get("experiment_id") or os.environ.get("EXPERIMENT_ID")
    config["experiment_variant"] = config.get("experiment_variant") or os.environ.get("EXPERIMENT_VARIANT")

    return config


class InstrumentationManager:
    """
    Central bootstrap that creates all instrumentation services and connects
    them to the OMS event bus.

    Usage in strategy main.py:
        instr = InstrumentationManager(oms, strategy_id, strategy_type)
        await instr.start()
        # ... run strategy ...
        await instr.stop()
    """

    def __init__(
        self,
        oms: "OMSService",
        strategy_id: str,
        strategy_type: str,
        data_provider=None,
        pg_store=None,
        family_strategy_ids: list[str] | None = None,
        get_regime_ctx=None,
        get_applied_config=None,
        write_daily_closeout_on_stop: bool = True,
        stop_sidecar_on_stop: bool = True,
    ):
        self._oms = oms
        self._strategy_id = strategy_id
        self._config = _load_config(strategy_id, strategy_type)
        self._config["family_id"] = "momentum"
        self._get_regime_ctx = get_regime_ctx
        self._get_applied_config = get_applied_config
        portfolio_rules_config = _resolve_applied_portfolio_rules_config(get_applied_config)
        self.lineage = lineage_from_config(
            self._config,
            family_id="momentum",
            strategy_id=self._config.get("strategy_id", strategy_id),
            portfolio_rules_config=portfolio_rules_config,
        )
        self._config["lineage"] = self.lineage
        self._pg_store = pg_store
        self._family_strategy_ids = list(family_strategy_ids or [])
        self._instrumentation_kits: weakref.WeakSet = weakref.WeakSet()

        self.snapshot_service = MarketSnapshotService(self._config, data_provider)
        self.process_scorer = ProcessScorer()
        self.trade_logger = TradeLogger(
            self._config, self.snapshot_service,
            process_scorer=self.process_scorer,
            strategy_type=strategy_type,
            pg_store=pg_store,
            family_strategy_ids=family_strategy_ids,
        )
        self.missed_logger = MissedOpportunityLogger(self._config, self.snapshot_service)
        self.order_logger = OrderLogger(self._config, strategy_type=strategy_type)
        self.experiment_registry = ExperimentRegistry()
        self.daily_builder = DailySnapshotBuilder(self._config, experiment_registry=self.experiment_registry, get_regime_ctx=get_regime_ctx, get_applied_config=get_applied_config)
        self.regime_classifier = RegimeClassifier(data_provider=data_provider)
        self.sidecar = Sidecar(self._config)

        # Phase 2B: config change detection — only monitor this strategy's config
        _strategy_config_map = {
            "nqdtc": ["strategies.momentum.nqdtc.config"],
            "vdubus": ["strategies.momentum.vdub.config"],
            "downturn": ["strategies.momentum.downturn.config"],
            "nq_regime": ["strategies.momentum.nq_regime.config"],
        }
        config_modules = _strategy_config_map.get(strategy_type, [])
        try:
            self.config_watcher = ConfigWatcher(
                bot_id=strategy_id,
                config_modules=config_modules,
                data_dir=self._config["data_dir"],
                lineage=self.lineage,
            )
        except Exception:
            self.config_watcher = None

        self._event_queue: Optional[asyncio.Queue] = None
        self._event_task: Optional[asyncio.Task] = None
        self._snapshot_task: Optional[asyncio.Task] = None
        self._running = False
        self._write_daily_closeout_on_stop = write_daily_closeout_on_stop
        self._stop_sidecar_on_stop = stop_sidecar_on_stop
        self._daily_closeout_written = False
        self._sidecar_forwarding_enabled = True

    def refresh_lineage(self, portfolio_rules_config=None) -> None:
        """Refresh runtime lineage after dynamic portfolio-rule changes."""
        try:
            if portfolio_rules_config is None:
                portfolio_rules_config = _resolve_applied_portfolio_rules_config(self._get_applied_config)
            lineage_config = dict(self._config)
            lineage_config.pop("lineage", None)
            refreshed = lineage_from_config(
                lineage_config,
                family_id="momentum",
                strategy_id=lineage_config.get("strategy_id", self._strategy_id),
                portfolio_rules_config=portfolio_rules_config,
            )
            refreshed = replace(
                refreshed,
                deployment_id=self.lineage.deployment_id or refreshed.deployment_id,
                trace_id=self.lineage.trace_id or refreshed.trace_id,
            )
            self.lineage = refreshed
            self._config["lineage"] = refreshed
            for component in (
                self.snapshot_service,
                self.trade_logger,
                self.missed_logger,
                self.order_logger,
                self.daily_builder,
            ):
                if hasattr(component, "_lineage"):
                    component._lineage = refreshed
            if self.config_watcher is not None and hasattr(self.config_watcher, "_lineage"):
                self.config_watcher._lineage = refreshed
            for kit in list(getattr(self, "_instrumentation_kits", ()) or ()):
                refresh = getattr(kit, "refresh_lineage", None)
                if callable(refresh):
                    refresh(refreshed)
        except Exception as exc:
            logger.warning("Failed to refresh instrumentation lineage: %s", exc)

    def _register_instrumentation_kit(self, kit) -> None:
        """Track facade kits so runtime lineage refresh reaches lazy loggers."""
        try:
            kits = getattr(self, "_instrumentation_kits", None)
            if kits is None:
                kits = weakref.WeakSet()
                self._instrumentation_kits = kits
            kits.add(kit)
        except Exception:
            logger.debug("Failed to register instrumentation kit", exc_info=True)

    def get_sidecar_diagnostics(self) -> Optional[dict]:
        """Return sidecar health diagnostics for heartbeat (#24)."""
        try:
            return self.sidecar.get_diagnostics()
        except Exception:
            return None

    def emit_heartbeat(
        self,
        active_positions: int,
        open_orders: int,
        uptime_s: float,
        error_count_1h: int,
        positions: Optional[list] = None,
        portfolio_exposure: Optional[dict] = None,
    ) -> None:
        """Proxy to facade kit for heartbeat emission with position state."""
        try:
            from .facade import InstrumentationKit
            kit = InstrumentationKit(self, strategy_type=self._config.get("strategy_type", ""))
            kit.emit_heartbeat(
                active_positions, open_orders, uptime_s, error_count_1h,
                positions=positions, portfolio_exposure=portfolio_exposure,
            )
        except Exception:
            pass

    async def start(self) -> None:
        """Subscribe to OMS events and start background tasks."""
        if self._running:
            return

        # Enforce HMAC auth in paper/live — match stock bootstrap behavior
        from libs.oms.persistence.db_config import get_environment
        env = get_environment()
        sidecar_forwarding_enabled = True
        if env in ("paper", "live") and not self.sidecar.hmac_secret:
            sidecar_forwarding_enabled = False
            logger.warning(
                "Sidecar forwarding disabled in %s mode: missing INSTRUMENTATION_HMAC_SECRET. "
                "Local startup instrumentation will continue.",
                env,
            )
        self._sidecar_forwarding_enabled = sidecar_forwarding_enabled

        self._running = True

        try:
            portfolio_rules_config = None
            try:
                if callable(self._get_applied_config):
                    portfolio_rules_config = self._get_applied_config()
            except Exception as e:
                logger.warning("Failed to read applied portfolio rules config for startup snapshot: %s", e)
            allocation_state, portfolio_state, positions = await collect_startup_snapshot_state(
                self._oms,
                strategy_ids=self._family_strategy_ids or [self._config.get("strategy_id", self._strategy_id)],
                default_strategy_id=self._config.get("strategy_id", self._strategy_id),
            )
            write_startup_events(
                self._config["data_dir"],
                self.lineage,
                effective_config={
                    "bot_id": self._config.get("bot_id", ""),
                    "strategy_id": self._config.get("strategy_id", ""),
                    "strategy_type": self._config.get("strategy_type", ""),
                    "market_snapshots": self._config.get("market_snapshots", {}),
                    "data_source_id": self._config.get("data_source_id", ""),
                },
                allocation_state=allocation_state,
                portfolio_state=portfolio_state,
                positions=positions,
                portfolio_rules_config=portfolio_rules_config,
            )
        except Exception as e:
            logger.warning("Failed to write startup instrumentation events: %s", e)

        try:
            self._event_queue = self._oms.stream_all_events()
            self._event_task = asyncio.create_task(self._event_loop())
        except Exception as e:
            logger.warning("Failed to subscribe to OMS events: %s", e)

        interval = self._config.get("market_snapshots", {}).get("interval_seconds", 60)
        self._snapshot_task = asyncio.create_task(self._periodic_snapshot_loop(interval))

        if sidecar_forwarding_enabled:
            try:
                self.sidecar.start()
            except Exception as e:
                self._sidecar_forwarding_enabled = False
                logger.warning("Failed to start sidecar: %s", e)

        logger.info("Instrumentation started for %s", self._strategy_id)

    async def stop(self) -> None:
        """Shutdown: build daily snapshot, stop background tasks, stop sidecar."""
        self._running = False

        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass

        if self._snapshot_task:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass

        if self._event_queue:
            try:
                self._oms.event_bus.unsubscribe_all(self._event_queue)
            except Exception:
                pass

        if self._write_daily_closeout_on_stop:
            await self.write_daily_closeout()

        if self._stop_sidecar_on_stop and self._sidecar_forwarding_enabled:
            self.flush_and_stop_sidecar()

        logger.info("Instrumentation stopped for %s", self._strategy_id)

    def flush_and_stop_sidecar(self) -> None:
        try:
            self.sidecar.run_once()
            self.sidecar.stop()
        except Exception as e:
            logger.warning("Sidecar stop error: %s", e)

    async def write_daily_closeout(
        self,
        *,
        oms_services: list | None = None,
        strategy_ids: list[str] | None = None,
        family_id: str = "momentum",
    ) -> None:
        if self._daily_closeout_written:
            return
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            snapshot = self.daily_builder.build(today)
            self.daily_builder.save(snapshot)
            await self._write_daily_reconciliation(
                today,
                snapshot,
                oms_services=oms_services,
                strategy_ids=strategy_ids,
                family_id=family_id,
            )
            self._daily_closeout_written = True
            logger.info("Daily snapshot saved for %s", today)
        except Exception as e:
            logger.warning("Failed to build daily snapshot: %s", e)

    async def _write_daily_reconciliation(
        self,
        today: str,
        snapshot,
        *,
        oms_services: list | None = None,
        strategy_ids: list[str] | None = None,
        family_id: str = "momentum",
    ) -> None:
        try:
            from libs.oms.instrumentation.daily_state import collect_family_daily_state
            from libs.oms.instrumentation.lifecycle import write_daily_reconciliation

            services = list(oms_services or [self._oms])
            portfolio_state, _, allocation_state = await collect_family_daily_state(
                services,
                strategy_ids=list(strategy_ids or []),
                default_strategy_id=self._strategy_id,
            )
            write_daily_reconciliation(
                self._config["data_dir"],
                replace(self.lineage, strategy_id=""),
                date_str=today,
                family_id=family_id,
                daily_snapshot=snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot),
                portfolio_state=portfolio_state,
                allocation_state=allocation_state,
            )
        except Exception as e:
            logger.warning("Failed to write daily reconciliation bundle: %s", e)

    _ORDER_STATUS_MAP = None  # lazily initialised

    async def _event_loop(self) -> None:
        """Process OMS events — logs RISK_DENIAL, order status, and fills."""
        from libs.oms.models.events import OMSEventType

        if InstrumentationManager._ORDER_STATUS_MAP is None:
            InstrumentationManager._ORDER_STATUS_MAP = {
                OMSEventType.ORDER_FILLED: "FILLED",
                OMSEventType.ORDER_PARTIALLY_FILLED: "PARTIAL_FILL",
                OMSEventType.ORDER_REJECTED: "REJECTED",
                OMSEventType.ORDER_CANCELLED: "CANCELLED",
                OMSEventType.ORDER_EXPIRED: "EXPIRED",
            }

        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                if event.event_type == OMSEventType.RISK_DENIAL:
                    self._handle_risk_denial(event)
                elif event.event_type == OMSEventType.RISK_DECISION:
                    self._handle_risk_decision(event)
                elif event.event_type == OMSEventType.RISK_HALT:
                    self._handle_risk_halt(event)
                elif event.event_type in self._ORDER_STATUS_MAP:
                    self._handle_order_status(event, self._ORDER_STATUS_MAP[event.event_type])
                elif event.event_type == OMSEventType.FILL:
                    self._handle_fill_event(event)
            except Exception as e:
                logger.warning("Error processing OMS event: %s", e)

    def _handle_risk_denial(self, event) -> None:
        """Log risk denials as missed opportunities with available context."""
        try:
            payload = event.payload or {}
            reason = payload.get("reason", "unknown")

            # Use enriched payload from IntentHandler, fall back to defaults
            symbols = self._config.get("market_snapshots", {}).get("symbols", [])
            pair = payload.get("symbol") or (symbols[0] if symbols else "NQ")
            side = payload.get("side", "UNKNOWN")
            signal = payload.get("signal_name", f"risk_denial_{self._strategy_id}")
            signal_strength = payload.get("signal_strength", 0.0)

            self.missed_logger.log_missed(
                pair=pair,
                side=side,
                signal=signal,
                signal_id=event.oms_order_id or "",
                signal_strength=signal_strength,
                blocked_by="risk_gateway",
                block_reason=reason,
                strategy_type=self._config.get("strategy_type"),
                exchange_timestamp=event.timestamp,
            )
        except Exception as e:
            logger.warning("Failed to log risk denial as missed: %s", e)

    def _handle_risk_decision(self, event) -> None:
        """Persist full OMS risk decisions for assistant audit."""
        try:
            timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, "isoformat") else str(event.timestamp)
            payload = enrich_payload(
                {
                    "timestamp": timestamp,
                    **dict(event.payload or {}),
                },
                lineage=dict(event.payload or {}).get("lineage"),
                event_type="risk_decision",
                scope="oms",
            )
            append_jsonl_event(
                self._config["data_dir"],
                "risk_decisions",
                "risk_decisions",
                payload,
            )
        except Exception as e:
            logger.warning("Failed to log risk decision: %s", e)

    def _handle_risk_halt(self, event) -> None:
        """Persist hard risk halts as first-class urgent events."""
        try:
            payload = dict(event.payload or {})
            write_risk_halt_event(
                self._config["data_dir"],
                payload.get("lineage") or self.lineage,
                reason=str(payload.get("reason") or "risk_halt"),
                strategy_id=str(event.strategy_id or payload.get("strategy_id") or ""),
                timestamp=getattr(event, "timestamp", None),
                details=payload,
            )
        except Exception as e:
            logger.warning("Failed to log risk halt: %s", e)

    def _handle_order_status(self, event, status_label: str) -> None:
        """Log order status transitions (FILLED, REJECTED, CANCELLED, etc.)."""
        try:
            payload = event.payload or {}
            self.order_logger.log_order(
                order_id=event.oms_order_id or "",
                pair=payload.get("symbol", "NQ"),
                side=payload.get("side", ""),
                order_type=payload.get("order_type", ""),
                status=status_label,
                requested_qty=payload.get("qty", 0),
                reject_reason=payload.get("reject_reason", ""),
                strategy_type=self._config.get("strategy_type", ""),
                exchange_timestamp=event.timestamp,
            )
        except Exception as e:
            logger.warning("Failed to log order status %s: %s", status_label, e)

    def _handle_fill_event(self, event) -> None:
        """Log fill events with execution details."""
        try:
            payload = event.payload or {}
            self.order_logger.log_order(
                order_id=event.oms_order_id or "",
                pair=payload.get("symbol", "NQ"),
                side=payload.get("side", ""),
                order_type=payload.get("order_type", ""),
                status="FILL",
                requested_qty=payload.get("requested_qty", 0),
                filled_qty=payload.get("qty", 0),
                fill_price=payload.get("price"),
                strategy_type=self._config.get("strategy_type", ""),
                exchange_timestamp=event.timestamp,
            )
        except Exception as e:
            logger.warning("Failed to log fill event: %s", e)

    async def _periodic_snapshot_loop(self, interval: int) -> None:
        """Capture market snapshots at regular intervals."""
        while self._running:
            try:
                self.snapshot_service.run_periodic()
            except Exception as e:
                logger.warning("Periodic snapshot failed: %s", e)
            # Config change detection (Phase 2B)
            try:
                if self.config_watcher:
                    self.config_watcher.check()
            except Exception as e:
                logger.warning("Config watcher check failed: %s", e)
            # Post-exit price backfill
            try:
                if hasattr(self.snapshot_service, '_data_provider') and self.snapshot_service._data_provider:
                    self.trade_logger.run_post_exit_backfill(self.snapshot_service._data_provider)
            except Exception as e:
                logger.warning("Post-exit backfill failed: %s", e)
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
