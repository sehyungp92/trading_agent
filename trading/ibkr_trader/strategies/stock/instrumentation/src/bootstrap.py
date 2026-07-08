"""Instrumentation bootstrap wires all components and integrates with the OMS event bus."""
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

from libs.oms.persistence.db_config import get_environment
from libs.instrumentation.event_contract import (
    append_jsonl_event,
    enrich_payload,
    write_risk_halt_event,
    write_startup_events,
)
from libs.instrumentation.lineage import lineage_from_config
from libs.instrumentation.startup_state import collect_startup_snapshot_state

from .config_watcher import ConfigWatcher
from .daily_snapshot import DailySnapshotBuilder
from .error_logger import ErrorLogger
from .experiment import ExperimentRegistry
from .market_snapshot import MarketSnapshotService
from .missed_opportunity import MissedOpportunityLogger
from .order_logger import OrderLogger
from .process_scorer import ProcessScorer
from .regime_classifier import RegimeClassifier
from .sidecar import Sidecar
from .trade_logger import TradeLogger

logger = logging.getLogger("instrumentation.bootstrap")

_STRATEGY_CONFIG_MODULES = {
    "IARIC_v1": ["strategy_iaric.config"],
    "ALCB_v1": ["strategy_alcb.config"],
    "strategy_iaric": ["strategy_iaric.config"],
    "strategy_alcb": ["strategy_alcb.config"],
}

_BOT_NAME_MAP = {
    "IARIC_v1": "IARIC v1",
    "ALCB_v1": "ALCB v1",
}

# Single bot_id for all strategies; strategy_id distinguishes them in events
_BOT_ID = "stock_trader"
_HMAC_SECRET_ENV = "INSTRUMENTATION_HMAC_SECRET"


def _resolve_applied_portfolio_rules_config(get_applied_config) -> object | None:
    if not callable(get_applied_config):
        return None
    try:
        return get_applied_config()
    except Exception as exc:
        logger.warning("Failed to read applied portfolio rules config for instrumentation lineage: %s", exc)
        return None
# Pass strategy_ids through unchanged to match registry keys.
_STRATEGY_ID_MAP = {
    "IARIC_v1": "IARIC_v1",
    "ALCB_v1": "ALCB_v1",
}


def _normalize_strategy_type(strategy_id: str, strategy_type: str) -> str:
    if strategy_type:
        return strategy_type
    return strategy_id.lower()


def _resolve_config_modules(strategy_id: str, strategy_type: str) -> list[str]:
    if strategy_id in _STRATEGY_CONFIG_MODULES:
        return list(_STRATEGY_CONFIG_MODULES[strategy_id])
    return list(_STRATEGY_CONFIG_MODULES.get(strategy_type, []))


def _load_config(strategy_id: str, strategy_type: str) -> dict:
    """Load instrumentation config and adapt it to the target strategy."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "instrumentation_config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    else:
        config = {}

    normalized_strategy_type = _normalize_strategy_type(strategy_id, strategy_type)
    _default_data_dir = str(Path(__file__).resolve().parent.parent / "data")
    data_dir = os.environ.get("INSTRUMENTATION_DATA_DIR") or config.get("data_dir") or _default_data_dir
    sidecar_config = dict(config.get("sidecar") or {})
    market_snapshots = dict(config.get("market_snapshots") or {})
    logging_config = dict(config.get("logging") or {})

    config["bot_id"] = _BOT_ID
    config["strategy_id"] = _STRATEGY_ID_MAP.get(strategy_id, normalized_strategy_type)
    config["bot_name"] = config.get("bot_name") or _BOT_NAME_MAP.get(strategy_id, strategy_id)
    config["strategy_type"] = normalized_strategy_type
    config["data_dir"] = data_dir
    config["data_source_id"] = config.get("data_source_id") or "ibkr_us_equities"
    config["portfolio_id"] = config.get("portfolio_id") or os.environ.get("PORTFOLIO_ID") or "paper_default"
    config["account_alias"] = (
        config.get("account_alias")
        or os.environ.get("ACCOUNT_ALIAS")
        or os.environ.get("TRADING_ACCOUNT_ALIAS")
        or os.environ.get("BROKER_ACCOUNT_ALIAS")
        or "paper_ibkr_1"
    )

    if not market_snapshots.get("symbols"):
        market_snapshots["symbols"] = ["SPY", "QQQ", "IWM"]
    market_snapshots.setdefault("interval_seconds", 60)
    config["market_snapshots"] = market_snapshots

    sidecar_config.setdefault("relay_url", "http://127.0.0.1:8000/events")
    sidecar_config["hmac_secret_env"] = _HMAC_SECRET_ENV
    sidecar_config["buffer_dir"] = os.environ.get("INSTRUMENTATION_BUFFER_DIR") or str(Path(data_dir) / ".sidecar_buffer")
    config["sidecar"] = sidecar_config

    logging_config.setdefault("file", str(Path(data_dir) / "instrumentation.log"))
    config["logging"] = logging_config

    config["experiment_id"] = config.get("experiment_id") or os.environ.get("EXPERIMENT_ID")
    config["experiment_variant"] = config.get("experiment_variant") or os.environ.get("EXPERIMENT_VARIANT")
    config["heartbeat_interval_seconds"] = int(
        os.environ.get("INSTRUMENTATION_HEARTBEAT_INTERVAL_SECONDS")
        or config.get("heartbeat_interval_seconds")
        or 30
    )
    config["daily_snapshot_checkpoint_interval_seconds"] = int(
        os.environ.get("INSTRUMENTATION_DAILY_SNAPSHOT_CHECKPOINT_INTERVAL_SECONDS")
        or config.get("daily_snapshot_checkpoint_interval_seconds")
        or 300
    )
    config["heartbeat_gap_multiplier"] = float(
        os.environ.get("INSTRUMENTATION_HEARTBEAT_GAP_MULTIPLIER")
        or config.get("heartbeat_gap_multiplier")
        or 2.5
    )

    return config


class InstrumentationManager:
    """Create instrumentation services, subscribe to OMS events, and manage the sidecar."""

    _ORDER_STATUS_MAP = None

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
    ) -> None:
        self._oms = oms
        self._strategy_id = strategy_id
        self._strategy_type = _normalize_strategy_type(strategy_id, strategy_type)
        self._config = _load_config(strategy_id, self._strategy_type)
        self._config["family_id"] = "stock"
        self._get_regime_ctx = get_regime_ctx
        self._get_applied_config = get_applied_config
        portfolio_rules_config = _resolve_applied_portfolio_rules_config(get_applied_config)
        self.lineage = lineage_from_config(
            self._config,
            family_id="stock",
            strategy_id=self._config.get("strategy_id", strategy_id),
            portfolio_rules_config=portfolio_rules_config,
        )
        self._config["lineage"] = self.lineage
        self.bot_id = self._config["bot_id"]
        self._pg_store = pg_store
        self._data_provider = None
        self._family_strategy_ids = list(family_strategy_ids or [])
        self._instrumentation_kits: weakref.WeakSet = weakref.WeakSet()

        self.error_logger = ErrorLogger(self._config)
        self.snapshot_service = MarketSnapshotService(self._config, None)
        self.process_scorer = ProcessScorer()
        self.trade_logger = TradeLogger(
            self._config,
            self.snapshot_service,
            process_scorer=self.process_scorer,
            strategy_type=self._strategy_type,
            error_logger=self.error_logger,
            pg_store=pg_store,
            family_strategy_ids=family_strategy_ids,
        )
        self.missed_logger = MissedOpportunityLogger(
            self._config,
            self.snapshot_service,
            error_logger=self.error_logger,
        )
        self.order_logger = OrderLogger(self._config, strategy_type=self._strategy_type)
        self.experiment_registry = ExperimentRegistry()
        self.daily_builder = DailySnapshotBuilder(
            self._config,
            experiment_registry=self.experiment_registry,
            get_regime_ctx=get_regime_ctx,
            get_applied_config=get_applied_config,
        )
        self.regime_classifier = RegimeClassifier(data_provider=None)
        self.sidecar = Sidecar(self._config)

        config_modules = _resolve_config_modules(strategy_id, self._strategy_type)
        try:
            self.config_watcher = ConfigWatcher(
                bot_id=self.bot_id,
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
        self._last_snapshot_checkpoint_at: float = 0.0
        self._write_daily_closeout_on_stop = write_daily_closeout_on_stop
        self._stop_sidecar_on_stop = stop_sidecar_on_stop
        self._daily_closeout_written = False
        self._sidecar_forwarding_enabled = True

        if data_provider is not None:
            self.attach_data_provider(data_provider)

    @property
    def config(self) -> dict:
        return self._config

    def refresh_lineage(self, portfolio_rules_config=None) -> None:
        """Refresh runtime lineage after dynamic portfolio-rule changes."""
        try:
            if portfolio_rules_config is None:
                portfolio_rules_config = _resolve_applied_portfolio_rules_config(self._get_applied_config)
            lineage_config = dict(self._config)
            lineage_config.pop("lineage", None)
            refreshed = lineage_from_config(
                lineage_config,
                family_id="stock",
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
                self.error_logger,
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

    def attach_data_provider(self, data_provider) -> None:
        self._data_provider = data_provider
        self.snapshot_service.set_data_provider(data_provider)
        self.regime_classifier.set_data_provider(data_provider)

    def set_data_provider(self, data_provider) -> None:
        self.attach_data_provider(data_provider)

    def get_sidecar_diagnostics(self) -> Optional[dict]:
        """Return sidecar health diagnostics for heartbeat emission."""
        try:
            return self.sidecar.get_diagnostics()
        except Exception:
            return None

    def recent_error_count_1h(self) -> int:
        """Return the rolling one-hour structured error count."""
        try:
            return self.error_logger.count_recent()
        except Exception:
            return 0

    def record_error(
        self,
        *,
        error_type: str,
        message: str,
        severity: str = "medium",
        category: str = "unknown",
        context: Optional[dict] = None,
        exc: BaseException | None = None,
        exchange_timestamp: Optional[datetime] = None,
    ) -> None:
        """Write a structured error event for bot/runtime failures."""
        try:
            self.error_logger.log_error(
                error_type=error_type,
                message=message,
                severity=severity,
                category=category,
                context=context,
                exc=exc,
                exchange_timestamp=exchange_timestamp,
            )
        except Exception:
            logger.warning("Failed to record structured error %s", error_type)

    def emit_heartbeat(
        self,
        active_positions: int,
        open_orders: int,
        uptime_s: float,
        error_count_1h: int,
        positions: Optional[list] = None,
        portfolio_exposure: Optional[dict] = None,
    ) -> None:
        """Proxy to the facade kit for heartbeat emission."""
        try:
            from .facade import InstrumentationKit

            kit = InstrumentationKit(self, strategy_type=self._strategy_type)
            kit.emit_heartbeat(
                active_positions,
                open_orders,
                uptime_s,
                error_count_1h,
                positions=positions,
                portfolio_exposure=portfolio_exposure,
            )
        except Exception:
            pass

    def on_indicator_snapshot(self, *args, **kwargs) -> None:
        """Proxy direct engine indicator decisions to the facade."""
        try:
            from .facade import InstrumentationKit

            kit = InstrumentationKit(self, strategy_type=self._strategy_type)
            kit.on_indicator_snapshot(*args, **kwargs)
        except Exception:
            pass

    async def start(self) -> None:
        """Subscribe to OMS events and start background tasks."""
        if self._running:
            return

        # Enforce HMAC auth in non-dev environments — relay will reject unsigned events
        env = get_environment()
        sidecar_forwarding_enabled = True
        if env in ("paper", "live") and not self.sidecar.hmac_secret:
            sidecar_forwarding_enabled = False
            logger.warning(
                "Sidecar forwarding disabled in %s mode: missing %s. "
                "Local startup instrumentation will continue.",
                env,
                _HMAC_SECRET_ENV,
            )
        self._sidecar_forwarding_enabled = sidecar_forwarding_enabled

        self._running = True

        try:
            portfolio_rules_config = None
            try:
                if callable(self._get_applied_config):
                    portfolio_rules_config = self._get_applied_config()
            except Exception as exc:
                logger.warning("Failed to read applied portfolio rules config for startup snapshot: %s", exc)
            allocation_state, portfolio_state, positions = await collect_startup_snapshot_state(
                self._oms,
                strategy_ids=self._family_strategy_ids or [self._config.get("strategy_id", self._strategy_id)],
                default_strategy_id=self._config.get("strategy_id", self._strategy_id),
            )
            write_startup_events(
                self._config["data_dir"],
                self.lineage,
                effective_config={
                    "bot_id": self.bot_id,
                    "strategy_id": self._config.get("strategy_id", ""),
                    "strategy_type": self._strategy_type,
                    "market_snapshots": self._config.get("market_snapshots", {}),
                    "data_source_id": self._config.get("data_source_id", ""),
                },
                allocation_state=allocation_state,
                portfolio_state=portfolio_state,
                positions=positions,
                portfolio_rules_config=portfolio_rules_config,
            )
        except Exception as exc:
            logger.warning("Failed to write startup instrumentation events: %s", exc)

        try:
            self._event_queue = self._oms.stream_all_events()
            self._event_task = asyncio.create_task(self._event_loop())
        except Exception as exc:
            logger.warning("Failed to subscribe to OMS events: %s", exc)

        interval = self._config.get("market_snapshots", {}).get("interval_seconds", 60)
        self._snapshot_task = asyncio.create_task(self._periodic_snapshot_loop(interval))

        if sidecar_forwarding_enabled:
            try:
                self.sidecar.validate_configuration(strict=False)
                self.sidecar.start()
            except Exception as exc:
                self._sidecar_forwarding_enabled = False
                logger.critical(
                    "Sidecar failed to start: %s - strategy will continue trading "
                    "WITHOUT event forwarding. Fix relay configuration ASAP.",
                    exc,
                )
                self.record_error(
                    error_type="sidecar_start_failure",
                    message=str(exc),
                    severity="critical",
                    category="config_error",
                    context={"component": "Sidecar.start", "environment": get_environment()},
                    exc=exc,
                )

        logger.info("Instrumentation started for %s", self.bot_id)

    async def stop(self) -> None:
        """Flush instrumentation artifacts, stop background tasks, and stop the sidecar."""
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

        logger.info("Instrumentation stopped for %s", self.bot_id)

    def flush_and_stop_sidecar(self) -> None:
        try:
            self.sidecar.run_once()
            self.sidecar.stop()
        except Exception as exc:
            logger.warning("Sidecar stop error: %s", exc)

    async def write_daily_closeout(
        self,
        *,
        oms_services: list | None = None,
        strategy_ids: list[str] | None = None,
        family_id: str = "stock",
    ) -> None:
        if self._daily_closeout_written:
            return
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            snapshot = self.daily_builder.build(today, snapshot_kind="final")
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
        except Exception as exc:
            logger.warning("Failed to build daily snapshot: %s", exc)

    async def _write_daily_reconciliation(
        self,
        today: str,
        snapshot,
        *,
        oms_services: list | None = None,
        strategy_ids: list[str] | None = None,
        family_id: str = "stock",
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
        except Exception as exc:
            logger.warning("Failed to write daily reconciliation bundle: %s", exc)

    async def _event_loop(self) -> None:
        """Process OMS events and fan them into instrumentation logs."""
        from libs.oms.models.events import OMSEventType

        if InstrumentationManager._ORDER_STATUS_MAP is None:
            InstrumentationManager._ORDER_STATUS_MAP = {
                OMSEventType.ORDER_CREATED: "CREATED",
                OMSEventType.ORDER_RISK_APPROVED: "RISK_APPROVED",
                OMSEventType.ORDER_ROUTED: "ROUTED",
                OMSEventType.ORDER_ACKED: "ACKED",
                OMSEventType.ORDER_WORKING: "WORKING",
                OMSEventType.ORDER_FILLED: "FILLED",
                OMSEventType.ORDER_PARTIALLY_FILLED: "PARTIAL_FILL",
                OMSEventType.ORDER_REJECTED: "REJECTED",
                OMSEventType.ORDER_CANCELLED: "CANCELLED",
                OMSEventType.ORDER_EXPIRED: "EXPIRED",
            }

        while self._running:
            try:
                if self._event_queue is None:
                    await asyncio.sleep(1.0)
                    continue
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
            except Exception as exc:
                logger.warning("Error processing OMS event: %s", exc)
                self.record_error(
                    error_type="instrumentation_event_loop_error",
                    message=str(exc),
                    severity="medium",
                    category="dependency",
                    context={"component": "InstrumentationManager._event_loop"},
                    exc=exc,
                    exchange_timestamp=getattr(event, "timestamp", None),
                )

    def _handle_risk_denial(self, event) -> None:
        """Log risk denials as missed opportunities."""
        try:
            payload = event.payload or {}
            symbols = self._config.get("market_snapshots", {}).get("symbols", [])
            pair = payload.get("symbol") or (symbols[0] if symbols else "SPY")
            reason = payload.get("reason", "unknown")

            self.missed_logger.log_missed(
                pair=pair,
                side=payload.get("side", "UNKNOWN"),
                signal=payload.get("signal_name", f"risk_denial_{self.bot_id}"),
                signal_id=event.oms_order_id or "",
                signal_strength=payload.get("signal_strength", 0.0),
                blocked_by="risk_gateway",
                block_reason=reason,
                strategy_type=self._strategy_type,
                exchange_timestamp=event.timestamp,
            )
        except Exception as exc:
            logger.warning("Failed to log risk denial as missed: %s", exc)

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
        except Exception as exc:
            logger.warning("Failed to log risk decision: %s", exc)

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
        except Exception as exc:
            logger.warning("Failed to log risk halt: %s", exc)

    def _handle_order_status(self, event, status_label: str) -> None:
        """Log order status transitions."""
        try:
            payload = event.payload or {}
            self.order_logger.log_order(
                order_id=event.oms_order_id or "",
                pair=payload.get("symbol", "UNKNOWN"),
                side=payload.get("side", ""),
                order_type=payload.get("order_type", ""),
                status=status_label,
                requested_qty=payload.get("qty", 0),
                reject_reason=payload.get("reject_reason", ""),
                strategy_type=self._strategy_type,
                exchange_timestamp=event.timestamp,
            )
        except Exception as exc:
            logger.warning("Failed to log order status %s: %s", status_label, exc)

    def _handle_fill_event(self, event) -> None:
        """Log fill events with execution details."""
        try:
            payload = event.payload or {}
            self.order_logger.log_order(
                order_id=event.oms_order_id or "",
                pair=payload.get("symbol", "UNKNOWN"),
                side=payload.get("side", ""),
                order_type=payload.get("order_type", ""),
                status="FILL",
                requested_qty=payload.get("requested_qty", payload.get("qty", 0)),
                filled_qty=payload.get("qty", 0),
                fill_price=payload.get("price"),
                strategy_type=self._strategy_type,
                exchange_timestamp=event.timestamp,
            )
        except Exception as exc:
            logger.warning("Failed to log fill event: %s", exc)

    async def _periodic_snapshot_loop(self, interval: int) -> None:
        """Capture market snapshots at regular intervals."""
        while self._running:
            try:
                self.snapshot_service.run_periodic()
            except Exception as exc:
                logger.warning("Periodic snapshot failed: %s", exc)
                self.record_error(
                    error_type="market_snapshot_error",
                    message=str(exc),
                    severity="low",
                    category="dependency",
                    context={"component": "MarketSnapshotService.run_periodic"},
                    exc=exc,
                )

            try:
                if self.config_watcher:
                    self.config_watcher.check()
            except Exception as exc:
                logger.warning("Config watcher check failed: %s", exc)
                self.record_error(
                    error_type="config_watcher_error",
                    message=str(exc),
                    severity="medium",
                    category="config_error",
                    context={"component": "ConfigWatcher.check"},
                    exc=exc,
                )

            try:
                if self._data_provider is not None:
                    self.trade_logger.run_post_exit_backfill(self._data_provider)
            except Exception as exc:
                logger.warning("Post-exit backfill failed: %s", exc)
                self.record_error(
                    error_type="post_exit_backfill_error",
                    message=str(exc),
                    severity="medium",
                    category="dependency",
                    context={"component": "TradeLogger.run_post_exit_backfill"},
                    exc=exc,
                )

            try:
                if self._data_provider is not None:
                    self.missed_logger.run_backfill(self._data_provider)
            except Exception as exc:
                logger.warning("Missed-opportunity backfill failed: %s", exc)
                self.record_error(
                    error_type="missed_backfill_error",
                    message=str(exc),
                    severity="medium",
                    category="dependency",
                    context={"component": "MissedOpportunityLogger.run_backfill"},
                    exc=exc,
                )

            try:
                self._maybe_checkpoint_daily_snapshot()
            except Exception as exc:
                logger.warning("Daily snapshot checkpoint failed: %s", exc)
                self.record_error(
                    error_type="daily_snapshot_checkpoint_error",
                    message=str(exc),
                    severity="medium",
                    category="dependency",
                    context={"component": "DailySnapshotBuilder.build/save"},
                    exc=exc,
                )

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    def _maybe_checkpoint_daily_snapshot(self) -> None:
        checkpoint_interval = int(
            self._config.get("daily_snapshot_checkpoint_interval_seconds", 0) or 0
        )
        if checkpoint_interval <= 0:
            return

        now_monotonic = asyncio.get_running_loop().time()
        if (
            self._last_snapshot_checkpoint_at > 0
            and (now_monotonic - self._last_snapshot_checkpoint_at) < checkpoint_interval
        ):
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        snapshot = self.daily_builder.build(today, snapshot_kind="checkpoint")
        self.daily_builder.save(snapshot)
        self._last_snapshot_checkpoint_at = now_monotonic
