"""InstrumentationContext — single injectable object bundling all services.

Passed as ``instrumentation=ctx`` to every strategy engine.  When ``None``,
engines silently skip all instrumentation calls.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("instrumentation.context")


@dataclass
class InstrumentationContext:
    """Bundles all instrumentation services into one injectable object."""

    snapshot_service: object = None   # MarketSnapshotService
    trade_logger: object = None       # TradeLogger
    missed_logger: object = None      # MissedOpportunityLogger
    process_scorer: object = None     # ProcessScorer
    daily_builder: object = None      # DailySnapshotBuilder
    regime_classifier: object = None  # RegimeClassifier
    sidecar: object = None            # Sidecar
    drawdown_tracker: object = None   # DrawdownTracker
    overnight_gap_tracker: object = None  # OvernightGapTracker
    coordination_logger: object = None   # CoordinationLogger
    order_logger: object = None           # OrderLogger
    indicator_logger: object = None       # IndicatorLogger
    filter_logger: object = None          # FilterLogger
    orderbook_logger: object = None       # OrderBookLogger
    experiment_registry: object = None    # ExperimentRegistry
    pg_store: object = None               # PgStore
    overlay_state_provider: object = None  # Callable[[], dict[str, bool]]
    post_exit_tracker: object = None      # PostExitTracker
    bot_id: str = ""
    data_dir: str = "instrumentation/data"
    lineage: object = None
    get_regime_ctx: object = None       # Callable[[], RegimeContext | None]
    get_applied_config: object = None   # Callable[[], PortfolioRulesConfig | None]
    oms: object = None                  # OMSService used for closeout reconciliation
    family_id: str = "swing"

    _started: bool = field(default=False, repr=False)
    _backfill_stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _backfill_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _async_stop_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _event_queue: Optional[asyncio.Queue] = field(default=None, repr=False)
    _event_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _sidecar_forwarding_enabled: bool = field(default=True, repr=False)

    def refresh_lineage(self, portfolio_rules_config=None) -> None:
        """Refresh runtime lineage after dynamic portfolio-rule changes."""
        if self.lineage is None or portfolio_rules_config is None:
            return
        try:
            from dataclasses import replace
            from libs.instrumentation.lineage import lineage_from_runtime

            refreshed = lineage_from_runtime(
                bot_id=getattr(self.lineage, "bot_id", self.bot_id or "swing_bot"),
                strategy_id=getattr(self.lineage, "strategy_id", ""),
                family_id=getattr(self.lineage, "family_id", self.family_id or "swing"),
                portfolio_id=getattr(self.lineage, "portfolio_id", ""),
                account_alias=getattr(self.lineage, "account_alias", ""),
                strategy_version=getattr(self.lineage, "strategy_version", ""),
                portfolio_rules_config=portfolio_rules_config,
                effective_strategy_config={
                    "bot_id": getattr(self.lineage, "bot_id", self.bot_id or "swing_bot"),
                    "strategy_id": getattr(self.lineage, "strategy_id", ""),
                    "family_id": getattr(self.lineage, "family_id", self.family_id or "swing"),
                    "runtime_refresh": "portfolio_rules",
                },
            )
            refreshed = replace(
                refreshed,
                deployment_id=getattr(self.lineage, "deployment_id", "") or refreshed.deployment_id,
                trace_id=getattr(self.lineage, "trace_id", "") or refreshed.trace_id,
                parameter_set_id=getattr(self.lineage, "parameter_set_id", "") or refreshed.parameter_set_id,
            )
            self.lineage = refreshed
            for component in (
                self.snapshot_service,
                self.trade_logger,
                self.missed_logger,
                self.daily_builder,
                self.coordination_logger,
                self.order_logger,
                self.indicator_logger,
                self.filter_logger,
                self.orderbook_logger,
            ):
                if hasattr(component, "_lineage"):
                    component._lineage = refreshed
        except Exception as e:
            logger.warning("Failed to refresh swing instrumentation lineage: %s", e)

    def start(self) -> None:
        """Start background services (sidecar thread, post-exit backfill)."""
        if self._started:
            return

        # Enforce HMAC auth in paper/live — match stock bootstrap behavior
        env = os.environ.get("TRADING_MODE", os.environ.get("TRADING_ENV", "dev"))
        sidecar_forwarding_enabled = True
        if env in ("paper", "live") and self.sidecar is not None:
            hmac_secret = getattr(self.sidecar, "hmac_secret", b"")
            if not hmac_secret:
                sidecar_forwarding_enabled = False
                logger.warning(
                    "Sidecar forwarding disabled in %s mode: missing INSTRUMENTATION_HMAC_SECRET. "
                    "Local startup instrumentation will continue.",
                    env,
                )
        self._sidecar_forwarding_enabled = sidecar_forwarding_enabled

        try:
            if self.lineage is not None:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(self._write_startup_events_async())
                else:
                    loop.create_task(self._write_startup_events_async())
        except Exception as e:
            logger.warning("Startup instrumentation events failed: %s", e)

        try:
            if self.oms is not None:
                self._event_queue = self.oms.stream_all_events()
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    logger.warning("Swing OMS event instrumentation requires a running event loop")
                else:
                    self._event_task = loop.create_task(self._event_loop())
        except Exception as e:
            logger.warning("Swing OMS event subscription failed: %s", e)

        try:
            if self.sidecar is not None and sidecar_forwarding_enabled:
                self.sidecar.start()
        except Exception as e:
            self._sidecar_forwarding_enabled = False
            logger.warning("Sidecar start failed: %s", e)
        try:
            if self.post_exit_tracker is not None:
                self._backfill_stop.clear()
                self._backfill_thread = threading.Thread(
                    target=self._backfill_loop, daemon=True,
                    name="post-exit-backfill",
                )
                self._backfill_thread.start()
                logger.info("Post-exit backfill thread started")
        except Exception as e:
            logger.warning("Post-exit backfill thread start failed: %s", e)
        self._started = True
        logger.info("InstrumentationContext started")

    async def _write_startup_events_async(self) -> None:
        from libs.instrumentation.event_contract import write_startup_events
        from libs.instrumentation.startup_state import collect_startup_snapshot_state

        portfolio_rules_config = None
        try:
            if callable(self.get_applied_config):
                portfolio_rules_config = self.get_applied_config()
        except Exception as e:
            logger.warning("Failed to read applied portfolio rules config for startup snapshot: %s", e)
        strategy_ids = list(getattr(self.oms, "_family_strategy_ids", []) or [])
        allocation_state, portfolio_state, positions = await collect_startup_snapshot_state(
            self.oms,
            strategy_ids=strategy_ids,
            default_strategy_id="",
        )
        write_startup_events(
            self.data_dir,
            self.lineage,
            effective_config={"bot_id": self.bot_id, "data_dir": self.data_dir},
            allocation_state=allocation_state,
            portfolio_state=portfolio_state,
            positions=positions,
            portfolio_rules_config=portfolio_rules_config,
        )

    def stop(self) -> None:
        """Stop background services."""
        if not self._started:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            if self._async_stop_task is None or self._async_stop_task.done():
                self._async_stop_task = loop.create_task(self.stop_async())
            logger.info("InstrumentationContext async stop scheduled from sync stop")
            return
        self._cancel_event_task()
        self._stop_backfill_thread()
        try:
            self._write_daily_closeout_sync()
        except Exception as e:
            logger.warning("Daily reconciliation closeout failed: %s", e)
        self._stop_sidecar()
        self._started = False
        logger.info("InstrumentationContext stopped")

    async def stop_async(self) -> None:
        """Async stop path used by family coordinators with an active OMS."""
        if not self._started:
            return
        await self._stop_event_task_async()
        self._stop_backfill_thread()
        try:
            await self._write_daily_closeout_async()
        except Exception as e:
            logger.warning("Daily reconciliation closeout failed: %s", e)
        self._stop_sidecar()
        self._started = False
        logger.info("InstrumentationContext stopped")

    def _cancel_event_task(self) -> None:
        if self._event_task is not None and not self._event_task.done():
            self._event_task.cancel()
        self._event_task = None
        self._event_queue = None

    async def _stop_event_task_async(self) -> None:
        task = self._event_task
        self._event_task = None
        self._event_queue = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _event_loop(self) -> None:
        from libs.oms.models.events import OMSEventType

        while True:
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
                if event.event_type == OMSEventType.RISK_DECISION:
                    self._handle_risk_decision(event)
                elif event.event_type == OMSEventType.RISK_HALT:
                    self._handle_risk_halt(event)
            except Exception as e:
                logger.warning("Swing OMS instrumentation event handling failed: %s", e)

    def _handle_risk_decision(self, event) -> None:
        try:
            from libs.instrumentation.event_contract import append_jsonl_event, enrich_payload

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
                self.data_dir,
                "risk_decisions",
                "risk_decisions",
                payload,
            )
        except Exception as e:
            logger.warning("Failed to persist swing risk decision: %s", e)

    def _handle_risk_halt(self, event) -> None:
        try:
            from libs.instrumentation.event_contract import write_risk_halt_event

            payload = dict(event.payload or {})
            write_risk_halt_event(
                self.data_dir,
                payload.get("lineage") or self.lineage,
                reason=str(payload.get("reason") or "risk_halt"),
                strategy_id=str(event.strategy_id or payload.get("strategy_id") or ""),
                timestamp=getattr(event, "timestamp", None),
                details=payload,
            )
        except Exception as e:
            logger.warning("Failed to persist swing risk halt: %s", e)

    def _stop_backfill_thread(self) -> None:
        try:
            if self._backfill_thread is not None:
                self._backfill_stop.set()
                self._backfill_thread.join(timeout=10)
                self._backfill_thread = None
                logger.info("Post-exit backfill thread stopped")
        except Exception as e:
            logger.warning("Post-exit backfill thread stop failed: %s", e)

    def _save_daily_snapshot(self):
        if self.daily_builder is None:
            return "", None
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        snapshot = self.daily_builder.build(today)
        self.daily_builder.save(snapshot)
        return today, snapshot

    async def _daily_reconciliation_state_async(self) -> tuple[dict, dict]:
        from libs.oms.instrumentation._shared import plain

        oms = self.oms
        repo = getattr(oms, "_oms_repo", None)
        strategy_ids = list(getattr(oms, "_family_strategy_ids", []) or [])
        positions = []
        if repo is not None:
            try:
                if strategy_ids:
                    positions = plain(await repo.get_positions_for_strategies(strategy_ids))
                elif hasattr(repo, "get_all_positions"):
                    positions = plain(await repo.get_all_positions())
            except Exception as exc:
                logger.warning("Failed to read OMS positions for daily reconciliation: %s", exc)
        portfolio_state = plain(getattr(oms, "_portfolio_risk_state", {})) if oms is not None else {}
        if not isinstance(portfolio_state, dict):
            portfolio_state = {}
        account_state = {}
        provider = getattr(oms, "_account_state_provider", None)
        if callable(provider):
            try:
                account_state = plain(provider())
            except Exception as exc:
                logger.warning("Failed to read account state for daily reconciliation: %s", exc)
        if isinstance(account_state, dict):
            portfolio_state.update(account_state)
        portfolio_state["positions"] = positions
        strategy_risk = plain(getattr(oms, "_strategy_risk_states", {})) if oms is not None else {}
        if strategy_risk:
            portfolio_state["strategy_risk"] = strategy_risk
        allocation_targets = plain(getattr(oms, "_allocation_targets", {})) if oms is not None else {}
        raw_nav = account_state.get("raw_nav") or account_state.get("equity") or account_state.get("net_liquidation")
        allocation_state = {
            "source": "daily_closeout",
            "targets": allocation_targets,
            "raw_nav": raw_nav,
            "allocated_nav": account_state.get("allocated_nav") or raw_nav,
            "account_state": account_state,
        }
        return portfolio_state, allocation_state

    def _daily_reconciliation_state_sync(self) -> tuple[dict, dict]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._daily_reconciliation_state_async())

        raise RuntimeError(
            "Synchronous swing daily closeout cannot refresh OMS state while an "
            "event loop is running; call stop_async() or let stop() schedule it."
        )

    def _write_daily_closeout_sync(self) -> None:
        today, snapshot = self._save_daily_snapshot()
        if snapshot is None:
            return
        if self.lineage is not None:
            from libs.oms.instrumentation.lifecycle import write_daily_reconciliation

            portfolio_state, allocation_state = self._daily_reconciliation_state_sync()
            write_daily_reconciliation(
                self.data_dir,
                self.lineage,
                date_str=today,
                family_id=self.family_id,
                daily_snapshot=snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot),
                portfolio_state=portfolio_state,
                allocation_state=allocation_state,
            )
        logger.info("Daily reconciliation snapshot saved for %s", today)

    async def _write_daily_closeout_async(self) -> None:
        today, snapshot = self._save_daily_snapshot()
        if snapshot is None:
            return
        if self.lineage is not None:
            from libs.oms.instrumentation.lifecycle import write_daily_reconciliation

            portfolio_state, allocation_state = await self._daily_reconciliation_state_async()
            write_daily_reconciliation(
                self.data_dir,
                self.lineage,
                date_str=today,
                family_id=self.family_id,
                daily_snapshot=snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot),
                portfolio_state=portfolio_state,
                allocation_state=allocation_state,
            )
        logger.info("Daily reconciliation snapshot saved for %s", today)

    def _stop_sidecar(self) -> None:
        if not self._sidecar_forwarding_enabled:
            return
        try:
            if self.sidecar is not None:
                try:
                    self.sidecar.run_once()
                except Exception as e:
                    logger.warning("Sidecar final flush failed: %s", e)
                self.sidecar.stop()
        except Exception as e:
            logger.warning("Sidecar stop failed: %s", e)

    def _backfill_loop(self) -> None:
        """Periodically run post-exit backfill (every 30 min)."""
        while not self._backfill_stop.wait(timeout=1800):
            try:
                self.post_exit_tracker.run_backfill()  # type: ignore[union-attr]
            except Exception as e:
                logger.warning("Post-exit backfill error: %s", e)
