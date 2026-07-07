"""Runtime-owned FastAPI lifespan wiring."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from trading_assistant.orchestrator.catchup import StartupCatchup, bootstrap_run_store_from_history
from trading_assistant.orchestrator.runtime_jobs import RuntimeSchedulerWiring
from trading_assistant.orchestrator.runtime_scheduler import create_scheduler
from trading_assistant.schemas.notifications import NotificationPayload, NotificationPriority

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def build_runtime_lifespan(runtime: Any, scheduler: RuntimeSchedulerWiring):
    """Build the app lifespan around runtime-owned collaborators."""

    @asynccontextmanager
    async def lifespan(app: Any):
        from filelock import FileLock, Timeout as LockTimeout

        config = runtime.config
        db_path = runtime.db_path
        queue = runtime.queue
        registry = runtime.registry
        scheduled_run_store = runtime.scheduled_run_store
        channel_adapters = runtime.channel_adapters
        telegram_adapter = runtime.telegram_adapter
        dispatcher = runtime.dispatcher
        notification_prefs = runtime.notification_preferences
        audit_consumer = runtime.audit_consumer
        event_stream = runtime.event_stream
        vps_receiver = runtime.vps_receiver
        subagent_mgr = runtime.subagent_mgr
        run_index = runtime.run_index

        lock_path = db_path / ".orchestrator.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        process_lock = FileLock(str(lock_path))
        try:
            process_lock.acquire(timeout=0)
        except LockTimeout as exc:
            raise RuntimeError(
                f"Another orchestrator is already running against {db_path}. "
                f"Lockfile: {lock_path}. Refusing to start a second process; "
                "this system is single-writer by design."
            ) from exc
        app.state.process_lock = process_lock
        try:
            await queue.initialize()
            await registry.initialize()
            await scheduled_run_store.initialize()

            try:
                recovered = await queue.recover_stale(timeout_seconds=300)
                if recovered:
                    logger.info("Startup: recovered %d stale events", recovered)
            except Exception:
                logger.warning("Startup recover_stale failed", exc_info=True)

            for bid, cfg in (config.bot_configs or {}).items():
                try:
                    from zoneinfo import ZoneInfo

                    ZoneInfo(cfg.timezone)
                except Exception as exc:
                    raise ValueError(f"Invalid timezone '{cfg.timezone}' for bot '{bid}'") from exc

            for adapter in channel_adapters:
                try:
                    await adapter.start()
                except Exception:
                    logger.warning("Failed to start %s adapter", type(adapter).__name__)

            app.state.telegram_healthy = True
            if telegram_adapter is not None and telegram_adapter._callback_router is not None:
                try:
                    await telegram_adapter.start_polling()
                except Exception:
                    logger.error("Failed to start Telegram polling; feedback loop disabled", exc_info=True)
                    app.state.telegram_healthy = False
                    try:
                        await dispatcher.dispatch(
                            NotificationPayload(
                                notification_type="telegram_polling_failure",
                                priority=NotificationPriority.CRITICAL,
                                title="Telegram polling offline",
                                body=(
                                    "Telegram polling failed at startup; suggestion "
                                    "approve/reject feedback is offline until restart."
                                ),
                            ),
                            notification_prefs,
                            _utc_now().hour,
                        )
                    except Exception:
                        logger.exception("Failed to dispatch Telegram degradation alert")

            audit_consumer.start(event_stream)

            if vps_receiver:
                try:
                    await vps_receiver.drain()
                except Exception:
                    logger.warning("Startup drain failed; will retry via scheduler")

            run_history_path = db_path / "data" / "run_history.jsonl"
            try:
                store_was_empty = await scheduled_run_store.is_empty()
                seeded_baseline = (
                    await bootstrap_run_store_from_history(
                        scheduled_run_store,
                        scheduler.job_specs,
                        run_history_path,
                    )
                    if store_was_empty
                    else None
                )
                if store_was_empty and await scheduled_run_store.get_baseline() is None:
                    await scheduled_run_store.set_baseline(seeded_baseline or _utc_now())

                catchup = StartupCatchup(
                    job_specs=scheduler.job_specs,
                    run_store=scheduled_run_store,
                )
                for occurrence in await catchup.build_plan(now=_utc_now()):
                    logger.info(
                        "Startup catch-up: running %s (%s @ %s)",
                        occurrence.spec.job_key,
                        occurrence.spec.scope_key,
                        occurrence.scheduled_for.isoformat(),
                    )
                    try:
                        await asyncio.wait_for(
                            scheduler.runner.run(
                                occurrence.spec,
                                scheduled_for=occurrence.scheduled_for,
                            ),
                            timeout=600,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Scheduled catch-up timed out after 600s for %s (%s); moving on",
                            occurrence.spec.job_key,
                            occurrence.spec.scope_key,
                        )
                    except Exception:
                        logger.warning(
                            "Scheduled catch-up failed for %s (%s)",
                            occurrence.spec.job_key,
                            occurrence.spec.scope_key,
                            exc_info=True,
                        )
            except Exception:
                logger.warning("Startup catch-up failed", exc_info=True)

            try:
                app.state.scheduler = create_scheduler(scheduler.scheduler_jobs)
            except Exception:
                logger.error("Failed to create scheduler", exc_info=True)
                app.state.scheduler = None
                if config.is_production:
                    raise RuntimeError("Scheduler creation failed in production") from None

        except Exception as exc:
            logger.critical("Startup failed: %s", exc, exc_info=True)
            try:
                await dispatcher.dispatch(
                    NotificationPayload(
                        notification_type="system_alert",
                        priority=NotificationPriority.CRITICAL,
                        title="Trading Assistant Startup Failed",
                        body=f"Startup failure: {exc}",
                    ),
                    notification_prefs,
                    0,
                )
            except Exception:
                pass
            raise

        try:
            yield
        finally:
            await audit_consumer.stop()
            if hasattr(app.state, "scheduler") and app.state.scheduler:
                app.state.scheduler.shutdown(wait=False)
            await subagent_mgr.cancel_all()
            for adapter in channel_adapters:
                try:
                    await adapter.stop()
                except Exception:
                    logger.warning("Failed to stop %s adapter", type(adapter).__name__)
            await queue.close()
            await registry.close()
            await scheduled_run_store.close()
            run_index.close()
            try:
                process_lock.release()
            except Exception:
                logger.warning("Failed to release orchestrator lockfile", exc_info=True)

    return lifespan
