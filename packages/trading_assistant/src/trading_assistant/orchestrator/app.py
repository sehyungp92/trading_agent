"""Orchestrator FastAPI entry point - wires all components together.

Run with: uvicorn trading_assistant.orchestrator.app:app --reload
For production, use create_app() factory to configure paths.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse

from trading_assistant.comms.dispatcher import NotificationDispatcher
from trading_assistant.orchestrator.agent_preferences import WORKFLOW_ORDER
from trading_assistant.orchestrator.config import AppConfig
from trading_assistant.orchestrator.event_validation import (
    QueueEventValidationError,
    normalize_queue_event,
)
from trading_assistant.orchestrator.input_sanitizer import InputSanitizer
from trading_assistant.orchestrator.runtime_validation import (
    runtime_config_summary,
    validate_auth_config,
)
from trading_assistant.orchestrator.runtime import (
    RuntimeBuildHooks,
    build_control_plane_runtime,
)
from trading_assistant.orchestrator.runtime_callbacks import register_runtime_callbacks
from trading_assistant.orchestrator.runtime_dispatch import wire_worker_dispatch
from trading_assistant.orchestrator.runtime_jobs import build_runtime_scheduler_wiring
from trading_assistant.orchestrator.runtime_scheduled_callbacks import (
    build_runtime_job_callbacks,
    expire_approvals_with_notification as _expire_approvals_with_notification,
    requires_monthly_outcome as _requires_monthly_outcome,
)
from trading_assistant.orchestrator.runtime_lifespan import build_runtime_lifespan
from trading_assistant.schemas.agent_preferences import (
    AgentPreferences,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
)
from trading_assistant.schemas.tasks import TaskStatus
from trading_assistant.schemas.notifications import (
    NotificationChannel,
    NotificationPreferences,
)

logger = logging.getLogger(__name__)

__all__ = [
    "app",
    "create_app",
    "_expire_approvals_with_notification",
    "_requires_monthly_outcome",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _normalize_queue_event(event: dict, allowed_bot_ids: set[str] | None = None) -> dict:
    try:
        return normalize_queue_event(event, allowed_bot_ids=allowed_bot_ids)
    except QueueEventValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _load_notification_prefs(prefs_path: Path) -> NotificationPreferences:
    """Load notification preferences from disk, or return defaults."""
    if prefs_path.exists():
        try:
            return NotificationPreferences(**json.loads(prefs_path.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Could not load notification prefs from %s, using defaults", prefs_path)
    return NotificationPreferences()


def _save_notification_prefs(prefs: NotificationPreferences, prefs_path: Path) -> None:
    """Persist notification preferences to disk."""
    prefs_path.parent.mkdir(parents=True, exist_ok=True)
    prefs_path.write_text(prefs.model_dump_json(indent=2), encoding="utf-8")


def _selection_from_env(provider_raw: str, model_raw: str = "") -> AgentSelection | None:
    provider_name = provider_raw.strip().lower()
    if not provider_name:
        return None
    try:
        provider = AgentProvider(provider_name)
    except ValueError:
        logger.warning("Ignoring invalid AGENT provider value: %r", provider_raw)
        return None
    model = model_raw.strip() or None
    return AgentSelection(provider=provider, model=model)


def _seed_agent_preferences(config: AppConfig) -> AgentPreferences:
    default_selection = _selection_from_env(
        config.agent_default_provider,
        config.agent_default_model,
    ) or AgentSelection(provider=AgentProvider.CODEX_PRO)

    overrides: dict[AgentWorkflow, AgentSelection | None] = {}
    workflow_env_map = (
        (AgentWorkflow.DAILY_ANALYSIS, config.daily_agent_provider, config.daily_agent_model),
        (AgentWorkflow.WEEKLY_ANALYSIS, config.weekly_agent_provider, config.weekly_agent_model),
        (
            AgentWorkflow.MONTHLY_VALIDATION,
            config.monthly_validation_agent_provider,
            config.monthly_validation_agent_model,
        ),
        (
            AgentWorkflow.MONTHLY_MODEL_REVIEW,
            config.monthly_model_review_agent_provider,
            config.monthly_model_review_agent_model,
        ),
        (
            AgentWorkflow.MONTHLY_VERIFIER,
            config.monthly_verifier_agent_provider,
            config.monthly_verifier_agent_model,
        ),
        (AgentWorkflow.TRIAGE, config.triage_agent_provider, config.triage_agent_model),
    )
    for workflow, provider_raw, model_raw in workflow_env_map:
        selection = _selection_from_env(provider_raw, model_raw)
        if selection is not None:
            overrides[workflow] = selection

    return AgentPreferences(default=default_selection, overrides=overrides)


def _load_agent_preferences(
    prefs_path: Path,
    config: AppConfig,
) -> AgentPreferences:
    """Load agent preferences from disk or build initial values from env."""
    if prefs_path.exists():
        try:
            return _active_agent_preferences(
                AgentPreferences(**json.loads(prefs_path.read_text(encoding="utf-8")))
            )
        except Exception:
            logger.warning("Could not load agent prefs from %s, using defaults", prefs_path)
    return _seed_agent_preferences(config)


def _save_agent_preferences(prefs: AgentPreferences, prefs_path: Path) -> None:
    """Persist agent preferences to disk."""
    prefs_path.parent.mkdir(parents=True, exist_ok=True)
    prefs_path.write_text(_active_agent_preferences(prefs).model_dump_json(indent=2), encoding="utf-8")


def _active_agent_preferences(prefs: AgentPreferences) -> AgentPreferences:
    """Drop persisted overrides for workflows that are no longer active."""
    active = set(WORKFLOW_ORDER)
    clean = prefs.model_copy(deep=True)
    clean.overrides = {
        workflow: selection
        for workflow, selection in clean.overrides.items()
        if workflow in active
    }
    clean.workflow_tuning = {
        workflow: tuning
        for workflow, tuning in clean.workflow_tuning.items()
        if workflow in active
    }
    return clean


def _unsupported_agent_workflows(prefs: AgentPreferences) -> list[str]:
    active = set(WORKFLOW_ORDER)
    return sorted({
        workflow.value
        for workflow in [*prefs.overrides.keys(), *prefs.workflow_tuning.keys()]
        if workflow not in active
    })


def _legacy_agent_workflows(payload: dict) -> list[str]:
    legacy: set[str] = set()
    for key in ("overrides", "workflow_tuning"):
        value = payload.get(key)
        if isinstance(value, dict) and "wfo" in value:
            legacy.add("wfo")
    return sorted(legacy)


def _register_channel_adapters(
    config: AppConfig,
    dispatcher: NotificationDispatcher,
) -> list:
    """Register communication channel adapters based on config. Returns adapters for lifecycle."""
    adapters = []

    if config.telegram_bot_token:
        from trading_assistant.comms.telegram_bot import TelegramBotAdapter, TelegramBotConfig
        from trading_assistant.comms.telegram_renderer import TelegramRenderer

        telegram = TelegramBotAdapter(config=TelegramBotConfig(
            token=config.telegram_bot_token,
            chat_id=config.telegram_chat_id,
        ))
        dispatcher.register_adapter(NotificationChannel.TELEGRAM, telegram)
        dispatcher.register_renderer(NotificationChannel.TELEGRAM, TelegramRenderer())
        adapters.append(telegram)
        logger.info("Telegram adapter registered (chat_id=%s)", config.telegram_chat_id)

    if config.discord_bot_token:
        from trading_assistant.comms.discord_bot import DiscordBotAdapter, DiscordBotConfig
        from trading_assistant.comms.renderer import PlainTextRenderer

        discord_adapter = DiscordBotAdapter(config=DiscordBotConfig(
            token=config.discord_bot_token,
            channel_id=int(config.discord_channel_id) if config.discord_channel_id else 0,
        ))
        dispatcher.register_adapter(NotificationChannel.DISCORD, discord_adapter)
        dispatcher.register_renderer(NotificationChannel.DISCORD, PlainTextRenderer())
        adapters.append(discord_adapter)
        logger.info("Discord adapter registered (channel_id=%s)", config.discord_channel_id)

    if config.smtp_host and config.smtp_user:
        from trading_assistant.comms.email_adapter import EmailAdapter, EmailConfig
        from trading_assistant.comms.renderer import PlainTextRenderer

        email = EmailAdapter(config=EmailConfig(
            smtp_host=config.smtp_host,
            smtp_port=config.smtp_port,
            username=config.smtp_user,
            password=config.smtp_pass,
            from_address=config.email_from,
        ))
        dispatcher.register_adapter(NotificationChannel.EMAIL, email)
        dispatcher.register_renderer(NotificationChannel.EMAIL, PlainTextRenderer())
        adapters.append(email)
        logger.info("Email adapter registered (smtp=%s)", config.smtp_host)

    return adapters


def _enforce_public_bind_requires_auth(config: AppConfig) -> None:
    """Backward-compatible wrapper for auth startup validation."""
    validate_auth_config(config)


def create_app(db_dir: str | None = None, config: AppConfig | None = None) -> FastAPI:
    """Factory function. Tests inject a temp directory for DB files."""
    runtime = build_control_plane_runtime(
        config=config,
        db_dir=db_dir,
        hooks=RuntimeBuildHooks(
            load_notification_preferences=_load_notification_prefs,
            save_notification_preferences=_save_notification_prefs,
            load_agent_preferences=_load_agent_preferences,
            register_channel_adapters=_register_channel_adapters,
        ),
    )
    config = runtime.config
    db_path = runtime.db_path
    memory_dir = runtime.memory_dir
    queue = runtime.queue
    registry = runtime.registry
    worker = runtime.worker
    event_stream = runtime.event_stream
    session_store = runtime.session_store
    subagent_mgr = runtime.subagent_mgr
    agent_runner = runtime.agent_runner
    brain = runtime.brain
    handlers = runtime.handlers
    prefs_path = runtime.prefs_path
    agent_prefs_path = runtime.agent_prefs_path
    callback_router = runtime.telegram_callback_router
    telegram_renderer = runtime.telegram_renderer
    vps_receiver = runtime.vps_receiver
    latency_tracker = runtime.latency_tracker
    structural_experiment_tracker = runtime.structural_experiment_tracker
    curated_dir = runtime.curated_dir
    raw_data_dir = runtime.raw_data_dir
    hypothesis_library = runtime.hypothesis_library
    data_dirs = runtime.data_dirs
    logger.info(
        "Resolved data dirs: db=%s raw=%s curated=%s legacy_curated=%s",
        db_path,
        raw_data_dir,
        curated_dir,
        data_dirs.legacy_curated_dir,
    )

    runtime_dispatch = wire_worker_dispatch(runtime)
    _reconcile_linked_subagent_tasks = runtime_dispatch.reconcile_linked_subagent_tasks
    callback_router = register_runtime_callbacks(
        runtime,
        save_agent_preferences=_save_agent_preferences,
    )

    scheduler_wiring = build_runtime_scheduler_wiring(
        runtime,
        build_runtime_job_callbacks(
            runtime,
            reconcile_linked_subagent_tasks=_reconcile_linked_subagent_tasks,
        ),
    )
    scheduled_job_specs = scheduler_wiring.job_specs

    lifespan = build_runtime_lifespan(runtime, scheduler_wiring)

    app = FastAPI(title="Trading Assistant Orchestrator", lifespan=lifespan)
    app.state.start_time = datetime.now(timezone.utc)

    @app.middleware("http")
    async def _require_api_key(request: Request, call_next):
        """Protect the control plane when ORCHESTRATOR_API_KEY is configured."""
        required_key = config.orchestrator_api_key
        if not required_key or request.url.path in {"/health", "/live", "/ready"}:
            return await call_next(request)

        provided_key = request.headers.get("X-Api-Key", "")
        if provided_key != required_key:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

        return await call_next(request)

    runtime.collaborators.update({
        "scheduled_job_specs": scheduled_job_specs,
        "reconcile_linked_subagent_tasks": _reconcile_linked_subagent_tasks,
        "telegram_callback_router": callback_router,
        "telegram_renderer": telegram_renderer,
        "vps_receiver": vps_receiver,
        "agent_preferences": agent_runner.get_preferences(),
    })
    runtime.attach_state(app)

    async def _evidence_pipeline_payload(
        *,
        relay_receiver: Any,
        relay_ready: bool,
        relay_consecutive_failures: int,
    ) -> dict:
        pipeline: dict[str, Any] = {
            "status": "ok",
            "assistant_status": "known",
            "relay": "configured" if config.relay_url else "disabled",
            "relay_ready": relay_ready,
            "relay_consecutive_failures": relay_consecutive_failures,
            "required_bot_ids": list(config.bot_ids),
            "missing_bot_ids": [],
            "queue_depth": None,
            "dead_letter_count": None,
            "oldest_pending_age_seconds": None,
        }
        if config.relay_url and not relay_ready:
            pipeline["status"] = "degraded"
            pipeline["missing_bot_ids"] = list(config.bot_ids)
        if config.relay_url and relay_receiver is None:
            pipeline["assistant_status"] = "unknown"

        try:
            pending = await queue.count_pending()
            dead_count = await queue.count_dead_letters()
            oldest_age = await queue.oldest_pending_age_seconds()
        except Exception:
            pipeline["status"] = "unknown"
            pipeline["assistant_status"] = "unknown"
            return pipeline

        pipeline["queue_depth"] = pending
        pipeline["dead_letter_count"] = dead_count
        pipeline["oldest_pending_age_seconds"] = oldest_age
        if dead_count > 0:
            pipeline["status"] = "degraded"
        elif oldest_age > 1800:
            pipeline["status"] = "warning"
        return pipeline

    async def _health_payload() -> dict:
        scheduler_obj = getattr(app.state, "scheduler", None) if hasattr(app.state, "scheduler") else None
        scheduler_ok = bool(scheduler_obj) and bool(getattr(scheduler_obj, "running", False))
        telegram_healthy = getattr(app.state, "telegram_healthy", True)
        relay_receiver = getattr(app.state, "vps_receiver", None)
        relay_ready = (
            bool(relay_receiver) and bool(getattr(relay_receiver, "is_healthy", False))
            if config.relay_url
            else True
        )
        relay_consecutive_failures = (
            int(getattr(relay_receiver, "consecutive_failures", 0))
            if relay_receiver is not None
            else 0
        )
        evidence_pipeline = await _evidence_pipeline_payload(
            relay_receiver=relay_receiver,
            relay_ready=relay_ready,
            relay_consecutive_failures=relay_consecutive_failures,
        )
        all_ok = scheduler_ok and telegram_healthy and relay_ready
        return {
            "status": "ok" if all_ok else "degraded",
            "scheduler": "running" if scheduler_ok else "unavailable",
            "telegram_healthy": telegram_healthy,
            "relay": "configured" if config.relay_url else "disabled",
            "relay_ready": relay_ready,
            "relay_consecutive_failures": relay_consecutive_failures,
            "evidence_pipeline": evidence_pipeline,
            **runtime_config_summary(config),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/health")
    async def health():
        return await _health_payload()

    @app.get("/live")
    async def live():
        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/ready")
    async def ready():
        payload = await _health_payload()
        return JSONResponse(
            status_code=200 if payload["status"] == "ok" else 503,
            content=payload,
        )

    @app.get("/metrics")
    async def metrics():
        """Operational metrics for monitoring the orchestrator itself."""
        from trading_assistant.schemas.orchestrator_metrics import BotLatencyStats, OrchestratorMetrics

        pending = await queue.count_pending()
        dead_count = await queue.count_dead_letters()
        oldest_age = await queue.oldest_pending_age_seconds()
        running = subagent_mgr.get_running()
        uptime = (datetime.now(timezone.utc) - app.state.start_time).total_seconds()
        error_rate = brain.get_error_rate_1h()

        agg = latency_tracker.get_aggregate_stats()
        per_bot = [
            BotLatencyStats(bot_id=bid, p50=s.p50, p95=s.p95, max=s.max, sample_count=s.sample_count)
            for bid, s in latency_tracker.get_all_stats().items()
        ]

        return OrchestratorMetrics(
            queue_depth=pending,
            dead_letter_count=dead_count,
            active_agents=len(running),
            error_rate_1h=error_rate,
            uptime_seconds=uptime,
            last_daily_analysis=brain.last_daily_analysis,
            last_weekly_analysis=brain.last_weekly_analysis,
            delivery_latency_p50=agg.p50,
            delivery_latency_p95=agg.p95,
            delivery_latency_max=agg.max,
            per_bot_latency=per_bot,
            oldest_pending_age_seconds=oldest_age,
        ).model_dump(mode="json")

    @app.post("/ingest")
    async def ingest_event(event: dict):
        """Direct event ingest - bypasses relay, useful for testing.

        Validates envelope (P1-4): bot_id must be in config.bot_ids or a
        known system source; payload must fit within 256 KB.
        """
        event = _normalize_queue_event(event, allowed_bot_ids=set(config.bot_ids))
        # Record latency for directly ingested events too
        ex_ts = event.get("exchange_timestamp", "")
        rx_ts = event.get("received_at", "")
        if ex_ts and rx_ts:
            latency_tracker.record(event.get("bot_id", "unknown"), ex_ts, rx_ts)
        inserted = await queue.enqueue(event)
        return {"inserted": inserted, "event_id": event.get("event_id")}

    @app.get("/events/pending")
    async def pending_events(limit: int = 20):
        return await queue.peek(limit=limit)

    @app.get("/tasks")
    async def list_tasks(status: str | None = None):
        if status:
            return [t.model_dump(mode="json") for t in await registry.list_by_status(TaskStatus(status))]
        return []

    @app.post("/process")
    async def trigger_processing(limit: int = 10):
        """Manually trigger event processing (for testing, normally done by scheduler)."""
        processed = await worker.process_batch(limit=limit)
        return {"processed": processed}

    @app.get("/events/stream")
    async def sse_stream(last_sequence: int = 0):
        from starlette.responses import StreamingResponse

        async def generate():
            # Send catch-up events
            for event in event_stream.get_recent(since_sequence=last_sequence):
                yield f"id: {event.sequence}\nevent: {event.event_type}\ndata: {event.model_dump_json()}\n\n"

            # Subscribe for new events
            q = event_stream.subscribe()
            try:
                while True:
                    event = await q.get()
                    yield f"id: {event.sequence}\nevent: {event.event_type}\ndata: {event.model_dump_json()}\n\n"
            finally:
                event_stream.unsubscribe(q)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/events/dead-letter")
    async def dead_letter_events(limit: int = 50):
        return await queue.get_dead_letters(limit=limit)

    @app.post("/events/dead-letter/{event_id}/reprocess")
    async def reprocess_dead_letter(event_id: str):
        success = await queue.reprocess_dead_letter(event_id)
        if not success:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Event not found in dead-letter queue")
        return {"status": "requeued", "event_id": event_id}

    @app.get("/agent/preferences")
    async def get_agent_preferences():
        return agent_runner.get_preferences_view().model_dump(mode="json")

    @app.put("/agent/preferences")
    async def update_agent_preferences(payload: dict):
        legacy = _legacy_agent_workflows(payload)
        if legacy:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported workflow override(s): {', '.join(legacy)}",
            )
        try:
            prefs = AgentPreferences.model_validate(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        unsupported = _unsupported_agent_workflows(prefs)
        if unsupported:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported workflow override(s): {', '.join(unsupported)}",
            )
        try:
            agent_runner.update_preferences(_active_agent_preferences(prefs))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        app.state.agent_preferences = agent_runner.get_preferences()
        _save_agent_preferences(app.state.agent_preferences, agent_prefs_path)
        return agent_runner.get_preferences_view().model_dump(mode="json")

    @app.get("/notifications/preferences")
    async def get_notification_preferences():
        return app.state.notification_preferences.model_dump()

    @app.put("/notifications/preferences")
    async def update_notification_preferences(prefs: NotificationPreferences):
        app.state.notification_preferences = prefs
        # Also update the handlers' reference
        handlers._notification_prefs = prefs
        # Persist to disk
        _save_notification_prefs(prefs, prefs_path)
        return prefs.model_dump()

    feedback_sanitizer = InputSanitizer()

    @app.post("/feedback")
    async def submit_feedback(body: dict):
        """Submit user feedback (approve/reject suggestions, corrections).

        All inbound text is passed through InputSanitizer (P0-3) before
        enqueueing, so prompt-injection patterns never reach the corrections
        store or downstream agent prompts.
        """
        text = body.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="'text' field is required")

        sanitized = feedback_sanitizer.sanitize(text, source="api")
        if not sanitized.safe:
            logger.warning("Blocked feedback (api): %s", sanitized.reason)
            raise HTTPException(
                status_code=400,
                detail=f"Feedback rejected: {sanitized.reason}",
            )

        import secrets

        event_time = _utc_now()
        feedback_event = {
            "event_type": "user_feedback",
            "bot_id": body.get("bot_id", "user"),
            "event_id": f"feedback-{event_time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(6)}",
            "payload": json.dumps({
                "text": sanitized.content,
                "report_id": body.get("report_id", "unknown"),
                "intent": sanitized.intent,
            }),
            "exchange_timestamp": event_time.isoformat(),
            "received_at": event_time.isoformat(),
        }
        inserted = await queue.enqueue(feedback_event)
        return {
            "inserted": inserted,
            "event_id": feedback_event["event_id"],
            "intent": sanitized.intent,
        }

    @app.get("/sessions")
    async def list_sessions(agent_type: str | None = None, date: str | None = None):
        return session_store.list_sessions(agent_type=agent_type, date=date)

    @app.get("/subagents")
    async def list_subagents():
        running = subagent_mgr.get_running()
        return [{"id": a.id, "agent_type": a.agent_type, "started_at": a.started_at.isoformat()} for a in running]

    @app.post("/subagents/{agent_id}/cancel")
    async def cancel_subagent(agent_id: str):
        success = await subagent_mgr.cancel(agent_id)
        if not success:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Agent not found or not running")
        return {"status": "cancelled", "agent_id": agent_id}

    @app.get("/learning/dashboard")
    async def learning_dashboard():
        """Learning system dashboard - ground truth, lessons, experiments, scorecard."""
        from trading_assistant.skills.learning_ledger import LearningLedger
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings_dir = memory_dir / "findings"
        ledger = LearningLedger(findings_dir)
        scorer = SuggestionScorer(findings_dir)

        trend = ledger.get_trend(weeks=12)
        lessons = ledger.get_lessons(weeks=4)
        latest = ledger.get_latest()
        scorecard = scorer.compute_scorecard()

        active_hypotheses = []
        try:
            records = hypothesis_library.get_active()
            active_hypotheses = [
                {
                    "hypothesis_id": getattr(h, "hypothesis_id", ""),
                    "title": getattr(h, "title", ""),
                    "category": getattr(h, "category", ""),
                    "effectiveness": getattr(h, "effectiveness", 0.0),
                }
                for h in records
            ]
        except Exception:
            pass

        active_experiments = []
        try:
            exps = structural_experiment_tracker.get_active_experiments()
            active_experiments = [
                {
                    "experiment_id": getattr(e, "experiment_id", ""),
                    "title": getattr(e, "title", ""),
                    "bot_id": getattr(e, "bot_id", ""),
                    "status": getattr(e, "status", ""),
                }
                for e in exps
            ]
        except Exception:
            pass

        prediction_accuracy = {}
        try:
            from trading_assistant.analysis.context_builder import ContextBuilder
            ctx = ContextBuilder(memory_dir)
            pkg = ctx.base_package()
            prediction_accuracy = pkg.data.get("prediction_accuracy_by_metric", {})
        except Exception:
            pass

        return {
            "ground_truth_trend": trend,
            "recent_lessons": lessons,
            "active_hypotheses": active_hypotheses,
            "active_experiments": active_experiments,
            "category_scorecard": scorecard.model_dump(mode="json"),
            "prediction_accuracy": prediction_accuracy,
            "net_improvement": latest.net_improvement if latest else None,
            "latest_entry": latest.model_dump(mode="json") if latest else None,
        }

    return app


# Default app instance for `uvicorn trading_assistant.orchestrator.app:app`
app = create_app()
