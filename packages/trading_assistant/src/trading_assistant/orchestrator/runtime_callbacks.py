"""Runtime callback-router registration."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable

from trading_assistant.comms.telegram_handlers import TelegramCallbackResponse, TelegramCallbackRouter
from trading_assistant.orchestrator.agent_preferences import WORKFLOW_ORDER
from trading_assistant.schemas.agent_preferences import (
    AgentPreferences,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
)

logger = logging.getLogger(__name__)


def register_runtime_callbacks(
    runtime: Any,
    *,
    save_agent_preferences: Callable[[AgentPreferences, Path], None],
) -> TelegramCallbackRouter | None:
    """Register Telegram settings, approval, and experiment callbacks."""
    callback_router = runtime.telegram_callback_router
    telegram_adapter = runtime.telegram_adapter
    telegram_renderer = runtime.telegram_renderer
    agent_runner = runtime.agent_runner
    agent_prefs_path = runtime.agent_prefs_path

    def _store_agent_preferences(preferences: AgentPreferences) -> None:
        agent_runner.update_preferences(preferences)
        save_agent_preferences(preferences, agent_prefs_path)
        runtime.collaborators["agent_preferences"] = agent_runner.get_preferences()

    def _render_agent_settings_response(
        scope: AgentWorkflow | str | None,
        *,
        edit_message: bool,
        answer: str,
    ) -> TelegramCallbackResponse:
        if telegram_renderer is None:
            return TelegramCallbackResponse(text="Telegram settings are unavailable.", answer=answer)
        text, keyboard = telegram_renderer.render_agent_settings(
            agent_runner.get_preferences_view(),
            scope=scope,
        )
        return TelegramCallbackResponse(
            text=text,
            keyboard=keyboard,
            answer=answer,
            edit_message=edit_message,
        )

    async def _settings_command(**kwargs) -> TelegramCallbackResponse:
        return _render_agent_settings_response(None, edit_message=False, answer="Agent settings")

    async def _settings_home(**kwargs) -> TelegramCallbackResponse:
        return _render_agent_settings_response(None, edit_message=True, answer="Agent settings")

    async def _settings_scope(scope_name: str) -> TelegramCallbackResponse:
        if scope_name == "global":
            return _render_agent_settings_response("global", edit_message=True, answer="Global provider")
        try:
            workflow = AgentWorkflow(scope_name)
        except ValueError as exc:
            return TelegramCallbackResponse(text=f"Unknown settings scope: {exc}", answer="Invalid scope")
        if workflow not in WORKFLOW_ORDER:
            return TelegramCallbackResponse(text="Unknown settings scope.", answer="Invalid scope")
        return _render_agent_settings_response(
            workflow,
            edit_message=True,
            answer=f"{scope_name} settings",
        )

    async def _settings_set(payload: str) -> TelegramCallbackResponse:
        scope_name, provider_name = (payload.split("|", 1) + [""])[:2]
        try:
            provider = AgentProvider(provider_name)
        except ValueError:
            return TelegramCallbackResponse(text="Unknown provider.", answer="Invalid provider")

        prefs = agent_runner.get_preferences()
        if scope_name == "global":
            prefs.default = AgentSelection(provider=provider)
            render_scope: AgentWorkflow | str | None = "global"
        else:
            try:
                workflow = AgentWorkflow(scope_name)
            except ValueError:
                return TelegramCallbackResponse(text="Unknown workflow.", answer="Invalid workflow")
            if workflow not in WORKFLOW_ORDER:
                return TelegramCallbackResponse(text="Unknown workflow.", answer="Invalid workflow")
            prefs.overrides[workflow] = AgentSelection(provider=provider)
            render_scope = workflow

        try:
            _store_agent_preferences(prefs)
        except ValueError as exc:
            return TelegramCallbackResponse(text=str(exc), answer="Provider unavailable")

        return _render_agent_settings_response(render_scope, edit_message=True, answer="Provider updated")

    async def _settings_clear(scope_name: str) -> TelegramCallbackResponse:
        try:
            workflow = AgentWorkflow(scope_name)
        except ValueError:
            return TelegramCallbackResponse(text="Unknown workflow.", answer="Invalid workflow")
        if workflow not in WORKFLOW_ORDER:
            return TelegramCallbackResponse(text="Unknown workflow.", answer="Invalid workflow")

        prefs = agent_runner.get_preferences()
        prefs.overrides.pop(workflow, None)
        _store_agent_preferences(prefs)
        return _render_agent_settings_response(workflow, edit_message=True, answer="Override cleared")

    if callback_router is not None:
        callback_router.register("cmd_settings", _settings_command)
        callback_router.register("agent_settings_home", _settings_home)
        callback_router.register("agent_settings_scope_", _settings_scope)
        callback_router.register("agent_settings_set_", _settings_set)
        callback_router.register("agent_settings_clear_", _settings_clear)

    approval_handler = runtime.approval_handler
    if approval_handler is not None:
        callback_router = callback_router or TelegramCallbackRouter()
        approval_tracker = runtime.approval_tracker
        deployment_monitor = runtime.deployment_monitor

        async def _on_approve_callback(request_id: str) -> str:
            response = await approval_handler.handle_approve(request_id)
            if deployment_monitor and "PR created:" in response:
                req = approval_tracker.get_by_id(request_id)
                if req and req.pr_url:
                    try:
                        dep_id = hashlib.sha256(f"dep-{request_id}".encode()).hexdigest()[:16]
                        deployment_monitor.create_deployment(
                            deployment_id=dep_id,
                            approval_request_id=request_id,
                            suggestion_id=req.suggestion_id,
                            pr_url=req.pr_url,
                            bot_id=req.bot_id,
                            param_changes=req.param_changes,
                            pr_number=0,
                        )
                        logger.info("Created deployment record %s for PR %s", dep_id, req.pr_url)
                    except Exception:
                        logger.warning("Failed to create deployment record for %s", request_id)
            return response

        async def _pending_command(**kwargs) -> str:
            pending = approval_tracker.get_pending()
            if not pending:
                return "No pending approval requests"
            lines = ["Pending approval requests:"]
            for request in pending:
                params = ", ".join(pc.get("param_name", "?") for pc in request.param_changes)
                lines.append(f"  [{request.request_id}] {request.bot_id}: {params}")
            return "\n".join(lines)

        callback_router.register("approve_suggestion_", _on_approve_callback)
        callback_router.register("reject_suggestion_", approval_handler.handle_reject)
        callback_router.register("detail_suggestion_", approval_handler.handle_detail)
        callback_router.register("cmd_pending", _pending_command)
        logger.info("Telegram approval callbacks registered")

    experiment_manager = runtime.experiment_manager
    if experiment_manager is not None and telegram_adapter is not None:
        callback_router = callback_router or TelegramCallbackRouter()

        async def _on_start_experiment(experiment_id: str) -> str:
            try:
                experiment_manager.activate_experiment(experiment_id)
                exp = experiment_manager.get_by_id(experiment_id)
                return f"Experiment started: {exp.title if exp else experiment_id}"
            except Exception as exc:
                return f"Failed to start experiment: {exc}"

        async def _on_cancel_experiment(experiment_id: str) -> str:
            try:
                experiment_manager.cancel_experiment(experiment_id)
                return f"Experiment cancelled: {experiment_id}"
            except Exception as exc:
                return f"Failed to cancel experiment: {exc}"

        callback_router.register("start_experiment_", _on_start_experiment)
        callback_router.register("cancel_experiment_", _on_cancel_experiment)
        logger.info("Experiment Telegram callbacks registered")

    if callback_router is not None:
        runtime.collaborators["telegram_callback_router"] = callback_router
        if telegram_adapter is not None:
            telegram_adapter.set_callback_router(callback_router)
    return callback_router
