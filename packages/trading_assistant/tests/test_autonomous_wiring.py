# tests/test_autonomous_wiring.py
"""Tests for autonomous pipeline wiring into handlers, app, and config."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from trading_assistant.orchestrator.config import AppConfig


class TestAppConfig:
    def test_autonomous_enabled_from_env(self):
        with patch.dict(os.environ, {"AUTONOMOUS_ENABLED": "true"}):
            config = AppConfig.from_env()
            assert config.autonomous_enabled is True

    def test_autonomous_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            config = AppConfig.from_env()
            assert config.autonomous_enabled is False

    def test_bot_config_dir_from_env(self):
        with patch.dict(os.environ, {"BOT_CONFIG_DIR": "/custom/configs"}):
            config = AppConfig.from_env()
            assert config.bot_config_dir == "/custom/configs"


class TestHandlersWiring:
    def test_handlers_accepts_autonomous_pipeline(self, tmp_path: Path):
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.schemas.notifications import NotificationPreferences

        runner = MagicMock()
        handlers_obj = None
        from trading_assistant.orchestrator.handlers import Handlers
        handlers_obj = Handlers(
            agent_runner=runner,
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=[],
            autonomous_pipeline=MagicMock(),
        )
        assert handlers_obj._autonomous_pipeline is not None

    def test_handlers_without_autonomous_pipeline(self, tmp_path: Path):
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.schemas.notifications import NotificationPreferences

        runner = MagicMock()
        from trading_assistant.orchestrator.handlers import Handlers
        handlers_obj = Handlers(
            agent_runner=runner,
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=[],
        )
        assert handlers_obj._autonomous_pipeline is None

    @pytest.mark.asyncio
    async def test_pipeline_called_with_suggestion_ids(self, tmp_path: Path):
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.schemas.notifications import NotificationPreferences
        from trading_assistant.orchestrator.handlers import Handlers

        pipeline = MagicMock()
        pipeline.process_new_suggestions = AsyncMock(return_value=[])

        runner = MagicMock()
        handlers_obj = Handlers(
            agent_runner=runner,
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=[],
            autonomous_pipeline=pipeline,
        )
        await handlers_obj._run_autonomous_pipeline({"s1": "test", "s2": "test2"}, "run1")
        pipeline.process_new_suggestions.assert_called_once_with(
            suggestion_ids=["s1", "s2"],
            run_id="run1",
        )

    @pytest.mark.asyncio
    async def test_pipeline_not_called_when_none(self, tmp_path: Path):
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.schemas.notifications import NotificationPreferences
        from trading_assistant.orchestrator.handlers import Handlers

        runner = MagicMock()
        handlers_obj = Handlers(
            agent_runner=runner,
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=[],
        )
        # Should be a no-op
        await handlers_obj._run_autonomous_pipeline({"s1": "test"}, "run1")


class TestRecordSuggestionsConfidence:
    def test_confidence_preserved_in_record(self, tmp_path: Path):
        """Verify that _record_suggestions carries confidence to SuggestionRecord."""
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.schemas.notifications import NotificationPreferences
        from trading_assistant.schemas.strategy_suggestions import StrategySuggestion, SuggestionTier
        from trading_assistant.skills.suggestion_tracker import SuggestionTracker
        from trading_assistant.orchestrator.handlers import Handlers

        sug_tracker = SuggestionTracker(store_dir=tmp_path / "findings")
        runner = MagicMock()
        handlers_obj = Handlers(
            agent_runner=runner,
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=sug_tracker,
        )

        suggestion = StrategySuggestion(
            tier=SuggestionTier.PARAMETER,
            bot_id="bot1",
            title="Increase X to 0.7",
            description="test",
            confidence=0.85,
        )
        ids = handlers_obj._record_suggestions([suggestion], "run1")
        assert len(ids) == 1
        all_recs = sug_tracker.load_all()
        assert all_recs[0].get("confidence") == 0.85


class TestAppWiring:
    def test_feature_flag_off_no_autonomous(self, tmp_path: Path):
        """When AUTONOMOUS_ENABLED is false, autonomous_pipeline should be None."""
        config = AppConfig(
            autonomous_enabled=False,
            allow_unauthenticated_local=True,
        )
        from trading_assistant.orchestrator.app import create_app
        app = create_app(db_dir=str(tmp_path), config=config)
        assert app.state.autonomous_pipeline is None

    def test_monthly_approval_gated_creates_shared_approval_without_autonomous(
        self,
        tmp_path: Path,
    ):
        """Monthly approval-gated mode gets approval plumbing without autonomous processing."""
        config = AppConfig(
            autonomous_enabled=False,
            monthly_validation_mode="approval_gated",
            allow_unauthenticated_local=True,
        )
        from trading_assistant.orchestrator.app import create_app

        app = create_app(db_dir=str(tmp_path), config=config)

        assert app.state.autonomous_pipeline is None
        assert app.state.approval_tracker is not None
        assert app.state.approval_handler is not None

    def test_monthly_approval_callbacks_register_without_autonomous(
        self,
        tmp_path: Path,
    ):
        """Telegram approval callbacks are shared infrastructure, not autonomous-only."""
        config = AppConfig(
            autonomous_enabled=False,
            monthly_validation_mode="approval_gated",
            telegram_bot_token="test_token",
            telegram_chat_id="123",
            allow_unauthenticated_local=True,
        )
        from trading_assistant.orchestrator.app import create_app

        app = create_app(db_dir=str(tmp_path), config=config)
        router = app.state.telegram_callback_router

        assert router is not None
        assert "approve_suggestion_" in router.handlers
        assert "reject_suggestion_" in router.handlers
        assert "cmd_pending" in router.handlers

    def test_feature_flag_on_creates_components(self, tmp_path: Path):
        """When AUTONOMOUS_ENABLED is true, autonomous components are created."""
        # Create bot config files
        cfg_dir = tmp_path / "data" / "bot_configs"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "test.yaml").write_text(yaml.dump({
            "bot_id": "test",
            "parameters": [{
                "param_name": "x",
                "param_type": "YAML_FIELD",
                "file_path": "config.yaml",
                "yaml_key": "x",
                "current_value": 0.5,
                "value_type": "float",
            }],
        }), encoding="utf-8")

        config = AppConfig(
            autonomous_enabled=True,
            bot_config_dir=str(cfg_dir),
            allow_unauthenticated_local=True,
        )
        from trading_assistant.orchestrator.app import create_app
        app = create_app(db_dir=str(tmp_path), config=config)
        assert app.state.autonomous_pipeline is not None
        assert app.state.approval_tracker is not None
        assert app.state.approval_handler is not None

    def test_telegram_bot_injected_into_pipeline(self, tmp_path: Path):
        """When Telegram token is set, the adapter is injected into AutonomousPipeline."""
        cfg_dir = tmp_path / "data" / "bot_configs"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "test.yaml").write_text(yaml.dump({
            "bot_id": "test",
            "parameters": [{
                "param_name": "x",
                "param_type": "YAML_FIELD",
                "file_path": "config.yaml",
                "yaml_key": "x",
                "current_value": 0.5,
                "value_type": "float",
            }],
        }), encoding="utf-8")

        config = AppConfig(
            autonomous_enabled=True,
            bot_config_dir=str(cfg_dir),
            telegram_bot_token="test_token",
            telegram_chat_id="123",
            allow_unauthenticated_local=True,
        )
        from trading_assistant.orchestrator.app import create_app
        app = create_app(db_dir=str(tmp_path), config=config)
        pipeline = app.state.autonomous_pipeline
        assert pipeline._telegram_bot is not None
        from trading_assistant.comms.telegram_bot import TelegramBotAdapter
        assert isinstance(pipeline._telegram_bot, TelegramBotAdapter)

    def test_telegram_bot_none_when_no_token(self, tmp_path: Path):
        """When Telegram token is not set, telegram_bot is None in pipeline."""
        cfg_dir = tmp_path / "data" / "bot_configs"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "test.yaml").write_text(yaml.dump({
            "bot_id": "test",
            "parameters": [{
                "param_name": "x",
                "param_type": "YAML_FIELD",
                "file_path": "config.yaml",
                "yaml_key": "x",
                "current_value": 0.5,
                "value_type": "float",
            }],
        }), encoding="utf-8")

        config = AppConfig(
            autonomous_enabled=True,
            bot_config_dir=str(cfg_dir),
            allow_unauthenticated_local=True,
        )
        from trading_assistant.orchestrator.app import create_app
        app = create_app(db_dir=str(tmp_path), config=config)
        pipeline = app.state.autonomous_pipeline
        assert pipeline._telegram_bot is None

    def test_callback_router_connected_to_adapter(self, tmp_path: Path):
        """When both autonomous and Telegram are enabled, callback router is wired."""
        cfg_dir = tmp_path / "data" / "bot_configs"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "test.yaml").write_text(yaml.dump({
            "bot_id": "test",
            "parameters": [{
                "param_name": "x",
                "param_type": "YAML_FIELD",
                "file_path": "config.yaml",
                "yaml_key": "x",
                "current_value": 0.5,
                "value_type": "float",
            }],
        }), encoding="utf-8")

        config = AppConfig(
            autonomous_enabled=True,
            bot_config_dir=str(cfg_dir),
            telegram_bot_token="test_token",
            telegram_chat_id="123",
            allow_unauthenticated_local=True,
        )
        from trading_assistant.orchestrator.app import create_app
        app = create_app(db_dir=str(tmp_path), config=config)
        pipeline = app.state.autonomous_pipeline
        adapter = pipeline._telegram_bot
        assert adapter._callback_router is not None
