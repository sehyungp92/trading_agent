# tests/test_app_wiring.py
"""Tests for app.py wiring: channel adapters, config, scanner, prefs persistence."""
import json
import os
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from trading_assistant.orchestrator.app import (
    _load_agent_preferences,
    _load_notification_prefs,
    _register_channel_adapters,
    _save_agent_preferences,
    _save_notification_prefs,
    create_app,
)
from trading_assistant.orchestrator.config import AppConfig
from trading_assistant.orchestrator import runtime_experiments, runtime_scheduled_callbacks
from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.paths import memory_root
from trading_assistant.schemas.agent_preferences import (
    AgentPreferences,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
)
from trading_assistant.schemas.notifications import (
    ChannelConfig,
    NotificationChannel,
    NotificationPreferences,
)
from trading_assistant.comms.dispatcher import NotificationDispatcher


def _local_config(**kwargs) -> AppConfig:
    return AppConfig(allow_unauthenticated_local=True, **kwargs)


class TestAppConfigFromEnv:
    def test_command_args_parse_from_env(self):
        env = {
            "CLAUDE_COMMAND_ARGS": '["--","claude"]',
            "CODEX_COMMAND_ARGS": '["--","codex"]',
            "MONTHLY_BACKTEST_COMMAND": '["python","-m","trading_assistant.skills.backtest_runner_client"]',
            "MONTHLY_OPTIMIZER_SEQUENCE_ENABLED": "false",
            "MONTHLY_WORKFLOW_CONTRACT_PATH": "MONTHLY_OPTIMIZER_WORKFLOW.md",
            "MONTHLY_WORKFLOW_CONTRACT_VERSION": "monthly_optimizer_workflow_contract_v1",
            "MONTHLY_STRATEGY_PLUGIN_CONTRACT_PATH": "strategy_plugin_contract.json",
            "LEARNING_REVIEW_MODE": "llm_review",
            "LEARNING_REVIEW_DISABLED_WORKFLOWS": "daily_analysis,monthly_validation",
        }

        with patch.dict(os.environ, env, clear=False):
            config = AppConfig.from_env()

        assert config.claude_command_args == ["--", "claude"]
        assert config.codex_command_args == ["--", "codex"]
        assert config.monthly_backtest_command == ["python", "-m", "trading_assistant.skills.backtest_runner_client"]
        assert config.monthly_optimizer_sequence_enabled is False
        assert config.monthly_workflow_contract_path == "MONTHLY_OPTIMIZER_WORKFLOW.md"
        assert config.monthly_workflow_contract_version == "monthly_optimizer_workflow_contract_v1"
        assert config.monthly_strategy_plugin_contract_path == "strategy_plugin_contract.json"
        assert config.learning_review_mode == "llm_review"
        assert config.learning_review_disabled_workflows == ["daily_analysis", "monthly_validation"]

    def test_invalid_command_args_env_falls_back_to_empty_list(self):
        env = {
            "CLAUDE_COMMAND_ARGS": '{"not":"a-list"}',
            "CODEX_COMMAND_ARGS": "not-json",
        }

        with patch.dict(os.environ, env, clear=False):
            config = AppConfig.from_env()

        assert config.claude_command_args == []
        assert config.codex_command_args == []

    def test_loads_dotenv_defaults_when_process_env_missing(self, tmp_path):
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text(
            "BOT_IDS=dotenv_bot\nCLAUDE_COMMAND=from-dotenv\n",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {}, clear=True):
            config = AppConfig.from_env(dotenv_path=dotenv_path)

        assert config.bot_ids == ["dotenv_bot"]
        assert config.claude_command == "from-dotenv"

    def test_process_env_overrides_dotenv(self, tmp_path):
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text("BOT_IDS=dotenv_bot\n", encoding="utf-8")

        with patch.dict(os.environ, {"BOT_IDS": "env_bot"}, clear=True):
            config = AppConfig.from_env(dotenv_path=dotenv_path)

        assert config.bot_ids == ["env_bot"]

    def test_allow_unauthenticated_local_from_env(self):
        with patch.dict(os.environ, {"ALLOW_UNAUTHENTICATED_LOCAL": "true"}, clear=False):
            config = AppConfig.from_env()

        assert config.allow_unauthenticated_local is True


class TestChannelAdapterRegistration:
    def test_telegram_registered_when_token_set(self):
        config = AppConfig(telegram_bot_token="fake-token", telegram_chat_id="12345")
        dispatcher = NotificationDispatcher()
        adapters = _register_channel_adapters(config, dispatcher)
        assert NotificationChannel.TELEGRAM in dispatcher.adapters
        assert len(adapters) == 1

    def test_discord_registered_when_token_set(self):
        config = AppConfig(discord_bot_token="fake-token", discord_channel_id="99999")
        dispatcher = NotificationDispatcher()
        adapters = _register_channel_adapters(config, dispatcher)
        assert NotificationChannel.DISCORD in dispatcher.adapters
        assert len(adapters) == 1

    def test_email_registered_when_smtp_set(self):
        config = AppConfig(
            smtp_host="smtp.test.com", smtp_user="user", smtp_pass="pass",
            email_from="from@test.com",
        )
        dispatcher = NotificationDispatcher()
        adapters = _register_channel_adapters(config, dispatcher)
        assert NotificationChannel.EMAIL in dispatcher.adapters
        assert len(adapters) == 1

    def test_no_adapters_when_no_config(self):
        config = AppConfig()
        dispatcher = NotificationDispatcher()
        adapters = _register_channel_adapters(config, dispatcher)
        assert len(dispatcher.adapters) == 0
        assert len(adapters) == 0

    def test_all_adapters_registered(self):
        config = AppConfig(
            telegram_bot_token="tg-token", telegram_chat_id="123",
            discord_bot_token="dc-token", discord_channel_id="456",
            smtp_host="smtp.test.com", smtp_user="user", smtp_pass="pass",
            email_from="from@test.com",
        )
        dispatcher = NotificationDispatcher()
        adapters = _register_channel_adapters(config, dispatcher)
        assert len(adapters) == 3
        assert NotificationChannel.TELEGRAM in dispatcher.adapters
        assert NotificationChannel.DISCORD in dispatcher.adapters
        assert NotificationChannel.EMAIL in dispatcher.adapters


class TestNotificationPreferencesPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        prefs_path = tmp_path / "data" / "notification_prefs.json"
        prefs = NotificationPreferences(channels=[
            ChannelConfig(channel=NotificationChannel.TELEGRAM, enabled=True, chat_id="123"),
        ])
        _save_notification_prefs(prefs, prefs_path)
        assert prefs_path.exists()
        loaded = _load_notification_prefs(prefs_path)
        assert len(loaded.channels) == 1
        assert loaded.channels[0].channel == NotificationChannel.TELEGRAM
        assert loaded.channels[0].chat_id == "123"

    def test_load_returns_defaults_when_file_missing(self, tmp_path):
        prefs_path = tmp_path / "nonexistent.json"
        loaded = _load_notification_prefs(prefs_path)
        assert loaded.channels == []

    def test_load_returns_defaults_on_corrupt_file(self, tmp_path):
        prefs_path = tmp_path / "prefs.json"
        prefs_path.write_text("not valid json{{{", encoding="utf-8")
        loaded = _load_notification_prefs(prefs_path)
        assert loaded.channels == []

    def test_save_creates_parent_dirs(self, tmp_path):
        prefs_path = tmp_path / "deep" / "nested" / "prefs.json"
        prefs = NotificationPreferences()
        _save_notification_prefs(prefs, prefs_path)
        assert prefs_path.exists()


class TestAgentPreferencesPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        prefs_path = tmp_path / "data" / "agent_preferences.json"
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO, model="gpt-5.4"),
            overrides={
                AgentWorkflow.TRIAGE: AgentSelection(
                    provider=AgentProvider.OPENROUTER,
                    model="minimax/minimax-m2.5",
                ),
            },
        )

        _save_agent_preferences(prefs, prefs_path)
        loaded = _load_agent_preferences(prefs_path, AppConfig())

        assert loaded.default.provider == AgentProvider.CODEX_PRO
        assert loaded.default.model == "gpt-5.4"
        assert loaded.overrides[AgentWorkflow.TRIAGE].provider == AgentProvider.OPENROUTER

    def test_load_seeds_from_env_when_file_missing(self, tmp_path):
        prefs_path = tmp_path / "missing.json"
        config = AppConfig(
            agent_default_provider="codex_pro",
            daily_agent_provider="openrouter",
            daily_agent_model="minimax/minimax-m2.5",
        )

        loaded = _load_agent_preferences(prefs_path, config)

        assert loaded.default.provider == AgentProvider.CODEX_PRO
        assert loaded.overrides[AgentWorkflow.DAILY_ANALYSIS].provider == AgentProvider.OPENROUTER

    def test_load_falls_back_to_seeded_defaults_on_corrupt_file(self, tmp_path):
        prefs_path = tmp_path / "prefs.json"
        prefs_path.write_text("not valid json", encoding="utf-8")
        config = AppConfig(agent_default_provider="claude_max", weekly_agent_provider="codex_pro")

        loaded = _load_agent_preferences(prefs_path, config)

        assert loaded.default.provider == AgentProvider.CLAUDE_MAX
        assert loaded.overrides[AgentWorkflow.WEEKLY_ANALYSIS].provider == AgentProvider.CODEX_PRO


class TestCreateAppWithConfig:
    def test_bot_ids_passed_to_handlers(self, tmp_path):
        config = _local_config(bot_ids=["bot_x", "bot_y"])
        app = create_app(db_dir=str(tmp_path), config=config)
        assert app.state.handlers._bots == ["bot_x", "bot_y"]

    def test_empty_config_still_creates_app(self, tmp_path):
        config = _local_config()
        app = create_app(db_dir=str(tmp_path), config=config)
        assert app.state.handlers._bots == []
        assert len(app.state.dispatcher.adapters) == 0

    def test_config_exposed_on_app_state(self, tmp_path):
        config = _local_config(bot_ids=["bot_z"])
        app = create_app(db_dir=str(tmp_path), config=config)
        assert app.state.config.bot_ids == ["bot_z"]

    def test_monthly_support_jobs_only_schedule_when_monthly_enabled(self, tmp_path):
        disabled_app = create_app(db_dir=str(tmp_path / "off"), config=_local_config(bot_ids=["bot1"]))
        disabled_job_keys = {spec.job_key for spec in disabled_app.state.scheduled_job_specs}
        assert "lineage_audit" not in disabled_job_keys
        assert "market_data_sync" not in disabled_job_keys
        assert "monthly_validation" not in disabled_job_keys

        enabled_app = create_app(
            db_dir=str(tmp_path / "on"),
            config=_local_config(
                bot_ids=["bot1"],
                monthly_validation_mode="shadow",
                monthly_backtest_command=["python", "runner.py", "{manifest}"],
                monthly_workflow_contract_path="MONTHLY_OPTIMIZER_WORKFLOW.md",
                monthly_workflow_contract_version="monthly_optimizer_workflow_contract_v1",
                monthly_strategy_plugin_contract_path="strategy_plugin_contract.json",
                backtest_max_parallel_strategies=3,
            ),
        )
        enabled_job_keys = {spec.job_key for spec in enabled_app.state.scheduled_job_specs}
        assert "lineage_audit" in enabled_job_keys
        assert "market_data_sync" in enabled_job_keys
        assert "monthly_validation" in enabled_job_keys
        assert enabled_app.state.handlers._monthly_optimizer_sequence_enabled is True
        assert enabled_app.state.handlers._monthly_backtest_command == ["python", "runner.py", "{manifest}"]
        assert enabled_app.state.handlers._monthly_workflow_contract_path == "MONTHLY_OPTIMIZER_WORKFLOW.md"
        assert (
            enabled_app.state.handlers._monthly_workflow_contract_version
            == "monthly_optimizer_workflow_contract_v1"
        )
        assert (
            enabled_app.state.handlers._monthly_strategy_plugin_contract_path
            == "strategy_plugin_contract.json"
        )
        assert enabled_app.state.handlers._backtest_max_parallel_strategies == 3

    @pytest.mark.asyncio
    async def test_monthly_scheduler_payload_scopes_approval_evidence(self, tmp_path):
        app = create_app(
            db_dir=str(tmp_path),
            config=_local_config(
                bot_ids=["ibkr", "crypto"],
                monthly_validation_mode="approval_gated",
                monthly_approval_scope_allowlist=["trading_stock_family"],
                monthly_approval_scope_map={
                    "ibkr": "trading_stock_family",
                    "crypto": "crypto_trader_portfolio",
                },
                monthly_deployment_metadata_install_report_paths=[
                    "reports/ibkr_metadata_install.json",
                ],
                monthly_operational_evidence_path="deployments/operational_evidence.json",
                monthly_relay_ingest_evidence_path="artifacts/relay_ingest.json",
                monthly_vps_host_id="ibkr-vps",
                monthly_assistant_host_id="local-assistant",
            ),
        )
        await app.state.queue.initialize()
        try:
            specs = {
                spec.name: spec
                for spec in app.state.scheduled_job_specs
                if spec.job_key == "monthly_validation"
            }
            scheduled_for = datetime(2026, 6, 2, 3, 0, tzinfo=timezone.utc)

            await specs["monthly_validation_ibkr"].execute(scheduled_for)
            await specs["monthly_validation_crypto"].execute(scheduled_for)
            rows = await app.state.queue.peek(10)
        finally:
            await app.state.queue.close()

        payloads = {row["bot_id"]: json.loads(row["payload"]) for row in rows}
        ibkr_payload = payloads["ibkr"]
        crypto_payload = payloads["crypto"]
        assert ibkr_payload["strategy_id"] == "trading_stock_family"
        assert ibkr_payload["approval_scope"] == "trading_stock_family"
        assert ibkr_payload["shadow"] is False
        assert ibkr_payload["approval_evidence_mode"] is True
        assert ibkr_payload["deployment_metadata_install_report_paths"] == [
            "reports/ibkr_metadata_install.json",
        ]
        assert ibkr_payload["operational_evidence_path"] == "deployments/operational_evidence.json"
        assert ibkr_payload["relay_ingest_evidence_path"] == "artifacts/relay_ingest.json"
        assert ibkr_payload["vps_host_id"] == "ibkr-vps"
        assert ibkr_payload["assistant_host_id"] == "local-assistant"
        assert crypto_payload["strategy_id"] == "crypto_trader_portfolio"
        assert crypto_payload["shadow"] is True
        assert crypto_payload["approval_evidence_mode"] is False
        assert "deployment_metadata_install_report_paths" not in crypto_payload

    def test_uses_normalized_curated_data_dir(self, tmp_path):
        (tmp_path / "curated").mkdir()
        app = create_app(db_dir=str(tmp_path), config=_local_config())
        assert app.state.handlers._curated_dir == tmp_path / "curated"
        assert app.state.handlers._raw_data_dir == tmp_path / "raw"

    def test_default_startup_uses_checked_package_memory(self, tmp_path):
        app = create_app(config=_local_config(data_dir=str(tmp_path / "runtime-data")))

        assert app.state.memory_dir == memory_root()
        assert app.state.handlers._memory_dir == memory_root()
        ctx = ContextBuilder(app.state.memory_dir)
        assert ctx.load_loop_contract_context(agent_type="monthly_model_review")
        assert ctx.load_recent_work_log_entries(
            agent_type="monthly_model_review",
            limit=5,
        )
        assert ctx.load_recent_performance_learning_entries(bot_id="bot1", limit=5)

    def test_uses_legacy_curated_dir_when_only_legacy_exists(self, tmp_path):
        legacy = tmp_path / "data" / "curated"
        legacy.mkdir(parents=True)
        app = create_app(db_dir=str(tmp_path), config=_local_config())
        assert app.state.handlers._curated_dir == legacy

    def test_prefers_normalized_curated_dir_when_both_exist(self, tmp_path):
        (tmp_path / "curated").mkdir()
        (tmp_path / "data" / "curated").mkdir(parents=True)
        app = create_app(db_dir=str(tmp_path), config=_local_config())
        assert app.state.handlers._curated_dir == tmp_path / "curated"

    def test_agent_runner_shares_event_stream(self, tmp_path):
        app = create_app(db_dir=str(tmp_path), config=_local_config())
        assert app.state.agent_runner._event_stream is app.state.event_stream

    def test_agent_runner_receives_launcher_args(self, tmp_path):
        config = AppConfig(
            allow_unauthenticated_local=True,
            claude_command="wsl.exe",
            claude_command_args=["--", "claude"],
            codex_command="wsl.exe",
            codex_command_args=["--", "codex"],
        )

        app = create_app(db_dir=str(tmp_path), config=config)

        assert app.state.agent_runner._claude_command == "wsl.exe"
        assert app.state.agent_runner._claude_command_args == ["--", "claude"]
        assert app.state.agent_runner._codex_command == "wsl.exe"
        assert app.state.agent_runner._codex_command_args == ["--", "codex"]

    def test_telegram_adapter_wired(self, tmp_path):
        config = _local_config(telegram_bot_token="token", telegram_chat_id="123")
        app = create_app(db_dir=str(tmp_path), config=config)
        assert NotificationChannel.TELEGRAM in app.state.dispatcher.adapters

    def test_prefs_loaded_from_disk(self, tmp_path):
        # Pre-create a prefs file
        prefs_path = tmp_path / "data" / "notification_prefs.json"
        prefs_path.parent.mkdir(parents=True)
        prefs_path.write_text(json.dumps({
            "channels": [{"channel": "telegram", "enabled": True, "chat_id": "saved-123"}]
        }), encoding="utf-8")

        config = _local_config()
        app = create_app(db_dir=str(tmp_path), config=config)
        assert len(app.state.notification_preferences.channels) == 1
        assert app.state.notification_preferences.channels[0].chat_id == "saved-123"

    def test_agent_prefs_loaded_from_disk(self, tmp_path):
        prefs_path = tmp_path / "data" / "agent_preferences.json"
        prefs_path.parent.mkdir(parents=True)
        prefs_path.write_text(json.dumps({
            "default": {"provider": "openrouter", "model": "minimax/minimax-m2.5"},
            "overrides": {
                "triage": {"provider": "codex_pro", "model": "gpt-5.4"},
            },
        }), encoding="utf-8")

        app = create_app(db_dir=str(tmp_path), config=_local_config())

        assert app.state.agent_preferences.default.provider == AgentProvider.OPENROUTER
        assert app.state.agent_preferences.overrides[AgentWorkflow.TRIAGE].provider == AgentProvider.CODEX_PRO

    def test_agent_prefs_seed_from_env_when_missing(self, tmp_path):
        config = AppConfig(
            allow_unauthenticated_local=True,
            agent_default_provider="zai_coding_plan",
            agent_default_model="glm-5",
            daily_agent_provider="claude_max",
            daily_agent_model="opus",
        )

        app = create_app(db_dir=str(tmp_path), config=config)

        assert app.state.agent_preferences.default.provider == AgentProvider.ZAI_CODING_PLAN
        assert app.state.agent_preferences.default.model == "glm-5"
        assert app.state.agent_preferences.overrides[AgentWorkflow.DAILY_ANALYSIS].provider == AgentProvider.CLAUDE_MAX
        assert app.state.agent_preferences.overrides[AgentWorkflow.DAILY_ANALYSIS].model == "opus"

    def test_telegram_settings_router_registered_when_telegram_enabled(self, tmp_path):
        config = _local_config(telegram_bot_token="token", telegram_chat_id="123")

        app = create_app(db_dir=str(tmp_path), config=config)

        assert app.state.telegram_callback_router is not None
        assert "cmd_settings" in app.state.telegram_callback_router.handlers
        assert "agent_settings_scope_" in app.state.telegram_callback_router.handlers

    def test_autonomous_app_exposes_repo_task_runner(self, tmp_path):
        config_dir = tmp_path / "bot_configs"
        config_dir.mkdir()
        (config_dir / "bot1.yaml").write_text(
            "bot_id: bot1\nrepo_dir: .\nparameters: []\n",
            encoding="utf-8",
        )
        config = AppConfig(
            allow_unauthenticated_local=True,
            autonomous_enabled=True,
            bot_config_dir=str(config_dir),
        )

        app = create_app(db_dir=str(tmp_path), config=config)

        assert app.state.repo_task_runner is not None
        assert app.state.approval_handler._repo_task_runner is app.state.repo_task_runner

    def test_outcome_wiring_records_selected_feedback_and_portfolio_proposal_outcomes(self):
        source = inspect.getsource(runtime_scheduled_callbacks)
        experiment_source = inspect.getsource(runtime_experiments)

        assert "measure_progressive(" in source
        assert "record_measurement_feedback(result)" in source
        assert "outcome_proposal_id = suggestion.get(\"proposal_id\")" in source
        assert "proposal_ledger.record_outcome" in source
        assert "experiment_results.jsonl" in experiment_source
        assert "structural_experiments.jsonl" in experiment_source
        assert "experiment_result_delta(result, experiment)" in experiment_source
        assert "structural_objective_delta(" in experiment_source

    @pytest.mark.asyncio
    async def test_lifespan_drains_relay_and_runs_startup_catchup(self, tmp_path):
        old_date = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        run_history_path = tmp_path / "data" / "run_history.jsonl"
        run_history_path.parent.mkdir(parents=True, exist_ok=True)
        run_history_path.write_text(json.dumps({
            "run_id": f"daily-{old_date}",
            "agent_type": "daily_analysis",
            "status": "completed",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }) + "\n", encoding="utf-8")

        config = _local_config(bot_ids=["bot1"], relay_url="https://relay.example")

        with (
            patch("trading_assistant.orchestrator.runtime.VPSReceiver") as MockReceiver,
            patch("trading_assistant.orchestrator.runtime_lifespan.create_scheduler", return_value=None),
        ):
            receiver = MockReceiver.return_value
            receiver.drain = AsyncMock()
            receiver.poll = AsyncMock()

            app = create_app(db_dir=str(tmp_path), config=config)

            async with app.router.lifespan_context(app):
                receiver.drain.assert_awaited_once()
                pending = await app.state.queue.peek(limit=20)

            assert any(event["event_type"] == "daily_analysis_trigger" for event in pending)
