# tests/test_config.py
"""Tests for AppConfig environment variable loading."""
from trading_assistant.orchestrator.config import AppConfig


class TestAppConfig:
    def test_from_env_reads_bot_ids(self, monkeypatch):
        monkeypatch.setenv("BOT_IDS", "bot_a,bot_b,bot_c")
        config = AppConfig.from_env()
        assert config.bot_ids == ["bot_a", "bot_b", "bot_c"]

    def test_from_env_empty_bot_ids(self, monkeypatch):
        monkeypatch.delenv("BOT_IDS", raising=False)
        config = AppConfig.from_env()
        assert config.bot_ids == []

    def test_from_env_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("BOT_IDS", " bot_a , bot_b ")
        config = AppConfig.from_env()
        assert config.bot_ids == ["bot_a", "bot_b"]

    def test_from_env_reads_relay_url(self, monkeypatch):
        monkeypatch.setenv("RELAY_URL", "https://relay.example.com")
        config = AppConfig.from_env()
        assert config.relay_url == "https://relay.example.com"

    def test_from_env_reads_telegram(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
        config = AppConfig.from_env()
        assert config.telegram_bot_token == "123:ABC"
        assert config.telegram_chat_id == "-100123"

    def test_from_env_reads_discord(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord-token")
        monkeypatch.setenv("DISCORD_CHANNEL_ID", "999888777")
        config = AppConfig.from_env()
        assert config.discord_bot_token == "discord-token"
        assert config.discord_channel_id == "999888777"

    def test_from_env_reads_smtp(self, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("SMTP_PORT", "465")
        monkeypatch.setenv("SMTP_USER", "user@example.com")
        monkeypatch.setenv("SMTP_PASS", "secret")
        monkeypatch.setenv("EMAIL_FROM", "from@example.com")
        monkeypatch.setenv("EMAIL_TO", "to@example.com")
        config = AppConfig.from_env()
        assert config.smtp_host == "smtp.example.com"
        assert config.smtp_port == 465
        assert config.smtp_user == "user@example.com"
        assert config.email_from == "from@example.com"

    def test_from_env_defaults(self, monkeypatch):
        # Clear all relevant env vars
        for key in ["BOT_IDS", "RELAY_URL", "TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN",
                     "SMTP_HOST", "SMTP_USER", "DATA_DIR", "LOG_LEVEL",
                     "ALLOW_UNAUTHENTICATED_LOCAL", "BIND_HOST", "UVICORN_HOST",
                     "ENVIRONMENT", "DIRECT_INGEST_ONLY"]:
            monkeypatch.delenv(key, raising=False)
        config = AppConfig.from_env()
        assert config.bot_ids == []
        assert config.relay_url == ""
        assert config.telegram_bot_token == ""
        assert config.data_dir == "data"
        assert config.log_level == "INFO"
        assert config.smtp_port == 587
        assert config.allow_unauthenticated_local is False
        assert config.bind_host == "127.0.0.1"
        assert config.configured_bind_host == ""
        assert config.bind_host_explicit is False
        assert config.environment == "development"
        assert config.direct_ingest_only is False

    def test_from_env_records_explicit_bind_host(self, monkeypatch):
        monkeypatch.setenv("BIND_HOST", "0.0.0.0")
        config = AppConfig.from_env()
        assert config.bind_host == "0.0.0.0"
        assert config.configured_bind_host == "0.0.0.0"
        assert config.bind_host_explicit is True

    def test_from_env_prefers_uvicorn_host_as_actual_launcher_host(self, monkeypatch):
        monkeypatch.setenv("BIND_HOST", "127.0.0.1")
        monkeypatch.setenv("UVICORN_HOST", "0.0.0.0")
        config = AppConfig.from_env()
        assert config.bind_host == "0.0.0.0"
        assert config.configured_bind_host == "127.0.0.1"
        assert config.uvicorn_host == "0.0.0.0"
        assert config.bind_host_explicit is True

    def test_from_env_reads_environment_and_direct_ingest_mode(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DIRECT_INGEST_ONLY", "true")
        config = AppConfig.from_env()
        assert config.environment == "production"
        assert config.is_production is True
        assert config.direct_ingest_only is True

    def test_from_env_reads_monthly_approval_scope_evidence_fields(self, monkeypatch):
        monkeypatch.setenv("MONTHLY_APPROVAL_SCOPE_ALLOWLIST", "trading_stock_family")
        monkeypatch.setenv(
            "MONTHLY_APPROVAL_SCOPE_MAP",
            "ibkr:trading_stock_family,crypto=crypto_trader_portfolio",
        )
        monkeypatch.setenv(
            "MONTHLY_DEPLOYMENT_METADATA_INSTALL_REPORTS",
            "reports/ibkr.json,reports/crypto.json",
        )
        monkeypatch.setenv("MONTHLY_OPERATIONAL_EVIDENCE_PATH", "deployments/ops.json")
        monkeypatch.setenv("MONTHLY_RELAY_INGEST_EVIDENCE_PATH", "artifacts/relay.json")
        monkeypatch.setenv("MONTHLY_VPS_HOST_ID", "ibkr-vps")
        monkeypatch.setenv("MONTHLY_ASSISTANT_HOST_ID", "assistant-local")

        config = AppConfig.from_env()

        assert config.monthly_approval_scope_allowlist == ["trading_stock_family"]
        assert config.monthly_approval_scope_map == {
            "ibkr": "trading_stock_family",
            "crypto": "crypto_trader_portfolio",
        }
        assert config.monthly_deployment_metadata_install_report_paths == [
            "reports/ibkr.json",
            "reports/crypto.json",
        ]
        assert config.monthly_operational_evidence_path == "deployments/ops.json"
        assert config.monthly_relay_ingest_evidence_path == "artifacts/relay.json"
        assert config.monthly_vps_host_id == "ibkr-vps"
        assert config.monthly_assistant_host_id == "assistant-local"

    def test_direct_construction(self):
        config = AppConfig(bot_ids=["a", "b"], relay_url="http://test")
        assert config.bot_ids == ["a", "b"]
        assert config.relay_url == "http://test"
