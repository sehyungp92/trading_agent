"""Tests for BotConfig schema and AppConfig bot_configs parsing."""
from __future__ import annotations


import pytest

from trading_assistant.schemas.bot_config import BotConfig
from trading_assistant.orchestrator.config import AppConfig, _parse_bot_timezones


class TestBotConfig:
    def test_defaults(self):
        bc = BotConfig(bot_id="test_bot")
        assert bc.timezone == "UTC"
        assert bc.market_close_local == "16:00"
        assert bc.daily_analysis_delay_minutes == 60

    def test_valid_timezone(self):
        bc = BotConfig(bot_id="k_stock_trader", timezone="Asia/Seoul")
        assert bc.timezone == "Asia/Seoul"

    def test_invalid_timezone_rejected(self):
        with pytest.raises(ValueError, match="Invalid IANA timezone"):
            BotConfig(bot_id="bad", timezone="Not/A/Zone")

    def test_market_close_format(self):
        bc = BotConfig(bot_id="bot", market_close_local="15:30")
        assert bc.market_close_local == "15:30"

    def test_market_close_bad_format(self):
        with pytest.raises(ValueError, match="HH:MM"):
            BotConfig(bot_id="bot", market_close_local="2530")

    def test_market_close_bad_hours(self):
        with pytest.raises(ValueError, match="HH:MM"):
            BotConfig(bot_id="bot", market_close_local="25:00")


class TestParseBotTimezones:
    def test_empty_string(self):
        result = _parse_bot_timezones("", ["bot_a"])
        assert "bot_a" in result
        assert result["bot_a"].timezone == "UTC"

    def test_single_mapping(self):
        result = _parse_bot_timezones("k_stock:Asia/Seoul", ["k_stock", "btc_bot"])
        assert result["k_stock"].timezone == "Asia/Seoul"
        assert result["btc_bot"].timezone == "UTC"

    def test_multiple_mappings(self):
        result = _parse_bot_timezones(
            "bot_a:Asia/Seoul,bot_b:US/Eastern", ["bot_a", "bot_b"],
        )
        assert result["bot_a"].timezone == "Asia/Seoul"
        assert result["bot_b"].timezone == "US/Eastern"

    def test_invalid_tz_skipped(self):
        result = _parse_bot_timezones("bot_a:Bad/Zone", ["bot_a"])
        # Falls back to default UTC
        assert result["bot_a"].timezone == "UTC"

    def test_malformed_entry_skipped(self):
        result = _parse_bot_timezones("nocolon", ["bot_a"])
        assert result["bot_a"].timezone == "UTC"

    def test_extra_whitespace(self):
        result = _parse_bot_timezones(" bot_a : Asia/Seoul , bot_b : UTC ", ["bot_a", "bot_b"])
        assert result["bot_a"].timezone == "Asia/Seoul"
        assert result["bot_b"].timezone == "UTC"


class TestAppConfigBotConfigs:
    def test_from_env_with_timezones(self, monkeypatch):
        monkeypatch.setenv("BOT_IDS", "k_stock,btc_bot")
        monkeypatch.setenv("BOT_TIMEZONES", "k_stock:Asia/Seoul")
        config = AppConfig.from_env()
        assert config.bot_ids == ["k_stock", "btc_bot"]
        assert config.bot_configs["k_stock"].timezone == "Asia/Seoul"
        assert config.bot_configs["btc_bot"].timezone == "UTC"

    def test_from_env_no_timezones(self, monkeypatch):
        monkeypatch.setenv("BOT_IDS", "bot_a")
        monkeypatch.delenv("BOT_TIMEZONES", raising=False)
        config = AppConfig.from_env()
        assert config.bot_configs["bot_a"].timezone == "UTC"

    def test_backward_compat_no_bot_configs(self):
        config = AppConfig(bot_ids=["bot_a"])
        assert config.bot_configs == {}
