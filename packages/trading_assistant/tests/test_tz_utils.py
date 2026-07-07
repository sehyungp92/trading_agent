"""Tests for timezone utility functions."""
from __future__ import annotations

from datetime import datetime, timezone


from trading_assistant.orchestrator.tz_utils import (
    bot_trading_date,
    to_local_hour,
    market_close_utc,
    analysis_trigger_utc,
    group_bots_by_analysis_time,
)
from trading_assistant.schemas.bot_config import BotConfig


class TestBotTradingDate:
    def test_utc_default(self):
        utc = datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc)
        assert bot_trading_date("UTC", utc) == "2026-03-08"

    def test_kst_same_day(self):
        # 2026-03-08 14:00 UTC = 2026-03-08 23:00 KST — still Mar 8
        utc = datetime(2026, 3, 8, 14, 0, tzinfo=timezone.utc)
        assert bot_trading_date("Asia/Seoul", utc) == "2026-03-08"

    def test_kst_date_boundary_crosses(self):
        # 2026-03-08 15:01 UTC = 2026-03-09 00:01 KST — crosses to Mar 9
        utc = datetime(2026, 3, 8, 15, 1, tzinfo=timezone.utc)
        assert bot_trading_date("Asia/Seoul", utc) == "2026-03-09"

    def test_us_eastern_behind_utc(self):
        # 2026-03-09 03:00 UTC = 2026-03-08 22:00 EST (or 23:00 EDT)
        # March 8 2026 — DST springs forward in US on Mar 8 2026 (2nd Sunday)
        utc = datetime(2026, 3, 9, 3, 0, tzinfo=timezone.utc)
        result = bot_trading_date("US/Eastern", utc)
        # Should be Mar 8 because 03:00 UTC is 22:00 or 23:00 Eastern on Mar 8
        assert result == "2026-03-08"


class TestToLocalHour:
    def test_utc_passthrough(self):
        t = datetime(2026, 3, 8, 14, 30, tzinfo=timezone.utc)
        assert to_local_hour(t, "UTC") == 14

    def test_kst_plus_nine(self):
        # 00:00 UTC = 09:00 KST
        t = datetime(2026, 3, 8, 0, 0, tzinfo=timezone.utc)
        assert to_local_hour(t, "Asia/Seoul") == 9

    def test_kst_wraps_day(self):
        # 18:00 UTC = 03:00 KST (next day)
        t = datetime(2026, 3, 8, 18, 0, tzinfo=timezone.utc)
        assert to_local_hour(t, "Asia/Seoul") == 3


class TestMarketCloseUtc:
    def test_kst_close(self):
        # KST market closes at 15:30 local = 06:30 UTC
        result = market_close_utc("Asia/Seoul", "15:30", "2026-03-08")
        assert result.hour == 6
        assert result.minute == 30
        assert result.tzinfo == timezone.utc

    def test_utc_close(self):
        result = market_close_utc("UTC", "16:00", "2026-03-08")
        assert result.hour == 16
        assert result.minute == 0


class TestAnalysisTriggerUtc:
    def test_default_delay(self):
        cfg = BotConfig(bot_id="k", timezone="Asia/Seoul", market_close_local="15:30")
        result = analysis_trigger_utc(cfg, "2026-03-08")
        # 15:30 KST = 06:30 UTC + 60 min = 07:30 UTC
        assert result.hour == 7
        assert result.minute == 30

    def test_custom_delay(self):
        cfg = BotConfig(
            bot_id="k", timezone="Asia/Seoul",
            market_close_local="15:30", daily_analysis_delay_minutes=30,
        )
        result = analysis_trigger_utc(cfg, "2026-03-08")
        # 06:30 + 30 min = 07:00 UTC
        assert result.hour == 7
        assert result.minute == 0


class TestGroupBotsByAnalysisTime:
    def test_single_group_utc(self):
        configs = {
            "bot_a": BotConfig(bot_id="bot_a"),
            "bot_b": BotConfig(bot_id="bot_b"),
        }
        groups = group_bots_by_analysis_time(configs)
        # Both UTC with 16:00 close + 60min delay = 17:00 UTC
        assert len(groups) == 1
        key = list(groups.keys())[0]
        assert set(groups[key]) == {"bot_a", "bot_b"}

    def test_mixed_timezones(self):
        configs = {
            "btc_bot": BotConfig(bot_id="btc_bot"),  # UTC 16:00+60 = 17:00
            "k_stock": BotConfig(
                bot_id="k_stock", timezone="Asia/Seoul",
                market_close_local="15:30",
            ),  # KST 15:30+60 = 16:30 KST = 07:30 UTC
        }
        groups = group_bots_by_analysis_time(configs)
        assert len(groups) == 2

    def test_empty(self):
        assert group_bots_by_analysis_time({}) == {}
