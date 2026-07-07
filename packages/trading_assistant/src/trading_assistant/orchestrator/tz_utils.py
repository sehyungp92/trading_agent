"""Timezone utility functions for per-bot date/hour conversions."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from trading_assistant.schemas.bot_config import BotConfig


def bot_trading_date(tz_name: str, utc_time: datetime | None = None) -> str:
    """Return the current YYYY-MM-DD in a bot's local timezone.

    Args:
        tz_name: IANA timezone string, e.g. ``"Asia/Seoul"``.
        utc_time: Optional UTC datetime. Defaults to ``datetime.now(UTC)``.
    """
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)
    local = utc_time.astimezone(ZoneInfo(tz_name))
    return local.strftime("%Y-%m-%d")


def to_local_hour(entry_time: datetime, tz_name: str) -> int:
    """Convert a UTC-aware datetime to local hour (0-23) in the given timezone."""
    local = entry_time.astimezone(ZoneInfo(tz_name))
    return local.hour


def market_close_utc(
    tz_name: str,
    close_local: str = "16:00",
    date: str | None = None,
) -> datetime:
    """Compute the UTC datetime of market close for a given timezone and date.

    Args:
        tz_name: IANA timezone, e.g. ``"Asia/Seoul"``.
        close_local: ``"HH:MM"`` in local time.
        date: ``"YYYY-MM-DD"`` — defaults to today in the bot's tz.
    """
    tz = ZoneInfo(tz_name)
    if date is None:
        date = bot_trading_date(tz_name)

    h, m = (int(x) for x in close_local.split(":"))
    local_dt = datetime(
        int(date[:4]), int(date[5:7]), int(date[8:10]),
        h, m, tzinfo=tz,
    )
    return local_dt.astimezone(timezone.utc)


def analysis_trigger_utc(bot_config: BotConfig, date: str | None = None) -> datetime:
    """Compute when daily analysis should trigger (UTC) for a bot.

    This is market_close + daily_analysis_delay_minutes.
    """
    close = market_close_utc(
        bot_config.timezone, bot_config.market_close_local, date,
    )
    return close + timedelta(minutes=bot_config.daily_analysis_delay_minutes)


def group_bots_by_analysis_time(
    bot_configs: dict[str, BotConfig],
) -> dict[str, list[str]]:
    """Group bots that trigger daily analysis at the same UTC hour.

    Returns ``{"{HH}:{MM}": [bot_id, ...], ...}``.
    Bots are grouped by the UTC hour:minute of their analysis trigger
    (using today's date for DST calculation).
    """
    groups: dict[str, list[str]] = {}
    for bot_id, cfg in bot_configs.items():
        trigger = analysis_trigger_utc(cfg)
        key = f"{trigger.hour:02d}:{trigger.minute:02d}"
        groups.setdefault(key, []).append(bot_id)
    return groups
