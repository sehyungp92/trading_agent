"""Per-bot timezone and market hours configuration."""
from __future__ import annotations

from pydantic import BaseModel, field_validator


class BotConfig(BaseModel):
    """Configuration for a single trading bot's timezone and market hours."""

    bot_id: str
    timezone: str = "UTC"  # IANA tz, e.g. "Asia/Seoul"
    market_close_local: str = "16:00"  # HH:MM in bot's local tz
    daily_analysis_delay_minutes: int = 60  # minutes after market close to trigger analysis
    strategy_version: str | None = None
    config_version: str | None = None
    code_sha: str | None = None

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, KeyError):
            raise ValueError(f"Invalid IANA timezone: {v!r}") from None
        return v

    @field_validator("market_close_local")
    @classmethod
    def validate_market_close(cls, v: str) -> str:
        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError(f"market_close_local must be HH:MM, got {v!r}")
        try:
            h, m = int(parts[0]), int(parts[1])
            if not (0 <= h <= 23 and 0 <= m <= 59):
                raise ValueError
        except ValueError:
            raise ValueError(
                f"market_close_local must be HH:MM with valid hours/minutes, got {v!r}"
            ) from None
        return v
