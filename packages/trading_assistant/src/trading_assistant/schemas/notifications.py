# schemas/notifications.py
"""Notification schemas — channels, preferences, control panel state, payloads.

Used by the communication layer (comms/) to route notifications to the right
channel with the right priority and format.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class NotificationChannel(str, Enum):
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"


class NotificationPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {"critical": 4, "high": 3, "normal": 2, "low": 1}[self.value]

    @property
    def bypasses_quiet_hours(self) -> bool:
        return self == NotificationPriority.CRITICAL


class ChannelConfig(BaseModel):
    """Per-channel notification configuration."""
    channel: NotificationChannel
    enabled: bool = True
    chat_id: str = ""
    quiet_hours_start: Optional[int] = None
    quiet_hours_end: Optional[int] = None

    def is_quiet_at(self, hour_utc: int) -> bool:
        if self.quiet_hours_start is None or self.quiet_hours_end is None:
            return False
        start = self.quiet_hours_start
        end = self.quiet_hours_end
        if start <= end:
            return start <= hour_utc < end
        else:
            return hour_utc >= start or hour_utc < end


class NotificationPreferences(BaseModel):
    """User notification preferences across all channels."""
    channels: list[ChannelConfig] = []

    def get_active_channels(self) -> list[ChannelConfig]:
        return [c for c in self.channels if c.enabled]

    def get_channels_for_priority(
        self, priority: NotificationPriority, current_hour_utc: int,
    ) -> list[ChannelConfig]:
        result: list[ChannelConfig] = []
        for cfg in self.channels:
            if not cfg.enabled:
                continue
            if cfg.is_quiet_at(current_hour_utc) and not priority.bypasses_quiet_hours:
                continue
            result.append(cfg)
        return result


class BotStatusLine(BaseModel):
    """Single bot status for the control panel."""
    bot_id: str
    status: str
    pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    summary: str = ""

    @property
    def status_emoji(self) -> str:
        return {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(self.status, "⚪")


class ControlPanelState(BaseModel):
    """State of the daily pinned control surface message."""
    date: str
    portfolio_pnl: float = 0.0
    portfolio_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0
    exposure_pct: float = 0.0
    daily_report_ready: bool = False
    alert_count: int = 0
    alert_summary: str = ""
    monthly_validation_status: str = ""
    pending_pr_count: int = 0
    risk_status: str = "OK"
    risk_detail: str = ""
    bot_statuses: list[BotStatusLine] = []


class NotificationPayload(BaseModel):
    """Generic notification payload that renderers transform into channel-specific format."""
    notification_type: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    title: str = ""
    body: str = ""
    data: dict = {}
    attachments: list[str] = []
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
