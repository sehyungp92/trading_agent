from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from trading_assistant.schemas.notifications import NotificationPayload, NotificationPriority

@dataclass
class OutboundMessage:
    """Message queued for outbound delivery."""
    payload: NotificationPayload
    priority: NotificationPriority = NotificationPriority.NORMAL
    channel_hint: str = ""  # optional channel preference
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class InboundMessage:
    """Message received from a communication channel."""
    source_channel: str  # "telegram", "discord"
    callback_data: str = ""
    text: str = ""
    user_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
