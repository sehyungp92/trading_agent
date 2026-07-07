from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.schemas.notifications import NotificationPayload


class AuditLoggerHook:
    """Logs every outbound notification to a JSONL audit file."""

    def __init__(self, log_path: str | Path) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def transform(self, payload: NotificationPayload) -> NotificationPayload:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notification_type": payload.notification_type,
            "priority": payload.priority.value,
            "title": payload.title,
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return payload  # Pass through unchanged
