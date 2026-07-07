# skills/proactive_scanner.py
"""Proactive notification scanner — morning, continuous, and evening scans."""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.notifications import NotificationPayload, NotificationPriority


@dataclass
class ScanResult:
    payloads: list[NotificationPayload] = field(default_factory=list)

    @property
    def has_notifications(self) -> bool:
        return len(self.payloads) > 0


_SEVERITY_TO_PRIORITY = {
    "CRITICAL": NotificationPriority.CRITICAL,
    "HIGH": NotificationPriority.HIGH,
    "MEDIUM": NotificationPriority.NORMAL,
    "LOW": NotificationPriority.LOW,
}


class ProactiveScanner:
    def morning_scan(
        self, events: list[dict], errors: list[dict], unusual_losses: list[dict],
    ) -> ScanResult:
        payloads: list[NotificationPayload] = []
        for error in errors:
            severity = error.get("severity", "MEDIUM").upper()
            priority = _SEVERITY_TO_PRIORITY.get(severity, NotificationPriority.NORMAL)
            payloads.append(NotificationPayload(
                notification_type="alert",
                priority=priority,
                title=f"{error.get('bot_id', '?')} — {error.get('error_type', 'Error')}",
                body=error.get("message", ""),
                data=error,
            ))
        for loss in unusual_losses:
            payloads.append(NotificationPayload(
                notification_type="alert",
                priority=NotificationPriority.HIGH,
                title=f"Unusual loss — {loss.get('bot_id', '?')}",
                body=f"PnL: ${loss.get('pnl', 0):.0f} — {loss.get('reason', '')}",
                data=loss,
            ))
        return ScanResult(payloads=payloads)

    def continuous_scan(self, alerts: list) -> ScanResult:
        payloads: list[NotificationPayload] = []
        low_alerts: list[str] = []
        for alert in alerts:
            severity_str = alert.severity.value.upper()
            priority = _SEVERITY_TO_PRIORITY.get(severity_str, NotificationPriority.NORMAL)
            if priority in (NotificationPriority.LOW,):
                low_alerts.append(alert.message)
                continue
            payloads.append(NotificationPayload(
                notification_type="alert",
                priority=priority,
                title=f"{alert.source} alert",
                body=alert.message,
            ))
        if low_alerts:
            payloads.append(NotificationPayload(
                notification_type="alert",
                priority=NotificationPriority.LOW,
                title=f"{len(low_alerts)} low-priority alert(s)",
                body="\n".join(f"- {m}" for m in low_alerts),
            ))
        return ScanResult(payloads=payloads)

    def evening_scan(self, date: str, daily_report_ready: bool) -> ScanResult:
        if daily_report_ready:
            return ScanResult(payloads=[
                NotificationPayload(
                    notification_type="daily_report",
                    priority=NotificationPriority.NORMAL,
                    title=f"Daily Report — {date}",
                    body=f"Your daily trading report for {date} is ready.",
                    data={"date": date},
                )
            ])
        else:
            return ScanResult(payloads=[
                NotificationPayload(
                    notification_type="alert",
                    priority=NotificationPriority.LOW,
                    title=f"Daily report pending — {date}",
                    body=f"Daily report for {date} is not ready yet. Analysis may still be running.",
                    data={"date": date},
                )
            ])

    def detect_unusual_losses(
        self,
        bot_id: str,
        recent_trade: TradeEvent,
        historical_losses: list[float],
        sigma_threshold: float = 2.0,
    ) -> NotificationPayload | None:
        """Detect if a recent loss is >2 sigma from the historical mean.

        Args:
            bot_id: The bot that produced this trade.
            recent_trade: The trade to evaluate.
            historical_losses: List of past loss PnL values (negative floats).
            sigma_threshold: Number of standard deviations for flagging.

        Returns:
            NotificationPayload if the loss is unusual, else None.
        """
        if not historical_losses or recent_trade.pnl >= 0:
            return None
        mean_loss = statistics.mean(historical_losses)
        if len(historical_losses) < 2:
            return None
        stdev = statistics.stdev(historical_losses)
        if stdev == 0:
            return None
        z_score = (recent_trade.pnl - mean_loss) / stdev
        if abs(z_score) < sigma_threshold:
            return None
        return NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.HIGH,
            title=f"Unusual loss — {bot_id}",
            body=(
                f"PnL: ${recent_trade.pnl:.0f} on {recent_trade.pair}. "
                f"This is {abs(z_score):.1f}\u03c3 from 30d mean (${mean_loss:.0f})."
            ),
        )

    def detect_repeated_errors(
        self,
        errors: list[dict],
        threshold: int = 3,
    ) -> list[NotificationPayload]:
        """Detect repeated errors of the same type within a short window.

        Groups errors by 'bot_id:error_type' and returns payloads for groups
        whose count meets or exceeds the threshold.

        Args:
            errors: List of dicts with keys: bot_id, error_type, timestamp.
            threshold: Minimum count to trigger a notification.

        Returns:
            List of NotificationPayload for each group exceeding the threshold.
        """
        type_counts: Counter[str] = Counter()
        for e in errors:
            key = f"{e.get('bot_id', 'unknown')}:{e.get('error_type', 'unknown')}"
            type_counts[key] += 1

        payloads: list[NotificationPayload] = []
        for key, count in type_counts.items():
            if count >= threshold:
                bot_id, error_type = key.split(":", 1)
                payloads.append(NotificationPayload(
                    notification_type="alert",
                    priority=NotificationPriority.HIGH,
                    title=f"Repeated errors — {bot_id}",
                    body=f"{error_type} occurred {count}x recently. Possible systemic issue.",
                ))
        return payloads

    def check_heartbeats(
        self,
        bot_heartbeats: dict[str, str],
        current_time: str,
        max_gap_hours: int = 2,
    ) -> list[NotificationPayload]:
        """Check for missing bot heartbeats.

        Args:
            bot_heartbeats: Mapping of bot_id to last-seen ISO timestamp.
            current_time: Current time as ISO string.
            max_gap_hours: Maximum acceptable gap in hours before alerting.

        Returns:
            List of NotificationPayload for bots whose heartbeat is stale.
            Priority is CRITICAL if >4h, HIGH if >2h.
        """
        now = datetime.fromisoformat(current_time)
        payloads: list[NotificationPayload] = []
        for bot_id, last_seen_str in bot_heartbeats.items():
            last_seen = datetime.fromisoformat(last_seen_str)
            gap = now - last_seen
            if gap > timedelta(hours=max_gap_hours):
                hours_ago = gap.total_seconds() / 3600
                payloads.append(NotificationPayload(
                    notification_type="alert",
                    priority=NotificationPriority.CRITICAL if hours_ago > 4 else NotificationPriority.HIGH,
                    title=f"Heartbeat missing — {bot_id}",
                    body=f"Last seen {hours_ago:.1f} hours ago. Check bot health.",
                ))
        return payloads
