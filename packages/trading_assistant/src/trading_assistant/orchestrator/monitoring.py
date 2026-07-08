"""Deterministic monitoring loop — no LLM calls.

Runs on a cron schedule (e.g., every 10 minutes) and produces alerts when:
  - Agent tasks are stuck (running beyond timeout)
  - Run folders are missing expected output files
  - VPS sidecar heartbeats are stale
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain
from trading_assistant.orchestrator.task_registry import TaskRegistry

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Alert:
    severity: AlertSeverity
    source: str  # which check produced this
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MonitoringCheck:
    """Individual monitoring checks. Compose as needed."""

    def __init__(
        self,
        registry: TaskRegistry | None = None,
        task_timeout_seconds: int = 3600,
        heartbeat_dir: str = "",
        heartbeat_max_age_seconds: int = 7200,
        queue=None,
        dead_letter_critical_threshold: int = 5,
        brain: OrchestratorBrain | None = None,
        heartbeat_md_path: str = "",
        relay_url: str = "",
        relay_timeout: float = 10.0,
        relay_disk_threshold_bytes: int = 500_000_000,
        bot_silence_threshold_seconds: int = 21600,
        relay_pending_age_threshold_seconds: float = 1800.0,
        latency_p95_threshold_seconds: float = 1800.0,
        latency_tracker=None,
        _relay_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._registry = registry
        self._task_timeout = task_timeout_seconds
        self._heartbeat_dir = heartbeat_dir
        self._heartbeat_max_age = heartbeat_max_age_seconds
        self._queue = queue
        self._dead_letter_critical = dead_letter_critical_threshold
        self._brain = brain
        self._heartbeat_md_path = heartbeat_md_path
        self._relay_url = relay_url
        self._relay_timeout = relay_timeout
        self._relay_disk_threshold = relay_disk_threshold_bytes
        self._bot_silence_threshold = bot_silence_threshold_seconds
        self._relay_pending_age_threshold = relay_pending_age_threshold_seconds
        self._latency_p95_threshold = latency_p95_threshold_seconds
        self._latency_tracker = latency_tracker
        self._relay_client_factory = _relay_client_factory

    async def check_stale_tasks(self) -> list[Alert]:
        """Find tasks stuck in RUNNING state beyond timeout."""
        if not self._registry:
            return []

        stale = await self._registry.find_stale(timeout_seconds=self._task_timeout)
        return [
            Alert(
                severity=AlertSeverity.HIGH,
                source="stale_task",
                message=f"Task '{task.id}' ({task.type}) has been running since {task.started_at}",
            )
            for task in stale
        ]

    def _read_heartbeats(self) -> list[tuple[str, datetime | None, float, str | None]]:
        """Read all heartbeat files. Returns [(bot_id, last_seen, age_seconds, error)]."""
        if not self._heartbeat_dir:
            return []
        hb_path = Path(self._heartbeat_dir)
        now = datetime.now(timezone.utc)
        results = []
        for hb_file in sorted(hb_path.glob("*.heartbeat")):
            bot_id = hb_file.stem
            try:
                last_seen = datetime.fromisoformat(hb_file.read_text().strip())
                if last_seen.tzinfo is None:
                    last_seen = last_seen.replace(tzinfo=timezone.utc)
                age = (now - last_seen).total_seconds()
                results.append((bot_id, last_seen, age, None))
            except (ValueError, OSError, TypeError) as e:
                results.append((bot_id, None, 0.0, str(e)))
        return results

    def check_heartbeats(self) -> list[Alert]:
        """Check VPS sidecar heartbeat freshness."""
        alerts: list[Alert] = []
        for bot_id, _last_seen, age, error in self._read_heartbeats():
            if error:
                alerts.append(Alert(
                    severity=AlertSeverity.HIGH,
                    source="heartbeat",
                    message=f"Cannot read heartbeat for '{bot_id}': {error}",
                ))
            elif age > self._heartbeat_max_age:
                alerts.append(Alert(
                    severity=AlertSeverity.CRITICAL,
                    source="heartbeat",
                    message=f"Bot '{bot_id}' last heartbeat was {age / 3600:.1f}h ago",
                ))
        return alerts

    def check_run_outputs(self, run_dir: str, expected_files: list[str]) -> list[Alert]:
        """Verify a run folder contains expected output files."""
        alerts: list[Alert] = []
        run_path = Path(run_dir)

        for filename in expected_files:
            if not (run_path / filename).exists():
                alerts.append(Alert(
                    severity=AlertSeverity.MEDIUM,
                    source="run_output",
                    message=f"Missing output file: {run_path / filename}",
                ))

        return alerts

    async def update_heartbeat_md(self) -> None:
        """Write a human-readable system status to heartbeat.md."""
        if not self._heartbeat_md_path:
            return

        now = datetime.now(timezone.utc)
        lines = [
            "# System Heartbeat",
            f"Last check: {now.isoformat()}",
            "",
            "## Bot Status",
        ]

        heartbeats = self._read_heartbeats()
        if heartbeats:
            for bot_id, _last_seen, age, error in heartbeats:
                if error:
                    lines.append(f"- {bot_id}: unknown (unreadable)")
                elif age > self._heartbeat_max_age:
                    lines.append(f"- {bot_id}: STALE (last seen {age / 3600:.1f}h ago)")
                else:
                    lines.append(f"- {bot_id}: healthy (last seen {int(age / 60)}min ago)")
        else:
            lines.append("- No heartbeat data available")

        lines.append("")
        lines.append("## Queue")

        pending = 0
        dead_count = 0
        if self._queue:
            try:
                pending = await self._queue.count_pending()
            except Exception:
                pass
            try:
                dead_count = await self._queue.count_dead_letters()
            except Exception:
                pass

        lines.append(f"- Pending events: {pending}")
        lines.append(f"- Dead letters: {dead_count}")

        lines.append("")
        lines.append("## Relay")
        if self._relay_url:
            lines.append(f"- URL: {self._relay_url}")
        else:
            lines.append("- Not configured")

        if self._latency_tracker:
            agg = self._latency_tracker.get_aggregate_stats()
            if agg.sample_count > 0:
                lines.append(f"- Delivery latency p50: {agg.p50:.1f}s, p95: {agg.p95:.1f}s ({agg.sample_count} samples)")
            else:
                lines.append("- No latency data")

        lines.append("")
        lines.append("## Last Analyses")

        if self._brain:
            lines.append(f"- Daily: {self._brain.last_daily_analysis or 'never'}")
            lines.append(f"- Weekly: {self._brain.last_weekly_analysis or 'never'}")
        else:
            lines.append("- Daily: unknown")
            lines.append("- Weekly: unknown")

        lines.append("")

        md_path = Path(self._heartbeat_md_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(lines), encoding="utf-8")

    async def check_dead_letters(self) -> list[Alert]:
        """Alert when events are stuck in dead-letter queue."""
        if not self._queue:
            return []
        try:
            count = await self._queue.count_dead_letters()
        except Exception:
            return []
        if count == 0:
            return []
        severity = AlertSeverity.CRITICAL if count >= self._dead_letter_critical else AlertSeverity.HIGH
        return [Alert(
            severity=severity,
            source="dead_letter",
            message=f"{count} event(s) in dead-letter queue — inspect and reprocess or discard",
        )]


    async def check_relay_health(self) -> list[Alert]:
        """Check relay health via its enriched /health endpoint.

        Produces alerts for:
        - Unreachable relay → CRITICAL
        - Disk threshold exceeded → HIGH
        - Per-bot silence > threshold → HIGH
        - Latency p95 > threshold → HIGH
        """
        if not self._relay_url:
            return []

        import httpx
        from trading_assistant.schemas.relay_health import RelayHealthResponse

        alerts: list[Alert] = []
        now = datetime.now(timezone.utc)

        try:
            if self._relay_client_factory:
                client = self._relay_client_factory()
            else:
                client = httpx.AsyncClient(
                    base_url=self._relay_url, timeout=self._relay_timeout,
                )

            async with client:
                resp = await client.get("/health")
                resp.raise_for_status()
                health = RelayHealthResponse(**resp.json())
        except Exception as exc:
            logger.warning("Relay health check failed: %s", exc)
            return [Alert(
                severity=AlertSeverity.CRITICAL,
                source="relay_health",
                message=f"Relay unreachable: {exc}",
            )]

        # Disk threshold
        if health.db_size_bytes > self._relay_disk_threshold:
            mb = health.db_size_bytes / (1024 * 1024)
            alerts.append(Alert(
                severity=AlertSeverity.HIGH,
                source="relay_health",
                message=f"Relay DB size {mb:.0f} MB exceeds threshold",
            ))

        if health.oldest_pending_age_seconds > self._relay_pending_age_threshold:
            alerts.append(Alert(
                severity=AlertSeverity.HIGH,
                source="relay_health",
                message=(
                    "Relay oldest pending event age is "
                    f"{health.oldest_pending_age_seconds:.0f}s "
                    f"with {health.pending_events} pending event(s)"
                ),
            ))

        # Per-bot silence
        for bot_id, last_ts in health.last_event_per_bot.items():
            try:
                last_dt = datetime.fromisoformat(last_ts)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                age = (now - last_dt).total_seconds()
                if age > self._bot_silence_threshold:
                    alerts.append(Alert(
                        severity=AlertSeverity.HIGH,
                        source="relay_health",
                        message=f"Bot '{bot_id}' last event was {age / 3600:.1f}h ago on relay",
                    ))
            except (ValueError, TypeError):
                pass

        # Latency p95 per bot
        if self._latency_tracker:
            all_stats = self._latency_tracker.get_all_stats()
            for bot_id, stats in all_stats.items():
                if stats.p95 > self._latency_p95_threshold:
                    alerts.append(Alert(
                        severity=AlertSeverity.HIGH,
                        source="relay_health",
                        message=f"Bot '{bot_id}' delivery latency p95 is {stats.p95:.0f}s",
                    ))

        return alerts


class MonitoringLoop:
    """Orchestrates all monitoring checks and collects alerts."""

    def __init__(
        self,
        checks: list[MonitoringCheck],
        event_stream: EventStream | None = None,
    ) -> None:
        self._checks = checks
        self._event_stream: EventStream | None = event_stream

    async def run_all(self) -> list[Alert]:
        """Run all checks and return combined alerts."""
        all_alerts: list[Alert] = []
        for check in self._checks:
            all_alerts.extend(await check.check_stale_tasks())
            all_alerts.extend(check.check_heartbeats())
            all_alerts.extend(await check.check_dead_letters())
            all_alerts.extend(await check.check_relay_health())
            await check.update_heartbeat_md()
        sorted_alerts = sorted(all_alerts, key=lambda a: list(AlertSeverity).index(a.severity))

        # Emit SSE events for CRITICAL/HIGH alerts
        if self._event_stream:
            for alert in sorted_alerts:
                if alert.severity in (AlertSeverity.CRITICAL, AlertSeverity.HIGH):
                    self._event_stream.broadcast("alert", {
                        "severity": alert.severity.value,
                        "source": alert.source,
                        "message": alert.message,
                    })

        return sorted_alerts
