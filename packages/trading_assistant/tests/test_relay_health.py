"""Tests for relay health monitoring (#27).

These test the MonitoringCheck.check_relay_health() method using a fake
HTTP relay (no dependency on trading/ibkr_trader/apps/relay/).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from trading_assistant.orchestrator.latency_tracker import LatencyTracker
from trading_assistant.orchestrator.monitoring import AlertSeverity, MonitoringCheck


# ---------------------------------------------------------------------------
# Fake relay /health
# ---------------------------------------------------------------------------

def _make_health_app(response: dict | None = None, status_code: int = 200):
    """Build a minimal Starlette app that serves a fixed /health response."""

    async def health(request: Request):
        if status_code != 200:
            return PlainTextResponse("error", status_code=status_code)
        return JSONResponse(response or {})

    return Starlette(routes=[Route("/health", health)])


def _client_factory_for(app) -> callable:
    transport = ASGITransport(app=app)
    return lambda: AsyncClient(transport=transport, base_url="http://relay")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckRelayHealth:
    async def test_unreachable_produces_critical(self):
        """Relay unreachable ??CRITICAL alert."""
        def bad_factory():
            return AsyncClient(base_url="http://localhost:1", timeout=0.1)

        check = MonitoringCheck(
            relay_url="http://localhost:1",
            _relay_client_factory=bad_factory,
        )
        alerts = await check.check_relay_health()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert "unreachable" in alerts[0].message.lower()

    async def test_healthy_relay_no_alerts(self):
        """Healthy relay with small DB and recent events ??no alerts."""
        now = datetime.now(timezone.utc).isoformat()
        app = _make_health_app({
            "status": "ok",
            "pending_events": 2,
            "per_bot_pending": {"bot-a": 2},
            "last_event_per_bot": {"bot-a": now},
            "oldest_pending_age_seconds": 10.0,
            "db_size_bytes": 1024,
            "uptime_seconds": 100.0,
        })
        check = MonitoringCheck(
            relay_url="http://relay",
            _relay_client_factory=_client_factory_for(app),
        )
        alerts = await check.check_relay_health()
        assert alerts == []

    async def test_disk_exceeded_produces_high(self):
        """DB size over threshold ??HIGH alert."""
        now = datetime.now(timezone.utc).isoformat()
        app = _make_health_app({
            "status": "ok",
            "pending_events": 0,
            "per_bot_pending": {},
            "last_event_per_bot": {"bot-a": now},
            "oldest_pending_age_seconds": 0.0,
            "db_size_bytes": 600_000_000,  # 600 MB, threshold default 500 MB
            "uptime_seconds": 100.0,
        })
        check = MonitoringCheck(
            relay_url="http://relay",
            _relay_client_factory=_client_factory_for(app),
        )
        alerts = await check.check_relay_health()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.HIGH
        assert "DB size" in alerts[0].message

    async def test_oldest_pending_age_produces_high(self):
        """Old pending relay events are part of operational monitoring."""
        now = datetime.now(timezone.utc).isoformat()
        app = _make_health_app({
            "status": "ok",
            "pending_events": 4,
            "per_bot_pending": {"bot-a": 4},
            "last_event_per_bot": {"bot-a": now},
            "oldest_pending_age_seconds": 3600.0,
            "db_size_bytes": 1024,
            "uptime_seconds": 100.0,
        })
        check = MonitoringCheck(
            relay_url="http://relay",
            _relay_client_factory=_client_factory_for(app),
        )
        alerts = await check.check_relay_health()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.HIGH
        assert "oldest pending" in alerts[0].message
        assert "4 pending" in alerts[0].message

    async def test_bot_silence_produces_high(self):
        """Bot with no events for >6h ??HIGH alert."""
        old = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
        app = _make_health_app({
            "status": "ok",
            "pending_events": 0,
            "per_bot_pending": {},
            "last_event_per_bot": {"bot-a": old},
            "oldest_pending_age_seconds": 0.0,
            "db_size_bytes": 1024,
            "uptime_seconds": 100.0,
        })
        check = MonitoringCheck(
            relay_url="http://relay",
            _relay_client_factory=_client_factory_for(app),
        )
        alerts = await check.check_relay_health()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.HIGH
        assert "bot-a" in alerts[0].message

    async def test_silence_within_threshold_no_alert(self):
        """Bot with event within 6h ??no alert."""
        recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        app = _make_health_app({
            "status": "ok",
            "pending_events": 0,
            "per_bot_pending": {},
            "last_event_per_bot": {"bot-a": recent},
            "oldest_pending_age_seconds": 0.0,
            "db_size_bytes": 1024,
            "uptime_seconds": 100.0,
        })
        check = MonitoringCheck(
            relay_url="http://relay",
            _relay_client_factory=_client_factory_for(app),
        )
        alerts = await check.check_relay_health()
        assert alerts == []

    async def test_latency_p95_exceeded_produces_high(self):
        """Latency p95 > threshold ??HIGH alert."""
        tracker = LatencyTracker()
        # Record a high-latency event (2h delay)
        ex = "2026-03-01T12:00:00+00:00"
        rx = "2026-03-01T14:00:00+00:00"
        tracker.record("bot-a", ex, rx)

        now = datetime.now(timezone.utc).isoformat()
        app = _make_health_app({
            "status": "ok",
            "pending_events": 0,
            "per_bot_pending": {},
            "last_event_per_bot": {"bot-a": now},
            "oldest_pending_age_seconds": 0.0,
            "db_size_bytes": 1024,
            "uptime_seconds": 100.0,
        })
        check = MonitoringCheck(
            relay_url="http://relay",
            latency_tracker=tracker,
            _relay_client_factory=_client_factory_for(app),
        )
        alerts = await check.check_relay_health()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.HIGH
        assert "latency" in alerts[0].message.lower()

    async def test_latency_within_threshold_no_alert(self):
        """Latency p95 within threshold ??no alert."""
        tracker = LatencyTracker()
        tracker.record("bot-a", "2026-03-01T12:00:00+00:00", "2026-03-01T12:00:10+00:00")

        now = datetime.now(timezone.utc).isoformat()
        app = _make_health_app({
            "status": "ok",
            "pending_events": 0,
            "per_bot_pending": {},
            "last_event_per_bot": {"bot-a": now},
            "oldest_pending_age_seconds": 0.0,
            "db_size_bytes": 1024,
            "uptime_seconds": 100.0,
        })
        check = MonitoringCheck(
            relay_url="http://relay",
            latency_tracker=tracker,
            _relay_client_factory=_client_factory_for(app),
        )
        alerts = await check.check_relay_health()
        assert alerts == []

    async def test_no_relay_url_skips(self):
        """Empty relay_url ??no checks, no alerts."""
        check = MonitoringCheck(relay_url="")
        alerts = await check.check_relay_health()
        assert alerts == []

    async def test_client_factory_used(self):
        """Verify _relay_client_factory is called instead of creating default client."""
        factory_called = False
        datetime.now(timezone.utc).isoformat()
        app = _make_health_app({
            "status": "ok",
            "pending_events": 0,
            "per_bot_pending": {},
            "last_event_per_bot": {},
            "oldest_pending_age_seconds": 0.0,
            "db_size_bytes": 0,
            "uptime_seconds": 0.0,
        })
        transport = ASGITransport(app=app)

        def tracking_factory():
            nonlocal factory_called
            factory_called = True
            return AsyncClient(transport=transport, base_url="http://relay")

        check = MonitoringCheck(
            relay_url="http://relay",
            _relay_client_factory=tracking_factory,
        )
        await check.check_relay_health()
        assert factory_called
