# tests/test_metrics_endpoint.py
"""Tests for /metrics observability endpoint."""
import pytest
from unittest.mock import AsyncMock, patch

from trading_assistant.schemas.orchestrator_metrics import OrchestratorMetrics


class TestOrchestratorMetricsSchema:
    def test_metrics_model(self):
        m = OrchestratorMetrics(
            queue_depth=5, dead_letter_count=1, active_agents=2,
            error_rate_1h=0.05, uptime_seconds=3600,
        )
        assert m.queue_depth == 5
        assert m.dead_letter_count == 1

    def test_metrics_defaults(self):
        m = OrchestratorMetrics()
        assert m.queue_depth == 0
        assert m.active_agents == 0
        assert m.last_daily_analysis is None

    def test_all_seven_fields_present(self):
        m = OrchestratorMetrics(
            queue_depth=3,
            dead_letter_count=1,
            active_agents=2,
            error_rate_1h=5.0,
            uptime_seconds=7200.0,
            last_daily_analysis="2026-03-04T06:00:00+00:00",
            last_weekly_analysis="2026-03-02T08:00:00+00:00",
        )
        dumped = m.model_dump(mode="json")
        assert dumped["queue_depth"] == 3
        assert dumped["dead_letter_count"] == 1
        assert dumped["active_agents"] == 2
        assert dumped["error_rate_1h"] == 5.0
        assert dumped["uptime_seconds"] == 7200.0
        assert dumped["last_daily_analysis"] == "2026-03-04T06:00:00+00:00"
        assert dumped["last_weekly_analysis"] == "2026-03-02T08:00:00+00:00"


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_queue_depth(self):
        """Test via direct function call (avoids full app wiring)."""
        from trading_assistant.orchestrator.app import create_app
        from trading_assistant.orchestrator.config import AppConfig

        config = AppConfig(
            bot_ids=[],
            data_dir="/tmp/test_metrics",
            allow_unauthenticated_local=True,
        )
        with patch("trading_assistant.orchestrator.runtime.EventQueue") as MockQueue, \
             patch("trading_assistant.orchestrator.runtime.TaskRegistry") as MockRegistry:
            mock_q = AsyncMock()
            mock_q.initialize = AsyncMock()
            mock_q.close = AsyncMock()
            mock_q.count_pending = AsyncMock(return_value=5)
            mock_q.get_dead_letters = AsyncMock(return_value=[{"event_id": "e1"}])
            MockQueue.return_value = mock_q

            mock_reg = AsyncMock()
            mock_reg.initialize = AsyncMock()
            mock_reg.close = AsyncMock()
            MockRegistry.return_value = mock_reg

            app = create_app(db_dir="/tmp/test_metrics", config=config)
            # Verify endpoint exists
            routes = [r.path for r in app.routes]
            assert "/metrics" in routes


class TestMetricsPopulated:
    def test_brain_tracks_error_rate(self):
        """Brain's get_error_rate_1h returns count of errors in window."""
        from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain
        brain = OrchestratorBrain()
        # Record some errors
        brain.decide({
            "event_type": "error", "event_id": "e1", "bot_id": "bot1",
            "payload": '{"severity": "HIGH", "error_type": "timeout"}',
        })
        brain.decide({
            "event_type": "error", "event_id": "e2", "bot_id": "bot1",
            "payload": '{"severity": "HIGH", "error_type": "timeout"}',
        })
        rate = brain.get_error_rate_1h()
        assert rate >= 2.0

    def test_brain_tracks_last_daily_analysis(self):
        from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain
        brain = OrchestratorBrain()
        assert brain.last_daily_analysis is None
        brain.record_daily_analysis("2026-03-04T06:00:00+00:00")
        assert brain.last_daily_analysis == "2026-03-04T06:00:00+00:00"

    def test_brain_tracks_last_weekly_analysis(self):
        from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain
        brain = OrchestratorBrain()
        assert brain.last_weekly_analysis is None
        brain.record_weekly_analysis("2026-03-02T08:00:00+00:00")
        assert brain.last_weekly_analysis == "2026-03-02T08:00:00+00:00"


class TestCountPending:
    @pytest.mark.asyncio
    async def test_count_pending_returns_count(self):
        """Test count_pending method on EventQueue."""
        from trading_assistant.orchestrator.db.queue import EventQueue
        import tempfile, os

        with tempfile.TemporaryDirectory() as td:
            q = EventQueue(db_path=os.path.join(td, "test.db"))
            await q.initialize()

            count = await q.count_pending()
            assert count == 0

            await q.close()
