"""Tests for heartbeat.md system status updates (A3)."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from trading_assistant.orchestrator.monitoring import MonitoringCheck


@pytest.fixture
def tmp_heartbeat(tmp_path):
    hb_dir = tmp_path / "heartbeats"
    hb_dir.mkdir()
    md_path = tmp_path / "memory" / "heartbeat.md"
    return hb_dir, md_path


@pytest.mark.asyncio
async def test_heartbeat_md_written_with_correct_format(tmp_heartbeat):
    hb_dir, md_path = tmp_heartbeat
    now = datetime.now(timezone.utc)
    (hb_dir / "bot_alpha.heartbeat").write_text(now.isoformat())

    check = MonitoringCheck(
        heartbeat_dir=str(hb_dir),
        heartbeat_md_path=str(md_path),
    )
    await check.update_heartbeat_md()

    content = md_path.read_text()
    assert "# System Heartbeat" in content
    assert "Last check:" in content
    assert "## Bot Status" in content
    assert "## Queue" in content
    assert "## Last Analyses" in content


@pytest.mark.asyncio
async def test_bot_status_shows_healthy_and_stale(tmp_heartbeat):
    hb_dir, md_path = tmp_heartbeat
    now = datetime.now(timezone.utc)
    # Healthy bot (recent heartbeat)
    (hb_dir / "bot_alpha.heartbeat").write_text(now.isoformat())
    # Stale bot (old heartbeat)
    stale_time = now - timedelta(hours=5)
    (hb_dir / "bot_beta.heartbeat").write_text(stale_time.isoformat())

    check = MonitoringCheck(
        heartbeat_dir=str(hb_dir),
        heartbeat_max_age_seconds=7200,
        heartbeat_md_path=str(md_path),
    )
    await check.update_heartbeat_md()

    content = md_path.read_text()
    assert "bot_alpha: healthy" in content
    assert "bot_beta: STALE" in content


@pytest.mark.asyncio
async def test_queue_stats_included(tmp_heartbeat):
    hb_dir, md_path = tmp_heartbeat
    queue = AsyncMock()
    queue.count_pending = AsyncMock(return_value=3)
    queue.count_dead_letters = AsyncMock(return_value=1)

    check = MonitoringCheck(
        heartbeat_dir=str(hb_dir),
        queue=queue,
        heartbeat_md_path=str(md_path),
    )
    await check.update_heartbeat_md()

    content = md_path.read_text()
    assert "Pending events: 3" in content
    assert "Dead letters: 1" in content


@pytest.mark.asyncio
async def test_last_analysis_timestamps_included(tmp_heartbeat):
    hb_dir, md_path = tmp_heartbeat
    brain = MagicMock()
    brain.last_daily_analysis = "2026-03-04T22:30:00+00:00"
    brain.last_weekly_analysis = "2026-03-02T10:00:00+00:00"

    check = MonitoringCheck(
        heartbeat_dir=str(hb_dir),
        brain=brain,
        heartbeat_md_path=str(md_path),
    )
    await check.update_heartbeat_md()

    content = md_path.read_text()
    assert "Daily: 2026-03-04" in content
    assert "Weekly: 2026-03-02" in content


@pytest.mark.asyncio
async def test_handles_missing_heartbeat_dir_gracefully(tmp_path):
    md_path = tmp_path / "memory" / "heartbeat.md"
    check = MonitoringCheck(
        heartbeat_dir=str(tmp_path / "nonexistent"),
        heartbeat_md_path=str(md_path),
    )
    await check.update_heartbeat_md()

    content = md_path.read_text()
    assert "No heartbeat data available" in content


@pytest.mark.asyncio
async def test_skips_when_no_md_path():
    check = MonitoringCheck()
    # Should not raise
    await check.update_heartbeat_md()
