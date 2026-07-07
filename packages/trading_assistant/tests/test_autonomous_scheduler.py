# tests/test_autonomous_scheduler.py
"""Tests for scheduler integration with autonomous pipeline."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from trading_assistant.orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs
from trading_assistant.schemas.approval import ApprovalRequest
from trading_assistant.skills.approval_tracker import ApprovalTracker


class TestAutonomousScheduler:
    def test_expiry_job_registered_when_enabled(self):
        config = SchedulerConfig()
        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=AsyncMock(),
            monitoring_fn=AsyncMock(),
            relay_fn=AsyncMock(),
            approval_expiry_fn=AsyncMock(),
        )
        job_names = [j["name"] for j in jobs]
        assert "approval_expiry" in job_names

    def test_expiry_job_not_registered_when_disabled(self):
        config = SchedulerConfig()
        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=AsyncMock(),
            monitoring_fn=AsyncMock(),
            relay_fn=AsyncMock(),
        )
        job_names = [j["name"] for j in jobs]
        assert "approval_expiry" not in job_names

    def test_expired_requests_transition(self, tmp_path: Path):
        tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        old = ApprovalRequest(
            request_id="r_old",
            suggestion_id="s1",
            bot_id="bot1",
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
        tracker.create_request(old)
        expired = tracker.expire_old(max_age_days=7)
        assert "r_old" in expired
        from trading_assistant.schemas.approval import ApprovalStatus
        assert tracker.get_by_id("r_old").status == ApprovalStatus.EXPIRED

    def test_expiry_empty_tracker(self, tmp_path: Path):
        tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        expired = tracker.expire_old(max_age_days=7)
        assert expired == []

    @pytest.mark.asyncio
    async def test_expiry_with_notification(self, tmp_path: Path):
        """Expiry helper sends Telegram notification for each expired request."""
        from trading_assistant.orchestrator.app import _expire_approvals_with_notification

        tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        old = ApprovalRequest(
            request_id="r_old",
            suggestion_id="s1",
            bot_id="bot1",
            param_changes=[{"param_name": "quality_min", "current": 0.6, "proposed": 0.7}],
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
        tracker.create_request(old)

        telegram_bot = AsyncMock()
        telegram_bot.send_message = AsyncMock(return_value=100)
        telegram_bot.edit_message = AsyncMock()

        await _expire_approvals_with_notification(
            tracker, telegram_bot, None, None,
        )
        telegram_bot.send_message.assert_called_once()
        call_text = telegram_bot.send_message.call_args[0][0]
        assert "Expired" in call_text
        assert "r_old" in call_text

    @pytest.mark.asyncio
    async def test_expiry_edits_original_card(self, tmp_path: Path):
        """Expiry edits the original Telegram approval card if message_id exists."""
        from trading_assistant.orchestrator.app import _expire_approvals_with_notification

        tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        old = ApprovalRequest(
            request_id="r_old",
            suggestion_id="s1",
            bot_id="bot1",
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
            message_id=55,
        )
        tracker.create_request(old)

        telegram_bot = AsyncMock()
        telegram_bot.send_message = AsyncMock(return_value=100)
        telegram_bot.edit_message = AsyncMock()

        await _expire_approvals_with_notification(
            tracker, telegram_bot, None, None,
        )
        telegram_bot.edit_message.assert_called_once()
        edit_args = telegram_bot.edit_message.call_args
        assert edit_args[0][0] == 55
        assert "EXPIRED" in edit_args[0][1]

    def test_pr_review_job_registered(self):
        config = SchedulerConfig()
        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=AsyncMock(),
            monitoring_fn=AsyncMock(),
            relay_fn=AsyncMock(),
            pr_review_check_fn=AsyncMock(),
        )
        job_names = [j["name"] for j in jobs]
        assert "pr_review_check" in job_names

    def test_pr_review_job_not_registered_when_disabled(self):
        config = SchedulerConfig()
        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=AsyncMock(),
            monitoring_fn=AsyncMock(),
            relay_fn=AsyncMock(),
        )
        job_names = [j["name"] for j in jobs]
        assert "pr_review_check" not in job_names

    def test_pr_review_interval_matches_config(self):
        config = SchedulerConfig(pr_review_check_interval_seconds=1800)
        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=AsyncMock(),
            monitoring_fn=AsyncMock(),
            relay_fn=AsyncMock(),
            pr_review_check_fn=AsyncMock(),
        )
        pr_job = [j for j in jobs if j["name"] == "pr_review_check"][0]
        assert pr_job["seconds"] == 1800

    @pytest.mark.asyncio
    async def test_expiry_no_notification_without_bot(self, tmp_path: Path):
        """Expiry without telegram_bot just expires silently."""
        from trading_assistant.orchestrator.app import _expire_approvals_with_notification

        tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        old = ApprovalRequest(
            request_id="r_old",
            suggestion_id="s1",
            bot_id="bot1",
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
        tracker.create_request(old)

        await _expire_approvals_with_notification(
            tracker, None, None, None,
        )
        from trading_assistant.schemas.approval import ApprovalStatus
        assert tracker.get_by_id("r_old").status == ApprovalStatus.EXPIRED
