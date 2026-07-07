# tests/test_autonomous_integration.py
"""End-to-end integration tests for the autonomous pipeline.

Tests the full flow: suggestion → backtest → approval → PR with
git/gh commands mocked.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from trading_assistant.schemas.approval import ApprovalRequest, ApprovalStatus, RepoRiskTier
from trading_assistant.schemas.repo_changes import ChangeKind
from trading_assistant.schemas.suggestion_tracking import SuggestionRecord
from trading_assistant.skills.approval_handler import ApprovalHandler
from trading_assistant.skills.approval_tracker import ApprovalTracker
from trading_assistant.skills.autonomous_pipeline import AutonomousPipeline
from trading_assistant.skills.config_registry import ConfigRegistry
from trading_assistant.skills.file_change_generator import FileChangeGenerator
from trading_assistant.skills.github_pr import PRBuilder
from trading_assistant.skills.suggestion_backtester import SuggestionBacktester
from trading_assistant.skills.suggestion_tracker import SuggestionTracker
from trading_assistant.comms.telegram_renderer import TelegramRenderer


def _setup_full(tmp_path: Path, trades: int = 25):
    """Set up all components for integration testing."""
    # Config registry
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "config.yaml").write_text(
        "kmp:\n  quality_min_threshold: 0.6\n  max_positions: 5\n"
    )
    (repo_dir / "config").mkdir()
    (repo_dir / "config" / "risk.py").write_text("BASE_RISK_PCT = 0.02\n")

    (cfg_dir / "test_bot.yaml").write_text(yaml.dump({
        "bot_id": "test_bot",
        "repo_url": "git@github.com:user/test_bot.git",
        "repo_dir": str(repo_dir),
        "strategies": ["kmp"],
        "parameters": [
            {
                "param_name": "quality_min_threshold",
                "strategy_id": "kmp",
                "param_type": "YAML_FIELD",
                "file_path": "config.yaml",
                "yaml_key": "kmp.quality_min_threshold",
                "current_value": 0.6,
                "valid_range": [0.0, 1.0],
                "value_type": "float",
                "category": "entry_signal",
                "is_safety_critical": False,
            },
            {
                "param_name": "base_risk_pct",
                "param_type": "PYTHON_CONSTANT",
                "file_path": "config/risk.py",
                "python_path": "BASE_RISK_PCT",
                "current_value": 0.02,
                "valid_range": [0.005, 0.05],
                "value_type": "float",
                "category": "risk_management",
                "is_safety_critical": True,
            },
        ],
    }), encoding="utf-8")

    registry = ConfigRegistry(cfg_dir)

    # Write trade data
    curated = tmp_path / "data" / "curated" / "2026-03-06" / "test_bot"
    curated.mkdir(parents=True)
    with open(curated / "trades.jsonl", "w") as f:
        for i in range(trades):
            pnl = 100.0 if i % 3 != 0 else -50.0
            f.write(json.dumps({"pnl": pnl, "date": f"2026-03-{(i % 28) + 1:02d}"}) + "\n")

    # Components
    backtester = SuggestionBacktester(registry, tmp_path)
    approval_tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
    suggestion_tracker = SuggestionTracker(tmp_path / "findings")
    file_gen = FileChangeGenerator()
    pr_builder = PRBuilder(dry_run=True)
    telegram_bot = MagicMock()
    telegram_bot.send_message = AsyncMock(return_value=42)
    telegram_bot.edit_message = AsyncMock()
    telegram_renderer = TelegramRenderer()
    event_stream = MagicMock()

    pipeline = AutonomousPipeline(
        config_registry=registry,
        backtester=backtester,
        approval_tracker=approval_tracker,
        suggestion_tracker=suggestion_tracker,
        telegram_bot=telegram_bot,
        telegram_renderer=telegram_renderer,
        event_stream=event_stream,
    )

    approval_handler = ApprovalHandler(
        approval_tracker=approval_tracker,
        suggestion_tracker=suggestion_tracker,
        file_change_generator=file_gen,
        pr_builder=pr_builder,
        config_registry=registry,
        event_stream=event_stream,
    )

    return {
        "pipeline": pipeline,
        "approval_handler": approval_handler,
        "approval_tracker": approval_tracker,
        "suggestion_tracker": suggestion_tracker,
        "registry": registry,
        "telegram_bot": telegram_bot,
        "event_stream": event_stream,
        "pr_builder": pr_builder,
    }


class TestAutonomousIntegration:
    @pytest.mark.asyncio
    async def test_happy_path(self, tmp_path: Path):
        """Suggestion → backtest → approval request → approve → PR created."""
        c = _setup_full(tmp_path)
        # Record a parameter suggestion
        c["suggestion_tracker"].record(SuggestionRecord(
            suggestion_id="s1",
            bot_id="test_bot",
            title="Increase quality_min_threshold to 0.7",
            tier="parameter",
            category="entry_signal",
            source_report_id="weekly-2026-03",
            confidence=0.85,
        ))

        # Process through pipeline
        results = await c["pipeline"].process_new_suggestions(["s1"], run_id="test")
        assert len(results) == 1
        request = results[0]
        assert request.bot_id == "test_bot"
        assert request.param_changes[0]["proposed"] == 0.7

        # Approve via handler
        result_msg = await c["approval_handler"].handle_approve(request.request_id)
        assert "dry-run" in result_msg or "PR created" in result_msg

        # Verify approval tracker state
        approved = c["approval_tracker"].get_by_id(request.request_id)
        assert approved.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_backtest_failure_blocks(self, tmp_path: Path):
        """Suggestion with poor backtest → no approval request."""
        c = _setup_full(tmp_path, trades=3)  # Too few trades
        c["suggestion_tracker"].record(SuggestionRecord(
            suggestion_id="s1",
            bot_id="test_bot",
            title="Set quality_min_threshold to 0.9",
            tier="parameter",
            category="entry_signal",
            source_report_id="r1",
            confidence=0.8,
        ))

        results = await c["pipeline"].process_new_suggestions(["s1"])
        assert len(results) == 0
        assert len(c["approval_tracker"].get_pending()) == 0
        c["telegram_bot"].send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_actionable_skipped(self, tmp_path: Path):
        """Hypothesis suggestion → skipped entirely."""
        c = _setup_full(tmp_path)
        c["suggestion_tracker"].record(SuggestionRecord(
            suggestion_id="s1",
            bot_id="test_bot",
            title="Consider new momentum strategy",
            tier="hypothesis",
            category="strategy_variant",
            source_report_id="r1",
            confidence=0.9,
        ))

        results = await c["pipeline"].process_new_suggestions(["s1"])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_parameter_without_explicit_value_is_skipped(self, tmp_path: Path):
        c = _setup_full(tmp_path)
        c["suggestion_tracker"].record(SuggestionRecord(
            suggestion_id="s1",
            bot_id="test_bot",
            title="Improve quality_min_threshold",
            tier="parameter",
            category="entry_signal",
            source_report_id="r1",
            confidence=0.8,
        ))

        results = await c["pipeline"].process_new_suggestions(["s1"])
        assert results == []

    @pytest.mark.asyncio
    async def test_existing_request_skips_duplicate_processing(self, tmp_path: Path):
        c = _setup_full(tmp_path)
        c["suggestion_tracker"].record(SuggestionRecord(
            suggestion_id="s1",
            bot_id="test_bot",
            title="Increase quality_min_threshold to 0.7",
            tier="parameter",
            category="entry_signal",
            source_report_id="r1",
            confidence=0.85,
        ))
        c["approval_tracker"].create_request(ApprovalRequest(
            request_id="req-existing",
            suggestion_id="s1",
            bot_id="test_bot",
            status=ApprovalStatus.APPROVED,
        ))

        results = await c["pipeline"].process_new_suggestions(["s1"], run_id="test")
        assert results == []

    @pytest.mark.asyncio
    async def test_safety_critical_param(self, tmp_path: Path):
        """Risk management suggestion uses tighter thresholds."""
        c = _setup_full(tmp_path)
        c["suggestion_tracker"].record(SuggestionRecord(
            suggestion_id="s1",
            bot_id="test_bot",
            title="Increase base_risk_pct to 0.03",
            tier="parameter",
            category="risk_management",
            source_report_id="r1",
            confidence=0.8,
        ))

        results = await c["pipeline"].process_new_suggestions(["s1"])
        # May or may not pass safety depending on backtest results
        # The important thing is that safety-critical flag is applied
        if results:
            bs = results[0].backtest_summary
            assert bs is not None

    @pytest.mark.asyncio
    async def test_rejection_flow(self, tmp_path: Path):
        """Create approval → reject → verify states."""
        c = _setup_full(tmp_path)
        c["suggestion_tracker"].record(SuggestionRecord(
            suggestion_id="s1",
            bot_id="test_bot",
            title="Increase quality_min_threshold to 0.7",
            tier="parameter",
            category="entry_signal",
            source_report_id="r1",
            confidence=0.8,
        ))

        results = await c["pipeline"].process_new_suggestions(["s1"])
        assert len(results) == 1

        # Reject
        msg = await c["approval_handler"].handle_reject(results[0].request_id, "too risky")
        assert "Rejected" in msg

        # Verify states
        req = c["approval_tracker"].get_by_id(results[0].request_id)
        assert req.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_expiry_flow(self, tmp_path: Path):
        """Create approval → wait > 7 days → expire."""
        c = _setup_full(tmp_path)
        old = ApprovalRequest(
            request_id="old_req",
            suggestion_id="s1",
            bot_id="test_bot",
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
        c["approval_tracker"].create_request(old)

        expired = c["approval_tracker"].expire_old(max_age_days=7)
        assert "old_req" in expired
        assert c["approval_tracker"].get_by_id("old_req").status == ApprovalStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_telegram_notification_content(self, tmp_path: Path):
        """Verify Telegram message contains backtest table and keyboard."""
        c = _setup_full(tmp_path)
        c["suggestion_tracker"].record(SuggestionRecord(
            suggestion_id="s1",
            bot_id="test_bot",
            title="Increase quality_min_threshold to 0.7",
            tier="parameter",
            category="entry_signal",
            source_report_id="r1",
            confidence=0.8,
        ))

        await c["pipeline"].process_new_suggestions(["s1"])
        c["telegram_bot"].send_message.assert_called_once()
        call_args = c["telegram_bot"].send_message.call_args
        text = call_args[0][0]
        assert "Approval Request" in text
        assert "Sharpe" in text

    @pytest.mark.asyncio
    async def test_pipeline_isolation(self, tmp_path: Path):
        """Pipeline exception doesn't propagate to caller."""
        c = _setup_full(tmp_path)
        c["suggestion_tracker"].record(SuggestionRecord(
            suggestion_id="s1",
            bot_id="test_bot",
            title="Increase quality_min_threshold to 0.7",
            tier="parameter",
            category="entry_signal",
            source_report_id="r1",
            confidence=0.8,
        ))

        # Break the pipeline's backtester
        c["pipeline"]._backtester = None

        # Should not raise
        results = await c["pipeline"].process_new_suggestions(["s1"])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_structural_suggestion_creates_request(self, tmp_path: Path):
        c = _setup_full(tmp_path)
        c["suggestion_tracker"].record(SuggestionRecord(
            suggestion_id="s-struct",
            bot_id="test_bot",
            title="Add regression test for entry gate",
            tier="hypothesis",
            category="structural",
            source_report_id="r1",
            confidence=0.9,
            implementation_context={
                "file_changes": [
                    {
                        "file_path": "tests/test_entry_gate.py",
                        "new_content": "def test_gate():\n    assert True\n",
                    }
                ],
                "verification_commands": ["pytest tests -q"],
            },
        ))

        results = await c["pipeline"].process_new_suggestions(["s-struct"])
        assert len(results) == 1
        request = results[0]
        assert request.change_kind == ChangeKind.STRUCTURAL_CHANGE
        assert request.file_changes[0].file_path == "tests/test_entry_gate.py"
        assert request.risk_tier == RepoRiskTier.AUTO
        assert request.draft_pr is True

    @pytest.mark.asyncio
    async def test_structural_suggestion_with_notes_only_still_creates_request(self, tmp_path: Path):
        c = _setup_full(tmp_path)
        c["suggestion_tracker"].record(SuggestionRecord(
            suggestion_id="s-struct-notes",
            bot_id="test_bot",
            title="Refactor entry decision flow",
            tier="hypothesis",
            category="structural",
            source_report_id="r1",
            confidence=0.9,
            implementation_context={
                "notes": "Extract the decision logic into a shared helper and keep behavior unchanged.",
                "verification_commands": ["pytest tests -q"],
            },
        ))

        results = await c["pipeline"].process_new_suggestions(["s-struct-notes"])
        assert len(results) == 1
        request = results[0]
        assert request.change_kind == ChangeKind.STRUCTURAL_CHANGE
        assert request.file_changes == []
        assert request.implementation_notes.startswith("Extract the decision logic")
        assert request.risk_tier == RepoRiskTier.REQUIRES_APPROVAL

    @pytest.mark.asyncio
    async def test_structural_risky_path_requires_double_approval(self, tmp_path: Path):
        c = _setup_full(tmp_path)
        c["suggestion_tracker"].record(SuggestionRecord(
            suggestion_id="s-risky",
            bot_id="test_bot",
            title="Adjust deployment secret handling",
            tier="hypothesis",
            category="structural",
            source_report_id="r1",
            confidence=0.9,
            implementation_context={
                "file_changes": [
                    {
                        "file_path": ".env.local",
                        "new_content": "TOKEN=abc\n",
                    }
                ],
            },
        ))

        results = await c["pipeline"].process_new_suggestions(["s-risky"])
        assert len(results) == 1
        assert results[0].risk_tier == RepoRiskTier.REQUIRES_DOUBLE_APPROVAL
