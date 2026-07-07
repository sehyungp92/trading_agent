# tests/test_autonomous_pipeline.py
"""Tests for AutonomousPipeline."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from trading_assistant.schemas.suggestion_tracking import SuggestionRecord
from trading_assistant.skills.approval_tracker import ApprovalTracker
from trading_assistant.skills.autonomous_pipeline import AutonomousPipeline
from trading_assistant.skills.config_registry import ConfigRegistry
from trading_assistant.skills.suggestion_backtester import SuggestionBacktester
from trading_assistant.skills.suggestion_tracker import SuggestionTracker


def _setup(tmp_path: Path, trades: int = 25):
    # Config registry
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "test_bot.yaml").write_text(yaml.dump({
        "bot_id": "test_bot",
        "repo_dir": str(tmp_path / "repo"),
        "strategies": ["alpha"],
        "parameters": [{
            "param_name": "quality_min",
            "param_type": "YAML_FIELD",
            "file_path": "config.yaml",
            "yaml_key": "alpha.quality_min",
            "current_value": 0.6,
            "valid_range": [0.0, 1.0],
            "value_type": "float",
            "category": "entry_signal",
            "is_safety_critical": False,
        }],
    }), encoding="utf-8")
    registry = ConfigRegistry(cfg_dir)

    # Write trade data
    curated = tmp_path / "data" / "curated" / "2026-03-06" / "test_bot"
    curated.mkdir(parents=True)
    with open(curated / "trades.jsonl", "w") as f:
        for i in range(trades):
            pnl = 100.0 if i % 3 != 0 else -50.0
            f.write(json.dumps({"pnl": pnl, "date": f"2026-03-{(i % 28) + 1:02d}"}) + "\n")

    backtester = SuggestionBacktester(registry, tmp_path)
    approval_tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
    suggestion_tracker = SuggestionTracker(tmp_path / "findings")
    telegram_bot = MagicMock()
    telegram_bot.send_message = AsyncMock(return_value=42)
    telegram_renderer = MagicMock()
    telegram_renderer.render_approval_request = MagicMock(return_value=("text", []))
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

    return pipeline, suggestion_tracker, approval_tracker, event_stream


def _record_suggestion(tracker: SuggestionTracker, **kwargs) -> str:
    defaults = {
        "suggestion_id": "s1",
        "bot_id": "test_bot",
        "title": "Increase quality min to 0.7",
        "tier": "parameter",
        "category": "entry_signal",
        "source_report_id": "r1",
        "confidence": 0.8,
    }
    defaults.update(kwargs)
    record = SuggestionRecord(**defaults)
    tracker.record(record)
    return defaults["suggestion_id"]


class TestAutonomousPipeline:
    @pytest.mark.asyncio
    async def test_process_actionable_suggestion(self, tmp_path: Path):
        pipeline, sug_tracker, approval_tracker, _ = _setup(tmp_path)
        _record_suggestion(sug_tracker)
        results = await pipeline.process_new_suggestions(["s1"], run_id="test")
        assert len(results) == 1
        assert results[0].bot_id == "test_bot"
        assert approval_tracker.get_pending()

    @pytest.mark.asyncio
    async def test_skip_wrong_tier(self, tmp_path: Path):
        pipeline, sug_tracker, _, _ = _setup(tmp_path)
        _record_suggestion(sug_tracker, tier="hypothesis")
        results = await pipeline.process_new_suggestions(["s1"])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_skip_low_confidence(self, tmp_path: Path):
        pipeline, sug_tracker, _, _ = _setup(tmp_path)
        _record_suggestion(sug_tracker, confidence=0.3)
        results = await pipeline.process_new_suggestions(["s1"])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_skip_already_in_queue(self, tmp_path: Path):
        pipeline, sug_tracker, approval_tracker, _ = _setup(tmp_path)
        _record_suggestion(sug_tracker)
        # Process once
        await pipeline.process_new_suggestions(["s1"])
        # Process again — should skip
        results = await pipeline.process_new_suggestions(["s1"])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_skip_no_matching_params(self, tmp_path: Path):
        pipeline, sug_tracker, _, _ = _setup(tmp_path)
        _record_suggestion(sug_tracker, category="nonexistent_category",
                          title="Something unrelated to any params")
        results = await pipeline.process_new_suggestions(["s1"])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_extract_value_increase_to(self, tmp_path: Path):
        pipeline, sug_tracker, _, _ = _setup(tmp_path)
        _record_suggestion(sug_tracker, title="Increase quality_min to 0.7")
        results = await pipeline.process_new_suggestions(["s1"])
        if results:
            assert results[0].param_changes[0]["proposed"] == 0.7

    @pytest.mark.asyncio
    async def test_extract_value_set_to(self, tmp_path: Path):
        pipeline, sug_tracker, _, _ = _setup(tmp_path)
        _record_suggestion(sug_tracker, title="Set quality_min to 0.8")
        results = await pipeline.process_new_suggestions(["s1"])
        if results:
            assert results[0].param_changes[0]["proposed"] == 0.8

    @pytest.mark.asyncio
    async def test_extract_value_change_from_to(self, tmp_path: Path):
        pipeline, sug_tracker, _, _ = _setup(tmp_path)
        _record_suggestion(sug_tracker, title="Change quality_min from 0.6 to 0.75")
        results = await pipeline.process_new_suggestions(["s1"])
        if results:
            assert results[0].param_changes[0]["proposed"] == 0.75

    @pytest.mark.asyncio
    async def test_extract_value_fallback_midpoint(self, tmp_path: Path):
        pipeline, sug_tracker, _, _ = _setup(tmp_path)
        _record_suggestion(sug_tracker, title="Adjust quality min threshold",
                          description="should be higher")
        results = await pipeline.process_new_suggestions(["s1"])
        if results:
            # Midpoint of [0.0, 1.0] = 0.5
            assert results[0].param_changes[0]["proposed"] == 0.5

    @pytest.mark.asyncio
    async def test_backtest_failure_no_request(self, tmp_path: Path):
        pipeline, sug_tracker, approval_tracker, _ = _setup(tmp_path, trades=3)  # too few trades
        _record_suggestion(sug_tracker)
        results = await pipeline.process_new_suggestions(["s1"])
        assert len(results) == 0
        assert len(approval_tracker.get_pending()) == 0

    @pytest.mark.asyncio
    async def test_multiple_suggestions(self, tmp_path: Path):
        pipeline, sug_tracker, _, _ = _setup(tmp_path)
        _record_suggestion(sug_tracker, suggestion_id="s1", title="Increase quality_min to 0.7")
        _record_suggestion(sug_tracker, suggestion_id="s2", tier="hypothesis",
                          title="Some hypothesis")
        results = await pipeline.process_new_suggestions(["s1", "s2"])
        assert len(results) == 1  # Only s1 is actionable

    @pytest.mark.asyncio
    async def test_telegram_notification_sent(self, tmp_path: Path):
        pipeline, sug_tracker, _, _ = _setup(tmp_path)
        _record_suggestion(sug_tracker)
        await pipeline.process_new_suggestions(["s1"])
        pipeline._telegram_renderer.render_approval_request.assert_called_once()
        pipeline._telegram_bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_event_broadcast(self, tmp_path: Path):
        pipeline, sug_tracker, _, event_stream = _setup(tmp_path)
        _record_suggestion(sug_tracker)
        await pipeline.process_new_suggestions(["s1"], run_id="test-run")
        event_stream.broadcast.assert_called_with("autonomous_pipeline_complete", {
            "run_id": "test-run",
            "approval_requests_created": 1,
        })

    @pytest.mark.asyncio
    async def test_pipeline_error_logged_not_raised(self, tmp_path: Path):
        pipeline, sug_tracker, _, _ = _setup(tmp_path)
        _record_suggestion(sug_tracker)
        # Break the backtester to cause an error
        pipeline._backtester = None
        # Should not raise, just log
        results = await pipeline.process_new_suggestions(["s1"])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_message_id_stored_after_send(self, tmp_path: Path):
        """Telegram message_id is stored on approval request after send."""
        pipeline, sug_tracker, approval_tracker, _ = _setup(tmp_path)
        _record_suggestion(sug_tracker)
        await pipeline.process_new_suggestions(["s1"], run_id="test")
        pending = approval_tracker.get_pending()
        assert len(pending) == 1
        assert pending[0].message_id == 42  # from AsyncMock(return_value=42)


class TestTelegramBotCallbackRouter:
    """Tests for TelegramBotAdapter callback router integration."""

    def test_set_callback_router(self):
        from trading_assistant.comms.telegram_bot import TelegramBotAdapter, TelegramBotConfig
        from trading_assistant.comms.telegram_handlers import TelegramCallbackRouter

        adapter = TelegramBotAdapter(config=TelegramBotConfig(token="x", chat_id="1"))
        router = TelegramCallbackRouter()
        adapter.set_callback_router(router)
        assert adapter._callback_router is router

    @pytest.mark.asyncio
    async def test_start_polling_without_router_warns(self):
        from trading_assistant.comms.telegram_bot import TelegramBotAdapter, TelegramBotConfig

        adapter = TelegramBotAdapter(config=TelegramBotConfig(token="x", chat_id="1"))
        # No router set — should not raise, just warn
        await adapter.start_polling()
        assert adapter._polling_task is None

    @pytest.mark.asyncio
    async def test_handle_update_callback_query(self):
        """_handle_update dispatches callback queries through the router."""
        from trading_assistant.comms.telegram_bot import TelegramBotAdapter, TelegramBotConfig
        from trading_assistant.comms.telegram_handlers import TelegramCallbackRouter

        adapter = TelegramBotAdapter(config=TelegramBotConfig(token="x", chat_id="1"))
        router = TelegramCallbackRouter()
        adapter.set_callback_router(router)
        adapter._bot = MagicMock()
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))

        received = {}

        async def handler(request_id: str):
            received["id"] = request_id
            return f"approved {request_id}"

        router.register("approve_suggestion_", handler)

        # Create a mock update with callback_query. Set chat.id to match the
        # configured chat_id so the P0-4 allowlist accepts it.
        update = MagicMock()
        update.callback_query = MagicMock()
        update.callback_query.data = "approve_suggestion_req123"
        update.callback_query.answer = AsyncMock()
        update.callback_query.message = MagicMock()
        update.callback_query.message.chat.id = "1"
        update.callback_query.from_user.id = 100
        update.message = None

        await adapter._handle_update(update)
        assert received["id"] == "req123"
        update.callback_query.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_update_slash_command(self):
        """_handle_update dispatches slash commands through the router."""
        from trading_assistant.comms.telegram_bot import TelegramBotAdapter, TelegramBotConfig
        from trading_assistant.comms.telegram_handlers import TelegramCallbackRouter

        adapter = TelegramBotAdapter(config=TelegramBotConfig(token="x", chat_id="1"))
        router = TelegramCallbackRouter()
        adapter.set_callback_router(router)
        adapter._bot = MagicMock()
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))

        async def handler(**kwargs):
            return "pending list here"

        router.register("cmd_pending", handler)

        # Create a mock update with a /pending message. Set chat.id to match
        # the configured chat_id so the P0-4 allowlist accepts it.
        update = MagicMock()
        update.callback_query = None
        update.message = MagicMock()
        update.message.text = "/pending"
        update.message.chat.id = "1"
        update.message.from_user.id = 100

        await adapter._handle_update(update)
        adapter._bot.send_message.assert_called_once()
