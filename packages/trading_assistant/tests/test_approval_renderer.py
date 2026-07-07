# tests/test_approval_renderer.py
"""Tests for TelegramRenderer.render_approval_request."""
from __future__ import annotations

import pytest

from trading_assistant.schemas.approval import (
    ApprovalRequest,
    BacktestComparison,
    BacktestContext,
)
from trading_assistant.schemas.simulation_metrics import SimulationMetrics
from trading_assistant.comms.telegram_renderer import TelegramRenderer


def _make_approval_request(**kwargs) -> ApprovalRequest:
    comparison = BacktestComparison(
        context=BacktestContext(
            suggestion_id="s1", bot_id="test_bot",
            param_name="quality_min", current_value=0.6, proposed_value=0.7,
            trade_count=25, data_days=30,
        ),
        baseline=SimulationMetrics(
            sharpe_ratio=1.0, max_drawdown_pct=-10.0,
            profit_factor=1.5, total_trades=25, win_count=15,
        ),
        proposed=SimulationMetrics(
            sharpe_ratio=1.2, max_drawdown_pct=-8.0,
            profit_factor=1.8, total_trades=25, win_count=17,
        ),
        passes_safety=kwargs.get("passes_safety", True),
    )
    return ApprovalRequest(
        request_id=kwargs.get("request_id", "req_abc123"),
        suggestion_id="s1",
        bot_id="test_bot",
        param_changes=[{"param_name": "quality_min", "current": 0.6, "proposed": 0.7}],
        backtest_summary=comparison,
    )


@pytest.fixture
def renderer() -> TelegramRenderer:
    return TelegramRenderer()


class TestApprovalRenderer:
    def test_renders_all_sections(self, renderer: TelegramRenderer):
        text, keyboard = renderer.render_approval_request(_make_approval_request())
        assert "Approval Request" in text
        assert "test" in text and "bot" in text
        assert "quality" in text and "min" in text
        assert "Sharpe" in text
        assert "PASS" in text

    def test_inline_keyboard_callbacks(self, renderer: TelegramRenderer):
        _, keyboard = renderer.render_approval_request(
            _make_approval_request(request_id="req_xyz")
        )
        callbacks = [btn["callback_data"] for row in keyboard for btn in row]
        assert "approve_suggestion_req_xyz" in callbacks
        assert "reject_suggestion_req_xyz" in callbacks
        assert "detail_suggestion_req_xyz" in callbacks

    def test_backtest_table_formatted(self, renderer: TelegramRenderer):
        text, _ = renderer.render_approval_request(_make_approval_request())
        assert "Sharpe" in text
        assert "MaxDD" in text
        assert "ProfitFact" in text
        assert "WinRate" in text

    def test_safety_pass_indicator(self, renderer: TelegramRenderer):
        text, _ = renderer.render_approval_request(
            _make_approval_request(passes_safety=True)
        )
        assert "PASS" in text

    def test_safety_fail_indicator(self, renderer: TelegramRenderer):
        text, _ = renderer.render_approval_request(
            _make_approval_request(passes_safety=False)
        )
        assert "FAIL" in text

    def test_truncation_at_limit(self, renderer: TelegramRenderer):
        req = _make_approval_request()
        # Add very long param changes to trigger truncation
        req.param_changes = [
            {"param_name": f"param_{i}", "current": i, "proposed": i + 1}
            for i in range(500)
        ]
        text, _ = renderer.render_approval_request(req)
        assert len(text) <= 4096

    def test_handles_missing_backtest(self, renderer: TelegramRenderer):
        req = ApprovalRequest(
            request_id="r1", suggestion_id="s1", bot_id="bot1",
            param_changes=[{"param_name": "x", "current": 1, "proposed": 2}],
        )
        text, keyboard = renderer.render_approval_request(req)
        assert "Approval Request" in text
        assert len(keyboard) == 2  # approve/reject row + details row

    def test_md2_escaping_applied(self, renderer: TelegramRenderer):
        req = _make_approval_request()
        req.bot_id = "test_bot.v2"
        text, _ = renderer.render_approval_request(req)
        assert "test\\_bot\\.v2" in text


class TestCallbackPrefixRouting:
    """Tests for prefix-based callback routing (Bug 2 fix)."""

    @pytest.mark.asyncio
    async def test_exact_match_still_works(self):
        from trading_assistant.comms.telegram_handlers import TelegramCallbackRouter

        router = TelegramCallbackRouter()
        called_with = {}

        async def handler(**kwargs):
            called_with["called"] = True
            return "ok"

        router.register("cmd_pending", handler)
        result = await router.dispatch("cmd_pending")
        assert result == "ok"
        assert called_with["called"]

    @pytest.mark.asyncio
    async def test_prefix_match_extracts_suffix(self):
        from trading_assistant.comms.telegram_handlers import TelegramCallbackRouter

        router = TelegramCallbackRouter()
        received_id = {}

        async def handler(request_id: str):
            received_id["value"] = request_id
            return f"handled {request_id}"

        router.register("approve_suggestion_", handler)
        result = await router.dispatch("approve_suggestion_abc123")
        assert result == "handled abc123"
        assert received_id["value"] == "abc123"

    @pytest.mark.asyncio
    async def test_prefix_match_reject(self):
        from trading_assistant.comms.telegram_handlers import TelegramCallbackRouter

        router = TelegramCallbackRouter()

        async def handler(request_id: str):
            return f"rejected {request_id}"

        router.register("reject_suggestion_", handler)
        result = await router.dispatch("reject_suggestion_xyz789")
        assert result == "rejected xyz789"

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self):
        from trading_assistant.comms.telegram_handlers import TelegramCallbackRouter

        router = TelegramCallbackRouter()
        result = await router.dispatch("unknown_callback")
        assert result is None

    @pytest.mark.asyncio
    async def test_exact_takes_priority_over_prefix(self):
        from trading_assistant.comms.telegram_handlers import TelegramCallbackRouter

        router = TelegramCallbackRouter()

        async def exact_handler(**kwargs):
            return "exact"

        async def prefix_handler(suffix: str):
            return "prefix"

        router.register("approve_suggestion_special", exact_handler)
        router.register("approve_suggestion_", prefix_handler)
        result = await router.dispatch("approve_suggestion_special")
        assert result == "exact"

    @pytest.mark.asyncio
    async def test_pending_in_slash_map(self):
        from trading_assistant.comms.telegram_handlers import TelegramCallbackRouter

        router = TelegramCallbackRouter()

        async def handler(**kwargs):
            return "pending list"

        router.register("cmd_pending", handler)
        result = await router.dispatch_slash("/pending")
        assert result == "pending list"

    @pytest.mark.asyncio
    async def test_help_includes_pending(self):
        from trading_assistant.comms.telegram_handlers import TelegramCallbackRouter

        router = TelegramCallbackRouter()

        async def handler(**kwargs):
            return "pending"

        router.register("cmd_pending", handler)
        result = await router.dispatch_slash("/help")
        assert "/pending" in result
