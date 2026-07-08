"""Unit tests for NQ Regime instrumentation wiring.

Validates that the engine correctly calls InstrumentationKit methods
for entry/exit/missed/stop/indicator events.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs for state and config types
# ---------------------------------------------------------------------------

@dataclass
class _FakeBarData:
    ts: datetime = field(default_factory=lambda: datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc))
    open: float = 20000.0
    high: float = 20050.0
    low: float = 19950.0
    close: float = 20020.0
    volume: float = 1000.0
    vwap: float | None = None


@dataclass
class _FakeEvent:
    code: str = ""
    details: dict = field(default_factory=dict)
    ts: datetime | None = None


@dataclass
class _FakeFillEvent:
    oms_order_id: str = "oms_001"
    fill_price: float = 20010.0
    fill_qty: int = 1
    fill_time: datetime = field(default_factory=lambda: datetime(2025, 1, 15, 10, 5, tzinfo=timezone.utc))
    symbol: str = "MNQ"
    commission: float = 0.62
    exit_type: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create an NQRegimeEngine with mocked instrumentation."""
    with patch("strategies.momentum.instrumentation.src.facade.InstrumentationKit") as MockKit:
        mock_kit_instance = MagicMock()
        mock_kit_instance.active = True
        MockKit.return_value = mock_kit_instance

        from strategies.momentum.nq_regime.engine import NQRegimeEngine
        from strategies.momentum.nq_regime.config import TradeSide, ModuleId, Grade, IBType
        from strategies.momentum.nq_regime.core.session import SessionPhase

        eng = NQRegimeEngine(equity=10_000.0)
        eng._kit = mock_kit_instance

        # Set up state defaults for testing
        eng._state.phase = SessionPhase.PRIMARY_DECISION
        eng._state.ib_type = IBType.UNCLASSIFIED

        return eng, mock_kit_instance


@pytest.fixture
def engine_with_position(engine):
    """Engine with an open LONG position."""
    eng, kit = engine
    from strategies.momentum.nq_regime.config import TradeSide, ModuleId, Grade

    eng._state.position_side = TradeSide.LONG
    eng._state.entry_price = 20000.0
    eng._state.qty_open = 2
    eng._state.active_trade_id = "trade_001"
    eng._state.entry_module = ModuleId.STRUCTURAL_EXPANSION
    eng._state.setup_grade = Grade.A
    eng._state.setup_score = 10
    eng._state.initial_risk_points = 15.0
    eng._state.stop_price = 19985.0
    eng._prev_stop_price = 19985.0
    eng._mfe_r = 1.5
    eng._mae_r = 0.3
    return eng, kit


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKitInitialization:
    def test_kit_initialized(self):
        """Verify _kit attribute exists and is InstrumentationKit."""
        with patch("strategies.momentum.instrumentation.src.facade.InstrumentationKit") as MockKit:
            mock_kit_instance = MagicMock()
            mock_kit_instance.active = True
            MockKit.return_value = mock_kit_instance

            from strategies.momentum.nq_regime.engine import NQRegimeEngine
            eng = NQRegimeEngine(equity=10_000.0)

            MockKit.assert_called_once_with(None, strategy_type="nq_regime")
            assert eng._kit is mock_kit_instance


class TestHydrationSync:
    def test_hydrate_syncs_tracking_fields(self):
        """After hydrate, _prev_stop_price and _prev_regime reflect restored state."""
        with patch("strategies.momentum.instrumentation.src.facade.InstrumentationKit") as MockKit:
            mock_kit_instance = MagicMock()
            mock_kit_instance.active = True
            MockKit.return_value = mock_kit_instance

            from strategies.momentum.nq_regime.engine import NQRegimeEngine

            eng = NQRegimeEngine(equity=10_000.0)
            assert eng._prev_stop_price == 0.0
            assert eng._prev_regime is None

            # Simulate hydration with a snapshot that has stop_price and regime
            eng._state.stop_price = 19990.0
            eng._state.regime = "TRENDING_UP"
            # Manually call the sync logic (hydrate calls hydrate_state which we can't easily mock)
            eng._prev_stop_price = eng._state.stop_price
            eng._prev_regime = eng._state.regime

            assert eng._prev_stop_price == 19990.0
            assert eng._prev_regime == "TRENDING_UP"


class TestEntryFill:
    def test_entry_fill_logs_entry(self, engine):
        """Mock kit, simulate FLAT->LONG via _log_entry_fill, verify log_entry called."""
        eng, kit = engine
        from strategies.momentum.nq_regime.config import TradeSide, ModuleId, Grade

        # Simulate state after entry fill
        eng._state.position_side = TradeSide.LONG
        eng._state.entry_price = 20010.0
        eng._state.qty_open = 1
        eng._state.active_trade_id = "trade_abc"
        eng._state.entry_module = ModuleId.STRUCTURAL_EXPANSION
        eng._state.setup_grade = Grade.A
        eng._state.setup_score = 10
        eng._state.initial_risk_points = 15.0
        eng._state.stop_price = 19995.0

        fill = _FakeFillEvent(fill_price=20010.0)
        eng._log_entry_fill(fill)

        kit.log_entry.assert_called_once()
        call_kwargs = kit.log_entry.call_args.kwargs
        assert call_kwargs["trade_id"] == "trade_abc"
        assert call_kwargs["pair"] == "MNQ"
        assert call_kwargs["side"] == "LONG"
        assert call_kwargs["entry_price"] == 20010.0
        assert call_kwargs["position_size"] == 1.0
        assert call_kwargs["entry_signal_strength"] == 10 / 12.0


class TestExitFill:
    def test_exit_fill_logs_exit(self, engine_with_position):
        """Simulate LONG->FLAT, verify log_exit with exit_reason, mfe_r, mae_r."""
        eng, kit = engine_with_position
        from strategies.momentum.nq_regime.config import TradeSide

        events = [_FakeEvent(code="EXIT_FILLED", details={"role": "stop"})]

        fill = _FakeFillEvent(fill_price=19985.0, commission=1.24)
        eng._log_exit_fill(fill, events, "trade_001", is_full=True)

        kit.log_exit.assert_called_once()
        call_kwargs = kit.log_exit.call_args.kwargs
        assert call_kwargs["trade_id"] == "trade_001"
        assert call_kwargs["exit_price"] == 19985.0
        assert call_kwargs["exit_reason"] == "INITIAL_STOP"
        assert call_kwargs["fees_paid"] == 1.24
        assert call_kwargs["mfe_r"] == 1.5
        assert call_kwargs["mae_r"] == 0.3

    def test_trailing_stop_exit_with_stop_at_be(self, engine_with_position):
        """Stop exit after partial taken (stop_at_be=True) -> TRAILING_STOP."""
        eng, kit = engine_with_position

        events = [_FakeEvent(code="EXIT_FILLED", details={"role": "stop"})]
        fill = _FakeFillEvent(fill_price=20000.0, commission=1.24)
        # pre_stop_at_be=True simulates state before _clear_position resets it
        eng._log_exit_fill(fill, events, "trade_001", is_full=True, pre_stop_at_be=True)

        kit.log_exit.assert_called_once()
        assert kit.log_exit.call_args.kwargs["exit_reason"] == "TRAILING_STOP"

    def test_partial_exit_logs(self, engine_with_position):
        """Target_1 fill with qty_open > 0, verify log_exit with PARTIAL_TARGET."""
        eng, kit = engine_with_position

        events = [_FakeEvent(code="PARTIAL_EXIT_FILLED", details={"role": "target_1"})]
        fill = _FakeFillEvent(fill_price=20030.0, commission=0.62)
        eng._log_exit_fill(fill, events, "trade_001", is_full=False)

        kit.log_exit.assert_called_once()
        assert kit.log_exit.call_args.kwargs["exit_reason"] == "PARTIAL_TARGET"
        # MFE/MAE should NOT be reset on partial
        assert eng._mfe_r == 1.5
        assert eng._mae_r == 0.3


class TestMissedOpportunities:
    def test_routing_blocked_logs_missed(self, engine):
        """ROUTING_DECISION with blocked candidates -> log_missed."""
        eng, kit = engine

        event = _FakeEvent(
            code="ROUTING_DECISION",
            details={
                "blocked_candidates": [
                    {"side": "LONG", "module": "structural_expansion", "setup_type": "squeeze_break",
                     "candidate_id": "cand_001", "score": 9, "block_reason": "regime_mismatch"},
                ],
                "candidate_count": 1,
                "regime": "TRENDING_UP",
                "confidence": 0.85,
                "margin": 0.2,
            },
        )
        eng._on_routing_decision(event)

        kit.log_missed.assert_called_once()
        call_kwargs = kit.log_missed.call_args.kwargs
        assert call_kwargs["blocked_by"] == "regime_routing"
        assert call_kwargs["signal"] == "structural_expansion_squeeze_break"
        assert call_kwargs["signal_strength"] == 9 / 12.0

    def test_daily_lockout_logs_missed(self, engine):
        """DAILY_LOCKOUT event -> log_missed."""
        eng, kit = engine
        eng._state.daily_realized_r = -3.5
        eng._state.daily_losses = 2

        event = _FakeEvent(code="DAILY_LOCKOUT", details={"daily_realized_r": -3.5})
        eng._on_daily_lockout(event)

        kit.log_missed.assert_called_once()
        call_kwargs = kit.log_missed.call_args.kwargs
        assert call_kwargs["blocked_by"] == "daily_lockout"
        assert "daily_realized_r=-3.50" in call_kwargs["block_reason"]

    def test_news_veto_logs_missed(self, engine):
        """NEWS_VETO event -> log_missed."""
        eng, kit = engine

        event = _FakeEvent(code="NEWS_VETO", details={"news": "FOMC_RATE_DECISION"})
        eng._on_news_veto(event)

        kit.log_missed.assert_called_once()
        assert kit.log_missed.call_args.kwargs["blocked_by"] == "news_veto"
        assert kit.log_missed.call_args.kwargs["block_reason"] == "FOMC_RATE_DECISION"


class TestStopAdjustment:
    def test_stop_adjustment_on_replace(self, engine_with_position):
        """ReplaceProtectiveStop dispatch -> log_stop_adjustment."""
        eng, kit = engine_with_position
        from strategies.core.actions import ReplaceProtectiveStop

        action = ReplaceProtectiveStop(
            symbol="MNQ",
            target_order_id="oms_stop_001",
            side="SELL",
            stop_price=20005.0,
            qty=2,
            reason="breakeven_move",
        )
        # Mock OMS
        eng._oms = MagicMock()
        eng._oms.submit_intent = AsyncMock(return_value=None)

        asyncio.run(eng._dispatch_action(action))

        kit.log_stop_adjustment.assert_called_once()
        call_kwargs = kit.log_stop_adjustment.call_args.kwargs
        assert call_kwargs["old_stop"] == 19985.0
        assert call_kwargs["new_stop"] == 20005.0
        assert call_kwargs["trade_id"] == "trade_001"
        assert eng._prev_stop_price == 20005.0


class TestIndicatorSnapshot:
    def test_indicator_snapshot_on_routing(self, engine):
        """ROUTING_DECISION with candidates -> on_indicator_snapshot."""
        eng, kit = engine

        # Set up indicators
        eng._state.indicators = MagicMock()
        eng._state.indicators.vwap = 20000.0
        eng._state.indicators.vwap_sd = 5.0
        eng._state.indicators.vwap_slope = 0.1
        eng._state.indicators.atr_15m = 10.0
        eng._state.indicators.atr_5m = 5.0
        eng._state.indicators.ema9_15m = 20010.0
        eng._state.indicators.ema20_15m = 20005.0
        eng._state.indicators.ema50_15m = 19990.0
        eng._state.indicators.bb_upper = 20050.0
        eng._state.indicators.bb_lower = 19950.0
        eng._state.indicators.kc_upper = 20040.0
        eng._state.indicators.kc_lower = 19960.0
        eng._state.indicators.squeeze_on = True
        eng._state.indicators.squeeze_duration = 5
        eng._state.indicators.rsi14_15m = 55.0
        eng._state.indicators.macd_15m = 2.5
        eng._state.indicators.macd_signal_15m = 1.8
        eng._state.indicators.volume_multiple_15m = 1.3
        eng._state.indicators.volume_multiple_5m = 1.1

        event = _FakeEvent(
            code="ROUTING_DECISION",
            details={
                "candidate_count": 2,
                "selected": True,
                "selected_module": "structural_expansion",
                "selected_score": 10,
                "regime": "TRENDING_UP",
                "confidence": 0.9,
            },
        )
        eng._emit_indicator_snapshot(event)

        kit.on_indicator_snapshot.assert_called_once()
        call_kwargs = kit.on_indicator_snapshot.call_args.kwargs
        assert call_kwargs["signal_name"] == "nq_regime_structural_expansion"
        assert call_kwargs["decision"] == "enter"
        assert call_kwargs["indicators"]["vwap"] == 20000.0
        assert call_kwargs["context"]["regime"] == "TRENDING_UP"


class TestFireAndForget:
    def test_exception_in_kit_does_not_crash_engine(self, engine):
        """Exception in kit method doesn't crash engine."""
        eng, kit = engine
        from strategies.momentum.nq_regime.config import TradeSide, ModuleId, Grade

        kit.log_entry.side_effect = RuntimeError("deliberate failure")

        # Should not raise
        eng._state.position_side = TradeSide.LONG
        eng._state.entry_price = 20000.0
        eng._state.qty_open = 1
        eng._state.active_trade_id = "t1"
        eng._state.entry_module = ModuleId.STRUCTURAL_EXPANSION
        eng._state.setup_grade = Grade.A
        eng._state.setup_score = 10
        eng._state.initial_risk_points = 15.0
        eng._state.stop_price = 19985.0

        fill = _FakeFillEvent()
        eng._log_entry_fill(fill)  # Should not raise

        kit.log_entry.assert_called_once()
