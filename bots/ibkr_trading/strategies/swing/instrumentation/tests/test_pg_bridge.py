"""Tests for PG-Instrumentation bridge (InstrumentedTradeRecorder).

Tests verify that:
1. PG writes always succeed and return trade_id even if instrumentation fails
2. Both inner recorder and kit are called for entry and exit
3. Instrumentation exceptions are swallowed and logged, not propagated
"""
import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from strategies.swing.instrumentation.src.pg_bridge import InstrumentedTradeRecorder


class TestInstrumentedTradeRecorder:
    """Test InstrumentedTradeRecorder (PG-Instrumentation bridge)."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock inner recorder (should be async)
        self.mock_inner = AsyncMock()
        self.mock_inner.record_entry = AsyncMock(return_value="trade_123")
        self.mock_inner.record_exit = AsyncMock()
        self.mock_inner.record = AsyncMock(return_value="trade_456")

        # Create mock kit (synchronous, as it is in the real code)
        self.mock_kit = MagicMock()
        self.mock_kit.log_entry = MagicMock()
        self.mock_kit.log_exit = MagicMock()

        # Create instrumented recorder
        self.recorder = InstrumentedTradeRecorder(self.mock_inner, self.mock_kit)

        # Common test parameters
        self.now = datetime.now(timezone.utc)

    @pytest.mark.asyncio
    async def test_init_stores_inner_recorder_and_kit(self):
        """Test that __init__ stores inner recorder and kit."""
        assert self.recorder._inner is self.mock_inner
        assert self.recorder._kit is self.mock_kit

    @pytest.mark.asyncio
    async def test_record_entry_calls_inner_first(self):
        """Test that record_entry calls inner.record_entry FIRST (before kit)."""
        # Track call order
        call_order = []

        self.mock_inner.record_entry.side_effect = lambda **kw: (
            call_order.append("inner"),
            "trade_123",
        )[1]
        self.mock_kit.log_entry.side_effect = lambda **kw: call_order.append("kit")

        trade_id = await self.recorder.record_entry(
            strategy_id="ATRSS",
            instrument="BTC/USDT",
            direction="LONG",
            quantity=1,
            entry_price=Decimal("50000"),
            entry_ts=self.now,
            setup_tag="breakout",
            entry_type="limit",
            meta={"test": True},
        )

        # Verify inner was called first
        assert call_order == ["inner", "kit"]
        assert trade_id == "trade_123"

    @pytest.mark.asyncio
    async def test_record_entry_calls_inner_and_kit(self):
        """Test that record_entry emits entry event to both inner and kit."""
        trade_id = await self.recorder.record_entry(
            strategy_id="ATRSS",
            instrument="BTC/USDT",
            direction="LONG",
            quantity=1,
            entry_price=Decimal("50000"),
            entry_ts=self.now,
            setup_tag="breakout",
            entry_type="limit",
            meta={"test": True},
        )

        # Verify both were called
        self.mock_inner.record_entry.assert_called_once()
        self.mock_kit.log_entry.assert_called_once()

        # Verify trade_id was returned
        assert trade_id == "trade_123"

    @pytest.mark.asyncio
    async def test_record_entry_passes_correct_params_to_inner(self):
        """Test that record_entry passes all params correctly to inner."""
        await self.recorder.record_entry(
            strategy_id="ATRSS",
            instrument="BTC/USDT",
            direction="LONG",
            quantity=5,
            entry_price=Decimal("50000.50"),
            entry_ts=self.now,
            setup_tag="breakout",
            entry_type="limit",
            meta={"custom": "data"},
            account_id="acct_001",
        )

        # Verify inner was called with correct params
        self.mock_inner.record_entry.assert_called_once_with(
            strategy_id="ATRSS",
            instrument="BTC/USDT",
            direction="LONG",
            quantity=5,
            entry_price=Decimal("50000.50"),
            entry_ts=self.now,
            setup_tag="breakout",
            entry_type="limit",
            meta={"custom": "data"},
            account_id="acct_001",
        )

    @pytest.mark.asyncio
    async def test_record_entry_passes_trade_id_to_kit(self):
        """Test that record_entry passes the trade_id from inner to kit."""
        await self.recorder.record_entry(
            strategy_id="ATRSS",
            instrument="BTC/USDT",
            direction="LONG",
            quantity=1,
            entry_price=Decimal("50000"),
            entry_ts=self.now,
        )

        # Verify kit was called with trade_id
        call_kwargs = self.mock_kit.log_entry.call_args[1]
        assert call_kwargs["trade_id"] == "trade_123"
        assert call_kwargs["pair"] == "BTC/USDT"
        assert call_kwargs["side"] == "LONG"
        assert call_kwargs["entry_price"] == 50000.0

    @pytest.mark.asyncio
    async def test_record_entry_pg_write_succeeds_even_if_kit_fails(self):
        """CRITICAL: Test that PG write succeeds even if instrumentation fails.

        This is the core guarantee of the bridge pattern: the primary PG write
        must always succeed and return trade_id, even if kit.log_entry raises.
        """
        # Make kit fail
        self.mock_kit.log_entry.side_effect = Exception("Kit crashed!")

        # record_entry should still succeed and return trade_id
        trade_id = await self.recorder.record_entry(
            strategy_id="ATRSS",
            instrument="BTC/USDT",
            direction="LONG",
            quantity=1,
            entry_price=Decimal("50000"),
            entry_ts=self.now,
        )

        # Verify PG write succeeded (inner.record_entry was called)
        self.mock_inner.record_entry.assert_called_once()

        # Verify trade_id was returned (from inner/PG)
        assert trade_id == "trade_123"

        # Verify kit was attempted (but failed silently)
        self.mock_kit.log_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_exit_calls_inner_and_kit(self):
        """Test that record_exit emits exit event to both inner and kit."""
        await self.recorder.record_exit(
            trade_id="trade_123",
            exit_price=Decimal("51000"),
            exit_ts=self.now,
            exit_reason="TAKE_PROFIT",
            realized_r=Decimal("2.0"),
            realized_usd=Decimal("100.00"),
            notes="Perfect exit",
            mae_r=Decimal("-0.5"),
            mfe_r=Decimal("3.0"),
            duration_seconds=3600,
            duration_bars=60,
            max_adverse_price=Decimal("49500"),
            max_favorable_price=Decimal("52000"),
        )

        # Verify both were called
        self.mock_inner.record_exit.assert_called_once()
        self.mock_kit.log_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_exit_calls_inner_first(self):
        """Test that record_exit calls inner.record_exit FIRST (before kit)."""
        # Track call order
        call_order = []

        async def inner_exit_impl(**kw):
            call_order.append("inner")

        self.mock_inner.record_exit.side_effect = inner_exit_impl
        self.mock_kit.log_exit.side_effect = lambda **kw: call_order.append("kit")

        await self.recorder.record_exit(
            trade_id="trade_123",
            exit_price=Decimal("51000"),
            exit_ts=self.now,
            exit_reason="TAKE_PROFIT",
            realized_r=Decimal("2.0"),
        )

        # Verify inner was called first
        assert call_order == ["inner", "kit"]

    @pytest.mark.asyncio
    async def test_record_exit_passes_correct_params_to_inner(self):
        """Test that record_exit passes all params correctly to inner."""
        await self.recorder.record_exit(
            trade_id="trade_123",
            exit_price=Decimal("51000.75"),
            exit_ts=self.now,
            exit_reason="STOP_LOSS",
            realized_r=Decimal("-1.5"),
            realized_usd=Decimal("-75.00"),
            notes="Hit stop",
            mae_r=Decimal("-2.0"),
            mfe_r=Decimal("1.0"),
            duration_seconds=1800,
            duration_bars=30,
            max_adverse_price=Decimal("49000"),
            max_favorable_price=Decimal("51500"),
        )

        # Verify inner was called with correct params
        self.mock_inner.record_exit.assert_called_once_with(
            trade_id="trade_123",
            exit_price=Decimal("51000.75"),
            exit_ts=self.now,
            exit_reason="STOP_LOSS",
            realized_r=Decimal("-1.5"),
            realized_usd=Decimal("-75.00"),
            notes="Hit stop",
            mae_r=Decimal("-2.0"),
            mfe_r=Decimal("1.0"),
            duration_seconds=1800,
            duration_bars=30,
            max_adverse_price=Decimal("49000"),
            max_favorable_price=Decimal("51500"),
        )

    @pytest.mark.asyncio
    async def test_record_exit_passes_trade_id_to_kit(self):
        """Test that record_exit passes trade_id and exit data to kit."""
        await self.recorder.record_exit(
            trade_id="trade_999",
            exit_price=Decimal("51000"),
            exit_ts=self.now,
            exit_reason="TAKE_PROFIT",
            realized_r=Decimal("2.0"),
        )

        # Verify kit was called with exit data
        call_kwargs = self.mock_kit.log_exit.call_args[1]
        assert call_kwargs["trade_id"] == "trade_999"
        assert call_kwargs["exit_price"] == 51000.0
        assert call_kwargs["exit_reason"] == "TAKE_PROFIT"

    @pytest.mark.asyncio
    async def test_record_exit_pg_write_succeeds_even_if_kit_fails(self):
        """CRITICAL: Test that PG write succeeds even if instrumentation fails.

        This is the core guarantee of the bridge pattern: the primary PG write
        (record_exit) must always succeed, even if kit.log_exit raises.
        """
        # Make kit fail
        self.mock_kit.log_exit.side_effect = Exception("Kit crashed!")

        # record_exit should still succeed (not raise)
        await self.recorder.record_exit(
            trade_id="trade_123",
            exit_price=Decimal("51000"),
            exit_ts=self.now,
            exit_reason="TAKE_PROFIT",
            realized_r=Decimal("2.0"),
        )

        # Verify PG write succeeded (inner.record_exit was called)
        self.mock_inner.record_exit.assert_called_once()

        # Verify kit was attempted (but failed silently)
        self.mock_kit.log_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_passthrough_to_inner(self):
        """Test that record() is a pass-through to inner.record()."""
        data = {
            "trade_id": "trade_456",
            "symbol": "ETH/USDT",
            "direction": "SHORT",
            "shares": 10,
            "entry_ts": self.now,
            "exit_ts": self.now,
            "entry_price": 2000,
            "exit_price": 2100,
            "r_multiple": 1.0,
            "realized_pnl": 1000,
        }

        trade_id = await self.recorder.record(data)

        # Verify inner.record was called with data
        self.mock_inner.record.assert_called_once_with(data)

        # Verify trade_id was returned
        assert trade_id == "trade_456"

    @pytest.mark.asyncio
    async def test_record_does_not_call_kit(self):
        """Test that record() does NOT call kit (it's a pass-through only)."""
        data = {
            "trade_id": "trade_789",
            "symbol": "BTC/USDT",
            "direction": "LONG",
        }

        await self.recorder.record(data)

        # Kit should NOT be called for record()
        self.mock_kit.log_entry.assert_not_called()
        self.mock_kit.log_exit.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_entry_with_none_kit(self):
        """Test that record_entry works even if kit is None."""
        recorder = InstrumentedTradeRecorder(self.mock_inner, None)

        trade_id = await recorder.record_entry(
            strategy_id="ATRSS",
            instrument="BTC/USDT",
            direction="LONG",
            quantity=1,
            entry_price=Decimal("50000"),
            entry_ts=self.now,
        )

        # Should still return trade_id from inner
        assert trade_id == "trade_123"

        # Inner should be called
        self.mock_inner.record_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_exit_with_none_kit(self):
        """Test that record_exit works even if kit is None."""
        recorder = InstrumentedTradeRecorder(self.mock_inner, None)

        # Should not raise even with kit=None
        await recorder.record_exit(
            trade_id="trade_123",
            exit_price=Decimal("51000"),
            exit_ts=self.now,
            exit_reason="TAKE_PROFIT",
            realized_r=Decimal("2.0"),
        )

        # Inner should be called
        self.mock_inner.record_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_entry_converts_decimal_to_float_for_kit(self):
        """Test that Decimal values are converted to float for kit."""
        await self.recorder.record_entry(
            strategy_id="ATRSS",
            instrument="BTC/USDT",
            direction="LONG",
            quantity=2,
            entry_price=Decimal("50000.123"),
            entry_ts=self.now,
        )

        # Verify kit received floats, not Decimals
        call_kwargs = self.mock_kit.log_entry.call_args[1]
        assert isinstance(call_kwargs["entry_price"], float)
        assert call_kwargs["entry_price"] == 50000.123
        assert isinstance(call_kwargs["position_size"], float)
        assert call_kwargs["position_size"] == 2.0

    @pytest.mark.asyncio
    async def test_record_exit_converts_decimal_to_float_for_kit(self):
        """Test that Decimal values are converted to float for kit on exit."""
        await self.recorder.record_exit(
            trade_id="trade_123",
            exit_price=Decimal("51000.456"),
            exit_ts=self.now,
            exit_reason="TAKE_PROFIT",
            realized_r=Decimal("2.5"),
        )

        # Verify kit received floats, not Decimals
        call_kwargs = self.mock_kit.log_exit.call_args[1]
        assert isinstance(call_kwargs["exit_price"], float)
        assert call_kwargs["exit_price"] == 51000.456
