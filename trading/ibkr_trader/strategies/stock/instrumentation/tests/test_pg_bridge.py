from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from strategies.stock.instrumentation.src.pg_bridge import InstrumentedTradeRecorder


@pytest.mark.asyncio
async def test_record_entry_writes_to_pg_before_instrumentation():
    call_order: list[str] = []
    inner = AsyncMock()
    inner.record_entry.side_effect = lambda **kwargs: (call_order.append("inner"), "trade_123")[1]
    kit = MagicMock()
    kit.log_entry.side_effect = lambda **kwargs: call_order.append("kit")

    recorder = InstrumentedTradeRecorder(
        inner,
        kit,
        strategy_id="IARIC_v1",
        strategy_type="strategy_iaric",
    )

    trade_id = await recorder.record_entry(
        strategy_id="IARIC_v1",
        instrument="AAPL",
        direction="LONG",
        quantity=10,
        entry_price=Decimal("101.25"),
        entry_ts=datetime.now(timezone.utc),
        meta={"entry_signal": "reclaim", "entry_signal_strength": 0.8},
    )

    assert trade_id == "trade_123"
    assert call_order == ["inner", "kit"]
    assert kit.log_entry.call_args.kwargs["entry_signal"] == "reclaim"


@pytest.mark.asyncio
async def test_record_entry_swallows_instrumentation_failures():
    inner = AsyncMock()
    inner.record_entry.return_value = "trade_456"
    kit = MagicMock()
    kit.log_entry.side_effect = RuntimeError("boom")

    recorder = InstrumentedTradeRecorder(
        inner,
        kit,
        strategy_id="ALCB_v1",
        strategy_type="strategy_alcb",
    )

    trade_id = await recorder.record_entry(
        strategy_id="ALCB_v1",
        instrument="MSFT",
        direction="LONG",
        quantity=5,
        entry_price=Decimal("250.10"),
        entry_ts=datetime.now(timezone.utc),
    )

    assert trade_id == "trade_456"
    inner.record_entry.assert_awaited_once()
    kit.log_entry.assert_called_once()


@pytest.mark.asyncio
async def test_record_exit_writes_to_pg_before_instrumentation():
    call_order: list[str] = []

    async def _inner_exit(**kwargs):
        call_order.append("inner")

    inner = AsyncMock()
    inner.record_exit.side_effect = _inner_exit
    kit = MagicMock()
    kit.log_exit.side_effect = lambda **kwargs: call_order.append("kit")

    recorder = InstrumentedTradeRecorder(
        inner,
        kit,
        strategy_id="IARIC_v1",
        strategy_type="strategy_iaric",
    )

    await recorder.record_exit(
        trade_id="trade_123",
        exit_price=Decimal("103.50"),
        exit_ts=datetime.now(timezone.utc),
        exit_reason="EXIT",
        realized_r=Decimal("1.5"),
    )

    assert call_order == ["inner", "kit"]


@pytest.mark.asyncio
async def test_record_exit_swallows_instrumentation_failures():
    inner = AsyncMock()
    inner.record_exit.return_value = None
    kit = MagicMock()
    kit.log_exit.side_effect = RuntimeError("boom")

    recorder = InstrumentedTradeRecorder(
        inner,
        kit,
        strategy_id="ALCB_v1",
        strategy_type="strategy_alcb",
    )

    await recorder.record_exit(
        trade_id="trade_789",
        exit_price=Decimal("99.50"),
        exit_ts=datetime.now(timezone.utc),
        exit_reason="STOP",
        realized_r=Decimal("-1.0"),
    )

    inner.record_exit.assert_awaited_once()
    kit.log_exit.assert_called_once()


@pytest.mark.asyncio
async def test_record_entry_falls_back_to_direct_instrumentation_without_pg():
    kit = MagicMock()
    recorder = InstrumentedTradeRecorder(
        None,
        kit,
        strategy_id="ALCB_v1",
        strategy_type="strategy_alcb",
    )

    trade_id = await recorder.record_entry(
        strategy_id="ALCB_v1",
        instrument="AAPL",
        direction="LONG",
        quantity=3,
        entry_price=Decimal("101.50"),
        entry_ts=datetime.now(timezone.utc),
        meta={"entry_signal": "breakout"},
    )

    assert trade_id
    kit.log_entry.assert_called_once()
    assert kit.log_entry.call_args.kwargs["trade_id"] == trade_id


@pytest.mark.asyncio
async def test_record_entry_forwards_stock_trader_context_fields():
    kit = MagicMock()
    recorder = InstrumentedTradeRecorder(
        None,
        kit,
        strategy_id="ALCB_v1",
        strategy_type="strategy_alcb",
    )
    entry_ts = datetime.now(timezone.utc)

    await recorder.record_entry(
        strategy_id="ALCB_v1",
        instrument="AAPL",
        direction="LONG",
        quantity=7,
        entry_price=Decimal("101.50"),
        entry_ts=entry_ts,
        meta={
            "entry_signal": "A_AVWAP_RETEST",
            "concurrent_positions": 2,
            "drawdown_pct": -1.5,
            "bar_id": "AAPL:20260319T143000",
            "entry_latency_ms": 450,
            "execution_timestamps": {
                "order_submitted_at": entry_ts.isoformat(),
                "fill_received_at": entry_ts.isoformat(),
            },
        },
    )

    kwargs = kit.log_entry.call_args.kwargs
    assert kwargs["concurrent_positions"] == 2
    assert kwargs["drawdown_pct"] == -1.5
    assert kwargs["bar_id"] == "AAPL:20260319T143000"
    assert kwargs["entry_latency_ms"] == 450
    assert kwargs["execution_timestamps"]["order_submitted_at"] == entry_ts.isoformat()


@pytest.mark.asyncio
async def test_record_exit_falls_back_to_direct_instrumentation_without_pg():
    kit = MagicMock()
    recorder = InstrumentedTradeRecorder(
        None,
        kit,
        strategy_id="IARIC_v1",
        strategy_type="strategy_iaric",
    )

    await recorder.record_exit(
        trade_id="fallback_trade",
        exit_price=Decimal("99.50"),
        exit_ts=datetime.now(timezone.utc),
        exit_reason="STOP",
        realized_r=Decimal("-1.0"),
    )

    kit.log_exit.assert_called_once()
    assert kit.log_exit.call_args.kwargs["trade_id"] == "fallback_trade"


@pytest.mark.asyncio
async def test_record_exit_forwards_exit_latency():
    kit = MagicMock()
    recorder = InstrumentedTradeRecorder(
        None,
        kit,
        strategy_id="ALCB_v1",
        strategy_type="strategy_alcb",
    )

    await recorder.record_exit(
        trade_id="trade-123",
        exit_price=Decimal("99.50"),
        exit_ts=datetime.now(timezone.utc),
        exit_reason="STOP",
        realized_r=Decimal("-1.0"),
        meta={"exit_latency_ms": 900, "fees_paid": 2.25},
    )

    kwargs = kit.log_exit.call_args.kwargs
    assert kwargs["exit_latency_ms"] == 900
    assert kwargs["fees_paid"] == 2.25
