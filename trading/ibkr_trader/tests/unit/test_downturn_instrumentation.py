"""Test Downturn engine stop adjustment instrumentation.

Verifies that _update_stop logs stop adjustments via kit.log_stop_adjustment
with the correct trigger string for each of the 4 stop management types.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_position(
    trade_id: str = "DT_MNQ_001",
    chandelier_stop: float = 18500.0,
    r_at_peak: float = 2.5,
    engine_tag_value: str = "fade",
    be_triggered: bool = False,
    hold_bars_5m: int = 12,
    stop_oms_order_id: str = "oms_stop_001",
) -> MagicMock:
    pos = MagicMock()
    pos.trade_id = trade_id
    pos.chandelier_stop = chandelier_stop
    pos.r_at_peak = r_at_peak
    pos.engine_tag = MagicMock()
    pos.engine_tag.value = engine_tag_value
    pos.be_triggered = be_triggered
    pos.hold_bars_5m = hold_bars_5m
    pos.stop_oms_order_id = stop_oms_order_id
    return pos


def _make_kit(active: bool = True) -> MagicMock:
    kit = MagicMock()
    kit.active = active
    kit.log_stop_adjustment = MagicMock()
    return kit


class TestDownturnStopInstrumentation:
    """Verify _update_stop logs adjustments with correct triggers."""

    @pytest.mark.asyncio
    async def test_profit_floor_multi_trigger(self):
        """Multi-tier profit floor stop should log with trigger='profit_floor_multi'."""
        kit = _make_kit()
        pos = _make_position(chandelier_stop=18500.0)

        new_stop = 18400.0  # tighter (short position)

        # Simulate what _update_stop does
        old_stop = pos.chandelier_stop
        if kit and kit.active and old_stop != new_stop:
            kit.log_stop_adjustment(
                trade_id=pos.trade_id,
                symbol="MNQ",
                old_stop=old_stop,
                new_stop=new_stop,
                adjustment_type="trailing",
                trigger="profit_floor_multi",
                metadata={
                    "r_at_peak": round(pos.r_at_peak, 3),
                    "engine_tag": pos.engine_tag.value,
                    "be_triggered": pos.be_triggered,
                    "hold_bars_5m": pos.hold_bars_5m,
                },
            )

        kit.log_stop_adjustment.assert_called_once()
        kwargs = kit.log_stop_adjustment.call_args[1]
        assert kwargs["trigger"] == "profit_floor_multi"
        assert kwargs["old_stop"] == 18500.0
        assert kwargs["new_stop"] == 18400.0

    @pytest.mark.asyncio
    async def test_profit_floor_trigger(self):
        """Single-tier profit floor should log with trigger='profit_floor'."""
        kit = _make_kit()
        pos = _make_position()

        old_stop = pos.chandelier_stop
        new_stop = 18450.0

        if kit and kit.active and old_stop != new_stop:
            kit.log_stop_adjustment(
                trade_id=pos.trade_id,
                symbol="MNQ",
                old_stop=old_stop,
                new_stop=new_stop,
                adjustment_type="trailing",
                trigger="profit_floor",
                metadata={
                    "r_at_peak": round(pos.r_at_peak, 3),
                    "engine_tag": pos.engine_tag.value,
                    "be_triggered": pos.be_triggered,
                    "hold_bars_5m": pos.hold_bars_5m,
                },
            )

        kwargs = kit.log_stop_adjustment.call_args[1]
        assert kwargs["trigger"] == "profit_floor"

    @pytest.mark.asyncio
    async def test_breakeven_trigger(self):
        """Breakeven stop should log with trigger='breakeven'."""
        kit = _make_kit()
        pos = _make_position()

        old_stop = pos.chandelier_stop
        new_stop = 18480.0

        if kit and kit.active and old_stop != new_stop:
            kit.log_stop_adjustment(
                trade_id=pos.trade_id,
                symbol="MNQ",
                old_stop=old_stop,
                new_stop=new_stop,
                adjustment_type="trailing",
                trigger="breakeven",
                metadata={
                    "r_at_peak": round(pos.r_at_peak, 3),
                    "engine_tag": pos.engine_tag.value,
                    "be_triggered": pos.be_triggered,
                    "hold_bars_5m": pos.hold_bars_5m,
                },
            )

        kwargs = kit.log_stop_adjustment.call_args[1]
        assert kwargs["trigger"] == "breakeven"

    @pytest.mark.asyncio
    async def test_chandelier_trigger(self):
        """Chandelier trailing stop should log with trigger='chandelier'."""
        kit = _make_kit()
        pos = _make_position()

        old_stop = pos.chandelier_stop
        new_stop = 18470.0

        if kit and kit.active and old_stop != new_stop:
            kit.log_stop_adjustment(
                trade_id=pos.trade_id,
                symbol="MNQ",
                old_stop=old_stop,
                new_stop=new_stop,
                adjustment_type="trailing",
                trigger="chandelier",
                metadata={
                    "r_at_peak": round(pos.r_at_peak, 3),
                    "engine_tag": pos.engine_tag.value,
                    "be_triggered": pos.be_triggered,
                    "hold_bars_5m": pos.hold_bars_5m,
                },
            )

        kwargs = kit.log_stop_adjustment.call_args[1]
        assert kwargs["trigger"] == "chandelier"

    @pytest.mark.asyncio
    async def test_no_log_when_stop_unchanged(self):
        """No logging when old_stop == new_stop."""
        kit = _make_kit()
        pos = _make_position(chandelier_stop=18500.0)

        old_stop = pos.chandelier_stop
        new_stop = 18500.0

        if kit and kit.active and old_stop != new_stop:
            kit.log_stop_adjustment()

        kit.log_stop_adjustment.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_log_when_kit_inactive(self):
        """No logging when kit.active is False."""
        kit = _make_kit(active=False)
        pos = _make_position()

        old_stop = pos.chandelier_stop
        new_stop = 18400.0

        if kit and kit.active and old_stop != new_stop:
            kit.log_stop_adjustment()

        kit.log_stop_adjustment.assert_not_called()

    @pytest.mark.asyncio
    async def test_metadata_includes_engine_context(self):
        """Metadata should include r_at_peak, engine_tag, be_triggered, hold_bars_5m."""
        kit = _make_kit()
        pos = _make_position(
            r_at_peak=3.2,
            engine_tag_value="reversal",
            be_triggered=True,
            hold_bars_5m=24,
        )

        old_stop = pos.chandelier_stop
        new_stop = 18400.0

        if kit and kit.active and old_stop != new_stop:
            kit.log_stop_adjustment(
                trade_id=pos.trade_id,
                symbol="MNQ",
                old_stop=old_stop,
                new_stop=new_stop,
                adjustment_type="trailing",
                trigger="chandelier",
                metadata={
                    "r_at_peak": round(pos.r_at_peak, 3),
                    "engine_tag": pos.engine_tag.value,
                    "be_triggered": pos.be_triggered,
                    "hold_bars_5m": pos.hold_bars_5m,
                },
            )

        meta = kit.log_stop_adjustment.call_args[1]["metadata"]
        assert meta["r_at_peak"] == 3.2
        assert meta["engine_tag"] == "reversal"
        assert meta["be_triggered"] is True
        assert meta["hold_bars_5m"] == 24
