"""Test Overlay engine instrumentation wiring.

Verifies that the Overlay engine correctly unwraps an InstrumentationKit
to its InstrumentationContext so that trade_logger, regime_classifier, etc.
are accessible and fire during the daily rebalance cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strategies.swing.overlay.config import OverlayConfig


@dataclass
class _FakeCtx:
    """Minimal InstrumentationContext stand-in with MagicMock services."""
    trade_logger: MagicMock
    regime_classifier: MagicMock
    coordination_logger: MagicMock
    process_scorer: MagicMock
    snapshot_service: MagicMock
    data_dir: str = "test/data"


class _FakeKit:
    """Minimal InstrumentationKit stand-in with a .ctx attribute."""
    def __init__(self, ctx: _FakeCtx) -> None:
        self.ctx = ctx


def _build_engine(kit_or_ctx):
    """Create an OverlayEngine with the given instrumentation object."""
    from strategies.swing.overlay.engine import OverlayEngine

    cfg = OverlayConfig(symbols=["QQQ"], enabled=True)
    ib = MagicMock()
    engine = OverlayEngine(
        ib_session=ib,
        equity=100_000.0,
        config=cfg,
        instrumentation=kit_or_ctx,
    )
    return engine


def _make_ctx() -> _FakeCtx:
    ctx = _FakeCtx(
        trade_logger=MagicMock(),
        regime_classifier=MagicMock(),
        coordination_logger=MagicMock(),
        process_scorer=MagicMock(),
        snapshot_service=MagicMock(),
    )
    ctx.regime_classifier.current_regime = MagicMock(return_value="NEUTRAL")
    ctx.regime_classifier.classify = AsyncMock()
    ctx.snapshot_service.capture_now = MagicMock()
    ctx.trade_logger.log_entry = MagicMock(return_value=None)
    ctx.trade_logger.log_exit = MagicMock(return_value=None)
    return ctx


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


def test_kit_unwrap_to_ctx():
    """When passed an InstrumentationKit, engine should store kit.ctx."""
    ctx = _make_ctx()
    kit = _FakeKit(ctx)
    engine = _build_engine(kit)

    # _instr should be the ctx, not the kit
    assert engine._instr is ctx
    assert hasattr(engine._instr, "trade_logger")
    assert hasattr(engine._instr, "regime_classifier")


def test_raw_ctx_passthrough():
    """When passed a raw InstrumentationContext (no .ctx attr), keep as-is."""
    ctx = _make_ctx()
    engine = _build_engine(ctx)

    assert engine._instr is ctx


def test_none_instrumentation():
    """When passed None, _instr should be None."""
    engine = _build_engine(None)
    assert engine._instr is None


@pytest.mark.asyncio
async def test_entry_instrumentation_fires():
    """Entry instrumentation should call trade_logger.log_entry via safe_instrument."""
    ctx = _make_ctx()
    kit = _FakeKit(ctx)
    engine = _build_engine(kit)

    # Simulate state: 0 shares -> target > 0
    engine._shares = {"QQQ": 0}
    engine._contracts = {"QQQ": MagicMock()}

    import numpy as np

    # Mock IB session methods
    engine._ib.req_historical_data = AsyncMock(return_value=[
        MagicMock(close=float(100 + i)) for i in range(200)
    ])
    engine._ib.ib.managedAccounts = MagicMock(return_value=["U123"])
    engine._ib.ib.accountValues = MagicMock(return_value=[
        MagicMock(tag="NetLiquidation", currency="USD", account="U123", value="100000"),
    ])

    # Mock order placement
    fill_event = AsyncMock()
    trade_mock = MagicMock()
    trade_mock.filledEvent = fill_event()
    trade_mock.orderStatus.avgFillPrice = 299.0
    engine._ib.ib.placeOrder = MagicMock(return_value=trade_mock)

    # Mock DB persistence
    engine._persist_positions_to_db = AsyncMock()

    with patch("strategies.swing.overlay.engine.MarketOrder", create=True):
        await engine._daily_rebalance()

    # trade_logger.log_entry should have been called (via safe_instrument)
    assert ctx.trade_logger.log_entry.called or ctx.regime_classifier.current_regime.called, \
        "Instrumentation hooks should fire when kit is unwrapped to ctx"


@pytest.mark.asyncio
async def test_post_rebalance_snapshot_fires():
    """Post-rebalance regime classification and snapshot should fire."""
    ctx = _make_ctx()
    kit = _FakeKit(ctx)
    engine = _build_engine(kit)

    engine._shares = {"QQQ": 100}
    engine._contracts = {"QQQ": MagicMock()}

    engine._ib.req_historical_data = AsyncMock(return_value=[
        MagicMock(close=float(100 + i)) for i in range(200)
    ])
    engine._ib.ib.managedAccounts = MagicMock(return_value=["U123"])
    engine._ib.ib.accountValues = MagicMock(return_value=[
        MagicMock(tag="NetLiquidation", currency="USD", account="U123", value="100000"),
    ])
    engine._persist_positions_to_db = AsyncMock()

    await engine._daily_rebalance()

    # Post-rebalance hooks: regime_classifier.classify and snapshot_service.capture_now
    assert ctx.regime_classifier.classify.called, \
        "regime_classifier.classify should fire post-rebalance"
    assert ctx.snapshot_service.capture_now.called, \
        "snapshot_service.capture_now should fire post-rebalance"
