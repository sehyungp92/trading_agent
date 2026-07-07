"""Integration tests for stock_trader bot compatibility with trading_assistant schemas.

Verifies that stock_trader's event payloads (with extra fields like fees_paid,
entry_slippage_bps, session_type, signal_id, strategy_id, etc.) are correctly
captured by the extended schemas, and that unknown fields are silently ignored.

stock_trader uses a single bot_id ("stock_trader") with strategy_id to
distinguish strategies (iaric, alcb, etc.), matching k_stock_trader's multi-strategy
pattern.
"""

from datetime import datetime, timezone

import pytest

from trading_assistant.schemas.events import (
    DailySnapshot,
    MissedOpportunityEvent,
    TradeEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_base_trade(**overrides) -> dict:
    """Minimal valid TradeEvent payload (shared across tests)."""
    base = dict(
        trade_id="t_iaric_001",
        bot_id="stock_trader",
        strategy_id="iaric",
        pair="AAPL",
        side="LONG",
        entry_time=datetime(2026, 3, 14, 10, 30, tzinfo=timezone.utc),
        exit_time=datetime(2026, 3, 14, 14, 0, tzinfo=timezone.utc),
        entry_price=185.50,
        exit_price=187.20,
        position_size=100,
        pnl=170.0,
        pnl_pct=0.92,
    )
    base.update(overrides)
    return base


def _make_base_missed(**overrides) -> dict:
    """Minimal valid MissedOpportunityEvent payload."""
    base = dict(
        bot_id="stock_trader",
        strategy_id="alcb",
        pair="MSFT",
        signal="breakout_long",
        signal_strength=0.78,
        blocked_by="spread_filter",
        hypothetical_entry=420.50,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# strategy_id field
# ---------------------------------------------------------------------------

class TestStrategyIdField:
    """Verify strategy_id distinguishes strategies within a multi-strategy bot."""

    @pytest.mark.parametrize("strategy_id", ["iaric", "alcb", "kmp", ""])
    def test_strategy_id_in_trade(self, strategy_id):
        trade = TradeEvent.model_validate(_make_base_trade(strategy_id=strategy_id))
        assert trade.strategy_id == strategy_id

    @pytest.mark.parametrize("strategy_id", ["iaric", "alcb", ""])
    def test_strategy_id_in_missed(self, strategy_id):
        event = MissedOpportunityEvent.model_validate(
            _make_base_missed(strategy_id=strategy_id)
        )
        assert event.strategy_id == strategy_id

    def test_strategy_id_defaults_empty(self):
        """Bots that don't set strategy_id get empty string."""
        payload = _make_base_trade()
        del payload["strategy_id"]
        trade = TradeEvent.model_validate(payload)
        assert trade.strategy_id == ""

    def test_strategy_id_roundtrip(self):
        original = TradeEvent.model_validate(_make_base_trade(strategy_id="alcb"))
        restored = TradeEvent.model_validate(original.model_dump(mode="json"))
        assert restored.strategy_id == "alcb"


# ---------------------------------------------------------------------------
# TradeEvent — stock_trader extra fields
# ---------------------------------------------------------------------------

class TestTradeEventStockTraderFields:
    """Verify TradeEvent captures stock_trader execution quality fields."""

    @pytest.mark.parametrize("field,value", [
        ("fees_paid", 1.25),
        ("entry_slippage_bps", 3.2),
        ("exit_slippage_bps", 1.8),
        ("entry_latency_ms", 45.0),
        ("session_type", "regular"),
        ("drawdown_pct", 2.5),
        ("signal_id", "sig_iaric_20260314_AAPL"),
    ])
    def test_field_captured(self, field, value):
        trade = TradeEvent.model_validate(_make_base_trade(**{field: value}))
        assert getattr(trade, field) == value

    def test_passed_filters_captured(self):
        filters = ["spread_ok", "volume_ok", "regime_ok"]
        trade = TradeEvent.model_validate(_make_base_trade(passed_filters=filters))
        assert trade.passed_filters == filters

    def test_filter_decisions_captured(self):
        decisions = [
            {"filter": "spread", "passed": True, "value": 0.02},
            {"filter": "volume", "passed": True, "value": 1500000},
        ]
        trade = TradeEvent.model_validate(_make_base_trade(filter_decisions=decisions))
        assert trade.filter_decisions == decisions
        assert trade.filter_decisions[0]["filter"] == "spread"

    def test_all_stock_trader_fields_together(self):
        """Full stock_trader payload with all extra fields."""
        trade = TradeEvent.model_validate(_make_base_trade(
            fees_paid=1.25,
            entry_slippage_bps=3.2,
            exit_slippage_bps=1.8,
            entry_latency_ms=45.0,
            session_type="regular",
            drawdown_pct=2.5,
            passed_filters=["spread_ok", "volume_ok"],
            filter_decisions=[{"filter": "spread", "passed": True, "value": 0.02}],
            signal_id="sig_001",
        ))
        assert trade.fees_paid == 1.25
        assert trade.entry_slippage_bps == 3.2
        assert trade.exit_slippage_bps == 1.8
        assert trade.entry_latency_ms == 45.0
        assert trade.session_type == "regular"
        assert trade.drawdown_pct == 2.5
        assert trade.passed_filters == ["spread_ok", "volume_ok"]
        assert trade.filter_decisions[0]["filter"] == "spread"
        assert trade.signal_id == "sig_001"

    @pytest.mark.parametrize("field,default", [
        ("fees_paid", 0.0),
        ("entry_slippage_bps", 0.0),
        ("exit_slippage_bps", 0.0),
        ("entry_latency_ms", 0.0),
        ("session_type", ""),
        ("drawdown_pct", 0.0),
        ("passed_filters", None),
        ("filter_decisions", None),
        ("signal_id", ""),
    ])
    def test_defaults_when_fields_absent(self, field, default):
        """Existing bots that don't emit these fields get zero/empty defaults."""
        trade = TradeEvent.model_validate(_make_base_trade())
        assert getattr(trade, field) == default

    def test_empty_passed_filters_vs_none(self):
        """Empty list is preserved, not collapsed to None."""
        trade = TradeEvent.model_validate(_make_base_trade(passed_filters=[]))
        assert trade.passed_filters == []

    def test_negative_slippage(self):
        """Negative slippage (price improvement) is valid."""
        trade = TradeEvent.model_validate(_make_base_trade(entry_slippage_bps=-2.0))
        assert trade.entry_slippage_bps == -2.0


# ---------------------------------------------------------------------------
# TradeEvent — JSON roundtrip with stock_trader fields
# ---------------------------------------------------------------------------

class TestTradeEventStockTraderRoundtrip:
    """Verify JSON serialization preserves stock_trader fields."""

    def test_roundtrip_with_all_extra_fields(self):
        original = TradeEvent.model_validate(_make_base_trade(
            fees_paid=2.50,
            entry_slippage_bps=4.1,
            exit_slippage_bps=1.5,
            entry_latency_ms=33.0,
            session_type="pre_market",
            drawdown_pct=3.1,
            signal_id="sig_002",
            passed_filters=["spread_ok"],
            filter_decisions=[{"filter": "vol", "passed": False, "value": 500}],
        ))
        data = original.model_dump(mode="json")
        restored = TradeEvent.model_validate(data)
        for field in (
            "fees_paid", "entry_slippage_bps", "exit_slippage_bps",
            "entry_latency_ms", "session_type", "drawdown_pct",
            "signal_id", "passed_filters", "filter_decisions",
            "strategy_id",
        ):
            assert getattr(restored, field) == getattr(original, field)


# ---------------------------------------------------------------------------
# TradeEvent — unknown fields silently ignored
# ---------------------------------------------------------------------------

class TestTradeEventUnknownFields:
    """Verify unknown fields from future bot versions are ignored."""

    def test_unknown_fields_ignored(self):
        payload = _make_base_trade(
            fees_paid=1.0,
            some_future_field="should_be_ignored",
            another_unknown=42,
        )
        trade = TradeEvent.model_validate(payload)
        assert trade.fees_paid == 1.0
        assert not hasattr(trade, "some_future_field")
        assert not hasattr(trade, "another_unknown")


# ---------------------------------------------------------------------------
# MissedOpportunityEvent — stock_trader extra fields
# ---------------------------------------------------------------------------

class TestMissedOpportunityStockTraderFields:
    """Verify MissedOpportunityEvent captures stock_trader extras."""

    @pytest.mark.parametrize("field,value", [
        ("signal_id", "sig_alcb_20260314_MSFT"),
        ("block_reason", "spread_too_wide"),
        ("backfill_status", "completed"),
        ("simulation_confidence", 0.85),
    ])
    def test_field_captured(self, field, value):
        event = MissedOpportunityEvent.model_validate(
            _make_base_missed(**{field: value})
        )
        assert getattr(event, field) == value

    def test_all_missed_extras_together(self):
        event = MissedOpportunityEvent.model_validate(_make_base_missed(
            signal_id="sig_003",
            block_reason="volume_insufficient",
            backfill_status="pending",
            simulation_confidence=0.72,
        ))
        assert event.signal_id == "sig_003"
        assert event.block_reason == "volume_insufficient"
        assert event.backfill_status == "pending"
        assert event.simulation_confidence == 0.72

    @pytest.mark.parametrize("field,default", [
        ("signal_id", ""),
        ("block_reason", ""),
        ("backfill_status", ""),
        ("simulation_confidence", 0.0),
    ])
    def test_defaults_when_fields_absent(self, field, default):
        event = MissedOpportunityEvent.model_validate(_make_base_missed())
        assert getattr(event, field) == default

    def test_unknown_fields_ignored(self):
        payload = _make_base_missed(
            signal_id="sig_004",
            unknown_future_field="ignored",
        )
        event = MissedOpportunityEvent.model_validate(payload)
        assert event.signal_id == "sig_004"
        assert not hasattr(event, "unknown_future_field")

    def test_simulation_confidence_boundary(self):
        """Confidence at boundary values."""
        for val in (0.0, 1.0):
            event = MissedOpportunityEvent.model_validate(
                _make_base_missed(simulation_confidence=val)
            )
            assert event.simulation_confidence == val

    def test_block_reason_distinct_from_blocked_by(self):
        """block_reason (freetext) and blocked_by (filter name) are independent."""
        event = MissedOpportunityEvent.model_validate(_make_base_missed(
            blocked_by="spread_filter",
            block_reason="spread was 15 bps, threshold is 10 bps",
        ))
        assert event.blocked_by == "spread_filter"
        assert "15 bps" in event.block_reason


# ---------------------------------------------------------------------------
# DailySnapshot — backwards compatibility
# ---------------------------------------------------------------------------

class TestDailySnapshotBackwardsCompat:
    """Verify DailySnapshot is unaffected by schema extensions."""

    def test_existing_snapshot_still_works(self):
        snapshot = DailySnapshot.model_validate(dict(
            date="2026-03-14",
            bot_id="stock_trader",
            total_trades=12,
            win_count=8,
            loss_count=4,
            gross_pnl=450.0,
            net_pnl=420.0,
        ))
        assert snapshot.bot_id == "stock_trader"
        assert snapshot.total_trades == 12
        assert snapshot.win_count == 8
        assert snapshot.loss_count == 4
        assert snapshot.gross_pnl == 450.0
        assert snapshot.net_pnl == 420.0


# ---------------------------------------------------------------------------
# Multi-strategy bot IDs
# ---------------------------------------------------------------------------

class TestMultiStrategyBotIds:
    """Verify multi-strategy bot patterns work across event types."""

    @pytest.mark.parametrize("bot_id,strategy_id", [
        ("stock_trader", "iaric"),
        ("stock_trader", "alcb"),
        ("k_stock_trader", "kmp"),
        ("k_stock_trader", "nulrimok"),
    ])
    def test_bot_strategy_in_trade_events(self, bot_id, strategy_id):
        trade = TradeEvent.model_validate(
            _make_base_trade(bot_id=bot_id, strategy_id=strategy_id)
        )
        assert trade.bot_id == bot_id
        assert trade.strategy_id == strategy_id

    @pytest.mark.parametrize("bot_id,strategy_id", [
        ("stock_trader", "iaric"),
        ("stock_trader", "alcb"),
    ])
    def test_bot_strategy_in_missed_events(self, bot_id, strategy_id):
        event = MissedOpportunityEvent.model_validate(
            _make_base_missed(bot_id=bot_id, strategy_id=strategy_id)
        )
        assert event.bot_id == bot_id
        assert event.strategy_id == strategy_id

    @pytest.mark.parametrize("bot_id", ["stock_trader", "k_stock_trader"])
    def test_bot_ids_in_snapshots(self, bot_id):
        snapshot = DailySnapshot.model_validate(dict(date="2026-03-14", bot_id=bot_id))
        assert snapshot.bot_id == bot_id
