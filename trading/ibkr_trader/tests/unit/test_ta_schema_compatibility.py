"""Verify that each family's TradeEvent.to_dict() and MissedOpportunityEvent.to_dict()
produce payloads compatible with the Trading Assistant's Pydantic schemas.

These tests catch silent data loss from field naming mismatches between
instrumentation (dataclass) and TA (Pydantic model).
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

# ---------------------------------------------------------------------------
# Helpers to build minimal valid dataclass instances for each family
# ---------------------------------------------------------------------------

def _make_event_metadata() -> dict:
    return {
        "event_id": "test123",
        "bot_id": "test_bot",
        "event_type": "trade",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _make_snapshot() -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pair": "NQ",
        "last_trade_price": 21000.0,
        "bid": 20999.5,
        "ask": 21000.5,
        "mid": 21000.0,
        "spread_bps": 0.24,
        "volume_24h": 1500000,
        "atr_14": 150.0,
        "funding_rate": 0.0,
        "open_interest": 500000,
    }


# ===========================================================================
# Gap 1+2: Momentum TradeEvent → TA TradeEvent compatibility
# ===========================================================================

class TestMomentumTradeEventCompat:
    """Momentum TradeEvent must include bot_id, TA aliases, and process_quality_score."""

    def _build_momentum_trade(self):
        from strategies.momentum.instrumentation.src.trade_logger import TradeEvent
        return TradeEvent(
            trade_id="m_001",
            event_metadata=_make_event_metadata(),
            entry_snapshot=_make_snapshot(),
            bot_id="momentum_nqdtc",
            pair="NQ",
            side="LONG",
            entry_time=datetime.now(timezone.utc).isoformat(),
            exit_time=datetime.now(timezone.utc).isoformat(),
            entry_price=21000.0,
            exit_price=21050.0,
            position_size=2.0,
            pnl=100.0,
            pnl_pct=0.238,
            entry_signal="class_m_bullish",
            entry_signal_id="sig_abc",
            entry_signal_strength=0.85,
            exit_reason="TAKE_PROFIT",
            strategy_type="nqdtc",
            stage="exit",
        )

    def test_bot_id_present(self):
        trade = self._build_momentum_trade()
        d = trade.to_dict()
        assert d["bot_id"] == "momentum_nqdtc", "bot_id must be set on momentum TradeEvent"

    def test_ta_aliases_present(self):
        trade = self._build_momentum_trade()
        d = trade.to_dict()
        assert "market_snapshot" in d, "market_snapshot alias missing"
        assert "spread_at_entry" in d, "spread_at_entry alias missing"
        assert "volume_24h" in d, "volume_24h alias missing"
        assert "funding_rate" in d, "funding_rate alias missing"
        assert "open_interest_delta" in d, "open_interest_delta alias missing"
        assert "signal_id" in d, "signal_id alias missing"

    def test_signal_id_matches_entry_signal_id(self):
        trade = self._build_momentum_trade()
        d = trade.to_dict()
        assert d["signal_id"] == "sig_abc"

    def test_process_quality_score_defaults_to_100(self):
        trade = self._build_momentum_trade()
        d = trade.to_dict()
        assert d["process_quality_score"] == 100

    def test_process_quality_score_preserved_when_set(self):
        trade = self._build_momentum_trade()
        trade.process_quality_score = 42
        trade.root_causes = ["late_entry"]
        trade.evidence_refs = ["slippage > 5bps"]
        d = trade.to_dict()
        assert d["process_quality_score"] == 42
        assert d["root_causes"] == ["late_entry"]
        assert d["evidence_refs"] == ["slippage > 5bps"]

    def test_fill_details_fields_exist(self):
        trade = self._build_momentum_trade()
        trade.entry_fill_details = {"slippage_bps": 1.5, "fill_latency_ms": 12, "fill_type": "limit"}
        trade.exit_fill_details = {"slippage_bps": 2.0, "fill_latency_ms": 8, "fill_type": "market"}
        d = trade.to_dict()
        assert d["entry_fill_details"]["slippage_bps"] == 1.5
        assert d["exit_fill_details"]["fill_type"] == "market"


# ===========================================================================
# Gap 3: signal_id alias across all families
# ===========================================================================

class TestSignalIdAlias:
    """All three families must emit signal_id in to_dict()."""

    def test_swing_signal_id(self):
        from strategies.swing.instrumentation.src.trade_logger import TradeEvent
        trade = TradeEvent(
            trade_id="s_001",
            bot_id="swing_bot",
            entry_signal_id="sig_swing",
        )
        d = trade.to_dict()
        assert d["signal_id"] == "sig_swing"

    def test_momentum_signal_id(self):
        from strategies.momentum.instrumentation.src.trade_logger import TradeEvent
        trade = TradeEvent(
            trade_id="m_001",
            event_metadata=_make_event_metadata(),
            entry_snapshot=_make_snapshot(),
            entry_signal_id="sig_momentum",
        )
        d = trade.to_dict()
        assert d["signal_id"] == "sig_momentum"

    def test_stock_signal_id(self):
        from strategies.stock.instrumentation.src.trade_logger import TradeEvent
        trade = TradeEvent(
            trade_id="st_001",
            event_metadata=_make_event_metadata(),
            entry_snapshot=_make_snapshot(),
            entry_signal_id="sig_stock",
        )
        d = trade.to_dict()
        assert d["signal_id"] == "sig_stock"


# ===========================================================================
# Gap 4: Swing fill details and signal evolution fields
# ===========================================================================

class TestSwingFillDetails:
    """Swing TradeEvent must support signal_evolution and fill details."""

    def test_signal_evolution_field_exists(self):
        from strategies.swing.instrumentation.src.trade_logger import TradeEvent
        trade = TradeEvent(
            trade_id="s_002",
            bot_id="swing_bot",
            signal_evolution=[{"bars_ago": 1, "close": 510.0, "rsi": 55.0}],
        )
        d = trade.to_dict()
        assert d["signal_evolution"] is not None
        assert len(d["signal_evolution"]) == 1

    def test_fill_details_fields_exist(self):
        from strategies.swing.instrumentation.src.trade_logger import TradeEvent
        trade = TradeEvent(
            trade_id="s_003",
            bot_id="swing_bot",
            entry_fill_details={"slippage_bps": 1.0, "fill_latency_ms": 10, "fill_type": "limit"},
            exit_fill_details={"slippage_bps": 2.0, "fill_latency_ms": 8, "fill_type": "stop"},
        )
        d = trade.to_dict()
        assert d["entry_fill_details"]["fill_type"] == "limit"
        assert d["exit_fill_details"]["fill_type"] == "stop"


# ===========================================================================
# Gap 6: margin_pct on missed opportunity events
# ===========================================================================

class TestMissedOpportunityMarginPct:
    """All families must support margin_pct on MissedOpportunityEvent."""

    def test_swing_margin_pct(self):
        from strategies.swing.instrumentation.src.missed_opportunity import MissedOpportunityEvent
        event = MissedOpportunityEvent(
            event_metadata=_make_event_metadata(),
            market_snapshot=_make_snapshot(),
            margin_pct=-5.2,
        )
        d = event.to_dict()
        assert d["margin_pct"] == -5.2

    def test_momentum_margin_pct(self):
        from strategies.momentum.instrumentation.src.missed_opportunity import MissedOpportunityEvent
        event = MissedOpportunityEvent(
            event_metadata=_make_event_metadata(),
            market_snapshot=_make_snapshot(),
            margin_pct=-3.1,
        )
        d = event.to_dict()
        assert d["margin_pct"] == -3.1

    def test_stock_margin_pct(self):
        from strategies.stock.instrumentation.src.missed_opportunity import MissedOpportunityEvent
        event = MissedOpportunityEvent(
            event_metadata=_make_event_metadata(),
            market_snapshot=_make_snapshot(),
            margin_pct=-8.0,
        )
        d = event.to_dict()
        assert d["margin_pct"] == -8.0


# ===========================================================================
# Gap 6: TA-compatible aliases on missed opportunities
# ===========================================================================

class TestMissedOpportunityAliases:
    """Swing and momentum must add hypothetical_entry and confidence aliases."""

    def test_swing_aliases(self):
        from strategies.swing.instrumentation.src.missed_opportunity import MissedOpportunityEvent
        event = MissedOpportunityEvent(
            event_metadata=_make_event_metadata(),
            market_snapshot=_make_snapshot(),
            hypothetical_entry_price=21000.5,
            simulation_confidence=0.7,
        )
        d = event.to_dict()
        assert d["hypothetical_entry"] == 21000.5
        assert d["confidence"] == 0.7

    def test_momentum_aliases(self):
        from strategies.momentum.instrumentation.src.missed_opportunity import MissedOpportunityEvent
        event = MissedOpportunityEvent(
            event_metadata=_make_event_metadata(),
            market_snapshot=_make_snapshot(),
            hypothetical_entry_price=21000.5,
            simulation_confidence=0.7,
        )
        d = event.to_dict()
        assert d["hypothetical_entry"] == 21000.5
        assert d["confidence"] == 0.7


# ===========================================================================
# Cross-family field coverage: required TA fields are non-empty
# ===========================================================================

class TestRequiredTAFields:
    """After to_dict(), TA-required fields must be populated."""

    @pytest.mark.parametrize("family", ["swing", "momentum", "stock"])
    def test_required_fields_present(self, family):
        if family == "swing":
            from strategies.swing.instrumentation.src.trade_logger import TradeEvent
            trade = TradeEvent(
                trade_id="t1", bot_id="bot1", entry_signal_id="sig1",
            )
        elif family == "momentum":
            from strategies.momentum.instrumentation.src.trade_logger import TradeEvent
            trade = TradeEvent(
                trade_id="t1",
                event_metadata=_make_event_metadata(),
                entry_snapshot=_make_snapshot(),
                bot_id="bot1",
                entry_signal_id="sig1",
            )
        else:
            from strategies.stock.instrumentation.src.trade_logger import TradeEvent
            trade = TradeEvent(
                trade_id="t1",
                event_metadata=_make_event_metadata(),
                entry_snapshot=_make_snapshot(),
                bot_id="bot1",
                entry_signal_id="sig1",
            )

        d = trade.to_dict()

        # These fields must be present and non-empty for TA validation
        assert d.get("bot_id"), f"{family}: bot_id must be non-empty"
        assert d.get("signal_id") == "sig1", f"{family}: signal_id must match entry_signal_id"
        assert isinstance(d.get("process_quality_score"), int), \
            f"{family}: process_quality_score must be int"
        assert d["process_quality_score"] >= 0, \
            f"{family}: process_quality_score must be >= 0"
