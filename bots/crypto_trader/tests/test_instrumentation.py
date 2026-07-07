"""Tests for the instrumentation package — types, collector, quality, emitter, sinks."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crypto_trader.core.models import SetupGrade, Side, Trade
from crypto_trader.instrumentation.types import (
    DailySnapshot,
    ErrorEvent,
    EventMetadata,
    FilterDecision,
    HealthReportSnapshot,
    InstrumentedTradeEvent,
    MarketContext,
    MissedOpportunityEvent,
    PipelineFunnelSnapshot,
    ROOT_CAUSE_TAXONOMY,
    SignalFactor,
)
from crypto_trader.instrumentation.collector import InstrumentationCollector
from crypto_trader.instrumentation.quality import (
    ProcessQualityScorer,
    SCORING_RULES,
)
from crypto_trader.instrumentation.emitter import EventEmitter
from crypto_trader.instrumentation.sinks import InMemorySink, JsonlSink


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(
    symbol: str = "BTC",
    direction: Side = Side.LONG,
    entry_price: float = 100.0,
    exit_price: float = 110.0,
    qty: float = 1.0,
    pnl: float = 10.0,
    r_multiple: float | None = 1.0,
    commission: float = 0.1,
    bars_held: int = 10,
    setup_grade: SetupGrade | None = SetupGrade.A,
    exit_reason: str = "trailing_stop",
    mfe_r: float | None = 1.5,
    mae_r: float | None = -0.3,
    confirmation_type: str = "engulfing",
    entry_method: str = "aggressive",
    confluences: list[str] | None = None,
    funding_paid: float = 0.0,
) -> Trade:
    now = datetime.now(timezone.utc)
    return Trade(
        trade_id="test_001",
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        qty=qty,
        entry_time=now,
        exit_time=now,
        pnl=pnl,
        r_multiple=r_multiple,
        commission=commission,
        bars_held=bars_held,
        setup_grade=setup_grade,
        exit_reason=exit_reason,
        confluences_used=confluences or ["ema_zone"],
        confirmation_type=confirmation_type,
        entry_method=entry_method,
        funding_paid=funding_paid,
        mae_r=mae_r,
        mfe_r=mfe_r,
    )


def _make_context(
    bias_direction: str | None = "LONG",
    funding_rate: float = 0.0,
    setup_grade: str = "A",
    setup_room_r: float = 2.0,
) -> MarketContext:
    return MarketContext(
        atr=100.0, adx=25.0, rsi=55.0,
        ema_fast=99.0, ema_mid=98.0, ema_slow=97.0,
        volume_ma=1000.0, funding_rate=funding_rate,
        bias_direction=bias_direction,
        setup_grade=setup_grade,
        setup_room_r=setup_room_r,
    )


def _make_indicators():
    """Minimal mock indicators for snapshot_context."""
    ind = MagicMock()
    ind.atr = 100.0
    ind.adx = 25.0
    ind.rsi = 55.0
    ind.ema_fast = 99.0
    ind.ema_mid = 98.0
    ind.ema_slow = 97.0
    ind.volume_ma = 1000.0
    return ind


# ===========================================================================
# Phase 1: Types
# ===========================================================================

class TestFilterDecision:
    def test_to_dict(self):
        fd = FilterDecision("warmup", True, threshold=200.0, actual_value=250.0,
                            margin_pct=25.0, reason="", context={})
        d = fd.to_dict()
        assert d["filter_name"] == "warmup"
        assert d["passed"] is True
        assert d["threshold"] == 200.0
        assert d["actual_value"] == 250.0
        assert d["margin_pct"] == 25.0

    def test_to_dict_no_threshold(self):
        fd = FilterDecision("bias", False, reason="no_bias")
        d = fd.to_dict()
        assert d["threshold"] is None
        assert d["actual_value"] is None
        assert d["margin_pct"] is None


class TestMarketContext:
    def test_to_dict_roundtrip(self):
        ctx = _make_context()
        d = ctx.to_dict()
        assert d["atr"] == 100.0
        assert d["bias_direction"] == "LONG"
        assert d["setup_grade"] == "A"

    def test_optional_fields_default_none(self):
        ctx = MarketContext()
        assert ctx.rsi is None
        assert ctx.regime_tier is None
        assert ctx.h4_context_direction is None


class TestSignalFactor:
    def test_to_dict(self):
        sf = SignalFactor("setup_room_r", 2.5)
        assert sf.to_dict() == {"factor": "setup_room_r", "value": 2.5}


class TestEventMetadata:
    def test_deterministic_hash(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        m1 = EventMetadata.create("bot1", "momentum", ts, "trade", "t001")
        m2 = EventMetadata.create("bot1", "momentum", ts, "trade", "t001")
        assert m1.event_id == m2.event_id

    def test_different_inputs_different_hash(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        m1 = EventMetadata.create("bot1", "momentum", ts, "trade", "t001")
        m2 = EventMetadata.create("bot2", "momentum", ts, "trade", "t001")
        assert m1.event_id != m2.event_id

    def test_to_dict(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        m = EventMetadata.create("bot1", "momentum", ts, "trade", "t001")
        d = m.to_dict()
        assert d["bot_id"] == "bot1"
        assert d["strategy_id"] == "momentum"
        assert d["assistant_strategy_id"] == "MomentumPullback_M15"
        assert "event_id" in d
        assert "trace_id" in d

    def test_event_id_length(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        m = EventMetadata.create("bot1", "momentum", ts, "trade", "t001")
        assert len(m.event_id) == 16


class TestInstrumentedTradeEvent:
    def test_to_dict_full(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "trade", "t001")
        event = InstrumentedTradeEvent(
            metadata=meta,
            trade_id="t001",
            pair="BTC",
            side="LONG",
            entry_price=100.0,
            exit_price=110.0,
            pnl=10.0,
            r_multiple=1.0,
            filter_decisions=[FilterDecision("warmup", True)],
            signal_factors=[SignalFactor("room_r", 2.0)],
            market_context=_make_context(),
        )
        d = event.to_dict()
        assert d["trade_id"] == "t001"
        assert len(d["filter_decisions"]) == 1
        assert len(d["signal_factors"]) == 1
        assert d["bias_direction"] == "LONG"
        assert d["market_context"]["atr"] == 100.0

    def test_to_dict_no_context(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "trade", "t001")
        event = InstrumentedTradeEvent(metadata=meta)
        d = event.to_dict()
        assert d["bias_direction"] is None
        assert d["market_context"] is None


class TestMissedOpportunityEvent:
    def test_to_dict(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "missed", "BTC")
        event = MissedOpportunityEvent(
            metadata=meta, pair="BTC", signal="momentum_B",
            blocked_by="confirmation", block_reason="no_pattern",
            hypothetical_entry=50000.0,
        )
        d = event.to_dict()
        assert d["pair"] == "BTC"
        assert d["blocked_by"] == "confirmation"
        assert d["backfill_status"] == "pending"


class TestDailySnapshot:
    def test_to_dict(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "portfolio", ts, "daily", "2025-01-01")
        snap = DailySnapshot(metadata=meta, date="2025-01-01", total_trades=5)
        d = snap.to_dict()
        assert d["total_trades"] == 5
        assert d["date"] == "2025-01-01"


class TestErrorEvent:
    def test_to_dict(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "error", "e001")
        err = ErrorEvent(metadata=meta, error_type="order_reject",
                         message="insufficient margin", severity="high")
        d = err.to_dict()
        assert d["error_type"] == "order_reject"
        assert d["severity"] == "high"


class TestPipelineFunnelSnapshot:
    def test_to_dict_includes_assistant_strategy_id(self):
        snap = PipelineFunnelSnapshot(
            strategy_id="breakout",
            timestamp="2026-05-10T00:00:00+00:00",
            period_start="2026-05-09T23:00:00+00:00",
            period_end="2026-05-10T00:00:00+00:00",
            funnel={"bars_received": {"BTC": 1}},
        )

        d = snap.to_dict()

        assert d["strategy_id"] == "breakout"
        assert d["assistant_strategy_id"] == "VolumeProfileBreakout_M30"


class TestRootCauseTaxonomy:
    def test_has_21_values(self):
        assert len(ROOT_CAUSE_TAXONOMY) == 21

    def test_known_values(self):
        for v in ["regime_mismatch", "premature_stop", "good_execution",
                   "normal_win", "exceptional_win", "normal_loss"]:
            assert v in ROOT_CAUSE_TAXONOMY

    def test_all_scoring_rules_in_taxonomy(self):
        for rule in SCORING_RULES:
            assert rule.root_cause in ROOT_CAUSE_TAXONOMY


# ===========================================================================
# Phase 1: Collector
# ===========================================================================

class TestInstrumentationCollector:
    def test_begin_bar_resets_decisions(self):
        c = InstrumentationCollector("momentum")
        c.record_gate("BTC", "warmup", True)
        c.begin_bar("BTC")
        assert c._current_decisions["BTC"] == []

    def test_record_gate_basic(self):
        c = InstrumentationCollector("momentum")
        c.begin_bar("BTC")
        c.record_gate("BTC", "warmup", True)
        c.record_gate("BTC", "bias", False, "no_bias")
        assert len(c._current_decisions["BTC"]) == 2
        assert c._current_decisions["BTC"][0].passed is True
        assert c._current_decisions["BTC"][1].passed is False
        assert c._current_decisions["BTC"][1].reason == "no_bias"

    def test_record_gate_margin_calc(self):
        c = InstrumentationCollector("momentum")
        c.begin_bar("BTC")
        c.record_gate("BTC", "warmup", False, threshold=200.0, actual_value=150.0)
        fd = c._current_decisions["BTC"][0]
        assert fd.margin_pct == pytest.approx(-25.0)

    def test_record_gate_zero_threshold(self):
        c = InstrumentationCollector("momentum")
        c.begin_bar("BTC")
        c.record_gate("BTC", "test", True, threshold=0.0, actual_value=5.0)
        fd = c._current_decisions["BTC"][0]
        assert fd.margin_pct is None  # Division by zero avoided

    def test_snapshot_context(self):
        c = InstrumentationCollector("momentum")
        ind = _make_indicators()
        c.snapshot_context("BTC", ind, bias_direction="LONG", bias_strength=0.8)
        ctx = c._current_context["BTC"]
        assert ctx.atr == 100.0
        assert ctx.bias_direction == "LONG"
        assert ctx.bias_strength == 0.8

    def test_record_signal_factor(self):
        c = InstrumentationCollector("momentum")
        c.begin_bar("BTC")
        c.record_signal_factor("BTC", "room_r", 2.5)
        c.record_signal_factor("BTC", "confluences", 0.5)
        assert len(c._current_signal_factors["BTC"]) == 2

    def test_record_entry_freezes_state(self):
        c = InstrumentationCollector("momentum")
        c.begin_bar("BTC")
        c.record_gate("BTC", "warmup", True)
        ind = _make_indicators()
        c.snapshot_context("BTC", ind)
        c.record_signal_factor("BTC", "room_r", 2.0)
        c.record_entry("BTC", {"key": "val"}, {"risk_pct": 0.01})

        assert len(c._entry_decisions["BTC"]) == 1
        assert c._entry_context["BTC"].atr == 100.0
        assert len(c._entry_signal_factors["BTC"]) == 1
        assert c._entry_config["BTC"] == {"key": "val"}

    def test_on_trade_closed_builds_event(self):
        c = InstrumentationCollector("momentum", bot_id="test_bot")
        c.begin_bar("BTC")
        c.record_gate("BTC", "warmup", True)
        ind = _make_indicators()
        c.snapshot_context("BTC", ind, bias_direction="LONG")
        c.record_entry("BTC", {}, {"risk_pct": 0.01})

        trade = _make_trade()
        event = c.on_trade_closed("BTC", trade, 85, ["normal_win"])

        assert isinstance(event, InstrumentedTradeEvent)
        assert event.trade_id == "test_001"
        assert event.process_quality_score == 85
        assert event.root_causes == ["normal_win"]
        assert len(event.filter_decisions) == 1
        assert event.market_context.bias_direction == "LONG"

    def test_on_trade_closed_emits_canonical_economics(self):
        c = InstrumentationCollector("momentum", bot_id="test_bot")
        trade = _make_trade(pnl=8.0, commission=0.5, funding_paid=2.0)

        event = c.on_trade_closed("BTC", trade)

        assert event.pnl == pytest.approx(7.5)
        assert event.price_pnl_gross == pytest.approx(10.0)
        assert event.total_fees == pytest.approx(0.5)
        assert event.funding_paid == pytest.approx(2.0)
        assert event.realized_pnl_net == pytest.approx(7.5)

    def test_on_trade_closed_clears_entry_state(self):
        c = InstrumentationCollector("momentum")
        c.begin_bar("BTC")
        c.record_entry("BTC", {}, {})
        trade = _make_trade()
        c.on_trade_closed("BTC", trade)
        assert "BTC" not in c._entry_decisions
        assert "BTC" not in c._entry_context

    def test_end_bar_no_missed_when_no_decisions(self):
        c = InstrumentationCollector("momentum")
        c.begin_bar("BTC")
        c.end_bar("BTC")
        assert c._missed_buffer == []

    def test_end_bar_no_missed_when_setup_fails(self):
        c = InstrumentationCollector("momentum")
        c.begin_bar("BTC")
        c.record_gate("BTC", "warmup", True)
        c.record_gate("BTC", "setup", False, "no_setup_detected")
        c.end_bar("BTC")
        assert c._missed_buffer == []

    def test_end_bar_missed_when_setup_passes_downstream_blocks(self):
        c = InstrumentationCollector("momentum")
        c.begin_bar("BTC", bar_close=50000.0)
        c.record_gate("BTC", "warmup", True)
        c.record_gate("BTC", "setup", True)
        c.record_gate("BTC", "confirmation", False, "no_confirmation_pattern")
        ind = _make_indicators()
        c.snapshot_context("BTC", ind, setup_grade="B", setup_room_r=1.5)
        c.end_bar("BTC")

        assert len(c._missed_buffer) == 1
        missed = c._missed_buffer[0]
        assert missed.pair == "BTC"
        assert missed.blocked_by == "confirmation"
        assert missed.hypothetical_entry == 50000.0

    def test_end_bar_no_missed_when_all_gates_pass(self):
        c = InstrumentationCollector("momentum")
        c.begin_bar("BTC")
        c.record_gate("BTC", "warmup", True)
        c.record_gate("BTC", "setup", True)
        c.record_gate("BTC", "confirmation", True)
        c.record_gate("BTC", "sizing", True)
        c.end_bar("BTC")
        assert c._missed_buffer == []

    def test_flush_missed_clears_buffer(self):
        c = InstrumentationCollector("momentum")
        c.begin_bar("BTC", bar_close=50000.0)
        c.record_gate("BTC", "setup", True)
        c.record_gate("BTC", "confirmation", False, "no_pattern")
        c.end_bar("BTC")
        assert len(c._missed_buffer) == 1

        missed = c.flush_missed()
        assert len(missed) == 1
        assert c._missed_buffer == []

    def test_emitter_property(self):
        c = InstrumentationCollector("momentum")
        assert c.emitter is None
        emitter = EventEmitter()
        c.emitter = emitter
        assert c.emitter is emitter


# ===========================================================================
# Phase 1: Quality Scorer
# ===========================================================================

class TestProcessQualityScorer:
    def test_default_score_is_100(self):
        scorer = ProcessQualityScorer()
        trade = _make_trade(r_multiple=1.0, mfe_r=1.2, setup_grade=SetupGrade.B)
        score, causes = scorer.score(trade, None, [], {})
        assert 0 <= score <= 100

    def test_premature_stop_penalty(self):
        scorer = ProcessQualityScorer()
        trade = _make_trade(bars_held=1, exit_reason="protective_stop", r_multiple=-0.8,
                            mfe_r=0.1, pnl=-5.0)
        score, causes = scorer.score(trade, None, [], {})
        assert "premature_stop" in causes
        assert score < 100

    def test_early_exit_penalty(self):
        scorer = ProcessQualityScorer()
        trade = _make_trade(mfe_r=3.0, r_multiple=0.3, pnl=3.0)
        score, causes = scorer.score(trade, None, [], {})
        assert "early_exit" in causes

    def test_regime_mismatch(self):
        scorer = ProcessQualityScorer()
        trade = _make_trade(direction=Side.LONG)
        ctx = _make_context(bias_direction="SHORT")
        score, causes = scorer.score(trade, ctx, [], {})
        assert "regime_mismatch" in causes

    def test_weak_signal(self):
        scorer = ProcessQualityScorer()
        trade = _make_trade(setup_grade=SetupGrade.B, r_multiple=-0.5, pnl=-5.0,
                            mfe_r=0.2)
        score, causes = scorer.score(trade, None, [], {})
        assert "weak_signal" in causes

    def test_funding_adverse(self):
        scorer = ProcessQualityScorer()
        trade = _make_trade(direction=Side.LONG)
        ctx = _make_context(funding_rate=0.001)  # Positive funding hurts longs
        score, causes = scorer.score(trade, ctx, [], {})
        assert "funding_adverse" in causes

    def test_funding_favorable(self):
        scorer = ProcessQualityScorer()
        trade = _make_trade(direction=Side.LONG)
        ctx = _make_context(funding_rate=-0.001)  # Negative funding helps longs
        score, causes = scorer.score(trade, ctx, [], {})
        assert "funding_favorable" in causes

    def test_strong_signal(self):
        scorer = ProcessQualityScorer()
        # A-grade, exit efficiency > 0.6 (r=1.0/mfe=1.2 = 0.83)
        trade = _make_trade(setup_grade=SetupGrade.A, r_multiple=1.0, mfe_r=1.2, pnl=10.0)
        score, causes = scorer.score(trade, None, [], {})
        assert "strong_signal" in causes

    def test_good_execution(self):
        scorer = ProcessQualityScorer()
        # exit efficiency > 0.7 (r=1.0/mfe=1.2 = 0.83)
        trade = _make_trade(r_multiple=1.0, mfe_r=1.2, pnl=10.0)
        score, causes = scorer.score(trade, None, [], {})
        assert "good_execution" in causes

    def test_exceptional_win_tag(self):
        scorer = ProcessQualityScorer()
        trade = _make_trade(r_multiple=4.0, mfe_r=4.5, pnl=40.0)
        score, causes = scorer.score(trade, None, [], {})
        assert "exceptional_win" in causes

    def test_normal_win_tag(self):
        scorer = ProcessQualityScorer()
        trade = _make_trade(r_multiple=1.5, mfe_r=2.0, pnl=15.0)
        score, causes = scorer.score(trade, None, [], {})
        assert "normal_win" in causes

    def test_normal_loss_tag(self):
        scorer = ProcessQualityScorer()
        trade = _make_trade(r_multiple=-0.8, mfe_r=0.2, pnl=-8.0)
        score, causes = scorer.score(trade, None, [], {})
        assert "normal_loss" in causes

    def test_regime_aligned_tag(self):
        scorer = ProcessQualityScorer()
        trade = _make_trade(direction=Side.LONG, r_multiple=1.0, pnl=10.0,
                            mfe_r=1.5)
        ctx = _make_context(bias_direction="LONG")
        score, causes = scorer.score(trade, ctx, [], {})
        assert "regime_aligned" in causes

    def test_score_clamped_0_100(self):
        scorer = ProcessQualityScorer()
        # Stack all penalties: premature_stop(-15) + early_exit(-20) + regime(-10) + ...
        trade = _make_trade(bars_held=1, exit_reason="protective_stop",
                            r_multiple=-0.5, mfe_r=3.0, pnl=-5.0,
                            setup_grade=SetupGrade.B)
        ctx = _make_context(bias_direction="SHORT", funding_rate=0.001)
        score, causes = scorer.score(trade, ctx, [], {})
        assert score >= 0
        assert score <= 100

    def test_risk_cap_hit(self):
        scorer = ProcessQualityScorer()
        trade = _make_trade()
        score, causes = scorer.score(trade, None, [], {"risk_clamped": True})
        assert "risk_cap_hit" in causes


# ===========================================================================
# Phase 2: Emitter + Sinks
# ===========================================================================

class TestInMemorySink:
    def test_collects_all_event_types(self):
        sink = InMemorySink()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "trade", "t001")

        sink.write_trade(InstrumentedTradeEvent(metadata=meta))
        sink.write_missed(MissedOpportunityEvent(metadata=meta))
        sink.write_daily(DailySnapshot(metadata=meta))
        sink.write_error(ErrorEvent(metadata=meta))

        assert len(sink.trades) == 1
        assert len(sink.missed) == 1
        assert len(sink.daily) == 1
        assert len(sink.errors) == 1


class TestJsonlSink:
    def test_writes_trade_to_file(self, tmp_path):
        sink = JsonlSink(tmp_path)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "trade", "t001")
        event = InstrumentedTradeEvent(metadata=meta, trade_id="t001")
        sink.write_trade(event)

        path = tmp_path / "instrumented_trades.jsonl"
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        assert data["trade_id"] == "t001"

    def test_writes_missed_to_file(self, tmp_path):
        sink = JsonlSink(tmp_path)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "missed", "BTC")
        event = MissedOpportunityEvent(metadata=meta, pair="BTC")
        sink.write_missed(event)

        path = tmp_path / "missed_opportunities.jsonl"
        assert path.exists()

    def test_writes_daily_to_file(self, tmp_path):
        sink = JsonlSink(tmp_path)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "portfolio", ts, "daily", "2025-01-01")
        sink.write_daily(DailySnapshot(metadata=meta, date="2025-01-01"))

        path = tmp_path / "daily_snapshots.jsonl"
        assert path.exists()

    def test_writes_error_to_file(self, tmp_path):
        sink = JsonlSink(tmp_path)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "error", "e001")
        sink.write_error(ErrorEvent(metadata=meta))

        path = tmp_path / "errors.jsonl"
        assert path.exists()

    def test_writes_funnel_and_health_files(self, tmp_path):
        sink = JsonlSink(tmp_path)
        sink.write_funnel(PipelineFunnelSnapshot(
            strategy_id="momentum",
            timestamp="2026-05-10T00:00:00+00:00",
            period_start="2026-05-09T23:00:00+00:00",
            period_end="2026-05-10T00:00:00+00:00",
        ))
        sink.write_health_report(HealthReportSnapshot(
            timestamp="2026-05-10T00:00:00+00:00",
            report={"assessment": "healthy"},
        ))

        assert (tmp_path / "pipeline_funnels.jsonl").exists()
        assert (tmp_path / "health_reports.jsonl").exists()

    def test_appends_multiple_events(self, tmp_path):
        sink = JsonlSink(tmp_path)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for i in range(3):
            meta = EventMetadata.create("bot1", "momentum", ts, "trade", f"t{i}")
            sink.write_trade(InstrumentedTradeEvent(metadata=meta, trade_id=f"t{i}"))

        path = tmp_path / "instrumented_trades.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_utf8_encoding(self, tmp_path):
        """Verify UTF-8 encoding (Windows cp949 crash prevention)."""
        sink = JsonlSink(tmp_path)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "error", "e001")
        # Include em-dash and other special chars
        sink.write_error(ErrorEvent(metadata=meta, message="error — details «here»"))

        path = tmp_path / "errors.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        assert "—" in data["message"]


class TestEventEmitter:
    def test_dispatches_to_multiple_sinks(self):
        emitter = EventEmitter()
        sink1 = InMemorySink()
        sink2 = InMemorySink()
        emitter.add_sink(sink1)
        emitter.add_sink(sink2)

        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "trade", "t001")
        emitter.emit_trade(InstrumentedTradeEvent(metadata=meta))

        assert len(sink1.trades) == 1
        assert len(sink2.trades) == 1

    def test_emit_missed(self):
        emitter = EventEmitter()
        sink = InMemorySink()
        emitter.add_sink(sink)

        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "missed", "BTC")
        emitter.emit_missed(MissedOpportunityEvent(metadata=meta))
        assert len(sink.missed) == 1

    def test_emit_daily(self):
        emitter = EventEmitter()
        sink = InMemorySink()
        emitter.add_sink(sink)

        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "portfolio", ts, "daily", "2025-01-01")
        emitter.emit_daily(DailySnapshot(metadata=meta))
        assert len(sink.daily) == 1

    def test_emit_error(self):
        emitter = EventEmitter()
        sink = InMemorySink()
        emitter.add_sink(sink)

        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "error", "e001")
        emitter.emit_error(ErrorEvent(metadata=meta))
        assert len(sink.errors) == 1

    def test_sink_error_does_not_crash_emitter(self):
        """If one sink errors, others still receive the event."""
        emitter = EventEmitter()
        bad_sink = MagicMock()
        bad_sink.write_trade.side_effect = RuntimeError("boom")
        good_sink = InMemorySink()
        emitter.add_sink(bad_sink)
        emitter.add_sink(good_sink)

        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "trade", "t001")
        emitter.emit_trade(InstrumentedTradeEvent(metadata=meta))

        assert len(good_sink.trades) == 1  # Good sink still received it


# ===========================================================================
# Phase 4: Sidecar
# ===========================================================================

class TestSidecarForwarder:
    def test_sidecar_polls_funnel_and_health_files(self):
        from crypto_trader.instrumentation.sidecar import _EVENT_FILES

        assert "instrumented_trades" in _EVENT_FILES
        assert "missed_opportunities" in _EVENT_FILES
        assert "daily_snapshots" in _EVENT_FILES
        assert "errors" in _EVENT_FILES
        assert "pipeline_funnels" in _EVENT_FILES
        assert "health_reports" in _EVENT_FILES

    def test_watermark_persistence(self, tmp_path):
        from crypto_trader.instrumentation.sidecar import SidecarForwarder

        s = SidecarForwarder(tmp_path, "http://localhost:8000", "bot1", "secret")

        # Simulate watermark save/load
        s._watermarks = {"instrumented_trades": 1234}
        s._save_watermarks()

        s2 = SidecarForwarder(tmp_path, "http://localhost:8000", "bot1", "secret")
        s2._load_watermarks()
        assert s2._watermarks["instrumented_trades"] == 1234

    def test_read_since_watermark(self, tmp_path):
        from crypto_trader.instrumentation.sidecar import SidecarForwarder

        # Write some JSONL data
        jsonl_path = tmp_path / "instrumented_trades.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write('{"event_id":"e1"}\n')
            f.write('{"event_id":"e2"}\n')
            f.write('{"event_id":"e3"}\n')

        s = SidecarForwarder(tmp_path, "http://localhost:8000", "bot1", "secret")

        # First read: gets all 3, returns (events, new_offset)
        events, offset = s._read_since_watermark(jsonl_path, "instrumented_trades")
        assert len(events) == 3
        # Simulate successful send — advance watermark manually
        s._watermarks["instrumented_trades"] = offset

        # Second read: empty (watermark was advanced)
        events, offset = s._read_since_watermark(jsonl_path, "instrumented_trades")
        assert len(events) == 0

        # Append new data
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write('{"event_id":"e4"}\n')

        # Third read: gets only new data
        events, offset = s._read_since_watermark(jsonl_path, "instrumented_trades")
        assert len(events) == 1
        assert events[0]["event_id"] == "e4"

    def test_read_since_watermark_resets_after_file_truncation(self, tmp_path):
        from crypto_trader.instrumentation.sidecar import SidecarForwarder

        jsonl_path = tmp_path / "instrumented_trades.jsonl"
        jsonl_path.write_text('{"event_id":"new"}\n', encoding="utf-8")

        s = SidecarForwarder(tmp_path, "http://localhost:8000", "bot1", "secret")
        s._watermarks["instrumented_trades"] = 10_000

        events, offset = s._read_since_watermark(jsonl_path, "instrumented_trades")

        assert events == [{"event_id": "new"}]
        assert 0 < offset < 10_000

    def test_sidecar_reads_telemetry_files(self, tmp_path):
        from crypto_trader.instrumentation.sidecar import SidecarForwarder

        (tmp_path / "pipeline_funnels.jsonl").write_text(
            '{"strategy_id":"momentum","timestamp":"2026-05-10T00:00:00+00:00","funnel":{}}\n',
            encoding="utf-8",
        )
        (tmp_path / "health_reports.jsonl").write_text(
            '{"timestamp":"2026-05-10T00:00:00+00:00","report":{"assessment":"healthy"}}\n',
            encoding="utf-8",
        )

        s = SidecarForwarder(tmp_path, "http://localhost:8000", "bot1", "secret")

        funnel_events, funnel_offset = s._read_since_watermark(
            tmp_path / "pipeline_funnels.jsonl",
            "pipeline_funnels",
        )
        health_events, health_offset = s._read_since_watermark(
            tmp_path / "health_reports.jsonl",
            "health_reports",
        )

        assert len(funnel_events) == 1
        assert funnel_events[0]["strategy_id"] == "momentum"
        assert funnel_offset > 0
        assert len(health_events) == 1
        assert health_events[0]["report"]["assessment"] == "healthy"
        assert health_offset > 0

    def test_sidecar_sends_telemetry_files_and_persists_watermarks(self, tmp_path):
        from crypto_trader.instrumentation.sidecar import SidecarForwarder

        (tmp_path / "pipeline_funnels.jsonl").write_text(
            '{"strategy_id":"momentum","timestamp":"2026-05-10T00:00:00+00:00"}\n',
            encoding="utf-8",
        )
        (tmp_path / "health_reports.jsonl").write_text(
            '{"timestamp":"2026-05-10T00:00:00+00:00","report":{"assessment":"healthy"}}\n',
            encoding="utf-8",
        )

        s = SidecarForwarder(tmp_path, "http://localhost:8000", "bot1", "secret")
        sent: list[tuple[str, list[dict]]] = []

        def fake_send(events: list[dict], event_type: str) -> bool:
            sent.append((event_type, events))
            return True

        s._send_batch = fake_send  # type: ignore[method-assign]
        s._poll_once()

        assert [event_type for event_type, _ in sent] == [
            "pipeline_funnels",
            "health_reports",
        ]
        assert s._watermarks["pipeline_funnels"] > 0
        assert s._watermarks["health_reports"] > 0

        s2 = SidecarForwarder(tmp_path, "http://localhost:8000", "bot1", "secret")
        s2._load_watermarks()
        assert s2._watermarks["pipeline_funnels"] == s._watermarks["pipeline_funnels"]
        assert s2._watermarks["health_reports"] == s._watermarks["health_reports"]

    def test_start_stop_lifecycle(self, tmp_path):
        from crypto_trader.instrumentation.sidecar import SidecarForwarder

        s = SidecarForwarder(tmp_path, "http://localhost:8000", "bot1", "secret",
                             poll_interval=0.1)
        s.start()
        assert s.is_running
        s.stop()
        assert not s.is_running


# ===========================================================================
# Phase 5: Daily Aggregator
# ===========================================================================

class TestDailyAggregator:
    def test_compute_snapshot_basic(self):
        from crypto_trader.instrumentation.daily_aggregator import DailyAggregator

        agg = DailyAggregator(bot_id="test")
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("test", "momentum", ts, "trade", "t1")

        agg.record_trade(InstrumentedTradeEvent(
            metadata=meta, pnl=10.0, commission=0.5,
            process_quality_score=90, root_causes=["normal_win"],
        ))
        agg.record_trade(InstrumentedTradeEvent(
            metadata=meta, pnl=-5.0, commission=0.5,
            process_quality_score=70, root_causes=["normal_loss"],
        ))
        agg.record_missed(MissedOpportunityEvent(metadata=meta))

        snap = agg.compute_snapshot("2025-01-01")
        assert snap.total_trades == 2
        assert snap.win_count == 1
        assert snap.loss_count == 1
        assert snap.net_pnl == 5.0
        assert snap.missed_count == 1
        assert snap.avg_process_quality == 80.0

    def test_compute_snapshot_uses_explicit_economic_fields(self):
        from crypto_trader.instrumentation.daily_aggregator import DailyAggregator

        agg = DailyAggregator(bot_id="test")
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("test", "momentum", ts, "trade", "t1")

        agg.record_trade(InstrumentedTradeEvent(
            metadata=meta, pnl=999.0, commission=99.0,
            price_pnl_gross=12.0, total_fees=1.0, funding_paid=2.0,
            realized_pnl_net=9.0,
            realized_r_net=1.25,
        ))
        agg.record_trade(InstrumentedTradeEvent(
            metadata=meta, pnl=999.0, commission=99.0,
            price_pnl_gross=1.0, total_fees=0.5, funding_paid=1.5,
            realized_pnl_net=-1.0,
            realized_r_net=-0.5,
        ))

        snap = agg.compute_snapshot("2025-01-01")

        assert snap.win_count == 1
        assert snap.loss_count == 1
        assert snap.gross_pnl == pytest.approx(13.0)
        assert snap.net_pnl == pytest.approx(8.0)
        assert snap.family_summary["gross_pnl"] == pytest.approx(13.0)
        assert snap.family_summary["net_pnl"] == pytest.approx(8.0)
        assert snap.family_summary["fees"] == pytest.approx(1.5)
        assert snap.family_summary["funding"] == pytest.approx(3.5)
        assert snap.family_summary["realized_R"] == pytest.approx(0.75)
        assert snap.per_strategy_summary["momentum"]["pnl"] == pytest.approx(8.0)
        assert snap.per_strategy_summary["momentum"]["gross_pnl"] == pytest.approx(13.0)
        assert snap.per_strategy_summary["momentum"]["net_pnl"] == pytest.approx(8.0)
        assert snap.per_strategy_summary["momentum"]["fees"] == pytest.approx(1.5)
        assert snap.per_strategy_summary["momentum"]["funding"] == pytest.approx(3.5)
        assert snap.per_strategy_summary["momentum"]["realized_R"] == pytest.approx(0.75)

    def test_compute_snapshot_resets_accumulators(self):
        from crypto_trader.instrumentation.daily_aggregator import DailyAggregator

        agg = DailyAggregator()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("test", "momentum", ts, "trade", "t1")
        agg.record_trade(InstrumentedTradeEvent(metadata=meta, pnl=10.0))

        snap1 = agg.compute_snapshot("2025-01-01")
        assert snap1.total_trades == 1

        snap2 = agg.compute_snapshot("2025-01-02")
        assert snap2.total_trades == 0  # Reset after first snapshot

    def test_compute_snapshot_is_scoped_to_requested_utc_date(self):
        from crypto_trader.instrumentation.daily_aggregator import DailyAggregator

        agg = DailyAggregator()
        day1 = datetime(2025, 1, 1, 23, 59, tzinfo=timezone.utc)
        day2 = day1 + timedelta(minutes=2)
        meta1 = EventMetadata.create("test", "momentum", day1, "trade", "t1")
        meta2 = EventMetadata.create("test", "momentum", day2, "trade", "t2")

        agg.record_trade(InstrumentedTradeEvent(metadata=meta1, pnl=10.0))
        agg.record_trade(InstrumentedTradeEvent(metadata=meta2, pnl=99.0))

        snap1 = agg.compute_snapshot("2025-01-01")
        snap2 = agg.compute_snapshot("2025-01-02")

        assert snap1.total_trades == 1
        assert snap1.net_pnl == pytest.approx(10.0)
        assert snap2.total_trades == 1
        assert snap2.net_pnl == pytest.approx(99.0)

    def test_compute_snapshot_dedupes_missed_updates_by_event_id(self):
        from crypto_trader.instrumentation.daily_aggregator import DailyAggregator

        agg = DailyAggregator(bot_id="test")
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        meta = EventMetadata.create("test", "momentum", ts, "missed", "BTC")
        agg.record_missed(MissedOpportunityEvent(metadata=meta, pair="BTC"))
        agg.record_missed(MissedOpportunityEvent(
            metadata=meta,
            pair="BTC",
            backfill_status="complete",
            would_have_hit_tp=True,
        ))

        snap = agg.compute_snapshot("2025-01-01")

        assert snap.missed_count == 1
        assert snap.missed_would_have_won == 1


# ===========================================================================
# Phase 5: Backfiller
# ===========================================================================

class TestMissedOpportunityBackfiller:
    def test_backfill_from_bars(self):
        from crypto_trader.instrumentation.backfill import MissedOpportunityBackfiller
        from crypto_trader.core.models import Bar, TimeFrame

        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "missed", "BTC")
        event = MissedOpportunityEvent(
            metadata=meta, pair="BTC", hypothetical_entry=50000.0,
        )

        # Create bars at 1h, 4h, 24h after signal
        from datetime import timedelta
        bars = [
            Bar(ts + timedelta(hours=1), "BTC", 50100, 50200, 50000, 50100, 100, TimeFrame.H1),
            Bar(ts + timedelta(hours=4), "BTC", 50500, 50600, 50400, 50500, 100, TimeFrame.H1),
            Bar(ts + timedelta(hours=24), "BTC", 51000, 51100, 50900, 51000, 100, TimeFrame.H1),
        ]

        MissedOpportunityBackfiller.backfill_from_bars([event], {"BTC": bars})

        assert event.outcome_1h is not None
        assert event.outcome_1h == pytest.approx(0.2, abs=0.01)  # (50100-50000)/50000*100
        assert event.outcome_24h is not None
        assert event.backfill_status == "complete"

    def test_backfill_no_bars_stays_pending(self):
        from crypto_trader.instrumentation.backfill import MissedOpportunityBackfiller

        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        meta = EventMetadata.create("bot1", "momentum", ts, "missed", "BTC")
        event = MissedOpportunityEvent(
            metadata=meta, pair="BTC", hypothetical_entry=50000.0,
        )

        MissedOpportunityBackfiller.backfill_from_bars([event], {})
        assert event.backfill_status == "pending"


# ===========================================================================
# Phase 6: Relay
# ===========================================================================

class TestRelayAuth:
    def test_sign_and_verify(self):
        from crypto_trader.relay.auth import sign_payload, verify_signature
        payload = {"bot_id": "bot1", "events": [{"x": 1}]}
        sig = sign_payload(payload, "secret123")
        assert verify_signature(payload, sig, "secret123")

    def test_verify_wrong_secret_fails(self):
        from crypto_trader.relay.auth import sign_payload, verify_signature
        payload = {"bot_id": "bot1", "events": [{"x": 1}]}
        sig = sign_payload(payload, "secret123")
        assert not verify_signature(payload, sig, "wrong_secret")

    def test_relay_auth_per_bot(self):
        from crypto_trader.relay.auth import RelayAuth
        auth = RelayAuth({"bot1": "s1", "bot2": "s2"})
        from crypto_trader.relay.auth import sign_payload
        payload = {"data": 1}
        sig = sign_payload(payload, "s1")
        assert auth.verify("bot1", payload, sig)
        assert not auth.verify("bot2", payload, sig)
        assert not auth.verify("bot3", payload, sig)


class TestRelayStore:
    def test_insert_and_get(self, tmp_path):
        from crypto_trader.relay.store import RelayStore
        store = RelayStore(tmp_path / "test.db")

        events = [{"metadata": {"event_id": "e1"}, "data": "hello"}]
        inserted = store.insert_events("bot1", "trades", events)
        assert inserted == 1

        result = store.get_events()
        assert len(result) == 1
        assert result[0]["payload"]["data"] == "hello"
        store.close()

    def test_dedup_by_event_id(self, tmp_path):
        from crypto_trader.relay.store import RelayStore
        store = RelayStore(tmp_path / "test.db")

        events = [{"metadata": {"event_id": "e1"}, "data": 1}]
        store.insert_events("bot1", "trades", events)
        inserted = store.insert_events("bot1", "trades", events)
        assert inserted == 0  # Duplicate

        result = store.get_events()
        assert len(result) == 1
        store.close()

    def test_dedup_prefers_top_level_event_id_over_nested_metadata(self, tmp_path):
        from crypto_trader.relay.store import RelayStore
        store = RelayStore(tmp_path / "test.db")

        first = {
            "schema_version": "assistant_event_v1",
            "event_id": "top_1",
            "logical_event_id": "logical_1",
            "event_type": "trade",
            "metadata": {"event_id": "stale_nested"},
            "payload": {
                "metadata": {"event_id": "stale_payload_nested"},
                "trade_id": "t1",
            },
        }
        duplicate_top = {
            **first,
            "metadata": {"event_id": "different_nested"},
            "payload": {"metadata": {"event_id": "different_payload_nested"}},
        }
        second_top_same_nested = {
            **first,
            "event_id": "top_2",
            "logical_event_id": "logical_2",
        }

        assert store.insert_events("bot1", "trade", [first]) == 1
        assert store.insert_events("bot1", "trade", [duplicate_top]) == 0
        assert store.insert_events("bot1", "trade", [second_top_same_nested]) == 1

        result = store.get_events()
        assert [event["event_id"] for event in result] == ["top_1", "top_2"]
        assert [event["logical_event_id"] for event in result] == ["logical_1", "logical_2"]
        store.close()

    def test_ack_events(self, tmp_path):
        from crypto_trader.relay.store import RelayStore
        store = RelayStore(tmp_path / "test.db")

        events = [
            {"metadata": {"event_id": "e1"}},
            {"metadata": {"event_id": "e2"}},
        ]
        store.insert_events("bot1", "trades", events)

        acked = store.ack_events(["e1"])
        assert acked == 1

        # Only e2 should be pending
        result = store.get_events()
        assert len(result) == 1
        assert result[0]["event_id"] == "e2"
        store.close()

    def test_get_health(self, tmp_path):
        from crypto_trader.relay.store import RelayStore
        store = RelayStore(tmp_path / "test.db")

        events = [
            {"metadata": {"event_id": "e1"}},
            {"metadata": {"event_id": "e2"}},
        ]
        store.insert_events("bot1", "trades", events)
        store.ack_events(["e1"])

        health = store.get_health()
        assert health["status"] == "ok"
        assert health["pending_events"] == 1
        assert health["total_events"] == 2
        assert health["pending"] == 1
        assert health["acked"] == 1
        assert "bot1" in health["per_bot"]
        assert health["per_bot_pending"]["bot1"] == 1
        assert health["last_event_per_bot"]["bot1"] is not None
        assert health["oldest_pending_age_seconds"] is not None
        assert health["db_size_bytes"] > 0
        assert health["uptime_seconds"] >= 0
        store.close()

    def test_purge_acked(self, tmp_path):
        from datetime import timedelta
        from crypto_trader.relay.store import RelayStore
        store = RelayStore(tmp_path / "test.db")

        events = [{"metadata": {"event_id": "e1"}}]
        store.insert_events("bot1", "trades", events)
        store.ack_events(["e1"])

        # Backdate received_at so it's older than the purge cutoff
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        store._conn.execute(
            "UPDATE events SET received_at = ? WHERE event_id = 'e1'",
            (old_ts,),
        )
        store._conn.commit()

        deleted = store.purge_acked(older_than_hours=1)
        assert deleted == 1

        result = store.get_events()
        assert len(result) == 0
        store.close()

    def test_watermark_based_get(self, tmp_path):
        from crypto_trader.relay.store import RelayStore
        store = RelayStore(tmp_path / "test.db")

        events = [
            {"metadata": {"event_id": f"e{i}"}}
            for i in range(5)
        ]
        store.insert_events("bot1", "trades", events)

        batch1 = store.get_events(since_id=0, limit=2)
        assert len(batch1) == 2
        last_id = batch1[-1]["id"]

        batch2 = store.get_events(since_id=last_id, limit=2)
        assert len(batch2) == 2

        batch3 = store.get_events(since_id=batch2[-1]["id"], limit=2)
        assert len(batch3) == 1
        store.close()

    def test_round_trips_pipeline_funnel_event(self, tmp_path):
        from crypto_trader.relay.store import RelayStore
        store = RelayStore(tmp_path / "relay.db")

        events = [{
            "strategy_id": "momentum",
            "timestamp": "2026-05-10T00:00:00+00:00",
            "funnel": {"bars_received": {"BTC": 10}},
        }]

        assert store.insert_events("crypto_trader", "pipeline_funnels", events) == 1
        out = store.get_events(limit=10)

        assert out[0]["event_type"] == "pipeline_funnels"
        assert out[0]["payload"]["strategy_id"] == "momentum"
        store.close()

    def test_round_trips_health_report_event(self, tmp_path):
        from crypto_trader.relay.store import RelayStore
        store = RelayStore(tmp_path / "relay.db")

        events = [{
            "timestamp": "2026-05-10T00:00:00+00:00",
            "report": {"assessment": "healthy"},
        }]

        assert store.insert_events("crypto_trader", "health_reports", events) == 1
        out = store.get_events(limit=10)

        assert out[0]["event_type"] == "health_reports"
        assert out[0]["payload"]["report"]["assessment"] == "healthy"
        store.close()


# ===========================================================================
# Phase 5: Live state path properties
# ===========================================================================

class TestPersistentStatePaths:
    def test_instrumentation_paths(self, tmp_path):
        from crypto_trader.live.state import PersistentState
        state = PersistentState(tmp_path)

        assert state.instrumented_trades_path == tmp_path / "instrumented_trades.jsonl"
        assert state.missed_opportunities_path == tmp_path / "missed_opportunities.jsonl"
        assert state.daily_snapshots_path == tmp_path / "daily_snapshots.jsonl"
        assert state.errors_path == tmp_path / "errors.jsonl"


# ===========================================================================
# Core events
# ===========================================================================

class TestCoreEvents:
    def test_instrumented_trade_emitted_event(self):
        from crypto_trader.core.events import InstrumentedTradeEmitted, Event
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        event = InstrumentedTradeEmitted(timestamp=ts, event="dummy")
        assert isinstance(event, Event)

    def test_missed_opportunity_emitted_event(self):
        from crypto_trader.core.events import MissedOpportunityEmitted, Event
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        event = MissedOpportunityEmitted(timestamp=ts, event="dummy")
        assert isinstance(event, Event)
