from __future__ import annotations

from datetime import datetime, timezone

from crypto_trader.core.models import TimeFrame
from crypto_trader.core.runtime_types import DecisionContext
from crypto_trader.instrumentation.collector import InstrumentationCollector
from crypto_trader.instrumentation.emitter import EventEmitter
from crypto_trader.instrumentation.sinks import InMemorySink


class _Indicators:
    atr = 100.0
    adx = 20.0
    rsi = 55.0
    ema_fast = 101.0
    ema_mid = 100.0
    ema_slow = 99.0
    volume_ma = 1_000.0


def test_missed_opportunity_uses_decision_identity_and_revisions() -> None:
    collector = InstrumentationCollector(
        "momentum",
        bot_id="bot1",
        lineage={"family_id": "crypto_perps", "portfolio_id": "p1"},
    )
    context = DecisionContext(
        decision_id="momentum|BTC|15m|2026-05-31T00:15:00+00:00",
        strategy_id="momentum",
        symbol="BTC",
        timeframe=TimeFrame.M15,
        decision_time=datetime(2026, 5, 31, 0, 15, tzinfo=timezone.utc),
        decision_key="key",
        metadata={"bar_id": "bar-1"},
    )

    collector.begin_decision_context(context)
    collector.begin_bar("BTC", bar_close=50_000)
    collector.record_gate("BTC", "setup", True)
    collector.record_gate("BTC", "confirmation", False, "no_pattern")
    collector.end_bar("BTC")

    missed = collector.flush_missed()[0]
    first_event_id = missed.metadata.event_id
    missed.outcome_1h = 1.0
    missed.backfill_status = "partial"
    missed.bump_revision()

    assert missed.decision_id == context.decision_id
    assert missed.bar_id == "bar-1"
    assert missed.logical_event_id
    assert missed.metadata.event_id != first_event_id
    assert missed.supersedes_event_id == first_event_id
    assert missed.revision == 1


def test_regime_transition_emits_only_on_material_context_change() -> None:
    collector = InstrumentationCollector("momentum", bot_id="bot1")
    memory = InMemorySink()
    emitter = EventEmitter()
    emitter.add_sink(memory)
    collector.emitter = emitter

    context = DecisionContext(
        decision_id="momentum|BTC|15m|2026-05-31T00:15:00+00:00",
        strategy_id="momentum",
        symbol="BTC",
        timeframe=TimeFrame.M15,
        decision_time=datetime(2026, 5, 31, 0, 15, tzinfo=timezone.utc),
        decision_key="key",
        metadata={"bar_id": "bar-1"},
    )

    collector.begin_decision_context(context)
    collector.begin_bar("BTC", bar_close=50_000)
    collector.snapshot_context("BTC", _Indicators(), bias_direction="LONG", bias_strength=0.81)
    collector.snapshot_context("BTC", _Indicators(), bias_direction="LONG", bias_strength=0.81)
    collector.snapshot_context("BTC", _Indicators(), bias_direction="SHORT", bias_strength=0.71)

    events = memory.events_by_type["regime_transition"]
    payloads = [event.to_dict()["payload"] for event in events]

    assert len(payloads) == 2
    assert payloads[0]["transition_kind"] == "initial"
    assert payloads[0]["new_state"]["bias_direction"] == "LONG"
    assert payloads[1]["transition_kind"] == "change"
    assert payloads[1]["previous_state"]["bias_direction"] == "LONG"
    assert payloads[1]["new_state"]["bias_direction"] == "SHORT"
    assert payloads[1]["bar_id"] == "bar-1"
