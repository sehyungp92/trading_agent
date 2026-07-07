# tests/test_opportunity_backfill.py
"""Tests for OpportunityBackfill service."""
from trading_assistant.schemas.events import MissedOpportunityEvent, MarketSnapshot
from trading_assistant.schemas.simulation_policy import (
    FillModel,
    SimulationPolicy,
    TPSLConfig,
    TPSLMethod,
)
from trading_assistant.skills.opportunity_backfill import OpportunityBackfill


def _event(
    entry: float = 40000.0,
    outcome_1h: float | None = 100.0,
    outcome_4h: float | None = 200.0,
    outcome_24h: float | None = 300.0,
    confidence: float = 0.8,
    snapshot: MarketSnapshot | None = None,
) -> MissedOpportunityEvent:
    return MissedOpportunityEvent(
        bot_id="bot1",
        pair="BTCUSDT",
        signal="breakout",
        signal_strength=0.8,
        hypothetical_entry=entry,
        outcome_1h=outcome_1h,
        outcome_4h=outcome_4h,
        outcome_24h=outcome_24h,
        confidence=confidence,
        market_snapshot=snapshot,
    )


def _snapshot(mid: float = 40000.0, ask: float = 40010.0, bid: float = 39990.0, atr: float = 500.0) -> MarketSnapshot:
    return MarketSnapshot(mid=mid, ask=ask, bid=bid, atr_14=atr)


class TestFillModel:
    def test_hypothetical_passthrough(self):
        policy = SimulationPolicy(bot_id="bot1", fill_model=FillModel.HYPOTHETICAL, slippage_bps=0, fees_bps=0)
        bf = OpportunityBackfill(policy)
        result = bf.compute(_event(entry=42000.0))
        assert result.adjusted_entry == 42000.0

    def test_mid_price_from_snapshot(self):
        policy = SimulationPolicy(bot_id="bot1", fill_model=FillModel.MID_PRICE, slippage_bps=0, fees_bps=0)
        bf = OpportunityBackfill(policy)
        result = bf.compute(_event(entry=42000.0), _snapshot(mid=40500.0))
        assert result.adjusted_entry == 40500.0

    def test_ask_for_long(self):
        policy = SimulationPolicy(bot_id="bot1", fill_model=FillModel.ASK_FOR_LONG, slippage_bps=0, fees_bps=0)
        bf = OpportunityBackfill(policy)
        result = bf.compute(_event(entry=42000.0), _snapshot(ask=40010.0))
        assert result.adjusted_entry == 40010.0

    def test_worst_case(self):
        policy = SimulationPolicy(bot_id="bot1", fill_model=FillModel.WORST_CASE, slippage_bps=0, fees_bps=0)
        bf = OpportunityBackfill(policy)
        result = bf.compute(_event(entry=42000.0), _snapshot(ask=40010.0))
        assert result.adjusted_entry == 40010.0


class TestFees:
    def test_round_trip_computed(self):
        policy = SimulationPolicy(bot_id="bot1", fill_model=FillModel.HYPOTHETICAL, fees_bps=10.0, slippage_bps=0)
        bf = OpportunityBackfill(policy)
        result = bf.compute(_event(entry=40000.0))
        # fees = 40000 * 1.0 * 10 / 10000 * 2 = 80.0
        assert result.fees == 80.0

    def test_zero_fees(self):
        policy = SimulationPolicy(bot_id="bot1", fill_model=FillModel.HYPOTHETICAL, fees_bps=0, slippage_bps=0)
        bf = OpportunityBackfill(policy)
        result = bf.compute(_event(entry=40000.0))
        assert result.fees == 0.0


class TestTPSL:
    def test_none_method_returns_none(self):
        policy = SimulationPolicy(bot_id="bot1", tpsl=TPSLConfig(method=TPSLMethod.NONE))
        bf = OpportunityBackfill(policy)
        result = bf.compute(_event())
        assert result.would_have_hit_tp is None
        assert result.would_have_hit_sl is None

    def test_fixed_pct_tp_hit(self):
        policy = SimulationPolicy(
            bot_id="bot1", fill_model=FillModel.HYPOTHETICAL, slippage_bps=0, fees_bps=0,
            tpsl=TPSLConfig(method=TPSLMethod.FIXED_PCT, tp_pct=0.5, sl_pct=0.5),
        )
        bf = OpportunityBackfill(policy)
        # tp_target = 40000 * 0.5 / 100 = 200; outcome_4h=200, max_gain=300 >=200 -> True
        result = bf.compute(_event(entry=40000.0, outcome_1h=100.0, outcome_4h=200.0, outcome_24h=300.0))
        assert result.would_have_hit_tp is True

    def test_atr_multiple(self):
        policy = SimulationPolicy(
            bot_id="bot1", fill_model=FillModel.HYPOTHETICAL, slippage_bps=0, fees_bps=0,
            tpsl=TPSLConfig(method=TPSLMethod.ATR_MULTIPLE, atr_tp_multiple=1.0, atr_sl_multiple=1.0),
        )
        bf = OpportunityBackfill(policy)
        # atr=500, tp_target=500, max_gain=300 < 500 -> False
        result = bf.compute(
            _event(entry=40000.0, outcome_1h=100.0, outcome_4h=200.0, outcome_24h=300.0),
            _snapshot(atr=500.0),
        )
        assert result.would_have_hit_tp is False


class TestNetOutcome:
    def test_reduced_by_fees(self):
        policy = SimulationPolicy(bot_id="bot1", fill_model=FillModel.HYPOTHETICAL, fees_bps=10.0, slippage_bps=0)
        bf = OpportunityBackfill(policy)
        result = bf.compute(_event(entry=40000.0, outcome_1h=100.0))
        # fees = 80.0, net = 100.0 - 80.0 = 20.0
        assert result.net_outcome_1h == 100.0 - 80.0

    def test_none_stays_none(self):
        policy = SimulationPolicy(bot_id="bot1", fill_model=FillModel.HYPOTHETICAL, fees_bps=10.0, slippage_bps=0)
        bf = OpportunityBackfill(policy)
        result = bf.compute(_event(entry=40000.0, outcome_1h=None, outcome_4h=None, outcome_24h=None))
        assert result.net_outcome_1h is None
        assert result.net_outcome_4h is None
        assert result.net_outcome_24h is None


class TestBatch:
    def test_processes_all_events(self):
        policy = SimulationPolicy(bot_id="bot1", fill_model=FillModel.HYPOTHETICAL, fees_bps=0, slippage_bps=0)
        bf = OpportunityBackfill(policy)
        events = [_event(entry=40000.0 + i * 100) for i in range(5)]
        results = bf.compute_batch(events)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.adjusted_entry == 40000.0 + i * 100
