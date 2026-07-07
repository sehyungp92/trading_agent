"""Tests for experiment ID / A-B test flag fields in TradeEvent and MissedOpportunityEvent."""

from strategies.momentum.instrumentation.src.trade_logger import TradeEvent
from strategies.momentum.instrumentation.src.missed_opportunity import MissedOpportunityEvent


def test_trade_event_has_experiment_fields():
    """TradeEvent should have experiment_id and experiment_variant fields."""
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={})
    assert hasattr(te, "experiment_id")
    assert hasattr(te, "experiment_variant")
    assert te.experiment_id is None
    assert te.experiment_variant is None


def test_trade_event_experiment_fields_set():
    """TradeEvent should accept and store experiment fields."""
    te = TradeEvent(
        trade_id="t1",
        event_metadata={},
        entry_snapshot={},
        experiment_id="exp_001",
        experiment_variant="variant_a"
    )
    assert te.experiment_id == "exp_001"
    assert te.experiment_variant == "variant_a"


def test_missed_event_has_experiment_fields():
    """MissedOpportunityEvent should have experiment_id and experiment_variant fields."""
    me = MissedOpportunityEvent(
        event_metadata={},
        market_snapshot={},
        pair="NQ",
        side="LONG",
        signal="Class_M",
        signal_id="s1",
        signal_strength=0.5,
        blocked_by="heat_cap",
        block_reason="heat > cap"
    )
    assert hasattr(me, "experiment_id")
    assert hasattr(me, "experiment_variant")
    assert me.experiment_id is None
    assert me.experiment_variant is None


def test_missed_event_experiment_fields_set():
    """MissedOpportunityEvent should accept and store experiment fields."""
    me = MissedOpportunityEvent(
        event_metadata={},
        market_snapshot={},
        pair="NQ",
        side="LONG",
        signal="Class_M",
        signal_id="s1",
        signal_strength=0.5,
        blocked_by="heat_cap",
        block_reason="heat > cap",
        experiment_id="exp_001",
        experiment_variant="variant_b"
    )
    assert me.experiment_id == "exp_001"
    assert me.experiment_variant == "variant_b"


def test_trade_event_to_dict_includes_experiment_fields():
    """TradeEvent.to_dict() should include experiment fields."""
    te = TradeEvent(
        trade_id="t1",
        event_metadata={},
        entry_snapshot={},
        experiment_id="exp_001",
        experiment_variant="variant_a"
    )
    d = te.to_dict()
    assert "experiment_id" in d
    assert "experiment_variant" in d
    assert d["experiment_id"] == "exp_001"
    assert d["experiment_variant"] == "variant_a"


def test_missed_event_to_dict_includes_experiment_fields():
    """MissedOpportunityEvent.to_dict() should include experiment fields."""
    me = MissedOpportunityEvent(
        event_metadata={},
        market_snapshot={},
        pair="NQ",
        side="LONG",
        signal="Class_M",
        signal_id="s1",
        signal_strength=0.5,
        blocked_by="heat_cap",
        block_reason="heat > cap",
        experiment_id="exp_001",
        experiment_variant="variant_b"
    )
    d = me.to_dict()
    assert "experiment_id" in d
    assert "experiment_variant" in d
    assert d["experiment_id"] == "exp_001"
    assert d["experiment_variant"] == "variant_b"
