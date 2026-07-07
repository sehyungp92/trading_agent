"""Tests for InteractionAnalyzer — coordinator event analysis."""
from __future__ import annotations

import pytest

from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.interaction_analysis import (
    CoordinatorAction,
    InteractionEffect,
    InteractionReport,
)
from trading_assistant.skills.interaction_analyzer import InteractionAnalyzer
from tests.factories import make_trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(
    pair: str = "AAPL",
    pnl: float = 100.0,
    side: str = "LONG",
    exit_reason: str = "TAKE_PROFIT",
    entry_time: str = "2026-01-05T10:00:00",
    exit_time: str = "2026-01-05T14:00:00",
    position_size: float = 10.0,
    entry_price: float = 150.0,
    exit_price: float = 160.0,
    post_exit_4h_price: float | None = None,
    market_regime: str = "trending",
) -> TradeEvent:
    return make_trade(
        trade_id=f"t-{pair}-{pnl}",
        bot_id="swing_multi_01",
        pair=pair,
        side=side,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        position_size=position_size,
        pnl=pnl,
        pnl_pct=pnl / (entry_price * position_size) * 100,
        exit_reason=exit_reason,
        market_regime=market_regime,
        post_exit_4h_price=post_exit_4h_price,
    )


def _make_action(
    action: str = "tighten_stop_be",
    symbol: str = "AAPL",
    rule: str = "rule_1",
    timestamp: str = "2026-01-05T11:00:00",
    trigger: str = "helix",
    target: str = "phoenix",
    details: dict | None = None,
) -> CoordinatorAction:
    return CoordinatorAction(
        timestamp=timestamp,
        action=action,
        trigger_strategy=trigger,
        target_strategy=target,
        symbol=symbol,
        rule=rule,
        details=details or {},
        outcome="applied",
    )


# ---------------------------------------------------------------------------
# Schema defaults
# ---------------------------------------------------------------------------

class TestSchemaDefaults:
    def test_coordinator_action_defaults(self):
        a = CoordinatorAction(action="test")
        assert a.timestamp == ""
        assert a.details == {}

    def test_interaction_effect_defaults(self):
        e = InteractionEffect(rule="r1", action_type="test", trigger_strategy="a", target_strategy="b")
        assert e.action_count == 0
        assert e.net_benefit == 0.0

    def test_interaction_report_defaults(self):
        r = InteractionReport(week_start="2026-01-01", week_end="2026-01-07")
        assert r.bot_id == "swing_multi_01"
        assert r.total_coordination_events == 0
        assert r.recommendation == ""


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_empty_events_returns_empty_report(self):
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        report = ia.compute([], [])
        assert report.total_coordination_events == 0
        assert report.effects == []

    def test_events_with_no_trades(self):
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        events = [_make_action()]
        report = ia.compute(events, [])
        assert report.total_coordination_events == 1
        assert len(report.effects) == 1
        assert report.effects[0].affected_trades == 0


# ---------------------------------------------------------------------------
# Stop-tightening analysis
# ---------------------------------------------------------------------------

class TestStopTightening:
    def test_stop_tightening_matched_trade(self):
        """Trade that was stopped out after tightening gets counterfactual analysis."""
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        trade = _make_trade(
            pnl=-50.0,
            exit_reason="STOP_LOSS",
            post_exit_4h_price=170.0,  # price went up after stop
        )
        action = _make_action(action="tighten_stop_be", symbol="AAPL")
        report = ia.compute([action], [trade])
        assert report.effects[0].affected_trades == 1

    def test_stop_tightening_trade_survived(self):
        """Trade that survived the tightened stop and hit TP."""
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        trade = _make_trade(pnl=200.0, exit_reason="TAKE_PROFIT")
        action = _make_action(action="tighten_stop_be", symbol="AAPL")
        report = ia.compute([action], [trade])
        assert report.effects[0].affected_trades == 1
        # No counterfactual difference since trade survived
        assert report.effects[0].net_benefit == 0.0

    def test_stop_tightening_no_matching_symbol(self):
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        trade = _make_trade(pair="MSFT")
        action = _make_action(symbol="AAPL")
        report = ia.compute([action], [trade])
        assert report.effects[0].affected_trades == 0


# ---------------------------------------------------------------------------
# Size-boost analysis
# ---------------------------------------------------------------------------

class TestSizeBoost:
    def test_size_boost_positive_trade(self):
        """Positive trade with size boost → net benefit from extra allocation."""
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        trade = _make_trade(pnl=125.0)  # boosted PnL
        action = _make_action(
            action="size_boost", symbol="AAPL",
            details={"boost_factor": 1.25},
        )
        report = ia.compute([action], [trade])
        effect = report.effects[0]
        assert effect.affected_trades == 1
        # Without boost: 125 / 1.25 = 100. Net benefit = 125 - 100 = 25
        assert effect.net_benefit == pytest.approx(25.0, abs=0.1)

    def test_size_boost_negative_trade(self):
        """Losing trade with size boost → net harm from extra allocation."""
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        trade = _make_trade(pnl=-125.0)
        action = _make_action(
            action="size_boost", symbol="AAPL",
            details={"boost_factor": 1.25},
        )
        report = ia.compute([action], [trade])
        effect = report.effects[0]
        # Without boost: -125 / 1.25 = -100. Net = -125 - (-100) = -25
        assert effect.net_benefit == pytest.approx(-25.0, abs=0.1)

    def test_size_boost_default_factor(self):
        """When no boost_factor in details, uses default 1.25."""
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        trade = _make_trade(pnl=125.0)
        action = _make_action(action="size_boost", symbol="AAPL")
        report = ia.compute([action], [trade])
        assert report.effects[0].net_benefit == pytest.approx(25.0, abs=0.1)


# ---------------------------------------------------------------------------
# Overlay regime summary
# ---------------------------------------------------------------------------

class TestOverlayRegime:
    def test_overlay_regime_grouping(self):
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        trades = [
            _make_trade(pnl=100.0, market_regime="trending"),
            _make_trade(pnl=-50.0, market_regime="trending", pair="MSFT"),
            _make_trade(pnl=200.0, market_regime="ranging", pair="GOOG"),
        ]
        report = ia.compute([_make_action()], trades)
        assert "trending" in report.overlay_regime_summary
        assert "ranging" in report.overlay_regime_summary
        assert report.overlay_regime_summary["trending"]["trades"] == 2
        assert report.overlay_regime_summary["ranging"]["trades"] == 1

    def test_overlay_win_rate_calculation(self):
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        trades = [
            _make_trade(pnl=100.0, market_regime="trending"),
            _make_trade(pnl=-50.0, market_regime="trending", pair="MSFT"),
        ]
        report = ia.compute([_make_action()], trades)
        assert report.overlay_regime_summary["trending"]["win_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Net benefit and recommendation
# ---------------------------------------------------------------------------

class TestNetBenefit:
    def test_positive_benefit_recommendation(self):
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        trade = _make_trade(pnl=125.0)
        action = _make_action(
            action="size_boost", symbol="AAPL",
            details={"boost_factor": 1.25},
        )
        report = ia.compute([action], [trade])
        assert report.net_coordinator_benefit > 0
        assert "net positive" in report.recommendation

    def test_negative_benefit_recommendation(self):
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        trade = _make_trade(pnl=-125.0)
        action = _make_action(
            action="size_boost", symbol="AAPL",
            details={"boost_factor": 1.25},
        )
        report = ia.compute([action], [trade])
        assert report.net_coordinator_benefit < 0
        assert "net negative" in report.recommendation

    def test_zero_benefit_recommendation(self):
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        trade = _make_trade(pnl=100.0, exit_reason="TAKE_PROFIT")
        action = _make_action(action="tighten_stop_be", symbol="AAPL")
        report = ia.compute([action], [trade])
        assert report.net_coordinator_benefit == 0.0
        assert "no measurable impact" in report.recommendation


# ---------------------------------------------------------------------------
# Multiple rules
# ---------------------------------------------------------------------------

class TestMultipleRules:
    def test_multiple_rules_produce_multiple_effects(self):
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        events = [
            _make_action(action="tighten_stop_be", rule="rule_1"),
            _make_action(action="size_boost", rule="rule_2", details={"boost_factor": 1.25}),
        ]
        trades = [_make_trade(pnl=125.0)]
        report = ia.compute(events, trades)
        assert len(report.effects) == 2
        assert report.total_coordination_events == 2

    def test_generic_action_type(self):
        ia = InteractionAnalyzer("2026-01-01", "2026-01-07")
        events = [_make_action(action="overlay_signal_change", rule="ema_crossover")]
        report = ia.compute(events, [])
        assert report.effects[0].action_type == "overlay_signal_change"
        assert report.effects[0].confidence == pytest.approx(0.2)
