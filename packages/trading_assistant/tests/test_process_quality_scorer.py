# tests/test_process_quality_scorer.py
"""Tests for ProcessQualityScorer — deterministic trade quality scoring."""
from datetime import datetime

from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.process_quality import RootCause
from trading_assistant.skills.process_quality_scorer import ProcessQualityScorer, ScorerConfig


def _trade(
    side: str = "LONG",
    regime: str = "trending_up",
    signal_strength: float = 0.8,
    spread: float = 5.0,
    exit_reason: str = "SIGNAL",
    blocked_by: str | None = None,
    pnl: float = 100.0,
) -> TradeEvent:
    return TradeEvent(
        trade_id="t1",
        bot_id="bot1",
        pair="BTCUSDT",
        side=side,
        entry_time=datetime(2025, 6, 1),
        exit_time=datetime(2025, 6, 1),
        entry_price=40000.0,
        exit_price=40100.0,
        position_size=0.1,
        pnl=pnl,
        pnl_pct=pnl / 4000 * 100,
        entry_signal_strength=signal_strength,
        market_regime=regime,
        exit_reason=exit_reason,
        spread_at_entry=spread,
        blocked_by=blocked_by,
    )


class TestRootCauseEnum:
    def test_has_10_members(self):
        assert len(RootCause) == 10

    def test_values_are_lowercase_snake_case(self):
        for member in RootCause:
            assert member.value == member.value.lower()
            assert " " not in member.value


class TestRegimeMismatch:
    def test_long_in_trending_up_no_penalty(self):
        scorer = ProcessQualityScorer()
        result = scorer.score(_trade(side="LONG", regime="trending_up"))
        assert RootCause.REGIME_MISMATCH not in result.root_causes

    def test_short_in_trending_up_penalty(self):
        scorer = ProcessQualityScorer()
        result = scorer.score(_trade(side="SHORT", regime="trending_up"))
        assert RootCause.REGIME_MISMATCH in result.root_causes
        assert result.score < 100


class TestWeakSignal:
    def test_above_threshold_no_penalty(self):
        scorer = ProcessQualityScorer()
        result = scorer.score(_trade(signal_strength=0.8))
        assert RootCause.WEAK_SIGNAL not in result.root_causes

    def test_below_threshold_penalty(self):
        scorer = ProcessQualityScorer()
        result = scorer.score(_trade(signal_strength=0.3))
        assert RootCause.WEAK_SIGNAL in result.root_causes

    def test_exact_boundary_no_penalty(self):
        scorer = ProcessQualityScorer()
        result = scorer.score(_trade(signal_strength=0.5))
        assert RootCause.WEAK_SIGNAL not in result.root_causes


class TestSlippageSpike:
    def test_normal_spread_no_penalty(self):
        scorer = ProcessQualityScorer()
        result = scorer.score(_trade(spread=10.0))
        assert RootCause.SLIPPAGE_SPIKE not in result.root_causes

    def test_high_spread_penalty(self):
        scorer = ProcessQualityScorer()
        result = scorer.score(_trade(spread=25.0))
        assert RootCause.SLIPPAGE_SPIKE in result.root_causes


class TestEarlyExit:
    def test_signal_exit_no_penalty(self):
        scorer = ProcessQualityScorer()
        result = scorer.score(_trade(exit_reason="SIGNAL"))
        assert RootCause.EARLY_EXIT not in result.root_causes

    def test_manual_exit_penalty(self):
        scorer = ProcessQualityScorer()
        result = scorer.score(_trade(exit_reason="MANUAL"))
        assert RootCause.EARLY_EXIT in result.root_causes


class TestCombined:
    def test_multiple_violations_stack(self):
        scorer = ProcessQualityScorer()
        result = scorer.score(_trade(
            side="SHORT",
            regime="trending_up",
            signal_strength=0.3,
            spread=25.0,
        ))
        assert RootCause.REGIME_MISMATCH in result.root_causes
        assert RootCause.WEAK_SIGNAL in result.root_causes
        assert RootCause.SLIPPAGE_SPIKE in result.root_causes
        assert result.score == max(0, 100 - 25 - 20 - 15)

    def test_score_floors_at_zero(self):
        config = ScorerConfig(deductions={
            "regime_mismatch": 50,
            "weak_signal": 50,
            "slippage_spike": 50,
            "early_exit": 50,
            "filter_blocked_good": 50,
        })
        scorer = ProcessQualityScorer(config)
        result = scorer.score(_trade(
            side="SHORT", regime="trending_up",
            signal_strength=0.1, spread=25.0, exit_reason="MANUAL",
        ))
        assert result.score == 0

    def test_evidence_refs_populated(self):
        scorer = ProcessQualityScorer()
        result = scorer.score(_trade(side="SHORT", regime="trending_up"))
        assert len(result.evidence_refs) > 0


class TestCustomConfig:
    def test_different_thresholds_change_scores(self):
        strict = ScorerConfig(signal_strength_min=0.9, max_spread_bps=5.0)
        lenient = ScorerConfig(signal_strength_min=0.1, max_spread_bps=100.0)
        trade = _trade(signal_strength=0.5, spread=15.0)
        strict_result = ProcessQualityScorer(strict).score(trade)
        lenient_result = ProcessQualityScorer(lenient).score(trade)
        assert strict_result.score < lenient_result.score
