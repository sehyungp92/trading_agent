# tests/test_strategy_suggestions.py
"""Tests for strategy suggestion schemas."""
from trading_assistant.schemas.strategy_suggestions import (
    SuggestionTier,
    StrategySuggestion,
    RefinementReport,
)


class TestSuggestionTier:
    def test_all_tiers_exist(self):
        assert SuggestionTier.PARAMETER == "parameter"
        assert SuggestionTier.FILTER == "filter"
        assert SuggestionTier.STRATEGY_VARIANT == "strategy_variant"
        assert SuggestionTier.HYPOTHESIS == "hypothesis"


class TestStrategySuggestion:
    def test_parameter_suggestion(self):
        s = StrategySuggestion(
            tier=SuggestionTier.PARAMETER,
            bot_id="bot2",
            title="Relax RSI threshold in ranging markets",
            description=(
                "Bot2's RSI threshold of 30 is too aggressive in ranging markets — "
                "entries at RSI 35 had better outcomes over 30 days"
            ),
            current_value="rsi_threshold=30",
            suggested_value="rsi_threshold=35",
            evidence_days=30,
            estimated_impact_pnl=120.0,
            confidence=0.8,
            simulation_assumptions=["mid_fill", "5bps_slippage", "fees_included"],
        )
        assert s.tier == SuggestionTier.PARAMETER
        assert s.confidence == 0.8

    def test_filter_suggestion(self):
        s = StrategySuggestion(
            tier=SuggestionTier.FILTER,
            bot_id="bot3",
            title="Relax volume filter from 2x to 1.5x avg",
            description="Volume filter blocked 47 entries this month. 31 would have been profitable.",
            current_value="volume_filter=2.0x",
            suggested_value="volume_filter=1.5x",
            evidence_days=30,
            estimated_impact_pnl=180.0,
            confidence=0.65,
        )
        assert s.tier == SuggestionTier.FILTER

    def test_strategy_variant(self):
        s = StrategySuggestion(
            tier=SuggestionTier.STRATEGY_VARIANT,
            bot_id="bot1",
            title="Add regime gate: mean-reversion when ADX < 20",
            description="EMA cross strategy loses in ranging. Consider regime gate.",
            requires_human_judgment=True,
        )
        assert s.requires_human_judgment is True

    def test_hypothesis(self):
        s = StrategySuggestion(
            tier=SuggestionTier.HYPOTHESIS,
            bot_id="",
            title="OI increase + negative funding precedes reversals",
            description="Data shows pattern. Needs backtest validation.",
            requires_human_judgment=True,
            confidence=0.3,
        )
        assert s.tier == SuggestionTier.HYPOTHESIS
        assert s.confidence == 0.3


class TestRefinementReport:
    def test_creates_report(self):
        report = RefinementReport(
            week_start="2026-02-23",
            week_end="2026-03-01",
            suggestions=[
                StrategySuggestion(
                    tier=SuggestionTier.PARAMETER,
                    bot_id="bot2",
                    title="Adjust RSI threshold",
                    description="Better outcomes at RSI 35",
                    confidence=0.8,
                ),
                StrategySuggestion(
                    tier=SuggestionTier.FILTER,
                    bot_id="bot3",
                    title="Relax volume filter",
                    description="Filter cost exceeds benefit",
                    confidence=0.65,
                ),
            ],
        )
        assert len(report.suggestions) == 2
        assert report.suggestions_by_tier["parameter"] == 1
        assert report.suggestions_by_tier["filter"] == 1

    def test_empty_report(self):
        report = RefinementReport(
            week_start="2026-02-23", week_end="2026-03-01"
        )
        assert len(report.suggestions) == 0
        assert report.suggestions_by_tier == {}

    def test_high_confidence_only(self):
        report = RefinementReport(
            week_start="2026-02-23",
            week_end="2026-03-01",
            suggestions=[
                StrategySuggestion(
                    tier=SuggestionTier.PARAMETER,
                    bot_id="bot1",
                    title="High conf",
                    description="...",
                    confidence=0.85,
                ),
                StrategySuggestion(
                    tier=SuggestionTier.HYPOTHESIS,
                    bot_id="bot2",
                    title="Low conf",
                    description="...",
                    confidence=0.3,
                ),
            ],
        )
        high_conf = [s for s in report.suggestions if s.confidence >= 0.7]
        assert len(high_conf) == 1
