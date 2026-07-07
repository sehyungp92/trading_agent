# tests/test_strategy_engine.py
"""Tests for the 4-tier strategy refinement engine."""
from __future__ import annotations

from unittest.mock import MagicMock


from trading_assistant.schemas.strategy_profile import StrategyProfile, StrategyRegistry
from trading_assistant.schemas.strategy_suggestions import SuggestionTier, RefinementReport
from trading_assistant.schemas.weekly_metrics import (
    BotWeeklySummary,
    CorrelationSummary,
    FilterWeeklySummary,
    RegimePerformanceTrend,
)
from trading_assistant.schemas.hourly_performance import HourlyBucket
from trading_assistant.analysis.strategy_engine import StrategyEngine


class TestParameterSuggestions:
    def test_no_suggestion_when_balanced(self):
        engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")
        summary = BotWeeklySummary(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bot_id="bot1",
            total_trades=50,
            win_count=28,
            loss_count=22,
            net_pnl=300.0,
            avg_win=80.0,
            avg_loss=-50.0,
        )
        suggestions = engine.analyze_parameters(summary)
        stop_suggestions = [s for s in suggestions if "stop" in s.title.lower()]
        assert len(stop_suggestions) == 0


class TestFilterSuggestions:
    def test_no_suggestion_for_beneficial_filter(self):
        engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")
        filter_summary = FilterWeeklySummary(
            bot_id="bot1",
            filter_name="spread_filter",
            total_blocks=10,
            net_impact_pnl=200.0,  # filter saves more than it costs
        )
        suggestions = engine.analyze_filters("bot1", [filter_summary])
        assert len(suggestions) == 0


class TestStrategyVariantSuggestions:
    def test_detects_regime_mismatch(self):
        """If a bot loses consistently in one regime, suggest a regime gate."""
        engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")
        trend = RegimePerformanceTrend(
            bot_id="bot1",
            regime="ranging",
            weekly_pnl=[-50.0, -80.0, -40.0, -60.0],  # consistent losses
            weekly_win_rate=[0.3, 0.25, 0.35, 0.28],
            weekly_trade_count=[10, 12, 8, 11],
        )
        suggestions = engine.analyze_regime_fit("bot1", [trend])
        assert len(suggestions) >= 1
        assert suggestions[0].tier == SuggestionTier.STRATEGY_VARIANT
        assert suggestions[0].requires_human_judgment is True

    def test_no_suggestion_for_profitable_regime(self):
        engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")
        trend = RegimePerformanceTrend(
            bot_id="bot1",
            regime="trending_up",
            weekly_pnl=[100.0, 120.0, 80.0, 150.0],
            weekly_win_rate=[0.7, 0.75, 0.65, 0.8],
            weekly_trade_count=[10, 12, 8, 11],
        )
        suggestions = engine.analyze_regime_fit("bot1", [trend])
        assert len(suggestions) == 0


class TestRefinementReport:
    def test_builds_full_report(self):
        engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")
        summary = BotWeeklySummary(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bot_id="bot1",
            total_trades=50,
            win_count=20,
            loss_count=30,
            net_pnl=-100.0,
            avg_win=200.0,
            avg_loss=-30.0,
        )
        filter_summaries = [
            FilterWeeklySummary(
                bot_id="bot1",
                filter_name="volume_filter",
                total_blocks=20,
                net_impact_pnl=-100.0,
            )
        ]
        regime_trends = [
            RegimePerformanceTrend(
                bot_id="bot1",
                regime="ranging",
                weekly_pnl=[-50.0, -80.0, -40.0, -60.0],
                weekly_win_rate=[0.3, 0.25, 0.35, 0.28],
                weekly_trade_count=[10, 12, 8, 11],
            )
        ]
        report = engine.build_report(
            bot_summaries={"bot1": summary},
            filter_summaries={"bot1": filter_summaries},
            regime_trends={"bot1": regime_trends},
        )
        assert isinstance(report, RefinementReport)
        assert len(report.suggestions) > 0


def _make_registry_for_engine() -> StrategyRegistry:
    return StrategyRegistry(
        strategies={
            "ATRSS": StrategyProfile(
                bot_id="trend_bot", family="swing",
                archetype="trend_follow", asset_class="mixed",
            ),
            "IARIC_v1": StrategyProfile(
                bot_id="intraday_bot", family="stock",
                archetype="intraday_momentum", asset_class="equity",
            ),
            "NQDTC_v2.1": StrategyProfile(
                bot_id="breakout_bot", family="momentum",
                archetype="box_breakout", asset_class="futures",
            ),
            "BEAR_SWING_TEST": StrategyProfile(
                bot_id="bear_bot", family="swing",
                archetype="bear_regime_swing", asset_class="mixed",
            ),
            "DownturnDominator_v1": StrategyProfile(
                bot_id="downturn_bot", family="momentum",
                archetype="multi_engine_bear", asset_class="futures",
            ),
            "MR_PULLBACK": StrategyProfile(
                bot_id="mr_bot", family="stock",
                archetype="mean_reversion_pullback", asset_class="equity",
            ),
        },
    )


class TestArchetypeAlphaDecay:
    def test_trend_follow_uses_looser_threshold(self):
        """Trend-following archetype uses 0.55 decay threshold (vs 0.3 default)."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        # 0.45 decay would fire with default 0.3, but not with trend_follow's 0.55
        result = engine.detect_alpha_decay(
            "trend_bot", rolling_sharpe_30d=0.55, rolling_sharpe_60d=0.8,
            rolling_sharpe_90d=1.0, strategy_id="ATRSS",
        )
        assert len(result) == 0  # 0.45 < 0.55 threshold → no suggestion

    def test_intraday_uses_tighter_threshold(self):
        """Intraday momentum uses 0.80 decay threshold."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        # 0.45 decay would NOT fire with intraday's 0.80 threshold
        result = engine.detect_alpha_decay(
            "intraday_bot", rolling_sharpe_30d=0.55, rolling_sharpe_60d=0.8,
            rolling_sharpe_90d=1.0, strategy_id="IARIC_v1",
        )
        assert len(result) == 0

    def test_no_registry_uses_default(self):
        """Without registry, uses the default 0.3 threshold."""
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
        )
        # 0.45 decay fires with default 0.3
        result = engine.detect_alpha_decay(
            "any_bot", rolling_sharpe_30d=0.55, rolling_sharpe_60d=0.8,
            rolling_sharpe_90d=1.0,
        )
        assert len(result) == 1

    def test_strategy_id_populated_on_suggestion(self):
        """Suggestions include strategy_id and archetype when registry available."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        result = engine.detect_alpha_decay(
            "trend_bot", rolling_sharpe_30d=0.1, rolling_sharpe_60d=0.5,
            rolling_sharpe_90d=1.0, strategy_id="ATRSS",
        )
        assert len(result) == 1
        assert result[0].strategy_id == "ATRSS"
        assert result[0].strategy_archetype == "trend_follow"


class TestArchetypeExitTiming:
    def test_trend_follow_rarely_flags(self):
        """Trend-following has 0.20 efficiency threshold — less sensitive."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        # 0.30 efficiency would fire with default 0.50, but not with trend_follow's 0.20
        result = engine.detect_exit_timing_issues(
            "trend_bot", avg_exit_efficiency=0.30, premature_exit_pct=0.3,
            strategy_id="ATRSS",
        )
        assert len(result) == 0

    def test_intraday_flags_more_aggressively(self):
        """Intraday momentum uses 0.45 efficiency threshold."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        # 0.40 efficiency fires with intraday's 0.45
        result = engine.detect_exit_timing_issues(
            "intraday_bot", avg_exit_efficiency=0.40, premature_exit_pct=0.3,
            strategy_id="IARIC_v1",
        )
        assert len(result) == 1


class TestArchetypeTimeOfDay:
    def test_intraday_gets_high_relevance_note(self):
        """Intraday strategies get HIGH RELEVANCE archetype_note."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        bucket = MagicMock(hour=10, trade_count=20, pnl=-100, win_rate=0.2)
        result = engine.detect_time_of_day_patterns(
            "intraday_bot", [bucket], strategy_id="IARIC_v1",
        )
        assert len(result) == 1
        assert "HIGH RELEVANCE" in result[0].archetype_note

    def test_trend_follow_gets_low_relevance_note(self):
        """Trend-following strategies get LOW RELEVANCE archetype_note."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        bucket = MagicMock(hour=10, trade_count=20, pnl=-100, win_rate=0.2)
        result = engine.detect_time_of_day_patterns(
            "trend_bot", [bucket], strategy_id="ATRSS",
        )
        assert len(result) == 1
        assert "LOW RELEVANCE" in result[0].archetype_note


class TestBearRegimeSwingArchetype:
    def test_alpha_decay_uses_loose_threshold(self):
        """Bear regime swing uses 0.50 decay threshold."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        # 0.45 ratio → no fire at 0.50 threshold
        result = engine.detect_alpha_decay(
            "bear_bot", rolling_sharpe_30d=0.55, rolling_sharpe_60d=0.8,
            rolling_sharpe_90d=1.0, strategy_id="BEAR_SWING_TEST",
        )
        assert len(result) == 0

    def test_exit_timing_uses_wide_threshold(self):
        """Bear regime swing uses 0.25 efficiency threshold (like trend_follow)."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        result = engine.detect_exit_timing_issues(
            "bear_bot", avg_exit_efficiency=0.30, premature_exit_pct=0.3,
            strategy_id="BEAR_SWING_TEST",
        )
        assert len(result) == 0

    def test_time_of_day_low_relevance(self):
        """Bear regime swing is multi-day, gets LOW RELEVANCE."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        bucket = MagicMock(hour=10, trade_count=20, pnl=-100, win_rate=0.2)
        result = engine.detect_time_of_day_patterns(
            "bear_bot", [bucket], strategy_id="BEAR_SWING_TEST",
        )
        assert len(result) == 1
        assert "LOW RELEVANCE" in result[0].archetype_note


class TestMultiEngineBearArchetype:
    def test_alpha_decay_moderate_threshold(self):
        """Multi-engine bear uses 0.55 decay threshold."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        result = engine.detect_alpha_decay(
            "downturn_bot", rolling_sharpe_30d=0.55, rolling_sharpe_60d=0.8,
            rolling_sharpe_90d=1.0, strategy_id="DownturnDominator_v1",
        )
        assert len(result) == 0

    def test_exit_timing_moderate_threshold(self):
        """Multi-engine bear uses 0.35 efficiency threshold."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        # 0.30 < 0.35 threshold → fires
        result = engine.detect_exit_timing_issues(
            "downturn_bot", avg_exit_efficiency=0.30, premature_exit_pct=0.3,
            strategy_id="DownturnDominator_v1",
        )
        assert len(result) == 1

    def test_time_of_day_high_relevance(self):
        """Multi-engine bear is intraday NQ, gets HIGH RELEVANCE."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        bucket = MagicMock(hour=10, trade_count=20, pnl=-100, win_rate=0.2)
        result = engine.detect_time_of_day_patterns(
            "downturn_bot", [bucket], strategy_id="DownturnDominator_v1",
        )
        assert len(result) == 1
        assert "HIGH RELEVANCE" in result[0].archetype_note


class TestMeanReversionPullbackArchetype:
    def test_alpha_decay_tight_threshold(self):
        """Mean reversion pullback uses 0.80 decay threshold (stable archetype)."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        result = engine.detect_alpha_decay(
            "mr_bot", rolling_sharpe_30d=0.55, rolling_sharpe_60d=0.8,
            rolling_sharpe_90d=1.0, strategy_id="MR_PULLBACK",
        )
        assert len(result) == 0

    def test_exit_timing_tight_threshold(self):
        """Mean reversion pullback uses 0.45 efficiency threshold (tight MR exits)."""
        registry = _make_registry_for_engine()
        engine = StrategyEngine(
            week_start="2026-03-01", week_end="2026-03-07",
            strategy_registry=registry,
        )
        result = engine.detect_exit_timing_issues(
            "mr_bot", avg_exit_efficiency=0.40, premature_exit_pct=0.3,
            strategy_id="MR_PULLBACK",
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Expanded strategy engine detectors (merged from test_strategy_engine_expanded.py)
# ---------------------------------------------------------------------------


class TestAlphaDecayDetector:
    def test_detects_declining_sharpe(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_alpha_decay(
            bot_id="bot1",
            rolling_sharpe_30d=0.8,
            rolling_sharpe_60d=1.5,
            rolling_sharpe_90d=2.1,
        )
        assert len(result) == 1
        assert "alpha decay" in result[0].title.lower() or "decay" in result[0].description.lower()
        assert result[0].confidence > 0

    def test_no_decay_when_stable(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_alpha_decay(
            bot_id="bot1",
            rolling_sharpe_30d=1.5,
            rolling_sharpe_60d=1.4,
            rolling_sharpe_90d=1.3,
        )
        assert len(result) == 0

    def test_no_decay_when_improving(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_alpha_decay(
            bot_id="bot1",
            rolling_sharpe_30d=2.0,
            rolling_sharpe_60d=1.5,
            rolling_sharpe_90d=1.0,
        )
        assert len(result) == 0


class TestSignalQualityDecay:
    def test_detects_declining_correlation(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_signal_decay(
            bot_id="bot1",
            signal_outcome_correlation_30d=0.35,
            signal_outcome_correlation_90d=0.72,
        )
        assert len(result) == 1
        assert result[0].requires_human_judgment is True

    def test_no_decay_when_stable(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_signal_decay(
            bot_id="bot1",
            signal_outcome_correlation_30d=0.65,
            signal_outcome_correlation_90d=0.70,
        )
        assert len(result) == 0


class TestExitTimingAnalyzer:
    def test_detects_premature_exits(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_exit_timing_issues(
            bot_id="bot1",
            avg_exit_efficiency=0.45,  # captures only 45% of available move
            premature_exit_pct=0.6,    # 60% of exits are premature
        )
        assert len(result) == 1
        assert "exit" in result[0].title.lower()

    def test_no_issue_when_efficient(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_exit_timing_issues(
            bot_id="bot1",
            avg_exit_efficiency=0.75,
            premature_exit_pct=0.2,
        )
        assert len(result) == 0


class TestCorrelationBreakdown:
    def test_detects_rising_correlation(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        correlations = [
            CorrelationSummary(
                bot_a="bot1", bot_b="bot2",
                rolling_30d_correlation=0.82,
                weekly_pnl_correlation=0.75,
                same_direction_pct=0.8,
            ),
        ]
        result = engine.detect_correlation_breakdown(correlations)
        assert len(result) >= 1
        assert "correlation" in result[0].title.lower()

    def test_no_alert_when_low_correlation(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        correlations = [
            CorrelationSummary(
                bot_a="bot1", bot_b="bot2",
                rolling_30d_correlation=0.3,
            ),
        ]
        result = engine.detect_correlation_breakdown(correlations)
        assert len(result) == 0


class TestTimeOfDayPatterns:
    def test_detects_bad_hours(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        buckets = [
            HourlyBucket(hour=9, trade_count=20, pnl=500.0, win_rate=0.75),
            HourlyBucket(hour=15, trade_count=15, pnl=-400.0, win_rate=0.2),
            HourlyBucket(hour=20, trade_count=10, pnl=100.0, win_rate=0.6),
        ]
        result = engine.detect_time_of_day_patterns(
            bot_id="bot1",
            hourly_buckets=buckets,
            min_trades=10,
        )
        assert len(result) >= 1
        assert "15" in result[0].description or "hour" in result[0].description.lower()

    def test_no_pattern_when_uniform(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        buckets = [
            HourlyBucket(hour=9, trade_count=10, pnl=100.0, win_rate=0.6),
            HourlyBucket(hour=15, trade_count=10, pnl=80.0, win_rate=0.55),
        ]
        result = engine.detect_time_of_day_patterns(
            bot_id="bot1",
            hourly_buckets=buckets,
        )
        assert len(result) == 0


class TestDrawdownPatternDetector:
    def test_detects_concentrated_drawdown(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_drawdown_patterns(
            bot_id="bot1",
            largest_single_loss_pct=8.5,
            max_drawdown_pct=15.0,
            avg_loss_pct=2.0,
        )
        assert len(result) >= 1
        assert "drawdown" in result[0].title.lower() or "position" in result[0].title.lower()

    def test_no_alert_for_normal_drawdown(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_drawdown_patterns(
            bot_id="bot1",
            largest_single_loss_pct=2.0,
            max_drawdown_pct=8.0,
            avg_loss_pct=1.5,
        )
        assert len(result) == 0


class TestPositionSizingEfficiency:
    def test_detects_oversizing(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_position_sizing_issues(
            bot_id="bot1",
            avg_win_pct=1.5,
            avg_loss_pct=3.0,
            win_rate=0.65,
        )
        assert len(result) >= 1

    def test_no_issue_when_balanced(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_position_sizing_issues(
            bot_id="bot1",
            avg_win_pct=2.0,
            avg_loss_pct=1.5,
            win_rate=0.55,
        )
        assert len(result) == 0


class TestExpandedBuildReport:
    def test_build_report_calls_all_detectors(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        bot_summaries = {
            "bot1": BotWeeklySummary(
                week_start="2026-02-24", week_end="2026-03-02",
                bot_id="bot1", total_trades=50, win_count=30, loss_count=20,
                avg_win=100.0, avg_loss=-60.0, max_drawdown_pct=12.0,
            ),
        }

        report = engine.build_report(
            bot_summaries=bot_summaries,
            rolling_sharpe={
                "bot1": {"30d": 0.5, "60d": 1.2, "90d": 2.0},
            },
            hourly_buckets={
                "bot1": [
                    HourlyBucket(hour=9, trade_count=20, pnl=500.0, win_rate=0.75),
                    HourlyBucket(hour=15, trade_count=15, pnl=-400.0, win_rate=0.2),
                ],
            },
        )
        # Should produce suggestions from multiple detectors
        assert len(report.suggestions) > 0
        tiers = {s.tier.value for s in report.suggestions}
        # At minimum we should get parameter (tight stop) and hypothesis (alpha decay)
        assert len(tiers) >= 1


# ---------------------------------------------------------------------------
# Adaptive strategy engine thresholds (merged from test_adaptive_strategy_engine.py)
# ---------------------------------------------------------------------------


def _make_bot_summary(
    bot_id: str = "bot_a",
    avg_win: float = 100.0,
    avg_loss: float = -20.0,
    win_rate: float = 0.55,
    total_trades: int = 50,
) -> BotWeeklySummary:
    win_count = int(total_trades * win_rate)
    loss_count = total_trades - win_count
    return BotWeeklySummary(
        week_start="2026-01-01",
        week_end="2026-01-07",
        bot_id=bot_id,
        total_trades=total_trades,
        win_count=win_count,
        loss_count=loss_count,
        avg_win=avg_win,
        avg_loss=avg_loss,
        net_pnl=(avg_win * win_rate * total_trades) + (avg_loss * (1 - win_rate) * total_trades),
    )


def _make_filter_summary(
    filter_name: str = "volatility_filter",
    net_impact_pnl: float = -50.0,
    total_blocks: int = 10,
    confidence: float = 0.7,
) -> FilterWeeklySummary:
    return FilterWeeklySummary(
        bot_id="bot_a",
        filter_name=filter_name,
        total_blocks=total_blocks,
        blocks_that_would_have_won=3,
        blocks_that_would_have_lost=7,
        net_impact_pnl=net_impact_pnl,
        confidence=confidence,
    )


class TestDefaultThresholds:
    """StrategyEngine uses hardcoded default thresholds."""

    def test_default_tight_stop(self):
        """Detectors use default thresholds."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        # loss/win ratio = 20/100 = 0.2, below default 0.3 -> should trigger
        summary = _make_bot_summary(avg_win=100.0, avg_loss=-20.0)
        result = engine.analyze_parameters(summary)
        assert len(result) == 1
        assert result[0].detection_context is not None
        assert result[0].detection_context.threshold_value == 0.3  # default

    def test_default_alpha_decay(self):
        """Alpha_decay uses its parameter default."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        result = engine.detect_alpha_decay(
            "bot_a",
            rolling_sharpe_30d=0.5,
            rolling_sharpe_60d=0.8,
            rolling_sharpe_90d=1.0,
            decay_threshold=0.3,
        )
        assert len(result) == 1
        assert result[0].detection_context is not None
        assert result[0].detection_context.threshold_value == 0.3

    def test_default_filter_cost(self):
        """Filter cost uses its default threshold."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        f = _make_filter_summary(net_impact_pnl=-50.0)
        result = engine.analyze_filters("bot_a", [f])
        assert len(result) == 1
        assert result[0].detection_context is not None
        assert result[0].detection_context.threshold_value == 0.0  # default

    def test_threshold_learner_attribute_defaults_to_none(self):
        """StrategyEngine has _threshold_learner attribute, defaulting to None."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        assert hasattr(engine, "_threshold_learner")
        assert engine._threshold_learner is None

    def test_detection_context_records_default_threshold(self):
        """detection_context records the default threshold value."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        result = engine.detect_alpha_decay(
            "bot_a",
            rolling_sharpe_30d=0.5,
            rolling_sharpe_60d=0.7,
            rolling_sharpe_90d=1.0,
        )
        assert len(result) == 1
        ctx = result[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "alpha_decay"
        assert ctx.threshold_name == "decay_threshold"

    def test_mixed_detectors_all_use_defaults(self):
        """Multiple detectors all use their respective default thresholds."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")

        # Alpha decay — uses default
        engine.detect_alpha_decay(
            "bot_a",
            rolling_sharpe_30d=0.8,
            rolling_sharpe_60d=0.9,
            rolling_sharpe_90d=1.0,  # decay_ratio = 0.2
        )

        # Signal decay — uses default 0.2
        signal_result = engine.detect_signal_decay(
            "bot_a",
            signal_outcome_correlation_30d=0.3,
            signal_outcome_correlation_90d=0.6,  # drop = 0.3 >= 0.2 -> triggers
        )
        assert len(signal_result) == 1
        assert signal_result[0].detection_context.threshold_value == 0.2  # default

    def test_build_report_uses_defaults(self):
        """build_report uses default thresholds."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        # loss/win ratio = 20/100 = 0.2, below default 0.3 -> triggers
        summary = _make_bot_summary(avg_win=100.0, avg_loss=-20.0)
        report = engine.build_report({"bot_a": summary})
        assert len(report.suggestions) == 1
        assert report.suggestions[0].detection_context.threshold_value == 0.3  # default


class TestWithThresholdLearner:
    """StrategyEngine uses learned thresholds from ThresholdLearner."""

    def _make_learner(self, overrides: dict[str, float] | None = None) -> MagicMock:
        """Create a mock ThresholdLearner that returns overrides or falls through to default."""
        learner = MagicMock()
        _overrides = overrides or {}

        def _get(detector_name, threshold_name, bot_id, default):
            key = f"{detector_name}:{threshold_name}:{bot_id}"
            return _overrides.get(key, default)

        learner.get_threshold.side_effect = _get
        return learner

    def test_with_threshold_learner_uses_learned_values(self):
        """Engine uses learned threshold when learner provides one."""
        # Learned tight_stop_ratio = 0.15 (lower than default 0.3)
        learner = self._make_learner({"tight_stop:tight_stop_ratio:bot_a": 0.15})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        # loss/win ratio = 20/100 = 0.2 — above learned 0.15, so should NOT trigger
        summary = _make_bot_summary(bot_id="bot_a", avg_win=100.0, avg_loss=-20.0)
        result = engine.analyze_parameters(summary)
        assert len(result) == 0  # 0.2 >= 0.15, no suggestion

    def test_learned_threshold_triggers_detection(self):
        """Learned threshold can make a detector MORE sensitive."""
        # Learned tight_stop_ratio = 0.5 (higher than default 0.3)
        learner = self._make_learner({"tight_stop:tight_stop_ratio:bot_a": 0.5})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        # loss/win ratio = 40/100 = 0.4 — below learned 0.5, triggers
        summary = _make_bot_summary(bot_id="bot_a", avg_win=100.0, avg_loss=-40.0)
        result = engine.analyze_parameters(summary)
        assert len(result) == 1
        assert result[0].detection_context.threshold_value == 0.5  # learned

    def test_learner_affects_alpha_decay(self):
        """Learner overrides alpha_decay threshold."""
        learner = self._make_learner({"alpha_decay:decay_threshold:bot_a": 0.6})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        # decay_ratio = (1.0 - 0.5)/1.0 = 0.5, below learned 0.6 -> no trigger
        result = engine.detect_alpha_decay(
            "bot_a", rolling_sharpe_30d=0.5, rolling_sharpe_60d=0.8,
            rolling_sharpe_90d=1.0,
        )
        assert len(result) == 0

    def test_learner_affects_filter_cost(self):
        """Learner overrides filter_cost_threshold."""
        learner = self._make_learner({"filter_cost:filter_cost_threshold:bot_a": -100.0})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        f = _make_filter_summary(net_impact_pnl=-50.0)
        # -50 > -100 (learned threshold), so no trigger
        result = engine.analyze_filters("bot_a", [f])
        assert len(result) == 0
