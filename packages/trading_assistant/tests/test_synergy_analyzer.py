# tests/test_synergy_analyzer.py
"""Tests for the synergy analyzer skill."""

from trading_assistant.schemas.synergy_analysis import (
    StrategyPairAnalysis,
    SynergyReport,
)
from trading_assistant.schemas.weekly_metrics import StrategyWeeklySummary
from trading_assistant.skills.synergy_analyzer import SynergyAnalyzer

_DATES = [
    "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27",
    "2026-02-28", "2026-03-01", "2026-03-02",
]


def _make_strat(
    strategy_id: str,
    bot_id: str,
    daily_pnl: dict[str, float],
    **kwargs,
) -> StrategyWeeklySummary:
    total = sum(daily_pnl.values())
    return StrategyWeeklySummary(
        strategy_id=strategy_id,
        bot_id=bot_id,
        net_pnl=total,
        daily_pnl=daily_pnl,
        **kwargs,
    )


class TestSynergyAnalyzerSchemas:
    def test_pair_analysis_defaults(self):
        p = StrategyPairAnalysis(strategy_a="a:x", strategy_b="b:y")
        assert p.classification == "neutral"

    def test_synergy_report_defaults(self):
        r = SynergyReport(week_start="2026-02-24", week_end="2026-03-02")
        assert r.total_strategies == 0


class TestSynergyAnalyzer:
    def test_empty_input(self):
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        report = analyzer.compute({})
        assert report.total_strategies == 0
        assert report.strategy_pairs == []

    def test_single_strategy(self):
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {
            "bot1": {
                "strat_a": _make_strat("strat_a", "bot1", {d: 10.0 for d in _DATES}),
            },
        }
        report = analyzer.compute(strats)
        assert report.total_strategies == 1
        assert len(report.strategy_pairs) == 0
        assert len(report.marginal_contributions) == 1

    def test_two_identical_strategies_redundant(self):
        # Use identical varying PnL (constant PnL has zero variance → undefined corr)
        daily_a = {_DATES[i]: 50.0 + i * 10 for i in range(7)}
        daily_b = {_DATES[i]: 50.0 + i * 10 for i in range(7)}
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {
            "bot1": {"strat_a": _make_strat("strat_a", "bot1", daily_a)},
            "bot2": {"strat_b": _make_strat("strat_b", "bot2", daily_b)},
        }
        report = analyzer.compute(strats)
        assert len(report.strategy_pairs) == 1
        pair = report.strategy_pairs[0]
        assert pair.correlation_30d > 0.99
        assert pair.classification in ("redundant", "cannibalistic")

    def test_two_uncorrelated_strategies_complementary(self):
        # One goes up, one goes down, alternating
        daily_a = {_DATES[i]: (100.0 if i % 2 == 0 else -100.0) for i in range(7)}
        daily_b = {_DATES[i]: (-100.0 if i % 2 == 0 else 100.0) for i in range(7)}
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {
            "bot1": {"strat_a": _make_strat("strat_a", "bot1", daily_a)},
            "bot2": {"strat_b": _make_strat("strat_b", "bot2", daily_b)},
        }
        report = analyzer.compute(strats)
        pair = report.strategy_pairs[0]
        assert pair.correlation_30d < 0.0
        assert pair.classification == "complementary"
        assert len(report.complementary_pairs) == 1

    def test_neutral_classification(self):
        daily_a = {_DATES[i]: float(i * 10) for i in range(7)}
        daily_b = {_DATES[i]: float(30 + (i % 3) * 20) for i in range(7)}
        analyzer = SynergyAnalyzer(
            "2026-02-24", "2026-03-02",
            correlation_threshold_redundant=0.95,
            correlation_threshold_complementary=-0.5,
        )
        strats = {
            "bot1": {"strat_a": _make_strat("strat_a", "bot1", daily_a)},
            "bot2": {"strat_b": _make_strat("strat_b", "bot2", daily_b)},
        }
        report = analyzer.compute(strats)
        pair = report.strategy_pairs[0]
        assert pair.classification == "neutral"

    def test_same_instrument_detection_same_bot(self):
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {
            "momentum_nq_01": {
                "strat_a": _make_strat("strat_a", "momentum_nq_01", {d: 50.0 for d in _DATES}),
                "strat_b": _make_strat("strat_b", "momentum_nq_01", {d: 30.0 for d in _DATES}),
            },
        }
        report = analyzer.compute(strats)
        pair = report.strategy_pairs[0]
        assert pair.same_instrument is True

    def test_different_bots_not_same_instrument(self):
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {
            "bot1": {"strat_a": _make_strat("strat_a", "bot1", {d: 50.0 for d in _DATES})},
            "bot2": {"strat_b": _make_strat("strat_b", "bot2", {d: 30.0 for d in _DATES})},
        }
        report = analyzer.compute(strats)
        pair = report.strategy_pairs[0]
        assert pair.same_instrument is False

    def test_marginal_contribution_calculation(self):
        # Strategy A: positive with variation
        daily_a = {_DATES[i]: 100.0 + i * 5 for i in range(7)}
        # Strategy B: noisy and losing — adds variance without helping return
        daily_b = {_DATES[i]: (-150.0 if i % 2 == 0 else 80.0) for i in range(7)}
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {
            "bot1": {"good": _make_strat("good", "bot1", daily_a)},
            "bot2": {"bad": _make_strat("bad", "bot2", daily_b)},
        }
        report = analyzer.compute(strats)
        contribs = {c.strategy_key: c for c in report.marginal_contributions}
        # Good strategy should have positive marginal Sharpe
        assert contribs["bot1:good"].marginal_sharpe > 0
        # Bad strategy adds noise and negative return → negative marginal Sharpe
        assert contribs["bot2:bad"].marginal_sharpe < 0

    def test_pnl_contribution_pct(self):
        daily_a = {d: 100.0 for d in _DATES}
        daily_b = {d: 100.0 for d in _DATES}
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {
            "bot1": {"strat_a": _make_strat("strat_a", "bot1", daily_a)},
            "bot2": {"strat_b": _make_strat("strat_b", "bot2", daily_b)},
        }
        report = analyzer.compute(strats)
        contribs = {c.strategy_key: c for c in report.marginal_contributions}
        # Equal PnL → 50% each
        assert abs(contribs["bot1:strat_a"].pnl_contribution_pct - 50.0) < 0.1

    def test_signal_overlap_pct(self):
        # Both active every day → 100% overlap
        daily_a = {d: 50.0 for d in _DATES}
        daily_b = {d: 30.0 for d in _DATES}
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {
            "bot1": {"strat_a": _make_strat("strat_a", "bot1", daily_a)},
            "bot2": {"strat_b": _make_strat("strat_b", "bot2", daily_b)},
        }
        report = analyzer.compute(strats)
        assert report.strategy_pairs[0].signal_overlap_pct == 100.0

    def test_multiple_pairs(self):
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {
            "bot1": {"a": _make_strat("a", "bot1", {d: 50.0 for d in _DATES})},
            "bot2": {"b": _make_strat("b", "bot2", {d: 30.0 for d in _DATES})},
            "bot3": {"c": _make_strat("c", "bot3", {d: 20.0 for d in _DATES})},
        }
        report = analyzer.compute(strats)
        # 3 strategies → 3 pairs (3 choose 2)
        assert len(report.strategy_pairs) == 3
        assert report.total_strategies == 3

    def test_60d_90d_correlation_none(self):
        """60d/90d correlations should be None for single-week data."""
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {
            "bot1": {"a": _make_strat("a", "bot1", {d: 50.0 for d in _DATES})},
            "bot2": {"b": _make_strat("b", "bot2", {d: 30.0 for d in _DATES})},
        }
        report = analyzer.compute(strats)
        pair = report.strategy_pairs[0]
        assert pair.correlation_60d is None
        assert pair.correlation_90d is None

    def test_serializes_to_json(self):
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {
            "bot1": {"a": _make_strat("a", "bot1", {d: 50.0 for d in _DATES})},
            "bot2": {"b": _make_strat("b", "bot2", {d: 30.0 for d in _DATES})},
        }
        report = analyzer.compute(strats)
        data = report.model_dump(mode="json")
        assert isinstance(data["strategy_pairs"], list)
        assert isinstance(data["marginal_contributions"], list)

    def test_recommendation_for_redundant(self):
        daily = {d: 50.0 for d in _DATES}
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {
            "bot1": {"a": _make_strat("a", "bot1", daily)},
            "bot2": {"b": _make_strat("b", "bot2", dict(daily))},
        }
        report = analyzer.compute(strats)
        pair = report.strategy_pairs[0]
        assert pair.recommendation != ""

    def test_recommendation_for_complementary(self):
        daily_a = {_DATES[i]: (100.0 if i % 2 == 0 else -100.0) for i in range(7)}
        daily_b = {_DATES[i]: (-100.0 if i % 2 == 0 else 100.0) for i in range(7)}
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {
            "bot1": {"a": _make_strat("a", "bot1", daily_a)},
            "bot2": {"b": _make_strat("b", "bot2", daily_b)},
        }
        report = analyzer.compute(strats)
        pair = report.strategy_pairs[0]
        assert "complement" in pair.recommendation.lower()
