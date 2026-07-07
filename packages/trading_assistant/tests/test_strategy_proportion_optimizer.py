# tests/test_strategy_proportion_optimizer.py
"""Tests for the strategy proportion optimizer skill."""

from trading_assistant.schemas.proportion_optimization import (
    StrategyAllocationRecommendation,
    IntraBotAllocationReport,
    ProportionOptimizationReport,
)
from trading_assistant.schemas.weekly_metrics import StrategyWeeklySummary
from trading_assistant.skills.strategy_proportion_optimizer import StrategyProportionOptimizer

_DATES = [
    "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27",
    "2026-02-28", "2026-03-01", "2026-03-02",
]


def _make_strat(
    strategy_id: str,
    bot_id: str,
    daily_pnl: dict[str, float],
) -> StrategyWeeklySummary:
    return StrategyWeeklySummary(
        strategy_id=strategy_id,
        bot_id=bot_id,
        net_pnl=sum(daily_pnl.values()),
        daily_pnl=daily_pnl,
    )


class TestProportionOptimizerSchemas:
    def test_strategy_recommendation_defaults(self):
        r = StrategyAllocationRecommendation(bot_id="b", strategy_id="s")
        assert r.evidence_period_days == 7

    def test_intra_bot_report_defaults(self):
        r = IntraBotAllocationReport(bot_id="b", week_start="x", week_end="y")
        assert r.special_notes == []

    def test_proportion_report_defaults(self):
        r = ProportionOptimizationReport(week_start="x", week_end="y")
        assert r.bot_reports == []


class TestStrategyProportionOptimizer:
    def test_empty_input(self):
        opt = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        report = opt.compute({})
        assert report.bot_reports == []

    def test_single_strategy_bot(self):
        opt = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        strats = {
            "bot1": {
                "only_strat": _make_strat("only_strat", "bot1", {d: 50.0 for d in _DATES}),
            },
        }
        report = opt.compute(strats)
        assert len(report.bot_reports) == 1
        recs = report.bot_reports[0].recommendations
        assert len(recs) == 1
        assert recs[0].suggested_unit_risk_pct == 1.0

    def test_equal_strategies(self):
        opt = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        daily = {d: 50.0 for d in _DATES}
        strats = {
            "bot1": {
                "a": _make_strat("a", "bot1", dict(daily)),
                "b": _make_strat("b", "bot1", dict(daily)),
            },
        }
        report = opt.compute(strats)
        recs = report.bot_reports[0].recommendations
        allocs = [r.suggested_unit_risk_pct for r in recs]
        # Equal performance → similar allocations
        assert abs(allocs[0] - allocs[1]) < 0.5

    def test_dominant_strategy_gets_more(self):
        opt = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        # Strategy A: consistently positive, low vol (small variation for valid Sharpe)
        daily_a = {_DATES[i]: 100.0 + i * 2 for i in range(7)}
        # Strategy B: volatile and losing
        daily_b = {_DATES[i]: (-200.0 if i % 2 == 0 else 50.0) for i in range(7)}
        strats = {
            "bot1": {
                "good": _make_strat("good", "bot1", daily_a),
                "bad": _make_strat("bad", "bot1", daily_b),
            },
        }
        report = opt.compute(strats)
        recs = {r.strategy_id: r for r in report.bot_reports[0].recommendations}
        assert recs["good"].suggested_unit_risk_pct >= recs["bad"].suggested_unit_risk_pct

    def test_constraint_min_pct(self):
        opt = StrategyProportionOptimizer(
            "2026-02-24", "2026-03-02", min_strategy_pct=0.5,
        )
        daily_a = {d: 1000.0 for d in _DATES}
        daily_b = {d: 1.0 for d in _DATES}
        strats = {
            "bot1": {
                "a": _make_strat("a", "bot1", daily_a),
                "b": _make_strat("b", "bot1", daily_b),
            },
        }
        report = opt.compute(strats)
        for rec in report.bot_reports[0].recommendations:
            assert rec.suggested_unit_risk_pct >= 0.1  # default min

    def test_constraint_max_pct(self):
        opt = StrategyProportionOptimizer(
            "2026-02-24", "2026-03-02", max_strategy_pct=2.0,
        )
        daily = {d: 100.0 for d in _DATES}
        strats = {
            "bot1": {
                "a": _make_strat("a", "bot1", daily),
                "b": _make_strat("b", "bot1", dict(daily)),
            },
        }
        report = opt.compute(strats)
        for rec in report.bot_reports[0].recommendations:
            assert rec.suggested_unit_risk_pct <= 2.0

    def test_nq_concentration_warning(self):
        opt = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        daily = {d: 50.0 for d in _DATES}
        strats = {
            "momentum_nq_01": {
                "long": _make_strat("long", "momentum_nq_01", dict(daily)),
                "short": _make_strat("short", "momentum_nq_01", dict(daily)),
            },
        }
        report = opt.compute(strats)
        bot_report = report.bot_reports[0]
        assert any("NQ concentration" in n for n in bot_report.special_notes)

    def test_no_concentration_warning_for_other_bots(self):
        opt = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        daily = {d: 50.0 for d in _DATES}
        strats = {
            "swing_multi_01": {
                "a": _make_strat("a", "swing_multi_01", dict(daily)),
                "b": _make_strat("b", "swing_multi_01", dict(daily)),
            },
        }
        report = opt.compute(strats)
        bot_report = report.bot_reports[0]
        assert not any("NQ concentration" in n for n in bot_report.special_notes)

    def test_rebalance_threshold(self):
        opt = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        daily_a = {d: 200.0 for d in _DATES}
        daily_b = {d: 10.0 for d in _DATES}
        strats = {
            "bot1": {
                "a": _make_strat("a", "bot1", daily_a),
                "b": _make_strat("b", "bot1", daily_b),
            },
        }
        # Current: very different from what optimizer will suggest
        report = opt.compute(strats, {"bot1": {"a": 0.1, "b": 3.0}})
        assert report.bot_reports[0].rebalance_needed is True

    def test_multiple_bots(self):
        opt = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        daily = {d: 50.0 for d in _DATES}
        strats = {
            "bot1": {"a": _make_strat("a", "bot1", dict(daily))},
            "bot2": {
                "x": _make_strat("x", "bot2", dict(daily)),
                "y": _make_strat("y", "bot2", dict(daily)),
            },
        }
        report = opt.compute(strats)
        assert len(report.bot_reports) == 2

    def test_sharpe_computed(self):
        opt = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        daily_a = {d: 100.0 + i * 10 for i, d in enumerate(_DATES)}
        daily_b = {d: 50.0 - i * 5 for i, d in enumerate(_DATES)}
        strats = {
            "bot1": {
                "a": _make_strat("a", "bot1", daily_a),
                "b": _make_strat("b", "bot1", daily_b),
            },
        }
        report = opt.compute(strats)
        bot = report.bot_reports[0]
        assert bot.suggested_bot_sharpe != 0.0

    def test_rationale_populated(self):
        opt = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        daily = {d: 50.0 for d in _DATES}
        strats = {"bot1": {"a": _make_strat("a", "bot1", daily)}}
        report = opt.compute(strats)
        assert report.bot_reports[0].recommendations[0].rationale != ""

    def test_evidence_period_days(self):
        opt = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        daily = {d: 50.0 for d in _DATES}
        strats = {
            "bot1": {
                "a": _make_strat("a", "bot1", daily),
                "b": _make_strat("b", "bot1", dict(daily)),
            },
        }
        report = opt.compute(strats)
        for rec in report.bot_reports[0].recommendations:
            assert rec.evidence_period_days == 7

    def test_serializes_to_json(self):
        opt = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        daily = {d: 50.0 for d in _DATES}
        strats = {
            "bot1": {
                "a": _make_strat("a", "bot1", daily),
                "b": _make_strat("b", "bot1", dict(daily)),
            },
        }
        report = opt.compute(strats)
        data = report.model_dump(mode="json")
        assert isinstance(data["bot_reports"], list)

    def test_empty_strategies_for_bot_skipped(self):
        opt = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        strats = {"bot1": {}}
        report = opt.compute(strats)
        assert report.bot_reports == []
