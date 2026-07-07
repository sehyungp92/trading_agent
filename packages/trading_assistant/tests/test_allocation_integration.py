# tests/test_allocation_integration.py
"""End-to-end integration tests for the allocation pipeline.

Flow: curated daily data → WeeklyMetricsBuilder → PortfolioAllocator
→ SynergyAnalyzer → StrategyProportionOptimizer → WeeklyPromptAssembler
"""
import json
from pathlib import Path


from trading_assistant.schemas.daily_metrics import BotDailySummary, PerStrategySummary
from trading_assistant.skills.build_weekly_metrics import WeeklyMetricsBuilder
from trading_assistant.skills.portfolio_allocator import PortfolioAllocator
from trading_assistant.skills.synergy_analyzer import SynergyAnalyzer
from trading_assistant.skills.strategy_proportion_optimizer import StrategyProportionOptimizer

_DATES = [
    "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27",
    "2026-02-28", "2026-03-01", "2026-03-02",
]

_BOTS = ["k_stock_trader", "momentum_nq_01", "swing_multi_01"]


def _make_daily_with_strategies(
    date: str, bot_id: str, strategies: dict[str, dict],
) -> BotDailySummary:
    per_strat = {}
    base_pnl = 0.0
    base_trades = 0
    base_wins = 0
    base_losses = 0
    for sid, vals in strategies.items():
        ps = PerStrategySummary(strategy_id=sid, **vals)
        per_strat[sid] = ps
        base_pnl += ps.net_pnl
        base_trades += ps.trades
        base_wins += ps.win_count
        base_losses += ps.loss_count

    return BotDailySummary(
        date=date,
        bot_id=bot_id,
        total_trades=base_trades,
        win_count=base_wins,
        loss_count=base_losses,
        net_pnl=base_pnl,
        gross_pnl=base_pnl + 10.0,
        avg_win=50.0,
        avg_loss=-30.0,
        max_drawdown_pct=3.0,
        per_strategy_summary=per_strat,
    )


def _build_seven_day_dailies() -> dict[str, list[BotDailySummary]]:
    """Create 7 days of daily data for 3 bots with per-strategy breakdowns."""
    dailies: dict[str, list[BotDailySummary]] = {b: [] for b in _BOTS}

    for i, date in enumerate(_DATES):
        # k_stock_trader: single strategy K4
        dailies["k_stock_trader"].append(
            _make_daily_with_strategies(date, "k_stock_trader", {
                "K4": {
                    "trades": 5 + i, "win_count": 3 + (i % 2), "loss_count": 2 + (i % 3),
                    "net_pnl": 100.0 + i * 15, "gross_pnl": 110.0 + i * 15,
                    "avg_win": 50.0, "avg_loss": -25.0,
                },
            })
        )

        # momentum_nq_01: two strategies on NQ
        dailies["momentum_nq_01"].append(
            _make_daily_with_strategies(date, "momentum_nq_01", {
                "momentum_long": {
                    "trades": 4, "win_count": 2 + (i % 2), "loss_count": 2 - (i % 2),
                    "net_pnl": 80.0 + i * 10, "gross_pnl": 90.0 + i * 10,
                    "avg_win": 60.0, "avg_loss": -20.0,
                    "symbols_traded": ["NQ"],
                },
                "momentum_short": {
                    "trades": 3, "win_count": 1, "loss_count": 2,
                    "net_pnl": -30.0 + i * 5, "gross_pnl": -20.0 + i * 5,
                    "avg_win": 40.0, "avg_loss": -35.0,
                    "symbols_traded": ["NQ"],
                },
            })
        )

        # swing_multi_01: two strategies on different instruments
        dailies["swing_multi_01"].append(
            _make_daily_with_strategies(date, "swing_multi_01", {
                "ATRSS": {
                    "trades": 2, "win_count": 1, "loss_count": 1,
                    "net_pnl": 60.0 + i * 8, "gross_pnl": 70.0 + i * 8,
                    "avg_win": 80.0, "avg_loss": -20.0,
                    "symbols_traded": ["MNQ"],
                },
                "Helix": {
                    "trades": 3, "win_count": 2, "loss_count": 1,
                    "net_pnl": 40.0 + i * 12, "gross_pnl": 50.0 + i * 12,
                    "avg_win": 45.0, "avg_loss": -10.0,
                    "symbols_traded": ["NQ"],
                },
            })
        )

    return dailies


class TestAllocationIntegration:
    def test_weekly_metrics_includes_per_strategy(self):
        """WeeklyMetricsBuilder aggregates per-strategy data correctly."""
        dailies = _build_seven_day_dailies()
        builder = WeeklyMetricsBuilder("2026-02-24", "2026-03-02", _BOTS)
        portfolio = builder.build_portfolio_summary(dailies)

        for bot_id in _BOTS:
            bot = portfolio.bot_summaries[bot_id]
            assert len(bot.per_strategy_summary) > 0
            for _sid, strat in bot.per_strategy_summary.items():
                assert strat.total_trades > 0
                assert len(strat.daily_pnl) == 7

    def test_portfolio_allocator_with_real_data(self):
        """PortfolioAllocator produces valid recommendations from built data."""
        dailies = _build_seven_day_dailies()
        builder = WeeklyMetricsBuilder("2026-02-24", "2026-03-02", _BOTS)
        portfolio = builder.build_portfolio_summary(dailies)

        allocator = PortfolioAllocator("2026-02-24", "2026-03-02")
        current = {bid: 100.0 / 3 for bid in _BOTS}
        report = allocator.compute(portfolio.bot_summaries, current)

        assert len(report.recommendations) == 3
        total = sum(r.suggested_allocation_pct for r in report.recommendations)
        assert abs(total - 100.0) < 0.5
        assert report.method == "risk_parity_calmar_tilt"

    def test_synergy_analyzer_with_real_data(self):
        """SynergyAnalyzer analyzes all strategy pairs from built data."""
        dailies = _build_seven_day_dailies()
        builder = WeeklyMetricsBuilder("2026-02-24", "2026-03-02", _BOTS)
        portfolio = builder.build_portfolio_summary(dailies)

        per_strat = {
            bid: s.per_strategy_summary
            for bid, s in portfolio.bot_summaries.items()
        }
        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        report = analyzer.compute(per_strat)

        # 5 strategies → 10 pairs (5 choose 2)
        assert report.total_strategies == 5
        assert len(report.strategy_pairs) == 10
        assert len(report.marginal_contributions) == 5

        # momentum_nq_01 strategies should be flagged as same instrument
        mt_pairs = [
            p for p in report.strategy_pairs
            if "momentum_nq_01" in p.strategy_a and "momentum_nq_01" in p.strategy_b
        ]
        assert len(mt_pairs) == 1
        assert mt_pairs[0].same_instrument is True

    def test_proportion_optimizer_with_real_data(self):
        """StrategyProportionOptimizer produces intra-bot recommendations."""
        dailies = _build_seven_day_dailies()
        builder = WeeklyMetricsBuilder("2026-02-24", "2026-03-02", _BOTS)
        portfolio = builder.build_portfolio_summary(dailies)

        per_strat = {
            bid: s.per_strategy_summary
            for bid, s in portfolio.bot_summaries.items()
        }
        optimizer = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        report = optimizer.compute(per_strat)

        # 3 bots
        assert len(report.bot_reports) == 3

        # momentum_nq_01 has NQ concentration warning
        mt_report = next(r for r in report.bot_reports if r.bot_id == "momentum_nq_01")
        assert any("NQ concentration" in n for n in mt_report.special_notes)

        # k_stock_trader has single strategy → 100% allocation
        kst_report = next(r for r in report.bot_reports if r.bot_id == "k_stock_trader")
        assert len(kst_report.recommendations) == 1
        assert kst_report.recommendations[0].suggested_unit_risk_pct == 1.0

    def test_full_pipeline_to_prompt_assembler(self, tmp_path: Path):
        """Full pipeline: daily data → weekly build → allocation → prompt package."""
        dailies = _build_seven_day_dailies()
        builder = WeeklyMetricsBuilder("2026-02-24", "2026-03-02", _BOTS)
        portfolio = builder.build_portfolio_summary(dailies)

        # Write curated data
        curated_dir = tmp_path / "curated"
        builder.write_weekly_curated(portfolio, curated_dir)

        # Run all allocation analyses
        allocator = PortfolioAllocator("2026-02-24", "2026-03-02")
        current = {bid: 100.0 / 3 for bid in _BOTS}
        alloc_report = allocator.compute(portfolio.bot_summaries, current)

        per_strat = {
            bid: s.per_strategy_summary
            for bid, s in portfolio.bot_summaries.items()
        }
        synergy = SynergyAnalyzer("2026-02-24", "2026-03-02")
        synergy_report = synergy.compute(per_strat)

        optimizer = StrategyProportionOptimizer("2026-02-24", "2026-03-02")
        proportion_report = optimizer.compute(per_strat)

        # Write allocation analysis to curated dir
        weekly_dir = curated_dir / "weekly" / "2026-02-24"
        alloc_data = {
            "portfolio_allocation": alloc_report.model_dump(mode="json"),
            "synergy_analysis": synergy_report.model_dump(mode="json"),
            "proportion_optimization": proportion_report.model_dump(mode="json"),
        }
        (weekly_dir / "allocation_analysis.json").write_text(
            json.dumps(alloc_data, indent=2, default=str),
        )

        # Create memory dir structure for assembler
        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test")

        # Assemble prompt
        from trading_assistant.analysis.weekly_prompt_assembler import WeeklyPromptAssembler

        assembler = WeeklyPromptAssembler(
            week_start="2026-02-24",
            week_end="2026-03-02",
            bots=_BOTS,
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=tmp_path / "runs",
        )
        package = assembler.assemble()

        # Verify allocation data is in the prompt package
        assert "allocation_analysis" in package.data
        assert "portfolio_allocation" in package.data["allocation_analysis"]
        assert "synergy_analysis" in package.data["allocation_analysis"]
        assert "proportion_optimization" in package.data["allocation_analysis"]
        assert "PORTFOLIO IMPROVEMENT ASSESSMENT" in package.instructions

    def test_all_results_json_serializable(self):
        """All analysis results can be serialized to JSON for curated output."""
        dailies = _build_seven_day_dailies()
        builder = WeeklyMetricsBuilder("2026-02-24", "2026-03-02", _BOTS)
        portfolio = builder.build_portfolio_summary(dailies)

        per_strat = {
            bid: s.per_strategy_summary
            for bid, s in portfolio.bot_summaries.items()
        }

        allocator = PortfolioAllocator("2026-02-24", "2026-03-02")
        alloc = allocator.compute(
            portfolio.bot_summaries, {b: 33.3 for b in _BOTS},
        )
        synergy = SynergyAnalyzer("2026-02-24", "2026-03-02").compute(per_strat)
        proportion = StrategyProportionOptimizer("2026-02-24", "2026-03-02").compute(per_strat)

        # All must serialize without error
        alloc_json = json.dumps(alloc.model_dump(mode="json"), default=str)
        synergy_json = json.dumps(synergy.model_dump(mode="json"), default=str)
        prop_json = json.dumps(proportion.model_dump(mode="json"), default=str)

        assert len(alloc_json) > 100
        assert len(synergy_json) > 100
        assert len(prop_json) > 100
