# tests/test_weekly_pipeline_integration.py
"""End-to-end integration test for the weekly analysis pipeline.

Flow: daily curated data → weekly metrics builder → strategy engine →
      weekly prompt assembler → complete context package.
"""
import json
from pathlib import Path


from trading_assistant.schemas.daily_metrics import (
    BotDailySummary,
)
from trading_assistant.skills.build_weekly_metrics import WeeklyMetricsBuilder
from trading_assistant.analysis.strategy_engine import StrategyEngine
from trading_assistant.analysis.weekly_prompt_assembler import WeeklyPromptAssembler


_DATES = [
    "2026-02-23", "2026-02-24", "2026-02-25", "2026-02-26",
    "2026-02-27", "2026-02-28", "2026-03-01",
]


def _setup_daily_data(base_dir: Path, bots: list[str]) -> None:
    """Create 7 days of daily curated data for each bot."""
    for date in _DATES:
        for bot_id in bots:
            bot_dir = base_dir / date / bot_id
            bot_dir.mkdir(parents=True)
            summary = BotDailySummary(
                date=date,
                bot_id=bot_id,
                total_trades=10,
                win_count=6,
                loss_count=4,
                gross_pnl=80.0,
                net_pnl=70.0,
                avg_win=30.0,
                avg_loss=-15.0,
                avg_process_quality=75.0,
                missed_count=3,
                missed_would_have_won=1,
            )
            (bot_dir / "summary.json").write_text(
                json.dumps(summary.model_dump(mode="json"), indent=2)
            )
            # Minimal files for completeness
            for fname in [
                "winners.json", "losers.json", "process_failures.json",
                "notable_missed.json", "regime_analysis.json",
                "filter_analysis.json", "root_cause_summary.json",
            ]:
                (bot_dir / fname).write_text("[]")

        # Portfolio risk card per day
        (base_dir / date / "portfolio_risk_card.json").write_text(
            json.dumps({"date": date, "concentration_score": 30.0})
        )


def _setup_memory(memory_dir: Path) -> None:
    """Create memory policies and corrections."""
    policy_dir = memory_dir / "policies" / "v1"
    policy_dir.mkdir(parents=True)
    (policy_dir / "agent.md").write_text("You are the trading assistant.")
    (policy_dir / "trading_rules.md").write_text("Max drawdown 15%.")
    (policy_dir / "soul.md").write_text("Conservative.")
    findings_dir = memory_dir / "findings"
    findings_dir.mkdir(parents=True)
    (findings_dir / "corrections.jsonl").write_text("")


def _setup_runs(runs_dir: Path) -> None:
    """Create 7 days of daily report stubs."""
    for index, date in enumerate(_DATES):
        run_dir = runs_dir / f"daily-{date}-run{index}"
        run_dir.mkdir(parents=True)
        (run_dir / "daily_report.md").write_text(f"# Daily Report {date}\nAll good.")


class TestWeeklyPipelineIntegration:
    def test_full_weekly_pipeline(self, tmp_path: Path):
        """End-to-end: daily data → weekly summary → strategy suggestions → prompt package."""
        curated_dir = tmp_path / "curated"
        memory_dir = tmp_path / "memory"
        runs_dir = tmp_path / "runs"
        bots = ["bot1", "bot2"]

        _setup_daily_data(curated_dir, bots)
        _setup_memory(memory_dir)
        _setup_runs(runs_dir)

        # Step 1: Build weekly metrics
        builder = WeeklyMetricsBuilder(
            week_start="2026-02-23", week_end="2026-03-01", bots=bots
        )

        dailies_by_bot = {}
        for bot_id in bots:
            dailies = []
            for date in _DATES:
                summary_path = curated_dir / date / bot_id / "summary.json"
                data = json.loads(summary_path.read_text())
                dailies.append(BotDailySummary.model_validate(data))
            dailies_by_bot[bot_id] = dailies

        portfolio_summary = builder.build_portfolio_summary(dailies_by_bot)
        assert portfolio_summary.total_trades == 140  # 10 * 7 * 2 bots
        assert portfolio_summary.total_net_pnl == 980.0  # 70 * 7 * 2

        # Write weekly curated data
        weekly_dir = builder.write_weekly_curated(portfolio_summary, curated_dir)
        assert (weekly_dir / "weekly_summary.json").exists()

        # Step 2: Run strategy engine
        engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")
        report = engine.build_report(bot_summaries=portfolio_summary.bot_summaries)
        # Write refinement report
        (weekly_dir / "refinement_report.json").write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, default=str)
        )

        # Step 3: Assemble weekly prompt
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=bots,
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
        )
        package = assembler.assemble()

        # Verify complete package
        assert package.system_prompt
        assert package.task_prompt
        assert package.data
        assert "weekly_summary" in package.data
        assert "refinement_report" in package.data
        assert "daily_reports" in package.data
        assert len(package.data["daily_reports"]) == 7
        assert "portfolio_risk_cards" in package.data
        assert len(package.data["portfolio_risk_cards"]) == 7
        assert package.instructions
        assert "weekly" in package.task_prompt.lower()
