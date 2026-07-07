# tests/test_weekly_allocation_wiring.py
"""Tests for weekly handler allocation wiring and prompt assembler integration."""
import json
from pathlib import Path
from unittest.mock import MagicMock


from trading_assistant.schemas.weekly_metrics import BotWeeklySummary, StrategyWeeklySummary, WeeklySummary


_DATES = [
    "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27",
    "2026-02-28", "2026-03-01", "2026-03-02",
]


def _make_portfolio_summary() -> WeeklySummary:
    """Create a portfolio summary with per-strategy data for testing."""
    strat_a = StrategyWeeklySummary(
        strategy_id="strat_a", bot_id="bot1",
        total_trades=20, net_pnl=500.0,
        daily_pnl={d: 500.0 / 7 + i for i, d in enumerate(_DATES)},
    )
    strat_b = StrategyWeeklySummary(
        strategy_id="strat_b", bot_id="bot1",
        total_trades=15, net_pnl=300.0,
        daily_pnl={d: 300.0 / 7 + i * 2 for i, d in enumerate(_DATES)},
    )
    strat_c = StrategyWeeklySummary(
        strategy_id="strat_c", bot_id="bot2",
        total_trades=10, net_pnl=200.0,
        daily_pnl={d: 200.0 / 7 + i for i, d in enumerate(_DATES)},
    )
    bot1 = BotWeeklySummary(
        week_start="2026-02-24", week_end="2026-03-02", bot_id="bot1",
        net_pnl=800.0, max_drawdown_pct=4.0,
        daily_pnl={d: 800.0 / 7 + i for i, d in enumerate(_DATES)},
        per_strategy_summary={"strat_a": strat_a, "strat_b": strat_b},
    )
    bot2 = BotWeeklySummary(
        week_start="2026-02-24", week_end="2026-03-02", bot_id="bot2",
        net_pnl=200.0, max_drawdown_pct=6.0,
        daily_pnl={d: 200.0 / 7 + i for i, d in enumerate(_DATES)},
        per_strategy_summary={"strat_c": strat_c},
    )
    return WeeklySummary(
        week_start="2026-02-24", week_end="2026-03-02",
        bot_summaries={"bot1": bot1, "bot2": bot2},
        total_net_pnl=1000.0,
    )


class TestRunAllocationAnalyses:
    def test_returns_all_three_analyses(self, tmp_path: Path):
        """Handler's _run_allocation_analyses returns all 3 analysis types."""
        from trading_assistant.orchestrator.handlers import Handlers

        handlers = Handlers(
            agent_runner=MagicMock(),
            event_stream=MagicMock(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1", "bot2"],
        )

        summary = _make_portfolio_summary()
        results = handlers._run_allocation_analyses(
            summary, "2026-02-24", "2026-03-02",
        )

        assert "portfolio_allocation" in results
        assert "synergy_analysis" in results
        assert "proportion_optimization" in results

    def test_portfolio_allocation_has_recommendations(self, tmp_path: Path):
        from trading_assistant.orchestrator.handlers import Handlers

        handlers = Handlers(
            agent_runner=MagicMock(),
            event_stream=MagicMock(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=["bot1", "bot2"],
        )

        results = handlers._run_allocation_analyses(
            _make_portfolio_summary(), "2026-02-24", "2026-03-02",
        )
        alloc = results["portfolio_allocation"]
        assert len(alloc["recommendations"]) == 2

    def test_synergy_analysis_has_pairs(self, tmp_path: Path):
        from trading_assistant.orchestrator.handlers import Handlers

        handlers = Handlers(
            agent_runner=MagicMock(),
            event_stream=MagicMock(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=["bot1", "bot2"],
        )

        results = handlers._run_allocation_analyses(
            _make_portfolio_summary(), "2026-02-24", "2026-03-02",
        )
        synergy = results["synergy_analysis"]
        # 3 strategies → 3 pairs
        assert synergy["total_strategies"] == 3

    def test_proportion_optimization_has_bot_reports(self, tmp_path: Path):
        from trading_assistant.orchestrator.handlers import Handlers

        handlers = Handlers(
            agent_runner=MagicMock(),
            event_stream=MagicMock(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=["bot1", "bot2"],
        )

        results = handlers._run_allocation_analyses(
            _make_portfolio_summary(), "2026-02-24", "2026-03-02",
        )
        prop = results["proportion_optimization"]
        assert len(prop["bot_reports"]) == 2

    def test_empty_bot_summaries_returns_empty(self, tmp_path: Path):
        from trading_assistant.orchestrator.handlers import Handlers

        handlers = Handlers(
            agent_runner=MagicMock(),
            event_stream=MagicMock(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
        )

        empty_summary = WeeklySummary(week_start="2026-02-24", week_end="2026-03-02")
        results = handlers._run_allocation_analyses(
            empty_summary, "2026-02-24", "2026-03-02",
        )
        assert results == {}

    def test_graceful_failure(self, tmp_path: Path):
        """If allocation analyses fail, handler returns empty dict."""
        from trading_assistant.orchestrator.handlers import Handlers

        handlers = Handlers(
            agent_runner=MagicMock(),
            event_stream=MagicMock(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
        )

        # Pass a non-WeeklySummary object that lacks bot_summaries
        results = handlers._run_allocation_analyses(
            object(), "2026-02-24", "2026-03-02",
        )
        assert results == {}


class TestWeeklyPromptAssemblerAllocation:
    def test_loads_allocation_analysis_file(self, tmp_path: Path):
        """Assembler loads allocation_analysis.json from curated dir."""
        from trading_assistant.analysis.weekly_prompt_assembler import WeeklyPromptAssembler

        weekly_dir = tmp_path / "curated" / "weekly" / "2026-02-24"
        weekly_dir.mkdir(parents=True)

        alloc_data = {
            "portfolio_allocation": {"recommendations": []},
            "synergy_analysis": {"total_strategies": 3},
        }
        (weekly_dir / "allocation_analysis.json").write_text(json.dumps(alloc_data))

        # Also create required weekly_summary.json
        (weekly_dir / "weekly_summary.json").write_text(json.dumps({
            "week_start": "2026-02-24", "week_end": "2026-03-02",
        }))

        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test")

        assembler = WeeklyPromptAssembler(
            week_start="2026-02-24",
            week_end="2026-03-02",
            bots=["bot1"],
            curated_dir=tmp_path / "curated",
            memory_dir=memory_dir,
            runs_dir=tmp_path / "runs",
        )
        package = assembler.assemble()
        assert "allocation_analysis" in package.data

    def test_instruction_14_present(self, tmp_path: Path):
        """Assembler instructions include allocation assessment."""
        from trading_assistant.analysis.weekly_prompt_assembler import WeeklyPromptAssembler

        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test")

        assembler = WeeklyPromptAssembler(
            week_start="2026-02-24",
            week_end="2026-03-02",
            bots=["bot1"],
            curated_dir=tmp_path / "curated",
            memory_dir=memory_dir,
            runs_dir=tmp_path / "runs",
        )
        package = assembler.assemble()
        assert "PORTFOLIO IMPROVEMENT ASSESSMENT" in package.instructions

    def test_allocation_in_context_files_list(self, tmp_path: Path):
        """allocation_analysis.json shows up in context_files when present."""
        from trading_assistant.analysis.weekly_prompt_assembler import WeeklyPromptAssembler

        weekly_dir = tmp_path / "curated" / "weekly" / "2026-02-24"
        weekly_dir.mkdir(parents=True)
        (weekly_dir / "allocation_analysis.json").write_text("{}")

        memory_dir = tmp_path / "memory"
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test")

        assembler = WeeklyPromptAssembler(
            week_start="2026-02-24",
            week_end="2026-03-02",
            bots=["bot1"],
            curated_dir=tmp_path / "curated",
            memory_dir=memory_dir,
            runs_dir=tmp_path / "runs",
        )
        package = assembler.assemble()
        assert any("allocation_analysis.json" in f for f in package.context_files)
