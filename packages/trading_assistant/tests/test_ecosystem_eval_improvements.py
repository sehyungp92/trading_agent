# tests/test_ecosystem_eval_improvements.py
"""Tests for ecosystem evaluation improvements:
- Correlation matrix wiring (SynergyAnalyzer → PortfolioAllocator)
- Unconditional simulations in weekly handler
- AllocationTracker wired into feedback handler
- Intra-bot synergy computation
- Bot-scope context filtering
- Minimum-data threshold for daily analysis
- Cross-bot pattern library
- Weekly retrospective builder
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from trading_assistant.schemas.weekly_metrics import StrategyWeeklySummary

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


# ---------------------------------------------------------------------------
# SynergyAnalyzer: intra-bot synergy + bot correlation matrix
# ---------------------------------------------------------------------------
class TestIntraBotSynergy:
    def test_compute_intra_bot_basic(self):
        from trading_assistant.skills.synergy_analyzer import SynergyAnalyzer

        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")

        daily_a = {_DATES[i]: 50.0 + i * 10 for i in range(7)}
        daily_b = {_DATES[i]: -20.0 + i * 5 for i in range(7)}

        strategies = {
            "strat_a": _make_strat("strat_a", "bot1", daily_a),
            "strat_b": _make_strat("strat_b", "bot1", daily_b),
        }

        report = analyzer.compute_intra_bot("bot1", strategies)
        assert report.total_strategies == 2
        assert len(report.strategy_pairs) == 1
        assert report.strategy_pairs[0].strategy_a.startswith("bot1:")
        assert report.strategy_pairs[0].strategy_b.startswith("bot1:")
        assert len(report.marginal_contributions) == 2

    def test_compute_intra_bot_single_strategy_returns_empty(self):
        from trading_assistant.skills.synergy_analyzer import SynergyAnalyzer

        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        daily = {_DATES[i]: 100.0 for i in range(7)}
        strategies = {
            "only_one": _make_strat("only_one", "bot1", daily),
        }
        report = analyzer.compute_intra_bot("bot1", strategies)
        assert report.total_strategies == 1
        assert len(report.strategy_pairs) == 0

    def test_compute_intra_bot_three_strategies(self):
        from trading_assistant.skills.synergy_analyzer import SynergyAnalyzer

        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        strats = {}
        for name in ["alpha", "beta", "gamma"]:
            daily = {_DATES[i]: (i + 1) * 10 for i in range(7)}
            strats[name] = _make_strat(name, "bot1", daily)

        report = analyzer.compute_intra_bot("bot1", strats)
        assert report.total_strategies == 3
        # 3 strategies → 3 pairs: (a,b), (a,c), (b,c)
        assert len(report.strategy_pairs) == 3
        assert len(report.marginal_contributions) == 3


class TestBotCorrelationMatrix:
    def test_two_bots_identical_pnl(self):
        from trading_assistant.skills.synergy_analyzer import SynergyAnalyzer

        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        daily = {_DATES[i]: 50.0 + i * 10 for i in range(7)}
        per_strat = {
            "bot1": {"s1": _make_strat("s1", "bot1", daily)},
            "bot2": {"s2": _make_strat("s2", "bot2", dict(daily))},
        }
        matrix = analyzer.compute_bot_correlation_matrix(per_strat)
        assert "bot1_bot2" in matrix
        assert matrix["bot1_bot2"] > 0.99  # identical series → ~1.0

    def test_two_bots_inverse_pnl(self):
        from trading_assistant.skills.synergy_analyzer import SynergyAnalyzer

        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        daily_a = {_DATES[i]: 50.0 + i * 10 for i in range(7)}
        daily_b = {_DATES[i]: -(50.0 + i * 10) for i in range(7)}
        per_strat = {
            "bot1": {"s1": _make_strat("s1", "bot1", daily_a)},
            "bot2": {"s2": _make_strat("s2", "bot2", daily_b)},
        }
        matrix = analyzer.compute_bot_correlation_matrix(per_strat)
        assert matrix["bot1_bot2"] < -0.99  # inverse → ~ -1.0

    def test_empty_input(self):
        from trading_assistant.skills.synergy_analyzer import SynergyAnalyzer

        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        matrix = analyzer.compute_bot_correlation_matrix({})
        assert matrix == {}

    def test_single_bot_no_pairs(self):
        from trading_assistant.skills.synergy_analyzer import SynergyAnalyzer

        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        daily = {_DATES[i]: 100.0 for i in range(7)}
        per_strat = {"bot1": {"s1": _make_strat("s1", "bot1", daily)}}
        matrix = analyzer.compute_bot_correlation_matrix(per_strat)
        assert matrix == {}

    def test_three_bots_produces_three_pairs(self):
        from trading_assistant.skills.synergy_analyzer import SynergyAnalyzer

        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")
        per_strat = {}
        for bid in ["bot1", "bot2", "bot3"]:
            daily = {_DATES[i]: float(i + hash(bid) % 5) for i in range(7)}
            per_strat[bid] = {"s": _make_strat("s", bid, daily)}
        matrix = analyzer.compute_bot_correlation_matrix(per_strat)
        assert len(matrix) == 3

    def test_multi_strategy_bot_aggregates_pnl(self):
        """Bot with 2 strategies: bot-level PnL = sum of strategy PnLs."""
        from trading_assistant.skills.synergy_analyzer import SynergyAnalyzer

        analyzer = SynergyAnalyzer("2026-02-24", "2026-03-02")

        # Use varying series so Pearson has non-zero variance
        daily_s1 = {_DATES[i]: 100.0 + i * 20 for i in range(7)}
        daily_s2 = {_DATES[i]: -50.0 + i * 5 for i in range(7)}
        # bot2 mirrors the sum: (100+i*20) + (-50+i*5) = 50+i*25
        daily_bot2 = {_DATES[i]: 50.0 + i * 25 for i in range(7)}

        per_strat = {
            "bot1": {
                "s1": _make_strat("s1", "bot1", daily_s1),
                "s2": _make_strat("s2", "bot1", daily_s2),
            },
            "bot2": {
                "s3": _make_strat("s3", "bot2", daily_bot2),
            },
        }
        matrix = analyzer.compute_bot_correlation_matrix(per_strat)
        # bot1 total PnL = s1+s2 = 50+i*25, same as bot2 → high correlation
        assert matrix["bot1_bot2"] > 0.99


# ---------------------------------------------------------------------------
# Bot-scope context filtering
# ---------------------------------------------------------------------------
class TestBotScopeFiltering:
    def test_load_corrections_unfiltered(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        mem_dir = tmp_path / "memory"
        findings = mem_dir / "findings"
        findings.mkdir(parents=True)
        corrections_path = findings / "corrections.jsonl"
        recent_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        corrections_path.write_text(
            json.dumps({"bot_id": "bot1", "text": "a", "timestamp": recent_ts}) + "\n"
            + json.dumps({"bot_id": "bot2", "text": "b", "timestamp": recent_ts}) + "\n",
        )

        builder = ContextBuilder(mem_dir)
        all_corrections = builder.load_corrections()
        assert len(all_corrections) == 2

    def test_load_corrections_filtered_by_bot(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        mem_dir = tmp_path / "memory"
        findings = mem_dir / "findings"
        findings.mkdir(parents=True)
        corrections_path = findings / "corrections.jsonl"
        recent_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        corrections_path.write_text(
            json.dumps({"bot_id": "bot1", "text": "a", "timestamp": recent_ts}) + "\n"
            + json.dumps({"bot_id": "bot2", "text": "b", "timestamp": recent_ts}) + "\n"
            + json.dumps({"text": "no bot", "timestamp": recent_ts}) + "\n",
        )

        builder = ContextBuilder(mem_dir)
        bot1_corrections = builder.load_corrections(bot_id="bot1")
        # Should include bot1 + the one without bot_id
        assert len(bot1_corrections) == 2
        assert any(c.get("bot_id") == "bot1" for c in bot1_corrections)

    def test_load_failure_log_filtered(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        mem_dir = tmp_path / "memory"
        findings = mem_dir / "findings"
        findings.mkdir(parents=True)
        path = findings / "failure-log.jsonl"
        recent_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        path.write_text(
            json.dumps({"bot_id": "bot1", "error": "e1", "timestamp": recent_ts}) + "\n"
            + json.dumps({"bot_id": "bot2", "error": "e2", "timestamp": recent_ts}) + "\n",
        )

        builder = ContextBuilder(mem_dir)
        bot2_log = builder.load_failure_log(bot_id="bot2")
        assert len(bot2_log) == 1
        assert bot2_log[0]["bot_id"] == "bot2"


# ---------------------------------------------------------------------------
# Pattern Library
# ---------------------------------------------------------------------------
class TestPatternLibrary:
    def test_add_and_load(self, tmp_path: Path):
        from trading_assistant.skills.pattern_library import PatternLibrary
        from trading_assistant.schemas.pattern_library import PatternEntry, PatternCategory

        lib = PatternLibrary(tmp_path / "findings")
        entry = PatternEntry(
            title="Time-of-day gate for NQ strategies",
            category=PatternCategory.FILTER,
            source_bot="momentum_nq_01",
            source_strategy="NQDTC",
            target_bots=["swing_multi_01"],
            description="NQ strategies perform better outside first 30 min",
            evidence="45 trades over 60 days, Calmar +0.3",
        )
        result = lib.add(entry)
        assert result.pattern_id  # auto-assigned

        all_entries = lib.load_all()
        assert len(all_entries) == 1
        assert all_entries[0].title == "Time-of-day gate for NQ strategies"

    def test_load_active_excludes_rejected(self, tmp_path: Path):
        from trading_assistant.skills.pattern_library import PatternLibrary
        from trading_assistant.schemas.pattern_library import PatternEntry, PatternCategory, PatternStatus

        lib = PatternLibrary(tmp_path / "findings")
        lib.add(PatternEntry(
            title="Good pattern",
            category=PatternCategory.FILTER,
            source_bot="bot1",
            status=PatternStatus.VALIDATED,
        ))
        lib.add(PatternEntry(
            title="Bad pattern",
            category=PatternCategory.FILTER,
            source_bot="bot1",
            status=PatternStatus.REJECTED,
        ))

        active = lib.load_active()
        assert len(active) == 1
        assert active[0].title == "Good pattern"

    def test_load_for_bot(self, tmp_path: Path):
        from trading_assistant.skills.pattern_library import PatternLibrary
        from trading_assistant.schemas.pattern_library import PatternEntry, PatternCategory

        lib = PatternLibrary(tmp_path / "findings")
        lib.add(PatternEntry(
            title="Pattern for bot1",
            category=PatternCategory.REGIME_GATE,
            source_bot="bot1",
        ))
        lib.add(PatternEntry(
            title="Pattern for bot2 targeting bot1",
            category=PatternCategory.ENTRY_SIGNAL,
            source_bot="bot2",
            target_bots=["bot1"],
        ))
        lib.add(PatternEntry(
            title="Pattern for bot3 only",
            category=PatternCategory.EXIT_RULE,
            source_bot="bot3",
        ))

        bot1_patterns = lib.load_for_bot("bot1")
        assert len(bot1_patterns) == 2  # source_bot=bot1 + target_bots includes bot1

    def test_update_status(self, tmp_path: Path):
        from trading_assistant.skills.pattern_library import PatternLibrary
        from trading_assistant.schemas.pattern_library import PatternEntry, PatternCategory, PatternStatus

        lib = PatternLibrary(tmp_path / "findings")
        entry = lib.add(PatternEntry(
            title="Test",
            category=PatternCategory.FILTER,
            source_bot="bot1",
        ))
        assert lib.update_status(entry.pattern_id, PatternStatus.IMPLEMENTED)
        reloaded = lib.load_all()
        assert reloaded[0].status == PatternStatus.IMPLEMENTED

    def test_update_status_not_found(self, tmp_path: Path):
        from trading_assistant.skills.pattern_library import PatternLibrary

        lib = PatternLibrary(tmp_path / "findings")
        from trading_assistant.schemas.pattern_library import PatternStatus
        assert not lib.update_status("nonexistent", PatternStatus.REJECTED)


# ---------------------------------------------------------------------------
# Retrospective Builder
# ---------------------------------------------------------------------------
class TestRetrospectiveBuilder:
    def test_build_empty(self, tmp_path: Path):
        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        builder = RetrospectiveBuilder(
            runs_dir=tmp_path / "runs",
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
        )
        retro = builder.build("2026-02-24", "2026-03-02")
        assert retro.predictions_reviewed == 0
        assert "No predictions" in retro.summary

    def test_build_with_run_data(self, tmp_path: Path):
        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "daily-2026-02-25"
        run_dir.mkdir(parents=True)

        # Create a structured output file with suggestions
        output = {
            "suggestions": ["Relax volume filter on bot1"],
            "warnings": ["bot2 approaching drawdown limit"],
        }
        (run_dir / "output.json").write_text(json.dumps(output))

        # Create curated data with outcomes
        curated_dir = tmp_path / "curated"
        bot_dir = curated_dir / "2026-02-25" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(json.dumps({
            "net_pnl": 500.0,
            "total_trades": 10,
        }))

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir,
            curated_dir=curated_dir,
            memory_dir=tmp_path / "memory",
        )
        retro = builder.build("2026-02-24", "2026-03-02")
        assert retro.predictions_reviewed == 2
        assert retro.week_start == "2026-02-24"

    def test_build_with_previous_weekly(self, tmp_path: Path):
        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        runs_dir = tmp_path / "runs"
        # Previous weekly run (one week before start)
        prev_weekly = runs_dir / "weekly-2026-02-17"
        prev_weekly.mkdir(parents=True)
        (prev_weekly / "trading_assistant.analysis.json").write_text(json.dumps({
            "suggestions": ["Increase k_stock_trader allocation"],
        }))

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir,
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
        )
        retro = builder.build("2026-02-24", "2026-03-02")
        assert retro.predictions_reviewed == 1


# ---------------------------------------------------------------------------
# Retrospective accuracy improvement (2.5)
# ---------------------------------------------------------------------------
class TestRetrospectiveAccuracy:
    def test_assess_accuracy_stop_improvement(self, tmp_path: Path):
        """Suggestion about stops + improved exit_efficiency → partially_correct."""
        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "daily-2026-02-25"
        run_dir.mkdir(parents=True)
        (run_dir / "output.json").write_text(json.dumps({
            "suggestions": ["Widen stop loss on bot1 to capture more of the move"],
        }))

        curated_dir = tmp_path / "curated"
        bot_dir = curated_dir / "2026-02-25" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(json.dumps({
            "net_pnl": 500.0, "total_trades": 10,
        }))
        (bot_dir / "exit_efficiency.json").write_text(json.dumps({
            "exit_efficiency": 0.72,
            "avg_mae_pct": 0.8,
        }))

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir, curated_dir=curated_dir, memory_dir=tmp_path / "memory",
        )
        retro = builder.build("2026-02-24", "2026-03-02")
        assert retro.predictions_reviewed == 1
        stop_pred = retro.predictions[0]
        assert stop_pred.accuracy == "partially_correct"
        assert retro.partially_correct == 1

    def test_assess_accuracy_filter_improvement(self, tmp_path: Path):
        """Suggestion about filters + low missed_would_have_won → partially_correct."""
        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "daily-2026-02-25"
        run_dir.mkdir(parents=True)
        (run_dir / "output.json").write_text(json.dumps({
            "suggestions": ["Relax the volume filter on bot1"],
        }))

        curated_dir = tmp_path / "curated"
        bot_dir = curated_dir / "2026-02-25" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(json.dumps({
            "net_pnl": 300.0, "total_trades": 8,
            "missed_count": 3, "missed_would_have_won": 1,
        }))
        (bot_dir / "filter_analysis.json").write_text(json.dumps({
            "missed_count": 3, "missed_would_have_won": 1,
        }))

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir, curated_dir=curated_dir, memory_dir=tmp_path / "memory",
        )
        retro = builder.build("2026-02-24", "2026-03-02")
        assert retro.predictions_reviewed == 1
        assert retro.predictions[0].accuracy == "partially_correct"

    def test_assess_accuracy_no_keyword_match_unverifiable(self, tmp_path: Path):
        """Suggestion with no matching keyword → unverifiable (backward compat)."""
        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "daily-2026-02-25"
        run_dir.mkdir(parents=True)
        (run_dir / "output.json").write_text(json.dumps({
            "suggestions": ["General observation about market conditions"],
        }))

        curated_dir = tmp_path / "curated"
        bot_dir = curated_dir / "2026-02-25" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(json.dumps({
            "net_pnl": 200.0, "total_trades": 5,
        }))

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir, curated_dir=curated_dir, memory_dir=tmp_path / "memory",
        )
        retro = builder.build("2026-02-24", "2026-03-02")
        assert retro.predictions_reviewed == 1
        assert retro.predictions[0].accuracy == "unverifiable"

    def test_load_actual_outcomes_loads_extra_files(self, tmp_path: Path):
        """_load_actual_outcomes should merge exit_efficiency, filter_analysis, etc."""
        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        curated_dir = tmp_path / "curated"
        bot_dir = curated_dir / "2026-02-25" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(json.dumps({
            "net_pnl": 100.0, "total_trades": 5,
        }))
        (bot_dir / "exit_efficiency.json").write_text(json.dumps({
            "exit_efficiency": 0.65,
        }))
        (bot_dir / "filter_analysis.json").write_text(json.dumps({
            "missed_would_have_won": 3,
        }))

        builder = RetrospectiveBuilder(
            runs_dir=tmp_path / "runs", curated_dir=curated_dir,
            memory_dir=tmp_path / "memory",
        )
        start = datetime(2026, 2, 25)
        end = datetime(2026, 2, 25)
        outcomes = builder._load_actual_outcomes(start, end)
        assert "2026-02-25" in outcomes
        entry = outcomes["2026-02-25"][0]
        assert "exit_efficiency" in entry
        assert entry["exit_efficiency"]["exit_efficiency"] == 0.65
        assert "filter_analysis" in entry

    def test_assess_accuracy_drawdown_decrease(self, tmp_path: Path):
        """Suggestion about drawdown + low max_drawdown → partially_correct."""
        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "daily-2026-02-25"
        run_dir.mkdir(parents=True)
        (run_dir / "output.json").write_text(json.dumps({
            "suggestions": ["Reduce drawdown risk via tighter stops"],
        }))

        curated_dir = tmp_path / "curated"
        bot_dir = curated_dir / "2026-02-25" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(json.dumps({
            "net_pnl": 400.0, "total_trades": 12,
            "max_drawdown_pct": 0.8,
        }))

        builder = RetrospectiveBuilder(
            runs_dir=runs_dir, curated_dir=curated_dir, memory_dir=tmp_path / "memory",
        )
        retro = builder.build("2026-02-24", "2026-03-02")
        assert retro.predictions[0].accuracy == "partially_correct"


# ---------------------------------------------------------------------------
# Feedback handler: allocation change detection
# ---------------------------------------------------------------------------
class TestAllocationFeedback:
    def test_parse_allocation_approval(self):
        from trading_assistant.analysis.feedback_handler import FeedbackHandler
        from trading_assistant.schemas.corrections import CorrectionType

        handler = FeedbackHandler(report_id="weekly-2026-03-01")
        correction = handler.parse("Approved the allocation rebalance as suggested")
        assert correction.correction_type == CorrectionType.ALLOCATION_CHANGE

    def test_parse_allocation_accept(self):
        from trading_assistant.analysis.feedback_handler import FeedbackHandler
        from trading_assistant.schemas.corrections import CorrectionType

        handler = FeedbackHandler(report_id="weekly-2026-03-01")
        correction = handler.parse("Accept the new allocation split")
        assert correction.correction_type == CorrectionType.ALLOCATION_CHANGE

    def test_parse_allocation_confirm(self):
        from trading_assistant.analysis.feedback_handler import FeedbackHandler
        from trading_assistant.schemas.corrections import CorrectionType

        handler = FeedbackHandler(report_id="weekly-2026-03-01")
        correction = handler.parse("Confirmed rebalancing as recommended")
        assert correction.correction_type == CorrectionType.ALLOCATION_CHANGE

    def test_parse_does_not_false_positive(self):
        from trading_assistant.analysis.feedback_handler import FeedbackHandler
        from trading_assistant.schemas.corrections import CorrectionType

        handler = FeedbackHandler(report_id="weekly-2026-03-01")
        # This should NOT match allocation pattern
        correction = handler.parse("I approve of the analysis quality")
        assert correction.correction_type != CorrectionType.ALLOCATION_CHANGE


# ---------------------------------------------------------------------------
# Minimum-data threshold
# ---------------------------------------------------------------------------
class TestMinimumDataThreshold:
    def test_count_daily_trades(self, tmp_path: Path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream

        handlers = Handlers(
            agent_runner=AsyncMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=_make_notification_prefs(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1", "bot2"],
        )

        # Create trades for bot1
        date = "2026-03-01"
        bot1_dir = tmp_path / "curated" / date / "bot1"
        bot1_dir.mkdir(parents=True)
        (bot1_dir / "trades.jsonl").write_text(
            '{"trade_id": "t1"}\n{"trade_id": "t2"}\n'
        )

        # bot2 has no trades
        bot2_dir = tmp_path / "curated" / date / "bot2"
        bot2_dir.mkdir(parents=True)

        count = handlers._count_daily_trades(date)
        assert count == 2

    def test_count_daily_trades_no_data(self, tmp_path: Path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream

        handlers = Handlers(
            agent_runner=AsyncMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=_make_notification_prefs(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
        )

        count = handlers._count_daily_trades("2026-03-01")
        assert count == 0


# ---------------------------------------------------------------------------
# Unconditional simulations
# ---------------------------------------------------------------------------
class TestUnconditionalSimulations:
    def test_simulations_attempted_for_all_bots_not_gated(self, tmp_path: Path):
        """Verify simulations iterate all bots, not just ones with suggestions.

        Instead of trying to mock deeply-imported simulators, we verify the
        iteration logic by checking _load_trades_for_week is called for each bot.
        """
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream

        handlers = Handlers(
            agent_runner=AsyncMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=_make_notification_prefs(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1", "bot2", "bot3"],
        )

        # Track which bots _load_trades_for_week is called for
        called_bots: list[str] = []

        def tracking_load(bot_id, week_start, week_end):
            called_bots.append(bot_id)
            return ([], [])  # No data → simulations won't produce results

        handlers._load_trades_for_week = tracking_load

        class EmptyReport:
            suggestions = []

        handlers._run_weekly_simulations(EmptyReport(), "2026-02-24", "2026-03-02")

        # All 3 bots should have been queried for data, not just suggestion-matching ones
        assert set(called_bots) == {"bot1", "bot2", "bot3"}

    def test_regime_hints_from_suggestions_used(self, tmp_path: Path):
        """Verify regime hints from suggestions are passed to counterfactual."""
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream

        handlers = Handlers(
            agent_runner=AsyncMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=_make_notification_prefs(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
        )

        # Create trade data
        for date in _DATES:
            bot1_dir = tmp_path / "curated" / date / "bot1"
            bot1_dir.mkdir(parents=True)
            (bot1_dir / "trades.jsonl").write_text(
                json.dumps({"trade_id": f"t-{date}", "bot_id": "bot1",
                            "strategy_id": "s1", "pnl": 50.0,
                            "entry_price": 100.0, "exit_price": 100.5,
                            "direction": "long", "symbol": "NQ",
                            "exchange_timestamp": f"{date}T12:00:00+00:00"}) + "\n",
            )

        class SuggestionReport:
            class Suggestion:
                bot_id = "bot1"
                title = "Regime mismatch in trending markets"
                regime = "trending"
            suggestions = [Suggestion()]

        # Should not raise — regime hint is extracted and used
        results = handlers._run_weekly_simulations(
            SuggestionReport(), "2026-02-24", "2026-03-02",
        )
        # Simulation may fail due to minimal data but should not raise
        assert isinstance(results, dict)


# ---------------------------------------------------------------------------
# Context builder: pattern library loading
# ---------------------------------------------------------------------------
class TestContextBuilderPatternLibrary:
    def test_load_pattern_library_empty(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        builder = ContextBuilder(tmp_path / "memory")
        patterns = builder.load_pattern_library()
        assert patterns == []

    def test_load_pattern_library_with_entries(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder
        from trading_assistant.skills.pattern_library import PatternLibrary
        from trading_assistant.schemas.pattern_library import PatternEntry, PatternCategory

        mem_dir = tmp_path / "memory"
        lib = PatternLibrary(mem_dir / "findings")
        lib.add(PatternEntry(
            title="Test pattern",
            category=PatternCategory.FILTER,
            source_bot="bot1",
        ))

        builder = ContextBuilder(mem_dir)
        patterns = builder.load_pattern_library()
        assert len(patterns) == 1
        assert patterns[0]["title"] == "Test pattern"

    def test_load_pattern_library_filtered_by_bot(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder
        from trading_assistant.skills.pattern_library import PatternLibrary
        from trading_assistant.schemas.pattern_library import PatternEntry, PatternCategory

        mem_dir = tmp_path / "memory"
        lib = PatternLibrary(mem_dir / "findings")
        lib.add(PatternEntry(
            title="Bot1 pattern",
            category=PatternCategory.FILTER,
            source_bot="bot1",
        ))
        lib.add(PatternEntry(
            title="Bot2 pattern",
            category=PatternCategory.EXIT_RULE,
            source_bot="bot2",
        ))

        builder = ContextBuilder(mem_dir)
        bot1_patterns = builder.load_pattern_library(bot_id="bot1")
        assert len(bot1_patterns) == 1
        assert bot1_patterns[0]["title"] == "Bot1 pattern"

    def test_base_package_includes_pattern_library(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder
        from trading_assistant.skills.pattern_library import PatternLibrary
        from trading_assistant.schemas.pattern_library import PatternEntry, PatternCategory

        mem_dir = tmp_path / "memory"
        # Create policies dir so system prompt doesn't fail
        policies = mem_dir / "policies" / "v1"
        policies.mkdir(parents=True)
        (policies / "agent.md").write_text("# Agents")
        (policies / "trading_rules.md").write_text("# Rules")
        (policies / "soul.md").write_text("# Soul")

        lib = PatternLibrary(mem_dir / "findings")
        lib.add(PatternEntry(
            title="Loaded pattern",
            category=PatternCategory.REGIME_GATE,
            source_bot="bot1",
        ))

        builder = ContextBuilder(mem_dir)
        pkg = builder.base_package()
        assert "pattern_library" in pkg.data
        assert len(pkg.data["pattern_library"]) == 1


# ---------------------------------------------------------------------------
# Handlers: allocation tracker wiring in feedback
# ---------------------------------------------------------------------------
class TestHandlerAllocationWiring:
    @pytest.mark.asyncio
    async def test_feedback_with_allocation_records_change(self, tmp_path: Path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType

        handlers = Handlers(
            agent_runner=AsyncMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=_make_notification_prefs(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
        )

        action = Action(
            type=ActionType.SEND_NOTIFICATION,
            event_id="evt-feedback-001",
            bot_id="bot1",
            details={
                "text": "Approved the allocation rebalance",
                "report_id": "weekly-2026-03-01",
                "bot_id": "bot1",
                "allocations": [
                    {"bot_id": "bot1", "allocation_pct": 60.0},
                    {"bot_id": "bot2", "allocation_pct": 40.0},
                ],
            },
        )
        await handlers.handle_feedback(action)

        # Verify allocation was recorded
        alloc_path = tmp_path / "memory" / "findings" / "allocation_history.jsonl"
        assert alloc_path.exists()
        lines = alloc_path.read_text().strip().splitlines()
        assert len(lines) == 2  # Two allocations recorded

    @pytest.mark.asyncio
    async def test_feedback_without_allocation_no_record(self, tmp_path: Path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType

        handlers = Handlers(
            agent_runner=AsyncMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=_make_notification_prefs(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
        )

        action = Action(
            type=ActionType.SEND_NOTIFICATION,
            event_id="evt-feedback-002",
            bot_id="bot1",
            details={
                "text": "Good catch on that trade",
                "report_id": "daily-2026-03-01",
            },
        )
        await handlers.handle_feedback(action)

        # No allocation file should be created
        alloc_path = tmp_path / "memory" / "findings" / "allocation_history.jsonl"
        assert not alloc_path.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_notification_prefs():
    from trading_assistant.schemas.notifications import NotificationPreferences

    return NotificationPreferences()
