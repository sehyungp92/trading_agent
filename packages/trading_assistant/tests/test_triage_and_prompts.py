# tests/test_triage_and_prompts.py
"""Tests for Phase 2: Focused Prompt Architecture.

Covers:
- analysis/daily_triage.py — DailyTriage deterministic pre-processing
- analysis/weekly_triage.py — WeeklyTriage weekly pattern detection
- analysis/prompt_assembler.py — DailyPromptAssembler triage integration
- analysis/weekly_prompt_assembler.py — WeeklyPromptAssembler triage integration
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


from trading_assistant.analysis.daily_triage import DailyTriage, SignificantEvent, TriageReport
from trading_assistant.analysis.weekly_triage import (
    BotWeekSummary,
    WeeklyAnomaly,
    WeeklyTriage,
    WeeklyTriageReport,
)
from trading_assistant.analysis.prompt_assembler import DailyPromptAssembler
from trading_assistant.analysis.weekly_prompt_assembler import WeeklyPromptAssembler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_summary(
    curated_dir: Path,
    date: str,
    bot_id: str,
    net_pnl: float = 100.0,
    total_trades: int = 10,
    win_count: int = 6,
    max_drawdown_pct: float = 5.0,
) -> None:
    """Write a minimal summary.json for a bot on a given date."""
    bot_dir = curated_dir / date / bot_id
    bot_dir.mkdir(parents=True, exist_ok=True)
    (bot_dir / "summary.json").write_text(
        json.dumps({
            "net_pnl": net_pnl,
            "total_trades": total_trades,
            "win_count": win_count,
            "max_drawdown_pct": max_drawdown_pct,
        })
    )


def _write_regime(
    curated_dir: Path, date: str, bot_id: str, dominant_regime: str
) -> None:
    """Write regime_analysis.json for a bot on a given date."""
    bot_dir = curated_dir / date / bot_id
    bot_dir.mkdir(parents=True, exist_ok=True)
    (bot_dir / "regime_analysis.json").write_text(
        json.dumps({"dominant_regime": dominant_regime})
    )


def _write_filter_analysis(
    curated_dir: Path,
    date: str,
    bot_id: str,
    filters: list[dict],
) -> None:
    """Write filter_analysis.json for a bot on a given date."""
    bot_dir = curated_dir / date / bot_id
    bot_dir.mkdir(parents=True, exist_ok=True)
    (bot_dir / "filter_analysis.json").write_text(
        json.dumps({"filters": filters})
    )


def _setup_memory(tmp_path: Path) -> Path:
    """Create a minimal memory directory with policies and findings."""
    memory_dir = tmp_path / "memory"
    (memory_dir / "policies" / "v1").mkdir(parents=True)
    (memory_dir / "policies" / "v1" / "agent.md").write_text(
        "You are a trading assistant agent."
    )
    (memory_dir / "policies" / "v1" / "trading_rules.md").write_text(
        "Max drawdown 10%. Max position 5%."
    )
    (memory_dir / "policies" / "v1" / "soul.md").write_text(
        "Conservative risk management."
    )
    (memory_dir / "findings").mkdir(parents=True)
    return memory_dir


def _trailing_dates(date: str, days: int) -> list[str]:
    """Return a list of date strings going back `days` from `date` (exclusive)."""
    dt = datetime.strptime(date, "%Y-%m-%d")
    return [(dt - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(1, days + 1)]


def _week_dates(end_date: str) -> list[str]:
    """Return 7 date strings ending on end_date (inclusive)."""
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return [(end - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(6, -1, -1)]


# ===========================================================================
# 1. DailyTriage tests (~20 tests)
# ===========================================================================


class TestDailyTriageRun:
    """Tests for DailyTriage.run() top-level behavior."""

    def test_run_returns_triage_report(self, tmp_path: Path):
        curated = tmp_path / "curated"
        _write_summary(curated, "2026-03-10", "bot-a")
        triage = DailyTriage(curated, "2026-03-10", ["bot-a"])
        report = triage.run()
        assert isinstance(report, TriageReport)

    def test_run_includes_routine_summary_with_bot_stats(self, tmp_path: Path):
        curated = tmp_path / "curated"
        _write_summary(curated, "2026-03-10", "bot-a", net_pnl=200, total_trades=8, win_count=5)
        triage = DailyTriage(curated, "2026-03-10", ["bot-a"])
        report = triage.run()
        assert "bot-a" in report.routine_summary
        assert "8 trades" in report.routine_summary
        assert "+200" in report.routine_summary

    def test_run_populates_relevant_data_keys_from_events(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        # Create a PnL anomaly: today = 1000, trailing = [100, 100, 100, 100, 100]
        _write_summary(curated, date, "bot-a", net_pnl=1000)
        for d in _trailing_dates(date, 5):
            _write_summary(curated, d, "bot-a", net_pnl=100)
        triage = DailyTriage(curated, date, ["bot-a"])
        report = triage.run()
        assert "summary.json" in report.relevant_data_keys

    def test_run_includes_execution_artifacts_when_present(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        bot_dir = curated / date / "bot-a"
        _write_summary(curated, date, "bot-a")
        bot_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("order_lifecycle.json", "process_quality.json", "parameter_changes.json"):
            (bot_dir / filename).write_text("{}", encoding="utf-8")

        triage = DailyTriage(curated, date, ["bot-a"])
        report = triage.run()

        assert "order_lifecycle.json" in report.relevant_data_keys
        assert "process_quality.json" in report.relevant_data_keys
        assert "parameter_changes.json" in report.relevant_data_keys

    def test_run_limits_focus_questions_to_max_5(self, tmp_path: Path):
        """Even with many events, focus_questions should not exceed 5."""
        curated = tmp_path / "curated"
        date = "2026-03-10"
        bots = ["bot-a", "bot-b", "bot-c", "bot-d", "bot-e", "bot-f"]
        # Create PnL anomalies for every bot
        for bot in bots:
            _write_summary(curated, date, bot, net_pnl=5000)
            for d in _trailing_dates(date, 5):
                _write_summary(curated, d, bot, net_pnl=50)
        triage = DailyTriage(curated, date, bots)
        report = triage.run()
        assert len(report.focus_questions) <= 5

    def test_handles_missing_curated_data_gracefully(self, tmp_path: Path):
        """No crash when curated directory does not exist."""
        curated = tmp_path / "curated"  # Not created on disk
        triage = DailyTriage(curated, "2026-03-10", ["bot-a"])
        report = triage.run()
        assert isinstance(report, TriageReport)
        assert report.significant_events == []


class TestSignificantEventDataclass:
    """Tests for SignificantEvent field structure."""

    def test_significant_event_has_correct_fields(self):
        event = SignificantEvent(
            event_type="pnl_anomaly",
            bot_id="bot-a",
            severity="high",
            description="Big loss",
            relevant_data_keys=["summary.json"],
        )
        assert event.event_type == "pnl_anomaly"
        assert event.bot_id == "bot-a"
        assert event.severity == "high"
        assert event.description == "Big loss"
        assert event.relevant_data_keys == ["summary.json"]

    def test_significant_event_defaults_relevant_data_keys(self):
        event = SignificantEvent(
            event_type="regime_shift", bot_id="bot-x", severity="medium",
            description="Regime changed",
        )
        assert event.relevant_data_keys == []


class TestCheckPnlAnomaly:
    """Tests for DailyTriage._check_pnl_anomaly."""

    def test_detects_positive_anomaly(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a", net_pnl=500)
        # Trailing: mean~100, small stdev → z-score >> 2
        trailing_pnls = [95, 105, 98, 102, 100]
        for i, d in enumerate(_trailing_dates(date, 5)):
            _write_summary(curated, d, "bot-a", net_pnl=trailing_pnls[i])
        triage = DailyTriage(curated, date, ["bot-a"])
        trailing = triage._load_trailing_summaries("bot-a")
        event = triage._check_pnl_anomaly("bot-a", 500.0, trailing)
        assert event is not None
        assert event.event_type == "pnl_anomaly"
        assert "gain" in event.description.lower()

    def test_detects_negative_anomaly(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a", net_pnl=-500)
        trailing_pnls = [95, 105, 98, 102, 100]
        for i, d in enumerate(_trailing_dates(date, 5)):
            _write_summary(curated, d, "bot-a", net_pnl=trailing_pnls[i])
        triage = DailyTriage(curated, date, ["bot-a"])
        trailing = triage._load_trailing_summaries("bot-a")
        event = triage._check_pnl_anomaly("bot-a", -500.0, trailing)
        assert event is not None
        assert "loss" in event.description.lower()

    def test_returns_none_when_fewer_than_3_trailing_days(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a", net_pnl=500)
        # Only 2 trailing days
        for d in _trailing_dates(date, 2):
            _write_summary(curated, d, "bot-a", net_pnl=100)
        triage = DailyTriage(curated, date, ["bot-a"], trailing_days=5)
        trailing = triage._load_trailing_summaries("bot-a")
        event = triage._check_pnl_anomaly("bot-a", 500.0, trailing)
        assert event is None

    def test_returns_none_when_z_score_below_threshold(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        # Today PnL is close to trailing mean — small z-score
        _write_summary(curated, date, "bot-a", net_pnl=105)
        for d in _trailing_dates(date, 5):
            _write_summary(curated, d, "bot-a", net_pnl=100)
        triage = DailyTriage(curated, date, ["bot-a"])
        trailing = triage._load_trailing_summaries("bot-a")
        event = triage._check_pnl_anomaly("bot-a", 105.0, trailing)
        assert event is None


class TestCheckDrawdownSpike:
    """Tests for DailyTriage._check_drawdown_spike."""

    def test_detects_drawdown_exceeding_trailing_max(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a", max_drawdown_pct=15.0)
        for d in _trailing_dates(date, 5):
            _write_summary(curated, d, "bot-a", max_drawdown_pct=5.0)
        triage = DailyTriage(curated, date, ["bot-a"])
        summary = json.loads(
            (curated / date / "bot-a" / "summary.json").read_text()
        )
        trailing = triage._load_trailing_summaries("bot-a")
        event = triage._check_drawdown_spike("bot-a", summary, trailing)
        assert event is not None
        assert event.event_type == "drawdown_spike"
        assert "15.0%" in event.description

    def test_returns_none_when_today_within_trailing_max(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a", max_drawdown_pct=4.0)
        for d in _trailing_dates(date, 5):
            _write_summary(curated, d, "bot-a", max_drawdown_pct=5.0)
        triage = DailyTriage(curated, date, ["bot-a"])
        summary = json.loads(
            (curated / date / "bot-a" / "summary.json").read_text()
        )
        trailing = triage._load_trailing_summaries("bot-a")
        event = triage._check_drawdown_spike("bot-a", summary, trailing)
        assert event is None


class TestCheckRegimeShift:
    """Tests for DailyTriage._check_regime_shift."""

    def test_detects_regime_change(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a")
        _write_regime(curated, date, "bot-a", "trending")
        for d in _trailing_dates(date, 5):
            _write_summary(curated, d, "bot-a")
            _write_regime(curated, d, "bot-a", "mean_reversion")
        triage = DailyTriage(curated, date, ["bot-a"])
        event = triage._check_regime_shift("bot-a")
        assert event is not None
        assert event.event_type == "regime_shift"
        assert "trending" in event.description.lower() or "regime" in event.description.lower()

    def test_returns_none_when_regime_unchanged(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a")
        _write_regime(curated, date, "bot-a", "trending")
        for d in _trailing_dates(date, 5):
            _write_summary(curated, d, "bot-a")
            _write_regime(curated, d, "bot-a", "trending")
        triage = DailyTriage(curated, date, ["bot-a"])
        event = triage._check_regime_shift("bot-a")
        assert event is None

    def test_returns_none_when_no_regime_data(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a")
        # No regime_analysis.json written
        triage = DailyTriage(curated, date, ["bot-a"])
        event = triage._check_regime_shift("bot-a")
        assert event is None


class TestCheckSuggestionConflicts:
    """Tests for DailyTriage._check_suggestion_conflicts."""

    def test_flags_deployed_suggestion_on_negative_pnl_day(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a", net_pnl=-200)
        suggestions = [
            {
                "suggestion_id": "abc12345",
                "bot_id": "bot-a",
                "status": "deployed",
                "title": "Widen stop loss",
            }
        ]
        triage = DailyTriage(curated, date, ["bot-a"], active_suggestions=suggestions)
        summary = {"net_pnl": -200}
        events = triage._check_suggestion_conflicts("bot-a", summary)
        assert len(events) >= 1
        assert events[0].event_type == "outcome_conflict"
        assert "abc12345" in events[0].description

    def test_ignores_non_deployed_suggestions(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a", net_pnl=-200)
        suggestions = [
            {
                "suggestion_id": "def456",
                "bot_id": "bot-a",
                "status": "proposed",
                "title": "Some idea",
            }
        ]
        triage = DailyTriage(curated, date, ["bot-a"], active_suggestions=suggestions)
        summary = {"net_pnl": -200}
        events = triage._check_suggestion_conflicts("bot-a", summary)
        assert len(events) == 0

    def test_ignores_deployed_suggestion_on_positive_pnl_day(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a", net_pnl=200)
        suggestions = [
            {
                "suggestion_id": "abc12345",
                "bot_id": "bot-a",
                "status": "deployed",
                "title": "Widen stop loss",
            }
        ]
        triage = DailyTriage(curated, date, ["bot-a"], active_suggestions=suggestions)
        summary = {"net_pnl": 200}
        events = triage._check_suggestion_conflicts("bot-a", summary)
        assert len(events) == 0


class TestCheckFilterAnomaly:
    """Tests for DailyTriage._check_filter_anomaly."""

    def test_detects_high_blocked_winners(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a")
        _write_filter_analysis(curated, date, "bot-a", [
            {"name": "trend_filter", "blocked_count": 5, "would_have_won": 4},
        ])
        triage = DailyTriage(curated, date, ["bot-a"])
        event = triage._check_filter_anomaly("bot-a")
        assert event is not None
        assert event.event_type == "pattern_break"
        assert "trend_filter" in event.description

    def test_returns_none_when_no_filter_data(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a")
        # No filter_analysis.json
        triage = DailyTriage(curated, date, ["bot-a"])
        event = triage._check_filter_anomaly("bot-a")
        assert event is None

    def test_returns_none_when_blocked_winners_low(self, tmp_path: Path):
        curated = tmp_path / "curated"
        date = "2026-03-10"
        _write_summary(curated, date, "bot-a")
        _write_filter_analysis(curated, date, "bot-a", [
            {"name": "trend_filter", "blocked_count": 2, "would_have_won": 1},
        ])
        triage = DailyTriage(curated, date, ["bot-a"])
        event = triage._check_filter_anomaly("bot-a")
        assert event is None


class TestGenerateFocusQuestions:
    """Tests for DailyTriage._generate_focus_questions."""

    def test_generates_questions_from_events(self, tmp_path: Path):
        curated = tmp_path / "curated"
        triage = DailyTriage(curated, "2026-03-10", ["bot-a"])
        events = [
            SignificantEvent("pnl_anomaly", "bot-a", "high", "Unusual loss: PnL=-500"),
        ]
        questions = triage._generate_focus_questions(events)
        assert len(questions) >= 1
        assert any("bot-a" in q for q in questions)

    def test_generates_default_question_when_no_events(self, tmp_path: Path):
        curated = tmp_path / "curated"
        triage = DailyTriage(curated, "2026-03-10", ["bot-a"])
        questions = triage._generate_focus_questions([])
        assert len(questions) >= 1
        # Default question should be generic
        assert any("pattern" in q.lower() or "miss" in q.lower() or "routine" in q.lower()
                    for q in questions)


# ===========================================================================
# 2. WeeklyTriage tests (~13 tests)
# ===========================================================================


def _setup_weekly_curated(curated_dir: Path, end_date: str, bot_id: str, pnls: list[float]) -> None:
    """Write 7 daily summaries for a bot across the week."""
    dates = _week_dates(end_date)
    for i, d in enumerate(dates):
        pnl = pnls[i] if i < len(pnls) else 0
        _write_summary(curated_dir, d, bot_id, net_pnl=pnl, total_trades=10, win_count=6)


class TestWeeklyTriageRun:
    """Tests for WeeklyTriage.run() top-level behavior."""

    def test_run_returns_weekly_triage_report(self, tmp_path: Path):
        curated = tmp_path / "curated"
        end_date = "2026-03-10"
        _setup_weekly_curated(curated, end_date, "bot-a", [100]*7)
        triage = WeeklyTriage(curated, end_date, ["bot-a"])
        report = triage.run()
        assert isinstance(report, WeeklyTriageReport)

    def test_run_has_all_report_fields(self, tmp_path: Path):
        curated = tmp_path / "curated"
        end_date = "2026-03-10"
        _setup_weekly_curated(curated, end_date, "bot-a", [100]*7)
        triage = WeeklyTriage(curated, end_date, ["bot-a"])
        report = triage.run()
        assert report.week_start is not None
        assert report.week_end == end_date
        assert isinstance(report.bot_summaries, list)
        assert isinstance(report.computed_summary, str)
        assert isinstance(report.anomalies, list)
        assert isinstance(report.retrospective_questions, list)
        assert isinstance(report.discovery_questions, list)
        assert isinstance(report.relevant_data_keys, list)

    def test_run_has_computed_summary(self, tmp_path: Path):
        curated = tmp_path / "curated"
        end_date = "2026-03-10"
        _setup_weekly_curated(curated, end_date, "bot-a", [100]*7)
        triage = WeeklyTriage(curated, end_date, ["bot-a"])
        report = triage.run()
        assert report.computed_summary != ""

    def test_handles_missing_data_gracefully(self, tmp_path: Path):
        curated = tmp_path / "curated"  # Not created
        triage = WeeklyTriage(curated, "2026-03-10", ["bot-a"])
        report = triage.run()
        assert isinstance(report, WeeklyTriageReport)
        assert len(report.bot_summaries) == 1
        assert report.bot_summaries[0].total_trades == 0


class TestComputeBotWeek:
    """Tests for WeeklyTriage._compute_bot_week."""

    def test_computes_total_pnl_and_win_rate(self, tmp_path: Path):
        curated = tmp_path / "curated"
        end_date = "2026-03-10"
        _setup_weekly_curated(curated, end_date, "bot-a", [100, 200, -50, 150, 100, -30, 80])
        triage = WeeklyTriage(curated, end_date, ["bot-a"])
        summary = triage._compute_bot_week("bot-a")
        assert summary.bot_id == "bot-a"
        assert abs(summary.total_pnl - 550.0) < 0.01
        assert summary.total_trades == 70  # 10 trades * 7 days
        assert abs(summary.win_rate - (42 / 70)) < 0.01  # 6 wins * 7 days / 70

    def test_detects_improving_trend(self, tmp_path: Path):
        curated = tmp_path / "curated"
        end_date = "2026-03-10"
        # First half low, second half much higher → improving
        pnls = [10, 10, 10, 100, 200, 200, 200]
        _setup_weekly_curated(curated, end_date, "bot-a", pnls)
        triage = WeeklyTriage(curated, end_date, ["bot-a"])
        summary = triage._compute_bot_week("bot-a")
        assert summary.trend == "improving"

    def test_detects_declining_trend(self, tmp_path: Path):
        curated = tmp_path / "curated"
        end_date = "2026-03-10"
        # First half high, second half much lower → declining
        pnls = [200, 200, 200, 10, 10, 10, 10]
        _setup_weekly_curated(curated, end_date, "bot-a", pnls)
        triage = WeeklyTriage(curated, end_date, ["bot-a"])
        summary = triage._compute_bot_week("bot-a")
        assert summary.trend == "declining"

    def test_stable_trend_when_no_significant_change(self, tmp_path: Path):
        curated = tmp_path / "curated"
        end_date = "2026-03-10"
        # With 7 values, first_half = [:3], second_half = [3:]
        # first_half = 300, second_half = 400 → need to balance
        # Use 4 values so first_half=[:2]=200, second_half=[2:]=200 → stable
        # Or use values where second_half is within 0.7-1.3x of first_half
        pnls = [100, 100, 100, 90, 100, 100, 110]  # first_half=300, second_half=400 still
        # Actually: 7 // 2 = 3, so first_half = sum([:3]) = 300
        # second_half = sum([3:]) = 400. 400 > 300*1.3=390 → improving
        # Need: second_half <= first_half * 1.3 AND second_half >= first_half * 0.7
        # first_half=3 items, second_half=4 items
        # Make first_half bigger: [130, 130, 130, 100, 100, 100, 100]
        # first_half=390, second_half=400 → 400 <= 390*1.3=507 and 400 >= 390*0.7=273 → stable
        pnls = [130, 130, 130, 100, 100, 100, 100]
        _setup_weekly_curated(curated, end_date, "bot-a", pnls)
        triage = WeeklyTriage(curated, end_date, ["bot-a"])
        summary = triage._compute_bot_week("bot-a")
        assert summary.trend == "stable"

    def test_tracks_regime_changes(self, tmp_path: Path):
        curated = tmp_path / "curated"
        end_date = "2026-03-10"
        dates = _week_dates(end_date)
        for d in dates:
            _write_summary(curated, d, "bot-a")
        # Alternating regimes → lots of changes
        _write_regime(curated, dates[0], "bot-a", "trending")
        _write_regime(curated, dates[1], "bot-a", "mean_reversion")
        _write_regime(curated, dates[2], "bot-a", "trending")
        _write_regime(curated, dates[3], "bot-a", "mean_reversion")
        _write_regime(curated, dates[4], "bot-a", "trending")
        _write_regime(curated, dates[5], "bot-a", "mean_reversion")
        _write_regime(curated, dates[6], "bot-a", "trending")
        triage = WeeklyTriage(curated, end_date, ["bot-a"])
        summary = triage._compute_bot_week("bot-a")
        assert summary.regime_changes >= 4


class TestCheckCrossBotDivergence:
    """Tests for WeeklyTriage._check_cross_bot_divergence."""

    def test_detects_performance_divergence(self, tmp_path: Path):
        curated = tmp_path / "curated"
        end_date = "2026-03-10"
        # Bot A does great, Bot B terrible, Bot C mediocre → large spread
        _setup_weekly_curated(curated, end_date, "bot-a", [500]*7)
        _setup_weekly_curated(curated, end_date, "bot-b", [-500]*7)
        _setup_weekly_curated(curated, end_date, "bot-c", [0]*7)
        triage = WeeklyTriage(curated, end_date, ["bot-a", "bot-b", "bot-c"])
        summaries = [triage._compute_bot_week(b) for b in ["bot-a", "bot-b", "bot-c"]]
        anomaly = triage._check_cross_bot_divergence(summaries)
        assert anomaly is not None
        assert anomaly.anomaly_type == "cross_bot_divergence"

    def test_returns_none_with_fewer_than_2_bots(self, tmp_path: Path):
        curated = tmp_path / "curated"
        end_date = "2026-03-10"
        _setup_weekly_curated(curated, end_date, "bot-a", [100]*7)
        triage = WeeklyTriage(curated, end_date, ["bot-a"])
        summaries = [triage._compute_bot_week("bot-a")]
        anomaly = triage._check_cross_bot_divergence(summaries)
        assert anomaly is None


class TestCheckSuggestionRetrospective:
    """Tests for WeeklyTriage._check_suggestion_retrospective."""

    def test_flags_negative_outcomes(self, tmp_path: Path):
        curated = tmp_path / "curated"
        outcomes = [
            {
                "suggestion_id": "abc12345",
                "bot_id": "bot-a",
                "verdict": "negative",
                "measurement_quality": "high",
            }
        ]
        triage = WeeklyTriage(curated, "2026-03-10", ["bot-a"], outcome_measurements=outcomes)
        anomalies = triage._check_suggestion_retrospective()
        assert len(anomalies) >= 1
        assert anomalies[0].anomaly_type == "suggestion_retrospective"
        assert anomalies[0].severity == "high"

    def test_flags_inconclusive_outcomes(self, tmp_path: Path):
        curated = tmp_path / "curated"
        outcomes = [
            {
                "suggestion_id": "def67890",
                "bot_id": "bot-a",
                "verdict": "inconclusive",
                "measurement_quality": "medium",
            }
        ]
        triage = WeeklyTriage(curated, "2026-03-10", ["bot-a"], outcome_measurements=outcomes)
        anomalies = triage._check_suggestion_retrospective()
        assert len(anomalies) >= 1
        assert anomalies[0].severity == "medium"
        assert "INCONCLUSIVE" in anomalies[0].description


class TestWeeklyTriageQuestions:
    """Tests for question generation methods."""

    def test_generate_retrospective_questions_from_anomalies(self, tmp_path: Path):
        curated = tmp_path / "curated"
        outcomes = [
            {
                "suggestion_id": "abc12345",
                "bot_id": "bot-a",
                "verdict": "negative",
                "measurement_quality": "high",
            }
        ]
        triage = WeeklyTriage(curated, "2026-03-10", ["bot-a"], outcome_measurements=outcomes)
        anomalies = triage._check_suggestion_retrospective()
        questions = triage._generate_retrospective_questions(anomalies)
        assert len(questions) >= 1
        assert any("wrong" in q.lower() or "evidence" in q.lower() for q in questions)

    def test_generate_discovery_questions_from_divergence(self, tmp_path: Path):
        curated = tmp_path / "curated"
        end_date = "2026-03-10"
        _setup_weekly_curated(curated, end_date, "bot-a", [500]*7)
        _setup_weekly_curated(curated, end_date, "bot-b", [-500]*7)
        _setup_weekly_curated(curated, end_date, "bot-c", [0]*7)
        triage = WeeklyTriage(curated, end_date, ["bot-a", "bot-b", "bot-c"])
        summaries = [triage._compute_bot_week(b) for b in ["bot-a", "bot-b", "bot-c"]]
        divergence = triage._check_cross_bot_divergence(summaries)
        anomalies = [divergence] if divergence else []
        questions = triage._generate_discovery_questions(summaries, anomalies)
        assert len(questions) >= 1
        assert any("diverge" in q.lower() or "factor" in q.lower() for q in questions)


# ===========================================================================
# 3. DailyPromptAssembler triage integration tests (~5 tests)
# ===========================================================================


class TestDailyPromptAssemblerTriageIntegration:
    """Tests for DailyPromptAssembler with triage_report parameter."""

    def test_assemble_works_without_triage(self, tmp_path: Path):
        memory_dir = _setup_memory(tmp_path)
        curated = tmp_path / "curated"
        curated.mkdir()
        assembler = DailyPromptAssembler(
            date="2026-03-10",
            bots=["bot-a"],
            curated_dir=curated,
            memory_dir=memory_dir,
        )
        pkg = assembler.assemble()
        assert pkg.instructions != ""
        # Fallback instructions should contain the legacy template text
        assert "triage" in pkg.instructions.lower() or "anomal" in pkg.instructions.lower()

    def test_assemble_with_triage_uses_focused_instructions(self, tmp_path: Path):
        memory_dir = _setup_memory(tmp_path)
        curated = tmp_path / "curated"
        curated.mkdir()
        triage_report = TriageReport(
            significant_events=[
                SignificantEvent("pnl_anomaly", "bot-a", "high", "Unusual loss: PnL=-500"),
            ],
            routine_summary="bot-a: 10 trades, PnL=-500, WR=40%",
            focus_questions=["Why did bot-a have an unusual loss?"],
            relevant_data_keys=["summary.json", "losers.json"],
        )
        assembler = DailyPromptAssembler(
            date="2026-03-10",
            bots=["bot-a"],
            curated_dir=curated,
            memory_dir=memory_dir,
        )
        pkg = assembler.assemble(triage_report=triage_report)
        assert "PNL_ANOMALY" in pkg.instructions
        assert "bot-a" in pkg.instructions

    def test_focused_instructions_include_significant_events(self, tmp_path: Path):
        memory_dir = _setup_memory(tmp_path)
        curated = tmp_path / "curated"
        curated.mkdir()
        triage_report = TriageReport(
            significant_events=[
                SignificantEvent("drawdown_spike", "bot-b", "high", "Drawdown 15% exceeds trailing max 5%"),
            ],
            routine_summary="bot-b: 12 trades",
            focus_questions=["Investigate drawdown spike"],
            relevant_data_keys=["summary.json"],
        )
        assembler = DailyPromptAssembler(
            date="2026-03-10",
            bots=["bot-b"],
            curated_dir=curated,
            memory_dir=memory_dir,
        )
        pkg = assembler.assemble(triage_report=triage_report)
        assert "DRAWDOWN_SPIKE" in pkg.instructions
        assert "15%" in pkg.instructions

    def test_focused_instructions_include_focus_questions(self, tmp_path: Path):
        memory_dir = _setup_memory(tmp_path)
        curated = tmp_path / "curated"
        curated.mkdir()
        triage_report = TriageReport(
            significant_events=[],
            routine_summary="Routine day",
            focus_questions=["Is the exit timing strategy still working?"],
            relevant_data_keys=["summary.json"],
        )
        assembler = DailyPromptAssembler(
            date="2026-03-10",
            bots=["bot-a"],
            curated_dir=curated,
            memory_dir=memory_dir,
        )
        pkg = assembler.assemble(triage_report=triage_report)
        assert "exit timing" in pkg.instructions

    def test_load_structured_data_filters_by_triage_relevant_data_keys(self, tmp_path: Path):
        memory_dir = _setup_memory(tmp_path)
        curated = tmp_path / "curated"
        date = "2026-03-10"
        bot_dir = curated / date / "bot-a"
        bot_dir.mkdir(parents=True)
        # Write several curated files
        (bot_dir / "summary.json").write_text('{"net_pnl": 100}')
        (bot_dir / "losers.json").write_text('[{"trade_id": "t1"}]')
        (bot_dir / "winners.json").write_text('[{"trade_id": "t2"}]')
        (bot_dir / "filter_analysis.json").write_text('{"filters": []}')

        # Triage only requests summary.json and losers.json
        triage_report = TriageReport(
            significant_events=[],
            routine_summary="",
            focus_questions=[],
            relevant_data_keys=["summary.json", "losers.json"],
        )
        assembler = DailyPromptAssembler(
            date=date,
            bots=["bot-a"],
            curated_dir=curated,
            memory_dir=memory_dir,
        )
        data = assembler._load_structured_data(triage_report)
        bot_data = data.get("bot-a", {})
        assert "summary" in bot_data
        assert "losers" in bot_data
        # winners.json and filter_analysis.json should NOT be loaded
        assert "winners" not in bot_data
        assert "filter_analysis" not in bot_data

    def test_load_structured_data_includes_execution_artifacts_when_requested(self, tmp_path: Path):
        memory_dir = _setup_memory(tmp_path)
        curated = tmp_path / "curated"
        date = "2026-03-10"
        bot_dir = curated / date / "bot-a"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text('{"net_pnl": 100}')
        (bot_dir / "order_lifecycle.json").write_text('{"reject_count": 1}')
        (bot_dir / "process_quality.json").write_text('{"low_score_count": 2}')
        (bot_dir / "parameter_changes.json").write_text('{"total_changes": 1}')

        triage_report = TriageReport(
            significant_events=[],
            routine_summary="",
            focus_questions=[],
            relevant_data_keys=[
                "summary.json",
                "order_lifecycle.json",
                "process_quality.json",
                "parameter_changes.json",
            ],
        )
        assembler = DailyPromptAssembler(
            date=date,
            bots=["bot-a"],
            curated_dir=curated,
            memory_dir=memory_dir,
        )
        data = assembler._load_structured_data(triage_report)
        bot_data = data.get("bot-a", {})

        assert "order_lifecycle" in bot_data
        assert "process_quality" in bot_data
        assert "parameter_changes" in bot_data


# ===========================================================================
# 4. WeeklyPromptAssembler triage integration tests (~5 tests)
# ===========================================================================


def _setup_weekly_dirs(tmp_path: Path):
    """Set up curated + memory + runs directories for weekly assembler tests."""
    curated = tmp_path / "curated"
    memory_dir = _setup_memory(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()

    end_date = "2026-03-10"
    dates = _week_dates(end_date)

    # Weekly summary
    weekly_dir = curated / "weekly" / "2026-03-04"
    weekly_dir.mkdir(parents=True)
    (weekly_dir / "weekly_summary.json").write_text(
        json.dumps({"total_net_pnl": 700, "total_trades": 50})
    )

    # Daily data + daily run reports
    for d in dates:
        _write_summary(curated, d, "bot-a", net_pnl=100)
        # Portfolio risk card
        portfolio_dir = curated / d
        portfolio_dir.mkdir(parents=True, exist_ok=True)
        (portfolio_dir / "portfolio_risk_card.json").write_text(
            json.dumps({"crowding_alerts": []})
        )
        # Run report
        run_dir = runs / f"daily-{d}-run"
        run_dir.mkdir(parents=True)
        (run_dir / "daily_report.md").write_text(f"Report for {d}")

    return curated, memory_dir, runs


class TestWeeklyPromptAssemblerTriageIntegration:
    """Tests for WeeklyPromptAssembler with triage_report parameter."""

    def test_assemble_works_without_triage(self, tmp_path: Path):
        curated, memory_dir, runs = _setup_weekly_dirs(tmp_path)
        assembler = WeeklyPromptAssembler(
            week_start="2026-03-04",
            week_end="2026-03-10",
            bots=["bot-a"],
            curated_dir=curated,
            memory_dir=memory_dir,
            runs_dir=runs,
        )
        pkg = assembler.assemble()
        assert pkg.instructions != ""
        assert "weekly" in pkg.task_prompt.lower()

    def test_assemble_with_triage_uses_focused_instructions(self, tmp_path: Path):
        curated, memory_dir, runs = _setup_weekly_dirs(tmp_path)
        triage_report = WeeklyTriageReport(
            week_start="2026-03-04",
            week_end="2026-03-10",
            bot_summaries=[
                BotWeekSummary(bot_id="bot-a", total_pnl=700, total_trades=50,
                               win_rate=0.6, trend="stable"),
            ],
            computed_summary="bot-a: 700 PnL, 50 trades, stable",
            anomalies=[
                WeeklyAnomaly("trajectory_change", "bot-a", "medium",
                              "Performance trajectory: improving"),
            ],
            retrospective_questions=["Were past predictions accurate?"],
            discovery_questions=["What new patterns emerged?"],
            relevant_data_keys=["summary.json"],
        )
        assembler = WeeklyPromptAssembler(
            week_start="2026-03-04",
            week_end="2026-03-10",
            bots=["bot-a"],
            curated_dir=curated,
            memory_dir=memory_dir,
            runs_dir=runs,
        )
        pkg = assembler.assemble(triage_report=triage_report)
        # Focused instructions should reference the computed summary
        assert "bot-a" in pkg.instructions or "700" in pkg.instructions

    def test_focused_instructions_include_anomalies(self, tmp_path: Path):
        curated, memory_dir, runs = _setup_weekly_dirs(tmp_path)
        triage_report = WeeklyTriageReport(
            week_start="2026-03-04",
            week_end="2026-03-10",
            computed_summary="Summary text",
            anomalies=[
                WeeklyAnomaly("cross_bot_divergence", "", "high",
                              "Large performance divergence: bot-a=+500 vs bot-b=-300"),
            ],
            retrospective_questions=[],
            discovery_questions=[],
        )
        assembler = WeeklyPromptAssembler(
            week_start="2026-03-04",
            week_end="2026-03-10",
            bots=["bot-a"],
            curated_dir=curated,
            memory_dir=memory_dir,
            runs_dir=runs,
        )
        pkg = assembler.assemble(triage_report=triage_report)
        assert "CROSS_BOT_DIVERGENCE" in pkg.instructions
        assert "divergence" in pkg.instructions.lower()

    def test_focused_instructions_include_retrospective_questions(self, tmp_path: Path):
        curated, memory_dir, runs = _setup_weekly_dirs(tmp_path)
        triage_report = WeeklyTriageReport(
            week_start="2026-03-04",
            week_end="2026-03-10",
            computed_summary="Summary",
            anomalies=[],
            retrospective_questions=[
                "Suggestion #abc had negative outcome. What went wrong?"
            ],
            discovery_questions=[],
        )
        assembler = WeeklyPromptAssembler(
            week_start="2026-03-04",
            week_end="2026-03-10",
            bots=["bot-a"],
            curated_dir=curated,
            memory_dir=memory_dir,
            runs_dir=runs,
        )
        pkg = assembler.assemble(triage_report=triage_report)
        assert "negative outcome" in pkg.instructions

    def test_focused_instructions_include_discovery_questions(self, tmp_path: Path):
        curated, memory_dir, runs = _setup_weekly_dirs(tmp_path)
        triage_report = WeeklyTriageReport(
            week_start="2026-03-04",
            week_end="2026-03-10",
            computed_summary="Summary",
            anomalies=[],
            retrospective_questions=[],
            discovery_questions=[
                "What market factor explains the cross-bot divergence?"
            ],
        )
        assembler = WeeklyPromptAssembler(
            week_start="2026-03-04",
            week_end="2026-03-10",
            bots=["bot-a"],
            curated_dir=curated,
            memory_dir=memory_dir,
            runs_dir=runs,
        )
        pkg = assembler.assemble(triage_report=triage_report)
        assert "market factor" in pkg.instructions
