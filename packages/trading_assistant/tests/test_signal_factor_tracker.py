"""Tests for signal factor tracker — schemas, JSONL persistence, and rolling analysis."""
from __future__ import annotations

import json
from pathlib import Path


from trading_assistant.schemas.signal_factor_history import (
    DailyFactorSnapshot,
    FactorDayStats,
    FactorRollingResult,
    SignalFactorRollingReport,
)
from trading_assistant.skills.signal_factor_tracker import SignalFactorTracker


# ---------------------------------------------------------------------------
# Schema round-trips
# ---------------------------------------------------------------------------

class TestSignalFactorSchemas:
    def test_factor_day_stats_round_trip(self):
        stats = FactorDayStats(
            factor_name="rsi", trade_count=20, win_rate=0.65,
            avg_pnl=12.5, total_pnl=250.0, avg_contribution=0.4,
        )
        data = stats.model_dump(mode="json")
        restored = FactorDayStats(**data)
        assert restored.factor_name == "rsi"
        assert restored.win_rate == 0.65

    def test_daily_factor_snapshot_round_trip(self):
        snap = DailyFactorSnapshot(
            date="2026-03-03", bot_id="bot_a",
            factors=[FactorDayStats(factor_name="rsi", trade_count=10, win_rate=0.5)],
        )
        data = snap.model_dump(mode="json")
        restored = DailyFactorSnapshot(**data)
        assert len(restored.factors) == 1

    def test_factor_rolling_result_round_trip(self):
        result = FactorRollingResult(
            factor_name="rsi", bot_id="bot_a",
            current_win_rate=0.55, rolling_30d_win_rate=0.60,
            win_rate_trend="stable", days_of_data=25,
        )
        data = result.model_dump(mode="json")
        restored = FactorRollingResult(**data)
        assert restored.win_rate_trend == "stable"

    def test_signal_factor_rolling_report_round_trip(self):
        report = SignalFactorRollingReport(
            bot_id="bot_a", date="2026-03-03", window_days=30,
            factors=[FactorRollingResult(factor_name="rsi", bot_id="bot_a")],
            alerts=["rsi degrading"],
        )
        data = report.model_dump(mode="json")
        restored = SignalFactorRollingReport(**data)
        assert len(restored.alerts) == 1


# ---------------------------------------------------------------------------
# Tracker persistence
# ---------------------------------------------------------------------------

class TestTrackerPersistence:
    def test_record_creates_file(self, tmp_path: Path):
        tracker = SignalFactorTracker(tmp_path)
        snap = DailyFactorSnapshot(
            date="2026-03-03", bot_id="bot_a",
            factors=[FactorDayStats(factor_name="rsi", trade_count=10, win_rate=0.5)],
        )
        tracker.record_daily(snap)
        assert (tmp_path / "signal_factor_history.jsonl").exists()

    def test_record_appends(self, tmp_path: Path):
        tracker = SignalFactorTracker(tmp_path)
        for i in range(3):
            snap = DailyFactorSnapshot(
                date=f"2026-03-0{i+1}", bot_id="bot_a",
                factors=[FactorDayStats(factor_name="rsi", trade_count=10, win_rate=0.5)],
            )
            tracker.record_daily(snap)
        lines = (tmp_path / "signal_factor_history.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3

    def test_load_filters_by_bot(self, tmp_path: Path):
        tracker = SignalFactorTracker(tmp_path)
        for bot in ["bot_a", "bot_b"]:
            snap = DailyFactorSnapshot(
                date="2026-03-03", bot_id=bot,
                factors=[FactorDayStats(factor_name="rsi", trade_count=10, win_rate=0.5)],
            )
            tracker.record_daily(snap)
        history = tracker.load_history("bot_a", days=30, as_of="2026-03-03")
        assert len(history) == 1
        assert history[0].bot_id == "bot_a"

    def test_load_filters_by_date(self, tmp_path: Path):
        tracker = SignalFactorTracker(tmp_path)
        for day in ["2026-01-15", "2026-03-01", "2026-03-03"]:
            snap = DailyFactorSnapshot(
                date=day, bot_id="bot_a",
                factors=[FactorDayStats(factor_name="rsi", trade_count=10, win_rate=0.5)],
            )
            tracker.record_daily(snap)
        # 30d window from 2026-03-03 = Jan 15 excluded
        history = tracker.load_history("bot_a", days=30, as_of="2026-03-03")
        dates = [h.date for h in history]
        assert "2026-01-15" not in dates
        assert "2026-03-01" in dates
        assert "2026-03-03" in dates


# ---------------------------------------------------------------------------
# Rolling analysis
# ---------------------------------------------------------------------------

class TestRollingAnalysis:
    def test_empty_history(self, tmp_path: Path):
        tracker = SignalFactorTracker(tmp_path)
        report = tracker.compute_rolling("bot_a", "2026-03-03")
        assert report.factors == []
        assert report.alerts == []

    def test_insufficient_days(self, tmp_path: Path):
        tracker = SignalFactorTracker(tmp_path)
        for i in range(5):
            snap = DailyFactorSnapshot(
                date=f"2026-03-0{i+1}", bot_id="bot_a",
                factors=[FactorDayStats(factor_name="rsi", trade_count=10, win_rate=0.5)],
            )
            tracker.record_daily(snap)
        report = tracker.compute_rolling("bot_a", "2026-03-05", min_days=10)
        assert report.factors == []

    def test_stable_factors(self, tmp_path: Path):
        tracker = SignalFactorTracker(tmp_path)
        for i in range(15):
            snap = DailyFactorSnapshot(
                date=f"2026-02-{17+i:02d}", bot_id="bot_a",
                factors=[FactorDayStats(factor_name="rsi", trade_count=10, win_rate=0.55)],
            )
            tracker.record_daily(snap)
        report = tracker.compute_rolling("bot_a", "2026-03-03", min_days=10)
        assert len(report.factors) == 1
        assert report.factors[0].win_rate_trend == "stable"
        assert report.factors[0].below_threshold is False

    def test_degrading_factor_flagged(self, tmp_path: Path):
        tracker = SignalFactorTracker(tmp_path)
        # First half: moderate win rate
        for i in range(8):
            snap = DailyFactorSnapshot(
                date=f"2026-02-{10+i:02d}", bot_id="bot_a",
                factors=[FactorDayStats(factor_name="rsi", trade_count=10, win_rate=0.45)],
            )
            tracker.record_daily(snap)
        # Second half: low win rate — drives rolling average below threshold
        for i in range(8):
            snap = DailyFactorSnapshot(
                date=f"2026-02-{18+i:02d}", bot_id="bot_a",
                factors=[FactorDayStats(factor_name="rsi", trade_count=10, win_rate=0.20)],
            )
            tracker.record_daily(snap)
        report = tracker.compute_rolling("bot_a", "2026-02-25", min_days=10)
        assert len(report.factors) == 1
        assert report.factors[0].win_rate_trend == "degrading"
        assert report.factors[0].below_threshold is True
        assert len(report.alerts) >= 1

    def test_trend_detection_improving(self, tmp_path: Path):
        tracker = SignalFactorTracker(tmp_path)
        # First half: low win rate
        for i in range(8):
            snap = DailyFactorSnapshot(
                date=f"2026-02-{10+i:02d}", bot_id="bot_a",
                factors=[FactorDayStats(factor_name="macd", trade_count=10, win_rate=0.30)],
            )
            tracker.record_daily(snap)
        # Second half: high win rate
        for i in range(8):
            snap = DailyFactorSnapshot(
                date=f"2026-02-{18+i:02d}", bot_id="bot_a",
                factors=[FactorDayStats(factor_name="macd", trade_count=10, win_rate=0.60)],
            )
            tracker.record_daily(snap)
        report = tracker.compute_rolling("bot_a", "2026-02-25", min_days=10)
        assert report.factors[0].win_rate_trend == "improving"


# ---------------------------------------------------------------------------
# Strategy engine integration
# ---------------------------------------------------------------------------

class TestStrategyEngineIntegration:
    def test_detect_factor_correlation_decay_positive(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        results = engine.detect_factor_correlation_decay(
            "bot_a",
            [
                {"factor_name": "rsi", "win_rate_trend": "degrading", "below_threshold": True,
                 "rolling_30d_win_rate": 0.30, "days_of_data": 25},
            ],
        )
        assert len(results) == 1
        assert results[0].tier.value == "hypothesis"
        assert "rsi" in results[0].description

    def test_detect_factor_correlation_decay_negative(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        results = engine.detect_factor_correlation_decay(
            "bot_a",
            [
                {"factor_name": "rsi", "win_rate_trend": "stable", "below_threshold": False,
                 "rolling_30d_win_rate": 0.55, "days_of_data": 25},
            ],
        )
        assert len(results) == 0

    def test_build_report_with_factor_rolling(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine
        from trading_assistant.schemas.weekly_metrics import BotWeeklySummary

        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        summaries = {
            "bot_a": BotWeeklySummary(
                week_start="2026-02-24", week_end="2026-03-02",
                bot_id="bot_a", total_trades=50,
                win_count=25, loss_count=25,
                avg_win=20, avg_loss=-10,
            ),
        }
        report = engine.build_report(
            summaries,
            factor_rolling={
                "bot_a": [
                    {"factor_name": "rsi", "win_rate_trend": "degrading", "below_threshold": True,
                     "rolling_30d_win_rate": 0.28, "days_of_data": 25},
                ],
            },
        )
        # Should have at least the factor decay suggestion
        factor_suggestions = [s for s in report.suggestions if "rsi" in s.description.lower()]
        assert len(factor_suggestions) >= 1


# ---------------------------------------------------------------------------
# Build pipeline integration
# ---------------------------------------------------------------------------

class TestBuildPipelineIntegration:
    def test_write_curated_records_history(self, tmp_path: Path):
        """write_curated with findings_dir records factor history."""
        from trading_assistant.schemas.events import TradeEvent

        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        curated_dir = tmp_path / "curated"
        builder = DailyMetricsBuilder(bot_id="bot_a", date="2026-03-03")
        trades = [
            TradeEvent(
                trade_id="t1", bot_id="bot_a", pair="BTC/USDT", side="long",
                pnl=100.0, pnl_pct=0.2, entry_price=50000, exit_price=50100,
                position_size=1.0,
                entry_time="2026-03-03T10:00:00Z", exit_time="2026-03-03T12:00:00Z",
                entry_signal="rsi_cross", exit_reason="tp",
                market_regime="trending", process_quality_score=80,
            ),
        ]
        builder.write_curated(trades, [], curated_dir, findings_dir=findings_dir)
        history_path = findings_dir / "signal_factor_history.jsonl"
        assert history_path.exists()
        records = history_path.read_text().strip().split("\n")
        assert len(records) >= 1
        data = json.loads(records[0])
        assert data["bot_id"] == "bot_a"
        assert data["date"] == "2026-03-03"

    def test_write_curated_skips_when_no_findings_dir(self, tmp_path: Path):
        """write_curated without findings_dir doesn't create history file."""
        from trading_assistant.schemas.events import TradeEvent

        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        curated_dir = tmp_path / "curated"
        builder = DailyMetricsBuilder(bot_id="bot_a", date="2026-03-03")
        trades = [
            TradeEvent(
                trade_id="t1", bot_id="bot_a", pair="BTC/USDT", side="long",
                pnl=100.0, pnl_pct=0.2, entry_price=50000, exit_price=50100,
                position_size=1.0,
                entry_time="2026-03-03T10:00:00Z", exit_time="2026-03-03T12:00:00Z",
                entry_signal="rsi_cross", exit_reason="tp",
                market_regime="trending", process_quality_score=80,
            ),
        ]
        builder.write_curated(trades, [], curated_dir)
        # No findings dir means no history file anywhere in tmp_path
        history_files = list(tmp_path.rglob("signal_factor_history.jsonl"))
        assert len(history_files) == 0


# ---------------------------------------------------------------------------
# Assembler integration
# ---------------------------------------------------------------------------

class TestAssemblerIntegration:
    def test_factor_trends_in_context_builder(self, tmp_path: Path):
        """ContextBuilder.load_signal_factor_history returns rolling data."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()

        tracker = SignalFactorTracker(findings_dir)
        for i in range(15):
            snap = DailyFactorSnapshot(
                date=f"2026-02-{17+i:02d}", bot_id="bot_a",
                factors=[FactorDayStats(factor_name="rsi", trade_count=10, win_rate=0.55)],
            )
            tracker.record_daily(snap)

        ctx = ContextBuilder(memory_dir)
        result = ctx.load_signal_factor_history("bot_a", "2026-03-03", findings_dir)
        assert isinstance(result, dict)
        assert result["bot_id"] == "bot_a"
        assert len(result["factors"]) == 1

    def test_factor_trends_empty_when_no_history(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        ctx = ContextBuilder(memory_dir)
        result = ctx.load_signal_factor_history("bot_a", "2026-03-03", tmp_path)
        assert result == {}
