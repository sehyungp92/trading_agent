# tests/test_ground_truth.py
"""Tests for ground truth computer, learning ledger, and context integration."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading_assistant.schemas.learning_ledger import (
    DiscardItem,
    GroundTruthSnapshot,
    RetrospectiveSynthesis,
    SynthesisItem,
)
from trading_assistant.skills.ground_truth_computer import GroundTruthComputer
from trading_assistant.skills.learning_ledger import LearningLedger


# ── Fixtures ──

@pytest.fixture
def curated_dir(tmp_path: Path) -> Path:
    d = tmp_path / "curated"
    d.mkdir()
    return d


@pytest.fixture
def findings_dir(tmp_path: Path) -> Path:
    d = tmp_path / "findings"
    d.mkdir()
    return d


@pytest.fixture
def memory_dir(tmp_path: Path) -> Path:
    d = tmp_path / "memory"
    (d / "findings").mkdir(parents=True)
    (d / "policies" / "v1").mkdir(parents=True)
    return d


def _write_daily_summary(
    curated_dir: Path, date: str, bot_id: str,
    net_pnl: float = 100.0, total_trades: int = 10,
    winning_trades: int = 6, max_drawdown_pct: float = 0.05,
    avg_process_quality: float = 70.0,
    avg_win: float = 25.0, avg_loss: float = -15.0,
    win_count: int | None = None, loss_count: int | None = None,
) -> None:
    bot_dir = curated_dir / date / bot_id
    bot_dir.mkdir(parents=True, exist_ok=True)
    if win_count is None:
        win_count = winning_trades
    if loss_count is None:
        loss_count = total_trades - winning_trades
    summary = {
        "net_pnl": net_pnl,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "max_drawdown_pct": max_drawdown_pct,
        "avg_process_quality": avg_process_quality,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "date": date,
    }
    (bot_dir / "summary.json").write_text(json.dumps(summary))


# ── GroundTruthSnapshot schema ──

class TestGroundTruthSnapshot:
    def test_default_values(self):
        s = GroundTruthSnapshot(snapshot_date="2026-01-01", bot_id="bot1")
        assert s.composite_score == 0.5
        assert s.trade_count == 0
        assert s.period_days == 30

    def test_computed_at_auto_set(self):
        s = GroundTruthSnapshot(snapshot_date="2026-01-01", bot_id="bot1")
        assert s.computed_at is not None
        assert s.computed_at.tzinfo is not None


# ── GroundTruthComputer ──

class TestGroundTruthComputer:
    def test_empty_data_returns_neutral(self, curated_dir: Path):
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-03-01")
        assert snap.composite_score == 0.5
        assert snap.trade_count == 0

    def test_insufficient_trades_returns_neutral(self, curated_dir: Path):
        # Only 5 trades (below MIN_TRADES=10)
        _write_daily_summary(curated_dir, "2026-03-01", "bot1", total_trades=5, winning_trades=3)
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-03-01")
        assert snap.composite_score == 0.5
        assert snap.trade_count == 5

    def test_deterministic_output(self, curated_dir: Path):
        for i in range(30):
            date = f"2026-02-{i+1:02d}"
            _write_daily_summary(curated_dir, date, "bot1")
        computer = GroundTruthComputer(curated_dir)
        snap1 = computer.compute_snapshot("bot1", "2026-02-28")
        snap2 = computer.compute_snapshot("bot1", "2026-02-28")
        assert snap1.composite_score == snap2.composite_score

    def test_composite_within_bounds(self, curated_dir: Path):
        for i in range(30):
            date = f"2026-02-{i+1:02d}"
            _write_daily_summary(curated_dir, date, "bot1", net_pnl=50 * (i % 3 - 1))
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-02-28")
        assert 0.0 <= snap.composite_score <= 1.0

    def test_metrics_populated(self, curated_dir: Path):
        for i in range(15):
            date = f"2026-02-{i+1:02d}"
            _write_daily_summary(curated_dir, date, "bot1", net_pnl=100.0)
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-02-15")
        assert snap.pnl_total > 0
        assert snap.trade_count > 0
        assert snap.win_rate > 0
        assert snap.calmar_ratio_30d >= 0
        assert snap.profit_factor >= 0

    def test_compute_all_bots(self, curated_dir: Path):
        for i in range(15):
            date = f"2026-02-{i+1:02d}"
            _write_daily_summary(curated_dir, date, "bot_a")
            _write_daily_summary(curated_dir, date, "bot_b", net_pnl=-50.0)
        computer = GroundTruthComputer(curated_dir)
        result = computer.compute_all_bots(["bot_a", "bot_b"], "2026-02-15")
        assert "bot_a" in result
        assert "bot_b" in result

    def test_calmar_computation(self, curated_dir: Path):
        # Create data with positive PnL and a drawdown
        pnls = [100, 100, -50, 100, 100, -30, 100, 100, 100, 100,
                100, 100, -50, 100, 100, -30, 100, 100, 100, 100]
        for i, pnl in enumerate(pnls):
            date = f"2026-02-{i+1:02d}"
            _write_daily_summary(curated_dir, date, "bot1", net_pnl=pnl)
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-02-20")
        assert snap.calmar_ratio_30d > 0
        assert isinstance(snap.calmar_ratio_30d, float)
        # Sharpe still tracked as informational
        assert isinstance(snap.sharpe_ratio_30d, float)

    def test_max_drawdown_computation(self, curated_dir: Path):
        # Create a drawdown pattern: gain, gain, loss, loss, loss
        pnls = [100, 100, -200, -100, -50, 100, 100, 100, 100, 100]
        for i, pnl in enumerate(pnls):
            date = f"2026-02-{i+1:02d}"
            _write_daily_summary(curated_dir, date, "bot1", net_pnl=pnl)
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-02-10", period_days=10)
        assert snap.max_drawdown_pct > 0

    def test_profit_factor_computation(self, curated_dir: Path):
        for i in range(15):
            date = f"2026-02-{i+1:02d}"
            _write_daily_summary(
                curated_dir, date, "bot1",
                total_trades=10, winning_trades=7,
                avg_win=30.0, avg_loss=-10.0,
                win_count=7, loss_count=3,
            )
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-02-15")
        # PF = (7 * 30) / (3 * 10) = 210 / 30 = 7.0
        assert abs(snap.profit_factor - 7.0) < 0.01
        # Win rate still tracked as informational
        assert abs(snap.win_rate - 0.7) < 0.01

    def test_missing_data_days_handled(self, curated_dir: Path):
        # Only write every other day
        for i in range(0, 30, 2):
            date = f"2026-02-{i+1:02d}"
            _write_daily_summary(curated_dir, date, "bot1")
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-02-28")
        assert snap.trade_count > 0

    def test_calmar_zero_drawdown_returns_zero(self, curated_dir: Path):
        """No drawdown → calmar = 0.0 (not infinity)."""
        # Monotonically increasing equity — no drawdown possible
        for i in range(15):
            date = f"2026-02-{i+1:02d}"
            _write_daily_summary(
                curated_dir, date, "bot1",
                net_pnl=100.0, max_drawdown_pct=0.0,
            )
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-02-15")
        assert snap.calmar_ratio_30d == 0.0

    def test_profit_factor_no_losses_capped(self, curated_dir: Path):
        """When there are no losses, PF is capped at _PF_CAP (10.0)."""
        for i in range(15):
            date = f"2026-02-{i+1:02d}"
            _write_daily_summary(
                curated_dir, date, "bot1",
                total_trades=10, winning_trades=10,
                avg_win=50.0, avg_loss=0.0,
                win_count=10, loss_count=0,
            )
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-02-15")
        assert snap.profit_factor == 10.0  # capped at _PF_CAP

    def test_profit_factor_computation_accuracy(self, curated_dir: Path):
        """Verify exact PF from known data: gross_wins / gross_losses."""
        for i in range(15):
            date = f"2026-02-{i+1:02d}"
            _write_daily_summary(
                curated_dir, date, "bot1",
                total_trades=10, winning_trades=6,
                avg_win=20.0, avg_loss=-10.0,
                win_count=6, loss_count=4,
            )
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-02-15")
        # PF = (6 * 20) / (4 * 10) = 120 / 40 = 3.0
        assert abs(snap.profit_factor - 3.0) < 0.01

    def test_informational_metrics_still_tracked(self, curated_dir: Path):
        """Sharpe and win_rate are still populated even though not weighted."""
        for i in range(20):
            date = f"2026-02-{i+1:02d}"
            pnl = 100.0 if i % 2 == 0 else -30.0
            _write_daily_summary(
                curated_dir, date, "bot1",
                net_pnl=pnl, total_trades=10, winning_trades=6,
            )
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-02-20")
        assert snap.sharpe_ratio_30d != 0.0  # varied returns → nonzero sharpe
        assert snap.win_rate > 0.0

    def test_expected_total_r_populated(self, curated_dir: Path):
        """expected_total_r is annualized net PnL."""
        for i in range(15):
            date = f"2026-02-{i+1:02d}"
            _write_daily_summary(curated_dir, date, "bot1", net_pnl=100.0)
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-02-15")
        # 15 days * 100 = 1500 total → annualized = 1500 * 365 / 30 = 18250
        assert snap.expected_total_r > 0
        assert abs(snap.expected_total_r - 18250.0) < 1.0

    def test_expectancy_populated(self, curated_dir: Path):
        """Expectancy is computed from aggregated curated data."""
        for i in range(15):
            date = f"2026-02-{i+1:02d}"
            _write_daily_summary(
                curated_dir, date, "bot1",
                total_trades=10, winning_trades=6,
                avg_win=25.0, avg_loss=-15.0,
                win_count=6, loss_count=4,
            )
        computer = GroundTruthComputer(curated_dir)
        snap = computer.compute_snapshot("bot1", "2026-02-15")
        # win_rate=0.6, avg_win/avg_loss=25/15≈1.667, expectancy≈1.0
        assert abs(snap.expectancy - 1.0) < 0.01

    def test_ground_truth_snapshot_new_fields_serialized(self):
        """GroundTruthSnapshot includes expected_total_r and expectancy."""
        s = GroundTruthSnapshot(
            snapshot_date="2026-01-01", bot_id="bot1",
            expected_total_r=18250.0, expectancy=1.0,
        )
        assert s.expected_total_r == 18250.0
        assert s.expectancy == 1.0
        data = s.model_dump()
        assert "expected_total_r" in data
        assert "expectancy" in data


# ── LearningLedger ──

class TestLearningLedger:
    def test_record_week(self, findings_dir: Path):
        ledger = LearningLedger(findings_dir)
        gt_start = {"bot1": GroundTruthSnapshot(
            snapshot_date="2026-02-01", bot_id="bot1", composite_score=0.45,
        )}
        gt_end = {"bot1": GroundTruthSnapshot(
            snapshot_date="2026-02-07", bot_id="bot1", composite_score=0.55,
        )}
        entry = ledger.record_week("2026-02-01", "2026-02-07", gt_start, gt_end)
        assert entry.entry_id
        assert entry.composite_delta["bot1"] == pytest.approx(0.1, abs=0.001)
        assert entry.net_improvement is True

    def test_dedup_by_entry_id(self, findings_dir: Path):
        ledger = LearningLedger(findings_dir)
        gt = {"bot1": GroundTruthSnapshot(
            snapshot_date="2026-02-01", bot_id="bot1",
        )}
        entry1 = ledger.record_week("2026-02-01", "2026-02-07", gt, gt)
        entry2 = ledger.record_week("2026-02-01", "2026-02-07", gt, gt)
        assert entry1.entry_id == entry2.entry_id
        # Only one entry in file
        lines = (findings_dir / "learning_ledger.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1

    def test_get_trend(self, findings_dir: Path):
        ledger = LearningLedger(findings_dir)
        for week in range(3):
            gt = {"bot1": GroundTruthSnapshot(
                snapshot_date=f"2026-02-{(week+1)*7:02d}",
                bot_id="bot1",
                composite_score=0.4 + week * 0.05,
            )}
            ledger.record_week(
                f"2026-02-{week*7+1:02d}",
                f"2026-02-{(week+1)*7:02d}",
                gt, gt,
            )
        trend = ledger.get_trend(weeks=12)
        assert "bot1" in trend
        assert len(trend["bot1"]) == 3

    def test_get_lessons(self, findings_dir: Path):
        ledger = LearningLedger(findings_dir)
        gt = {"bot1": GroundTruthSnapshot(
            snapshot_date="2026-02-07", bot_id="bot1",
        )}
        ledger.record_week(
            "2026-02-01", "2026-02-07", gt, gt,
            lessons_for_next_week=["Avoid filter changes in trending regimes"],
        )
        lessons = ledger.get_lessons()
        assert len(lessons) == 1
        assert "filter" in lessons[0].lower()

    def test_get_lessons_dedup(self, findings_dir: Path):
        ledger = LearningLedger(findings_dir)
        gt = {"bot1": GroundTruthSnapshot(
            snapshot_date="2026-02-07", bot_id="bot1",
        )}
        ledger.record_week(
            "2026-02-01", "2026-02-07", gt, gt,
            lessons_for_next_week=["lesson A", "lesson B"],
        )
        gt2 = {"bot1": GroundTruthSnapshot(
            snapshot_date="2026-02-14", bot_id="bot1",
        )}
        ledger.record_week(
            "2026-02-08", "2026-02-14", gt2, gt2,
            lessons_for_next_week=["lesson A", "lesson C"],  # A is duplicate
        )
        lessons = ledger.get_lessons()
        assert len(lessons) == 3  # A, C, B (no dup)

    def test_get_latest(self, findings_dir: Path):
        ledger = LearningLedger(findings_dir)
        assert ledger.get_latest() is None
        gt = {"bot1": GroundTruthSnapshot(
            snapshot_date="2026-02-07", bot_id="bot1",
        )}
        ledger.record_week("2026-02-01", "2026-02-07", gt, gt)
        latest = ledger.get_latest()
        assert latest is not None
        assert latest.week_start == "2026-02-01"

    def test_jsonl_persistence(self, findings_dir: Path):
        ledger = LearningLedger(findings_dir)
        gt = {"bot1": GroundTruthSnapshot(
            snapshot_date="2026-02-07", bot_id="bot1", composite_score=0.6,
        )}
        ledger.record_week("2026-02-01", "2026-02-07", gt, gt)

        # New ledger instance should load persisted data
        ledger2 = LearningLedger(findings_dir)
        latest = ledger2.get_latest()
        assert latest is not None
        assert latest.week_start == "2026-02-01"

    def test_negative_improvement(self, findings_dir: Path):
        ledger = LearningLedger(findings_dir)
        gt_start = {"bot1": GroundTruthSnapshot(
            snapshot_date="2026-02-01", bot_id="bot1", composite_score=0.6,
        )}
        gt_end = {"bot1": GroundTruthSnapshot(
            snapshot_date="2026-02-07", bot_id="bot1", composite_score=0.4,
        )}
        entry = ledger.record_week("2026-02-01", "2026-02-07", gt_start, gt_end)
        assert entry.net_improvement is False
        assert entry.composite_delta["bot1"] < 0

    def test_what_worked_what_failed(self, findings_dir: Path):
        ledger = LearningLedger(findings_dir)
        gt = {"bot1": GroundTruthSnapshot(
            snapshot_date="2026-02-07", bot_id="bot1",
        )}
        entry = ledger.record_week(
            "2026-02-01", "2026-02-07", gt, gt,
            what_worked=["Exit timing adjustments improved Sharpe"],
            what_failed=["Filter threshold change had no effect"],
        )
        assert len(entry.what_worked) == 1
        assert len(entry.what_failed) == 1


# ── Schema models ──

class TestLearningLedgerSchemas:
    def test_synthesis_item(self):
        item = SynthesisItem(
            suggestion_id="abc123", bot_id="bot1", category="exit_timing",
            title="Tighten trailing stop", outcome_verdict="positive",
            ground_truth_delta=0.05, mechanism="Reduced tail losses",
        )
        assert item.ground_truth_delta == 0.05

    def test_discard_item(self):
        item = DiscardItem(
            bot_id="bot1", category="filter_threshold",
            failure_count=4, reason="4 attempts, 0 improvements",
        )
        assert item.failure_count == 4

    def test_retrospective_synthesis(self):
        synthesis = RetrospectiveSynthesis(
            week_start="2026-02-01", week_end="2026-02-07",
            what_worked=[SynthesisItem(suggestion_id="s1", title="Good")],
            discard=[DiscardItem(category="stop_loss", failure_count=3, reason="fails")],
            lessons=["Don't change stops during trending markets"],
        )
        assert len(synthesis.what_worked) == 1
        assert len(synthesis.discard) == 1


# ── Context injection ──

class TestContextInjection:
    def test_ground_truth_trend_in_base_package(self, memory_dir: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        # Write a ledger entry
        findings = memory_dir / "findings"
        ledger = LearningLedger(findings)
        gt = {"bot1": GroundTruthSnapshot(
            snapshot_date="2026-02-07", bot_id="bot1", composite_score=0.6,
        )}
        ledger.record_week(
            "2026-02-01", "2026-02-07", gt, gt,
            lessons_for_next_week=["Lesson 1"],
        )

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "ground_truth_trend" in pkg.data
        assert "composite_trend" in pkg.data["ground_truth_trend"]
        assert "recent_lessons" in pkg.data["ground_truth_trend"]

    def test_ground_truth_trend_empty_when_no_data(self, memory_dir: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "ground_truth_trend" not in pkg.data

    def test_retrospective_synthesis_in_base_package(self, memory_dir: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        # Write synthesis entry
        synthesis_path = memory_dir / "findings" / "retrospective_synthesis.jsonl"
        synthesis = RetrospectiveSynthesis(
            week_start="2026-02-01", week_end="2026-02-07",
            lessons=["lesson1"],
        )
        synthesis_path.write_text(synthesis.model_dump_json() + "\n")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "last_week_synthesis" in pkg.data
        assert pkg.data["last_week_synthesis"]["week_start"] == "2026-02-01"

    def test_strategy_ideas_in_base_package(self, memory_dir: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        # Use a recent timestamp so the 90-day temporal window in
        # load_strategy_ideas does not filter the entry out as the suite ages.
        recent_ts = datetime.now(timezone.utc).isoformat()
        ideas_path = memory_dir / "findings" / "strategy_ideas.jsonl"
        idea = {
            "idea_id": "idea1", "title": "Regime-Filtered ORB",
            "status": "proposed",
            "timestamp": recent_ts,
        }
        ideas_path.write_text(json.dumps(idea) + "\n")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "strategy_ideas" in pkg.data
        assert len(pkg.data["strategy_ideas"]) == 1

    def test_retired_strategy_ideas_excluded(self, memory_dir: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        recent_ts = datetime.now(timezone.utc).isoformat()
        ideas_path = memory_dir / "findings" / "strategy_ideas.jsonl"
        active = {"idea_id": "a1", "title": "Active", "status": "proposed",
                  "timestamp": recent_ts}
        retired = {"idea_id": "r1", "title": "Retired", "status": "retired",
                   "timestamp": recent_ts}
        ideas_path.write_text(
            json.dumps(active) + "\n" + json.dumps(retired) + "\n"
        )

        ctx = ContextBuilder(memory_dir)
        ideas = ctx.load_strategy_ideas()
        assert len(ideas) == 1
        assert ideas[0]["idea_id"] == "a1"
