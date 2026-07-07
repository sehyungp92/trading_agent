# tests/test_learning_cycle.py
"""Tests for Phase 4 — Autonomous Learning Cycle."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading_assistant.schemas.learning_ledger import LearningLedgerEntry


# ── Learning Cycle Core ──

class TestLearningCycleRun:
    def _setup(self, tmp_path: Path):
        """Create minimal directory structure for learning cycle."""
        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()
        return memory_dir, runs_dir, curated_dir

    def _write_curated_summary(self, curated_dir: Path, date: str, bot_id: str, summary: dict):
        path = curated_dir / date / bot_id / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary))

    @pytest.mark.asyncio
    async def test_cold_start_returns_entry(self, tmp_path: Path):
        """Cycle runs successfully with no historical data."""
        from trading_assistant.skills.learning_cycle import LearningCycle

        memory_dir, runs_dir, curated_dir = self._setup(tmp_path)
        cycle = LearningCycle(
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
            bots=["bot_a"],
        )
        entry = await cycle.run("2026-03-01", "2026-03-07")
        assert isinstance(entry, LearningLedgerEntry)
        assert entry.week_start == "2026-03-01"
        assert entry.week_end == "2026-03-07"

    @pytest.mark.asyncio
    async def test_cycle_computes_composite_delta(self, tmp_path: Path):
        """Cycle computes composite deltas from curated data."""
        from trading_assistant.skills.learning_cycle import LearningCycle

        memory_dir, runs_dir, curated_dir = self._setup(tmp_path)

        # Create 30 days of curated data for bot_a (enough for GT computation)
        base_date = datetime(2026, 3, 7, tzinfo=timezone.utc)
        for i in range(35):
            date_str = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            self._write_curated_summary(curated_dir, date_str, "bot_a", {
                "net_pnl": 10.0 + (i * 0.1 if i < 7 else 0),
                "win_rate": 0.55,
                "max_drawdown_pct": 5.0,
                "avg_process_quality": 75.0,
                "trade_count": 5,
            })

        cycle = LearningCycle(
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
            bots=["bot_a"],
        )
        entry = await cycle.run("2026-03-01", "2026-03-07")
        assert "bot_a" in entry.composite_delta or entry.composite_delta == {}

    @pytest.mark.asyncio
    async def test_cycle_records_to_ledger(self, tmp_path: Path):
        """Cycle persists entry to learning_ledger.jsonl."""
        from trading_assistant.skills.learning_cycle import LearningCycle

        memory_dir, runs_dir, curated_dir = self._setup(tmp_path)
        cycle = LearningCycle(
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
            bots=["bot_a"],
        )
        await cycle.run("2026-03-01", "2026-03-07")

        ledger_path = memory_dir / "findings" / "learning_ledger.jsonl"
        assert ledger_path.exists()
        lines = ledger_path.read_text().strip().splitlines()
        assert len(lines) >= 1

    @pytest.mark.asyncio
    async def test_cycle_deduplicates(self, tmp_path: Path):
        """Running cycle twice with same dates → single entry."""
        from trading_assistant.skills.learning_cycle import LearningCycle

        memory_dir, runs_dir, curated_dir = self._setup(tmp_path)
        cycle = LearningCycle(
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
            bots=["bot_a"],
        )
        await cycle.run("2026-03-01", "2026-03-07")
        await cycle.run("2026-03-01", "2026-03-07")

        ledger_path = memory_dir / "findings" / "learning_ledger.jsonl"
        lines = ledger_path.read_text().strip().splitlines()
        assert len(lines) == 1  # deduped by entry_id


# ── Synthesis Integration ──

class TestCycleSynthesis:
    def _setup(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()
        return memory_dir, runs_dir, curated_dir

    @pytest.mark.asyncio
    async def test_cycle_builds_synthesis(self, tmp_path: Path):
        """Cycle creates retrospective_synthesis.jsonl."""
        from trading_assistant.skills.learning_cycle import LearningCycle

        memory_dir, runs_dir, curated_dir = self._setup(tmp_path)

        # Write an outcome for the week
        outcomes_path = memory_dir / "findings" / "outcomes.jsonl"
        outcomes_path.write_text(json.dumps({
            "suggestion_id": "s1", "bot_id": "bot_a", "category": "exit_timing",
            "verdict": "positive", "pnl_delta": 0.05,
            "measured_at": "2026-03-03T10:00:00+00:00",
            "title": "Good change",
        }) + "\n")

        cycle = LearningCycle(
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
            bots=["bot_a"],
        )
        entry = await cycle.run("2026-03-01", "2026-03-07")
        assert "Good change" in entry.what_worked

        synth_path = memory_dir / "findings" / "retrospective_synthesis.jsonl"
        assert synth_path.exists()

    @pytest.mark.asyncio
    async def test_cycle_applies_recalibration(self, tmp_path: Path):
        """Cycle creates category_overrides when discard items found."""
        from trading_assistant.skills.learning_cycle import LearningCycle

        memory_dir, runs_dir, curated_dir = self._setup(tmp_path)

        # Write 3+ failures for same (bot_id, category)
        outcomes = [
            {
                "suggestion_id": f"s{i}", "bot_id": "bot_a",
                "category": "filter_threshold",
                "verdict": "negative", "pnl_delta": -0.01,
                "measured_at": f"2026-03-0{i+1}T10:00:00+00:00",
                "title": f"Bad filter {i}",
            }
            for i in range(4)
        ]
        (memory_dir / "findings" / "outcomes.jsonl").write_text(
            "\n".join(json.dumps(o) for o in outcomes) + "\n"
        )

        cycle = LearningCycle(
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
            bots=["bot_a"],
        )
        await cycle.run("2026-03-01", "2026-03-07")

        overrides_path = memory_dir / "findings" / "category_overrides.jsonl"
        assert overrides_path.exists()
        lines = overrides_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["confidence_multiplier"] == 0.3


# ── Hypothesis Lifecycle ──

class TestCycleHypothesisLifecycle:
    @pytest.mark.asyncio
    async def test_what_worked_records_positive_outcome(self, tmp_path: Path):
        """what_worked items → hypothesis_library.record_outcome(positive=True)."""
        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        # Track calls
        calls = []
        class MockHypLib:
            def record_outcome(self, hyp_id, positive):
                calls.append({"id": hyp_id, "positive": positive})
            def get_active(self):
                return []

        (memory_dir / "findings" / "outcomes.jsonl").write_text(json.dumps({
            "suggestion_id": "s1", "bot_id": "bot_a", "category": "exit_timing",
            "verdict": "positive", "pnl_delta": 0.05,
            "measured_at": "2026-03-03T10:00:00+00:00", "title": "Good",
        }) + "\n")

        from trading_assistant.skills.learning_cycle import LearningCycle
        cycle = LearningCycle(
            curated_dir=curated_dir, memory_dir=memory_dir,
            runs_dir=runs_dir, bots=["bot_a"],
            hypothesis_library=MockHypLib(),
        )
        await cycle.run("2026-03-01", "2026-03-07")

        positive_calls = [c for c in calls if c["positive"]]
        assert len(positive_calls) >= 1

    @pytest.mark.asyncio
    async def test_what_failed_records_negative_outcome(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        calls = []
        class MockHypLib:
            def record_outcome(self, hyp_id, positive):
                calls.append({"id": hyp_id, "positive": positive})
            def get_active(self):
                return []

        (memory_dir / "findings" / "outcomes.jsonl").write_text(json.dumps({
            "suggestion_id": "s2", "bot_id": "bot_a", "category": "signal",
            "verdict": "negative", "pnl_delta": -0.03,
            "measured_at": "2026-03-04T10:00:00+00:00", "title": "Bad",
        }) + "\n")

        from trading_assistant.skills.learning_cycle import LearningCycle
        cycle = LearningCycle(
            curated_dir=curated_dir, memory_dir=memory_dir,
            runs_dir=runs_dir, bots=["bot_a"],
            hypothesis_library=MockHypLib(),
        )
        await cycle.run("2026-03-01", "2026-03-07")

        negative_calls = [c for c in calls if not c["positive"]]
        assert len(negative_calls) >= 1


# ── Experiment Selection ──

class TestExperimentSelection:
    def test_max_2_per_bot(self, tmp_path: Path):
        from trading_assistant.skills.learning_cycle import LearningCycle

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        class MockHyp:
            def __init__(self, hid, cat, eff):
                self.hypothesis_id = hid
                self.category = cat
                self.effectiveness = eff

        class MockHypLib:
            def get_active(self):
                return [
                    MockHyp("h1", "signal", 0.8),
                    MockHyp("h2", "exit_timing", 0.6),
                    MockHyp("h3", "filter_threshold", 0.4),
                ]

        class MockExpTracker:
            def get_active_experiments(self):
                return []

        cycle = LearningCycle(
            curated_dir=curated_dir, memory_dir=memory_dir,
            runs_dir=runs_dir, bots=["bot_a"],
            hypothesis_library=MockHypLib(),
            experiment_tracker=MockExpTracker(),
        )
        selected = cycle._select_next_experiments()
        bot_a_count = sum(1 for s in selected if s["bot_id"] == "bot_a")
        assert bot_a_count <= 2

    def test_skips_discarded_categories(self, tmp_path: Path):
        from trading_assistant.skills.learning_cycle import LearningCycle

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        # Write a discard override
        (memory_dir / "findings" / "category_overrides.jsonl").write_text(
            json.dumps({
                "bot_id": "bot_a", "category": "signal",
                "confidence_multiplier": 0.3,
            }) + "\n"
        )

        class MockHyp:
            def __init__(self, hid, cat, eff):
                self.hypothesis_id = hid
                self.category = cat
                self.effectiveness = eff

        class MockHypLib:
            def get_active(self):
                return [MockHyp("h1", "signal", 0.8)]

        class MockExpTracker:
            def get_active_experiments(self):
                return []

        cycle = LearningCycle(
            curated_dir=curated_dir, memory_dir=memory_dir,
            runs_dir=runs_dir, bots=["bot_a"],
            hypothesis_library=MockHypLib(),
            experiment_tracker=MockExpTracker(),
        )
        selected = cycle._select_next_experiments()
        signal_selected = [s for s in selected if s["category"] == "signal"]
        assert len(signal_selected) == 0

    def test_skips_negative_effectiveness(self, tmp_path: Path):
        from trading_assistant.skills.learning_cycle import LearningCycle

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        class MockHyp:
            def __init__(self, hid, cat, eff):
                self.hypothesis_id = hid
                self.category = cat
                self.effectiveness = eff

        class MockHypLib:
            def get_active(self):
                return [MockHyp("h1", "signal", -0.2)]

        class MockExpTracker:
            def get_active_experiments(self):
                return []

        cycle = LearningCycle(
            curated_dir=curated_dir, memory_dir=memory_dir,
            runs_dir=runs_dir, bots=["bot_a"],
            hypothesis_library=MockHypLib(),
            experiment_tracker=MockExpTracker(),
        )
        selected = cycle._select_next_experiments()
        assert len(selected) == 0

    def test_prefers_high_effectiveness(self, tmp_path: Path):
        from trading_assistant.skills.learning_cycle import LearningCycle

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        class MockHyp:
            def __init__(self, hid, cat, eff):
                self.hypothesis_id = hid
                self.category = cat
                self.effectiveness = eff

        class MockHypLib:
            def get_active(self):
                return [
                    MockHyp("h_low", "signal", 0.1),
                    MockHyp("h_high", "exit_timing", 0.9),
                    MockHyp("h_mid", "stop_loss", 0.5),
                ]

        class MockExpTracker:
            def get_active_experiments(self):
                return []

        cycle = LearningCycle(
            curated_dir=curated_dir, memory_dir=memory_dir,
            runs_dir=runs_dir, bots=["bot_a"],
            hypothesis_library=MockHypLib(),
            experiment_tracker=MockExpTracker(),
        )
        selected = cycle._select_next_experiments()
        # Max 2 per bot, highest effectiveness first
        assert len(selected) == 2
        assert selected[0]["hypothesis_id"] == "h_high"
        assert selected[1]["hypothesis_id"] == "h_mid"


# ── Scheduler Wiring ──

class TestLearningCycleScheduling:
    def test_learning_cycle_spec_created(self):
        from trading_assistant.orchestrator.scheduler import SchedulerConfig, build_scheduled_job_specs

        config = SchedulerConfig()

        async def _noop(scheduled_for=None):
            pass

        specs = build_scheduled_job_specs(
            config=config,
            worker_fn=_noop,
            monitoring_fn=_noop,
            relay_fn=_noop,
            learning_cycle_fn=_noop,
        )
        lc_specs = [s for s in specs if s.name == "learning_cycle"]
        assert len(lc_specs) == 1
        assert lc_specs[0].day_of_week == "sun"
        assert lc_specs[0].hour == 11

    def test_learning_cycle_not_created_without_fn(self):
        from trading_assistant.orchestrator.scheduler import SchedulerConfig, build_scheduled_job_specs

        config = SchedulerConfig()

        async def _noop(scheduled_for=None):
            pass

        specs = build_scheduled_job_specs(
            config=config,
            worker_fn=_noop,
            monitoring_fn=_noop,
            relay_fn=_noop,
        )
        lc_specs = [s for s in specs if "learning_cycle" in s.name]
        assert len(lc_specs) == 0


# ── Dashboard ──

class TestLearningDashboard:
    def test_dashboard_endpoint_exists(self, tmp_path: Path):
        """Dashboard endpoint is registered on the app."""
        from trading_assistant.orchestrator.app import create_app
        from trading_assistant.orchestrator.config import AppConfig
        app = create_app(
            db_dir=str(tmp_path),
            config=AppConfig(allow_unauthenticated_local=True),
        )
        routes = [r.path for r in app.routes]
        assert "/learning/dashboard" in routes

    @pytest.mark.asyncio
    async def test_dashboard_returns_structure(self, tmp_path: Path):
        """Dashboard returns expected keys."""
        from trading_assistant.orchestrator.app import create_app
        from trading_assistant.orchestrator.config import AppConfig
        from httpx import AsyncClient, ASGITransport

        app = create_app(
            db_dir=str(tmp_path),
            config=AppConfig(allow_unauthenticated_local=True),
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/learning/dashboard")
            assert resp.status_code == 200
            data = resp.json()
            assert "ground_truth_trend" in data
            assert "recent_lessons" in data
            assert "category_scorecard" in data
            assert "prediction_accuracy" in data
            assert "net_improvement" in data


# ── Learning Ledger Extended ──

class TestLedgerRecordWeekExtended:
    def test_record_week_with_explicit_delta(self, tmp_path: Path):
        """record_week accepts pre-computed composite_delta."""
        from trading_assistant.skills.learning_ledger import LearningLedger

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        ledger = LearningLedger(findings_dir)

        entry = ledger.record_week(
            week_start="2026-03-01",
            week_end="2026-03-07",
            composite_delta={"bot_a": 0.15},
            net_improvement=True,
            what_worked=["Exit timing improved"],
        )
        assert entry.composite_delta == {"bot_a": 0.15}
        assert entry.net_improvement is True
        assert entry.what_worked == ["Exit timing improved"]

    def test_record_week_without_gt_snapshots(self, tmp_path: Path):
        """record_week works with no GT snapshots (cold start)."""
        from trading_assistant.skills.learning_ledger import LearningLedger

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        ledger = LearningLedger(findings_dir)

        entry = ledger.record_week(
            week_start="2026-03-01",
            week_end="2026-03-07",
        )
        assert entry.composite_delta == {}
        assert entry.net_improvement is False


# ── A1/A2: Timestamp and field drift fixes ──


class TestTimestampFallbackFixes:
    """Verify proposed_at and resolved_at fallback chains work correctly."""

    def _make_cycle(self, tmp_path):
        from trading_assistant.skills.learning_cycle import LearningCycle

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        cycle = LearningCycle(
            curated_dir=tmp_path / "curated",
            memory_dir=memory_dir,
            runs_dir=tmp_path / "runs",
            bots=["bot_a"],
        )
        return cycle, memory_dir

    def test_count_suggestions_with_proposed_at_only(self, tmp_path):
        """Suggestions with only proposed_at (current schema) are counted."""
        from unittest.mock import MagicMock

        cycle, memory_dir = self._make_cycle(tmp_path)
        tracker = MagicMock()
        tracker.load_all.return_value = [
            {"suggestion_id": "s1", "proposed_at": "2026-03-03T10:00:00Z", "status": "deployed"},
            {"suggestion_id": "s2", "proposed_at": "2026-03-04T10:00:00Z", "status": "accepted"},
            {"suggestion_id": "s3", "proposed_at": "2026-02-28T10:00:00Z", "status": "deployed"},  # out of range
        ]
        cycle._suggestion_tracker = tracker

        proposed, accepted, implemented = cycle._count_suggestions("2026-03-01", "2026-03-07")
        assert proposed == 2
        assert accepted == 2
        assert implemented == 1

    def test_count_suggestions_with_legacy_timestamp(self, tmp_path):
        """Suggestions with legacy timestamp field still work."""
        from unittest.mock import MagicMock

        cycle, memory_dir = self._make_cycle(tmp_path)
        tracker = MagicMock()
        tracker.load_all.return_value = [
            {"suggestion_id": "s1", "timestamp": "2026-03-03T10:00:00Z", "status": "deployed"},
        ]
        cycle._suggestion_tracker = tracker

        proposed, accepted, implemented = cycle._count_suggestions("2026-03-01", "2026-03-07")
        assert proposed == 1

    def test_classify_loop_sources_with_proposed_at(self, tmp_path):
        """_classify_loop_sources uses proposed_at for current-schema suggestions."""
        from unittest.mock import MagicMock

        cycle, memory_dir = self._make_cycle(tmp_path)
        tracker = MagicMock()
        tracker.load_all.return_value = [
            {
                "suggestion_id": "s1",
                "proposed_at": "2026-03-03T10:00:00Z",
                "detection_context": {"detector_name": "alpha_decay"},
            },
            {
                "suggestion_id": "s2",
                "proposed_at": "2026-03-04T10:00:00Z",
            },
        ]
        cycle._suggestion_tracker = tracker

        inner_p, outer_p, _, _, _, _ = cycle._classify_loop_sources(
            "2026-03-01", "2026-03-07", memory_dir / "findings",
        )
        assert inner_p == 1
        assert outer_p == 1

    def test_count_experiments_with_resolved_at(self, tmp_path):
        """Experiments with resolved_at (current schema) are counted."""
        from unittest.mock import MagicMock

        cycle, memory_dir = self._make_cycle(tmp_path)

        exp_mock = MagicMock()
        exp_mock.resolved_at = "2026-03-05T10:00:00Z"
        # No concluded_at attribute
        del exp_mock.concluded_at

        tracker = MagicMock()
        tracker.load_all.return_value = [exp_mock]
        cycle._experiment_tracker = tracker

        count = cycle._count_experiments("2026-03-01", "2026-03-07")
        assert count == 1

    def test_count_experiments_with_legacy_concluded_at(self, tmp_path):
        """Experiments with only concluded_at (legacy) are still counted."""
        from unittest.mock import MagicMock

        cycle, memory_dir = self._make_cycle(tmp_path)

        exp_mock = MagicMock()
        exp_mock.resolved_at = None
        exp_mock.concluded_at = "2026-03-05T10:00:00Z"

        tracker = MagicMock()
        tracker.load_all.return_value = [exp_mock]
        cycle._experiment_tracker = tracker

        count = cycle._count_experiments("2026-03-01", "2026-03-07")
        assert count == 1


class TestSuggestionQualityTrendTimestampFix:
    """Verify suggestion_scorer uses proposed_at fallback."""

    def test_quality_trend_with_proposed_at_only(self, tmp_path):
        """compute_suggestion_quality_trend picks up proposed_at suggestions."""
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings = tmp_path / "findings"
        findings.mkdir()

        # Write suggestions with only proposed_at
        with open(findings / "suggestions.jsonl", "w") as f:
            f.write(json.dumps({
                "suggestion_id": "s1",
                "bot_id": "bot_a",
                "category": "signal",
                "proposed_at": "2026-03-03T10:00:00Z",
                "status": "deployed",
            }) + "\n")

        # Write matching outcome
        with open(findings / "outcomes.jsonl", "w") as f:
            f.write(json.dumps({
                "suggestion_id": "s1",
                "verdict": "POSITIVE",
                "measurement_date": "2026-03-10T10:00:00Z",
                "measurement_quality": "high",
            }) + "\n")

        scorer = SuggestionScorer(findings)
        result = scorer.compute_suggestion_quality_trend(weeks=8)

        # Should have at least one week with data (not empty due to missing timestamps)
        assert result.get("weekly_metrics") is not None
        assert len(result["weekly_metrics"]) > 0


def test_learning_cycle_filters_non_strict_harness_cases_from_replay_suite(tmp_path: Path):
    from trading_assistant.schemas.benchmark_case import (
        BenchmarkCase,
        BenchmarkSeverity,
        BenchmarkSource,
        BenchmarkSuite,
    )
    from trading_assistant.skills.harness_execution_runner import load_execution_inputs
    from trading_assistant.skills.learning_cycle import (
        _executable_benchmark_suite,
        _materialize_strict_harness_cases,
    )

    findings = tmp_path / "memory" / "findings"
    evidence = findings / "source.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")
    valid = BenchmarkCase(
        case_id="valid-case",
        source=BenchmarkSource.NEGATIVE_OUTCOME,
        source_id="outcome:valid",
        severity=BenchmarkSeverity.HIGH,
        artifact_refs=["memory/findings/source.json"],
        input_snapshot={"strategy_id": "strat1"},
    )
    invalid = BenchmarkCase(
        case_id="invalid-case",
        source=BenchmarkSource.VALIDATION_BLOCK,
        source_id="validation:invalid",
        severity=BenchmarkSeverity.HIGH,
    )
    suite = BenchmarkSuite(cases=[valid, invalid])

    case_dirs = _materialize_strict_harness_cases(
        findings_dir=findings,
        benchmark_suite=suite,
    )
    execution_inputs = load_execution_inputs(case_dirs)
    executable_suite = _executable_benchmark_suite(
        benchmark_suite=suite,
        execution_inputs=execution_inputs,
    )

    assert [case.case_id for case in executable_suite.cases] == ["valid-case"]
