"""Tests for benchmark case schema and compiler."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.schemas.benchmark_case import BenchmarkCase, BenchmarkSeverity, BenchmarkSource
from trading_assistant.skills.benchmark_compiler import BenchmarkCompiler


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _recent_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _old_ts() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()


class TestBenchmarkCaseSchema:
    def test_deterministic_case_id(self):
        id1 = BenchmarkCase.make_case_id(BenchmarkSource.VALIDATION_BLOCK, "test:123")
        id2 = BenchmarkCase.make_case_id(BenchmarkSource.VALIDATION_BLOCK, "test:123")
        assert id1 == id2
        assert len(id1) == 16

    def test_different_source_different_id(self):
        id1 = BenchmarkCase.make_case_id(BenchmarkSource.VALIDATION_BLOCK, "test:123")
        id2 = BenchmarkCase.make_case_id(BenchmarkSource.NEGATIVE_OUTCOME, "test:123")
        assert id1 != id2


class TestEmptyFindings:
    def test_empty_dir_returns_empty_suite(self, tmp_path):
        compiler = BenchmarkCompiler(tmp_path / "findings")
        suite = compiler.compile()
        assert suite.cases == []
        assert suite.source_summary["validation_blocks"] == 0


class TestValidationBlocks:
    def test_high_block_rate_creates_case(self, tmp_path):
        findings = tmp_path / "findings"
        _write_jsonl(findings / "validation_log.jsonl", [{
            "date": "2026-04-01",
            "approved_count": 1,
            "blocked_count": 5,
            "blocked_details": [{"title": "Bad suggestion", "reason": "rejected", "bot_id": "b1"}],
            "timestamp": _recent_ts(),
        }])

        compiler = BenchmarkCompiler(findings)
        suite = compiler.compile()
        assert len(suite.cases) == 1
        assert suite.cases[0].source == BenchmarkSource.VALIDATION_BLOCK
        assert suite.cases[0].severity == BenchmarkSeverity.HIGH

    def test_low_total_skipped(self, tmp_path):
        """Total < 4 → not enough volume."""
        findings = tmp_path / "findings"
        _write_jsonl(findings / "validation_log.jsonl", [{
            "date": "2026-04-01",
            "approved_count": 0,
            "blocked_count": 2,
            "blocked_details": [],
            "timestamp": _recent_ts(),
        }])

        compiler = BenchmarkCompiler(findings)
        suite = compiler.compile()
        assert len(suite.cases) == 0

    def test_low_block_rate_skipped(self, tmp_path):
        """Block rate < 75% → skip."""
        findings = tmp_path / "findings"
        _write_jsonl(findings / "validation_log.jsonl", [{
            "date": "2026-04-01",
            "approved_count": 3,
            "blocked_count": 2,
            "blocked_details": [],
            "timestamp": _recent_ts(),
        }])

        compiler = BenchmarkCompiler(findings)
        suite = compiler.compile()
        assert len(suite.cases) == 0


class TestNegativeOutcomes:
    def test_negative_verdict_creates_critical_case(self, tmp_path):
        findings = tmp_path / "findings"
        _write_jsonl(findings / "suggestions.jsonl", [{
            "suggestion_id": "s123",
            "bot_id": "bot1",
            "category": "parameter",
            "source_report_id": "daily-2026-03-01",
            "detection_context": {
                "source_provider": "claude_max",
                "source_model": "sonnet",
            },
        }])
        run_dir = tmp_path / "runs" / "daily-2026-03-01"
        run_dir.mkdir(parents=True)
        (run_dir / "parsed_analysis.json").write_text("{}", encoding="utf-8")
        _write_jsonl(findings / "outcomes.jsonl", [{
            "suggestion_id": "s123",
            "implemented_date": "2026-03-01",
            "measurement_date": "2026-04-01",
            "pnl_before": 100.0,
            "pnl_after": 50.0,
            "win_rate_before": 0.6,
            "win_rate_after": 0.4,
            "verdict": "negative",
            "measurement_quality": "high",
            "timestamp": _recent_ts(),
        }])

        compiler = BenchmarkCompiler(findings)
        suite = compiler.compile()
        assert len(suite.cases) == 1
        assert suite.cases[0].severity == BenchmarkSeverity.CRITICAL
        assert suite.cases[0].source == BenchmarkSource.NEGATIVE_OUTCOME
        assert suite.cases[0].source_run_id == "daily-2026-03-01"
        assert "runs/daily-2026-03-01/parsed_analysis.json" in suite.cases[0].artifact_refs
        assert "category:parameter" in suite.cases[0].case_tags
        assert suite.cases[0].output_snapshot["parsed_analysis"] == {}

    def test_low_quality_negative_skipped(self, tmp_path):
        findings = tmp_path / "findings"
        _write_jsonl(findings / "outcomes.jsonl", [{
            "suggestion_id": "s123",
            "verdict": "negative",
            "measurement_quality": "low",
            "timestamp": _recent_ts(),
        }])

        compiler = BenchmarkCompiler(findings)
        suite = compiler.compile()
        assert len(suite.cases) == 0

    def test_positive_verdict_skipped(self, tmp_path):
        findings = tmp_path / "findings"
        _write_jsonl(findings / "outcomes.jsonl", [{
            "suggestion_id": "s123",
            "verdict": "positive",
            "measurement_quality": "high",
            "timestamp": _recent_ts(),
        }])

        compiler = BenchmarkCompiler(findings)
        suite = compiler.compile()
        assert len(suite.cases) == 0


class TestCalibrationMisses:
    def test_large_delta_creates_case(self, tmp_path):
        findings = tmp_path / "findings"
        _write_jsonl(findings / "recalibrations.jsonl", [{
            "suggestion_id": "s456",
            "original_confidence": 0.9,
            "revised_confidence": 0.3,
            "category": "parameter",
            "bot_id": "bot1",
            "lessons_learned": ["Overconfident on parameter changes"],
            "timestamp": _recent_ts(),
        }])

        compiler = BenchmarkCompiler(findings)
        suite = compiler.compile()
        assert len(suite.cases) == 1
        assert suite.cases[0].severity == BenchmarkSeverity.MEDIUM
        assert suite.cases[0].source == BenchmarkSource.CALIBRATION_MISS

    def test_small_delta_skipped(self, tmp_path):
        """Delta < 0.3 → skip."""
        findings = tmp_path / "findings"
        _write_jsonl(findings / "recalibrations.jsonl", [{
            "suggestion_id": "s456",
            "original_confidence": 0.5,
            "revised_confidence": 0.6,
            "category": "parameter",
            "timestamp": _recent_ts(),
        }])

        compiler = BenchmarkCompiler(findings)
        suite = compiler.compile()
        assert len(suite.cases) == 0


class TestTransferFailures:
    def test_negative_transfer_creates_case(self, tmp_path):
        findings = tmp_path / "findings"
        _write_jsonl(findings / "transfer_outcomes.jsonl", [{
            "pattern_id": "p789",
            "source_bot": "bot1",
            "target_bot": "bot2",
            "verdict": "negative",
            "pnl_delta_7d": -50.0,
            "win_rate_delta_7d": -0.1,
            "regime_matched": True,
            "measured_at": _recent_ts(),
        }])

        compiler = BenchmarkCompiler(findings)
        suite = compiler.compile()
        assert len(suite.cases) == 1
        assert suite.cases[0].severity == BenchmarkSeverity.HIGH
        assert suite.cases[0].bot_id == "bot2"

    def test_positive_transfer_skipped(self, tmp_path):
        findings = tmp_path / "findings"
        _write_jsonl(findings / "transfer_outcomes.jsonl", [{
            "pattern_id": "p789",
            "source_bot": "bot1",
            "target_bot": "bot2",
            "verdict": "positive",
            "pnl_delta_7d": 50.0,
            "measured_at": _recent_ts(),
        }])

        compiler = BenchmarkCompiler(findings)
        suite = compiler.compile()
        assert len(suite.cases) == 0


class TestDiscardedHarnessExperiments:
    def test_discarded_harness_experiment_creates_negative_case(self, tmp_path):
        findings = tmp_path / "findings"
        _write_jsonl(findings / "discarded_harness_experiments.jsonl", [{
            "experiment_id": "exp123",
            "variant_name": "unsafe_patch",
            "hypothesis": "Unsafe patch should never be promoted",
            "changed_components": ["validator", "prompt_patch"],
            "score_delta": -0.12,
            "aggregate_score": 0.22,
            "baseline_score": 0.34,
            "kept": False,
            "discard_reason": "Rejected by governance hard-fail: approval bypass",
            "governance_regressions": ["approval bypass"],
            "anti_overfitting_assessment": "Does not generalize safely.",
            "future_warning_tags": ["governance_regression"],
            "program_ref": "memory/policies/v1/harness_program.md",
            "recorded_at": _recent_ts(),
        }])

        suite = BenchmarkCompiler(findings).compile()

        assert len(suite.cases) == 1
        case = suite.cases[0]
        assert case.source == BenchmarkSource.DISCARDED_HARNESS_EXPERIMENT
        assert case.severity == BenchmarkSeverity.CRITICAL
        assert "component:validator" in case.case_tags
        assert "governance:approval_bypass" in case.case_tags
        assert case.score_profile["score_delta"] == -0.12


class TestDeduplication:
    def test_same_source_data_no_duplicates(self, tmp_path):
        findings = tmp_path / "findings"
        entry = {
            "suggestion_id": "s123",
            "verdict": "negative",
            "measurement_quality": "high",
            "timestamp": _recent_ts(),
        }
        _write_jsonl(findings / "outcomes.jsonl", [entry, entry])

        compiler = BenchmarkCompiler(findings)
        suite = compiler.compile()
        assert len(suite.cases) == 1


class TestLookbackFilter:
    def test_old_entries_excluded(self, tmp_path):
        findings = tmp_path / "findings"
        _write_jsonl(findings / "outcomes.jsonl", [{
            "suggestion_id": "sold",
            "verdict": "negative",
            "measurement_quality": "high",
            "timestamp": _old_ts(),
        }])

        compiler = BenchmarkCompiler(findings)
        suite = compiler.compile(lookback_days=90)
        assert len(suite.cases) == 0


class TestIncrementalSave:
    def test_compile_and_save_incremental(self, tmp_path):
        findings = tmp_path / "findings"
        _write_jsonl(findings / "outcomes.jsonl", [{
            "suggestion_id": "s1",
            "verdict": "negative",
            "measurement_quality": "high",
            "timestamp": _recent_ts(),
        }])

        compiler = BenchmarkCompiler(findings)
        count1 = compiler.compile_and_save()
        assert count1 == 1

        # Second call with same data — no new cases
        count2 = compiler.compile_and_save()
        assert count2 == 0

        # Verify file has exactly 1 line
        lines = (findings / "benchmark_cases.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1

    def test_compile_and_save_adds_new(self, tmp_path):
        findings = tmp_path / "findings"
        _write_jsonl(findings / "outcomes.jsonl", [{
            "suggestion_id": "s1",
            "verdict": "negative",
            "measurement_quality": "high",
            "timestamp": _recent_ts(),
        }])

        compiler = BenchmarkCompiler(findings)
        compiler.compile_and_save()

        # Add a new entry
        with open(findings / "outcomes.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "suggestion_id": "s2",
                "verdict": "negative",
                "measurement_quality": "medium",
                "timestamp": _recent_ts(),
            }) + "\n")

        count2 = compiler.compile_and_save()
        assert count2 == 1

        lines = (findings / "benchmark_cases.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
