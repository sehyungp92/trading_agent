# tests/test_triage_runner.py
"""Tests for triage runner — main pipeline orchestrating severity → complexity → route."""
from pathlib import Path

from trading_assistant.schemas.bug_triage import (
    BugSeverity,
    BugComplexity,
    ErrorEvent,
    TriageOutcome,
    TriageResult,
)
from trading_assistant.skills.run_bug_triage import TriageRunner


class TestTriageRunnerCritical:
    def test_critical_crash_returns_alerted(self, tmp_path: Path):
        runner = TriageRunner(
            source_root=tmp_path,
            failure_log_path=tmp_path / "failure-log.jsonl",
        )
        event = ErrorEvent(
            bot_id="bot1", error_type="SystemExit",
            message="process crashed", stack_trace="...",
            context={"crash": True},
        )
        result = runner.triage(event)
        assert result.severity == BugSeverity.CRITICAL
        assert result.outcome == TriageOutcome.ALERTED

    def test_critical_connection_lost_returns_alerted(self, tmp_path: Path):
        runner = TriageRunner(
            source_root=tmp_path,
            failure_log_path=tmp_path / "failure-log.jsonl",
        )
        event = ErrorEvent(
            bot_id="bot1", error_type="ConnectionError",
            message="connection lost to exchange",
            stack_trace="...",
        )
        result = runner.triage(event)
        assert result.severity == BugSeverity.CRITICAL
        assert result.outcome == TriageOutcome.ALERTED


class TestTriageRunnerHigh:
    def test_high_obvious_fix_returns_known_fix(self, tmp_path: Path):
        runner = TriageRunner(
            source_root=tmp_path,
            failure_log_path=tmp_path / "failure-log.jsonl",
        )
        event = ErrorEvent(
            bot_id="bot1", error_type="ImportError",
            message="No module named requests",
            stack_trace="...",
        )
        # Manually push severity to HIGH (e.g., via repeated errors)
        result = runner.triage(event, severity_override=BugSeverity.HIGH)
        assert result.severity == BugSeverity.HIGH
        assert result.outcome == TriageOutcome.KNOWN_FIX

    def test_high_single_function_returns_needs_investigation(self, tmp_path: Path):
        runner = TriageRunner(
            source_root=tmp_path,
            failure_log_path=tmp_path / "failure-log.jsonl",
        )
        event = ErrorEvent(
            bot_id="bot1", error_type="APIError",
            message="exchange returned 500",
            stack_trace='Traceback:\n  File "connector.py", line 55\nAPIError: 500',
            source_file="connector.py",
            source_line=55,
        )
        result = runner.triage(event)
        assert result.severity == BugSeverity.HIGH
        assert result.outcome == TriageOutcome.NEEDS_INVESTIGATION


class TestTriageRunnerMediumLow:
    def test_medium_queued_for_daily(self, tmp_path: Path):
        runner = TriageRunner(
            source_root=tmp_path,
            failure_log_path=tmp_path / "failure-log.jsonl",
        )
        event = ErrorEvent(
            bot_id="bot1", error_type="RuntimeError",
            message="division by zero",
            stack_trace="Traceback:\n  File calc.py:10\nRuntimeError: division by zero",
        )
        result = runner.triage(event)
        assert result.severity == BugSeverity.MEDIUM
        assert result.outcome == TriageOutcome.QUEUED_FOR_DAILY

    def test_low_queued_for_weekly(self, tmp_path: Path):
        runner = TriageRunner(
            source_root=tmp_path,
            failure_log_path=tmp_path / "failure-log.jsonl",
        )
        event = ErrorEvent(
            bot_id="bot1", error_type="DeprecationWarning",
            message="module X is deprecated",
            stack_trace="...",
        )
        result = runner.triage(event)
        assert result.severity == BugSeverity.LOW
        assert result.outcome == TriageOutcome.QUEUED_FOR_WEEKLY


class TestTriageRunnerWithRepeatedErrors:
    def test_repeated_errors_promote_to_high(self, tmp_path: Path):
        runner = TriageRunner(
            source_root=tmp_path,
            failure_log_path=tmp_path / "failure-log.jsonl",
            repeated_error_threshold=3,
        )
        event = ErrorEvent(
            bot_id="bot1", error_type="RuntimeError",
            message="some recurring error",
            stack_trace="...",
        )
        # Record multiple errors to trigger promotion
        for _ in range(3):
            runner.record_error(event)

        result = runner.triage(event)
        assert result.severity == BugSeverity.HIGH


class TestTriageRunnerRecordsToFailureLog:
    def test_triage_records_to_log(self, tmp_path: Path):
        log_path = tmp_path / "failure-log.jsonl"
        runner = TriageRunner(
            source_root=tmp_path,
            failure_log_path=log_path,
        )
        event = ErrorEvent(
            bot_id="bot1", error_type="RuntimeError",
            message="fail", stack_trace="...",
        )
        runner.triage(event)
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1


class TestTriageRunnerPastRejections:
    def test_includes_past_rejections_in_result(self, tmp_path: Path):
        log_path = tmp_path / "failure-log.jsonl"
        runner = TriageRunner(
            source_root=tmp_path,
            failure_log_path=log_path,
        )
        # Seed a past rejection
        from trading_assistant.skills.failure_log import FailureLog
        fl = FailureLog(log_path)
        fl.record_rejection(
            TriageResult(
                error_event=ErrorEvent(
                    bot_id="bot1", error_type="ImportError",
                    message="no foo", stack_trace="s",
                ),
                severity=BugSeverity.HIGH,
                complexity=BugComplexity.OBVIOUS_FIX,
                outcome=TriageOutcome.KNOWN_FIX,
            ),
            reason="wrong version",
        )

        event = ErrorEvent(
            bot_id="bot1", error_type="ImportError",
            message="no foo", stack_trace="s",
        )
        result = runner.triage(event, severity_override=BugSeverity.HIGH)
        assert len(result.past_rejections) >= 1
        assert any("wrong version" in r for r in result.past_rejections)
