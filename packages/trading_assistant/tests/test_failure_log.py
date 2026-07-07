# tests/test_failure_log.py
"""Tests for failure log — .assistant/failure-log.jsonl persistence."""
import json
from pathlib import Path

from trading_assistant.schemas.bug_triage import BugSeverity, BugComplexity, TriageOutcome, ErrorEvent, TriageResult
from trading_assistant.skills.failure_log import FailureLog, FailureEntry


class TestFailureEntry:
    def test_creates_from_triage_result(self):
        tr = TriageResult(
            error_event=ErrorEvent(
                bot_id="bot1", error_type="E", message="m", stack_trace="s",
            ),
            severity=BugSeverity.HIGH,
            complexity=BugComplexity.OBVIOUS_FIX,
            outcome=TriageOutcome.KNOWN_FIX,
            suggested_fix="fix X",
        )
        entry = FailureEntry.from_triage_result(tr, rejection_reason="wrong fix")
        assert entry.bot_id == "bot1"
        assert entry.error_type == "E"
        assert entry.outcome == TriageOutcome.KNOWN_FIX
        assert entry.rejection_reason == "wrong fix"
        assert entry.issue_url == ""


class TestFailureLogWrite:
    def test_append_creates_file(self, tmp_path: Path):
        log_path = tmp_path / ".assistant" / "failure-log.jsonl"
        log = FailureLog(log_path)
        tr = TriageResult(
            error_event=ErrorEvent(
                bot_id="bot1", error_type="E", message="m", stack_trace="s",
            ),
            severity=BugSeverity.HIGH,
            complexity=BugComplexity.OBVIOUS_FIX,
            outcome=TriageOutcome.KNOWN_FIX,
        )
        log.record_triage(tr)
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["bot_id"] == "bot1"

    def test_append_multiple(self, tmp_path: Path):
        log_path = tmp_path / "failure-log.jsonl"
        log = FailureLog(log_path)
        for i in range(3):
            tr = TriageResult(
                error_event=ErrorEvent(
                    bot_id=f"bot{i}", error_type="E", message="m", stack_trace="s",
                ),
                severity=BugSeverity.HIGH,
                complexity=BugComplexity.OBVIOUS_FIX,
                outcome=TriageOutcome.KNOWN_FIX,
            )
            log.record_triage(tr)
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_record_rejection(self, tmp_path: Path):
        log_path = tmp_path / "failure-log.jsonl"
        log = FailureLog(log_path)
        tr = TriageResult(
            error_event=ErrorEvent(
                bot_id="bot1", error_type="ImportError", message="no foo", stack_trace="s",
            ),
            severity=BugSeverity.HIGH,
            complexity=BugComplexity.OBVIOUS_FIX,
            outcome=TriageOutcome.KNOWN_FIX,
            pr_url="https://github.com/user/repo/pull/42",
        )
        log.record_rejection(tr, reason="wrong version pinned")
        lines = log_path.read_text().strip().splitlines()
        data = json.loads(lines[0])
        assert data["rejection_reason"] == "wrong version pinned"
        assert data["pr_url"] == "https://github.com/user/repo/pull/42"

    def test_records_issue_url(self, tmp_path: Path):
        log_path = tmp_path / "failure-log.jsonl"
        log = FailureLog(log_path)
        tr = TriageResult(
            error_event=ErrorEvent(
                bot_id="bot1", error_type="RuntimeError", message="investigate", stack_trace="s",
            ),
            severity=BugSeverity.HIGH,
            complexity=BugComplexity.MULTI_FILE,
            outcome=TriageOutcome.NEEDS_INVESTIGATION,
            github_issue_url="https://github.com/user/repo/issues/7",
        )
        log.record_triage(tr)
        data = json.loads(log_path.read_text().strip().splitlines()[0])
        assert data["issue_url"] == "https://github.com/user/repo/issues/7"


class TestFailureLogRead:
    def test_get_past_rejections(self, tmp_path: Path):
        log_path = tmp_path / "failure-log.jsonl"
        log = FailureLog(log_path)

        for i in range(3):
            tr = TriageResult(
                error_event=ErrorEvent(
                    bot_id="bot1", error_type="ImportError", message="no foo", stack_trace="s",
                ),
                severity=BugSeverity.HIGH,
                complexity=BugComplexity.OBVIOUS_FIX,
                outcome=TriageOutcome.KNOWN_FIX,
            )
            log.record_rejection(tr, reason=f"rejection {i}")

        rejections = log.get_past_rejections(error_type="ImportError", limit=10)
        assert len(rejections) == 3
        assert all(r.rejection_reason.startswith("rejection") for r in rejections)

    def test_get_past_rejections_filters_by_error_type(self, tmp_path: Path):
        log_path = tmp_path / "failure-log.jsonl"
        log = FailureLog(log_path)

        for etype in ["ImportError", "RuntimeError", "ImportError"]:
            tr = TriageResult(
                error_event=ErrorEvent(
                    bot_id="bot1", error_type=etype, message="m", stack_trace="s",
                ),
                severity=BugSeverity.HIGH,
                complexity=BugComplexity.OBVIOUS_FIX,
                outcome=TriageOutcome.KNOWN_FIX,
            )
            log.record_rejection(tr, reason=f"rej-{etype}")

        import_rejections = log.get_past_rejections(error_type="ImportError")
        assert len(import_rejections) == 2

    def test_get_past_rejections_empty_file(self, tmp_path: Path):
        log_path = tmp_path / "failure-log.jsonl"
        log = FailureLog(log_path)
        assert log.get_past_rejections(error_type="E") == []

    def test_get_past_rejections_respects_limit(self, tmp_path: Path):
        log_path = tmp_path / "failure-log.jsonl"
        log = FailureLog(log_path)

        for i in range(10):
            tr = TriageResult(
                error_event=ErrorEvent(
                    bot_id="bot1", error_type="E", message="m", stack_trace="s",
                ),
                severity=BugSeverity.HIGH,
                complexity=BugComplexity.OBVIOUS_FIX,
                outcome=TriageOutcome.KNOWN_FIX,
            )
            log.record_rejection(tr, reason=f"rej-{i}")

        rejections = log.get_past_rejections(error_type="E", limit=5)
        assert len(rejections) == 5
