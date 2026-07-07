# tests/test_bug_complexity_classifier.py
"""Tests for bug complexity classifier."""
from trading_assistant.schemas.bug_triage import (
    BugComplexity,
    BugSeverity,
    ErrorCategory,
    ErrorEvent,
    TriageOutcome,
)
from trading_assistant.skills.bug_complexity_classifier import BugComplexityClassifier


class TestObviousFix:
    def test_import_error_is_obvious_fix_at_high(self):
        """HIGH + OBVIOUS_FIX → KNOWN_FIX (complexity-driven routing only at HIGH)."""
        e = ErrorEvent(
            bot_id="bot1", error_type="ImportError",
            message="No module named 'requests'",
            stack_trace="...",
            source_file="requirements.txt",
        )
        result = BugComplexityClassifier().classify(e, BugSeverity.HIGH, ErrorCategory.DEPENDENCY)
        assert result.complexity == BugComplexity.OBVIOUS_FIX
        assert result.outcome == TriageOutcome.KNOWN_FIX

    def test_config_error_is_obvious_fix_at_high(self):
        """HIGH + OBVIOUS_FIX → KNOWN_FIX."""
        e = ErrorEvent(
            bot_id="bot1", error_type="ConfigError",
            message="missing key 'api_key' in config.yaml",
            stack_trace="...",
        )
        result = BugComplexityClassifier().classify(e, BugSeverity.HIGH, ErrorCategory.CONFIG_ERROR)
        assert result.complexity == BugComplexity.OBVIOUS_FIX
        assert result.outcome == TriageOutcome.KNOWN_FIX

    def test_deprecation_warning_is_obvious_fix(self):
        e = ErrorEvent(
            bot_id="bot1", error_type="DeprecationWarning",
            message="use new_func instead of old_func",
            stack_trace="...",
        )
        result = BugComplexityClassifier().classify(e, BugSeverity.LOW, ErrorCategory.DEPRECATION)
        assert result.complexity == BugComplexity.OBVIOUS_FIX
        assert result.outcome == TriageOutcome.QUEUED_FOR_WEEKLY


class TestSingleFunction:
    def test_runtime_error_single_file_at_high(self):
        """HIGH + SINGLE_FUNCTION → NEEDS_INVESTIGATION."""
        e = ErrorEvent(
            bot_id="bot1", error_type="RuntimeError",
            message="division by zero",
            stack_trace="Traceback:\n  File \"calc.py\", line 10\n    return x / y\nRuntimeError: division by zero",
            source_file="calc.py",
            source_line=10,
        )
        result = BugComplexityClassifier().classify(e, BugSeverity.HIGH, ErrorCategory.UNKNOWN)
        assert result.complexity == BugComplexity.SINGLE_FUNCTION
        assert result.outcome == TriageOutcome.NEEDS_INVESTIGATION

    def test_api_error_single_endpoint(self):
        e = ErrorEvent(
            bot_id="bot1", error_type="APIError",
            message="exchange returned 429",
            stack_trace="Traceback:\n  File \"connector.py\", line 55\nAPIError: 429",
            source_file="connector.py",
            source_line=55,
        )
        result = BugComplexityClassifier().classify(e, BugSeverity.HIGH, ErrorCategory.API_ERROR)
        assert result.complexity == BugComplexity.SINGLE_FUNCTION
        assert result.outcome == TriageOutcome.NEEDS_INVESTIGATION


class TestStateDependentOrMultiFile:
    def test_stuck_position_is_state_dependent_at_high(self):
        """HIGH + STATE_DEPENDENT → NEEDS_HUMAN."""
        e = ErrorEvent(
            bot_id="bot1", error_type="TimeoutError",
            message="position stuck",
            stack_trace="...",
        )
        result = BugComplexityClassifier().classify(e, BugSeverity.HIGH, ErrorCategory.STUCK_POSITION)
        assert result.complexity == BugComplexity.STATE_DEPENDENT
        assert result.outcome == TriageOutcome.NEEDS_HUMAN

    def test_connection_lost_is_needs_human_at_high(self):
        """HIGH + STATE_DEPENDENT → NEEDS_HUMAN."""
        e = ErrorEvent(
            bot_id="bot1", error_type="ConnectionError",
            message="connection lost",
            stack_trace="...",
        )
        result = BugComplexityClassifier().classify(e, BugSeverity.HIGH, ErrorCategory.CONNECTION_LOST)
        assert result.complexity == BugComplexity.STATE_DEPENDENT
        assert result.outcome == TriageOutcome.NEEDS_HUMAN


class TestSeverityBasedRouting:
    def test_critical_always_alerted(self):
        """CRITICAL errors -> ALERTED regardless of complexity."""
        e = ErrorEvent(
            bot_id="bot1", error_type="SystemExit",
            message="crash", stack_trace="...",
        )
        result = BugComplexityClassifier().classify(e, BugSeverity.CRITICAL, ErrorCategory.CRASH)
        assert result.outcome == TriageOutcome.ALERTED

    def test_low_queued_for_weekly(self):
        """LOW errors -> QUEUED_FOR_WEEKLY."""
        e = ErrorEvent(
            bot_id="bot1", error_type="UserWarning",
            message="some warning", stack_trace="...",
        )
        result = BugComplexityClassifier().classify(e, BugSeverity.LOW, ErrorCategory.WARNING)
        assert result.outcome == TriageOutcome.QUEUED_FOR_WEEKLY

    def test_medium_queued_for_daily(self):
        """MEDIUM obvious_fix -> QUEUED_FOR_DAILY (not auto-fix, just queue)."""
        e = ErrorEvent(
            bot_id="bot1", error_type="RuntimeError",
            message="some error", stack_trace="...",
        )
        result = BugComplexityClassifier().classify(e, BugSeverity.MEDIUM, ErrorCategory.UNKNOWN)
        assert result.outcome == TriageOutcome.QUEUED_FOR_DAILY


class TestComplexityResult:
    def test_result_has_complexity_and_outcome(self):
        e = ErrorEvent(
            bot_id="bot1", error_type="ImportError",
            message="no module foo", stack_trace="...",
        )
        result = BugComplexityClassifier().classify(e, BugSeverity.HIGH, ErrorCategory.DEPENDENCY)
        assert hasattr(result, "complexity")
        assert hasattr(result, "outcome")
        assert hasattr(result, "rationale")
