# tests/test_severity_classifier.py
"""Tests for deterministic severity classifier."""
from trading_assistant.schemas.bug_triage import BugSeverity, ErrorCategory, ErrorEvent
from trading_assistant.skills.severity_classifier import SeverityClassifier


class TestCriticalSeverity:
    def test_crash_is_critical(self):
        e = ErrorEvent(
            bot_id="bot1", error_type="SystemExit", message="process crashed",
            stack_trace="...", context={"crash": True},
        )
        result = SeverityClassifier().classify(e)
        assert result.severity == BugSeverity.CRITICAL
        assert result.category == ErrorCategory.CRASH

    def test_stuck_position_is_critical(self):
        e = ErrorEvent(
            bot_id="bot1", error_type="TimeoutError",
            message="position stuck open for 6 hours",
            stack_trace="...", context={"stuck_position": True},
        )
        result = SeverityClassifier().classify(e)
        assert result.severity == BugSeverity.CRITICAL
        assert result.category == ErrorCategory.STUCK_POSITION

    def test_connection_lost_is_critical(self):
        e = ErrorEvent(
            bot_id="bot1", error_type="ConnectionError",
            message="connection lost to exchange",
            stack_trace="...",
        )
        result = SeverityClassifier().classify(e)
        assert result.severity == BugSeverity.CRITICAL
        assert result.category == ErrorCategory.CONNECTION_LOST


class TestHighSeverity:
    def test_unexpected_loss_is_high(self):
        e = ErrorEvent(
            bot_id="bot1", error_type="TradingError",
            message="unexpected loss exceeding threshold",
            stack_trace="...", context={"unexpected_loss": True},
        )
        result = SeverityClassifier().classify(e)
        assert result.severity == BugSeverity.HIGH
        assert result.category == ErrorCategory.UNEXPECTED_LOSS

    def test_api_error_is_high(self):
        e = ErrorEvent(
            bot_id="bot1", error_type="APIError",
            message="exchange API returned 500",
            stack_trace="...",
        )
        result = SeverityClassifier().classify(e)
        assert result.severity == BugSeverity.HIGH
        assert result.category == ErrorCategory.API_ERROR


class TestMediumSeverity:
    def test_config_error_is_medium(self):
        e = ErrorEvent(
            bot_id="bot1", error_type="ConfigError",
            message="invalid config value for trailing_stop_pct",
            stack_trace="...",
        )
        result = SeverityClassifier().classify(e)
        assert result.severity == BugSeverity.MEDIUM
        assert result.category == ErrorCategory.CONFIG_ERROR

    def test_generic_runtime_error_is_medium(self):
        e = ErrorEvent(
            bot_id="bot1", error_type="RuntimeError",
            message="division by zero",
            stack_trace="Traceback...\n  File calc.py:10\nRuntimeError: division by zero",
        )
        result = SeverityClassifier().classify(e)
        assert result.severity == BugSeverity.MEDIUM


class TestLowSeverity:
    def test_warning_is_low(self):
        e = ErrorEvent(
            bot_id="bot1", error_type="UserWarning",
            message="some generic warning",
            stack_trace="...",
        )
        result = SeverityClassifier().classify(e)
        assert result.severity == BugSeverity.LOW
        assert result.category == ErrorCategory.WARNING

    def test_deprecation_is_low(self):
        e = ErrorEvent(
            bot_id="bot1", error_type="DeprecationWarning",
            message="module X is deprecated",
            stack_trace="...",
        )
        result = SeverityClassifier().classify(e)
        assert result.severity == BugSeverity.LOW
        assert result.category == ErrorCategory.DEPRECATION


class TestMutatesEvent:
    def test_classify_sets_severity_and_category_on_event(self):
        e = ErrorEvent(
            bot_id="bot1", error_type="SystemExit",
            message="crash", stack_trace="...", context={"crash": True},
        )
        assert e.severity is None
        result = SeverityClassifier().classify(e)
        # Returns a ClassificationResult, does NOT mutate the event
        assert e.severity is None  # original unchanged
        assert result.severity == BugSeverity.CRITICAL
