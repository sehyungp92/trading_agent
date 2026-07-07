# skills/run_bug_triage.py
"""Triage runner — main bug triage pipeline.

Orchestrates: severity classification → error rate check → complexity classification →
outcome routing → failure log recording → context building.

This is the skill the orchestrator worker calls when it receives a SPAWN_TRIAGE action.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from trading_assistant.schemas.bug_triage import (
    BugSeverity,
    ErrorEvent,
    TriageResult,
)
from trading_assistant.skills.bug_complexity_classifier import BugComplexityClassifier
from trading_assistant.skills.error_rate_tracker import ErrorRateTracker
from trading_assistant.skills.failure_log import FailureLog
from trading_assistant.skills.severity_classifier import SeverityClassifier
from trading_assistant.skills.triage_context_builder import TriageContextBuilder


class TriageRunner:
    """Main triage pipeline — the entry point for bug triage."""

    def __init__(
        self,
        source_root: Path,
        failure_log_path: Path,
        repeated_error_threshold: int = 3,
        error_rate_window_seconds: int = 3600,
    ) -> None:
        self._severity_classifier = SeverityClassifier()
        self._complexity_classifier = BugComplexityClassifier()
        self._error_rate_tracker = ErrorRateTracker(
            window_seconds=error_rate_window_seconds,
            threshold=repeated_error_threshold,
        )
        self._failure_log = FailureLog(failure_log_path)
        self._context_builder = TriageContextBuilder(source_root=source_root)

    def record_error(self, event: ErrorEvent) -> None:
        """Record an error for rate tracking (called by worker on every error event)."""
        self._error_rate_tracker.record(event)

    def triage(
        self,
        event: ErrorEvent,
        severity_override: Optional[BugSeverity] = None,
    ) -> TriageResult:
        """Run the full triage pipeline for an error event.

        Args:
            event: The error event to triage.
            severity_override: If set, skip severity classification and use this value.
        """
        # Step 1: Classify severity
        if severity_override is not None:
            severity = severity_override
            classification = self._severity_classifier.classify(event)
            category = classification.category
        else:
            classification = self._severity_classifier.classify(event)
            severity = classification.severity
            category = classification.category

            # Step 2: Check error rate — promote to HIGH if repeated
            if severity.rank < BugSeverity.HIGH.rank and self._error_rate_tracker.is_repeated(event.bot_id):
                severity = BugSeverity.HIGH

        # Step 3: Load past rejections
        past_rejections = self._failure_log.get_past_rejections(
            error_type=event.error_type, limit=5,
        )
        rejection_texts = [r.rejection_reason for r in past_rejections if r.rejection_reason]

        # Step 4: Classify complexity and determine outcome
        complexity_result = self._complexity_classifier.classify(event, severity, category)

        # Step 5: Build triage result
        result = TriageResult(
            error_event=event,
            severity=severity,
            complexity=complexity_result.complexity,
            outcome=complexity_result.outcome,
            past_rejections=rejection_texts,
        )

        # Step 6: Record to failure log
        self._failure_log.record_triage(result)

        return result
