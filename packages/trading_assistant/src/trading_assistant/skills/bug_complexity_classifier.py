# skills/bug_complexity_classifier.py
"""Bug complexity classifier — determines fix approach based on error characteristics.

Decision matrix:
  CRITICAL -> always ALERTED (immediate telegram)
  HIGH + OBVIOUS_FIX -> KNOWN_FIX (spawn fix agent)
  HIGH + SINGLE_FUNCTION -> NEEDS_INVESTIGATION (create GitHub issue)
  HIGH + STATE_DEPENDENT/MULTI_FILE -> NEEDS_HUMAN (Telegram alert)
  MEDIUM -> QUEUED_FOR_DAILY
  LOW -> QUEUED_FOR_WEEKLY
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from trading_assistant.schemas.bug_triage import (
    BugComplexity,
    BugSeverity,
    ErrorCategory,
    ErrorEvent,
    TriageOutcome,
)

_OBVIOUS_FIX_CATEGORIES = {
    ErrorCategory.DEPENDENCY,
    ErrorCategory.CONFIG_ERROR,
    ErrorCategory.DEPRECATION,
}

_STATE_DEPENDENT_CATEGORIES = {
    ErrorCategory.STUCK_POSITION,
    ErrorCategory.CONNECTION_LOST,
}

_MULTI_FILE_PATTERNS = re.compile(
    r"multi.*service|cross.*module|distributed|race.*condition|deadlock",
    re.I,
)


@dataclass
class ComplexityResult:
    complexity: BugComplexity
    outcome: TriageOutcome
    rationale: str


class BugComplexityClassifier:
    """Classifies bug complexity and determines triage outcome."""

    def classify(
        self,
        event: ErrorEvent,
        severity: BugSeverity,
        category: ErrorCategory,
    ) -> ComplexityResult:
        """Classify complexity and determine outcome.

        Args:
            event: The error event to classify.
            severity: Already-classified severity from SeverityClassifier.
            category: Already-classified category from SeverityClassifier.
        """
        complexity = self._assess_complexity(event, category)
        outcome = self._determine_outcome(severity, complexity)
        rationale = self._build_rationale(severity, complexity, category)
        return ComplexityResult(
            complexity=complexity, outcome=outcome, rationale=rationale,
        )

    def _assess_complexity(
        self, event: ErrorEvent, category: ErrorCategory,
    ) -> BugComplexity:
        """Assess how complex the bug is to fix."""
        if category in _OBVIOUS_FIX_CATEGORIES:
            return BugComplexity.OBVIOUS_FIX

        if category in _STATE_DEPENDENT_CATEGORIES:
            return BugComplexity.STATE_DEPENDENT

        text = f"{event.error_type} {event.message}"
        if _MULTI_FILE_PATTERNS.search(text):
            return BugComplexity.MULTI_FILE

        # If we have a single source file, it's likely a single-function issue
        if event.source_file and event.source_line > 0:
            return BugComplexity.SINGLE_FUNCTION

        # Count unique files in stack trace as a heuristic
        file_refs = re.findall(r'File "([^"]+)"', event.stack_trace)
        unique_files = {f for f in file_refs if not f.startswith("<")}
        if len(unique_files) > 3:
            return BugComplexity.MULTI_FILE
        if len(unique_files) == 1:
            return BugComplexity.SINGLE_FUNCTION

        return BugComplexity.UNKNOWN

    def _determine_outcome(
        self, severity: BugSeverity, complexity: BugComplexity,
    ) -> TriageOutcome:
        """Map (severity, complexity) to a triage outcome."""
        if severity == BugSeverity.CRITICAL:
            return TriageOutcome.ALERTED

        if severity == BugSeverity.LOW:
            return TriageOutcome.QUEUED_FOR_WEEKLY

        if severity == BugSeverity.MEDIUM:
            return TriageOutcome.QUEUED_FOR_DAILY

        # HIGH severity — outcome depends on complexity
        if complexity == BugComplexity.OBVIOUS_FIX:
            return TriageOutcome.KNOWN_FIX
        if complexity == BugComplexity.SINGLE_FUNCTION:
            return TriageOutcome.NEEDS_INVESTIGATION
        if complexity in (BugComplexity.MULTI_FILE, BugComplexity.STATE_DEPENDENT):
            return TriageOutcome.NEEDS_HUMAN

        return TriageOutcome.NEEDS_INVESTIGATION  # UNKNOWN complexity

    def _build_rationale(
        self,
        severity: BugSeverity,
        complexity: BugComplexity,
        category: ErrorCategory,
    ) -> str:
        return (
            f"Severity={severity.value}, complexity={complexity.value}, "
            f"category={category.value} → outcome determined by decision matrix"
        )
