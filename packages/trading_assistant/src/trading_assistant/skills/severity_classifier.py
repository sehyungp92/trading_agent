# skills/severity_classifier.py
"""Deterministic severity classifier — no LLM calls.

Routes error events into severity buckets based on error_type, message patterns,
and context flags. This is the first stage of the triage pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from trading_assistant.schemas.bug_triage import BugSeverity, ErrorCategory, ErrorEvent

# Patterns for each category — checked in priority order (most severe first)
_CRITICAL_PATTERNS: list[tuple[re.Pattern, ErrorCategory]] = [
    (re.compile(r"crash|segfault|abort|SystemExit|SIGKILL|SIGSEGV", re.I), ErrorCategory.CRASH),
    (re.compile(r"stuck.*(position|order)|position.*stuck|open.*(?:too|for)\s+\d+\s*h", re.I), ErrorCategory.STUCK_POSITION),
    (re.compile(r"connection.*(lost|refused|reset|timeout)|lost.*connection|cannot.*reach.*exchange", re.I), ErrorCategory.CONNECTION_LOST),
]

_HIGH_PATTERNS: list[tuple[re.Pattern, ErrorCategory]] = [
    (re.compile(r"unexpected.*loss|loss.*exceed|abnormal.*drawdown", re.I), ErrorCategory.UNEXPECTED_LOSS),
    (re.compile(r"API.*(?:error|500|502|503|429)|exchange.*(?:error|reject|refused)", re.I), ErrorCategory.API_ERROR),
]

_MEDIUM_PATTERNS: list[tuple[re.Pattern, ErrorCategory]] = [
    (re.compile(r"config.*(?:error|invalid|missing)|invalid.*config", re.I), ErrorCategory.CONFIG_ERROR),
    (re.compile(r"import.*error|module.*not.*found|dependency", re.I), ErrorCategory.DEPENDENCY),
]

_LOW_PATTERNS: list[tuple[re.Pattern, ErrorCategory]] = [
    (re.compile(r"deprecat", re.I), ErrorCategory.DEPRECATION),
    (re.compile(r"warning|UserWarning|FutureWarning", re.I), ErrorCategory.WARNING),
]


@dataclass
class ClassificationResult:
    severity: BugSeverity
    category: ErrorCategory


class SeverityClassifier:
    """Deterministic error severity classifier."""

    def classify(self, event: ErrorEvent) -> ClassificationResult:
        """Classify an error event's severity and category.

        Checks context flags first (most reliable), then error_type + message patterns.
        Returns a ClassificationResult; does NOT mutate the event.
        """
        # 1. Check explicit context flags (highest priority)
        ctx = event.context
        if ctx.get("crash"):
            return ClassificationResult(BugSeverity.CRITICAL, ErrorCategory.CRASH)
        if ctx.get("stuck_position"):
            return ClassificationResult(BugSeverity.CRITICAL, ErrorCategory.STUCK_POSITION)
        if ctx.get("connection_lost"):
            return ClassificationResult(BugSeverity.CRITICAL, ErrorCategory.CONNECTION_LOST)
        if ctx.get("unexpected_loss"):
            return ClassificationResult(BugSeverity.HIGH, ErrorCategory.UNEXPECTED_LOSS)

        # 2. Pattern match against error_type + message
        text = f"{event.error_type} {event.message}"

        for pattern, category in _CRITICAL_PATTERNS:
            if pattern.search(text):
                return ClassificationResult(BugSeverity.CRITICAL, category)

        for pattern, category in _HIGH_PATTERNS:
            if pattern.search(text):
                return ClassificationResult(BugSeverity.HIGH, category)

        for pattern, category in _MEDIUM_PATTERNS:
            if pattern.search(text):
                return ClassificationResult(BugSeverity.MEDIUM, category)

        for pattern, category in _LOW_PATTERNS:
            if pattern.search(text):
                return ClassificationResult(BugSeverity.LOW, category)

        # 3. Default: MEDIUM with UNKNOWN category
        return ClassificationResult(BugSeverity.MEDIUM, ErrorCategory.UNKNOWN)
