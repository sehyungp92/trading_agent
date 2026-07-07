"""Input sanitizer — all inbound messages are untrusted.

Blocks prompt injection patterns and classifies message intent.
This is a deterministic first-pass filter; it does NOT use LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SanitizedInput:
    safe: bool
    intent: str = "unknown"
    content: str = ""
    source: str = ""
    reason: str = ""
    requires_confirmation: bool = False


class InputSanitizer:
    """Deterministic input filter for prompt injection defense."""

    BLOCKED_PATTERNS: list[str] = [
        r"ignore previous instructions",
        r"override.*rules",
        r"system prompt",
        r"you are now",
        r"pretend to be",
        r"disregard.*above",
        r"forget.*instructions",
        r"new instructions",
        r"act as",
        r"jailbreak",
    ]

    INTENT_PATTERNS: dict[str, list[str]] = {
        "report_request": [
            r"\breport\b",
            r"\bsummary\b",
            r"\bstatus\b",
            r"\bshow\b.*\b(daily|weekly|bot|pnl|performance)\b",
            r"\bhow.*doing\b",
        ],
        "feedback": [
            r"\bactually\b",
            r"\bwas.*hedge\b",
            r"\bwrong\b.*\b(classification|regime|tag)\b",
            r"\bgood catch\b",
            r"\btrade\s*#",
        ],
        "approval": [
            r"^approve\b",
            r"^yes\b",
            r"^confirm\b",
            r"^lgtm\b",
            r"\bapprove\s+(all|pr|change)\b",
        ],
        "rejection": [
            r"^reject\b",
            r"^no\b",
            r"^deny\b",
            r"^cancel\b",
        ],
        "operational": [
            r"\brestart\b",
            r"\bstop\b.*\bbot\b",
            r"\bstart\b.*\bbot\b",
            r"\bdeploy\b",
            r"\bkill\b",
            r"\bscale\b",
        ],
    }

    def sanitize(self, message: str, source: str) -> SanitizedInput:
        """Check message for injection patterns, then classify intent."""
        # 1. Block prompt injection attempts
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return SanitizedInput(
                    safe=False,
                    reason=f"Blocked pattern: {pattern}",
                    content=message,
                    source=source,
                )

        # 2. Classify intent
        intent = self._classify_intent(message)

        # 3. Operational intents require confirmation
        if intent == "operational":
            return SanitizedInput(
                safe=True,
                requires_confirmation=True,
                intent=intent,
                content=message,
                source=source,
            )

        return SanitizedInput(
            safe=True,
            intent=intent,
            content=message,
            source=source,
        )

    def _classify_intent(self, message: str) -> str:
        """Match message against intent patterns. Returns first match or 'unknown'."""
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return intent
        return "unknown"
