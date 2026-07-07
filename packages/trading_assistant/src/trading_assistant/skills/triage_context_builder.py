# skills/triage_context_builder.py
"""Triage context builder — assembles context for the bug triage agent.

Collects: error summary, stack trace, source code snippet around the error line,
recent git log, and past rejection reasons for similar errors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from trading_assistant.schemas.bug_triage import BugSeverity, ErrorCategory, ErrorEvent
from trading_assistant.skills.failure_log import FailureEntry


@dataclass
class TriageContext:
    """Context package for the triage agent."""

    error_event_summary: str
    stack_trace: str
    source_snippet: str
    recent_git_log: str
    past_rejections: list[str] = field(default_factory=list)


class TriageContextBuilder:
    """Builds context for the bug triage agent."""

    def __init__(
        self,
        source_root: Path,
        context_lines: int = 10,
    ) -> None:
        self._source_root = source_root
        self._context_lines = context_lines

    def build(
        self,
        event: ErrorEvent,
        severity: BugSeverity,
        category: ErrorCategory,
        past_rejections: list[FailureEntry],
    ) -> TriageContext:
        summary = (
            f"[{severity.value.upper()}] {event.error_type}: {event.message} "
            f"(bot={event.bot_id}, category={category.value})"
        )

        source_snippet = self._extract_source(event.source_file, event.source_line)
        rejection_texts = [
            f"Past rejection: {r.rejection_reason}" for r in past_rejections if r.rejection_reason
        ]

        return TriageContext(
            error_event_summary=summary,
            stack_trace=event.stack_trace,
            source_snippet=source_snippet,
            recent_git_log="",  # populated by caller if git is available
            past_rejections=rejection_texts,
        )

    def _extract_source(self, source_file: str, source_line: int) -> str:
        """Extract source code around the error line."""
        if not source_file or source_line <= 0:
            return ""

        path = self._source_root / source_file
        if not path.exists():
            return ""

        try:
            all_lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return ""

        start = max(0, source_line - 1 - self._context_lines)
        end = min(len(all_lines), source_line + self._context_lines)

        snippet_lines: list[str] = []
        for i in range(start, end):
            line_num = i + 1
            marker = " >>> " if line_num == source_line else "     "
            snippet_lines.append(f"{line_num:4d}{marker}{all_lines[i]}")

        return "\n".join(snippet_lines)
