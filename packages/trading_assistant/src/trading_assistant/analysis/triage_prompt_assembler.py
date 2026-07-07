# analysis/triage_prompt_assembler.py
"""Triage prompt assembler — builds context package for the triage agent runtime.

Uses ContextBuilder for shared policy loading. Adds triage-specific
context: stack trace, source snippet, git log, past rejections.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.analysis.evidence_memory import EvidenceMemory
from trading_assistant.schemas.bug_triage import BugComplexity, BugSeverity
from trading_assistant.schemas.prompt_package import PromptPackage
from trading_assistant.skills.triage_context_builder import TriageContext

_TRIAGE_INSTRUCTIONS = """\
1. Analyze the error event summary and severity classification
2. Read the stack trace to identify the root cause
3. If source code is provided, review the code around the error
4. Check past rejections — do NOT repeat previously rejected fixes
5. Based on the complexity assessment:
   - OBVIOUS_FIX: Propose a specific fix with exact file paths and code changes
   - SINGLE_FUNCTION: Identify the root cause, suggest investigation steps
   - MULTI_FILE / STATE_DEPENDENT: Summarize findings, recommend human intervention
6. Output: triage_result.json with outcome, affected_files, and suggested_fix"""


class TriagePromptAssembler:
    """Assembles the full context package for a bug triage agent invocation."""

    def __init__(self, memory_dir: Path) -> None:
        self._memory_dir = memory_dir
        self._evidence = EvidenceMemory(memory_dir)
        self._ctx = ContextBuilder(memory_dir, evidence_memory=self._evidence)

    def assemble(
        self,
        context: TriageContext,
        severity: BugSeverity,
        complexity: BugComplexity,
        session_store=None,
        bot_id: str = "",
    ) -> PromptPackage:
        """Build the complete prompt package."""
        pkg = self._ctx.base_package(session_store=session_store, agent_type="triage", bot_id=bot_id)
        pkg.task_prompt = self._build_task_prompt(severity, complexity)
        pkg.data.update({"context": self._build_context(context)})
        pkg.instructions = _TRIAGE_INSTRUCTIONS
        pkg.metadata["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if bot_id:
            pkg.metadata["bot_ids"] = [bot_id]
        return pkg

    def _build_task_prompt(
        self, severity: BugSeverity, complexity: BugComplexity,
    ) -> str:
        return (
            f"Triage this error event.\n"
            f"Severity: {severity.value.upper()}\n"
            f"Complexity: {complexity.value}\n"
            f"Follow the instructions to produce a triage result."
        )

    def _build_context(self, context: TriageContext) -> str:
        sections: list[str] = []

        sections.append(f"## Error Summary\n{context.error_event_summary}")
        sections.append(f"## Stack Trace\n```\n{context.stack_trace}\n```")

        if context.source_snippet:
            sections.append(f"## Source Code\n```python\n{context.source_snippet}\n```")

        if context.recent_git_log:
            sections.append(f"## Recent Git Log\n```\n{context.recent_git_log}\n```")

        if context.past_rejections:
            rejections = "\n".join(f"- {r}" for r in context.past_rejections)
            sections.append(f"## Past Rejections\n{rejections}")

        return "\n\n".join(sections)
