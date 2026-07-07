# tests/test_triage_prompt_assembler.py
"""Tests for triage prompt assembler — packages context for Claude."""
from pathlib import Path

from trading_assistant.schemas.bug_triage import BugSeverity, BugComplexity
from trading_assistant.skills.triage_context_builder import TriageContext
from trading_assistant.analysis.triage_prompt_assembler import TriagePromptAssembler


class TestTriagePromptAssembler:
    def test_assemble_returns_required_keys(self, tmp_path: Path):
        ctx = TriageContext(
            error_event_summary="[HIGH] RuntimeError: division by zero (bot=bot1)",
            stack_trace="Traceback...",
            source_snippet="10 >>> return x / y",
            recent_git_log="abc123 fix: previous fix",
            past_rejections=[],
        )
        asm = TriagePromptAssembler(memory_dir=tmp_path)
        result = asm.assemble(ctx, BugSeverity.HIGH, BugComplexity.OBVIOUS_FIX)

        assert result.system_prompt is not None
        assert result.task_prompt
        assert result.data.get("context")

    def test_task_prompt_includes_severity_and_complexity(self, tmp_path: Path):
        ctx = TriageContext(
            error_event_summary="[HIGH] ImportError: no module foo",
            stack_trace="...",
            source_snippet="",
            recent_git_log="",
            past_rejections=[],
        )
        asm = TriagePromptAssembler(memory_dir=tmp_path)
        result = asm.assemble(ctx, BugSeverity.HIGH, BugComplexity.OBVIOUS_FIX)

        assert "HIGH" in result.task_prompt
        assert "obvious_fix" in result.task_prompt

    def test_includes_past_rejections_in_context(self, tmp_path: Path):
        ctx = TriageContext(
            error_event_summary="[HIGH] ImportError: no module foo",
            stack_trace="...",
            source_snippet="",
            recent_git_log="",
            past_rejections=["Past rejection: wrong version pinned"],
        )
        asm = TriagePromptAssembler(memory_dir=tmp_path)
        result = asm.assemble(ctx, BugSeverity.HIGH, BugComplexity.OBVIOUS_FIX)

        assert "wrong version pinned" in result.data["context"]

    def test_includes_stack_trace_in_context(self, tmp_path: Path):
        ctx = TriageContext(
            error_event_summary="[HIGH] RuntimeError: x",
            stack_trace="Traceback (most recent call last):\n  File foo.py:10\nRuntimeError: x",
            source_snippet="",
            recent_git_log="",
            past_rejections=[],
        )
        asm = TriagePromptAssembler(memory_dir=tmp_path)
        result = asm.assemble(ctx, BugSeverity.HIGH, BugComplexity.SINGLE_FUNCTION)

        assert "Traceback" in result.data["context"]

    def test_includes_source_snippet_in_context(self, tmp_path: Path):
        ctx = TriageContext(
            error_event_summary="[HIGH] E: m",
            stack_trace="...",
            source_snippet="  10 >>> return x / y",
            recent_git_log="",
            past_rejections=[],
        )
        asm = TriagePromptAssembler(memory_dir=tmp_path)
        result = asm.assemble(ctx, BugSeverity.HIGH, BugComplexity.SINGLE_FUNCTION)

        assert "return x / y" in result.data["context"]

    def test_bot_id_passed_to_metadata(self, tmp_path: Path):
        ctx = TriageContext(
            error_event_summary="[HIGH] RuntimeError: x",
            stack_trace="...",
            source_snippet="",
            recent_git_log="",
            past_rejections=[],
        )
        asm = TriagePromptAssembler(memory_dir=tmp_path)
        result = asm.assemble(ctx, BugSeverity.HIGH, BugComplexity.OBVIOUS_FIX, bot_id="bot_a")

        assert result.metadata["bot_ids"] == ["bot_a"]

    def test_empty_bot_id_omits_bot_ids_metadata(self, tmp_path: Path):
        ctx = TriageContext(
            error_event_summary="[HIGH] RuntimeError: x",
            stack_trace="...",
            source_snippet="",
            recent_git_log="",
            past_rejections=[],
        )
        asm = TriagePromptAssembler(memory_dir=tmp_path)
        result = asm.assemble(ctx, BugSeverity.HIGH, BugComplexity.OBVIOUS_FIX)

        assert "bot_ids" not in result.metadata

    def test_loads_policies_for_system_prompt(self, tmp_path: Path):
        policy_dir = tmp_path / "policies" / "v1"
        policy_dir.mkdir(parents=True)
        (policy_dir / "agent.md").write_text("Be helpful.")

        asm = TriagePromptAssembler(memory_dir=tmp_path)
        ctx = TriageContext(
            error_event_summary="e", stack_trace="s",
            source_snippet="", recent_git_log="", past_rejections=[],
        )
        result = asm.assemble(ctx, BugSeverity.HIGH, BugComplexity.OBVIOUS_FIX)

        assert "Be helpful" in result.system_prompt
