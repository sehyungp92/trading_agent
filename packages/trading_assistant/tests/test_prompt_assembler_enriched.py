# tests/test_prompt_assembler_enriched.py
"""Tests for enriched prompt assemblers — failure log + rejected suggestions in context."""

from trading_assistant.analysis.prompt_assembler import DailyPromptAssembler
from trading_assistant.analysis.weekly_prompt_assembler import WeeklyPromptAssembler


class TestDailyAssemblerEnriched:
    def test_includes_failure_log_in_data(self, tmp_path):
        memory_dir = tmp_path / "memory"
        policies = memory_dir / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = memory_dir / "findings"
        findings.mkdir()
        (findings / "failure-log.jsonl").write_text(
            '{"error_type":"timeout","bot_id":"bot1","outcome":"known_fix"}\n'
        )
        curated = tmp_path / "curated"
        curated.mkdir()

        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory_dir,
        )
        pkg = assembler.assemble()
        assert "failure_log" in pkg.data

    def test_includes_rejected_suggestions_in_data(self, tmp_path):
        memory_dir = tmp_path / "memory"
        policies = memory_dir / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = memory_dir / "findings"
        findings.mkdir()
        (findings / "suggestions.jsonl").write_text(
            '{"suggestion_id":"s001","bot_id":"bot1","title":"Widen stop","status":"rejected","rejection_reason":"No evidence"}\n'
        )
        curated = tmp_path / "curated"
        curated.mkdir()

        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory_dir,
        )
        pkg = assembler.assemble()
        assert "rejected_suggestions" in pkg.data

    def test_instructions_reference_rejected_suggestions(self, tmp_path):
        memory_dir = tmp_path / "memory"
        policies = memory_dir / "policies" / "v1"
        policies.mkdir(parents=True)
        curated = tmp_path / "curated"
        curated.mkdir()

        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory_dir,
        )
        pkg = assembler.assemble()
        assert "rejected" in pkg.instructions.lower() or "previously rejected" in pkg.instructions.lower()


class TestWeeklyAssemblerEnriched:
    def test_includes_failure_log_in_data(self, tmp_path):
        memory_dir = tmp_path / "memory"
        policies = memory_dir / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = memory_dir / "findings"
        findings.mkdir()
        (findings / "failure-log.jsonl").write_text(
            '{"error_type":"timeout","outcome":"known_fix"}\n'
        )
        curated = tmp_path / "curated"
        curated.mkdir()
        runs = tmp_path / "runs"
        runs.mkdir()

        assembler = WeeklyPromptAssembler(
            week_start="2026-02-24",
            week_end="2026-03-02",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory_dir,
            runs_dir=runs,
        )
        pkg = assembler.assemble()
        assert "failure_log" in pkg.data

    def test_instructions_reference_structural_analysis(self, tmp_path):
        memory_dir = tmp_path / "memory"
        policies = memory_dir / "policies" / "v1"
        policies.mkdir(parents=True)
        curated = tmp_path / "curated"
        curated.mkdir()
        runs = tmp_path / "runs"
        runs.mkdir()

        assembler = WeeklyPromptAssembler(
            week_start="2026-02-24",
            week_end="2026-03-02",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory_dir,
            runs_dir=runs,
        )
        pkg = assembler.assemble()
        lower = pkg.instructions.lower()
        assert "structural" in lower or "signal decay" in lower
