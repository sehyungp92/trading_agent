"""Tests for memory consolidation (M4) and memory index."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from trading_assistant.orchestrator.memory_consolidator import MemoryConsolidator

def _write_entries(path: Path, entries: list[dict]) -> None:
    lines = [json.dumps(e) for e in entries]
    path.write_text("\n".join(lines), encoding="utf-8")

@pytest.fixture
def findings_dir(tmp_path: Path) -> Path:
    d = tmp_path / "findings"
    d.mkdir()
    return d

@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a realistic project directory structure."""
    base = tmp_path / "project"
    base.mkdir()
    (base / "data" / "curated" / "2026-03-01" / "bot1").mkdir(parents=True)
    (base / "data" / "curated" / "2026-03-01" / "bot1" / "summary.json").write_text("{}")
    (base / "data" / "curated" / "2026-03-01" / "bot2").mkdir(parents=True)
    (base / "data" / "curated" / "2026-03-02" / "bot1").mkdir(parents=True)
    (base / "data" / "curated" / "weekly" / "2026-02-24").mkdir(parents=True)
    (base / "data" / "curated" / "weekly" / "2026-02-24" / "weekly_summary.json").write_text("{}")
    (base / ".assistant" / "sessions" / "daily_analysis" / "2026-03-01").mkdir(parents=True)
    (base / ".assistant" / "sessions" / "daily_analysis" / "2026-03-01" / "sessions.jsonl").write_text('{"session_id":"s1"}\n')
    (base / ".assistant" / "sessions" / "daily_analysis" / "2026-03-02").mkdir(parents=True)
    (base / ".assistant" / "sessions" / "daily_analysis" / "2026-03-02" / "sessions.jsonl").write_text('{"session_id":"s2"}\n')
    (base / ".assistant" / "sessions" / "monthly_validation" / "2026-03-01").mkdir(parents=True)
    (base / ".assistant" / "sessions" / "monthly_validation" / "2026-03-01" / "sessions.jsonl").write_text('{"session_id":"s3"}\n')
    (base / "runs" / "daily-2026-03-01").mkdir(parents=True)
    (base / "runs" / "monthly-validation-bot1-2026-03-01").mkdir(parents=True)
    findings = base / "findings"
    findings.mkdir()
    _write_entries(findings / "corrections.jsonl", [{"bot_id": f"b{i}"} for i in range(5)])
    _write_entries(findings / "triage_log.jsonl", [{"err": f"e{i}"} for i in range(3)])
    (base / "heartbeats").mkdir()
    (base / "heartbeats" / "bot1.heartbeat").write_text("2026-03-02T10:00:00+00:00")
    (base / "heartbeats" / "bot2.heartbeat").write_text("2026-03-02T09:30:00+00:00")
    return base

class TestNeedsConsolidation:
    def test_returns_false_below_threshold(self, findings_dir: Path):
        entries = [{"bot_id": f"bot-{i}", "error_type": "RuntimeError"} for i in range(50)]
        _write_entries(findings_dir / "corrections.jsonl", entries)
        consolidator = MemoryConsolidator(findings_dir, threshold=100)
        assert consolidator.needs_consolidation() is False

    def test_returns_true_above_threshold(self, findings_dir: Path):
        entries = [{"bot_id": f"bot-{i}", "error_type": "RuntimeError"} for i in range(150)]
        _write_entries(findings_dir / "corrections.jsonl", entries)
        consolidator = MemoryConsolidator(findings_dir, threshold=100)
        assert consolidator.needs_consolidation() is True

    def test_returns_false_for_missing_file(self, findings_dir: Path):
        consolidator = MemoryConsolidator(findings_dir, threshold=100)
        assert consolidator.needs_consolidation("nonexistent.jsonl") is False

class TestConsolidate:
    def test_produces_summary(self, findings_dir: Path):
        entries = [
            {"bot_id": "bot-a", "error_type": "ImportError", "root_causes": ["missing_dep"]}
            for _ in range(60)
        ] + [
            {"bot_id": "bot-b", "error_type": "RuntimeError", "root_causes": ["null_ref"]}
            for _ in range(50)
        ]
        _write_entries(findings_dir / "corrections.jsonl", entries)
        consolidator = MemoryConsolidator(findings_dir, threshold=100)
        summary = consolidator.consolidate()
        assert summary is not None
        assert summary.total_entries == 110
        assert len(summary.top_bots) == 2
        assert summary.top_bots[0].key == "bot-a"

    def test_returns_none_below_threshold(self, findings_dir: Path):
        entries = [{"bot_id": "bot-a"} for _ in range(50)]
        _write_entries(findings_dir / "corrections.jsonl", entries)
        consolidator = MemoryConsolidator(findings_dir, threshold=100)
        assert consolidator.consolidate() is None

    def test_writes_markdown_file(self, findings_dir: Path):
        entries = [{"bot_id": "bot-a", "error_type": "Err"} for _ in range(150)]
        _write_entries(findings_dir / "corrections.jsonl", entries)
        consolidator = MemoryConsolidator(findings_dir, threshold=100)
        consolidator.consolidate()
        md_path = findings_dir / "patterns_consolidated.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "Memory Consolidation Summary" in content
        assert "bot-a" in content

    def test_counts_correction_types(self, findings_dir: Path):
        entries = [
            {"correction_type": "trade_reclassify"} for _ in range(80)
        ] + [
            {"correction_type": "filter_adjustment"} for _ in range(30)
        ]
        _write_entries(findings_dir / "corrections.jsonl", entries)
        consolidator = MemoryConsolidator(findings_dir, threshold=100)
        summary = consolidator.consolidate()
        assert summary is not None
        types = {p.key: p.count for p in summary.top_error_types}
        assert types["trade_reclassify"] == 80

    def test_handles_malformed_json(self, findings_dir: Path):
        lines = [json.dumps({"bot_id": f"b{i}"}) for i in range(110)]
        lines.insert(50, "NOT VALID JSON")
        (findings_dir / "corrections.jsonl").write_text("\n".join(lines))
        consolidator = MemoryConsolidator(findings_dir, threshold=100)
        summary = consolidator.consolidate()
        assert summary is not None
        assert summary.total_entries == 110

    def test_returns_none_for_missing_file(self, findings_dir: Path):
        consolidator = MemoryConsolidator(findings_dir, threshold=100)
        assert consolidator.consolidate("nonexistent.jsonl") is None

    def test_consolidate_updates_index(self, findings_dir: Path):
        entries = [{"bot_id": "bot-a"} for _ in range(150)]
        _write_entries(findings_dir / "corrections.jsonl", entries)
        consolidator = MemoryConsolidator(findings_dir, threshold=100)
        consolidator.consolidate()
        index_path = findings_dir.parent / "memory" / "index.json"
        assert index_path.exists()


class TestRebuildIndex:
    def test_infers_project_root_from_memory_findings(self, tmp_path: Path):
        findings = tmp_path / "memory" / "findings"
        findings.mkdir(parents=True)
        (tmp_path / "curated" / "2026-03-01" / "bot1").mkdir(parents=True)

        consolidator = MemoryConsolidator(findings)
        index = consolidator.rebuild_index()

        assert index.curated_dates_by_bot["bot1"] == ["2026-03-01"]
        assert (tmp_path / "memory" / "index.json").exists()

    def test_scans_curated_dates(self, project_dir: Path):
        consolidator = MemoryConsolidator(project_dir / "findings", base_dir=project_dir)
        index = consolidator.rebuild_index()
        assert "bot1" in index.curated_dates_by_bot
        assert "2026-03-01" in index.curated_dates_by_bot["bot1"]
        assert "2026-03-02" in index.curated_dates_by_bot["bot1"]
        assert "bot2" in index.curated_dates_by_bot
        assert index.curated_dates_by_bot["bot2"] == ["2026-03-01"]

    def test_scans_weekly_dates(self, project_dir: Path):
        consolidator = MemoryConsolidator(project_dir / "findings", base_dir=project_dir)
        index = consolidator.rebuild_index()
        assert index.weekly_dates == ["2026-02-24"]

    def test_scans_sessions(self, project_dir: Path):
        consolidator = MemoryConsolidator(project_dir / "findings", base_dir=project_dir)
        index = consolidator.rebuild_index()
        assert "daily_analysis" in index.sessions_by_agent_type
        assert len(index.sessions_by_agent_type["daily_analysis"]) == 2
        assert "monthly_validation" in index.sessions_by_agent_type
        assert index.total_sessions == 3

    def test_scans_runs(self, project_dir: Path):
        consolidator = MemoryConsolidator(project_dir / "findings", base_dir=project_dir)
        index = consolidator.rebuild_index()
        assert "daily-2026-03-01" in index.run_ids
        assert "monthly-validation-bot1-2026-03-01" in index.run_ids

    def test_counts_findings(self, project_dir: Path):
        consolidator = MemoryConsolidator(project_dir / "findings", base_dir=project_dir)
        index = consolidator.rebuild_index()
        assert index.findings_counts["corrections.jsonl"] == 5
        assert index.findings_counts["triage_log.jsonl"] == 3
        assert index.total_findings == 8

    def test_scans_heartbeats(self, project_dir: Path):
        consolidator = MemoryConsolidator(project_dir / "findings", base_dir=project_dir)
        index = consolidator.rebuild_index()
        assert "bot1" in index.heartbeat_last_seen
        assert "bot2" in index.heartbeat_last_seen
        assert index.heartbeat_last_seen["bot1"] == "2026-03-02T10:00:00+00:00"

    def test_summary_stats(self, project_dir: Path):
        consolidator = MemoryConsolidator(project_dir / "findings", base_dir=project_dir)
        index = consolidator.rebuild_index()
        assert index.total_curated_days == 2
        assert index.total_sessions == 3
        assert index.total_findings == 8
        assert index.last_consolidated != ""

    def test_writes_index_json(self, project_dir: Path):
        consolidator = MemoryConsolidator(project_dir / "findings", base_dir=project_dir)
        consolidator.rebuild_index()
        index_path = project_dir / "memory" / "index.json"
        assert index_path.exists()
        loaded = json.loads(index_path.read_text())
        assert loaded["total_curated_days"] == 2

    def test_load_index(self, project_dir: Path):
        consolidator = MemoryConsolidator(project_dir / "findings", base_dir=project_dir)
        consolidator.rebuild_index()
        loaded = MemoryConsolidator.load_index(project_dir)
        assert loaded is not None
        assert loaded.total_curated_days == 2
        assert "bot1" in loaded.curated_dates_by_bot

    def test_load_index_returns_none_if_missing(self, tmp_path: Path):
        assert MemoryConsolidator.load_index(tmp_path) is None

    def test_handles_empty_project(self, tmp_path: Path):
        findings = tmp_path / "findings"
        findings.mkdir()
        consolidator = MemoryConsolidator(findings, base_dir=tmp_path)
        index = consolidator.rebuild_index()
        assert index.total_curated_days == 0
        assert index.total_sessions == 0
        assert index.total_findings == 0
        assert index.run_ids == []

    def test_ignores_non_date_directories(self, project_dir: Path):
        # Create a non-date directory in curated
        (project_dir / "data" / "curated" / "readme.txt").write_text("ignore me")
        (project_dir / "data" / "curated" / "templates").mkdir()
        consolidator = MemoryConsolidator(project_dir / "findings", base_dir=project_dir)
        index = consolidator.rebuild_index()
        # Should still only find the 2 date dirs
        assert index.total_curated_days == 2
