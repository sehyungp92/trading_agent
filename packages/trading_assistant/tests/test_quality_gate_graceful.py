"""Tests for quality gate graceful degradation."""
from pathlib import Path

from trading_assistant.analysis.quality_gate import QualityGate


def _setup_bot_dir(base: Path, date: str, bot_id: str, files: list[str]) -> None:
    bot_dir = base / date / bot_id
    bot_dir.mkdir(parents=True)
    for f in files:
        if f.endswith(".jsonl"):
            (bot_dir / f).write_text("")
        else:
            (bot_dir / f).write_text("{}")
    (base / date / "portfolio_risk_card.json").write_text("{}")


ALL_FILES = [
    "summary.json", "winners.json", "losers.json", "process_failures.json",
    "notable_missed.json", "regime_analysis.json", "filter_analysis.json",
    "root_cause_summary.json", "hourly_performance.json", "slippage_stats.json",
    "factor_attribution.json", "exit_efficiency.json", "trades.jsonl", "missed.jsonl",
]


class TestGracefulDegradation:
    def test_partial_bots_reported_passes_with_degradation(self, tmp_path):
        _setup_bot_dir(tmp_path, "2026-03-01", "bot1", ALL_FILES)
        gate = QualityGate("r1", "2026-03-01", ["bot1", "bot2"], tmp_path)
        checklist = gate.run()
        assert checklist.can_proceed is True
        assert checklist.data_completeness < 1.0
        assert "bot2" in checklist.missing_bots

    def test_all_bots_present_full_completeness(self, tmp_path):
        _setup_bot_dir(tmp_path, "2026-03-01", "bot1", ALL_FILES)
        _setup_bot_dir(tmp_path, "2026-03-01", "bot2", ALL_FILES)
        gate = QualityGate("r1", "2026-03-01", ["bot1", "bot2"], tmp_path)
        checklist = gate.run()
        assert checklist.can_proceed is True
        assert checklist.data_completeness == 1.0
        assert checklist.missing_bots == []

    def test_missing_files_reduces_completeness(self, tmp_path):
        _setup_bot_dir(tmp_path, "2026-03-01", "bot1", ALL_FILES)
        _setup_bot_dir(tmp_path, "2026-03-01", "bot2", ALL_FILES[:5])
        gate = QualityGate("r1", "2026-03-01", ["bot1", "bot2"], tmp_path)
        checklist = gate.run()
        assert checklist.can_proceed is True
        assert 0.5 < checklist.data_completeness < 1.0

    def test_no_bots_present_still_proceeds_with_zero_completeness(self, tmp_path):
        gate = QualityGate("r1", "2026-03-01", ["bot1", "bot2"], tmp_path)
        checklist = gate.run()
        assert checklist.can_proceed is True
        assert checklist.data_completeness == 0.0

    def test_expected_files_includes_hourly_and_slippage(self, tmp_path):
        _setup_bot_dir(tmp_path, "2026-03-01", "bot1", ALL_FILES[:8])
        gate = QualityGate("r1", "2026-03-01", ["bot1"], tmp_path)
        checklist = gate.run()
        assert checklist.data_completeness < 1.0
