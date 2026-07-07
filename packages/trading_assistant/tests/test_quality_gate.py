# tests/test_quality_gate.py
"""Tests for the report quality gate."""
import json
from pathlib import Path


from trading_assistant.analysis.quality_gate import QualityGate


class TestQualityGate:
    def test_all_pass_with_complete_data(self, tmp_path: Path):
        """Full bot data present → all checks pass."""
        bot_dir = tmp_path / "2026-03-01" / "bot1"
        bot_dir.mkdir(parents=True)

        # Write all expected files
        (bot_dir / "summary.json").write_text(json.dumps({
            "bot_id": "bot1", "date": "2026-03-01", "total_trades": 10,
            "error_count": 0, "avg_process_quality": 85,
        }))
        (bot_dir / "winners.json").write_text("[]")
        (bot_dir / "losers.json").write_text("[]")
        (bot_dir / "process_failures.json").write_text("[]")
        (bot_dir / "notable_missed.json").write_text("[]")
        (bot_dir / "regime_analysis.json").write_text("{}")
        (bot_dir / "filter_analysis.json").write_text("{}")
        (bot_dir / "root_cause_summary.json").write_text(json.dumps({
            "distribution": {"normal_win": 6, "normal_loss": 4}, "total_trades": 10,
        }))
        (bot_dir / "hourly_performance.json").write_text("{}")
        (bot_dir / "slippage_stats.json").write_text("{}")
        (bot_dir / "factor_attribution.json").write_text("{}")
        (bot_dir / "exit_efficiency.json").write_text("{}")
        (bot_dir / "trades.jsonl").write_text("")
        (bot_dir / "missed.jsonl").write_text("")

        # Portfolio risk card at date level
        risk_dir = tmp_path / "2026-03-01"
        (risk_dir / "portfolio_risk_card.json").write_text(json.dumps({
            "date": "2026-03-01", "total_exposure_pct": 30.0, "crowding_alerts": [],
        }))

        gate = QualityGate(
            report_id="daily-2026-03-01",
            date="2026-03-01",
            expected_bots=["bot1"],
            curated_dir=tmp_path,
        )
        checklist = gate.run()

        assert checklist.overall == "PASS"

    def test_fails_when_bot_missing(self, tmp_path: Path):
        gate = QualityGate(
            report_id="daily-2026-03-01",
            date="2026-03-01",
            expected_bots=["bot1", "bot2"],
            curated_dir=tmp_path,
        )
        checklist = gate.run()
        assert checklist.overall == "FAIL"
        assert any("all_bots_reported" in issue for issue in checklist.blocking_issues)

    def test_fails_when_risk_card_missing(self, tmp_path: Path):
        bot_dir = tmp_path / "2026-03-01" / "bot1"
        bot_dir.mkdir(parents=True)
        for f in ["summary.json", "winners.json", "losers.json", "process_failures.json",
                   "notable_missed.json", "regime_analysis.json", "filter_analysis.json",
                   "root_cause_summary.json"]:
            (bot_dir / f).write_text("{}" if not f.endswith("s.json") else "[]")

        gate = QualityGate(
            report_id="daily-2026-03-01",
            date="2026-03-01",
            expected_bots=["bot1"],
            curated_dir=tmp_path,
        )
        checklist = gate.run()
        assert checklist.overall == "FAIL"
        assert any("portfolio_risk_card" in issue for issue in checklist.blocking_issues)

    def test_fails_when_curated_file_missing(self, tmp_path: Path):
        bot_dir = tmp_path / "2026-03-01" / "bot1"
        bot_dir.mkdir(parents=True)
        # Only write some files, skip root_cause_summary
        for f in ["summary.json", "winners.json", "losers.json", "process_failures.json",
                   "notable_missed.json", "regime_analysis.json", "filter_analysis.json"]:
            (bot_dir / f).write_text("{}" if not f.endswith("s.json") else "[]")

        (tmp_path / "2026-03-01" / "portfolio_risk_card.json").write_text("{}")

        gate = QualityGate(
            report_id="daily-2026-03-01",
            date="2026-03-01",
            expected_bots=["bot1"],
            curated_dir=tmp_path,
        )
        checklist = gate.run()
        assert checklist.overall == "FAIL"

    def test_writes_checklist_json(self, tmp_path: Path):
        gate = QualityGate(
            report_id="daily-2026-03-01",
            date="2026-03-01",
            expected_bots=["bot1"],
            curated_dir=tmp_path,
        )
        checklist = gate.run()
        output_path = tmp_path / "2026-03-01" / "report_checklist.json"
        gate.write_checklist(checklist, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["report_id"] == "daily-2026-03-01"
