# tests/test_prompt_assembler.py
"""Tests for the daily analysis prompt assembler."""
import json
from pathlib import Path

import pytest

from trading_assistant.analysis.prompt_assembler import DailyPromptAssembler


@pytest.fixture
def curated_dir(tmp_path: Path) -> Path:
    """Set up a minimal curated data directory."""
    date_dir = tmp_path / "data" / "curated" / "2026-03-01"

    # Bot1 data
    bot_dir = date_dir / "bot1"
    bot_dir.mkdir(parents=True)
    (bot_dir / "summary.json").write_text(json.dumps({
        "bot_id": "bot1", "date": "2026-03-01", "total_trades": 10,
        "win_count": 6, "loss_count": 4, "net_pnl": 150.0,
    }))
    (bot_dir / "winners.json").write_text(json.dumps([
        {"trade_id": "t1", "pnl": 200.0, "pair": "BTCUSDT"},
    ]))
    (bot_dir / "losers.json").write_text(json.dumps([
        {"trade_id": "t2", "pnl": -50.0, "pair": "ETHUSDT"},
    ]))
    (bot_dir / "process_failures.json").write_text("[]")
    (bot_dir / "notable_missed.json").write_text("[]")
    (bot_dir / "regime_analysis.json").write_text(json.dumps({"regime_pnl": {"trending_up": 150.0}}))
    (bot_dir / "filter_analysis.json").write_text(json.dumps({"filter_block_counts": {}}))
    (bot_dir / "root_cause_summary.json").write_text(json.dumps({"distribution": {"normal_win": 6}}))
    (bot_dir / "factor_attribution.json").write_text(json.dumps({"factors": [{"name": "rsi", "contribution": 0.3}]}))
    (bot_dir / "exit_efficiency.json").write_text(json.dumps({"avg_exit_score": 0.72, "trades_analyzed": 10}))
    (bot_dir / "hourly_performance.json").write_text(json.dumps({"buckets": [{"hour": 14, "pnl": 50.0}]}))
    (bot_dir / "slippage_stats.json").write_text(json.dumps({"avg_slippage_bps": 2.5}))

    # Portfolio risk card
    (date_dir / "portfolio_risk_card.json").write_text(json.dumps({
        "date": "2026-03-01", "total_exposure_pct": 30.0, "crowding_alerts": [],
    }))

    return tmp_path / "data" / "curated"


@pytest.fixture
def memory_dir(tmp_path: Path) -> Path:
    """Set up a minimal memory directory."""
    mem = tmp_path / "memory"
    policies = mem / "policies" / "v1"
    policies.mkdir(parents=True)
    (policies / "agent.md").write_text("You are a trading analyst.")
    (policies / "trading_rules.md").write_text("Max 3 suggestions.")
    (policies / "soul.md").write_text("Be helpful.")

    findings = mem / "findings"
    findings.mkdir(parents=True)
    (findings / "corrections.jsonl").write_text("")
    (findings / "prompt_patterns.jsonl").write_text("")

    return mem


class TestDailyPromptAssembler:
    def test_assembles_system_prompt(self, curated_dir: Path, memory_dir: Path):
        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()

        assert prompt.system_prompt
        assert "You are a trading analyst" in prompt.system_prompt
        assert "Max 3 suggestions" in prompt.system_prompt

    def test_assembles_task_prompt(self, curated_dir: Path, memory_dir: Path):
        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()

        assert prompt.task_prompt
        assert "bot1" in prompt.task_prompt
        assert "2026-03-01" in prompt.task_prompt

    def test_includes_structured_data(self, curated_dir: Path, memory_dir: Path):
        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()

        assert prompt.data
        assert "bot1" in prompt.data
        assert "summary" in prompt.data["bot1"]

    def test_includes_portfolio_risk_card(self, curated_dir: Path, memory_dir: Path):
        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()

        assert "portfolio_risk_card" in prompt.data

    def test_skips_malformed_bot_json_and_records_error(
        self, curated_dir: Path, memory_dir: Path,
    ):
        bad_path = curated_dir / "2026-03-01" / "bot1" / "winners.json"
        bad_path.write_text("{not json", encoding="utf-8")

        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()

        assert "summary" in prompt.data["bot1"]
        assert "winners" not in prompt.data["bot1"]
        assert prompt.data["data_load_errors"]
        assert str(bad_path) == prompt.data["data_load_errors"][0]["path"]
        assert str(bad_path) not in prompt.context_files

    def test_skips_malformed_portfolio_json_and_keeps_valid_files(
        self, curated_dir: Path, memory_dir: Path,
    ):
        bad_path = curated_dir / "2026-03-01" / "portfolio_risk_card.json"
        bad_path.write_text("{not json", encoding="utf-8")

        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()

        assert "summary" in prompt.data["bot1"]
        assert "portfolio_risk_card" not in prompt.data
        assert any(e["path"] == str(bad_path) for e in prompt.data["data_load_errors"])
        assert str(bad_path) not in prompt.context_files

    def test_includes_instructions(self, curated_dir: Path, memory_dir: Path):
        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()

        assert prompt.instructions
        assert "ANALYTICAL TASKS" in prompt.instructions
        assert "actionable" in prompt.instructions.lower()

    def test_includes_corrections_context(self, curated_dir: Path, memory_dir: Path):
        corrections_path = memory_dir / "findings" / "corrections.jsonl"
        corrections_path.write_text(json.dumps({
            "correction_type": "trade_reclassify",
            "original_report_id": "daily-2026-02-28",
            "raw_text": "That was a hedge",
            "timestamp": "2026-02-28T12:00:00Z",
        }) + "\n")

        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            corrections_lookback_days=30,
        )
        prompt = assembler.assemble()
        assert len(prompt.corrections) == 1

    def test_context_file_list(self, curated_dir: Path, memory_dir: Path):
        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()

        assert len(prompt.context_files) > 0

    def test_includes_factor_attribution(self, curated_dir: Path, memory_dir: Path):
        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()
        assert "factor_attribution" in prompt.data["bot1"]
        assert prompt.data["bot1"]["factor_attribution"]["factors"][0]["name"] == "rsi"

    def test_includes_exit_efficiency(self, curated_dir: Path, memory_dir: Path):
        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()
        assert "exit_efficiency" in prompt.data["bot1"]

    def test_includes_hourly_performance(self, curated_dir: Path, memory_dir: Path):
        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()
        assert "hourly_performance" in prompt.data["bot1"]

    def test_includes_slippage_stats(self, curated_dir: Path, memory_dir: Path):
        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()
        assert "slippage_stats" in prompt.data["bot1"]

    def test_instructions_reference_evidence_base(self, curated_dir: Path, memory_dir: Path):
        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()
        assert "evidence base" in prompt.instructions.lower()

    def test_instructions_reference_exit_efficiency(self, curated_dir: Path, memory_dir: Path):
        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()
        assert "exit" in prompt.instructions.lower()


# ---------------------------------------------------------------------------
# Learning loop gap closure — prompt instruction tests
# ---------------------------------------------------------------------------

class TestDailyInstructionGapClosures:
    """Verify daily prompt instructions reference validation_patterns and reveal blocking rules."""

    @pytest.fixture()
    def assembler(self, curated_dir: Path, memory_dir: Path):
        return DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )

    def test_instructions_reference_validation_patterns(self, assembler):
        pkg = assembler.assemble()
        assert "validation_patterns" in pkg.instructions
        assert "3+ blocks" in pkg.instructions

    def test_constraints_reveal_blocking(self, assembler):
        pkg = assembler.assemble()
        assert "BLOCKED by validator" in pkg.instructions

    def test_structured_output_critical_warning(self, assembler):
        pkg = assembler.assemble()
        assert "CRITICAL" in pkg.instructions
        assert "LOST" in pkg.instructions

    def test_outcome_quality_note_in_constraints(self, assembler):
        pkg = assembler.assemble()
        assert "HIGH/MEDIUM quality" in pkg.instructions
