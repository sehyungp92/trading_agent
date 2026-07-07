# tests/test_weekly_prompt_assembler.py
"""Tests for the weekly prompt assembler."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from trading_assistant.analysis.weekly_prompt_assembler import WeeklyPromptAssembler


@pytest.fixture
def setup_dirs(tmp_path: Path):
    """Create directory structure with weekly curated data and daily reports."""
    curated = tmp_path / "curated"
    memory = tmp_path / "memory"
    runs = tmp_path / "runs"

    # Weekly curated data
    weekly_dir = curated / "weekly" / "2026-02-23"
    weekly_dir.mkdir(parents=True)
    (weekly_dir / "weekly_summary.json").write_text(
        json.dumps({"week_start": "2026-02-23", "total_net_pnl": 500.0})
    )
    (weekly_dir / "refinement_report.json").write_text(
        json.dumps({"suggestions": [{"title": "Adjust RSI"}]})
    )
    (weekly_dir / "week_over_week.json").write_text(
        json.dumps({"pnl_delta": 100.0})
    )

    # 7 daily reports in the real runs/daily-<date>* layout
    week_start_dt = datetime.strptime("2026-02-23", "%Y-%m-%d")
    week_dates = [(week_start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    for index, date in enumerate(week_dates):
        run_dir = runs / f"daily-{date}-run{index}"
        run_dir.mkdir(parents=True)
        if index == len(week_dates) - 1:
            (run_dir / "response.md").write_text(f"# Daily Report {date}\nFallback path.")
        else:
            (run_dir / "daily_report.md").write_text(f"# Daily Report {date}\nAll good.")

    # Portfolio risk cards (7 days)
    for date in week_dates:
        date_dir = curated / date
        date_dir.mkdir(parents=True)
        (date_dir / "portfolio_risk_card.json").write_text(
            json.dumps({"date": date, "concentration_score": 30.0})
        )

    # Memory policies
    policy_dir = memory / "policies" / "v1"
    policy_dir.mkdir(parents=True)
    (policy_dir / "agent.md").write_text("You are the trading assistant.")
    (policy_dir / "trading_rules.md").write_text("Max drawdown 15%.")
    (policy_dir / "soul.md").write_text("Conservative approach.")

    # Corrections
    findings_dir = memory / "findings"
    findings_dir.mkdir(parents=True)
    (findings_dir / "corrections.jsonl").write_text(
        json.dumps({"correction_type": "positive_reinforcement", "raw_text": "good catch"})
        + "\n"
    )

    return curated, memory, runs


class TestWeeklyPromptAssembler:
    def test_assembles_complete_package(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1", "bot2"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()

        assert package.system_prompt
        assert package.task_prompt
        assert package.data
        assert package.instructions
        assert package.corrections is not None
        assert package.context_files is not None

    def test_skips_malformed_weekly_json_and_records_error(self, setup_dirs):
        curated, memory, runs = setup_dirs
        bad_path = curated / "weekly" / "2026-02-23" / "weekly_summary.json"
        bad_path.write_text("{not json", encoding="utf-8")
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )

        package = assembler.assemble()

        assert "weekly_summary" not in package.data
        assert any(e["path"] == str(bad_path) for e in package.data["data_load_errors"])
        assert str(bad_path) not in package.context_files

    def test_system_prompt_includes_policies(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()
        assert "trading assistant" in package.system_prompt
        assert "Max drawdown" in package.system_prompt

    def test_data_includes_weekly_summary(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()
        assert "weekly_summary" in package.data
        assert package.data["weekly_summary"]["total_net_pnl"] == 500.0

    def test_data_includes_daily_reports(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()
        assert "daily_reports" in package.data
        assert len(package.data["daily_reports"]) == 7
        assert any(
            "Fallback path." in report["content"]
            for report in package.data["daily_reports"]
        )

    def test_prefers_daily_report_before_response(self, setup_dirs):
        curated, memory, runs = setup_dirs
        preferred_run = runs / "daily-2026-02-24-preferred"
        preferred_run.mkdir(parents=True)
        (preferred_run / "daily_report.md").write_text("preferred report")
        (preferred_run / "response.md").write_text("legacy fallback")

        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()
        joined = "\n".join(
            report["content"] for report in package.data["daily_reports"]
        )
        assert "preferred report" in joined
        assert "legacy fallback" not in joined

    def test_data_includes_risk_cards(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()
        assert "portfolio_risk_cards" in package.data
        assert len(package.data["portfolio_risk_cards"]) == 7

    def test_non_crypto_health_summaries_use_generic_operational_key(self, setup_dirs):
        curated, memory, runs = setup_dirs
        bot_dir = curated / "2026-02-23" / "ibkr"
        bot_dir.mkdir(parents=True, exist_ok=True)
        (bot_dir / "health_summary.json").write_text(json.dumps({
            "latest_assessment": "degraded",
            "high_severity_alert_count": 2,
        }))
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["ibkr"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )

        package = assembler.assemble()

        assert package.data["health_summaries"][0]["bot_id"] == "ibkr"
        assert package.data["health_summaries"][0]["data"]["latest_assessment"] == "degraded"
        assert "crypto_health_summaries" not in package.data

    def test_data_includes_weekly_funnel_conversion_summary(self, setup_dirs):
        curated, memory, runs = setup_dirs
        for day, setups, fills, closed in [
            ("2026-02-23", 10, 5, 4),
            ("2026-02-24", 20, 10, 6),
        ]:
            bot_dir = curated / day / "crypto"
            bot_dir.mkdir(parents=True, exist_ok=True)
            (bot_dir / "funnel_analysis.json").write_text(json.dumps({
                "stage_totals": {
                    "setups_detected": setups,
                    "confirmations": setups,
                    "entries_attempted": fills,
                    "fills": fills,
                    "trades_closed": closed,
                },
                "per_strategy_breakdown": {
                    "strat1": {
                        "stage_totals": {
                            "setups_detected": setups,
                            "fills": fills,
                            "trades_closed": closed,
                        },
                    },
                },
            }))
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["crypto"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )

        package = assembler.assemble()

        summary = package.data["funnel_weekly_summary"]
        assert summary["coverage_days"] == 2
        assert summary["stage_totals"]["setups_detected"] == 30
        assert summary["stage_totals"]["fills"] == 15
        assert summary["conversion_rates"]["setup_to_fill"] == 0.5
        assert summary["conversion_rates"]["fill_to_close"] == round(10 / 15, 4)
        assert summary["per_strategy_breakdown"]["strat1"]["conversion_rates"]["setup_to_fill"] == 0.5
        assert package.data["crypto_funnel_weekly_summary"] == summary

    def test_data_includes_cross_bot_weekly_funnel_conversion_summary(self, setup_dirs):
        curated, memory, runs = setup_dirs
        cases = [
            ("ibkr", "strat_ibkr", 12, 6, 3),
            ("k_stock", "strat_k", 8, 4, 4),
        ]
        for bot, strategy_id, setups, fills, closed in cases:
            bot_dir = curated / "2026-02-23" / bot
            bot_dir.mkdir(parents=True, exist_ok=True)
            (bot_dir / "funnel_analysis.json").write_text(json.dumps({
                "stage_totals": {
                    "setups_seen": setups,
                    "fills": fills,
                    "trades_closed": closed,
                },
                "per_strategy_breakdown": {
                    strategy_id: {
                        "stage_totals": {
                            "setups_seen": setups,
                            "fills": fills,
                            "trades_closed": closed,
                        },
                    },
                },
            }))
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["ibkr", "k_stock"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )

        package = assembler.assemble()
        summary = package.data["funnel_weekly_summary"]

        assert summary["bots_with_coverage"] == ["ibkr", "k_stock"]
        assert "crypto_funnel_weekly_summary" not in package.data
        assert summary["stage_totals"]["setups_detected"] == 20
        assert summary["conversion_rates"]["setup_to_fill"] == 0.5
        assert summary["conversion_rates"]["fill_to_close"] == round(7 / 10, 4)
        assert summary["per_strategy_breakdown"]["strat_k"]["conversion_rates"]["fill_to_close"] == 1.0

    def test_data_includes_learning_sufficiency_authority_and_gaps(self, setup_dirs):
        curated, memory, runs = setup_dirs
        manifest_path = runs / "monthly-bot1-strat1-2026-02" / "learning_sufficiency_manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "run_month": "2026-02",
            "eligibility": "diagnostics_only",
            "supported_learning_capabilities": ["execution_learning"],
            "blocked_learning_capabilities": ["filter_threshold_learning"],
            "runtime_evidence_support": {
                event_class: {
                    "support_state": "observed",
                    "configured_event_types": [event_class],
                    "observed_event_count": 1,
                    "event_value_classifications": {event_class: "learning_authority"},
                }
                for event_class in ("filter_decision", "order", "fill", "orderbook_context", "trade")
            },
            "known_gaps": [{
                "blocked_learning_capability": "filter_threshold_learning",
                "expected_learning_value": "critical",
                "event_type": "pipeline_funnel",
                "missing_field": "denominator_coverage",
                "remediation": "Emit funnel snapshots for bot1/strat1.",
            }],
        }))
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )

        package = assembler.assemble()

        sufficiency = package.data["learning_sufficiency"]
        assert sufficiency["scope_count"] == 1
        assert sufficiency["supported_capabilities_by_scope"][0]["capabilities"] == ["execution_learning"]
        assert sufficiency["blocked_capabilities_by_scope"][0]["capabilities"] == ["filter_threshold_learning"]
        assert sufficiency["top_learning_gaps"][0]["expected_learning_value"] == "critical"
        assert str(manifest_path) in package.context_files
        assert "LEARNING SUFFICIENCY AUTHORITY" in package.instructions
        assert "supported_learning_capabilities" in package.instructions
        assert "blocked_learning_capabilities" in package.instructions
        assert "diagnostics-only evidence" in package.instructions

    def test_learning_sufficiency_prompt_rejects_non_authority_runtime_sources(self, setup_dirs):
        curated, memory, runs = setup_dirs
        manifest_path = runs / "monthly-bot1-strat1-2026-02" / "learning_sufficiency_manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "run_month": "2026-02",
            "eligibility": "learning_authoritative",
            "supported_learning_capabilities": ["execution_learning"],
            "blocked_learning_capabilities": [],
            "runtime_evidence_support": {
                "trade": {
                    "support_state": "observed",
                    "configured_event_types": ["trade"],
                    "event_value_classifications": {"trade": "operational_health"},
                }
            },
        }))
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )

        sufficiency = assembler.assemble().data["learning_sufficiency"]

        assert sufficiency["supported_capabilities_by_scope"] == []
        assert sufficiency["blocked_capabilities_by_scope"][0]["eligibility"] == "diagnostics_only"
        assert sufficiency["blocked_capabilities_by_scope"][0]["capabilities"] == [
            "runtime_source_classification_violation"
        ]

    def test_learning_sufficiency_prompt_rejects_missing_runtime_support(self, setup_dirs):
        curated, memory, runs = setup_dirs
        manifest_path = runs / "monthly-bot1-strat1-2026-02" / "learning_sufficiency_manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "run_month": "2026-02",
            "eligibility": "learning_authoritative",
            "supported_learning_capabilities": ["execution_learning"],
            "blocked_learning_capabilities": [],
        }))
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )

        sufficiency = assembler.assemble().data["learning_sufficiency"]

        assert sufficiency["supported_capabilities_by_scope"] == []
        assert sufficiency["blocked_capabilities_by_scope"][0]["eligibility"] == "diagnostics_only"
        assert sufficiency["blocked_capabilities_by_scope"][0]["capabilities"] == [
            "runtime_source_classification_violation"
        ]

    def test_task_prompt_mentions_weekly(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()
        assert "weekly" in package.task_prompt.lower()

    def test_package_includes_weekly_focus_rotation_contract(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-06-01",
            week_end="2026-06-07",
            bots=["k_stock_trader", "trading"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()

        assert package.metadata["weekly_focus_id"] == "k_stock_and_trading_stock"
        assert package.data["weekly_focus"]["focus_id"] == "k_stock_and_trading_stock"
        assert package.data["weekly_focus"]["week_start"] == "2026-06-01"
        assert "trading_stock" in package.data["weekly_focus"]["portfolio_families"]
        assert len(package.data["weekly_focus_rotation"]) == 4
        assert "Active weekly focus" in package.task_prompt
        assert "WEEKLY PORTFOLIO FOCUS ROTATION" in package.instructions
        assert "monthly_search_brief only" in package.instructions
        assert "satisfy approval gates" in package.instructions

    def test_corrections_loaded(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()
        assert len(package.corrections) == 1


# ---------------------------------------------------------------------------
# Learning loop gap closure — weekly prompt instruction tests
# ---------------------------------------------------------------------------

class TestWeeklyInstructionGapClosures:
    """Verify weekly prompt instructions have gap closure language."""

    @pytest.fixture()
    def assembler(self, tmp_path: Path):
        curated = tmp_path / "curated"
        curated.mkdir()
        memory = tmp_path / "memory"
        (memory / "policies" / "v1").mkdir(parents=True)
        runs = tmp_path / "runs"
        runs.mkdir()
        return WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )

    def test_instructions_reference_validation_patterns(self, assembler):
        pkg = assembler.assemble()
        assert "validation_patterns" in pkg.instructions

    def test_weekly_constraints_reveal_blocking(self, assembler):
        pkg = assembler.assemble()
        assert "BLOCKED" in pkg.instructions

    def test_require_detector_engagement(self, assembler):
        pkg = assembler.assemble()
        assert "AGREE or DISAGREE" in pkg.instructions

    def test_structured_output_critical_warning(self, assembler):
        pkg = assembler.assemble()
        assert "CRITICAL" in pkg.instructions
        assert "LOST" in pkg.instructions

    def test_spurious_outcomes_quality_note(self, assembler):
        pkg = assembler.assemble()
        assert "low/insufficient measurement quality" in pkg.instructions

    def test_outcome_quality_note_in_constraints(self, assembler):
        """Weekly constraints should state outcome_measurements is quality-filtered."""
        pkg = assembler.assemble()
        assert "HIGH/MEDIUM quality data" in pkg.instructions
