"""Tests for generic context builder (H3), strategy registry enrichment, session history, and triage context."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.schemas.prompt_package import PromptPackage
from trading_assistant.analysis.prompt_assembler import DailyPromptAssembler
from trading_assistant.analysis.weekly_prompt_assembler import WeeklyPromptAssembler
from trading_assistant.schemas.strategy_profile import (
    ArchetypeExpectation,
    CoordinationConfig,
    CoordinationSignal,
    PortfolioConfig,
    StrategyProfile,
    StrategyRegistry,
)
from trading_assistant.orchestrator.session_store import SessionStore
from trading_assistant.schemas.bug_triage import ErrorEvent, BugSeverity, ErrorCategory
from trading_assistant.skills.failure_log import FailureEntry
from trading_assistant.skills.triage_context_builder import TriageContextBuilder, TriageContext


@pytest.fixture
def memory_dir(tmp_path: Path) -> Path:
    """Create a minimal memory directory with policies and corrections."""
    policy_dir = tmp_path / "policies" / "v1"
    policy_dir.mkdir(parents=True)
    (policy_dir / "agent.md").write_text("You are a trading assistant.")
    (policy_dir / "trading_rules.md").write_text("Never risk more than 2%.")
    (policy_dir / "soul.md").write_text("Be cautious and precise.")

    findings_dir = tmp_path / "findings"
    findings_dir.mkdir()
    now = datetime.now(timezone.utc)
    corrections = [
        {
            "date": (now - timedelta(days=2)).date().isoformat(),
            "correction": "Bot-A filter was too aggressive",
        },
        {
            "date": (now - timedelta(days=1)).date().isoformat(),
            "correction": "Risk limit should be 1.5% not 2%",
        },
    ]
    lines = [json.dumps(c) for c in corrections]
    (findings_dir / "corrections.jsonl").write_text("\n".join(lines))

    return tmp_path


@pytest.fixture()
def strategy_registry() -> StrategyRegistry:
    return StrategyRegistry(
        strategies={
            "ATRSS": StrategyProfile(
                display_name="ATR Swing",
                bot_id="swing_multi_01",
                family="swing",
                archetype="trend_follow",
                asset_class="mixed",
                symbols=["QQQ"],
                preferred_regimes=["trending_up"],
                adverse_regimes=["ranging"],
            ),
            "IARIC_v1": StrategyProfile(
                display_name="IARIC",
                bot_id="stock_trader",
                family="stock",
                archetype="intraday_momentum",
                asset_class="equity",
            ),
            "DownturnDominator_v1": StrategyProfile(
                display_name="Downturn Multi-Engine Bear",
                bot_id="momentum_nq_01",
                family="momentum",
                archetype="multi_engine_bear",
                asset_class="futures",
                sub_engines=["REVERSAL", "BREAKDOWN", "FADE"],
                key_metadata_fields=["engine_tag", "composite_regime_at_entry"],
            ),
        },
        coordination=CoordinationConfig(
            signals=[
                CoordinationSignal(
                    trigger={"strategy": "ATRSS", "event": "ENTRY_FILL"},
                    target={"strategy": "AKC_HELIX", "action": "TIGHTEN_STOP_BE"},
                    condition="SAME_SYMBOL",
                ),
            ],
        ),
        portfolio=PortfolioConfig(
            heat_cap_R=2.5,
            portfolio_daily_stop_R=3.0,
            portfolio_weekly_stop_R=5.0,
            family_allocations={"swing": 0.5, "stock": 0.5},
        ),
        archetype_expectations={
            "trend_follow": ArchetypeExpectation(
                expected_win_rate=(0.35, 0.50),
                expected_payoff_ratio=(1.8, 3.0),
                regime_sensitivity="high",
            ),
            "bear_regime_swing": ArchetypeExpectation(
                expected_win_rate=(0.30, 0.45),
                expected_payoff_ratio=(2.0, 4.0),
                regime_sensitivity="high",
            ),
            "multi_engine_bear": ArchetypeExpectation(
                expected_win_rate=(0.35, 0.50),
                expected_payoff_ratio=(1.5, 3.0),
                regime_sensitivity="high",
            ),
            "mean_reversion_pullback": ArchetypeExpectation(
                expected_win_rate=(0.55, 0.70),
                expected_payoff_ratio=(0.8, 1.5),
                regime_sensitivity="medium",
            ),
        },
    )


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(base_dir=str(tmp_path / "sessions"))


class TestContextBuilder:
    def test_build_system_prompt_loads_all_policies(self, memory_dir: Path):
        ctx = ContextBuilder(memory_dir)
        prompt = ctx.build_system_prompt()
        assert "--- agent.md ---" in prompt
        assert "trading assistant" in prompt
        assert "--- trading_rules.md ---" in prompt
        assert "--- soul.md ---" in prompt

    def test_build_system_prompt_skips_missing_files(self, tmp_path: Path):
        policy_dir = tmp_path / "policies" / "v1"
        policy_dir.mkdir(parents=True)
        (policy_dir / "agent.md").write_text("Agent only.")
        ctx = ContextBuilder(tmp_path)
        prompt = ctx.build_system_prompt()
        assert "--- agent.md ---" in prompt
        assert "trading_rules.md" not in prompt

    def test_load_corrections(self, memory_dir: Path):
        ctx = ContextBuilder(memory_dir)
        corrections = ctx.load_corrections()
        assert len(corrections) == 2
        # Temporal decay sorts by recency (most recent first)
        assert corrections[0]["correction"] == "Risk limit should be 1.5% not 2%"

    def test_load_corrections_returns_empty_when_missing(self, tmp_path: Path):
        ctx = ContextBuilder(tmp_path)
        corrections = ctx.load_corrections()
        assert corrections == []

    def test_list_policy_files(self, memory_dir: Path):
        ctx = ContextBuilder(memory_dir)
        files = ctx.list_policy_files()
        assert len(files) == 3
        assert any("agent.md" in f for f in files)

    def test_runtime_metadata(self, memory_dir: Path):
        ctx = ContextBuilder(memory_dir)
        meta = ctx.runtime_metadata()
        assert "assembled_at" in meta
        assert meta["timezone"] == "UTC"

    def test_base_package_returns_prompt_package(self, memory_dir: Path):
        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert isinstance(pkg, PromptPackage)
        assert "trading assistant" in pkg.system_prompt
        assert len(pkg.corrections) == 2
        assert len(pkg.context_files) == 3
        assert pkg.metadata["timezone"] == "UTC"

    def test_base_package_large_history_latency(self, memory_dir: Path):
        findings_dir = memory_dir / "findings"
        now = datetime.now(timezone.utc)
        old_records = [
            json.dumps({
                "timestamp": (now - timedelta(days=120)).isoformat(),
                "bot_id": "bot1",
                "correction": f"old correction {i}",
            })
            for i in range(12_000)
        ]
        recent_records = [
            json.dumps({
                "timestamp": (now - timedelta(minutes=i)).isoformat(),
                "bot_id": "bot1",
                "correction": f"recent correction {i}",
            })
            for i in range(1_200)
        ]
        (findings_dir / "corrections.jsonl").write_text(
            "\n".join(old_records + recent_records),
            encoding="utf-8",
        )
        failure_records = [
            json.dumps({
                "timestamp": (now - timedelta(minutes=i)).isoformat(),
                "bot_id": "bot1",
                "error_type": "synthetic",
            })
            for i in range(3_000)
        ]
        (findings_dir / "failure-log.jsonl").write_text(
            "\n".join(failure_records),
            encoding="utf-8",
        )

        started = time.perf_counter()
        pkg = ContextBuilder(memory_dir).base_package(
            agent_type="daily_analysis",
            bot_id="bot1",
            record_retrieval=False,
        )
        elapsed = time.perf_counter() - started

        assert isinstance(pkg, PromptPackage)
        assert elapsed < 5.0
        assert pkg.corrections
        assert all("recent correction" in item["correction"] for item in pkg.corrections)
        assert len(pkg.data.get("failure_log", [])) <= 50

    def test_assemblers_use_context_builder(self, memory_dir: Path, tmp_path: Path):
        """Integration test: DailyPromptAssembler uses ContextBuilder."""
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot-a"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        pkg = assembler.assemble()
        assert isinstance(pkg, PromptPackage)
        assert "trading assistant" in pkg.system_prompt
        assert len(pkg.corrections) == 2
        assert "bot-a" in pkg.task_prompt


class TestLoadFailureLog:
    def test_loads_failure_log_entries(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        log_path = findings / "failure-log.jsonl"
        log_path.write_text(
            '{"error_type":"timeout","bot_id":"bot1","outcome":"known_fix"}\n'
            '{"error_type":"api_error","bot_id":"bot2","outcome":"needs_human"}\n'
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        entries = ctx.load_failure_log()
        assert len(entries) == 2
        assert entries[0]["error_type"] == "timeout"

    def test_missing_failure_log_returns_empty(self, tmp_path):
        ctx = ContextBuilder(memory_dir=tmp_path)
        assert ctx.load_failure_log() == []


class TestLoadRejectedSuggestions:
    def test_loads_rejected_suggestions(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        suggestions_path = findings / "suggestions.jsonl"
        suggestions_path.write_text(
            '{"suggestion_id":"s001","bot_id":"bot1","title":"Widen stop","status":"rejected","rejection_reason":"No evidence"}\n'
            '{"suggestion_id":"s002","bot_id":"bot1","title":"Remove filter","status":"deployed"}\n'
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        rejected = ctx.load_rejected_suggestions()
        assert len(rejected) == 1
        assert rejected[0]["title"] == "Widen stop"

    def test_missing_suggestions_file_returns_empty(self, tmp_path):
        ctx = ContextBuilder(memory_dir=tmp_path)
        assert ctx.load_rejected_suggestions() == []


class TestLoadRecentProposalOutcomes:
    def test_includes_old_candidate_with_recent_outcome(self, tmp_path: Path):
        from trading_assistant.schemas.proposal_ledger import (
            ProposalCandidate,
            ProposalKind,
            ProposalOutcome,
            ProposalSource,
        )
        from trading_assistant.skills.proposal_ledger import ProposalLedger

        findings = tmp_path / "findings"
        ledger = ProposalLedger(findings)
        ledger.record_candidate(ProposalCandidate(
            proposal_id="old-candidate",
            source=ProposalSource.LLM_DAILY,
            kind=ProposalKind.PARAMETER_CHANGE,
            bot_id="bot1",
            title="Old proposal with fresh evidence",
            proposed_at=datetime.now(timezone.utc) - timedelta(days=45),
        ))
        ledger.record_outcome("old-candidate", ProposalOutcome(
            proposal_id="old-candidate",
            verdict="positive",
            objective_delta=0.08,
            measured_at=datetime.now(timezone.utc),
        ))

        ctx = ContextBuilder(memory_dir=tmp_path)
        outcomes = ctx.load_recent_proposal_outcomes("bot1", days=30)

        assert len(outcomes) == 1
        assert outcomes[0]["proposal_id"] == "old-candidate"
        assert outcomes[0]["title"] == "Old proposal with fresh evidence"


class TestBasePackageWithFailureLog:
    def test_base_package_includes_failure_log(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        (findings / "failure-log.jsonl").write_text(
            '{"error_type":"timeout","outcome":"known_fix"}\n'
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        pkg = ctx.base_package()
        assert "failure_log" in pkg.data
        assert len(pkg.data["failure_log"]) == 1

    def test_base_package_includes_rejected_suggestions(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        (findings / "suggestions.jsonl").write_text(
            '{"suggestion_id":"s001","status":"rejected","rejection_reason":"x"}\n'
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        pkg = ctx.base_package()
        assert "rejected_suggestions" in pkg.data
        assert len(pkg.data["rejected_suggestions"]) == 1


class TestLoadAllocationHistory:
    def test_loads_allocation_records(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        now = datetime.now(timezone.utc)
        (findings / "allocation_history.jsonl").write_text(
            json.dumps(
                {
                    "date": (now - timedelta(days=2)).date().isoformat(),
                    "bot_id": "bot-a",
                    "allocation_pct": 25.0,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "date": (now - timedelta(days=1)).date().isoformat(),
                    "bot_id": "bot-a",
                    "allocation_pct": 30.0,
                }
            )
            + "\n"
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        records = ctx.load_allocation_history()
        assert len(records) == 2
        # Temporal decay: most recent first
        assert records[0]["date"] == (now - timedelta(days=1)).date().isoformat()

    def test_missing_file_returns_empty(self, tmp_path):
        ctx = ContextBuilder(memory_dir=tmp_path)
        assert ctx.load_allocation_history() == []

    def test_temporal_decay_excludes_old_records(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        recent_date = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
        (findings / "allocation_history.jsonl").write_text(
            '{"date":"2020-01-01","bot_id":"bot-a","allocation_pct":10.0}\n'
            + json.dumps(
                {
                    "date": recent_date,
                    "bot_id": "bot-a",
                    "allocation_pct": 25.0,
                }
            )
            + "\n"
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        records = ctx.load_allocation_history()
        assert len(records) == 1
        assert records[0]["date"] == recent_date

    def test_base_package_includes_allocation_history(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        recent_date = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
        (findings / "allocation_history.jsonl").write_text(
            json.dumps(
                {
                    "date": recent_date,
                    "bot_id": "bot-a",
                    "allocation_pct": 25.0,
                }
            )
            + "\n"
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        pkg = ctx.base_package()
        assert "allocation_history" in pkg.data
        assert len(pkg.data["allocation_history"]) == 1

    def test_base_package_excludes_empty_allocation_history(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        ctx = ContextBuilder(memory_dir=tmp_path)
        pkg = ctx.base_package()
        assert "allocation_history" not in pkg.data


class TestLoadSearchReports:
    def test_loads_recent_search_reports(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        reports = [
            {"suggestion_id": f"s{i}", "bot_id": "bot1", "param_name": "p1",
             "routing": "approve", "best_value": 0.7, "discard_reason": "",
             "exploration_summary": "ok", "searched_at": "2026-03-01T00:00:00Z"}
            for i in range(8)
        ]
        with (findings / "search_reports.jsonl").open("w") as f:
            for r in reports:
                f.write(json.dumps(r) + "\n")
        ctx = ContextBuilder(memory_dir=tmp_path)
        result = ctx.load_search_reports(lookback_n=5)
        assert len(result) == 5  # Only last 5

    def test_filters_by_bot_id(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        reports = [
            {"bot_id": "bot1", "param_name": "p1", "routing": "approve"},
            {"bot_id": "bot2", "param_name": "p2", "routing": "discard"},
        ]
        with (findings / "search_reports.jsonl").open("w") as f:
            for r in reports:
                f.write(json.dumps(r) + "\n")
        ctx = ContextBuilder(memory_dir=tmp_path)
        result = ctx.load_search_reports(bot_id="bot1")
        assert len(result) == 1
        assert result[0]["param_name"] == "p1"

    def test_base_package_includes_search_reports(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        (findings / "search_reports.jsonl").write_text(
            json.dumps({"bot_id": "bot1", "param_name": "p1", "routing": "approve"}) + "\n"
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        pkg = ctx.base_package()
        assert "search_reports" in pkg.data
        assert len(pkg.data["search_reports"]) == 1


class TestLoadBacktestReliability:
    def test_loads_per_category_reliability(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        records = []
        for i in range(5):
            records.append({
                "suggestion_id": f"s{i}", "bot_id": "bot1",
                "param_category": "signal", "predicted_improvement": 1.1,
                "predicted_routing": "approve",
                "prediction_correct": i < 4,  # 4/5 = 0.80
            })
        with (findings / "backtest_calibration.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        ctx = ContextBuilder(memory_dir=tmp_path)
        result = ctx.load_backtest_reliability()
        assert "signal" in result
        assert result["signal"] == 0.8

    def test_excludes_categories_below_min_samples(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        records = [
            {"bot_id": "bot1", "param_category": "exit",
             "prediction_correct": True},
            {"bot_id": "bot1", "param_category": "exit",
             "prediction_correct": False},
        ]
        with (findings / "backtest_calibration.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        ctx = ContextBuilder(memory_dir=tmp_path)
        result = ctx.load_backtest_reliability()
        assert "exit" not in result  # Only 2 samples, need >= 3

    def test_base_package_includes_backtest_reliability(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        records = [
            {"bot_id": "bot1", "param_category": "signal",
             "prediction_correct": True}
            for _ in range(4)
        ]
        with (findings / "backtest_calibration.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        ctx = ContextBuilder(memory_dir=tmp_path)
        pkg = ctx.base_package()
        assert "backtest_reliability" in pkg.data
        assert pkg.data["backtest_reliability"]["signal"] == 1.0


# --- Tests from test_context_enrichment.py (strategy registry context enrichment) ---


class TestBasePackageInjection:
    def test_strategy_profiles_injected(self, tmp_path: Path, strategy_registry: StrategyRegistry) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test", encoding="utf-8")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package(strategy_registry=strategy_registry)

        assert "strategy_profiles" in pkg.data
        assert "ATRSS" in pkg.data["strategy_profiles"]
        assert pkg.data["strategy_profiles"]["ATRSS"]["bot_id"] == "swing_multi_01"

    def test_coordination_rules_injected(self, tmp_path: Path, strategy_registry: StrategyRegistry) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test", encoding="utf-8")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package(strategy_registry=strategy_registry)

        assert "coordination_rules" in pkg.data
        assert len(pkg.data["coordination_rules"]["signals"]) == 1

    def test_archetype_expectations_injected(self, tmp_path: Path, strategy_registry: StrategyRegistry) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test", encoding="utf-8")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package(strategy_registry=strategy_registry)

        assert "archetype_expectations" in pkg.data
        assert "trend_follow" in pkg.data["archetype_expectations"]

    def test_portfolio_risk_config_injected(self, tmp_path: Path, strategy_registry: StrategyRegistry) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test", encoding="utf-8")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package(strategy_registry=strategy_registry)

        assert "portfolio_risk_config" in pkg.data
        assert pkg.data["portfolio_risk_config"]["heat_cap_R"] == 2.5

    def test_no_registry_no_injection(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test", encoding="utf-8")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package(strategy_registry=None)

        assert "strategy_profiles" not in pkg.data
        assert "coordination_rules" not in pkg.data

    def test_empty_registry_no_injection(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test", encoding="utf-8")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package(strategy_registry=StrategyRegistry())

        assert "strategy_profiles" not in pkg.data

    def test_new_archetype_expectations_injected(self, tmp_path: Path, strategy_registry: StrategyRegistry) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test", encoding="utf-8")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package(strategy_registry=strategy_registry)

        expectations = pkg.data["archetype_expectations"]
        for arch in ("bear_regime_swing", "multi_engine_bear", "mean_reversion_pullback"):
            assert arch in expectations, f"Missing {arch} in archetype_expectations"

    def test_profile_metadata_in_context(self, tmp_path: Path, strategy_registry: StrategyRegistry) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test", encoding="utf-8")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package(strategy_registry=strategy_registry)

        profiles = pkg.data["strategy_profiles"]
        downturn = profiles["DownturnDominator_v1"]
        assert downturn["sub_engines"] == ["REVERSAL", "BREAKDOWN", "FADE"]
        assert downturn["key_metadata_fields"] == ["engine_tag", "composite_regime_at_entry"]


class TestContextPriority:
    def test_strategy_profiles_in_priority_list(self) -> None:
        assert "strategy_profiles" in ContextBuilder._CONTEXT_PRIORITY

    def test_strategy_profiles_after_ground_truth(self) -> None:
        idx_gt = ContextBuilder._CONTEXT_PRIORITY.index("ground_truth_trend")
        idx_sp = ContextBuilder._CONTEXT_PRIORITY.index("strategy_profiles")
        assert idx_sp > idx_gt  # strategy_profiles comes after ground_truth_trend


class TestAssemblerPassthrough:
    def test_daily_assembler_accepts_strategy_registry(self, tmp_path: Path, strategy_registry: StrategyRegistry) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test", encoding="utf-8")

        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["swing_multi_01"],
            curated_dir=tmp_path / "curated",
            memory_dir=memory_dir,
            strategy_registry=strategy_registry,
        )
        assert assembler.strategy_registry is strategy_registry

    def test_weekly_assembler_accepts_strategy_registry(self, tmp_path: Path, strategy_registry: StrategyRegistry) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test", encoding="utf-8")
        (memory_dir / "policies" / "v1" / "agent.md").write_text("test", encoding="utf-8")

        assembler = WeeklyPromptAssembler(
            week_start="2026-03-01",
            week_end="2026-03-07",
            bots=["swing_multi_01"],
            curated_dir=tmp_path / "curated",
            memory_dir=memory_dir,
            runs_dir=tmp_path / "runs",
            strategy_registry=strategy_registry,
        )
        assert assembler.strategy_registry is strategy_registry


# --- Tests from test_session_history_context.py (session history in context, A5) ---


class TestSessionHistoryContext:
    def test_get_recent_sessions_returns_sessions(self, session_store):
        # Record a session
        session_store.record_invocation(
            session_id="sess-1",
            agent_type="daily_analysis",
            prompt_package={"task": "test"},
            response="Analysis complete for all bots",
            duration_ms=5000,
        )

        result = session_store.get_recent_sessions("daily_analysis", days=7)
        assert len(result) == 1
        assert result[0]["agent_type"] == "daily_analysis"
        assert result[0]["duration_ms"] == 5000
        assert "Analysis complete" in result[0]["response_summary"]

    def test_load_session_history_formats_text(self, memory_dir, session_store):
        session_store.record_invocation(
            session_id="sess-1",
            agent_type="daily_analysis",
            prompt_package={},
            response="Daily report generated",
            duration_ms=3000,
            metadata={
                "provider": "claude_max",
                "effective_model": "sonnet",
                "first_output_ms": 125,
                "tool_call_count": 2,
                "stream_event_count": 8,
                "auth_mode": "claude.ai:max",
            },
        )

        ctx = ContextBuilder(memory_dir)
        result = ctx.load_session_history(session_store, "daily_analysis")
        assert "Recent daily_analysis sessions" in result
        assert "3000ms" in result
        assert "claude_max/sonnet" in result
        assert "first 125ms" in result
        assert "tools 2" in result
        assert "stream 8" in result
        assert "claude.ai:max" in result

    def test_base_package_includes_session_history(self, memory_dir, session_store):
        session_store.record_invocation(
            session_id="sess-1",
            agent_type="daily_analysis",
            prompt_package={},
            response="Daily report output",
            duration_ms=2000,
        )

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package(session_store=session_store, agent_type="daily_analysis")
        assert "session_history" in pkg.data
        assert "daily_analysis" in pkg.data["session_history"]

    def test_empty_session_history_handled_gracefully(self, memory_dir, session_store):
        ctx = ContextBuilder(memory_dir)
        result = ctx.load_session_history(session_store, "nonexistent_type")
        assert result == ""

    def test_session_summary_truncated(self, memory_dir, session_store):
        long_response = "x" * 1000
        session_store.record_invocation(
            session_id="sess-1",
            agent_type="daily_analysis",
            prompt_package={},
            response=long_response,
            duration_ms=1000,
        )

        ctx = ContextBuilder(memory_dir)
        result = ctx.load_session_history(session_store, "daily_analysis")
        # response_summary is capped at 200 chars, then we take 100 in format
        assert len(result) < 500

    def test_filters_by_agent_type(self, session_store):
        session_store.record_invocation(
            session_id="sess-1",
            agent_type="daily_analysis",
            prompt_package={},
            response="daily",
            duration_ms=1000,
        )
        session_store.record_invocation(
            session_id="sess-2",
            agent_type="weekly_analysis",
            prompt_package={},
            response="weekly",
            duration_ms=2000,
        )

        daily = session_store.get_recent_sessions("daily_analysis", days=7)
        weekly = session_store.get_recent_sessions("weekly_analysis", days=7)

        assert all(s["agent_type"] == "daily_analysis" for s in daily)
        assert all(s["agent_type"] == "weekly_analysis" for s in weekly)


# --- Tests from test_triage_context_builder.py (triage context builder) ---


class TestTriageContext:
    def test_has_required_fields(self):
        ctx = TriageContext(
            error_event_summary="RuntimeError: division by zero in calc.py:10",
            stack_trace="Traceback...",
            source_snippet="",
            recent_git_log="",
            past_rejections=[],
        )
        assert ctx.error_event_summary != ""
        assert ctx.stack_trace != ""


class TestTriageContextBuilder:
    def test_builds_context_with_source_file(self, tmp_path: Path):
        # Create a fake source file
        src = tmp_path / "calc.py"
        src.write_text("def divide(x, y):\n    return x / y\n")

        e = ErrorEvent(
            bot_id="bot1", error_type="RuntimeError",
            message="division by zero",
            stack_trace='Traceback:\n  File "calc.py", line 2\n    return x / y\nRuntimeError: division by zero',
            source_file="calc.py",
            source_line=2,
        )

        builder = TriageContextBuilder(source_root=tmp_path)
        ctx = builder.build(e, BugSeverity.HIGH, ErrorCategory.UNKNOWN, [])

        assert "RuntimeError" in ctx.error_event_summary
        assert "division by zero" in ctx.error_event_summary
        assert "return x / y" in ctx.source_snippet

    def test_builds_context_without_source_file(self, tmp_path: Path):
        e = ErrorEvent(
            bot_id="bot1", error_type="ConnectionError",
            message="connection lost",
            stack_trace="...",
        )

        builder = TriageContextBuilder(source_root=tmp_path)
        ctx = builder.build(e, BugSeverity.CRITICAL, ErrorCategory.CONNECTION_LOST, [])

        assert ctx.source_snippet == ""
        assert "ConnectionError" in ctx.error_event_summary

    def test_includes_past_rejections(self, tmp_path: Path):
        e = ErrorEvent(
            bot_id="bot1", error_type="ImportError",
            message="no module foo",
            stack_trace="...",
        )

        past = [
            FailureEntry(
                bot_id="bot1", error_type="ImportError", message="no module foo",
                outcome="known_fix", rejection_reason="wrong version",
            ),
        ]

        builder = TriageContextBuilder(source_root=tmp_path)
        ctx = builder.build(e, BugSeverity.HIGH, ErrorCategory.DEPENDENCY, past)

        assert len(ctx.past_rejections) == 1
        assert "wrong version" in ctx.past_rejections[0]

    def test_source_snippet_includes_surrounding_lines(self, tmp_path: Path):
        src = tmp_path / "module.py"
        lines = [f"line {i}\n" for i in range(1, 21)]
        src.write_text("".join(lines))

        e = ErrorEvent(
            bot_id="bot1", error_type="E", message="m",
            stack_trace="...", source_file="module.py", source_line=10,
        )

        builder = TriageContextBuilder(source_root=tmp_path, context_lines=3)
        ctx = builder.build(e, BugSeverity.HIGH, ErrorCategory.UNKNOWN, [])

        # Should include lines 7-13 (3 before, target, 3 after)
        assert "line 7" in ctx.source_snippet
        assert "line 10" in ctx.source_snippet
        assert "line 13" in ctx.source_snippet


# ---------------------------------------------------------------------------
# Phase 1: Outcome measurement quality gate
# ---------------------------------------------------------------------------

class TestOutcomeQualityFilter:
    """Tests for load_outcome_measurements() quality filtering."""

    def _write_outcomes(self, tmp_path: Path, entries: list[dict]) -> None:
        findings = tmp_path / "findings"
        findings.mkdir(exist_ok=True)
        lines = [json.dumps(e) for e in entries]
        (findings / "outcomes.jsonl").write_text("\n".join(lines))

    def test_excludes_low_and_insufficient(self, tmp_path: Path):
        self._write_outcomes(tmp_path, [
            {"suggestion_id": "s1", "verdict": "positive", "measurement_quality": "high"},
            {"suggestion_id": "s2", "verdict": "negative", "measurement_quality": "low"},
            {"suggestion_id": "s3", "verdict": "positive", "measurement_quality": "insufficient"},
        ])
        ctx = ContextBuilder(memory_dir=tmp_path)
        reliable, low_q = ctx.load_outcome_measurements()
        assert len(reliable) == 1
        assert reliable[0]["suggestion_id"] == "s1"
        assert len(low_q) == 2

    def test_includes_high_and_medium(self, tmp_path: Path):
        self._write_outcomes(tmp_path, [
            {"suggestion_id": "s1", "measurement_quality": "high"},
            {"suggestion_id": "s2", "measurement_quality": "medium"},
        ])
        ctx = ContextBuilder(memory_dir=tmp_path)
        reliable, low_q = ctx.load_outcome_measurements()
        assert len(reliable) == 2
        assert len(low_q) == 0

    def test_backward_compat_no_quality_field(self, tmp_path: Path):
        """Entries without measurement_quality should be included in reliable."""
        self._write_outcomes(tmp_path, [
            {"suggestion_id": "s1", "verdict": "positive"},
            {"suggestion_id": "s2", "verdict": "negative", "measurement_quality": "low"},
        ])
        ctx = ContextBuilder(memory_dir=tmp_path)
        reliable, low_q = ctx.load_outcome_measurements()
        assert len(reliable) == 1
        assert reliable[0]["suggestion_id"] == "s1"
        assert len(low_q) == 1

    def test_low_quality_in_spurious_via_base_package(self, tmp_path: Path):
        """LOW/INSUFFICIENT outcomes should appear in spurious_outcomes in base_package."""
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        self._write_outcomes(tmp_path, [
            {"suggestion_id": "s1", "measurement_quality": "high"},
            {"suggestion_id": "s2", "measurement_quality": "low"},
            {"suggestion_id": "s3", "measurement_quality": "insufficient"},
        ])
        ctx = ContextBuilder(memory_dir=tmp_path)
        pkg = ctx.base_package()
        # Reliable outcomes only
        assert len(pkg.data.get("outcome_measurements", [])) == 1
        # Low-quality merged into spurious
        spurious = pkg.data.get("spurious_outcomes", [])
        assert len(spurious) == 2
        sids = {e["suggestion_id"] for e in spurious}
        assert sids == {"s2", "s3"}

    def test_empty_file_returns_empty_tuples(self, tmp_path: Path):
        ctx = ContextBuilder(memory_dir=tmp_path)
        reliable, low_q = ctx.load_outcome_measurements()
        assert reliable == []
        assert low_q == []

    def test_dedup_still_works_with_quality_filter(self, tmp_path: Path):
        """Last-write-wins dedup should still apply before quality filtering."""
        self._write_outcomes(tmp_path, [
            {"suggestion_id": "s1", "measurement_quality": "low", "verdict": "negative"},
            {"suggestion_id": "s1", "measurement_quality": "high", "verdict": "positive"},
        ])
        ctx = ContextBuilder(memory_dir=tmp_path)
        reliable, low_q = ctx.load_outcome_measurements()
        # s1 was overwritten to high quality
        assert len(reliable) == 1
        assert reliable[0]["measurement_quality"] == "high"
        assert len(low_q) == 0


class TestBotAwareCardRetrieval:
    """Phase B: bot_id parameter flows through base_package to ranked_for_prompt."""

    def test_base_package_passes_bot_id_to_ranked_for_prompt(self, tmp_path: Path):
        """base_package(bot_id='bot_a') should pass bot_id to card_store.ranked_for_prompt."""
        policy_dir = tmp_path / "policies" / "v1"
        policy_dir.mkdir(parents=True)
        (policy_dir / "agent.md").write_text("agent")
        (policy_dir / "soul.md").write_text("soul")
        (policy_dir / "trading_rules.md").write_text("rules")
        (tmp_path / "findings").mkdir()

        from unittest.mock import patch

        ctx = ContextBuilder(tmp_path)
        with patch("trading_assistant.skills.learning_card_store.LearningCardStore") as MockStore:
            mock_instance = MockStore.return_value
            mock_instance.ranked_for_prompt.return_value = []
            ctx.base_package(bot_id="bot_a", agent_type="daily_analysis")
            mock_instance.ranked_for_prompt.assert_called_once_with(
                limit=10,
                bot_id="bot_a",
                workflow="daily_analysis",
                tags=["workflow:daily_analysis", "bot:bot_a"],
            )

    def test_base_package_defaults_bot_id_to_empty(self, tmp_path: Path):
        """base_package() without bot_id should default to empty string."""
        policy_dir = tmp_path / "policies" / "v1"
        policy_dir.mkdir(parents=True)
        (policy_dir / "agent.md").write_text("agent")
        (policy_dir / "soul.md").write_text("soul")
        (policy_dir / "trading_rules.md").write_text("rules")
        (tmp_path / "findings").mkdir()

        from unittest.mock import patch

        ctx = ContextBuilder(tmp_path)
        with patch("trading_assistant.skills.learning_card_store.LearningCardStore") as MockStore:
            mock_instance = MockStore.return_value
            mock_instance.ranked_for_prompt.return_value = []
            ctx.base_package(agent_type="weekly_analysis")
            mock_instance.ranked_for_prompt.assert_called_once_with(
                limit=10,
                bot_id="",
                workflow="weekly_analysis",
                tags=["workflow:weekly_analysis"],
            )

    def test_daily_assembler_single_bot_passes_bot_id(self, tmp_path: Path):
        """Daily assembler with single bot should pass bots[0] as bot_id."""
        from unittest.mock import patch

        memory_dir = tmp_path / "memory"
        policy_dir = memory_dir / "policies" / "v1"
        policy_dir.mkdir(parents=True)
        (policy_dir / "agent.md").write_text("agent")
        (policy_dir / "soul.md").write_text("soul")
        (policy_dir / "trading_rules.md").write_text("rules")
        (memory_dir / "findings").mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        assembler = DailyPromptAssembler(
            date="2026-03-01", bots=["single_bot"], curated_dir=curated, memory_dir=memory_dir,
        )
        with patch.object(ContextBuilder, "base_package", wraps=assembler._ctx.base_package) as mock_bp:
            assembler.assemble()
            mock_bp.assert_called_once()
            assert mock_bp.call_args.kwargs.get("bot_id") == "single_bot"
