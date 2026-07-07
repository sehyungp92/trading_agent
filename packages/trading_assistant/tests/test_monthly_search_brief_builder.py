from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.strategy_change_ledger import RollbackStatus, StrategyChangeRecord, StrategyChangeRecordType
from trading_assistant.schemas.weekly_focus_rotation import weekly_focus_for_week
from trading_assistant.skills.monthly_search_brief_builder import MonthlySearchBriefBuilder
from trading_assistant.skills.strategy_change_ledger import StrategyChangeLedger


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_monthly_search_brief_is_bounded_and_non_authoritative(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    _write_jsonl(findings / "retrospective_synthesis.jsonl", [{
        "week_start": "2026-05-18",
        "keep": [{"category": "filter_threshold", "summary": "Filter recovered in trend regime"}],
        "discard": [{"category": "stop_loss", "summary": "Noisy one-week loss pattern"}],
    }])
    _write_jsonl(findings / "outcome_priors.jsonl", [{
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "mutation_family": "stop_loss",
        "category": "stop_loss",
        "negative_count": 1,
    }])
    _write_jsonl(findings / "hypotheses.jsonl", [{
        "id": "h-filter",
        "title": "Filter threshold hypothesis",
        "category": "filter_threshold",
        "description": "",
        "status": "active",
    }])
    _write_jsonl(findings / "structural_experiments.jsonl", [{
        "experiment_id": "exp-1",
        "bot_id": "bot1",
        "title": "Regime gate experiment",
        "hypothesis_id": "h-regime",
        "status": "active",
    }])
    _write_jsonl(findings / "category_overrides.jsonl", [{
        "bot_id": "bot1",
        "category": "stop_loss",
        "confidence_multiplier": 0.3,
    }])
    _write_jsonl(findings / "monthly_outcomes.jsonl", [{
        "outcome_id": "out-rollback",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "category": "stop_loss",
        "mutation_family": "stop_loss",
        "verdict": "rollback",
        "run_month": "2026-04",
    }])

    brief = MonthlySearchBriefBuilder(findings).build(
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        week_start="2026-06-01",
    )

    assert brief.report_only is True
    assert brief.weekly_focus["focus_id"] == "k_stock_and_trading_stock"
    assert brief.weekly_focus["week_start"] == "2026-06-01"
    assert "k_stock_olr_kalcb" in brief.weekly_focus["portfolio_families"]
    assert "trading_stock" in brief.weekly_focus["portfolio_families"]
    assert brief.experiment_focus_hints
    assert brief.negative_priors
    assert any(item.get("seed_type") == "active_hypothesis" for item in brief.seed_candidates)
    assert any(item.get("seed_type") == "active_structural_experiment" for item in brief.seed_candidates)
    assert any(item.get("category") == "stop_loss" for item in brief.confidence_caps)
    assert any(item.get("source_weekly_signal_id") == "out-rollback" for item in brief.rollback_candidates)
    assert all(item.get("authority") == "search_order_only" for item in brief.seed_candidates)
    assert "if OOS repair triggers, inspect negative-prior families early in the ablation queue" in brief.phase_order_hints
    guidance = brief.to_optimizer_guidance()
    assert guidance["authority"] == "search_order_only"
    assert guidance["plan_requirements"]["weekly_focus_id"] == "k_stock_and_trading_stock"
    assert "trading_stock" in guidance["plan_requirements"]["focus_portfolio_families"]
    assert "IARIC_v1" in guidance["plan_requirements"]["focus_strategy_ids"]
    assert guidance["rollback_candidates"]
    assert "stop_loss" in guidance["plan_requirements"]["rollback_families"]
    plan = brief.apply_to_experiment_plan_payload(
        {
            "run_id": "run-1",
            "score_components": ["expected_return"],
            "candidate_families": [],
            "gate_expectations": [],
            "overfit_risks": [],
            "evidence_paths": [],
        },
        brief_path="/tmp/monthly_search_brief.json",
    )
    assert "/tmp/monthly_search_brief.json" in plan["evidence_paths"]
    assert plan["source_weekly_signal_ids"]
    assert any(item.startswith("brief_") for item in plan["phase_order"])
    assert any(item["family"] == "filter_threshold" for item in plan["candidate_families"])
    assert any(item["family"] == "trading_stock" for item in plan["candidate_families"])
    assert any(item["family"] == "k_stock_olr_kalcb" for item in plan["candidate_families"])
    assert any("approval gates" in expectation for expectation in plan["gate_expectations"])
    assert any("weekly_focus" in expectation for expectation in plan["gate_expectations"])
    assert any("stop_loss" in risk for risk in plan["overfit_risks"])


def test_monthly_search_brief_reads_strategy_change_ledger_projection(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    record = StrategyChangeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        record_type=StrategyChangeRecordType.ROLLBACK,
        rollback_status=RollbackStatus.RECOMMENDED,
        decision_reason="monthly result recommended rollback",
        run_id="monthly-bot1-strat1-2026-04",
        run_month="2026-04",
    )
    StrategyChangeLedger(findings).record(record)

    brief = MonthlySearchBriefBuilder(findings).build(
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
    )

    assert any(
        item.get("source_weekly_signal_id") == record.record_id
        for item in brief.rollback_candidates
    )


def test_weekly_focus_rotation_contract_cycles_by_portfolio_family() -> None:
    assert weekly_focus_for_week("2026-06-01").focus_id == "k_stock_and_trading_stock"
    assert weekly_focus_for_week("2026-06-08").focus_id == "trading_momentum"
    assert weekly_focus_for_week("2026-06-15").focus_id == "trading_swing"
    assert weekly_focus_for_week("2026-06-22").focus_id == "crypto_trader"
    assert weekly_focus_for_week("2026-06-29").focus_id == "k_stock_and_trading_stock"
