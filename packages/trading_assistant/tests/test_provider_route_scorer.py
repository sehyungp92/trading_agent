from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus
from trading_assistant.skills.monthly_outcome_measurer import MonthlyOutcomeMeasurer
from trading_assistant.skills.provider_route_scorer import ProviderRouteScorer


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def test_recompute_scores_from_validation_outcomes_and_recalibration(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    _write_jsonl(findings / "suggestions.jsonl", [{
        "suggestion_id": "s1",
        "source_report_id": "daily-2026-04-20",
        "detection_context": {
            "source_provider": "codex_pro",
            "source_model": "gpt-5.4",
        },
    }])
    _write_jsonl(findings / "validation_log.jsonl", [{
        "agent_type": "daily_analysis",
        "provider": "codex_pro",
        "model": "gpt-5.4",
        "approved_count": 4,
        "blocked_count": 1,
    }])
    _write_jsonl(findings / "outcomes.jsonl", [{
        "suggestion_id": "s1",
        "source_run_id": "daily-2026-04-20",
        "source_provider": "codex_pro",
        "source_model": "gpt-5.4",
        "verdict": "positive",
        "measurement_quality": "high",
    }])
    _write_jsonl(findings / "recalibrations.jsonl", [{
        "suggestion_id": "s1",
        "revised_confidence": 0.6,
        "original_confidence": 0.7,
    }])

    scores = ProviderRouteScorer(findings).recompute()

    assert len(scores) == 1
    assert scores[0]["workflow"] == "daily_analysis"
    assert scores[0]["provider"] == "codex_pro"
    assert scores[0]["sample_count"] == 3
    assert scores[0]["composite_score"] > 0.7


def test_recompute_scores_include_provider_benchmarks(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    _write_jsonl(findings / "provider_benchmark_results.jsonl", [
        {
            "workflow": "monthly_model_review",
            "provider": "codex_pro",
            "model": "gpt-5.4",
            "benchmark_score": 0.92,
            "sample_count": 8,
        },
        {
            "workflow": "monthly_model_review",
            "provider": "claude_max",
            "model": "sonnet",
            "benchmark_score": 0.4,
            "sample_count": 8,
            "governance_failures": ["approval bypass"],
        },
    ])

    scores = ProviderRouteScorer(findings).recompute()

    by_provider = {score["provider"]: score for score in scores}
    assert by_provider["codex_pro"]["benchmark_quality"] == 0.92
    assert by_provider["codex_pro"]["sample_count"] == 8
    assert by_provider["claude_max"]["benchmark_quality"] == 0.0
    assert by_provider["codex_pro"]["composite_score"] > by_provider["claude_max"]["composite_score"]


def test_recompute_scores_include_monthly_outcomes_and_model_review_validation(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    _write_jsonl(findings / "monthly_outcomes.jsonl", [
        {
            "run_id": "monthly-validation-bot1-2026-04",
            "source": "monthly",
            "provider": "codex_pro",
            "model": "gpt-5.4",
            "verdict": "keep",
            "confidence": 0.9,
        },
        {
            "run_id": "monthly-validation-bot1-2026-05",
            "source": "monthly",
            "provider": "claude_max",
            "model": "sonnet",
            "verdict": "rollback",
            "confidence": 0.9,
        },
    ])
    _write_jsonl(findings / "monthly_model_review_validations.jsonl", [
        {
            "workflow": "monthly_model_review",
            "provider": "codex_pro",
            "model": "gpt-5.4",
            "valid": True,
        },
        {
            "workflow": "monthly_model_review",
            "provider": "claude_max",
            "model": "sonnet",
            "valid": False,
            "issues": [{"message": "invented evidence"}],
        },
    ])

    scores = ProviderRouteScorer(findings).recompute()

    by_workflow_provider = {
        (score["workflow"], score["provider"]): score
        for score in scores
    }
    assert by_workflow_provider[("monthly_validation", "codex_pro")]["outcome_quality"] == 1.0
    assert by_workflow_provider[("monthly_validation", "claude_max")]["outcome_quality"] == 0.0
    assert by_workflow_provider[("monthly_model_review", "codex_pro")]["benchmark_quality"] == 1.0
    assert by_workflow_provider[("monthly_model_review", "claude_max")]["benchmark_quality"] == 0.0


def test_recompute_scores_include_operational_monthly_outcome_provider_attribution(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    result = MonthlyValidationResult(
        run_id="monthly-bot1-strat1-2026-04",
        run_month="2026-04",
        bot_id="bot1",
        strategy_id="strat1",
        status=MonthlyValidationStatus.KEEP,
        model_review_provider="codex_pro",
        model_review_model="gpt-5.4",
    )
    MonthlyOutcomeMeasurer(findings).record_from_monthly_validation(
        result,
        strategy_change_record_id="change-1",
    )

    scores = ProviderRouteScorer(findings).recompute()

    by_workflow_provider = {
        (score["workflow"], score["provider"]): score
        for score in scores
    }
    assert by_workflow_provider[("monthly_validation", "codex_pro")]["outcome_quality"] == 1.0


def test_monthly_provider_recommendation_uses_higher_default_sample_threshold(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    _write_jsonl(findings / "provider_route_scores.jsonl", [
        {
            "workflow": "monthly_model_review",
            "provider": "codex_pro",
            "model": "gpt-5.4",
            "composite_score": 0.95,
            "sample_count": 9,
        },
    ])

    recommendation = ProviderRouteScorer(findings).recommend_provider(
        "monthly_model_review",
        requested_provider="claude_max",
    )

    assert recommendation is None


def test_provider_recommendation_includes_route_change_audit_fields(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    _write_jsonl(findings / "provider_route_scores.jsonl", [
        {
            "workflow": "daily_analysis",
            "provider": "claude_max",
            "model": "sonnet",
            "composite_score": 0.52,
            "benchmark_quality": 0.5,
            "sample_count": 6,
        },
        {
            "workflow": "daily_analysis",
            "provider": "codex_pro",
            "model": "gpt-5.4",
            "composite_score": 0.72,
            "benchmark_quality": 0.9,
            "sample_count": 7,
        },
    ])

    recommendation = ProviderRouteScorer(findings).recommend_provider(
        "daily_analysis",
        requested_provider="claude_max",
    )

    assert recommendation is not None
    assert recommendation["provider"] == "codex_pro"
    assert recommendation["requested_provider"] == "claude_max"
    assert recommendation["requested_composite_score"] == 0.52
    assert recommendation["score_gap"] == 0.2
    assert "rollback_condition" in recommendation
