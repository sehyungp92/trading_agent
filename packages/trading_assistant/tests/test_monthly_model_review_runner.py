from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.backtest_artifacts import BacktestArtifactIndex
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus
from trading_assistant.skills.monthly_model_review_runner import (
    MonthlyModelReviewInvocationResult,
    MonthlyModelReviewRunner,
)


def test_existing_model_review_preserves_invocation_attribution(tmp_path: Path) -> None:
    review_path = tmp_path / "model_review.json"
    review_path.write_text("{}", encoding="utf-8")
    (tmp_path / "model_review_invocation.json").write_text(
        json.dumps({
            "provider": "codex_pro",
            "model": "gpt-5.4",
            "runtime": "codex_cli",
            "cost_usd": 0.17,
        }),
        encoding="utf-8",
    )

    result = MonthlyModelReviewRunner().run(
        monthly_result=MonthlyValidationResult(
            run_id="monthly-bot1-strat1-2026-04",
            run_month="2026-04",
            bot_id="bot1",
            strategy_id="strat1",
            status=MonthlyValidationStatus.KEEP,
        ),
        artifact_index=BacktestArtifactIndex(
            run_id="monthly-bot1-strat1-2026-04",
            artifact_root=str(tmp_path),
        ),
        artifact_root=tmp_path,
        existing_review_path=review_path,
    )

    assert result.provider == "codex_pro"
    assert result.model == "gpt-5.4"
    assert result.runtime == "codex_cli"
    assert result.cost_usd == 0.17


def test_model_review_invocation_records_cost(tmp_path: Path) -> None:
    selected_path = tmp_path / "selected_candidates.json"
    selected_path.write_text(
        json.dumps({
            "candidates": [
                {
                    "candidate_id": "cand1",
                    "source": "smoke_repair",
                    "bot_id": "bot1",
                    "strategy_id": "strat1",
                    "evidence_paths": [str(tmp_path / "candidate_results.jsonl")],
                    "rollback_plan": "restore incumbent",
                    "replay_or_experiment_plan": "measure next month",
                    "acceptance_criteria": ["positive OOS"],
                }
            ]
        }),
        encoding="utf-8",
    )
    (tmp_path / "candidate_results.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "rejected_candidates.jsonl").write_text("", encoding="utf-8")

    def invoke(_package, _run_id):
        return MonthlyModelReviewInvocationResult(
            response="{}",
            provider="codex_pro",
            model="gpt-5.4",
            runtime="codex_cli",
            cost_usd=0.23,
        )

    review_path = tmp_path / "model_review.json"
    result = MonthlyModelReviewRunner(invoker=invoke).run(
        monthly_result=MonthlyValidationResult(
            run_id="monthly-bot1-strat1-2026-04",
            run_month="2026-04",
            bot_id="bot1",
            strategy_id="strat1",
            status=MonthlyValidationStatus.REPAIR,
        ),
        artifact_index=BacktestArtifactIndex(
            run_id="monthly-bot1-strat1-2026-04",
            artifact_root=str(tmp_path),
        ),
        artifact_root=tmp_path,
        existing_review_path=review_path,
    )

    metadata = json.loads((tmp_path / "model_review_invocation.json").read_text(encoding="utf-8"))
    assert result.invoked is True
    assert result.cost_usd == 0.23
    assert metadata["cost_usd"] == 0.23
