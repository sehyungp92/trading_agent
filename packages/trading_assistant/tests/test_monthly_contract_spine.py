from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_assistant.orchestrator.agent_runner import AgentResult
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.handlers import Handlers
from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType
from trading_assistant.schemas.notifications import NotificationPreferences
from trading_assistant.schemas.strategy_change_ledger import StrategyChangeRecordType
from trading_assistant.skills.learning_review_orchestrator import LearningReviewOrchestrator
from trading_assistant.skills.monthly_search_brief_builder import MonthlySearchBriefBuilder
from trading_assistant.skills.provider_route_scorer import ProviderRouteScorer
from trading_assistant.skills.run_recall_summarizer import RunRecallSummarizer
from trading_assistant.skills.strategy_change_ledger import StrategyChangeLedger
from tests.test_monthly_candidate_generation import _write_fixture_runner, _write_inputs


@pytest.mark.asyncio
async def test_scheduled_monthly_contract_spine_feeds_review_recall_search_and_provider_scores(
    tmp_path: Path,
) -> None:
    curated, findings, market_root, repo = _write_inputs(tmp_path)
    _write_fixture_runner(repo, write_model_review=False)

    async def invoke_model_review(agent_type, prompt_package, run_id, **_kwargs):
        assert agent_type == "monthly_model_review"
        request = prompt_package.data
        evidence_path = request["selected_candidates"][0]["evidence_paths"][0]
        response = json.dumps({
            "run_id": request["run_id"],
            "bot_id": request["bot_id"],
            "strategy_id": request["strategy_id"],
            "candidate_reviews": [{
                "candidate_id": "cand-smoke-1",
                "recommendation": "approval packet is coherent",
                "evidence_paths": [evidence_path],
                "expected_objective_impact": {"latest_month_oos": 0.18},
                "risk_classification": "medium",
                "replay_or_experiment_plan": "Measure next completed month.",
                "acceptance_criteria": ["positive latest OOS"],
                "rollback_plan": "Restore config version cv1.",
                "routing": "experiment",
            }],
        })
        return AgentResult(
            response=response,
            run_dir=tmp_path / "runs" / run_id,
            success=True,
            provider="codex_pro",
            runtime="codex_cli",
            effective_model="gpt-5.4",
        )

    agent_runner = MagicMock()
    agent_runner.invoke = AsyncMock(side_effect=invoke_model_review)
    dispatcher = AsyncMock()
    dispatcher.dispatch = AsyncMock(return_value=[])
    runs_dir = tmp_path / "runs"
    handlers = Handlers(
        agent_runner=agent_runner,
        event_stream=EventStream(),
        dispatcher=dispatcher,
        notification_prefs=NotificationPreferences(),
        curated_dir=curated,
        memory_dir=tmp_path / "memory",
        runs_dir=runs_dir,
        source_root=tmp_path,
        bots=["bot1"],
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
        monthly_validation_mode="approval_gated",
        monthly_optimizer_sequence_enabled=False,
    )

    await handlers.handle_monthly_validation(Action(
        type=ActionType.SPAWN_MONTHLY_VALIDATION,
        event_id="evt-monthly-spine",
        bot_id="bot1",
        details={
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "run_month": "2026-04",
            "backtest_command": ["python", "fixture_runner.py", "{manifest}"],
            "optimizer_sequence_enabled": False,
            "shadow": False,
        },
    ))

    run_dir = runs_dir / "monthly-bot1-strat1-2026-04"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["monthly_artifact_roots"]
    assert metadata["monthly_stage_status"]
    artifact_root = Path(metadata["monthly_artifact_roots"][0])
    assert (artifact_root / "selected_candidates.json").exists()
    assert (artifact_root / "model_review_validation.json").exists()
    invocation = json.loads((artifact_root / "model_review_invocation.json").read_text(encoding="utf-8"))
    assert invocation["provider"] == "codex_pro"
    assert invocation["model"] == "gpt-5.4"

    monthly_reviews = [
        record for record in StrategyChangeLedger(findings).get_for_strategy("bot1", "strat1")
        if record.record_type == StrategyChangeRecordType.MONTHLY_REVIEW
    ]
    assert monthly_reviews
    assert any(Path(path).name == "monthly_report.md" for path in monthly_reviews[0].evidence_paths)

    today = datetime.now(timezone.utc).date().isoformat()
    review = LearningReviewOrchestrator(findings, runs_dir=runs_dir).run(
        week_start=today,
        week_end=today,
    )
    action_types = {action["type"] for action in review["actions"]}
    assert "monthly_candidate_recall" in action_types

    cards = RunRecallSummarizer(tmp_path / "memory").summarize(
        workflow="monthly_validation",
        bot_id="bot1",
        strategy_id="strat1",
    )
    assert cards
    assert any(card.run_id == "monthly-bot1-strat1-2026-04" for card in cards)

    brief = MonthlySearchBriefBuilder(findings).build(
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
    )
    assert any(
        item.get("source_weekly_signal_id") == monthly_reviews[0].record_id
        for item in brief.rollback_candidates
    )

    scores = ProviderRouteScorer(findings).recompute()
    by_workflow_provider = {
        (score["workflow"], score["provider"]): score
        for score in scores
    }
    assert by_workflow_provider[("monthly_model_review", "codex_pro")]["benchmark_quality"] == 1.0
