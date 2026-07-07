"""Canonical imports for shared approval and replay schemas."""
from __future__ import annotations

import pytest

from trading_assistant.schemas.approval import ApprovalRequest, ApprovalStatus, BacktestComparison, BacktestContext
from trading_assistant.schemas.bot_profile import BotConfigProfile
from trading_assistant.schemas.parameter_definition import ParameterDefinition, ParameterType
from trading_assistant.schemas.repo_changes import FileChange, FileChangeMode, PRRequest
from trading_assistant.schemas.repo_task import RepoTaskContext
from trading_assistant.schemas.simulation_metrics import SimulationMetrics


def test_shared_schema_canonical_imports_instantiate() -> None:
    metrics = SimulationMetrics(total_trades=10, win_count=6, sharpe_ratio=1.2)
    param = ParameterDefinition(
        param_name="quality_min",
        bot_id="bot1",
        param_type=ParameterType.YAML_FIELD,
        file_path="config.yaml",
        yaml_key="strategy.quality_min",
    )
    profile = BotConfigProfile(bot_id="bot1", parameters=[param])
    repo_task = RepoTaskContext(task_id="task1", repo_dir="/tmp/repo")
    change = FileChange(
        file_path="config.yaml",
        change_mode=FileChangeMode.YAML_FIELD,
        new_content="quality_min: 0.7",
    )
    comparison = BacktestComparison(
        context=BacktestContext(suggestion_id="s1", bot_id="bot1", param_name="quality_min"),
        baseline=SimulationMetrics(sharpe_ratio=1.0, total_trades=10, win_count=5),
        proposed=metrics,
    )
    request = ApprovalRequest(
        request_id="r1",
        suggestion_id="s1",
        bot_id="bot1",
        repo_task=repo_task,
        file_changes=[change],
        backtest_summary=comparison,
    )
    pr_request = PRRequest(
        approval_request_id=request.request_id,
        suggestion_id=request.suggestion_id,
        bot_id=request.bot_id,
        repo_dir=repo_task.repo_dir,
        branch_name="codex/test",
        title="Test PR",
        repo_task=repo_task,
        file_changes=[change],
    )

    assert profile.get_parameter("quality_min") == param
    assert request.status == ApprovalStatus.PENDING
    assert request.backtest_summary.sharpe_change_pct == pytest.approx(20.0)
    assert pr_request.file_changes == [change]
