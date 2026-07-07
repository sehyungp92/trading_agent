# tests/test_autonomous_schemas.py
"""Tests for autonomous pipeline schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from trading_assistant.schemas.approval import (
    ApprovalRequest,
    ApprovalStatus,
    BacktestComparison,
    BacktestContext,
)
from trading_assistant.schemas.bot_profile import BotConfigProfile
from trading_assistant.schemas.parameter_definition import ParameterDefinition, ParameterType
from trading_assistant.schemas.repo_changes import (
    PreflightResult,
    PRResult,
    PRReviewStatus,
    ReviewState,
)
from trading_assistant.schemas.simulation_metrics import SimulationMetrics


class TestParameterDefinition:
    def test_yaml_field_requires_yaml_key(self):
        with pytest.raises(ValidationError, match="yaml_key"):
            ParameterDefinition(
                param_name="x",
                bot_id="bot1",
                param_type=ParameterType.YAML_FIELD,
                file_path="config.yaml",
            )

    def test_python_constant_requires_python_path(self):
        with pytest.raises(ValidationError, match="python_path"):
            ParameterDefinition(
                param_name="x",
                bot_id="bot1",
                param_type=ParameterType.PYTHON_CONSTANT,
                file_path="config.py",
            )

    def test_valid_range_order(self):
        with pytest.raises(ValidationError, match="valid_range"):
            ParameterDefinition(
                param_name="x",
                bot_id="bot1",
                param_type=ParameterType.YAML_FIELD,
                file_path="config.yaml",
                yaml_key="x",
                valid_range=(1.0, 0.5),
            )

    def test_valid_yaml_definition(self):
        p = ParameterDefinition(
            param_name="quality_min",
            bot_id="bot1",
            param_type=ParameterType.YAML_FIELD,
            file_path="config.yaml",
            yaml_key="kmp.quality_min",
            current_value=0.6,
            valid_range=(0.0, 1.0),
            value_type="float",
            category="entry_signal",
        )
        assert p.param_name == "quality_min"
        assert p.yaml_key == "kmp.quality_min"


class TestBacktestComparison:
    def test_auto_computes_change_pcts(self):
        baseline = SimulationMetrics(
            sharpe_ratio=1.0, max_drawdown_pct=-10.0,
            profit_factor=1.5, total_trades=20, win_count=12,
        )
        proposed = SimulationMetrics(
            sharpe_ratio=1.2, max_drawdown_pct=-8.0,
            profit_factor=1.8, total_trades=20, win_count=14,
        )
        ctx = BacktestContext(
            suggestion_id="s1", bot_id="bot1",
            param_name="x", current_value=0.5, proposed_value=0.7,
        )
        comp = BacktestComparison(
            context=ctx, baseline=baseline, proposed=proposed,
        )
        assert comp.sharpe_change_pct == pytest.approx(20.0)
        assert comp.profit_factor_change_pct == pytest.approx(20.0)

    def test_passes_safety_logic(self):
        ctx = BacktestContext(suggestion_id="s1", bot_id="bot1", param_name="x")
        comp = BacktestComparison(
            context=ctx,
            baseline=SimulationMetrics(sharpe_ratio=1.0),
            proposed=SimulationMetrics(sharpe_ratio=1.2),
            passes_safety=True,
        )
        assert comp.passes_safety is True

    def test_zero_baseline_no_division_error(self):
        ctx = BacktestContext(suggestion_id="s1", bot_id="bot1", param_name="x")
        comp = BacktestComparison(
            context=ctx,
            baseline=SimulationMetrics(),
            proposed=SimulationMetrics(sharpe_ratio=1.0),
        )
        assert comp.sharpe_change_pct == 0.0


class TestApprovalRequest:
    def test_lifecycle_transitions(self):
        req = ApprovalRequest(
            request_id="r1", suggestion_id="s1", bot_id="bot1",
        )
        assert req.status == ApprovalStatus.PENDING
        assert req.resolved_at is None

        req.status = ApprovalStatus.APPROVED
        assert req.status == ApprovalStatus.APPROVED

    def test_serialization_roundtrip(self):
        req = ApprovalRequest(
            request_id="r1", suggestion_id="s1", bot_id="bot1",
            param_changes=[{"param_name": "x", "current": 0.5, "proposed": 0.7}],
        )
        data = req.model_dump(mode="json")
        restored = ApprovalRequest(**data)
        assert restored.request_id == "r1"
        assert restored.param_changes == req.param_changes


class TestPreflightResult:
    def test_default_passes(self):
        r = PreflightResult()
        assert r.passed is True
        assert r.checks == []
        assert r.reasons == []

    def test_failed_with_reasons(self):
        r = PreflightResult(
            passed=False,
            checks=[{"name": "remote", "passed": False, "detail": "unreachable"}],
            reasons=["Remote unreachable"],
        )
        assert r.passed is False
        assert len(r.checks) == 1
        assert r.reasons == ["Remote unreachable"]


class TestPRReviewStatus:
    def test_construction(self):
        status = PRReviewStatus(pr_number=42, pr_url="https://github.com/x/y/pull/42")
        assert status.pr_number == 42
        assert status.review_state == ReviewState.PENDING
        assert status.needs_attention is False

    def test_changes_requested_needs_attention(self):
        status = PRReviewStatus(
            pr_number=42,
            review_state=ReviewState.CHANGES_REQUESTED,
            needs_attention=True,
            actionable_comments=["Fix the type"],
        )
        assert status.needs_attention is True
        assert len(status.actionable_comments) == 1


class TestPRResultExtended:
    def test_new_fields_default_none(self):
        r = PRResult(success=True)
        assert r.pr_number is None
        assert r.preflight is None
        assert r.existing_pr_url is None

    def test_with_preflight(self):
        pf = PreflightResult(passed=False, reasons=["dirty tree"])
        r = PRResult(success=False, preflight=pf, error="preflight failed")
        assert r.preflight.passed is False

    def test_with_existing_pr_url(self):
        r = PRResult(success=True, existing_pr_url="https://github.com/x/y/pull/42")
        assert r.existing_pr_url == "https://github.com/x/y/pull/42"


class TestBotConfigProfile:
    def test_parameter_lookup(self):
        profile = BotConfigProfile(
            bot_id="bot1",
            parameters=[
                ParameterDefinition(
                    param_name="quality_min",
                    bot_id="bot1",
                    param_type=ParameterType.YAML_FIELD,
                    file_path="config.yaml",
                    yaml_key="kmp.quality_min",
                    category="entry_signal",
                ),
            ],
        )
        assert profile.get_parameter("quality_min") is not None
        assert profile.get_parameter("nonexistent") is None
        assert len(profile.get_parameters_by_category("entry_signal")) == 1
        assert len(profile.get_parameters_by_category("risk")) == 0
