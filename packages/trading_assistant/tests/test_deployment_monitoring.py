"""Tests for DeploymentRecord lineage extension."""
from __future__ import annotations

from trading_assistant.schemas.deployment_monitoring import DeploymentRecord, DeploymentStatus


def _kwargs() -> dict:
    return {
        "deployment_id": "dep_001",
        "approval_request_id": "ar_001",
        "pr_url": "https://example.com/pr/1",
        "bot_id": "bot_a",
    }


def test_deployment_record_defaults_lineage_empty() -> None:
    rec = DeploymentRecord(**_kwargs())
    assert rec.affected_population == []
    assert rec.variant_id is None
    assert rec.parameter_set_id is None
    assert rec.strategy_version is None
    assert rec.config_version is None
    assert rec.status == DeploymentStatus.PENDING_MERGE


def test_deployment_record_round_trip_with_lineage() -> None:
    kwargs = _kwargs()
    kwargs.update(
        affected_population=["ev_1", "ev_2"],
        variant_id="A",
        parameter_set_id="ps_42",
        strategy_version="v3",
        config_version="2026-05",
    )
    rec = DeploymentRecord(**kwargs)
    redo = DeploymentRecord(**rec.model_dump(mode="json"))
    assert redo.affected_population == ["ev_1", "ev_2"]
    assert redo.variant_id == "A"
    assert redo.parameter_set_id == "ps_42"
    assert redo.strategy_version == "v3"
    assert redo.config_version == "2026-05"
