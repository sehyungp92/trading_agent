from __future__ import annotations

import json

from libs.instrumentation.deployment_metadata import (
    build_deployment_metadata,
    write_deployment_metadata,
)
from libs.instrumentation.lineage import LineageContext


def _lineage() -> LineageContext:
    return LineageContext(
        bot_id="stock_trader",
        strategy_id="IARIC_v1",
        family_id="stock",
        portfolio_id="paper_default",
        strategy_version="IARIC_v1.0.0",
        config_version="cfg_1",
        deployment_id="dep_1",
        code_sha="abc123",
        trace_id="trace_1",
        proposal_ids=("proposal-1",),
        source_weekly_signal_ids=("weekly-1",),
        strategy_change_record_ids=("change-1",),
        candidate_ids=("candidate-1",),
        monthly_search_brief_id="brief-1",
    )


def _env() -> dict[str, str]:
    return {
        "SOURCE_CONTROL_ORIGIN": "git@github.com:example/trading.git",
        "DEPLOYED_COMMIT_SHA": "abc123",
        "SOURCE_CONTROL_COMMIT_SHA": "abc123",
        "SOURCE_CONTROL_WORKTREE_CLEAN": "true",
        "STRATEGY_PLUGIN_CONTRACT_HASH": "feedface",
        "EMISSION_ENVIRONMENT": "production_vps",
        "RUNTIME_HOST_FINGERPRINT": "host_test",
        "DEPLOYMENT_BOT_ID": "not_trading",
    }


def test_deployment_metadata_matches_approval_artifact_contract(tmp_path) -> None:
    metadata = build_deployment_metadata(
        _lineage(),
        bridge_id="trading_stock_family",
        repo_root=tmp_path,
        effective_config={"risk": {"unit": 100}},
        env=_env(),
    )

    assert metadata["metadata_source"] == "vps_live_bot_runtime_deployment_metadata_v1"
    assert metadata["emission_environment"] == "production_vps"
    assert metadata["repo_url"] == "https://github.com/example/trading"
    assert metadata["source_control_origin"] == metadata["repo_url"]
    assert metadata["source_control_worktree_clean"] is True
    assert metadata["bot_id"] == "trading"
    assert metadata["portfolio_id"] == "stock"
    assert metadata["strategy_id"] == "trading_stock_family"
    assert metadata["telemetry_schema_version"] == "trading_live_shadow_contract_v1"
    assert metadata["strategy_plugin_contract_hash"] == "feedface"
    assert metadata["assistant_driven"] is True
    assert metadata["assistant_lineage"]["proposal_ids"] == ["proposal-1"]
    assert metadata["assistant_lineage"]["source_weekly_signal_ids"] == ["weekly-1"]
    assert metadata["assistant_lineage"]["weekly_signal_ids"] == ["weekly-1"]
    assert metadata["assistant_lineage"]["strategy_change_record_ids"] == ["change-1"]
    assert metadata["assistant_lineage"]["candidate_ids"] == ["candidate-1"]
    assert metadata["assistant_lineage"]["monthly_search_brief_id"] == "brief-1"
    assert metadata["approval_ready"] is True


def test_deployment_metadata_derives_live_vps_environment_from_trading_mode(tmp_path) -> None:
    env = _env()
    env.pop("EMISSION_ENVIRONMENT")
    env["TRADING_MODE"] = "live"

    metadata = build_deployment_metadata(
        _lineage(),
        bridge_id="trading_stock_family",
        repo_root=tmp_path,
        effective_config={"risk": {"unit": 100}},
        env=env,
    )

    assert metadata["emission_environment"] == "production_vps"


def test_write_deployment_metadata_uses_bridge_subdirectory(tmp_path) -> None:
    path = write_deployment_metadata(
        tmp_path,
        _lineage(),
        bridge_id="trading_stock_family",
        repo_root=tmp_path,
        effective_config={"risk": {"unit": 100}},
        env=_env(),
    )

    assert path == tmp_path / "deployments" / "trading_stock_family" / "deployment_metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["deployment_id"] == "dep_1"
