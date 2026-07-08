from __future__ import annotations

from pathlib import Path

import pytest

from trading_assistant_backtest.validation.bridge_readiness import run_bridge_readiness_audit
from trading_assistant_backtest.validation.decision_parity_run import (
    run_crypto_trend_decision_parity_validation,
)
from tests.paths import MONOREPO_ROOT, package_workspace

AGENT_ROOT = MONOREPO_ROOT
CRYPTO_CONTRACT = (
    package_workspace("trading_assistant_backtest")
    / "contracts"
    / "crypto_trend_v1"
    / "strategy_plugin_contract.json"
)
CRYPTO_DEPLOYMENT = CRYPTO_CONTRACT.parent / "deployment_metadata.json"
TRADING_REPO = AGENT_ROOT / "trading" / "ibkr_trader"
K_STOCK_REPO = AGENT_ROOT / "trading" / "k_stock_trader"


def test_bridge_readiness_audit_reports_formal_shadow_bridges_and_failures(
    tmp_path: Path,
) -> None:
    if not CRYPTO_CONTRACT.exists():
        pytest.skip("persisted crypto trend strategy contract is not available")
    if not TRADING_REPO.exists() or not K_STOCK_REPO.exists():
        pytest.skip("trading and k_stock_trader reference repos are not available")

    crypto_artifact_root = tmp_path / "crypto_trend_v1" / "decision_parity"
    run_crypto_trend_decision_parity_validation(
        contract_path=CRYPTO_CONTRACT,
        deployment_metadata_path=CRYPTO_DEPLOYMENT,
        artifact_root=crypto_artifact_root,
    )

    result = run_bridge_readiness_audit(
        agent_root=AGENT_ROOT,
        artifact_root=tmp_path / "bridge_readiness",
        crypto_parity_artifact_root=crypto_artifact_root,
    )
    bridges = {bridge["repo_id"]: bridge for bridge in result["bridges"]}

    assert result["ok"] is True
    assert Path(result["artifact_path"]).exists()
    assert result["approval_ready_bridges"] == []
    assert result["shadow_validated_bridges"] == [
        "crypto_trend_v1",
        "crypto_momentum_v1",
        "crypto_breakout_v1",
        "trading_stock_family",
        "k_stock_olr_kalcb",
        "trading_momentum_family",
        "trading_swing_family",
    ]
    assert result["configured_shadow_validated_bridges"] == [
        "crypto_trend_v1",
        "crypto_momentum_v1",
        "crypto_breakout_v1",
        "trading_stock_family",
        "k_stock_olr_kalcb",
        "trading_momentum_family",
        "trading_swing_family",
    ]
    assert result["recommended_next_bridge"] == "replay_backed_evaluator_and_fixture_coverage"
    assert result["weekly_focus_rotation"][0]["bridge_ids"] == [
        "k_stock_olr_kalcb",
        "trading_stock_family",
    ]

    assert bridges["crypto_trend_v1"]["status"] == "formal_decision_parity_passed"
    assert bridges["crypto_trend_v1"]["maturity"] == "shadow_validated"
    assert bridges["crypto_trend_v1"]["approval_ready"] is False
    assert bridges["crypto_momentum_v1"]["status"] == "formal_decision_parity_passed"
    assert bridges["crypto_breakout_v1"]["status"] == "formal_decision_parity_passed"

    trading_stock = bridges["trading_stock_family"]
    assert trading_stock["status"] == "formal_decision_parity_passed"
    assert trading_stock["maturity"] == "shadow_validated"
    assert trading_stock["eligible_for_optimizer"] is True
    assert trading_stock["approval_ready"] is False
    assert trading_stock["errors"] == []
    assert trading_stock["weekly_focus_week"] == 1
    assert "strategy:IARIC_v1" in trading_stock["supported_scope"]
    assert trading_stock["approval_blockers"] == [
        "plugin_maturity_is_shadow_validated_not_approval_ready"
    ]

    k_stock = bridges["k_stock_olr_kalcb"]
    assert k_stock["status"] == "formal_decision_parity_passed"
    assert k_stock["maturity"] == "shadow_validated"
    assert k_stock["eligible_for_optimizer"] is True
    assert k_stock["approval_ready"] is False
    assert k_stock["errors"] == []
    assert k_stock["weekly_focus_week"] == 1
    assert "strategy:OLR" in k_stock["supported_scope"]
    assert "portfolio:OLR_KALCB" in k_stock["supported_scope"]

    momentum = bridges["trading_momentum_family"]
    assert momentum["status"] == "formal_decision_parity_passed"
    assert momentum["errors"] == []
    assert momentum["eligible_for_optimizer"] is True
    assert momentum["approval_ready"] is False
    assert momentum["weekly_focus_week"] == 2
    assert "asset:futures" in momentum["supported_scope"]
    assert "strategy:NQ_REGIME" in momentum["supported_scope"]

    swing = bridges["trading_swing_family"]
    assert swing["status"] == "formal_decision_parity_passed"
    assert swing["weekly_focus_week"] == 3
    assert "strategy:AKC_HELIX" in swing["supported_scope"]
    assert "strategy:OVERLAY" in swing["supported_scope"]

