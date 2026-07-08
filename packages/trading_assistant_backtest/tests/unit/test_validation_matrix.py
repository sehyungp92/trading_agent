from __future__ import annotations

from pathlib import Path

import pytest

from trading_assistant_backtest.validation.validation_matrix import (
    _approval_grade_validation_complete,
    _bridge_artifact_path,
    _historical_walk_forward_leakage_ok,
    run_validation_matrix_audit,
)
from tests.paths import MONOREPO_ROOT, package_workspace

AGENT_ROOT = MONOREPO_ROOT
TRADING_REPO = AGENT_ROOT / "trading" / "ibkr_trader"
K_STOCK_REPO = AGENT_ROOT / "trading" / "k_stock_trader"
CRYPTO_REPO = AGENT_ROOT / "trading" / "crypto_trader"
DATA_REPO = package_workspace("trading_assistant_data")


def test_validation_matrix_marks_runnable_tests_and_real_blockers(tmp_path: Path) -> None:
    if (
        not TRADING_REPO.exists()
        or not K_STOCK_REPO.exists()
        or not CRYPTO_REPO.exists()
        or not DATA_REPO.exists()
    ):
        pytest.skip("reference trading repos are not available")

    result = run_validation_matrix_audit(
        agent_root=AGENT_ROOT,
        artifact_root=tmp_path / "validation_matrix",
    )
    scopes = {scope["scope_id"]: scope for scope in result["scopes"]}

    assert result["ok"] is True
    assert result["runnable_validations_passed"] is True
    assert result["all_validation_tests_runnable_for_all_scopes"] is True
    assert result["approval_grade_validation_complete"] is False
    assert result["structural_candidates_approval_ready"] is False
    assert Path(result["artifact_path"]).exists()

    expected_full_family_counts = {
        "k_stock_olr_kalcb": 516,
        "trading_stock_family": 611,
        "trading_momentum_family": 2,
        "trading_swing_family": 6,
    }
    for scope_id, expected_slice_count in expected_full_family_counts.items():
        tests = scopes[scope_id]["tests"]
        assert tests["data_reproduction"]["result"] == "pass"
        assert tests["data_reproduction"]["reason"] == "authoritative data bundle reproduced"
        authority = tests["data_reproduction"]["full_family_authority"]
        assert authority["status"] == "pass"
        assert authority["selected_slice_count"] == expected_slice_count
        assert authority["non_authoritative_count"] == 0
        assert authority["missing_requirement_count"] == 0
        assert tests["decision_parity"]["result"] == "pass"
        assert tests["decision_parity"]["reason"] == "formal_decision_parity_passed"
        assert tests["decision_parity"]["approval_ready"] is False
        assert tests["incumbent_replay"]["result"] == "pass"
        assert tests["round_reproduction"]["result"] == "pass"
        assert tests["historical_walk_forward"]["result"] == "pass"
        assert tests["historical_walk_forward"]["reason"] == (
            "historical_walk_forward_replay_evidence_passed"
        )

    stock_scope = (
        scopes["trading_stock_family"]["tests"]["data_reproduction"]
        ["full_family_authority"]["requirement_scope"]
    )
    stock_derived = stock_scope["derived_trading_stock_scope"]
    assert stock_derived["live_intraday_symbol_count"] == 98
    assert stock_derived["live_intraday_requirement_count"] == 294
    assert stock_derived["daily_reference_context_symbol_count"] == 317
    assert stock_derived["legacy_archive_source_comparison_count"] == 611
    assert stock_derived["excluded_swing_symbols_present"] == []

    crypto = scopes["crypto_trader_portfolio"]["tests"]
    assert crypto["data_reproduction"]["result"] == "pass"
    assert crypto["decision_parity"]["result"] == "pass"
    assert crypto["decision_parity"]["reason"] == "formal_decision_parity_passed"
    assert crypto["decision_parity"]["covered_strategies"] == ["trend", "momentum", "breakout"]
    assert crypto["decision_parity"]["missing_strategies"] == []
    assert crypto["incumbent_replay"]["result"] == "pass"
    assert crypto["round_reproduction"]["result"] == "pass"
    assert crypto["historical_walk_forward"]["result"] == "pass"

    assert result["meaningful_remaining_gaps"] == []
    blocked = set()
    assert (
        "data_reproduction",
        "smoke_data_reproduced_full_family_authority_blocked",
    ) not in blocked
    assert (
        "incumbent_replay",
        "smoke_replay_passed_full_family_authority_blocked",
    ) not in blocked
    assert (
        "decision_parity",
        "formal_decision_parity_passed_but_not_approval_ready",
    ) not in blocked
    assert (
        "historical_walk_forward",
        "historical_walk_forward_leakage_checks_failed",
    ) not in blocked
    assert (
        "data_reproduction",
        "authoritative_data_bundle_missing_for_scope",
    ) not in blocked
    assert (
        "data_reproduction",
        "data_reproduction_report_missing_for_expected_bundle",
    ) not in blocked
    assert (
        "incumbent_replay",
        "missing_replay_backed_evaluator_and_old_diagnostics_baseline",
    ) not in blocked
    approval_gaps = {
        (row["scope_id"], row["reason"])
        for row in result["approval_remaining_gaps"]
    }
    contract_gaps = {
        gap for gap in approval_gaps if gap[1] == "strategy_contract_not_approval_ready"
    }
    assert len(contract_gaps) == 5
    assert (
        "crypto_trader_portfolio",
        "strategy_contract_not_approval_ready",
    ) in approval_gaps


def test_validation_matrix_approval_ready_requires_optimizer_p6_p7() -> None:
    passed_test = {"result": "pass", "approval_ready": True}
    scope = {
        "tests": {
            "data_reproduction": {"result": "pass"},
            "incumbent_replay": {"result": "pass"},
            "decision_parity": passed_test,
            "round_reproduction": {"result": "pass"},
            "historical_walk_forward": {"result": "pass"},
        },
        "optimizer_approval_readiness": {
            "ready": False,
            "checks": [
                {
                    "name": "scope:optimizer_p7_repair_confirmatory_round_complete",
                    "passed": False,
                    "errors": ["round_n_plus_1_recommendation.json missing"],
                }
            ],
        },
    }

    assert _approval_grade_validation_complete([scope]) is False

    scope["optimizer_approval_readiness"]["ready"] = True
    assert _approval_grade_validation_complete([scope]) is True


def test_validation_matrix_prefers_root_deployment_metadata_for_optimizer_context(
    tmp_path: Path,
) -> None:
    agent_root = tmp_path / "repo"
    root_contract = (
        agent_root
        / "contracts"
        / "strategy_plugins"
        / "trading_stock_family"
        / "strategy_plugin_contract.json"
    )
    root_metadata = root_contract.with_name("deployment_metadata.json")
    package_metadata = (
        agent_root
        / "packages"
        / "trading_assistant_backtest"
        / "contracts"
        / "trading_stock_family"
        / "deployment_metadata.json"
    )
    root_contract.parent.mkdir(parents=True)
    package_metadata.parent.mkdir(parents=True)
    root_contract.write_text("{}", encoding="utf-8")
    root_metadata.write_text('{"source":"root"}', encoding="utf-8")
    package_metadata.write_text('{"source":"package-fixture"}', encoding="utf-8")

    selected = _bridge_artifact_path(
        "trading_stock_family",
        {
            "evidence": [
                {
                    "relative_path": (
                        "contracts/strategy_plugins/trading_stock_family/"
                        "strategy_plugin_contract.json"
                    )
                }
            ]
        },
        agent_root=agent_root,
        artifact_name="deployment_metadata.json",
    )

    assert selected == root_metadata


def test_validation_matrix_rejects_historical_report_with_failed_leakage_checks(
    tmp_path: Path,
) -> None:
    walk_forward = tmp_path / "historical_walk_forward_report.json"
    walk_forward.write_text(
        """
        {
          "schema_version": "historical_walk_forward_report_v1",
          "status": "pass",
          "leakage_checks": {
            "status": "pass",
            "window_order_strictly_increasing": true,
            "bundle_checksums_unique": false
          }
        }
        """,
        encoding="utf-8",
    )
    report = {
        "tests": {
            "historical_walk_forward": {
                "ok": True,
                "artifact_paths": [str(walk_forward)],
            }
        }
    }

    assert _historical_walk_forward_leakage_ok(report, "historical_walk_forward") is False


def test_validation_matrix_resolves_legacy_workspace_artifact_paths(tmp_path: Path) -> None:
    agent_root = tmp_path / "repo"
    final_walk_forward = (
        agent_root
        / "packages"
        / "trading_assistant_backtest"
        / "artifacts"
        / "validation"
        / "replay_evidence"
        / "scope"
        / "historical_walk_forward_report.json"
    )
    final_walk_forward.parent.mkdir(parents=True)
    final_walk_forward.write_text(
        """
        {
          "schema_version": "historical_walk_forward_report_v1",
          "status": "pass",
          "leakage_checks": {
            "status": "pass",
            "bundle_checksums_unique": false
          }
        }
        """,
        encoding="utf-8",
    )
    legacy_walk_forward = (
        agent_root
        / "trading_assistant_backtest"
        / "artifacts"
        / "validation"
        / "replay_evidence"
        / "scope"
        / "historical_walk_forward_report.json"
    )
    report = {
        "tests": {
            "historical_walk_forward": {
                "ok": True,
                "artifact_paths": [str(legacy_walk_forward)],
            }
        }
    }

    assert (
        _historical_walk_forward_leakage_ok(
            report,
            "historical_walk_forward",
            agent_root=agent_root,
        )
        is False
    )

