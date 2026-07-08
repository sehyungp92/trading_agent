from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import trading_assistant_backtest.validation.approval_grade_audit as audit_module
from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.validation.approval_grade_audit import (
    _deployment_metadata_checks,
)
from trading_assistant_backtest.validation.deployment_metadata_contract import (
    live_deployment_metadata_errors,
)
from trading_assistant_backtest.validation.deployment_metadata_emit import (
    emit_runtime_deployment_metadata,
)
from trading_assistant_backtest.validation.deployment_metadata_install import (
    validate_and_maybe_install_deployment_metadata,
)
from trading_assistant_backtest.validation.optimizer_evidence import (
    build_optimizer_manifest_index,
    optimizer_evidence_checks,
)


def test_approval_grade_audit_refreshes_validation_and_bridge_reports_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = {"matrix": 0, "bridge": 0}

    def fake_matrix(**_kwargs):
        calls["matrix"] += 1
        tests = {
            test_name: {"result": "blocked", "reason": "fixture_blocked"}
            for test_name in audit_module.APPROVAL_TESTS
        }
        return {
            "scopes": [
                {
                    "scope_id": "crypto_trader_portfolio",
                    "tests": tests,
                    "optimizer_approval_readiness": {
                        "ready": False,
                        "checks": [
                            {
                                "name": (
                                    "crypto_trader_portfolio:"
                                    "optimizer_p6_true_fold_scoring_complete"
                                ),
                                "passed": False,
                                "errors": ["fresh optimizer fixture blocker"],
                            }
                        ],
                    },
                }
            ]
        }

    def fake_bridge(**_kwargs):
        calls["bridge"] += 1
        return {"bridges": []}

    cached_matrix = (
        tmp_path
        / "artifacts"
        / "validation"
        / "approval_grade"
        / "source_reports"
        / "validation_matrix"
        / "validation_matrix_report.json"
    )
    cached_matrix.parent.mkdir(parents=True)
    cached_matrix.write_text(
        json.dumps({"scopes": [{"scope_id": "stale_cached_scope"}]}),
        encoding="utf-8",
    )
    cached_bridge = (
        tmp_path
        / "artifacts"
        / "validation"
        / "approval_grade"
        / "source_reports"
        / "bridge_readiness"
        / "bridge_readiness_report.json"
    )
    cached_bridge.parent.mkdir(parents=True)
    cached_bridge.write_text(json.dumps({"bridges": [{"repo_id": "stale"}]}), encoding="utf-8")
    monkeypatch.setattr(audit_module, "run_validation_matrix_audit", fake_matrix)
    monkeypatch.setattr(audit_module, "run_bridge_readiness_audit", fake_bridge)

    report = audit_module.run_approval_grade_audit(
        agent_root=tmp_path,
        artifact_root=tmp_path / "audit",
        scope_ids=("crypto_trader_portfolio",),
    )

    assert calls == {"matrix": 1, "bridge": 1}
    assert report["source_reports_refreshed"] is True
    assert report["approval_grade"] is False


def test_approval_grade_audit_rejects_local_shadow_deployment_metadata(
    tmp_path: Path,
) -> None:
    contract_path, metadata_path = _write_crypto_contract_pair(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "metadata_source": "local clean live-repo checkout shadow snapshot",
            "repo_url": "local://trading/crypto_trader",
            "strategy_plugin_contract_hash": sha256_file(contract_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    checks = _deployment_metadata_checks("crypto_trend_v1", tmp_path)
    by_name = {check["name"]: check for check in checks}

    assert by_name["crypto_trend_v1:deployment_metadata_live_emitted"]["passed"] is False
    assert by_name["crypto_trend_v1:deployment_repo_url_not_local_shadow"]["passed"] is False
    assert by_name["crypto_trend_v1:deployment_sha_matches_contract"]["passed"] is True
    assert by_name["crypto_trend_v1:deployment_contract_hash_matches"]["passed"] is True


def test_approval_grade_audit_accepts_live_emitted_metadata_contract_fields(
    tmp_path: Path,
) -> None:
    contract_path, metadata_path = _write_crypto_contract_pair(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "metadata_source": "vps_live_bot_runtime_deployment_metadata_v1",
            "emission_environment": "paper_vps",
            "emission_context": "runtime_startup",
            "emitted_at_utc": "2026-05-31T12:00:00Z",
            "live_runtime_started_at_utc": "2026-05-31T12:00:00Z",
            "runtime_entrypoint": "crypto-trader paper",
            "runtime_instance_id": "crypto-paper-vps-1",
            "runtime_host_fingerprint": "host-" + "d" * 16,
            "source_control_origin": "https://github.com/example/crypto_trader.git",
            "source_control_commit_sha": "a" * 40,
            "source_control_worktree_clean": True,
            "repo_url": "https://github.com/example/crypto_trader.git",
            "strategy_plugin_contract_hash": sha256_file(contract_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    checks = _deployment_metadata_checks("crypto_trend_v1", tmp_path)

    assert all(check["passed"] for check in checks)


def test_approval_grade_audit_rejects_generic_runtime_metadata_without_live_provenance(
    tmp_path: Path,
) -> None:
    contract_path, metadata_path = _write_crypto_contract_pair(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "metadata_source": "runtime_deployment_metadata_v1",
            "repo_url": "https://github.com/example/crypto_trader.git",
            "strategy_plugin_contract_hash": sha256_file(contract_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    checks = _deployment_metadata_checks("crypto_trend_v1", tmp_path)
    by_name = {check["name"]: check for check in checks}

    assert by_name["crypto_trend_v1:deployment_metadata_live_emitted"]["passed"] is False
    assert any(
        "metadata_source must be one of" in error
        for error in by_name["crypto_trend_v1:deployment_metadata_live_emitted"]["errors"]
    )


def test_deployment_metadata_installer_refuses_shadow_metadata(
    tmp_path: Path,
) -> None:
    _contract_path, metadata_path = _write_crypto_contract_pair(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "metadata_source": "local clean live-repo checkout shadow snapshot",
            "repo_url": "local://trading/crypto_trader",
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = validate_and_maybe_install_deployment_metadata(
        agent_root=tmp_path,
        bridge_id="crypto_trend_v1",
        metadata_path=metadata_path,
        artifact_root=tmp_path / "artifacts",
        install=True,
    )

    assert report["ok"] is False
    assert report["installed"] is False
    assert any(
        check["name"] == "live_emission_provenance" and check["passed"] is False
        for check in report["checks"]
    )


def test_runtime_deployment_metadata_emitter_produces_installable_live_shape(
    tmp_path: Path,
) -> None:
    repo_path, commit_sha = _write_git_live_repo(tmp_path)
    contract_path, _metadata_path = _write_crypto_contract_pair(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["live_repo_commit_sha"] = commit_sha
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    config_path = tmp_path / "live_config.json"
    config_path.write_text(json.dumps({"risk": {"max_notional": 1000}}), encoding="utf-8")
    output_path = tmp_path / "runtime" / "deployment_metadata.json"

    report = emit_runtime_deployment_metadata(
        repo_path=repo_path,
        contract_path=contract_path,
        config_path=config_path,
        output_path=output_path,
        bot_id="crypto_portfolio",
        strategy_id="btc_1m",
        strategy_version="crypto_trend_v1",
        config_version="trend_config",
        telemetry_schema_version="trade_event_v1",
        runtime_entrypoint="crypto-trader paper",
        runtime_instance_id="crypto-paper-vps-1",
        deployment_id="deployment-1",
        emission_environment="paper_vps",
    )
    metadata = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["ok"] is True
    assert live_deployment_metadata_errors(metadata) == []
    assert metadata["deployed_commit_sha"] == commit_sha
    assert metadata["repo_url"] == "https://github.com/acme/crypto_trader.git"
    assert metadata["strategy_plugin_contract_hash"] == _sha256_file(contract_path)

    install_report = validate_and_maybe_install_deployment_metadata(
        agent_root=tmp_path,
        bridge_id="crypto_trend_v1",
        metadata_path=output_path,
        artifact_root=tmp_path / "artifacts",
        install=True,
    )

    assert install_report["ok"] is True
    assert install_report["installed"] is True


def test_deployment_metadata_installer_accepts_complete_schema_list(
    tmp_path: Path,
) -> None:
    contract_path, metadata_path = _write_crypto_contract_pair(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["required_telemetry_schemas"] = ["trade_event_v1", "assistant_event_v1"]
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["telemetry_schema_version"] = "trade_event_v1"
    metadata["telemetry_schema_versions"] = ["trade_event_v1", "assistant_event_v1"]
    metadata["strategy_plugin_contract_hash"] = sha256_file(contract_path)
    runtime_metadata_path = tmp_path / "runtime" / "deployment_metadata.json"
    runtime_metadata_path.parent.mkdir(parents=True)
    runtime_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    install_report = validate_and_maybe_install_deployment_metadata(
        agent_root=tmp_path,
        bridge_id="crypto_trend_v1",
        metadata_path=runtime_metadata_path,
        artifact_root=tmp_path / "artifacts",
        install=True,
    )

    assert install_report["ok"] is True
    assert install_report["installed"] is True


def test_deployment_metadata_installer_rejects_missing_required_schema(
    tmp_path: Path,
) -> None:
    contract_path, metadata_path = _write_crypto_contract_pair(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["required_telemetry_schemas"] = ["trade_event_v1", "assistant_event_v1"]
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["telemetry_schema_version"] = "trade_event_v1"
    metadata["telemetry_schema_versions"] = ["trade_event_v1"]
    metadata["strategy_plugin_contract_hash"] = sha256_file(contract_path)
    runtime_metadata_path = tmp_path / "runtime" / "deployment_metadata.json"
    runtime_metadata_path.parent.mkdir(parents=True)
    runtime_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    install_report = validate_and_maybe_install_deployment_metadata(
        agent_root=tmp_path,
        bridge_id="crypto_trend_v1",
        metadata_path=runtime_metadata_path,
        artifact_root=tmp_path / "artifacts",
        install=True,
    )

    assert install_report["ok"] is False
    assert install_report["installed"] is False
    assert any(
        check["name"] == "telemetry_schema_matches_contract"
        and check["passed"] is False
        and any("assistant_event_v1" in error for error in check["errors"])
        for check in install_report["checks"]
    )


def test_runtime_deployment_metadata_emitter_rejects_dirty_checkout(
    tmp_path: Path,
) -> None:
    repo_path, commit_sha = _write_git_live_repo(tmp_path)
    contract_path, _metadata_path = _write_crypto_contract_pair(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["live_repo_commit_sha"] = commit_sha
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    config_path = tmp_path / "live_config.json"
    config_path.write_text(json.dumps({"risk": {"max_notional": 1000}}), encoding="utf-8")
    (repo_path / "strategy.py").write_text("VALUE = 2\n", encoding="utf-8")

    report = emit_runtime_deployment_metadata(
        repo_path=repo_path,
        contract_path=contract_path,
        config_path=config_path,
        output_path=tmp_path / "runtime" / "deployment_metadata.json",
        bot_id="crypto_portfolio",
        strategy_id="btc_1m",
        strategy_version="crypto_trend_v1",
        config_version="trend_config",
        telemetry_schema_version="trade_event_v1",
        runtime_entrypoint="crypto-trader paper",
        runtime_instance_id="crypto-paper-vps-1",
        deployment_id="deployment-1",
        emission_environment="paper_vps",
    )

    assert report["ok"] is False
    assert any(
        "source_control_worktree_clean must be true" in error
        for check in report["checks"]
        for error in check["errors"]
    )


def test_approval_grade_audit_blocks_missing_optimizer_p6_p7_evidence(
    tmp_path: Path,
) -> None:
    root = _write_optimizer_root(tmp_path, "crypto_trader_portfolio")

    checks = optimizer_evidence_checks("crypto_trader_portfolio", tmp_path)
    by_name = {check["name"]: check for check in checks}

    assert root.exists()
    assert by_name[
        "crypto_trader_portfolio:optimizer_p6_true_fold_scoring_complete"
    ]["passed"] is False
    assert by_name[
        "crypto_trader_portfolio:optimizer_p7_repair_confirmatory_round_complete"
    ]["passed"] is False
    assert any(
        "fold_score_matrix.json" in error
        for error in by_name[
            "crypto_trader_portfolio:optimizer_p6_true_fold_scoring_complete"
        ]["errors"]
    )
    assert any(
        "selection_oos_evaluation.json" in error
        for error in by_name[
            "crypto_trader_portfolio:optimizer_p7_repair_confirmatory_round_complete"
        ]["errors"]
    )


def test_approval_grade_audit_accepts_complete_optimizer_p6_p7_evidence(
    tmp_path: Path,
) -> None:
    root = _write_optimizer_root(tmp_path, "crypto_trader_portfolio")
    _write_optimizer_p6_p7_artifacts(root)

    checks = optimizer_evidence_checks("crypto_trader_portfolio", tmp_path)

    assert all(check["passed"] for check in checks)


def test_approval_grade_audit_rejects_unevaluated_round_patch(
    tmp_path: Path,
) -> None:
    root = _write_optimizer_root(tmp_path, "crypto_trader_portfolio")
    _write_optimizer_p6_p7_artifacts(root, adopted=True)
    recommendation_path = root / "round_n_plus_1_recommendation.json"
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    recommendation["evaluated_patch_fingerprint"] = "not-the-selected-evaluation"
    recommendation_path.write_text(json.dumps(recommendation), encoding="utf-8")

    checks = optimizer_evidence_checks("crypto_trader_portfolio", tmp_path)
    by_name = {check["name"]: check for check in checks}

    p7 = by_name["crypto_trader_portfolio:optimizer_p7_repair_confirmatory_round_complete"]
    assert p7["passed"] is False
    assert any("fingerprint" in error for error in p7["errors"])


def test_approval_grade_audit_rejects_smoke_optimizer_roots(
    tmp_path: Path,
) -> None:
    smoke_root = (
        tmp_path
        / "trading_assistant_backtest"
        / "artifacts"
        / "validation"
        / "monthly_smoke"
        / "crypto_trader_portfolio"
    )
    smoke_root.mkdir(parents=True)
    (smoke_root / "artifact_index.json").write_text(
        json.dumps({"artifact_root": str(smoke_root)}),
        encoding="utf-8",
    )
    _write_optimizer_p6_p7_artifacts(smoke_root, adopted=True)
    _write_optimizer_manifest(smoke_root, "crypto_trader_portfolio")
    generic_monthly_root = (
        tmp_path
        / "trading_assistant_backtest"
        / "artifacts"
        / "monthly"
        / "crypto_trader_portfolio"
    )
    generic_monthly_root.mkdir(parents=True)
    _write_optimizer_p6_p7_artifacts(generic_monthly_root, adopted=True)
    _write_optimizer_manifest(generic_monthly_root, "crypto_trader_portfolio")

    checks = optimizer_evidence_checks("crypto_trader_portfolio", tmp_path)
    by_name = {check["name"]: check for check in checks}

    assert "crypto_trader_portfolio" not in build_optimizer_manifest_index(tmp_path)
    assert by_name[
        "crypto_trader_portfolio:optimizer_p6_true_fold_scoring_complete"
    ]["passed"] is False
    assert any(
        "optimizer_run_manifest" in error
        for error in by_name[
            "crypto_trader_portfolio:optimizer_p6_true_fold_scoring_complete"
        ]["errors"]
    )


def test_optimizer_evidence_rejects_smoke_run_mode_in_approved_root(
    tmp_path: Path,
) -> None:
    root = _write_optimizer_root(tmp_path, "crypto_trader_portfolio")
    _write_optimizer_p6_p7_artifacts(root, adopted=True)
    manifest_path = root / "optimizer_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_mode"] = "smoke_repair"
    manifest["approval_grade_optimizer_run"] = True
    manifest["smoke_mode"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    checks = optimizer_evidence_checks("crypto_trader_portfolio", tmp_path)
    by_name = {check["name"]: check for check in checks}

    p6 = by_name["crypto_trader_portfolio:optimizer_p6_true_fold_scoring_complete"]
    p7 = by_name["crypto_trader_portfolio:optimizer_p7_repair_confirmatory_round_complete"]
    assert p6["passed"] is False
    assert p7["passed"] is False
    assert any("smoke optimizer mode" in error for error in p6["errors"])


def test_optimizer_evidence_requires_promoted_context_hash_set(
    tmp_path: Path,
) -> None:
    root = _write_optimizer_root(tmp_path, "crypto_trader_portfolio")
    _write_optimizer_p6_p7_artifacts(root, adopted=True)
    expected_context = {
        "run_month": "2026-04",
        "data_bundle_checksums": ["bundle-sha"],
        "bridge_contract_hashes": {
            "crypto_trend_v1": "trend-contract-hash",
            "crypto_momentum_v1": "momentum-contract-hash",
            "crypto_breakout_v1": "breakout-contract-hash",
        },
        "deployment_metadata_hashes": {
            "crypto_trend_v1": "trend-metadata-hash",
            "crypto_momentum_v1": "momentum-metadata-hash",
            "crypto_breakout_v1": "breakout-metadata-hash",
        },
    }

    checks = optimizer_evidence_checks(
        "crypto_trader_portfolio",
        tmp_path,
        expected_context=expected_context,
    )
    errors = [error for check in checks for error in check["errors"]]

    assert any("missing promoted strategy contract hash" in error for error in errors)
    assert any("missing promoted deployment metadata hash" in error for error in errors)

    manifest_path = root / "optimizer_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "bridge_contract_hashes": expected_context["bridge_contract_hashes"],
            "strategy_plugin_contract_hashes": expected_context["bridge_contract_hashes"],
            "deployment_metadata_hashes": expected_context["deployment_metadata_hashes"],
            "bridge_deployment_metadata_hashes": expected_context[
                "deployment_metadata_hashes"
            ],
            "data_bundle_checksums": ["bundle-sha"],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    checks = optimizer_evidence_checks(
        "crypto_trader_portfolio",
        tmp_path,
        expected_context=expected_context,
    )

    assert all(check["passed"] for check in checks)


def test_optimizer_evidence_blocks_promoted_context_hash_mismatch(
    tmp_path: Path,
) -> None:
    root = _write_optimizer_root(tmp_path, "crypto_trader_portfolio")
    _write_optimizer_p6_p7_artifacts(root, adopted=True)
    expected_context = {
        "run_month": "2026-04",
        "data_bundle_checksums": ["bundle-sha"],
        "bridge_contract_hashes": {
            "crypto_trader_portfolio": "wrong-contract-hash",
        },
        "deployment_metadata_hashes": {
            "crypto_trader_portfolio": "wrong-metadata-hash",
        },
    }

    checks = optimizer_evidence_checks(
        "crypto_trader_portfolio",
        tmp_path,
        expected_context=expected_context,
    )
    errors = [error for check in checks for error in check["errors"]]

    assert any("promoted strategy contract hash mismatch" in error for error in errors)
    assert any("promoted deployment metadata hash mismatch" in error for error in errors)
    assert any(not check["passed"] for check in checks)


def test_approval_grade_audit_recomputes_copied_evaluated_patch_fingerprint(
    tmp_path: Path,
) -> None:
    root = _write_optimizer_root(tmp_path, "crypto_trader_portfolio")
    _write_optimizer_p6_p7_artifacts(root, adopted=True)
    copied_wrong = "f" * 64
    fold_rows = [
        json.loads(line)
        for line in (root / "fold_candidate_results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    for row in fold_rows:
        row["candidate"]["evaluated_patch_fingerprint"] = copied_wrong
    (root / "fold_candidate_results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in fold_rows) + "\n",
        encoding="utf-8",
    )
    selected = json.loads((root / "selected_candidates.json").read_text(encoding="utf-8"))
    selected[0]["evaluated_patch_fingerprint"] = copied_wrong
    (root / "selected_candidates.json").write_text(json.dumps(selected), encoding="utf-8")
    recommendation_path = root / "round_n_plus_1_recommendation.json"
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    recommendation["evaluated_patch_fingerprint"] = copied_wrong
    recommendation_path.write_text(json.dumps(recommendation), encoding="utf-8")

    checks = optimizer_evidence_checks("crypto_trader_portfolio", tmp_path)

    assert any(not check["passed"] for check in checks)
    assert any(
        "canonical" in error or "evaluated patch fingerprint" in error
        for check in checks
        for error in check["errors"]
    )


def _write_crypto_contract_pair(root: Path) -> tuple[Path, Path]:
    contract_dir = root / "contracts" / "strategy_plugins" / "crypto_trend_v1"
    contract_dir.mkdir(parents=True)
    contract_path = contract_dir / "strategy_plugin_contract.json"
    metadata_path = contract_dir / "deployment_metadata.json"
    contract = {
        "plugin_id": "crypto-trend-v1",
        "live_repo_path": "live",
        "live_repo_commit_sha": "a" * 40,
        "backtest_adapter_path": "adapter.py",
        "backtest_adapter_commit_sha": "b" * 64,
        "config_schema_version": "config_v1",
        "decision_api_version": "decision_v1",
        "required_telemetry_schemas": ["trade_event_v1"],
        "supported_symbols": ["BTC"],
        "supported_timeframes": ["1m"],
        "parity_fixture_set": ["fixture.json"],
        "maturity": "approval_ready",
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    metadata = {
        "bot_id": "crypto_portfolio",
        "strategy_id": "btc_1m",
        "repo_url": "https://github.com/example/crypto_trader.git",
        "deployed_commit_sha": "a" * 40,
        "config_hash": "c" * 64,
        "strategy_version": "crypto_trend_v1",
        "config_version": "trend_config",
        "telemetry_schema_version": "trade_event_v1",
        "deployment_id": "deployment-1",
        "strategy_plugin_contract_path": "strategy_plugin_contract.json",
        "strategy_plugin_contract_hash": "",
        "metadata_source": "vps_live_bot_runtime_deployment_metadata_v1",
        "emission_environment": "paper_vps",
        "emission_context": "runtime_startup",
        "emitted_at_utc": "2026-05-31T12:00:00Z",
        "live_runtime_started_at_utc": "2026-05-31T12:00:00Z",
        "runtime_entrypoint": "crypto-trader paper",
        "runtime_instance_id": "crypto-paper-vps-1",
        "runtime_host_fingerprint": "host-" + "d" * 16,
        "source_control_origin": "https://github.com/example/crypto_trader.git",
        "source_control_commit_sha": "a" * 40,
        "source_control_worktree_clean": True,
        "dry_run": False,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return contract_path, metadata_path


def _write_git_live_repo(root: Path) -> tuple[Path, str]:
    repo_path = root / "live_repo"
    repo_path.mkdir()
    _git(repo_path, "init")
    _git(repo_path, "config", "user.email", "test@example.invalid")
    _git(repo_path, "config", "user.name", "Test User")
    _git(repo_path, "remote", "add", "origin", "https://github.com/acme/crypto_trader.git")
    (repo_path / "strategy.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo_path, "add", "strategy.py")
    _git(repo_path, "commit", "-m", "initial strategy")
    return repo_path, _git(repo_path, "rev-parse", "HEAD")


def _git(repo_path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _write_optimizer_root(root: Path, scope_id: str) -> Path:
    artifact_root = (
        root
        / "trading_assistant_backtest"
        / "artifacts"
        / "validation"
        / "optimizer"
        / scope_id
    )
    artifact_root.mkdir(parents=True)
    contract_path = artifact_root / "strategy_plugin_contract.json"
    contract_path.write_text(json.dumps({"scope_id": scope_id}), encoding="utf-8")
    deployment_path = artifact_root / "deployment_metadata.json"
    deployment_path.write_text(json.dumps({"scope_id": scope_id}), encoding="utf-8")
    run_manifest_path = artifact_root / "run_manifest.json"
    run_manifest = {
        "run_id": f"{scope_id}-2026-04",
        "run_month": "2026-04",
        "bot_id": scope_id,
        "strategy_id": scope_id,
        "data_bundle_checksum": "bundle-sha",
    }
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    _write_optimizer_manifest(artifact_root, scope_id)
    (artifact_root / "artifact_index.json").write_text(
        json.dumps({"artifact_root": str(artifact_root)}),
        encoding="utf-8",
    )
    return artifact_root


def _write_optimizer_manifest(artifact_root: Path, scope_id: str) -> None:
    contract_path = artifact_root / "strategy_plugin_contract.json"
    if not contract_path.exists():
        contract_path.write_text(json.dumps({"scope_id": scope_id}), encoding="utf-8")
    deployment_path = artifact_root / "deployment_metadata.json"
    if not deployment_path.exists():
        deployment_path.write_text(json.dumps({"scope_id": scope_id}), encoding="utf-8")
    run_manifest_path = artifact_root / "run_manifest.json"
    if not run_manifest_path.exists():
        run_manifest = {
            "run_id": f"{scope_id}-2026-04",
            "run_month": "2026-04",
            "bot_id": scope_id,
            "strategy_id": scope_id,
            "data_bundle_checksum": "bundle-sha",
        }
        run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    else:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    (artifact_root / "optimizer_run_manifest.json").write_text(json.dumps({
        "schema_version": "optimizer_approval_run_manifest_v1",
        "run_id": run_manifest["run_id"],
        "scope_id": scope_id,
        "scope_aliases": [scope_id],
        "bot_id": scope_id,
        "strategy_id": scope_id,
        "run_month": "2026-04",
        "run_mode": "phased_auto",
        "optimizer_mode": "approval_grade",
        "approval_mode": "manual_required",
        "approval_grade_optimizer_run": True,
        "smoke_mode": False,
        "artifact_root": str(artifact_root),
        "run_manifest_path": str(run_manifest_path),
        "run_manifest_hash": _sha256_file(run_manifest_path),
        "data_bundle_checksum": "bundle-sha",
        "strategy_plugin_contract_path": str(contract_path),
        "strategy_plugin_contract_hash": _sha256_file(contract_path),
        "strategy_plugin_contract_hashes": {
            scope_id: _sha256_file(contract_path),
        },
        "bridge_contract_hashes": {
            scope_id: _sha256_file(contract_path),
        },
        "deployment_metadata_path": str(deployment_path),
        "deployment_metadata_hash": _sha256_file(deployment_path),
        "deployment_metadata_hashes": {
            scope_id: _sha256_file(deployment_path),
        },
        "bridge_deployment_metadata_hashes": {
            scope_id: _sha256_file(deployment_path),
        },
    }), encoding="utf-8")


def _write_optimizer_p6_p7_artifacts(root: Path, *, adopted: bool = False) -> None:
    patch_fingerprint, evaluated_fingerprint = _optimizer_patch_fingerprints()
    candidate_replay = {
        "trade_count": 4,
        "net_return": 1.12,
        "max_drawdown": 0.04,
        "profit_factor": 1.6,
        "objective_score": 1.12,
        "trade_hash": "fixture-candidate-trades",
        "order_hash": "fixture-candidate-orders",
        "coverage": [{"rows": 4}],
        "parameter_patch": _optimizer_config_patch(),
        "evaluated_parameter_patch": _optimizer_config_patch(),
        "parameter_patch_fingerprint": patch_fingerprint,
        "evaluated_patch_fingerprint": evaluated_fingerprint,
        "evaluated_parameters": _optimizer_evaluated_parameters(),
    }
    (root / "fold_manifest.json").write_text(json.dumps({
        "folds": [
            {"fold_id": "fold_1", "purged": True},
            {"fold_id": "fold_2", "purged": True},
        ],
    }), encoding="utf-8")
    (root / "fold_score_matrix.json").write_text(json.dumps({
        "selection_oos_excluded_from_first_pass": True,
        "scoring_windows": [
            {"fold_id": "fold_1", "purged": True},
            {"fold_id": "fold_2", "purged": True},
        ],
        "candidate_count": 1,
        "candidates": [{"candidate_id": "cand-1"}],
    }), encoding="utf-8")
    (root / "fold_candidate_results.jsonl").write_text(
        json.dumps({
            "candidate_id": "cand-1",
            "fold_id": "fold_1",
            "selection_oos_used_in_first_pass": False,
            "candidate": candidate_replay,
        }) + "\n" + json.dumps({
            "candidate_id": "cand-1",
            "fold_id": "fold_2",
            "selection_oos_used_in_first_pass": False,
            "candidate": candidate_replay,
        }) + "\n",
        encoding="utf-8",
    )
    (root / "selection_oos_evaluation.json").write_text(json.dumps({
        "selection_oos_used_after_fold_ranking": True,
        "selection_oos_used_in_first_pass": False,
        "primary_candidate_id": "cand-1" if adopted else "",
    }), encoding="utf-8")
    (root / "selection_oos_repair_trigger.json").write_text(json.dumps({
        "status": "not_triggered",
        "triggered": False,
        "thresholds": {"objective_drop_threshold": -0.05},
        "measured_degradation": {"objective_delta_vs_fold_mean": 0.02},
    }), encoding="utf-8")
    (root / "repair_failure_attribution.json").write_text(json.dumps({
        "status": "complete",
    }), encoding="utf-8")
    (root / "accepted_mutation_chain.json").write_text(json.dumps({
        "accepted_mutations": [],
    }), encoding="utf-8")
    (root / "repair_candidate_results.jsonl").write_text("", encoding="utf-8")
    (root / "repair_checkpoint.json").write_text(json.dumps({
        "repair_triggered": False,
    }), encoding="utf-8")
    if adopted:
        (root / "selected_candidates.json").write_text(json.dumps([{
            "candidate_id": "cand-1",
            "source": "phased_auto",
            "parameter_patch": _optimizer_config_patch(),
            "evaluated_parameter_patch": _optimizer_config_patch(),
            "evaluated_parameters": _optimizer_evaluated_parameters(),
            "parameter_patch_fingerprint": patch_fingerprint,
            "evaluated_patch_fingerprint": evaluated_fingerprint,
        }]), encoding="utf-8")
        (root / "confirmatory_rerank.json").write_text(json.dumps({
            "primary_candidate_id": "cand-1",
            "primary_source": "phased_auto",
            "compared_candidate_ids": ["cand-1"],
            "variants": [{"candidate_id": "cand-1"}],
            "adopted_candidate_id": "cand-1",
            "adopted_source": "phased_auto",
        }), encoding="utf-8")
        (root / "rounds_manifest.json").write_text(json.dumps({
            "adopted_candidate_id": "cand-1",
            "records": [{"candidate_id": "cand-1", "source": "phased_auto"}],
        }), encoding="utf-8")
        config_patch_path = root / "round_n_plus_1" / "config_patch.json"
        config_patch_path.parent.mkdir(parents=True, exist_ok=True)
        config_patch_path.write_text(json.dumps(_optimizer_config_patch()), encoding="utf-8")
        (root / "round_n_plus_1_recommendation.json").write_text(json.dumps({
            "status": "optimized_backtest_recommendation",
            "adopted_candidate_id": "cand-1",
            "config_patch_path": str(config_patch_path),
            "parameter_patch_fingerprint": patch_fingerprint,
            "evaluated_patch_fingerprint": evaluated_fingerprint,
            "evaluated_parameters": _optimizer_evaluated_parameters(),
        }), encoding="utf-8")
    else:
        (root / "confirmatory_rerank.json").write_text(json.dumps({
            "primary_candidate_id": "",
            "variants": [],
            "no_adoption_reason": "no replay-backed candidate passed purged folds",
        }), encoding="utf-8")
        (root / "rounds_manifest.json").write_text(json.dumps({
            "adopted_candidate_id": "",
            "no_adoption_reason": "no replay-backed candidate passed purged folds",
        }), encoding="utf-8")
        (root / "round_n_plus_1_recommendation.json").write_text(json.dumps({
            "status": "no_adoption",
            "no_adoption_reason": "no replay-backed candidate passed purged folds",
        }), encoding="utf-8")


def _optimizer_config_patch() -> dict:
    return {
        "family": "filter_repair",
        "filter_threshold_bps_delta": -2.0,
        "position_weight_multiplier": 1.05,
    }


def _optimizer_evaluated_parameters() -> dict:
    return {
        "threshold_bps": 8.0,
        "position_weight": 1.05,
        "max_positions": 1,
    }


def _optimizer_patch_fingerprints() -> tuple[str, str]:
    patch = _optimizer_config_patch()
    evaluated = {
        "parameter_patch": patch,
        "evaluated_parameters": _optimizer_evaluated_parameters(),
    }
    return _stable_json_hash(patch), _stable_json_hash(evaluated)


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _stable_json_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

