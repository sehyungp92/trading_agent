from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import trading_assistant_backtest.validation.approval_evidence_spine as spine
from trading_assistant_backtest.file_hashes import sha256_file


def test_approval_evidence_bundle_blocks_when_optimizer_manifest_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch, tmp_path, optimizer_ready=False)
    _write_production_evidence(tmp_path, optimizer=False)

    bundle = spine.run_approval_evidence_spine(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "bundle",
    )

    assert bundle["eligible_for_promotion"] is False
    assert any("optimizer_approval_manifest" in blocker for blocker in bundle["blockers"])
    assert (tmp_path / "bundle" / "approval_evidence_bundle.json").exists()


def test_approval_evidence_bundle_blocks_local_shadow_deployment_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch, tmp_path)
    _write_production_evidence(tmp_path, metadata_live=False)

    bundle = spine.run_approval_evidence_spine(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "bundle",
    )

    assert bundle["eligible_for_promotion"] is False
    assert any("repo_url is local/shadow-only" in blocker for blocker in bundle["blockers"])


def test_approval_evidence_bundle_blocks_missing_scheduled_shadow_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch, tmp_path)
    _write_production_evidence(tmp_path, scheduled_shadow=False)

    bundle = spine.run_approval_evidence_spine(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "bundle",
    )

    assert bundle["eligible_for_promotion"] is False
    assert any(
        "scheduled_shadow_cycle_report.json missing" in blocker
        for blocker in bundle["blockers"]
    )


def test_approval_evidence_bundle_blocks_fixture_only_scheduled_shadow_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch, tmp_path)
    _write_production_evidence(tmp_path, scheduled_shadow_source="decision_parity_fixture")

    bundle = spine.run_approval_evidence_spine(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "bundle",
    )

    assert bundle["eligible_for_promotion"] is False
    assert any(
        "source_kind must be 'monthly_validation_shadow'" in blocker
        for blocker in bundle["blockers"]
    )


def test_approval_evidence_bundle_keeps_ptg7_context_from_masking_production_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch, tmp_path)
    _write_production_evidence(tmp_path, learning_authoritative=False)

    bundle = spine.run_approval_evidence_spine(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "bundle",
    )
    checks = {check["name"]: check for check in bundle["required_checks"]}

    assert checks["ptg7_fixture_context_present_not_authority"]["passed"] is True
    assert bundle["eligible_for_promotion"] is False
    assert any("learning_sufficiency_authoritative" in blocker for blocker in bundle["blockers"])


def test_approval_evidence_bundle_blocks_manual_approval_ready_maturity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch, tmp_path, maturity="approval_ready", approval_ready=True)
    _write_production_evidence(tmp_path)

    bundle = spine.run_approval_evidence_spine(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "bundle",
    )

    assert bundle["eligible_for_promotion"] is False
    assert any("maturity must be shadow_validated" in blocker for blocker in bundle["blockers"])


def test_approval_evidence_bundle_blocks_when_live_config_verifier_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch, tmp_path, live_config_ok=False)
    _write_production_evidence(tmp_path)

    bundle = spine.run_approval_evidence_spine(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "bundle",
    )

    assert bundle["eligible_for_promotion"] is False
    assert any(
        "IARIC_v1 phase3 evidence status is missing" in blocker
        for blocker in bundle["blockers"]
    )


def test_live_config_verifier_keeps_unrelated_ibkr_failures_contextual(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout=(
                "PASS ibkr:IARIC_v1 - promotion manifest ok\n"
                "PASS ibkr:ALCB_v1 - promotion manifest ok\n"
                "FAIL ibkr:ATRSS - latest round is not frozen\n"
                "\nLive-config promotion check failed:\n"
                "- ibkr:ATRSS latest round is not frozen (missing)\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(spine.subprocess, "run", fake_run)

    path = spine._load_or_run_live_config_verifier(
        agent_root=tmp_path,
        source_root=tmp_path / "source_reports",
        bot_id="ibkr",
        scope=spine._scope_for("trading_stock_family"),
        refresh=True,
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    check = spine._live_config_promotion_check(path)

    assert report["ok"] is True
    assert report["failures"] == []
    assert report["out_of_scope_failures"] == [
        "ibkr:ATRSS latest round is not frozen (missing)",
    ]
    assert report["observed_scoped_strategy_ids"] == ["ALCB_v1", "IARIC_v1"]
    assert check["passed"] is True


def test_live_config_verifier_blocks_selected_scope_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout=(
                "PASS ibkr:IARIC_v1 - promotion manifest ok\n"
                "FAIL ibkr:ALCB_v1 - phase3 evidence status is missing\n"
                "\nLive-config promotion check failed:\n"
                "- ibkr:ALCB_v1 phase3 evidence status is missing\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(spine.subprocess, "run", fake_run)

    path = spine._load_or_run_live_config_verifier(
        agent_root=tmp_path,
        source_root=tmp_path / "source_reports",
        bot_id="ibkr",
        scope=spine._scope_for("trading_stock_family"),
        refresh=True,
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    check = spine._live_config_promotion_check(path)

    assert report["ok"] is False
    assert report["failures"] == ["ibkr:ALCB_v1 phase3 evidence status is missing"]
    assert check["passed"] is False
    assert check["errors"] == ["ibkr:ALCB_v1 phase3 evidence status is missing"]


def test_next_actions_distinguish_existing_fixture_manifest_with_missing_runtime_case() -> None:
    actions = spine._next_required_actions(
        {},
        [
            (
                "production_fixture_breadth_complete: fixture-set manifest missing "
                "required case class: live_shadow_telemetry_source"
            )
        ],
    )

    assert any("matching live/shadow runtime telemetry source" in action for action in actions)
    assert not any(
        "Add a production-derived parity fixture-set manifest" in action
        for action in actions
    )


def test_approval_evidence_bundle_blocks_production_fixture_hash_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch, tmp_path)
    _write_production_evidence(tmp_path, fixture_hash_ok=False)

    bundle = spine.run_approval_evidence_spine(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "bundle",
    )

    assert bundle["eligible_for_promotion"] is False
    assert any("source record hash mismatch" in blocker for blocker in bundle["blockers"])


def test_approval_evidence_bundle_blocks_placeholder_relay_auth(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch, tmp_path)
    _write_production_evidence(
        tmp_path,
        relay_overrides={"auth": {"secret_fingerprint": "change-me"}},
    )

    bundle = spine.run_approval_evidence_spine(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "bundle",
    )

    assert bundle["eligible_for_promotion"] is False
    assert any("placeholder HMAC secret fingerprint" in blocker for blocker in bundle["blockers"])


def test_approval_evidence_bundle_blocks_stale_relay_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch, tmp_path)
    _write_production_evidence(
        tmp_path,
        relay_overrides={
            "freshness": {
                "ok": True,
                "max_event_age_seconds": spine.RELAY_EVIDENCE_MAX_AGE_SECONDS + 1,
            }
        },
    )

    bundle = spine.run_approval_evidence_spine(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "bundle",
    )

    assert bundle["eligible_for_promotion"] is False
    assert any(
        "relay ingest evidence max_event_age_seconds is stale" in blocker
        for blocker in bundle["blockers"]
    )


def test_approval_evidence_bundle_blocks_relay_evidence_not_linked_to_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch, tmp_path)
    _write_production_evidence(
        tmp_path,
        relay_overrides={"deployment_metadata_hash": "0" * 64},
    )

    bundle = spine.run_approval_evidence_spine(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "bundle",
    )

    assert bundle["eligible_for_promotion"] is False
    assert any("deployment metadata hash mismatch" in blocker for blocker in bundle["blockers"])


def test_approval_evidence_bundle_becomes_eligible_only_when_every_input_is_green(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch, tmp_path)
    evidence = _write_production_evidence(tmp_path)

    bundle = spine.run_approval_evidence_spine(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "bundle",
    )

    assert bundle["eligible_for_promotion"] is True
    assert bundle["promotion_decision"] == "eligible"
    assert all(check["passed"] for check in bundle["required_checks"])
    assert str(evidence["optimizer_manifest"]) in bundle["evidence_hashes"]
    assert str(evidence["fixture_source"]) in bundle["evidence_hashes"]
    assert bundle["source_reports"]["bridge_readiness"].endswith("bridge_readiness_report.json")
    assert bundle["source_reports"]["validation_matrix"].endswith("validation_matrix_report.json")
    assert "Promote via guarded bridge promotion command." in bundle["next_required_actions"]


def _install_fakes(
    monkeypatch,
    agent_root: Path,
    *,
    maturity: str = "shadow_validated",
    approval_ready: bool = False,
    optimizer_ready: bool = True,
    live_config_ok: bool = True,
) -> None:
    def fake_bridge(*, artifact_root: Path, **_kwargs):
        artifact_root.mkdir(parents=True, exist_ok=True)
        report_path = artifact_root / "bridge_readiness_report.json"
        report = {
            "ok": True,
            "artifact_path": str(report_path),
            "bridges": [
                {
                    "repo_id": "trading_stock_family",
                    "status": "formal_decision_parity_passed",
                    "maturity": maturity,
                    "approval_ready": approval_ready,
                    "audit_passed": True,
                    "errors": [],
                    "evidence": [
                        {
                            "path": str(
                                agent_root
                                / "contracts"
                                / "strategy_plugins"
                                / "trading_stock_family"
                                / "strategy_plugin_contract.json"
                            )
                        },
                        {
                            "path": str(
                                agent_root
                                / "artifacts"
                                / "validation"
                                / "decision_parity_matrix"
                                / "trading_stock_family"
                                / "decision_parity"
                                / "decision_parity_report.json"
                            )
                        },
                    ],
                }
            ],
        }
        _write_json(report_path, report)
        return report

    def fake_matrix(*, artifact_root: Path, **_kwargs):
        artifact_root.mkdir(parents=True, exist_ok=True)
        report_path = artifact_root / "validation_matrix_report.json"
        artifact_path = agent_root / "validation-artifact.json"
        _write_json(artifact_path, {"ok": True})
        tests = {
            test: {"result": "pass", "artifact_paths": [str(artifact_path)]}
            for test in spine.VALIDATION_TESTS
        }
        report = {
            "ok": True,
            "artifact_path": str(report_path),
            "scopes": [
                {
                    "scope_id": "trading_stock_family",
                    "tests": tests,
                    "optimizer_evidence_context": {"run_month": "2026-06"},
                    "optimizer_approval_readiness": {"ready": optimizer_ready, "checks": []},
                }
            ],
        }
        _write_json(report_path, report)
        return report

    def fake_audit(*, artifact_root: Path, **_kwargs):
        artifact_root.mkdir(parents=True, exist_ok=True)
        report_path = artifact_root / "approval_grade_audit_report.json"
        report = {
            "schema_version": "approval_grade_audit_v1",
            "artifact_path": str(report_path),
            "approval_grade": False,
            "next_required_actions": ["Promote via guarded bridge promotion command."],
        }
        _write_json(report_path, report)
        return report

    def fake_ptg7(*, source_root: Path, **_kwargs):
        report_path = source_root / "ptg7_gate_report.json"
        command_path = source_root / spine.PTG7_COMMAND_REPORT
        _write_json(
            report_path,
            {
                "schema_version": "approval_ready_pilot_ptg7_gate_report_v1",
                "status": "blocked",
                "implementation_status": "pass",
                "pilot_bridge_id": "trading_stock_family",
            },
        )
        _write_json(command_path, {"ok": False, "returncode": 1})
        return report_path, command_path

    def fake_live_config(*, source_root: Path, **_kwargs):
        report_path = source_root / spine.LIVE_CONFIG_VERIFICATION_REPORT
        _write_json(
            report_path,
            (
                {"schema_version": "live_config_promotion_verification_v1", "ok": True}
                if live_config_ok
                else {
                    "schema_version": "live_config_promotion_verification_v1",
                    "ok": False,
                    "returncode": 1,
                    "failures": ["IARIC_v1 phase3 evidence status is missing"],
                }
            ),
        )
        return report_path

    def fake_operational(*, source_root: Path, **_kwargs):
        report_path = source_root / spine.OPERATIONAL_VERIFICATION_REPORT
        _write_json(
            report_path,
            {"schema_version": "operational_deployment_verification_v1", "ok": True},
        )
        return report_path

    def fake_latest_optimizer_root(*_args, **_kwargs):
        root = (
            agent_root
            / "trading_assistant_backtest"
            / "artifacts"
            / "validation"
            / "optimizer"
            / "trading_stock_family"
        )
        return root if optimizer_ready else None

    def fake_optimizer_checks(*_args, **_kwargs):
        return [
            {
                "name": "trading_stock_family:optimizer_p6_true_fold_scoring_complete",
                "passed": optimizer_ready,
                "errors": []
                if optimizer_ready
                else [
                    "missing explicit approval-grade optimizer_run_manifest.json "
                    "for promoted scope"
                ],
            },
            {
                "name": "trading_stock_family:optimizer_p7_repair_confirmatory_round_complete",
                "passed": optimizer_ready,
                "errors": []
                if optimizer_ready
                else [
                    "missing explicit approval-grade optimizer_run_manifest.json "
                    "for promoted scope"
                ],
            },
        ]

    monkeypatch.setattr(spine, "run_bridge_readiness_audit", fake_bridge)
    monkeypatch.setattr(spine, "run_validation_matrix_audit", fake_matrix)
    monkeypatch.setattr(spine, "run_approval_grade_audit", fake_audit)
    monkeypatch.setattr(spine, "_load_or_run_ptg7_gate", fake_ptg7)
    monkeypatch.setattr(spine, "_load_or_run_live_config_verifier", fake_live_config)
    monkeypatch.setattr(spine, "_load_or_run_operational_verifier", fake_operational)
    monkeypatch.setattr(spine, "build_optimizer_manifest_index", lambda _root: {})
    monkeypatch.setattr(spine, "latest_optimizer_artifact_root", fake_latest_optimizer_root)
    monkeypatch.setattr(spine, "optimizer_evidence_checks", fake_optimizer_checks)


def _write_production_evidence(
    root: Path,
    *,
    metadata_live: bool = True,
    scheduled_shadow: bool = True,
    scheduled_shadow_source: str = "monthly_validation_shadow",
    optimizer: bool = True,
    learning_authoritative: bool = True,
    fixture_hash_ok: bool = True,
    relay_overrides: dict | None = None,
) -> dict[str, Path]:
    contract_path, metadata_path = _write_contract_pair(root, metadata_live=metadata_live)
    parity_report = (
        root
        / "artifacts"
        / "validation"
        / "decision_parity_matrix"
        / "trading_stock_family"
        / "decision_parity"
        / "decision_parity_report.json"
    )
    _write_json(parity_report, {"status": "pass"})
    install_report = (
        root
        / "artifacts"
        / "validation"
        / "deployment_metadata_install"
        / "deployment_metadata_install_report.json"
    )
    _write_json(
        install_report,
        {
            "schema_version": "deployment_metadata_install_report_v1",
            "bridge_id": "trading_stock_family",
            "ok": True,
            "installed": True,
            "installed_path": str(metadata_path),
            "metadata_path": str(metadata_path),
            "contract_path": str(contract_path),
        },
    )
    operational_evidence = root / "deployments" / "operational_evidence.json"
    _write_json(operational_evidence, {"schema_version": "trading_agent_operational_evidence_v1"})
    learning_manifest = (
        root
        / "artifacts"
        / "learning_sufficiency"
        / "phase2_manifests"
        / "ibkr"
        / "2026-06"
        / "trading_stock_family"
        / "learning_sufficiency_manifest.json"
    )
    _write_json(
        learning_manifest,
        {
            "manifest_version": "learning_sufficiency_manifest_v1",
            "eligibility": (
                "learning_authoritative" if learning_authoritative else "diagnostics_only"
            ),
            "gaps": [{"remediation": "Provide real runtime trade outcome lineage."}],
        },
    )
    telemetry_manifest = learning_manifest.with_name("telemetry_manifest.json")
    _write_json(
        telemetry_manifest,
        {
            "manifest_version": "telemetry_manifest_v1",
            "bot_id": "ibkr",
            "strategy_id": "trading_stock_family",
            "total_events": 3,
            "event_counts_by_type": {"order": 1, "fill": 1, "trade": 1},
            "missing_field_counts": {"event_id": 0, "bot_id": 0, "event_type": 0},
        },
    )
    optimizer_manifest = (
        root
        / "trading_assistant_backtest"
        / "artifacts"
        / "validation"
        / "optimizer"
        / "trading_stock_family"
        / "optimizer_run_manifest.json"
    )
    if optimizer:
        _write_json(optimizer_manifest, {"schema_version": "optimizer_approval_run_manifest_v1"})
    monthly_result = root / "monthly_validation_result.json"
    relay_ingest = root / "relay_ingest_evidence.json"
    _write_json(monthly_result, {"ok": True})
    relay_payload = {
        "ok": True,
        "bot_id": "ibkr",
        "event_id": "relay-heartbeat-1",
        "effective_config_hash": "b" * 64,
        "deployment_id": "deployment-1",
        "runtime_instance_id": "ibkr-paper-vps-1",
        "deployment_metadata_hash": sha256_file(metadata_path),
        "auth": {"secret_fingerprint": "hmac-sha256:abcdef1234567890"},
        "freshness": {"ok": True, "max_event_age_seconds": 60},
        "generated_at": datetime.now(UTC).isoformat(),
        "classification_counts": {"enqueued": 3, "duplicate": 0},
        "dropped_field_counts": {},
        "unknown_field_counts": {},
        "sample_events": [
            {
                "event_id": "evt-1",
                "bot_id": "ibkr",
                "event_type": "order",
                "payload": {"status": "accepted"},
            }
        ],
    }
    if relay_overrides:
        relay_payload.update(relay_overrides)
    _write_json(
        relay_ingest,
        relay_payload,
    )
    if scheduled_shadow:
        shadow = (
            root
            / "artifacts"
            / "validation"
            / "scheduled_shadow"
            / "trading_stock_family"
            / "run-1"
            / "scheduled_shadow_cycle_report.json"
        )
        _write_json(
            shadow,
            {
                "schema_version": "scheduled_shadow_cycle_report_v1",
                "scope_id": "trading_stock_family",
                "bridge_ids": ["trading_stock_family"],
                "run_id": "run-1",
                "run_month": "2026-06",
                "bot_id": "ibkr",
                "monthly_validation_result_path": str(monthly_result),
                "deployment_metadata_install_report_paths": [str(install_report)],
                "operational_evidence_path": str(operational_evidence),
                "relay_ingest_evidence_path": str(relay_ingest),
                "learning_sufficiency_manifest_path": str(learning_manifest),
                "optimizer_run_manifest_path": str(optimizer_manifest),
                "approval_evidence_mode": True,
                "uses_live_vps_metadata": True,
                "adoption_disabled": True,
                "source_kind": scheduled_shadow_source,
                "ok": True,
                "blockers": [],
            },
        )
    fixture_source = root / "fixture-source.json"
    _write_json(fixture_source, {"case": "accepted_entry"})
    fixture_manifest = (
        root
        / "artifacts"
        / "validation"
        / "decision_parity_matrix"
        / "trading_stock_family"
        / "production_fixture_set_manifest.json"
    )
    case_classes = [
        "accepted_entry",
        "blocked_no_trade",
        "risk_portfolio_denial",
        "exit_close",
        "order_fill",
        "live_shadow_telemetry_source",
    ]
    _write_json(
        fixture_manifest,
        {
            "schema_version": "production_fixture_set_manifest_v1",
            "status": "pass",
            "ok": True,
            "source_kind": "production_derived_live_shadow",
            "case_classes": case_classes,
            "source_records": [
                {
                    "path": str(fixture_source),
                    "sha256": (
                        sha256_file(fixture_source)
                        if fixture_hash_ok
                        else "0" * 64
                    ),
                }
            ],
        },
    )
    return {
        "contract": contract_path,
        "metadata": metadata_path,
        "optimizer_manifest": optimizer_manifest,
        "fixture_source": fixture_source,
        "learning_manifest": learning_manifest,
    }


def _write_contract_pair(root: Path, *, metadata_live: bool) -> tuple[Path, Path]:
    contract_dir = root / "contracts" / "strategy_plugins" / "trading_stock_family"
    contract_path = contract_dir / "strategy_plugin_contract.json"
    metadata_path = contract_dir / "deployment_metadata.json"
    contract = {
        "contract_version": "strategy_plugin_contract_v1",
        "plugin_id": "trading-stock-family",
        "live_repo_path": "trading/ibkr_trader",
        "live_repo_commit_sha": "a" * 40,
        "backtest_adapter_path": "adapter.py",
        "backtest_adapter_commit_sha": "b" * 64,
        "config_schema_version": "config_v1",
        "decision_api_version": "decision_v1",
        "required_telemetry_schemas": ["trade_event_v1", "assistant_event_v1"],
        "supported_symbols": ["MSFT"],
        "supported_timeframes": ["5m"],
        "parity_fixture_set": ["fixture.json"],
        "maturity": "shadow_validated",
    }
    _write_json(contract_path, contract)
    metadata = {
        "bot_id": "trading",
        "strategy_id": "trading_stock_family",
        "repo_url": "https://github.com/example/ibkr_trading.git",
        "deployed_commit_sha": "a" * 40,
        "config_hash": "c" * 64,
        "strategy_version": "trading_stock_family",
        "config_version": "config_v1",
        "telemetry_schema_version": "trade_event_v1",
        "telemetry_schema_versions": ["trade_event_v1", "assistant_event_v1"],
        "deployment_id": "deployment-1",
        "strategy_plugin_contract_path": "strategy_plugin_contract.json",
        "strategy_plugin_contract_hash": sha256_file(contract_path),
        "metadata_source": "vps_live_bot_runtime_deployment_metadata_v1",
        "emission_environment": "paper_vps",
        "emission_context": "runtime_startup",
        "emitted_at_utc": "2026-06-30T12:00:00Z",
        "live_runtime_started_at_utc": "2026-06-30T12:00:00Z",
        "runtime_entrypoint": "ibkr-trading paper",
        "runtime_instance_id": "ibkr-paper-vps-1",
        "runtime_host_fingerprint": "host-" + "d" * 16,
        "source_control_origin": "https://github.com/example/ibkr_trading.git",
        "source_control_commit_sha": "a" * 40,
        "source_control_worktree_clean": True,
        "dry_run": False,
    }
    if not metadata_live:
        metadata.update(
            {
                "metadata_source": "local clean live-repo checkout shadow snapshot",
                "repo_url": "local://trading/ibkr_trader",
                "source_control_origin": "local://trading/ibkr_trader",
            }
        )
    _write_json(metadata_path, metadata)
    return contract_path, metadata_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
