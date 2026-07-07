from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from trading_assistant.contracts.monthly_runner_contract import validate_manifest_file
from trading_assistant.schemas.data_bundle_manifest import DataBundleManifest, DataBundleSlice, DataBundleStatus
from trading_assistant.schemas.decision_parity import DECISION_PARITY_DIMENSIONS, DecisionParityReport
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest, MonthlyRunMode
from trading_assistant.schemas.strategy_plugin_contract import StrategyPluginContract, StrategyPluginMaturity
from trading_assistant.skills.backtest_runner_client import BacktestRunnerClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _manifest(tmp_path: Path, *, mode: MonthlyRunMode = MonthlyRunMode.INCUMBENT_VALIDATION, failure: str = "") -> tuple[MonthlyRunManifest, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact_root = tmp_path / "artifacts"
    data_bundle_path = tmp_path / "data_bundle_manifest.json"
    strategy_contract_path = tmp_path / "strategy_plugin_contract.json"
    bundle = DataBundleManifest(
        data_repo_path=str(tmp_path / "market_data"),
        data_repo_commit_sha="fixture-data-sha",
        slice_manifests=[
            DataBundleSlice(
                manifest_path=str(tmp_path / "slice.json"),
                manifest_id="slice-1",
                source="fixture",
                market="equity",
                symbol="AAPL",
                timeframe="1m",
                checksum="slice-sha",
                calendar="XNYS",
                authoritative=True,
            )
        ],
        calendars=["XNYS"],
        fee_model_version="fees_v1",
        slippage_model_version="slippage_v1",
        adjustment_policy="split_adjusted",
        status=DataBundleStatus.AUTHORITATIVE,
    )
    data_bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    strategy_contract_path.write_text(StrategyPluginContract(
        plugin_id="strat1-plugin",
        live_repo_path=str(tmp_path / "live"),
        live_repo_commit_sha="fixture-live-sha",
        backtest_adapter_path="adapters/strat1.py",
        backtest_adapter_commit_sha="fixture-backtest-sha",
        config_schema_version="config_v1",
        decision_api_version="decision_api_v1",
        required_telemetry_schemas=["trade_event_v1"],
        supported_symbols=["AAPL"],
        supported_timeframes=["1m"],
        parity_fixture_set=[str(tmp_path / "parity_fixture.json")],
        maturity=StrategyPluginMaturity.APPROVAL_READY,
    ).model_dump_json(indent=2), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "tests.contract_fixtures.monthly_runner_contract.runner",
        "--manifest",
        "{manifest}",
    ]
    if failure:
        command.extend(["--failure", failure])
    manifest = MonthlyRunManifest(
        run_id="monthly-bot1-strat1-2026-04",
        run_month="2026-04",
        mode=mode,
        bot_id="bot1",
        strategy_id="strat1",
        latest_month_start=date(2026, 4, 1),
        latest_month_end=date(2026, 4, 30),
        calibration_start=date(2026, 1, 1),
        calibration_end=date(2026, 3, 31),
        in_sample_start=date(2026, 1, 1),
        in_sample_end=date(2026, 3, 31),
        selection_oos_start=date(2026, 4, 1),
        selection_oos_end=date(2026, 4, 30),
        market_data_manifest_path=str(data_bundle_path),
        data_bundle_manifest_path=str(data_bundle_path),
        data_bundle_checksum=bundle.bundle_checksum,
        data_manifest_checksum=bundle.bundle_checksum,
        telemetry_manifest_path=str(tmp_path / "telemetry.json"),
        backtest_repo_path=str(PROJECT_ROOT),
        backtest_repo_commit_sha="fixture-backtest-sha",
        trading_repo_commit_sha="fixture-live-sha",
        control_plane_commit_sha="fixture-control-sha",
        artifact_root=str(artifact_root),
        backtest_command=command,
        strategy_plugin_id="strat1-plugin",
        strategy_plugin_contract_path=str(strategy_contract_path),
        round_id="round_1",
        prior_round_id="round_0",
        next_round_id="round_2",
    )
    manifest_path = artifact_root / "run_manifest.json"
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest, manifest_path


def test_fixture_runner_valid_incumbent_emits_required_artifacts(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(tmp_path)

    result = BacktestRunnerClient(timeout_seconds=30).run(manifest, manifest_path)

    assert result.success is True
    assert result.artifact_index is not None
    assert result.artifact_index.missing_required() == []


def test_contract_fails_missing_artifact_index(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(tmp_path, failure="missing_index")

    result = BacktestRunnerClient(timeout_seconds=30).run(manifest, manifest_path)

    assert result.success is False
    assert "missing artifact_index.json" in result.error


def test_contract_fails_path_outside_artifact_root(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(tmp_path, failure="outside_path")

    result = BacktestRunnerClient(timeout_seconds=30).run(manifest, manifest_path)

    assert result.success is False
    assert "outside artifact_root" in result.error


def test_contract_fails_stale_and_malformed_artifacts(tmp_path: Path) -> None:
    stale_manifest, stale_path = _manifest(tmp_path / "stale", failure="stale_artifact")
    stale = BacktestRunnerClient(timeout_seconds=30).run(stale_manifest, stale_path)
    assert stale.success is False
    assert "stale artifacts older than run manifest" in stale.error

    malformed_manifest, malformed_path = _manifest(tmp_path / "malformed", failure="malformed_json")
    malformed = BacktestRunnerClient(timeout_seconds=30).run(malformed_manifest, malformed_path)
    assert malformed.success is False
    assert "malformed required artifacts" in malformed.error


def test_optimizer_contract_validates_with_cli(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(tmp_path, mode=MonthlyRunMode.PHASED_AUTO)
    result = BacktestRunnerClient(timeout_seconds=30).run(manifest, manifest_path)
    assert result.success is True

    validation = validate_manifest_file(manifest_path)
    assert validation.valid is True

    completed = subprocess.run(
        [sys.executable, "-m", "trading_assistant.contracts.validate_monthly_runner", "--manifest", str(manifest_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["valid"] is True


def test_optimizer_contract_requires_core_sequence_artifacts(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(
        tmp_path,
        mode=MonthlyRunMode.PHASED_AUTO,
        failure="missing_optimizer_diagnostics",
    )
    result = BacktestRunnerClient(timeout_seconds=30).run(manifest, manifest_path)
    assert result.success is True

    validation = validate_manifest_file(manifest_path)

    assert validation.valid is False
    assert any("missing core phase4 optimizer artifacts" in error for error in validation.errors)


def test_optimizer_contract_fails_structural_candidate_without_patches(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(
        tmp_path,
        mode=MonthlyRunMode.PHASED_AUTO,
        failure="structural_missing_patch",
    )

    result = BacktestRunnerClient(timeout_seconds=30).run(manifest, manifest_path)

    assert result.success is True
    validation = validate_manifest_file(manifest_path)
    assert validation.valid is False
    assert any("structural candidate missing required lineage" in error for error in validation.errors)


def test_optimizer_contract_fails_structural_candidate_with_mismatched_decision_parity(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(
        tmp_path,
        mode=MonthlyRunMode.PHASED_AUTO,
        failure="decision_parity_mismatch",
    )

    result = BacktestRunnerClient(timeout_seconds=30).run(manifest, manifest_path)

    assert result.success is True
    validation = validate_manifest_file(manifest_path)
    assert validation.valid is False
    assert any("decision parity strategy_plugin_id" in error for error in validation.errors)


def test_optimizer_contract_fails_structural_candidate_without_deployment_metadata(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(
        tmp_path,
        mode=MonthlyRunMode.PHASED_AUTO,
        failure="decision_parity_mismatch",
    )

    result = BacktestRunnerClient(timeout_seconds=30).run(manifest, manifest_path)

    assert result.success is True
    validation = validate_manifest_file(manifest_path)
    assert validation.valid is False
    assert any(
        "structural candidate requires deployment metadata evidence" in error
        for error in validation.errors
    )


def test_optimizer_contract_fails_structural_candidate_with_missing_decision_parity_evidence(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(
        tmp_path,
        mode=MonthlyRunMode.PHASED_AUTO,
        failure="decision_parity_mismatch",
    )
    result = BacktestRunnerClient(timeout_seconds=30).run(manifest, manifest_path)
    assert result.success is True

    missing = tmp_path / "artifacts" / "missing_decision_evidence.json"
    (tmp_path / "artifacts" / "decision_parity_report.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "candidate_id": "fixture-structural",
        "strategy_plugin_id": manifest.strategy_plugin_id,
        "live_repo_commit_sha": manifest.trading_repo_commit_sha,
        "backtest_adapter_commit_sha": manifest.backtest_repo_commit_sha,
        "status": "pass",
        "evidence_paths": [str(missing)],
        "checks": [
            {
                "dimension": dimension,
                "status": "pass",
                "match_rate": 1.0,
                "mismatch_count": 0,
                "evidence_paths": [str(missing)],
            }
            for dimension in sorted(DECISION_PARITY_DIMENSIONS)
        ],
    }), encoding="utf-8")

    validation = validate_manifest_file(manifest_path)

    assert validation.valid is False
    assert any("decision parity evidence path does not exist" in error for error in validation.errors)


def test_mature_strategy_plugin_contract_requires_full_lineage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mature strategy plugin contract missing required fields"):
        StrategyPluginContract(
            plugin_id="strat1-plugin",
            backtest_adapter_path="adapters/strat1.py",
            config_schema_version="config_v1",
            decision_api_version="decision_api_v1",
            maturity=StrategyPluginMaturity.APPROVAL_READY,
        )


def test_authoritative_data_bundle_requires_frozen_repo_and_policy_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="authoritative data bundle missing required fields"):
        DataBundleManifest(
            data_repo_path=str(tmp_path / "market_data"),
            slice_manifests=[
                DataBundleSlice(
                    manifest_path=str(tmp_path / "slice.json"),
                    source="fixture",
                    market="equity",
                    symbol="AAPL",
                    timeframe="1m",
                    authoritative=True,
                )
            ],
            status=DataBundleStatus.AUTHORITATIVE,
        )


def test_passing_decision_parity_requires_lineage_and_evidence() -> None:
    with pytest.raises(ValueError, match="decision parity pass missing required fields"):
        DecisionParityReport(
            run_id="run1",
            candidate_id="cand1",
            status="pass",
            checks=[
                {
                    "dimension": dimension,
                    "status": "pass",
                    "match_rate": 1.0,
                    "mismatch_count": 0,
                    "evidence_paths": ["diagnostics.json"],
                }
                for dimension in sorted(DECISION_PARITY_DIMENSIONS)
            ],
        )
