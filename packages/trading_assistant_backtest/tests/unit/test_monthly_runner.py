from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from tests.paths import MONOREPO_ROOT, package_workspace
from trading_assistant_backtest.auto.types import Candidate, CandidateEvaluation, PhaseSpec
from trading_assistant_backtest.contract_loader import validate_manifest_file
from trading_assistant_backtest.contract_models import (
    DataBundleManifest,
    DataBundleSlice,
    DataBundleStatus,
    MonthlyRunManifest,
    MonthlyRunMode,
)
from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.artifact_writer import ArtifactWriter
from trading_assistant_backtest.monthly_execution.optimizer_sequence import (
    _optimizer_run_manifest_payload,
    write_optimizer_artifacts,
)
from trading_assistant_backtest.monthly_execution.replay_context import ReplayEvaluationContext
from trading_assistant_backtest.monthly_execution.structural_registry import (
    STRUCTURAL_PARITY_BUILDERS,
)
from trading_assistant_backtest.replay.types import ReplayResult, WindowSpec
from trading_assistant_backtest.strategies.bar_replay import BarReplayConfig, _candidate_params
from trading_assistant_backtest.strategies.contracts import load_strategy_plugin_contract
from trading_assistant_backtest.strategies.crypto.trend import (
    DECISION_API_VERSION as CRYPTO_TREND_DECISION_API_VERSION,
)
from trading_assistant_backtest.strategies.crypto.trend import (
    PLUGIN_ID as CRYPTO_TREND_PLUGIN_ID,
)
from trading_assistant_backtest.strategies.deployment import load_deployment_metadata
from trading_assistant_backtest.strategies.krx.olr_kalcb import (
    DECISION_API_VERSION as K_STOCK_DECISION_API_VERSION,
)
from trading_assistant_backtest.strategies.krx.olr_kalcb import (
    PLUGIN_ID as K_STOCK_PLUGIN_ID,
)
from trading_assistant_backtest.strategies.plugin_semantics import (
    evaluated_patch_payload,
    round_n_plus_1_payload,
)
from trading_assistant_backtest.strategies.trading.stock import (
    DECISION_API_VERSION as TRADING_STOCK_DECISION_API_VERSION,
)
from trading_assistant_backtest.strategies.trading.stock import (
    PLUGIN_ID as TRADING_STOCK_PLUGIN_ID,
)

PROJECT_ROOT = package_workspace("trading_assistant_backtest")
AGENT_ROOT = MONOREPO_ROOT
UPSTREAM_CONTROL_PLANE = package_workspace("trading_assistant")
SUBPROCESS_TIMEOUT_SECONDS = int(os.environ.get("TA_MONTHLY_RUNNER_TEST_TIMEOUT_SECONDS", "90"))
PERSISTED_CRYPTO_CONTRACT = (
    PROJECT_ROOT / "contracts" / "crypto_trend_v1" / "strategy_plugin_contract.json"
)
PERSISTED_CRYPTO_DEPLOYMENT = (
    PROJECT_ROOT / "contracts" / "crypto_trend_v1" / "deployment_metadata.json"
)


def _env() -> dict[str, str]:
    env = dict(os.environ)
    paths = [
        PROJECT_ROOT / "src",
        MONOREPO_ROOT / "packages" / "trading_contracts" / "src",
    ]
    existing = env.get("PYTHONPATH")
    path_items = [str(path) for path in paths]
    if existing:
        path_items.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(path_items)
    return env


def _manifest(
    tmp_path: Path,
    *,
    mode: MonthlyRunMode,
    checksum: str | None = None,
    write_slice: bool = True,
    strategy_contract: dict | None = None,
    strategy_plugin_id: str = "strat1-plugin",
    trading_repo_commit_sha: str = "",
    backtest_repo_commit_sha: str = "",
) -> Path:
    artifact_root = tmp_path / "artifacts"
    data_bundle_path = tmp_path / "data_bundle_manifest.json"
    slice_path = tmp_path / "slice.json"
    if write_slice:
        slice_path.write_text(json.dumps({"slice_id": "slice-1"}), encoding="utf-8")
    bundle = DataBundleManifest(
        data_repo_path=str(tmp_path / "market_data"),
        data_repo_commit_sha="fixture-data-sha",
        slice_manifests=[
            DataBundleSlice(
                manifest_path=str(slice_path),
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
    strategy_contract_path = ""
    if strategy_contract is not None:
        path = tmp_path / "strategy_plugin_contract.json"
        path.write_text(json.dumps(strategy_contract), encoding="utf-8")
        strategy_contract_path = str(path)
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
        data_bundle_checksum=checksum if checksum is not None else bundle.bundle_checksum,
        data_manifest_checksum=checksum if checksum is not None else bundle.bundle_checksum,
        telemetry_manifest_path=str(tmp_path / "telemetry.json"),
        artifact_root=str(artifact_root),
        strategy_plugin_id=strategy_plugin_id,
        strategy_plugin_contract_path=strategy_contract_path,
        trading_repo_commit_sha=trading_repo_commit_sha,
        backtest_repo_commit_sha=backtest_repo_commit_sha,
        round_id="round_1",
        prior_round_id="round_0",
        next_round_id="round_2",
    )
    artifact_root.mkdir(parents=True)
    manifest_path = artifact_root / "run_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest_path


def _run_runner(
    manifest_path: Path, module: str = "trading_assistant_backtest.monthly"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, "--manifest", str(manifest_path)],
        cwd=PROJECT_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _stable_json_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_native_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "trading_assistant_backtest.monthly", "--help"],
        cwd=PROJECT_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 0
    assert "--manifest" in completed.stdout


def test_structural_parity_dispatch_includes_formal_week1_bridges() -> None:
    assert STRUCTURAL_PARITY_BUILDERS[K_STOCK_PLUGIN_ID][0] == K_STOCK_DECISION_API_VERSION
    assert STRUCTURAL_PARITY_BUILDERS[TRADING_STOCK_PLUGIN_ID][0] == (
        TRADING_STOCK_DECISION_API_VERSION
    )


def test_native_cli_help_from_agent_root() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "trading_assistant_backtest.monthly", "--help"],
        cwd=AGENT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 0
    assert "--manifest" in completed.stdout


def test_compatibility_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "backtests.shared.monthly_repair", "--help"],
        cwd=PROJECT_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 0
    assert "--manifest" in completed.stdout


def test_validate_only_resolves_retained_legacy_smoke_manifest_paths() -> None:
    manifest_path = (
        PROJECT_ROOT
        / "artifacts"
        / "validation"
        / "monthly_smoke"
        / "k_stock_olr_kalcb"
        / "run_manifest.json"
    )
    if not manifest_path.exists():
        pytest.skip("retained monthly smoke manifest is not available")

    validation = validate_manifest_file(manifest_path)

    assert validation.valid is True, validation.errors
    assert Path(validation.artifact_index_path).is_file()


def test_incumbent_validation_emits_required_artifacts(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, mode=MonthlyRunMode.INCUMBENT_VALIDATION)

    completed = _run_runner(manifest_path)

    assert completed.returncode == 0
    validation = validate_manifest_file(manifest_path)
    assert validation.valid is True
    artifact_root = manifest_path.parent
    assert (artifact_root / "artifact_index.json").exists()
    assert (
        json.loads((artifact_root / "selected_candidates.json").read_text(encoding="utf-8")) == []
    )


def test_phased_auto_emits_no_adoption_optimizer_contract(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, mode=MonthlyRunMode.PHASED_AUTO)

    completed = _run_runner(manifest_path, "backtests.shared.monthly_repair")

    assert completed.returncode == 0
    artifact_root = manifest_path.parent
    assert validate_manifest_file(manifest_path).valid is True
    confirmatory = json.loads(
        (artifact_root / "confirmatory_rerank.json").read_text(encoding="utf-8")
    )
    rounds = json.loads((artifact_root / "rounds_manifest.json").read_text(encoding="utf-8"))
    fold = json.loads((artifact_root / "fold_manifest.json").read_text(encoding="utf-8"))
    candidates = _read_jsonl(artifact_root / "candidate_results.jsonl")
    attempts = _read_jsonl(artifact_root / "candidate_attempts.jsonl")
    observability = json.loads(
        (artifact_root / "runner_observability.json").read_text(encoding="utf-8")
    )
    assert confirmatory["no_adoption_reason"]
    assert confirmatory["primary_source"] == "phased_auto"
    assert set(confirmatory["compared_candidate_ids"]) == {
        candidate["candidate_id"] for candidate in candidates
    }
    assert rounds["no_adoption_reason"] == confirmatory["no_adoption_reason"]
    assert fold["selection_oos_start"] == "2026-04-01"
    assert fold["folds"][0]["purged"] is True
    assert candidates
    assert candidates[0]["decision"] == "reject"
    assert "replay-backed evaluator" in candidates[0]["reason"]
    assert attempts[0]["candidate_id"] == candidates[0]["candidate_id"]
    assert {entry["attempt_id"] for entry in observability} == {
        attempt["attempt_id"] for attempt in attempts
    }


def test_optimizer_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, mode=MonthlyRunMode.PHASED_AUTO, checksum="wrong")

    completed = _run_runner(manifest_path)

    assert completed.returncode == 2
    coverage = json.loads(
        (manifest_path.parent / "coverage_manifest.json").read_text(encoding="utf-8")
    )
    assert coverage["status"] == "blocked"
    assert "checksum" in coverage["errors"][0]


def test_optimizer_missing_slice_manifest_fails_closed(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, mode=MonthlyRunMode.PHASED_AUTO, write_slice=False)

    completed = _run_runner(manifest_path)

    assert completed.returncode == 2
    coverage = json.loads(
        (manifest_path.parent / "coverage_manifest.json").read_text(encoding="utf-8")
    )
    assert any("slice manifest missing" in error for error in coverage["errors"])


def test_strategy_plugin_contract_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest_path = _manifest(
        tmp_path,
        mode=MonthlyRunMode.PHASED_AUTO,
        strategy_contract={
            "plugin_id": "wrong-plugin",
            "backtest_adapter_path": "adapters/strat1.py",
            "config_schema_version": "config_v1",
            "decision_api_version": "decision_api_v1",
            "maturity": "diagnostic",
        },
    )

    completed = _run_runner(manifest_path)

    assert completed.returncode == 2
    stderr = (manifest_path.parent / "stderr.log").read_text(encoding="utf-8")
    assert "plugin_id does not match" in stderr


def test_smoke_repair_emits_repair_ablation_matrix(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, mode=MonthlyRunMode.SMOKE_REPAIR)

    completed = _run_runner(manifest_path)

    assert completed.returncode == 0
    artifact_root = manifest_path.parent
    confirmatory = json.loads(
        (artifact_root / "confirmatory_rerank.json").read_text(encoding="utf-8")
    )
    assert confirmatory["repair_triggered"] is True
    assert confirmatory["primary_source"] == "smoke_repair"
    assert (artifact_root / "repair_ablation_matrix.jsonl").read_text(encoding="utf-8").strip()
    assert validate_manifest_file(manifest_path).valid is True


def test_replay_params_follow_concrete_parameter_patch_not_family() -> None:
    config = BarReplayConfig(
        family="trading_stock_family",
        market="equity",
        source="fixture",
        replay_engine_version="fixture",
        diagnostics_schema_version="fixture",
        result_schema_version="fixture",
        supported_symbols=("AAPL",),
        supported_timeframes=("1m",),
        threshold_bps=10.0,
        position_weight=1.0,
        max_positions=1,
        quantity=1.0,
    )
    loosened = Candidate(
        candidate_id="candidate-loosened",
        family="filter_repair",
        payload={"parameter_patch": {"filter_threshold_bps_delta": -3.0}},
    )
    tightened = Candidate(
        candidate_id="candidate-tightened",
        family="filter_repair",
        payload={"parameter_patch": {"filter_threshold_bps_delta": 4.0}},
    )

    loosened_params = _candidate_params(loosened, config)
    tightened_params = _candidate_params(tightened, config)

    assert loosened_params["threshold_bps"] == 7.0
    assert tightened_params["threshold_bps"] == 14.0
    assert loosened_params != tightened_params


def test_round_n_plus_1_payload_requires_evaluated_patch() -> None:
    candidate = Candidate(
        candidate_id="candidate-patch",
        family="filter_repair",
        payload={"parameter_patch": {"filter_threshold_bps_delta": -2.0}},
    )

    with pytest.raises(ValueError, match="concrete patch"):
        round_n_plus_1_payload(candidate)

    patch_payload = evaluated_patch_payload(
        candidate,
        {"threshold_bps": 8.0, "position_weight": 1.0, "max_positions": 1},
        scope_family="trading_stock_family",
    )
    evaluated = Candidate(
        candidate_id=candidate.candidate_id,
        family=candidate.family,
        payload={**candidate.payload, **patch_payload},
    )

    payload = round_n_plus_1_payload(evaluated)

    assert payload["config_patch"]["filter_threshold_bps_delta"] == -2.0
    assert payload["parameter_patch_fingerprint"] == patch_payload["parameter_patch_fingerprint"]
    assert payload["evaluated_patch_fingerprint"] == patch_payload["evaluated_patch_fingerprint"]


class _RerankFakePlugin:
    def __init__(self) -> None:
        self.selection_incumbent_calls = 0

    def build_phase_specs(self, diagnostics, plan, search_brief):
        return [PhaseSpec(phase_id="phase", candidate_families=["phase_good"])]

    def build_repair_candidates(self, failure_analysis, accepted_mutation_chain):
        return [
            Candidate(
                candidate_id="repair-candidate",
                family="repair",
                payload={"phase_id": "repair", "candidate_family": "repair"},
            )
        ]

    def build_confirmatory_variants(self, candidate, context):
        return [
            Candidate(
                candidate_id="repair-confirmatory",
                family="repair_confirmatory",
                payload={
                    "phase_id": "confirmatory",
                    "candidate_family": "repair_confirmatory",
                    "source_candidate_id": candidate.candidate_id,
                    "variant_type": "local_parameter_perturbation",
                },
            )
        ]

    def run_incumbent(self, window: WindowSpec, baseline) -> ReplayResult:
        if window.name == "selection_oos":
            self.selection_incumbent_calls += 1
        return ReplayResult(
            run_id=f"incumbent-{window.name}",
            window=window,
            trade_count=4,
            net_return=1.0,
            max_drawdown=0.05,
            profit_factor=1.5,
            objective_score=1.0,
            diagnostics={"trade_hash": f"incumbent-{window.name}", "order_hash": "orders"},
        )

    def evaluate_candidate(self, candidate: Candidate, window: WindowSpec) -> CandidateEvaluation:
        scores = {
            "phase-phase_good-1": 1.20,
            "repair-candidate": 1.10,
            "repair-confirmatory": 1.05,
        }
        score = scores[candidate.candidate_id]
        parameter_patch = {
            "family": candidate.family,
            "fixture_candidate_id": candidate.candidate_id,
        }
        evaluated_parameters = {
            "threshold_bps": score,
            "position_weight": 1.0,
            "max_positions": 1,
        }
        parameter_fingerprint = _stable_json_hash(parameter_patch)
        evaluated_fingerprint = _stable_json_hash(
            {
                "parameter_patch": parameter_patch,
                "evaluated_parameters": evaluated_parameters,
            }
        )
        replay = {
            "trade_count": 4,
            "net_return": score,
            "max_drawdown": 0.04,
            "profit_factor": 1.6,
            "objective_score": score,
            "objective_profile_id": "test.profile",
            "immutable_score": {
                "profile_id": "test.profile",
                "profile_version": "immutable_score_profiles_v1",
                "objective_score": score,
                "score_component_cap": 3,
                "renormalized_components": [],
            },
            "trade_hash": f"{candidate.candidate_id}-{window.name}",
            "order_hash": f"orders-{candidate.candidate_id}-{window.name}",
            "coverage": [{"rows": 4}],
            "parameter_patch": parameter_patch,
            "evaluated_parameter_patch": parameter_patch,
            "parameter_patch_fingerprint": parameter_fingerprint,
            "evaluated_patch_fingerprint": evaluated_fingerprint,
            "evaluated_parameters": evaluated_parameters,
        }
        return CandidateEvaluation(
            candidate=Candidate(
                candidate_id=candidate.candidate_id,
                family=candidate.family,
                payload={
                    **candidate.payload,
                    "parameter_patch": parameter_patch,
                    "evaluated_parameter_patch": parameter_patch,
                    "evaluated_parameters": evaluated_parameters,
                    "parameter_patch_fingerprint": parameter_fingerprint,
                    "evaluated_patch_fingerprint": evaluated_fingerprint,
                    "replay_result": replay,
                },
            ),
            objective_score=score,
            passed=True,
            reasons=[f"fixture score {score}"],
        )

    def write_round_n_plus_1(self, candidate: Candidate, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_manifest = output_dir / "candidate_manifest.json"
        config_patch = output_dir / "config_patch.json"
        rollback = output_dir / "rollback_plan.json"
        recommendation = output_dir / "recommendation.json"
        patch_payload = candidate.payload.get("evaluated_parameter_patch")
        if not isinstance(patch_payload, dict):
            patch_payload = {
                "family": candidate.family,
                "fixture_candidate_id": candidate.candidate_id,
            }
        candidate_manifest.write_text(json.dumps(candidate.payload), encoding="utf-8")
        config_patch.write_text(json.dumps(patch_payload), encoding="utf-8")
        rollback.write_text(json.dumps({"candidate_id": candidate.candidate_id}), encoding="utf-8")
        recommendation.write_text(
            json.dumps({"candidate_id": candidate.candidate_id}),
            encoding="utf-8",
        )
        return {
            "next_config_hash": f"next-{candidate.candidate_id}",
            "candidate_manifest_path": str(candidate_manifest),
            "config_patch_path": str(config_patch),
            "rollback_plan_path": str(rollback),
            "path": str(recommendation),
            "parameter_patch_fingerprint": candidate.payload.get(
                "parameter_patch_fingerprint",
                "",
            ),
            "evaluated_patch_fingerprint": candidate.payload.get(
                "evaluated_patch_fingerprint",
                "",
            ),
        }

    def run_diagnostics(self, replay: ReplayResult) -> dict:
        return {"trade_count": replay.trade_count}


def test_repair_triggered_rerank_keeps_phase_winner_and_caches_selection_oos(
    tmp_path: Path,
) -> None:
    manifest_path = _manifest(tmp_path, mode=MonthlyRunMode.SMOKE_REPAIR)
    manifest = MonthlyRunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    plugin = _RerankFakePlugin()
    selection_window = WindowSpec(
        "latest_month",
        manifest.latest_month_start,
        manifest.latest_month_end,
    )
    replay_context = ReplayEvaluationContext(
        plugin=plugin,
        baseline={},
        incumbent=ReplayResult(
            run_id="incumbent",
            window=selection_window,
            trade_count=4,
            objective_score=1.0,
        ),
        baseline_score=1.0,
        replay_backed=True,
    )

    write_optimizer_artifacts(
        ArtifactWriter(manifest, manifest_path.parent),
        manifest,
        manifest_path=manifest_path,
        data_errors=[],
        planner_mode="deterministic",
        replay_context=replay_context,
    )

    confirmatory = json.loads(
        (manifest_path.parent / "confirmatory_rerank.json").read_text(encoding="utf-8")
    )
    rounds = json.loads((manifest_path.parent / "rounds_manifest.json").read_text(encoding="utf-8"))
    recommendation = json.loads(
        (manifest_path.parent / "round_n_plus_1_recommendation.json").read_text(
            encoding="utf-8"
        )
    )
    selected = json.loads(
        (manifest_path.parent / "selected_candidates.json").read_text(encoding="utf-8")
    )

    assert plugin.selection_incumbent_calls == 1
    assert confirmatory["repair_triggered"] is True
    assert confirmatory["primary_candidate_id"] == "repair-candidate"
    assert confirmatory["adopted_candidate_id"] == "phase-phase_good-1"
    assert confirmatory["adopted_source"] == "phased_auto"
    assert set(confirmatory["compared_candidate_ids"]) == {
        "phase-phase_good-1",
        "repair-candidate",
        "repair-confirmatory",
    }
    assert rounds["records"][0]["source"] == "phased_auto"
    assert recommendation["adopted_candidate_id"] == "phase-phase_good-1"
    assert recommendation["evaluated_patch_fingerprint"]
    assert selected[0]["effective_objective_version"] == "immutable_score_profiles_v1"
    assert selected[0]["objective_profile_id"] == "test.profile"
    assert selected[0]["immutable_score"]["profile_id"] == "test.profile"


def test_structural_review_emits_blocked_parity_artifacts(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, mode=MonthlyRunMode.STRUCTURAL_REVIEW)

    completed = _run_runner(manifest_path)

    assert completed.returncode == 0
    artifact_root = manifest_path.parent
    parity = json.loads((artifact_root / "decision_parity_report.json").read_text(encoding="utf-8"))
    gate = json.loads(
        (artifact_root / "structural_selection_gate.json").read_text(encoding="utf-8")
    )
    assert parity["status"] == "insufficient_data"
    assert gate["status"] == "blocked"
    assert gate["selection_allowed"] is False
    assert "decision parity report status is insufficient_data" in gate["blocking_reasons"]
    assert {
        item["artifact_name"]: item["usable_for_structural_selection"]
        for item in gate["patch_checks"]
    } == {
        "live_repo_patch.diff": False,
        "backtest_adapter_patch.diff": False,
        "config_patch.diff": False,
    }
    assert (artifact_root / "live_repo_patch.diff").exists()
    assert validate_manifest_file(manifest_path).valid is True


def test_structural_review_rejects_selection_allowed_without_real_patches(
    tmp_path: Path,
) -> None:
    manifest_path = _manifest(tmp_path, mode=MonthlyRunMode.STRUCTURAL_REVIEW)

    completed = _run_runner(manifest_path)
    assert completed.returncode == 0
    artifact_root = manifest_path.parent
    gate_path = artifact_root / "structural_selection_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["status"] = "selection_ready"
    gate["selection_allowed"] = True
    gate["decision_parity"]["status"] = "pass"
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")

    validation = validate_manifest_file(manifest_path)

    assert validation.valid is False
    assert any(
        "structural selection gate allowed selection with unusable patch artifacts" in error
        for error in validation.errors
    )


def test_structural_review_emits_crypto_trend_decision_parity(tmp_path: Path) -> None:
    crypto_trader_repo = AGENT_ROOT / "bots" / "crypto_trader"
    if not crypto_trader_repo.exists():
        pytest.skip("crypto_trader reference repo is not available")
    live_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=crypto_trader_repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    ).stdout.strip()
    fixture_path = tmp_path / "crypto_trend_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "fixture_id": "trend_entry_market",
                "symbol": "BTC",
                "timeframe": "1h",
                "timestamp": "2026-03-15T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    manifest_path = _manifest(
        tmp_path,
        mode=MonthlyRunMode.STRUCTURAL_REVIEW,
        strategy_plugin_id=CRYPTO_TREND_PLUGIN_ID,
        trading_repo_commit_sha=live_sha,
        backtest_repo_commit_sha="backtest-sha",
        strategy_contract={
            "plugin_id": CRYPTO_TREND_PLUGIN_ID,
            "live_repo_path": str(crypto_trader_repo),
            "live_repo_commit_sha": live_sha,
            "backtest_adapter_path": "src/trading_assistant_backtest/strategies/crypto/trend.py",
            "backtest_adapter_commit_sha": "backtest-sha",
            "config_schema_version": "crypto_trend_config_v1",
            "decision_api_version": CRYPTO_TREND_DECISION_API_VERSION,
            "parity_fixture_set": [str(fixture_path)],
            "maturity": "diagnostic",
        },
    )

    completed = _run_runner(manifest_path)

    assert completed.returncode == 0
    parity = json.loads(
        (manifest_path.parent / "decision_parity_report.json").read_text(encoding="utf-8")
    )
    gate = json.loads(
        (manifest_path.parent / "structural_selection_gate.json").read_text(encoding="utf-8")
    )
    assert parity["status"] == "pass"
    assert gate["selection_allowed"] is False
    assert "live_repo_patch.diff is not a real patch artifact" in gate["blocking_reasons"]
    assert parity["strategy_plugin_id"] == CRYPTO_TREND_PLUGIN_ID
    assert parity["live_repo_commit_sha"] == live_sha
    assert {check["dimension"] for check in parity["checks"]} == {
        "signals",
        "filters",
        "entries",
        "exits",
        "stops",
        "sizing",
        "risk_caps",
        "order_intent",
    }


def test_optimizer_run_manifest_payload_emits_all_crypto_bridge_hashes(
    tmp_path: Path,
) -> None:
    crypto_bridge_ids = {"crypto_trend_v1", "crypto_momentum_v1", "crypto_breakout_v1"}
    expected_contract_paths = {
        bridge_id: PROJECT_ROOT / "contracts" / bridge_id / "strategy_plugin_contract.json"
        for bridge_id in crypto_bridge_ids
    }
    expected_metadata_paths = {
        bridge_id: PROJECT_ROOT / "contracts" / bridge_id / "deployment_metadata.json"
        for bridge_id in crypto_bridge_ids
    }
    if not all(path.exists() for path in [
        *expected_contract_paths.values(),
        *expected_metadata_paths.values(),
    ]):
        pytest.skip("persisted crypto bridge artifacts are not available")
    manifest = MonthlyRunManifest(
        run_id="monthly-crypto_portfolio-btc_1m-2026-05-shadow",
        run_month="2026-05",
        mode=MonthlyRunMode.PHASED_AUTO,
        bot_id="crypto_portfolio",
        strategy_id="btc_1m",
        latest_month_start=date(2026, 5, 1),
        latest_month_end=date(2026, 5, 30),
        in_sample_start=date(2026, 2, 1),
        in_sample_end=date(2026, 4, 30),
        selection_oos_start=date(2026, 5, 1),
        selection_oos_end=date(2026, 5, 30),
        market_data_manifest_path=str(tmp_path / "data_bundle_manifest.json"),
        data_bundle_manifest_path=str(tmp_path / "data_bundle_manifest.json"),
        data_bundle_checksum="bundle-sha",
        data_manifest_checksum="bundle-sha",
        telemetry_manifest_path=str(tmp_path / "telemetry.json"),
        artifact_root=str(tmp_path / "artifacts"),
        strategy_plugin_id=CRYPTO_TREND_PLUGIN_ID,
        strategy_plugin_contract_path=str(PERSISTED_CRYPTO_CONTRACT),
        deployment_metadata_path=str(PERSISTED_CRYPTO_DEPLOYMENT),
        round_id="round_shadow_1",
        prior_round_id="round_shadow_0",
        next_round_id="round_shadow_2",
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    payload = _optimizer_run_manifest_payload(
        manifest,
        artifact_root=tmp_path / "artifacts",
        manifest_path=manifest_path,
        planner_mode="fixture",
    )

    assert payload["scope_id"] == "crypto_trader_portfolio"
    assert set(payload["bridge_contract_paths"]) == crypto_bridge_ids
    assert set(payload["bridge_deployment_metadata_paths"]) == crypto_bridge_ids
    assert payload["bridge_contract_hashes"] == {
        bridge_id: sha256_file(path)
        for bridge_id, path in expected_contract_paths.items()
    }
    assert payload["bridge_deployment_metadata_hashes"] == {
        bridge_id: sha256_file(path)
        for bridge_id, path in expected_metadata_paths.items()
    }


def test_shadow_monthly_cycle_uses_committed_bundle_and_mature_crypto_contract(
    tmp_path: Path,
) -> None:
    bundle_path = (
        package_workspace("trading_assistant_data")
        / "data"
        / "bundles"
        / "monthly"
        / "2026-05"
        / "crypto_portfolio"
        / "btc_1m"
        / "data_bundle_manifest.json"
    )
    crypto_trader_repo = AGENT_ROOT / "bots" / "crypto_trader"
    if not bundle_path.exists():
        pytest.skip("committed BTC 1m data bundle is not available")
    if not crypto_trader_repo.exists():
        pytest.skip("crypto_trader reference repo is not available")
    if not PERSISTED_CRYPTO_CONTRACT.exists() or not PERSISTED_CRYPTO_DEPLOYMENT.exists():
        pytest.skip("persisted crypto trend shadow artifacts are not available")
    contract, contract_errors = load_strategy_plugin_contract(PERSISTED_CRYPTO_CONTRACT)
    assert contract_errors == []
    assert contract is not None
    deployment = load_deployment_metadata(PERSISTED_CRYPTO_DEPLOYMENT)
    assert contract.maturity.value == "shadow_validated"
    assert contract.eligible_for_optimizer is True
    assert contract.eligible_for_approval is False
    assert deployment.deployed_commit_sha == contract.live_repo_commit_sha
    assert len(contract.parity_fixture_set) >= 4
    bundle = DataBundleManifest.model_validate(
        json.loads(bundle_path.read_text(encoding="utf-8"))
    )
    artifact_root = tmp_path / "shadow_artifacts"
    manifest = MonthlyRunManifest(
        run_id="monthly-crypto_portfolio-btc_1m-2026-05-shadow",
        run_month="2026-05",
        mode=MonthlyRunMode.PHASED_AUTO,
        bot_id=deployment.bot_id,
        strategy_id=deployment.strategy_id,
        strategy_version=deployment.strategy_version,
        config_version=deployment.config_version,
        config_hash=deployment.config_hash,
        deployment_id=deployment.deployment_id,
        latest_month_start=date(2026, 5, 1),
        latest_month_end=date(2026, 5, 30),
        in_sample_start=date(2026, 2, 1),
        in_sample_end=date(2026, 4, 30),
        selection_oos_start=date(2026, 5, 1),
        selection_oos_end=date(2026, 5, 30),
        market_data_manifest_path=str(bundle_path),
        data_bundle_manifest_path=str(bundle_path),
        data_bundle_checksum=bundle.bundle_checksum,
        data_manifest_checksum=bundle.bundle_checksum,
        telemetry_manifest_path=str(tmp_path / "telemetry.json"),
        artifact_root=str(artifact_root),
        strategy_plugin_id=contract.plugin_id,
        strategy_plugin_contract_path=str(PERSISTED_CRYPTO_CONTRACT),
        trading_repo_commit_sha=contract.live_repo_commit_sha,
        backtest_repo_commit_sha=contract.backtest_adapter_commit_sha,
        deployment_metadata_path=str(PERSISTED_CRYPTO_DEPLOYMENT),
        monthly_search_guidance={
            "plan_requirements": {"candidate_families": ["filter_repair", "exit_repair"]}
        },
        round_id="round_shadow_1",
        prior_round_id="round_shadow_0",
        next_round_id="round_shadow_2",
    )
    artifact_root.mkdir(parents=True)
    manifest_path = artifact_root / "run_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    completed = _run_runner(manifest_path)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert validate_manifest_file(manifest_path).valid is True
    if (UPSTREAM_CONTROL_PLANE / "contracts" / "validate_monthly_runner.py").exists():
        control_validation = subprocess.run(
            [
                sys.executable,
                "-m",
                "contracts.validate_monthly_runner",
                "--manifest",
                str(manifest_path),
            ],
            cwd=UPSTREAM_CONTROL_PLANE,
            capture_output=True,
            text=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
        assert control_validation.returncode == 0, (
            control_validation.stdout + control_validation.stderr
        )
    coverage = json.loads((artifact_root / "coverage_manifest.json").read_text(encoding="utf-8"))
    candidates = _read_jsonl(artifact_root / "candidate_results.jsonl")
    attempts = _read_jsonl(artifact_root / "candidate_attempts.jsonl")
    observability = json.loads(
        (artifact_root / "runner_observability.json").read_text(encoding="utf-8")
    )
    confirmatory = json.loads(
        (artifact_root / "confirmatory_rerank.json").read_text(encoding="utf-8")
    )
    optimizer_manifest = json.loads(
        (artifact_root / "optimizer_run_manifest.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (artifact_root / "end_of_round_diagnostics.json").read_text(encoding="utf-8")
    )
    crypto_bridge_ids = {"crypto_trend_v1", "crypto_momentum_v1", "crypto_breakout_v1"}
    expected_contract_hashes = {
        bridge_id: sha256_file(
            PROJECT_ROOT
            / "contracts"
            / bridge_id
            / "strategy_plugin_contract.json"
        )
        for bridge_id in crypto_bridge_ids
    }
    expected_metadata_hashes = {
        bridge_id: sha256_file(
            PROJECT_ROOT
            / "contracts"
            / bridge_id
            / "deployment_metadata.json"
        )
        for bridge_id in crypto_bridge_ids
    }

    assert coverage["status"] == "pass"
    assert coverage["data_bundle_checksum"] == bundle.bundle_checksum
    assert optimizer_manifest["scope_id"] == "crypto_trader_portfolio"
    assert set(optimizer_manifest["bridge_contract_hashes"]) == crypto_bridge_ids
    assert set(optimizer_manifest["bridge_deployment_metadata_hashes"]) == crypto_bridge_ids
    assert optimizer_manifest["bridge_contract_hashes"] == expected_contract_hashes
    assert optimizer_manifest["bridge_deployment_metadata_hashes"] == expected_metadata_hashes
    assert {candidate["candidate_id"] for candidate in candidates} == {
        "signal_quality-filter_repair-1",
        "signal_quality-exit_repair-2",
    }
    assert {attempt["state"] for attempt in attempts} == {"failed"}
    assert {entry["attempt_id"] for entry in observability} == {
        attempt["attempt_id"] for attempt in attempts
    }
    assert confirmatory["adopted_candidate_id"] == ""
    assert set(confirmatory["compared_candidate_ids"]) == {
        candidate["candidate_id"] for candidate in candidates
    }
    assert "replay-backed evaluation" in confirmatory["no_adoption_reason"]
    assert "candidate adoption remains disabled" in confirmatory["no_adoption_reason"]
    assert candidates[0]["deterministic_gate_inputs"]["diagnostic_only"] is False
    assert candidates[0]["baseline_score"] != 0.0
    assert (artifact_root / "frozen_baseline.json").exists()
    assert (artifact_root / "round_reproduction_report.json").exists()
    assert diagnostics["failure_analysis"]["primary_failure"] == "shadow_replay_candidate_gate"


def test_upstream_contract_validator_accepts_phased_auto_no_adoption(tmp_path: Path) -> None:
    if not (UPSTREAM_CONTROL_PLANE / "contracts" / "validate_monthly_runner.py").exists():
        pytest.skip("sibling trading_assistant contract validator is not available")
    manifest_path = _manifest(tmp_path, mode=MonthlyRunMode.PHASED_AUTO)
    completed = _run_runner(manifest_path, "backtests.shared.monthly_repair")
    assert completed.returncode == 0

    validation = subprocess.run(
        [
            sys.executable,
            "-m",
            "contracts.validate_monthly_runner",
            "--manifest",
            str(manifest_path),
        ],
        cwd=UPSTREAM_CONTROL_PLANE,
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert json.loads(validation.stdout)["valid"] is True

