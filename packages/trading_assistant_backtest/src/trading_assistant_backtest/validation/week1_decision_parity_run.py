"""Formal week-1 decision parity validation for stock/KRX bridges."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from trading_assistant_backtest.contract_models import (
    DECISION_PARITY_DIMENSIONS,
    DecisionParityCheck,
    DecisionParityReport,
    DecisionParityStatus,
    MonthlyRunManifest,
    MonthlyRunMode,
    StrategyPluginMaturity,
)
from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.strategies.contracts import load_strategy_plugin_contract
from trading_assistant_backtest.strategies.deployment import (
    deployment_metadata_errors,
    load_deployment_metadata,
)
from trading_assistant_backtest.strategies.krx.olr_kalcb import (
    DECISION_API_VERSION as K_STOCK_DECISION_API_VERSION,
)
from trading_assistant_backtest.strategies.krx.olr_kalcb import (
    PLUGIN_ID as K_STOCK_PLUGIN_ID,
)
from trading_assistant_backtest.strategies.krx.olr_kalcb import (
    build_k_stock_olr_kalcb_decision_parity_report,
)
from trading_assistant_backtest.strategies.live_clone import (
    LiveRepoCheckoutSpec,
    LiveRepoCloneManager,
    validate_clean_checkout,
)
from trading_assistant_backtest.strategies.trading.momentum import (
    DECISION_API_VERSION as TRADING_MOMENTUM_DECISION_API_VERSION,
)
from trading_assistant_backtest.strategies.trading.momentum import (
    PLUGIN_ID as TRADING_MOMENTUM_PLUGIN_ID,
)
from trading_assistant_backtest.strategies.trading.momentum import (
    build_trading_momentum_decision_parity_report,
)
from trading_assistant_backtest.strategies.trading.stock import (
    DECISION_API_VERSION as TRADING_STOCK_DECISION_API_VERSION,
)
from trading_assistant_backtest.strategies.trading.stock import (
    PLUGIN_ID as TRADING_STOCK_PLUGIN_ID,
)
from trading_assistant_backtest.strategies.trading.stock import (
    build_trading_stock_decision_parity_report,
)
from trading_assistant_backtest.strategies.trading.swing import (
    DECISION_API_VERSION as TRADING_SWING_DECISION_API_VERSION,
)
from trading_assistant_backtest.strategies.trading.swing import (
    PLUGIN_ID as TRADING_SWING_PLUGIN_ID,
)
from trading_assistant_backtest.strategies.trading.swing import (
    build_trading_swing_decision_parity_report,
)
from trading_assistant_backtest.paths import monorepo_root, resolve_workspace_path

DEFAULT_WEEK1_CONTRACTS = (
    Path("contracts/strategy_plugins/k_stock_olr_kalcb/strategy_plugin_contract.json"),
    Path("contracts/strategy_plugins/trading_stock_family/strategy_plugin_contract.json"),
)


@dataclass(frozen=True)
class Week1Adapter:
    plugin_id: str
    decision_api_version: str
    min_fixture_count: int
    builder: Callable[..., DecisionParityReport]


ADAPTERS = {
    K_STOCK_PLUGIN_ID: Week1Adapter(
        plugin_id=K_STOCK_PLUGIN_ID,
        decision_api_version=K_STOCK_DECISION_API_VERSION,
        min_fixture_count=2,
        builder=build_k_stock_olr_kalcb_decision_parity_report,
    ),
    TRADING_STOCK_PLUGIN_ID: Week1Adapter(
        plugin_id=TRADING_STOCK_PLUGIN_ID,
        decision_api_version=TRADING_STOCK_DECISION_API_VERSION,
        min_fixture_count=2,
        builder=build_trading_stock_decision_parity_report,
    ),
    TRADING_MOMENTUM_PLUGIN_ID: Week1Adapter(
        plugin_id=TRADING_MOMENTUM_PLUGIN_ID,
        decision_api_version=TRADING_MOMENTUM_DECISION_API_VERSION,
        min_fixture_count=2,
        builder=build_trading_momentum_decision_parity_report,
    ),
    TRADING_SWING_PLUGIN_ID: Week1Adapter(
        plugin_id=TRADING_SWING_PLUGIN_ID,
        decision_api_version=TRADING_SWING_DECISION_API_VERSION,
        min_fixture_count=2,
        builder=build_trading_swing_decision_parity_report,
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run formal week-1 live/backtest decision parity validations."
    )
    parser.add_argument("--contract", type=Path, action="append", default=None)
    parser.add_argument("--artifact-root", type=Path, default=_default_artifact_root())
    parser.add_argument("--run-id-prefix", default="decision-parity-week1-shadow")
    args = parser.parse_args(argv)

    contracts = args.contract or [_repo_root() / path for path in DEFAULT_WEEK1_CONTRACTS]
    result = run_week1_decision_parity_validations(
        contract_paths=contracts,
        artifact_root=args.artifact_root,
        run_id_prefix=args.run_id_prefix,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["ok"] else 1


def run_week1_decision_parity_validations(
    *,
    contract_paths: Iterable[Path],
    artifact_root: Path,
    run_id_prefix: str = "decision-parity-week1-shadow",
) -> dict[str, Any]:
    results = []
    for contract_path in contract_paths:
        resolved = Path(contract_path)
        if not resolved.is_absolute():
            resolved = _repo_root() / resolved
        plugin_slug = resolved.parent.name
        results.append(
            run_week1_decision_parity_validation(
                contract_path=resolved,
                artifact_root=Path(artifact_root) / plugin_slug / "decision_parity",
                run_id=f"{run_id_prefix}-{plugin_slug}",
            )
        )
    return {
        "ok": all(item["ok"] for item in results),
        "results": results,
    }


def run_week1_decision_parity_validation(
    *,
    contract_path: Path,
    artifact_root: Path,
    deployment_metadata_path: Path | None = None,
    run_id: str = "decision-parity-week1-shadow",
) -> dict[str, Any]:
    """Validate one persisted week-1 bridge and emit durable parity artifacts."""

    contract_path = Path(contract_path).resolve()
    deployment_metadata_path = (
        Path(deployment_metadata_path).resolve()
        if deployment_metadata_path is not None
        else contract_path.with_name("deployment_metadata.json").resolve()
    )
    artifact_root = Path(artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    contract, contract_errors = load_strategy_plugin_contract(contract_path)
    checks.append(_check("contract_loads_cleanly", not contract_errors, contract_errors))
    if contract is None:
        return _write_summary(
            artifact_root,
            {"ok": False, "run_id": run_id, "checks": checks, "decision_parity_report_path": ""},
        )

    adapter = ADAPTERS.get(contract.plugin_id)
    adapter_errors = [] if adapter else [f"unsupported week-1 plugin_id: {contract.plugin_id}"]
    if adapter and contract.decision_api_version != adapter.decision_api_version:
        adapter_errors.append("contract decision_api_version does not match wired adapter")
    checks.append(_check("wired_week1_adapter_available", not adapter_errors, adapter_errors))
    if adapter is None or adapter_errors:
        return _write_summary(
            artifact_root,
            {"ok": False, "run_id": run_id, "checks": checks, "decision_parity_report_path": ""},
        )

    try:
        deployment = load_deployment_metadata(deployment_metadata_path)
        checks.append(_check("deployment_metadata_loads_cleanly", True, []))
    except Exception as exc:
        checks.append(_check("deployment_metadata_loads_cleanly", False, [str(exc)]))
        return _write_summary(
            artifact_root,
            {"ok": False, "run_id": run_id, "checks": checks, "decision_parity_report_path": ""},
        )

    manifest = MonthlyRunManifest(
        run_id=run_id,
        run_month="2026-05",
        mode=MonthlyRunMode.STRUCTURAL_REVIEW,
        bot_id=deployment.bot_id,
        strategy_id=deployment.strategy_id,
        strategy_version=deployment.strategy_version,
        config_version=deployment.config_version,
        config_hash=deployment.config_hash,
        deployment_id=deployment.deployment_id,
        latest_month_start=date(2026, 5, 1),
        latest_month_end=date(2026, 5, 30),
        market_data_manifest_path="decision_parity_fixture_set",
        telemetry_manifest_path="decision_parity_fixture_set",
        artifact_root=str(artifact_root),
        strategy_plugin_id=contract.plugin_id,
        strategy_plugin_contract_path=str(contract_path),
        strategy_plugin_contract_version=contract.contract_version,
        trading_repo_commit_sha=contract.live_repo_commit_sha,
        backtest_repo_commit_sha=contract.backtest_adapter_commit_sha,
        deployment_metadata_path=str(deployment_metadata_path),
    )

    metadata_errors = deployment_metadata_errors(manifest, contract)
    checks.append(
        _check("deployment_metadata_matches_contract", not metadata_errors, metadata_errors)
    )

    live_repo_path = Path(contract.live_repo_path)
    try:
        live_repo_path = _prepare_clean_live_checkout(
            repo_url=deployment.repo_url,
            commit_sha=contract.live_repo_commit_sha,
            artifact_root=artifact_root,
        )
        checkout_errors = validate_clean_checkout(live_repo_path, contract.live_repo_commit_sha)
    except Exception as exc:
        checkout_errors = [f"could not prepare clean pinned live repo checkout: {exc}"]
    checks.append(
        _check("live_repo_checkout_clean_at_pinned_commit", not checkout_errors, checkout_errors)
    )

    fixture_paths = _fixture_paths_for_live_checkout(
        contract.parity_fixture_set,
        original_repo_path=Path(contract.live_repo_path),
        validation_repo_path=live_repo_path,
    )
    fixture_errors = [
        f"fixture missing from clean live repo checkout: {path}"
        for path in fixture_paths
        if not Path(path).exists()
    ]
    checks.append(
        _check(
            "parity_fixtures_available_in_clean_checkout",
            not fixture_errors,
            fixture_errors,
        )
    )

    adapter_path = _resolve_adapter_path(contract.backtest_adapter_path)
    adapter_hash = sha256_file(adapter_path)
    hash_errors = (
        []
        if adapter_hash == contract.backtest_adapter_commit_sha
        else ["contract backtest_adapter_commit_sha does not match adapter file hash"]
    )
    checks.append(_check("contract_matches_backtest_adapter_hash", not hash_errors, hash_errors))

    maturity_errors: list[str] = []
    if contract.maturity != StrategyPluginMaturity.SHADOW_VALIDATED:
        maturity_errors.append(f"contract maturity is {contract.maturity.value}")
    if contract.eligible_for_approval:
        maturity_errors.append("contract is unexpectedly eligible for approval")
    checks.append(
        _check(
            "plugin_remains_shadow_validated_not_approval_ready",
            not maturity_errors,
            maturity_errors,
        )
    )

    report_path = artifact_root / "decision_parity_report.json"
    try:
        report = adapter.builder(
            manifest,
            candidate_id="strategy-plugin-contract",
            fixture_paths=fixture_paths,
            live_repo_path=live_repo_path,
            live_repo_commit_sha=contract.live_repo_commit_sha,
            backtest_adapter_commit_sha=contract.backtest_adapter_commit_sha,
        )
    except Exception as exc:
        fixture_count = len(set(contract.parity_fixture_set))
        error = f"decision parity builder failed: {type(exc).__name__}: {exc}"
        report = _failed_decision_parity_report(
            manifest,
            evidence_paths=contract.parity_fixture_set,
            live_repo_commit_sha=contract.live_repo_commit_sha,
            backtest_adapter_commit_sha=contract.backtest_adapter_commit_sha,
            error=error,
        )
        checks.append(_check("live_and_adapter_traces_match_required_dimensions", False, [error]))
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return _write_summary(
            artifact_root,
            {
                "ok": False,
                "run_id": run_id,
                "strategy_plugin_id": contract.plugin_id,
                "contract_maturity": contract.maturity.value,
                "eligible_for_optimizer": contract.eligible_for_optimizer,
                "eligible_for_approval": contract.eligible_for_approval,
                "live_repo_commit_sha": contract.live_repo_commit_sha,
                "validated_live_repo_path": str(live_repo_path),
                "backtest_adapter_commit_sha": contract.backtest_adapter_commit_sha,
                "fixture_count": fixture_count,
                "decision_parity_status": report.status.value,
                "decision_parity_report_path": str(report_path),
                "checks": checks,
            },
        )
    parity_errors: list[str] = []
    if report.status != DecisionParityStatus.PASS:
        parity_errors.append(f"decision parity status is {report.status.value}")
    dimensions = {check.dimension for check in report.checks}
    missing_dimensions = DECISION_PARITY_DIMENSIONS - dimensions
    if missing_dimensions:
        parity_errors.append(
            "decision parity missing dimensions: " + ", ".join(sorted(missing_dimensions))
        )
    fixture_count = len({path for check in report.checks for path in check.evidence_paths})
    if fixture_count < adapter.min_fixture_count:
        parity_errors.append("decision parity fixture coverage is narrower than expected")
    checks.append(
        _check(
            "live_and_adapter_traces_match_required_dimensions",
            not parity_errors,
            parity_errors,
        )
    )

    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    ok = all(item["passed"] for item in checks)
    return _write_summary(
        artifact_root,
        {
            "ok": ok,
            "run_id": run_id,
            "strategy_plugin_id": contract.plugin_id,
            "contract_maturity": contract.maturity.value,
            "eligible_for_optimizer": contract.eligible_for_optimizer,
            "eligible_for_approval": contract.eligible_for_approval,
            "live_repo_commit_sha": contract.live_repo_commit_sha,
            "validated_live_repo_path": str(live_repo_path),
            "backtest_adapter_commit_sha": contract.backtest_adapter_commit_sha,
            "fixture_count": fixture_count,
            "decision_parity_status": report.status.value,
            "decision_parity_report_path": str(report_path),
            "checks": checks,
        },
    )


def _check(name: str, passed: bool, errors: list[str]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "errors": errors}


def _failed_decision_parity_report(
    manifest: MonthlyRunManifest,
    *,
    evidence_paths: list[str],
    live_repo_commit_sha: str,
    backtest_adapter_commit_sha: str,
    error: str,
) -> DecisionParityReport:
    return DecisionParityReport(
        run_id=manifest.run_id,
        candidate_id="strategy-plugin-contract",
        strategy_plugin_id=manifest.strategy_plugin_id,
        live_repo_commit_sha=live_repo_commit_sha,
        backtest_adapter_commit_sha=backtest_adapter_commit_sha,
        status=DecisionParityStatus.FAIL,
        evidence_paths=evidence_paths,
        checks=[
            DecisionParityCheck(
                dimension=dimension,
                status=DecisionParityStatus.FAIL,
                match_rate=0.0,
                mismatch_count=1,
                notes=error,
                evidence_paths=evidence_paths,
            )
            for dimension in sorted(DECISION_PARITY_DIMENSIONS)
        ],
    )


def _write_summary(artifact_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    summary_path = artifact_root / "decision_parity_validation_summary.json"
    payload["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _resolve_adapter_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else _repo_root() / path


def _prepare_clean_live_checkout(
    *,
    repo_url: str,
    commit_sha: str,
    artifact_root: Path,
) -> Path:
    source = _clone_source(repo_url)
    manager = LiveRepoCloneManager(Path(tempfile.gettempdir()) / "ta_live_checkouts")
    return manager.prepare(
        LiveRepoCheckoutSpec(
            repo_url=source,
            commit_sha=commit_sha,
            checkout_root=manager.checkout_root,
        )
    )


def _clone_source(repo_url: str) -> str:
    if not repo_url.startswith("local://"):
        return repo_url
    raw = repo_url.removeprefix("local://")
    if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    path = Path(raw)
    if not path.is_absolute():
        path = resolve_workspace_path(monorepo_root(), path)
    return str(path.resolve())


def _fixture_paths_for_live_checkout(
    fixture_paths: Iterable[str],
    *,
    original_repo_path: Path,
    validation_repo_path: Path,
) -> list[str]:
    original_repo = original_repo_path.resolve()
    validation_repo = validation_repo_path.resolve()
    remapped: list[str] = []
    for raw_path in fixture_paths:
        path = Path(raw_path).resolve()
        try:
            rel_path = path.relative_to(original_repo)
        except ValueError:
            remapped.append(str(path))
            continue
        remapped.append(str(validation_repo / rel_path))
    return remapped


def _default_artifact_root() -> Path:
    return _repo_root() / "artifacts" / "validation" / "week1"


def _repo_root() -> Path:
    return monorepo_root()


if __name__ == "__main__":
    raise SystemExit(main())
