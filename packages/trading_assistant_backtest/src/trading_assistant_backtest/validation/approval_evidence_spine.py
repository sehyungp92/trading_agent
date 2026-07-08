"""Production approval evidence bundle for guarded bridge promotion."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_contracts import relay_evidence

from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.paths import (
    monorepo_root,
    normalize_workspace_path,
    resolve_workspace_path,
)
from trading_assistant_backtest.validation.approval_grade_audit import (
    CONTRACT_PATHS,
    run_approval_grade_audit,
)
from trading_assistant_backtest.validation.bridge_readiness import run_bridge_readiness_audit
from trading_assistant_backtest.validation.deployment_metadata_contract import (
    live_deployment_metadata_errors,
    telemetry_schema_contract_errors,
)
from trading_assistant_backtest.validation.optimizer_evidence import (
    OPTIMIZER_RUN_MANIFEST,
    build_optimizer_manifest_index,
    latest_optimizer_artifact_root,
    optimizer_evidence_checks,
)
from trading_assistant_backtest.validation.telemetry_conformance import (
    run_telemetry_conformance_check,
)
from trading_assistant_backtest.validation.validation_matrix import (
    SCOPES,
    VALIDATION_TESTS,
    ValidationScope,
    run_validation_matrix_audit,
)

SCHEMA_VERSION = "approval_evidence_bundle_v1"
PILOT_SCOPE_ID = "trading_stock_family"
SCHEDULED_SHADOW_REPORT = "scheduled_shadow_cycle_report.json"
PRODUCTION_FIXTURE_MANIFEST = "production_fixture_set_manifest.json"
LIVE_CONFIG_VERIFICATION_REPORT = "live_config_promotion_verification.json"
OPERATIONAL_VERIFICATION_REPORT = "operational_deployment_verification.json"
PTG7_COMMAND_REPORT = "ptg7_command_result.json"
RELAY_EVIDENCE_MAX_AGE_SECONDS = relay_evidence.RELAY_EVIDENCE_MAX_AGE_SECONDS

SCOPE_BOTS = {
    "trading_stock_family": "ibkr",
    "trading_momentum_family": "ibkr",
    "trading_swing_family": "ibkr",
    "k_stock_olr_kalcb": "k_stock",
    "crypto_trader_portfolio": "crypto",
}
FIXTURE_CASE_ALIASES = {
    "accepted_entry": {"accepted_entry", "entry_accept", "accepted_trade_entry"},
    "blocked_no_trade": {"blocked_no_trade", "blocked_trade", "no_trade_blocked"},
    "risk_portfolio_denial": {
        "portfolio_denial",
        "risk_denial",
        "risk_portfolio_denial",
    },
    "exit_close": {"close", "exit_close", "position_exit"},
    "order_fill_or_explicit_non_fill": {
        "explicit_non_fill",
        "fill",
        "non_fill",
        "order_fill",
        "order_fill_or_explicit_non_fill",
    },
    "live_shadow_telemetry_source": {
        "live_shadow_telemetry_source",
        "live_telemetry",
        "runtime_telemetry",
        "shadow_telemetry",
    },
}


@dataclass(frozen=True)
class SourceContext:
    bridge_readiness: dict[str, Any]
    validation_matrix: dict[str, Any]
    approval_grade_audit: dict[str, Any]
    ptg7_gate_report_path: Path
    ptg7_command_report_path: Path
    live_config_verification_path: Path
    operational_verification_path: Path
    deployment_metadata_install_reports: list[Path]
    operational_evidence_path: Path
    scheduled_shadow_report_path: Path | None
    telemetry_conformance_report_path: Path
    optimizer_artifact_root: Path | None
    learning_sufficiency_manifest_path: Path | None
    production_fixture_manifest_path: Path | None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fail-closed production approval evidence bundle for a strategy scope."
        )
    )
    parser.add_argument("--agent-root", type=Path, default=monorepo_root())
    parser.add_argument("--scope", default=PILOT_SCOPE_ID, choices=sorted(_scope_by_id()))
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Use cached source reports when present instead of refreshing them.",
    )
    parser.add_argument(
        "--require-eligible",
        action="store_true",
        help="Exit nonzero when the generated bundle is not eligible for promotion.",
    )
    args = parser.parse_args(argv)

    bundle = run_approval_evidence_spine(
        agent_root=args.agent_root,
        scope_id=args.scope,
        artifact_root=args.artifact_root,
        refresh=not args.use_cached,
    )
    print(json.dumps(bundle, indent=2, sort_keys=True, default=str))
    return 1 if args.require_eligible and not bundle["eligible_for_promotion"] else 0


def run_approval_evidence_spine(
    *,
    agent_root: Path,
    scope_id: str = PILOT_SCOPE_ID,
    artifact_root: Path | None = None,
    refresh: bool = True,
) -> dict[str, Any]:
    """Compose existing audits and production inputs into one promotion record."""

    agent_root = Path(agent_root).resolve()
    scope = _scope_for(scope_id)
    artifact_root = (
        Path(artifact_root).resolve()
        if artifact_root is not None
        else resolve_workspace_path(
            agent_root,
            Path("artifacts") / "validation" / "approval_evidence" / scope_id,
        )
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    source_root = artifact_root / "source_reports"
    source_root.mkdir(parents=True, exist_ok=True)

    context = _collect_source_context(
        agent_root=agent_root,
        scope=scope,
        artifact_root=artifact_root,
        source_root=source_root,
        refresh=refresh,
    )

    checks = [
        _source_reports_loadable_check(context),
        _validation_matrix_green_check(scope, context.validation_matrix, agent_root),
        _bridge_readiness_check(scope, context.bridge_readiness, agent_root),
        _contract_shadow_maturity_check(scope, context.bridge_readiness, agent_root),
        _deployment_metadata_installed_check(
            scope,
            context.deployment_metadata_install_reports,
            agent_root,
        ),
        _live_config_promotion_check(context.live_config_verification_path),
        _operational_deployment_check(
            context.operational_evidence_path,
            context.operational_verification_path,
        ),
        _scheduled_shadow_cycle_check(context.scheduled_shadow_report_path, agent_root),
        _telemetry_conformance_check(context.telemetry_conformance_report_path),
        _production_fixture_breadth_check(context.production_fixture_manifest_path, agent_root),
        _optimizer_approval_manifest_check(scope, context, agent_root),
        _ptg7_fixture_context_check(scope, context.ptg7_gate_report_path),
        _learning_sufficiency_check(context.learning_sufficiency_manifest_path),
    ]
    blockers = _blockers_from_checks(checks)
    source_reports = _source_reports_payload(context)
    evidence_paths = _evidence_paths_for_bundle(source_reports, checks)
    evidence_hashes = _evidence_hashes(evidence_paths, agent_root)
    bundle_path = artifact_root / "approval_evidence_bundle.json"
    eligible_for_promotion = all(check["passed"] for check in checks)

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "scope_id": scope.scope_id,
        "bridge_ids": list(scope.decision_bridge_ids or (scope.decision_bridge_id,)),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "eligible_for_promotion": eligible_for_promotion,
        "promotion_decision": "eligible" if eligible_for_promotion else "blocked",
        "blockers": blockers,
        "source_reports": source_reports,
        "required_checks": checks,
        "evidence_hashes": evidence_hashes,
        "next_required_actions": _next_required_actions(
            context.approval_grade_audit,
            blockers,
        ),
        "artifact_path": str(bundle_path),
    }
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return bundle


def _collect_source_context(
    *,
    agent_root: Path,
    scope: ValidationScope,
    artifact_root: Path,
    source_root: Path,
    refresh: bool,
) -> SourceContext:
    bridge_root = source_root / "bridge_readiness"
    matrix_root = source_root / "validation_matrix"
    approval_root = source_root / "approval_grade"
    bridge_report_path = bridge_root / "bridge_readiness_report.json"
    matrix_report_path = matrix_root / "validation_matrix_report.json"
    approval_report_path = approval_root / "approval_grade_audit_report.json"

    if refresh:
        bridge_report = run_bridge_readiness_audit(
            agent_root=agent_root,
            artifact_root=bridge_root,
        )
        bridge_report_path = Path(str(bridge_report.get("artifact_path") or bridge_report_path))
        matrix_report = run_validation_matrix_audit(
            agent_root=agent_root,
            artifact_root=matrix_root,
            bridge_readiness_report_path=bridge_report_path,
        )
        matrix_report_path = Path(str(matrix_report.get("artifact_path") or matrix_report_path))
        approval_report = run_approval_grade_audit(
            agent_root=agent_root,
            artifact_root=approval_root,
            scope_ids=(scope.scope_id,),
            refresh=True,
        )
        approval_report_path = Path(
            str(approval_report.get("artifact_path") or approval_report_path)
        )
    else:
        bridge_report = _read_json(bridge_report_path)
        matrix_report = _read_json(matrix_report_path)
        approval_report = _read_json(approval_report_path)

    ptg7_gate_report_path, ptg7_command_report_path = _load_or_run_ptg7_gate(
        agent_root=agent_root,
        artifact_root=artifact_root,
        source_root=source_root,
        refresh=refresh,
    )
    live_config_verification_path = _load_or_run_live_config_verifier(
        agent_root=agent_root,
        source_root=source_root,
        bot_id=SCOPE_BOTS.get(scope.scope_id, ""),
        scope=scope,
        refresh=refresh,
    )
    operational_verification_path = _load_or_run_operational_verifier(
        agent_root=agent_root,
        source_root=source_root,
        refresh=refresh,
    )
    deployment_metadata_install_reports = _find_deployment_metadata_install_reports(
        agent_root,
        source_root,
    )
    scheduled_shadow_report_path = _latest_scheduled_shadow_report(agent_root, scope.scope_id)
    learning_sufficiency_manifest_path = _latest_learning_sufficiency_manifest(
        agent_root,
        scope.scope_id,
    )
    production_fixture_manifest_path = _production_fixture_manifest(
        agent_root,
        scope.decision_bridge_id,
    )
    telemetry_manifest_path = (
        learning_sufficiency_manifest_path.with_name("telemetry_manifest.json")
        if learning_sufficiency_manifest_path is not None
        else None
    )
    telemetry_conformance_report = run_telemetry_conformance_check(
        agent_root=agent_root,
        scope_id=scope.scope_id,
        artifact_root=source_root / "telemetry_conformance",
        telemetry_manifest_path=telemetry_manifest_path,
        scheduled_shadow_report_path=scheduled_shadow_report_path,
    )
    telemetry_conformance_report_path = Path(
        str(
            telemetry_conformance_report.get("artifact_path")
            or source_root / "telemetry_conformance" / "telemetry_conformance_report.json"
        )
    )
    optimizer_artifact_root = latest_optimizer_artifact_root(
        scope.scope_id,
        agent_root,
        expected_context=_matrix_scope(matrix_report, scope.scope_id).get(
            "optimizer_evidence_context"
        ),
        manifest_index=build_optimizer_manifest_index(agent_root),
    )
    return SourceContext(
        bridge_readiness=bridge_report,
        validation_matrix=matrix_report,
        approval_grade_audit=approval_report,
        ptg7_gate_report_path=ptg7_gate_report_path,
        ptg7_command_report_path=ptg7_command_report_path,
        live_config_verification_path=live_config_verification_path,
        operational_verification_path=operational_verification_path,
        deployment_metadata_install_reports=deployment_metadata_install_reports,
        operational_evidence_path=resolve_workspace_path(
            agent_root,
            Path("deployments") / "operational_evidence.json",
        ),
        scheduled_shadow_report_path=scheduled_shadow_report_path,
        telemetry_conformance_report_path=telemetry_conformance_report_path,
        optimizer_artifact_root=optimizer_artifact_root,
        learning_sufficiency_manifest_path=learning_sufficiency_manifest_path,
        production_fixture_manifest_path=production_fixture_manifest_path,
    )


def _source_reports_payload(context: SourceContext) -> dict[str, Any]:
    optimizer_manifest = (
        context.optimizer_artifact_root / OPTIMIZER_RUN_MANIFEST
        if context.optimizer_artifact_root is not None
        else None
    )
    return {
        "bridge_readiness": str(context.bridge_readiness.get("artifact_path") or ""),
        "validation_matrix": str(context.validation_matrix.get("artifact_path") or ""),
        "approval_grade_audit": str(context.approval_grade_audit.get("artifact_path") or ""),
        "ptg7_gate_report": str(context.ptg7_gate_report_path),
        "ptg7_command_report": str(context.ptg7_command_report_path),
        "live_config_promotion_verification": str(context.live_config_verification_path),
        "operational_deployment_evidence": (
            str(context.operational_evidence_path)
            if context.operational_evidence_path.exists()
            else ""
        ),
        "operational_deployment_verification": str(context.operational_verification_path),
        "deployment_metadata_install_reports": [
            str(path) for path in context.deployment_metadata_install_reports
        ],
        "scheduled_shadow_cycle": (
            str(context.scheduled_shadow_report_path)
            if context.scheduled_shadow_report_path is not None
            else ""
        ),
        "telemetry_conformance": str(context.telemetry_conformance_report_path),
        "optimizer_run_manifest": (
            str(optimizer_manifest)
            if optimizer_manifest is not None and optimizer_manifest.exists()
            else ""
        ),
        "production_fixture_set_manifest": (
            str(context.production_fixture_manifest_path)
            if context.production_fixture_manifest_path is not None
            else ""
        ),
        "learning_sufficiency_manifest": (
            str(context.learning_sufficiency_manifest_path)
            if context.learning_sufficiency_manifest_path is not None
            else ""
        ),
    }


def _source_reports_loadable_check(context: SourceContext) -> dict[str, Any]:
    required = [
        str(context.bridge_readiness.get("artifact_path") or ""),
        str(context.validation_matrix.get("artifact_path") or ""),
        str(context.approval_grade_audit.get("artifact_path") or ""),
        str(context.ptg7_command_report_path),
        str(context.live_config_verification_path),
        str(context.operational_verification_path),
    ]
    errors: list[str] = []
    evidence_paths: list[str] = []
    for raw_path in required:
        if not raw_path:
            errors.append("source report path missing")
            continue
        path = Path(raw_path)
        evidence_paths.append(str(path))
        if not path.exists() or not path.is_file():
            errors.append(f"source report missing: {path}")
        elif _read_json(path) == {}:
            errors.append(f"source report malformed or empty: {path}")
    return _check("source_reports_loadable", not errors, evidence_paths, errors)


def _validation_matrix_green_check(
    scope: ValidationScope,
    matrix: dict[str, Any],
    agent_root: Path,
) -> dict[str, Any]:
    matrix_scope = _matrix_scope(matrix, scope.scope_id)
    if not matrix_scope:
        return _check(
            "validation_matrix_green",
            False,
            [],
            [f"scope missing from validation matrix: {scope.scope_id}"],
        )
    tests = matrix_scope.get("tests") or {}
    evidence_paths: list[str] = []
    errors: list[str] = []
    for test_name in VALIDATION_TESTS:
        test = tests.get(test_name) or {}
        evidence_paths.extend(_string_paths(test.get("artifact_paths")))
        if test.get("result") != "pass":
            errors.append(
                f"{test_name} result is {test.get('result', 'missing')}: "
                f"{test.get('reason', 'missing')}"
            )
    errors.extend(_missing_evidence_errors(evidence_paths, agent_root))
    return _check("validation_matrix_green", not errors, evidence_paths, errors)


def _bridge_readiness_check(
    scope: ValidationScope,
    bridge_report: dict[str, Any],
    agent_root: Path,
) -> dict[str, Any]:
    bridge_by_id = {
        str(bridge.get("repo_id") or ""): bridge
        for bridge in bridge_report.get("bridges", [])
        if isinstance(bridge, dict)
    }
    bridge_ids = scope.decision_bridge_ids or (scope.decision_bridge_id,)
    errors: list[str] = []
    evidence_paths: list[str] = []
    for bridge_id in bridge_ids:
        bridge = bridge_by_id.get(bridge_id)
        if bridge is None:
            errors.append(f"bridge readiness entry missing: {bridge_id}")
            continue
        evidence_paths.extend(
            str(item.get("path") or "")
            for item in bridge.get("evidence", [])
            if isinstance(item, dict) and item.get("path")
        )
        if bridge.get("status") != "formal_decision_parity_passed":
            errors.append(f"{bridge_id} bridge status is {bridge.get('status', 'missing')}")
        if bridge.get("audit_passed") is not True:
            errors.extend(str(error) for error in bridge.get("errors", []) if error)
    errors.extend(_missing_evidence_errors(evidence_paths, agent_root))
    return _check("formal_decision_parity_passing", not errors, evidence_paths, errors)


def _contract_shadow_maturity_check(
    scope: ValidationScope,
    bridge_report: dict[str, Any],
    agent_root: Path,
) -> dict[str, Any]:
    bridge_by_id = {
        str(bridge.get("repo_id") or ""): bridge
        for bridge in bridge_report.get("bridges", [])
        if isinstance(bridge, dict)
    }
    evidence_paths: list[str] = []
    errors: list[str] = []
    for bridge_id in scope.decision_bridge_ids or (scope.decision_bridge_id,):
        bridge = bridge_by_id.get(bridge_id) or {}
        evidence_paths.extend(_contract_evidence_paths(bridge_id, agent_root))
        maturity = str(bridge.get("maturity") or "missing")
        approval_ready = bridge.get("approval_ready") is True
        if maturity != "shadow_validated" or approval_ready:
            errors.append(
                f"{bridge_id} maturity must be shadow_validated before guarded promotion; "
                f"found maturity={maturity!r}, approval_ready={approval_ready}"
            )
    errors.extend(_missing_evidence_errors(evidence_paths, agent_root))
    return _check("contract_maturity_shadow_validated", not errors, evidence_paths, errors)


def _deployment_metadata_installed_check(
    scope: ValidationScope,
    install_reports: list[Path],
    agent_root: Path,
) -> dict[str, Any]:
    bridge_ids = scope.decision_bridge_ids or (scope.decision_bridge_id,)
    evidence_paths = [str(path) for path in install_reports]
    errors: list[str] = []
    install_by_bridge: dict[str, list[dict[str, Any]]] = {}
    for path in install_reports:
        report = _read_json(path)
        if report:
            install_by_bridge.setdefault(str(report.get("bridge_id") or ""), []).append(report)
    for bridge_id in bridge_ids:
        reports = install_by_bridge.get(bridge_id, [])
        if not reports:
            errors.append(f"{bridge_id} deployment_metadata_install_report missing")
        elif not any(
            report.get("ok") is True and report.get("installed") is True
            for report in reports
        ):
            errors.append(
                f"{bridge_id} deployment metadata was not installed from a passing report"
            )
        contract_dir = CONTRACT_PATHS.get(bridge_id)
        if contract_dir is None:
            errors.append(f"{bridge_id} contract path is unknown")
            continue
        resolved = resolve_workspace_path(agent_root, contract_dir)
        contract_path = resolved / "strategy_plugin_contract.json"
        metadata_path = resolved / "deployment_metadata.json"
        evidence_paths.extend([str(contract_path), str(metadata_path)])
        contract = _read_json(contract_path)
        metadata = _read_json(metadata_path)
        if not metadata:
            errors.append(f"{bridge_id} installed deployment_metadata.json missing or malformed")
            continue
        live_errors = live_deployment_metadata_errors(metadata)
        errors.extend(f"{bridge_id}: {error}" for error in live_errors)
        repo_url = str(metadata.get("repo_url") or "")
        if not repo_url or repo_url.startswith("local://"):
            errors.append(f"{bridge_id}: repo_url is local/shadow-only: {repo_url!r}")
        if contract:
            contract_hash = sha256_file(contract_path)
            if metadata.get("strategy_plugin_contract_hash") != contract_hash:
                errors.append(f"{bridge_id}: strategy_plugin_contract_hash mismatch")
            telemetry_errors = telemetry_schema_contract_errors(metadata, contract)
            errors.extend(f"{bridge_id}: {error}" for error in telemetry_errors)
        if not str(metadata.get("config_hash") or "").strip():
            errors.append(f"{bridge_id}: config_hash missing")
    errors.extend(_missing_evidence_errors(evidence_paths, agent_root))
    return _check(
        "deployment_metadata_promotion_grade_installed",
        not errors,
        evidence_paths,
        errors,
    )


def _live_config_promotion_check(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    evidence_paths = [str(path)]
    errors: list[str] = []
    if not report:
        errors.append("live-config promotion verification report missing or malformed")
    elif report.get("not_applicable_reason"):
        errors = []
    elif report.get("ok") is not True:
        failures = report.get("failures")
        if isinstance(failures, list) and failures:
            errors.extend(str(item) for item in failures)
        else:
            errors.append(
                f"verify_live_config_promotions failed with returncode "
                f"{report.get('returncode', 'missing')}"
            )
    return _check("live_config_promotion_verification_passed", not errors, evidence_paths, errors)


def _operational_deployment_check(evidence_path: Path, verification_path: Path) -> dict[str, Any]:
    evidence_paths = [str(verification_path)]
    if evidence_path.exists():
        evidence_paths.append(str(evidence_path))
    report = _read_json(verification_path)
    errors: list[str] = []
    if not evidence_path.exists():
        errors.append(f"operational deployment evidence missing: {evidence_path}")
    if not report:
        errors.append("operational deployment verification report missing or malformed")
    elif report.get("ok") is not True:
        failures = report.get("errors") or report.get("failures")
        if isinstance(failures, list) and failures:
            errors.extend(str(item) for item in failures)
        else:
            errors.append(
                f"verify_operational_deployment_evidence failed with returncode "
                f"{report.get('returncode', 'missing')}"
            )
    return _check("operational_deployment_evidence_complete", not errors, evidence_paths, errors)


def _scheduled_shadow_cycle_check(path: Path | None, agent_root: Path) -> dict[str, Any]:
    if path is None:
        return _check(
            "scheduled_shadow_cycle_production_grade",
            False,
            [],
            ["production scheduled_shadow_cycle_report.json missing"],
        )
    report = _read_json(path)
    evidence_paths = [str(path)]
    errors: list[str] = []
    if not report:
        errors.append(f"scheduled shadow report missing or malformed: {path}")
        return _check("scheduled_shadow_cycle_production_grade", False, evidence_paths, errors)
    expected = {
        "schema_version": "scheduled_shadow_cycle_report_v1",
        "approval_evidence_mode": True,
        "uses_live_vps_metadata": True,
        "adoption_disabled": True,
        "source_kind": "monthly_validation_shadow",
        "ok": True,
    }
    for key, expected_value in expected.items():
        if report.get(key) != expected_value:
            errors.append(f"{key} must be {expected_value!r}; found {report.get(key)!r}")
    if "fixture" in str(report.get("source_kind") or "").lower():
        errors.append("scheduled shadow source is fixture-scoped, not production monthly shadow")
    evidence_paths.extend(_string_paths(report.get("deployment_metadata_install_report_paths")))
    for field in (
        "monthly_validation_result_path",
        "operational_evidence_path",
        "relay_ingest_evidence_path",
        "learning_sufficiency_manifest_path",
        "optimizer_run_manifest_path",
    ):
        raw_path = str(report.get(field) or "")
        if raw_path:
            evidence_paths.append(raw_path)
        else:
            errors.append(f"{field} missing")
    blockers = report.get("blockers")
    if isinstance(blockers, list) and blockers:
        errors.extend(f"scheduled shadow blocker: {blocker}" for blocker in blockers)
    errors.extend(_missing_evidence_errors(evidence_paths, agent_root))
    errors.extend(_relay_ingest_evidence_errors(report, agent_root))
    return _check("scheduled_shadow_cycle_production_grade", not errors, evidence_paths, errors)


def _telemetry_conformance_check(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    evidence_paths = [str(path)]
    errors: list[str] = []
    if not report:
        errors.append(f"telemetry conformance report missing or malformed: {path}")
        return _check("telemetry_conformance_for_approval_evidence", False, evidence_paths, errors)
    if report.get("ok") is not True:
        blockers = report.get("blockers")
        if isinstance(blockers, list) and blockers:
            errors.extend(str(blocker) for blocker in blockers)
        else:
            errors.append("telemetry conformance report is not ok")
    for raw_path in (
        report.get("telemetry_manifest_path"),
        report.get("scheduled_shadow_report_path"),
        report.get("relay_ingest_evidence_path"),
    ):
        if raw_path:
            evidence_paths.append(str(raw_path))
    return _check(
        "telemetry_conformance_for_approval_evidence",
        not errors,
        evidence_paths,
        errors,
    )


def _production_fixture_breadth_check(path: Path | None, agent_root: Path) -> dict[str, Any]:
    if path is None:
        return _check(
            "production_fixture_breadth_complete",
            False,
            [],
            ["production fixture-set manifest missing"],
        )
    manifest = _read_json(path)
    evidence_paths = [str(path)]
    errors: list[str] = []
    if not manifest:
        errors.append(f"production fixture-set manifest missing or malformed: {path}")
        return _check("production_fixture_breadth_complete", False, evidence_paths, errors)
    if manifest.get("ok") is False:
        errors.append("production fixture-set manifest is not ok")
    status = str(manifest.get("status") or "").strip().lower()
    if status and status not in {"ok", "pass", "passed"}:
        errors.append(f"production fixture-set manifest status is {status}")
    source_kind = str(manifest.get("source_kind") or manifest.get("fixture_scope") or "").lower()
    if source_kind and "fixture" in source_kind and "production" not in source_kind:
        errors.append("fixture-set manifest is fixture-only, not production-derived")
    case_classes = _fixture_case_classes(manifest)
    for required, aliases in FIXTURE_CASE_ALIASES.items():
        if not case_classes.intersection(aliases):
            errors.append(f"fixture-set manifest missing required case class: {required}")
    source_records = _source_records(manifest)
    if not source_records:
        errors.append("fixture-set manifest must include hashed source records")
    for index, record in enumerate(source_records):
        raw_path = str(record.get("path") or record.get("evidence_path") or "")
        expected_hash = str(record.get("sha256") or record.get("hash") or "")
        if raw_path:
            evidence_paths.append(raw_path)
        else:
            errors.append(f"source_records[{index}].path missing")
            continue
        if not expected_hash:
            errors.append(f"source_records[{index}].sha256 missing")
            continue
        resolved = normalize_workspace_path(agent_root, raw_path)
        if not resolved.exists() or not resolved.is_file():
            errors.append(f"source record missing: {raw_path}")
        elif sha256_file(resolved) != expected_hash:
            errors.append(f"source record hash mismatch: {raw_path}")
    errors.extend(_missing_evidence_errors(evidence_paths, agent_root))
    return _check("production_fixture_breadth_complete", not errors, evidence_paths, errors)


def _optimizer_approval_manifest_check(
    scope: ValidationScope,
    context: SourceContext,
    agent_root: Path,
) -> dict[str, Any]:
    matrix_scope = _matrix_scope(context.validation_matrix, scope.scope_id)
    expected_context = matrix_scope.get("optimizer_evidence_context")
    checks = optimizer_evidence_checks(
        scope.scope_id,
        agent_root,
        expected_context=expected_context,
    )
    errors = [error for check in checks for error in check.get("errors", []) if error]
    evidence_paths: list[str] = []
    if context.optimizer_artifact_root is not None:
        evidence_paths.append(str(context.optimizer_artifact_root / OPTIMIZER_RUN_MANIFEST))
    errors.extend(_missing_evidence_errors(evidence_paths, agent_root))
    return _check("optimizer_approval_manifest_p6_p7_complete", not errors, evidence_paths, errors)


def _ptg7_fixture_context_check(scope: ValidationScope, path: Path) -> dict[str, Any]:
    report = _read_json(path)
    evidence_paths = [str(path)]
    errors: list[str] = []
    if not report:
        errors.append("PTG-7 gate report missing or malformed")
    elif report.get("schema_version") != "approval_ready_pilot_ptg7_gate_report_v1":
        errors.append(f"unexpected PTG-7 schema_version: {report.get('schema_version', 'missing')}")
    elif scope.scope_id == PILOT_SCOPE_ID and report.get("pilot_bridge_id") != scope.scope_id:
        errors.append(
            f"PTG-7 pilot_bridge_id {report.get('pilot_bridge_id', 'missing')!r} "
            f"does not match scope {scope.scope_id!r}"
        )
    else:
        fixture_context_passed = any(
            isinstance(check, dict)
            and check.get("name") == "fixture_scoped_pilot_evidence_complete"
            and check.get("passed") is True
            for check in report.get("checks", [])
        )
        implementation_status = str(report.get("implementation_status") or "")
        if (
            not fixture_context_passed
            and implementation_status
            and implementation_status != "pass"
        ):
            errors.append(f"PTG-7 implementation_status is {implementation_status}")
    return _check("ptg7_fixture_context_present_not_authority", not errors, evidence_paths, errors)


def _learning_sufficiency_check(path: Path | None) -> dict[str, Any]:
    if path is None:
        return _check(
            "learning_sufficiency_authoritative",
            False,
            [],
            ["learning_sufficiency_manifest.json missing for selected scope"],
        )
    manifest = _read_json(path)
    evidence_paths = [str(path)]
    errors: list[str] = []
    if not manifest:
        errors.append("learning sufficiency manifest missing or malformed")
    elif manifest.get("eligibility") != "learning_authoritative":
        errors.append(
            "learning sufficiency is "
            f"{manifest.get('eligibility', 'missing')!r}, not 'learning_authoritative'"
        )
        gaps = manifest.get("gaps")
        if isinstance(gaps, list) and gaps:
            for gap in gaps[:5]:
                if isinstance(gap, dict):
                    errors.append(str(gap.get("remediation") or gap.get("gap_id") or gap))
    return _check("learning_sufficiency_authoritative", not errors, evidence_paths, errors)


def _load_or_run_ptg7_gate(
    *,
    agent_root: Path,
    artifact_root: Path,
    source_root: Path,
    refresh: bool,
) -> tuple[Path, Path]:
    report_path = source_root / "ptg7_gate_report.json"
    command_path = source_root / PTG7_COMMAND_REPORT
    if not refresh:
        return report_path, command_path
    tool_path = agent_root / "tools" / "check_approval_ready_pilot_ptg7.py"
    if not tool_path.exists():
        _write_json(
            report_path,
            {
                "schema_version": "approval_ready_pilot_ptg7_gate_report_v1",
                "status": "blocked",
                "implementation_status": "blocked",
                "pilot_bridge_id": PILOT_SCOPE_ID,
                "implementation_failures": [f"missing tool: {tool_path}"],
            },
        )
        _write_json(command_path, {"ok": False, "returncode": 127, "command": []})
        return report_path, command_path
    command = [
        sys.executable,
        str(tool_path),
        "--output",
        str(report_path),
        "--pilot-root",
        str(artifact_root / "ptg7_pilot"),
    ]
    completed = subprocess.run(
        command,
        cwd=agent_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if not report_path.exists():
        _write_json(
            report_path,
            {
                "schema_version": "approval_ready_pilot_ptg7_gate_report_v1",
                "status": "blocked",
                "implementation_status": "blocked",
                "pilot_bridge_id": PILOT_SCOPE_ID,
                "implementation_failures": ["PTG-7 command did not emit report"],
            },
        )
    _write_json(
        command_path,
        {
            "schema_version": "approval_evidence_external_command_v1",
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "output_report_path": str(report_path),
        },
    )
    return report_path, command_path


def _load_or_run_live_config_verifier(
    *,
    agent_root: Path,
    source_root: Path,
    bot_id: str,
    scope: ValidationScope,
    refresh: bool,
) -> Path:
    report_path = source_root / LIVE_CONFIG_VERIFICATION_REPORT
    if not refresh:
        return report_path
    if not bot_id:
        _write_json(
            report_path,
            {
                "schema_version": "live_config_promotion_verification_v1",
                "ok": True,
                "not_applicable_reason": "selected scope has no live-config promotion verifier bot",
            },
        )
        return report_path
    tool_path = agent_root / "tools" / "verify_live_config_promotions.py"
    command = [sys.executable, str(tool_path), "--bot", bot_id, "--strict"]
    completed = subprocess.run(
        command,
        cwd=agent_root,
        capture_output=True,
        text=True,
        check=False,
    )
    all_failures = _failure_lines(completed.stdout, completed.stderr)
    scoped_failures, out_of_scope_failures, unscoped_failures = (
        _partition_live_config_failures(all_failures, scope.strategies)
    )
    observed_scoped_strategies = _observed_live_config_strategies(
        completed.stdout,
        scope.strategies,
    )
    missing_scoped_strategies = sorted(set(scope.strategies) - observed_scoped_strategies)
    scoped_failures.extend(
        f"live-config verifier did not report scoped strategy: {strategy}"
        for strategy in missing_scoped_strategies
    )
    blocking_failures = scoped_failures + unscoped_failures
    _write_json(
        report_path,
        {
            "schema_version": "live_config_promotion_verification_v1",
            "bot_id": bot_id,
            "scope_id": scope.scope_id,
            "scoped_strategy_ids": list(scope.strategies),
            "ok": completed.returncode == 0 or not blocking_failures,
            "returncode": completed.returncode,
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "failures": blocking_failures,
            "all_failures": all_failures,
            "out_of_scope_failures": out_of_scope_failures,
            "observed_scoped_strategy_ids": sorted(observed_scoped_strategies),
        },
    )
    return report_path


def _load_or_run_operational_verifier(
    *,
    agent_root: Path,
    source_root: Path,
    refresh: bool,
) -> Path:
    report_path = source_root / OPERATIONAL_VERIFICATION_REPORT
    if not refresh:
        return report_path
    tool_path = agent_root / "tools" / "verify_operational_deployment_evidence.py"
    command = [sys.executable, str(tool_path)]
    completed = subprocess.run(
        command,
        cwd=agent_root,
        capture_output=True,
        text=True,
        check=False,
    )
    parsed = _loads_json(completed.stdout)
    errors = []
    if isinstance(parsed, dict):
        errors = [str(error) for error in parsed.get("errors", []) if error]
    _write_json(
        report_path,
        {
            "schema_version": "operational_deployment_verification_v1",
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "errors": errors or _failure_lines(completed.stdout, completed.stderr),
        },
    )
    return report_path


def _find_deployment_metadata_install_reports(agent_root: Path, source_root: Path) -> list[Path]:
    roots = [
        source_root,
        resolve_workspace_path(
            agent_root,
            Path("artifacts") / "validation" / "deployment_metadata_install",
        ),
        resolve_workspace_path(
            agent_root,
            Path("trading_assistant_backtest")
            / "artifacts"
            / "validation"
            / "deployment_metadata_install",
        ),
    ]
    reports: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("deployment_metadata_install_report.json"):
            reports[str(path.resolve())] = path.resolve()
    return sorted(reports.values(), key=lambda item: str(item))


def _latest_scheduled_shadow_report(agent_root: Path, scope_id: str) -> Path | None:
    roots = [
        resolve_workspace_path(
            agent_root,
            Path("artifacts") / "validation" / "scheduled_shadow" / scope_id,
        ),
        resolve_workspace_path(
            agent_root,
            Path("trading_assistant")
            / "artifacts"
            / "validation"
            / "scheduled_shadow"
            / scope_id,
        ),
    ]
    candidates = [
        path
        for root in roots
        if root.exists()
        for path in root.rglob(SCHEDULED_SHADOW_REPORT)
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _latest_learning_sufficiency_manifest(agent_root: Path, scope_id: str) -> Path | None:
    root = resolve_workspace_path(agent_root, Path("artifacts") / "learning_sufficiency")
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.rglob("learning_sufficiency_manifest.json")
        if path.is_file() and path.parent.name == scope_id
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (str(item.parent.parent.name), item.stat().st_mtime))


def _production_fixture_manifest(agent_root: Path, bridge_id: str) -> Path | None:
    path = resolve_workspace_path(
        agent_root,
        Path("artifacts")
        / "validation"
        / "decision_parity_matrix"
        / bridge_id
        / PRODUCTION_FIXTURE_MANIFEST,
    )
    return path if path.exists() else None


def _matrix_scope(matrix: dict[str, Any], scope_id: str) -> dict[str, Any]:
    for scope in matrix.get("scopes", []):
        if isinstance(scope, dict) and scope.get("scope_id") == scope_id:
            return scope
    return {}


def _scope_for(scope_id: str) -> ValidationScope:
    scope = _scope_by_id().get(scope_id)
    if scope is None:
        raise ValueError(f"unknown scope: {scope_id}")
    return scope


def _scope_by_id() -> dict[str, ValidationScope]:
    return {scope.scope_id: scope for scope in SCOPES}


def _contract_evidence_paths(bridge_id: str, agent_root: Path) -> list[str]:
    contract_dir = CONTRACT_PATHS.get(bridge_id)
    if contract_dir is None:
        return []
    resolved = resolve_workspace_path(agent_root, contract_dir)
    return [
        str(resolved / "strategy_plugin_contract.json"),
        str(resolved / "deployment_metadata.json"),
    ]


def _evidence_paths_for_bundle(
    source_reports: dict[str, Any],
    checks: list[dict[str, Any]],
) -> list[str]:
    paths: list[str] = []
    for value in source_reports.values():
        if isinstance(value, str) and value:
            paths.append(value)
        elif isinstance(value, list):
            paths.extend(str(item) for item in value if item)
    for check in checks:
        paths.extend(_string_paths(check.get("evidence_paths")))
    return list(dict.fromkeys(paths))


def _evidence_hashes(paths: list[str], agent_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw_path in paths:
        path = normalize_workspace_path(agent_root, raw_path)
        if path.exists() and path.is_file():
            hashes[str(path)] = sha256_file(path)
    return dict(sorted(hashes.items()))


def _missing_evidence_errors(paths: list[str], agent_root: Path) -> list[str]:
    errors: list[str] = []
    for raw_path in paths:
        if not raw_path:
            continue
        path = normalize_workspace_path(agent_root, raw_path)
        if not path.exists() or not path.is_file():
            errors.append(f"evidence path missing: {raw_path}")
    return errors


def _relay_ingest_evidence_errors(report: dict[str, Any], agent_root: Path) -> list[str]:
    raw_path = str(report.get("relay_ingest_evidence_path") or "")
    if not raw_path:
        return []
    path = normalize_workspace_path(agent_root, raw_path)
    if not path.exists() or not path.is_file():
        return []
    evidence = _read_json(path)
    if not evidence:
        return [f"relay ingest evidence missing or malformed: {raw_path}"]

    metadata_refs = _scheduled_shadow_metadata_refs(report, agent_root)
    return relay_evidence.validate_relay_ingest_evidence(
        evidence,
        expected_bot_id=str(report.get("bot_id") or ""),
        deployment_ids=metadata_refs["deployment_ids"],
        runtime_instance_ids=metadata_refs["runtime_instance_ids"],
        deployment_metadata_hashes=metadata_refs["hashes"],
    )


def _scheduled_shadow_metadata_refs(
    report: dict[str, Any],
    agent_root: Path,
) -> dict[str, set[str]]:
    refs = {"deployment_ids": set(), "runtime_instance_ids": set(), "hashes": set()}
    for raw_path in _string_paths(report.get("deployment_metadata_install_report_paths")):
        install_path = normalize_workspace_path(agent_root, raw_path)
        install = _read_json(install_path)
        for key in ("metadata_path", "installed_path"):
            raw_metadata_path = str(install.get(key) or "")
            if not raw_metadata_path:
                continue
            metadata_path = normalize_workspace_path(agent_root, raw_metadata_path)
            if not metadata_path.exists() or not metadata_path.is_file():
                continue
            metadata = _read_json(metadata_path)
            if metadata.get("deployment_id"):
                refs["deployment_ids"].add(str(metadata["deployment_id"]))
            if metadata.get("runtime_instance_id"):
                refs["runtime_instance_ids"].add(str(metadata["runtime_instance_id"]))
            refs["hashes"].add(sha256_file(metadata_path))
    return refs


def _fixture_case_classes(manifest: dict[str, Any]) -> set[str]:
    classes: set[str] = set()
    for value in manifest.get("case_classes") or []:
        classes.add(str(value).strip())
    for item in manifest.get("cases") or []:
        if not isinstance(item, dict):
            continue
        classes.add(str(item.get("case_class") or item.get("class") or "").strip())
    return {item for item in classes if item}


def _source_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ("source_records", "source_evidence", "source_artifacts"):
        value = manifest.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    return records


def _next_required_actions(approval_audit: dict[str, Any], blockers: list[str]) -> list[str]:
    actions = [
        str(action)
        for action in approval_audit.get("next_required_actions", [])
        if str(action).strip()
    ]
    for blocker in blockers:
        if "optimizer_approval_manifest" in blocker:
            actions.append(
                "Generate a real approval-grade optimizer_run_manifest.json "
                "with P6/P7 evidence."
            )
        elif "scheduled_shadow_cycle" in blocker:
            actions.append(
                "Run a production scheduled shadow monthly validation cycle "
                "with adoption disabled."
            )
        elif "learning_sufficiency" in blocker:
            actions.append(
                "Promote pilot learning sufficiency to learning_authoritative "
                "with real runtime evidence."
            )
        elif "deployment_metadata" in blocker:
            actions.append("Install live/VPS-emitted deployment metadata for the selected bridge.")
        elif "production_fixture_breadth" in blocker:
            if (
                "production fixture-set manifest missing" in blocker
                or "production fixture-set manifest missing or malformed" in blocker
            ):
                actions.append(
                    "Add a production-derived parity fixture-set manifest for the pilot bridge."
                )
            elif "live_shadow_telemetry_source" in blocker:
                actions.append(
                    "Attach at least one matching live/shadow runtime telemetry source "
                    "to the production fixture-set manifest."
                )
            else:
                actions.append(
                    "Complete the production fixture-set manifest with observed, "
                    "hashed production case evidence."
                )
    return list(dict.fromkeys(actions))


def _blockers_from_checks(checks: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for check in checks:
        if check["passed"]:
            continue
        errors = check.get("errors") or ["failed"]
        blockers.extend(f"{check['name']}: {error}" for error in errors)
    return blockers


def _check(
    name: str,
    passed: bool,
    evidence_paths: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "evidence_paths": list(dict.fromkeys(path for path in evidence_paths if path)),
        "errors": list(dict.fromkeys(str(error) for error in errors if str(error))),
    }


def _string_paths(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    item = str(value or "")
    return [item] if item else []


def _failure_lines(stdout: str, stderr: str) -> list[str]:
    lines = [
        line.strip()
        for line in f"{stdout}\n{stderr}".splitlines()
        if line.strip().startswith("- ") or "failed" in line.lower()
    ]
    return [line.removeprefix("- ").strip() for line in lines]


def _partition_live_config_failures(
    failures: list[str],
    scoped_strategy_ids: tuple[str, ...],
) -> tuple[list[str], list[str], list[str]]:
    scoped_lookup = {strategy.lower(): strategy for strategy in scoped_strategy_ids}
    scoped: list[str] = []
    out_of_scope: list[str] = []
    unscoped: list[str] = []
    for failure in failures:
        strategy_id = _live_config_failure_strategy(failure)
        if strategy_id == "":
            continue
        if strategy_id is None:
            unscoped.append(failure)
        elif strategy_id.lower() in scoped_lookup:
            scoped.append(failure)
        else:
            out_of_scope.append(failure)
    return scoped, out_of_scope, unscoped


def _live_config_failure_strategy(failure: str) -> str | None:
    text = failure.strip()
    if text.lower().rstrip(":") == "live-config promotion check failed":
        return ""
    label = text.split(maxsplit=1)[0] if text else ""
    if ":" not in label:
        return None
    return label.split(":", 1)[1].strip() or None


def _observed_live_config_strategies(
    stdout: str,
    scoped_strategy_ids: tuple[str, ...],
) -> set[str]:
    scoped_lookup = {strategy.lower(): strategy for strategy in scoped_strategy_ids}
    observed: set[str] = set()
    for line in stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 2 or parts[0] not in {"PASS", "FAIL"} or ":" not in parts[1]:
            continue
        strategy_id = parts[1].split(":", 1)[1].strip().lower()
        if strategy_id in scoped_lookup:
            observed.add(scoped_lookup[strategy_id])
    return observed


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _loads_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
