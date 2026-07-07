"""Fail-closed approval-grade audit for strategy learning-loop promotion."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.validation.bridge_readiness import run_bridge_readiness_audit
from trading_assistant_backtest.validation.deployment_metadata_contract import (
    live_deployment_metadata_errors,
    telemetry_schema_contract_errors,
)
from trading_assistant_backtest.validation.optimizer_evidence import (
    optimizer_readiness_summary,
)
from trading_assistant_backtest.validation.validation_matrix import (
    SCOPES,
    ValidationScope,
    run_validation_matrix_audit,
)
from trading_assistant_backtest.paths import monorepo_root, package_root, resolve_workspace_path

CONTRACT_PATHS = {
    "crypto_trend_v1": Path("contracts/strategy_plugins/crypto_trend_v1"),
    "crypto_momentum_v1": Path("contracts/strategy_plugins/crypto_momentum_v1"),
    "crypto_breakout_v1": Path("contracts/strategy_plugins/crypto_breakout_v1"),
    "k_stock_olr_kalcb": Path("contracts/strategy_plugins/k_stock_olr_kalcb"),
    "trading_stock_family": Path("contracts/strategy_plugins/trading_stock_family"),
    "trading_momentum_family": Path(
        "contracts/strategy_plugins/trading_momentum_family"
    ),
    "trading_swing_family": Path("contracts/strategy_plugins/trading_swing_family"),
}

APPROVAL_TESTS = (
    "data_reproduction",
    "incumbent_replay",
    "decision_parity",
    "round_reproduction",
    "historical_walk_forward",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether strategy scopes are approval-grade for structural candidates."
    )
    parser.add_argument("--agent-root", type=Path, default=_default_agent_root())
    parser.add_argument("--artifact-root", type=Path, default=_default_artifact_root())
    parser.add_argument(
        "--scope",
        action="append",
        choices=(*[scope.scope_id for scope in SCOPES], "all"),
        default=None,
        help="Scope to audit. Repeatable. Defaults to all active scopes.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Regenerate source reports first. This is the default for approval audits.",
    )
    parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Diagnostic-only: reuse cached validation/bridge reports instead of refreshing.",
    )
    args = parser.parse_args(argv)

    report = run_approval_grade_audit(
        agent_root=args.agent_root,
        artifact_root=args.artifact_root,
        scope_ids=_scope_ids(args.scope),
        refresh=args.refresh or not args.use_cached,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["approval_grade"] else 1


def run_approval_grade_audit(
    *,
    agent_root: Path,
    artifact_root: Path,
    scope_ids: tuple[str, ...] | None = None,
    refresh: bool = True,
) -> dict[str, Any]:
    """Emit a promotion-grade report without mutating contracts."""

    agent_root = Path(agent_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    matrix = _load_or_run_validation_matrix(
        agent_root=agent_root,
        refresh=refresh,
    )
    bridge_report = _load_or_run_bridge_readiness(
        agent_root=agent_root,
        refresh=refresh,
    )
    bridge_by_id = {bridge["repo_id"]: bridge for bridge in bridge_report.get("bridges", [])}
    scope_by_id = {scope.scope_id: scope for scope in SCOPES}
    wanted = scope_ids or tuple(scope.scope_id for scope in SCOPES)
    matrix_scope_by_id = {scope["scope_id"]: scope for scope in matrix.get("scopes", [])}

    audited_scopes = []
    for scope_id in wanted:
        scope = scope_by_id[scope_id]
        matrix_scope = matrix_scope_by_id.get(scope_id, {})
        bridge_ids = scope.decision_bridge_ids or (scope.decision_bridge_id,)
        audited_scopes.append(
            _audit_scope(
                scope,
                matrix_scope=matrix_scope,
                bridges=[bridge_by_id.get(bridge_id, {}) for bridge_id in bridge_ids],
                agent_root=agent_root,
            )
        )

    report_path = artifact_root / "approval_grade_audit_report.json"
    approval_grade = bool(audited_scopes) and all(
        scope["approval_grade"] for scope in audited_scopes
    )
    report = {
        "schema_version": "approval_grade_audit_v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "artifact_path": str(report_path),
        "source_reports_refreshed": refresh,
        "approval_grade": approval_grade,
        "structural_candidates_approval_ready": approval_grade,
        "audited_scope_count": len(audited_scopes),
        "approved_scopes": [
            scope["scope_id"] for scope in audited_scopes if scope["approval_grade"]
        ],
        "blocked_scopes": [
            scope["scope_id"] for scope in audited_scopes if not scope["approval_grade"]
        ],
        "checks": _summary_checks(audited_scopes),
        "next_required_actions": _next_required_actions(audited_scopes),
        "scopes": audited_scopes,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def _audit_scope(
    scope: ValidationScope,
    *,
    matrix_scope: dict[str, Any],
    bridges: list[dict[str, Any]],
    agent_root: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    tests = matrix_scope.get("tests") or {}
    for test_name in APPROVAL_TESTS:
        test = tests.get(test_name) or {}
        passed = test.get("result") == "pass"
        checks.append(
            _check(
                f"{test_name}_approval_grade_pass",
                passed,
                []
                if passed
                else [
                    f"{test_name} result is {test.get('result', 'missing')}: "
                    f"{test.get('reason', 'missing')}"
                ],
            )
        )

    for bridge in bridges:
        bridge_id = str(bridge.get("repo_id") or "")
        checks.extend(_bridge_checks(bridge_id, bridge, agent_root))
    optimizer = matrix_scope.get("optimizer_approval_readiness") or {}
    optimizer_checks = optimizer.get("checks") or [
        _check(
            f"{scope.scope_id}:optimizer_evidence_context_bound",
            False,
            ["validation matrix did not include optimizer approval readiness checks"],
        )
    ]
    checks.extend(optimizer_checks)

    return {
        "scope_id": scope.scope_id,
        "repo_id": scope.repo_id,
        "portfolio_id": scope.portfolio_id,
        "strategies": list(scope.strategies),
        "bridge_ids": [
            bridge.get("repo_id", "")
            for bridge in bridges
            if bridge.get("repo_id")
        ],
        "approval_grade": all(check["passed"] for check in checks),
        "optimizer_approval_readiness": optimizer_readiness_summary(checks),
        "checks": checks,
    }


def _bridge_checks(
    bridge_id: str,
    bridge: dict[str, Any],
    agent_root: Path,
) -> list[dict[str, Any]]:
    checks = [
        _check(
            f"{bridge_id}:formal_decision_parity_passed",
            bridge.get("status") == "formal_decision_parity_passed",
            []
            if bridge.get("status") == "formal_decision_parity_passed"
            else [f"bridge status is {bridge.get('status', 'missing')}"],
        ),
        _check(
            f"{bridge_id}:contract_maturity_approval_ready",
            bridge.get("maturity") == "approval_ready" and bridge.get("approval_ready") is True,
            []
            if bridge.get("maturity") == "approval_ready" and bridge.get("approval_ready") is True
            else [
                f"bridge maturity is {bridge.get('maturity', 'missing')}; "
                f"approval_ready={bridge.get('approval_ready', False)}"
            ],
        ),
        _check(
            f"{bridge_id}:source_checkout_clean",
            bridge.get("source_checkout_clean") is True,
            [] if bridge.get("source_checkout_clean") is True else bridge.get("errors", []),
        ),
    ]
    checks.extend(_deployment_metadata_checks(bridge_id, agent_root))
    return checks


def _deployment_metadata_checks(bridge_id: str, agent_root: Path) -> list[dict[str, Any]]:
    contract_dir = CONTRACT_PATHS.get(bridge_id)
    if contract_dir is None:
        return [_check(f"{bridge_id}:deployment_metadata_present", False, ["unknown bridge id"])]
    resolved = resolve_workspace_path(agent_root, contract_dir)
    contract_path = resolved / "strategy_plugin_contract.json"
    metadata_path = resolved / "deployment_metadata.json"
    if not contract_path.exists() or not metadata_path.exists():
        return [
            _check(
                f"{bridge_id}:deployment_metadata_present",
                False,
                [f"missing contract or deployment metadata under {resolved}"],
            )
        ]
    contract = _read_json(contract_path)
    metadata = _read_json(metadata_path)
    live_emission_errors = live_deployment_metadata_errors(metadata)
    repo_url = str(metadata.get("repo_url") or "")
    contract_hash = sha256_file(contract_path)
    telemetry_errors = telemetry_schema_contract_errors(metadata, contract)
    return [
        _check(
            f"{bridge_id}:deployment_metadata_live_emitted",
            not live_emission_errors,
            []
            if not live_emission_errors
            else live_emission_errors,
        ),
        _check(
            f"{bridge_id}:deployment_repo_url_not_local_shadow",
            bool(repo_url) and not repo_url.startswith("local://"),
            []
            if bool(repo_url) and not repo_url.startswith("local://")
            else [f"repo_url is local/shadow-only: {repo_url!r}"],
        ),
        _check(
            f"{bridge_id}:deployment_sha_matches_contract",
            metadata.get("deployed_commit_sha") == contract.get("live_repo_commit_sha"),
            []
            if metadata.get("deployed_commit_sha") == contract.get("live_repo_commit_sha")
            else ["deployed_commit_sha does not match contract live_repo_commit_sha"],
        ),
        _check(
            f"{bridge_id}:deployment_contract_hash_matches",
            metadata.get("strategy_plugin_contract_hash") == contract_hash,
            []
            if metadata.get("strategy_plugin_contract_hash") == contract_hash
            else ["strategy_plugin_contract_hash does not match contract file"],
        ),
        _check(
            f"{bridge_id}:deployment_config_hash_present",
            bool(str(metadata.get("config_hash") or "").strip()),
            [] if metadata.get("config_hash") else ["config_hash missing"],
        ),
        _check(
            f"{bridge_id}:deployment_telemetry_schema_matches_contract",
            not telemetry_errors,
            telemetry_errors,
        ),
    ]


def _summary_checks(scopes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _check(
            "all_selected_scopes_approval_grade",
            bool(scopes) and all(scope["approval_grade"] for scope in scopes),
            [
                scope["scope_id"]
                for scope in scopes
                if not scope["approval_grade"]
            ],
        )
    ]


def _next_required_actions(scopes: list[dict[str, Any]]) -> list[str]:
    failed_names = {
        check["name"]
        for scope in scopes
        for check in scope["checks"]
        if check["passed"] is False
    }
    failed_replay_scopes = [
        scope["scope_id"]
        for scope in scopes
        if any(
            check["passed"] is False
            and check["name"] in {
                "incumbent_replay_approval_grade_pass",
                "round_reproduction_approval_grade_pass",
                "historical_walk_forward_approval_grade_pass",
            }
            for check in scope["checks"]
        )
    ]
    actions: list[str] = []
    if any(name.startswith("data_reproduction_") for name in failed_names):
        actions.append(
            "Promote full-family data bundles to authoritative for every audited scope."
        )
    if any(
        name.startswith(prefix)
        for name in failed_names
        for prefix in (
            "incumbent_replay_",
            "round_reproduction_",
            "historical_walk_forward_",
        )
    ):
        if failed_replay_scopes == ["k_stock_olr_kalcb"]:
            actions.append(
                "Fill or authoritatively explain the March/April KIS 5m gaps, rebuild "
                "k_stock OLR/KALCB authoritative portfolio bundles, and rerun "
                "historical walk-forward."
            )
        else:
            actions.append(
                "Replace incomplete replay evidence with production replay baselines, "
                "round reproduction, and multi-month walk-forward evidence for the "
                f"blocked scope(s): {', '.join(failed_replay_scopes)}."
            )
    if any(name.endswith(":contract_maturity_approval_ready") for name in failed_names):
        actions.append(
            "Promote strategy contracts from shadow_validated to approval_ready only after "
            "their deployment, data, parity, and replay evidence passes."
        )
    if any(":optimizer_p6_" in name or ":optimizer_p7_" in name for name in failed_names):
        actions.append(
            "Complete P6/P7 monthly optimizer evidence: true two-fold in-sample scoring, "
            "post-ranking selection-OOS repair trigger evidence, confirmatory rerank, "
            "and one round_N+1 recommendation or deterministic no-adoption reason."
        )
    if any(
        name.endswith(":deployment_metadata_live_emitted")
        or name.endswith(":deployment_repo_url_not_local_shadow")
        for name in failed_names
    ):
        actions.append(
            "Persist VPS/live-bot emitted deployment metadata with non-local repo URLs, "
            "deployed SHAs, config hashes, telemetry schemas, and contract hashes."
        )
    return actions


def _load_or_run_validation_matrix(*, agent_root: Path, refresh: bool) -> dict[str, Any]:
    report_path = (
        agent_root
        / "trading_assistant_backtest"
        / "artifacts"
        / "validation"
        / "validation_matrix"
        / "validation_matrix_report.json"
    )
    if report_path.exists() and not refresh:
        return _read_json(report_path)
    return run_validation_matrix_audit(
        agent_root=agent_root,
        artifact_root=report_path.parent,
    )


def _load_or_run_bridge_readiness(*, agent_root: Path, refresh: bool) -> dict[str, Any]:
    report_path = (
        agent_root
        / "trading_assistant_backtest"
        / "artifacts"
        / "validation"
        / "bridge_readiness"
        / "bridge_readiness_report.json"
    )
    if report_path.exists() and not refresh:
        return _read_json(report_path)
    return run_bridge_readiness_audit(
        agent_root=agent_root,
        artifact_root=report_path.parent,
    )


def _check(name: str, passed: bool, errors: list[str]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "errors": errors}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _scope_ids(raw: list[str] | None) -> tuple[str, ...] | None:
    if not raw or "all" in raw:
        return None
    return tuple(dict.fromkeys(raw))


def _default_artifact_root() -> Path:
    return package_root() / "artifacts" / "validation" / "approval_grade"


def _default_agent_root() -> Path:
    return monorepo_root()


def _repo_root() -> Path:
    return monorepo_root()


if __name__ == "__main__":
    raise SystemExit(main())
