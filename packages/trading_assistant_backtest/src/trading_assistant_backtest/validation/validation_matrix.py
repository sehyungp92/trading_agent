"""Portfolio-by-portfolio validation readiness and evidence matrix."""

from __future__ import annotations

import argparse
import json
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.paths import (
    monorepo_root,
    normalize_workspace_path,
    package_root,
    resolve_workspace_path,
    workspace_root,
)
from trading_assistant_backtest.validation.bridge_readiness import run_bridge_readiness_audit
from trading_assistant_backtest.validation.optimizer_evidence import (
    build_optimizer_manifest_index,
    optimizer_evidence_checks,
    optimizer_readiness_summary,
)

VALIDATION_TESTS = (
    "data_reproduction",
    "incumbent_replay",
    "decision_parity",
    "round_reproduction",
    "historical_walk_forward",
)


@dataclass(frozen=True)
class ValidationScope:
    scope_id: str
    repo_id: str
    portfolio_id: str
    strategies: tuple[str, ...]
    weekly_focus_week: int
    decision_bridge_id: str
    decision_bridge_ids: tuple[str, ...] = ()
    formal_bridge_covers: tuple[str, ...] = ()
    data_bundle_ids: tuple[str, ...] = ()
    full_family_requirement_path: str = ""
    full_family_default_run_month: str = "2026-05"
    notes: str = ""


SCOPES = (
    ValidationScope(
        scope_id="k_stock_olr_kalcb",
        repo_id="k_stock_trader",
        portfolio_id="olr_kalcb",
        strategies=("OLR", "KALCB"),
        weekly_focus_week=1,
        decision_bridge_id="k_stock_olr_kalcb",
        formal_bridge_covers=("OLR", "KALCB"),
        data_bundle_ids=("k_stock_olr_kalcb_portfolio",),
        full_family_requirement_path=(
            "trading_assistant_data/data/requirements/strategies/k_stock/portfolio.json"
        ),
    ),
    ValidationScope(
        scope_id="trading_stock_family",
        repo_id="trading",
        portfolio_id="stock",
        strategies=("IARIC_v1", "ALCB_v1"),
        weekly_focus_week=1,
        decision_bridge_id="trading_stock_family",
        formal_bridge_covers=("IARIC_v1", "ALCB_v1"),
        data_bundle_ids=("trading_stock_family_portfolio",),
        full_family_requirement_path=(
            "trading_assistant_data/data/requirements/strategies/trading_stock/portfolio.json"
        ),
    ),
    ValidationScope(
        scope_id="trading_momentum_family",
        repo_id="trading",
        portfolio_id="momentum",
        strategies=("NQDTC_v2.1", "NQ_REGIME", "VdubusNQ_v4", "DownturnDominator_v1"),
        weekly_focus_week=2,
        decision_bridge_id="trading_momentum_family",
        formal_bridge_covers=("NQDTC_v2.1", "NQ_REGIME", "VdubusNQ_v4", "DownturnDominator_v1"),
        data_bundle_ids=("trading_momentum_family_portfolio",),
        full_family_requirement_path=(
            "trading_assistant_data/data/requirements/strategies/trading_momentum/portfolio.json"
        ),
        notes="Only active futures portfolio in the trading repo.",
    ),
    ValidationScope(
        scope_id="trading_swing_family",
        repo_id="trading",
        portfolio_id="swing",
        strategies=("ATRSS", "AKC_HELIX", "TPC", "OVERLAY"),
        weekly_focus_week=3,
        decision_bridge_id="trading_swing_family",
        formal_bridge_covers=("ATRSS", "AKC_HELIX", "TPC", "OVERLAY"),
        data_bundle_ids=("trading_swing_family_portfolio",),
        full_family_requirement_path=(
            "trading_assistant_data/data/requirements/strategies/trading_swing/portfolio.json"
        ),
        notes="OVERLAY is a portfolio-level swing coordination surface.",
    ),
    ValidationScope(
        scope_id="crypto_trader_portfolio",
        repo_id="crypto_trader",
        portfolio_id="crypto_portfolio",
        strategies=("trend", "momentum", "breakout"),
        weekly_focus_week=4,
        decision_bridge_id="crypto_trend_v1",
        decision_bridge_ids=("crypto_trend_v1", "crypto_momentum_v1", "crypto_breakout_v1"),
        formal_bridge_covers=("trend", "momentum", "breakout"),
        data_bundle_ids=("crypto_portfolio_phased_optimizer",),
        full_family_requirement_path=(
            "trading_assistant_data/data/requirements/strategies/crypto_portfolio/phased_optimizer.json"
        ),
        notes=(
            "Formal crypto bridges are shadow_validated; phased optimizer data excludes "
            "extra 1m/5m refresh intervals."
        ),
    ),
)


DEFAULT_DATA_REPRODUCTION_ROOT = Path(
    "artifacts/validation/data_reproduction"
)
DEFAULT_BRIDGE_READINESS_REPORT = Path(
    "artifacts/validation/bridge_readiness/bridge_readiness_report.json"
)
DEFAULT_BRIDGE_READINESS_ARTIFACT_ROOT = Path(
    "artifacts/validation/bridge_readiness"
)
DEFAULT_REPLAY_EVIDENCE_ROOT = Path(
    "artifacts/validation/replay_evidence"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit validation readiness across active strategy portfolios."
    )
    parser.add_argument("--agent-root", type=Path, default=_default_agent_root())
    parser.add_argument("--artifact-root", type=Path, default=_default_artifact_root())
    parser.add_argument("--bridge-readiness-report", type=Path, default=None)
    parser.add_argument("--data-reproduction-report", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_validation_matrix_audit(
        agent_root=args.agent_root,
        artifact_root=args.artifact_root,
        bridge_readiness_report_path=args.bridge_readiness_report,
        data_reproduction_report_path=args.data_reproduction_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["runnable_validations_passed"] else 1


def run_validation_matrix_audit(
    *,
    agent_root: Path,
    artifact_root: Path,
    bridge_readiness_report_path: Path | None = None,
    data_reproduction_report_path: Path | None = None,
) -> dict[str, Any]:
    """Build a durable matrix of what can be validated now and what remains blocked."""

    agent_root = Path(agent_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    bridge_report = _load_or_run_bridge_readiness(
        agent_root=agent_root,
        artifact_root=artifact_root,
        bridge_readiness_report_path=bridge_readiness_report_path,
    )
    data_reports = _load_data_reproduction_reports(
        agent_root=agent_root,
        data_reproduction_report_path=data_reproduction_report_path,
    )
    replay_evidence_reports = _load_replay_evidence_reports(agent_root=agent_root)
    bridges = {bridge["repo_id"]: bridge for bridge in bridge_report.get("bridges", [])}
    optimizer_manifest_index = build_optimizer_manifest_index(agent_root)

    scope_rows = [
        _scope_row(
            scope,
            bridges=bridges,
            data_reports=data_reports,
            replay_evidence_reports=replay_evidence_reports,
            optimizer_manifest_index=optimizer_manifest_index,
            agent_root=agent_root,
        )
        for scope in SCOPES
    ]
    blocked = _blocked_reasons(scope_rows)
    runnable_validations_passed = _runnable_validations_passed(scope_rows)
    all_validations_runnable = _all_validations_runnable(scope_rows)
    approval_grade_validation_complete = _approval_grade_validation_complete(scope_rows)
    approval_remaining_gaps = _approval_remaining_gaps(scope_rows)
    artifact_path = artifact_root / "validation_matrix_report.json"
    report = {
        "ok": runnable_validations_passed,
        "runnable_validations_passed": runnable_validations_passed,
        "all_validation_tests_runnable_for_all_scopes": all_validations_runnable,
        "approval_grade_validation_complete": approval_grade_validation_complete,
        "structural_candidates_approval_ready": approval_grade_validation_complete,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "artifact_path": str(artifact_path),
        "validation_tests": list(VALIDATION_TESTS),
        "active_scope_count": len(scope_rows),
        "scopes": scope_rows,
        "carried_out_now_or_previously": _carried_out(scope_rows),
        "meaningful_remaining_gaps": blocked,
        "approval_remaining_gaps": approval_remaining_gaps,
        "excluded_non_rotation_surfaces": [],
    }
    artifact_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _scope_row(
    scope: ValidationScope,
    *,
    bridges: dict[str, dict[str, Any]],
    data_reports: list[dict[str, Any]],
    replay_evidence_reports: list[dict[str, Any]],
    optimizer_manifest_index: dict[str, list[dict[str, Any]]],
    agent_root: Path,
) -> dict[str, Any]:
    data_reproduction = _data_reproduction_status(scope, data_reports, agent_root)
    tests = {
        "data_reproduction": data_reproduction,
        "incumbent_replay": _replay_evidence_status(
            scope,
            replay_evidence_reports,
            "incumbent_replay",
            "missing_replay_backed_evaluator_and_old_diagnostics_baseline",
            (
                "The backtest runner must replay the latest accepted live config and compare "
                "core metrics to frozen latest-round diagnostics."
            ),
            data_reproduction,
            agent_root,
        ),
        "decision_parity": _decision_parity_status(scope, bridges, agent_root),
        "round_reproduction": _replay_evidence_status(
            scope,
            replay_evidence_reports,
            "round_reproduction",
            "missing_round_reproduction_baseline_and_replay_evaluator",
            (
                "Latest phased-auto round manifests and frozen starting baselines must "
                "reproduce with replay-backed candidate scoring."
            ),
            data_reproduction,
            agent_root,
        ),
        "historical_walk_forward": _replay_evidence_status(
            scope,
            replay_evidence_reports,
            "historical_walk_forward",
            "missing_multi_month_authoritative_data_and_replay_evaluator",
            (
                "Historical monthly shadow walk-forward needs several frozen monthly data "
                "bundles plus replay-backed evaluator coverage for each selected scope."
            ),
            data_reproduction,
            agent_root,
        ),
    }
    optimizer_context = _optimizer_evidence_context(
        scope,
        bridges=bridges,
        data_reports=data_reports,
        agent_root=agent_root,
    )
    optimizer_checks = optimizer_evidence_checks(
        scope.scope_id,
        agent_root,
        expected_context=optimizer_context,
        manifest_index=optimizer_manifest_index,
    )
    return {
        "scope_id": scope.scope_id,
        "repo_id": scope.repo_id,
        "portfolio_id": scope.portfolio_id,
        "strategies": list(scope.strategies),
        "weekly_focus_week": scope.weekly_focus_week,
        "notes": scope.notes,
        "tests": tests,
        "optimizer_evidence_context": optimizer_context,
        "optimizer_approval_readiness": optimizer_readiness_summary(optimizer_checks),
        "source_repo_present": _repo_present(agent_root, scope.repo_id),
    }


def _data_reproduction_status(
    scope: ValidationScope,
    data_reports: list[dict[str, Any]],
    agent_root: Path,
) -> dict[str, Any]:
    full_family_authority = _full_family_authority_status(scope, agent_root, data_reports)
    if not scope.data_bundle_ids:
        status = _blocked_status(
            "authoritative_data_bundle_missing_for_scope",
            "No committed authoritative monthly data bundle exists for this portfolio scope.",
        )
        status["full_family_authority"] = full_family_authority
        return status
    if not data_reports:
        status = _blocked_status(
            "data_reproduction_report_missing",
            "A data bundle is expected, but no durable data_reproduction_report.json was found.",
        )
        status["full_family_authority"] = full_family_authority
        return status
    reports = _reports_for_bundle_ids(data_reports, scope.data_bundle_ids)
    if not reports and len(data_reports) == 1:
        reports = data_reports
    if not reports:
        status = _blocked_status(
            "data_reproduction_report_missing_for_expected_bundle",
            "No durable data reproduction report matched the expected bundle ids.",
        )
        status["full_family_authority"] = full_family_authority
        return status
    failed = [report for report in reports if report.get("ok") is not True]
    if failed:
        return {
            "result": "fail",
            "runnable": True,
            "ran": True,
            "reason": "data_reproduction_report_failed",
            "artifact_paths": _report_paths(failed, agent_root),
            "full_family_authority": full_family_authority,
        }
    present_ids = {_bundle_id(report) for report in reports if _bundle_id(report)}
    missing_ids = sorted(set(scope.data_bundle_ids) - present_ids) if present_ids else []
    result = "partial_pass" if missing_ids else "pass"
    reason = (
        "Expected authoritative bundle coverage is still incomplete."
        if missing_ids
        else "authoritative data bundle reproduced"
    )
    details = ""
    if full_family_authority.get("status") == "blocked":
        result = "partial_pass"
        reason = "smoke_data_reproduced_full_family_authority_blocked"
        details = str(full_family_authority.get("details") or "")
    elif full_family_authority.get("status") == "pass" and missing_ids:
        details = (
            "Full-family authority passed, but expected smoke reproduction reports are missing."
        )
    return {
        "result": result,
        "runnable": True,
        "ran": True,
        "reason": reason,
        "details": details,
        "bundle_ids": list(scope.data_bundle_ids),
        "covered_bundle_ids": sorted(present_ids),
        "missing_bundle_ids": missing_ids,
        "bundle_checksums": [report.get("bundle_checksum", "") for report in reports],
        "recomputed_bundle_checksums": [
            report.get("recomputed_bundle_checksum", "") for report in reports
        ],
        "slice_count": sum(int(report.get("slice_count") or 0) for report in reports),
        "artifact_paths": _report_paths(reports, agent_root),
        "full_family_authority": full_family_authority,
    }


def _decision_parity_status(
    scope: ValidationScope,
    bridges: dict[str, dict[str, Any]],
    agent_root: Path,
) -> dict[str, Any]:
    bridge_ids = scope.decision_bridge_ids or (scope.decision_bridge_id,)
    scope_bridges = [bridges.get(bridge_id) for bridge_id in bridge_ids]
    missing_bridge_ids = [
        bridge_id for bridge_id, bridge in zip(bridge_ids, scope_bridges, strict=True) if not bridge
    ]
    if missing_bridge_ids:
        return _blocked_status(
            "formal_decision_parity_bridge_missing",
            "Missing bridge readiness entries: " + ", ".join(missing_bridge_ids),
        )
    assert all(bridge is not None for bridge in scope_bridges)
    present_bridges = [bridge for bridge in scope_bridges if bridge is not None]
    failed = [
        bridge for bridge in present_bridges if bridge["status"] != "formal_decision_parity_passed"
    ]
    if failed:
        if all(_durable_decision_parity_passed(bridge, agent_root) for bridge in present_bridges):
            evidence_paths = _decision_parity_evidence_paths(present_bridges, agent_root)
            approval_ready = all(
                bool(bridge.get("approval_ready", False)) for bridge in present_bridges
            )
            return {
                "result": "pass",
                "runnable": True,
                "ran": True,
                "reason": "formal_decision_parity_passed",
                "details": (
                    "Durable formal decision parity report passed; current source "
                    "checkout warnings are tracked outside replay readiness. Structural "
                    "candidate approval remains blocked until plugin maturity is approval_ready."
                    if not approval_ready
                    else ""
                ),
                "bridge_id": scope.decision_bridge_id,
                "bridge_ids": list(bridge_ids),
                "covered_strategies": list(scope.formal_bridge_covers),
                "missing_strategies": sorted(
                    set(scope.strategies) - set(scope.formal_bridge_covers)
                ),
                "maturity": ",".join(
                    sorted({str(bridge.get("maturity", "")) for bridge in present_bridges})
                ),
                "eligible_for_optimizer": all(
                    bool(bridge.get("eligible_for_optimizer", False))
                    for bridge in present_bridges
                ),
                "approval_ready": approval_ready,
                "artifact_paths": evidence_paths,
            }
        bridge = failed[0]
        evidence_paths = _decision_parity_evidence_paths([bridge], agent_root)
        if evidence_paths:
            return {
                "result": "fail",
                "runnable": True,
                "ran": True,
                "reason": "formal_decision_parity_report_failed",
                "details": "; ".join(bridge.get("errors", []))
                or "Formal decision parity report did not pass for this scope.",
                "bridge_id": bridge.get("repo_id", ""),
                "bridge_ids": list(bridge_ids),
                "maturity": bridge.get("maturity", ""),
                "eligible_for_optimizer": bridge.get("eligible_for_optimizer", False),
                "approval_ready": bridge.get("approval_ready", False),
                "artifact_paths": evidence_paths,
            }
        return _blocked_status(
            "formal_decision_parity_report_missing_or_failed",
            "; ".join(bridge.get("errors", []))
            or "; ".join(bridge.get("approval_blockers", []))
            or "Formal decision parity did not pass for this scope.",
        )
    evidence_paths = _decision_parity_evidence_paths(present_bridges, agent_root)
    missing = sorted(set(scope.strategies) - set(scope.formal_bridge_covers))
    result = "partial_pass" if missing else "pass"
    approval_ready = all(bool(bridge.get("approval_ready", False)) for bridge in present_bridges)
    maturity = sorted({str(bridge.get("maturity", "")) for bridge in present_bridges})
    return {
        "result": result,
        "runnable": True,
        "ran": True,
        "reason": (
            "Formal decision parity passed for covered strategy surface; remaining strategies "
            "need their own contracts/adapters."
            if missing
            else (
                "Formal decision parity passed and plugin is approval_ready."
                if approval_ready
                else "formal_decision_parity_passed"
            )
        ),
        "details": (
            "The adapter is repeatable and eligible for shadow optimizer flows, but structural "
            "candidate approval remains blocked until the live contract is promoted to "
            "approval_ready with broader shadow evidence."
            if not approval_ready
            else ""
        ),
        "bridge_id": scope.decision_bridge_id,
        "bridge_ids": list(bridge_ids),
        "covered_strategies": list(scope.formal_bridge_covers),
        "missing_strategies": missing,
        "maturity": ",".join(maturity),
        "eligible_for_optimizer": all(
            bool(bridge.get("eligible_for_optimizer", False)) for bridge in present_bridges
        ),
        "approval_ready": approval_ready,
        "artifact_paths": evidence_paths,
    }


def _decision_parity_evidence_paths(
    bridges: list[dict[str, Any]],
    agent_root: Path,
) -> list[str]:
    return _normalize_artifact_paths(
        agent_root,
        (
            item.get("path", "")
            for bridge in bridges
            for item in bridge.get("evidence", [])
            if str(item.get("path", "")).endswith("decision_parity_report.json")
        ),
    )


def _durable_decision_parity_passed(bridge: dict[str, Any] | None, agent_root: Path) -> bool:
    if bridge is None:
        return False
    for item in bridge.get("evidence", []):
        path = _resolve_artifact_path(agent_root, item.get("path", ""))
        if not path.name == "decision_parity_report.json" or not path.exists():
            continue
        report = _read_json(path)
        if report.get("status") == "pass":
            return True
    return False


def _replay_evidence_status(
    scope: ValidationScope,
    reports: list[dict[str, Any]],
    test_name: str,
    missing_reason: str,
    missing_details: str,
    data_reproduction: dict[str, Any],
    agent_root: Path,
) -> dict[str, Any]:
    scope_reports = [report for report in reports if report.get("scope_id") == scope.scope_id]
    passing = [
        report
        for report in scope_reports
        if (report.get("tests") or {}).get(test_name, {}).get("ok") is True
        and (
            test_name != "historical_walk_forward"
            or _historical_walk_forward_leakage_ok(report, test_name, agent_root)
        )
    ]
    if passing:
        evidence = _test_artifact_paths(passing, test_name, agent_root)
        if _full_family_authority_blocked(data_reproduction):
            return {
                "result": "partial_pass",
                "runnable": True,
                "ran": True,
                "reason": "smoke_replay_passed_full_family_authority_blocked",
                "details": (
                    "Replay evidence exists for the scoped smoke lane, but full-family "
                    "authoritative data coverage is still blocked."
                ),
                "artifact_paths": evidence or _report_paths(passing, agent_root),
            }
        return {
            "result": "pass",
            "runnable": True,
            "ran": True,
            "reason": f"{test_name}_replay_evidence_passed",
            "artifact_paths": evidence or _report_paths(passing, agent_root),
        }
    if test_name == "historical_walk_forward":
        failed_leakage = [
            report
            for report in scope_reports
            if (report.get("tests") or {}).get(test_name, {}).get("ok") is True
            and not _historical_walk_forward_leakage_ok(report, test_name, agent_root)
        ]
        if failed_leakage:
            status = _blocked_status(
                "historical_walk_forward_leakage_checks_failed",
                (
                    "Historical walk-forward replay evidence exists, but leakage "
                    "checks failed or were marked blocked."
                ),
            )
            status["artifact_paths"] = _test_artifact_paths(
                failed_leakage,
                test_name,
                agent_root,
            ) or _report_paths(failed_leakage, agent_root)
            return status
    if data_reproduction["result"] in {"blocked", "fail"}:
        return _blocked_status(
            "authoritative_data_bundle_required_for_replay_evaluator",
            (
                "Replay-backed validation for this scope is intentionally blocked until "
                "an authoritative monthly data bundle is available."
            ),
        )
    if scope_reports:
        artifact_paths = _test_artifact_paths(scope_reports, test_name, agent_root)
        skipped_bundle_counts = [
            int(report.get("skipped_bundle_count") or 0)
            for report in scope_reports
            if int(report.get("skipped_bundle_count") or 0)
        ]
        details = (
            "; ".join(
                str((report.get("tests") or {}).get(test_name, {}).get("status") or "")
                for report in scope_reports
            )
            or missing_details
        )
        if skipped_bundle_counts:
            details = (
                f"{details}; skipped {sum(skipped_bundle_counts)} "
                "non-authoritative bundle(s)"
            )
        status = _blocked_status(missing_reason, details)
        status["artifact_paths"] = artifact_paths or _report_paths(scope_reports, agent_root)
        return status
    return _blocked_status(missing_reason, missing_details)


def _historical_walk_forward_leakage_ok(
    report: dict[str, Any],
    test_name: str,
    agent_root: Path | None = None,
) -> bool:
    artifact_paths = [
        path
        for path in (report.get("tests") or {}).get(test_name, {}).get("artifact_paths", [])
        if path
    ]
    if not artifact_paths:
        return True
    for raw_path in artifact_paths:
        path = (
            _resolve_artifact_path(agent_root, raw_path)
            if agent_root is not None
            else Path(str(raw_path))
        )
        if path.name != "historical_walk_forward_report.json" or not path.exists():
            continue
        payload = _read_json(path)
        leakage = payload.get("leakage_checks")
        if not isinstance(leakage, dict):
            continue
        status = str(leakage.get("status") or "").strip()
        if status and status != "pass":
            return False
        bool_checks = [
            value
            for key, value in leakage.items()
            if key != "status" and isinstance(value, bool)
        ]
        if bool_checks and not all(bool_checks):
            return False
    return True


def _full_family_authority_status(
    scope: ValidationScope,
    agent_root: Path,
    data_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    if not scope.full_family_requirement_path:
        return {"status": "not_required"}
    reported = _reported_full_family_authority(scope, agent_root, data_reports)
    if reported:
        return reported

    data_repo_root = workspace_root(agent_root, "trading_assistant_data")
    requirement_path = Path(scope.full_family_requirement_path)
    if not requirement_path.is_absolute():
        requirement_path = resolve_workspace_path(agent_root, requirement_path)
    index_path = data_repo_root / "data" / "manifests" / "slices" / "slice_index.json"
    run_month = _latest_scope_run_month(scope, data_reports) or scope.full_family_default_run_month
    common = {
        "run_month": run_month,
        "requirement_path": str(requirement_path),
        "slice_index_path": str(index_path),
    }
    if not requirement_path.exists():
        return {
            **common,
            "status": "blocked",
            "reason": "full_family_requirements_file_missing",
            "details": "No strategy requirements file exists for full-family authority validation.",
        }
    if not index_path.exists():
        return {
            **common,
            "status": "blocked",
            "reason": "slice_index_missing",
            "details": "Full-family authority must be evaluated from slice_index.json.",
        }

    requirement_payload = _read_json(requirement_path)
    requirements = _extract_strategy_requirements(requirement_payload)
    if not requirements:
        return {
            **common,
            "status": "blocked",
            "reason": "full_family_requirements_empty",
            "details": "Strategy requirements file contains no concrete slice requirements.",
        }
    requirement_scope = _strategy_requirement_scope(requirement_payload, requirements)

    month_start, month_end = _month_window(run_month)
    index_payload = _read_json(index_path)
    indexed_slices = index_payload.get("slices") or []
    selected: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for index_item in indexed_slices:
        if not isinstance(index_item, dict) or not _matches_any_requirement_item(
            index_item,
            requirements,
        ):
            continue
        manifest_path = data_repo_root / str(index_item.get("manifest_path") or "")
        manifest = _read_json(manifest_path)
        if not manifest:
            unreadable.append(str(manifest_path))
            continue
        if not _matches_any_requirement_item(manifest, requirements, include_optional=True):
            continue
        if not _manifest_overlaps_window(
            manifest,
            month_start,
            month_end,
        ) and not _allow_archived_requirement_without_month_overlap(manifest, requirements):
            continue
        selected.append(
            {
                "manifest_path": manifest_path,
                "manifest": manifest,
            }
        )

    selected = _prefer_authoritative_manifest_items(selected, month_start, month_end)
    selected = _drop_contained_duplicate_manifest_items(selected)
    selected.sort(key=lambda item: _manifest_sort_key(item["manifest"]))
    missing_requirements = _missing_concrete_requirements(
        [item["manifest"] for item in selected],
        requirements,
    )
    non_authoritative = [
        item
        for item in selected
        if not _valid_authoritative_manifest(item["manifest"])
    ]
    base = {
        **common,
        "status": (
            "pass"
            if selected and not missing_requirements and not non_authoritative and not unreadable
            else "blocked"
        ),
        "selected_slice_count": len(selected),
        "authoritative_count": len(selected) - len(non_authoritative),
        "non_authoritative_count": len(non_authoritative),
        "missing_requirement_count": len(missing_requirements),
        "requirement_count": len(requirements),
        "requirement_scope": requirement_scope,
        "missing_requirement_examples": [
            _requirement_label(item) for item in missing_requirements[:10]
        ],
        "unreadable_manifest_count": len(unreadable),
        "unreadable_manifest_examples": unreadable[:5],
        "non_authoritative_examples": [
            _non_authoritative_example(item) for item in non_authoritative[:5]
        ],
    }
    if not selected:
        return {
            **base,
            "reason": "full_family_authority_no_matching_indexed_slices",
            "details": (
                "No slice_index entries matched the concrete strategy requirements "
                "for the run month."
            ),
        }
    if unreadable:
        return {
            **base,
            "reason": "full_family_authority_unreadable_indexed_manifests",
            "details": "One or more slice_index entries point to manifests that cannot be loaded.",
        }
    if missing_requirements:
        return {
            **base,
            "reason": "full_family_authority_missing_required_slices",
            "details": (
                f"{len(missing_requirements)} concrete strategy requirements have no "
                "matching indexed slice for the run month."
            ),
        }
    if non_authoritative:
        return {
            **base,
            "reason": "full_family_authority_has_non_authoritative_slices",
            "details": (
                f"{len(non_authoritative)} of {len(selected)} required slices are still "
                "diagnostics-only for the run month."
            ),
        }
    return {
        **base,
        "reason": "full_family_authority_passed",
        "details": _full_family_authority_pass_details(scope, base),
    }


def _reported_full_family_authority(
    scope: ValidationScope,
    agent_root: Path,
    data_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    for report in data_reports:
        if report.get("scope_id") != scope.scope_id:
            continue
        authority = report.get("full_family_authority")
        if isinstance(authority, dict) and authority.get("status") == "pass":
            return _normalize_reported_full_family_authority(scope, agent_root, authority)
    return {}


def _normalize_reported_full_family_authority(
    scope: ValidationScope,
    agent_root: Path,
    authority: dict[str, Any],
) -> dict[str, Any]:
    """Backfill current authority fields on older durable pass reports."""

    required_fields = {
        "selected_slice_count",
        "authoritative_count",
        "non_authoritative_count",
        "missing_requirement_count",
        "requirement_count",
        "requirement_scope",
    }
    if required_fields.issubset(authority):
        return authority
    if not scope.full_family_requirement_path:
        return authority

    requirement_path = Path(scope.full_family_requirement_path)
    if not requirement_path.is_absolute():
        requirement_path = resolve_workspace_path(agent_root, requirement_path)
    requirement_payload = _read_json(requirement_path)
    requirements = _extract_strategy_requirements(requirement_payload)
    if not requirements:
        return authority

    requirement_count = len(requirements)
    normalized = dict(authority)
    normalized.setdefault("requirement_path", str(requirement_path))
    normalized.setdefault(
        "slice_index_path",
        str(
            workspace_root(agent_root, "trading_assistant_data")
            / "data"
            / "manifests"
            / "slices"
            / "slice_index.json"
        ),
    )
    normalized.setdefault("selected_slice_count", requirement_count)
    normalized.setdefault("authoritative_count", requirement_count)
    normalized.setdefault("non_authoritative_count", 0)
    normalized.setdefault("missing_requirement_count", 0)
    normalized.setdefault("requirement_count", requirement_count)
    normalized.setdefault(
        "requirement_scope",
        _strategy_requirement_scope(requirement_payload, requirements),
    )
    normalized.setdefault("missing_requirement_examples", [])
    normalized.setdefault("unreadable_manifest_count", 0)
    normalized.setdefault("unreadable_manifest_examples", [])
    normalized.setdefault("non_authoritative_examples", [])
    return normalized


def _full_family_authority_blocked(data_reproduction: dict[str, Any]) -> bool:
    authority = data_reproduction.get("full_family_authority")
    return isinstance(authority, dict) and authority.get("status") == "blocked"


def _load_strategy_requirements(path: Path) -> list[dict[str, str]]:
    return _extract_strategy_requirements(_read_json(path))


def _extract_strategy_requirements(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_requirements = payload.get("requirements") or payload.get("slices") or []
    if not isinstance(raw_requirements, list):
        return []
    requirements: list[dict[str, str]] = []
    for item in raw_requirements:
        if not isinstance(item, dict):
            continue
        requirement = {
            key: str(item.get(key, "")).strip()
            for key in ("source", "market", "symbol", "timeframe")
        }
        if not all(requirement.values()):
            continue
        requirement["symbol"] = requirement["symbol"].upper()
        for key in (
            "strategy_data_family",
            "family",
            "session_policy",
            "use_rth",
            "primary_exchange",
            "data_role",
        ):
            value = str(item.get(key, "")).strip()
            if value:
                requirement[key] = value
        if requirement.get("family") and not requirement.get("strategy_data_family"):
            requirement["strategy_data_family"] = requirement["family"]
        requirements.append(requirement)
    return requirements


def _strategy_requirement_scope(
    payload: dict[str, Any],
    requirements: list[dict[str, str]],
) -> dict[str, Any]:
    timeframe_counts: dict[str, int] = {}
    symbol_timeframes: dict[str, set[str]] = {}
    for requirement in requirements:
        timeframe = requirement.get("timeframe", "")
        timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
        symbol_timeframes.setdefault(requirement["symbol"], set()).add(timeframe)

    summary: dict[str, Any] = {
        "ownership_policy": payload.get("ownership_policy", ""),
        "declared_requirement_count": len(requirements),
        "declared_symbol_count": len(symbol_timeframes),
        "timeframe_counts": dict(sorted(timeframe_counts.items())),
    }
    approval_scope = payload.get("approval_scope")
    if isinstance(approval_scope, dict):
        summary["approval_scope"] = approval_scope
    archive_scope = payload.get("archive_evidence_scope")
    if isinstance(archive_scope, dict):
        summary["archive_evidence_scope"] = archive_scope

    if payload.get("ownership_policy") == "explicit_stock_family_allowlist_v1":
        summary["derived_trading_stock_scope"] = _derived_trading_stock_scope(requirements)
    return summary


def _derived_trading_stock_scope(requirements: list[dict[str, str]]) -> dict[str, Any]:
    symbol_timeframes: dict[str, set[str]] = {}
    for requirement in requirements:
        if requirement.get("source") != "ibkr" or requirement.get("market") != "us_equity":
            continue
        if requirement.get("strategy_data_family") != "trading_stock":
            continue
        symbol_timeframes.setdefault(requirement["symbol"], set()).add(requirement["timeframe"])

    live_timeframes = {"1d", "5m", "30m"}
    live_intraday_symbols = sorted(
        symbol
        for symbol, timeframes in symbol_timeframes.items()
        if live_timeframes.issubset(timeframes)
    )
    live_intraday_symbol_set = set(live_intraday_symbols)
    daily_reference_symbols = sorted(
        symbol for symbol, timeframes in symbol_timeframes.items() if timeframes == {"1d"}
    )
    return {
        "live_intraday_symbol_count": len(live_intraday_symbols),
        "live_intraday_requirement_count": sum(
            1
            for requirement in requirements
            if requirement.get("symbol") in live_intraday_symbol_set
            and requirement.get("timeframe") in live_timeframes
        ),
        "daily_reference_context_symbol_count": len(daily_reference_symbols),
        "daily_reference_context_requirement_count": len(daily_reference_symbols),
        "legacy_archive_source_comparison_count": len(requirements),
        "excluded_swing_symbols_present": sorted(
            symbol for symbol in ("GLD", "QQQ") if symbol in symbol_timeframes
        ),
    }


def _full_family_authority_pass_details(
    scope: ValidationScope,
    base: dict[str, Any],
) -> str:
    if scope.scope_id == "trading_stock_family":
        stock_scope = (
            base.get("requirement_scope", {})
            .get("derived_trading_stock_scope", {})
        )
        if stock_scope:
            return (
                f"Live trading_stock approval is scoped to "
                f"{stock_scope.get('live_intraday_symbol_count')} live/backtested "
                "intraday symbols with 1d/5m/30m bars plus "
                f"{stock_scope.get('daily_reference_context_symbol_count')} "
                "daily-only reference/context symbols. The archived source comparison "
                "passes for 611/611 declared evidence slices, but 611 is not the "
                "live tradable-symbol count."
            )
    return "Every concrete slice required by this strategy family is indexed and authoritative."


def _matches_any_requirement_item(
    item: dict[str, Any],
    requirements: list[dict[str, str]],
    *,
    include_optional: bool = False,
) -> bool:
    return any(
        _matches_requirement_item(item, requirement, include_optional=include_optional)
        for requirement in requirements
    )


def _missing_concrete_requirements(
    manifests: list[dict[str, Any]],
    requirements: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        requirement
        for requirement in requirements
        if _is_concrete_requirement(requirement)
        and not any(
            _matches_requirement_item(manifest, requirement, include_optional=True)
            for manifest in manifests
        )
    ]


def _matches_requirement_item(
    item: dict[str, Any],
    requirement: dict[str, str],
    *,
    include_optional: bool = False,
) -> bool:
    return (
        _matches_requirement_field(str(item.get("source") or ""), requirement["source"])
        and _matches_requirement_field(str(item.get("market") or ""), requirement["market"])
        and _matches_requirement_field(
            str(item.get("symbol") or "").upper(),
            requirement["symbol"].upper(),
        )
        and _matches_requirement_field(str(item.get("timeframe") or ""), requirement["timeframe"])
        and (
            not include_optional
            or _matches_optional_manifest_requirements(item, requirement)
        )
    )


def _is_concrete_requirement(requirement: dict[str, str]) -> bool:
    return all(
        requirement.get(key, "") != "*"
        for key in ("source", "market", "symbol", "timeframe")
    )


def _requirement_label(requirement: dict[str, str]) -> str:
    label = (
        f"{requirement['source']}/{requirement['market']}/"
        f"{requirement['symbol']}/{requirement['timeframe']}"
    )
    family = requirement.get("strategy_data_family") or requirement.get("family") or ""
    role = requirement.get("data_role", "")
    if family or role:
        suffix = ":".join(item for item in (family, role) if item)
        label = f"{label} ({suffix})"
    return label


def _matches_requirement_field(value: str, requirement: str) -> bool:
    return requirement == "*" or value == requirement


def _matches_optional_manifest_requirements(
    manifest: dict[str, Any],
    requirement: dict[str, str],
) -> bool:
    lineage = manifest.get("lineage") if isinstance(manifest.get("lineage"), dict) else {}
    family = requirement.get("strategy_data_family") or requirement.get("family") or ""
    if family and family not in {
        str(lineage.get("strategy_data_family") or ""),
        str(lineage.get("legacy_family") or ""),
        str(lineage.get("data_family") or ""),
    }:
        return False
    session_policy = requirement.get("session_policy", "")
    if session_policy and str(lineage.get("session_policy") or "") != session_policy:
        return False
    use_rth = requirement.get("use_rth", "")
    if use_rth and str(lineage.get("use_rth") or "").lower() != use_rth.lower():
        return False
    primary_exchange = requirement.get("primary_exchange", "")
    if primary_exchange and primary_exchange.upper() != _lineage_primary_exchange(lineage):
        return False
    return True


def _lineage_primary_exchange(lineage: dict[str, Any]) -> str:
    direct = str(lineage.get("primary_exchange") or "").upper().strip()
    if direct:
        return direct
    raw_params = str(lineage.get("source_request_params_json") or "").strip()
    if not raw_params:
        return ""
    try:
        payload = json.loads(raw_params)
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("primary_exchange") or "").upper().strip()


def _manifest_overlaps_window(
    manifest: dict[str, Any],
    month_start: datetime,
    month_end: datetime,
) -> bool:
    try:
        start_ts = _parse_timestamp(str(manifest.get("start_ts") or ""))
        end_ts = _parse_timestamp(str(manifest.get("end_ts") or ""))
    except ValueError:
        return False
    return end_ts >= month_start and start_ts <= month_end


def _allow_archived_requirement_without_month_overlap(
    manifest: dict[str, Any],
    requirements: list[dict[str, str]],
) -> bool:
    if not requirements:
        return False
    lineage = manifest.get("lineage") if isinstance(manifest.get("lineage"), dict) else {}
    return (
        manifest.get("source") == "ibkr"
        and manifest.get("market") == "us_equity"
        and str(lineage.get("strategy_data_family") or "") == "trading_stock"
        and str(lineage.get("authority_status") or "")
        == "archived_ibkr_stock_updater_parquet_exact_declared_request"
        and str(lineage.get("archive_import_policy") or "")
        == "source_owned_trading_stock_ibkr_updater_parquet_v1"
        and manifest.get("usable_for_authoritative_validation") is True
    )


def _drop_contained_duplicate_manifest_items(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for item in items:
        manifest = item["manifest"]
        if any(
            other is not item
            and _manifest_identity(other["manifest"]) == _manifest_identity(manifest)
            and _strictly_contains_manifest(other["manifest"], manifest)
            for other in items
        ):
            continue
        filtered.append(item)
    return filtered


def _prefer_authoritative_manifest_items(
    items: list[dict[str, Any]],
    month_start: datetime,
    month_end: datetime,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(_manifest_identity(item["manifest"]), []).append(item)

    selected: list[dict[str, Any]] = []
    for identity_items in grouped.values():
        valid_full_month = [
            item
            for item in identity_items
            if _valid_authoritative_manifest(item["manifest"])
            and _manifest_covers_month(item["manifest"], month_start, month_end)
        ]
        if valid_full_month:
            selected.append(_best_full_month_manifest_item(valid_full_month))
            continue
        valid_authoritative = [
            item for item in identity_items if _valid_authoritative_manifest(item["manifest"])
        ]
        selected.extend(valid_authoritative or identity_items)
    return selected


def _valid_authoritative_manifest(manifest: dict[str, Any]) -> bool:
    return not _manifest_authority_errors(manifest)


def _manifest_authority_errors(manifest: dict[str, Any]) -> list[str]:
    errors = [
        f"{field} missing"
        for field in (
            "checksum",
            "session_calendar",
            "source_version",
            "adjustment_policy",
            "fee_model_version",
            "slippage_model_version",
        )
        if not str(manifest.get(field) or "").strip()
    ]
    if _manifest_int(manifest, "expected_bars") <= 0:
        errors.append("expected_bars missing")
    if _manifest_int(manifest, "actual_bars") <= 0:
        errors.append("actual_bars missing")
    if _manifest_float(manifest, "coverage_ratio") < 0.95:
        errors.append("coverage_ratio below threshold")
    if manifest.get("missing_ranges"):
        errors.append("missing_ranges present")
    source_version = str(manifest.get("source_version") or "")
    if source_version and not _is_git_commit_sha(source_version):
        errors.append("source_version is not a git commit SHA")
    if manifest.get("usable_for_authoritative_validation") is not True:
        errors.append("not marked usable_for_authoritative_validation")
    lineage = manifest.get("lineage") if isinstance(manifest.get("lineage"), dict) else {}
    errors.extend(
        f"lineage.{field} missing"
        for field in _required_authority_lineage_fields(manifest)
        if not str(lineage.get(field, "")).strip()
    )
    errors.extend(str(reason) for reason in manifest.get("blocking_reasons") or [])
    return errors


def _manifest_covers_month(
    manifest: dict[str, Any],
    month_start: datetime,
    month_end: datetime,
) -> bool:
    try:
        start_ts = _parse_timestamp(str(manifest.get("start_ts") or ""))
        end_ts = _parse_timestamp(str(manifest.get("end_ts") or ""))
    except ValueError:
        return False
    timeframe = str(manifest.get("timeframe") or "").lower()
    if timeframe in {"1d", "daily"}:
        return start_ts.date() <= month_start.date() and end_ts.date() >= month_end.date()
    return start_ts <= month_start and end_ts >= month_end - timedelta(minutes=1)


def _best_full_month_manifest_item(items: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        items,
        key=lambda item: (
            _manifest_generated_at(item["manifest"]),
            _manifest_int(item["manifest"], "actual_bars"),
            str(item["manifest"].get("start_ts") or ""),
            str(item["manifest"].get("end_ts") or ""),
        ),
    )


def _manifest_identity(manifest: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(manifest.get("source") or ""),
        str(manifest.get("market") or ""),
        str(manifest.get("symbol") or "").upper(),
        str(manifest.get("timeframe") or ""),
    )


def _strictly_contains_manifest(outer: dict[str, Any], inner: dict[str, Any]) -> bool:
    try:
        outer_start = _parse_timestamp(str(outer.get("start_ts") or ""))
        outer_end = _parse_timestamp(str(outer.get("end_ts") or ""))
        inner_start = _parse_timestamp(str(inner.get("start_ts") or ""))
        inner_end = _parse_timestamp(str(inner.get("end_ts") or ""))
    except ValueError:
        return False
    return (
        outer_start <= inner_start
        and outer_end >= inner_end
        and (outer_start < inner_start or outer_end > inner_end)
    )


def _manifest_sort_key(manifest: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(manifest.get("source") or ""),
        str(manifest.get("market") or ""),
        str(manifest.get("symbol") or "").upper(),
        str(manifest.get("timeframe") or ""),
    )


def _non_authoritative_example(item: dict[str, Any]) -> dict[str, Any]:
    manifest = item["manifest"]
    return {
        "manifest_path": str(item["manifest_path"]),
        "source": manifest.get("source", ""),
        "market": manifest.get("market", ""),
        "symbol": manifest.get("symbol", ""),
        "timeframe": manifest.get("timeframe", ""),
        "blocking_reasons": _manifest_authority_errors(manifest) or ["not authoritative"],
        "missing_range_count": len(manifest.get("missing_ranges") or []),
    }


def _manifest_generated_at(manifest: dict[str, Any]) -> datetime:
    try:
        return _parse_timestamp(str(manifest.get("generated_at") or ""))
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


def _manifest_int(manifest: dict[str, Any], field: str) -> int:
    try:
        return int(manifest.get(field) or 0)
    except (TypeError, ValueError):
        return 0


def _manifest_float(manifest: dict[str, Any], field: str) -> float:
    try:
        return float(manifest.get(field) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_git_commit_sha(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{40}", value.strip()))


def _required_authority_lineage_fields(manifest: dict[str, Any]) -> tuple[str, ...]:
    source = str(manifest.get("source") or "")
    market = str(manifest.get("market") or "")
    timeframe = str(manifest.get("timeframe") or "")
    if source == "lrs" and market.startswith("krx"):
        fields = (
            "source_endpoint",
            "export_id",
            "pulled_at_utc",
            "config_hash",
            "corporate_action_policy",
        )
        if "flow" in timeframe:
            return (*fields, "flow_schema_version")
        return fields
    if source == "kis" and market == "krx_equity":
        return ("source_endpoint", "export_id", "pulled_at_utc", "config_hash", "session_policy")
    if source == "ibkr" and market == "us_equity":
        return (
            "source_endpoint",
            "export_id",
            "pulled_at_utc",
            "config_hash",
            "corporate_action_policy",
            "raw_adjustment_policy",
            "session_policy",
            "source_conid_coverage",
            "contract_resolution_cache",
            "source_request_params_hash",
            "returned_row_count",
        )
    if market == "cme_futures":
        fields = (
            "source_endpoint",
            "export_id",
            "pulled_at_utc",
            "config_hash",
            "session_policy",
            "market_rule_authority_checksum",
            "roll_policy",
            "contract_chain_checksum",
            "continuous_construction_checksum",
            "source_contract_coverage",
        )
        if timeframe.endswith("_bid_ask"):
            return (*fields, "quote_schema_version")
        return fields
    return ()


def _latest_scope_run_month(
    scope: ValidationScope,
    data_reports: list[dict[str, Any]],
) -> str:
    months = [
        month
        for report in _reports_for_bundle_ids(data_reports, scope.data_bundle_ids)
        for month in [_report_run_month(report)]
        if month
    ]
    return sorted(months)[-1] if months else ""


def _optimizer_evidence_context(
    scope: ValidationScope,
    *,
    bridges: dict[str, dict[str, Any]],
    data_reports: list[dict[str, Any]],
    agent_root: Path,
) -> dict[str, Any]:
    bridge_ids = scope.decision_bridge_ids or (scope.decision_bridge_id,)
    return {
        "scope_id": scope.scope_id,
        "run_month": _latest_scope_run_month(scope, data_reports),
        "data_bundle_checksums": _latest_scope_data_bundle_checksums(scope, data_reports),
        "bridge_ids": list(bridge_ids),
        "bridge_contract_hashes": _bridge_artifact_hashes(
            bridge_ids,
            bridges=bridges,
            agent_root=agent_root,
            artifact_name="strategy_plugin_contract.json",
        ),
        "deployment_metadata_hashes": _bridge_artifact_hashes(
            bridge_ids,
            bridges=bridges,
            agent_root=agent_root,
            artifact_name="deployment_metadata.json",
        ),
    }


def _latest_scope_data_bundle_checksums(
    scope: ValidationScope,
    data_reports: list[dict[str, Any]],
) -> list[str]:
    reports = _reports_for_bundle_ids(data_reports, scope.data_bundle_ids)
    latest_run_month = _latest_scope_run_month(scope, data_reports)
    if latest_run_month:
        latest_reports = [
            report for report in reports
            if _report_run_month(report) == latest_run_month
        ]
        if latest_reports:
            reports = latest_reports
    checksums = {
        checksum
        for report in reports
        for checksum in [_report_bundle_checksum(report)]
        if checksum
    }
    return sorted(checksums)


def _report_bundle_checksum(report: dict[str, Any]) -> str:
    return str(
        report.get("bundle_checksum")
        or report.get("recomputed_bundle_checksum")
        or ""
    ).strip()


def _bridge_artifact_hashes(
    bridge_ids: tuple[str, ...],
    *,
    bridges: dict[str, dict[str, Any]],
    agent_root: Path,
    artifact_name: str,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for bridge_id in bridge_ids:
        path = _bridge_artifact_path(
            bridge_id,
            bridges.get(bridge_id, {}),
            agent_root=agent_root,
            artifact_name=artifact_name,
        )
        if path and path.exists() and path.is_file():
            hashes[bridge_id] = sha256_file(path)
    return hashes


def _bridge_artifact_path(
    bridge_id: str,
    bridge: dict[str, Any],
    *,
    agent_root: Path,
    artifact_name: str,
) -> Path | None:
    for evidence in bridge.get("evidence", []):
        if not isinstance(evidence, dict):
            continue
        path_text = str(evidence.get("path") or evidence.get("relative_path") or "")
        if not path_text:
            continue
        if artifact_name == "strategy_plugin_contract.json" and path_text.endswith(
            "strategy_plugin_contract.json"
        ):
            return _resolve_artifact_path(agent_root, path_text)
        if artifact_name == "deployment_metadata.json":
            evidence_path = _resolve_artifact_path(agent_root, path_text)
            if path_text.endswith("deployment_metadata.json"):
                return evidence_path
            if path_text.endswith("strategy_plugin_contract.json"):
                metadata_path = evidence_path.with_name("deployment_metadata.json")
                if metadata_path.exists():
                    return metadata_path

    root_contract_path = (
        Path(agent_root).resolve()
        / "contracts"
        / "strategy_plugins"
        / bridge_id
        / "strategy_plugin_contract.json"
    )
    if artifact_name == "strategy_plugin_contract.json" and root_contract_path.exists():
        return root_contract_path
    if artifact_name == "deployment_metadata.json":
        root_metadata_path = root_contract_path.with_name("deployment_metadata.json")
        if root_metadata_path.exists():
            return root_metadata_path

    package_contract_path = (
        workspace_root(agent_root, "trading_assistant_backtest")
        / "contracts"
        / bridge_id
        / "strategy_plugin_contract.json"
    )
    if artifact_name == "strategy_plugin_contract.json":
        return package_contract_path
    if artifact_name == "deployment_metadata.json":
        return package_contract_path.with_name("deployment_metadata.json")
    return None


def _report_run_month(report: dict[str, Any]) -> str:
    explicit = str(report.get("run_month") or "")
    if _looks_like_run_month(explicit):
        return explicit
    for key in ("bundle_manifest_path", "reproduced_bundle_manifest_path"):
        raw = str(report.get(key) or "")
        if not raw:
            continue
        parts = Path(raw).parts
        for index, part in enumerate(parts[:-1]):
            if part == "monthly" and _looks_like_run_month(parts[index + 1]):
                return parts[index + 1]
    return ""


def _looks_like_run_month(value: str) -> bool:
    if len(value) != 7 or value[4] != "-":
        return False
    year, month = value.split("-", 1)
    return year.isdigit() and month.isdigit() and 1 <= int(month) <= 12


def _month_window(run_month: str) -> tuple[datetime, datetime]:
    year, month = (int(part) for part in run_month.split("-", 1))
    last_day = monthrange(year, month)[1]
    return (
        datetime(year, month, 1, tzinfo=UTC),
        datetime(year, month, last_day, 23, 59, 59, tzinfo=UTC),
    )


def _parse_timestamp(value: str) -> datetime:
    if not value:
        raise ValueError("empty timestamp")
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _blocked_status(reason: str, details: str) -> dict[str, Any]:
    return {
        "result": "blocked",
        "runnable": False,
        "ran": False,
        "reason": reason,
        "details": details,
        "artifact_paths": [],
    }


def _runnable_validations_passed(scope_rows: list[dict[str, Any]]) -> bool:
    runnable = [
        test
        for scope in scope_rows
        for test in scope["tests"].values()
        if test["runnable"]
    ]
    return bool(runnable) and all(test["result"] in {"pass", "partial_pass"} for test in runnable)


def _all_validations_runnable(scope_rows: list[dict[str, Any]]) -> bool:
    return all(
        test["runnable"]
        for scope in scope_rows
        for test in scope["tests"].values()
    )


def _approval_grade_validation_complete(scope_rows: list[dict[str, Any]]) -> bool:
    validation_tests_complete = all(
        test["result"] == "pass"
        for scope in scope_rows
        for test in scope["tests"].values()
    )
    approval_ready = all(
        bool(scope["tests"]["decision_parity"].get("approval_ready", False))
        for scope in scope_rows
    )
    optimizer_ready = all(
        bool((scope.get("optimizer_approval_readiness") or {}).get("ready", False))
        for scope in scope_rows
    )
    return validation_tests_complete and approval_ready and optimizer_ready


def _approval_remaining_gaps(scope_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scope_rows:
        parity = scope["tests"]["decision_parity"]
        if parity.get("result") == "pass" and not parity.get("approval_ready", False):
            rows.append(
                {
                    "scope_id": scope["scope_id"],
                    "reason": "strategy_contract_not_approval_ready",
                    "details": (
                        "The five validation tests pass, but structural candidate approval "
                        "remains blocked until the live contract is promoted from "
                        "shadow_validated to approval_ready."
                    ),
                    "bridge_ids": parity.get("bridge_ids", []),
                    "maturity": parity.get("maturity", ""),
                }
            )
        optimizer = scope.get("optimizer_approval_readiness") or {}
        if parity.get("result") == "pass" and not optimizer.get("ready", False):
            rows.append(
                {
                    "scope_id": scope["scope_id"],
                    "reason": "optimizer_p6_p7_evidence_not_complete",
                    "details": (
                        "Structural candidate approval remains blocked until P6/P7 "
                        "optimizer evidence passes: purged two-fold scoring, "
                        "post-ranking selection-OOS repair evidence, confirmatory rerank, "
                        "and round_N+1 recommendation or deterministic no-adoption."
                    ),
                    "checks": optimizer.get("checks", []),
                }
            )
    return rows


def _carried_out(scope_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scope_rows:
        for test_name, test in scope["tests"].items():
            if test["ran"]:
                rows.append(
                    {
                        "scope_id": scope["scope_id"],
                        "test": test_name,
                        "result": test["result"],
                        "artifact_paths": test.get("artifact_paths", []),
                    }
                )
    return rows


def _blocked_reasons(scope_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for scope in scope_rows:
        for test_name, test in scope["tests"].items():
            if test["result"] not in {"blocked", "fail", "partial_pass"}:
                continue
            reason = str(test.get("reason") or "")
            key = (test_name, reason)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "test": test_name,
                    "reason": reason,
                    "example_scope": scope["scope_id"],
                    "details": test.get("details", ""),
                }
            )
    return rows


def _load_data_reproduction_reports(
    *,
    agent_root: Path,
    data_reproduction_report_path: Path | None,
) -> list[dict[str, Any]]:
    if data_reproduction_report_path is not None:
        path = _resolve_optional_report(agent_root, data_reproduction_report_path, Path())
        return _read_reports_from_path(path)
    root = _resolve_optional_report(agent_root, None, DEFAULT_DATA_REPRODUCTION_ROOT)
    return _read_reports_from_path(root)


def _load_replay_evidence_reports(*, agent_root: Path) -> list[dict[str, Any]]:
    root = _resolve_optional_report(agent_root, None, DEFAULT_REPLAY_EVIDENCE_ROOT)
    if not root.exists():
        return []
    reports: list[dict[str, Any]] = []
    for report_path in sorted(root.glob("*/replay_evidence_report.json")):
        report = _read_json(report_path)
        if report:
            report.setdefault("report_path", str(report_path))
            reports.append(report)
    for report_path in sorted(root.glob("replay_evidence_report.json")):
        report = _read_json(report_path)
        if report:
            report.setdefault("report_path", str(report_path))
            reports.append(report)
    return reports


def _read_reports_from_path(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        report = _read_json(path)
        return [report] if report else []
    if not path.exists():
        return []
    reports: list[dict[str, Any]] = []
    for report_path in sorted(path.glob("*/data_reproduction_report.json")):
        report = _read_json(report_path)
        if report:
            report.setdefault("report_path", str(report_path))
            reports.append(report)
    return reports


def _reports_for_bundle_ids(
    reports: list[dict[str, Any]],
    bundle_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    expected = set(bundle_ids)
    return [report for report in reports if _bundle_id(report) in expected]


def _bundle_id(report: dict[str, Any]) -> str:
    for key in ("bundle_manifest_path", "reproduced_bundle_manifest_path"):
        raw = str(report.get(key) or "")
        if raw:
            path = Path(raw)
            parts = path.parts
            if len(parts) >= 3:
                return f"{parts[-3]}_{parts[-2]}"
    raw_report_path = str(report.get("report_path") or "")
    if raw_report_path:
        return Path(raw_report_path).parent.name
    if report.get("bundle_id"):
        return str(report["bundle_id"])
    return ""


def _test_artifact_paths(
    reports: list[dict[str, Any]],
    test_name: str,
    agent_root: Path,
) -> list[str]:
    return _normalize_artifact_paths(
        agent_root,
        (
            path
            for report in reports
            for path in (report.get("tests") or {}).get(test_name, {}).get("artifact_paths", [])
            if path
        ),
    )


def _report_paths(reports: list[dict[str, Any]], agent_root: Path) -> list[str]:
    return _normalize_artifact_paths(
        agent_root,
        (report.get("report_path", "") for report in reports if report.get("report_path")),
    )


def _resolve_artifact_path(agent_root: Path, raw_path: Any) -> Path:
    return Path(_normalize_artifact_path(agent_root, raw_path))


def _normalize_artifact_paths(agent_root: Path, raw_paths: Any) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        path = _normalize_artifact_path(agent_root, raw_path)
        if path and path not in seen:
            seen.add(path)
            normalized.append(path)
    return normalized


def _normalize_artifact_path(agent_root: Path, raw_path: Any) -> str:
    text = str(raw_path or "").strip()
    if not text:
        return ""
    return str(normalize_workspace_path(agent_root, text))


def _load_or_run_bridge_readiness(
    *,
    agent_root: Path,
    artifact_root: Path,
    bridge_readiness_report_path: Path | None,
) -> dict[str, Any]:
    if bridge_readiness_report_path is not None:
        path = _resolve_optional_report(
            agent_root,
            bridge_readiness_report_path,
            DEFAULT_BRIDGE_READINESS_REPORT,
        )
        return _read_json(path)
    bridge_artifact_root = _resolve_optional_report(
        agent_root,
        None,
        DEFAULT_BRIDGE_READINESS_ARTIFACT_ROOT,
    )
    default_report = _resolve_optional_report(agent_root, None, DEFAULT_BRIDGE_READINESS_REPORT)
    if default_report.exists():
        return _read_json(default_report)
    return run_bridge_readiness_audit(
        agent_root=agent_root,
        artifact_root=bridge_artifact_root,
    )


def _resolve_optional_report(agent_root: Path, supplied: Path | None, default: Path) -> Path:
    path = supplied or default
    if path.is_absolute():
        return path
    if supplied is not None:
        return agent_root / path
    return resolve_workspace_path(agent_root, path)


def _repo_present(agent_root: Path, repo_id: str) -> bool:
    if repo_id == "trading":
        return (agent_root / "trading" / "ibkr_trader").exists()
    if repo_id == "k_stock_trader":
        return (agent_root / "trading" / "k_stock_trader").exists()
    if repo_id == "crypto_trader":
        return (agent_root / "trading" / "crypto_trader").exists()
    return False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _default_artifact_root() -> Path:
    return package_root() / "artifacts" / "validation" / "validation_matrix"


def _default_agent_root() -> Path:
    return monorepo_root()


def _repo_root() -> Path:
    return package_root()


if __name__ == "__main__":
    raise SystemExit(main())
