"""Guarded promotion from shadow_validated to approval_ready."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.paths import (
    monorepo_root,
    normalize_workspace_path,
    resolve_workspace_path,
)
from trading_assistant_backtest.validation.approval_evidence_spine import (
    PILOT_SCOPE_ID,
    run_approval_evidence_spine,
)
from trading_assistant_backtest.validation.approval_grade_audit import (
    CONTRACT_PATHS,
    run_approval_grade_audit,
)
from trading_assistant_backtest.validation.bridge_readiness import run_bridge_readiness_audit
from trading_assistant_backtest.validation.validation_matrix import run_validation_matrix_audit

SCHEMA_VERSION = "bridge_promotion_report_v1"
PRE_PROMOTION_SCHEMA_VERSION = "bridge_pre_promotion_report_v1"
POST_PROMOTION_SCHEMA_VERSION = "bridge_post_promotion_report_v1"
TARGET_MATURITY = "approval_ready"
SOURCE_MATURITY = "shadow_validated"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote a strategy bridge through an eligible approval evidence bundle."
    )
    parser.add_argument("--agent-root", type=Path, default=monorepo_root())
    parser.add_argument(
        "--scope",
        "--bridge-id",
        dest="scope",
        default=PILOT_SCOPE_ID,
        choices=sorted(CONTRACT_PATHS),
    )
    parser.add_argument(
        "--approval-evidence",
        "--approval-evidence-bundle",
        dest="approval_evidence",
        type=Path,
        default=None,
    )
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument(
        "--refresh-evidence",
        action="store_true",
        help="Regenerate the approval evidence bundle before evaluating promotion.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Mutate the strategy plugin contract maturity after all checks pass.",
    )
    args = parser.parse_args(argv)

    report = promote_bridge(
        agent_root=args.agent_root,
        scope_id=args.scope,
        approval_evidence_path=args.approval_evidence,
        artifact_root=args.artifact_root,
        refresh_evidence=args.refresh_evidence,
        write=args.write,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.write and not report["promoted"]:
        return 1
    return 0


def promote_bridge(
    *,
    agent_root: Path,
    scope_id: str = PILOT_SCOPE_ID,
    approval_evidence_path: Path | None = None,
    artifact_root: Path | None = None,
    refresh_evidence: bool = False,
    write: bool = False,
) -> dict[str, Any]:
    agent_root = Path(agent_root).resolve()
    artifact_root = _artifact_root(agent_root, scope_id, artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    contract_path = resolve_workspace_path(
        agent_root,
        CONTRACT_PATHS[scope_id],
    ) / "strategy_plugin_contract.json"

    bundle = _load_or_refresh_bundle(
        agent_root=agent_root,
        scope_id=scope_id,
        approval_evidence_path=approval_evidence_path,
        refresh_evidence=refresh_evidence,
    )
    pre_contract = _read_json(contract_path)
    pre_checks = _pre_promotion_checks(
        agent_root=agent_root,
        scope_id=scope_id,
        contract_path=contract_path,
        contract=pre_contract,
        bundle=bundle,
    )
    eligible = all(check["passed"] for check in pre_checks)
    dry_run = not write
    pre_report_path = artifact_root / "pre_promotion_report.json"
    pre_contract_hash = sha256_file(contract_path, missing_ok=True)
    pre_report = {
        "schema_version": PRE_PROMOTION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scope_id": scope_id,
        "contract_path": _display_path(agent_root, contract_path),
        "approval_evidence_path": str(bundle.get("artifact_path") or approval_evidence_path or ""),
        "write_requested": write,
        "dry_run": dry_run,
        "eligible": eligible,
        "checks": pre_checks,
        "contract_hash_before": pre_contract_hash,
    }
    pre_report_path.write_text(
        json.dumps(pre_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    promoted = False
    post_report: dict[str, Any] = {}
    if eligible and write:
        post_contract = deepcopy(pre_contract)
        post_contract["maturity"] = TARGET_MATURITY
        _write_contract(contract_path, post_contract)
        promoted = True
        post_report = _post_promotion_report(
            agent_root=agent_root,
            scope_id=scope_id,
            artifact_root=artifact_root,
            contract_path=contract_path,
            pre_contract=pre_contract,
            post_contract=post_contract,
            pre_contract_hash=pre_contract_hash,
        )
    elif write:
        post_report = {
            "schema_version": POST_PROMOTION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "scope_id": scope_id,
            "promoted": False,
            "post_validation_skipped": True,
            "reason": "pre-promotion checks failed",
        }
        (artifact_root / "post_promotion_report.json").write_text(
            json.dumps(post_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    report_path = artifact_root / "bridge_promotion_report.json"
    blockers = [
        f"{check['name']}: {error}"
        for check in pre_checks
        if not check["passed"]
        for error in check["errors"]
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "scope_id": scope_id,
        "contract_path": _display_path(agent_root, contract_path),
        "approval_evidence_path": str(bundle.get("artifact_path") or approval_evidence_path or ""),
        "write_requested": write,
        "dry_run": dry_run,
        "eligible": eligible,
        "promoted": promoted,
        "promotion_decision": (
            "promoted"
            if promoted
            else "eligible_dry_run"
            if eligible
            else "blocked"
        ),
        "blockers": blockers,
        "pre_promotion_report": str(pre_report_path),
        "post_promotion_report": str(artifact_root / "post_promotion_report.json")
        if post_report
        else "",
        "artifact_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _pre_promotion_checks(
    *,
    agent_root: Path,
    scope_id: str,
    contract_path: Path,
    contract: dict[str, Any],
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        _check(
            "approval_evidence_schema",
            bundle.get("schema_version") == "approval_evidence_bundle_v1",
            ["approval evidence bundle missing or malformed"],
        ),
        _check(
            "approval_evidence_scope",
            str(bundle.get("scope_id") or "") == scope_id,
            [f"approval evidence scope does not match {scope_id}"],
        ),
        _check(
            "approval_evidence_eligible",
            bool(bundle.get("eligible_for_promotion")) is True and not bundle.get("blockers"),
            [str(item) for item in bundle.get("blockers") or ["bundle is not eligible"]],
        ),
        _check(
            "contract_present",
            bool(contract),
            [f"contract missing or malformed: {_display_path(agent_root, contract_path)}"],
        ),
        _check(
            "contract_source_maturity",
            str(contract.get("maturity") or "") == SOURCE_MATURITY,
            [f"contract maturity must be {SOURCE_MATURITY} before guarded promotion"],
        ),
        _evidence_hashes_check(agent_root, bundle),
        _contract_hash_bound_check(agent_root, contract_path, bundle),
    ]
    return checks


def _post_promotion_report(
    *,
    agent_root: Path,
    scope_id: str,
    artifact_root: Path,
    contract_path: Path,
    pre_contract: dict[str, Any],
    post_contract: dict[str, Any],
    pre_contract_hash: str,
) -> dict[str, Any]:
    bridge_root = artifact_root / "post_bridge_readiness"
    bridge = run_bridge_readiness_audit(agent_root=agent_root, artifact_root=bridge_root)
    bridge_path = Path(
        str(bridge.get("artifact_path") or bridge_root / "bridge_readiness_report.json")
    )
    matrix_root = artifact_root / "post_validation_matrix"
    matrix = run_validation_matrix_audit(
        agent_root=agent_root,
        artifact_root=matrix_root,
        bridge_readiness_report_path=bridge_path,
    )
    audit_root = artifact_root / "post_approval_grade_audit"
    audit = run_approval_grade_audit(
        agent_root=agent_root,
        artifact_root=audit_root,
        scope_ids=(scope_id,),
        refresh=True,
    )
    post_report_path = artifact_root / "post_promotion_report.json"
    report = {
        "schema_version": POST_PROMOTION_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "scope_id": scope_id,
        "promoted": True,
        "contract_path": _display_path(agent_root, contract_path),
        "contract_hash_before": pre_contract_hash,
        "contract_hash_after": sha256_file(contract_path),
        "mutation_scope_ok": _only_maturity_changed(pre_contract, post_contract),
        "bridge_readiness_report": str(bridge.get("artifact_path") or ""),
        "validation_matrix_report": str(matrix.get("artifact_path") or ""),
        "approval_grade_audit_report": str(audit.get("artifact_path") or ""),
        "post_validation_ok": (
            bool(bridge.get("ok"))
            and bool(matrix.get("ok"))
            and bool(audit.get("ok"))
        ),
    }
    post_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _evidence_hashes_check(agent_root: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    evidence_hashes = bundle.get("evidence_hashes")
    if not isinstance(evidence_hashes, dict) or not evidence_hashes:
        errors.append("approval evidence bundle has no evidence_hashes")
        return _check("evidence_hashes_match", False, errors)
    for raw_path, expected in evidence_hashes.items():
        path = normalize_workspace_path(agent_root, raw_path)
        if not path.exists() or not path.is_file():
            errors.append(f"evidence path missing: {raw_path}")
            continue
        actual = sha256_file(path)
        if actual != str(expected):
            errors.append(f"evidence hash mismatch: {raw_path}")
    return _check("evidence_hashes_match", not errors, errors)


def _contract_hash_bound_check(
    agent_root: Path,
    contract_path: Path,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    evidence_hashes = bundle.get("evidence_hashes") if isinstance(bundle, dict) else {}
    if not isinstance(evidence_hashes, dict):
        evidence_hashes = {}
    actual = sha256_file(contract_path, missing_ok=True)
    matching_keys = [
        raw_path
        for raw_path in evidence_hashes
        if normalize_workspace_path(agent_root, raw_path).resolve() == contract_path.resolve()
    ]
    if not matching_keys:
        return _check(
            "contract_hash_bound_to_approval_evidence",
            False,
            ["approval evidence bundle does not hash the strategy plugin contract"],
        )
    errors = [
        "approval evidence contract hash does not match current contract"
        for raw_path in matching_keys
        if str(evidence_hashes[raw_path]) != actual
    ]
    return _check("contract_hash_bound_to_approval_evidence", not errors, errors)


def _check(name: str, passed: bool, errors: list[str]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "errors": [] if passed else errors}


def _load_or_refresh_bundle(
    *,
    agent_root: Path,
    scope_id: str,
    approval_evidence_path: Path | None,
    refresh_evidence: bool,
) -> dict[str, Any]:
    if refresh_evidence:
        return run_approval_evidence_spine(agent_root=agent_root, scope_id=scope_id, refresh=True)
    path = (
        normalize_workspace_path(agent_root, approval_evidence_path)
        if approval_evidence_path is not None
        else agent_root
        / "artifacts"
        / "validation"
        / "approval_evidence"
        / scope_id
        / "approval_evidence_bundle.json"
    )
    return _read_json(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_contract(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _only_maturity_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    before_rest = {key: value for key, value in before.items() if key != "maturity"}
    after_rest = {key: value for key, value in after.items() if key != "maturity"}
    return before_rest == after_rest and after.get("maturity") == TARGET_MATURITY


def _artifact_root(agent_root: Path, scope_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return normalize_workspace_path(agent_root, explicit)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        agent_root
        / "artifacts"
        / "validation"
        / "bridge_promotion"
        / scope_id
        / stamp
    )


def _display_path(agent_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(agent_root.resolve()).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
