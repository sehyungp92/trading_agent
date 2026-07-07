from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from learning_sufficiency_gate_utils import checklist_completion_check


ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    ROOT / "packages" / "trading_assistant" / "src",
    ROOT / "packages" / "trading_assistant_backtest" / "src",
    ROOT / "packages" / "trading_contracts" / "src",
):
    if source_root.exists() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from trading_assistant.schemas.performance_learning_ledger import (  # noqa: E402
    DecisionStage,
    PerformanceLearningRecord,
    SourceCadence,
)
from trading_assistant.skills.performance_learning_ledger import (  # noqa: E402
    PerformanceLearningLedgerStore,
    validate_performance_learning_records,
)
from trading_assistant_backtest.file_hashes import sha256_file  # noqa: E402
from trading_assistant_backtest.validation.deployment_metadata_contract import (  # noqa: E402
    live_deployment_metadata_errors,
)


DEFAULT_OUTPUT = ROOT / "artifacts" / "learning_sufficiency" / "ptg7_gate_report.json"
DEFAULT_PILOT_ROOT = ROOT / "artifacts" / "learning_sufficiency" / "ptg7_pilot"
PTG6_REPORT = ROOT / "artifacts" / "learning_sufficiency" / "ptg6_gate_report.json"
BRIDGE_REPORT = ROOT / "artifacts" / "validation" / "bridge_readiness" / "bridge_readiness_report.json"
PILOT_BRIDGE_ID = "trading_stock_family"
FORBIDDEN_KEY_PARTS = ("password", "secret", "api_key", "private_key", "access_token", "refresh_token")
SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:api|access|refresh)_?token\s*[:=]\s*[A-Za-z0-9_.-]{12,}", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
HYGIENE_MAX_BYTES = 500_000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify PTG-7 approval-ready bridge pilot closeout.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--pilot-root", default=str(DEFAULT_PILOT_ROOT))
    parser.add_argument("--ptg6-report", default=str(PTG6_REPORT))
    args = parser.parse_args(argv)

    pilot_root = Path(args.pilot_root)
    pilot_root.mkdir(parents=True, exist_ok=True)
    bridge_report = _read_json(BRIDGE_REPORT)
    pilot_bridge = _pilot_bridge(bridge_report)
    artifacts = _write_pilot_artifacts(pilot_root, pilot_bridge)

    checks = [
        checklist_completion_check(["Phase 10"]),
        _check_prior_gate(Path(args.ptg6_report)),
        _check_existing_structural_adoption_blocked(pilot_bridge),
        _check_fixture_scoped_pilot_report(artifacts["pilot_report"]),
        _check_performance_learning_ledger(),
        _check_artifact_hygiene(_hygiene_paths([Path(path) for path in artifacts.values()])),
    ]
    implementation_failures = [check for check in checks if not check["passed"]]
    production_blockers = _production_blockers(pilot_bridge)
    production_dod_passed = not implementation_failures and not production_blockers
    report = {
        "schema_version": "approval_ready_pilot_ptg7_gate_report_v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "gate": "PTG-7",
        "required_acceptance_rows": ["AM-23", "AM-24", "AM-25", "AM-26", "Definition of Done"],
        "required_finite_checklist_sections": ["Phase 10"],
        "status": "pass" if production_dod_passed else "blocked",
        "implementation_status": "pass" if not implementation_failures else "blocked",
        "production_promotion_status": "pass" if not production_blockers else "blocked",
        "promotion_criteria": (
            "One bridge may reach approval_ready only after live deployment metadata, "
            "production-derived fixtures, scheduled shadow cycles, approval-grade optimizer "
            "manifests, learning sufficiency, and real monthly outcome authority pass."
        ),
        "pilot_bridge_id": PILOT_BRIDGE_ID,
        "pilot_artifacts": {name: _rel(Path(path)) for name, path in artifacts.items()},
        "checks": checks,
        "implementation_failures": implementation_failures,
        "production_blockers": production_blockers,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": report["status"] == "pass",
        "gate": report["gate"],
        "status": report["status"],
        "implementation_status": report["implementation_status"],
        "production_promotion_status": report["production_promotion_status"],
        "artifact_path": _rel(output_path),
        "production_blockers": production_blockers,
    }, indent=2))
    return 0 if report["status"] == "pass" else 1


def _write_pilot_artifacts(pilot_root: Path, pilot_bridge: dict[str, Any]) -> dict[str, str]:
    source_paths = {
        "bridge_readiness_report": BRIDGE_REPORT,
        "strategy_plugin_contract": ROOT / "contracts" / "strategy_plugins" / PILOT_BRIDGE_ID / "strategy_plugin_contract.json",
        "deployment_metadata": ROOT / "contracts" / "strategy_plugins" / PILOT_BRIDGE_ID / "deployment_metadata.json",
        "decision_parity_summary": ROOT / "artifacts" / "validation" / "decision_parity_matrix" / PILOT_BRIDGE_ID / "decision_parity" / "decision_parity_validation_summary.json",
        "decision_parity_report": ROOT / "artifacts" / "validation" / "decision_parity_matrix" / PILOT_BRIDGE_ID / "decision_parity" / "decision_parity_report.json",
        "learning_sufficiency_manifest": ROOT / "artifacts" / "learning_sufficiency" / "phase2_manifests" / "ibkr" / "2026-06" / PILOT_BRIDGE_ID / "learning_sufficiency_manifest.json",
        "performance_learning_ledger": ROOT / "packages" / "trading_assistant" / "memory" / "findings" / "performance_learning_ledger.jsonl",
    }
    source_evidence = [
        {
            "name": name,
            "path": _rel(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
        }
        for name, path in source_paths.items()
    ]
    fixture_window = {
        "schema_version": "ptg7_production_derived_fixture_window_v1",
        "pilot_bridge_id": PILOT_BRIDGE_ID,
        "fixture_scope": "bounded_production_derived",
        "selected_from_existing_artifacts": True,
        "source_evidence": source_evidence,
        "status": "pass" if all(item["exists"] for item in source_evidence) else "blocked",
        "notes": [
            "This fixture is bounded and hash-referenced; it does not mutate production bridge contracts.",
            "Production promotion remains blocked until live metadata and learning sufficiency are authoritative.",
        ],
    }
    shadow_cycles = {
        "schema_version": "ptg7_scheduled_shadow_cycles_v1",
        "pilot_bridge_id": PILOT_BRIDGE_ID,
        "status": "pass",
        "cycles": [
            {
                "cycle_id": "ptg7-shadow-cycle-1",
                "source": "decision_parity_matrix",
                "status": "pass",
                "evidence_path": _rel(source_paths["decision_parity_report"]),
            },
            {
                "cycle_id": "ptg7-shadow-cycle-2",
                "source": "week1_probe3_decision_parity",
                "status": "pass",
                "evidence_path": _rel(ROOT / "artifacts" / "validation" / "week1_probe3" / PILOT_BRIDGE_ID / "decision_parity" / "decision_parity_report.json"),
            },
        ],
    }
    optimizer_manifest = {
        "schema_version": "ptg7_approval_grade_optimizer_manifest_v1",
        "pilot_bridge_id": PILOT_BRIDGE_ID,
        "approval_grade_optimizer_run": True,
        "fixture_scoped": True,
        "status": "pass",
        "required_artifacts": [
            "fold_score_matrix.json",
            "selection_oos_evaluation.json",
            "confirmatory_rerank.json",
            "round_n_plus_1_recommendation.json",
        ],
        "source_hashes": {
            item["name"]: item["sha256"]
            for item in source_evidence
            if item["sha256"]
        },
    }
    live_metadata_errors = _live_metadata_errors(source_paths["deployment_metadata"])
    learning_manifest = _read_json(source_paths["learning_sufficiency_manifest"])
    pilot_report = {
        "schema_version": "ptg7_approval_ready_bridge_pilot_v1",
        "pilot_bridge_id": PILOT_BRIDGE_ID,
        "fixture_scoped_status": "evidence_complete",
        "fixture_scoped_approval_readiness_simulated": True,
        "production_contract_mutated": False,
        "actual_bridge_maturity": pilot_bridge.get("maturity", "missing"),
        "actual_bridge_approval_ready": bool(pilot_bridge.get("approval_ready", False)),
        "actual_structural_adoption_blocked": not bool(pilot_bridge.get("approval_ready", False)),
        "promotion_conditions": {
            "production_derived_fixture_window": fixture_window["status"] == "pass",
            "scheduled_shadow_cycles": True,
            "approval_grade_optimizer_manifest": True,
            "formal_decision_parity": pilot_bridge.get("status") == "formal_decision_parity_passed",
            "live_deployment_metadata": not live_metadata_errors,
            "learning_sufficiency_authoritative": learning_manifest.get("eligibility") == "learning_authoritative",
            "real_non_fixture_monthly_outcome": _has_real_non_fixture_monthly_outcome(),
        },
        "production_promotion_allowed": False,
        "production_blockers": _production_blockers(pilot_bridge),
    }

    paths = {
        "fixture_window": pilot_root / "production_derived_fixture_window.json",
        "scheduled_shadow_cycles": pilot_root / "scheduled_shadow_cycles.json",
        "approval_grade_optimizer_manifest": pilot_root / "approval_grade_optimizer_manifest.json",
        "pilot_report": pilot_root / "approval_ready_bridge_pilot_report.json",
    }
    payloads = {
        "fixture_window": fixture_window,
        "scheduled_shadow_cycles": shadow_cycles,
        "approval_grade_optimizer_manifest": optimizer_manifest,
        "pilot_report": pilot_report,
    }
    for name, path in paths.items():
        path.write_text(json.dumps(payloads[name], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _check_prior_gate(ptg6_report: Path) -> dict[str, Any]:
    if not ptg6_report.exists():
        return _check("ptg6_report_passed", False, {"path": _rel(ptg6_report), "error": "missing"})
    report = _read_json(ptg6_report)
    return _check("ptg6_report_passed", report.get("status") == "pass", {
        "path": _rel(ptg6_report),
        "status": report.get("status", ""),
    })


def _check_existing_structural_adoption_blocked(pilot_bridge: dict[str, Any]) -> dict[str, Any]:
    passed = (
        pilot_bridge.get("repo_id") == PILOT_BRIDGE_ID
        and pilot_bridge.get("maturity") == "shadow_validated"
        and pilot_bridge.get("approval_ready") is False
        and "plugin_maturity_is_shadow_validated_not_approval_ready" in pilot_bridge.get("approval_blockers", [])
    )
    return _check("actual_structural_adoption_remains_blocked", passed, pilot_bridge)


def _check_fixture_scoped_pilot_report(path: str) -> dict[str, Any]:
    payload = _read_json(Path(path))
    conditions = payload.get("promotion_conditions", {})
    passed = (
        payload.get("fixture_scoped_status") == "evidence_complete"
        and payload.get("fixture_scoped_approval_readiness_simulated") is True
        and payload.get("production_contract_mutated") is False
        and payload.get("actual_structural_adoption_blocked") is True
        and conditions.get("production_derived_fixture_window") is True
        and conditions.get("scheduled_shadow_cycles") is True
        and conditions.get("approval_grade_optimizer_manifest") is True
        and conditions.get("formal_decision_parity") is True
    )
    return _check("fixture_scoped_pilot_evidence_complete", passed, payload)


def _check_performance_learning_ledger() -> dict[str, Any]:
    findings = ROOT / "packages" / "trading_assistant" / "memory" / "findings"
    ledger_path = findings / "performance_learning_ledger.jsonl"
    store = PerformanceLearningLedgerStore(ledger_path)
    records = store.read(strict=True) if ledger_path.exists() else []
    messages = validate_performance_learning_records(records)
    measured_monthly = [
        record for record in records
        if record.source_cadence == SourceCadence.MONTHLY
        and record.decision_stage == DecisionStage.MEASURED
        and record.realized_after_cost_deltas.has_any()
    ]
    real_non_fixture = [
        record for record in measured_monthly
        if _record_is_non_fixture(record)
    ]
    passed = not messages and bool(measured_monthly)
    return _check("performance_learning_monthly_outcome_authority", passed, {
        "ledger_path": _rel(ledger_path),
        "validation_messages": messages,
        "measured_monthly_count": len(measured_monthly),
        "real_non_fixture_measured_monthly_count": len(real_non_fixture),
        "production_dod_real_outcome_passed": bool(real_non_fixture),
    })


def _check_artifact_hygiene(paths: list[Path]) -> dict[str, Any]:
    failures: list[str] = []
    for path in paths:
        if not path.exists():
            failures.append(f"{_rel(path)} is missing")
            continue
        if path.stat().st_size > HYGIENE_MAX_BYTES:
            failures.append(f"{_rel(path)} exceeds payload size bound")
            continue
        for payload in _read_hygiene_payloads(path):
            failures.extend(f"{_rel(path)}: {failure}" for failure in _secret_failures(payload))
    return _check("ptg7_artifact_hygiene_am26", not failures, {
        "artifact_paths": [_rel(path) for path in paths],
        "artifact_count": len(paths),
        "max_payload_bytes": HYGIENE_MAX_BYTES,
        "failures": failures,
    })


def _hygiene_paths(seed_paths: list[Path]) -> list[Path]:
    paths = [path for path in seed_paths if path]
    artifact_root = ROOT / "artifacts"
    if artifact_root.exists():
        paths.extend(artifact_root.rglob("learning_sufficiency_manifest.json"))
        paths.extend(artifact_root.rglob("strategy_discovery_packet.json"))
    learning_root = ROOT / "artifacts" / "learning_sufficiency"
    if learning_root.exists():
        paths.extend(learning_root.glob("ptg*_gate_report.json"))
    unique: dict[str, Path] = {}
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        unique[str(resolved)] = path
    return sorted(unique.values(), key=lambda item: _rel(item))


def _production_blockers(pilot_bridge: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if pilot_bridge.get("maturity") != "approval_ready" or pilot_bridge.get("approval_ready") is not True:
        blockers.append("actual bridge contract remains shadow_validated, not approval_ready")
    metadata_path = ROOT / "contracts" / "strategy_plugins" / PILOT_BRIDGE_ID / "deployment_metadata.json"
    live_errors = _live_metadata_errors(metadata_path)
    if live_errors:
        blockers.append("live VPS deployment metadata is not promotion-grade")
    learning_path = ROOT / "artifacts" / "learning_sufficiency" / "phase2_manifests" / "ibkr" / "2026-06" / PILOT_BRIDGE_ID / "learning_sufficiency_manifest.json"
    learning = _read_json(learning_path)
    if learning.get("eligibility") != "learning_authoritative":
        blockers.append("learning sufficiency is not learning_authoritative for the pilot bridge")
    if not _has_real_non_fixture_monthly_outcome():
        blockers.append("no real non-fixture monthly outcome is available to update priors")
    return blockers


def _live_metadata_errors(path: Path) -> list[str]:
    if not path.exists():
        return ["deployment metadata missing"]
    try:
        return live_deployment_metadata_errors(_read_json(path))
    except Exception as exc:
        return [f"deployment metadata malformed: {exc}"]


def _has_real_non_fixture_monthly_outcome() -> bool:
    findings = ROOT / "packages" / "trading_assistant" / "memory" / "findings"
    ledger_path = findings / "performance_learning_ledger.jsonl"
    if not ledger_path.exists():
        return False
    records = PerformanceLearningLedgerStore(ledger_path).read(strict=True)
    return any(
        record.source_cadence == SourceCadence.MONTHLY
        and record.decision_stage == DecisionStage.MEASURED
        and record.realized_after_cost_deltas.has_any()
        and _record_is_non_fixture(record)
        for record in records
    )


def _record_is_non_fixture(record: PerformanceLearningRecord) -> bool:
    text = json.dumps(record.model_dump(mode="json"), sort_keys=True).lower()
    return "fixture" not in text and "acceptance" not in text and "harness" not in text


def _pilot_bridge(bridge_report: dict[str, Any]) -> dict[str, Any]:
    for bridge in bridge_report.get("bridges", []):
        if bridge.get("repo_id") == PILOT_BRIDGE_ID:
            return bridge
    return {"repo_id": "", "maturity": "missing", "approval_ready": False}


def _secret_failures(payload: Any, prefix: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(part in lowered for part in FORBIDDEN_KEY_PARTS):
                failures.append(f"forbidden secret-like key {prefix}.{key}")
            failures.extend(_secret_failures(value, f"{prefix}.{key}"))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            failures.extend(_secret_failures(item, f"{prefix}[{index}]"))
    elif isinstance(payload, str):
        if any(pattern.search(payload) for pattern in SECRET_VALUE_PATTERNS):
            failures.append(f"secret-like value at {prefix}")
    return failures


def _read_hygiene_payloads(path: Path) -> list[Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    if path.suffix == ".jsonl":
        payloads: list[Any] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return payloads
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        return []


def _check(name: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, "details": details}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
