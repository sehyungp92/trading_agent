from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from learning_sufficiency_gate_utils import checklist_completion_check


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "artifacts" / "learning_sufficiency" / "phase2_manifests" / "manifest_index.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "learning_sufficiency" / "ptg3_gate_report.json"

REQUIRED_CHECKS = {
    "join_coverage": [
        "decision_to_trade_join",
        "decision_to_order_join",
        "order_to_fill_join",
        "risk_portfolio_join",
    ],
    "denominator_coverage": ["denominator_coverage"],
    "after_cost_coverage": ["after_cost_coverage"],
}
KNOWN_STATUSES = {"pass", "partial", "missing", "not_applicable"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify PTG-3 learning sufficiency gate artifacts.")
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    index_path = Path(args.index)
    output_path = Path(args.output)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = [row for row in index.get("manifests", []) if isinstance(row, dict)]

    row_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    readiness_failures: list[dict[str, Any]] = []
    for row in rows:
        result = _check_manifest_row(row)
        row_results.append(result)
        if result["failures"]:
            failures.append({
                "manifest_path": result["manifest_path"],
                "failures": result["failures"],
            })
        if result["readiness_failures"]:
            readiness_failures.append({
                "manifest_path": result["manifest_path"],
                "failures": result["readiness_failures"],
            })
    checklist_check = checklist_completion_check(["Phase 3", "Phase 4", "Phase 5"])

    report = {
        "schema_version": "learning_sufficiency_ptg3_gate_report_v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_index": _rel(index_path),
        "gate": "PTG-3",
        "required_acceptance_rows": ["AM-06", "AM-07", "AM-08", "AM-09", "AM-10", "AM-11", "AM-25"],
        "required_finite_checklist_sections": ["Phase 3", "Phase 4", "Phase 5"],
        "status": "pass" if not failures and not readiness_failures and checklist_check["passed"] else "blocked",
        "fail_closed_contract_status": "pass" if not failures else "blocked",
        "readiness_status": "pass" if not readiness_failures else "blocked",
        "checklist_status": "pass" if checklist_check["passed"] else "blocked",
        "active_scope_count": index.get("active_scope_count", len(rows)),
        "manifest_count": len(rows),
        "promotion_criteria": (
            "Join, denominator, funnel, and after-cost authority checks are present, "
            "fail closed with gaps when insufficient, and pass readiness only when "
            "AM-08 denominator and after-cost authority checks are satisfied."
        ),
        "failures": failures,
        "readiness_failures": readiness_failures,
        "checks": [checklist_check],
        "manifests": row_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": report["status"] == "pass",
        "gate": report["gate"],
        "status": report["status"],
        "artifact_path": _rel(output_path),
    }, indent=2))
    return 0 if report["status"] == "pass" else 1


def _check_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    manifest_path = ROOT / str(row.get("manifest_path") or "")
    failures: list[str] = []
    if not manifest_path.exists():
        return {
            "manifest_path": _rel(manifest_path),
            "bot_id": row.get("bot_id", ""),
            "strategy_id": row.get("strategy_id", ""),
            "eligibility": "",
            "check_statuses": {},
            "known_gap_count": 0,
            "blocked_learning_capabilities": row.get("blocked_learning_capabilities", []),
            "failures": ["manifest file is missing"],
            "readiness_failures": ["manifest file is missing"],
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    known_gaps = [gap for gap in manifest.get("known_gaps", []) if isinstance(gap, dict)]
    gap_check_ids = {
        str((gap.get("details") or {}).get("check_id") or "")
        for gap in known_gaps
    }
    check_statuses: dict[str, str] = {}
    readiness_failures: list[str] = []

    for section, check_ids in REQUIRED_CHECKS.items():
        source = manifest if section == "after_cost_coverage" else manifest.get(section, {})
        for check_id in check_ids:
            check = source.get(check_id, {}) if isinstance(source, dict) else {}
            status = str(check.get("status") or "")
            check_statuses[check_id] = status
            if not status:
                failures.append(f"{check_id} is missing")
            elif status not in KNOWN_STATUSES:
                failures.append(f"{check_id} has unknown status {status!r}")
            elif status not in {"pass", "not_applicable"} and check_id not in gap_check_ids:
                failures.append(f"{check_id} is {status} without a learning gap record")
            if check_id in {"denominator_coverage", "after_cost_coverage"} and status != "pass":
                readiness_failures.append(f"{check_id} is {status or 'missing'}")

    blocked = [str(item) for item in manifest.get("blocked_learning_capabilities", [])]
    if blocked and not known_gaps:
        failures.append("blocked learning capabilities have no gap records")

    bot_id = str(manifest.get("bot_id") or row.get("bot_id") or "")
    strategy_id = str(manifest.get("strategy_id") or row.get("strategy_id") or "")
    family_id = str(manifest.get("family_id") or row.get("family_id") or "")
    if any("k_stock" in value.lower() or "kalcb" in value.lower() or value.upper() == "OLR" for value in (bot_id, strategy_id, family_id)):
        if not check_statuses.get("denominator_coverage"):
            failures.append("k-stock denominator coverage is absent")

    return {
        "manifest_path": _rel(manifest_path),
        "bot_id": bot_id,
        "strategy_id": strategy_id,
        "eligibility": manifest.get("eligibility", ""),
        "check_statuses": check_statuses,
        "known_gap_count": len(known_gaps),
        "blocked_learning_capabilities": blocked,
        "failures": failures,
        "readiness_failures": readiness_failures,
    }


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
