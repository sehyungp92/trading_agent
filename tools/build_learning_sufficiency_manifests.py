from __future__ import annotations

import argparse
import ast
import json
import sys
from calendar import monthrange
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    ROOT / "packages" / "trading_assistant" / "src",
    ROOT / "packages" / "trading_contracts" / "src",
):
    if source_root.exists() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from trading_assistant.orchestrator.learning_sufficiency_audit import (  # noqa: E402
    CAPABILITY_REQUIREMENTS,
    CHECK_RUNTIME_EVENT_CLASSES,
    LearningSufficiencyAuditor,
    canonical_runtime_event_class,
)


DEFAULT_BASELINE = ROOT / "artifacts" / "learning_sufficiency" / "baseline_capability_matrix.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "learning_sufficiency" / "phase2_manifests"
DEFAULT_CURATED_DIR = ROOT / "packages" / "trading_assistant" / "memory" / "data" / "curated"
DEFAULT_RAW_DIR = ROOT / "packages" / "trading_assistant" / "memory" / "data" / "raw"
DEFAULT_FINDINGS_DIR = ROOT / "artifacts" / "learning_sufficiency" / "phase2_findings"
CHECK_ALIASES = {
    "after_cost_authority": ("after_cost_coverage",),
    "runtime_evidence_coverage": (
        "trade_outcome_lineage",
        "missed_opportunity_lineage",
        "deployment_metadata_coverage",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit learning-sufficiency manifests for active scopes.")
    parser.add_argument("--run-month", default=_latest_completed_month())
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--curated-dir", default=str(DEFAULT_CURATED_DIR))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--findings-dir", default=str(DEFAULT_FINDINGS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args(argv)

    run_month = args.run_month
    window_start, window_end = _month_window(run_month)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    findings_dir = Path(args.findings_dir)
    findings_dir.mkdir(parents=True, exist_ok=True)
    auditor = LearningSufficiencyAuditor(
        Path(args.curated_dir),
        findings_dir,
        raw_data_dir=Path(args.raw_dir),
    )

    scopes = _active_scopes(Path(args.baseline))
    baseline = _read_json(Path(args.baseline))
    capability_rows = {
        str(row.get("contract_id") or ""): row
        for row in baseline.get("capability_matrix", [])
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        contract_id = str(scope.get("contract_id") or scope.get("strategy_id") or "unknown")
        bot_id = str(scope.get("bot_id") or "")
        strategy_id = str(scope.get("strategy_id") or contract_id)
        artifact_root = output_root / bot_id / run_month / strategy_id
        artifact_root.mkdir(parents=True, exist_ok=True)
        telemetry_path = artifact_root / "telemetry_manifest.json"
        manifest_path = artifact_root / "learning_sufficiency_manifest.json"
        expected_session_path = artifact_root / "expected_active_sessions.json"
        runtime_support_path = artifact_root / "runtime_evidence_support.json"
        expected_session_path.write_text(
            json.dumps(_expected_session_payload(scope, window_start, window_end), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        runtime_support_path.write_text(
            json.dumps(
                _runtime_support_payload(scope, capability_rows.get(contract_id, {})),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        deployment_metadata_path = ROOT / str(scope.get("deployment_metadata_path") or "")
        deployment_metadata_paths = [deployment_metadata_path] if deployment_metadata_path.exists() else []
        contract_path = ROOT / str(scope.get("contract_path") or "")
        manifest = auditor.build_manifest(
            bot_id=bot_id,
            strategy_id=strategy_id,
            family_id=str(scope.get("family_id") or ""),
            portfolio_id=str(scope.get("portfolio_id") or ""),
            run_month=run_month,
            window_start=window_start,
            window_end=window_end,
            telemetry_manifest_path=telemetry_path,
            output_path=manifest_path,
            deployment_metadata_paths=deployment_metadata_paths,
            strategy_contract_path=contract_path if contract_path.exists() else None,
            expected_session_paths=[expected_session_path],
            runtime_support_paths=[runtime_support_path],
        )
        rows.append({
            "contract_id": contract_id,
            "bot_id": bot_id,
            "strategy_id": strategy_id,
            "family_id": scope.get("family_id", ""),
            "portfolio_id": scope.get("portfolio_id", ""),
            "manifest_path": _rel(manifest_path),
            "telemetry_manifest_path": _rel(telemetry_path),
            "expected_session_path": _rel(expected_session_path),
            "runtime_support_path": _rel(runtime_support_path),
            "eligibility": manifest.eligibility.value,
            "supported_learning_capabilities": manifest.supported_learning_capabilities,
            "blocked_learning_capabilities": manifest.blocked_learning_capabilities,
            "known_gap_count": len(manifest.known_gaps),
            "event_counts_by_type": manifest.event_counts_by_type,
        })

    missing_manifest_rows = [row for row in rows if not (ROOT / row["manifest_path"]).exists()]
    missing_gap_rows = [
        row
        for row in rows
        if row["blocked_learning_capabilities"] and not row["known_gap_count"]
    ]
    index = {
        "schema_version": "learning_sufficiency_phase2_manifest_index_v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_month": run_month,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "active_scope_count": len(scopes),
        "manifest_count": len(rows),
        "phase_gate": {
            "gate": "PTG-2",
            "required_acceptance_rows": ["AM-04", "AM-05", "AM-16", "AM-25"],
            "status": "pass" if not missing_manifest_rows and not missing_gap_rows else "blocked",
            "missing_manifest_rows": missing_manifest_rows,
            "missing_gap_rows": missing_gap_rows,
        },
        "manifests": rows,
    }
    index_path = output_root / "manifest_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": index["phase_gate"]["status"] == "pass",
        "manifest_count": len(rows),
        "artifact_path": _rel(index_path),
    }, indent=2))
    return 0 if index["phase_gate"]["status"] == "pass" else 1


def _active_scopes(baseline_path: Path) -> list[dict[str, Any]]:
    if baseline_path.exists():
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        scopes = payload.get("active_scopes", [])
        if isinstance(scopes, list):
            return [scope for scope in scopes if isinstance(scope, dict)]
    rows: list[dict[str, Any]] = []
    for contract_path in sorted((ROOT / "contracts" / "strategy_plugins").glob("*/strategy_plugin_contract.json")):
        contract_id = contract_path.parent.name
        deployment_path = _deployment_metadata_path(contract_id)
        deployment = _read_json(deployment_path) if deployment_path else {}
        rows.append({
            "contract_id": contract_id,
            "bot_id": deployment.get("bot_id") or _bot_from_contract(contract_id),
            "strategy_id": deployment.get("strategy_id") or contract_id,
            "family_id": contract_id if contract_id.startswith("trading_") else "",
            "portfolio_id": deployment.get("portfolio_id", ""),
            "deployment_metadata_path": _rel(deployment_path) if deployment_path else "",
        })
    return rows


def _deployment_metadata_path(contract_id: str) -> Path | None:
    matches = sorted(
        (ROOT / "deployments").glob(f"*/generated/runtime_deployment_metadata/{contract_id}/deployment_metadata.json")
    )
    return matches[0] if matches else None


def _runtime_support_payload(scope: dict[str, Any], capability_row: dict[str, Any]) -> dict[str, Any]:
    classifications = _runtime_event_value_classifications(scope)
    return {
        "schema_version": "runtime_evidence_support_v1",
        "bot_id": str(scope.get("bot_id") or ""),
        "strategy_id": str(scope.get("strategy_id") or ""),
        "declares_complete_runtime_support": True,
        "support_source_paths": [_rel(path) for path in _runtime_support_source_paths(scope) if path.exists()],
        "event_value_classifications": classifications,
        "capabilities": _capabilities_with_alias_credit(capability_row.get("capabilities", {}), classifications),
    }


def _capabilities_with_alias_credit(capabilities: Any, classifications: dict[str, str]) -> dict[str, Any]:
    if not isinstance(capabilities, dict):
        return {}
    result: dict[str, Any] = {}
    for name, details in capabilities.items():
        if not isinstance(details, dict):
            result[str(name)] = details
            continue
        row = dict(details)
        required = _required_runtime_events(str(name), row.get("required_event_types"))
        row["required_event_types"] = required
        missing = [
            event_type for event_type in required
            if not _has_learning_authority_source_for_class(classifications, canonical_runtime_event_class(event_type))
        ]
        row["missing_configured_event_types"] = missing
        if not missing:
            row["configured"] = True
            if str(row.get("status") or "").strip().lower() == "unsupported":
                row["status"] = "observed" if row.get("observed") else "configured_unobserved"
        result[str(name)] = row
    return result


def _required_runtime_events(name: str, fallback: Any) -> list[str]:
    check_ids = CAPABILITY_REQUIREMENTS.get(name, CHECK_ALIASES.get(name, (name,)))
    mapped = [
        canonical_runtime_event_class(event_type)
        for check_id in check_ids
        for event_type in CHECK_RUNTIME_EVENT_CLASSES.get(check_id, ())
    ]
    if mapped:
        return sorted(dict.fromkeys(mapped))
    return sorted(dict.fromkeys(canonical_runtime_event_class(event_type) for event_type in _string_list(fallback)))


def _has_learning_authority_source_for_class(classifications: dict[str, str], event_class: str) -> bool:
    return any(
        canonical_runtime_event_class(event_type) == event_class
        and str(value_class).strip().lower() == "learning_authority"
        for event_type, value_class in classifications.items()
    )


def _expected_session_payload(scope: dict[str, Any], window_start: date, window_end: date) -> dict[str, Any]:
    bot_id = str(scope.get("bot_id") or "")
    return {
        "schema_version": "expected_active_sessions_v1",
        "source": "phase_manifest_calendar_projection",
        "bot_id": bot_id,
        "strategy_id": str(scope.get("strategy_id") or ""),
        "expected_session_days": [
            day.isoformat()
            for day in _expected_active_session_days(bot_id, window_start, window_end)
        ],
    }


def _expected_active_session_days(bot_id: str, window_start: date, window_end: date) -> list[date]:
    days: list[date] = []
    current = window_start
    while current <= window_end:
        if bot_id == "crypto" or current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _runtime_support_source_paths(scope: dict[str, Any]) -> list[Path]:
    bot_id = str(scope.get("bot_id") or "")
    contract_id = str(scope.get("contract_id") or scope.get("strategy_id") or "")
    if bot_id == "crypto":
        return [
            ROOT / "bots" / "crypto_trader" / "src" / "crypto_trader" / "instrumentation" / "sidecar.py",
            ROOT / "bots" / "crypto_trader" / "src" / "crypto_trader" / "instrumentation" / "async_postgres_sink.py",
        ]
    if bot_id == "k_stock":
        return [ROOT / "bots" / "k_stock_trader" / "instrumentation" / "src" / "event_contract.py"]
    if bot_id == "ibkr":
        family = "stock"
        if "momentum" in contract_id:
            family = "momentum"
        elif "swing" in contract_id:
            family = "swing"
        return [ROOT / "bots" / "ibkr_trading" / "strategies" / family / "instrumentation" / "src" / "sidecar.py"]
    return []


def _runtime_event_value_classifications(scope: dict[str, Any]) -> dict[str, str]:
    classifications: dict[str, str] = {}
    for path in _runtime_support_source_paths(scope):
        for variable in ("EVENT_VALUE_CLASSES", "_EVENT_VALUE_CLASSES"):
            for event_type, value_class in _literal_dict(path, variable).items():
                classifications[str(event_type)] = str(value_class)
    return dict(sorted(classifications.items()))


def _literal_dict(path: Path, variable: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            matches = any(isinstance(target, ast.Name) and target.id == variable for target in node.targets)
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            matches = node.target.id == variable
            value_node = node.value
        else:
            continue
        if matches and value_node is not None:
            value = ast.literal_eval(value_node)
            return value if isinstance(value, dict) else {}
    return {}


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _bot_from_contract(contract_id: str) -> str:
    if contract_id.startswith("crypto_"):
        return "crypto"
    if contract_id.startswith("k_stock_"):
        return "k_stock"
    if contract_id.startswith("trading_"):
        return "ibkr"
    return ""


def _month_window(run_month: str) -> tuple[date, date]:
    year, month = (int(part) for part in run_month.split("-", 1))
    return date(year, month, 1), date(year, month, monthrange(year, month)[1])


def _latest_completed_month() -> str:
    today = datetime.now(UTC).date()
    year = today.year
    month = today.month - 1
    if month == 0:
        year -= 1
        month = 12
    return f"{year:04d}-{month:02d}"


def _rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
