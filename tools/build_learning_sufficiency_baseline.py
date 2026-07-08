from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    ROOT / "packages" / "trading_assistant_backtest" / "src",
    ROOT / "packages" / "trading_assistant" / "src",
    ROOT / "packages" / "trading_assistant_data" / "src",
    ROOT / "packages" / "trading_contracts" / "src",
    ROOT / "packages" / "trading_backtest" / "src",
):
    if source_root.exists() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from trading_assistant_backtest.validation.validation_matrix import (  # noqa: E402
    run_validation_matrix_audit,
)
from trading_assistant.orchestrator.learning_sufficiency_audit import (  # noqa: E402
    CAPABILITY_REQUIREMENTS,
    CHECK_RUNTIME_EVENT_CLASSES,
    canonical_runtime_event_class,
)


OUTPUT_PATH = ROOT / "artifacts" / "learning_sufficiency" / "baseline_capability_matrix.json"
VALIDATION_ARTIFACT_ROOT = ROOT / "artifacts" / "learning_sufficiency" / "phase0_validation_matrix"

CHECK_ALIASES = {
    "after_cost_authority": ("after_cost_coverage",),
    "runtime_evidence_coverage": (
        "trade_outcome_lineage",
        "missed_opportunity_lineage",
        "deployment_metadata_coverage",
    ),
}


def _required_runtime_events(name: str) -> set[str]:
    check_ids = CAPABILITY_REQUIREMENTS.get(name, CHECK_ALIASES.get(name, (name,)))
    return {
        canonical_runtime_event_class(event_type)
        for check_id in check_ids
        for event_type in CHECK_RUNTIME_EVENT_CLASSES.get(check_id, ())
    }


LEARNING_CAPABILITY_REQUIREMENTS = {
    name: _required_runtime_events(name)
    for name in (
        "trade_outcome_lineage",
        "missed_opportunity_lineage",
        "decision_to_order_join",
        "order_to_fill_join",
        "risk_portfolio_join",
        "filter_threshold_learning",
        "denominator_coverage",
        "after_cost_authority",
        "counterfactual_coverage",
        "proposal_trace_coverage",
        "runtime_evidence_coverage",
    )
}

EVENT_FILE_ALIASES = {
    "trades.jsonl": "trade",
    "missed.jsonl": "missed_opportunity",
    "filter_decisions.json": "filter_decision",
    "order_lifecycle.json": "order",
    "slippage_latency.json": "fill",
    "orderbook_summary.json": "orderbook_context",
    "funnel_summary.json": "pipeline_funnel",
    "pipeline_funnel.json": "pipeline_funnel",
    "pipeline_funnels.json": "pipeline_funnel",
    "portfolio_rules_summary.json": "portfolio_rule",
    "rule_blocks_summary.json": "portfolio_rule",
    "summary.json": "daily_snapshot",
    "health_report.json": "heartbeat",
}


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    validation_matrix = run_validation_matrix_audit(
        agent_root=ROOT,
        artifact_root=VALIDATION_ARTIFACT_ROOT,
    )
    scopes = _active_scopes()
    configured_by_scope = {
        scope["contract_id"]: _configured_events_for_scope(scope)
        for scope in scopes
    }
    observed_counts = _observed_event_counts()
    curated_inventory = _curated_file_inventory()
    capability_rows = [
        _capability_row(scope, configured_by_scope[scope["contract_id"]], observed_counts)
        for scope in scopes
    ]
    payload = {
        "schema_version": "learning_sufficiency_phase0_baseline_v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": {
            "contracts_root": _rel(ROOT / "contracts" / "strategy_plugins"),
            "deployment_metadata_root": _rel(ROOT / "deployments"),
            "validation_matrix_artifact_path": validation_matrix.get("artifact_path", ""),
        },
        "phase_gate": {
            "gate": "PTG-0",
            "required_acceptance_rows": ["AM-01", "AM-14", "AM-15", "AM-25"],
            "status": "pass" if validation_matrix.get("runnable_validations_passed") else "blocked",
            "regression_safety": {
                "runnable_validations_passed": validation_matrix.get(
                    "runnable_validations_passed", False
                ),
                "approval_grade_validation_complete": validation_matrix.get(
                    "approval_grade_validation_complete", False
                ),
                "approval_remaining_gaps": validation_matrix.get("approval_remaining_gaps", []),
            },
        },
        "active_scopes": scopes,
        "event_type_inventory": {
            scope["contract_id"]: {
                "configured_event_types": sorted(configured_by_scope[scope["contract_id"]]),
                "observed_event_counts": {
                    event_type: observed_counts.get(event_type, 0)
                    for event_type in sorted(configured_by_scope[scope["contract_id"]])
                },
            }
            for scope in scopes
        },
        "curated_file_inventory": curated_inventory,
        "capability_matrix": capability_rows,
        "material_missing_capabilities": _material_missing_capabilities(capability_rows),
        "strategies_with_no_denominator_funnel_evidence": _scopes_missing(
            capability_rows, "denominator_coverage",
        ),
        "strategies_with_no_order_fill_join_evidence": _scopes_missing(
            capability_rows, "order_to_fill_join",
        ),
        "strategies_with_no_after_cost_authority": _scopes_missing(
            capability_rows, "after_cost_authority",
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "artifact_path": _rel(OUTPUT_PATH)}, indent=2))
    return 0


def _active_scopes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contract_path in sorted((ROOT / "contracts" / "strategy_plugins").glob("*/strategy_plugin_contract.json")):
        contract = _read_json(contract_path)
        contract_id = contract_path.parent.name
        deployment_path = _deployment_metadata_path(contract_id)
        deployment = _read_json(deployment_path) if deployment_path else {}
        rows.append({
            "contract_id": contract_id,
            "plugin_id": contract.get("plugin_id", ""),
            "bot_id": deployment.get("bot_id") or _bot_from_live_repo(contract.get("live_repo_path", "")),
            "family_id": _family_id(contract_id),
            "strategy_id": deployment.get("strategy_id") or contract_id,
            "portfolio_id": deployment.get("portfolio_id", ""),
            "bridge_id": contract_id,
            "maturity": contract.get("maturity", ""),
            "approval_ready": bool(deployment.get("approval_ready", False)),
            "deployment_id": deployment.get("deployment_id", ""),
            "contract_path": _rel(contract_path),
            "deployment_metadata_path": _rel(deployment_path) if deployment_path else "",
            "required_telemetry_schemas": contract.get("required_telemetry_schemas", []),
        })
    return rows


def _deployment_metadata_path(contract_id: str) -> Path | None:
    matches = sorted(
        (ROOT / "deployments").glob(f"*/generated/runtime_deployment_metadata/{contract_id}/deployment_metadata.json")
    )
    return matches[0] if matches else None


def _bot_from_live_repo(live_repo_path: str) -> str:
    if "crypto_trader" in live_repo_path:
        return "crypto"
    if "k_stock_trader" in live_repo_path:
        return "k_stock"
    if "ibkr_trading" in live_repo_path:
        return "ibkr"
    return ""


def _family_id(contract_id: str) -> str:
    if contract_id.startswith("trading_"):
        return contract_id
    if contract_id.startswith("crypto_"):
        return "crypto_trader_portfolio"
    if contract_id.startswith("k_stock_"):
        return "k_stock"
    return ""


def _configured_events_for_scope(scope: dict[str, Any]) -> set[str]:
    contract_id = scope["contract_id"]
    bot_id = scope["bot_id"]
    paths: list[tuple[Path, str]] = []
    if bot_id == "crypto":
        paths.append((
            ROOT / "trading" / "crypto_trader" / "src" / "crypto_trader" / "instrumentation" / "sidecar.py",
            "_EVENT_FILE_MAP",
        ))
    elif bot_id == "k_stock":
        paths.append((
            ROOT / "trading" / "k_stock_trader" / "instrumentation" / "src" / "event_contract.py",
            "EVENT_DIRS",
        ))
    elif bot_id == "ibkr":
        family = "stock"
        if "momentum" in contract_id:
            family = "momentum"
        elif "swing" in contract_id:
            family = "swing"
        paths.append((
            ROOT
            / "trading"
            / "ibkr_trading"
            / "strategies"
            / family
            / "instrumentation"
            / "src"
            / "sidecar.py",
            "_DIR_TO_EVENT_TYPE",
        ))
    events: set[str] = {"deployment"}
    for path, variable in paths:
        mapping = _literal_dict(path, variable)
        if variable == "EVENT_DIRS":
            events.update(str(key) for key in mapping)
        else:
            events.update(str(value) for value in mapping.values())
    return events


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
        if not matches or value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _observed_event_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    bounded_roots = [
        ROOT / "packages" / "trading_assistant" / "memory",
        ROOT / "artifacts" / "validation",
    ]
    for base in bounded_roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            event_type = EVENT_FILE_ALIASES.get(path.name)
            if event_type is None:
                parent = path.parent.name
                event_type = _event_type_from_directory(parent)
            if event_type is None:
                continue
            counts[event_type] += _record_count(path)
    deployment_count = len(list((ROOT / "deployments").glob("*/generated/runtime_deployment_metadata/*/deployment_metadata.json")))
    if deployment_count:
        counts["deployment"] += deployment_count
    return counts


def _event_type_from_directory(directory: str) -> str | None:
    directory = directory.lower()
    aliases = {
        "trades": "trade",
        "missed": "missed_opportunity",
        "filter_decisions": "filter_decision",
        "orders": "order",
        "fills": "fill",
        "portfolio_rules": "portfolio_rule",
        "pipeline_funnel": "pipeline_funnel",
        "pipeline_funnels": "pipeline_funnel",
        "post_exit": "post_exit",
        "orderbook": "orderbook_context",
        "deployments": "deployment",
    }
    return aliases.get(directory)


def _record_count(path: Path) -> int:
    try:
        if path.suffix == ".jsonl":
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        return 1
    except OSError:
        return 0


def _curated_file_inventory() -> dict[str, Any]:
    inventory: dict[str, Any] = {"roots": [], "files_by_name": {}}
    roots = [ROOT / "packages" / "trading_assistant" / "memory", ROOT / "artifacts" / "validation"]
    counts: Counter[str] = Counter()
    for root in roots:
        if not root.exists():
            continue
        inventory["roots"].append(_rel(root))
        for path in root.rglob("*"):
            if path.is_file() and path.name in EVENT_FILE_ALIASES:
                counts[path.name] += 1
    inventory["files_by_name"] = dict(sorted(counts.items()))
    return inventory


def _capability_row(
    scope: dict[str, Any],
    configured_event_types: set[str],
    observed_counts: Counter[str],
) -> dict[str, Any]:
    capabilities: dict[str, Any] = {}
    for capability, required in LEARNING_CAPABILITY_REQUIREMENTS.items():
        configured = _covers_required(configured_event_types, required)
        observed = _covers_required(
            {event_type for event_type, count in observed_counts.items() if count > 0},
            required,
        )
        if observed:
            status = "observed"
        elif configured:
            status = "configured_unobserved"
        else:
            status = "unsupported"
        capabilities[capability] = {
            "status": status,
            "configured": configured,
            "observed": observed,
            "required_event_types": sorted(required),
            "missing_configured_event_types": _missing_required(configured_event_types, required),
            "missing_observed_event_types": _missing_required(
                {event_type for event_type, count in observed_counts.items() if count > 0},
                required,
            ),
        }
    return {
        "contract_id": scope["contract_id"],
        "bot_id": scope["bot_id"],
        "strategy_id": scope["strategy_id"],
        "portfolio_id": scope["portfolio_id"],
        "capabilities": capabilities,
    }


def _covers_required(available: set[str], required: set[str]) -> bool:
    return not _missing_required(available, required)


def _missing_required(available: set[str], required: set[str]) -> list[str]:
    canonical_available = {canonical_runtime_event_class(event_type) for event_type in available}
    return sorted(
        event_type for event_type in required
        if canonical_runtime_event_class(event_type) not in canonical_available
    )


def _material_missing_capabilities(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for row in rows:
        for capability, status in row["capabilities"].items():
            if status["status"] == "observed":
                continue
            missing.append({
                "contract_id": row["contract_id"],
                "strategy_id": row["strategy_id"],
                "capability": capability,
                "status": status["status"],
                "missing_observed_event_types": status["missing_observed_event_types"],
            })
    return missing


def _scopes_missing(rows: list[dict[str, Any]], capability: str) -> list[str]:
    return [
        row["contract_id"]
        for row in rows
        if row["capabilities"][capability]["status"] != "observed"
    ]


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
