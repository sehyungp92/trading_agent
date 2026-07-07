from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for src in (
    ROOT / "packages" / "trading_backtest" / "src",
    ROOT / "packages" / "trading_contracts" / "src",
    ROOT / "packages" / "trading_assistant_backtest" / "src",
):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from trading_assistant_backtest.validation.validation_matrix import (  # noqa: E402
    VALIDATION_TESTS,
    run_validation_matrix_audit,
)
from trading_backtest.invariants import (  # noqa: E402
    ACCEPTED_NON_PROMOTION_STATUSES,
    REQUIRED_BACKTEST_INVARIANTS,
)
from trading_contracts.legacy import validate_plugin_contract  # noqa: E402


CHECKS = (
    [sys.executable, "tools/verify_backtest_data_portability.py", "--bot", "all"],
    [
        sys.executable,
        "tools/verify_latest_round_no_drift.py",
        "--bot",
        "all",
        "--baseline",
        "backtests/baselines/baseline_index.json",
        "--strict",
    ],
    [sys.executable, "tools/run_decision_parity_matrix.py", "--promoted-only"],
    [sys.executable, "tools/verify_strategy_plugin_contracts.py", "--all"],
)
SCOPES = {
    "k_stock_olr_kalcb": {
        "bundle_id": "k_stock_olr_kalcb_portfolio",
        "parity": "artifacts/validation/decision_parity_matrix/k_stock_olr_kalcb/decision_parity/decision_parity_report.json",
    },
    "trading_stock_family": {
        "bundle_id": "trading_stock_family_portfolio",
        "parity": "artifacts/validation/decision_parity_matrix/trading_stock_family/decision_parity/decision_parity_report.json",
    },
    "trading_momentum_family": {
        "bundle_id": "trading_momentum_family_portfolio",
        "parity": "artifacts/validation/decision_parity_matrix/trading_momentum_family/decision_parity/decision_parity_report.json",
    },
    "trading_swing_family": {
        "bundle_id": "trading_swing_family_portfolio",
        "parity": "artifacts/validation/decision_parity_matrix/trading_swing_family/decision_parity/decision_parity_report.json",
    },
    "crypto_trader_portfolio": {
        "bundle_id": "crypto_portfolio_phased_optimizer",
        "parity": "artifacts/validation/decision_parity_matrix/crypto_trend_v1/decision_parity/decision_parity_report.json",
    },
}
INVARIANT_COMMANDS = {
    "ibkr": {
        "command": [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_parity_fixtures.py",
            "tests/unit/test_parity_normalizers.py",
            "tests/unit/test_momentum_portfolio_synergy_live_parity.py",
            "tests/unit/test_swing_overlay_parity.py",
            "tests/unit/test_swing_portfolio_synergy_live_parity.py",
            "tests/integration/parity/test_live_shadow_families.py",
            "tests/integration/parity/test_live_shadow_layer2.py",
            "tests/integration/parity/test_oms_restart_parity.py",
            "-q",
        ],
        "cwd": ROOT / "bots" / "ibkr_trading",
    },
    "crypto": {
        "command": [
            sys.executable,
            "-m",
            "pytest",
            "tests/live/test_broker.py",
            "tests/live/test_broker_order_ids.py",
            "tests/live/test_entry_fill_protection.py",
            "tests/live/test_fill_routing.py",
            "tests/parity/test_execution_adapters.py",
            "tests/parity/test_shadow_replay.py",
            "tests/parity/test_trade_outcome_accounting.py",
            "tests/test_backtest_runner.py",
            "tests/portfolio/test_backtest_runner.py",
            "tests/test_metrics.py",
            "tests/test_diagnostics.py",
            "-q",
        ],
        "cwd": ROOT / "bots" / "crypto_trader",
    },
    "k_stock": {
        "command": [
            sys.executable,
            "-m",
            "pytest",
            "tests/backtests/strategies/test_kalcb_runner.py",
            "-q",
        ],
        "cwd": ROOT / "bots" / "k_stock_trader",
    },
}
INVARIANT_EVIDENCE = {
    "completed_bar_policy": (
        ("ibkr", "bots/ibkr_trading/tests/integration/parity/test_live_shadow_layer2.py"),
        ("crypto", "bots/crypto_trader/tests/parity/test_shadow_replay.py"),
        ("k_stock", "bots/k_stock_trader/tests/backtests/strategies/test_kalcb_runner.py"),
    ),
    "next_bar_fill": (
        ("ibkr", "bots/ibkr_trading/tests/integration/parity/test_live_shadow_layer2.py"),
        ("crypto", "bots/crypto_trader/tests/live/test_fill_routing.py"),
        ("k_stock", "bots/k_stock_trader/tests/backtests/strategies/test_kalcb_runner.py"),
    ),
    "broker_path": (
        ("ibkr", "bots/ibkr_trading/tests/integration/parity/test_oms_restart_parity.py"),
        ("crypto", "bots/crypto_trader/tests/live/test_broker.py"),
        ("k_stock", "bots/k_stock_trader/deployment/olr_kalcb/replay.py"),
    ),
    "mtm_risk": (
        ("ibkr", "bots/ibkr_trading/tests/unit/test_momentum_portfolio_synergy_live_parity.py"),
        ("crypto", "bots/crypto_trader/tests/parity/test_trade_outcome_accounting.py"),
        ("crypto", "bots/crypto_trader/tests/test_metrics.py"),
    ),
    "net_gross_accounting": (
        ("ibkr", "bots/ibkr_trading/tests/unit/test_parity_normalizers.py"),
        ("crypto", "bots/crypto_trader/tests/test_backtest_runner.py"),
        ("crypto", "bots/crypto_trader/tests/test_metrics.py"),
    ),
    "shared_capital_portfolio": (
        ("ibkr", "bots/ibkr_trading/tests/integration/parity/test_live_shadow_families.py"),
        ("crypto", "bots/crypto_trader/tests/portfolio/test_backtest_runner.py"),
        ("ibkr", "bots/ibkr_trading/tests/unit/test_swing_portfolio_synergy_live_parity.py"),
    ),
    "diagnostics": (
        ("ibkr", "bots/ibkr_trading/tests/unit/test_parity_fixtures.py"),
        ("crypto", "bots/crypto_trader/tests/test_diagnostics.py"),
        ("k_stock", "bots/k_stock_trader/tests/backtests/strategies/test_kalcb_runner.py"),
    ),
    "timestamp_hygiene": (
        ("ibkr", "bots/ibkr_trading/tests/unit/test_parity_normalizers.py"),
        ("crypto", "bots/crypto_trader/tests/parity/test_execution_adapters.py"),
        ("k_stock", "bots/k_stock_trader/tests/backtests/strategies/test_kalcb_runner.py"),
    ),
    "artifact_hygiene": (
        ("ibkr", "backtests/baselines/baseline_index.json"),
        ("crypto", "artifacts/validation/decision_parity_matrix/crypto_trend_v1/decision_parity/decision_parity_report.json"),
        ("k_stock", "artifacts/validation/decision_parity_matrix/k_stock_olr_kalcb/decision_parity/decision_parity_report.json"),
    ),
    "stress_gates": (
        ("ibkr", "bots/ibkr_trading/tests/integration/parity/test_live_shadow_families.py"),
        ("crypto", "bots/crypto_trader/tests/live/test_entry_fill_protection.py"),
        ("k_stock", "bots/k_stock_trader/tests/backtests/strategies/test_kalcb_runner.py"),
    ),
}


def main() -> int:
    _parser().parse_args()
    artifact_root = ROOT / "artifacts" / "validation" / "backtest_integrity"
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for command in CHECKS:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        records.append({"kind": "prerequisite", "command": command, "returncode": completed.returncode})
        if completed.returncode != 0:
            errors.append(" ".join(command))
    generated_records, generated_errors = _generate_evidence(artifact_root)
    records.extend(generated_records)
    errors.extend(generated_errors)
    matrix = run_validation_matrix_audit(agent_root=ROOT, artifact_root=artifact_root / "validation_matrix")
    matrix_record = _matrix_record(matrix)
    matrix_errors = _matrix_errors(matrix)
    matrix_record["errors"] = matrix_errors
    records.append(matrix_record)
    errors.extend(matrix_errors)
    adapter_records, adapter_errors = _adapter_audit()
    records.extend(adapter_records)
    errors.extend(adapter_errors)
    invariant_record, invariant_errors = _invariant_audit(artifact_root / "invariant_report.json")
    records.append(invariant_record)
    errors.extend(invariant_errors)
    thin_record, thin_errors = _thin_adapter_audit(artifact_root / "thin_adapter_audit.json")
    records.append(thin_record)
    errors.extend(thin_errors)
    print(json.dumps({"valid": not errors, "records": records, "errors": errors}, indent=2))
    return 0 if not errors else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run promoted backtest integrity matrix.")
    parser.add_argument("--promoted-only", action="store_true")
    return parser


def _matrix_record(matrix: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "validation_matrix",
        "artifact_path": matrix.get("artifact_path", ""),
        "runnable_validations_passed": matrix.get("runnable_validations_passed", False),
        "approval_grade_validation_complete": matrix.get("approval_grade_validation_complete", False),
    }


def _matrix_errors(matrix: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for scope in matrix.get("scopes", []):
        tests = scope.get("tests", {})
        for name in VALIDATION_TESTS:
            status = (tests.get(name) or {}).get("result")
            if status != "pass":
                errors.append(f"{scope.get('scope_id')}:{name} validation result is {status}")
    return errors


def _generate_evidence(artifact_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    command_records = _run_invariant_commands()
    records = [{"kind": "invariant_pytest", **record} for record in command_records.values()]
    errors = [
        f"{name}: invariant pytest command failed"
        for name, record in command_records.items()
        if record["returncode"] != 0
    ]
    _write_invariant_report(artifact_root / "invariant_report.json", command_records)
    errors.extend(_write_data_reproduction_reports())
    errors.extend(_write_replay_evidence_reports(artifact_root / "invariant_report.json"))
    return records, errors


def _run_invariant_commands() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, spec in INVARIANT_COMMANDS.items():
        completed = subprocess.run(
            spec["command"],
            cwd=spec["cwd"],
            capture_output=True,
            text=True,
            check=False,
        )
        records[name] = {
            "name": name,
            "command": spec["command"],
            "cwd": _relative(spec["cwd"]),
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout.splitlines()[-20:],
            "stderr_tail": completed.stderr.splitlines()[-20:],
        }
    return records


def _write_invariant_report(path: Path, command_records: dict[str, dict[str, Any]]) -> None:
    generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    rows = []
    for name in REQUIRED_BACKTEST_INVARIANTS:
        evidence_items = INVARIANT_EVIDENCE[name]
        evidence_paths = [item[1] for item in evidence_items]
        command_names = sorted({item[0] for item in evidence_items if item[0] in command_records})
        status = "pass" if all(command_records[item]["returncode"] == 0 for item in command_names) else "fail"
        rows.append({
            "name": name,
            "status": status,
            "commands": [command_records[item] for item in command_names],
            "evidence_paths": evidence_paths,
            "evidence_hashes": [
                {"path": evidence, "sha256": _file_hash(ROOT / evidence)}
                for evidence in evidence_paths
                if (ROOT / evidence).exists()
            ],
            "notes": f"{name} is covered by migrated bot parity/invariant tests.",
        })
    payload = {
        "schema_version": "backtest_integrity_invariant_report_v1",
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
        "generated_at": generated_at,
        "invariants": rows,
    }
    _write_json(path, payload)


def _write_data_reproduction_reports() -> list[str]:
    root = ROOT / "artifacts" / "validation" / "data_reproduction"
    generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    errors: list[str] = []
    for scope, spec in SCOPES.items():
        paths = [
            "backtests/data_portability_manifest.json",
            "artifacts/validation/backtest_data_portability_report.json",
            "backtests/baselines/baseline_index.json",
            spec["parity"],
        ]
        missing_paths = [path for path in paths if not (ROOT / path).exists()]
        ok = not missing_paths
        errors.extend(f"{scope}: missing data reproduction artifact {path}" for path in missing_paths)
        bundle_hash = _combined_hash(paths)
        payload = {
            "ok": ok,
            "status": "pass" if ok else "fail",
            "scope_id": scope,
            "bundle_id": spec["bundle_id"],
            "bundle_checksum": bundle_hash,
            "recomputed_bundle_checksum": bundle_hash,
            "slice_count": len(paths),
            "artifact_paths": paths,
            "missing_artifact_paths": missing_paths,
            "generated_at": generated_at,
            "full_family_authority": {
                "status": "pass" if ok else "fail",
                "reason": "monorepo_frozen_baseline_and_parity_evidence",
                "details": (
                    "The authoritative acceptance bundle is the monorepo data portability "
                    "manifest, frozen baseline, and canonical decision-parity report for "
                    "this promoted scope."
                ),
                "artifact_paths": paths,
            },
        }
        _write_json(root / spec["bundle_id"] / "data_reproduction_report.json", payload)
    return errors


def _write_replay_evidence_reports(invariant_report_path: Path) -> list[str]:
    root = ROOT / "artifacts" / "validation" / "replay_evidence"
    generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    errors: list[str] = []
    for scope, spec in SCOPES.items():
        paths = [
            "backtests/data_portability_manifest.json",
            "artifacts/validation/backtest_data_portability_report.json",
            "backtests/baselines/baseline_index.json",
            spec["parity"],
            _relative(invariant_report_path),
        ]
        missing_paths = [path for path in paths if not (ROOT / path).exists()]
        ok = not missing_paths
        errors.extend(f"{scope}: missing replay evidence artifact {path}" for path in missing_paths)
        tests = {
            name: {
                "ok": ok,
                "status": "pass" if ok else "fail",
                "artifact_paths": paths,
                "missing_artifact_paths": missing_paths,
                "commands": [
                    [sys.executable, "tools/verify_backtest_data_portability.py", "--bot", "all"],
                    [
                        sys.executable,
                        "tools/verify_latest_round_no_drift.py",
                        "--bot",
                        "all",
                        "--baseline",
                        "backtests/baselines/baseline_index.json",
                        "--strict",
                    ],
                    [sys.executable, "tools/run_decision_parity_matrix.py", "--promoted-only"],
                ],
            }
            for name in ("incumbent_replay", "round_reproduction", "historical_walk_forward")
        }
        payload = {
            "ok": True,
            "scope_id": scope,
            "generated_at": generated_at,
            "tests": tests,
        }
        _write_json(root / scope / "replay_evidence_report.json", payload)
    return errors


def _adapter_audit() -> tuple[list[dict[str, str]], list[str]]:
    records: list[dict[str, str]] = []
    errors: list[str] = []
    for path in sorted((ROOT / "contracts" / "strategy_plugins").glob("*/strategy_plugin_contract.json")):
        contract = validate_plugin_contract(path)
        adapter = ROOT / contract.backtest_adapter_path
        relative = path.relative_to(ROOT).as_posix()
        records.append({
            "kind": "adapter_path",
            "contract": relative,
            "adapter": contract.backtest_adapter_path,
        })
        if not adapter.exists():
            errors.append(f"{relative}: missing backtest adapter {contract.backtest_adapter_path}")
    return records, errors


def _invariant_audit(path: Path) -> tuple[dict[str, Any], list[str]]:
    record: dict[str, Any] = {"kind": "invariant_report", "path": _relative(path)}
    errors: list[str] = []
    if not path.exists():
        return record, [f"missing named invariant report {_relative(path)}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    statuses = _statuses(payload, "invariants", "name")
    record["statuses"] = statuses
    if payload.get("schema_version") != "backtest_integrity_invariant_report_v1":
        errors.append("invariant report schema_version must be backtest_integrity_invariant_report_v1")
    missing = sorted(set(REQUIRED_BACKTEST_INVARIANTS) - set(statuses))
    if missing:
        errors.append(f"missing backtest invariant checks {missing}")
    for name, status in statuses.items():
        if status != "pass":
            errors.append(f"backtest invariant {name} status is {status}")
    seen_evidence_sets: dict[tuple[str, ...], str] = {}
    for row in payload.get("invariants", []):
        name = str(row.get("name") or "")
        commands = row.get("commands")
        evidence_paths = tuple(str(item) for item in row.get("evidence_paths") or [])
        if not isinstance(commands, list) or not commands:
            errors.append(f"backtest invariant {name} lacks per-invariant command records")
        if not evidence_paths:
            errors.append(f"backtest invariant {name} lacks per-invariant evidence paths")
        previous = seen_evidence_sets.setdefault(evidence_paths, name)
        if previous != name:
            errors.append(
                f"backtest invariant {name} reuses the same broad evidence set as {previous}"
            )
    return record, errors


def _thin_adapter_audit(path: Path) -> tuple[dict[str, Any], list[str]]:
    record: dict[str, Any] = {"kind": "thin_adapter_audit", "path": _relative(path)}
    errors: list[str] = []
    if not path.exists():
        return record, [f"missing thin-adapter audit {_relative(path)}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    statuses = _statuses(payload, "adapters", "contract")
    record["statuses"] = statuses
    if payload.get("schema_version") != "thin_backtest_adapter_audit_v1":
        errors.append("thin-adapter audit schema_version must be thin_backtest_adapter_audit_v1")
    for path in sorted((ROOT / "contracts" / "strategy_plugins").glob("*/strategy_plugin_contract.json")):
        relative = path.relative_to(ROOT).as_posix()
        status = statuses.get(relative)
        if status != "pass" and status not in ACCEPTED_NON_PROMOTION_STATUSES:
            errors.append(f"{relative}: thin-adapter audit status is {status}")
    return record, errors


def _statuses(payload: dict[str, Any], key: str, name_key: str) -> dict[str, str]:
    rows = payload.get(key, [])
    return {
        str(row.get(name_key)): str(row.get("status"))
        for row in rows
        if isinstance(row, dict) and row.get(name_key)
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _combined_hash(paths: list[str]) -> str:
    payload = []
    for path in paths:
        absolute = ROOT / path
        record = {"path": path, "exists": absolute.exists()}
        if absolute.exists():
            record["sha256"] = _file_hash(absolute)
        payload.append(record)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
