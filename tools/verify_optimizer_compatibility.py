from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "artifacts" / "validation" / "optimizer_compatibility"
DIMENSIONS = {
    "cumulative_mutations",
    "gate_decisions",
    "selected_candidates",
    "canonical_round_outputs",
}
SCOPES = {
    "ibkr": ("trading_stock_family", "trading_momentum_family", "trading_swing_family"),
    "crypto": ("crypto_trader_portfolio",),
    "k_stock": ("k_stock_olr_kalcb",),
}


def main() -> int:
    args = _parser().parse_args()
    no_drift = _run_no_drift(args.bot)
    records: list[dict[str, Any]] = [no_drift]
    errors = [] if no_drift["returncode"] == 0 else ["latest-round no-drift prerequisite failed"]
    for scope in _selected_scopes(args.bot):
        record, scope_errors = _evidence_record(scope, args.fixture_set)
        records.append(record)
        errors.extend(scope_errors)
    result = {
        "valid": not errors,
        "bot": args.bot,
        "fixture_set": args.fixture_set,
        "check": "optimizer_runner_equivalence",
        "records": records,
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify optimizer runner equivalence evidence.")
    parser.add_argument("--bot", choices=["all", "ibkr", "crypto", "k_stock"], default="all")
    parser.add_argument("--fixture-set", default="smoke")
    return parser


def _run_no_drift(bot: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "tools/verify_latest_round_no_drift.py",
        "--bot",
        bot,
        "--baseline",
        "backtests/baselines/baseline_index.json",
        "--strict",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "scope": "latest_round_no_drift_prerequisite",
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-20:],
        "stderr_tail": completed.stderr.splitlines()[-20:],
    }


def _selected_scopes(bot: str) -> list[str]:
    selected = SCOPES if bot == "all" else {bot: SCOPES[bot]}
    return [scope for scopes in selected.values() for scope in scopes]


def _evidence_record(scope: str, fixture_set: str) -> tuple[dict[str, Any], list[str]]:
    path = EVIDENCE_ROOT / f"{scope}.{fixture_set}.json"
    generation = _generate_evidence(scope, fixture_set)
    record: dict[str, Any] = {
        "scope": scope,
        "evidence_path": _relative(path),
        "generation": generation,
    }
    errors: list[str] = []
    if generation["returncode"] != 0:
        errors.append(f"{scope}: optimizer equivalence fixture generation failed")
    if not path.exists():
        errors.append(f"{scope}: missing optimizer runner equivalence evidence {_relative(path)}")
        record["status"] = "missing"
        return record, errors
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{scope}: invalid optimizer equivalence JSON: {exc}")
        record["status"] = "invalid"
        return record, errors
    record.update({
        "status": payload.get("status", ""),
        "legacy_runner": payload.get("legacy_runner", ""),
        "adapter_runner": payload.get("adapter_runner", ""),
        "wrapped_legacy_runners": payload.get("wrapped_legacy_runners", []),
        "dimensions": _dimension_statuses(payload),
    })
    if payload.get("schema_version") != "optimizer_runner_equivalence_matrix_v1":
        errors.append(f"{scope}: schema_version must be optimizer_runner_equivalence_matrix_v1")
    if payload.get("scope_id") != scope:
        errors.append(f"{scope}: scope_id mismatch")
    if payload.get("fixture_set") != fixture_set:
        errors.append(f"{scope}: fixture_set mismatch")
    if not payload.get("legacy_runner") or not payload.get("adapter_runner"):
        errors.append(f"{scope}: legacy_runner and adapter_runner are required")
    wrapped = payload.get("wrapped_legacy_runners")
    if not isinstance(wrapped, list) or not wrapped:
        errors.append(f"{scope}: wrapped_legacy_runners are required")
    else:
        for item in wrapped:
            source_path = ROOT / str(item.get("source_path", ""))
            if not source_path.exists():
                errors.append(f"{scope}: wrapped legacy runner source is missing: {item.get('source_path')}")
    statuses = _dimension_statuses(payload)
    missing = sorted(DIMENSIONS - set(statuses))
    if missing:
        errors.append(f"{scope}: missing optimizer equivalence dimensions {missing}")
    for dimension, status in statuses.items():
        if status != "pass":
            errors.append(f"{scope}: {dimension} status is {status}")
    if payload.get("status") != "pass":
        errors.append(f"{scope}: optimizer equivalence status is {payload.get('status')}")
    if "Archived" in str(payload.get("adapter_runner", "")):
        errors.append(f"{scope}: archived-output smoke adapter cannot satisfy runner equivalence")
    execution = payload.get("execution_evidence")
    if not isinstance(execution, dict):
        errors.append(f"{scope}: missing same-input legacy/adapter execution_evidence")
    else:
        for key in ("legacy_command", "adapter_command", "input_hashes", "output_hashes", "compared_payloads"):
            if not execution.get(key):
                errors.append(f"{scope}: execution_evidence missing {key}")
    for item in payload.get("comparisons", []):
        notes = str(item.get("notes") or "").lower() if isinstance(item, dict) else ""
        if "frozen legacy optimizer output" in notes or "archived-smoke" in notes:
            errors.append(f"{scope}: archived/frozen-output comparison is preflight only")
    return record, errors


def _generate_evidence(scope: str, fixture_set: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "tools/run_optimizer_equivalence_fixture.py",
        "--scope",
        scope,
        "--fixture-set",
        fixture_set,
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-20:],
        "stderr_tail": completed.stderr.splitlines()[-20:],
    }


def _dimension_statuses(payload: dict[str, Any]) -> dict[str, str]:
    raw = payload.get("comparisons", [])
    if isinstance(raw, dict):
        return {str(key): str(value.get("status") if isinstance(value, dict) else value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {
            str(item.get("dimension")): str(item.get("status"))
            for item in raw
            if isinstance(item, dict) and item.get("dimension")
        }
    return {}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
