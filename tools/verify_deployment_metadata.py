from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from migration_support import ROOT, file_sha256, read_json

for src in (
    ROOT / "packages" / "trading_deployment" / "src",
    ROOT / "packages" / "trading_assistant_backtest" / "src",
):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from trading_deployment.metadata import FAIL_CLOSED_CHECKS, combined_artifact_hash  # noqa: E402
from trading_assistant_backtest.validation.deployment_metadata_contract import (  # noqa: E402
    live_deployment_metadata_errors,
    telemetry_schema_contract_errors,
)


BOTS = {
    "ibkr": {
        "effective": "deployments/ibkr/generated/strategies.effective.json",
        "dependency_report": "deployments/ibkr/generated/dependency_report.json",
        "runtime_metadata": "deployments/ibkr/generated/runtime_deployment_metadata",
        "promotions": "contracts/promotions/ibkr",
        "contracts": ("trading_momentum_family", "trading_stock_family", "trading_swing_family"),
        "runtime_command_marker": "ibkr_trading.runtime",
    },
    "crypto": {
        "effective": "deployments/crypto/generated/live_config.effective.json",
        "dependency_report": "deployments/crypto/generated/dependency_report.json",
        "runtime_metadata": "deployments/crypto/generated/runtime_deployment_metadata",
        "promotions": "contracts/promotions/crypto",
        "contracts": ("crypto_breakout_v1", "crypto_momentum_v1", "crypto_trend_v1"),
        "runtime_command_marker": "crypto_trader.cli",
    },
    "k_stock": {
        "effective": "deployments/k_stock/generated/olr_kalcb.effective.json",
        "dependency_report": "deployments/k_stock/generated/dependency_report.json",
        "runtime_metadata": "deployments/k_stock/generated/runtime_deployment_metadata",
        "promotions": "contracts/promotions/k_stock",
        "contracts": ("k_stock_olr_kalcb",),
        "runtime_command_marker": "k_stock_trader.olr_kalcb_runtime",
    },
}
def main() -> int:
    args = _parser().parse_args()
    generation = _generate_runtime_metadata(args.bot)
    selected = list(BOTS) if args.bot == "all" else [args.bot]
    records: list[dict[str, Any]] = [{"kind": "runtime_metadata_orchestration", **generation}]
    if generation["returncode"] != 0:
        errors = [
            "runtime deployment metadata emission failed; existing generated metadata was not validated because it may be stale"
        ]
        result = {"valid": False, "records": records, "errors": errors}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    errors: list[str] = []
    for bot in selected:
        record, bot_errors = _record(bot, BOTS[bot])
        records.append(record)
        errors.extend(bot_errors)
    result = {"valid": not errors, "records": records, "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify deployment metadata inputs.")
    parser.add_argument("--bot", choices=["all", *BOTS], default="all")
    return parser


def _generate_runtime_metadata(bot: str) -> dict[str, Any]:
    command = [sys.executable, "tools/run_runtime_deployment_metadata_matrix.py", "--bot", bot]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-20:],
        "stderr_tail": completed.stderr.splitlines()[-20:],
    }


def _record(bot: str, spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    effective_path = ROOT / spec["effective"]
    report_path = ROOT / spec["dependency_report"]
    promotion_paths = sorted((ROOT / spec["promotions"]).glob("*.json"))
    contract_paths = [
        ROOT / "contracts" / "strategy_plugins" / contract / "strategy_plugin_contract.json"
        for contract in spec["contracts"]
    ]
    if not effective_path.exists():
        errors.append(f"{bot}: missing effective config artifact")
        effective = {}
    else:
        effective = read_json(effective_path)
    if not report_path.exists():
        errors.append(f"{bot}: missing image dependency report")
        dependency_report = {}
    else:
        dependency_report = read_json(report_path)
    if not promotion_paths:
        errors.append(f"{bot}: missing promotion manifests")
    for path in contract_paths:
        if not path.exists():
            errors.append(f"{bot}: missing strategy plugin contract {path.relative_to(ROOT).as_posix()}")
    materialized_hash = str(effective.get("materialized_config_hash") or "")
    metadata_records = []
    for path in contract_paths:
        if not path.exists():
            continue
        metadata_record, metadata_errors = _metadata_record(
            bot,
            path,
            ROOT / spec["runtime_metadata"],
            materialized_hash,
            spec["runtime_command_marker"],
            promotion_hashes={
                path.relative_to(ROOT).as_posix(): file_sha256(path) for path in promotion_paths
            },
        )
        metadata_records.append(metadata_record)
        errors.extend(metadata_errors)
    if dependency_report.get("assistant_packages_present") is True:
        errors.append(f"{bot}: dependency report contains assistant control-plane/backtest packages")
    if not materialized_hash:
        errors.append(f"{bot}: effective config artifact lacks materialized_config_hash")
    return (
        {
            "bot": bot,
            "effective_config": spec["effective"],
            "materialized_config_hash": materialized_hash,
            "promotion_hashes": {
                path.relative_to(ROOT).as_posix(): file_sha256(path) for path in promotion_paths
            },
            "contract_hashes": {
                path.relative_to(ROOT).as_posix(): file_sha256(path)
                for path in contract_paths
                if path.exists()
            },
            "dependency_report": spec["dependency_report"],
            "runtime_metadata": metadata_records,
        },
        errors,
    )


def _metadata_record(
    bot: str,
    contract_path: Path,
    runtime_metadata_root: Path,
    materialized_hash: str,
    runtime_command_marker: str,
    *,
    promotion_hashes: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    contract_id = contract_path.parent.name
    metadata_path = runtime_metadata_root / contract_id / "deployment_metadata.json"
    relative = metadata_path.relative_to(ROOT).as_posix()
    errors: list[str] = []
    if not metadata_path.exists():
        return {"path": relative, "valid": False}, [f"{bot}: missing runtime deployment metadata {relative}"]
    metadata = read_json(metadata_path)
    errors.extend(_metadata_validation_errors(
        bot,
        relative,
        metadata,
        contract_path,
        materialized_hash,
        promotion_hashes,
        runtime_command_marker,
    ))
    probe_records, probe_errors = _fail_closed_probe_records(
        bot,
        relative,
        metadata,
        contract_path,
        materialized_hash,
        promotion_hashes,
        runtime_command_marker,
    )
    errors.extend(probe_errors)
    return (
        {
            "path": relative,
            "valid": not errors,
            "metadata_source": metadata.get("metadata_source", ""),
            "repo_url": metadata.get("repo_url", ""),
            "runtime_emission": _runtime_emission_record(metadata),
            "fail_closed_probes": probe_records,
        },
        errors,
    )


def _runtime_emission_record(metadata: dict[str, Any]) -> dict[str, Any]:
    emission = metadata.get("runtime_emission")
    if not isinstance(emission, dict):
        return {}
    return {
        "source": emission.get("source", ""),
        "command": emission.get("command", []),
        "returncode": emission.get("returncode"),
        "stdout_tail": emission.get("stdout_tail", []),
        "stderr_tail": emission.get("stderr_tail", []),
        "metadata_path": emission.get("metadata_path", ""),
        "raw_runtime_artifact_hash": emission.get("raw_runtime_artifact_hash", ""),
    }


def _metadata_validation_errors(
    bot: str,
    relative: str,
    metadata: dict[str, Any],
    contract_path: Path,
    materialized_hash: str,
    promotion_hashes: dict[str, str],
    runtime_command_marker: str,
) -> list[str]:
    errors: list[str] = []
    errors.extend(f"{bot}: {relative}: {error}" for error in live_deployment_metadata_errors(metadata))
    contract = read_json(contract_path)
    errors.extend(
        f"{bot}: {relative}: {error}"
        for error in telemetry_schema_contract_errors(metadata, contract)
    )
    expected_contract_hash = file_sha256(contract_path)
    if metadata.get("strategy_plugin_contract_hash") != expected_contract_hash:
        errors.append(f"{bot}: {relative}: strategy_plugin_contract_hash does not match contract")
    declared_config_hash = str(metadata.get("materialized_config_hash") or metadata.get("config_hash") or "")
    if materialized_hash and declared_config_hash != materialized_hash:
        errors.append(f"{bot}: {relative}: materialized config hash does not match effective config")
    expected_promotion_hash = combined_artifact_hash(promotion_hashes)
    if metadata.get("promotion_hash") != expected_promotion_hash:
        errors.append(f"{bot}: {relative}: promotion_hash does not match promotion manifests")
    if metadata.get("paper_live_fail_closed") is not True:
        errors.append(f"{bot}: {relative}: paper_live_fail_closed must be true")
    fail_closed_checks = set(metadata.get("fail_closed_checks") or [])
    missing_checks = sorted(set(FAIL_CLOSED_CHECKS) - fail_closed_checks)
    if missing_checks:
        errors.append(f"{bot}: {relative}: fail_closed_checks missing {missing_checks}")
    for field in ("image_version", "promotion_hash", "runtime_entrypoint"):
        if not str(metadata.get(field) or "").strip():
            errors.append(f"{bot}: {relative}: missing {field}")
    errors.extend(_runtime_emission_errors(bot, relative, metadata, runtime_command_marker))
    return errors


def _fail_closed_probe_records(
    bot: str,
    relative: str,
    metadata: dict[str, Any],
    contract_path: Path,
    materialized_hash: str,
    promotion_hashes: dict[str, str],
    runtime_command_marker: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    mutations = {
        "missing_image_version": {"image_version": ""},
        "stale_materialized_config_hash": {
            "materialized_config_hash": "stale",
            "config_hash": "stale",
        },
        "stale_promotion_hash": {"promotion_hash": "stale"},
        "stale_strategy_plugin_contract_hash": {"strategy_plugin_contract_hash": "stale"},
        "paper_live_fail_closed_false": {"paper_live_fail_closed": False},
        "dirty_worktree_flag": {"source_control_worktree_clean": False},
        "missing_telemetry_schemas": {
            "telemetry_schema_version": "",
            "telemetry_schema_versions": [],
        },
        "helper_runtime_emission": {
            "runtime_emission": {
                "source": "tools/run_runtime_deployment_metadata_matrix.py",
                "command": [sys.executable, "tools/run_runtime_deployment_metadata_matrix.py"],
                "returncode": 0,
                "metadata_path": relative,
            }
        },
    }
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for name, update in mutations.items():
        mutated = {**metadata, **update}
        probe_errors = _metadata_validation_errors(
            bot,
            relative,
            mutated,
            contract_path,
            materialized_hash,
            promotion_hashes,
            runtime_command_marker,
        )
        records.append({"mutation": name, "status": "pass" if probe_errors else "fail"})
        if not probe_errors:
            errors.append(f"{bot}: {relative}: fail-closed probe did not reject {name}")
    return records, errors


def _runtime_emission_errors(
    bot: str,
    relative: str,
    metadata: dict[str, Any],
    runtime_command_marker: str,
) -> list[str]:
    emission = metadata.get("runtime_emission")
    if not isinstance(emission, dict):
        return [f"{bot}: {relative}: missing runtime_emission startup evidence"]
    source = str(emission.get("source") or "")
    if "generate_runtime_deployment_metadata.py" in source or source.startswith("tools/"):
        return [f"{bot}: {relative}: runtime_emission source is local generator, not runtime startup"]
    command = emission.get("command")
    if not isinstance(command, list) or not command:
        return [f"{bot}: {relative}: runtime_emission missing startup command"]
    joined_command = " ".join(str(part) for part in command)
    if (
        "tools/run_runtime_deployment_metadata_matrix.py" in joined_command
        or "tools/generate_runtime_deployment_metadata.py" in joined_command
    ):
        return [f"{bot}: {relative}: runtime_emission command is a local helper, not bot startup"]
    if runtime_command_marker not in joined_command:
        return [f"{bot}: {relative}: runtime_emission command does not invoke {runtime_command_marker}"]
    if emission.get("returncode") != 0:
        return [f"{bot}: {relative}: runtime_emission command did not pass"]
    if str(emission.get("metadata_path") or "") != relative:
        return [f"{bot}: {relative}: runtime_emission metadata_path mismatch"]
    return []


def _worktree_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == ""

if __name__ == "__main__":
    raise SystemExit(main())
