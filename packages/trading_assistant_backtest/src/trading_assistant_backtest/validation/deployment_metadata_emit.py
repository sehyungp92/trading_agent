"""Emit runtime deployment metadata from a live bot checkout.

This helper is intentionally read-only: it inspects source-control state,
hashes the pinned strategy-plugin contract and config, and writes the metadata
artifact consumed by the approval audit. It does not install or promote the
artifact; ``deployment_metadata_install`` owns that fail-closed step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.validation.deployment_metadata_contract import (
    APPROVAL_EMISSION_ENVIRONMENTS,
    APPROVAL_METADATA_SOURCES,
    live_deployment_metadata_errors,
    telemetry_schema_contract_errors,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit approval-grade runtime deployment_metadata.json from a live checkout."
    )
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bot-id", required=True)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--strategy-version", required=True)
    parser.add_argument("--config-version", required=True)
    parser.add_argument("--telemetry-schema-version", required=True)
    parser.add_argument("--runtime-entrypoint", required=True)
    parser.add_argument("--runtime-instance-id", required=True)
    parser.add_argument("--deployment-id", default="")
    parser.add_argument("--repo-url", default="")
    parser.add_argument(
        "--metadata-source",
        choices=sorted(APPROVAL_METADATA_SOURCES),
        default="live_bot_runtime_deployment_metadata_v1",
    )
    parser.add_argument(
        "--emission-environment",
        choices=sorted(APPROVAL_EMISSION_ENVIRONMENTS),
        default="live_bot",
    )
    parser.add_argument("--emission-context", default="runtime_startup")
    parser.add_argument("--live-runtime-started-at-utc", default="")
    parser.add_argument("--runtime-host-fingerprint", default="")
    parser.add_argument(
        "--contract-path-in-metadata",
        default="strategy_plugin_contract.json",
        help=(
            "Path stored in deployment metadata. Use a relative path such as "
            "strategy_plugin_contract.json when the metadata will be installed beside the contract."
        ),
    )
    args = parser.parse_args(argv)

    report = emit_runtime_deployment_metadata(
        repo_path=args.repo_path,
        contract_path=args.contract,
        config_path=args.config,
        output_path=args.output,
        bot_id=args.bot_id,
        strategy_id=args.strategy_id,
        strategy_version=args.strategy_version,
        config_version=args.config_version,
        telemetry_schema_version=args.telemetry_schema_version,
        runtime_entrypoint=args.runtime_entrypoint,
        runtime_instance_id=args.runtime_instance_id,
        deployment_id=args.deployment_id,
        repo_url=args.repo_url,
        metadata_source=args.metadata_source,
        emission_environment=args.emission_environment,
        emission_context=args.emission_context,
        live_runtime_started_at_utc=args.live_runtime_started_at_utc,
        runtime_host_fingerprint=args.runtime_host_fingerprint,
        contract_path_in_metadata=args.contract_path_in_metadata,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def emit_runtime_deployment_metadata(
    *,
    repo_path: Path,
    contract_path: Path,
    config_path: Path,
    output_path: Path,
    bot_id: str,
    strategy_id: str,
    strategy_version: str,
    config_version: str,
    telemetry_schema_version: str,
    runtime_entrypoint: str,
    runtime_instance_id: str,
    deployment_id: str = "",
    repo_url: str = "",
    metadata_source: str = "live_bot_runtime_deployment_metadata_v1",
    emission_environment: str = "live_bot",
    emission_context: str = "runtime_startup",
    live_runtime_started_at_utc: str = "",
    runtime_host_fingerprint: str = "",
    contract_path_in_metadata: str = "strategy_plugin_contract.json",
) -> dict[str, Any]:
    repo_path = Path(repo_path).resolve()
    contract_path = Path(contract_path).resolve()
    config_path = Path(config_path).resolve()
    output_path = Path(output_path).resolve()
    emitted_at = _utc_now()
    runtime_started = live_runtime_started_at_utc or emitted_at
    contract = _read_json(contract_path)
    source_control_origin = repo_url or _git_text(repo_path, "config", "--get", "remote.origin.url")
    source_control_commit_sha = _git_text(repo_path, "rev-parse", "HEAD")
    worktree_status = _git_text(repo_path, "status", "--porcelain")
    clean = worktree_status == ""
    metadata = {
        "metadata_source": metadata_source,
        "emission_environment": emission_environment,
        "emission_context": emission_context,
        "emitted_at_utc": emitted_at,
        "live_runtime_started_at_utc": runtime_started,
        "runtime_entrypoint": runtime_entrypoint,
        "runtime_instance_id": runtime_instance_id,
        "runtime_host_fingerprint": runtime_host_fingerprint or _host_fingerprint(),
        "bot_id": bot_id,
        "strategy_id": strategy_id,
        "repo_url": source_control_origin,
        "source_control_origin": source_control_origin,
        "source_control_commit_sha": source_control_commit_sha,
        "source_control_worktree_clean": clean,
        "deployed_commit_sha": source_control_commit_sha,
        "config_hash": sha256_file(config_path, missing_ok=True),
        "strategy_version": strategy_version,
        "config_version": config_version,
        "telemetry_schema_version": telemetry_schema_version,
        "telemetry_schema_versions": [telemetry_schema_version] if telemetry_schema_version else [],
        "deployment_id": deployment_id,
        "assistant_lineage": {
            "weekly_signal_ids": [],
            "monthly_search_brief_id": "",
            "proposal_ids": [],
            "suggestion_ids": [],
            "hypothesis_ids": [],
            "experiment_id": "",
            "variant_id": "",
            "parameter_set_id": "",
            "deployment_id": deployment_id,
            "strategy_change_record_ids": [],
            "monthly_outcome_id": "",
        },
        "strategy_plugin_contract_path": contract_path_in_metadata,
        "strategy_plugin_contract_hash": sha256_file(contract_path, missing_ok=True),
        "dry_run": False,
    }
    checks = _emission_checks(
        metadata=metadata,
        contract=contract,
        contract_path=contract_path,
        config_path=config_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "schema_version": "runtime_deployment_metadata_emit_report_v1",
        "generated_at": emitted_at,
        "metadata_path": str(output_path),
        "contract_path": str(contract_path),
        "config_path": str(config_path),
        "repo_path": str(repo_path),
        "ok": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _emission_checks(
    *,
    metadata: dict[str, Any],
    contract: dict[str, Any],
    contract_path: Path,
    config_path: Path,
) -> list[dict[str, Any]]:
    live_errors = live_deployment_metadata_errors(metadata)
    telemetry_errors = telemetry_schema_contract_errors(metadata, contract)
    return [
        _check("metadata_live_emission_contract", not live_errors, live_errors),
        _check(
            "contract_present",
            bool(contract),
            [f"missing or malformed contract: {contract_path}"] if not contract else [],
        ),
        _check(
            "config_present",
            config_path.exists(),
            [f"missing config: {config_path}"] if not config_path.exists() else [],
        ),
        _check(
            "deployed_sha_matches_contract",
            metadata.get("deployed_commit_sha") == contract.get("live_repo_commit_sha"),
            []
            if metadata.get("deployed_commit_sha") == contract.get("live_repo_commit_sha")
            else ["deployed commit does not match strategy plugin contract live_repo_commit_sha"],
        ),
        _check(
            "telemetry_schema_matches_contract",
            not telemetry_errors,
            telemetry_errors,
        ),
    ]


def _check(name: str, passed: bool, errors: list[str]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "errors": errors}


def _git_text(repo_path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _host_fingerprint() -> str:
    raw = platform.node() or "unknown-host"
    return "host-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
