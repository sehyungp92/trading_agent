"""Validate and install live-emitted deployment metadata artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.validation.approval_grade_audit import CONTRACT_PATHS
from trading_assistant_backtest.validation.deployment_metadata_contract import (
    live_deployment_metadata_errors,
    telemetry_schema_contract_errors,
)
from trading_assistant_backtest.paths import monorepo_root, package_root, resolve_workspace_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a runtime-emitted deployment_metadata.json and optionally install it "
            "beside the matching strategy_plugin_contract.json."
        )
    )
    parser.add_argument("--agent-root", type=Path, default=_default_agent_root())
    parser.add_argument("--bridge-id", required=True, choices=sorted(CONTRACT_PATHS))
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=_default_artifact_root())
    parser.add_argument(
        "--install",
        action="store_true",
        help="Replace contracts/<bridge-id>/deployment_metadata.json when validation passes.",
    )
    args = parser.parse_args(argv)

    report = validate_and_maybe_install_deployment_metadata(
        agent_root=args.agent_root,
        bridge_id=args.bridge_id,
        metadata_path=args.metadata,
        artifact_root=args.artifact_root,
        install=args.install,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def validate_and_maybe_install_deployment_metadata(
    *,
    agent_root: Path,
    bridge_id: str,
    metadata_path: Path,
    artifact_root: Path,
    install: bool = False,
) -> dict[str, Any]:
    agent_root = Path(agent_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(metadata_path).resolve()
    contract_dir = resolve_workspace_path(agent_root, CONTRACT_PATHS[bridge_id])
    contract_path = contract_dir / "strategy_plugin_contract.json"
    installed_path = contract_dir / "deployment_metadata.json"

    checks: list[dict[str, Any]] = []
    contract = _read_json(contract_path)
    metadata = _read_json(metadata_path)
    checks.append(
        _check(
            "contract_present",
            bool(contract),
            [f"missing: {contract_path}"] if not contract else [],
        )
    )
    checks.append(
        _check(
            "metadata_present",
            bool(metadata),
            [f"missing: {metadata_path}"] if not metadata else [],
        )
    )

    if contract and metadata:
        checks.extend(
            _metadata_checks(
                contract=contract,
                metadata=metadata,
                contract_path=contract_path,
            )
        )

    ok = all(check["passed"] for check in checks)
    installed = False
    if ok and install:
        installed_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata_path, installed_path)
        installed = True

    report_path = artifact_root / "deployment_metadata_install_report.json"
    report = {
        "schema_version": "deployment_metadata_install_report_v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "bridge_id": bridge_id,
        "metadata_path": str(metadata_path),
        "contract_path": str(contract_path),
        "installed_path": str(installed_path),
        "install_requested": install,
        "installed": installed,
        "ok": ok,
        "checks": checks,
        "artifact_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _metadata_checks(
    *,
    contract: dict[str, Any],
    metadata: dict[str, Any],
    contract_path: Path,
) -> list[dict[str, Any]]:
    live_errors = live_deployment_metadata_errors(metadata)
    contract_hash = sha256_file(contract_path)
    telemetry_errors = telemetry_schema_contract_errors(metadata, contract)
    repo_url = str(metadata.get("repo_url") or "")
    repo_url_ok = bool(repo_url) and not repo_url.startswith("local://")
    return [
        _check("live_emission_provenance", not live_errors, live_errors),
        _check(
            "repo_url_non_local",
            repo_url_ok,
            [] if repo_url_ok else [f"repo_url is local/shadow-only: {repo_url!r}"],
        ),
        _check(
            "deployed_sha_matches_contract",
            metadata.get("deployed_commit_sha") == contract.get("live_repo_commit_sha"),
            []
            if metadata.get("deployed_commit_sha") == contract.get("live_repo_commit_sha")
            else ["deployed_commit_sha does not match contract live_repo_commit_sha"],
        ),
        _check(
            "contract_hash_matches",
            metadata.get("strategy_plugin_contract_hash") == contract_hash,
            []
            if metadata.get("strategy_plugin_contract_hash") == contract_hash
            else ["strategy_plugin_contract_hash does not match local contract file"],
        ),
        _check(
            "config_hash_present",
            bool(str(metadata.get("config_hash") or "").strip()),
            [] if metadata.get("config_hash") else ["config_hash missing"],
        ),
        _check(
            "telemetry_schema_matches_contract",
            not telemetry_errors,
            telemetry_errors,
        ),
    ]


def _check(name: str, passed: bool, errors: list[str]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "errors": errors}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _default_artifact_root() -> Path:
    return package_root() / "artifacts" / "validation" / "deployment_metadata_install"


def _default_agent_root() -> Path:
    return monorepo_root()


def _repo_root() -> Path:
    return package_root()


if __name__ == "__main__":
    raise SystemExit(main())
