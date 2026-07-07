"""Read-only live deployment metadata checks."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from trading_assistant_backtest.contract_models import MonthlyRunManifest, StrategyPluginContract
from trading_assistant_backtest.file_hashes import sha256_file


class DeploymentMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    bot_id: str
    strategy_id: str
    repo_url: str
    deployed_commit_sha: str
    config_hash: str
    strategy_version: str = ""
    config_version: str = ""
    telemetry_schema_version: str = ""
    deployment_id: str = ""
    strategy_plugin_contract_path: str = ""
    strategy_plugin_contract_hash: str = ""

    @model_validator(mode="after")
    def _normalize(self) -> DeploymentMetadata:
        for attr in (
            "bot_id",
            "strategy_id",
            "repo_url",
            "deployed_commit_sha",
            "config_hash",
            "strategy_version",
            "config_version",
            "telemetry_schema_version",
            "deployment_id",
            "strategy_plugin_contract_path",
            "strategy_plugin_contract_hash",
        ):
            setattr(self, attr, str(getattr(self, attr) or "").strip())
        return self


def load_deployment_metadata(path: str | Path) -> DeploymentMetadata:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DeploymentMetadata.model_validate(payload)


def deployment_metadata_errors(
    manifest: MonthlyRunManifest,
    contract: StrategyPluginContract | None = None,
) -> list[str]:
    path_text = str(getattr(manifest, "deployment_metadata_path", "") or "").strip()
    if not path_text:
        return []
    path = Path(path_text)
    if not path.exists():
        return [f"deployment metadata path is missing: {path}"]
    try:
        metadata = load_deployment_metadata(path)
    except Exception as exc:
        return [f"deployment metadata is invalid: {exc}"]

    errors = _required_errors(metadata)
    expected = {
        "bot_id": manifest.bot_id,
        "strategy_id": manifest.strategy_id,
        "deployment_id": manifest.deployment_id,
        "strategy_version": manifest.strategy_version,
        "config_version": manifest.config_version,
        "config_hash": getattr(manifest, "config_hash", ""),
    }
    for attr, value in expected.items():
        if value and getattr(metadata, attr) and getattr(metadata, attr) != value:
            errors.append(f"deployment metadata {attr} does not match run manifest")
    if manifest.trading_repo_commit_sha and (
        metadata.deployed_commit_sha != manifest.trading_repo_commit_sha
    ):
        errors.append("deployment metadata deployed_commit_sha does not match run manifest")
    if contract is not None and contract.live_repo_commit_sha:
        if metadata.deployed_commit_sha != contract.live_repo_commit_sha:
            errors.append("deployment metadata deployed_commit_sha does not match plugin contract")
    errors.extend(_contract_artifact_errors(metadata, manifest, path))
    return errors


def _required_errors(metadata: DeploymentMetadata) -> list[str]:
    missing = [
        attr
        for attr in (
            "bot_id",
            "strategy_id",
            "repo_url",
            "deployed_commit_sha",
            "config_hash",
            "strategy_version",
            "config_version",
            "telemetry_schema_version",
            "strategy_plugin_contract_path",
            "strategy_plugin_contract_hash",
        )
        if not getattr(metadata, attr)
    ]
    return ["deployment metadata missing required fields: " + ", ".join(missing)] if missing else []


def _contract_artifact_errors(
    metadata: DeploymentMetadata,
    manifest: MonthlyRunManifest,
    metadata_path: Path,
) -> list[str]:
    errors: list[str] = []
    declared_path = Path(metadata.strategy_plugin_contract_path)
    if not declared_path.is_absolute():
        declared_path = metadata_path.parent / declared_path
    if manifest.strategy_plugin_contract_path:
        manifest_path = Path(manifest.strategy_plugin_contract_path)
        if declared_path.resolve() != manifest_path.resolve():
            errors.append(
                "deployment metadata strategy_plugin_contract_path does not match run manifest"
            )
    if declared_path.exists():
        digest = sha256_file(declared_path)
        if digest != metadata.strategy_plugin_contract_hash:
            errors.append(
                "deployment metadata strategy_plugin_contract_hash does not match contract artifact"
            )
    else:
        errors.append("deployment metadata strategy_plugin_contract_path does not exist")
    return errors
