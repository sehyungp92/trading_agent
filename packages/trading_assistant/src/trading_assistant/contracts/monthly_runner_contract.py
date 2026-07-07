"""Shared monthly runner artifact contract validator."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from trading_assistant.schemas.backtest_artifacts import BacktestArtifactIndex, missing_required_artifact_keys
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest, MonthlyRunMode
from trading_assistant.skills.backtest_runner_client import BacktestRunnerClient
from trading_assistant.skills.monthly_optimizer_runner import MonthlyOptimizerRunner


@dataclass(frozen=True)
class MonthlyRunnerContractValidation:
    manifest_path: str
    artifact_index_path: str
    valid: bool
    errors: list[str] = field(default_factory=list)


def validate_manifest_file(manifest_path: Path) -> MonthlyRunnerContractValidation:
    manifest_path = Path(manifest_path)
    manifest = MonthlyRunManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    index_path = Path(manifest.artifact_root) / "artifact_index.json"
    if not index_path.exists():
        return MonthlyRunnerContractValidation(
            manifest_path=str(manifest_path),
            artifact_index_path=str(index_path),
            valid=False,
            errors=["missing artifact_index.json"],
        )
    try:
        raw_index = json.loads(index_path.read_text(encoding="utf-8"))
        missing_keys = missing_required_artifact_keys(raw_index)
        if missing_keys:
            raise ValueError(f"artifact index missing required keys: {', '.join(missing_keys)}")
        index = BacktestArtifactIndex.model_validate(raw_index)
    except Exception as exc:
        return MonthlyRunnerContractValidation(
            manifest_path=str(manifest_path),
            artifact_index_path=str(index_path),
            valid=False,
            errors=[f"malformed artifact_index.json: {exc}"],
        )
    errors = validate_manifest_artifacts(manifest, index, manifest_path=manifest_path)
    return MonthlyRunnerContractValidation(
        manifest_path=str(manifest_path),
        artifact_index_path=str(index_path),
        valid=not errors,
        errors=errors,
    )


def validate_manifest_artifacts(
    manifest: MonthlyRunManifest,
    artifact_index: BacktestArtifactIndex,
    *,
    manifest_path: Path | None = None,
) -> list[str]:
    errors = [
        *BacktestRunnerClient.validate_artifact_contract(
            manifest,
            artifact_index,
            manifest_path=Path(manifest_path or Path(manifest.artifact_root) / "run_manifest.json"),
        ),
    ]
    if manifest.mode in {
        MonthlyRunMode.PHASED_AUTO,
        MonthlyRunMode.SMOKE_REPAIR,
        MonthlyRunMode.STRUCTURAL_REVIEW,
    }:
        sequence = MonthlyOptimizerRunner().validate_artifacts(
            manifest,
            artifact_index,
            manifest_path=manifest_path,
        )
        errors.extend(sequence.blocking_reasons)
    return list(dict.fromkeys(error for error in errors if error))
