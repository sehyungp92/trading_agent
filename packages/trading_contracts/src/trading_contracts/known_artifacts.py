"""Repository-wide validation for known migration contract artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from trading_contracts.legacy import (
    validate_deployment_manifest,
    validate_plugin_contract,
    validate_promotion_manifest,
    validate_rounds_manifest,
)


Validator = Callable[[str | Path], BaseModel]


def known_artifact_targets(repo_root: str | Path) -> list[tuple[str, Path, Validator]]:
    root = Path(repo_root)
    targets: list[tuple[str, Path, Validator]] = []

    for path in sorted((root / "backtests" / "baselines").rglob("rounds_manifest.json")):
        targets.append(("rounds_manifest", path, validate_rounds_manifest))

    for path in sorted((root / "contracts" / "strategy_plugins").rglob("strategy_plugin_contract.json")):
        targets.append(("strategy_plugin_contract", path, validate_plugin_contract))

    for path in sorted((root / "contracts" / "promotions").rglob("*.json")):
        targets.append(("promotion_manifest", path, validate_promotion_manifest))

    deployment_manifest = (
        root / "backtests" / "baselines" / "crypto" / "portfolio" / "round_3" / "deployment_manifest.json"
    )
    if deployment_manifest.exists():
        targets.append(("deployment_manifest", deployment_manifest, validate_deployment_manifest))

    return targets


def validate_known_artifacts(repo_root: str | Path) -> dict:
    root = Path(repo_root).resolve()
    records: list[dict] = []
    errors: list[dict] = []
    for role, path, validator in known_artifact_targets(root):
        try:
            model = validator(path)
        except Exception as exc:
            errors.append({
                "role": role,
                "path": str(path.relative_to(root)),
                "error": str(exc),
            })
            continue
        records.append({
            "role": role,
            "path": str(path.relative_to(root)),
            "model": type(model).__name__,
        })
    return {
        "valid": not errors,
        "repo_root": str(root),
        "validated_count": len(records),
        "error_count": len(errors),
        "records": records,
        "errors": errors,
    }
