"""Verify materialized effective live config artifacts against current files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trading_contracts.canonical import canonical_json_sha256, file_sha256, load_json

from trading_config.generator import BOT_SPECS, build_effective_snapshot
from trading_config.models import CONFIG_MERGE_ORDER, EffectiveConfigSnapshot


def verify_effective_configs(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    errors: list[dict[str, str]] = []
    records: list[dict[str, str]] = []
    for spec in BOT_SPECS:
        path = root / spec.generated_path
        if not path.exists():
            errors.append({"bot_id": spec.bot_id, "path": spec.generated_path, "error": "missing artifact"})
            continue
        try:
            snapshot = EffectiveConfigSnapshot.model_validate(load_json(path))
            _verify_snapshot(root, snapshot)
            expected = build_effective_snapshot(
                root,
                spec,
                sorted((root / spec.canonical_dir).glob("*.json")),
            )
            _verify_materialized(snapshot, expected)
        except Exception as exc:
            errors.append({"bot_id": spec.bot_id, "path": spec.generated_path, "error": str(exc)})
            continue
        records.append({
            "bot_id": spec.bot_id,
            "path": spec.generated_path,
            "effective_config_hash": snapshot.effective_config_hash,
            "materialized_config_hash": snapshot.materialized_config_hash,
        })
    return {
        "valid": not errors,
        "validated_count": len(records),
        "error_count": len(errors),
        "records": records,
        "errors": errors,
    }


def _verify_snapshot(root: Path, snapshot: EffectiveConfigSnapshot) -> None:
    source_records = [record.model_dump(mode="json") for record in snapshot.source_files]
    promotion_records = [record.model_dump(mode="json") for record in snapshot.promotion_manifests]
    for record in source_records:
        path = root / record["path"]
        actual = file_sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"source file hash drift for {record['path']}")
        if record.get("canonical_json_sha256"):
            canonical = canonical_json_sha256(load_json(path))
            if canonical != record["canonical_json_sha256"]:
                raise ValueError(f"source canonical JSON drift for {record['path']}")
    for record in promotion_records:
        path = root / record["path"]
        actual = file_sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"promotion manifest hash drift for {record['path']}")
        canonical = canonical_json_sha256(load_json(path))
        if canonical != record["canonical_json_sha256"]:
            raise ValueError(f"promotion canonical JSON drift for {record['path']}")
    recomputed = canonical_json_sha256({
        "bot_id": snapshot.bot_id,
        "merge_order": CONFIG_MERGE_ORDER,
        "source_files": source_records,
        "promotion_manifests": promotion_records,
        "materialized_config_hash": snapshot.materialized_config_hash,
    })
    if recomputed != snapshot.effective_config_hash:
        raise ValueError(f"effective config hash drift for {snapshot.bot_id}")


def _verify_materialized(snapshot: EffectiveConfigSnapshot, expected: EffectiveConfigSnapshot) -> None:
    if snapshot.materialized_config_hash != expected.materialized_config_hash:
        raise ValueError(f"materialized config hash drift for {snapshot.bot_id}")
    if snapshot.materialized_config != expected.materialized_config:
        raise ValueError(f"materialized config payload drift for {snapshot.bot_id}")
    if snapshot.effective_config_hash != expected.effective_config_hash:
        raise ValueError(f"effective config hash drift for {snapshot.bot_id}")
