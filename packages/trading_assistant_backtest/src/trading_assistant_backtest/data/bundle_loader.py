"""Data bundle manifest loading and coverage emission."""

from __future__ import annotations

import json
from pathlib import Path

from trading_assistant_backtest.contract_models import (
    DataBundleManifest,
    DataBundleStatus,
    MonthlyRunManifest,
)


def load_data_bundle(manifest: MonthlyRunManifest) -> DataBundleManifest | None:
    path = Path(manifest.data_bundle_manifest_path or manifest.market_data_manifest_path)
    if not path.exists():
        return None
    bundle = DataBundleManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    _resolve_slice_manifest_paths(bundle, path)
    return bundle


def data_bundle_errors(
    manifest: MonthlyRunManifest, bundle: DataBundleManifest | None
) -> list[str]:
    errors: list[str] = []
    if bundle is None:
        if manifest.optimizer_mode:
            errors.append(
                "optimizer modes require data_bundle_manifest_path or market_data_manifest_path"
            )
        return errors
    if manifest.optimizer_mode and bundle.status != DataBundleStatus.AUTHORITATIVE:
        errors.append(f"data bundle is not authoritative: {bundle.status.value}")
    if manifest.optimizer_mode and (missing := bundle.authoritative_contract_errors()):
        errors.append("authoritative data bundle missing required fields: " + ", ".join(missing))
    if manifest.optimizer_mode:
        bundle_path = Path(manifest.data_bundle_manifest_path or manifest.market_data_manifest_path)
        errors.extend(_slice_manifest_errors(bundle, bundle_path))
    expected = manifest.data_bundle_checksum or manifest.data_manifest_checksum
    if expected and bundle.bundle_checksum != expected:
        errors.append("data bundle checksum does not match run manifest")
    return errors


def coverage_payload(
    manifest: MonthlyRunManifest,
    bundle: DataBundleManifest | None,
    *,
    errors: list[str],
) -> dict:
    if bundle is None:
        return {
            "run_id": manifest.run_id,
            "status": "blocked" if errors else "diagnostics_only",
            "data_bundle_checksum": manifest.data_bundle_checksum
            or manifest.data_manifest_checksum,
            "errors": errors,
        }
    return {
        "run_id": manifest.run_id,
        "status": "pass" if not errors else "blocked",
        "bundle_id": bundle.bundle_id,
        "data_bundle_checksum": bundle.bundle_checksum,
        "bundle_checksum": bundle.bundle_checksum,
        "data_repo_path": bundle.data_repo_path,
        "data_repo_commit_sha": bundle.data_repo_commit_sha,
        "data_repo_branch": bundle.data_repo_branch,
        "calendars": bundle.calendars,
        "fee_model_version": bundle.fee_model_version,
        "slippage_model_version": bundle.slippage_model_version,
        "adjustment_policy": bundle.adjustment_policy,
        "slice_manifest_ids": [item.manifest_id for item in bundle.slice_manifests],
        "slice_manifests": [
            {
                "manifest_path": item.manifest_path,
                "manifest_id": item.manifest_id,
                "source": item.source,
                "market": item.market,
                "symbol": item.symbol,
                "timeframe": item.timeframe,
                "checksum": item.checksum,
                "calendar": item.calendar,
                "authoritative": item.authoritative,
            }
            for item in bundle.slice_manifests
        ],
        "errors": errors,
    }


def _slice_manifest_errors(bundle: DataBundleManifest, bundle_path: Path) -> list[str]:
    errors: list[str] = []
    for item in bundle.slice_manifests:
        path = Path(item.manifest_path)
        if not path.exists():
            errors.append(f"slice manifest missing: {path}")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"slice manifest malformed: {path}: {exc}")
    for slice_index in _slice_index_candidates(bundle, bundle_path):
        try:
            json.loads(slice_index.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"slice_index.json malformed: {slice_index}: {exc}")
    return errors


def _resolve_slice_manifest_paths(bundle: DataBundleManifest, bundle_path: Path) -> None:
    for item in bundle.slice_manifests:
        path = Path(item.manifest_path)
        if path.is_absolute():
            continue
        item.manifest_path = str(_resolve_bundle_relative_path(bundle, bundle_path, path))


def _resolve_bundle_relative_path(
    bundle: DataBundleManifest,
    bundle_path: Path,
    relative_path: Path,
) -> Path:
    roots = _bundle_candidate_roots(bundle, bundle_path)
    for root in roots:
        candidate = (root / relative_path).resolve()
        if candidate.exists():
            return candidate
    return (roots[0] / relative_path).resolve()


def _bundle_candidate_roots(bundle: DataBundleManifest, bundle_path: Path) -> list[Path]:
    raw_roots: list[Path] = []
    if bundle.data_repo_path:
        data_repo_root = Path(bundle.data_repo_path)
        if data_repo_root.is_absolute():
            raw_roots.append(data_repo_root)
        else:
            for parent in [bundle_path.parent, *bundle_path.parents]:
                raw_roots.append(parent / data_repo_root)
    raw_roots.extend([bundle_path.parent, *bundle_path.parents])

    roots: list[Path] = []
    seen: set[str] = set()
    for root in raw_roots:
        resolved = root.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        roots.append(resolved)
    return roots


def _slice_index_candidates(bundle: DataBundleManifest, bundle_path: Path) -> list[Path]:
    candidates = [
        bundle_path.parent / "slice_index.json",
        Path(bundle.slice_manifests[0].manifest_path).parent / "slice_index.json",
    ]
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result
