"""Canonical slice manifest writer helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from trading_assistant_data.io import read_json, write_json
from trading_assistant_data.manifests import MarketDataManifest, write_model


def slice_manifest_path(repo_root: Path, manifest: MarketDataManifest) -> Path:
    start = manifest.start_ts.strftime("%Y%m%dT%H%M%SZ")
    end = manifest.end_ts.strftime("%Y%m%dT%H%M%SZ")
    return (
        Path(repo_root)
        / "data"
        / "manifests"
        / "slices"
        / manifest.source
        / manifest.market
        / manifest.symbol
        / manifest.timeframe
        / f"{start}_{end}.market_data_manifest.json"
    )


def write_slice_manifest(repo_root: Path, manifest: MarketDataManifest) -> Path:
    path = slice_manifest_path(repo_root, manifest)
    write_model(path, manifest)
    return path


def update_slice_index(repo_root: Path, writes: Iterable[object]) -> None:
    entries_by_id = {}
    for write in writes:
        manifest = write.manifest
        entries_by_id[manifest.manifest_id] = {
            "manifest_id": manifest.manifest_id,
            "manifest_path": _rel(write.manifest_path, repo_root),
            "source": manifest.source,
            "market": manifest.market,
            "symbol": manifest.symbol,
            "timeframe": manifest.timeframe,
            "checksum": manifest.checksum,
            "canonical_paths": [_rel(path, repo_root) for path in write.canonical_paths],
        }
        family = str((manifest.lineage or {}).get("strategy_data_family") or "").strip()
        if family:
            entries_by_id[manifest.manifest_id]["strategy_data_family"] = family
    if not entries_by_id:
        return

    index_path = Path(repo_root) / "data" / "manifests" / "slices" / "slice_index.json"
    if index_path.exists():
        try:
            payload = read_json(index_path)
        except ValueError:
            payload = {"schema_version": "slice_index_v1", "slices": []}
    else:
        payload = {"schema_version": "slice_index_v1", "slices": []}
    entries = list(entries_by_id.values())
    manifest_ids = {entry["manifest_id"] for entry in entries}
    manifest_paths = {entry["manifest_path"] for entry in entries}
    canonical_paths = {path for entry in entries for path in entry["canonical_paths"]}
    payload["slices"] = [
        item
        for item in payload.get("slices", [])
        if item.get("manifest_id") not in manifest_ids
        and item.get("manifest_path") not in manifest_paths
        and not canonical_paths.intersection(item.get("canonical_paths", []))
    ]
    payload["slices"].extend(entries)
    payload["slices"] = list({item["manifest_path"]: item for item in payload["slices"]}.values())
    payload["slices"].sort(
        key=lambda item: (
            item["source"],
            item["market"],
            item["symbol"],
            item["timeframe"],
            item["manifest_path"],
        )
    )
    write_json(index_path, payload)


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()
