"""Canonical data hygiene utilities."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .io import read_json, write_json


def clean_stale_cme_bid_ask_aliases(*, repo_root: Path, dry_run: bool = False) -> dict[str, Any]:
    """Remove stale CME bid/ask aliases written under timeframe=1m.

    The accepted convention is `kind=bid_ask/timeframe=1m_bid_ask`; any
    `kind=bid_ask/timeframe=1m` partition is ambiguous with trade bars and must not be
    trusted by bundle selection.
    """

    repo = Path(repo_root).resolve()
    index_path = repo / "data" / "manifests" / "slices" / "slice_index.json"
    payload = read_json(index_path) if index_path.exists() else {"schema_version": "slice_index_v1", "slices": []}
    slices = list(payload.get("slices", []))
    stale_entries = [item for item in slices if _is_stale_index_entry(item)]
    stale_manifest_paths = sorted(
        {
            str(item.get("manifest_path", ""))
            for item in stale_entries
            if str(item.get("manifest_path", "")).strip()
        }
    )
    remaining = [item for item in slices if item not in stale_entries]
    remaining_manifest_paths = {str(item.get("manifest_path", "")) for item in remaining}
    removable_manifests = [
        repo / path for path in stale_manifest_paths if path not in remaining_manifest_paths and (repo / path).exists()
    ]
    stale_dirs = sorted(
        path
        for path in (
            repo
            / "data"
            / "canonical"
            / "bars"
            / "market=cme_futures"
            / "source=ibkr"
            / "kind=bid_ask"
        ).glob("symbol=*/timeframe=1m")
        if path.is_dir()
    )

    for path in [*removable_manifests, *stale_dirs]:
        _assert_safe_cleanup_target(repo, path)

    if not dry_run:
        payload["slices"] = remaining
        write_json(index_path, payload)
        for path in removable_manifests:
            path.unlink(missing_ok=True)
        for path in stale_dirs:
            shutil.rmtree(path)

    return {
        "name": "clean_stale_cme_bid_ask_aliases",
        "dry_run": dry_run,
        "removed_index_entries": len(stale_entries),
        "deleted_manifest_paths": [_rel(path, repo) for path in removable_manifests],
        "deleted_canonical_dirs": [_rel(path, repo) for path in stale_dirs],
    }


def _is_stale_index_entry(item: dict[str, Any]) -> bool:
    canonical_paths = [str(path) for path in item.get("canonical_paths", [])]
    return (
        item.get("source") == "ibkr"
        and item.get("market") == "cme_futures"
        and item.get("timeframe") == "1m"
        and any("kind=bid_ask" in path and "timeframe=1m/" in path.replace("\\", "/") for path in canonical_paths)
    )


def _assert_safe_cleanup_target(repo_root: Path, path: Path) -> None:
    target = path.resolve()
    data_root = (repo_root / "data").resolve()
    if not target.is_relative_to(data_root):
        raise ValueError(f"cleanup target escapes data root: {target}")
    parts = set(target.parts)
    if path.is_dir():
        required = {"canonical", "bars", "market=cme_futures", "source=ibkr", "kind=bid_ask", "timeframe=1m"}
        if not required.issubset(parts):
            raise ValueError(f"unexpected canonical cleanup target: {target}")
    elif target.suffix != ".json" or "manifests" not in parts or "slices" not in parts:
        raise ValueError(f"unexpected manifest cleanup target: {target}")


def _rel(path: Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
