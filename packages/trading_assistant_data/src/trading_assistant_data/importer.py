"""Reference snapshot importer."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .checksums import checksum_paths, sha256_file
from .repo import git_commit_sha


@dataclass(frozen=True)
class CopySpec:
    source: str
    destination: str
    recursive: bool = False
    pattern: str = "*"


COPY_SPECS: tuple[CopySpec, ...] = (
    CopySpec("trading/data/raw", "trading/data/raw", pattern="*.parquet"),
    CopySpec("trading/backtests/momentum/data/raw", "trading/backtests/momentum/data/raw"),
    CopySpec("trading/backtests/swing/data/raw", "trading/backtests/swing/data/raw"),
    CopySpec("trading/backtests/stock/data/raw", "trading/backtests/stock/data/raw", pattern="*.parquet"),
    CopySpec("trading/backtests/regime/data/raw", "trading/backtests/regime/data/raw"),
    CopySpec("crypto_trader/data/candles/BTC", "crypto_trader/data/candles/BTC", pattern="*.parquet"),
    CopySpec("crypto_trader/data/candles/ETH", "crypto_trader/data/candles/ETH", pattern="*.parquet"),
    CopySpec("crypto_trader/data/candles/SOL", "crypto_trader/data/candles/SOL", pattern="*.parquet"),
    CopySpec("crypto_trader/data/funding", "crypto_trader/data/funding", pattern="*.parquet"),
    CopySpec("k_stock_trader/data/krx_daily_parquet", "k_stock_trader/data/krx_daily_parquet", recursive=True),
    CopySpec("k_stock_trader/data/kis_intraday_parquet", "k_stock_trader/data/kis_intraday_parquet", recursive=True),
)


DOC_COPY_SPECS: tuple[CopySpec, ...] = (
    CopySpec("trading/data/strategy-registry.json", "trading_strategy_registry_2026-05-30.json"),
    CopySpec("k_stock_trader/config/olr_kalcb/olr_deployment_universe_103.yaml", "olr_deployment_universe_103.yaml"),
    CopySpec("k_stock_trader/config/universe_103.yaml", "k_stock_universe_103.yaml"),
)


def import_reference_snapshot(
    *,
    repo_root: Path,
    snapshot: str,
    references_root: Path,
    dry_run: bool = False,
) -> dict:
    references_root = _resolve_snapshot_root(references_root)
    snapshot_root = Path(repo_root) / "data" / "imported" / f"reference_snapshot_{snapshot}"
    docs_root = Path(repo_root) / "docs" / "reference_inputs"
    copied: list[Path] = []
    skipped: list[str] = []
    planned: list[dict[str, str]] = []

    for spec in COPY_SPECS:
        source_root = references_root / spec.source
        dest_root = snapshot_root / spec.destination
        if not source_root.exists():
            skipped.append(f"missing source: {source_root}")
            continue
        for source in _iter_sources(source_root, spec):
            destination = dest_root / source.relative_to(source_root)
            planned.append({"source": str(source), "destination": str(destination)})
            if dry_run:
                continue
            if _copy_if_changed(source, destination):
                copied.append(destination)

    for spec in DOC_COPY_SPECS:
        source = references_root / spec.source
        destination = docs_root / spec.destination
        if not source.exists():
            skipped.append(f"missing source: {source}")
            continue
        planned.append({"source": str(source), "destination": str(destination)})
        if dry_run:
            continue
        if _copy_if_changed(source, destination):
            copied.append(destination)

    import_manifest = snapshot_root / "import_manifest.json"
    all_files = []
    if snapshot_root.exists():
        all_files.extend(
            path
            for path in snapshot_root.rglob("*")
            if path.is_file() and path != import_manifest
        )
    if docs_root.exists():
        all_files.extend(path for path in docs_root.rglob("*") if path.is_file())
    manifest = {
        "snapshot": snapshot,
        "references_root": str(references_root),
        "references_root_commit_sha": git_commit_sha(references_root),
        "source_repo_commits": _source_repo_commits(references_root),
        "imported_root": str(snapshot_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "planned_file_count": len(planned),
        "copied_file_count": len(copied),
        "file_count": len(all_files),
        "bytes": sum(path.stat().st_size for path in all_files) if not dry_run else 0,
        "skipped": skipped,
        "files": checksum_paths(all_files, root=repo_root) if not dry_run else planned,
    }
    if not dry_run:
        from .io import write_json

        write_json(import_manifest, manifest)
    return manifest


def _source_repo_commits(references_root: Path) -> dict[str, str]:
    source_names = sorted(
        {Path(spec.source).parts[0] for spec in (*COPY_SPECS, *DOC_COPY_SPECS)}
    )
    return {
        name: git_commit_sha(references_root / name)
        for name in source_names
        if (references_root / name).exists()
    }


def _resolve_snapshot_root(path: Path) -> Path:
    candidate = Path(path).resolve()
    if (candidate / "trading").exists():
        return candidate
    raise FileNotFoundError(f"could not resolve references root: {path}")


def _iter_sources(root: Path, spec: CopySpec) -> Iterable[Path]:
    if spec.recursive:
        return (path for path in root.rglob(spec.pattern) if path.is_file())
    return (path for path in root.glob(spec.pattern) if path.is_file())


def _copy_if_changed(source: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        try:
            if sha256_file(destination) == sha256_file(source):
                return False
        except OSError:
            pass
    shutil.copy2(source, destination)
    return True
