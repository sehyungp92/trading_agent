"""Seed parquet manifest helpers for regime/crisis live startup."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SEED_MANIFEST_NAME = "regime_seed_manifest.json"
SEED_FILES = ("macro_df.parquet", "market_df.parquet", "strat_ret_df.parquet")

REQUIRED_COLUMNS = {
    "macro_df.parquet": ("GROWTH", "INFLATION"),
    "market_df.parquet": ("VIX", "SPREAD", "SLOPE_10Y2Y"),
    "strat_ret_df.parquet": ("SPY", "EFA", "TLT", "GLD", "CASH"),
}
RETURN_AS_OF_COLS = ("SPY", "EFA", "TLT", "GLD")


def write_seed_manifest(
    data_dir: Path,
    *,
    generated_by: str,
    source_versions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and write a manifest for regime seed parquets."""
    manifest = build_seed_manifest(
        data_dir,
        generated_by=generated_by,
        source_versions=source_versions or {},
    )
    path = data_dir / SEED_MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_seed_manifest(
    data_dir: Path,
    *,
    generated_by: str,
    source_versions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a manifest describing required seed parquet files."""
    data_dir = Path(data_dir)
    frames = _load_required_frames(data_dir)
    summaries = {
        filename: _frame_summary(frames[filename], REQUIRED_COLUMNS[filename])
        for filename in SEED_FILES
    }
    file_hashes = {
        filename: _sha256_file(data_dir / filename)
        for filename in SEED_FILES
    }
    data_as_of = _latest_common_return_as_of(frames["strat_ret_df.parquet"])
    if data_as_of is None:
        raise ValueError(
            "strat_ret_df.parquet has no common non-null return date for "
            f"{', '.join(RETURN_AS_OF_COLS)}"
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": generated_by,
        "data_as_of": data_as_of.date().isoformat(),
        "row_counts": {
            filename: int(summaries[filename]["row_count"])
            for filename in SEED_FILES
        },
        "files": {
            filename: {
                **summaries[filename],
                "sha256": file_hashes[filename],
            }
            for filename in SEED_FILES
        },
        "required_columns": {
            filename: list(cols)
            for filename, cols in REQUIRED_COLUMNS.items()
        },
        "source_versions": source_versions or {},
    }


def validate_seed_data_dir(
    data_dir: Path,
    *,
    require_manifest: bool = False,
    validate_hashes: bool = True,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Validate seed files and manifest, returning ``(ok, status, manifest)``."""
    data_dir = Path(data_dir)
    missing_files = [name for name in SEED_FILES if not (data_dir / name).exists()]
    if missing_files:
        return False, f"seed_files=missing:{','.join(missing_files)}", None

    manifest_path = data_dir / SEED_MANIFEST_NAME
    if not manifest_path.exists():
        if require_manifest:
            return False, "seed_manifest=missing", None
        return True, "seed_manifest=missing", None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"seed_manifest=invalid_json:{exc}", None

    try:
        current = build_seed_manifest(
            data_dir,
            generated_by="runtime_validation",
            source_versions=manifest.get("source_versions", {}),
        )
    except Exception as exc:
        return False, f"seed_manifest=validation_failed:{exc}", manifest

    problems: list[str] = []
    if manifest.get("data_as_of") != current.get("data_as_of"):
        problems.append(
            "data_as_of_mismatch:"
            f"{manifest.get('data_as_of')}!={current.get('data_as_of')}"
        )

    for filename in SEED_FILES:
        manifest_file = (manifest.get("files") or {}).get(filename) or {}
        current_file = current["files"][filename]
        if manifest_file.get("row_count") != current_file.get("row_count"):
            problems.append(f"{filename}:row_count_mismatch")
        if validate_hashes and manifest_file.get("sha256") != current_file.get("sha256"):
            problems.append(f"{filename}:sha256_mismatch")

    if problems:
        return False, "seed_manifest=stale:" + ",".join(problems), manifest

    return True, f"seed_manifest=ok:data_as_of={manifest.get('data_as_of')}", manifest


def bootstrap_seed_data_dir(target_dir: Path, seed_dir: Path | None) -> str:
    """Copy image seed files into the writable runtime cache when fresher.

    Docker named volumes can hide files copied into ``/app/data`` after the
    first deploy. Keeping an immutable image seed under ``/app/seed`` and
    bootstrapping from it keeps the writable volume current without overwriting
    a cache that is already fresher than the image.
    """
    if seed_dir is None:
        return "seed_bootstrap=disabled"
    target_dir = Path(target_dir)
    seed_dir = Path(seed_dir)
    if not seed_dir.exists():
        return "seed_bootstrap=seed_dir_missing"
    try:
        if target_dir.resolve() == seed_dir.resolve():
            return "seed_bootstrap=same_dir"
    except Exception:
        pass

    seed_missing = [name for name in SEED_FILES if not (seed_dir / name).exists()]
    if seed_missing:
        return "seed_bootstrap=seed_files_missing:" + ",".join(seed_missing)

    seed_as_of = _data_as_of_from_dir(seed_dir)
    target_as_of = _data_as_of_from_dir(target_dir)
    target_missing = any(not (target_dir / name).exists() for name in SEED_FILES)
    if not target_missing and target_as_of is not None and seed_as_of is not None:
        if target_as_of >= seed_as_of:
            return f"seed_bootstrap=current:data_as_of={target_as_of.date().isoformat()}"

    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in (*SEED_FILES, SEED_MANIFEST_NAME):
        src = seed_dir / filename
        if src.exists():
            shutil.copy2(src, target_dir / filename)

    copied_as_of = seed_as_of.date().isoformat() if seed_as_of is not None else "unknown"
    return f"seed_bootstrap=copied:data_as_of={copied_as_of}"


def manifest_data_as_of(data_dir: Path) -> pd.Timestamp | None:
    """Return manifest data_as_of when available."""
    path = Path(data_dir) / SEED_MANIFEST_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        value = data.get("data_as_of")
        return pd.Timestamp(value) if value else None
    except Exception:
        return None


def _data_as_of_from_dir(data_dir: Path) -> pd.Timestamp | None:
    manifest_as_of = manifest_data_as_of(data_dir)
    if manifest_as_of is not None:
        return manifest_as_of
    path = Path(data_dir) / "strat_ret_df.parquet"
    if not path.exists():
        return None
    try:
        return _latest_common_return_as_of(pd.read_parquet(path))
    except Exception:
        return None


def _load_required_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for filename in SEED_FILES:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required seed file missing: {path}")
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        missing = [col for col in REQUIRED_COLUMNS[filename] if col not in df.columns]
        if missing:
            raise ValueError(f"{filename} missing required columns: {missing}")
        frames[filename] = df.sort_index()
    return frames


def _frame_summary(df: pd.DataFrame, required_cols: tuple[str, ...]) -> dict[str, Any]:
    required = df.loc[:, list(required_cols)].dropna(how="any")
    latest_required = required.index.max() if not required.empty else None
    return {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "start": df.index.min().date().isoformat() if len(df) else "",
        "end": df.index.max().date().isoformat() if len(df) else "",
        "latest_required_as_of": (
            pd.Timestamp(latest_required).date().isoformat()
            if latest_required is not None else ""
        ),
    }


def _latest_common_return_as_of(strat_ret_df: pd.DataFrame) -> pd.Timestamp | None:
    missing = [col for col in RETURN_AS_OF_COLS if col not in strat_ret_df.columns]
    if missing:
        return None
    valid = strat_ret_df.loc[:, list(RETURN_AS_OF_COLS)].dropna(how="any")
    if valid.empty:
        return None
    return pd.Timestamp(valid.index.max())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
