"""Data-only LRS SQLite daily export entry points."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def export_table(sqlite_path: Path, table: str, output_path: Path) -> Path:
    import sqlite3

    with sqlite3.connect(sqlite_path) as connection:
        frame = pd.read_sql_query(f"SELECT * FROM {table}", connection)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, engine="pyarrow", index=False)
    return output_path


def export_table_with_lineage(
    sqlite_path: Path,
    table: str,
    output_path: Path,
    *,
    export_id: str,
    schema_version: str,
    config: dict | None = None,
) -> dict:
    """Export one LRS table and persist local research lineage metadata."""

    if not table.replace("_", "").isalnum():
        raise ValueError(f"unsafe LRS table name: {table}")
    output = export_table(sqlite_path, table, output_path)
    config_payload = {
        "sqlite_path": str(sqlite_path),
        "table": table,
        "schema_version": schema_version,
        "config": config or {},
    }
    payload = {
        "schema_version": "lrs_local_export_lineage_v1",
        "export_id": export_id,
        "source_kind": "local_research_export",
        "source_db_checksum": _sha256_file(sqlite_path),
        "source_db_path": str(sqlite_path),
        "query": f"SELECT * FROM {table}",
        "query_config_hash": _sha256_json(config_payload),
        "pulled_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "table": table,
        "table_schema_version": schema_version,
        "output_path": str(output),
        "output_checksum": _sha256_file(output),
    }
    lineage_path = output.with_suffix(".lineage.json")
    lineage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
