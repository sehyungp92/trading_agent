from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


VOLATILE_HASH_KEYS = {
    "broker_order_id",
    "database_id",
    "db_id",
    "local_path",
    "manifest_path",
    "optimized_config_path",
    "original_order_id",
    "path",
    "recorded_at",
    "session_path",
    "source_manifest",
    "source_path",
    "write_timestamp",
    "log_sequence",
}
TEXT_ARTIFACT_SUFFIXES = frozenset(
    {
        ".json",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)


def canonical_json_hash(value: Any, *, exclude_keys: set[str] | None = None) -> str:
    payload = _normalize(value, exclude_keys=_excluded_keys(exclude_keys))
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    data = target.read_bytes()
    if _is_text_artifact(target, data):
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def json_file_hash(path: str | Path, *, exclude_keys: set[str] | None = None) -> str:
    target = Path(path)
    if not target.exists() or not target.read_text(encoding="utf-8").strip():
        return canonical_json_hash(None)
    return canonical_json_hash(json.loads(target.read_text(encoding="utf-8")), exclude_keys=exclude_keys)


def jsonl_file_hash(path: str | Path, *, exclude_keys: set[str] | None = None) -> str:
    target = Path(path)
    if not target.exists():
        return canonical_json_hash([])
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return canonical_json_hash(rows, exclude_keys=exclude_keys)


def _excluded_keys(extra: set[str] | None) -> set[str]:
    return VOLATILE_HASH_KEYS | set(extra or ())


def _is_text_artifact(path: Path, data: bytes) -> bool:
    if path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES:
        return True
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _normalize(value: Any, *, exclude_keys: set[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item, exclude_keys=exclude_keys)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in exclude_keys
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item, exclude_keys=exclude_keys) for item in value]
    return value


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)
