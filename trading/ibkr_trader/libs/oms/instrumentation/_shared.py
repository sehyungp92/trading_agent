"""Shared helpers for OMS instrumentation payload builders."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_RAW_ACCOUNT_KEYS = {
    "account",
    "account_id",
    "broker_account",
    "broker_account_id",
    "ibkr_account",
    "ibkr_account_id",
}


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return plain(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(plain(item) for item in value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def event_time(value: Any = None) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if value:
        return str(value)
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def direction_from_qty(qty: Any) -> str:
    qty_f = as_float(qty)
    if qty_f > 0:
        return "LONG"
    if qty_f < 0:
        return "SHORT"
    return "FLAT"


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(data_dir: str | Path, date_str: str) -> dict[str, str]:
    root = Path(data_dir)
    candidates = {
        "trades": root / "trades" / f"trades_{date_str}.jsonl",
        "orders": root / "orders" / f"orders_{date_str}.jsonl",
        "fills": root / "fills" / f"fills_{date_str}.jsonl",
        "daily": root / "daily" / f"daily_{date_str}.jsonl",
        "portfolio_rules": root / "portfolio_rules" / f"rules_{date_str}.jsonl",
        "positions": root / "positions" / f"positions_{date_str}.jsonl",
        "portfolio": root / "portfolio" / f"portfolio_snapshots_{date_str}.jsonl",
        "allocations": root / "allocations" / f"allocations_{date_str}.jsonl",
    }
    return {name: digest for name, path in candidates.items() if (digest := file_sha256(path))}


def stable_payload_hash(prefix: str, value: Any, length: int = 16) -> str:
    raw = json.dumps(plain(value), sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:length]}"


def account_alias_for(account_id: Any = "", existing_alias: Any = "") -> str:
    alias = str(existing_alias or "").strip()
    if alias:
        return alias
    raw = str(account_id or "").strip()
    if not raw:
        return ""
    return stable_payload_hash("acct_", raw, length=12)


def redact_account_payload(value: Any, account_alias: str = "") -> Any:
    """Remove raw broker account IDs from OMS instrumentation payloads.

    Snapshot events are replay artifacts; they should correlate by account_alias
    only. If a raw account id is present and no alias was supplied, derive a
    deterministic opaque alias so snapshots remain joinable without leaking the
    broker account value.
    """
    value = plain(value)
    if isinstance(value, Mapping):
        local_alias = account_alias_for(value.get("account_id", ""), account_alias or value.get("account_alias", ""))
        result: dict[str, Any] = {}
        raw_account_seen = False
        for key, raw in value.items():
            key_str = str(key)
            key_l = key_str.lower()
            if key_l == "account_alias":
                continue
            if key_l in _RAW_ACCOUNT_KEYS:
                raw_account_seen = True
                local_alias = account_alias_for(raw, local_alias)
                continue
            result[key_str] = redact_account_payload(raw, local_alias)
        if local_alias or raw_account_seen:
            result["account_alias"] = local_alias
        return result
    if isinstance(value, list):
        return [redact_account_payload(item, account_alias) for item in value]
    return value
