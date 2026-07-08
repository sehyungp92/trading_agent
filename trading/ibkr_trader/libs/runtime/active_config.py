"""Active runtime config persistence for operator-facing truth."""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


ACTIVE_CONFIG_VERSION = "2026-06-04"


@dataclass(frozen=True)
class ActiveRuntimeConfigRecord:
    account_id: str
    config_scope: str
    scope_id: str
    runtime_env: str
    payload: dict[str, Any]
    config_version: str = ACTIVE_CONFIG_VERSION
    deployment_id: str = ""
    source_hash: str = ""
    applied_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None


def hash_config_payload(payload: Mapping[str, Any]) -> str:
    """Return a stable content hash for an active config payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def active_config_expiry(hours: float = 48.0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=hours)


def build_account_runtime_config(
    *,
    account_id: str,
    heat_cap_R: float,
    portfolio_daily_stop_R: float,
    portfolio_weekly_stop_R: float,
    global_standdown: bool,
    account_urd: float,
    source: str = "config/portfolio.yaml + runtime overrides",
) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "heat_cap_R": float(heat_cap_R),
        "portfolio_daily_stop_R": float(portfolio_daily_stop_R),
        "portfolio_weekly_stop_R": float(portfolio_weekly_stop_R),
        "global_standdown": bool(global_standdown),
        "account_urd": float(account_urd),
        "source": source,
    }


def build_family_runtime_config(
    *,
    account_id: str,
    family_id: str,
    family_allocation_pct: float,
    family_nav: float,
    family_heat_cap_R: float,
    family_daily_stop_R: float,
    family_weekly_stop_R: float,
    active_strategy_ids: list[str],
    paper_only_filtered: list[str] | None = None,
    source: str = "family coordinator bootstrap",
) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "family_id": family_id,
        "family_allocation_pct": float(family_allocation_pct),
        "family_nav": float(family_nav),
        "family_heat_cap_R": float(family_heat_cap_R),
        "family_daily_stop_R": float(family_daily_stop_R),
        "family_weekly_stop_R": float(family_weekly_stop_R),
        "active_strategy_ids": list(active_strategy_ids),
        "paper_only_filtered": list(paper_only_filtered or []),
        "source": source,
    }


def build_strategy_runtime_config(
    *,
    account_id: str,
    strategy_id: str,
    family_id: str,
    enabled: bool,
    live: bool,
    allocated_nav: float,
    unit_risk_dollars: float,
    max_heat_R: float,
    max_daily_loss_R: float,
    max_weekly_loss_R: float,
    risk_per_trade: float,
    regime_overlays: Mapping[str, Any] | None = None,
    source: str = "strategy coordinator bootstrap",
) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "strategy_id": strategy_id,
        "family_id": family_id,
        "enabled": bool(enabled),
        "live": bool(live),
        "allocated_nav": float(allocated_nav),
        "unit_risk_dollars": float(unit_risk_dollars),
        "max_heat_R": float(max_heat_R),
        "strategy_heat_cap_R": float(max_heat_R),
        "max_daily_loss_R": float(max_daily_loss_R),
        "max_weekly_loss_R": float(max_weekly_loss_R),
        "risk_per_trade": float(risk_per_trade),
        "regime_overlays": dict(regime_overlays or {}),
        "source": source,
    }


async def upsert_active_runtime_config(
    pool,
    record: ActiveRuntimeConfigRecord,
) -> None:
    """Persist the latest active config record if a DB pool is available."""
    if pool is None:
        return
    execute = getattr(pool, "execute", None)
    if not callable(execute):
        return
    source_hash = record.source_hash or hash_config_payload(record.payload)
    result = execute(
        """
        INSERT INTO active_runtime_config (
            account_id, config_scope, scope_id, runtime_env, config_version,
            deployment_id, source_hash, payload, applied_at, expires_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10)
        ON CONFLICT (account_id, config_scope, scope_id, runtime_env) DO UPDATE SET
            config_version = EXCLUDED.config_version,
            deployment_id = EXCLUDED.deployment_id,
            source_hash = EXCLUDED.source_hash,
            payload = EXCLUDED.payload,
            applied_at = EXCLUDED.applied_at,
            expires_at = EXCLUDED.expires_at
        """,
        record.account_id,
        record.config_scope,
        record.scope_id,
        record.runtime_env,
        record.config_version,
        record.deployment_id or None,
        source_hash,
        json.dumps(record.payload, sort_keys=True, default=str),
        record.applied_at,
        record.expires_at,
    )
    if inspect.isawaitable(result):
        await result


def build_active_runtime_config_artifact(
    records: list[ActiveRuntimeConfigRecord],
) -> dict[str, Any]:
    """Build a live-compatible backtest artifact payload."""
    return {
        "schema_version": ACTIVE_CONFIG_VERSION,
        "records": [
            {
                "account_id": record.account_id,
                "config_scope": record.config_scope,
                "scope_id": record.scope_id,
                "runtime_env": record.runtime_env,
                "config_version": record.config_version,
                "deployment_id": record.deployment_id,
                "source_hash": record.source_hash or hash_config_payload(record.payload),
                "payload": record.payload,
                "applied_at": record.applied_at.isoformat(),
                "expires_at": record.expires_at.isoformat() if record.expires_at else None,
            }
            for record in records
        ],
    }


def write_active_runtime_config_artifact(
    output_dir: str | Path,
    records: list[ActiveRuntimeConfigRecord],
) -> Path:
    """Write active_runtime_config.json for replay/backtest outputs."""
    path = Path(output_dir) / "active_runtime_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_active_runtime_config_artifact(records), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return path
