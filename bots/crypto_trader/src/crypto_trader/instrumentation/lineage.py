"""Lineage and redaction helpers for assistant instrumentation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SECRET_KEY_PARTS = (
    "secret",
    "private_key",
    "api_key",
    "token",
    "password",
    "dsn",
    "wallet_address",
)

RISK_CONFIG_KEYS = {
    "heat_cap_R",
    "directional_cap_R",
    "portfolio_daily_stop_R",
    "max_total_positions",
    "dd_tiers",
    "symbol_collision",
    "symbol_exposure_cap_R",
    "terminal_accounting_mode",
    "priority_headroom_R",
    "priority_reserve_threshold",
}

ALLOCATION_CONFIG_KEYS = {
    "strategies",
    "initial_equity",
}


def stable_json_dumps(value: Any) -> str:
    """Serialize a value with deterministic key ordering."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any, *, length: int = 16) -> str:
    """Return a compact stable hash for JSON-like data."""
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()[:length]


def redact_secrets(value: Any) -> Any:
    """Recursively redact secret-bearing keys from mappings."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(part in key_text for part in SECRET_KEY_PARTS):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    return value


def strip_secret_fields(value: Any) -> Any:
    """Recursively remove secret-bearing fields from snapshot payloads."""
    if isinstance(value, dict):
        stripped: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(part in key_text for part in SECRET_KEY_PARTS):
                continue
            stripped[key] = strip_secret_fields(item)
        return stripped
    if isinstance(value, list):
        return [strip_secret_fields(item) for item in value]
    if isinstance(value, tuple):
        return [strip_secret_fields(item) for item in value]
    return value


def read_json_file(path: Path | None) -> dict[str, Any]:
    """Best-effort JSON file reader used for config lineage."""
    if path is None:
        return {}
    try:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {"value": data}
    except Exception:
        return {}


def object_to_plain(value: Any) -> Any:
    """Convert config dataclasses and path-heavy objects into plain structures."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict(redacted=True)
        except TypeError:
            try:
                return value.to_dict()
            except TypeError:
                pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): object_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [object_to_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: object_to_plain(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def subset_keys(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: payload[key] for key in sorted(keys) if key in payload}


def current_code_sha(cwd: Path | None = None) -> str:
    """Return the git HEAD SHA when available, otherwise an explicit unknown marker."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class LineageContext:
    """Stable lineage values shared by assistant instrumentation events."""

    bot_id: str = ""
    family_id: str = "crypto_perps"
    portfolio_id: str = "default"
    account_alias: str = "default"
    exchange: str = "hyperliquid"
    venue_environment: str = "testnet"
    code_sha: str = "unknown"
    deployment_id: str = ""
    config_version: str = ""
    portfolio_config_version: str = ""
    risk_config_version: str = ""
    allocation_version: str = ""
    strategy_config_versions: dict[str, str] = field(default_factory=dict)
    symbol_universe: list[str] = field(default_factory=list)
    deployment_manifest_version: str = ""

    def base(self) -> dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "family_id": self.family_id,
            "portfolio_id": self.portfolio_id,
            "account_alias": self.account_alias,
            "exchange": self.exchange,
            "venue_environment": self.venue_environment,
            "code_sha": self.code_sha,
            "deployment_id": self.deployment_id,
            "config_version": self.config_version,
            "portfolio_config_version": self.portfolio_config_version,
            "risk_config_version": self.risk_config_version,
            "allocation_version": self.allocation_version,
            "symbol_universe": list(self.symbol_universe),
        }

    def for_strategy(self, strategy_id: str) -> dict[str, Any]:
        payload = self.base()
        payload["strategy_id"] = strategy_id
        payload["strategy_config_version"] = self.strategy_config_versions.get(strategy_id, "")
        return payload

    def for_portfolio(self) -> dict[str, Any]:
        return self.base()

    def metadata_defaults(self, strategy_id: str = "") -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "portfolio_id": self.portfolio_id,
            "account_alias": self.account_alias,
            "config_version": self.config_version,
            "deployment_id": self.deployment_id,
            "code_sha": self.code_sha,
            "lineage": self.for_strategy(strategy_id) if strategy_id else self.for_portfolio(),
        }


def from_live_engine_inputs(
    *,
    config: Any,
    portfolio_config: Any | None = None,
    strategy_configs: dict[str, Any] | None = None,
    deployment_manifest: dict[str, Any] | None = None,
    cwd: Path | None = None,
) -> LineageContext:
    """Build a lineage context from live engine startup inputs."""
    config_plain = redact_secrets(object_to_plain(config))
    portfolio_plain = redact_secrets(object_to_plain(portfolio_config or {}))
    strategy_plain = {
        strategy_id: redact_secrets(object_to_plain(value))
        for strategy_id, value in (strategy_configs or {}).items()
    }
    deployment_plain = redact_secrets(deployment_manifest or {})

    family_id = str(getattr(config, "family_id", "") or config_plain.get("family_id") or "crypto_perps")
    portfolio_id = str(getattr(config, "portfolio_id", "") or config_plain.get("portfolio_id") or "default")
    account_alias = str(
        getattr(config, "account_alias", "")
        or config_plain.get("account_alias")
        or getattr(config, "bot_id", "")
        or "default"
    )
    bot_id = str(getattr(config, "bot_id", "") or config_plain.get("bot_id") or "")
    environment = "testnet" if bool(getattr(config, "is_testnet", True)) else "mainnet"
    code_sha = current_code_sha(cwd)

    strategy_versions = {
        strategy_id: stable_hash(payload)
        for strategy_id, payload in strategy_plain.items()
    }
    portfolio_version = stable_hash(portfolio_plain)
    risk_version = stable_hash(subset_keys(portfolio_plain, RISK_CONFIG_KEYS))
    allocation_version = stable_hash(subset_keys(portfolio_plain, ALLOCATION_CONFIG_KEYS))
    config_version = stable_hash({
        "live_config": config_plain,
        "portfolio_config": portfolio_plain,
        "strategy_configs": strategy_plain,
    })
    deployment_version = stable_hash(deployment_plain) if deployment_plain else ""
    deployment_id = stable_hash({
        "deployment_manifest": deployment_plain,
        "code_sha": code_sha,
        "config_version": config_version,
    })

    return LineageContext(
        bot_id=bot_id,
        family_id=family_id,
        portfolio_id=portfolio_id,
        account_alias=account_alias,
        venue_environment=environment,
        code_sha=code_sha,
        deployment_id=deployment_id,
        config_version=config_version,
        portfolio_config_version=portfolio_version,
        risk_config_version=risk_version,
        allocation_version=allocation_version,
        strategy_config_versions=strategy_versions,
        symbol_universe=list(getattr(config, "symbols", []) or config_plain.get("symbols") or []),
        deployment_manifest_version=deployment_version,
    )
