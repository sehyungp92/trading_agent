"""Live trading configuration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from trading_contracts.relay_acceptance import contains_placeholder, validate_relay_config

_CREDENTIAL_PLACEHOLDER_RE = re.compile(
    r"(your|placeholder|changeme|change_me|example|here|<|>)",
    re.IGNORECASE,
)
_WALLET_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_PRIVATE_KEY_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")


@dataclass
class LiveConfig:
    """Configuration for live/paper trading on Hyperliquid."""

    wallet_address: str = ""
    private_key: str | None = None  # None = read-only mode
    is_testnet: bool = True
    poll_interval_sec: float = 15.0  # candle poll frequency
    fill_poll_interval_sec: float = 30.0
    fill_query_overlap_sec: float = 300.0
    equity_snapshot_interval_sec: float = 300.0  # 5 minutes
    health_check_interval_sec: float = 60.0
    health_report_interval_sec: float = 300.0
    funnel_report_interval_sec: float = 3600.0
    rate_limit_per_sec: float = 5.0
    max_slippage_pct: float = 0.005  # 0.5% for market orders
    reconciliation_policy: str = "block"
    allow_manual_flatten: bool = False
    strict_live_parity: bool = False
    require_native_oca: bool = False

    # Strategy configs
    strategy_configs: dict[str, Path] = field(default_factory=dict)
    portfolio_config_path: Path | None = None
    deployment_manifest_path: Path | None = None

    # Trading universe
    symbols: list[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    state_dir: Path = field(default_factory=lambda: Path("data/live_state"))
    asset_meta_path: Path | None = None

    # Instrumentation / relay (optional)
    bot_id: str = ""
    family_id: str = "crypto_perps"
    portfolio_id: str = "default"
    account_alias: str = "default"
    relay_url: str = ""
    relay_secret: str = ""

    # PostgreSQL (optional; empty string = disabled)
    postgres_dsn: str = ""
    postgres_async_enabled: bool = True
    postgres_queue_capacity: int = 5000
    postgres_flush_timeout_sec: float = 5.0

    @property
    def base_url(self) -> str:
        from hyperliquid.utils import constants

        return constants.TESTNET_API_URL if self.is_testnet else constants.MAINNET_API_URL

    def validate(self, *, require_relay: bool = False) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        wallet_address = str(self.wallet_address or "").strip()
        private_key = "" if self.private_key is None else str(self.private_key).strip()

        if not wallet_address:
            errors.append("wallet_address is required")
        elif _is_placeholder(wallet_address):
            errors.append("wallet_address must be replaced with a real 0x wallet address")
        elif not _WALLET_RE.fullmatch(wallet_address):
            errors.append("wallet_address must be 0x followed by 40 hex characters")

        if self.private_key is None or not private_key:
            errors.append("private_key is required for trading (None = read-only)")
        elif _is_placeholder(private_key):
            errors.append("private_key must be replaced with a real 0x private key")
        elif not _PRIVATE_KEY_RE.fullmatch(private_key):
            errors.append("private_key must be 0x followed by 64 hex characters")

        if not self.symbols:
            errors.append("at least one symbol required")
        if require_relay or any((self.relay_url, self.relay_secret)):
            relay_errors = validate_relay_config(
                relay_url=self.relay_url,
                hmac_secret=self.relay_secret,
                bot_id=self.bot_id,
                require=True,
                allow_loopback=False,
                secret_field_name="relay_secret",
            )
            errors.extend(
                error.replace("relay_url is required", "relay_url is required when relay is configured")
                .replace("relay_secret is required", "relay_secret is required when relay is configured")
                .replace("bot_id is required", "bot_id is required when relay is configured")
                for error in relay_errors
            )
        if self.reconciliation_policy not in {"block", "cancel_unmanaged_orders", "flatten_unmanaged_positions"}:
            errors.append("reconciliation_policy must be block, cancel_unmanaged_orders, or flatten_unmanaged_positions")
        if self.reconciliation_policy == "flatten_unmanaged_positions" and not self.allow_manual_flatten:
            errors.append("allow_manual_flatten=true is required for flatten_unmanaged_positions")
        if not self.is_testnet and self.asset_meta_path is None:
            errors.append("asset_meta_path is required for mainnet parity")
        if self.postgres_queue_capacity <= 0:
            errors.append("postgres_queue_capacity must be positive")
        if self.postgres_flush_timeout_sec <= 0:
            errors.append("postgres_flush_timeout_sec must be positive")
        return errors

    @classmethod
    def from_dict(cls, d: dict) -> LiveConfig:
        """Deserialize from dict."""
        strategy_configs = {
            k: Path(v) for k, v in d.get("strategy_configs", {}).items()
        }
        return cls(
            wallet_address=d.get("wallet_address", ""),
            private_key=d.get("private_key"),
            is_testnet=d.get("is_testnet", True),
            poll_interval_sec=d.get("poll_interval_sec", 15.0),
            fill_poll_interval_sec=d.get("fill_poll_interval_sec", 30.0),
            fill_query_overlap_sec=d.get("fill_query_overlap_sec", 300.0),
            equity_snapshot_interval_sec=d.get("equity_snapshot_interval_sec", 300.0),
            health_check_interval_sec=d.get("health_check_interval_sec", 60.0),
            health_report_interval_sec=d.get("health_report_interval_sec", 300.0),
            funnel_report_interval_sec=d.get("funnel_report_interval_sec", 3600.0),
            rate_limit_per_sec=d.get("rate_limit_per_sec", 5.0),
            max_slippage_pct=d.get("max_slippage_pct", 0.005),
            reconciliation_policy=d.get("reconciliation_policy", "block"),
            allow_manual_flatten=d.get("allow_manual_flatten", False),
            strict_live_parity=d.get("strict_live_parity", False),
            require_native_oca=d.get("require_native_oca", False),
            strategy_configs=strategy_configs,
            portfolio_config_path=Path(d["portfolio_config_path"]) if d.get("portfolio_config_path") else None,
            deployment_manifest_path=Path(d["deployment_manifest_path"]) if d.get("deployment_manifest_path") else None,
            symbols=d.get("symbols", ["BTC", "ETH", "SOL"]),
            data_dir=Path(d.get("data_dir", "data")),
            state_dir=Path(d.get("state_dir", "data/live_state")),
            asset_meta_path=Path(d["asset_meta_path"]) if d.get("asset_meta_path") else None,
            bot_id=d.get("bot_id", ""),
            family_id=d.get("family_id", "crypto_perps"),
            portfolio_id=d.get("portfolio_id", "default"),
            account_alias=d.get("account_alias", "default"),
            relay_url=d.get("relay_url", ""),
            relay_secret=d.get("relay_secret", ""),
            postgres_dsn=os.environ.get("POSTGRES_DSN") or d.get("postgres_dsn", ""),
            postgres_async_enabled=d.get("postgres_async_enabled", True),
            postgres_queue_capacity=int(d.get("postgres_queue_capacity", 5000)),
            postgres_flush_timeout_sec=float(d.get("postgres_flush_timeout_sec", 5.0)),
        )

    def to_dict(self, *, redacted: bool = False) -> dict:
        """Serialize to dict.

        ``redacted=True`` is intended for assistant config snapshots and removes
        credential-bearing fields entirely.
        """
        payload = {
            "wallet_address": self.wallet_address,
            "is_testnet": self.is_testnet,
            "poll_interval_sec": self.poll_interval_sec,
            "fill_poll_interval_sec": self.fill_poll_interval_sec,
            "fill_query_overlap_sec": self.fill_query_overlap_sec,
            "equity_snapshot_interval_sec": self.equity_snapshot_interval_sec,
            "health_check_interval_sec": self.health_check_interval_sec,
            "health_report_interval_sec": self.health_report_interval_sec,
            "funnel_report_interval_sec": self.funnel_report_interval_sec,
            "rate_limit_per_sec": self.rate_limit_per_sec,
            "max_slippage_pct": self.max_slippage_pct,
            "reconciliation_policy": self.reconciliation_policy,
            "allow_manual_flatten": self.allow_manual_flatten,
            "strict_live_parity": self.strict_live_parity,
            "require_native_oca": self.require_native_oca,
            "strategy_configs": {k: str(v) for k, v in self.strategy_configs.items()},
            "portfolio_config_path": str(self.portfolio_config_path) if self.portfolio_config_path else None,
            "deployment_manifest_path": str(self.deployment_manifest_path) if self.deployment_manifest_path else None,
            "symbols": self.symbols,
            "data_dir": str(self.data_dir),
            "state_dir": str(self.state_dir),
            "asset_meta_path": str(self.asset_meta_path) if self.asset_meta_path else None,
            "bot_id": self.bot_id,
            "family_id": self.family_id,
            "portfolio_id": self.portfolio_id,
            "account_alias": self.account_alias,
            "relay_url": self.relay_url,
            "relay_secret": self.relay_secret,
            "postgres_dsn": self.postgres_dsn,
            "postgres_async_enabled": self.postgres_async_enabled,
            "postgres_queue_capacity": self.postgres_queue_capacity,
            "postgres_flush_timeout_sec": self.postgres_flush_timeout_sec,
        }
        if redacted:
            for key in ("wallet_address", "relay_secret", "postgres_dsn"):
                payload.pop(key, None)
        return payload


def _is_placeholder(value: str) -> bool:
    return bool(_CREDENTIAL_PLACEHOLDER_RE.search(value)) or contains_placeholder(value)
