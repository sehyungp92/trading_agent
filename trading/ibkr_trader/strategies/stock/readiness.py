from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from libs.config.models import StrategyRegistryConfig

_ET = ZoneInfo("America/New_York")
_ACCOUNT_PLACEHOLDER_TOKENS = ("PLACEHOLDER", "YOUR_ACCOUNT", "CHANGEME")


@dataclass(frozen=True)
class StockReadinessFailure:
    category: str
    identifier: str
    detail: str

    @property
    def check_name(self) -> str:
        prefix = {
            "account-config": "stock-account-config",
            "artifact-readiness": "stock-artifact-readiness",
        }[self.category]
        return f"{prefix}:{self.identifier}"


def stock_trade_date(timestamp: datetime | None = None) -> date:
    ts = timestamp or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_ET).date()


def validate_stock_readiness(
    registry: StrategyRegistryConfig,
    *,
    live: bool = False,
    strategy_ids: tuple[str, ...] | None = None,
    trade_date: date | None = None,
) -> tuple[dict[str, Any], list[StockReadinessFailure]]:
    selected_ids = set(strategy_ids) if strategy_ids is not None else None
    manifests = [
        manifest
        for manifest in registry.enabled_strategies(live=live)
        if manifest.family == "stock"
        and manifest.artifact_config
        and (selected_ids is None or manifest.strategy_id in selected_ids)
    ]

    artifacts: dict[str, Any] = {}
    failures: list[StockReadinessFailure] = []
    if not manifests:
        return artifacts, failures

    group_to_strategies: dict[str, list[str]] = {}
    for manifest in manifests:
        group_to_strategies.setdefault(manifest.connection_group, []).append(manifest.strategy_id)

    for group_name, strategy_list in group_to_strategies.items():
        account_id = registry.connection_groups[group_name].account_id
        reason = _account_config_reason(account_id)
        if reason is None:
            continue
        failures.append(
            StockReadinessFailure(
                category="account-config",
                identifier=group_name,
                detail=f"{reason}; strategies={', '.join(sorted(strategy_list))}",
            )
        )

    current_trade_date = trade_date or stock_trade_date()
    for manifest in manifests:
        artifact_type = str(manifest.artifact_config.get("type", "")).strip().lower()
        try:
            artifacts[manifest.strategy_id] = _load_artifact(
                artifact_type=artifact_type,
                trade_date=current_trade_date,
            )
        except Exception as exc:
            failures.append(
                StockReadinessFailure(
                    category="artifact-readiness",
                    identifier=manifest.strategy_id,
                    detail=(
                        f"{artifact_type or 'artifact'} unavailable for "
                        f"{current_trade_date.isoformat()}: {exc}"
                    ),
                )
            )

    return artifacts, failures


def _account_config_reason(account_id: str | None) -> str | None:
    if account_id is None or not str(account_id).strip():
        return "account_id is blank"
    value = str(account_id).strip()
    if value.startswith("${") and value.endswith("}"):
        return f"account_id is unresolved placeholder {value}"
    if any(token in value.upper() for token in _ACCOUNT_PLACEHOLDER_TOKENS):
        return f"account_id is placeholder-like {_redact_account_id(value)}"
    return None


def _redact_account_id(account_id: str) -> str:
    if len(account_id) <= 4:
        return "***"
    return f"{account_id[:2]}...{account_id[-2:]}"


def _load_artifact(*, artifact_type: str, trade_date: date) -> Any:
    if artifact_type == "watchlist":
        from strategies.stock.iaric.artifact_store import load_watchlist_artifact
        from strategies.stock.iaric.config import StrategySettings

        return load_watchlist_artifact(trade_date, settings=StrategySettings())

    if artifact_type == "candidate":
        from strategies.stock.alcb.artifact_store import load_candidate_artifact
        from strategies.stock.alcb.config import StrategySettings

        return load_candidate_artifact(trade_date, settings=StrategySettings())

    raise ValueError(f"unsupported artifact_config.type={artifact_type!r}")
