"""Execution adapter contracts for parity between simulation and live."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from crypto_trader.core.models import OrderType
from crypto_trader.core.order_semantics import (
    EXIT_OCA_POLICY,
    EXIT_ORDER_TAGS,
    validate_strategy_scoped_oca_group,
)
from crypto_trader.core.runtime_types import ExecutionReport, OrderIntent


@dataclass(frozen=True, slots=True)
class ExecutionCapabilities:
    """Adapter order semantics available without local emulation."""

    market: bool = True
    limit: bool = True
    stop_market: bool = True
    stop_limit: bool = False
    reduce_only: bool = False
    oca: bool = False
    bracket: bool = False
    ttl: bool = False
    partial_fills: bool = True


@runtime_checkable
class ExecutionAdapter(Protocol):
    """Adapter-neutral order and execution-report interface."""

    @property
    def capabilities(self) -> ExecutionCapabilities:
        ...

    def submit(self, intent: OrderIntent) -> list[ExecutionReport]:
        ...

    def cancel(self, client_order_id: str) -> list[ExecutionReport]:
        ...

    def sync_open_orders(self) -> list[ExecutionReport]:
        ...

    def sync_positions(self) -> list[dict]:
        ...

    def sync_fills(self, watermark: datetime) -> list[ExecutionReport]:
        ...


def unsupported_order_intent_reasons(
    intent: OrderIntent,
    capabilities: ExecutionCapabilities,
) -> list[str]:
    """Return adapter capability failures for a concrete order intent."""
    checks = [
        (intent.order_type == OrderType.STOP_LIMIT and not capabilities.stop_limit,
         "stop_limit_not_supported_live"),
        (intent.reduce_only and not capabilities.reduce_only,
         "reduce_only_not_enforced_live"),
        (bool(intent.oca_group) and not capabilities.oca and not _uses_broker_managed_oca(intent),
         "oca_not_supported_live"),
        (bool(intent.bracket_group) and not capabilities.bracket,
         "bracket_not_supported_live"),
        (intent.ttl_bars is not None and not capabilities.ttl,
         "ttl_not_supported_live"),
    ]
    return [reason for active, reason in checks if active]


def _uses_broker_managed_oca(intent: OrderIntent) -> bool:
    group = str(intent.oca_group or "").strip()
    metadata_group = str(intent.metadata.get("oca_group") or "").strip()
    tag = str(intent.metadata.get("tag") or "").strip()
    return (
        bool(group)
        and metadata_group == group
        and str(intent.metadata.get("oca_policy") or "") == EXIT_OCA_POLICY
        and not bool(intent.metadata.get("native_oca_required", False))
        and bool(intent.reduce_only)
        and bool(intent.metadata.get("exit_only", False))
        and tag in EXIT_ORDER_TAGS
        and not validate_strategy_scoped_oca_group(
            group,
            strategy_id=intent.strategy_id,
            symbol=intent.symbol,
        )
    )
