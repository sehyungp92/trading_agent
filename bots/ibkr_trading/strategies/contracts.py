"""Strategy contracts for the compatibility-first kernel scaffold."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from libs.config.models import PortfolioConfig, StrategyManifest, StrategyRegistryConfig


@dataclass(slots=True)
class RuntimeContext:
    """Dependency surface for strategy plugin factories."""

    manifest: StrategyManifest
    registry: StrategyRegistryConfig
    portfolio: PortfolioConfig
    session: Any = None
    market_data: Any = None
    oms: Any = None
    state_store: Any = None
    instrumentation: Any = None
    contracts: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    logger: Any = None
    clock: Any = None
    db_pool: Any = None
    account_gate: Any = None
    family_coordinator: Any = None
    regime_service: Any = None
    crisis_service: Any = None
    trade_recorder: Any = None
    heartbeat: Any = None
    runtime_overrides: Any = None
    require_instrumentation: bool = False


@dataclass(slots=True)
class CoordinatorRuntimeOverrides:
    """Optional offline/test providers for family coordinators.

    Production runtime leaves this unset. Parity tests use it to keep the
    deployed coordinator wiring offline without changing production configs.
    """

    adapter_factory: Callable[..., Any] | None = None
    calendar_factory: Callable[[], Any] | None = None
    equity_provider: Callable[[], float] | None = None
    build_oms_service: Callable[..., Any] | None = None
    build_multi_strategy_oms: Callable[..., Any] | None = None
    strategy_ids: Sequence[str] | None = None
    strategy_ids_provider: Callable[[], Sequence[str]] | None = None
    portfolio_rules_provider: Callable[[], Any] | None = None
    stock_artifact_provider: Callable[[], dict[str, Any]] | None = None
    overlay_rebalance_provider: Callable[[], dict[str, Any]] | None = None
    state_dir_overrides: dict[str, Path] = field(default_factory=dict)
    instrumentation_data_dir: Path | None = None
    disable_instrumentation: bool = False
    disable_market_data: bool = False
    disable_background_tasks: bool = False


@runtime_checkable
class StrategyPlugin(Protocol):
    """Full lifecycle protocol for strategy engines managed by the runtime."""

    strategy_id: str

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def health_status(self) -> dict[str, Any]: ...

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        """Restore state from a persisted snapshot (optional)."""
        ...

    def snapshot_state(self) -> dict[str, Any]:
        """Return serializable state for persistence (optional)."""
        ...

    async def on_market_data(self, event: Any) -> None:
        """Handle incoming market data tick/bar (optional)."""
        ...

    async def on_order_event(self, event: Any) -> None:
        """Handle order status change (optional)."""
        ...

    async def on_fill_event(self, event: Any) -> None:
        """Handle fill notification (optional)."""
        ...
