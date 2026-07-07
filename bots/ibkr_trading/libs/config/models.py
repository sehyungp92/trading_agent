"""Typed models for monorepo runtime configuration."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractTemplate(BaseModel):
    symbol: str
    sec_type: str = "FUT"
    exchange: str
    currency: str = "USD"
    multiplier: float = 1.0
    tick_size: float
    tick_value: float
    trading_class: str | None = None
    primary_exchange: str | None = None


class ExchangeRoute(BaseModel):
    root_symbol: str
    exchange: str
    primary_exchange: str | None = None
    trading_class: str | None = None
    local_symbol_pattern: str | None = None


class ConnectionGroupConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int
    account_id: str | None = None
    readonly: bool = False
    reconnect_max_retries: int = 10
    reconnect_base_delay_s: float = 1.0
    reconnect_max_delay_s: float = 60.0
    market_data_type: int = 1  # 1=real-time, 2=frozen, 3=delayed, 4=delayed-frozen
    pacing_orders_per_sec: float = 5.0
    pacing_messages_per_sec: float = 45.0


class StrategyRiskManifest(BaseModel):
    unit_risk_dollars: float
    daily_stop_R: float = 2.0
    max_working_orders: int = 2
    max_heat_R: float = 0.0
    portfolio_daily_stop_R: float | None = None
    priority: int | None = None
    allowed_order_types: dict[str, list[str]] = Field(default_factory=dict)
    session_block: dict[str, Any] = Field(default_factory=dict)


class StrategyAllocationManifest(BaseModel):
    base_risk_pct: float | None = None
    equity_offset: float = 0.0
    continuation_half_size: bool | None = None
    continuation_size_mult: float | None = None


class StrategyManifest(BaseModel):
    strategy_id: str
    system_id: str
    family: str
    connection_group: str
    display_name: str
    module_path: str | None = None
    enabled: bool = True
    paper_mode: bool = False
    asset_class: str = "unknown"
    symbols: list[str] = Field(default_factory=list)
    risk: StrategyRiskManifest
    allocation: StrategyAllocationManifest = Field(default_factory=StrategyAllocationManifest)
    artifact_config: dict[str, Any] = Field(default_factory=dict)
    engine_config: dict[str, Any] = Field(default_factory=dict)
    dashboard_metadata: dict[str, Any] = Field(default_factory=dict)
    deployment_tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_manifest(self) -> "StrategyManifest":
        if not self.strategy_id.strip():
            raise ValueError("strategy_id must not be blank")
        if not self.system_id.strip():
            raise ValueError(f"{self.strategy_id}: system_id must not be blank")
        if not self.family.strip():
            raise ValueError(f"{self.strategy_id}: family must not be blank")
        if not self.connection_group.strip():
            raise ValueError(f"{self.strategy_id}: connection_group must not be blank")
        return self


class TriggerRef(BaseModel):
    strategy: str
    event: str


class TargetRef(BaseModel):
    strategy: str
    action: str


class CoordinationSignal(BaseModel):
    trigger: TriggerRef
    target: TargetRef
    condition: str
    params: dict[str, Any] = Field(default_factory=dict)


class CooldownPair(BaseModel):
    strategies: list[str]
    minutes: int
    session_only: bool = True
    session_window: list[str] = Field(default_factory=list)


class DirectionFilterRule(BaseModel):
    observer: str
    reference: str
    agree_mult: float = 1.0
    oppose_mult: float = 0.0


class ChopThrottleRule(BaseModel):
    enabled: bool = False
    strategy: str | None = None
    score_threshold: int = 0
    size_mult: float = 1.0


class CoordinationConfig(BaseModel):
    signals: list[CoordinationSignal] = Field(default_factory=list)
    cooldown_pairs: list[CooldownPair] = Field(default_factory=list)
    direction_filter: DirectionFilterRule | None = None
    directional_cap_R: float = 0.0
    chop_throttle: ChopThrottleRule = Field(default_factory=ChopThrottleRule)


class PortfolioRiskConfig(BaseModel):
    heat_cap_R: float = 2.5
    portfolio_daily_stop_R: float = 3.0
    portfolio_weekly_stop_R: float = 5.0
    account_urd_dollars: float = 200.0
    global_standdown: bool = False


class PortfolioCapitalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allocation_check_equity: float = 100000.0
    paper_initial_equity: float = 30000.0
    family_allocations: dict[str, float] = Field(default_factory=dict)
    strategy_allocations: dict[str, float] = Field(default_factory=dict)


class PortfolioConfig(BaseModel):
    risk: PortfolioRiskConfig = Field(default_factory=PortfolioRiskConfig)
    capital: PortfolioCapitalConfig = Field(default_factory=PortfolioCapitalConfig)
    drawdown_tiers: list[tuple[float, float]] = Field(default_factory=list)
    coordination: CoordinationConfig = Field(default_factory=CoordinationConfig)


class StrategyRegistryConfig(BaseModel):
    connection_groups: dict[str, ConnectionGroupConfig]
    strategies: dict[str, StrategyManifest]

    @model_validator(mode="after")
    def validate_registry(self) -> "StrategyRegistryConfig":
        if not self.connection_groups:
            raise ValueError("At least one connection group is required")
        if not self.strategies:
            raise ValueError("At least one strategy manifest is required")

        seen_client_ids: dict[int, str] = {}
        for group_name, group in self.connection_groups.items():
            prior = seen_client_ids.get(group.client_id)
            if prior is not None:
                raise ValueError(
                    f"Duplicate client_id {group.client_id} for groups {prior} and {group_name}"
                )
            seen_client_ids[group.client_id] = group_name

        for key, manifest in self.strategies.items():
            if key != manifest.strategy_id:
                raise ValueError(
                    f"Strategy key {key!r} does not match embedded strategy_id {manifest.strategy_id!r}"
                )
            if manifest.connection_group not in self.connection_groups:
                raise ValueError(
                    f"{manifest.strategy_id}: unknown connection_group {manifest.connection_group!r}"
                )
        return self

    def enabled_strategies(self, *, live: bool = False) -> list[StrategyManifest]:
        """Return enabled strategies, optionally filtering out paper_mode in live."""
        return [
            manifest for manifest in self.strategies.values()
            if manifest.enabled and (not live or not manifest.paper_mode)
        ]
