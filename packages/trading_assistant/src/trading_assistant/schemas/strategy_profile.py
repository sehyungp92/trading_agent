"""Strategy profile schemas — archetype-aware metadata for analysis context."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class StrategyArchetype(str, Enum):
    TREND_FOLLOW = "trend_follow"
    DIVERGENCE_SWING = "divergence_swing"
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    INTRADAY_MOMENTUM = "intraday_momentum"
    OPENING_RANGE_BREAKOUT = "opening_range_breakout"
    MULTI_TF_MOMENTUM = "multi_tf_momentum"
    BOX_BREAKOUT = "box_breakout"
    VWAP_PULLBACK = "vwap_pullback"
    FLOW_FOLLOWING = "flow_following"
    BEAR_REGIME_SWING = "bear_regime_swing"
    MULTI_ENGINE_BEAR = "multi_engine_bear"
    MEAN_REVERSION_PULLBACK = "mean_reversion_pullback"
    MOMENTUM_PULLBACK_CRYPTO = "momentum_pullback_crypto"
    INSTITUTIONAL_ANCHOR = "institutional_anchor"
    VOLUME_PROFILE_BREAKOUT = "volume_profile_breakout"


class HoldingPeriod(str, Enum):
    INTRADAY = "intraday"
    MULTI_DAY = "multi_day"
    SWING = "swing"


class StrategyRisk(BaseModel):
    unit_risk_dollars: float = 0.0
    daily_stop_R: float = 0.0
    max_heat_R: float = 0.0
    tp_ratio: float = 0.0


class StrategyAllocation(BaseModel):
    base_risk_pct: float = 0.0


class ArchetypeExpectation(BaseModel):
    expected_win_rate: tuple[float, float] = (0.0, 1.0)
    expected_payoff_ratio: tuple[float, float] = (0.0, 10.0)
    regime_sensitivity: str = "medium"
    typical_holding_bars: tuple[int, int] = (1, 1000)


class CoordinationSignal(BaseModel):
    trigger: dict = {}
    target: dict = {}
    condition: str = ""
    params: dict = {}


class CooldownPair(BaseModel):
    strategies: list[str] = []
    minutes: int = 0
    session_only: bool = False
    session_window: list[str] = []


class DirectionFilter(BaseModel):
    observer: str = ""
    reference: str = ""
    agree_mult: float = 1.0
    oppose_mult: float = 0.0


class StockCoordination(BaseModel):
    directional_cap_R: float = 0.0
    symbol_collision_action: str = ""
    strategies: list[str] = []


class CoordinationConfig(BaseModel):
    signals: list[CoordinationSignal] = []
    cooldown_pairs: list[CooldownPair] = []
    direction_filter: DirectionFilter | None = None
    stock_coordination: StockCoordination | None = None


class PortfolioConfig(BaseModel):
    heat_cap_R: float = 0.0
    portfolio_daily_stop_R: float = 0.0
    portfolio_weekly_stop_R: float = 0.0
    family_allocations: dict[str, float] = {}
    drawdown_tiers: list[list[float]] = []


class ExitTier(BaseModel):
    """A single TP/exit tier in a strategy's exit profile."""

    tier_name: str = ""
    tier_type: str = ""  # "take_profit", "trailing_stop", "time_stop"
    r_target: float = 0.0
    partial_pct: float = 1.0  # fraction of position to exit


class ExitProfile(BaseModel):
    """Exit configuration for a strategy — TP tiers, trailing stops, etc."""

    tiers: list[ExitTier] = []
    has_chandelier: bool = False
    regime_multipliers: dict[str, float] = {}  # regime -> multiplier on trailing


class StrategyProfile(BaseModel):
    display_name: str = ""
    bot_id: str = ""
    family: str = ""
    archetype: StrategyArchetype | str = ""
    asset_class: str = ""
    symbols: list[str] = []
    preferred_regimes: list[str] = []
    adverse_regimes: list[str] = []
    holding_period: HoldingPeriod | str = ""
    risk: StrategyRisk = StrategyRisk()
    allocation: StrategyAllocation = StrategyAllocation()
    entry_types: list[str] = []
    sub_engines: list[str] = []
    regime_model: str = ""
    key_metadata_fields: list[str] = []
    analysis_focus: list[str] = []
    macro_regime_sensitivity: dict[str, str | float] = {}  # G/R/S/D -> label or sizing multiplier
    exit_profile: ExitProfile | None = None


class StrategyRegistry(BaseModel):
    """Registry of all strategy profiles with coordination and portfolio config."""

    strategies: dict[str, StrategyProfile] = {}
    coordination: CoordinationConfig = CoordinationConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    archetype_expectations: dict[str, ArchetypeExpectation] = {}

    def is_active(self, strategy_id: str) -> bool:
        """Return True if strategy_id is currently registered (i.e. live)."""
        return strategy_id in self.strategies

    def strategies_for_bot(self, bot_id: str) -> dict[str, StrategyProfile]:
        """Return all strategies deployed on a given bot_id."""
        return {
            sid: profile
            for sid, profile in self.strategies.items()
            if profile.bot_id == bot_id
        }

    def strategies_in_family(self, family: str) -> dict[str, StrategyProfile]:
        """Return all strategies in a given family."""
        return {
            sid: profile
            for sid, profile in self.strategies.items()
            if profile.family == family
        }

    def archetype_for_strategy(self, strategy_id: str) -> StrategyArchetype | None:
        """Return the archetype enum for a strategy, or None if not found."""
        profile = self.strategies.get(strategy_id)
        if profile and profile.archetype:
            arch = profile.archetype
            if isinstance(arch, StrategyArchetype):
                return arch
            try:
                return StrategyArchetype(arch)
            except ValueError:
                return None
        return None

    def expectations_for_archetype(self, archetype: str) -> ArchetypeExpectation | None:
        """Return expected performance ranges for an archetype."""
        return self.archetype_expectations.get(archetype)
