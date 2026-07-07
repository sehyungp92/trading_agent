"""Trend strategy configuration — nested dataclasses with maximally loose defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class IndicatorParams:
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    adx_period: int = 14
    atr_period: int = 14
    rsi_period: int = 14
    volume_ma_period: int = 20
    atr_avg_period: int = 20


@dataclass
class RegimeParams:
    a_min_adx: float = 12.0
    b_min_adx: float = 12.0
    b_adx_rising_required: bool = False
    require_ema_cross: bool = False
    require_structure: bool = False
    structure_lookback: int = 10
    no_trade_max_adx: float = 10.0
    h1_regime_enabled: bool = True
    h1_min_adx: float = 22.0


@dataclass
class TrendSetupParams:
    impulse_min_bars: int = 3
    impulse_min_atr_move: float = 0.8
    impulse_lookback: int = 30
    pullback_max_retrace: float = 0.75
    pullback_max_bars: int = 15
    pullback_rsi_low: float = 30.0
    pullback_rsi_high: float = 65.0
    min_confluences: int = 0
    require_orderly_pullback: bool = False
    strict_orderly_pullback: bool = False
    orderly_min_countertrend_bars: int = 2
    orderly_max_body_frac: float = 0.85
    orderly_max_countertrend_volume_ratio: float = 0.95
    min_room_r: float = 1.0
    min_room_r_a: float = 2.0
    require_completed_impulse: bool = True
    use_weighted_confluence: bool = False
    min_setup_score_b: float = 1.5
    min_setup_score_a: float = 2.5
    weekly_room_filter_enabled: bool = False
    min_weekly_room_r: float = 1.0


@dataclass
class TrendConfirmationParams:
    enable_engulfing: bool = True
    enable_hammer: bool = False
    enable_ema_reclaim: bool = True
    enable_structure_break: bool = True
    require_confirmation: bool = False
    require_confirmation_for_b: bool = False
    require_volume_confirm: bool = False
    enforce_volume_on_trigger: bool = False
    volume_threshold_mult: float = 1.0
    hammer_wick_ratio: float = 2.0
    max_bars_after_setup: int = 3


@dataclass
class TrendEntryParams:
    entry_on_close: bool = True
    entry_on_break: bool = False
    max_bars_after_confirmation: int = 2
    mode: str = "legacy"


@dataclass
class TrendStopParams:
    atr_mult: float = 2.0
    use_swing: bool = True
    use_farther: bool = True
    min_stop_atr: float = 1.0
    swing_lookback: int = 10
    buffer_pct: float = 0.001


@dataclass
class TrendExitParams:
    tp1_r: float = 0.8
    tp1_frac: float = 0.25
    tp2_r: float = 2.0
    tp2_frac: float = 0.50
    runner_frac: float = 0.25
    time_stop_bars: int = 20
    time_stop_min_progress_r: float = 0.3
    time_stop_action: str = "reduce"
    be_after_tp1: bool = True
    be_buffer_r: float = 0.2
    be_min_bars_above: int = 4
    ema_failsafe_enabled: bool = True
    ema_failsafe_period: int = 20
    ema_failsafe_min_expansion_r: float = 1.0
    quick_exit_enabled: bool = True
    quick_exit_bars: int = 12
    quick_exit_max_mfe_r: float = 0.15
    quick_exit_max_r: float = -0.2
    scratch_exit_enabled: bool = False
    scratch_peak_r: float = 0.25
    scratch_floor_r: float = 0.0
    scratch_min_bars: int = 4
    mfe_lock_exit_enabled: bool = False
    mfe_lock_trigger_r: float = 1.0
    mfe_lock_floor_r: float = 0.2
    mfe_lock_min_bars: int = 2


@dataclass
class TrendTrailParams:
    trail_r_adaptive: bool = True
    trail_use_mfe_for_adaptive: bool = False
    trail_buffer_wide: float = 1.2
    trail_buffer_tight: float = 0.1
    trail_r_ceiling: float = 1.5
    trail_activation_r: float = 0.3
    trail_activation_bars: int = 8
    structure_trail_enabled: bool = False
    structure_swing_lookback: int = 5
    ema_trail_period: int = 20


@dataclass
class TrendRiskParams:
    risk_pct_a: float = 0.015
    risk_pct_b: float = 0.01
    max_risk_pct: float = 0.02
    max_leverage_major: float = 15.0
    max_leverage_alt: float = 12.0
    min_leverage: float = 0.1
    major_symbols: tuple[str, ...] = ("BTC", "ETH")


@dataclass
class TrendLimitParams:
    max_concurrent_positions: int = 5
    max_daily_loss_pct: float = 0.04
    max_consecutive_losses: int = 6
    max_trades_per_day: int = 10
    max_correlated_risk_pct: float = 0.05


@dataclass
class TrendPerpFilterParams:
    funding_filter_enabled: bool = False
    funding_extreme_threshold: float = 0.001
    funding_reduce_risk_mult: float = 0.5
    relative_strength_filter_enabled: bool = False
    relative_strength_lookback: int = 24
    relative_strength_min_delta: float = 0.0


@dataclass
class TrendReentryParams:
    enabled: bool = True
    cooldown_bars: int = 3
    max_loss_r: float = 1.5
    max_reentries: int = 1
    min_confluences_override: int = 0
    max_wait_bars: int = 0
    require_same_direction: bool = False
    only_after_scratch_exit: bool = False
    risk_scale: float = 1.0


@dataclass
class TrendSymbolFilterParams:
    btc_direction: str = "both"
    eth_direction: str = "both"
    sol_direction: str = "both"


@dataclass
class TrendConfig:
    h1_indicators: IndicatorParams = field(
        default_factory=lambda: IndicatorParams(ema_fast=20, ema_mid=50, ema_slow=200)
    )
    d1_indicators: IndicatorParams = field(
        default_factory=lambda: IndicatorParams(ema_fast=21, ema_mid=50, ema_slow=50)
    )
    m15_indicators: IndicatorParams = field(
        default_factory=lambda: IndicatorParams(ema_fast=20, ema_mid=50, ema_slow=200)
    )
    regime: RegimeParams = field(default_factory=RegimeParams)
    setup: TrendSetupParams = field(default_factory=TrendSetupParams)
    confirmation: TrendConfirmationParams = field(default_factory=TrendConfirmationParams)
    entry: TrendEntryParams = field(default_factory=TrendEntryParams)
    stops: TrendStopParams = field(default_factory=TrendStopParams)
    exits: TrendExitParams = field(default_factory=TrendExitParams)
    trail: TrendTrailParams = field(default_factory=TrendTrailParams)
    risk: TrendRiskParams = field(default_factory=TrendRiskParams)
    limits: TrendLimitParams = field(default_factory=TrendLimitParams)
    filters: TrendPerpFilterParams = field(default_factory=TrendPerpFilterParams)
    reentry: TrendReentryParams = field(default_factory=TrendReentryParams)
    symbol_filter: TrendSymbolFilterParams = field(default_factory=TrendSymbolFilterParams)
    symbols: list[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])

    def to_dict(self) -> dict:
        """Serialize config to a plain dict (round-trips with from_dict)."""
        raw = asdict(self)
        for section in raw.values():
            if isinstance(section, dict):
                for k, v in section.items():
                    if isinstance(v, tuple):
                        section[k] = list(v)
        return raw

    @classmethod
    def from_dict(cls, d: dict) -> TrendConfig:
        sub_map = {
            "h1_indicators": IndicatorParams,
            "d1_indicators": IndicatorParams,
            "m15_indicators": IndicatorParams,
            "regime": RegimeParams,
            "setup": TrendSetupParams,
            "confirmation": TrendConfirmationParams,
            "entry": TrendEntryParams,
            "stops": TrendStopParams,
            "exits": TrendExitParams,
            "trail": TrendTrailParams,
            "risk": TrendRiskParams,
            "limits": TrendLimitParams,
            "filters": TrendPerpFilterParams,
            "reentry": TrendReentryParams,
            "symbol_filter": TrendSymbolFilterParams,
        }
        kwargs: dict = {}
        for key, val in d.items():
            if key in sub_map and isinstance(val, dict):
                kwargs[key] = sub_map[key](**val)
            else:
                kwargs[key] = val
        return cls(**kwargs)
