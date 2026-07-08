"""Momentum strategy configuration — nested dataclasses with spec defaults."""

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
class BiasParams:
    min_4h_conditions: int = 2
    min_1h_conditions: int = 2
    h4_ema_slope_lookback: int = 5
    h4_structure_lookback: int = 20
    h1_adx_threshold: float = 15.0


@dataclass
class SetupParams:
    min_confluences_a: int = 2
    min_confluences_b: int = 0
    min_room_a: float = 2.0
    min_room_b: float = 1.2
    fib_low: float = 0.382
    fib_high: float = 0.618
    use_vwap: bool = False
    reject_mid_nowhere: bool = True
    reject_parabolic_extension: bool = False
    reject_impulsive_breakdown: bool = True
    reject_extended_reaction: bool = False
    impulse_lookback: int = 20
    swing_lookback: int = 10
    use_rsi_pullback_filter: bool = True
    rsi_pullback_threshold: float = 40.0


@dataclass
class ConfirmationParams:
    enable_engulfing: bool = True
    enable_hammer: bool = True
    enable_inside_bar: bool = True
    enable_micro_shift: bool = True
    enable_base_break: bool = True
    require_volume_confirm: bool = True
    enforce_volume_on_trigger: bool = False
    enforce_volume_on_weak_confirmations: bool = False
    volume_threshold_mult: float = 1.0
    require_zone_proximity: bool = False
    zone_proximity_atr: float = 0.5
    hammer_wick_ratio: float = 2.0
    micro_shift_min_bars: int = 3
    base_break_min_bars: int = 2
    base_break_max_bars: int = 4
    min_confluences_for_weak: int = 2
    weak_confirmations: tuple[str, ...] = ("micro_structure_shift", "shooting_star")


@dataclass
class EntryParams:
    entry_on_close: bool = True
    entry_on_break: bool = False
    max_bars_after_confirmation: int = 3
    mode: str = "legacy"


@dataclass
class StopParams:
    atr_buffer_mult: float = 0.3
    alt_atr_buffer_mult: float = 0.5
    atr_buffer_min: float = 0.15
    atr_buffer_max: float = 0.6
    min_stop_atr_mult: float = 2.0
    swing_lookback: int = 10
    major_symbols: tuple[str, ...] = ("BTC", "ETH")


@dataclass
class ExitParams:
    tp1_r: float = 1.2
    tp1_frac: float = 0.16
    tp2_r: float = 2.5
    tp2_frac: float = 0.20
    be_acceptance_bars: int = 2
    proof_lock_enabled: bool = False
    proof_lock_trigger_r: float = 0.5
    proof_lock_stop_r: float = 0.0
    proof_lock_min_bars: int = 2
    followthrough_exit_enabled: bool = False
    followthrough_peak_r: float = 0.35
    followthrough_bars: int = 4
    followthrough_floor_r: float = -0.1
    followthrough_scope: str = "all"
    mfe_retrace_exit_enabled: bool = False
    mfe_retrace_trigger_r: float = 1.5
    mfe_retrace_giveback_r: float = 1.25
    mfe_retrace_min_r: float = 0.75
    mfe_retrace_min_bars: int = 6
    mfe_retrace_scope: str = "all"
    soft_time_stop_bars: int = 12
    soft_time_stop_min_r: float = 0.5
    hard_time_stop_bars: int = 20
    hard_time_stop_min_r: float = 1.0
    enable_structure_break_exit: bool = True
    enable_reversal_candle_exit: bool = True
    enable_counter_volume_exit: bool = True
    enable_h1_thesis_exit: bool = True
    enable_disorderly_exit: bool = True
    enable_funding_exit: bool = True
    structure_break_body_atr_mult: float = 1.5
    reversal_body_atr_mult: float = 1.0
    reversal_volume_mult: float = 1.5
    be_buffer_r: float = 0.2
    # Quick exit for stagnant trades
    quick_exit_enabled: bool = True
    quick_exit_bars: int = 6
    quick_exit_max_mfe_r: float = 0.15
    quick_exit_max_r: float = -0.3


@dataclass
class TrailParams:
    trail_mode: str = "components"
    trail_behind_structure: bool = True
    trail_behind_ema: bool = True
    trail_ema_period: int = 25
    trail_chandelier_lookback: int = 20
    trail_chandelier_atr_mult: float = 2.5
    trail_atr_buffer: float = 0.5
    trail_use_tightest: bool = True
    trail_activation_bars: int = 3
    trail_activation_r: float = 0.5
    trail_warmup_bars: int = 5
    trail_warmup_buffer_mult: float = 1.0
    # R-adaptive buffer: scales inversely with R-multiple
    trail_r_adaptive: bool = True
    trail_r_basis: str = "current"
    trail_buffer_wide: float = 1.5
    trail_buffer_tight: float = 0.3
    trail_r_ceiling: float = 2.0
    runner_trail_enabled: bool = False
    runner_trail_scope: str = "all"
    runner_trigger_r: float = 1.5
    runner_trail_r_basis: str = "mfe"
    runner_trail_buffer_wide: float = 1.0
    runner_trail_buffer_tight: float = 0.2
    runner_trail_r_ceiling: float = 1.0
    # MFE-floor: prevent trail from getting too tight on proven trades
    trail_mfe_floor_enabled: bool = False
    trail_mfe_floor_threshold: float = 0.8
    trail_mfe_floor_buffer: float = 0.5


@dataclass
class RiskParams:
    risk_pct_a: float = 0.02
    risk_pct_b: float = 0.0125
    max_leverage_major: float = 10.0
    min_leverage: float = 1.5
    max_leverage_alt: float = 8.0
    min_liquidation_buffer_atr: float = 3.0
    max_correlated_risk: float = 0.03
    max_gross_risk: float = 0.05
    max_concurrent_positions: int = 3
    major_symbols: tuple[str, ...] = ("BTC", "ETH")


@dataclass
class SessionParams:
    london_start: int = 8
    london_end: int = 16
    ny_start: int = 13
    ny_end: int = 21
    overlap_start: int = 13
    overlap_end: int = 16
    reduced_window_require_a: bool = False


@dataclass
class FilterParams:
    atr_expansion_mult: float = 2.5
    atr_compression_mult: float = 0.3
    adx_chop_threshold: float = 10.0
    funding_extreme_threshold: float = 0.01
    funding_moderate_threshold: float = 0.005


@dataclass
class DailyLimitParams:
    max_consecutive_losses: int = 2
    max_daily_loss_pct: float = 0.015
    max_trades_per_day: int = 4


@dataclass
class FundingHoldParams:
    avoid_adverse_funding: bool = True
    funding_exit_threshold: float = 0.01


@dataclass
class ReentryParams:
    """Re-entry after stop-out within valid trends."""
    enabled: bool = True
    cooldown_bars: int = 3
    max_loss_r: float = 1.5
    max_reentries: int = 1
    min_confluences_override: int = 0


@dataclass
class SymbolFilterParams:
    """Per-symbol direction gating. Values: 'both', 'long_only', 'short_only', 'disabled'"""
    btc_direction: str = "both"
    eth_direction: str = "both"
    sol_direction: str = "both"


@dataclass
class MomentumConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    indicators: IndicatorParams = field(default_factory=IndicatorParams)
    bias: BiasParams = field(default_factory=BiasParams)
    setup: SetupParams = field(default_factory=SetupParams)
    confirmation: ConfirmationParams = field(default_factory=ConfirmationParams)
    entry: EntryParams = field(default_factory=EntryParams)
    stops: StopParams = field(default_factory=StopParams)
    exits: ExitParams = field(default_factory=ExitParams)
    trail: TrailParams = field(default_factory=TrailParams)
    risk: RiskParams = field(default_factory=RiskParams)
    session: SessionParams = field(default_factory=SessionParams)
    filters: FilterParams = field(default_factory=FilterParams)
    daily_limits: DailyLimitParams = field(default_factory=DailyLimitParams)
    funding_hold: FundingHoldParams = field(default_factory=FundingHoldParams)
    symbol_filter: SymbolFilterParams = field(default_factory=SymbolFilterParams)
    reentry: ReentryParams = field(default_factory=ReentryParams)

    def to_dict(self) -> dict:
        """Serialize config to a plain dict (round-trips with from_dict)."""
        raw = asdict(self)
        # Convert tuples back to lists for JSON compatibility
        for section in raw.values():
            if isinstance(section, dict):
                for k, v in section.items():
                    if isinstance(v, tuple):
                        section[k] = list(v)
        return raw

    @classmethod
    def from_dict(cls, d: dict) -> MomentumConfig:
        sub_map = {
            "indicators": IndicatorParams,
            "bias": BiasParams,
            "setup": SetupParams,
            "confirmation": ConfirmationParams,
            "entry": EntryParams,
            "stops": StopParams,
            "exits": ExitParams,
            "trail": TrailParams,
            "risk": RiskParams,
            "session": SessionParams,
            "filters": FilterParams,
            "daily_limits": DailyLimitParams,
            "funding_hold": FundingHoldParams,
            "symbol_filter": SymbolFilterParams,
            "reentry": ReentryParams,
        }
        kwargs: dict = {}
        for key, val in d.items():
            if key in sub_map and isinstance(val, dict):
                kwargs[key] = sub_map[key](**val)
            else:
                kwargs[key] = val
        return cls(**kwargs)
