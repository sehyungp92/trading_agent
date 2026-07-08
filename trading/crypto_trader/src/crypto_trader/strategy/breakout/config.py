"""Breakout strategy configuration — nested dataclasses with maximally loose defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class IndicatorParams:
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 100
    adx_period: int = 14
    atr_period: int = 14
    rsi_period: int = 14
    volume_ma_period: int = 20
    atr_avg_period: int = 20


@dataclass
class ProfileParams:
    """Volume profile computation parameters."""
    lookback_bars: int = 36          # Baked R1 (was 24)
    num_bins: int = 50               # Price bins for volume histogram
    value_area_pct: float = 0.70     # 70% of volume = Value Area
    hvn_threshold_pct: float = 1.2   # Bins with vol >= 1.2x avg = HVN (more zones → more trades)
    lvn_threshold_pct: float = 0.5   # Bins with vol <= 0.5x avg = LVN
    recalc_interval_bars: int = 2    # Recalculate every 2 M30 bars (1h)
    min_bars: int = 20               # Min bars for valid profile


@dataclass
class BalanceParams:
    """Balance area (consolidation) detection parameters."""
    min_bars_in_zone: int = 4        # Baked round 3 (was 6)
    zone_width_atr: float = 1.2      # Zone width as ATR multiple (spec: ~1.2)
    min_touches: int = 2             # Min returns to zone center (loose)
    max_zone_age_bars: int = 24      # Zone expires after 12h at M30
    require_volume_contraction: bool = False
    contraction_threshold: float = 0.8
    dedup_atr_frac: float = 0.3      # Dedup zones within this ATR fraction


@dataclass
class ContextParams:
    """H4 directional bias parameters."""
    h4_adx_threshold: float = 12.0   # Low threshold for permissive default
    allow_countertrend: bool = True   # Allow B-grade countertrend trades
    require_ema_alignment: bool = False
    strong_min_adx: float = 20.0     # ADX threshold for "strong" bias


@dataclass
class BreakoutSetupParams:
    """Breakout validation parameters."""
    min_breakout_atr: float = 0.2    # Min breakout distance from edge
    max_breakout_atr: float = 2.0    # Max (too far = chase)
    require_volume_surge: bool = False
    volume_surge_mult: float = 1.3   # Volume multiple for surge confluence
    min_lvn_runway_atr: float = 0.3  # Min LVN space ahead (loose)
    min_confluences_a_plus: int = 4
    min_confluences_a: int = 2
    min_confluences_b: int = 0
    min_room_r_a: float = 1.8       # Min projected R for A-grade
    min_room_r_b: float = 1.2       # Min projected R for B-grade
    body_ratio_min: float = 0.4675   # Baked round 3 (was 0.30)
    relaxed_body_enabled: bool = False
    relaxed_body_min: float = 0.35
    relaxed_body_min_confluences: int = 5
    relaxed_body_min_room_r: float = 1.4
    relaxed_body_require_volume_surge: bool = True
    relaxed_body_risk_scale: float = 0.5


@dataclass
class BreakoutConfirmParams:
    """Confirmation parameters for both entry models."""
    enable_model1: bool = True       # Breakout-close (aggressive)
    enable_model2: bool = False      # Disabled — 25% WR, -1.80R on 4 trades in R1
    retest_max_bars: int = 6         # Max bars to wait for retest
    retest_zone_atr: float = 0.5     # How close to edge = "retest" (ATR frac)
    retest_require_rejection: bool = False
    retest_require_volume_decline: bool = False
    volume_decline_threshold: float = 0.8
    model1_require_volume: bool = True           # Gate on volume (not rubber-stamp)
    model1_min_volume_mult: float = 1.0          # At least average volume
    model1_require_direction_close: bool = False  # Baked R1 (was True; 12→17 trades)


@dataclass
class BreakoutEntryParams:
    """Entry order generation parameters."""
    model1_entry_on_close: bool = True   # MARKET order for Model 1
    model2_entry_on_close: bool = True   # MARKET on retest close
    model2_entry_on_break: bool = False  # STOP order at breakout level
    max_bars_after_signal: int = 3       # Chase rule


@dataclass
class BreakoutStopParams:
    """Stop placement parameters."""
    use_balance_edge: bool = True    # Stop at opposite balance edge
    use_retest_extreme: bool = True  # Model 2: retest candle extreme
    use_atr_stop: bool = True        # ATR-based fallback
    atr_mult: float = 1.0            # Baked round 3 (was 1.3)
    use_farther: bool = False        # Use closer stop (avg loser -0.469R → ~-0.25R)
    min_stop_atr: float = 0.8       # Min stop distance (ATR)
    buffer_pct: float = 0.001        # Tiny buffer beyond stop


@dataclass
class BreakoutExitParams:
    """Exit management parameters."""
    tp1_r: float = 0.8               # Match trend pattern (was 0.5 — fired on losers)
    tp1_frac: float = 0.3            # Keep 70% runner (was 0.65 — killed upside)
    tp2_r: float = 2.0               # Let winners run (match trend 2.0R)
    tp2_frac: float = 0.4            # 40% at TP2, 30% trail runner
    runner_frac: float = 0.3         # 1.0 - 0.3 - 0.4 = 0.3
    time_stop_bars: int = 16         # Max bars before time exit (8h at M30; baked round 2)
    time_stop_min_progress_r: float = 0.3
    time_stop_action: str = "reduce" # Keep 50% alive (match trend)
    be_after_tp1: bool = True
    be_buffer_r: float = 0.525       # Baked R1 (was 0.3)
    be_min_bars_above: int = 2
    invalidation_exit: bool = True   # Exit if price re-enters balance
    invalidation_depth_atr: float = 1.2  # Require deeper penetration (was 0.8)
    invalidation_min_bars: int = 3       # Don't invalidate before trade develops
    early_lock_enabled: bool = False
    early_lock_mfe_r: float = 0.35
    early_lock_stop_r: float = 0.1
    quick_exit_enabled: bool = True   # Cut stagnant trades early
    quick_exit_bars: int = 4
    quick_exit_max_mfe_r: float = 0.2   # Slightly more permissive (was 0.15)
    quick_exit_max_r: float = -0.2      # Exit sooner on negative R (was -0.3)


@dataclass
class BreakoutTrailParams:
    """Trailing stop parameters."""
    trail_r_adaptive: bool = True
    trail_buffer_wide: float = 1.5
    trail_buffer_tight: float = 0.1575   # Baked round 4
    trail_r_ceiling: float = 2.0
    trail_activation_r: float = 0.3    # Earlier trail (was 0.5; losers peaking 0.3R get protection)
    trail_activation_bars: int = 4     # Breakout momentum fades faster (was 6)
    structure_trail_enabled: bool = False  # Baked round 3 (was True)
    structure_swing_lookback: int = 5
    ema_trail_period: int = 20


@dataclass
class BreakoutRiskParams:
    """Position sizing and risk parameters."""
    risk_pct_a_plus: float = 0.0225  # 2.25% for A+ (perp-appropriate)
    risk_pct_a: float = 0.01875     # 1.875% for A
    risk_pct_b: float = 0.015       # 1.5% for B (optimal from risk sweep)
    max_risk_pct: float = 0.025
    max_leverage_major: float = 10.0
    max_leverage_alt: float = 8.0
    min_leverage: float = 0.1
    major_symbols: tuple[str, ...] = ("BTC", "ETH")


@dataclass
class BreakoutLimitParams:
    """Daily and session operating limits."""
    max_concurrent_positions: int = 3
    max_daily_loss_pct: float = 0.025   # 2.5% daily drawdown
    max_consecutive_losses: int = 3
    max_trades_per_day: int = 5
    max_correlated_risk_pct: float = 0.0125  # 1.25% correlated
    max_weekly_loss_pct: float = 0.05        # 5% weekly


@dataclass
class BreakoutFilterParams:
    """Optional filters (all disabled by default for optimization)."""
    session_filter_enabled: bool = False
    session_avoid_hours: tuple[int, ...] = ()
    funding_filter_enabled: bool = False
    funding_extreme_threshold: float = 0.001
    oi_filter_enabled: bool = False      # Stub — no OI data


@dataclass
class BreakoutReentryParams:
    """Re-entry after stop-out."""
    enabled: bool = True
    cooldown_bars: int = 3
    max_wait_bars: int = 12
    max_loss_r: float = 1.5
    max_reentries: int = 1
    min_confluences_override: int = 0
    risk_scale: float = 1.0


@dataclass
class BreakoutSymbolFilterParams:
    """Per-symbol direction gating."""
    btc_direction: str = "both"
    eth_direction: str = "long_only"  # Baked round 4 fix (6 shorts: 0% WR, -1.71R)
    sol_direction: str = "both"
    btc_relaxed_body_direction: str = "disabled"
    eth_relaxed_body_direction: str = "disabled"
    sol_relaxed_body_direction: str = "disabled"


@dataclass
class BreakoutConfig:
    """Top-level breakout strategy configuration."""
    m30_indicators: IndicatorParams = field(
        default_factory=lambda: IndicatorParams(ema_fast=20, ema_mid=50, ema_slow=100)
    )
    h4_indicators: IndicatorParams = field(
        default_factory=lambda: IndicatorParams(ema_fast=20, ema_mid=50, ema_slow=50)
    )
    profile: ProfileParams = field(default_factory=ProfileParams)
    balance: BalanceParams = field(default_factory=BalanceParams)
    context: ContextParams = field(default_factory=ContextParams)
    setup: BreakoutSetupParams = field(default_factory=BreakoutSetupParams)
    confirmation: BreakoutConfirmParams = field(default_factory=BreakoutConfirmParams)
    entry: BreakoutEntryParams = field(default_factory=BreakoutEntryParams)
    stops: BreakoutStopParams = field(default_factory=BreakoutStopParams)
    exits: BreakoutExitParams = field(default_factory=BreakoutExitParams)
    trail: BreakoutTrailParams = field(default_factory=BreakoutTrailParams)
    risk: BreakoutRiskParams = field(default_factory=BreakoutRiskParams)
    limits: BreakoutLimitParams = field(default_factory=BreakoutLimitParams)
    filters: BreakoutFilterParams = field(default_factory=BreakoutFilterParams)
    reentry: BreakoutReentryParams = field(default_factory=BreakoutReentryParams)
    symbol_filter: BreakoutSymbolFilterParams = field(default_factory=BreakoutSymbolFilterParams)
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
    def from_dict(cls, d: dict) -> BreakoutConfig:
        sub_map = {
            "m30_indicators": IndicatorParams,
            "h4_indicators": IndicatorParams,
            "profile": ProfileParams,
            "balance": BalanceParams,
            "context": ContextParams,
            "setup": BreakoutSetupParams,
            "confirmation": BreakoutConfirmParams,
            "entry": BreakoutEntryParams,
            "stops": BreakoutStopParams,
            "exits": BreakoutExitParams,
            "trail": BreakoutTrailParams,
            "risk": BreakoutRiskParams,
            "limits": BreakoutLimitParams,
            "filters": BreakoutFilterParams,
            "reentry": BreakoutReentryParams,
            "symbol_filter": BreakoutSymbolFilterParams,
        }
        kwargs: dict = {}
        for key, val in d.items():
            if key in sub_map and isinstance(val, dict):
                kwargs[key] = sub_map[key](**val)
            else:
                kwargs[key] = val
        return cls(**kwargs)
