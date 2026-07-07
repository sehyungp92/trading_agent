"""TPC strategy constants and per-symbol configuration."""
from __future__ import annotations

from dataclasses import dataclass

from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry

STRATEGY_ID = "TPC"


@dataclass(frozen=True)
class TPCSymbolConfig:
    symbol: str

    ma_50_period: int = 50
    ma_100_period: int = 100
    rsi_period: int = 14
    adx_period: int = 14
    rsi_long_band: tuple[float, float] = (45.0, 70.0)
    rsi_short_band: tuple[float, float] = (30.0, 55.0)
    rsi_a_plus_long_max: float = 75.0
    rsi_a_plus_short_min: float = 25.0
    min_adx_4h: float = 0.0
    max_adx_4h: float = 0.0
    require_di_alignment: bool = True
    min_ma50_slope_atr_4h: float = 0.0
    min_ma100_slope_atr_4h: float = 0.06
    max_extension_atr_mult: float = 2.0

    ema_20_period: int = 20
    ema_50_period: int = 50
    fib_a_low: float = 0.33
    fib_a_high: float = 0.8
    fib_b_low: float = 0.236
    fib_b_high: float = 0.382
    fib_c_low: float = 0.118
    fib_c_high: float = 0.236
    pullback_min_bars_1h: int = 3
    pullback_max_bars_1h: int = 10
    type_b_enabled: bool = False
    type_b_requires_a_plus: bool = True
    type_c_enabled: bool = True
    type_c_mode: str = "real_reentry"
    type_c_requires_a_plus: bool = False
    second_entry_min_wait_bars_15m: int = 1
    second_entry_max_wait_bars_15m: int = 16
    second_entry_require_vwap: bool = True
    second_entry_require_structure: bool = True
    second_entry_score_min: int = 15
    second_entry_min_source_score: float = 16.0
    second_entry_requires_source_a_plus: bool = False
    second_entry_structure_buffer_atr_mult: float = 0.05
    pullback_orderly_required: bool = False
    pullback_volume_contract_max: float = 0.0
    type_a_value_hits_min: int = 1
    type_b_value_hits_min: int = 1
    type_c_value_hits_min: int = 1
    pb30_pullback_enabled: bool = True
    pb30_ema20_value_touch_enabled: bool = False
    pb30_pullback_min_bars_30m: int = 6
    pb30_pullback_max_bars_30m: int = 20
    pb30_pullback_orderly_required: bool = True
    pb30_fib_a_low: float = 0.0
    pb30_fib_a_high: float = 0.0
    pb30_type_a_value_hits_min: int = 0
    pb30_ema20_context_enabled: bool = False
    pb30_ema20_context_mode: str = "touch"
    pb30_ema20_context_lookback_bars_30m: int = 0
    pb30_ema20_context_distance_atr: float = 0.15
    pb30_ma_transition_enabled: bool = False
    pb30_ma_transition_mode: str = "fast_slope"
    pb30_ma_transition_lookback_bars_30m: int = 6
    pb30_ma_transition_min_slope_atr: float = 0.0
    pb30_ma_transition_window_bars_30m: int = 12
    pb30_confirmation_required: int = 1
    pb30_confirmation_combo_mode: str = "structure_or_vwap"
    pb30_entry_order_model: str = ""
    pb30_ema20_value_touch_entry_order_model: str = "ema20_30m_value_touch_market"
    ema20_value_touch_entry_enabled: bool = False
    ema20_value_touch_entry_order_model: str = "ema20_value_touch_market"
    ema20_value_touch_confirmation_required: int = -1
    ema20_value_touch_confirmation_combo_mode: str = ""
    max_stop_atr_mult: float = 1.5
    min_stop_atr_mult: float = 0.15
    daily_room_min_r: float = 2.0

    confirmation_required: int = 1
    confirmation_max_count: int = 0
    volume_expansion_mult: float = 1.3
    confirmation_combo_mode: str = "structure_or_vwap"
    require_vwap_confirmation: bool = False
    require_structure_confirmation: bool = False
    require_micro_break_confirmation: bool = False
    require_volume_confirmation: bool = False
    vwap_anchor_hour: int = 9
    vwap_anchor_minute: int = 30
    atr_period: int = 14
    stop_buffer_atr_mult: float = 0.25
    signal_stop_buffer_atr_mult: float = 0.12
    initial_stop_source: str = "signal"
    entry_order_model: str = "structure_stop"
    entry_stop_limit_atr_mult: float = 0.08
    entry_adaptive_stop_limit_min_atr_mult: float = 0.08
    entry_adaptive_stop_limit_max_atr_mult: float = 0.30
    entry_order_ttl_hours: float = 4.0

    t1_r: float = 1.5
    t1_partial_pct: float = 0.45
    t2_r: float = 2.0
    t2_partial_pct: float = 0.275
    t1_stop_r: float = 0.4
    profit_floor_ladder: tuple[tuple[float, float], ...] = ()
    trail_after_t1_30m_bars: int = 8
    trail_after_t2_1h_bars: int = 0
    trail_use_vwap_after_t1: bool = True
    max_hold_bars_15m: int = 36
    time_stop_min_mfe_r: float = 0.5
    runner_max_hold_bars_15m: int = 0
    stall_exit_bars_15m: int = 52
    stall_exit_min_mfe_r: float = 1.0
    stall_exit_max_current_r: float = 0.2
    mfe_giveback_trigger_r: float = 2.0
    mfe_giveback_retain_frac: float = 0.40
    mfe_giveback_lock_r: float = 0.5
    mfe_giveback_after_t1_only: bool = False
    addon_enabled: bool = False
    addon_trigger_r: float = 1.75
    addon_size_mult: float = 0.25
    addon_min_score: int = 17
    addon_requires_t1: bool = True
    addon_require_vwap_hold: bool = True
    addon_require_structure_hold: bool = True
    addon_max_total_risk_pct: float = 0.03
    addon_max_notional_pct: float = 0.0

    score_a_plus_min: int = 13
    score_a_min: int = 10
    score_b_min: int = 9
    score_model: str = "alpha7"
    min_short_score: int = 0
    shorts_require_a_plus: bool = False
    longs_enabled: bool = True
    shorts_enabled: bool = True
    risk_a_plus_pct: float = 0.020
    risk_a_pct: float = 0.012
    risk_b_pct: float = 0.009
    max_risk_pct: float = 0.020
    dynamic_risk_enabled: bool = False
    dynamic_risk_score_floor: float = 14.0
    dynamic_risk_score_ceiling: float = 21.0
    dynamic_risk_min_mult: float = 0.75
    dynamic_risk_max_mult: float = 1.50
    dynamic_risk_curve: float = 1.0
    max_daily_loss_r: float = 2.0
    max_weekly_loss_r: float = 5.0
    max_open_risk_r: float = 1.5
    max_position_notional_pct: float = 6.0

    asset_context_enabled: bool = False
    asset_context_min_score: float = -1.0
    asset_context_symbol: str = ""

    primary_windows_et: tuple[tuple[int, int, int, int], ...] = ()
    avoid_windows_et: tuple[tuple[int, int, int, int], ...] = ((11, 0, 12, 0),)
    news_avoidance_minutes: int = 75

    tick_size: float = 0.01
    multiplier: float = 1.0
    exchange: str = "SMART"
    sec_type: str = "STK"
    primary_exchange: str = ""
    contract_expiry: str = ""


SYMBOL_CONFIGS: dict[str, TPCSymbolConfig] = {
    "QQQ": TPCSymbolConfig(
        symbol="QQQ",
        stop_buffer_atr_mult=0.25,
        t1_r=0.9,
        primary_windows_et=((9, 35, 12, 15), (13, 30, 15, 45)),
        avoid_windows_et=((11, 0, 12, 0),),
        type_b_enabled=True,
        type_b_requires_a_plus=False,
        confirmation_required=2,
        fib_b_low=0.2,
        fib_b_high=0.38,
        min_short_score=12,
        daily_room_min_r=1.6,
        require_di_alignment=False,
        min_ma100_slope_atr_4h=0.03,
        max_extension_atr_mult=2.25,
        t1_partial_pct=0.70,
        t1_stop_r=0.35,
        addon_min_score=12,
        asset_context_enabled=True,
        asset_context_min_score=-0.1,
        shorts_require_a_plus=True,
        asset_context_symbol="NQ",
        primary_exchange="NASDAQ",
    ),
    "GLD": TPCSymbolConfig(
        symbol="GLD",
        stop_buffer_atr_mult=0.20,
        t1_r=1.1,
        t1_partial_pct=0.50,
        primary_windows_et=((8, 0, 11, 30), (13, 0, 16, 0)),
        entry_order_model="market_next_bar",
        daily_room_min_r=3.0,
        confirmation_required=2,
        confirmation_combo_mode="structure_or_vwap",
        score_a_min=11,
        score_b_min=10,
        asset_context_enabled=True,
        require_structure_confirmation=True,
        max_extension_atr_mult=1.5,
        min_ma50_slope_atr_4h=0.04,
        min_ma100_slope_atr_4h=0.06,
        require_di_alignment=True,
        asset_context_symbol="GC",
        primary_exchange="ARCA",
    ),
}


def build_instruments() -> dict[str, Instrument]:
    instruments: dict[str, Instrument] = {}
    for sym, cfg in SYMBOL_CONFIGS.items():
        inst = Instrument(
            symbol=sym,
            root=sym,
            venue=cfg.exchange,
            tick_size=cfg.tick_size,
            tick_value=cfg.tick_size * cfg.multiplier,
            multiplier=cfg.multiplier,
            contract_expiry=cfg.contract_expiry,
        )
        InstrumentRegistry.register(inst)
        instruments[sym] = inst
    return instruments
