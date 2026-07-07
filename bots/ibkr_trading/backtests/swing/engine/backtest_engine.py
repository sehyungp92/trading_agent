"""Core single-symbol bar-by-bar backtesting engine.

Mirrors strategy/engine.py logic but uses SimBroker instead of OMS/IBKR.
All strategy logic is called via the pure functions in strategy/*.py.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace as _dc_replace
from datetime import datetime, timedelta, timezone

import numpy as np

from backtests.shared.parity.legacy_result_outputs import trade_outcomes_from_records
from strategies.swing.atrss.core import logic as atrss_core_logic
from strategies.swing.atrss.core.state import (
    ATRSSAddOnARequest,
    ATRSSCoreState,
    ATRSSEntryRequest,
    ATRSSFill,
    ATRSSFlattenRequest,
    ATRSSOrderUpdate,
    ATRSSPartialExitRequest,
    ATRSSStopUpdateRequest,
)
from libs.broker_ibkr.risk_support.tick_rules import round_to_tick
from strategies.swing.atrss import allocator, signals, stops
from strategies.swing.atrss.config import (
    ADDON_A_SIZE_MULT,
    ADDON_B_SIZE_MULT,
    ARM_WINDOW_HOURS,
    BREAKOUT_REQUIRE_DIRECTIONAL_CANDLE,
    BREAKOUT_RETRACE_ENTRY_FRAC,
    BREAKOUT_RETRACE_LIMIT_FRAC,
    DYNAMIC_RISK_STRONG_TREND_MULT,
    DYNAMIC_RISK_WEAK_TREND_MULT,
    EARLY_STALL_CHECK_HOURS,
    EARLY_STALL_MFE_THRESHOLD,
    EARLY_STALL_PARTIAL_FRAC,
    FIXED_QTY_ADDON_B_ENABLED,
    FIXED_QTY_REGIME_SCALING_ENABLED,
    FIXED_QTY_STRONG_TREND_MULT,
    FIXED_QTY_WEAK_TREND_MULT,
    MAX_ENTRY_SLIP_ATR,
    MAX_HOLD_HOURS,
    MOMENTUM_TOLERANCE_ATR,
    ORDER_EXPIRY_HOURS,
    QUALITY_GATE_THRESHOLD,
    RECOVERY_TOLERANCE_ATR,
    RECOVERY_TOLERANCE_ATR_STRONG,
    RECOVERY_TOLERANCE_ATR_TREND,
    STALL_CHECK_HOURS,
    STALL_MFE_THRESHOLD,
    TP1_FRAC,
    TP1_R,
    TP2_FRAC,
    TP2_R,
    TREND_STOP_TIGHTENING,
    SymbolConfig,
)
from strategies.swing.atrss.indicators import compute_daily_state, compute_hourly_state
from strategies.swing.atrss.models import (
    BreakoutArmState,
    Candidate,
    CandidateType,
    DailyState,
    Direction,
    HourlyState,
    LegType,
    PositionBook,
    PositionLeg,
    ReentryState,
    Regime,
)

from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from backtests.swing.config import AblationFlags, BacktestConfig, SlippageConfig
from backtests.swing.data.preprocessing import NumpyBars
from backtests.swing.engine.sim_broker import (
    FillResult,
    FillStatus,
    OrderSide,
    OrderType,
    SimBroker,
    SimOrder,
)

logger = logging.getLogger(__name__)

# Pre-cached timezone for hot-path use (avoids per-call import + construction)
try:
    from zoneinfo import ZoneInfo as _ZoneInfo
    _ET_TZ = _ZoneInfo("America/New_York")
except ImportError:
    _ET_TZ = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Trade record for post-hoc analysis
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """One completed trade (entry + exit)."""

    symbol: str = ""
    direction: int = 0
    entry_type: str = ""       # PULLBACK, BREAKOUT, REVERSE
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    qty: int = 0
    initial_stop: float = 0.0
    exit_reason: str = ""      # STOP, FLATTEN_BIAS_FLIP, FLATTEN_TIME_DECAY
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars_held: int = 0
    commission: float = 0.0
    addon_a_qty: int = 0
    addon_b_qty: int = 0
    leg_type: str = "BASE"
    adx_entry: float = 0.0
    score_entry: float = 0.0
    touch_distance_atr: float = 0.0
    di_agrees: bool = False
    quality_score: float = 0.0
    regime_entry: str = ""
    signal_time: datetime | None = None
    fill_time: datetime | None = None
    signal_bar_index: int = -1
    fill_bar_index: int = -1
    campaign_id: str = ""

    def __post_init__(self) -> None:
        if self.gross_pnl == 0.0 and self.pnl_dollars != 0.0:
            self.gross_pnl = self.pnl_dollars
        if self.net_pnl == 0.0 and (self.pnl_dollars != 0.0 or self.commission != 0.0):
            self.net_pnl = self.pnl_dollars - self.commission


@dataclass
class SignalFunnelStats:
    """Bar-level accounting of where potential signals get filtered."""
    total_bars: int = 0
    bars_nan: int = 0
    bars_warmup: int = 0
    bars_in_position: int = 0
    bars_entry_restricted: int = 0
    bars_bias_flat: int = 0
    bars_regime_range: int = 0
    bars_regime_trend: int = 0
    bars_regime_strong: int = 0
    pullback_signals: int = 0
    breakout_signals: int = 0
    reverse_signals: int = 0
    rejected_momentum: int = 0
    rejected_reentry: int = 0
    rejected_sizing: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_expired: int = 0
    orders_limit_rejected: int = 0
    bars_shorts_disabled: int = 0
    rejected_quality: int = 0
    breakout_arms_created: int = 0
    breakout_arms_expired: int = 0
    breakout_arms_converted: int = 0


@dataclass
class SymbolResult:
    """Result of backtesting a single symbol."""

    symbol: str
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    total_commission: float = 0.0
    # Diagnostic: daily bias distribution
    bias_days_long: int = 0
    bias_days_short: int = 0
    bias_days_flat: int = 0
    funnel: SignalFunnelStats | None = None
    order_metadata: list[dict] = field(default_factory=list)
    decision_stream: list[dict] = field(default_factory=list)
    trade_outcomes: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ablation context manager — patches module constants
# ---------------------------------------------------------------------------

class _AblationPatch:
    """Temporarily patch strategy module constants for ablation testing.

    Also applies param_overrides for module-level constants (cooldown hours,
    voucher window, confirm days, ADX slope gate).
    Safe in single-threaded / per-process execution.
    """

    def __init__(self, flags: AblationFlags, param_overrides: dict[str, float] | None = None):
        self.flags = flags
        self.overrides = param_overrides or {}
        self._patches: list[tuple[object, str, object]] = []

    def __enter__(self):
        import strategies.swing.atrss.config as scfg
        import strategies.swing.atrss.signals as ssig
        import strategies.swing.atrss.allocator as salloc

        f = self.flags
        ov = self.overrides

        if not f.conviction_gating:
            self._patch(scfg, "SCORE_REVERSE_MIN", 0)
            self._patch(ssig, "SCORE_REVERSE_MIN", 0)

        if not f.fast_confirm:
            self._patch(scfg, "FAST_CONFIRM_SCORE", 999)
            import strategies.swing.atrss.indicators as sind
            self._patch(sind, "FAST_CONFIRM_SCORE", 999)

        if not f.short_safety:
            self._patch(ssig, "short_safety_ok", lambda d: True)

        # --- Apply param_overrides for module-level constants ---
        if "cooldown_strong" in ov or "cooldown_trend" in ov or "cooldown_range" in ov:
            new_cd = dict(scfg.COOLDOWN_HOURS)
            if "cooldown_strong" in ov:
                new_cd["STRONG_TREND"] = int(ov["cooldown_strong"])
            if "cooldown_trend" in ov:
                new_cd["TREND"] = int(ov["cooldown_trend"])
            if "cooldown_range" in ov:
                new_cd["RANGE"] = int(ov["cooldown_range"])
            self._patch(scfg, "COOLDOWN_HOURS", new_cd)
            self._patch(ssig, "COOLDOWN_HOURS", new_cd)

        if "voucher_valid_hours" in ov:
            val = int(ov["voucher_valid_hours"])
            self._patch(scfg, "VOUCHER_VALID_HOURS", val)
            self._patch(ssig, "VOUCHER_VALID_HOURS", val)

        if "confirm_days_normal" in ov:
            import strategies.swing.atrss.indicators as sind
            val = int(ov["confirm_days_normal"])
            self._patch(scfg, "CONFIRM_DAYS_NORMAL", val)
            self._patch(sind, "CONFIRM_DAYS_NORMAL", val)

        if "pullback_lookback" in ov:
            import strategies.swing.atrss.indicators as sind
            val = int(ov["pullback_lookback"])
            self._patch(scfg, "PULLBACK_LOOKBACK", val)
            self._patch(sind, "PULLBACK_LOOKBACK", val)
        if "pullback_touch_tolerance_atr" in ov:
            import strategies.swing.atrss.indicators as sind
            val = float(ov["pullback_touch_tolerance_atr"])
            self._patch(scfg, "PULLBACK_TOUCH_TOLERANCE_ATR", val)
            self._patch(sind, "PULLBACK_TOUCH_TOLERANCE_ATR", val)
        if "pullback_touch_tolerance_pct" in ov:
            import strategies.swing.atrss.indicators as sind
            val = float(ov["pullback_touch_tolerance_pct"])
            self._patch(scfg, "PULLBACK_TOUCH_TOLERANCE_PCT", val)
            self._patch(sind, "PULLBACK_TOUCH_TOLERANCE_PCT", val)

        if "adx_slope_gate" in ov:
            import strategies.swing.atrss.indicators as sind
            val = float(ov["adx_slope_gate"])
            self._patch(scfg, "ADX_STRONG_SLOPE_FLOOR", val)
            self._patch(sind, "ADX_STRONG_SLOPE_FLOOR", val)

        # --- ADX regime thresholds (adx_on/adx_off now per-symbol in SymbolConfig) ---
        if "adx_strong" in ov:
            import strategies.swing.atrss.indicators as sind
            self._patch(scfg, "ADX_STRONG", int(ov["adx_strong"]))
            self._patch(sind, "ADX_STRONG", int(ov["adx_strong"]))

        # --- Entry thresholds ---
        if "score_reverse_min" in ov:
            val = int(ov["score_reverse_min"])
            self._patch(scfg, "SCORE_REVERSE_MIN", val)
            self._patch(ssig, "SCORE_REVERSE_MIN", val)
        if "fast_confirm_score" in ov:
            import strategies.swing.atrss.indicators as sind
            val = int(ov["fast_confirm_score"])
            self._patch(scfg, "FAST_CONFIRM_SCORE", val)
            self._patch(sind, "FAST_CONFIRM_SCORE", val)
        if "fast_confirm_adx" in ov:
            import strategies.swing.atrss.indicators as sind
            val = int(ov["fast_confirm_adx"])
            self._patch(scfg, "FAST_CONFIRM_ADX", val)
            self._patch(sind, "FAST_CONFIRM_ADX", val)
        if "di_min" in ov:
            import strategies.swing.atrss.indicators as sind
            val = int(ov["di_min"])
            self._patch(scfg, "DI_MIN", val)
            self._patch(sind, "DI_MIN", val)
        if "sep_min" in ov:
            import strategies.swing.atrss.indicators as sind
            val = float(ov["sep_min"])
            self._patch(scfg, "SEP_MIN", val)
            self._patch(sind, "SEP_MIN", val)
        if "adx_min_struct" in ov:
            import strategies.swing.atrss.indicators as sind
            val = int(ov["adx_min_struct"])
            self._patch(scfg, "ADX_MIN_STRUCT", val)
            self._patch(sind, "ADX_MIN_STRUCT", val)

        # --- BE / stop management ---
        if "be_trigger_r" in ov:
            self._patch(scfg, "BE_TRIGGER_R", float(ov["be_trigger_r"]))
        if "chandelier_trigger_r" in ov:
            self._patch(scfg, "CHANDELIER_TRIGGER_R", float(ov["chandelier_trigger_r"]))
        if "be_atr_offset" in ov:
            self._patch(scfg, "BE_ATR_OFFSET", float(ov["be_atr_offset"]))

        # --- Module-level constants used by backtest_engine itself ---
        import sys
        _self_mod = sys.modules[__name__]

        if "tp1_r" in ov:
            val = float(ov["tp1_r"])
            self._patch(scfg, "TP1_R", val)
            self._patch(_self_mod, "TP1_R", val)
        if "tp1_frac" in ov:
            val = float(ov["tp1_frac"])
            self._patch(scfg, "TP1_FRAC", val)
            self._patch(_self_mod, "TP1_FRAC", val)
        if "tp2_r" in ov:
            val = float(ov["tp2_r"])
            self._patch(scfg, "TP2_R", val)
            self._patch(_self_mod, "TP2_R", val)
        if "tp2_frac" in ov:
            val = float(ov["tp2_frac"])
            self._patch(scfg, "TP2_FRAC", val)
            self._patch(_self_mod, "TP2_FRAC", val)
        if "max_hold_hours" in ov:
            val = int(ov["max_hold_hours"])
            self._patch(scfg, "MAX_HOLD_HOURS", val)
            self._patch(_self_mod, "MAX_HOLD_HOURS", val)
        if "early_stall_check_hours" in ov:
            val = int(ov["early_stall_check_hours"])
            self._patch(scfg, "EARLY_STALL_CHECK_HOURS", val)
            self._patch(_self_mod, "EARLY_STALL_CHECK_HOURS", val)
        if "early_stall_mfe_threshold" in ov:
            val = float(ov["early_stall_mfe_threshold"])
            self._patch(scfg, "EARLY_STALL_MFE_THRESHOLD", val)
            self._patch(_self_mod, "EARLY_STALL_MFE_THRESHOLD", val)
        if "early_stall_partial_frac" in ov:
            val = float(ov["early_stall_partial_frac"])
            self._patch(scfg, "EARLY_STALL_PARTIAL_FRAC", val)
            self._patch(_self_mod, "EARLY_STALL_PARTIAL_FRAC", val)
        if "stall_check_hours" in ov:
            val = int(ov["stall_check_hours"])
            self._patch(scfg, "STALL_CHECK_HOURS", val)
            self._patch(_self_mod, "STALL_CHECK_HOURS", val)
        if "stall_mfe_threshold" in ov:
            val = float(ov["stall_mfe_threshold"])
            self._patch(scfg, "STALL_MFE_THRESHOLD", val)
            self._patch(_self_mod, "STALL_MFE_THRESHOLD", val)
        if "order_expiry_hours" in ov:
            val = int(ov["order_expiry_hours"])
            self._patch(scfg, "ORDER_EXPIRY_HOURS", val)
            self._patch(_self_mod, "ORDER_EXPIRY_HOURS", val)
        if "max_entry_slip_atr" in ov:
            val = float(ov["max_entry_slip_atr"])
            self._patch(scfg, "MAX_ENTRY_SLIP_ATR", val)
            self._patch(_self_mod, "MAX_ENTRY_SLIP_ATR", val)
        if "trend_stop_tightening" in ov:
            val = float(ov["trend_stop_tightening"])
            self._patch(scfg, "TREND_STOP_TIGHTENING", val)
            self._patch(_self_mod, "TREND_STOP_TIGHTENING", val)
        if "quality_gate_threshold" in ov:
            val = float(ov["quality_gate_threshold"])
            self._patch(scfg, "QUALITY_GATE_THRESHOLD", val)
            self._patch(_self_mod, "QUALITY_GATE_THRESHOLD", val)
        if "max_portfolio_heat" in ov:
            val = float(ov["max_portfolio_heat"])
            self._patch(scfg, "MAX_PORTFOLIO_HEAT", val)
            self._patch(salloc, "MAX_PORTFOLIO_HEAT", val)
        if "fixed_qty_regime_scaling" in ov:
            val = bool(ov["fixed_qty_regime_scaling"])
            self._patch(scfg, "FIXED_QTY_REGIME_SCALING_ENABLED", val)
            self._patch(_self_mod, "FIXED_QTY_REGIME_SCALING_ENABLED", val)
        if "fixed_qty_strong_trend_mult" in ov:
            val = float(ov["fixed_qty_strong_trend_mult"])
            self._patch(scfg, "FIXED_QTY_STRONG_TREND_MULT", val)
            self._patch(_self_mod, "FIXED_QTY_STRONG_TREND_MULT", val)
        if "fixed_qty_weak_trend_mult" in ov:
            val = float(ov["fixed_qty_weak_trend_mult"])
            self._patch(scfg, "FIXED_QTY_WEAK_TREND_MULT", val)
            self._patch(_self_mod, "FIXED_QTY_WEAK_TREND_MULT", val)
        if "dynamic_risk_strong_trend_mult" in ov:
            val = float(ov["dynamic_risk_strong_trend_mult"])
            self._patch(scfg, "DYNAMIC_RISK_STRONG_TREND_MULT", val)
            self._patch(_self_mod, "DYNAMIC_RISK_STRONG_TREND_MULT", val)
        if "dynamic_risk_weak_trend_mult" in ov:
            val = float(ov["dynamic_risk_weak_trend_mult"])
            self._patch(scfg, "DYNAMIC_RISK_WEAK_TREND_MULT", val)
            self._patch(_self_mod, "DYNAMIC_RISK_WEAK_TREND_MULT", val)

        # --- Constants used by signals module ---
        if "momentum_tolerance_atr" in ov:
            val = float(ov["momentum_tolerance_atr"])
            self._patch(scfg, "MOMENTUM_TOLERANCE_ATR", val)
            self._patch(_self_mod, "MOMENTUM_TOLERANCE_ATR", val)
            self._patch(ssig, "MOMENTUM_TOLERANCE_ATR", val)
        if "pullback_momentum_filter" in ov:
            val = bool(ov["pullback_momentum_filter"])
            self._patch(scfg, "PULLBACK_MOMENTUM_FILTER_ENABLED", val)
            self._patch(ssig, "PULLBACK_MOMENTUM_FILTER_ENABLED", val)
        if "recovery_tolerance_atr" in ov:
            val = float(ov["recovery_tolerance_atr"])
            self._patch(scfg, "RECOVERY_TOLERANCE_ATR", val)
            self._patch(_self_mod, "RECOVERY_TOLERANCE_ATR", val)
            self._patch(ssig, "RECOVERY_TOLERANCE_ATR", val)
        if "recovery_tolerance_atr_trend" in ov:
            val = float(ov["recovery_tolerance_atr_trend"])
            self._patch(scfg, "RECOVERY_TOLERANCE_ATR_TREND", val)
            self._patch(_self_mod, "RECOVERY_TOLERANCE_ATR_TREND", val)
            self._patch(ssig, "RECOVERY_TOLERANCE_ATR_TREND", val)
        if "recovery_tolerance_atr_strong" in ov:
            val = float(ov["recovery_tolerance_atr_strong"])
            self._patch(scfg, "RECOVERY_TOLERANCE_ATR_STRONG", val)
            self._patch(_self_mod, "RECOVERY_TOLERANCE_ATR_STRONG", val)
            self._patch(ssig, "RECOVERY_TOLERANCE_ATR_STRONG", val)
        if "addon_a_r" in ov:
            val = float(ov["addon_a_r"])
            self._patch(scfg, "ADDON_A_R", val)
            self._patch(ssig, "ADDON_A_R", val)
        if "addon_b_r" in ov:
            val = float(ov["addon_b_r"])
            self._patch(scfg, "ADDON_B_R", val)
            self._patch(ssig, "ADDON_B_R", val)
        if "addon_a_size_mult" in ov:
            val = float(ov["addon_a_size_mult"])
            self._patch(scfg, "ADDON_A_SIZE_MULT", val)
            self._patch(_self_mod, "ADDON_A_SIZE_MULT", val)
        if "addon_b_size_mult" in ov:
            val = float(ov["addon_b_size_mult"])
            self._patch(scfg, "ADDON_B_SIZE_MULT", val)
            self._patch(_self_mod, "ADDON_B_SIZE_MULT", val)
        if "fixed_qty_addon_b" in ov:
            val = bool(ov["fixed_qty_addon_b"])
            self._patch(scfg, "FIXED_QTY_ADDON_B_ENABLED", val)
            self._patch(_self_mod, "FIXED_QTY_ADDON_B_ENABLED", val)

        # --- Breakout trigger variants ---
        if "breakout_retrace_entry_frac" in ov:
            val = float(ov["breakout_retrace_entry_frac"])
            self._patch(scfg, "BREAKOUT_RETRACE_ENTRY_FRAC", val)
            self._patch(ssig, "BREAKOUT_RETRACE_ENTRY_FRAC", val)
            self._patch(_self_mod, "BREAKOUT_RETRACE_ENTRY_FRAC", val)
        if "breakout_retrace_limit_frac" in ov:
            val = float(ov["breakout_retrace_limit_frac"])
            self._patch(scfg, "BREAKOUT_RETRACE_LIMIT_FRAC", val)
            self._patch(ssig, "BREAKOUT_RETRACE_LIMIT_FRAC", val)
            self._patch(_self_mod, "BREAKOUT_RETRACE_LIMIT_FRAC", val)
        if "breakout_require_directional_candle" in ov:
            val = bool(ov["breakout_require_directional_candle"])
            self._patch(scfg, "BREAKOUT_REQUIRE_DIRECTIONAL_CANDLE", val)
            self._patch(ssig, "BREAKOUT_REQUIRE_DIRECTIONAL_CANDLE", val)
            self._patch(_self_mod, "BREAKOUT_REQUIRE_DIRECTIONAL_CANDLE", val)
        if "breakout_direct_entry" in ov:
            val = bool(ov["breakout_direct_entry"])
            self._patch(scfg, "BREAKOUT_DIRECT_ENTRY", val)
            self._patch(ssig, "BREAKOUT_DIRECT_ENTRY", val)

        if "rank_mode" in ov:
            import strategies.swing.atrss.allocator as salloc
            val = str(ov["rank_mode"])
            self._patch(scfg, "CANDIDATE_RANK_MODE", val)
            self._patch(salloc, "CANDIDATE_RANK_MODE", val)

        # --- Dict-type constants used by stops module ---
        import strategies.swing.atrss.stops as ssto
        if "profit_floor" in ov:
            val = {float(k): float(v) for k, v in ov["profit_floor"].items()}
            self._patch(scfg, "PROFIT_FLOOR", val)
            self._patch(ssto, "PROFIT_FLOOR", val)
        if "profit_floor_short" in ov:
            val = {float(k): float(v) for k, v in ov["profit_floor_short"].items()}
            self._patch(scfg, "PROFIT_FLOOR_SHORT", val)
            self._patch(ssto, "PROFIT_FLOOR_SHORT", val)

        return self

    def __exit__(self, *exc):
        for obj, attr, orig in reversed(self._patches):
            setattr(obj, attr, orig)
        self._patches.clear()

    def _patch(self, obj, attr: str, value):
        self._patches.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, value)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Single-symbol bar-by-bar backtest engine."""

    def __init__(
        self,
        symbol: str,
        cfg: SymbolConfig,
        bt_config: BacktestConfig,
        point_value: float,
        indicator_cache: dict | None = None,
    ):
        self.symbol = symbol
        self.cfg = cfg
        self.bt_config = bt_config
        self.point_value = point_value
        self.flags = bt_config.flags

        # State
        self.equity = bt_config.initial_equity
        self.sizing_equity = bt_config.initial_equity
        self.position: PositionBook = PositionBook(symbol=symbol)
        self.reentry: ReentryState = ReentryState()
        self.daily_state: DailyState | None = None
        self.hourly_state: HourlyState | None = None
        self.prev_trend_dir: Direction = Direction.FLAT
        # Use ETF commission rate when trading ETFs
        from dataclasses import replace as _replace
        slippage = bt_config.slippage
        if cfg.sec_type == "STK":
            slippage = _replace(slippage,
                                commission_per_contract=slippage.commission_per_share_etf)
        self.broker = SimBroker(slippage_config=slippage)
        self.breakout_arm: BreakoutArmState = BreakoutArmState()

        # Apply param_overrides that affect SymbolConfig (now per-symbol)
        _ov = bt_config.param_overrides
        _cfg_kw: dict = {}
        for _fn in ("adx_on", "adx_off", "daily_ema_fast", "daily_ema_slow",
                     "adx_period", "atr_daily_period", "atr_hourly_period",
                     "ema_mom_period", "ema_pull_strong", "ema_pull_normal",
                     "donchian_period", "chand_mult", "base_risk_pct",
                     "limit_ticks", "limit_pct"):
            if _fn in _ov:
                _cfg_kw[_fn] = type(getattr(cfg, _fn))(_ov[_fn])
        # Per-symbol ATR multiplier overrides
        _dm = f"daily_mult_{symbol}"
        _hm = f"hourly_mult_{symbol}"
        if _dm in _ov:
            _cfg_kw["daily_mult"] = float(_ov[_dm])
        if _hm in _ov:
            _cfg_kw["hourly_mult"] = float(_ov[_hm])
        # Ablation: disable hysteresis gap → adx_off = adx_on
        if not bt_config.flags.hysteresis_gap:
            _cfg_kw["adx_off"] = _cfg_kw.get("adx_on", cfg.adx_on)
        if _cfg_kw:
            self.cfg = _dc_replace(cfg, **_cfg_kw)

        # Overridable module-level constants (resolved from param_overrides)
        import strategies.swing.atrss.config as _scfg
        self._be_trigger_r: float = float(_ov.get("be_trigger_r", _scfg.BE_TRIGGER_R))
        self._chandelier_trigger_r: float = float(_ov.get("chandelier_trigger_r", _scfg.CHANDELIER_TRIGGER_R))

        # MFE/MAE tracking for trade records
        self._mae_price: float = 0.0
        # Prior bar close for gap-and-go detection
        self._prior_close: float = 0.0

        # Pending order metadata (instance-level)
        self._pending_initial_stops: dict[str, float] = {}
        self._pending_entry_types: dict[str, str] = {}
        self._last_entry_type: str = ""

        # Daily state history (for shadow tracker)
        self._daily_state_by_idx: dict[int, DailyState] = {}

        # Deferred candidate mode for synchronized portfolio
        self._defer_submissions: bool = False
        self._deferred_candidates: list[Candidate] = []

        # Results
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[float] = []
        self.timestamps: list = []
        self.total_commission: float = 0.0

        # Diagnostic: daily bias counters
        self._bias_days_long: int = 0
        self._bias_days_short: int = 0
        self._bias_days_flat: int = 0

        # Shadow tracking callback: callable(symbol, direction, filter_names, time, entry, stop)
        self.on_rejection: object | None = None

        # Quality gate tracking
        self._last_quality_score: float = 0.0

        # Entry context for trade records (MFE cohort diagnostic)
        self._pending_entry_context: dict[str, dict] = {}
        self._last_entry_context: dict = {}

        # Order metadata for fill rate diagnostic
        self._order_metadata: list[dict] = []
        self._order_metadata_by_id: dict[str, dict] = {}

        # Signal funnel counters
        self._funnel = SignalFunnelStats()

        # Deferred exit tracking (broker-mediated discretionary exits)
        self._flatten_pending: bool = False
        self._pending_flatten_info: dict | None = None  # {reason, reverse_entry?}
        self._pending_partial_info: dict | None = None   # {reason}

        # Core parity: decision event capture via shared replay driver
        self._core_state = ATRSSCoreState()
        self._decision_events: list = []

        # Pre-computed constants for hot path
        self._hourly_lookback = max(cfg.ema_pull_normal, cfg.atr_hourly_period) + 5
        self._daily_lookback = max(cfg.daily_ema_slow, cfg.atr_daily_period) + 5

        # Optional shared indicator cache for optimization runs.
        # When provided, caches DailyState/HourlyState by (symbol, idx).
        # Safe only when indicator params (ema periods, atr periods, etc.)
        # are identical across candidates.  Caller is responsible for
        # clearing the cache when indicator-affecting params change.
        self._indicator_cache = indicator_cache

    def run(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        daily_idx_map: np.ndarray,
    ) -> SymbolResult:
        """Run the backtest over all hourly bars.

        Args:
            daily: Daily OHLCV numpy arrays
            hourly: Hourly OHLCV numpy arrays
            daily_idx_map: For each hourly bar, index of the last completed daily bar
        """
        warmup_d = self.bt_config.warmup_daily
        warmup_h = self.bt_config.warmup_hourly

        with _AblationPatch(self.flags, self.bt_config.param_overrides):
            self._run_loop(daily, hourly, daily_idx_map, warmup_d, warmup_h)

        return SymbolResult(
            symbol=self.symbol,
            trades=self.trades,
            equity_curve=np.array(self.equity_curve),
            timestamps=np.array(self.timestamps),
            total_commission=self.total_commission,
            bias_days_long=self._bias_days_long,
            bias_days_short=self._bias_days_short,
            bias_days_flat=self._bias_days_flat,
            funnel=self._funnel,
            order_metadata=self._order_metadata,
            decision_stream=normalize_decision_stream(self._decision_events),
            trade_outcomes=trade_outcomes_from_records(self.trades),
        )

    # ------------------------------------------------------------------
    # Core parity: replay driver delegation
    # ------------------------------------------------------------------

    def _replay_core_step(
        self,
        *,
        bar_input: dict | None = None,
        order_updates: list | None = None,
        fills: list | None = None,
    ):
        result = run_replay(
            self._core_state,
            steps=[
                ReplayStep(
                    bar_input=bar_input,
                    order_updates=order_updates or [],
                    fills=fills or [],
                )
            ],
            on_bar=lambda state, payload: atrss_core_logic.on_bar(state, **payload),
            on_order_update=atrss_core_logic.on_order_update,
            on_fill=atrss_core_logic.on_fill,
        )
        self._core_state = result.state
        self._decision_events.extend(result.events)
        return result

    def _notify_core_order_submitted(
        self,
        order_id: str,
        *,
        bar_time: datetime | None,
        order_role: str,
        metadata: dict | None = None,
    ) -> None:
        self._replay_core_step(order_updates=[ATRSSOrderUpdate(
            oms_order_id=order_id,
            status="submitted",
            symbol=self.symbol,
            timestamp=bar_time,
            order_role=order_role,
            metadata=metadata or {},
        )])

    @staticmethod
    def _core_order_role(order_tag: str) -> str:
        if order_tag in ("addon_a", "addon_b"):
            return "add_on"
        if order_tag == "protective_stop":
            return "stop"
        if order_tag in ("entry", "flatten", "partial"):
            return order_tag
        return "unknown"

    # ------------------------------------------------------------------
    # Step-by-step API for synchronized portfolio mode
    # ------------------------------------------------------------------

    _last_daily_idx: int = -1

    def step_bar(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        daily_idx_map: np.ndarray,
        bar_idx: int,
        warmup_d: int,
        warmup_h: int,
    ) -> list[Candidate]:
        """Process one hourly bar. Returns candidates for portfolio allocation."""
        self._deferred_candidates = []
        self._defer_submissions = True
        self._funnel.total_bars += 1

        bar_time = self._to_datetime(hourly.times[bar_idx])
        O, H, L, C = hourly.opens[bar_idx], hourly.highs[bar_idx], hourly.lows[bar_idx], hourly.closes[bar_idx]

        if np.isnan(O) or np.isnan(H) or np.isnan(L):
            self._funnel.bars_nan += 1
            self.equity_curve.append(self._mtm_equity(C))
            self.timestamps.append(hourly.times[bar_idx])
            self._defer_submissions = False
            return []

        d_idx = int(daily_idx_map[bar_idx])
        if d_idx != self._last_daily_idx and d_idx >= warmup_d:
            self._update_daily(daily, d_idx)
            self._last_daily_idx = d_idx

        if self.daily_state is None:
            self._funnel.bars_warmup += 1
            self.equity_curve.append(self._mtm_equity(C))
            self.timestamps.append(hourly.times[bar_idx])
            self._defer_submissions = False
            return []

        if bar_idx >= warmup_h:
            cache = self._indicator_cache
            cache_key = (self.symbol, "h", bar_idx)
            if cache is not None and cache_key in cache:
                self.hourly_state = cache[cache_key]
            else:
                start = max(0, bar_idx - self._hourly_lookback)
                self.hourly_state = compute_hourly_state(
                    hourly.closes[start:bar_idx + 1],
                    hourly.highs[start:bar_idx + 1],
                    hourly.lows[start:bar_idx + 1],
                    self.daily_state, self.cfg,
                    bar_time,
                    hourly.opens[start:bar_idx + 1],
                )
                if cache is not None:
                    cache[cache_key] = self.hourly_state
        else:
            self._funnel.bars_warmup += 1
            self.equity_curve.append(self._mtm_equity(C))
            self.timestamps.append(hourly.times[bar_idx])
            self._defer_submissions = False
            return []

        h = self.hourly_state

        # Process fills
        fills = self.broker.process_bar(self.symbol, bar_time, O, H, L, C, self.cfg.tick_size)
        for fr in fills:
            self._handle_fill(fr, bar_time)

        # Update reset flags
        if h.close < h.ema_pull:
            self.reentry.reset_seen_long = True
        if h.close > h.ema_pull:
            self.reentry.reset_seen_short = True
        if not self.flags.reset_requirement:
            self.reentry.reset_seen_long = True
            self.reentry.reset_seen_short = True
        if not self.flags.cooldown:
            self.reentry.last_exit_time = None

        # Update breakout arm state (spec S7.2-7.3)
        self._update_breakout_arm(h, self.daily_state, bar_time)

        # Manage position (add-on A direct, reverse/addon_b deferred)
        if self.position.direction != Direction.FLAT:
            self._funnel.bars_in_position += 1
            self._manage_position(h, self.daily_state, bar_time)
        else:
            # Generate candidates (deferred) — only if not in position at bar start
            self._generate_candidates(h, self.daily_state, bar_time)

        self.equity_curve.append(self._mtm_equity(C))
        self.timestamps.append(hourly.times[bar_idx])

        self._defer_submissions = False
        if not self._deferred_candidates:
            return []
        result = self._deferred_candidates
        self._deferred_candidates = []
        return result

    def _run_loop(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        daily_idx_map: np.ndarray,
        warmup_d: int,
        warmup_h: int,
    ) -> None:
        """Main bar-by-bar loop."""
        last_daily_idx = -1

        for i in range(len(hourly)):
            self._funnel.total_bars += 1
            bar_time = self._to_datetime(hourly.times[i])
            O = hourly.opens[i]
            H = hourly.highs[i]
            L = hourly.lows[i]
            C = hourly.closes[i]

            # Skip NaN bars (gap-filled)
            if np.isnan(O) or np.isnan(H) or np.isnan(L):
                self._funnel.bars_nan += 1
                self.equity_curve.append(self._mtm_equity(C))
                self.timestamps.append(hourly.times[i])
                continue

            # --- 1. Update daily state if new daily bar ---
            d_idx = int(daily_idx_map[i])
            new_daily_bar = d_idx != last_daily_idx and d_idx >= warmup_d
            if new_daily_bar:
                self._update_daily(daily, d_idx)
                last_daily_idx = d_idx

            if self.daily_state is None:
                self._funnel.bars_warmup += 1
                self.equity_curve.append(self._mtm_equity(C))
                self.timestamps.append(hourly.times[i])
                continue

            # --- 2. Compute hourly state ---
            if i >= warmup_h:
                cache = self._indicator_cache
                cache_key = (self.symbol, "h", i)
                if cache is not None and cache_key in cache:
                    self.hourly_state = cache[cache_key]
                else:
                    start = max(0, i - self._hourly_lookback)
                    h_closes = hourly.closes[start:i + 1]
                    h_highs = hourly.highs[start:i + 1]
                    h_lows = hourly.lows[start:i + 1]
                    h_opens = hourly.opens[start:i + 1]
                    self.hourly_state = compute_hourly_state(
                        h_closes, h_highs, h_lows,
                        self.daily_state, self.cfg,
                        bar_time, h_opens,
                    )
                    if cache is not None:
                        cache[cache_key] = self.hourly_state
            else:
                self._funnel.bars_warmup += 1
                self.equity_curve.append(self._mtm_equity(C))
                self.timestamps.append(hourly.times[i])
                continue

            h = self.hourly_state

            # Diagnostic: log daily regime state on first hourly bar of each day
            if new_daily_bar:
                d = self.daily_state
                logger.debug(
                    "%s %s | ADX=%.1f | Regime=%s | Donchian_Hi=%.2f | "
                    "EMA_fast=%.2f | Dist_ATR=%.2f",
                    self.symbol, str(bar_time.date()),
                    d.adx, d.regime.value, h.donchian_high,
                    d.ema_fast, h.dist_atr,
                )

            # --- 3. Process pending orders ---
            fills = self.broker.process_bar(
                self.symbol, bar_time, O, H, L, C, self.cfg.tick_size,
            )
            for fr in fills:
                self._handle_fill(fr, bar_time)

            # --- 4. Update reset flags ---
            if h.close < h.ema_pull:
                self.reentry.reset_seen_long = True
            if h.close > h.ema_pull:
                self.reentry.reset_seen_short = True

            # Ablation: reset requirement disabled
            if not self.flags.reset_requirement:
                self.reentry.reset_seen_long = True
                self.reentry.reset_seen_short = True

            # Ablation: cooldown disabled
            if not self.flags.cooldown:
                self.reentry.last_exit_time = None

            # --- 4b. Update breakout arm state (spec S7.2-7.3) ---
            self._update_breakout_arm(h, self.daily_state, bar_time)

            # --- 5. Manage open position ---
            if self.position.direction != Direction.FLAT:
                self._funnel.bars_in_position += 1
                self._manage_position(h, self.daily_state, bar_time)
            else:
                # --- 6. Generate candidates (only if not in position at bar start) ---
                self._generate_candidates(h, self.daily_state, bar_time)

            # Record equity
            self.equity_curve.append(self._mtm_equity(C))
            self.timestamps.append(hourly.times[i])

    def _mtm_equity(self, current_price: float) -> float:
        """Return equity with unrealized P&L from open legs."""
        if self.position.direction == Direction.FLAT or np.isnan(current_price):
            return self.equity
        d = 1 if self.position.direction == Direction.LONG else -1
        unrealized = sum(
            (current_price - leg.entry_price) * d * self.point_value * leg.qty
            for leg in self.position.legs
        )
        return self.equity + unrealized

    # ------------------------------------------------------------------
    # Daily state update
    # ------------------------------------------------------------------

    def _update_daily(self, daily: NumpyBars, d_idx: int) -> None:
        """Recompute daily state from array slice ending at d_idx."""
        prev = self.daily_state
        self.prev_trend_dir = prev.trend_dir if prev else Direction.FLAT

        cache = self._indicator_cache
        cache_key = (self.symbol, "d", d_idx)
        if cache is not None and cache_key in cache:
            self.daily_state = cache[cache_key]
        else:
            start = max(0, d_idx - self._daily_lookback)
            d_closes = daily.closes[start:d_idx + 1]
            d_highs = daily.highs[start:d_idx + 1]
            d_lows = daily.lows[start:d_idx + 1]
            daily_bar_date = str(daily.times[d_idx])[:10]
            self.daily_state = compute_daily_state(
                d_closes, d_highs, d_lows, prev, self.cfg, daily_bar_date,
            )
            if cache is not None:
                cache[cache_key] = self.daily_state

        self._daily_state_by_idx[d_idx] = self.daily_state

        # Count confirmed bias days
        td = self.daily_state.trend_dir
        if td == Direction.LONG:
            self._bias_days_long += 1
        elif td == Direction.SHORT:
            self._bias_days_short += 1
        else:
            self._bias_days_flat += 1

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    def _handle_fill(self, fr: FillResult, bar_time: datetime) -> None:
        """Process a filled, rejected, or expired order."""
        # Track funnel stats for entry/addon_b orders
        if fr.order.tag in ("entry", "addon_b"):
            if fr.status == FillStatus.FILLED:
                self._funnel.orders_filled += 1
            elif fr.status == FillStatus.EXPIRED:
                self._funnel.orders_expired += 1
            elif fr.status == FillStatus.REJECTED:
                self._funnel.orders_limit_rejected += 1

        # Update order metadata for fill rate diagnostic
        om = self._order_metadata_by_id.get(fr.order.order_id)
        if om is not None:
            if fr.status == FillStatus.FILLED:
                om["status"] = "FILLED"
                om["fill_price"] = fr.fill_price
                om["fill_time"] = bar_time
            elif fr.status == FillStatus.EXPIRED:
                om["status"] = "EXPIRED"
            elif fr.status == FillStatus.REJECTED:
                om["status"] = "REJECTED"

        if fr.status in (FillStatus.EXPIRED, FillStatus.REJECTED, FillStatus.CANCELLED):
            self._replay_core_step(order_updates=[ATRSSOrderUpdate(
                oms_order_id=fr.order.order_id,
                status=fr.status.name.lower(),
                symbol=self.symbol,
                timestamp=bar_time,
                order_role=self._core_order_role(fr.order.tag),
            )])
            return

        # FILLED
        order = fr.order
        # Commission tracked here for entry/stop/addon fills.
        # flatten/partial commission is tracked by _close_position/_partial_close_base.
        if order.tag not in ("flatten", "partial"):
            self.total_commission += fr.commission
            # Debit entry/addon commission from equity (stop commission
            # is already captured via PnL formula in _handle_stop_fill)
            if order.tag != "protective_stop":
                self.equity -= fr.commission

        if order.tag == "protective_stop":
            self._handle_stop_fill(fr, bar_time)
        elif order.tag in ("entry", "addon_b"):
            self._handle_entry_fill(fr, bar_time)
        elif order.tag == "addon_a":
            self._handle_addon_fill(fr, bar_time, LegType.ADDON_A)
        elif order.tag == "flatten":
            self._handle_flatten_fill(fr, bar_time)
        elif order.tag == "partial":
            self._handle_partial_fill(fr, bar_time)

    @staticmethod
    def _commission_share(total_commission: float, qty: int, total_qty: int) -> float:
        if total_commission <= 0 or qty <= 0 or total_qty <= 0:
            return 0.0
        return total_commission * qty / total_qty

    def _handle_entry_fill(self, fr: FillResult, bar_time: datetime) -> None:
        """Process entry fill: create position, check bad fill, place stop."""
        order = fr.order
        fill_price = fr.fill_price

        is_addon_b = order.tag == "addon_b"
        direction = order.direction

        # Bad-fill slippage guard (spec Section 6): you ARE filled, then panic-flatten
        bad_fill = False
        if self.flags.slippage_abort:
            trigger_price = order.stop_price
            atrh = self.hourly_state.atrh if self.hourly_state else 0
            max_slip_pct = self.cfg.max_entry_slip_pct * trigger_price
            max_slip_atr = MAX_ENTRY_SLIP_ATR * atrh
            max_slip = min(max_slip_pct, max_slip_atr)
            actual_slip = abs(fill_price - trigger_price)
            if max_slip > 0 and actual_slip > max_slip:
                bad_fill = True

        initial_stop = self._pending_initial_stops.pop(order.order_id, 0.0)
        entry_type = self._pending_entry_types.pop(order.order_id, "PULLBACK")
        entry_ctx = self._pending_entry_context.pop(order.order_id, {})
        self._last_entry_context = entry_ctx
        portfolio_size_mult = float(entry_ctx.get("portfolio_size_mult", 1.0) or 1.0)

        if is_addon_b:
            # Add-on B fill
            if self.position.direction != Direction.FLAT:
                leg = PositionLeg(
                    leg_type=LegType.ADDON_B,
                    qty=order.qty,
                    entry_price=fill_price,
                    initial_stop=initial_stop,
                    fill_time=bar_time,
                    entry_commission=fr.commission,
                )
                setattr(leg, "portfolio_size_mult", portfolio_size_mult)
                setattr(leg, "signal_time", entry_ctx.get("signal_time"))
                self.position.legs.append(leg)
                self.position.addon_b_done = True
                # Update protective stop qty
                self._update_protective_stop()
                # --- Core parity: notify addon B fill ---
                self._replay_core_step(fills=[ATRSSFill(
                    oms_order_id=order.order_id,
                    fill_price=fill_price,
                    fill_qty=order.qty,
                    symbol=self.symbol,
                    fill_time=bar_time,
                    commission=fr.commission,
                )])
        else:
            # Base entry
            leg = PositionLeg(
                leg_type=LegType.BASE,
                qty=order.qty,
                entry_price=fill_price,
                initial_stop=initial_stop,
                fill_time=bar_time,
                entry_commission=fr.commission,
            )
            setattr(leg, "portfolio_size_mult", portfolio_size_mult)
            setattr(leg, "signal_time", entry_ctx.get("signal_time"))
            self.position = PositionBook(
                symbol=self.symbol,
                direction=Direction(direction),
                legs=[leg],
                current_stop=initial_stop,
                mfe_price=fill_price,
                entry_time=bar_time,
            )
            self._mae_price = fill_price

            # Consume voucher if applicable
            if not self.flags.voucher_system:
                pass  # Skip voucher logic
            else:
                d = self.daily_state
                td = d.trend_dir if d else Direction.FLAT
                if signals._has_valid_voucher(self.reentry, Direction(direction), bar_time, td):
                    signals.consume_voucher(self.reentry, Direction(direction))

            # Bad fill: position was created (you're filled), now panic-flatten
            if bad_fill:
                logger.debug(
                    "%s BAD FILL: panic flatten at %.4f (filled %.4f)",
                    self.symbol, fill_price, fill_price,
                )
                # --- Core parity: notify entry fill before bad-fill flatten ---
                self._replay_core_step(fills=[ATRSSFill(
                    oms_order_id=order.order_id,
                    fill_price=fill_price,
                    fill_qty=order.qty,
                    symbol=self.symbol,
                    fill_time=bar_time,
                    commission=fr.commission,
                )])
                self._close_position(fill_price, bar_time, "BAD_FILL_FLATTEN")
                return

            # Place protective stop
            stop_side = OrderSide.SELL if direction == Direction.LONG else OrderSide.BUY
            stop_order = SimOrder(
                order_id=self.broker.next_order_id(),
                symbol=self.symbol,
                side=stop_side,
                order_type=OrderType.STOP,
                qty=order.qty,
                stop_price=initial_stop,
                tick_size=self.cfg.tick_size,
                submit_time=bar_time,
                ttl_hours=0,
                tag="protective_stop",
            )
            self.broker.submit_order(stop_order)
            # --- Core parity: notify entry fill + track stop order ---
            self._replay_core_step(fills=[ATRSSFill(
                oms_order_id=order.order_id,
                fill_price=fill_price,
                fill_qty=order.qty,
                symbol=self.symbol,
                fill_time=bar_time,
                commission=fr.commission,
            )])
            self._notify_core_order_submitted(
                stop_order.order_id,
                bar_time=bar_time,
                order_role="stop",
                metadata={"qty": stop_order.qty, "stop_price": stop_order.stop_price},
            )

    def _handle_addon_fill(self, fr: FillResult, bar_time: datetime, leg_type: LegType) -> None:
        """Process Add-on A fill."""
        if self.position.direction == Direction.FLAT:
            return
        order = fr.order
        leg = PositionLeg(
            leg_type=leg_type,
            qty=order.qty,
            entry_price=fr.fill_price,
            initial_stop=self.position.current_stop,
            fill_time=bar_time,
            entry_commission=fr.commission,
        )
        setattr(leg, "signal_time", order.submit_time)
        self.position.legs.append(leg)
        self.position.addon_a_done = True
        # NOTE: commission already tracked in _handle_fill() — do NOT double-count
        self._update_protective_stop()
        # --- Core parity: notify addon A fill ---
        self._replay_core_step(fills=[ATRSSFill(
            oms_order_id=fr.order.order_id,
            fill_price=fr.fill_price,
            fill_qty=order.qty,
            symbol=self.symbol,
            fill_time=bar_time,
            commission=fr.commission,
        )])

    # ------------------------------------------------------------------
    # Broker-mediated discretionary exits (Rec 2)
    # ------------------------------------------------------------------

    def _submit_flatten(
        self, reason: str, bar_time: datetime,
        reverse_entry_info: tuple | None = None,
    ) -> None:
        """Submit a MARKET order to flatten the position (fills next bar)."""
        pos = self.position
        if pos.direction == Direction.FLAT or self._flatten_pending:
            return
        total_qty = sum(leg.qty for leg in pos.legs)
        side = OrderSide.SELL if pos.direction == Direction.LONG else OrderSide.BUY
        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=side,
            order_type=OrderType.MARKET,
            qty=total_qty,
            tick_size=self.cfg.tick_size,
            submit_time=bar_time,
            tag="flatten",
        )
        self.broker.submit_order(order)
        self._flatten_pending = True
        self._pending_flatten_info = {
            "reason": reason,
            "reverse_entry_info": reverse_entry_info,
        }
        # --- Core parity: notify flatten request ---
        self._replay_core_step(bar_input={
            "bar_ts": bar_time,
            "flatten_request": ATRSSFlattenRequest(symbol=self.symbol, reason=reason),
        })
        self._notify_core_order_submitted(
            order.order_id,
            bar_time=bar_time,
            order_role="flatten",
            metadata={"symbol": self.symbol, "reason": reason, "qty": total_qty},
        )

    def _submit_partial_exit(
        self, frac: float, reason: str, bar_time: datetime,
    ) -> None:
        """Submit a MARKET order for a partial exit (fills next bar)."""
        pos = self.position
        base = pos.base_leg
        if base is None or base.qty < 1 or self._flatten_pending:
            return
        # qty=1: fall back to flatten
        if base.qty == 1:
            self._submit_flatten(reason, bar_time)
            return
        partial_qty = max(1, int(base.qty * frac))
        if partial_qty >= base.qty:
            partial_qty = base.qty - 1  # Keep at least 1
        side = OrderSide.SELL if pos.direction == Direction.LONG else OrderSide.BUY
        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=side,
            order_type=OrderType.MARKET,
            qty=partial_qty,
            tick_size=self.cfg.tick_size,
            submit_time=bar_time,
            tag="partial",
        )
        self.broker.submit_order(order)
        self._pending_partial_info = {"reason": reason}
        # --- Core parity: notify partial exit request ---
        self._replay_core_step(bar_input={
            "bar_ts": bar_time,
            "partial_exit_request": ATRSSPartialExitRequest(
                client_order_id=order.order_id,
                symbol=self.symbol,
                qty=partial_qty,
                reason=reason,
            ),
        })
        self._notify_core_order_submitted(
            order.order_id,
            bar_time=bar_time,
            order_role="partial",
            metadata={
                "symbol": self.symbol,
                "direction": pos.direction,
                "type": "PARTIAL_CLOSE",
                "partial_qty": partial_qty,
                "reason": reason,
            },
        )

    def _handle_flatten_fill(self, fr: FillResult, bar_time: datetime) -> None:
        """Process fill from a broker-mediated flatten order."""
        info = self._pending_flatten_info or {}
        reason = info.get("reason", "FLATTEN")
        self._close_position(fr.fill_price, bar_time, reason, exit_commission=fr.commission)
        self._flatten_pending = False
        self._pending_flatten_info = None
        # --- Core parity: notify flatten fill ---
        self._replay_core_step(fills=[ATRSSFill(
            oms_order_id=fr.order.order_id,
            fill_price=fr.fill_price,
            fill_qty=fr.order.qty,
            symbol=self.symbol,
            fill_time=bar_time,
            commission=fr.commission,
            exit_type=reason,
        )])
        # Submit deferred reverse entry (bias flip)
        reverse = info.get("reverse_entry_info")
        if reverse:
            ctype, direction, h_snap, d_snap = reverse
            self._submit_entry(ctype, direction, h_snap, d_snap, bar_time)

    def _handle_partial_fill(self, fr: FillResult, bar_time: datetime) -> None:
        """Process fill from a broker-mediated partial exit order."""
        info = self._pending_partial_info or {}
        reason = info.get("reason", "PARTIAL")
        self._pending_partial_info = None
        self._partial_close_base(
            fr.fill_price, bar_time, frac=0.0, reason=reason,
            exit_commission=fr.commission, override_qty=fr.order.qty,
        )
        # --- Core parity: notify partial fill ---
        self._replay_core_step(fills=[ATRSSFill(
            oms_order_id=fr.order.order_id,
            fill_price=fr.fill_price,
            fill_qty=fr.order.qty,
            symbol=self.symbol,
            fill_time=bar_time,
            commission=fr.commission,
        )])

    def _handle_stop_fill(self, fr: FillResult, bar_time: datetime) -> None:
        """Process protective stop fill — close the position."""
        if self.position.direction == Direction.FLAT:
            return

        pos = self.position
        fill_price = fr.fill_price
        risk_per_unit = pos.base_risk_per_unit
        total_qty = max(sum(leg.qty for leg in pos.legs), 1)

        # Cancel all pending entry orders for this symbol
        self.broker.cancel_all(self.symbol)

        # Record trade(s)
        for leg in pos.legs:
            exit_commission = self._commission_share(fr.commission, leg.qty, total_qty)
            trade_commission = leg.entry_commission + exit_commission
            if pos.direction == Direction.LONG:
                pnl_pts = fill_price - leg.entry_price
            else:
                pnl_pts = leg.entry_price - fill_price

            pnl_dollars = pnl_pts * self.point_value * leg.qty
            r_mult = pnl_pts / risk_per_unit if risk_per_unit > 0 else 0

            # MAE in R
            if pos.direction == Direction.LONG:
                mae_pts = leg.entry_price - self._mae_price
            else:
                mae_pts = self._mae_price - leg.entry_price
            mae_r = mae_pts / risk_per_unit if risk_per_unit > 0 else 0

            addon_a = sum(l.qty for l in pos.legs if l.leg_type == LegType.ADDON_A)
            addon_b = sum(l.qty for l in pos.legs if l.leg_type == LegType.ADDON_B)

            trade = TradeRecord(
                symbol=self.symbol,
                direction=int(pos.direction),
                entry_type=self._last_entry_type,
                entry_time=leg.fill_time,
                exit_time=bar_time,
                entry_price=leg.entry_price,
                exit_price=fill_price,
                qty=leg.qty,
                initial_stop=leg.initial_stop,
                exit_reason="STOP",
                pnl_points=pnl_pts,
                pnl_dollars=pnl_dollars,
                r_multiple=r_mult,
                mfe_r=pos.mfe,
                mae_r=mae_r,
                bars_held=pos.bars_held,
                commission=trade_commission,
                addon_a_qty=addon_a,
                addon_b_qty=addon_b,
                leg_type=leg.leg_type.value,
                adx_entry=self._last_entry_context.get("adx_entry", 0.0),
                score_entry=self._last_entry_context.get("score_entry", 0.0),
                touch_distance_atr=self._last_entry_context.get("touch_distance_atr", 0.0),
                di_agrees=self._last_entry_context.get("di_agrees", False),
                quality_score=self._last_entry_context.get("quality_score", 0.0),
                regime_entry=self._last_entry_context.get("regime_entry", ""),
                signal_time=getattr(leg, "signal_time", None),
                fill_time=leg.fill_time,
            )
            setattr(trade, "portfolio_size_mult", self._last_entry_context.get("portfolio_size_mult", 1.0))
            self.trades.append(trade)
            self.equity += pnl_dollars - exit_commission
            if not self._defer_submissions:
                self.sizing_equity = self.equity

        # Update reentry state
        self.reentry.last_exit_time = bar_time
        self.reentry.last_exit_dir = pos.direction
        self.reentry.reset_seen_long = False
        self.reentry.reset_seen_short = False
        self.reentry.last_exit_mfe = pos.mfe
        self.reentry.last_exit_reason = "STOP"

        # Grant voucher if MFE >= 1R
        if pos.mfe >= 1.0:
            if pos.direction == Direction.LONG:
                self.reentry.voucher_long = True
            else:
                self.reentry.voucher_short = True
            self.reentry.voucher_granted_time = bar_time

        # Clear position
        self.position = PositionBook(symbol=self.symbol)
        self._mae_price = 0.0
        # --- Core parity: notify stop fill ---
        self._replay_core_step(fills=[ATRSSFill(
            oms_order_id=fr.order.order_id,
            fill_price=fr.fill_price,
            fill_qty=fr.order.qty,
            symbol=self.symbol,
            fill_time=bar_time,
            commission=fr.commission,
            exit_type="STOP",
        )])

    def _close_position(
        self, exit_price: float, bar_time: datetime, reason: str,
        exit_commission: float | None = None,
    ) -> None:
        """Flatten the position at a given price (bias flip, time decay, etc)."""
        pos = self.position
        if pos.direction == Direction.FLAT:
            return

        risk_per_unit = pos.base_risk_per_unit

        # Cancel all pending orders
        self.broker.cancel_all(self.symbol)

        # Compute exit commission if not provided (e.g. BAD_FILL_FLATTEN)
        total_qty = sum(leg.qty for leg in pos.legs)
        if exit_commission is None:
            exit_commission = self.broker._compute_commission(total_qty)
        self.total_commission += exit_commission

        for leg in pos.legs:
            leg_exit_commission = self._commission_share(exit_commission, leg.qty, total_qty)
            trade_commission = leg.entry_commission + leg_exit_commission
            if pos.direction == Direction.LONG:
                pnl_pts = exit_price - leg.entry_price
            else:
                pnl_pts = leg.entry_price - exit_price

            pnl_dollars = pnl_pts * self.point_value * leg.qty
            r_mult = pnl_pts / risk_per_unit if risk_per_unit > 0 else 0

            if pos.direction == Direction.LONG:
                mae_pts = leg.entry_price - self._mae_price
            else:
                mae_pts = self._mae_price - leg.entry_price
            mae_r = mae_pts / risk_per_unit if risk_per_unit > 0 else 0

            addon_a = sum(l.qty for l in pos.legs if l.leg_type == LegType.ADDON_A)
            addon_b = sum(l.qty for l in pos.legs if l.leg_type == LegType.ADDON_B)

            trade = TradeRecord(
                symbol=self.symbol,
                direction=int(pos.direction),
                entry_type=self._last_entry_type,
                entry_time=leg.fill_time,
                exit_time=bar_time,
                entry_price=leg.entry_price,
                exit_price=exit_price,
                qty=leg.qty,
                initial_stop=leg.initial_stop,
                exit_reason=reason,
                pnl_points=pnl_pts,
                pnl_dollars=pnl_dollars,
                r_multiple=r_mult,
                mfe_r=pos.mfe,
                mae_r=mae_r,
                bars_held=pos.bars_held,
                commission=trade_commission,
                addon_a_qty=addon_a,
                addon_b_qty=addon_b,
                leg_type=leg.leg_type.value,
                adx_entry=self._last_entry_context.get("adx_entry", 0.0),
                score_entry=self._last_entry_context.get("score_entry", 0.0),
                touch_distance_atr=self._last_entry_context.get("touch_distance_atr", 0.0),
                di_agrees=self._last_entry_context.get("di_agrees", False),
                quality_score=self._last_entry_context.get("quality_score", 0.0),
                regime_entry=self._last_entry_context.get("regime_entry", ""),
                signal_time=getattr(leg, "signal_time", None),
                fill_time=leg.fill_time,
            )
            setattr(trade, "portfolio_size_mult", self._last_entry_context.get("portfolio_size_mult", 1.0))
            self.trades.append(trade)
            self.equity += pnl_dollars - leg_exit_commission
            if not self._defer_submissions:
                self.sizing_equity = self.equity

        # Reentry state (no voucher for flatten)
        self.reentry.last_exit_time = bar_time
        self.reentry.last_exit_dir = pos.direction
        self.reentry.reset_seen_long = False
        self.reentry.reset_seen_short = False
        self.reentry.last_exit_mfe = pos.mfe
        self.reentry.last_exit_reason = reason

        self.position = PositionBook(symbol=self.symbol)
        self._mae_price = 0.0

    def _partial_close_base(
        self, exit_price: float, bar_time: datetime, frac: float,
        reason: str = "EARLY_STALL_PARTIAL",
        exit_commission: float | None = None,
        override_qty: int | None = None,
    ) -> None:
        """Exit a fraction of the base leg to reduce exposure.

        When qty=1, falls back to full close (can't split a single contract).
        override_qty: if set, use this qty instead of computing from frac.
        exit_commission: if set, use this instead of computing from broker.
        """
        pos = self.position
        base = pos.base_leg
        if base is None or base.qty < 1:
            return

        # qty=1: fall back to full close
        if base.qty == 1:
            self._close_position(exit_price, bar_time, reason, exit_commission=exit_commission)
            return

        if override_qty is not None:
            partial_qty = override_qty
        else:
            partial_qty = max(1, int(base.qty * frac))
        if partial_qty >= base.qty:
            partial_qty = base.qty - 1  # Keep at least 1
        base_qty_before = base.qty

        # Compute exit commission if not provided
        if exit_commission is None:
            exit_commission = self.broker._compute_commission(partial_qty)
        self.total_commission += exit_commission
        entry_commission = self._commission_share(base.entry_commission, partial_qty, base_qty_before)

        risk_per_unit = pos.base_risk_per_unit
        if pos.direction == Direction.LONG:
            pnl_pts = exit_price - base.entry_price
            mae_pts = base.entry_price - self._mae_price
        else:
            pnl_pts = base.entry_price - exit_price
            mae_pts = self._mae_price - base.entry_price

        pnl_dollars = pnl_pts * self.point_value * partial_qty
        r_mult = pnl_pts / risk_per_unit if risk_per_unit > 0 else 0
        mae_r = mae_pts / risk_per_unit if risk_per_unit > 0 else 0

        trade = TradeRecord(
            symbol=self.symbol,
            direction=int(pos.direction),
            entry_type=self._last_entry_type,
            entry_time=base.fill_time,
            exit_time=bar_time,
            entry_price=base.entry_price,
            exit_price=exit_price,
            qty=partial_qty,
            initial_stop=base.initial_stop,
            exit_reason=reason,
            pnl_points=pnl_pts,
            pnl_dollars=pnl_dollars,
            r_multiple=r_mult,
            mfe_r=pos.mfe,
            mae_r=mae_r,
            bars_held=pos.bars_held,
            commission=entry_commission + exit_commission,
            addon_a_qty=0,
            addon_b_qty=0,
            leg_type="BASE",
            adx_entry=self._last_entry_context.get("adx_entry", 0.0),
            score_entry=self._last_entry_context.get("score_entry", 0.0),
            touch_distance_atr=self._last_entry_context.get("touch_distance_atr", 0.0),
            di_agrees=self._last_entry_context.get("di_agrees", False),
            quality_score=self._last_entry_context.get("quality_score", 0.0),
            regime_entry=self._last_entry_context.get("regime_entry", ""),
            signal_time=getattr(base, "signal_time", None),
            fill_time=base.fill_time,
        )
        setattr(trade, "portfolio_size_mult", self._last_entry_context.get("portfolio_size_mult", 1.0))
        self.trades.append(trade)
        self.equity += pnl_dollars - exit_commission
        if not self._defer_submissions:
            self.sizing_equity = self.equity

        # Reduce base leg qty
        base.qty -= partial_qty
        base.entry_commission = max(base.entry_commission - entry_commission, 0.0)

        # Update protective stop to reflect reduced position size
        self._update_protective_stop()

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_position(self, h: HourlyState, d: DailyState, bar_time: datetime) -> None:
        """Update MFE/MAE, stops, add-ons, bias flip, time decay."""
        pos = self.position

        # Skip management while waiting for a deferred flatten to fill
        if self._flatten_pending:
            if self._is_rth(bar_time):
                pos.bars_held += 1
            return

        if self._is_rth(bar_time):
            pos.bars_held += 1

        # Update MFE price
        if pos.direction == Direction.LONG:
            if h.high > pos.mfe_price or pos.mfe_price == 0:
                pos.mfe_price = h.high
            if h.low < self._mae_price or self._mae_price == 0:
                self._mae_price = h.low
        else:
            if h.low < pos.mfe_price or pos.mfe_price == 0:
                pos.mfe_price = h.low
            if h.high > self._mae_price or self._mae_price == 0:
                self._mae_price = h.high

        base = pos.base_leg
        if base is None:
            return

        risk_per_unit = pos.base_risk_per_unit
        if risk_per_unit > 0:
            if pos.direction == Direction.LONG:
                pos.mfe = (pos.mfe_price - base.entry_price) / risk_per_unit
            else:
                pos.mfe = (base.entry_price - pos.mfe_price) / risk_per_unit

        # --- Current R (shared by stall exit, time decay, addon A) ---
        if risk_per_unit > 0:
            if pos.direction == Direction.LONG:
                cur_r = (h.close - base.entry_price) / risk_per_unit
            else:
                cur_r = (base.entry_price - h.close) / risk_per_unit
        else:
            cur_r = 0.0

        # --- Catastrophic loss cap (spec §13.1a) ---
        if cur_r < -2.0:
            self._submit_flatten("FLATTEN_CATASTROPHIC_CAP", bar_time)
            return

        # --- Partial profit-taking at fixed R targets (only when qty > 1) ---
        base_qty = base.qty if base else 0
        if base_qty > 1 and not pos.tp1_done and pos.mfe >= TP1_R and cur_r >= TP1_R * 0.8:
            self._submit_partial_exit(TP1_FRAC, "TP1", bar_time)
            pos.tp1_done = True

        if base_qty > 1 and not pos.tp2_done and pos.tp1_done and pos.mfe >= TP2_R and cur_r >= TP2_R * 0.8:
            self._submit_partial_exit(TP2_FRAC, "TP2", bar_time)
            pos.tp2_done = True

        # --- Early stall partial exit (non-developing trade, partial risk reduction) ---
        if (
            self.flags.early_stall_exit
            and not pos.early_partial_done
            and pos.bars_held >= EARLY_STALL_CHECK_HOURS
            and pos.mfe < EARLY_STALL_MFE_THRESHOLD
            and cur_r <= 0.2
        ):
            self._submit_partial_exit(EARLY_STALL_PARTIAL_FRAC, "EARLY_STALL_PARTIAL", bar_time)
            pos.early_partial_done = True

        # --- Stall exit (non-developing trade) ---
        if self.flags.stall_exit and pos.bars_held >= STALL_CHECK_HOURS:
            if pos.mfe < STALL_MFE_THRESHOLD and cur_r <= 0.2:
                self._submit_flatten("FLATTEN_STALL", bar_time)
                return

        # --- Time decay exit ---
        if self.flags.time_decay and pos.bars_held >= MAX_HOLD_HOURS:
            if cur_r < 1.0:
                self._submit_flatten("FLATTEN_TIME_DECAY", bar_time)
                return

        new_stop = pos.current_stop

        # --- BE trigger at configurable R ---
        if not pos.be_triggered and pos.mfe >= self._be_trigger_r:
            be_stop = stops.compute_be_stop(
                pos.direction, base.entry_price, d.atr20, self.cfg.tick_size,
            )
            if pos.direction == Direction.LONG and be_stop > pos.current_stop:
                new_stop = be_stop
                pos.be_triggered = True
            elif pos.direction == Direction.SHORT and be_stop < pos.current_stop:
                new_stop = be_stop
                pos.be_triggered = True

        # --- Add-on A at MFE >= be_trigger_r ---
        if self.flags.addon_a and pos.be_triggered and not pos.addon_a_done and pos.mfe >= self._be_trigger_r:
            if signals.addon_a_eligible(pos, h, d, current_r=cur_r):
                self._submit_addon_a(h, d, bar_time)

        # --- Chandelier trailing at configurable R (regime-adaptive multiplier) ---
        if pos.be_triggered and pos.mfe >= self._chandelier_trigger_r:
            effective_chand_mult = self.cfg.chand_mult
            if d.regime == Regime.STRONG_TREND:
                effective_chand_mult *= 1.15  # wider trail, let winners run
            elif d.regime == Regime.TREND:
                effective_chand_mult *= 0.85  # tighter trail, protect gains
            chandelier = stops.compute_chandelier_stop(
                pos.direction, d, effective_chand_mult, self.cfg.tick_size,
            )
            if pos.direction == Direction.LONG and chandelier > new_stop:
                new_stop = chandelier
            elif pos.direction == Direction.SHORT and chandelier < new_stop:
                new_stop = chandelier

        # --- Add-on B ---
        if self.flags.addon_b and not pos.addon_b_done:
            if signals.addon_b_eligible(pos, h, d):
                self._submit_addon_b_entry(h, d, bar_time)

        # --- Profit floor (spec Section 10.4) ---
        if risk_per_unit > 0:
            floor_stop = stops.apply_profit_floor(
                pos.direction, base.entry_price, risk_per_unit,
                pos.mfe, new_stop, self.cfg.tick_size,
            )
            if pos.direction == Direction.LONG and floor_stop > new_stop:
                new_stop = floor_stop
            elif pos.direction == Direction.SHORT and floor_stop < new_stop:
                new_stop = floor_stop

        # Update protective stop if changed
        if new_stop != pos.current_stop:
            pos.current_stop = new_stop
            # --- Core parity: notify stop update ---
            self._replay_core_step(bar_input={
                "bar_ts": bar_time,
                "stop_update": ATRSSStopUpdateRequest(
                    symbol=self.symbol,
                    stop_price=new_stop,
                    qty=pos.total_qty,
                    reason="trailing_stop_update",
                ),
            })
            self._update_protective_stop()

        # --- Bias flip exit + reverse ---
        if d.trend_dir != Direction.FLAT and d.trend_dir != pos.direction and d.trend_dir != self.prev_trend_dir:
            reverse_entry_info = None
            if signals.reverse_entry_ok(h, d):
                self._funnel.reverse_signals += 1
                reverse_entry_info = (CandidateType.REVERSE, d.trend_dir, h, d)
            self._submit_flatten("FLATTEN_BIAS_FLIP", bar_time, reverse_entry_info=reverse_entry_info)

    # ------------------------------------------------------------------
    # Add-on A submission (market fill at current close)
    # ------------------------------------------------------------------

    def _submit_addon_a(self, h: HourlyState, d: DailyState, bar_time: datetime) -> None:
        """Submit Add-on A as a market order (fills at next bar open)."""
        pos = self.position
        base = pos.base_leg
        if base is None:
            return

        if self.bt_config.fixed_qty is not None:
            desired = 1  # Fixed-qty mode: always add 1 unit
        else:
            desired = math.ceil(base.qty * ADDON_A_SIZE_MULT)
            if desired <= 0:
                # Allow qty=1 with risk cap guard
                addon_risk = abs(h.close - pos.current_stop) * self.point_value
                base_risk = abs(base.entry_price - base.initial_stop) * self.point_value * base.qty
                max_risk = min(0.25 * base_risk, 0.0015 * self.sizing_equity)
                if base.qty >= 1 and addon_risk <= max_risk:
                    desired = 1
                else:
                    return

        side = OrderSide.BUY if pos.direction == Direction.LONG else OrderSide.SELL
        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=side,
            order_type=OrderType.MARKET,
            qty=desired,
            tick_size=self.cfg.tick_size,
            submit_time=bar_time,
            ttl_hours=0,
            tag="addon_a",
        )
        self.broker.submit_order(order)
        pos.addon_a_done = True
        # --- Core parity: notify add-on A request ---
        addon_req = ATRSSAddOnARequest(
            client_order_id=order.order_id,
            symbol=self.symbol,
            direction=pos.direction,
            qty=desired,
            entry_price=0.0,
            stop_price=pos.current_stop,
        )
        self._replay_core_step(bar_input={"bar_ts": bar_time, "add_on_a_request": addon_req})
        self._notify_core_order_submitted(
            order.order_id,
            bar_time=bar_time,
            order_role="add_on",
            metadata={
                "symbol": self.symbol,
                "direction": pos.direction,
                "type": CandidateType.ADDON_A,
                "trigger_price": 0.0,
                "initial_stop": pos.current_stop,
                "qty": desired,
            },
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(self, h: HourlyState, d: DailyState, bar_time: datetime) -> None:
        """Generate entry candidates when no position is open."""
        # Entry time restriction (spec Section 14.1)
        if self._is_entry_restricted(bar_time):
            self._funnel.bars_entry_restricted += 1
            return

        # Per-symbol time/day blocking
        if self.cfg.blocked_hours_et or self.cfg.blocked_weekdays:
            dt_et = bar_time.astimezone(_ET_TZ)
            if dt_et.hour in self.cfg.blocked_hours_et:
                self._funnel.bars_entry_restricted += 1
                return
            if dt_et.weekday() in self.cfg.blocked_weekdays:
                self._funnel.bars_entry_restricted += 1
                return

        # Funnel: classify bar by bias and regime
        if d.trend_dir == Direction.FLAT:
            self._funnel.bars_bias_flat += 1
        else:
            if d.regime == Regime.RANGE:
                self._funnel.bars_regime_range += 1
            elif d.regime == Regime.TREND:
                self._funnel.bars_regime_trend += 1
            elif d.regime == Regime.STRONG_TREND:
                self._funnel.bars_regime_strong += 1

        # Skip SHORT generation if disabled for this symbol
        if d.trend_dir == Direction.SHORT and not self.cfg.shorts_enabled:
            self._funnel.bars_shorts_disabled += 1
            return

        # Per-symbol short gate
        if d.trend_dir == Direction.SHORT and not signals.short_symbol_gate(self.symbol, d, h):
            self._funnel.bars_shorts_disabled += 1
            return

        # Pullback signal with rejection attribution
        pb_dir = signals.pullback_signal(h, d)

        if pb_dir != Direction.FLAT:
            self._funnel.pullback_signals += 1
            self._evaluate_and_submit(CandidateType.PULLBACK, pb_dir, h, d, bar_time)
        elif self.on_rejection and d.trend_dir != Direction.FLAT:
            self._record_pullback_rejection(h, d, bar_time)

        # Breakout pullback trigger while armed (spec Sections 7.2-7.3)
        if self.flags.breakout_entries and self.breakout_arm.breakout_armed_dir != Direction.FLAT:
            bo_dir = signals.breakout_pullback_signal(
                h, d, self.breakout_arm.breakout_armed_dir,
                arm_high=self.breakout_arm.breakout_arm_high,
                arm_low=self.breakout_arm.breakout_arm_low,
            )
            if bo_dir != Direction.FLAT:
                self._funnel.breakout_signals += 1
                self._funnel.breakout_arms_converted += 1
                self._evaluate_and_submit(CandidateType.BREAKOUT, bo_dir, h, d, bar_time)
                # Disarm after trigger
                self.breakout_arm.breakout_armed_dir = Direction.FLAT
                self.breakout_arm.breakout_armed_until = None
            elif self.on_rejection and d.trend_dir != Direction.FLAT:
                self._record_breakout_rejection(h, d, bar_time)

    def _record_pullback_rejection(
        self, h: HourlyState, d: DailyState, bar_time: datetime,
    ) -> None:
        """Identify why pullback_signal returned FLAT."""
        direction = d.trend_dir
        reasons: list[str] = []
        if d.regime == Regime.STRONG_TREND:
            tol = RECOVERY_TOLERANCE_ATR_STRONG
        elif d.regime == Regime.TREND:
            tol = RECOVERY_TOLERANCE_ATR_TREND
        else:
            reasons.append("regime_off")
            tol = RECOVERY_TOLERANCE_ATR
        if not reasons and direction == Direction.LONG:
            if not h.recent_pull_touch_long:
                reasons.append("no_pullback_touch")
            if not (h.close > h.ema_pull - tol * h.atrh):
                reasons.append("no_ema_pull_recovery")
        elif not reasons and direction == Direction.SHORT:
            if not h.recent_pull_touch_short:
                reasons.append("no_pullback_touch")
            if not (h.close < h.ema_pull + tol * h.atrh):
                reasons.append("no_ema_pull_recovery")
            if not signals.short_safety_ok(d):
                reasons.append("short_safety")
        if not reasons:
            reasons.append("bias_not_confirmed")
        self._record_rejection(direction, reasons, h, d, bar_time)

    def _record_breakout_rejection(
        self, h: HourlyState, d: DailyState, bar_time: datetime,
    ) -> None:
        """Identify why breakout pullback trigger did not fire while armed."""
        direction = d.trend_dir
        arm = self.breakout_arm
        reasons: list[str] = []
        breakout_range = arm.breakout_arm_high - arm.breakout_arm_low
        if breakout_range <= 0:
            reasons.append("zero_arm_range")
        elif arm.breakout_armed_dir == Direction.LONG:
            retrace_entry = arm.breakout_arm_high - BREAKOUT_RETRACE_ENTRY_FRAC * breakout_range
            retrace_limit = arm.breakout_arm_high - BREAKOUT_RETRACE_LIMIT_FRAC * breakout_range
            if not (h.low <= retrace_entry):
                reasons.append("no_retrace_touch")
            if not (h.close > retrace_limit):
                reasons.append("no_retrace_recovery")
            if BREAKOUT_REQUIRE_DIRECTIONAL_CANDLE and not (h.close > h.open):
                reasons.append("not_bullish_bar")
        elif arm.breakout_armed_dir == Direction.SHORT:
            retrace_entry = arm.breakout_arm_low + BREAKOUT_RETRACE_ENTRY_FRAC * breakout_range
            retrace_limit = arm.breakout_arm_low + BREAKOUT_RETRACE_LIMIT_FRAC * breakout_range
            if not (h.high >= retrace_entry):
                reasons.append("no_retrace_touch")
            if not (h.close < retrace_limit):
                reasons.append("no_retrace_recovery")
            if BREAKOUT_REQUIRE_DIRECTIONAL_CANDLE and not (h.close < h.open):
                reasons.append("not_bearish_bar")
            if not signals.short_safety_ok(d):
                reasons.append("short_safety")
        if not reasons:
            reasons.append("no_armed_direction")
        self._record_rejection(direction, reasons, h, d, bar_time)

    def _record_rejection(
        self, direction: Direction, reasons: list[str],
        h: HourlyState, d: DailyState, bar_time: datetime,
    ) -> None:
        """Compute trigger/stop and invoke the on_rejection callback."""
        if not self.on_rejection:
            return
        trigger = round_to_tick(
            h.high + self.cfg.tick_size if direction == Direction.LONG else h.low - self.cfg.tick_size,
            self.cfg.tick_size,
        )
        d_mult = self.cfg.daily_mult
        h_mult = self.cfg.hourly_mult
        if d.regime == Regime.TREND:
            d_mult *= TREND_STOP_TIGHTENING
            h_mult *= TREND_STOP_TIGHTENING
        stop = stops.compute_initial_stop(
            direction, trigger, h, d.atr20, h.atrh,
            d_mult, h_mult, self.cfg.tick_size,
        )
        self.on_rejection(self.symbol, direction, reasons, bar_time, trigger, stop)

    def _evaluate_and_submit(
        self, ctype: CandidateType, direction: Direction,
        h: HourlyState, d: DailyState, bar_time: datetime,
    ) -> None:
        """Evaluate post-signal filters, submit if none fail, else record all failures."""
        failed: list[str] = []

        # Momentum filter: pullback entries exempt (touch+recovery already confirms structure)
        if self.flags.momentum_filter and ctype != CandidateType.PULLBACK and not signals.momentum_ok(h, direction):
            self._funnel.rejected_momentum += 1
            failed.append("momentum_filter")

        if not signals.same_direction_reentry_allowed(
            self.reentry, direction, bar_time, d.regime, d.trend_dir,
        ):
            self._funnel.rejected_reentry += 1
            failed.append("reentry_gate")

        # Quality gate
        quality_score = 0.0
        if self.flags.quality_gate:
            quality_score = signals.compute_entry_quality(h, d, direction)
            if quality_score < QUALITY_GATE_THRESHOLD:
                self._funnel.rejected_quality += 1
                failed.append("quality_gate")
        self._last_quality_score = quality_score

        if not failed:
            self._submit_entry(ctype, direction, h, d, bar_time)
        else:
            self._record_rejection(direction, failed, h, d, bar_time)

    # ------------------------------------------------------------------
    # Entry order submission
    # ------------------------------------------------------------------

    def _make_candidate(
        self,
        ctype: CandidateType,
        direction: Direction,
        h: HourlyState,
        d: DailyState,
        bar_time: datetime,
    ) -> Candidate | None:
        """Build a Candidate without submitting."""
        if direction == Direction.LONG:
            trigger = round_to_tick(h.high + self.cfg.tick_size, self.cfg.tick_size)
        else:
            trigger = round_to_tick(h.low - self.cfg.tick_size, self.cfg.tick_size)

        # Tighter stops in TREND regime (not STRONG_TREND)
        d_mult = self.cfg.daily_mult
        h_mult = self.cfg.hourly_mult
        if d.regime == Regime.TREND:
            d_mult *= TREND_STOP_TIGHTENING
            h_mult *= TREND_STOP_TIGHTENING

        initial_stop = stops.compute_initial_stop(
            direction, trigger, h, d.atr20, h.atrh,
            d_mult, h_mult, self.cfg.tick_size,
        )
        if self.bt_config.fixed_qty is not None:
            qty = self.bt_config.fixed_qty
            if FIXED_QTY_REGIME_SCALING_ENABLED:
                if d.regime == Regime.STRONG_TREND and d.score >= 60:
                    qty = max(1, int(round(qty * FIXED_QTY_STRONG_TREND_MULT)))
                elif d.regime == Regime.TREND and d.score < 45:
                    qty = max(1, int(round(qty * FIXED_QTY_WEAK_TREND_MULT)))
        else:
            # Regime-adaptive risk sizing
            risk_pct = self.cfg.base_risk_pct
            if d.regime == Regime.STRONG_TREND and d.score >= 60:
                risk_pct *= DYNAMIC_RISK_STRONG_TREND_MULT
            elif d.regime == Regime.TREND and d.score < 45:
                risk_pct *= DYNAMIC_RISK_WEAK_TREND_MULT
            qty = allocator.compute_position_size(
                trigger, initial_stop, self.sizing_equity, risk_pct, self.point_value,
            )
        # Per-symbol size reduction by month
        if self.cfg.size_reduction_months and bar_time is not None:
            month = bar_time.month
            for m, frac in self.cfg.size_reduction_months:
                if month == m:
                    qty = max(1, int(qty * frac))
                    break
        if qty <= 0:
            return None

        return Candidate(
            symbol=self.symbol, type=ctype, direction=direction,
            trigger_price=trigger, initial_stop=initial_stop, qty=qty,
            signal_bar=h, time=bar_time, rank_score=d.score,
            atrh=h.atrh, tick_size=self.cfg.tick_size,
        )

    def submit_candidate(self, cand: Candidate, bar_time: datetime) -> None:
        """Submit an accepted Candidate to the broker."""
        self._funnel.orders_submitted += 1
        tick = self.cfg.tick_size
        side = OrderSide.BUY if cand.direction == Direction.LONG else OrderSide.SELL
        tag = "addon_b" if cand.type == CandidateType.ADDON_B else "entry"
        order_id = self.broker.next_order_id()

        if self.bt_config.slippage.use_stop_market:
            # J2 variant: stop-market (optimistic, no limit rejection)
            order = SimOrder(
                order_id=order_id, symbol=self.symbol, side=side,
                order_type=OrderType.STOP, qty=cand.qty,
                stop_price=cand.trigger_price,
                tick_size=tick, submit_time=bar_time,
                ttl_hours=ORDER_EXPIRY_HOURS, tag=tag,
            )
        else:
            limit_offset = max(
                self.cfg.limit_ticks * tick,
                self.cfg.limit_pct * cand.trigger_price,
            )
            if cand.direction == Direction.LONG:
                limit_price = cand.trigger_price + limit_offset
            else:
                limit_price = cand.trigger_price - limit_offset

            order = SimOrder(
                order_id=order_id, symbol=self.symbol, side=side,
                order_type=OrderType.STOP_LIMIT, qty=cand.qty,
                stop_price=cand.trigger_price, limit_price=limit_price,
                tick_size=tick, submit_time=bar_time,
                ttl_hours=ORDER_EXPIRY_HOURS, tag=tag,
            )
        self.broker.submit_order(order)
        self._pending_initial_stops[order_id] = cand.initial_stop
        self._pending_entry_types[order_id] = cand.type.value
        self._last_entry_type = cand.type.value

        # Capture entry context for MFE cohort diagnostic
        d = self.daily_state
        h_ctx = self.hourly_state
        td_atr = abs(h_ctx.close - h_ctx.ema_pull) / h_ctx.atrh if h_ctx and h_ctx.atrh > 0 else 0.0
        if cand.direction == Direction.LONG:
            di_ok = d.plus_di > d.minus_di if d else False
        else:
            di_ok = d.minus_di > d.plus_di if d else False
        self._pending_entry_context[order_id] = {
            "adx_entry": d.adx if d else 0.0,
            "score_entry": d.score if d else 0.0,
            "touch_distance_atr": td_atr,
            "di_agrees": di_ok,
            "quality_score": self._last_quality_score,
            "regime_entry": d.regime.value if d else "",
            "portfolio_size_mult": float(getattr(cand, "portfolio_size_mult", 1.0) or 1.0),
            "signal_time": bar_time,
        }

        # Order metadata for fill rate diagnostic
        market_price = h_ctx.close if h_ctx else 0.0
        atr_at_submit = h_ctx.atrh if h_ctx else 0.0
        trigger_dist = abs(cand.trigger_price - market_price)
        trigger_dist_atr = trigger_dist / atr_at_submit if atr_at_submit > 0 else 0.0
        om = {
            "order_id": order_id,
            "qty": cand.qty,
            "trigger_price": cand.trigger_price,
            "limit_price": getattr(order, "limit_price", None),
            "market_price_at_submit": market_price,
            "atr_at_submit": atr_at_submit,
            "submit_time": bar_time,
            "trigger_dist_atr": trigger_dist_atr,
            "direction": int(cand.direction),
            "entry_type": cand.type.value,
            "status": "PENDING",
            "fill_price": None,
            "fill_time": None,
        }
        self._order_metadata.append(om)
        self._order_metadata_by_id[order_id] = om

        # --- Core parity: notify entry request ---
        _order_type_str = "STOP" if self.bt_config.slippage.use_stop_market else "STOP_LIMIT"
        entry_req = ATRSSEntryRequest(
            client_order_id=order_id,
            symbol=self.symbol,
            candidate=cand,
            limit_price=getattr(order, "limit_price", 0.0),
            order_type=_order_type_str,
        )
        self._replay_core_step(bar_input={"bar_ts": bar_time, "entry_request": entry_req})
        self._notify_core_order_submitted(
            order_id,
            bar_time=bar_time,
            order_role="add_on" if cand.type == CandidateType.ADDON_B else "entry",
            metadata={
                "symbol": self.symbol,
                "direction": cand.direction,
                "type": cand.type,
                "trigger_price": cand.trigger_price,
                "initial_stop": cand.initial_stop,
                "qty": cand.qty,
            },
        )

    def _submit_entry(
        self,
        ctype: CandidateType,
        direction: Direction,
        h: HourlyState,
        d: DailyState,
        bar_time: datetime,
    ) -> None:
        """Build and submit an entry order (or defer in sync mode)."""
        cand = self._make_candidate(ctype, direction, h, d, bar_time)
        if cand is None:
            self._funnel.rejected_sizing += 1
            return

        if self._defer_submissions:
            self._deferred_candidates.append(cand)
            return

        self.submit_candidate(cand, bar_time)

    def _submit_addon_b_entry(
        self,
        h: HourlyState,
        d: DailyState,
        bar_time: datetime,
    ) -> None:
        """Submit Add-on B as a stop-limit order (or defer in sync mode)."""
        if self.bt_config.fixed_qty is not None and not FIXED_QTY_ADDON_B_ENABLED:
            return
        pos = self.position
        direction = pos.direction

        if direction == Direction.LONG:
            trigger = round_to_tick(h.high + self.cfg.tick_size, self.cfg.tick_size)
        else:
            trigger = round_to_tick(h.low - self.cfg.tick_size, self.cfg.tick_size)

        # Tighter stops in TREND regime (not STRONG_TREND)
        d_mult = self.cfg.daily_mult
        h_mult = self.cfg.hourly_mult
        if d.regime == Regime.TREND:
            d_mult *= TREND_STOP_TIGHTENING
            h_mult *= TREND_STOP_TIGHTENING

        initial_stop = stops.compute_initial_stop(
            direction, trigger, h, d.atr20, h.atrh,
            d_mult, h_mult, self.cfg.tick_size,
        )
        base = pos.base_leg
        if self.bt_config.fixed_qty is not None:
            if not base:
                return
            qty = max(1, int(base.qty * ADDON_B_SIZE_MULT))
        else:
            qty = allocator.compute_position_size(
                trigger, initial_stop, self.sizing_equity, self.cfg.base_risk_pct, self.point_value,
            )
        if base:
            qty = min(qty, max(1, int(base.qty * ADDON_B_SIZE_MULT)))
        if qty <= 0:
            return

        cand = Candidate(
            symbol=self.symbol, type=CandidateType.ADDON_B, direction=direction,
            trigger_price=trigger, initial_stop=initial_stop, qty=qty,
            signal_bar=h, time=bar_time, rank_score=d.score if d else 0,
            atrh=h.atrh, tick_size=self.cfg.tick_size,
        )

        if self._defer_submissions:
            self._deferred_candidates.append(cand)
            return

        self.submit_candidate(cand, bar_time)

    # ------------------------------------------------------------------
    # Protective stop management
    # ------------------------------------------------------------------

    def _update_protective_stop(self) -> None:
        """Update the protective stop order to match current_stop and total_qty."""
        pos = self.position
        if pos.direction == Direction.FLAT:
            return

        # Cancel existing protective stops
        self.broker.cancel_orders(self.symbol, tag="protective_stop")

        # Place new one
        stop_side = OrderSide.SELL if pos.direction == Direction.LONG else OrderSide.BUY
        stop_order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=stop_side,
            order_type=OrderType.STOP,
            qty=pos.total_qty,
            stop_price=pos.current_stop,
            tick_size=self.cfg.tick_size,
            submit_time=pos.entry_time,
            ttl_hours=0,
            tag="protective_stop",
        )
        self.broker.submit_order(stop_order)
        # --- Core parity: track replacement stop through the shared order update path ---
        self._notify_core_order_submitted(
            stop_order.order_id,
            bar_time=stop_order.submit_time,
            order_role="stop",
            metadata={"qty": stop_order.qty, "stop_price": stop_order.stop_price},
        )

    # ------------------------------------------------------------------
    # Breakout arm state management (spec S7.2-7.3)
    # ------------------------------------------------------------------

    def _update_breakout_arm(self, h: HourlyState, d: DailyState, bar_time: datetime) -> None:
        """Check for arm events and expiry. Runs every bar regardless of position."""
        if not self.flags.breakout_entries:
            return

        arm = self.breakout_arm

        # Expire stale arm
        if (arm.breakout_armed_dir != Direction.FLAT
                and arm.breakout_armed_until
                and bar_time > arm.breakout_armed_until):
            self._funnel.breakout_arms_expired += 1
            arm.breakout_armed_dir = Direction.FLAT
            arm.breakout_armed_until = None

        # Check for new arm event (refreshes window if already armed)
        arm_dir = signals.check_breakout_arm(h, d)
        if arm_dir != Direction.FLAT:
            self._funnel.breakout_arms_created += 1
            arm.breakout_armed_dir = arm_dir
            arm.breakout_armed_until = bar_time + timedelta(hours=ARM_WINDOW_HOURS)
            arm.breakout_arm_high = h.high
            arm.breakout_arm_low = h.low

    # ------------------------------------------------------------------
    # RTH / entry restriction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rth(dt: datetime) -> bool:
        """Check if datetime falls within NYSE RTH (09:30-16:00 ET)."""
        dt_et = dt.astimezone(_ET_TZ)
        if dt_et.weekday() >= 5:  # weekend
            return False
        t = dt_et.hour * 60 + dt_et.minute
        return 570 <= t < 960  # 09:30=570, 16:00=960

    @staticmethod
    def _is_entry_restricted(dt: datetime) -> bool:
        """True during first 5 min after open or last 5 min before close."""
        dt_et = dt.astimezone(_ET_TZ)
        if dt_et.weekday() >= 5:
            return True
        t = dt_et.hour * 60 + dt_et.minute
        if t < 575:   # before 09:35
            return True
        if t >= 955:   # at or after 15:55
            return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_datetime(ts) -> datetime:
        """Convert numpy datetime64 to Python datetime."""
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float, np.integer, np.floating)):
            # numpy.datetime64.item() returns an integer nanosecond offset for
            # ns-resolution arrays. Treat large numeric timestamps explicitly
            # instead of silently falling back to wall-clock time.
            value = float(ts)
            if abs(value) > 1e14:
                value /= 1e9
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if hasattr(ts, 'astype'):
            # numpy datetime64
            unix_epoch = np.datetime64(0, 'ns')
            one_second = np.timedelta64(1, 's')
            seconds = (ts - unix_epoch) / one_second
            return datetime.fromtimestamp(float(seconds), tz=timezone.utc)
        return datetime.now(timezone.utc)
