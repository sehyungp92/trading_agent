"""Downturn Dominator backtest engine — 3 sub-engines, one class.

Primary loop on 5m bars with multi-TF boundary callbacks.
Uses SimBroker for order fill simulation.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np

from backtests.momentum.config_downturn import DownturnBacktestConfig
from backtests.momentum.data.preprocessing import NumpyBars
from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.execution_adapters import ParitySimOrder, neutral_action_to_sim_order
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from backtests.shared.parity.trade_outcomes import normalize_trade_outcome_stream
from strategies.core.actions import SubmitEntry
from strategies.momentum.downturn.indicators import (
    IncrementalATR,
    IncrementalEMA,
    compute_adx,
    compute_adx_suite,
    compute_atr,
    compute_box_adaptive_length,
    compute_chop_score,
    compute_displacement_metric,
    compute_ema,
    compute_ema_array,
    compute_extension,
    compute_macd_hist,
    compute_momentum_slope_ok,
    compute_session_vwap,
    compute_sma,
    compute_trend_strength,
    compute_vwap_anchored,
    highest,
    lowest,
    percentile_rank,
)
from strategies.momentum.downturn.bt_models import (
    BreakdownBoxState,
    BreakdownSignal,
    CompositeRegime,
    CorrectionWindow,
    DownturnRegimeCtx,
    DownturnResult,
    DownturnSignalEvent,
    DownturnTradeRecord,
    EngineCounters,
    EngineTag,
    FadeSignal,
    FadeState,
    Regime4H,
    ReversalSignal,
    ReversalState,
    VolState,
)
from strategies.momentum.downturn.regime import (
    check_bear_structure_override,
    check_drawdown_override,
    check_fast_crash_override,
    classify_4h_regime,
    classify_daily_trend,
    compute_bear_conviction,
    compute_composite_regime,
    compute_regime_on,
    compute_strong_bear,
    compute_vol_factor,
    compute_vol_state,
    regime_sizing_mult,
)
from strategies.momentum.downturn.signals import (
    compute_entry_subtype_stop,
    detect_breakdown_short,
    detect_fade_short,
    detect_momentum_impulse,
    detect_reversal_short,
    update_box_state,
)
from strategies.momentum.downturn.stops import (
    check_catastrophic_exit,
    check_climax_exit,
    check_stale_exit,
    check_vwap_failure_exit,
    compute_adaptive_lock_pct,
    compute_breakeven_stop,
    compute_chandelier_regime_mult,
    compute_multi_tier_profit_floor,
    compute_profit_floor_stop,
    compute_tiered_tp_schedule,
    update_chandelier_trail,
)
from strategies.momentum.downturn.core import logic as downturn_core_logic
from strategies.momentum.downturn.core.state import (
    DownturnCoreState,
    DownturnEntryRequest,
    DownturnFill,
    DownturnOrderUpdate,
    DownturnStopUpdateRequest,
)
from backtests.momentum.engine.sim_broker import (
    FillResult,
    FillStatus,
    OrderSide,
    OrderType,
    SimBroker,
    SimOrder,
)
from backtests.momentum.models import Direction

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def _sim_order_from_parity(order: ParitySimOrder) -> SimOrder:
    return SimOrder(
        order_id=order.order_id,
        symbol=order.symbol,
        side=OrderSide[order.side.name],
        order_type=OrderType[order.order_type.name],
        qty=order.qty,
        stop_price=order.stop_price,
        limit_price=order.limit_price,
        tick_size=order.tick_size,
        submit_time=order.submit_time,
        ttl_hours=order.ttl_hours,
        ttl_minutes=order.ttl_minutes,
        tag=order.tag,
        oca_group=order.oca_group,
        invalidation_price=order.invalidation_price,
        triggered_ts=order.triggered_ts,
    )


# ---------------------------------------------------------------------------
# Active position tracking
# ---------------------------------------------------------------------------

class _ActivePosition:
    """Tracks a single open position with exit management."""

    __slots__ = (
        "engine_tag", "signal_class", "direction", "entry_price", "stop0",
        "qty", "remaining_qty", "entry_time", "entry_bar_idx",
        "composite_regime", "vol_state", "in_correction", "predator",
        "tp_schedule", "tp_idx", "chandelier_stop", "be_triggered",
        "hold_bars_5m", "hold_bars_1h", "hold_bars_30m", "hold_bars_4h",
        "mfe_price", "mae_price", "commission",
        "consecutive_above_vwap", "r_at_peak", "scaled_out",
        "exit_trigger",
    )

    def __init__(
        self,
        engine_tag: EngineTag,
        signal_class: str,
        entry_price: float,
        stop0: float,
        qty: int,
        entry_time: datetime,
        entry_bar_idx: int,
        composite_regime: CompositeRegime,
        vol_state: VolState,
        in_correction: bool,
        predator: bool,
        tp_schedule: list[tuple[float, float]],
    ):
        self.engine_tag = engine_tag
        self.signal_class = signal_class
        self.direction = Direction.SHORT
        self.entry_price = entry_price
        self.stop0 = stop0
        self.qty = qty
        self.remaining_qty = qty
        self.entry_time = entry_time
        self.entry_bar_idx = entry_bar_idx
        self.composite_regime = composite_regime
        self.vol_state = vol_state
        self.in_correction = in_correction
        self.predator = predator
        self.tp_schedule = tp_schedule
        self.tp_idx = 0
        self.chandelier_stop = stop0
        self.be_triggered = False
        self.hold_bars_5m = 0
        self.hold_bars_1h = 0
        self.hold_bars_30m = 0
        self.hold_bars_4h = 0
        self.mfe_price = entry_price
        self.mae_price = entry_price
        self.commission = 0.0
        self.consecutive_above_vwap = 0
        self.r_at_peak = 0.0
        self.scaled_out = False
        self.exit_trigger: str = ""

    @property
    def risk_per_unit(self) -> float:
        return abs(self.stop0 - self.entry_price) or 1.0

    def r_state(self, current_price: float) -> float:
        """Current R-multiple (positive = profitable for short)."""
        return (self.entry_price - current_price) / self.risk_per_unit


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DownturnEngine:
    """Downturn Dominator backtest engine."""

    def __init__(self, symbol: str, config: DownturnBacktestConfig):
        self.symbol = symbol
        self.config = config
        self.flags = config.flags
        self.po = config.param_overrides
        self.broker = SimBroker(slippage_config=config.slippage)
        self._core_state = DownturnCoreState(symbol=symbol)
        self._decision_events: list[Any] = []

        # State
        self._position: Optional[_ActivePosition] = None
        self._regime = DownturnRegimeCtx()
        self._reversal = ReversalState()
        self._box = BreakdownBoxState()
        self._fade = FadeState()

        # Indicator caches
        self._atr_d: float = 0.0
        self._atr_d_baseline: float = 0.0
        self._atr_d_history: list[float] = []
        self._atr_d_pctl: float = 0.5
        self._atr_30m: float = 0.0
        self._atr_15m: float = 0.0
        self._atr_1h: float = 0.0
        self._atr_4h: float = 0.0
        self._ema_fast_d: float = 0.0
        self._ema_slow_d: float = 0.0
        self._sma200_d: float = 0.0
        self._ema20_1h: float = 0.0
        self._trend_strength_3d: list[float] = []
        self._vwap_cross_count: int = 0
        self._mom15: list[float] = []
        self._mom15_slope_ok: bool = False
        self._atr_4h_fast: float = 0.0
        self._atr_4h_slow: float = 0.0
        self._short_sma_d: float = 0.0
        self._session_start_15m: int = 0
        self._correction_windows: list[CorrectionWindow] = []

        # Fast-crash override state (computed on daily boundary)
        self._fast_crash_active: bool = False
        # Bear conviction score (computed on daily boundary)
        self._bear_conviction: float = 0.0
        self._prev_ema_fast_d: float = 0.0
        # Daily ADX suite for conviction scoring
        self._daily_adx: float = 0.0
        self._daily_plus_di: float = 0.0
        self._daily_minus_di: float = 0.0
        # Bear structure override state (ADX hysteresis + paths B/C)
        self._regime_on: bool = False
        self._bear_structure_active: bool = False
        # R6: Drawdown override state (computed on daily boundary)
        self._drawdown_override_active: bool = False
        # R6: Track momentum impulse signal for sig_class tagging
        self._momentum_impulse_pending: bool = False
        # R6 Rev2: Progressive SMA warmup flag (True when data < sma200_period)
        self._progressive_sma_warmup: bool = False
        # R6 Rev2: Bars since last entry fill (for momentum cooldown gate)
        self._bars_since_last_entry: int = 999
        # Early abort when portfolio DD exceeds threshold (optimizer perf)
        self._max_dd_abort: float = config.max_dd_abort
        self._peak_equity: float = config.initial_equity
        self._abort: bool = False

        # Incremental indicators (O(1) per boundary update)
        self._inc_atr_15m = IncrementalATR(14)
        self._inc_ema_15m_fast = IncrementalEMA(5)
        self._inc_ema_15m_slow = IncrementalEMA(13)
        self._inc_atr_30m = IncrementalATR(14)
        self._inc_atr_30m_fast = IncrementalATR(5)
        self._inc_atr_1h = IncrementalATR(14)
        self._inc_ema_1h_20 = IncrementalEMA(20)

        # 4H pivot tracking for reversal
        self._4h_highs: list[float] = []
        self._4h_lows: list[float] = []
        self._4h_macd_hist: list[float] = []

        # Daily risk
        self._daily_loss: float = 0.0
        self._daily_trades: int = 0
        self._circuit_breaker_tripped: bool = False

        # Counters
        self._reversal_ctr = EngineCounters()
        self._breakdown_ctr = EngineCounters()
        self._fade_ctr = EngineCounters()

        # Results
        self._trades: list[DownturnTradeRecord] = []
        self._signals: list[DownturnSignalEvent] = []
        self._total_commission: float = 0.0

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    def run(
        self,
        five_min: NumpyBars,
        fifteen_min: NumpyBars,
        thirty_min: NumpyBars,
        hourly: NumpyBars,
        four_hour: NumpyBars,
        daily: NumpyBars,
        fifteen_min_idx_map: np.ndarray,
        thirty_min_idx_map: np.ndarray,
        hourly_idx_map: np.ndarray,
        four_hour_idx_map: np.ndarray,
        daily_idx_map: np.ndarray,
        daily_es: NumpyBars | None = None,
        daily_es_idx_map: np.ndarray | None = None,
    ) -> DownturnResult:
        """Run backtest over 5m bars."""
        cfg = self.config
        n = len(five_min.closes)
        warmup = cfg.warmup_days * 78  # ~78 5m bars per day

        equity = cfg.initial_equity
        equity_curve = np.full(n, equity)
        self._realized_pnl = 0.0  # running total for O(1) equity updates
        self._current_equity = equity  # tracked for dynamic leverage cap
        correction_windows = self._compute_correction_windows(daily)
        self._correction_windows = correction_windows

        # TF boundary trackers
        last_d_idx = -1
        last_fh_idx = -1
        last_h_idx = -1
        last_m30_idx = -1
        last_m15_idx = -1

        for t in range(warmup, n):
            bar_time_raw = five_min.times[t]
            if isinstance(bar_time_raw, np.datetime64):
                bar_time = datetime.utcfromtimestamp(
                    bar_time_raw.astype("datetime64[s]").astype("int64")
                ).replace(tzinfo=timezone.utc)
            else:
                bar_time = bar_time_raw

            O = five_min.opens[t]
            H = five_min.highs[t]
            L = five_min.lows[t]
            Cl = five_min.closes[t]

            # Detect TF boundaries
            d_idx = int(daily_idx_map[t])
            fh_idx = int(four_hour_idx_map[t])
            h_idx = int(hourly_idx_map[t])
            m30_idx = int(thirty_min_idx_map[t])
            m15_idx = int(fifteen_min_idx_map[t])

            new_d = d_idx != last_d_idx
            new_4h = fh_idx != last_fh_idx
            new_1h = h_idx != last_h_idx
            new_30m = m30_idx != last_m30_idx
            new_15m = m15_idx != last_m15_idx

            # Process fills
            fills = self.broker.process_bar(
                self.symbol, bar_time, O, H, L, Cl, cfg.tick_size,
            )
            for fill in fills:
                self._handle_fill(fill, bar_time, Cl, equity, correction_windows)

            # Boundary callbacks
            if new_d:
                self._on_daily_boundary(
                    daily, d_idx,
                    daily_es, int(daily_es_idx_map[t]) if daily_es_idx_map is not None else -1,
                )
                last_d_idx = d_idx
            if new_4h:
                self._on_4h_boundary(four_hour, fh_idx)
                last_fh_idx = fh_idx
            if new_1h:
                self._on_1h_boundary(hourly, h_idx, Cl)
                last_h_idx = h_idx
            if new_30m:
                self._on_30m_boundary(thirty_min, m30_idx)
                last_m30_idx = m30_idx
            if new_15m:
                self._on_15m_boundary(fifteen_min, m15_idx)
                last_m15_idx = m15_idx

            # Position management
            if self._position is not None:
                self._position.hold_bars_5m += 1
                self._manage_position(t, bar_time, H, L, Cl, hourly, h_idx)

            # R6 Rev2: Increment bars-since-last-entry counter
            self._bars_since_last_entry += 1

            # Signal detection + entry
            if self._position is None and self._can_enter(bar_time):
                signal = self._evaluate_signals(
                    t, bar_time, Cl, H, L,
                    fifteen_min, m15_idx,
                    thirty_min, m30_idx,
                    four_hour, fh_idx,
                    daily, d_idx,
                )
                if signal is not None:
                    self._submit_entry(signal, t, bar_time, Cl, equity, correction_windows)

            # Update equity (O(1) via running total)
            mark_pnl = 0.0
            if self._position is not None:
                pos = self._position
                mark_pnl = (pos.entry_price - Cl) * pos.remaining_qty * cfg.point_value
            self._current_equity = cfg.initial_equity + self._realized_pnl + mark_pnl
            equity_curve[t] = self._current_equity

            # Early abort: stop backtest if portfolio DD exceeds threshold
            if self._max_dd_abort > 0:
                if self._current_equity > self._peak_equity:
                    self._peak_equity = self._current_equity
                elif self._peak_equity > 0:
                    dd = (self._peak_equity - self._current_equity) / self._peak_equity
                    if dd > self._max_dd_abort:
                        self._abort = True
                        break

        # Close any remaining position at last close
        if self._position is not None:
            self._force_close(five_min.closes[-1], n - 1, correction_windows)

        return DownturnResult(
            symbol=self.symbol,
            trades=self._trades,
            signal_events=self._signals,
            decision_stream=[] if cfg.skip_parity_output else normalize_decision_stream(self._decision_events),
            trade_outcomes=[] if cfg.skip_parity_output else normalize_trade_outcome_stream(self._trades),
            equity_curve=equity_curve,
            timestamps=five_min.times,
            total_commission=self._total_commission,
            correction_windows=correction_windows,
            reversal_counters=self._reversal_ctr,
            breakdown_counters=self._breakdown_ctr,
            fade_counters=self._fade_ctr,
        )

    def _replay_core_step(
        self,
        *,
        bar_input: dict[str, Any] | None = None,
        order_updates: list[DownturnOrderUpdate] | None = None,
        fills: list[DownturnFill] | None = None,
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
            on_bar=lambda state, payload: downturn_core_logic.on_bar(state, **payload),
            on_order_update=downturn_core_logic.on_order_update,
            on_fill=downturn_core_logic.on_fill,
        )
        self._core_state = result.state
        if not self.config.skip_parity_output:
            self._decision_events.extend(result.events)
        return result

    # -------------------------------------------------------------------
    # Boundary callbacks
    # -------------------------------------------------------------------

    def _on_daily_boundary(
        self, daily: NumpyBars, d_idx: int,
        daily_es: NumpyBars | None, es_idx: int,
    ) -> None:
        """Daily: vol state, trend, extension, risk reset."""
        if d_idx < 1:
            return

        po = self.po
        end = d_idx + 1

        # ATR daily
        period = 14
        if end > period:
            self._atr_d = compute_atr(daily.highs[:end], daily.lows[:end], daily.closes[:end], period)
        self._atr_d_history.append(self._atr_d)
        if len(self._atr_d_history) > 60 and self._atr_d_baseline == 0:
            self._atr_d_baseline = float(np.median(self._atr_d_history[-60:]))

        # Vol state + cache pctl for chop score
        if len(self._atr_d_history) >= 60:
            atr_med = float(np.median(self._atr_d_history[-60:]))
            atr_pct = percentile_rank(self._atr_d, np.array(self._atr_d_history[-60:]), 60)
            self._atr_d_pctl = atr_pct
            self._regime.vol_state = compute_vol_state(atr_pct, self._atr_d, atr_med)
            self._regime.vol_factor = compute_vol_factor(
                self._atr_d_baseline, self._atr_d, atr_pct,
            )

        # EMA fast/slow
        ema_fast_p = int(po.get("ema_fast_period", 20))
        ema_slow_p = int(po.get("ema_slow_period", 50))
        sma200_p = int(po.get("sma200_period", 200))
        self._ema_fast_d = compute_ema(daily.closes[:end], ema_fast_p)
        self._ema_slow_d = compute_ema(daily.closes[:end], ema_slow_p)
        if end >= sma200_p:
            self._sma200_d = compute_sma(daily.closes[:end], sma200_p)
            self._progressive_sma_warmup = False
        elif self.flags.progressive_sma:
            progressive_min = int(po.get("progressive_sma_min", 50))
            if end >= progressive_min:
                self._sma200_d = float(np.mean(daily.closes[:end]))
            # R6 Rev2: flag warmup period for direct regime override
            self._progressive_sma_warmup = True

        # Daily trend with 2-bar persistence
        close_d = daily.closes[d_idx]

        # Short SMA for alternative trend signal
        if self.flags.short_sma_trend:
            short_sma_p = int(po.get("short_sma_period", 50))
            if end >= short_sma_p:
                self._short_sma_d = compute_sma(daily.closes[:end], short_sma_p)
                self._regime.short_trend = -1 if close_d < self._short_sma_d else 0
            else:
                self._regime.short_trend = 0
        if self._sma200_d > 0:
            trend, consec = classify_daily_trend(
                close_d, self._sma200_d,
                self._regime.daily_trend, self._regime.daily_trend_consec,
            )
            self._regime.daily_trend = trend
            self._regime.daily_trend_consec = consec

        # Extension
        self._regime.extension_short, self._regime.extension_long = compute_extension(
            close_d, self._ema_fast_d, self._atr_d,
        )

        # Trend strength
        ts = compute_trend_strength(self._ema_fast_d, self._ema_slow_d, self._atr_d)
        self._regime.trend_strength = ts
        self._trend_strength_3d.append(ts)

        # Fast-crash override paths E/F/G
        if self.flags.fast_crash_override:
            self._fast_crash_active = check_fast_crash_override(
                daily.closes[:end], self._ema_fast_d,
                self._atr_d, self._atr_d_baseline, self.po,
            )
        else:
            self._fast_crash_active = False

        # Bear conviction scoring (daily ADX suite + conviction)
        if self.flags.conviction_scoring:
            self._daily_adx, self._daily_plus_di, self._daily_minus_di = (
                compute_adx_suite(daily.highs[:end], daily.lows[:end], daily.closes[:end], 14)
            )
            self._bear_conviction = compute_bear_conviction(
                self._daily_adx, self._daily_plus_di, self._daily_minus_di,
                self._ema_fast_d, self._ema_slow_d, close_d,
                prev_ema_fast=self._prev_ema_fast_d,
            )

        # Bear structure override (ADX hysteresis + paths B/C + BEAR_FORMING)
        if self.flags.bear_structure_override:
            # Compute daily ADX suite if not already done by conviction_scoring
            if not self.flags.conviction_scoring:
                self._daily_adx, self._daily_plus_di, self._daily_minus_di = (
                    compute_adx_suite(daily.highs[:end], daily.lows[:end], daily.closes[:end], 14)
                )
                self._bear_conviction = compute_bear_conviction(
                    self._daily_adx, self._daily_plus_di, self._daily_minus_di,
                    self._ema_fast_d, self._ema_slow_d, close_d,
                    prev_ema_fast=self._prev_ema_fast_d,
                )
            # ADX hysteresis
            adx_on = self.po.get("bear_structure_adx_on", 25.0)
            adx_off = self.po.get("bear_structure_adx_off", 15.0)
            self._regime_on = compute_regime_on(
                self._daily_adx, self._regime_on, adx_on, adx_off,
            )
            # Structure check
            self._bear_structure_active = check_bear_structure_override(
                self._daily_adx, self._daily_plus_di, self._daily_minus_di,
                close_d, self._ema_fast_d, self._ema_slow_d,
                self._regime_on, self._bear_conviction, self.po,
            )
        else:
            self._bear_structure_active = False
        self._prev_ema_fast_d = self._ema_fast_d

        # R6: Real-time drawdown override
        if self.flags.drawdown_regime_override:
            dd_lookback = int(po.get("drawdown_lookback", 20))
            dd_threshold = po.get("drawdown_threshold", 0.03)
            self._drawdown_override_active = check_drawdown_override(
                daily.closes[:end], dd_lookback, dd_threshold,
            )
        else:
            self._drawdown_override_active = False

        # Daily risk reset
        self._daily_loss = 0.0
        self._daily_trades = 0
        self._circuit_breaker_tripped = False

    def _on_4h_boundary(self, four_hour: NumpyBars, fh_idx: int) -> None:
        """4H: regime, reversal pivot tracking, strong_bear."""
        if fh_idx < 2:
            return

        po = self.po
        end = fh_idx + 1

        # ADX (period is always 14; thresholds are separate params)
        adx_val = compute_adx(four_hour.highs[:end], four_hour.lows[:end], four_hour.closes[:end], 14)

        # EMA50 slope on 4H
        ema50_arr = compute_ema_array(four_hour.closes[:end], 50)
        slope_4h = ema50_arr[-1] - ema50_arr[-2] if len(ema50_arr) >= 2 else 0.0
        slope_dir = 1 if slope_4h > 0 else -1

        # 4H regime (pass configurable thresholds)
        self._regime.regime_4h = classify_4h_regime(
            adx_val, slope_4h,
            adx_trending_threshold=po.get("adx_trending_threshold", 25.0),
            adx_range_threshold=po.get("adx_range_threshold", 15.0),
        )

        # Composite regime
        self._regime.composite_regime = compute_composite_regime(
            self._regime.regime_4h, self._regime.daily_trend, slope_dir,
            short_trend=self._regime.short_trend if self.flags.short_sma_trend else 0,
        )

        # ATR 4H fast/slow for vol coil
        self._atr_4h = compute_atr(four_hour.highs[:end], four_hour.lows[:end], four_hour.closes[:end], 14)
        self._atr_4h_fast = compute_atr(four_hour.highs[:end], four_hour.lows[:end], four_hour.closes[:end], 5)
        self._atr_4h_slow = compute_atr(four_hour.highs[:end], four_hour.lows[:end], four_hour.closes[:end], 20)

        # MACD on 4H for divergence
        _, _, hist = compute_macd_hist(four_hour.closes[:end])
        self._4h_highs.append(four_hour.highs[fh_idx])
        self._4h_lows.append(four_hour.lows[fh_idx])
        self._4h_macd_hist.append(hist)

        # Reversal pivot tracking: find H1/H2 pairs
        self._update_reversal_pivots()

        # Strong bear
        alignment = 1.0 if self._regime.composite_regime == CompositeRegime.ALIGNED_BEAR else 0.0
        self._regime.strong_bear = compute_strong_bear(self._regime.trend_strength, alignment)

        # Disable reversal in strong bear (unless flag allows it) or shock
        self._reversal.disabled = (
            (self._regime.strong_bear and not self.flags.allow_reversal_strong_bear)
            or self._regime.vol_state == VolState.SHOCK
        )

    def _on_1h_boundary(self, hourly: NumpyBars, h_idx: int, close_5m: float) -> None:
        """1H: chandelier trail, EMA20."""
        if h_idx < 2:
            return

        end = h_idx + 1
        self._atr_1h = self._inc_atr_1h.update(
            hourly.highs[h_idx], hourly.lows[h_idx], hourly.closes[h_idx],
        )
        self._ema20_1h = self._inc_ema_1h_20.update(hourly.closes[h_idx])

        # Update position hold bars
        if self._position is not None:
            self._position.hold_bars_1h += 1

    def _on_30m_boundary(self, thirty_min: NumpyBars, m30_idx: int) -> None:
        """30m: box state, chop score, ATR."""
        if m30_idx < 2:
            return

        end = m30_idx + 1
        self._atr_30m = self._inc_atr_30m.update(
            thirty_min.highs[m30_idx], thirty_min.lows[m30_idx], thirty_min.closes[m30_idx],
        )

        # Adaptive box length
        if self._atr_30m > 0:
            atr_fast = self._inc_atr_30m_fast.update(
                thirty_min.highs[m30_idx], thirty_min.lows[m30_idx], thirty_min.closes[m30_idx],
            )
            atr_ratio = atr_fast / self._atr_30m if self._atr_30m > 0 else 1.0
            adaptive_L = compute_box_adaptive_length(atr_ratio)
        else:
            adaptive_L = 32

        # Update box state
        self._box = update_box_state(
            self._box,
            thirty_min.highs[m30_idx],
            thirty_min.lows[m30_idx],
            thirty_min.closes[m30_idx],
            self._atr_30m,
            adaptive_L,
            self.po,
        )

        # Box VWAP
        if self._box.active and self._box.age > 0:
            start = max(0, m30_idx - self._box.age)
            self._box.vwap_box = compute_session_vwap(
                thirty_min.highs[start:end], thirty_min.lows[start:end],
                thirty_min.closes[start:end], thirty_min.volumes[start:end], 0,
            )

        # Chop score
        if len(self._atr_d_history) >= 60:
            atr_pctl = percentile_rank(self._atr_d, np.array(self._atr_d_history[-60:]), 60)
        else:
            atr_pctl = 0.5
        # Count VWAP crosses (simplified)
        self._vwap_cross_count = 0  # tracked in 15m boundary

        # Update position hold bars
        if self._position is not None:
            self._position.hold_bars_30m += 1

    def _on_15m_boundary(self, fifteen_min: NumpyBars, m15_idx: int) -> None:
        """15m: session VWAP, fade state, momentum."""
        if m15_idx < 2:
            return

        end = m15_idx + 1

        # Session VWAP (reset each day via session start tracking)
        self._fade.vwap_session = compute_session_vwap(
            fifteen_min.highs[self._session_start_15m:end],
            fifteen_min.lows[self._session_start_15m:end],
            fifteen_min.closes[self._session_start_15m:end],
            fifteen_min.volumes[self._session_start_15m:end],
            0,
        )
        self._fade.vwap_used = self._fade.vwap_session

        # Touch tracking
        vwap = self._fade.vwap_used
        touched = fifteen_min.highs[m15_idx] >= vwap if vwap > 0 else False
        self._fade.touch_bars.append(touched)
        if len(self._fade.touch_bars) > 8:
            self._fade.touch_bars = self._fade.touch_bars[-8:]

        # Consecutive above VWAP
        if vwap > 0 and fifteen_min.closes[m15_idx] > vwap:
            self._fade.consecutive_above_vwap += 1
        else:
            self._fade.consecutive_above_vwap = 0

        # ATR 15m (incremental O(1))
        self._atr_15m = self._inc_atr_15m.update(
            fifteen_min.highs[m15_idx], fifteen_min.lows[m15_idx], fifteen_min.closes[m15_idx],
        )

        # Momentum (incremental EMA difference as proxy)
        close_15m = fifteen_min.closes[m15_idx]
        ema_fast = self._inc_ema_15m_fast.update(close_15m)
        ema_slow = self._inc_ema_15m_slow.update(close_15m)
        self._mom15.append(ema_fast - ema_slow)
        if len(self._mom15) > 3:
            self._mom15_slope_ok = compute_momentum_slope_ok(
                np.array(self._mom15[-10:]), len(self._mom15[-10:]) - 1, 3,
            )
        else:
            self._mom15_slope_ok = False

        # VWAP cross counting for chop score
        if len(self._mom15) >= 2:
            if self._mom15[-1] * self._mom15[-2] < 0:
                self._vwap_cross_count += 1

        # Track session start (simplified: detect new day)
        if m15_idx > 0:
            prev_time = fifteen_min.times[m15_idx - 1]
            curr_time = fifteen_min.times[m15_idx]
            if isinstance(prev_time, np.datetime64):
                prev_day = prev_time.astype("datetime64[D]")
                curr_day = curr_time.astype("datetime64[D]")
                if curr_day != prev_day:
                    self._session_start_15m = m15_idx
                    self._vwap_cross_count = 0

    # -------------------------------------------------------------------
    # Reversal pivot tracking
    # -------------------------------------------------------------------

    def _update_reversal_pivots(self) -> None:
        """Scan 4H highs for H1/H2 divergence pairs."""
        if len(self._4h_highs) < 10:
            return

        # Simple pivot detection: local highs in last 20 bars
        highs = self._4h_highs[-20:]
        macd = self._4h_macd_hist[-20:]

        # Find last two swing highs
        pivots = []
        for i in range(2, len(highs) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                pivots.append((i, highs[i], macd[i] if i < len(macd) else 0.0))

        if len(pivots) >= 2:
            p1 = pivots[-2]
            p2 = pivots[-1]
            self._reversal.h1_price = p1[1]
            self._reversal.h1_idx = len(self._4h_highs) - 20 + p1[0]
            self._reversal.h2_price = p2[1]
            self._reversal.h2_idx = len(self._4h_highs) - 20 + p2[0]
            self._reversal.macd_at_h1 = p1[2]
            self._reversal.macd_at_h2 = p2[2]
            lows_slice = self._4h_lows[self._reversal.h1_idx:self._reversal.h2_idx + 1]
            self._reversal.l_between = min(lows_slice) if lows_slice else 0.0
            self._reversal.divergence_arm_active = True

    # -------------------------------------------------------------------
    # Signal evaluation
    # -------------------------------------------------------------------

    def _can_enter(self, bar_time: datetime) -> bool:
        """Check common entry gates."""
        # Circuit breaker
        if self.flags.daily_circuit_breaker and self._circuit_breaker_tripped:
            return False

        # Vol shock block
        if self.flags.use_shock_block and self._regime.vol_state == VolState.SHOCK:
            return False

        # Session window
        if self.flags.use_entry_windows:
            can, _ = self._classify_session(bar_time)
            if not can:
                return False

        # Directional entry cap (max N short entries per day)
        if self.flags.directional_entry_caps:
            max_daily = int(self.po.get("max_daily_entries", 3))
            if self._daily_trades >= max_daily:
                return False

        # News blackout (simplified: skip first/last 5 min of RTH)
        if self.flags.use_news_blackout:
            try:
                et = bar_time.astimezone(ET)
                mins = et.hour * 60 + et.minute
                # Skip 09:30-09:35 and 15:55-16:00 ET (high-impact windows)
                if 570 <= mins < 575 or 955 <= mins < 960:
                    return False
            except Exception:
                pass

        # Friction gate: require minimum ATR for tradeable conditions
        if self.flags.friction_gate and self._atr_d > 0:
            min_atr_pctl = self.po.get("friction_min_atr_pctl", 0.10)
            if self._atr_d_pctl < min_atr_pctl:
                return False

        vol_gate = float(getattr(self.flags, "vol_percentile_gate", 0.0) or 0.0)
        if vol_gate > 0 and self._atr_d_pctl * 100.0 < vol_gate:
            return False

        conviction_gate = float(getattr(self.flags, "regime_confidence_gate", 0.0) or 0.0)
        if conviction_gate > 0 and self._bear_conviction < conviction_gate:
            return False

        # Block counter-regime entries (consistently lose money)
        if self.flags.block_counter_regime:
            if self._regime.composite_regime == CompositeRegime.COUNTER:
                # Allow through if reversal-in-correction exemption applies
                # (actual per-signal filtering happens in _evaluate_signals)
                if not (self.flags.allow_reversal_in_correction
                        and self._in_correction(bar_time, self._correction_windows)):
                    return False

        return True

    def _evaluate_signals(
        self, t: int, bar_time: datetime, close: float, high: float, low: float,
        fifteen_min: NumpyBars, m15_idx: int,
        thirty_min: NumpyBars, m30_idx: int,
        four_hour: NumpyBars, fh_idx: int,
        daily: NumpyBars, d_idx: int,
    ) -> Optional[ReversalSignal | BreakdownSignal | FadeSignal]:
        """Evaluate signals in priority order: Breakdown > Reversal > Fade."""

        self._momentum_impulse_pending = False

        # Correction-window regime override: treat NEUTRAL/RANGE as EMERGING_BEAR
        # during correction windows to unlock fade/reversal signals
        effective_regime = self._regime.composite_regime
        if (self.flags.correction_regime_override
                and effective_regime in (CompositeRegime.NEUTRAL, CompositeRegime.RANGE)
                and self._in_correction(bar_time, self._correction_windows)):
            effective_regime = CompositeRegime.EMERGING_BEAR

        # Fast-crash override (independent of correction windows)
        if (self.flags.fast_crash_override and self._fast_crash_active
                and effective_regime in (CompositeRegime.NEUTRAL, CompositeRegime.RANGE)):
            # Conviction gate: require minimum bear conviction before overriding
            if self.flags.conviction_scoring:
                threshold = self.po.get("conviction_threshold", 50)
                if self._bear_conviction >= threshold:
                    effective_regime = CompositeRegime.EMERGING_BEAR
            else:
                effective_regime = CompositeRegime.EMERGING_BEAR

        # Bear structure override (gradual corrections -- paths B/C + BEAR_FORMING)
        if (self.flags.bear_structure_override and self._bear_structure_active
                and effective_regime in (CompositeRegime.NEUTRAL, CompositeRegime.RANGE)):
            effective_regime = CompositeRegime.EMERGING_BEAR

        # R6: Drawdown override (real-time rolling-high drawdown detection)
        if (self.flags.drawdown_regime_override and self._drawdown_override_active
                and effective_regime in (
                    CompositeRegime.NEUTRAL, CompositeRegime.RANGE, CompositeRegime.COUNTER,
                )):
            effective_regime = CompositeRegime.EMERGING_BEAR

        # R6 Rev2: Progressive SMA regime override during warmup period
        # When data < sma200_period AND price is below progressive SMA,
        # directly override to EMERGING_BEAR (bypasses composite regime hierarchy)
        if (self.flags.progressive_sma
                and self._progressive_sma_warmup
                and self._sma200_d > 0
                and daily.closes[d_idx] < self._sma200_d
                and effective_regime in (
                    CompositeRegime.NEUTRAL, CompositeRegime.RANGE, CompositeRegime.COUNTER,
                )):
            effective_regime = CompositeRegime.EMERGING_BEAR

        # Chop check (atr_d_pctl cached on daily boundary)
        chop = compute_chop_score(self._atr_d_pctl, self._vwap_cross_count)

        # Counter-regime: if allow_reversal_in_correction let us through _can_enter,
        # only reversal signals are allowed; block breakdown and fade.
        counter_reversal_only = (
            self.flags.block_counter_regime
            and self._regime.composite_regime == CompositeRegime.COUNTER
            and self.flags.allow_reversal_in_correction
            and self._in_correction(bar_time, self._correction_windows)
        )

        # 1. Breakdown
        if self.flags.breakdown_engine and self._box.active and m30_idx > 0 and not counter_reversal_only:
            close_30m = thirty_min.closes[m30_idx]
            disp = compute_displacement_metric(close_30m, self._box.vwap_box, self._atr_30m)
            bar_range = thirty_min.highs[m30_idx] - thirty_min.lows[m30_idx]
            body = abs(thirty_min.closes[m30_idx] - thirty_min.opens[m30_idx])
            body_ratio = body / bar_range if bar_range > 0 else 0.5
            avg_vol = float(np.mean(thirty_min.volumes[max(0, m30_idx - 20):m30_idx + 1]))
            rvol = thirty_min.volumes[m30_idx] / avg_vol if avg_vol > 0 else 1.0

            sig = detect_breakdown_short(
                self._box, close_30m, disp, self._box.displacement_history,
                chop, bar_range, body_ratio, rvol, self._atr_30m,
                self.flags, self.po,
            )
            if sig is not None:
                self._breakdown_ctr.signals_detected += 1
                self._record_signal(EngineTag.BREAKDOWN, "box_breakdown", bar_time, True)
                return sig
            elif close_30m < self._box.range_low:
                self._breakdown_ctr.gates_blocked += 1
                self._record_signal(EngineTag.BREAKDOWN, "box_breakdown", bar_time, False)

        # 2. Reversal (only during transition, not strong bear — unless flag allows it)
        strong_bear_blocks = (
            self._regime.strong_bear and not self.flags.allow_reversal_strong_bear
        )
        if (self.flags.reversal_engine
                and not strong_bear_blocks
                and self._regime.vol_state != VolState.SHOCK):
            ts_3d = self._trend_strength_3d[-4] if len(self._trend_strength_3d) >= 4 else 0.0
            sig = detect_reversal_short(
                self._reversal,
                self._regime.trend_strength, ts_3d,
                daily.closes[d_idx] if d_idx >= 0 else close,
                self._ema_fast_d, self._atr_d,
                self._atr_4h_fast, self._atr_4h_slow,
                self.flags, self.po,
            )
            if sig is not None:
                self._reversal_ctr.signals_detected += 1
                self._record_signal(EngineTag.REVERSAL, "classic_divergence", bar_time, True)
                return sig

        # 3. Fade (workhorse during established bear)
        if self.flags.fade_engine and m15_idx > 0 and not counter_reversal_only:
            close_15m = fifteen_min.closes[m15_idx]
            lookback = int(self.po.get("rejection_lookback_bars", 8))
            start = max(0, m15_idx - lookback + 1)
            high_recent = fifteen_min.highs[start:m15_idx + 1]

            atr_15m = self._atr_15m
            mom_ok = self._mom15_slope_ok
            _, session_mult = self._classify_session(bar_time)
            session_type = "core" if session_mult >= 1.0 else "extended"

            sig = detect_fade_short(
                self._fade, close_15m, high_recent,
                effective_regime, mom_ok,
                self._regime.extension_short, atr_15m, session_type,
                self.flags, self.po,
            )
            if sig is not None:
                self._fade_ctr.signals_detected += 1
                self._record_signal(EngineTag.FADE, "vwap_rejection", bar_time, True)
                return sig

            # R6: Momentum impulse — alternative fade trigger (no VWAP rejection needed)
            # R6 Rev2: cooldown gate — only fire if no entry in last N bars
            momentum_cooldown = int(self.po.get("momentum_cooldown_bars", 36))
            if (self.flags.momentum_signal and sig is None
                    and self._bars_since_last_entry >= momentum_cooldown):
                close_5ago = (
                    fifteen_min.closes[m15_idx - 5]
                    if m15_idx >= 5 else fifteen_min.closes[0]
                )
                roc_5bar = (
                    (close_15m - close_5ago) / close_5ago if close_5ago > 0 else 0.0
                )
                ema_fast_15m = self._inc_ema_15m_fast.value
                if detect_momentum_impulse(
                    close_15m, ema_fast_15m, roc_5bar,
                    effective_regime, self.po,
                ):
                    # Build a FadeSignal with momentum_impulse class
                    sig = FadeSignal(
                        vwap_used=self._fade.vwap_used,
                        rejection_close=close_15m,
                        class_mult=0.70,
                        predator_present=False,
                    )
                    self._fade_ctr.signals_detected += 1
                    self._record_signal(EngineTag.FADE, "momentum_impulse", bar_time, True)
                    self._momentum_impulse_pending = True
                    return sig

        return None

    # -------------------------------------------------------------------
    # Entry submission
    # -------------------------------------------------------------------

    def _submit_entry(
        self, signal, t: int, bar_time: datetime, close: float,
        equity: float, correction_windows: list[CorrectionWindow],
    ) -> None:
        """Submit entry order via broker."""
        cfg = self.config
        po = self.po

        if isinstance(signal, BreakdownSignal):
            tag = EngineTag.BREAKDOWN
            sig_class = "box_breakdown"
        elif isinstance(signal, ReversalSignal):
            tag = EngineTag.REVERSAL
            sig_class = "classic_divergence"
        else:
            tag = EngineTag.FADE
            sig_class = "momentum_impulse" if self._momentum_impulse_pending else "vwap_rejection"

        # Compute entry/stop
        atr = self._atr_30m if tag == EngineTag.BREAKDOWN else self._atr_1h
        trigger_buffer_ticks = max(0.0, float(po.get("trigger_low_buffer_ticks", 2.0)))
        low_recent = close - trigger_buffer_ticks * cfg.tick_size
        entry_price, stop0, entry_type = compute_entry_subtype_stop(
            tag, signal, close, atr, low_recent, cfg.tick_size, po,
        )

        # Position sizing
        risk_per_unit = abs(stop0 - entry_price) * cfg.point_value
        if risk_per_unit <= 0:
            return

        base_risk = po.get("base_risk_pct", 0.01)
        regime_mult = regime_sizing_mult(self._regime.composite_regime, po)
        vol_factor = self._regime.vol_factor if self.flags.use_volatility_states else 1.0
        strong_bonus = 1.25 if self._regime.strong_bear and self.flags.use_strong_bear_bonus else 1.0

        risk_dollars = equity * base_risk * regime_mult * vol_factor * strong_bonus

        # Determine if in correction window (needed for sizing adjustments)
        in_correction = self._in_correction(bar_time, correction_windows)

        # Correction-window sizing adjustments
        if in_correction and self.flags.correction_sizing_bonus:
            corr_bonus = po.get("correction_sizing_mult", 1.30)
            risk_dollars *= corr_bonus
        if not in_correction and self.flags.non_correction_penalty:
            non_corr_mult = po.get("non_correction_sizing_mult", 0.60)
            risk_dollars *= non_corr_mult

        qty = max(1, int(risk_dollars / risk_per_unit))
        if self.config.max_contracts > 0:
            qty = min(qty, self.config.max_contracts)
        if self.config.max_notional_leverage > 0:
            notional_per = entry_price * self.config.point_value
            max_qty = max(1, int(self._current_equity * self.config.max_notional_leverage / notional_per))
            qty = min(qty, max_qty)

        # TP schedule
        tp_sched = compute_tiered_tp_schedule(tag, self._regime.composite_regime, po)

        # Build pending position info (will activate on fill)
        predator = getattr(signal, "predator_present", False)

        # Submit order
        oid = self.broker.next_order_id()
        limit_offset_ticks = max(0.0, float(po.get("entry_limit_offset_ticks", 4.0)))
        limit_offset = limit_offset_ticks * cfg.tick_size
        ttl_bars = max(1, int(po.get("entry_ttl_bars", 72)))
        entry_request = DownturnEntryRequest(
            client_order_id=oid,
            symbol=self.symbol,
            engine_tag=tag,
            signal_class=sig_class,
            qty=qty,
            entry_price=entry_price,
            stop0=stop0,
            tif="DAY",
            order_type="STOP" if entry_type == "stop_market" else "STOP_LIMIT",
            price=entry_price if entry_type != "stop_market" else None,
            limit_price=entry_price - limit_offset if entry_type != "stop_market" else None,
            stop_price=entry_price,
            submitted_bar_idx=t,
            ttl_bars=ttl_bars,
            composite_regime=self._regime.composite_regime,
            vol_state=self._regime.vol_state,
            in_correction=in_correction,
            predator=predator,
            tp_schedule=tp_sched,
            signal_strength=getattr(signal, "class_mult", 0.5),
        )
        replay = self._replay_core_step(
            bar_input={
                "bar_count_5m": t,
                "bar_ts": bar_time,
                "entry_request": entry_request,
            }
        )
        submit_action = next((action for action in replay.actions if isinstance(action, SubmitEntry)), None)
        if submit_action is None:
            return
        order = _sim_order_from_parity(
            neutral_action_to_sim_order(
                submit_action,
                tick_size=cfg.tick_size,
                submit_time=bar_time,
            )
        )
        order.tag = "entry"
        self.broker.submit_order(order)
        self._replay_core_step(
            order_updates=[
                DownturnOrderUpdate(
                    oms_order_id=order.order_id,
                    status="accepted",
                    timestamp=bar_time,
                    order_role="entry",
                    accepted_entry=entry_request,
                )
            ]
        )

        # Store pending position data on order for fill handler
        order._pending_data = {
            "engine_tag": tag,
            "signal_class": sig_class,
            "stop0": stop0,
            "composite_regime": self._regime.composite_regime,
            "vol_state": self._regime.vol_state,
            "in_correction": in_correction,
            "predator": predator,
            "tp_schedule": tp_sched,
        }

        ctr = self._get_counter(tag)
        ctr.entries_placed += 1

    # -------------------------------------------------------------------
    # Fill handling
    # -------------------------------------------------------------------

    def _handle_fill(
        self, fill: FillResult, bar_time: datetime, close: float,
        equity: float, correction_windows: list[CorrectionWindow],
    ) -> None:
        """Route fill to entry or exit handler."""
        if fill.status == FillStatus.FILLED:
            if fill.order.tag == "entry" and self._position is None:
                self._replay_core_step(
                    fills=[
                        DownturnFill(
                            oms_order_id=fill.order.order_id,
                            fill_price=fill.fill_price,
                            fill_qty=fill.order.qty,
                            commission=fill.commission,
                            fill_time=bar_time,
                        )
                    ]
                )
                self._on_entry_fill(fill, bar_time, correction_windows)
            elif fill.order.tag == "protective_stop" and self._position is not None:
                self._replay_core_step(
                    fills=[
                        DownturnFill(
                            oms_order_id=fill.order.order_id,
                            fill_price=fill.fill_price,
                            fill_qty=fill.order.qty,
                            commission=fill.commission,
                            fill_time=bar_time,
                            exit_type="stop",
                        )
                    ]
                )
                self._on_exit_fill(fill, bar_time, "stop")
            elif fill.order.tag.startswith("tp") and self._position is not None:
                if self._position.remaining_qty <= fill.order.qty:
                    self._replay_core_step(
                        fills=[
                            DownturnFill(
                                oms_order_id=fill.order.order_id,
                                fill_price=fill.fill_price,
                                fill_qty=fill.order.qty,
                                commission=fill.commission,
                                fill_time=bar_time,
                                exit_type=fill.order.tag,
                            )
                        ]
                    )
                self._on_tp_fill(fill, bar_time)
        elif fill.status in (FillStatus.EXPIRED, FillStatus.CANCELLED, FillStatus.REJECTED):
            self._replay_core_step(
                order_updates=[
                    DownturnOrderUpdate(
                        oms_order_id=fill.order.order_id,
                        status=fill.status.name.lower(),
                        timestamp=bar_time,
                    )
                ]
            )

    def _on_entry_fill(
        self, fill: FillResult, bar_time: datetime,
        correction_windows: list[CorrectionWindow],
    ) -> None:
        """Handle entry fill — create active position."""
        self._bars_since_last_entry = 0  # R6 Rev2: reset cooldown counter

        data = getattr(fill.order, "_pending_data", {})
        if not data:
            return

        self._position = _ActivePosition(
            engine_tag=data["engine_tag"],
            signal_class=data["signal_class"],
            entry_price=fill.fill_price,
            stop0=data["stop0"],
            qty=fill.order.qty,
            entry_time=bar_time,
            entry_bar_idx=0,
            composite_regime=data["composite_regime"],
            vol_state=data["vol_state"],
            in_correction=data["in_correction"],
            predator=data["predator"],
            tp_schedule=data["tp_schedule"],
        )
        self._position.commission += fill.commission
        self._total_commission += fill.commission

        # Submit protective stop
        self._submit_protective_stop(data["stop0"], fill.order.qty, bar_time)

        ctr = self._get_counter(data["engine_tag"])
        ctr.entries_filled += 1

    def _on_exit_fill(self, fill: FillResult, bar_time: datetime, exit_type: str) -> None:
        """Handle exit fill — close position, record trade."""
        pos = self._position
        if pos is None:
            return

        # R6: Exit type tagging — distinguish profit_floor exits from initial stop hits
        if exit_type == "stop" and pos.exit_trigger:
            exit_type = pos.exit_trigger

        pos.commission += fill.commission
        self._total_commission += fill.commission

        # Use original qty (not remaining_qty which may be 0 after TP partial decrement)
        close_qty = pos.qty
        pnl = (pos.entry_price - fill.fill_price) * close_qty * self.config.point_value
        pnl -= pos.commission
        r_mult = pos.r_state(fill.fill_price)
        self._realized_pnl += pnl

        self._trades.append(DownturnTradeRecord(
            symbol=self.symbol,
            direction=Direction.SHORT,
            entry_price=pos.entry_price,
            exit_price=fill.fill_price,
            entry_time=pos.entry_time,
            exit_time=bar_time,
            qty=pos.qty,
            pnl=pnl,
            r_multiple=r_mult,
            stop0=pos.stop0,
            commission=pos.commission,
            entry_type="stop_market" if pos.engine_tag == EngineTag.REVERSAL else "stop_limit",
            exit_type=exit_type,
            hold_bars=pos.hold_bars_1h,
            hold_bars_5m=pos.hold_bars_5m,
            mfe=pos.r_at_peak,
            mae=(pos.entry_price - pos.mae_price) / pos.risk_per_unit if pos.risk_per_unit > 0 else 0.0,
            engine_tag=pos.engine_tag,
            composite_regime_at_entry=pos.composite_regime,
            vol_state_at_entry=pos.vol_state,
            in_correction_window=pos.in_correction,
            predator_present=pos.predator,
            signal_class=pos.signal_class,
        ))

        # Daily risk tracking
        self._daily_loss += min(0, pnl)
        self._daily_trades += 1
        if self.flags.daily_circuit_breaker:
            cb_threshold = self.po.get("circuit_breaker_threshold", -3000.0)
            if self._daily_loss <= cb_threshold:
                self._circuit_breaker_tripped = True

        # Cancel remaining orders
        self.broker.cancel_all(self.symbol)
        self._position = None

    def _on_tp_fill(self, fill: FillResult, bar_time: datetime) -> None:
        """Handle TP fill — partial close."""
        pos = self._position
        if pos is None:
            return

        pos.commission += fill.commission
        self._total_commission += fill.commission
        pos.remaining_qty -= fill.order.qty
        if pos.remaining_qty <= 0:
            self._on_exit_fill(fill, bar_time, f"tp{pos.tp_idx + 1}")
            return
        pos.tp_idx += 1

    # -------------------------------------------------------------------
    # Position management
    # -------------------------------------------------------------------

    def _manage_position(
        self, t: int, bar_time: datetime, H: float, L: float, Cl: float,
        hourly: NumpyBars, h_idx: int,
    ) -> None:
        """Manage open position: TPs, chandelier, stale, climax exits."""
        pos = self._position
        if pos is None:
            return

        # Track MFE/MAE (for shorts: lower price = better)
        if Cl < pos.mfe_price:
            pos.mfe_price = Cl
        if Cl > pos.mae_price:
            pos.mae_price = Cl

        r = pos.r_state(Cl)
        pos.r_at_peak = max(pos.r_at_peak, r)

        # R6: Min hold period — skip all exits except catastrophic for first N bars
        # Uses 5m bar count: default 6 bars = 30 minutes
        if self.flags.min_hold_period:
            min_bars = int(self.po.get("min_hold_bars", 6))
            if pos.hold_bars_5m < min_bars:
                if check_catastrophic_exit(r):
                    self._submit_market_exit(pos.remaining_qty, bar_time, "catastrophic")
                return

        # Profit floor: multi-tier replaces single-tier when enabled
        if self.flags.multi_tier_profit_floor and pos.risk_per_unit > 0:
            mt_stop = compute_multi_tier_profit_floor(
                pos.entry_price, pos.r_at_peak, pos.risk_per_unit,
                self.config.tick_size, self.po,
            )
            if mt_stop is not None and mt_stop < pos.chandelier_stop:
                pos.chandelier_stop = mt_stop
                pos.exit_trigger = "profit_floor"
                self._update_protective_stop(mt_stop, pos.remaining_qty, bar_time)
        elif self.flags.profit_floor_trail and pos.risk_per_unit > 0:
            # R6: Adaptive lock_pct — increase capture from big winners
            if self.flags.adaptive_profit_floor:
                base_lock = self.po.get("profit_floor_lock_pct", 0.40)
                adapted_lock = compute_adaptive_lock_pct(pos.r_at_peak, base_lock, self.po)
                po_adapted = {**self.po, "profit_floor_lock_pct": adapted_lock}
            else:
                po_adapted = self.po
            pf_stop = compute_profit_floor_stop(
                pos.entry_price, r, pos.risk_per_unit,
                self.config.tick_size, po_adapted,
            )
            if pf_stop is not None and pf_stop < pos.chandelier_stop:
                pos.chandelier_stop = pf_stop
                pos.exit_trigger = "profit_floor"
                self._update_protective_stop(pf_stop, pos.remaining_qty, bar_time)

        # Breakeven stop after configurable R threshold (default +1.0R)
        be_trigger_r = self.po.get("be_trigger_r", 1.0)
        if not pos.be_triggered and r >= be_trigger_r and self._atr_1h > 0:
            be_stop = compute_breakeven_stop(
                pos.entry_price, self._atr_1h, self.config.tick_size, self.po,
            )
            if be_stop < pos.chandelier_stop:
                pos.chandelier_stop = be_stop
                self._update_protective_stop(pos.chandelier_stop, pos.remaining_qty, bar_time)
            pos.be_triggered = True

        # Chandelier trail (1H basis)
        if self.flags.chandelier_trailing and self._atr_1h > 0 and h_idx > 14:
            end = h_idx + 1
            lookback = int(self.po.get("chandelier_lookback", 14))
            start = max(0, end - lookback)
            ll = float(np.min(hourly.lows[start:end]))

            # Regime-adaptive chandelier width
            r_mult = None
            if self.flags.regime_adaptive_chandelier:
                r_mult = compute_chandelier_regime_mult(
                    self._regime.composite_regime, self.po,
                )

            new_stop = update_chandelier_trail(
                ll, self._atr_1h, r, self._regime.strong_bear,
                pos.chandelier_stop, self.config.tick_size, self.po,
                tp1_hit=(pos.tp_idx > 0),
                regime_mult=r_mult,
            )
            if new_stop < pos.chandelier_stop:
                pos.chandelier_stop = new_stop
                self._update_protective_stop(new_stop, pos.remaining_qty, bar_time)

        # Scale-out: partial profit lock at target R
        if (self.flags.scale_out_enabled and not pos.scaled_out
                and pos.remaining_qty > 1):
            so_target = self.po.get("scale_out_target_r", 3.0)
            if r >= so_target:
                so_pct = self.po.get("scale_out_pct", 0.30)
                qty_out = max(1, int(pos.qty * so_pct))
                qty_out = min(qty_out, pos.remaining_qty - 1)  # keep at least 1
                if qty_out > 0:
                    self._submit_market_exit(qty_out, bar_time, "scale_out")
                    # Widen chandelier for runner
                    so_bonus = self.po.get("scale_out_trail_bonus", 0.3)
                    pos.chandelier_stop += so_bonus * self._atr_1h
                    pos.scaled_out = True

        # Tiered TPs
        if self.flags.tiered_exits and pos.tp_idx < len(pos.tp_schedule):
            tp_r, tp_pct = pos.tp_schedule[pos.tp_idx]
            if r >= tp_r:
                qty_close = max(1, int(pos.qty * tp_pct))
                qty_close = min(qty_close, pos.remaining_qty)
                self._submit_market_exit(qty_close, bar_time, f"tp{pos.tp_idx + 1}")

        # Stale exit
        if self.flags.stale_exit:
            bars = pos.hold_bars_1h
            if pos.engine_tag == EngineTag.BREAKDOWN:
                bars = pos.hold_bars_30m
            elif pos.engine_tag == EngineTag.REVERSAL:
                bars = pos.hold_bars_4h
            if check_stale_exit(pos.engine_tag, bars, r, self.po):
                # Tag profitable stale exits as tp0 when flag enabled
                exit_tag = "stale"
                if self.flags.stale_to_tp and r > 0:
                    exit_tag = "tp0"
                self._submit_market_exit(pos.remaining_qty, bar_time, exit_tag)
                return

        # Climax exit
        if self.flags.climax_exit:
            if check_climax_exit(Cl, self._ema20_1h, self._atr_1h, r, self.po):
                self._submit_market_exit(pos.remaining_qty, bar_time, "climax")
                return

        # VWAP failure exit (Fade only)
        if (self.flags.vwap_failure_exit
                and pos.engine_tag == EngineTag.FADE):
            if check_vwap_failure_exit(self._fade.consecutive_above_vwap, r):
                self._submit_market_exit(pos.remaining_qty, bar_time, "vwap_failure")
                return

        # Catastrophic exit
        if check_catastrophic_exit(r):
            self._submit_market_exit(pos.remaining_qty, bar_time, "catastrophic")
            return

    # -------------------------------------------------------------------
    # Order helpers
    # -------------------------------------------------------------------

    def _submit_protective_stop(self, stop_price: float, qty: int, bar_time: datetime) -> None:
        oid = self.broker.next_order_id()
        order = SimOrder(
            order_id=oid, symbol=self.symbol, side=OrderSide.BUY,
            order_type=OrderType.STOP, qty=qty,
            stop_price=stop_price, tick_size=self.config.tick_size,
            submit_time=bar_time, ttl_hours=0, tag="protective_stop",
        )
        self.broker.submit_order(order)
        self._replay_core_step(
            order_updates=[
                DownturnOrderUpdate(
                    oms_order_id=oid,
                    status="accepted",
                    timestamp=bar_time,
                    order_role="stop",
                )
            ]
        )

    def _update_protective_stop(self, new_stop: float, qty: int, bar_time: datetime) -> None:
        self._replay_core_step(
            bar_input={
                "bar_count_5m": self._core_state.bar_count_5m,
                "bar_ts": bar_time,
                "stop_update": DownturnStopUpdateRequest(
                    stop_price=new_stop,
                    qty=qty,
                    reason=self._position.exit_trigger if self._position is not None else "stop_update",
                ),
            }
        )
        self.broker.cancel_orders(self.symbol, tag="protective_stop")
        self._submit_protective_stop(new_stop, qty, bar_time)

    def _submit_market_exit(self, qty: int, bar_time: datetime, tag: str) -> None:
        self._replay_core_step(
            bar_input={
                "bar_count_5m": self._core_state.bar_count_5m,
                "bar_ts": bar_time,
                "flatten_reason": tag,
            }
        )
        oid = self.broker.next_order_id()
        order = SimOrder(
            order_id=oid, symbol=self.symbol, side=OrderSide.BUY,
            order_type=OrderType.MARKET, qty=qty,
            tick_size=self.config.tick_size,
            submit_time=bar_time, ttl_hours=1, tag=tag,
        )
        self.broker.submit_order(order)

    def _force_close(self, close: float, bar_idx: int, correction_windows: list) -> None:
        """Force close at end of backtest."""
        if self._position is None:
            return
        pos = self._position
        self._replay_core_step(
            fills=[
                DownturnFill(
                    oms_order_id="force_close",
                    fill_price=close,
                    fill_qty=pos.qty,
                    fill_time=None,
                    exit_type="eob",
                )
            ]
        )
        pnl = (pos.entry_price - close) * pos.qty * self.config.point_value - pos.commission
        self._realized_pnl += pnl
        r_mult = pos.r_state(close)
        self._trades.append(DownturnTradeRecord(
            symbol=self.symbol, direction=Direction.SHORT,
            entry_price=pos.entry_price, exit_price=close,
            entry_time=pos.entry_time, exit_time=None,
            qty=pos.qty, pnl=pnl, r_multiple=r_mult,
            stop0=pos.stop0, commission=pos.commission,
            exit_type="eob", hold_bars=pos.hold_bars_1h, hold_bars_5m=pos.hold_bars_5m,
            mfe=pos.r_at_peak, engine_tag=pos.engine_tag,
            composite_regime_at_entry=pos.composite_regime,
            vol_state_at_entry=pos.vol_state,
            in_correction_window=pos.in_correction,
            predator_present=pos.predator,
            signal_class=pos.signal_class,
        ))
        self.broker.cancel_all(self.symbol)
        self._position = None

    # -------------------------------------------------------------------
    # Session classification  (Spec §1.1)
    # -------------------------------------------------------------------

    def _classify_session(self, bar_time: datetime) -> tuple[bool, float]:
        """Classify session for entry windows and sizing.

        Returns (can_enter, session_size_mult).
        """
        try:
            et = bar_time.astimezone(ET)
        except Exception:
            return True, 1.0

        h, m = et.hour, et.minute
        mins = h * 60 + m

        # Dead zones (no entry)
        if self.flags.use_dead_zones:
            # 09:25-09:35 ET (opening volatility)
            if 565 <= mins < 575:
                return False, 0.0
            # 15:50-16:00 ET (closing)
            if 950 <= mins < 960:
                return False, 0.0

        # Core RTH: 09:35-15:50 ET
        if 575 <= mins < 950:
            return True, 1.0

        # ETH morning: 04:00-09:25
        if 240 <= mins < 565:
            return True, 0.75

        # ETH evening: 18:00-20:00
        if 1080 <= mins < 1200:
            return True, 0.50

        return False, 0.0

    # -------------------------------------------------------------------
    # Correction window detection
    # -------------------------------------------------------------------

    def _compute_correction_windows(self, daily: NumpyBars) -> list[CorrectionWindow]:
        """Identify NQ correction periods: >3% drop from 20-day rolling high."""
        windows = []
        n = len(daily.closes)
        if n < 20:
            return windows

        rolling_high = np.full(n, np.nan)
        for i in range(19, n):
            rolling_high[i] = np.max(daily.highs[i - 19:i + 1])

        in_correction = False
        start_idx = 0
        peak = 0.0
        trough = float("inf")

        for i in range(20, n):
            rh = rolling_high[i]
            close = daily.closes[i]
            if rh > 0 and (rh - close) / rh >= 0.03:
                if not in_correction:
                    in_correction = True
                    start_idx = i
                    peak = rh
                    trough = close
                else:
                    trough = min(trough, close)
            else:
                if in_correction:
                    pct = (peak - trough) / peak * 100 if peak > 0 else 0
                    start_time = self._np_to_datetime(daily.times[start_idx])
                    end_time = self._np_to_datetime(daily.times[i - 1])
                    if start_time and end_time:
                        windows.append(CorrectionWindow(start_time, end_time, pct))
                    in_correction = False

        # Handle ongoing correction at end
        if in_correction:
            pct = (peak - trough) / peak * 100 if peak > 0 else 0
            start_time = self._np_to_datetime(daily.times[start_idx])
            end_time = self._np_to_datetime(daily.times[-1])
            if start_time and end_time:
                windows.append(CorrectionWindow(start_time, end_time, pct))

        return windows

    def _in_correction(self, bar_time: datetime, windows: list[CorrectionWindow]) -> bool:
        """Check if bar_time falls within any correction window."""
        for w in windows:
            if w.start_date <= bar_time <= w.end_date:
                return True
        return False

    @staticmethod
    def _np_to_datetime(ts) -> datetime | None:
        if isinstance(ts, np.datetime64):
            epoch = ts.astype("datetime64[s]").astype("int64")
            return datetime.utcfromtimestamp(epoch).replace(tzinfo=timezone.utc)
        return ts if isinstance(ts, datetime) else None

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _get_counter(self, tag: EngineTag) -> EngineCounters:
        if tag == EngineTag.REVERSAL:
            return self._reversal_ctr
        elif tag == EngineTag.BREAKDOWN:
            return self._breakdown_ctr
        return self._fade_ctr

    def _record_signal(
        self, tag: EngineTag, sig_class: str, bar_time: datetime, entered: bool,
    ) -> None:
        if self.config.track_signals:
            self._signals.append(DownturnSignalEvent(
                engine_tag=tag,
                direction=Direction.SHORT,
                signal_class=sig_class,
                regime_at_signal=self._regime.composite_regime,
                timestamp=bar_time,
                entered=entered,
            ))
