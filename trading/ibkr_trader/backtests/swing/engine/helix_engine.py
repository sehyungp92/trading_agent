"""Core single-symbol bar-by-bar Helix backtesting engine.

Translates strategy_2/engine.py's async event-driven architecture into a
synchronous bar-by-bar loop.  Uses SimBroker for fill simulation (shared
with the ATRSS engine).

All strategy logic is called via the pure functions in strategy_2/*.py.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.legacy_result_outputs import trade_outcomes_from_records
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from libs.broker_ibkr.risk_support.tick_rules import round_to_tick

from strategies.swing.akc_helix import signals, stops
from strategies.swing.akc_helix.allocator import (
    apply_initial_risk_basis,
    compute_position_size,
    compute_risk_r,
    compute_unit1_risk,
)
from strategies.swing.akc_helix.circuit import roll_circuit_breaker_window
from strategies.swing.akc_helix.config import (
    ADD_1H_R,
    ADD_4H_R,
    ADD_OVERNIGHT_R,
    ADD_RISK_FRAC,
    ADX_UPPER_GATE,
    BE_ATR1H_OFFSET,
    CLASS_B_BAIL_BARS,
    CLASS_B_BAIL_R_THRESH,
    CLASS_B_MIN_ADX,
    CLASS_B_MIN_PIVOT_SEP_BARS,
    CLASS_B_MOM_LOOKBACK,
    CLASS_C_MIN_HOLD_BARS,
    CLASS_D_BAIL_BARS,
    CLASS_D_BAIL_R_THRESH,
    CLASS_D_HIST_SIGN_GATE,
    CLASS_D_MIN_ADX,
    CLASS_D_REGIME_STREAK_MIN,
    CLASS_D_SHORT_MIN_ADX,
    CONSEC_STOPS_HALVE,
    DAILY_STOP_R,
    EARLY_STALE_BARS,
    EMA_4H_FAST,
    EMA_4H_SLOW,
    EMERGENCY_STOP_R,
    PARTIAL_2P5_FRAC,
    PARTIAL_5_FRAC,
    PARTIAL_5_TRAIL_BONUS,
    R_BE,
    R_BE_1H,
    R_PARTIAL_2P5,
    R_PARTIAL_5,
    RTS_FAIL_FLATTEN_R,
    RTS_GUARD_FADE_BARS,
    RTS_GUARD_FLOOR_R,
    RTS_GUARD_MAX_MFE_R,
    RTS_GUARD_MFE_R,
    RTS_GUARD_MIN_BARS,
    RTS_GUARD_MIN_GIVEBACK_R,
    STALE_1H_BARS,
    STALE_4H_BARS,
    STALE_FLATTEN_R_FLOOR,
    STALE_R_THRESH,
    STOP_4H_MULT,
    R_BAND_HIGH,
    R_BAND_MID,
    TRAIL_BASE_CLASS_B,
    TRAIL_BASE_CLASS_D,
    TRAIL_BASE_HIGH_R,
    TRAIL_BASE_LOW_R,
    TRAIL_BASE_MID_R,
    TRAIL_FADE_FLOOR,
    TRAIL_FADE_MIN_R,
    TRAIL_FADE_MIN_R_CLASS_D,
    TRAIL_FADE_ONSET_BARS,
    TRAIL_FADE_PENALTY,
    TRAIL_FADE_PENALTY_CLASS_D,
    TRAIL_PROFIT_DELAY_BARS,
    TRAIL_R_DIV,
    TRAIL_R_DIV_CLASS_B,
    TRAIL_R_DIV_CLASS_D,
    TRAIL_R_DIV_HIGH_R,
    TRAIL_R_DIV_LOW_R,
    TRAIL_R_DIV_MID_R,
    TRAIL_MIN,
    TRAIL_STALL_FLOOR,
    TRAIL_STALL_ONSET,
    TRAIL_STALL_ONSET_CLASS_B,
    TRAIL_STALL_ONSET_CLASS_D,
    TRAIL_STALL_RATE,
    TRAIL_TIMEDECAY_FLOOR,
    TRAIL_TIMEDECAY_ONSET,
    TRAIL_TIMEDECAY_RATE,
    TTL_1H_HOURS,
    TTL_4H_HOURS,
    WEEKLY_STOP_R,
    SymbolConfig,
)
from strategies.swing.akc_helix.indicators import (
    atr,
    compute_daily_state,
    compute_regime_4h,
    ema,
    macd,
    scan_pivots,
)
from strategies.swing.akc_helix.core import logic as helix_core_logic
from strategies.swing.akc_helix.core.state import (
    AKCHelixCoreState,
    AKCHelixEntryRequest,
    AKCHelixFill,
    AKCHelixFlattenRequest,
    AKCHelixOrderUpdate,
    AKCHelixPartialExitRequest,
    AKCHelixStopUpdateRequest,
)
from strategies.swing.akc_helix.models import (
    CircuitBreakerState,
    DailyState,
    Direction,
    Pivot,
    PivotStore,
    Regime,
    SetupClass,
    SetupInstance,
    SetupState,
    TFState,
)

from backtests.swing.config import SlippageConfig
from backtests.swing.config_helix import HelixAblationFlags, HelixBacktestConfig
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


# ---------------------------------------------------------------------------
# Param override patching (matches ATRSS _AblationPatch pattern)
# ---------------------------------------------------------------------------

class _AblationPatch:
    """Temporarily patch strategy_2 module constants for param_overrides.

    Monkeypatches constants in strategy_2.config (and this engine module's
    own top-level bindings) within a context manager. Restores originals
    on exit. Safe in single-threaded / per-process execution.
    """

    def __init__(self, flags: HelixAblationFlags, param_overrides: dict[str, float] | None = None):
        self.flags = flags
        self.overrides = param_overrides or {}
        self._patches: list[tuple[object, str, object]] = []

    def __enter__(self):
        import sys
        import strategies.swing.akc_helix.config as scfg
        import strategies.swing.akc_helix.signals as ssig
        import strategies.swing.akc_helix.stops as sstops

        engine_mod = sys.modules[__name__]

        for key, val in self.overrides.items():
            upper_key = key.upper()
            # Patch source module
            if hasattr(scfg, upper_key):
                self._patch(scfg, upper_key, val)
            # Patch engine module's own top-level imported binding
            if hasattr(engine_mod, upper_key):
                self._patch(engine_mod, upper_key, val)
            # Patch signals module's cached bindings (Class D momentum lookback, div filters, stops).
            if hasattr(ssig, upper_key):
                self._patch(ssig, upper_key, val)
            # Patch stops module's cached bindings (TRAIL_BASE, R_BE, etc.)
            if hasattr(sstops, upper_key):
                self._patch(sstops, upper_key, val)
        return self

    def __exit__(self, *exc):
        for obj, attr, orig in reversed(self._patches):
            setattr(obj, attr, orig)
        self._patches.clear()

    def _patch(self, obj, attr: str, value):
        self._patches.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, value)


# ---------------------------------------------------------------------------
# Trade record for post-hoc analysis (Helix-specific fields)
# ---------------------------------------------------------------------------

@dataclass
class HelixTradeRecord:
    """One completed Helix trade (entry + exit)."""

    symbol: str = ""
    direction: int = 0
    setup_class: str = ""       # A, B, C, D
    origin_tf: str = ""         # 1H or 4H
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    entry_price: float = 0.0
    avg_entry_price: float = 0.0
    exit_price: float = 0.0
    qty: int = 0
    initial_stop: float = 0.0
    exit_reason: str = ""       # STOP, STALE, REGIME_FLIP, FLATTEN
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    r_multiple: float = 0.0
    net_pnl_dollars: float = 0.0
    net_r_multiple: float = 0.0
    base_unit1_risk_dollars: float = 0.0
    target_initial_risk_dollars: float = 0.0
    actual_initial_risk_dollars: float = 0.0
    risk_utilization: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars_held: int = 0
    commission: float = 0.0
    # Helix-specific
    qty_partial_1: int = 0      # qty exited at +2.5R
    qty_partial_2: int = 0      # qty exited at +5R
    add_on_qty: int = 0
    add_on_price: float = 0.0
    setup_size_mult: float = 1.0
    adx_at_entry: float = 0.0
    div_mag_norm: float = 0.0
    regime_4h_at_entry: str = ""
    regime_at_entry: str = ""


@dataclass
class HelixSymbolResult:
    """Result of backtesting a single symbol with Helix."""

    symbol: str
    trades: list[HelixTradeRecord] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    total_commission: float = 0.0
    # Diagnostic counters
    setups_detected: int = 0
    setups_armed: int = 0
    setups_filled: int = 0
    setups_expired: int = 0
    regime_days_bull: int = 0
    regime_days_bear: int = 0
    regime_days_chop: int = 0
    decision_stream: list[dict] = field(default_factory=list)
    trade_outcomes: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Active position tracking during backtest
# ---------------------------------------------------------------------------

@dataclass
class _ActivePosition:
    """Internal position tracking during the bar-by-bar loop."""

    setup: SetupInstance
    fill_price: float = 0.0
    avg_entry_price: float = 0.0
    qty_open: int = 0
    initial_stop: float = 0.0
    current_stop: float = 0.0
    r_price: float = 0.0        # |entry - stop0| for R calc
    entry_time: datetime | None = None
    bars_held_1h: int = 0
    bars_held_4h: int = 0
    mfe_price: float = 0.0
    mae_price: float = 0.0
    trail_active: bool = False
    partial_2p5_done: bool = False
    partial_5_done: bool = False
    trailing_mult_bonus: float = 0.0
    add_done: bool = False
    add_qty: int = 0
    add_fill_price: float = 0.0
    realized_pnl: float = 0.0
    regime_at_entry: str = ""
    stop_order_tag: str = ""
    qty_partial_1: int = 0      # qty exited at +2.5R
    qty_partial_2: int = 0      # qty exited at +5R
    commission: float = 0.0
    bars_at_r1: int = 0   # for trailing profit delay
    bars_neg_fading_hist: int = 0  # bars with negative AND declining histogram
    bar_of_max_mfe: int = 0       # bar count when peak MFE was set


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class HelixEngine:
    """Single-symbol bar-by-bar Helix backtest engine.

    Processes 1H bars sequentially, updating indicators and detecting
    setups on each bar.  Uses SimBroker for fill simulation.
    """

    def __init__(
        self,
        symbol: str,
        cfg: SymbolConfig,
        bt_config: HelixBacktestConfig,
        point_value: float = 1.0,
    ) -> None:
        self.symbol = symbol
        self.cfg = cfg
        self.bt_config = bt_config
        self.point_value = point_value
        self.flags = bt_config.flags

        # Use ETF commission rate when trading ETFs
        from dataclasses import replace as _replace
        slippage = bt_config.slippage
        if cfg.is_etf:
            slippage = _replace(slippage,
                                commission_per_contract=slippage.commission_per_share_etf)
        self.broker = SimBroker(slippage_config=slippage)
        self.equity = bt_config.initial_equity
        self.sizing_equity = bt_config.initial_equity

        # State
        self.daily_state: DailyState | None = None
        self.tf_1h = TFState(tf_label="1H")
        self.tf_4h = TFState(tf_label="4H")
        self.pivots_1h = PivotStore()
        self.pivots_4h = PivotStore()
        self.regime_4h: Regime = Regime.CHOP
        self.div_mag_history: list[float] = []
        self.circuit_breaker = CircuitBreakerState()
        # Pivot dedup: prevent re-detection with same L2/H2 pivot
        self._last_b_long_l2_ts: datetime | None = None
        self._last_b_short_h2_ts: datetime | None = None
        self._last_d_long_l2_ts: datetime | None = None
        self._last_d_short_h2_ts: datetime | None = None

        # Position tracking
        self.active_position: _ActivePosition | None = None
        self.pending_setup: SetupInstance | None = None
        self._pending_flatten_reason: str | None = None

        # Results
        self.trades: list[HelixTradeRecord] = []
        self.equity_curve: list[float] = []
        self.timestamps: list = []
        self.total_commission: float = 0.0

        # Diagnostic counters
        self.setups_detected: int = 0
        self.setups_armed: int = 0
        self.setups_filled: int = 0
        self.setups_expired: int = 0
        self.regime_days_bull: int = 0
        self.regime_days_bear: int = 0
        self.regime_days_chop: int = 0
        self._regime_streak: int = 0         # consecutive days in current regime
        self._prev_regime: Regime | None = None

        # Daily state index cache for shadow tracker
        self._daily_state_by_idx: dict[int, DailyState] = {}

        # Shadow tracker callback for filter attribution
        self.on_rejection: callable | None = None

        # Bar index tracking for 4H detection
        self._prev_4h_idx: int = -1

        # Core state for thin-driver event capture
        self._core_state = AKCHelixCoreState()
        self._decision_events: list = []

    def run(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        four_hour: NumpyBars,
        daily_idx_map: np.ndarray,
        four_hour_idx_map: np.ndarray,
    ) -> HelixSymbolResult:
        """Run the full backtest over the hourly bar series."""
        warmup_d = self.bt_config.warmup_daily
        warmup_h = self.bt_config.warmup_hourly

        with _AblationPatch(self.flags, self.bt_config.param_overrides):
            # Precompute full indicator arrays once (avoids recomputing
            # 200-bar windows on every bar)
            self._precompute_indicators(hourly, four_hour)

            for t in range(len(hourly)):
                self._step_bar(
                    daily, hourly, four_hour,
                    daily_idx_map, four_hour_idx_map, t,
                    warmup_d, warmup_h,
                )

            # Close any remaining position at last bar's close
            if self.active_position is not None:
                self._flatten_at_end_of_data(
                    hourly.closes[-1],
                    self._to_datetime(hourly.times[-1]),
                )

        return HelixSymbolResult(
            symbol=self.symbol,
            trades=self.trades,
            equity_curve=np.array(self.equity_curve),
            timestamps=np.array(self.timestamps),
            total_commission=self.total_commission,
            setups_detected=self.setups_detected,
            setups_armed=self.setups_armed,
            setups_filled=self.setups_filled,
            setups_expired=self.setups_expired,
            regime_days_bull=self.regime_days_bull,
            regime_days_bear=self.regime_days_bear,
            regime_days_chop=self.regime_days_chop,
            decision_stream=normalize_decision_stream(self._decision_events),
            trade_outcomes=trade_outcomes_from_records(self.trades),
        )

    # ------------------------------------------------------------------
    # Core replay driver (thin-driver event capture)
    # ------------------------------------------------------------------

    def _replay_core_step(self, *, bar_input=None, order_updates=None, fills=None):
        """Delegate to shared core logic for decision event capture."""
        result = run_replay(
            self._core_state,
            steps=[ReplayStep(bar_input=bar_input, order_updates=order_updates or [], fills=fills or [])],
            on_bar=lambda state, payload: helix_core_logic.on_bar(state, **payload),
            on_order_update=helix_core_logic.on_order_update,
            on_fill=helix_core_logic.on_fill,
        )
        self._core_state = result.state
        self._decision_events.extend(result.events)
        return result

    def _sync_core_stop(self, setup_id: str, stop_order_id: str) -> None:
        """Update core state stop tracking after placing a new stop order."""
        setup = self._core_state.active_setups.get(setup_id)
        if setup is not None:
            setup.stop_order_id = stop_order_id
        self._core_state.order_to_setup[stop_order_id] = setup_id

    # ------------------------------------------------------------------
    # Indicator precomputation
    # ------------------------------------------------------------------

    def _precompute_indicators(
        self,
        hourly: NumpyBars,
        four_hour: NumpyBars,
    ) -> None:
        """Precompute full-length indicator arrays for 1H and 4H.

        Avoids recomputing 200-bar ATR/MACD windows on every bar.
        Also precomputes datetime conversions (biggest bottleneck: 4ms/bar).
        """
        from strategies.swing.akc_helix.config import ATR_DAILY_PERIOD

        # 1H full arrays
        self._pre_1h_atr = atr(hourly.highs, hourly.lows, hourly.closes, ATR_DAILY_PERIOD)
        line, sig, hist = macd(hourly.closes)
        self._pre_1h_macd_line = line
        self._pre_1h_macd_sig = sig
        self._pre_1h_macd_hist = hist

        # 1H datetime conversions (eliminates 4ms/bar numpy→datetime overhead)
        self._pre_1h_datetimes = [self._to_datetime(hourly.times[i]) for i in range(len(hourly.times))]

        # 4H full arrays
        if len(four_hour.closes) > 0:
            self._pre_4h_atr = atr(four_hour.highs, four_hour.lows, four_hour.closes, ATR_DAILY_PERIOD)
            line4, sig4, hist4 = macd(four_hour.closes)
            self._pre_4h_macd_line = line4
            self._pre_4h_macd_sig = sig4
            self._pre_4h_macd_hist = hist4
            # EMA for regime
            if len(four_hour.closes) >= EMA_4H_SLOW:
                self._pre_4h_ema_fast = ema(four_hour.closes, EMA_4H_FAST)
                self._pre_4h_ema_slow = ema(four_hour.closes, EMA_4H_SLOW)
            else:
                self._pre_4h_ema_fast = None
                self._pre_4h_ema_slow = None
            # 4H datetime conversions
            self._pre_4h_datetimes = [self._to_datetime(four_hour.times[i]) for i in range(len(four_hour.times))]
        else:
            self._pre_4h_atr = np.array([])
            self._pre_4h_macd_line = np.array([])
            self._pre_4h_macd_sig = np.array([])
            self._pre_4h_macd_hist = np.array([])
            self._pre_4h_ema_fast = None
            self._pre_4h_ema_slow = None
            self._pre_4h_datetimes = []

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------

    def _step_bar(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        four_hour: NumpyBars,
        daily_idx_map: np.ndarray,
        four_hour_idx_map: np.ndarray,
        t: int,
        warmup_d: int,
        warmup_h: int,
    ) -> None:
        """Process one hourly bar."""
        # Use precomputed datetime if available (avoids 0.02ms numpy→datetime per call)
        bar_time = self._pre_1h_datetimes[t] if hasattr(self, '_pre_1h_datetimes') else self._to_datetime(hourly.times[t])
        O = hourly.opens[t]
        H = hourly.highs[t]
        L = hourly.lows[t]
        C = hourly.closes[t]

        # Skip NaN bars (gaps)
        if np.isnan(O) or np.isnan(C):
            self.equity_curve.append(self._mtm_equity(C if not np.isnan(C) else 0.0))
            self.timestamps.append(hourly.times[t])
            return

        # 1. Process fills from SimBroker
        fills = self.broker.process_bar(
            self.symbol, bar_time, O, H, L, C, self.cfg.tick_size,
        )
        for fill in fills:
            self._handle_fill(fill, bar_time, C)

        # 2. Update daily state on boundary
        d_idx = int(daily_idx_map[t])
        if d_idx >= warmup_d:
            if self.daily_state is None or (t > 0 and daily_idx_map[t] != daily_idx_map[t - 1]):
                end = d_idx + 1
                start = max(0, end - warmup_d)
                d_state = compute_daily_state(
                    daily.closes[start:end],
                    daily.highs[start:end],
                    daily.lows[start:end],
                    self.daily_state,
                )
                self.daily_state = d_state
                self._daily_state_by_idx[d_idx] = d_state

                # Track regime distribution and streak
                if d_state.regime == Regime.BULL:
                    self.regime_days_bull += 1
                elif d_state.regime == Regime.BEAR:
                    self.regime_days_bear += 1
                else:
                    self.regime_days_chop += 1
                if d_state.regime == self._prev_regime:
                    self._regime_streak += 1
                else:
                    self._regime_streak = 1
                    self._prev_regime = d_state.regime

        # 3. Update 4H state on boundary
        fh_idx = int(four_hour_idx_map[t])
        is_4h_boundary = fh_idx != self._prev_4h_idx
        self._prev_4h_idx = fh_idx

        if is_4h_boundary and fh_idx >= self.bt_config.warmup_4h:
            self._update_tf_state_4h(four_hour, fh_idx)

        # 4. Update 1H state
        if t >= warmup_h:
            self._update_tf_state_1h(hourly, t)

        # 5. Update active position tracking
        if self.active_position is not None:
            self._manage_active_position(bar_time, H, L, C, is_4h_boundary)

        # Skip signal detection during warmup
        if t < warmup_h or d_idx < warmup_d:
            self.equity_curve.append(self._mtm_equity(C))
            self.timestamps.append(hourly.times[t])
            return

        # 6. Manage pending setup (TTL, structure invalidation)
        if self.pending_setup is not None:
            self._manage_pending_setup(bar_time)

        # 7. Detect new setups (only if no active position and no pending)
        if self.active_position is None and self.pending_setup is None and self.daily_state is not None:
            self._detect_and_arm(bar_time, is_4h_boundary)

        # Record equity (mark-to-market)
        self.equity_curve.append(self._mtm_equity(C))
        self.timestamps.append(hourly.times[t])

    def _mtm_equity(self, current_price: float) -> float:
        """Return equity with unrealized P&L from open position."""
        if self.active_position is None or current_price == 0.0:
            return self.equity
        pos = self.active_position
        basis = self._position_cost_basis(pos)
        d = 1 if pos.setup.direction == Direction.LONG else -1
        return self.equity + (current_price - basis) * d * self.point_value * pos.qty_open

    def _position_cost_basis(self, pos: _ActivePosition) -> float:
        """Average entry for open quantity, including add-on fills."""
        return pos.avg_entry_price or pos.fill_price

    def _apply_add_fill_cost_basis(self, pos: _ActivePosition, fill_price: float, qty: int) -> None:
        """Fold an add-on fill into the open position using average-cost accounting."""
        if qty <= 0:
            return
        old_qty = max(int(pos.qty_open), 0)
        old_basis = self._position_cost_basis(pos)
        new_qty = old_qty + int(qty)
        pos.avg_entry_price = ((old_basis * old_qty) + (float(fill_price) * int(qty))) / new_qty
        pos.qty_open = new_qty

    def _cap_add_qty(self, pos: _ActivePosition, qty: int) -> int:
        """Respect the configured absolute position cap for pyramid add-ons."""
        qty = max(0, int(qty))
        max_contracts = int(getattr(self.cfg, "max_contracts", 0) or 0)
        if max_contracts <= 0:
            return qty
        remaining = max(0, max_contracts - int(pos.qty_open))
        return min(qty, remaining)

    def _cap_entry_qty_to_initial_risk(self, setup: SetupInstance, fill_price: float, qty: int) -> int:
        """Cap entry quantity using the actual fill-to-stop distance."""
        qty = max(0, int(qty))
        if qty <= 0 or not getattr(self.bt_config, "enforce_initial_risk_cap", True):
            return qty
        target = float(getattr(setup, "target_initial_risk_dollars", 0.0) or 0.0)
        if target <= 0.0:
            target = float(getattr(setup, "unit1_risk_dollars", 0.0) or 0.0)
        if target <= 0.0:
            return qty
        risk_per_unit = abs(float(fill_price) - float(getattr(setup, "stop0", fill_price))) * self.point_value
        if risk_per_unit <= 0.0:
            return 0
        buffer = max(0.0, float(getattr(self.bt_config, "initial_risk_cap_buffer", 0.0) or 0.0))
        cap_qty = int((target * (1.0 + buffer)) // risk_per_unit)
        max_contracts = int(getattr(self.cfg, "max_contracts", 0) or 0)
        if max_contracts > 0:
            cap_qty = min(cap_qty, max_contracts)
        return min(qty, max(0, cap_qty))

    def _net_trade_r(self, gross_pnl_dollars: float, commission: float, setup: SetupInstance) -> tuple[float, float]:
        """Return fee-net dollars and R for a closed trade."""
        net_pnl = float(gross_pnl_dollars) - float(commission)
        unit_risk = float(getattr(setup, "unit1_risk_dollars", 0.0) or 0.0)
        net_r = net_pnl / unit_risk if unit_risk > 0.0 else 0.0
        return net_pnl, net_r

    # ------------------------------------------------------------------
    # Indicator updates
    # ------------------------------------------------------------------

    def _update_tf_state_1h(self, hourly: NumpyBars, t: int) -> None:
        """Update 1H TFState and scan pivots using precomputed arrays."""
        lookback = min(t + 1, 200)
        start = t + 1 - lookback

        highs = hourly.highs[start:t + 1]
        lows = hourly.lows[start:t + 1]

        # Slice precomputed arrays instead of recomputing
        atr_slice = self._pre_1h_atr[start:t + 1]
        line_slice = self._pre_1h_macd_line[start:t + 1]
        hist_slice = self._pre_1h_macd_hist[start:t + 1]

        self.tf_1h.atr = float(atr_slice[-1])
        self.tf_1h.macd_line = float(line_slice[-1])
        self.tf_1h.macd_signal = float(self._pre_1h_macd_sig[t])
        self.tf_1h.macd_hist = float(hist_slice[-1])
        self.tf_1h.close = float(hourly.closes[t])
        self.tf_1h.bar_time = self._pre_1h_datetimes[t]
        self.tf_1h.macd_line_history = [float(v) for v in line_slice[-50:]]
        self.tf_1h.macd_hist_history = [float(v) for v in hist_slice[-50:]]

        chandelier_lb = max(self.cfg.chandelier_lookback, 30)
        self.tf_1h.highs = [float(v) for v in highs[-chandelier_lb:]]
        self.tf_1h.lows = [float(v) for v in lows[-chandelier_lb:]]

        # Check for pivot at current bar only (O(1) instead of scanning 200 bars)
        from strategies.swing.akc_helix.indicators import confirmed_pivot
        bar_times = self._pre_1h_datetimes[start:t + 1]
        local_idx = lookback - 1  # current bar index within the slice
        p = confirmed_pivot(highs, lows, local_idx, line_slice, hist_slice, atr_slice, bar_times)
        if p is not None:
            if p.kind.value == "H":
                last_ts = self.pivots_1h.highs[-1].ts if self.pivots_1h.highs else datetime.min
                if p.ts > last_ts:
                    self.pivots_1h.add(p)
            elif p.kind.value == "L":
                last_ts = self.pivots_1h.lows[-1].ts if self.pivots_1h.lows else datetime.min
                if p.ts > last_ts:
                    self.pivots_1h.add(p)

    def _update_tf_state_4h(self, four_hour: NumpyBars, fh_idx: int) -> None:
        """Update 4H TFState, scan pivots, and compute 4H regime (v2.0)."""
        lookback = min(fh_idx + 1, 200)
        start = fh_idx + 1 - lookback

        highs = four_hour.highs[start:fh_idx + 1]
        lows = four_hour.lows[start:fh_idx + 1]

        # Slice precomputed arrays instead of recomputing
        atr_slice = self._pre_4h_atr[start:fh_idx + 1]
        line_slice = self._pre_4h_macd_line[start:fh_idx + 1]
        sig_slice = self._pre_4h_macd_sig[start:fh_idx + 1]
        hist_slice = self._pre_4h_macd_hist[start:fh_idx + 1]

        self.tf_4h.atr = float(atr_slice[-1])
        self.tf_4h.macd_line = float(line_slice[-1])
        self.tf_4h.macd_signal = float(self._pre_4h_macd_sig[fh_idx])
        self.tf_4h.macd_hist = float(hist_slice[-1])
        self.tf_4h.close = float(four_hour.closes[fh_idx])
        self.tf_4h.bar_time = self._pre_4h_datetimes[fh_idx]
        self.tf_4h.macd_line_history = [float(v) for v in line_slice[-50:]]
        self.tf_4h.macd_hist_history = [float(v) for v in hist_slice[-50:]]

        chandelier_lb = max(self.cfg.chandelier_lookback, 30)
        self.tf_4h.highs = [float(v) for v in highs[-chandelier_lb:]]
        self.tf_4h.lows = [float(v) for v in lows[-chandelier_lb:]]

        # Compute 4H regime using precomputed EMAs
        if self._pre_4h_ema_fast is not None and fh_idx < len(self._pre_4h_ema_fast):
            self.regime_4h = compute_regime_4h(
                float(four_hour.closes[fh_idx]),
                float(self._pre_4h_ema_fast[fh_idx]),
                float(self._pre_4h_ema_slow[fh_idx]),
            )

        # Check for pivot at current bar only (O(1) instead of scanning 200 bars)
        from strategies.swing.akc_helix.indicators import confirmed_pivot
        bar_times = self._pre_4h_datetimes[start:fh_idx + 1]
        local_idx = lookback - 1
        p = confirmed_pivot(highs, lows, local_idx, line_slice, hist_slice, atr_slice, bar_times)
        if p is not None:
            if p.kind.value == "H":
                last_ts = self.pivots_4h.highs[-1].ts if self.pivots_4h.highs else datetime.min
                if p.ts > last_ts:
                    self.pivots_4h.add(p)
            elif p.kind.value == "L":
                last_ts = self.pivots_4h.lows[-1].ts if self.pivots_4h.lows else datetime.min
                if p.ts > last_ts:
                    self.pivots_4h.add(p)

    # ------------------------------------------------------------------
    # Setup detection & arming
    # ------------------------------------------------------------------

    def _record_rejection(self, setup, filter_name: str, bar_time: datetime) -> None:
        """Record a gate rejection for shadow tracking."""
        if self.on_rejection is not None:
            cls = setup.setup_class
            cls_str = cls.name if hasattr(cls, 'name') else str(cls)
            self.on_rejection(
                symbol=self.symbol,
                direction=int(setup.direction),
                filter_names=[filter_name],
                time=bar_time,
                entry_price=setup.bos_level,
                stop_price=setup.stop0,
                origin_tf=setup.origin_tf,
                setup_class=cls_str,
            )

    def _detect_and_arm(self, bar_time: datetime, is_4h_boundary: bool) -> None:
        """Detect new setups and arm the highest-priority one (A > C > B > D)."""
        daily = self.daily_state
        if daily is None:
            return

        # Circuit breaker check
        if not self.flags.disable_circuit_breaker:
            cb = roll_circuit_breaker_window(self.circuit_breaker, bar_time)
            self.circuit_breaker = cb
            if cb.paused_until and bar_time < cb.paused_until:
                return

        # Corridor inversion: disable 4H if min_stop_floor > corridor_cap
        _4h_disabled = False
        if daily.atr_d > 0:
            cap_dollars = 1.4 * daily.atr_d * self.point_value
            if self.cfg.min_stop_floor_dollars > cap_dollars:
                _4h_disabled = True

        # ADX upper gate: skip all setups when ADX overextended (999 = disabled)
        if ADX_UPPER_GATE < 999 and daily.adx > ADX_UPPER_GATE:
            return

        # Extreme vol: disable 1H-origin classes (B, D) when vol_pct > 95th
        from strategies.swing.akc_helix.config import EXTREME_VOL_PCT
        _1h_disabled = daily.vol_pct > EXTREME_VOL_PCT

        candidates: list[SetupInstance] = []

        # Priority order: A > C > B > D (spec s10.6)

        # Class A: 4H hidden divergence continuation (only on 4H boundary)
        if is_4h_boundary and not _4h_disabled and not self.flags.disable_class_a:
            setup_4h = signals.detect_class_a(
                self.symbol, self.pivots_4h, daily, self.tf_4h,
                self.cfg, self.div_mag_history, bar_time,
            )
            if setup_4h is not None:
                self.div_mag_history.append(setup_4h.div_mag_norm)
                candidates.append(setup_4h)

        # Class C: 4H classic divergence reversal (only on 4H boundary, gated)
        if not candidates and is_4h_boundary and not _4h_disabled and not self.flags.disable_class_c:
            setup_c = signals.detect_class_c(
                self.symbol, self.pivots_4h, daily, self.tf_4h,
                self.cfg, self.div_mag_history, bar_time,
            )
            if setup_c is not None:
                self.div_mag_history.append(setup_c.div_mag_norm)
                candidates.append(setup_c)

        # Class B: 1H hidden divergence continuation (every bar, if no 4H candidate)
        if not candidates and not _1h_disabled and not self.flags.disable_class_b:
            setup_b = signals.detect_class_b(
                self.symbol, self.pivots_1h, daily, self.tf_1h,
                self.cfg, self.div_mag_history, bar_time,
            )
            if setup_b is not None:
                # Quality filter: reject Class B in CHOP, counter-trend, or low ADX
                _b_rejected = False
                if daily.regime == Regime.CHOP:
                    _b_rejected = True
                elif setup_b.direction == Direction.LONG and daily.regime == Regime.BEAR:
                    _b_rejected = True
                elif setup_b.direction == Direction.SHORT and daily.regime == Regime.BULL:
                    _b_rejected = True
                elif daily.adx < CLASS_B_MIN_ADX:
                    _b_rejected = True

                # Momentum gate: MACD line must trend in trade direction
                if not _b_rejected:
                    ml_hist = self.tf_1h.macd_line_history
                    if len(ml_hist) >= CLASS_B_MOM_LOOKBACK + 1:
                        if setup_b.direction == Direction.LONG:
                            if self.tf_1h.macd_line <= ml_hist[-1 - CLASS_B_MOM_LOOKBACK]:
                                _b_rejected = True
                        else:  # SHORT
                            if self.tf_1h.macd_line >= ml_hist[-1 - CLASS_B_MOM_LOOKBACK]:
                                _b_rejected = True
                    else:
                        _b_rejected = True

                if _b_rejected:
                    if self.on_rejection is not None:
                        cls = setup_b.setup_class
                        cls_str = cls.name if hasattr(cls, 'name') else str(cls)
                        reasons = []
                        if daily.regime == Regime.CHOP:
                            reasons.append("class_b_chop")
                        elif (setup_b.direction == Direction.LONG and daily.regime == Regime.BEAR) or \
                             (setup_b.direction == Direction.SHORT and daily.regime == Regime.BULL):
                            reasons.append("class_b_counter_trend")
                        elif daily.adx < CLASS_B_MIN_ADX:
                            reasons.append("class_b_low_adx")
                        else:
                            reasons.append("class_b_momentum_gate")
                        self.on_rejection(
                            symbol=self.symbol, direction=int(setup_b.direction),
                            filter_names=reasons, time=bar_time,
                            entry_price=setup_b.bos_level, stop_price=setup_b.stop0,
                            origin_tf=setup_b.origin_tf, setup_class=cls_str,
                        )
                    setup_b = None
            if setup_b is not None:
                # Pivot dedup: skip if same L2/H2 as last B detection
                p2_ts = setup_b.pivot_2.ts if setup_b.pivot_2 else None
                if setup_b.direction == Direction.LONG:
                    if p2_ts != self._last_b_long_l2_ts:
                        self._last_b_long_l2_ts = p2_ts
                        candidates.append(setup_b)
                else:
                    if p2_ts != self._last_b_short_h2_ts:
                        self._last_b_short_h2_ts = p2_ts
                        candidates.append(setup_b)

        # Class D: 1H momentum continuation (every bar, if no higher-priority candidate)
        if not candidates and not _1h_disabled and not self.flags.disable_class_d:
            setup_1h = signals.detect_class_d(
                self.symbol, self.pivots_1h, daily, self.tf_1h,
                self.cfg, bar_time,
            )
            if setup_1h is not None:
                d_rejections: list[str] = []
                if CLASS_D_MIN_ADX > 0 and daily.adx < CLASS_D_MIN_ADX:
                    d_rejections.append("class_d_low_adx")
                if (
                    setup_1h.direction == Direction.SHORT
                    and CLASS_D_SHORT_MIN_ADX > 0
                    and daily.adx < CLASS_D_SHORT_MIN_ADX
                ):
                    d_rejections.append("class_d_short_low_adx")
                if CLASS_D_HIST_SIGN_GATE:
                    hist = self.tf_1h.macd_hist
                    if setup_1h.direction == Direction.LONG and hist <= 0:
                        d_rejections.append("class_d_hist_sign")
                    elif setup_1h.direction == Direction.SHORT and hist >= 0:
                        d_rejections.append("class_d_hist_sign")
                if (
                    CLASS_D_REGIME_STREAK_MIN > 0
                    and self._regime_streak < CLASS_D_REGIME_STREAK_MIN
                ):
                    d_rejections.append("class_d_regime_streak")
                if d_rejections:
                    self._record_rejection(setup_1h, d_rejections[0], bar_time)
                    setup_1h = None

            if setup_1h is not None:
                # Pivot dedup: skip if same L2/H2 as last D detection
                p2_ts = setup_1h.pivot_2.ts if setup_1h.pivot_2 else None
                if setup_1h.direction == Direction.LONG:
                    if p2_ts != self._last_d_long_l2_ts:
                        self._last_d_long_l2_ts = p2_ts
                        candidates.append(setup_1h)
                else:
                    if p2_ts != self._last_d_short_h2_ts:
                        self._last_d_short_h2_ts = p2_ts
                        candidates.append(setup_1h)

        if not candidates:
            return

        # Take first (highest priority: 4H > 1H)
        setup = candidates[0]

        # ── USO-specific gates ──
        # (a) Block counter-regime entries on USO
        if self.symbol == "USO" and daily:
            is_counter = (
                (setup.direction == Direction.LONG and daily.regime == Regime.BEAR)
                or (setup.direction == Direction.SHORT and daily.regime == Regime.BULL)
            )
            if is_counter:
                self._record_rejection(setup, "uso_counter_regime", bar_time)
                return
        # (b) Disable Class C on USO (4 trades, 0% WR, -0.682 avg R)
        if self.symbol == "USO" and setup.setup_class == SetupClass.CLASS_C:
            self._record_rejection(setup, "uso_class_c", bar_time)
            return
        # (c) Regime stability gate: require 3+ consecutive regime days for
        #     counter-regime Class A entries (blocks whipsaw clusters)
        if (setup.setup_class == SetupClass.CLASS_A and daily
                and self._regime_streak < 3):
            is_counter_a = (
                (setup.direction == Direction.LONG and daily.regime == Regime.BEAR)
                or (setup.direction == Direction.SHORT and daily.regime == Regime.BULL)
            )
            if is_counter_a:
                self._record_rejection(setup, "regime_stability_class_a", bar_time)
                return

        self.setups_detected += 1

        # Compute unit1 risk and position size
        vf = daily.vol_factor
        unit1_risk = compute_unit1_risk(self.sizing_equity, self.cfg.base_risk_pct, vf)
        target_initial_risk = unit1_risk * setup.setup_size_mult
        setup.base_unit1_risk_dollars = unit1_risk
        setup.target_initial_risk_dollars = target_initial_risk
        setup.actual_initial_risk_dollars = 0.0
        setup.risk_utilization = 0.0
        setup.unit1_risk_dollars = target_initial_risk if target_initial_risk > 0 else unit1_risk
        setup.vol_factor_at_placement = vf

        if self.bt_config.fixed_qty is not None:
            setup.qty_planned = self.bt_config.fixed_qty
        else:
            setup.qty_planned = compute_position_size(
                setup.bos_level, setup.stop0, unit1_risk,
                setup.setup_size_mult, self.point_value,
                self.cfg.max_contracts,
            )

        if setup.qty_planned <= 0:
            self._record_rejection(setup, "qty_zero", bar_time)
            return

        # Circuit breaker halving
        if not self.flags.disable_circuit_breaker:
            cb = roll_circuit_breaker_window(self.circuit_breaker, bar_time)
            self.circuit_breaker = cb
            if cb.halved_until and bar_time < cb.halved_until:
                setup.qty_planned = max(1, setup.qty_planned // 2)

        # Corridor cap check (unless disabled)
        if not self.flags.disable_corridor_cap:
            entry_to_stop = abs(setup.bos_level - setup.stop0)
            cap_mult = signals._corridor_cap_mult(daily, setup.direction)
            corridor_cap = cap_mult * daily.atr_d
            if corridor_cap > 0 and entry_to_stop > corridor_cap:
                self._record_rejection(setup, "corridor_cap", bar_time)
                return

        # R price
        setup.r_price = abs(setup.bos_level - setup.stop0)
        if setup.r_price <= 0:
            self._record_rejection(setup, "r_price_zero", bar_time)
            return

        # Minimum stop distance: 0.3% of entry price.
        # Prevents tiny stops that create massive position sizes and gap risk.
        min_r_price = 0.003 * setup.bos_level
        if setup.r_price < min_r_price:
            self._record_rejection(setup, "min_stop_distance", bar_time)
            return

        # TTL
        if setup.origin_tf == "4H":
            setup.expiry_ts = bar_time + timedelta(hours=TTL_4H_HOURS)
            ttl_hours = TTL_4H_HOURS
        else:
            setup.expiry_ts = bar_time + timedelta(hours=TTL_1H_HOURS)
            ttl_hours = TTL_1H_HOURS

        # Place entry order via SimBroker
        tick = self.cfg.tick_size
        if setup.direction == Direction.LONG:
            trigger = round_to_tick(setup.bos_level, tick, "up")
            side = OrderSide.BUY
        else:
            trigger = round_to_tick(setup.bos_level, tick, "down")
            side = OrderSide.SELL

        # ETF: stop-market, Futures: stop-limit
        if self.cfg.is_etf:
            order = SimOrder(
                order_id=self.broker.next_order_id(),
                symbol=self.symbol,
                side=side,
                order_type=OrderType.STOP,
                qty=setup.qty_planned,
                stop_price=trigger,
                tick_size=tick,
                submit_time=bar_time,
                ttl_hours=ttl_hours,
                tag="entry",
            )
        else:
            from strategies.swing.akc_helix.config import HIGH_VOL_PCT
            if daily.vol_pct > HIGH_VOL_PCT:
                offset_ticks = self.cfg.offset_wide_ticks
            else:
                offset_ticks = self.cfg.offset_tight_ticks
            limit_offset = offset_ticks * tick
            if setup.direction == Direction.LONG:
                limit_price = trigger + limit_offset
            else:
                limit_price = trigger - limit_offset
            order = SimOrder(
                order_id=self.broker.next_order_id(),
                symbol=self.symbol,
                side=side,
                order_type=OrderType.STOP_LIMIT,
                qty=setup.qty_planned,
                stop_price=trigger,
                limit_price=limit_price,
                tick_size=tick,
                submit_time=bar_time,
                ttl_hours=ttl_hours,
                tag="entry",
            )

        self.broker.submit_order(order)
        setup.state = SetupState.ARMED
        setup.armed_ts = bar_time
        self.pending_setup = setup
        self.setups_armed += 1

        # Core notification: entry requested
        entry_req = AKCHelixEntryRequest(
            client_order_id=order.order_id,
            setup=setup,
            order_role="entry",
            qty=order.qty,
        )
        self._replay_core_step(bar_input={"bar_ts": bar_time, "entry_request": entry_req})

    # ------------------------------------------------------------------
    # Pending setup management
    # ------------------------------------------------------------------

    def _manage_pending_setup(self, bar_time: datetime) -> None:
        """Check TTL expiry and structure invalidation of pending setup."""
        setup = self.pending_setup
        if setup is None:
            return

        # TTL expiry (broker handles order TTL, but we track setup-level)
        if setup.expiry_ts and bar_time >= setup.expiry_ts:
            self.broker.cancel_orders(self.symbol, tag="entry")
            self.pending_setup = None
            self.setups_expired += 1
            return

        # Structure invalidation
        tf_key = "4H" if setup.origin_tf == "4H" else "1H"
        pivots = self.pivots_4h if tf_key == "4H" else self.pivots_1h
        if signals.is_structure_invalidated(setup, pivots):
            self.broker.cancel_orders(self.symbol, tag="entry")
            self.pending_setup = None
            return

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    def _handle_fill(self, fill: FillResult, bar_time: datetime, close: float) -> None:
        """Route a fill result to the appropriate handler."""
        if fill.status in (FillStatus.EXPIRED, FillStatus.REJECTED, FillStatus.CANCELLED):
            if fill.order.tag == "entry":
                self.pending_setup = None
                if fill.status == FillStatus.EXPIRED:
                    self.setups_expired += 1
            _role_map = {"entry": "entry", "add_on": "add", "partial": "partial",
                         "protective_stop": "stop", "flatten": "flatten"}
            self._replay_core_step(order_updates=[AKCHelixOrderUpdate(
                oms_order_id=fill.order.order_id, status=fill.status.name.lower(),
                symbol=self.symbol, timestamp=bar_time,
                order_role=_role_map.get(fill.order.tag, "unknown"),
            )])
            return

        if fill.status != FillStatus.FILLED:
            return

        if fill.order.tag == "entry":
            self._on_entry_fill(fill, bar_time)
        elif fill.order.tag == "protective_stop":
            self._on_stop_fill(fill, bar_time)
        elif fill.order.tag == "partial":
            self._on_partial_fill(fill, bar_time)
        elif fill.order.tag == "add_on":
            self._on_add_fill(fill, bar_time)
        elif fill.order.tag == "flatten":
            self._on_flatten_fill(fill, bar_time)

    def _on_entry_fill(self, fill: FillResult, bar_time: datetime) -> None:
        """Handle entry fill: create position, place protective stop."""
        setup = self.pending_setup
        if setup is None:
            return

        self.pending_setup = None
        accepted_qty = self._cap_entry_qty_to_initial_risk(setup, fill.fill_price, fill.order.qty)
        if accepted_qty <= 0:
            self._record_rejection(setup, "entry_initial_risk_cap", bar_time)
            self.broker.cancel_all(self.symbol)
            return
        if accepted_qty != fill.order.qty:
            fill.order.qty = accepted_qty
            fill.commission = self.broker._compute_commission(accepted_qty)
            setup.qty_planned = accepted_qty

        self.setups_filled += 1
        self.total_commission += fill.commission
        self.equity -= fill.commission

        actual_r_price = abs(fill.fill_price - setup.stop0)
        if actual_r_price > 0:
            setup.r_price = actual_r_price
        apply_initial_risk_basis(
            setup,
            fill.fill_price,
            fill.order.qty,
            self.point_value,
            setup.target_initial_risk_dollars,
        )

        pos = _ActivePosition(
            setup=setup,
            fill_price=fill.fill_price,
            avg_entry_price=fill.fill_price,
            qty_open=fill.order.qty,
            initial_stop=setup.stop0,
            current_stop=setup.stop0,
            r_price=setup.r_price,
            entry_time=bar_time,
            mfe_price=fill.fill_price,
            mae_price=fill.fill_price,
            regime_at_entry=self.daily_state.regime.value if self.daily_state else "",
            commission=fill.commission,
        )
        self.active_position = pos

        # Place protective stop
        stop_side = OrderSide.SELL if setup.direction == Direction.LONG else OrderSide.BUY
        stop_order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=stop_side,
            order_type=OrderType.STOP,
            qty=fill.order.qty,
            stop_price=setup.stop0,
            tick_size=self.cfg.tick_size,
            submit_time=bar_time,
            ttl_hours=0,  # No TTL for protective stops
            tag="protective_stop",
        )
        self.broker.submit_order(stop_order)
        pos.stop_order_tag = "protective_stop"

        # Core notification: entry fill + stop registration
        self._replay_core_step(fills=[AKCHelixFill(
            oms_order_id=fill.order.order_id, fill_price=fill.fill_price,
            fill_qty=fill.order.qty, point_value=self.point_value, symbol=self.symbol,
            fill_time=bar_time, commission=fill.commission,
            order_role="entry",
        )])
        self._sync_core_stop(setup.setup_id, stop_order.order_id)

    def _on_stop_fill(self, fill: FillResult, bar_time: datetime) -> None:
        """Handle protective stop fill."""
        pos = self.active_position
        if pos is None:
            return

        self.total_commission += fill.commission
        self.equity -= fill.commission

        # Compute R and PnL
        setup = pos.setup
        pv = self.point_value
        cost_basis = self._position_cost_basis(pos)
        if setup.direction == Direction.LONG:
            pnl_points = fill.fill_price - cost_basis
        else:
            pnl_points = cost_basis - fill.fill_price

        pnl_dollars = pnl_points * pv * pos.qty_open + pos.realized_pnl
        r_multiple = pnl_dollars / setup.unit1_risk_dollars if setup.unit1_risk_dollars > 0 else 0.0
        trade_commission = pos.commission + fill.commission
        net_pnl_dollars, net_r_multiple = self._net_trade_r(pnl_dollars, trade_commission, setup)

        # MFE/MAE in R
        r_base = pos.r_price
        if r_base > 0:
            if setup.direction == Direction.LONG:
                mfe_r = (pos.mfe_price - pos.fill_price) / r_base
                mae_r = (pos.fill_price - pos.mae_price) / r_base
            else:
                mfe_r = (pos.fill_price - pos.mfe_price) / r_base
                mae_r = (pos.mae_price - pos.fill_price) / r_base
        else:
            mfe_r = mae_r = 0.0

        self.equity += pnl_points * pv * pos.qty_open

        trade = HelixTradeRecord(
            symbol=self.symbol,
            direction=setup.direction,
            setup_class=setup.setup_class.value,
            origin_tf=setup.origin_tf,
            entry_time=pos.entry_time,
            exit_time=bar_time,
            entry_price=pos.fill_price,
            avg_entry_price=cost_basis,
            exit_price=fill.fill_price,
            qty=setup.qty_planned,
            initial_stop=pos.initial_stop,
            exit_reason="STOP",
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            r_multiple=r_multiple,
            net_pnl_dollars=net_pnl_dollars,
            net_r_multiple=net_r_multiple,
            base_unit1_risk_dollars=setup.base_unit1_risk_dollars,
            target_initial_risk_dollars=setup.target_initial_risk_dollars,
            actual_initial_risk_dollars=setup.actual_initial_risk_dollars,
            risk_utilization=setup.risk_utilization,
            mfe_r=mfe_r,
            mae_r=mae_r,
            bars_held=pos.bars_held_1h,
            commission=trade_commission,
            qty_partial_1=pos.qty_partial_1,
            qty_partial_2=pos.qty_partial_2,
            add_on_qty=pos.add_qty,
            add_on_price=pos.add_fill_price,
            setup_size_mult=setup.setup_size_mult,
            adx_at_entry=setup.adx_at_entry,
            div_mag_norm=setup.div_mag_norm,
            regime_4h_at_entry=setup.regime_4h_at_entry or "",
            regime_at_entry=pos.regime_at_entry,
        )
        self.trades.append(trade)

        # Core notification: stop fill
        self._replay_core_step(fills=[AKCHelixFill(
            oms_order_id=fill.order.order_id, fill_price=fill.fill_price,
            fill_qty=pos.qty_open, point_value=self.point_value, symbol=self.symbol,
            fill_time=bar_time, commission=fill.commission,
            order_role="stop", exit_type="STOP",
        )])

        # Update circuit breaker
        self._update_circuit_breaker(net_r_multiple, bar_time)

        # Cancel any remaining orders for this symbol
        self._pending_flatten_reason = None
        self.broker.cancel_all(self.symbol)
        self.active_position = None

    def _on_partial_fill(self, fill: FillResult, bar_time: datetime) -> None:
        """Handle partial exit fill."""
        pos = self.active_position
        if pos is None:
            return

        self.total_commission += fill.commission
        self.equity -= fill.commission
        pos.commission += fill.commission

        # Record realized PnL from partial
        pv = self.point_value
        cost_basis = self._position_cost_basis(pos)
        if pos.setup.direction == Direction.LONG:
            partial_pnl = (fill.fill_price - cost_basis) * pv * fill.order.qty
        else:
            partial_pnl = (cost_basis - fill.fill_price) * pv * fill.order.qty

        pos.realized_pnl += partial_pnl
        self.equity += partial_pnl
        pos.qty_open -= fill.order.qty

        # Update protective stop qty
        self.broker.cancel_orders(self.symbol, tag="protective_stop")
        if pos.qty_open > 0:
            stop_side = OrderSide.SELL if pos.setup.direction == Direction.LONG else OrderSide.BUY
            stop_order = SimOrder(
                order_id=self.broker.next_order_id(),
                symbol=self.symbol,
                side=stop_side,
                order_type=OrderType.STOP,
                qty=pos.qty_open,
                stop_price=pos.current_stop,
                tick_size=self.cfg.tick_size,
                submit_time=bar_time,
                ttl_hours=0,
                tag="protective_stop",
            )
            self.broker.submit_order(stop_order)

        # Core notification: partial fill + stop registration
        self._replay_core_step(fills=[AKCHelixFill(
            oms_order_id=fill.order.order_id, fill_price=fill.fill_price,
            fill_qty=fill.order.qty, point_value=self.point_value, symbol=self.symbol,
            fill_time=bar_time, commission=fill.commission,
            order_role="partial",
        )])
        if pos.qty_open > 0:
            self._sync_core_stop(pos.setup.setup_id, stop_order.order_id)

    def _on_add_fill(self, fill: FillResult, bar_time: datetime) -> None:
        """Handle add-on entry fill."""
        pos = self.active_position
        if pos is None:
            return

        self.total_commission += fill.commission
        self.equity -= fill.commission
        pos.commission += fill.commission
        self._apply_add_fill_cost_basis(pos, fill.fill_price, fill.order.qty)
        pos.add_qty = fill.order.qty
        pos.add_fill_price = fill.fill_price

        # Move stop to at least breakeven on add fill.
        # The add was placed because the trade was already profitable;
        # locking in BE on the original position means the add-on risk
        # is the ONLY remaining risk (~0.5R max).
        setup = pos.setup
        be_level = pos.fill_price + (BE_ATR1H_OFFSET * self.tf_1h.atr
                                     if setup.direction == Direction.LONG
                                     else -BE_ATR1H_OFFSET * self.tf_1h.atr)
        if setup.direction == Direction.LONG:
            new_stop = max(pos.current_stop, be_level)
        else:
            new_stop = min(pos.current_stop, be_level)
        pos.current_stop = new_stop

        # Update protective stop to include add-on qty
        self.broker.cancel_orders(self.symbol, tag="protective_stop")
        stop_side = OrderSide.SELL if setup.direction == Direction.LONG else OrderSide.BUY
        stop_order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=stop_side,
            order_type=OrderType.STOP,
            qty=pos.qty_open,
            stop_price=pos.current_stop,
            tick_size=self.cfg.tick_size,
            submit_time=bar_time,
            ttl_hours=0,
            tag="protective_stop",
        )
        self.broker.submit_order(stop_order)

        # Core notification: add fill + stop registration
        self._replay_core_step(fills=[AKCHelixFill(
            oms_order_id=fill.order.order_id, fill_price=fill.fill_price,
            fill_qty=fill.order.qty, point_value=self.point_value, symbol=self.symbol,
            fill_time=bar_time, commission=fill.commission,
            order_role="add",
        )])
        self._sync_core_stop(setup.setup_id, stop_order.order_id)

    # ------------------------------------------------------------------
    # Active position management (spec s13, s14, s15)
    # ------------------------------------------------------------------

    def _manage_active_position(
        self, bar_time: datetime, H: float, L: float, C: float,
        is_4h_boundary: bool,
    ) -> None:
        """Per-bar management: R calc, BE, partials, trailing, stale, adds."""
        pos = self.active_position
        if pos is None:
            return

        setup = pos.setup
        pos.bars_held_1h += 1
        if is_4h_boundary:
            pos.bars_held_4h += 1

        # Update MFE/MAE
        if setup.direction == Direction.LONG:
            if H > pos.mfe_price:
                pos.mfe_price = H
                pos.bar_of_max_mfe = pos.bars_held_1h
            if L < pos.mae_price:
                pos.mae_price = L
        else:
            if L < pos.mfe_price:
                pos.mfe_price = L
                pos.bar_of_max_mfe = pos.bars_held_1h
            if H > pos.mae_price:
                pos.mae_price = H

        # Compute current R
        r_base = pos.r_price
        if r_base <= 0:
            return

        if setup.direction == Direction.LONG:
            r_now = (C - pos.fill_price) / r_base
        else:
            r_now = (pos.fill_price - C) / r_base

        # R_state includes realized PnL from partials
        pv = self.point_value
        cost_basis = self._position_cost_basis(pos)
        if setup.direction == Direction.LONG:
            unrealized = (C - cost_basis) * pv * pos.qty_open
        else:
            unrealized = (cost_basis - C) * pv * pos.qty_open
        r_state = (pos.realized_pnl + unrealized) / setup.unit1_risk_dollars \
            if setup.unit1_risk_dollars > 0 else r_now

        # Catastrophic loss protection: flatten if loss exceeds emergency stop
        if r_now < EMERGENCY_STOP_R:
            self._submit_flatten(pos, bar_time, "STOP")
            return

        # Class B bail trigger: exit early if trade hasn't shown momentum
        if (setup.setup_class == SetupClass.CLASS_B
                and pos.bars_held_1h >= CLASS_B_BAIL_BARS
                and r_now < CLASS_B_BAIL_R_THRESH
                and not pos.trail_active):
            self._submit_flatten(pos, bar_time, "STALE")
            return

        # Class D bail trigger: exit early if momentum reverses (0 = disabled)
        if (CLASS_D_BAIL_BARS > 0
                and setup.setup_class == SetupClass.CLASS_D
                and pos.bars_held_1h >= CLASS_D_BAIL_BARS
                and r_now < CLASS_D_BAIL_R_THRESH
                and not pos.trail_active):
            self._submit_flatten(pos, bar_time, "STALE")
            return

        new_stop = pos.current_stop

        # Per-class BE threshold: R_BE for 4H origin, R_BE_1H for 1H origin
        be_threshold = R_BE_1H if setup.origin_tf == "1H" else R_BE
        if r_now >= be_threshold and not pos.trail_active:
            be_stop = stops.compute_be_stop(
                setup.direction, pos.fill_price, self.tf_1h.atr, self.cfg.tick_size,
            )
            if setup.direction == Direction.LONG and be_stop > new_stop:
                new_stop = be_stop
            elif setup.direction == Direction.SHORT and be_stop < new_stop:
                new_stop = be_stop

        # +2.5R → partial 50% (spec s13.3)
        if r_now >= R_PARTIAL_2P5 and not pos.partial_2p5_done and not self.flags.disable_partial_2p5r:
            partial_qty = max(1, int(pos.qty_open * PARTIAL_2P5_FRAC))
            self._submit_partial_exit(pos, partial_qty, bar_time)
            pos.partial_2p5_done = True
            pos.qty_partial_1 = partial_qty

            # Ratchet stop
            ratchet = stops.compute_ratchet_stop(
                setup.direction, pos.fill_price, pos.r_price, self.cfg.tick_size,
            )
            if setup.direction == Direction.LONG and ratchet > new_stop:
                new_stop = ratchet
            elif setup.direction == Direction.SHORT and ratchet < new_stop:
                new_stop = ratchet

        # +5R → partial 25% + trail bonus (spec s13.4)
        if r_now >= R_PARTIAL_5 and not pos.partial_5_done and not self.flags.disable_partial_5r:
            partial_qty = max(1, int(pos.qty_open * PARTIAL_5_FRAC))
            self._submit_partial_exit(pos, partial_qty, bar_time)
            pos.partial_5_done = True
            pos.qty_partial_2 = partial_qty
            pos.trailing_mult_bonus += PARTIAL_5_TRAIL_BONUS

        # Track bars at +1R for trailing profit delay (cumulative, not consecutive)
        if r_now >= be_threshold:
            pos.bars_at_r1 += 1

        # Track momentum fade: histogram negative AND declining in trade direction
        hist_list = self.tf_1h.macd_hist_history
        if len(hist_list) >= 2:
            if setup.direction == Direction.LONG:
                fading = self.tf_1h.macd_hist < 0 and hist_list[-1] < hist_list[-2]
            else:
                fading = self.tf_1h.macd_hist > 0 and hist_list[-1] > hist_list[-2]
            if fading:
                pos.bars_neg_fading_hist += 1
            else:
                pos.bars_neg_fading_hist = 0

        # Trailing chandelier (spec s14) — activate after profit delay
        if setup.direction == Direction.LONG:
            max_mfe_r = (pos.mfe_price - pos.fill_price) / r_base
        else:
            max_mfe_r = (pos.fill_price - pos.mfe_price) / r_base

        if stops.should_flatten_rts_failure(
            max_mfe_r=max_mfe_r,
            current_r=r_now,
            bars_held=pos.bars_held_1h,
            fading_bars=pos.bars_neg_fading_hist,
            trail_active=pos.trail_active,
            min_mfe_r=RTS_GUARD_MFE_R,
            min_giveback_r=RTS_GUARD_MIN_GIVEBACK_R,
            min_bars=RTS_GUARD_MIN_BARS,
            fade_bars=RTS_GUARD_FADE_BARS,
            max_mfe_r_limit=RTS_GUARD_MAX_MFE_R,
            flatten_r=RTS_FAIL_FLATTEN_R,
        ):
            self._submit_flatten(pos, bar_time, "RTS_FAIL")
            return

        if stops.should_arm_rts_guard(
            max_mfe_r=max_mfe_r,
            current_r=r_now,
            bars_held=pos.bars_held_1h,
            fading_bars=pos.bars_neg_fading_hist,
            trail_active=pos.trail_active,
            min_mfe_r=RTS_GUARD_MFE_R,
            min_giveback_r=RTS_GUARD_MIN_GIVEBACK_R,
            min_bars=RTS_GUARD_MIN_BARS,
            fade_bars=RTS_GUARD_FADE_BARS,
            max_mfe_r_limit=RTS_GUARD_MAX_MFE_R,
        ):
            guard_stop = stops.compute_rts_guard_stop(
                direction=setup.direction,
                avg_entry=pos.fill_price,
                r_price=pos.r_price,
                current_price=C,
                tick_size=self.cfg.tick_size,
                floor_r=RTS_GUARD_FLOOR_R,
            )
            if guard_stop is not None:
                if setup.direction == Direction.LONG and guard_stop > new_stop:
                    new_stop = guard_stop
                elif setup.direction == Direction.SHORT and guard_stop < new_stop:
                    new_stop = guard_stop

        if r_now >= be_threshold and pos.bars_at_r1 >= TRAIL_PROFIT_DELAY_BARS and not self.flags.disable_chandelier_trailing:
            pos.trail_active = True
            momentum_strong = False
            if r_state > 2.0 and len(self.tf_1h.macd_line_history) >= 6:
                momentum_strong = stops.is_momentum_strong(
                    self.tf_1h.macd_line, self.tf_1h.macd_line_history[-6],
                    self.tf_1h.macd_hist,
                    direction=setup.direction,
                )

            # Regime deterioration
            regime_deteriorated = False
            daily = self.daily_state
            if daily and pos.regime_at_entry:
                was_aligned = (
                    (setup.direction == Direction.LONG and pos.regime_at_entry == "BULL")
                    or (setup.direction == Direction.SHORT and pos.regime_at_entry == "BEAR")
                )
                if was_aligned and daily.regime == Regime.CHOP:
                    regime_deteriorated = True

            # Regime flip: daily regime opposes position direction
            regime_flipped = False
            if daily:
                regime_flipped = (
                    (setup.direction == Direction.LONG and daily.regime == Regime.BEAR)
                    or (setup.direction == Direction.SHORT and daily.regime == Regime.BULL)
                )

            trail_mult = stops.compute_trailing_mult(
                r_state, momentum_strong, regime_deteriorated, regime_flipped,
                pos.trailing_mult_bonus,
            )

            # R-band trailing profile override (spec s14.R)
            # Order: check HIGH first so unconfigured MID band keeps global defaults
            if R_BAND_MID > 0 and R_BAND_HIGH > 0:
                if r_state >= R_BAND_HIGH and TRAIL_BASE_HIGH_R > 0:
                    trail_mult = max(TRAIL_MIN, TRAIL_BASE_HIGH_R - r_state / (TRAIL_R_DIV_HIGH_R or TRAIL_R_DIV))
                elif r_state < R_BAND_MID and TRAIL_BASE_LOW_R > 0:
                    trail_mult = max(TRAIL_MIN, TRAIL_BASE_LOW_R - r_state / (TRAIL_R_DIV_LOW_R or TRAIL_R_DIV))
                elif TRAIL_BASE_MID_R > 0:
                    trail_mult = max(TRAIL_MIN, TRAIL_BASE_MID_R - r_state / (TRAIL_R_DIV_MID_R or TRAIL_R_DIV))

            # Class-specific trailing base/div override (narrower than R-band)
            cls_name = setup.setup_class.name if hasattr(setup.setup_class, 'name') else str(setup.setup_class)
            if cls_name == "D" and TRAIL_BASE_CLASS_D > 0:
                trail_mult = max(TRAIL_MIN, TRAIL_BASE_CLASS_D - r_state / (TRAIL_R_DIV_CLASS_D or TRAIL_R_DIV))
            elif cls_name == "B" and TRAIL_BASE_CLASS_B > 0:
                trail_mult = max(TRAIL_MIN, TRAIL_BASE_CLASS_B - r_state / (TRAIL_R_DIV_CLASS_B or TRAIL_R_DIV))

            # Select class-specific inline layer params
            if cls_name == "D" and TRAIL_STALL_ONSET_CLASS_D > 0:
                _fade_penalty = TRAIL_FADE_PENALTY_CLASS_D or TRAIL_FADE_PENALTY
                _fade_min_r = TRAIL_FADE_MIN_R_CLASS_D or TRAIL_FADE_MIN_R
                _stall_onset = TRAIL_STALL_ONSET_CLASS_D
            elif cls_name == "B" and TRAIL_STALL_ONSET_CLASS_B > 0:
                _fade_penalty = TRAIL_FADE_PENALTY
                _fade_min_r = TRAIL_FADE_MIN_R
                _stall_onset = TRAIL_STALL_ONSET_CLASS_B
            else:
                _fade_penalty = TRAIL_FADE_PENALTY
                _fade_min_r = TRAIL_FADE_MIN_R
                _stall_onset = TRAIL_STALL_ONSET

            # Momentum fade tightening: if momentum dying and R > threshold, tighten
            if pos.bars_neg_fading_hist >= TRAIL_FADE_ONSET_BARS and r_state > _fade_min_r:
                trail_mult = max(TRAIL_FADE_FLOOR, trail_mult - _fade_penalty)
            # Time-decay trailing: after onset bars at +1R, tighten per bar
            if pos.bars_at_r1 > TRAIL_TIMEDECAY_ONSET:
                decay = (pos.bars_at_r1 - TRAIL_TIMEDECAY_ONSET) * TRAIL_TIMEDECAY_RATE
                trail_mult = max(TRAIL_TIMEDECAY_FLOOR, trail_mult - decay)
            # Stalled winner decay: profitable but no new MFE for onset+ bars
            bars_since_peak = pos.bars_held_1h - pos.bar_of_max_mfe
            if r_state > 0.5 and bars_since_peak >= _stall_onset:
                stall_decay = min(1.0, bars_since_peak * TRAIL_STALL_RATE)
                trail_mult = max(TRAIL_STALL_FLOOR, trail_mult - stall_decay)
            chandelier = stops.compute_chandelier_stop(
                setup.direction, self.tf_1h.highs, self.tf_1h.lows,
                self.cfg.chandelier_lookback, self.tf_1h.atr,
                trail_mult, self.cfg.tick_size,
            )
            if setup.direction == Direction.LONG and chandelier > new_stop:
                new_stop = chandelier
            elif setup.direction == Direction.SHORT and chandelier < new_stop:
                new_stop = chandelier

        # Class C min hold: prevent early exits before reversal develops
        class_c_min_hold = setup.setup_class == SetupClass.CLASS_C and pos.bars_held_1h < CLASS_C_MIN_HOLD_BARS

        # Regime flip exit: daily regime opposes position direction
        daily = self.daily_state
        if daily and not class_c_min_hold:
            if (
                (setup.direction == Direction.LONG and daily.regime == Regime.BEAR)
                or (setup.direction == Direction.SHORT and daily.regime == Regime.BULL)
            ):
                self._submit_flatten(pos, bar_time, "REGIME_FLIP")
                return

        # Early stale: if N+ bars and trail never activated, flatten losers.
        if pos.bars_held_1h >= EARLY_STALE_BARS and not pos.trail_active and r_state < 0 and not class_c_min_hold:
            self._submit_flatten(pos, bar_time, "STALE")
            return

        # Stale exit (graduated response) — skip during Class C min hold
        stale_bars = STALE_1H_BARS if setup.origin_tf == "1H" else STALE_4H_BARS
        bars_held = pos.bars_held_1h if setup.origin_tf == "1H" else pos.bars_held_4h
        if bars_held >= stale_bars and r_state < STALE_R_THRESH and not class_c_min_hold:
            if r_state >= STALE_FLATTEN_R_FLOOR:
                if not pos.trail_active:
                    # Trail never activated — flatten rather than hold indefinitely
                    self._submit_flatten(pos, bar_time, "STALE")
                    return
                # else: trail is active and will tighten naturally
            else:
                # Below floor: flatten immediately
                self._submit_flatten(pos, bar_time, "STALE")
                return

        # Update stop if tightened
        if new_stop != pos.current_stop:
            safe = False
            if setup.direction == Direction.LONG and new_stop > pos.current_stop:
                safe = True
            elif setup.direction == Direction.SHORT and new_stop < pos.current_stop:
                safe = True
            if safe:
                pos.current_stop = new_stop
                # Replace protective stop order
                self.broker.cancel_orders(self.symbol, tag="protective_stop")
                stop_side = OrderSide.SELL if setup.direction == Direction.LONG else OrderSide.BUY
                stop_order = SimOrder(
                    order_id=self.broker.next_order_id(),
                    symbol=self.symbol,
                    side=stop_side,
                    order_type=OrderType.STOP,
                    qty=pos.qty_open,
                    stop_price=new_stop,
                    tick_size=self.cfg.tick_size,
                    submit_time=bar_time,
                    ttl_hours=0,
                    tag="protective_stop",
                )
                self.broker.submit_order(stop_order)

                # Core notification: stop update + registration
                self._replay_core_step(bar_input={"bar_ts": bar_time, "stop_update": AKCHelixStopUpdateRequest(
                    setup_id=setup.setup_id, symbol=self.symbol,
                    stop_price=new_stop, qty=pos.qty_open, reason="trailing",
                )})
                self._sync_core_stop(setup.setup_id, stop_order.order_id)

        # Add-on check (simplified: time + R + price gate)
        if not pos.add_done and not self.flags.disable_add_ons and pos.qty_open > 0:
            min_r = ADD_4H_R if setup.origin_tf == "4H" else ADD_1H_R
            from strategies.swing.akc_helix.config import ADD_MIN_BARS, ADD_MAX_BARS, ADD_PRICE_GATE_ATR_MULT
            in_time_window = ADD_MIN_BARS <= pos.bars_held_1h <= ADD_MAX_BARS
            if r_now >= min_r and in_time_window:
                self._try_add_simplified(pos, bar_time, C)

    def _submit_partial_exit(
        self, pos: _ActivePosition, qty: int, bar_time: datetime,
    ) -> None:
        """Submit a market order for partial exit."""
        if qty <= 0 or qty > pos.qty_open:
            return
        exit_side = OrderSide.SELL if pos.setup.direction == Direction.LONG else OrderSide.BUY
        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=exit_side,
            order_type=OrderType.MARKET,
            qty=qty,
            tick_size=self.cfg.tick_size,
            submit_time=bar_time,
            ttl_hours=0,
            tag="partial",
        )
        self.broker.submit_order(order)

        # Core notification: partial exit requested
        self._replay_core_step(bar_input={"bar_ts": bar_time, "partial_exit_request": AKCHelixPartialExitRequest(
            client_order_id=order.order_id, setup_id=pos.setup.setup_id,
            symbol=self.symbol, qty=qty, reason="partial",
        )})
        self._core_state.order_to_setup[order.order_id] = pos.setup.setup_id

    def _submit_flatten(self, pos: _ActivePosition, bar_time: datetime, reason: str) -> None:
        """Submit a next-bar market order for a discretionary full exit."""
        if pos.qty_open <= 0 or self._pending_flatten_reason:
            return
        self.broker.cancel_orders(self.symbol, tag="protective_stop")
        exit_side = OrderSide.SELL if pos.setup.direction == Direction.LONG else OrderSide.BUY
        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=exit_side,
            order_type=OrderType.MARKET,
            qty=pos.qty_open,
            tick_size=self.cfg.tick_size,
            submit_time=bar_time,
            ttl_hours=0,
            tag="flatten",
        )
        self.broker.submit_order(order)
        self._pending_flatten_reason = reason

        # Core notification: flatten requested
        self._replay_core_step(bar_input={"bar_ts": bar_time, "flatten_request": AKCHelixFlattenRequest(
            setup_id=pos.setup.setup_id, symbol=self.symbol, reason=reason,
        )})

    def _on_flatten_fill(self, fill: FillResult, bar_time: datetime) -> None:
        """Handle a queued flatten fill."""
        pos = self.active_position
        if pos is None:
            return
        reason = self._pending_flatten_reason or "FLATTEN"
        self._pending_flatten_reason = None
        self._flatten_position(
            pos,
            fill.fill_price,
            bar_time,
            reason,
            exit_commission=fill.commission,
        )

    def _flatten_at_end_of_data(self, last_price: float, bar_time: datetime) -> None:
        """Apply market-exit friction at end of data."""
        pos = self.active_position
        if pos is None or pos.qty_open <= 0:
            return
        # Core notification: flatten requested (end of data)
        self._replay_core_step(bar_input={"bar_ts": bar_time, "flatten_request": AKCHelixFlattenRequest(
            setup_id=pos.setup.setup_id, symbol=self.symbol, reason="END_OF_DATA",
        )})
        exit_side = OrderSide.SELL if pos.setup.direction == Direction.LONG else OrderSide.BUY
        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=exit_side,
            order_type=OrderType.MARKET,
            qty=pos.qty_open,
            tick_size=self.cfg.tick_size,
            submit_time=bar_time,
            ttl_hours=0,
            tag="flatten",
        )
        fill = self.broker.fill_market_order(order, bar_time, last_price, self.cfg.tick_size)
        self._pending_flatten_reason = None
        self._flatten_position(
            pos,
            fill.fill_price,
            bar_time,
            "END_OF_DATA",
            exit_commission=fill.commission,
        )
        if self.equity_curve:
            self.equity_curve[-1] = self.equity
        else:
            self.equity_curve.append(self.equity)
            self.timestamps.append(np.datetime64(bar_time))

    def _try_add(self, pos: _ActivePosition, bar_time: datetime) -> None:
        """Detect and place add-on entry (spec s15.2) — original structural version."""
        setup = pos.setup
        daily = self.daily_state
        if daily is None:
            return

        add = signals.detect_add_setup(
            self.symbol, setup.direction, self.pivots_1h, self.tf_1h,
            setup.bos_level, self.cfg, daily, bar_time,
        )
        if add is None:
            return

        # Add risk = 0.50 * Unit1Risk
        add_risk = ADD_RISK_FRAC * setup.unit1_risk_dollars
        risk_per_contract = abs(add.bos_level - add.stop0) * self.point_value
        if risk_per_contract <= 0:
            return
        add_qty = max(1, int(add_risk / risk_per_contract))
        add_qty = self._cap_add_qty(pos, add_qty)
        if add_qty <= 0:
            return

        tick = self.cfg.tick_size
        side = OrderSide.BUY if setup.direction == Direction.LONG else OrderSide.SELL

        if setup.direction == Direction.LONG:
            trigger = round_to_tick(add.bos_level, tick, "up")
            limit_price = trigger + self.cfg.offset_tight_ticks * tick
        else:
            trigger = round_to_tick(add.bos_level, tick, "down")
            limit_price = trigger - self.cfg.offset_tight_ticks * tick

        from strategies.swing.akc_helix.config import TTL_ADD_HOURS
        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=side,
            order_type=OrderType.STOP_LIMIT,
            qty=add_qty,
            stop_price=trigger,
            limit_price=limit_price,
            tick_size=tick,
            submit_time=bar_time,
            ttl_hours=TTL_ADD_HOURS,
            tag="add_on",
        )
        self.broker.submit_order(order)
        pos.add_done = True

        # Core notification: add-on entry requested
        add_req = AKCHelixEntryRequest(
            client_order_id=order.order_id,
            setup=pos.setup,
            order_role="add",
            qty=order.qty,
        )
        self._replay_core_step(bar_input={"bar_ts": bar_time, "entry_request": add_req})

    def _try_add_simplified(
        self, pos: _ActivePosition, bar_time: datetime, current_price: float,
    ) -> None:
        """Simplified add: price gate + market entry, no pivot requirement."""
        setup = pos.setup
        from strategies.swing.akc_helix.config import ADD_PRICE_GATE_ATR_MULT

        # Price gate: price must have moved beyond BoS + 0.5×ATR1H in trade direction
        price_offset = ADD_PRICE_GATE_ATR_MULT * self.tf_1h.atr
        if setup.direction == Direction.LONG:
            if current_price < setup.bos_level + price_offset:
                return
        else:
            if current_price > setup.bos_level - price_offset:
                return

        # Add risk = 0.50 * Unit1Risk
        add_risk = ADD_RISK_FRAC * setup.unit1_risk_dollars
        risk_per_contract = pos.r_price * self.point_value
        if risk_per_contract <= 0:
            return
        add_qty = max(1, int(add_risk / risk_per_contract))
        add_qty = self._cap_add_qty(pos, add_qty)
        if add_qty <= 0:
            return

        # Market order for immediate fill
        side = OrderSide.BUY if setup.direction == Direction.LONG else OrderSide.SELL
        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol,
            side=side,
            order_type=OrderType.MARKET,
            qty=add_qty,
            tick_size=self.cfg.tick_size,
            submit_time=bar_time,
            ttl_hours=0,
            tag="add_on",
        )
        self.broker.submit_order(order)
        pos.add_done = True

        # Core notification: add-on entry requested (simplified)
        add_req = AKCHelixEntryRequest(
            client_order_id=order.order_id,
            setup=pos.setup,
            order_role="add",
            qty=order.qty,
        )
        self._replay_core_step(bar_input={"bar_ts": bar_time, "entry_request": add_req})

    # ------------------------------------------------------------------
    # Position flatten (market exit)
    # ------------------------------------------------------------------

    def _flatten_position(
        self,
        pos: _ActivePosition,
        exit_price: float,
        bar_time: datetime,
        reason: str,
        *,
        exit_commission: float | None = None,
    ) -> None:
        """Flatten entire position at the given price."""
        setup = pos.setup
        pv = self.point_value
        cost_basis = self._position_cost_basis(pos)

        if setup.direction == Direction.LONG:
            pnl_points = exit_price - cost_basis
        else:
            pnl_points = cost_basis - exit_price

        # Compute exit commission for remaining open qty
        if exit_commission is None:
            exit_commission = self.broker._compute_commission(pos.qty_open)
        self.total_commission += exit_commission
        pos.commission += exit_commission

        pnl_dollars = pnl_points * pv * pos.qty_open + pos.realized_pnl
        r_multiple = pnl_dollars / setup.unit1_risk_dollars if setup.unit1_risk_dollars > 0 else 0.0
        net_pnl_dollars, net_r_multiple = self._net_trade_r(pnl_dollars, pos.commission, setup)

        r_base = pos.r_price
        if r_base > 0:
            if setup.direction == Direction.LONG:
                mfe_r = (pos.mfe_price - pos.fill_price) / r_base
                mae_r = (pos.fill_price - pos.mae_price) / r_base
            else:
                mfe_r = (pos.fill_price - pos.mfe_price) / r_base
                mae_r = (pos.mae_price - pos.fill_price) / r_base
        else:
            mfe_r = mae_r = 0.0

        self.equity += pnl_points * pv * pos.qty_open - exit_commission

        trade = HelixTradeRecord(
            symbol=self.symbol,
            direction=setup.direction,
            setup_class=setup.setup_class.value,
            origin_tf=setup.origin_tf,
            entry_time=pos.entry_time,
            exit_time=bar_time,
            entry_price=pos.fill_price,
            avg_entry_price=cost_basis,
            exit_price=exit_price,
            qty=setup.qty_planned,
            initial_stop=pos.initial_stop,
            exit_reason=reason,
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            r_multiple=r_multiple,
            net_pnl_dollars=net_pnl_dollars,
            net_r_multiple=net_r_multiple,
            base_unit1_risk_dollars=setup.base_unit1_risk_dollars,
            target_initial_risk_dollars=setup.target_initial_risk_dollars,
            actual_initial_risk_dollars=setup.actual_initial_risk_dollars,
            risk_utilization=setup.risk_utilization,
            mfe_r=mfe_r,
            mae_r=mae_r,
            bars_held=pos.bars_held_1h,
            commission=pos.commission,
            qty_partial_1=pos.qty_partial_1,
            qty_partial_2=pos.qty_partial_2,
            add_on_qty=pos.add_qty,
            add_on_price=pos.add_fill_price,
            setup_size_mult=setup.setup_size_mult,
            adx_at_entry=setup.adx_at_entry,
            div_mag_norm=setup.div_mag_norm,
            regime_4h_at_entry=setup.regime_4h_at_entry or "",
            regime_at_entry=pos.regime_at_entry,
        )
        self.trades.append(trade)

        # Update circuit breaker
        self._update_circuit_breaker(net_r_multiple, bar_time)

        # Cancel all remaining orders
        self._pending_flatten_reason = None
        self.broker.cancel_all(self.symbol)
        self.active_position = None

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def _update_circuit_breaker(self, r_multiple: float, bar_time: datetime) -> None:
        """Update circuit breaker state after a trade close."""
        if self.flags.disable_circuit_breaker:
            return

        cb = roll_circuit_breaker_window(self.circuit_breaker, bar_time)
        cb.daily_realized_r += r_multiple
        cb.weekly_realized_r += r_multiple
        self.circuit_breaker = cb

        if r_multiple < 0:
            cb.consecutive_stops += 1
            if cb.consecutive_stops >= CONSEC_STOPS_HALVE:
                cb.halved_until = bar_time + timedelta(hours=24)
        else:
            cb.consecutive_stops = 0

        if cb.daily_realized_r <= DAILY_STOP_R:
            cb.paused_until = bar_time + timedelta(hours=24)

        if cb.weekly_realized_r <= WEEKLY_STOP_R:
            cb.paused_until = bar_time + timedelta(hours=48)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_datetime(ts) -> datetime:
        """Convert numpy datetime64 or pandas Timestamp to datetime."""
        if isinstance(ts, datetime):
            return ts
        if hasattr(ts, 'to_pydatetime'):
            return ts.to_pydatetime()
        # numpy datetime64
        return pd.Timestamp(ts).to_pydatetime()
