"""Regime-following backtest engine.

Enters on regime confirmation (trend_dir flip), exits on regime flip.
Uses tight trailing stops for risk control only — no pullback signals,
no quality gates, no cooldown, no add-ons.

Investigation findings driving this design:
- Regime B&H captures 221% of QQQ buy-and-hold
- Chandelier at 1.2-1.5x outperforms 2.2-3.2x baseline
- TimeExit@80-120h is the best exit rule inside trades
- Trades held 160+h have 83-100% WR
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import numpy as np

from backtests.shared.parity.legacy_result_outputs import (
    decision_stream_from_trades,
    trade_outcomes_from_records,
)
from libs.broker_ibkr.risk_support.tick_rules import round_to_tick
from strategies.swing.atrss import stops
from strategies.swing.atrss.config import SYMBOL_CONFIGS, SymbolConfig
from strategies.swing.atrss.indicators import compute_daily_state, compute_hourly_state
from strategies.swing.atrss.models import DailyState, Direction, HourlyState, Regime

from backtests.swing.config import SlippageConfig
from backtests.swing.config_regime import RegimeConfig
from backtests.swing.data.preprocessing import NumpyBars
from backtests.swing.engine.backtest_engine import SymbolResult, TradeRecord
from backtests.swing.engine.portfolio_engine import PortfolioData, PortfolioResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RegimeEngine:
    """Single-symbol regime-following backtest engine."""

    def __init__(
        self,
        symbol: str,
        cfg: SymbolConfig,
        regime_config: RegimeConfig,
        point_value: float,
    ):
        self.symbol = symbol
        self.cfg = cfg
        self.regime_config = regime_config
        self.point_value = point_value

        # State
        self.equity = regime_config.initial_equity
        self.daily_state: DailyState | None = None
        self.hourly_state: HourlyState | None = None
        self.prev_trend_dir: Direction = Direction.FLAT
        self.prev_regime: Regime = Regime.RANGE

        # Position tracking (simple — no PositionBook overhead)
        self._in_position = False
        self._pos_direction: int = 0  # +1 LONG, -1 SHORT
        self._pos_entry_price: float = 0.0
        self._pos_initial_stop: float = 0.0
        self._pos_current_stop: float = 0.0
        self._pos_qty: int = 0
        self._pos_entry_time: datetime | None = None
        self._pos_risk_per_unit: float = 0.0
        self._pos_bars_held: int = 0
        self._pos_mfe_price: float = 0.0
        self._pos_mfe_r: float = 0.0
        self._pos_mae_price: float = 0.0
        self._pos_be_triggered: bool = False
        self._pos_regime_entry: str = ""

        # Resolve per-symbol overrides for chand_mult
        ov = regime_config.param_overrides
        sym_key = f"chand_mult_{symbol}"
        if sym_key in ov:
            self._chand_mult = float(ov[sym_key])
        elif "chand_mult" in ov:
            self._chand_mult = float(ov["chand_mult"])
        else:
            self._chand_mult = regime_config.chand_mult

        # Daily state history (for investigation tools compatibility)
        self._daily_state_by_idx: dict[int, DailyState] = {}

        # Results
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[float] = []
        self.timestamps: list = []
        self.total_commission: float = 0.0

        # Slippage config
        self._slip = regime_config.slippage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        daily_idx_map: np.ndarray,
    ) -> SymbolResult:
        """Run the backtest over all hourly bars."""
        warmup_d = self.regime_config.warmup_daily
        warmup_h = self.regime_config.warmup_hourly
        self._run_loop(daily, hourly, daily_idx_map, warmup_d, warmup_h)

        return SymbolResult(
            symbol=self.symbol,
            trades=self.trades,
            equity_curve=np.array(self.equity_curve),
            timestamps=np.array(self.timestamps),
            total_commission=self.total_commission,
            decision_stream=decision_stream_from_trades(self.trades, timeframe="1h"),
            trade_outcomes=trade_outcomes_from_records(self.trades),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        daily_idx_map: np.ndarray,
        warmup_d: int,
        warmup_h: int,
    ) -> None:
        last_daily_idx = -1

        for i in range(len(hourly)):
            bar_time = self._to_datetime(hourly.times[i])
            O = hourly.opens[i]
            H = hourly.highs[i]
            L = hourly.lows[i]
            C = hourly.closes[i]

            # Skip NaN bars
            if np.isnan(O) or np.isnan(H) or np.isnan(L):
                self.equity_curve.append(self.equity)
                self.timestamps.append(hourly.times[i])
                continue

            # 1. Update daily state when d_idx changes
            d_idx = int(daily_idx_map[i])
            daily_changed = False
            if d_idx != last_daily_idx and d_idx >= warmup_d:
                self._update_daily(daily, d_idx)
                last_daily_idx = d_idx
                daily_changed = True

            if self.daily_state is None:
                self.equity_curve.append(self.equity)
                self.timestamps.append(hourly.times[i])
                continue

            # 2. Compute hourly state
            if i >= warmup_h:
                start = max(0, i - max(self.cfg.ema_pull_normal, self.cfg.atr_hourly_period) - 5)
                self.hourly_state = compute_hourly_state(
                    hourly.closes[start:i + 1],
                    hourly.highs[start:i + 1],
                    hourly.lows[start:i + 1],
                    self.daily_state, self.cfg,
                    bar_time,
                    hourly.opens[start:i + 1],
                )
            else:
                self.equity_curve.append(self.equity)
                self.timestamps.append(hourly.times[i])
                continue

            h = self.hourly_state
            d = self.daily_state

            # 3. Position management or entry
            if self._in_position:
                self._manage_position(h, d, bar_time, daily_changed)
            elif daily_changed:
                self._check_entry(h, d, bar_time)

            self.equity_curve.append(self.equity)
            self.timestamps.append(hourly.times[i])

    # ------------------------------------------------------------------
    # Daily state update
    # ------------------------------------------------------------------

    def _update_daily(self, daily: NumpyBars, d_idx: int) -> None:
        start = max(0, d_idx - max(self.cfg.daily_ema_slow, self.cfg.atr_daily_period) - 5)
        d_closes = daily.closes[start:d_idx + 1]
        d_highs = daily.highs[start:d_idx + 1]
        d_lows = daily.lows[start:d_idx + 1]

        prev = self.daily_state
        self.prev_trend_dir = prev.trend_dir if prev else Direction.FLAT
        self.prev_regime = prev.regime if prev else Regime.RANGE
        daily_bar_date = str(daily.times[d_idx])[:10]
        self.daily_state = compute_daily_state(
            d_closes, d_highs, d_lows, prev, self.cfg, daily_bar_date,
        )
        self._daily_state_by_idx[d_idx] = self.daily_state

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_position(
        self, h: HourlyState, d: DailyState, bar_time: datetime, daily_changed: bool,
    ) -> None:
        if not self._in_position:
            return

        # Count RTH bars held
        if self._is_rth(bar_time):
            self._pos_bars_held += 1

        # Update MFE / MAE
        if self._pos_direction == Direction.LONG:
            if h.high > self._pos_mfe_price or self._pos_mfe_price == 0:
                self._pos_mfe_price = h.high
            if h.low < self._pos_mae_price or self._pos_mae_price == 0:
                self._pos_mae_price = h.low
        else:
            if h.low < self._pos_mfe_price or self._pos_mfe_price == 0:
                self._pos_mfe_price = h.low
            if h.high > self._pos_mae_price or self._pos_mae_price == 0:
                self._pos_mae_price = h.high

        risk_per_unit = self._pos_risk_per_unit
        if risk_per_unit > 0:
            if self._pos_direction == Direction.LONG:
                self._pos_mfe_r = (self._pos_mfe_price - self._pos_entry_price) / risk_per_unit
                cur_r = (h.close - self._pos_entry_price) / risk_per_unit
            else:
                self._pos_mfe_r = (self._pos_entry_price - self._pos_mfe_price) / risk_per_unit
                cur_r = (self._pos_entry_price - h.close) / risk_per_unit
        else:
            cur_r = 0.0

        # (a) Check stop hit — highest priority
        if self._pos_direction == Direction.LONG:
            if h.low <= self._pos_current_stop:
                self._close_position(self._pos_current_stop, bar_time, "STOP")
                return
        else:
            if h.high >= self._pos_current_stop:
                self._close_position(self._pos_current_stop, bar_time, "STOP")
                return

        # (b) Update trailing stop (every bar)
        new_stop = self._pos_current_stop

        # BE trigger
        if not self._pos_be_triggered and self._pos_mfe_r >= self.regime_config.be_trigger_r:
            be_stop = stops.compute_be_stop(
                self._pos_direction, self._pos_entry_price, d.atr20, self.cfg.tick_size,
            )
            if self._pos_direction == Direction.LONG and be_stop > new_stop:
                new_stop = be_stop
                self._pos_be_triggered = True
            elif self._pos_direction == Direction.SHORT and be_stop < new_stop:
                new_stop = be_stop
                self._pos_be_triggered = True

        # Chandelier trailing
        if self._pos_be_triggered and self._pos_mfe_r >= self.regime_config.chandelier_trigger_r:
            chandelier = stops.compute_chandelier_stop(
                self._pos_direction, d, self._chand_mult, self.cfg.tick_size,
            )
            if self._pos_direction == Direction.LONG and chandelier > new_stop:
                new_stop = chandelier
            elif self._pos_direction == Direction.SHORT and chandelier < new_stop:
                new_stop = chandelier

        # Profit floor
        if risk_per_unit > 0:
            floor_stop = stops.apply_profit_floor(
                self._pos_direction, self._pos_entry_price, risk_per_unit,
                self._pos_mfe_r, new_stop, self.cfg.tick_size,
            )
            if self._pos_direction == Direction.LONG and floor_stop > new_stop:
                new_stop = floor_stop
            elif self._pos_direction == Direction.SHORT and floor_stop < new_stop:
                new_stop = floor_stop

        self._pos_current_stop = new_stop

        # (c) Time-based forced exit
        if (
            self.regime_config.time_exit_hours > 0
            and self._pos_bars_held >= self.regime_config.time_exit_hours
            and cur_r < self.regime_config.time_exit_min_r
        ):
            self._close_position(h.close, bar_time, "TIME_EXIT")
            return

        # (d) Regime change exits (only on daily bar change)
        if daily_changed:
            # Regime downgrade exit
            if self.regime_config.regime_downgrade_exit:
                downgraded = False
                if (
                    self.prev_regime == Regime.STRONG_TREND
                    and d.regime in (Regime.TREND, Regime.RANGE)
                ):
                    downgraded = True
                elif self.prev_regime == Regime.TREND and d.regime == Regime.RANGE:
                    downgraded = True
                if downgraded:
                    self._close_position(h.close, bar_time, "REGIME_DOWNGRADE")
                    return

            # Regime flip exit (trend_dir changed away from position)
            if (
                d.trend_dir != Direction.FLAT
                and d.trend_dir != self._pos_direction
                and d.trend_dir != self.prev_trend_dir
            ):
                saved_dir = self._pos_direction
                self._close_position(h.close, bar_time, "REGIME_FLIP")
                # Check if new direction is opposite — open reverse
                if d.trend_dir == -saved_dir:
                    self._enter_position(d.trend_dir, h.close, h, d, bar_time)
                return

    # ------------------------------------------------------------------
    # Entry check
    # ------------------------------------------------------------------

    def _check_entry(self, h: HourlyState, d: DailyState, bar_time: datetime) -> None:
        """Check for entry on daily bar change (regime confirmation)."""
        if self._in_position:
            return

        # trend_dir just flipped to LONG (from FLAT or SHORT)
        if d.trend_dir == Direction.LONG and self.prev_trend_dir != Direction.LONG:
            self._enter_position(Direction.LONG, h.close, h, d, bar_time)
            return

        # trend_dir just flipped to SHORT
        if (
            self.regime_config.shorts_enabled
            and d.trend_dir == Direction.SHORT
            and self.prev_trend_dir != Direction.SHORT
        ):
            self._enter_position(Direction.SHORT, h.close, h, d, bar_time)

    # ------------------------------------------------------------------
    # Enter position
    # ------------------------------------------------------------------

    def _enter_position(
        self,
        direction: int,
        entry_price: float,
        h: HourlyState,
        d: DailyState,
        bar_time: datetime,
    ) -> None:
        # Apply slippage (unfavorable direction)
        slip_ticks = self._slip.slip_ticks_normal
        if bar_time.hour in self._slip.illiquid_hours:
            slip_ticks = self._slip.slip_ticks_illiquid
        slip_amount = slip_ticks * self.cfg.tick_size
        if direction == Direction.LONG:
            entry_price = round_to_tick(entry_price + slip_amount, self.cfg.tick_size, "up")
        else:
            entry_price = round_to_tick(entry_price - slip_amount, self.cfg.tick_size, "down")

        # Initial stop: entry - daily_mult * d.atr20 (simple ATR stop)
        stop_dist = self.cfg.daily_mult * d.atr20
        if direction == Direction.LONG:
            initial_stop = round_to_tick(entry_price - stop_dist, self.cfg.tick_size, "down")
        else:
            initial_stop = round_to_tick(entry_price + stop_dist, self.cfg.tick_size, "up")

        risk_per_unit = abs(entry_price - initial_stop)
        if risk_per_unit <= 0:
            return

        # Sizing
        if self.regime_config.fixed_qty is not None:
            qty = self.regime_config.fixed_qty
        else:
            risk_dollars = self.cfg.base_risk_pct * self.equity
            qty = max(1, int(risk_dollars / (risk_per_unit * self.point_value)))

        # Commission
        commission = qty * self._slip.commission_per_contract
        self.equity -= commission
        self.total_commission += commission

        # Set position state
        self._in_position = True
        self._pos_direction = direction
        self._pos_entry_price = entry_price
        self._pos_initial_stop = initial_stop
        self._pos_current_stop = initial_stop
        self._pos_qty = qty
        self._pos_entry_time = bar_time
        self._pos_risk_per_unit = risk_per_unit
        self._pos_bars_held = 0
        self._pos_mfe_price = 0.0
        self._pos_mfe_r = 0.0
        self._pos_mae_price = 0.0
        self._pos_be_triggered = False
        self._pos_regime_entry = d.regime.value

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------

    def _close_position(self, exit_price: float, bar_time: datetime, reason: str) -> None:
        if not self._in_position:
            return

        # Apply slippage on exit (unfavorable direction)
        slip_ticks = self._slip.slip_ticks_normal
        if bar_time.hour in self._slip.illiquid_hours:
            slip_ticks = self._slip.slip_ticks_illiquid
        slip_amount = slip_ticks * self.cfg.tick_size
        if self._pos_direction == Direction.LONG:
            exit_price = round_to_tick(exit_price - slip_amount, self.cfg.tick_size, "down")
        else:
            exit_price = round_to_tick(exit_price + slip_amount, self.cfg.tick_size, "up")

        # PnL
        if self._pos_direction == Direction.LONG:
            pnl_pts = exit_price - self._pos_entry_price
            mae_pts = self._pos_entry_price - self._pos_mae_price if self._pos_mae_price > 0 else 0.0
        else:
            pnl_pts = self._pos_entry_price - exit_price
            mae_pts = self._pos_mae_price - self._pos_entry_price if self._pos_mae_price > 0 else 0.0

        pnl_dollars = pnl_pts * self.point_value * self._pos_qty
        r_mult = pnl_pts / self._pos_risk_per_unit if self._pos_risk_per_unit > 0 else 0.0
        mae_r = mae_pts / self._pos_risk_per_unit if self._pos_risk_per_unit > 0 else 0.0

        # Commission on exit
        commission = self._pos_qty * self._slip.commission_per_contract
        self.total_commission += commission

        trade = TradeRecord(
            symbol=self.symbol,
            direction=int(self._pos_direction),
            entry_type="REGIME",
            entry_time=self._pos_entry_time,
            exit_time=bar_time,
            entry_price=self._pos_entry_price,
            exit_price=exit_price,
            qty=self._pos_qty,
            initial_stop=self._pos_initial_stop,
            exit_reason=reason,
            pnl_points=pnl_pts,
            pnl_dollars=pnl_dollars,
            r_multiple=r_mult,
            mfe_r=self._pos_mfe_r,
            mae_r=mae_r,
            bars_held=self._pos_bars_held,
            commission=commission,
            regime_entry=self._pos_regime_entry,
        )
        self.trades.append(trade)
        self.equity += pnl_dollars - commission

        # Reset position
        self._in_position = False
        self._pos_direction = 0
        self._pos_entry_price = 0.0
        self._pos_initial_stop = 0.0
        self._pos_current_stop = 0.0
        self._pos_qty = 0
        self._pos_entry_time = None
        self._pos_risk_per_unit = 0.0
        self._pos_bars_held = 0
        self._pos_mfe_price = 0.0
        self._pos_mfe_r = 0.0
        self._pos_mae_price = 0.0
        self._pos_be_triggered = False
        self._pos_regime_entry = ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rth(dt: datetime) -> bool:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        dt_et = dt.astimezone(et)
        if dt_et.weekday() >= 5:
            return False
        t = dt_et.hour * 60 + dt_et.minute
        return 570 <= t < 960  # 09:30-16:00

    @staticmethod
    def _to_datetime(ts) -> datetime:
        if isinstance(ts, datetime):
            return ts
        if hasattr(ts, 'astype'):
            unix_epoch = np.datetime64(0, 'ns')
            one_second = np.timedelta64(1, 's')
            seconds = (ts - unix_epoch) / one_second
            return datetime.fromtimestamp(float(seconds), tz=timezone.utc)
        return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Portfolio runner
# ---------------------------------------------------------------------------

def _get_point_value(symbol: str) -> float:
    cfg = SYMBOL_CONFIGS.get(symbol)
    return cfg.multiplier if cfg else 1.0


def _apply_overrides(cfg: SymbolConfig, overrides: dict[str, float]) -> SymbolConfig:
    if not overrides:
        return cfg
    from dataclasses import asdict
    changes: dict[str, object] = {}
    for key, value in overrides.items():
        suffix = f"_{cfg.symbol}"
        field_name = key[:-len(suffix)] if key.endswith(suffix) else key
        if hasattr(cfg, field_name):
            current = getattr(cfg, field_name)
            changes[field_name] = int(round(value)) if isinstance(current, int) else float(value)
    if not changes:
        return cfg
    d = asdict(cfg)
    d.update(changes)
    return SymbolConfig(**d)


def _combine_equity_curves(
    results: dict[str, SymbolResult],
    initial_equity: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not results:
        return np.array([initial_equity]), np.array([])
    max_len = max(len(r.equity_curve) for r in results.values())
    combined = np.full(max_len, initial_equity, dtype=np.float64)
    for r in results.values():
        n = len(r.equity_curve)
        if n == 0:
            continue
        padded = np.full(max_len, r.equity_curve[-1] if n > 0 else initial_equity)
        padded[:n] = r.equity_curve
        combined += (padded - initial_equity)
    longest_sym = max(results, key=lambda s: len(results[s].timestamps))
    combined_ts = results[longest_sym].timestamps
    return combined, combined_ts


def run_regime_independent(
    data: PortfolioData,
    config: RegimeConfig,
) -> PortfolioResult:
    """Run each symbol independently with the regime-following engine."""
    results: dict[str, SymbolResult] = {}

    for sym in config.symbols:
        if sym not in data.hourly or sym not in data.daily:
            logger.warning("No data for %s, skipping", sym)
            continue

        cfg = SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            logger.warning("No SymbolConfig for %s, skipping", sym)
            continue
        cfg = _apply_overrides(cfg, config.param_overrides)

        engine = RegimeEngine(
            symbol=sym,
            cfg=cfg,
            regime_config=config,
            point_value=_get_point_value(sym),
        )
        results[sym] = engine.run(
            daily=data.daily[sym],
            hourly=data.hourly[sym],
            daily_idx_map=data.daily_idx_maps[sym],
        )

    combined_equity, combined_ts = _combine_equity_curves(results, config.initial_equity)
    return PortfolioResult(
        symbol_results=results,
        combined_equity=combined_equity,
        combined_timestamps=combined_ts,
    )
