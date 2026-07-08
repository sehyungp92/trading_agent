"""Shadow trade simulation for rejected candidates.

Tracks what would have happened if a filter had NOT rejected a candidate.
Uses full stop lifecycle (BE, chandelier, profit floor, time decay,
bias flip) to match the main backtest engine's exit logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

import strategies.swing.atrss.config as _scfg
from strategies.swing.atrss.config import (
    MAX_HOLD_HOURS,
    ORDER_EXPIRY_HOURS,
    PROFIT_FLOOR,
    SymbolConfig,
)
from strategies.swing.atrss.models import DailyState, Direction, Regime
from strategies.swing.atrss import stops

logger = logging.getLogger(__name__)


@dataclass
class ShadowCandidate:
    """A candidate that was rejected by one or more filters."""

    symbol: str
    direction: int
    filter_names: list[str]
    time: datetime
    entry_price: float
    stop_price: float

    @property
    def filter_name(self) -> str:
        return self.filter_names[0] if self.filter_names else ""


@dataclass
class ShadowResult:
    """Outcome of simulating a shadow candidate."""

    candidate: ShadowCandidate
    filled: bool = False
    fill_price: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    bars_held: int = 0


@dataclass
class FilterStats:
    """Aggregated stats for one filter."""

    filter_name: str
    rejected_count: int = 0
    simulated_count: int = 0
    filled_count: int = 0
    avg_shadow_r: float = 0.0
    pct_above_1r: float = 0.0
    pct_above_2r: float = 0.0
    net_missed_expectancy: float = 0.0
    net_avoided_loss: float = 0.0


class ShadowTracker:
    """Track rejected candidates and simulate their outcomes."""

    def __init__(self):
        self.rejections: list[ShadowCandidate] = []
        self.results: list[ShadowResult] = []

    def record_rejection(
        self,
        symbol: str,
        direction: int,
        filter_names: str | list[str],
        time: datetime,
        entry_price: float,
        stop_price: float,
    ) -> None:
        """Log a rejected candidate with all failed filter names."""
        if isinstance(filter_names, str):
            filter_names = [filter_names]
        self.rejections.append(ShadowCandidate(
            symbol=symbol, direction=direction, filter_names=filter_names,
            time=time, entry_price=entry_price, stop_price=stop_price,
        ))

    def simulate_shadows(
        self,
        hourly_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        hourly_times: dict[str, np.ndarray],
        configs: dict[str, SymbolConfig],
        point_values: dict[str, float],
        daily_states: dict[str, dict[int, DailyState]] | None = None,
        daily_idx_maps: dict[str, np.ndarray] | None = None,
    ) -> list[ShadowResult]:
        """Forward-simulate each rejected candidate.

        When daily_states and daily_idx_maps are provided, uses full stop
        lifecycle (BE, chandelier, regime collapse, time decay, bias flip).
        Otherwise falls back to simplified stop + time-decay exit.
        """
        self.results.clear()

        for cand in self.rejections:
            sym = cand.symbol
            if sym not in hourly_data or sym not in hourly_times:
                continue

            opens, highs, lows, closes, _ = hourly_data[sym]
            times = hourly_times[sym]
            cfg = configs.get(sym)
            if cfg is None:
                continue

            d_states = daily_states.get(sym) if daily_states else None
            d_idx_map = daily_idx_maps.get(sym) if daily_idx_maps else None

            result = self._simulate_one(
                cand, opens, highs, lows, closes, times, cfg,
                d_states, d_idx_map,
            )
            self.results.append(result)

        return self.results

    def _simulate_one(
        self,
        cand: ShadowCandidate,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        times: np.ndarray,
        cfg: SymbolConfig,
        daily_states: dict[int, DailyState] | None,
        daily_idx_map: np.ndarray | None,
    ) -> ShadowResult:
        """Simulate a single shadow candidate with full stop lifecycle."""
        cand_ts = np.datetime64(cand.time, 'ns') if not isinstance(cand.time, np.datetime64) else cand.time
        start_idx = np.searchsorted(times, cand_ts, side='right')

        if start_idx >= len(times):
            return ShadowResult(candidate=cand)

        r_base = abs(cand.entry_price - cand.stop_price)
        if r_base <= 0:
            return ShadowResult(candidate=cand)

        # Phase 1: check if entry would fill within TTL
        fill_price = 0.0
        fill_idx = -1
        for i in range(start_idx, min(start_idx + ORDER_EXPIRY_HOURS, len(opens))):
            if cand.direction == Direction.LONG:
                if highs[i] >= cand.entry_price:
                    fill_price = max(opens[i], cand.entry_price)
                    fill_idx = i
                    break
            else:
                if lows[i] <= cand.entry_price:
                    fill_price = min(opens[i], cand.entry_price)
                    fill_idx = i
                    break

        if fill_idx < 0:
            return ShadowResult(candidate=cand, filled=False)

        # Phase 2: full stop lifecycle simulation
        current_stop = cand.stop_price
        mfe_price = fill_price
        mfe_r = 0.0
        be_triggered = False
        has_daily = daily_states is not None and daily_idx_map is not None

        for j in range(fill_idx + 1, len(closes)):
            bars_held = j - fill_idx

            # Update MFE
            if cand.direction == Direction.LONG:
                if highs[j] > mfe_price:
                    mfe_price = highs[j]
                cur_mfe = (mfe_price - fill_price) / r_base
            else:
                if lows[j] < mfe_price:
                    mfe_price = lows[j]
                cur_mfe = (fill_price - mfe_price) / r_base
            mfe_r = max(mfe_r, cur_mfe)

            # Get daily state for this bar
            d = None
            if has_daily and j < len(daily_idx_map):
                d_idx = int(daily_idx_map[j])
                d = daily_states.get(d_idx)

            # --- BE trigger at configurable R ---
            if not be_triggered and cur_mfe >= _scfg.BE_TRIGGER_R and d is not None:
                be_stop = stops.compute_be_stop(
                    cand.direction, fill_price, d.atr20, cfg.tick_size,
                )
                if cand.direction == Direction.LONG and be_stop > current_stop:
                    current_stop = be_stop
                    be_triggered = True
                elif cand.direction == Direction.SHORT and be_stop < current_stop:
                    current_stop = be_stop
                    be_triggered = True

            # --- Chandelier trailing at configurable R ---
            if be_triggered and cur_mfe >= _scfg.CHANDELIER_TRIGGER_R and d is not None:
                chand = stops.compute_chandelier_stop(
                    cand.direction, d, cfg.chand_mult, cfg.tick_size,
                )
                if cand.direction == Direction.LONG and chand > current_stop:
                    current_stop = chand
                elif cand.direction == Direction.SHORT and chand < current_stop:
                    current_stop = chand

            # --- Profit floor ---
            if r_base > 0:
                floor_stop = stops.apply_profit_floor(
                    cand.direction, fill_price, r_base, cur_mfe, current_stop, cfg.tick_size,
                )
                if cand.direction == Direction.LONG and floor_stop > current_stop:
                    current_stop = floor_stop
                elif cand.direction == Direction.SHORT and floor_stop < current_stop:
                    current_stop = floor_stop

            # --- Check stop fill ---
            if cand.direction == Direction.LONG:
                if lows[j] <= current_stop:
                    pnl = current_stop - fill_price
                    return ShadowResult(
                        candidate=cand, filled=True, fill_price=fill_price,
                        exit_price=current_stop, exit_reason="STOP",
                        r_multiple=pnl / r_base, mfe_r=mfe_r, bars_held=bars_held,
                    )
            else:
                if highs[j] >= current_stop:
                    pnl = fill_price - current_stop
                    return ShadowResult(
                        candidate=cand, filled=True, fill_price=fill_price,
                        exit_price=current_stop, exit_reason="STOP",
                        r_multiple=pnl / r_base, mfe_r=mfe_r, bars_held=bars_held,
                    )

            # --- Time decay exit ---
            if bars_held >= MAX_HOLD_HOURS:
                if cand.direction == Direction.LONG:
                    cur_profit_r = (closes[j] - fill_price) / r_base
                else:
                    cur_profit_r = (fill_price - closes[j]) / r_base
                if cur_profit_r < 1.0:
                    pnl = closes[j] - fill_price if cand.direction == Direction.LONG else fill_price - closes[j]
                    return ShadowResult(
                        candidate=cand, filled=True, fill_price=fill_price,
                        exit_price=closes[j], exit_reason="TIME_DECAY",
                        r_multiple=pnl / r_base, mfe_r=mfe_r, bars_held=bars_held,
                    )

            # --- Bias flip exit ---
            if d is not None and d.trend_dir != Direction.FLAT and d.trend_dir != cand.direction:
                pnl = closes[j] - fill_price if cand.direction == Direction.LONG else fill_price - closes[j]
                return ShadowResult(
                    candidate=cand, filled=True, fill_price=fill_price,
                    exit_price=closes[j], exit_reason="BIAS_FLIP",
                    r_multiple=pnl / r_base, mfe_r=mfe_r, bars_held=bars_held,
                )

        # End of data
        last_close = closes[-1]
        pnl = last_close - fill_price if cand.direction == Direction.LONG else fill_price - last_close
        return ShadowResult(
            candidate=cand, filled=True, fill_price=fill_price,
            exit_price=last_close, exit_reason="END_OF_DATA",
            r_multiple=pnl / r_base, mfe_r=mfe_r,
            bars_held=len(closes) - fill_idx,
        )

    def get_filter_summary(self) -> dict[str, FilterStats]:
        """Compute per-filter stats from simulation results.

        Each candidate contributes to stats for ALL its failed filters,
        enabling marginal-value analysis conditional on other filters.
        """
        by_filter: dict[str, list[ShadowResult]] = {}
        for r in self.results:
            for name in r.candidate.filter_names:
                by_filter.setdefault(name, []).append(r)

        rej_counts: dict[str, int] = {}
        for c in self.rejections:
            for name in c.filter_names:
                rej_counts[name] = rej_counts.get(name, 0) + 1

        summaries: dict[str, FilterStats] = {}
        for name, results in by_filter.items():
            filled = [r for r in results if r.filled]
            r_multiples = [r.r_multiple for r in filled]

            stats = FilterStats(filter_name=name)
            stats.rejected_count = rej_counts.get(name, len(results))
            stats.simulated_count = len(results)
            stats.filled_count = len(filled)

            if r_multiples:
                arr = np.array(r_multiples)
                stats.avg_shadow_r = float(np.mean(arr))
                stats.pct_above_1r = float(np.mean(arr > 1.0)) * 100
                stats.pct_above_2r = float(np.mean(arr >= 2.0)) * 100
                stats.net_missed_expectancy = float(np.sum(arr[arr > 0]))
                stats.net_avoided_loss = float(np.sum(np.abs(arr[arr < 0])))

            summaries[name] = stats

        return summaries
