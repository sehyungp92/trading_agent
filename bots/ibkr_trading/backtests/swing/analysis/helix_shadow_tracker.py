"""Shadow trade simulation for Helix rejected candidates.

Tracks what would have happened if a gate had NOT rejected a candidate.
Uses the Helix stop lifecycle: BE at +1R, chandelier trailing with
adaptive multiplier, R-based partials.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from strategies.swing.akc_helix import stops
from strategies.swing.akc_helix.config import (
    R_BE,
    R_PARTIAL_2P5,
    STALE_1H_BARS,
    TTL_1H_HOURS,
    TTL_4H_HOURS,
    SymbolConfig,
)
from strategies.swing.akc_helix.models import DailyState, Direction, Regime

logger = logging.getLogger(__name__)


@dataclass
class HelixShadowCandidate:
    """A candidate that was rejected by one or more gates."""

    symbol: str
    direction: int
    filter_names: list[str]
    time: datetime
    entry_price: float
    stop_price: float
    origin_tf: str = "1H"      # "1H" or "4H"
    setup_class: str = ""      # A/B/C/D

    @property
    def filter_name(self) -> str:
        return self.filter_names[0] if self.filter_names else ""


@dataclass
class HelixShadowResult:
    """Outcome of simulating a shadow candidate."""

    candidate: HelixShadowCandidate
    filled: bool = False
    fill_price: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    bars_held: int = 0


@dataclass
class FilterStats:
    """Aggregated stats for one filter/gate."""

    filter_name: str
    rejected_count: int = 0
    simulated_count: int = 0
    filled_count: int = 0
    avg_shadow_r: float = 0.0
    pct_above_1r: float = 0.0
    pct_above_2r: float = 0.0
    net_missed_expectancy: float = 0.0
    net_avoided_loss: float = 0.0


class HelixShadowTracker:
    """Track rejected Helix candidates and simulate their outcomes."""

    def __init__(self):
        self.rejections: list[HelixShadowCandidate] = []
        self.results: list[HelixShadowResult] = []

    def record_rejection(
        self,
        symbol: str,
        direction: int,
        filter_names: str | list[str],
        time: datetime,
        entry_price: float,
        stop_price: float,
        origin_tf: str = "1H",
        setup_class: str = "",
    ) -> None:
        """Log a rejected candidate with all failed gate names."""
        if isinstance(filter_names, str):
            filter_names = [filter_names]
        self.rejections.append(HelixShadowCandidate(
            symbol=symbol, direction=direction, filter_names=filter_names,
            time=time, entry_price=entry_price, stop_price=stop_price,
            origin_tf=origin_tf, setup_class=setup_class,
        ))

    def simulate_shadows(
        self,
        hourly_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        hourly_times: dict[str, np.ndarray],
        configs: dict[str, SymbolConfig],
        point_values: dict[str, float],
        daily_states: dict[str, dict[int, DailyState]] | None = None,
        daily_idx_maps: dict[str, np.ndarray] | None = None,
    ) -> list[HelixShadowResult]:
        """Forward-simulate each rejected candidate using Helix stop lifecycle."""
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
        cand: HelixShadowCandidate,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        times: np.ndarray,
        cfg: SymbolConfig,
        daily_states: dict[int, DailyState] | None,
        daily_idx_map: np.ndarray | None,
    ) -> HelixShadowResult:
        """Simulate one shadow candidate with Helix stop lifecycle."""
        cand_ts = np.datetime64(cand.time, 'ns') if not isinstance(cand.time, np.datetime64) else cand.time
        start_idx = int(np.searchsorted(times, cand_ts, side='right'))

        if start_idx >= len(times):
            return HelixShadowResult(candidate=cand)

        r_base = abs(cand.entry_price - cand.stop_price)
        if r_base <= 0:
            return HelixShadowResult(candidate=cand)

        # TTL for fill window
        ttl = TTL_4H_HOURS if cand.origin_tf == "4H" else TTL_1H_HOURS

        # Phase 1: check entry fill within TTL
        fill_price = 0.0
        fill_idx = -1
        for i in range(start_idx, min(start_idx + ttl, len(opens))):
            if cand.direction == Direction.LONG:
                if highs[i] >= cand.entry_price:
                    fill_price = max(float(opens[i]), cand.entry_price)
                    fill_idx = i
                    break
            else:
                if lows[i] <= cand.entry_price:
                    fill_price = min(float(opens[i]), cand.entry_price)
                    fill_idx = i
                    break

        if fill_idx < 0:
            return HelixShadowResult(candidate=cand, filled=False)

        # Phase 2: full Helix stop lifecycle
        current_stop = cand.stop_price
        mfe_price = fill_price
        mfe_r = 0.0
        be_triggered = False
        trail_active = False
        has_daily = daily_states is not None and daily_idx_map is not None
        chandelier_highs: list[float] = []
        chandelier_lows: list[float] = []

        for j in range(fill_idx + 1, len(closes)):
            bars_held = j - fill_idx
            chandelier_highs.append(float(highs[j]))
            chandelier_lows.append(float(lows[j]))

            # Update MFE
            if cand.direction == Direction.LONG:
                if highs[j] > mfe_price:
                    mfe_price = float(highs[j])
                cur_mfe = (mfe_price - fill_price) / r_base
            else:
                if lows[j] < mfe_price:
                    mfe_price = float(lows[j])
                cur_mfe = (fill_price - mfe_price) / r_base
            mfe_r = max(mfe_r, cur_mfe)

            # Current R
            if cand.direction == Direction.LONG:
                r_now = (float(closes[j]) - fill_price) / r_base
            else:
                r_now = (fill_price - float(closes[j])) / r_base

            # Daily state
            d = None
            if has_daily and j < len(daily_idx_map):
                d_idx = int(daily_idx_map[j])
                d = daily_states.get(d_idx)

            # BE at +1R (spec s13.2)
            if not be_triggered and cur_mfe >= R_BE:
                atr_1h_approx = r_base * 2.0  # rough approximation
                be_stop = stops.compute_be_stop(
                    Direction(cand.direction), fill_price,
                    atr_1h_approx, cfg.tick_size,
                )
                if cand.direction == Direction.LONG and be_stop > current_stop:
                    current_stop = be_stop
                    be_triggered = True
                elif cand.direction == Direction.SHORT and be_stop < current_stop:
                    current_stop = be_stop
                    be_triggered = True

            # Chandelier trailing (simplified)
            if be_triggered and cur_mfe >= R_BE:
                trail_active = True
                lookback = min(cfg.chandelier_lookback, len(chandelier_highs))
                trail_mult = max(2.0, 4.0 - r_now / 5.0)
                atr_approx = r_base * 2.0
                chandelier = stops.compute_chandelier_stop(
                    Direction(cand.direction),
                    chandelier_highs, chandelier_lows,
                    lookback, atr_approx, trail_mult, cfg.tick_size,
                )
                if cand.direction == Direction.LONG and chandelier > current_stop:
                    current_stop = chandelier
                elif cand.direction == Direction.SHORT and chandelier < current_stop:
                    current_stop = chandelier

            # Check stop fill
            if cand.direction == Direction.LONG:
                if lows[j] <= current_stop:
                    pnl = current_stop - fill_price
                    return HelixShadowResult(
                        candidate=cand, filled=True, fill_price=fill_price,
                        exit_price=current_stop, exit_reason="STOP",
                        r_multiple=pnl / r_base, mfe_r=mfe_r, bars_held=bars_held,
                    )
            else:
                if highs[j] >= current_stop:
                    pnl = fill_price - current_stop
                    return HelixShadowResult(
                        candidate=cand, filled=True, fill_price=fill_price,
                        exit_price=current_stop, exit_reason="STOP",
                        r_multiple=pnl / r_base, mfe_r=mfe_r, bars_held=bars_held,
                    )

            # Stale exit
            if bars_held >= STALE_1H_BARS and r_now < 0.5:
                if cand.direction == Direction.LONG:
                    pnl = float(closes[j]) - fill_price
                else:
                    pnl = fill_price - float(closes[j])
                return HelixShadowResult(
                    candidate=cand, filled=True, fill_price=fill_price,
                    exit_price=float(closes[j]), exit_reason="STALE",
                    r_multiple=pnl / r_base, mfe_r=mfe_r, bars_held=bars_held,
                )

            # Regime flip exit
            if d is not None:
                if cand.direction == Direction.LONG and d.regime == Regime.BEAR:
                    pnl = float(closes[j]) - fill_price
                    return HelixShadowResult(
                        candidate=cand, filled=True, fill_price=fill_price,
                        exit_price=float(closes[j]), exit_reason="REGIME_FLIP",
                        r_multiple=pnl / r_base, mfe_r=mfe_r, bars_held=bars_held,
                    )
                elif cand.direction == Direction.SHORT and d.regime == Regime.BULL:
                    pnl = fill_price - float(closes[j])
                    return HelixShadowResult(
                        candidate=cand, filled=True, fill_price=fill_price,
                        exit_price=float(closes[j]), exit_reason="REGIME_FLIP",
                        r_multiple=pnl / r_base, mfe_r=mfe_r, bars_held=bars_held,
                    )

        # End of data
        last_close = float(closes[-1])
        if cand.direction == Direction.LONG:
            pnl = last_close - fill_price
        else:
            pnl = fill_price - last_close
        return HelixShadowResult(
            candidate=cand, filled=True, fill_price=fill_price,
            exit_price=last_close, exit_reason="END_OF_DATA",
            r_multiple=pnl / r_base, mfe_r=mfe_r,
            bars_held=len(closes) - fill_idx,
        )

    def get_filter_summary(self) -> dict[str, FilterStats]:
        """Compute per-gate stats from simulation results."""
        by_filter: dict[str, list[HelixShadowResult]] = {}
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
