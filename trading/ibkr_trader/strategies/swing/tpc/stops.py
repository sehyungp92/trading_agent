from __future__ import annotations

import numpy as np

from strategies.swing._shared.etf_core import BarWindow
from strategies.swing._shared.models import Direction
from strategies.swing.tpc.config import TPCSymbolConfig
from strategies.swing.tpc.signals import PullbackCandidate


def compute_initial_stop(
    pullback: PullbackCandidate,
    direction: Direction,
    atr_4h: float,
    cfg: TPCSymbolConfig,
) -> float:
    buffer = cfg.stop_buffer_atr_mult * atr_4h
    if direction == Direction.LONG:
        return pullback.low - buffer
    return pullback.high + buffer


def validate_stop(stop: float, entry: float, atr_4h: float, cfg: TPCSymbolConfig) -> bool:
    return atr_4h > 0 and abs(entry - stop) <= cfg.max_stop_atr_mult * atr_4h


def compute_breakeven_stop(entry: float, direction: Direction) -> float:
    del direction
    return entry


def compute_trailing_stop(
    current_stop: float,
    bars_1h: BarWindow | None,
    bars_30m: BarWindow | None,
    direction: Direction,
    vwap: float,
    daily_levels: list[float] | None = None,
) -> float:
    candidates = [current_stop]
    if bars_1h is not None and len(bars_1h) >= 3:
        candidates.append(float(np.nanmin(bars_1h.lows[-3:]) if direction == Direction.LONG else np.nanmax(bars_1h.highs[-3:])))
    if bars_30m is not None and len(bars_30m) >= 4:
        candidates.append(float(np.nanmin(bars_30m.lows[-4:]) if direction == Direction.LONG else np.nanmax(bars_30m.highs[-4:])))
    if not np.isnan(vwap):
        candidates.append(vwap)
    if daily_levels:
        candidates.extend(daily_levels)
    if direction == Direction.LONG:
        return max(c for c in candidates if np.isfinite(c))
    return min(c for c in candidates if np.isfinite(c))

