from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def compute_ema(series: np.ndarray, period: int) -> np.ndarray:
    """EMA with SMA seed shared by live overlay and backtest mirror."""
    out = np.full_like(series, np.nan, dtype=float)
    if len(series) < period:
        return out
    out[period - 1] = np.mean(series[:period])
    alpha = 2.0 / (period + 1)
    for idx in range(period, len(series)):
        out[idx] = series[idx] * alpha + out[idx - 1] * (1 - alpha)
    return out


def allocate_weighted_targets(
    symbols: Sequence[str],
    *,
    signals: Mapping[str, bool],
    prices: Mapping[str, float],
    portfolio_equity: float,
    max_equity_pct: float,
    weights: Mapping[str, float] | None = None,
) -> dict[str, int]:
    """Allocate whole-share overlay targets from shared signal/weight semantics."""
    available = max(portfolio_equity * max_equity_pct, 0.0)
    if weights is None:
        bullish_weights = {symbol: 1.0 for symbol in symbols if signals.get(symbol, False)}
    else:
        bullish_weights = {
            symbol: float(weights.get(symbol, 1.0))
            for symbol in symbols
            if signals.get(symbol, False)
        }
    total_weight = sum(bullish_weights.values())

    targets: dict[str, int] = {}
    for symbol in symbols:
        price = float(prices.get(symbol, 0.0) or 0.0)
        if signals.get(symbol, False) and price > 0.0 and total_weight > 0.0:
            allocation = available * bullish_weights[symbol] / total_weight
            targets[symbol] = int(allocation / price)
        else:
            targets[symbol] = 0
    return targets
