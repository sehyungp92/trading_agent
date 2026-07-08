from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np


@dataclass
class ScalpPerformanceMetrics:
    total_trades: int = 0
    net_profit: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_commission: float = 0.0
    expectancy_dollar: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    trades_per_month: float = 0.0
    avg_r: float = 0.0
    edge_velocity: float = 0.0


def extract_metrics(
    trades: Sequence,
    equity_curve: Sequence[float],
    timestamps: Sequence,
    initial_equity: float,
) -> ScalpPerformanceMetrics:
    if not trades:
        return ScalpPerformanceMetrics()
    pnls = np.asarray([float(getattr(trade, "pnl_dollars", 0.0)) for trade in trades], dtype=float)
    commissions = np.asarray([float(getattr(trade, "commission", 0.0)) for trade in trades], dtype=float)
    r_values = np.asarray([float(getattr(trade, "r_multiple", 0.0)) for trade in trades], dtype=float)
    gross_profit = float(np.sum(pnls[pnls > 0]))
    gross_loss = float(abs(np.sum(pnls[pnls < 0])))
    equity = np.asarray(equity_curve, dtype=float)
    dd_pct = _max_drawdown_pct(equity)
    months = _months_covered(timestamps)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1.0) if len(equity) > 1 else np.array([])
    sharpe = _sharpe(returns)
    downside = returns[returns < 0]
    sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(252 * 390)) if len(downside) > 1 and np.std(downside) > 0 else 0.0
    net_profit = float(np.sum(pnls))
    calmar = (net_profit / initial_equity) / dd_pct if initial_equity > 0 and dd_pct > 0 else 0.0
    tpm = len(trades) / months if months > 0 else float(len(trades))
    expectancy = float(np.mean(pnls))
    return ScalpPerformanceMetrics(
        total_trades=len(trades),
        net_profit=net_profit,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        total_commission=float(np.sum(commissions)),
        expectancy_dollar=expectancy,
        profit_factor=gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        win_rate=float(np.mean(pnls > 0)),
        max_drawdown_pct=dd_pct,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        trades_per_month=tpm,
        avg_r=float(np.mean(r_values)) if len(r_values) else 0.0,
        edge_velocity=expectancy * tpm,
    )


def _max_drawdown_pct(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1.0)
    return float(np.max(dd))


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or np.std(returns) <= 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 390))


def _months_covered(timestamps: Sequence) -> float:
    if timestamps is None or len(timestamps) == 0:
        return 0.0
    first = _coerce_datetime(timestamps[0])
    last = _coerce_datetime(timestamps[-1])
    if first is None or last is None or last <= first:
        return 1.0
    return max((last - first).days / 30.4375, 1.0)


def _coerce_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        import pandas as pd

        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.to_pydatetime()
    except Exception:
        return None
