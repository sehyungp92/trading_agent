"""Composite scoring for automated backtesting.

Weights:
  - Net profit (25%): absolute profitability, norm: log(1+R)/log(base)
  - Profit factor (20%): win quality, norm: log(PF) / log(base)
  - Edge t-stat (15%): statistical significance of R-multiple edge
  - Win rate (15%): consistency, norm: win_rate / ceiling
  - Calmar ratio (10%): risk-adjusted return, norm: calmar / ceiling
  - Frequency (10%): trade count, norm: clip(total_trades / freq_ceiling, 0, 1)
  - Inverse drawdown (5%): low DD reward, norm: 1.0 - max_dd / ceiling

Normalization ceilings are strategy-specific via ScoreNormalization.
Default ceilings are used for general stock strategies, while IARIC uses a
separate normalization profile tuned for its capital-efficient regime.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backtests.stock.analysis.metrics import PerformanceMetrics, compute_metrics
from backtests.stock.models import TradeRecord


@dataclass(frozen=True)
class ScoreNormalization:
    """Strategy-specific normalization ceilings for composite scoring."""

    calmar_ceiling: float = 50.0
    pf_log_base: float = 4.0
    dd_ceiling: float = 0.30
    return_log_base: float = 6.0
    wr_ceiling: float = 0.70
    tstat_ceiling: float = 8.0
    freq_ceiling: float = 120.0


DEFAULT_NORM = ScoreNormalization()

# IARIC normalization is wider on profitability/risk metrics and now uses a
# 150-trade ceiling for the immutable frequency component.
IARIC_NORM = ScoreNormalization(
    calmar_ceiling=100.0,
    pf_log_base=8.0,
    dd_ceiling=0.05,
    freq_ceiling=150.0,
)


@dataclass(frozen=True)
class CompositeScore:
    calmar_component: float
    pf_component: float
    inv_dd_component: float
    net_profit_component: float
    wr_component: float = 0.0
    edge_tstat_component: float = 0.0
    freq_component: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


_W_CALMAR = 0.10
_W_PF = 0.20
_W_INV_DD = 0.05
_W_NET_PROFIT = 0.25
_W_WR = 0.15
_W_EDGE = 0.15
_W_FREQ = 0.10


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def composite_score(
    metrics: PerformanceMetrics,
    initial_equity: float = 10_000.0,
    r_multiples: np.ndarray | None = None,
    norm: ScoreNormalization | None = None,
) -> CompositeScore:
    """Compute the composite score."""
    if norm is None:
        norm = DEFAULT_NORM

    if metrics.total_trades < 25:
        return CompositeScore(0, 0, 0, 0, rejected=True, reject_reason=f"Too few trades: {metrics.total_trades} < 25")
    if metrics.max_drawdown_pct > 0.35:
        return CompositeScore(0, 0, 0, 0, rejected=True, reject_reason=f"Max DD too high: {metrics.max_drawdown_pct:.1%} > 35%")
    if metrics.profit_factor < 0.8:
        return CompositeScore(0, 0, 0, 0, rejected=True, reject_reason=f"PF too low: {metrics.profit_factor:.2f} < 0.80")

    calmar_raw = _clip01(metrics.calmar / norm.calmar_ceiling)
    pf_raw = _clip01(math.log(metrics.profit_factor) / math.log(norm.pf_log_base)) if metrics.profit_factor > 1.0 else 0.0
    inv_dd_raw = _clip01(1.0 - metrics.max_drawdown_pct / norm.dd_ceiling)
    return_ratio = metrics.net_profit / initial_equity
    np_raw = _clip01(math.log(1.0 + return_ratio) / math.log(norm.return_log_base))
    wr_raw = _clip01(metrics.win_rate / norm.wr_ceiling)

    edge_raw = 0.0
    if r_multiples is not None and len(r_multiples) > 1:
        std_r = float(np.std(r_multiples, ddof=1))
        if std_r > 0:
            avg_r = float(np.mean(r_multiples))
            t_stat = (avg_r / std_r) * math.sqrt(len(r_multiples))
            edge_raw = _clip01(t_stat / norm.tstat_ceiling)

    freq_raw = _clip01(metrics.total_trades / norm.freq_ceiling)

    total = (
        _W_CALMAR * calmar_raw
        + _W_PF * pf_raw
        + _W_INV_DD * inv_dd_raw
        + _W_NET_PROFIT * np_raw
        + _W_WR * wr_raw
        + _W_EDGE * edge_raw
        + _W_FREQ * freq_raw
    )

    return CompositeScore(
        calmar_component=calmar_raw,
        pf_component=pf_raw,
        inv_dd_component=inv_dd_raw,
        net_profit_component=np_raw,
        wr_component=wr_raw,
        edge_tstat_component=edge_raw,
        freq_component=freq_raw,
        total=total,
    )


def extract_metrics(
    trades: list[TradeRecord],
    equity_curve: np.ndarray,
    timestamps: np.ndarray,
    initial_equity: float,
) -> PerformanceMetrics:
    """Standard metrics extraction from engine result."""
    if not trades:
        return PerformanceMetrics()

    pnls = np.array([t.pnl_net for t in trades])
    risks = np.array([t.risk_per_share * t.quantity for t in trades])
    hold_hours = np.array([t.hold_hours for t in trades])
    commissions = np.array([t.commission for t in trades])
    symbols = [t.symbol for t in trades]

    return compute_metrics(
        pnls,
        risks,
        hold_hours,
        commissions,
        equity_curve,
        timestamps,
        initial_equity,
        trade_symbols=symbols,
    )


def compute_r_multiples(trades: list[TradeRecord]) -> np.ndarray:
    """Compute per-trade R-multiples (pnl / risk)."""
    if not trades:
        return np.array([])
    pnls = np.array([t.pnl_net for t in trades])
    risks = np.array([t.risk_per_share * t.quantity for t in trades])
    return pnls / np.where(risks == 0, 1.0, risks)
