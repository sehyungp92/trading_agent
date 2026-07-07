"""Downturn composite scoring.

Seven score components, each tied to a structural objective:
  net_return      : whole-strategy expected return
  correction_pnl  : realized correction-window alpha
  edge            : trade quality via profit factor
  frequency       : enough trades to avoid brittle sparse configs
  coverage        : breadth of captured correction windows
  alpha_capture   : available downturn move captured, with weak-entry suppression
  risk            : drawdown-adjusted survivability
"""
from __future__ import annotations

from dataclasses import dataclass

from backtests.momentum.analysis.downturn_diagnostics import DownturnMetrics


@dataclass(frozen=True)
class DownturnCompositeScore:
    """Frozen composite score."""

    net_return: float = 0.0
    correction_pnl: float = 0.0
    edge: float = 0.0
    frequency: float = 0.0
    coverage: float = 0.0
    alpha_capture: float = 0.0
    risk: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


BASE_WEIGHTS = {
    "net_return": 0.20,
    "correction_pnl": 0.18,
    "edge": 0.15,
    "frequency": 0.12,
    "coverage": 0.12,
    "alpha_capture": 0.15,
    "risk": 0.08,
}


def _clip01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def _normalised_weights(weight_overrides: dict[str, float] | None) -> dict[str, float]:
    weights = dict(BASE_WEIGHTS)
    if weight_overrides:
        for key, value in weight_overrides.items():
            if key in weights:
                weights[key] = float(value)
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total > 0 else weights


def composite_score(
    metrics: DownturnMetrics,
    weight_overrides: dict[str, float] | None = None,
) -> DownturnCompositeScore:
    """Compute the seven-component optimization score."""
    weights = _normalised_weights(weight_overrides)

    if metrics.total_trades < 50:
        return DownturnCompositeScore(
            rejected=True, reject_reason=f"too_few_trades ({metrics.total_trades})",
        )
    if metrics.correction_pnl_pct < 0:
        return DownturnCompositeScore(
            rejected=True,
            reject_reason=f"negative_correction_pnl ({metrics.correction_pnl_pct:.2f}%)",
        )
    if metrics.max_dd_pct > 0.25:
        return DownturnCompositeScore(
            rejected=True,
            reject_reason=f"max_dd_exceeded ({metrics.max_dd_pct:.2%})",
        )
    if metrics.profit_factor < 1.0:
        return DownturnCompositeScore(
            rejected=True,
            reject_reason=f"profit_factor_below_1 ({metrics.profit_factor:.2f})",
        )

    net_return_c = _clip01(metrics.net_return_pct / 100.0)
    correction_pnl_c = _clip01(metrics.correction_pnl_pct / 80.0)
    edge_c = _clip01((metrics.profit_factor - 1.0) / 2.0)
    frequency_c = _clip01(metrics.total_trades / 160.0)
    coverage_c = _clip01(metrics.correction_coverage / 0.75)

    correction_capture_c = _clip01(metrics.correction_capture_ratio / 0.25)
    bear_capture_c = _clip01(metrics.bear_capture_ratio / 0.08)
    low_mfe_rate = getattr(metrics, "low_mfe_trade_rate", 0.0)
    weak_entry_suppression_c = _clip01(1.0 - low_mfe_rate / 0.55)
    alpha_capture_c = (
        0.45 * correction_capture_c
        + 0.35 * bear_capture_c
        + 0.20 * weak_entry_suppression_c
    )

    drawdown_c = _clip01(1.0 - metrics.max_dd_pct / 0.16)
    calmar_c = _clip01(metrics.calmar / 4.0)
    sharpe_c = _clip01((metrics.sharpe + 0.2) / 1.8)
    risk_c = 0.55 * drawdown_c + 0.25 * calmar_c + 0.20 * sharpe_c

    components = {
        "net_return": net_return_c,
        "correction_pnl": correction_pnl_c,
        "edge": edge_c,
        "frequency": frequency_c,
        "coverage": coverage_c,
        "alpha_capture": alpha_capture_c,
        "risk": risk_c,
    }
    total = sum(weights[key] * components[key] for key in BASE_WEIGHTS)

    return DownturnCompositeScore(
        net_return=net_return_c,
        correction_pnl=correction_pnl_c,
        edge=edge_c,
        frequency=frequency_c,
        coverage=coverage_c,
        alpha_capture=alpha_capture_c,
        risk=risk_c,
        total=total,
    )
