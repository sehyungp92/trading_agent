from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .phase_candidates import ROUND_TARGETS, SCORE_WEIGHTS

SCORE_COMPONENTS: tuple[str, ...] = tuple(SCORE_WEIGHTS.keys())


@dataclass(frozen=True)
class PortfolioScore:
    components: dict[str, float]
    total: float
    rejected: bool = False
    reject_reason: str = ""


def score_portfolio_metrics(
    metrics: dict[str, float],
    *,
    scoring_weights: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> PortfolioScore:
    weights = _normalise_weights(scoring_weights or SCORE_WEIGHTS)
    if len(weights) > 7:
        raise ValueError(f"Stock portfolio synergy score has {len(weights)} components; max is 7.")

    rejection = _reject_reason(metrics, hard_rejects or {})
    components = _components(metrics)
    total = sum(weights.get(name, 0.0) * components.get(name, 0.0) for name in SCORE_COMPONENTS)
    if rejection:
        return PortfolioScore(components, 0.0, rejected=True, reject_reason=rejection)
    return PortfolioScore(components, total)


def _components(metrics: dict[str, float]) -> dict[str, float]:
    net_return_pct = _metric(metrics, "net_return_pct")
    total_r_per_month = _metric(metrics, "total_r_per_month")
    active_tpm = _metric(metrics, "active_trades_per_month")
    max_dd_pct = _metric(metrics, "max_drawdown_pct")
    pf = _metric(metrics, "profit_factor")
    trade_capture = _metric(metrics, "trade_capture_ratio")
    positive_block = _metric(metrics, "positive_alpha_block_rate")
    candidate_discrimination = _metric(metrics, "candidate_discrimination")
    active_strategies = _metric(metrics, "active_strategy_count")
    max_trade_share = _metric(metrics, "max_strategy_trade_share")
    max_risk_share = _metric(metrics, "max_strategy_risk_share")
    sharpe = _metric(metrics, "sharpe")
    positive_slices = _metric(metrics, "positive_slices")

    dd_quality = _clip(1.0 - max_dd_pct / ROUND_TARGETS["hard_max_drawdown_pct"])
    capture_quality = _clip(0.55 * trade_capture + 0.45 * (1.0 - positive_block / 0.24))
    discrimination_quality = _clip(candidate_discrimination)
    strategy_count_quality = _clip(active_strategies / 2.0)
    trade_share_quality = _clip(1.0 - max(0.0, max_trade_share - 0.60) / 0.30)
    risk_share_quality = _clip(1.0 - max(0.0, max_risk_share - 0.60) / 0.30)
    sharpe_quality = _clip(sharpe / 5.5)
    slice_quality = _clip(positive_slices / 4.0)

    return {
        "alpha_return": _clip(0.55 * net_return_pct / 2.50 + 0.45 * total_r_per_month / 16.0),
        "trade_frequency": _clip(active_tpm / 60.0),
        "drawdown_control": dd_quality,
        "profit_factor_quality": _clip((pf - 1.0) / 2.75),
        "synergy_capture": _clip(0.65 * capture_quality + 0.35 * discrimination_quality),
        "allocation_balance": _clip(
            0.40 * strategy_count_quality
            + 0.30 * trade_share_quality
            + 0.30 * risk_share_quality
        ),
        "robustness": _clip(0.45 * sharpe_quality + 0.35 * dd_quality + 0.20 * slice_quality),
    }


def _reject_reason(metrics: dict[str, float], hard_rejects: dict[str, float]) -> str:
    max_dd_limit = float(hard_rejects.get("max_drawdown_pct", ROUND_TARGETS["hard_max_drawdown_pct"]))
    if _metric(metrics, "max_drawdown_pct") > max_dd_limit:
        return f"max drawdown {_metric(metrics, 'max_drawdown_pct'):.2%} > {max_dd_limit:.2%}"

    min_strategies = float(hard_rejects.get("min_active_strategies", 2.0))
    if _metric(metrics, "active_strategy_count") < min_strategies:
        return f"active strategies {_metric(metrics, 'active_strategy_count'):.0f} < {min_strategies:.0f}"

    min_tpm = hard_rejects.get("min_active_trades_per_month")
    if min_tpm is not None and _metric(metrics, "active_trades_per_month") < float(min_tpm):
        return f"active trades/month {_metric(metrics, 'active_trades_per_month'):.2f} < {float(min_tpm):.2f}"

    min_pf = hard_rejects.get("min_profit_factor")
    if min_pf is not None and _metric(metrics, "profit_factor") < float(min_pf):
        return f"profit factor {_metric(metrics, 'profit_factor'):.2f} < {float(min_pf):.2f}"

    max_positive_block = hard_rejects.get("max_positive_alpha_block_rate")
    if max_positive_block is not None and _metric(metrics, "positive_alpha_block_rate") > float(max_positive_block):
        return (
            "positive-alpha block rate "
            f"{_metric(metrics, 'positive_alpha_block_rate'):.2%} > {float(max_positive_block):.2%}"
        )

    max_trade_share = hard_rejects.get("max_strategy_trade_share")
    if max_trade_share is not None and _metric(metrics, "max_strategy_trade_share") > float(max_trade_share):
        return (
            f"strategy trade share {_metric(metrics, 'max_strategy_trade_share'):.2%} "
            f"> {float(max_trade_share):.2%}"
        )

    return ""


def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    unexpected = sorted(set(weights) - set(SCORE_COMPONENTS))
    if unexpected:
        raise ValueError(f"Unknown score component(s): {', '.join(unexpected)}")
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        return {name: 1.0 / len(SCORE_COMPONENTS) for name in SCORE_COMPONENTS}
    return {name: float(weights.get(name, 0.0)) / total for name in SCORE_COMPONENTS}


def _metric(metrics: dict[str, float], key: str) -> float:
    try:
        value = float(metrics.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if isfinite(value) else 0.0


def _clip(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return min(max(value, 0.0), 1.0)

