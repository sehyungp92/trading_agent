"""Worker shim for TPC round 8 30m-pullback optimisation."""
from __future__ import annotations

from pathlib import Path

from backtests.shared.auto.types import ScoredCandidate

from .round8_plugin import ROUND8_SCORING_WEIGHTS, Round8TPCPlugin, _round8_composite_score

_plugin: Round8TPCPlugin | None = None


def init_worker(data_dir_str: str, equity: float, start_date: str | None = None, end_date: str | None = None) -> None:
    global _plugin
    _plugin = Round8TPCPlugin(
        Path(data_dir_str),
        initial_equity=equity,
        max_workers=1,
        start_date=start_date,
        end_date=end_date,
    )


def score_candidate(args: tuple) -> ScoredCandidate:
    name, candidate_muts, base_muts, phase, weights, hard_rejects = args
    muts = dict(base_muts)
    muts.update(candidate_muts)
    clean_weights = weights or ROUND8_SCORING_WEIGHTS
    if _plugin:
        metrics, score = _plugin.score_mutations(muts, phase, hard_rejects, clean_weights)
    else:
        metrics = _empty_metrics()
        score = _round8_composite_score(metrics, hard_rejects, clean_weights)
    return ScoredCandidate(
        name=name,
        score=0.0 if score.rejected else score.total,
        rejected=score.rejected,
        reject_reason=score.reject_reason,
        metrics=dict(metrics),
    )


def _empty_metrics() -> dict[str, float]:
    return {
        "total_trades": 0.0,
        "net_return_pct": 0.0,
        "net_return_per_month": 0.0,
        "total_pnl_dollars": 0.0,
        "pnl_per_trade": 0.0,
        "profit_factor": 0.0,
        "dollar_profit_factor": 0.0,
        "avg_r": 0.0,
        "total_r": 0.0,
        "r_per_month": 0.0,
        "win_rate": 0.0,
        "max_dd_pct": 0.0,
        "sharpe": 0.0,
        "trades_per_month": 0.0,
        "expectancy_frequency": 0.0,
        "excellent_trades": 0.0,
        "excellent_trades_per_month": 0.0,
        "excellent_rate": 0.0,
        "two_r_plus_rate": 0.0,
        "mfe_capture": 0.0,
        "avg_mfe_r": 0.0,
        "avg_mae_r": 0.0,
        "never_worked_rate": 1.0,
        "low_mfe_loss_rate": 1.0,
        "right_then_lost_rate": 1.0,
        "top5_winner_share": 1.0,
        "dollar_top5_winner_share": 1.0,
        "max_symbol_trade_share": 1.0,
        "gld_trade_share": 1.0,
        "qqq_trade_count": 0.0,
        "qqq_excellent_trades": 0.0,
        "worst_symbol_avg_r": 0.0,
        "worst_year_pnl_pct": -99.0,
    }
