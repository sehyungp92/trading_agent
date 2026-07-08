from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

import numpy as np

from backtests.stock.auto.scoring import extract_metrics
from backtests.shared.auto.types import ScoredCandidate

from .phase_scoring import (
    merge_pullback_metrics,
    score_pullback_phase,
    score_v2r1_pullback_phase,
    score_v2r2_pullback_phase,
    score_v2r3_pullback_phase,
    score_v2r4_pullback_phase,
    score_v3r1_pullback_phase,
    score_v4r1_pullback_phase,
    score_v5r1_pullback_phase,
    score_v5r2_pullback_phase,
)

_worker_replay = None
_worker_config = None
_worker_equity: float = 0.0
_worker_phase: int = 0
_worker_hard_rejects: dict | None = None
_worker_scoring_weights: dict | None = None
_worker_round_name: str = "r4"


def init_worker(
    data_dir_str: str,
    start_date: str,
    end_date: str,
    equity: float,
    phase: int = 0,
    hard_rejects: dict | None = None,
    scoring_weights: dict | None = None,
    round_name: str = "r4",
) -> None:
    global _worker_replay, _worker_config, _worker_equity, _worker_phase
    global _worker_hard_rejects, _worker_scoring_weights, _worker_round_name

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.stock.config_iaric import IARICBacktestConfig
    from backtests.stock.data.replay_cache import load_research_replay_bundle

    data_dir = Path(data_dir_str)
    _worker_replay = load_research_replay_bundle(data_dir).data
    _worker_equity = equity
    _worker_phase = phase
    _worker_hard_rejects = hard_rejects or {}
    _worker_scoring_weights = scoring_weights or {}
    _worker_round_name = round_name
    _worker_config = IARICBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=equity,
        tier=3,
        data_dir=data_dir,
    )


def score_candidate(args: tuple[str, dict, dict]) -> ScoredCandidate:
    name, candidate_muts, base_muts = args

    try:
        from backtests.stock.auto.config_mutator import mutate_iaric_config
        from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_iaric_config(_worker_config, all_muts)
        result = IARICPullbackEngine(config, _worker_replay, collect_diagnostics=True).run()
        perf = extract_metrics(
            result.trades,
            result.equity_curve,
            result.timestamps,
            _worker_equity,
        )
        avg_r = float(np.mean([float(t.r_multiple) for t in result.trades])) if result.trades else 0.0
        reject_reason = _phase_reject_reason(perf, _worker_hard_rejects, avg_r=avg_r)
        if reject_reason:
            return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=reject_reason)

        merged_metrics = merge_pullback_metrics(
            perf,
            result.trades,
            candidate_ledger=result.candidate_ledger,
            selection_attribution=result.selection_attribution,
        )
        if _worker_round_name == "v5r2":
            score_fn = score_v5r2_pullback_phase
        elif _worker_round_name == "v5r1":
            score_fn = score_v5r1_pullback_phase
        elif _worker_round_name == "v4r1":
            score_fn = score_v4r1_pullback_phase
        elif _worker_round_name == "v3r1":
            score_fn = score_v3r1_pullback_phase
        elif _worker_round_name == "v2r4":
            score_fn = score_v2r4_pullback_phase
        elif _worker_round_name == "v2r3":
            score_fn = score_v2r3_pullback_phase
        elif _worker_round_name == "v2r2":
            score_fn = score_v2r2_pullback_phase
        elif _worker_round_name == "v2r1":
            score_fn = score_v2r1_pullback_phase
        else:
            score_fn = score_pullback_phase
        score = score_fn(_worker_phase, merged_metrics, _worker_scoring_weights)

        return ScoredCandidate(
            name=name,
            score=score,
            metrics=merged_metrics,
        )

    except Exception:
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=traceback.format_exc())


def _phase_reject_reason(metrics, hard_rejects: dict | None, *, avg_r: float | None = None) -> str:
    rejects = hard_rejects or {}

    min_trades = int(rejects.get("min_trades", 0))
    if metrics.total_trades < min_trades:
        return f"phase{_worker_phase}_too_few_trades ({metrics.total_trades} < {min_trades})"

    max_dd = rejects.get("max_dd_pct")
    if max_dd is not None and metrics.max_drawdown_pct > float(max_dd):
        return f"phase{_worker_phase}_max_dd ({metrics.max_drawdown_pct:.2%} > {float(max_dd):.2%})"

    min_pf = rejects.get("min_pf")
    if min_pf is not None and metrics.profit_factor < float(min_pf):
        return f"phase{_worker_phase}_low_pf ({metrics.profit_factor:.2f} < {float(min_pf):.2f})"

    min_net_profit = rejects.get("min_net_profit")
    if min_net_profit is not None and metrics.net_profit < float(min_net_profit):
        return f"phase{_worker_phase}_low_net_profit ({metrics.net_profit:.2f} < {float(min_net_profit):.2f})"

    min_sharpe = rejects.get("min_sharpe")
    if min_sharpe is not None and metrics.sharpe < float(min_sharpe):
        return f"phase{_worker_phase}_low_sharpe ({metrics.sharpe:.2f} < {float(min_sharpe):.2f})"

    min_expectancy = rejects.get("min_expectancy")
    if min_expectancy is not None and metrics.expectancy < float(min_expectancy):
        return f"phase{_worker_phase}_low_expectancy ({metrics.expectancy:.3f} < {float(min_expectancy):.3f})"

    _avg_r = avg_r if avg_r is not None else getattr(metrics, "avg_r", 0.0)

    min_avg_r_thresh = rejects.get("min_avg_r")
    if min_avg_r_thresh is not None and _avg_r < float(min_avg_r_thresh):
        return f"phase{_worker_phase}_low_avg_r ({_avg_r:.4f} < {float(min_avg_r_thresh):.4f})"

    min_expected_total_r = rejects.get("min_expected_total_r")
    if min_expected_total_r is not None:
        actual_etr = _avg_r * metrics.total_trades
        if actual_etr < float(min_expected_total_r):
            return f"phase{_worker_phase}_low_expected_total_r ({actual_etr:.2f} < {float(min_expected_total_r):.2f})"

    return ""
