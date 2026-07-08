"""Greedy forward selection for optimal swing portfolio config.

Algorithm:
  1. Start with live_parity base config (default UnifiedBacktestConfig)
  2. Score the base → baseline_score
  3. For each round, test every remaining candidate IN PARALLEL
  4. Keep the best candidate if it improves score; stop when none do
  5. Output the final optimal mutations and comparison table

Uses multiprocessing to evaluate candidates concurrently within each round.
Each worker loads data independently (6s one-time cost), then reuses it.
With 12 cores and 10 candidates, round 1 runs in ~350s instead of ~3500s.

Usage:
    from backtests.swing.auto.greedy_optimize import run_greedy
    result = run_greedy(data_dir, candidates, initial_equity=10_000)
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker process globals (initialized once per worker via _init_worker)
# ---------------------------------------------------------------------------
_worker_data = None
_worker_equity: float = 0.0
_worker_return_basis: str = "equity_curve"
_worker_scoring_kwargs: dict = {}
_worker_score_profile: str = "generic"


def _init_worker(
    data_dir_str: str,
    equity: float,
    return_basis: str = "equity_curve",
    scoring_kwargs: dict | None = None,
    score_profile: str = "generic",
) -> None:
    """Initialize a worker process: install aliases, load data."""
    global _worker_data, _worker_equity, _worker_return_basis, _worker_scoring_kwargs, _worker_score_profile

    from backtests.swing.config_unified import UnifiedBacktestConfig
    from backtests.swing.engine.unified_portfolio_engine import load_unified_data

    _worker_equity = equity
    _worker_return_basis = return_basis
    _worker_scoring_kwargs = dict(scoring_kwargs or {})
    _worker_score_profile = score_profile
    config = UnifiedBacktestConfig(initial_equity=equity, data_dir=Path(data_dir_str))
    _worker_data = load_unified_data(config)


def _worker_score(mutations: dict) -> tuple[float, bool, str]:
    """Score a config in a worker process. Returns (score, rejected, reject_reason)."""
    global _worker_data, _worker_equity, _worker_return_basis, _worker_scoring_kwargs, _worker_score_profile

    from backtests.swing.auto.config_mutator import mutate_unified_config
    from backtests.swing.auto.scoring import extract_metrics
    from backtests.swing.config_unified import UnifiedBacktestConfig
    from backtests.swing.engine.unified_portfolio_engine import run_unified

    try:
        config = UnifiedBacktestConfig(initial_equity=_worker_equity)
        if mutations:
            config = mutate_unified_config(config, mutations)
        result = run_unified(_worker_data, config)
        trades = _collect_trades(result)
        metrics = extract_metrics(
            trades, result.combined_equity,
            result.combined_timestamps, _worker_equity,
        )
        net_profit_override = _score_net_profit(
            config,
            result,
            _worker_equity,
            _worker_return_basis,
        )
        score = _score_result(
            metrics,
            config,
            result,
            _worker_equity,
            net_profit_override=net_profit_override,
            score_profile=_worker_score_profile,
            scoring_kwargs=_worker_scoring_kwargs,
        )
        return score.total if not score.rejected else 0.0, score.rejected, score.reject_reason
    except Exception:
        return 0.0, True, traceback.format_exc()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GreedyRound:
    """Result of a single greedy selection round."""
    round_num: int
    candidates_tested: int
    best_name: str
    best_score: float
    best_delta_pct: float
    kept: bool
    all_scores: list[tuple[str, float, float]]  # (name, score, delta_pct)


@dataclass
class GreedyResult:
    """Full result of greedy forward selection."""
    base_score: float
    final_mutations: dict
    final_score: float
    kept_features: list[str]
    rounds: list[GreedyRound]
    final_trades: int = 0
    final_pf: float = 0.0
    final_dd_pct: float = 0.0
    final_return_pct: float = 0.0
    final_sharpe: float = 0.0


@dataclass(frozen=True)
class _ScoreResult:
    total: float
    rejected: bool = False
    reject_reason: str = ""


DEFAULT_PORTFOLIO_SYNERGY_WEIGHTS: dict[str, float] = {
    "alpha_quality": 0.27,
    "frequency_quality": 0.23,
    "drawdown_quality": 0.18,
    "pf_quality": 0.11,
    "balance_quality": 0.10,
    "capture_quality": 0.07,
    "robustness_quality": 0.04,
}


# ---------------------------------------------------------------------------
# Trade collection helper
# ---------------------------------------------------------------------------

def _collect_trades(result) -> list:
    """Collect trades from a UnifiedPortfolioResult."""
    all_trades = []
    for attr in ('atrss_trades', 'helix_trades', 'tpc_trades'):
        trades = getattr(result, attr, [])
        if isinstance(trades, list):
            all_trades.extend(trades)
    strategy_results = getattr(result, 'strategy_results', {})
    if isinstance(strategy_results, dict) and not all_trades:
        for sr in strategy_results.values():
            all_trades.extend(getattr(sr, 'trades', []))
    return all_trades


def _score_net_profit(config, result, initial_equity: float, return_basis: str) -> float | None:
    """Return a scoring PnL override, or None to use the default equity curve."""
    if return_basis == "equity_curve":
        return None
    if return_basis == "static_initial_strategy_risk":
        return _static_initial_strategy_net_profit(config, result, initial_equity)
    raise ValueError(f"Unknown swing greedy return_basis: {return_basis}")


def _static_initial_strategy_net_profit(config, result, initial_equity: float) -> float:
    """Score strategy R on initial unit risk to avoid rewarding later compounding."""
    strategy_results = getattr(result, "strategy_results", {}) or {}
    slot_attrs = {
        "ATRSS": "atrss",
        "AKC_HELIX": "helix",
        "TPC": "tpc",
    }
    total = 0.0
    for strategy_id, slot_attr in slot_attrs.items():
        sr = strategy_results.get(strategy_id)
        if sr is None:
            continue
        slot = getattr(config, slot_attr, None)
        unit_risk_pct = float(getattr(slot, "unit_risk_pct", 0.0) or 0.0)
        total_r = float(getattr(sr, "total_r", 0.0) or 0.0)
        total += total_r * float(initial_equity) * unit_risk_pct
    return total


def _score_result(
    metrics,
    config,
    result,
    initial_equity: float,
    *,
    net_profit_override: float | None,
    score_profile: str,
    scoring_kwargs: dict,
):
    if score_profile == "generic":
        from backtests.swing.auto.scoring import composite_score

        return composite_score(
            metrics,
            initial_equity,
            equity_curve=result.combined_equity,
            net_profit_override=net_profit_override,
            **scoring_kwargs,
        )
    if score_profile == "portfolio_synergy_alpha_frequency":
        return _portfolio_synergy_score(
            metrics,
            config,
            result,
            initial_equity,
            net_profit_override=net_profit_override,
            scoring_kwargs=scoring_kwargs,
        )
    raise ValueError(f"Unknown swing greedy score_profile: {score_profile}")


def _portfolio_synergy_score(
    metrics,
    config,
    result,
    initial_equity: float,
    *,
    net_profit_override: float | None,
    scoring_kwargs: dict,
) -> _ScoreResult:
    """Score portfolio synergy for alpha, frequency, balance, and controlled risk."""
    hard_dd = float(scoring_kwargs.get("max_drawdown_hard_pct", 0.15))
    drawdown_comfort = scoring_kwargs.get("drawdown_comfort_pct")
    alpha_return_target_pct = max(float(scoring_kwargs.get("alpha_return_target_pct", 240.0)), 1e-9)
    min_pf = float(scoring_kwargs.get("min_profit_factor", 1.75))
    min_trades = int(scoring_kwargs.get("min_trades", 120))
    required_strategies = tuple(scoring_kwargs.get("required_strategies", ()))
    min_required_strategy_trades = int(scoring_kwargs.get("min_required_strategy_trades", 1))
    max_single_strategy_static_pnl_share = scoring_kwargs.get("max_single_strategy_static_pnl_share")
    trades_per_month_target = max(float(scoring_kwargs.get("trades_per_month_target", 8.0)), 1e-9)
    total_r_per_month_target = max(float(scoring_kwargs.get("total_r_per_month_target", 4.5)), 1e-9)
    accept_rate_target = max(float(scoring_kwargs.get("accept_rate_target", 0.22)), 1e-9)
    strategy_active_trade_floor = max(float(scoring_kwargs.get("strategy_active_trade_floor", 10.0)), 1e-9)
    strategy_min_trade_target = max(float(scoring_kwargs.get("strategy_min_trade_target", 20.0)), 1e-9)
    pf_quality_floor = float(scoring_kwargs.get("pf_quality_floor", 1.4))
    pf_quality_target = max(float(scoring_kwargs.get("pf_quality_target", 3.6)), pf_quality_floor + 1e-9)
    sharpe_quality_target = max(float(scoring_kwargs.get("sharpe_quality_target", 2.2)), 1e-9)
    strategy_results = getattr(result, "strategy_results", {}) or {}
    if metrics.max_drawdown_pct > hard_dd:
        return _ScoreResult(0.0, True, f"Max DD too high: {metrics.max_drawdown_pct:.1%} > {hard_dd:.1%}")
    if metrics.profit_factor < min_pf:
        return _ScoreResult(0.0, True, f"PF too low: {metrics.profit_factor:.2f} < {min_pf:.2f}")
    if metrics.total_trades < min_trades:
        return _ScoreResult(0.0, True, f"Too few trades: {metrics.total_trades} < {min_trades}")
    for strategy_id in required_strategies:
        sr = strategy_results.get(strategy_id)
        strategy_trades = int(getattr(sr, "total_trades", 0) or 0) if sr is not None else 0
        if strategy_trades < min_required_strategy_trades:
            return _ScoreResult(
                0.0,
                True,
                (
                    f"{strategy_id} too few trades: {strategy_trades} "
                    f"< {min_required_strategy_trades}"
                ),
            )

    net_profit = float(net_profit_override if net_profit_override is not None else metrics.net_profit)
    return_pct = net_profit / initial_equity * 100.0 if initial_equity > 0 else 0.0
    months = _backtest_months(getattr(result, "combined_timestamps", None))
    trades_per_month = float(metrics.total_trades) / months if months > 0 else 0.0

    total_fired = 0.0
    total_accepted = 0.0
    total_r = 0.0
    trade_counts: list[float] = []
    static_pnls: list[float] = []
    for strategy_id in ("ATRSS", "AKC_HELIX", "TPC"):
        sr = strategy_results.get(strategy_id)
        if sr is None:
            trade_counts.append(0.0)
            static_pnls.append(0.0)
            continue
        fired = float(
            getattr(
                sr,
                "entry_requests",
                getattr(sr, "entry_signals_fired", 0.0),
            )
            or 0.0
        )
        accepted = float(getattr(sr, "entries_accepted_by_portfolio", 0.0) or 0.0)
        total_fired += max(fired, accepted)
        total_accepted += accepted
        total_r += float(getattr(sr, "total_r", 0.0) or 0.0)
        trade_counts.append(float(getattr(sr, "total_trades", 0.0) or 0.0))
        static_pnls.append(
            max(
                _strategy_static_pnl(config, strategy_id, getattr(sr, "total_r", 0.0), initial_equity),
                0.0,
            )
        )
    positive_static_total = sum(static_pnls)
    if max_single_strategy_static_pnl_share is not None and positive_static_total > 0.0:
        max_share = max(static_pnls) / positive_static_total
        if max_share > float(max_single_strategy_static_pnl_share):
            return _ScoreResult(
                0.0,
                True,
                (
                    f"Single-strategy static PnL share too high: {max_share:.1%} "
                    f"> {float(max_single_strategy_static_pnl_share):.1%}"
                ),
            )

    total_r_per_month = total_r / months if months > 0 else 0.0
    accept_rate = total_accepted / total_fired if total_fired > 0 else 0.0
    pf_quality = _clip((float(metrics.profit_factor) - pf_quality_floor) / (pf_quality_target - pf_quality_floor))
    drawdown_quality = _drawdown_quality(metrics.max_drawdown_pct, hard_dd, drawdown_comfort)
    frequency_quality = (
        0.55 * _clip(trades_per_month / trades_per_month_target)
        + 0.45 * _clip(total_r_per_month / total_r_per_month_target)
    )
    alpha_quality = _clip(return_pct / alpha_return_target_pct)
    balance_quality = (
        0.55 * _participation_quality(
            trade_counts,
            active_floor=strategy_active_trade_floor,
            min_trade_target=strategy_min_trade_target,
        )
        + 0.45 * _entropy_quality(static_pnls)
    )
    capture_quality = 0.60 * _clip(accept_rate / accept_rate_target) + 0.40 * pf_quality
    robustness_quality = 0.50 * _clip(float(metrics.sharpe) / sharpe_quality_target) + 0.50 * drawdown_quality
    components = {
        "alpha_quality": alpha_quality,
        "frequency_quality": frequency_quality,
        "drawdown_quality": drawdown_quality,
        "pf_quality": pf_quality,
        "balance_quality": balance_quality,
        "capture_quality": capture_quality,
        "robustness_quality": robustness_quality,
    }
    weights = _portfolio_synergy_weights(scoring_kwargs.get("score_weights"))
    total = sum(weights[key] * components[key] for key in weights)
    return _ScoreResult(total)


def _drawdown_quality(max_drawdown_pct: float, hard_dd: float, comfort_pct: float | None) -> float:
    if comfort_pct is None:
        return _clip((hard_dd - max_drawdown_pct) / max(hard_dd - 0.06, 1e-9))
    comfort = min(max(float(comfort_pct), 0.0), hard_dd)
    if max_drawdown_pct <= comfort:
        return 1.0
    return _clip((hard_dd - max_drawdown_pct) / max(hard_dd - comfort, 1e-9))


def _strategy_static_pnl(config, strategy_id: str, total_r: float, initial_equity: float) -> float:
    slot_attrs = {"ATRSS": "atrss", "AKC_HELIX": "helix", "TPC": "tpc"}
    slot = getattr(config, slot_attrs.get(strategy_id, ""), None)
    unit_risk_pct = float(getattr(slot, "unit_risk_pct", 0.0) or 0.0)
    return float(total_r or 0.0) * float(initial_equity) * unit_risk_pct


def _portfolio_synergy_weights(overrides) -> dict[str, float]:
    weights = dict(DEFAULT_PORTFOLIO_SYNERGY_WEIGHTS)
    if overrides:
        for key, value in dict(overrides).items():
            if key not in weights:
                raise ValueError(f"Unknown portfolio synergy score weight: {key}")
            weights[key] = max(float(value), 0.0)
    total = sum(weights.values())
    if total <= 0.0:
        return dict(DEFAULT_PORTFOLIO_SYNERGY_WEIGHTS)
    return {key: value / total for key, value in weights.items()}


def _participation_quality(
    trade_counts: list[float],
    *,
    active_floor: float = 10.0,
    min_trade_target: float = 20.0,
) -> float:
    if not trade_counts:
        return 0.0
    active = sum(1 for count in trade_counts if count >= active_floor)
    min_trade_quality = _clip(min(trade_counts) / min_trade_target)
    return 0.70 * (active / len(trade_counts)) + 0.30 * min_trade_quality


def _entropy_quality(values: list[float]) -> float:
    positives = [max(float(value), 0.0) for value in values]
    total = sum(positives)
    if total <= 0.0 or len(positives) <= 1:
        return 0.0
    shares = [value / total for value in positives if value > 0.0]
    entropy = -sum(share * float(np.log(share)) for share in shares)
    return _clip(entropy / float(np.log(len(positives))))


def _backtest_months(timestamps) -> float:
    if timestamps is None or len(timestamps) < 2:
        return 1.0
    arr = np.asarray(timestamps)
    if np.issubdtype(arr.dtype, np.datetime64):
        start = arr[0].astype("datetime64[ns]")
        end = arr[-1].astype("datetime64[ns]")
        days = float((end - start) / np.timedelta64(1, "D"))
    elif np.issubdtype(arr.dtype, np.number):
        delta = float(arr[-1] - arr[0])
        # Engine timestamps are commonly nanoseconds since epoch when numeric.
        scale = 1e9 if abs(delta) > 1e12 else 1.0
        days = delta / scale / 86_400.0
    else:
        import pandas as pd

        start_ts = _coerce_timestamp(arr[0], pd)
        end_ts = _coerce_timestamp(arr[-1], pd)
        if start_ts.tzinfo is not None:
            start_ts = start_ts.tz_convert("UTC").tz_localize(None)
        if end_ts.tzinfo is not None:
            end_ts = end_ts.tz_convert("UTC").tz_localize(None)
        days = (end_ts - start_ts).total_seconds() / 86_400.0
    return max(days / 30.4375, 1.0)


def _coerce_timestamp(value, pd):
    """Normalize mixed replay timestamps without assuming one array dtype."""

    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, (int, np.integer, float, np.floating)):
        # Unified replay timestamps are nanoseconds when magnitudes are epoch-sized.
        unit = "ns" if abs(float(value)) > 1e12 else None
        return pd.Timestamp(value, unit=unit) if unit else pd.Timestamp(value)
    return pd.Timestamp(value)


def _clip(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return min(max(float(value), 0.0), 1.0)



# ---------------------------------------------------------------------------
# Single-process scoring (for baseline and final run)
# ---------------------------------------------------------------------------

def _score_config(
    data,
    mutations: dict,
    initial_equity: float,
    return_basis: str = "equity_curve",
    scoring_kwargs: dict | None = None,
    score_profile: str = "generic",
):
    """Build config, run unified engine, return (score, metrics, scoring_pnl)."""
    from backtests.swing.auto.config_mutator import mutate_unified_config
    from backtests.swing.auto.scoring import extract_metrics
    from backtests.swing.config_unified import UnifiedBacktestConfig
    from backtests.swing.engine.unified_portfolio_engine import run_unified

    config = UnifiedBacktestConfig(initial_equity=initial_equity)
    if mutations:
        config = mutate_unified_config(config, mutations)
    result = run_unified(data, config)
    trades = _collect_trades(result)
    metrics = extract_metrics(
        trades, result.combined_equity,
        result.combined_timestamps, initial_equity,
    )
    net_profit_override = _score_net_profit(config, result, initial_equity, return_basis)
    score = _score_result(
        metrics,
        config,
        result,
        initial_equity,
        net_profit_override=net_profit_override,
        score_profile=score_profile,
        scoring_kwargs=dict(scoring_kwargs or {}),
    )
    score_net_profit = net_profit_override if net_profit_override is not None else metrics.net_profit
    return score, metrics, score_net_profit


# ---------------------------------------------------------------------------
# Main greedy loop
# ---------------------------------------------------------------------------

def run_greedy(
    data,
    candidates: list[tuple[str, dict]],
    initial_equity: float = 10_000.0,
    base_mutations: dict | None = None,
    data_dir: Path | None = None,
    max_workers: int | None = None,
    return_basis: str = "equity_curve",
    scoring_kwargs: dict | None = None,
    score_profile: str = "generic",
    verbose: bool = True,
) -> GreedyResult:
    """Run greedy forward selection to find optimal portfolio config.

    Args:
        data: UnifiedPortfolioData (pre-loaded, used for baseline/final only)
        candidates: List of (name, mutations_dict) to test
        initial_equity: Starting equity
        base_mutations: Optional starting mutations (default: live_parity base)
        data_dir: Path to bar data (required for parallel workers to load data)
        max_workers: Number of parallel workers (default: cpu_count - 1)
        return_basis: "equity_curve" for existing compounded scoring, or
            "static_initial_strategy_risk" to score strategy total R against
            each sleeve's initial unit risk.
        scoring_kwargs: Optional keyword arguments forwarded to composite_score.
        score_profile: "generic" or "portfolio_synergy_alpha_frequency".
        verbose: Print progress

    Returns:
        GreedyResult with optimal mutations and round-by-round history
    """
    if base_mutations is None:
        base_mutations = {}
    if max_workers is None:
        max_workers = min(len(candidates), max(1, (os.cpu_count() or 4) - 1))
    if return_basis not in {"equity_curve", "static_initial_strategy_risk"}:
        raise ValueError(f"Unknown swing greedy return_basis: {return_basis}")
    if score_profile not in {"generic", "portfolio_synergy_alpha_frequency"}:
        raise ValueError(f"Unknown swing greedy score_profile: {score_profile}")
    scoring_kwargs = dict(scoring_kwargs or {})

    if verbose:
        print(f"\n{'='*60}")
        print("GREEDY FORWARD SELECTION: SWING PORTFOLIO")
        print(f"{'='*60}")
        print(f"Base: live_parity + {len(base_mutations)} mutations")
        print(f"Candidate pool: {len(candidates)} features")
        print(f"Workers: {max_workers} parallel processes")
        print(f"{'='*60}\n")

    # Compute baseline in main process (data already loaded)
    base_score_obj, base_metrics, base_score_net_profit = _score_config(
        data,
        base_mutations,
        initial_equity,
        return_basis,
        scoring_kwargs,
        score_profile,
    )

    if base_score_obj.rejected:
        print(f"WARNING: Base config rejected ({base_score_obj.reject_reason})")
        baseline_score = 0.0
    else:
        baseline_score = base_score_obj.total

    if verbose:
        print(f"Baseline: score={baseline_score:.4f}")
        if base_metrics:
            print(f"  trades={base_metrics.total_trades}, "
                  f"PF={base_metrics.profit_factor:.2f}, "
                  f"DD={base_metrics.max_drawdown_pct:.1%}, "
                  f"return={base_score_net_profit / initial_equity:.1%}, "
                  f"Sharpe={base_metrics.sharpe:.2f}")
        print()

    # Initialize worker pool (workers load data once, reuse for all rounds)
    if data_dir is None:
        data_dir = Path("backtests/swing/data/raw")

    print(f"Spawning {max_workers} worker processes (each loads data ~6s)...")
    t_pool = time.time()
    pool = mp.Pool(
        processes=max_workers,
        initializer=_init_worker,
        initargs=(str(data_dir), initial_equity, return_basis, scoring_kwargs, score_profile),
    )
    print(f"Pool ready in {time.time() - t_pool:.1f}s\n")

    # Greedy loop
    current_mutations = dict(base_mutations)
    current_score = baseline_score
    remaining = list(candidates)
    kept_features: list[str] = []
    rounds: list[GreedyRound] = []

    try:
        round_num = 0
        while remaining:
            round_num += 1
            if verbose:
                print(f"Round {round_num}: Testing {len(remaining)} candidates "
                      f"(parallel across {min(max_workers, len(remaining))} workers)...")

            t0 = time.time()

            # Build merged mutation dicts for all candidates
            tasks = []
            for name, cand_mutations in remaining:
                merged = {**current_mutations, **cand_mutations}
                tasks.append(merged)

            # Evaluate all candidates in parallel
            results = pool.map(_worker_score, tasks)

            # Collect scores
            round_scores: list[tuple[str, float, float]] = []
            for (name, _), (score_val, rejected, reason) in zip(remaining, results):
                if rejected and score_val == 0.0 and reason:
                    logger.warning("Candidate %s: %s", name, reason[:200])
                delta = (score_val - current_score) / current_score if current_score > 0 else 0.0
                round_scores.append((name, score_val, delta))

            elapsed = time.time() - t0

            # Sort by score descending
            round_scores.sort(key=lambda x: x[1], reverse=True)

            if verbose:
                for name, score_val, delta in round_scores:
                    marker = " ***" if score_val == round_scores[0][1] else ""
                    print(f"  {name:35s} score={score_val:.4f}  delta={delta:+.2%}{marker}")
                print(f"  ({elapsed:.1f}s)")

            best_name, best_score, best_delta = round_scores[0]

            if best_score > current_score:
                prev_score = current_score
                best_cand_mutations = next(m for n, m in remaining if n == best_name)
                current_mutations = {**current_mutations, **best_cand_mutations}
                current_score = best_score
                kept_features.append(best_name)
                remaining = [(n, m) for n, m in remaining if n != best_name]

                rounds.append(GreedyRound(
                    round_num=round_num,
                    candidates_tested=len(round_scores),
                    best_name=best_name,
                    best_score=best_score,
                    best_delta_pct=best_delta,
                    kept=True,
                    all_scores=round_scores,
                ))

                if verbose:
                    print(f"  -> KEEP {best_name} (score={best_score:.4f}, "
                          f"{best_delta:+.2%} vs {prev_score:.4f})\n")
            else:
                rounds.append(GreedyRound(
                    round_num=round_num,
                    candidates_tested=len(round_scores),
                    best_name=best_name,
                    best_score=best_score,
                    best_delta_pct=best_delta,
                    kept=False,
                    all_scores=round_scores,
                ))
                if verbose:
                    print("  -> No candidate improves score. Stopping.\n")
                break
    finally:
        pool.close()
        pool.join()

    # Final scoring in main process for detailed metrics
    _, final_metrics, final_score_net_profit = _score_config(
        data,
        current_mutations,
        initial_equity,
        return_basis,
        scoring_kwargs,
        score_profile,
    )

    result = GreedyResult(
        base_score=baseline_score,
        final_mutations=current_mutations,
        final_score=current_score,
        kept_features=kept_features,
        rounds=rounds,
    )

    if final_metrics:
        result.final_trades = final_metrics.total_trades
        result.final_pf = final_metrics.profit_factor
        result.final_dd_pct = final_metrics.max_drawdown_pct
        result.final_return_pct = final_score_net_profit / initial_equity
        result.final_sharpe = final_metrics.sharpe

    if verbose:
        _print_summary(result, baseline_score)

    return result


def _print_summary(result: GreedyResult, baseline_score: float) -> None:
    """Print final summary."""
    print(f"{'='*60}")
    print("OPTIMAL PORTFOLIO CONFIG")
    print(f"{'='*60}")
    print(f"Added: {', '.join(result.kept_features) if result.kept_features else '(none)'}")
    print(f"Rounds: {len(result.rounds)} ({sum(1 for r in result.rounds if r.kept)} kept)")
    print()
    print(f"Final score:    {result.final_score:.4f}")
    print(f"Baseline score: {baseline_score:.4f}")
    if baseline_score > 0:
        print(f"Improvement:    {(result.final_score - baseline_score) / baseline_score:+.2%}")
    print()
    print(f"Trades:  {result.final_trades}")
    print(f"PF:      {result.final_pf:.2f}")
    print(f"DD:      {result.final_dd_pct:.1%}")
    print(f"Return:  {result.final_return_pct:.1%}")
    print(f"Sharpe:  {result.final_sharpe:.2f}")
    print()
    print("Final mutations:")
    for k, v in sorted(result.final_mutations.items()):
        print(f"  {k}: {v}")
    print(f"{'='*60}")


def save_result(result: GreedyResult, output_path: Path) -> None:
    """Save greedy result to JSON."""
    data = {
        "base_score": result.base_score,
        "final_score": result.final_score,
        "improvement_pct": (
            (result.final_score - result.base_score) / result.base_score
            if result.base_score > 0 else 0.0
        ),
        "kept_features": result.kept_features,
        "final_mutations": result.final_mutations,
        "final_trades": result.final_trades,
        "final_pf": result.final_pf,
        "final_dd_pct": result.final_dd_pct,
        "final_return_pct": result.final_return_pct,
        "final_sharpe": result.final_sharpe,
        "rounds": [
            {
                "round": r.round_num,
                "candidates_tested": r.candidates_tested,
                "best_name": r.best_name,
                "best_score": r.best_score,
                "best_delta_pct": r.best_delta_pct,
                "kept": r.kept,
            }
            for r in result.rounds
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"\nResult saved to {output_path}")


# ---------------------------------------------------------------------------
# Predefined candidate pool from auto experiment results
# ---------------------------------------------------------------------------

# Candidates ordered by individual delta (highest first).
# Strategy-specific params are mapped to unified config routing keys.
PORTFOLIO_CANDIDATES: list[tuple[str, dict]] = [
    # Helix stale 4H detection
    ("helix_stale_4h_4", {"helix_param.STALE_4H_BARS": 4}),
    # Portfolio-level risk adjustments
    ("helix_risk_1.2pct", {"helix.unit_risk_pct": 0.012}),
    ("helix_risk_1.0pct", {"helix.unit_risk_pct": 0.010}),
    ("overlay_max_70pct", {"overlay_max_pct": 0.70}),
]
