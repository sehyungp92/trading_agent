"""Greedy forward selection for optimal config discovery.

Algorithm:
  1. Start with a base config (e.g. full_recalibration mutations)
  2. Score the base → baseline_score
  3. For each round, test every remaining candidate merged with base
  4. Keep the best candidate if it improves score; stop when none do
  5. Output the final optimal mutations and comparison table

Usage:
    from backtests.stock.auto.greedy_optimize import run_greedy
    result = run_greedy(replay, "iaric", 1, base_mutations, candidates)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from datetime import date, datetime, time as dt_time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backtests.stock.auto.config_mutator import (
    mutate_alcb_config,
    mutate_iaric_config,
)
from backtests.stock.auto.scoring import (
    CompositeScore, IARIC_NORM, composite_score, compute_r_multiples,
    extract_metrics,
)
from backtests.stock.models import TradeRecord

logger = logging.getLogger(__name__)


_GREEDY_CHECKPOINT_VERSION = 2
_GREEDY_ALGORITHM_VERSION = "causal-selection-next-bar-netpnl-v1"


# ---------------------------------------------------------------------------
# Parallel evaluation support (multiprocessing)
# ---------------------------------------------------------------------------
_mp_replay = None  # Worker-local replay engine
_mp_cfg_kwargs = None  # Worker-local config kwargs


def _init_eval_worker(data_dir_str: str, cfg_kwargs: dict) -> None:
    """Initialize a multiprocessing worker: load data once per process."""
    global _mp_replay, _mp_cfg_kwargs
    from backtests.stock.engine.research_replay import ResearchReplayEngine
    _mp_replay = ResearchReplayEngine(data_dir=Path(data_dir_str))
    _mp_replay.load_all_data()
    cfg_kwargs["data_dir"] = Path(cfg_kwargs["data_dir"])
    _mp_cfg_kwargs = cfg_kwargs


def _eval_candidate_mp(args: tuple) -> tuple[str, float, float, float, float, bool, str | None]:
    """Evaluate one candidate in a worker process."""
    name, merged_mutations, initial_equity = args
    try:
        config = _make_config(merged_mutations, _mp_cfg_kwargs)
        trades, eq, ts = _run_engine(
            _mp_replay, _mp_cfg_kwargs["strategy"],
            _mp_cfg_kwargs["tier"], config,
        )
        metrics = extract_metrics(trades, eq, ts, initial_equity)
        r_mult = compute_r_multiples(trades)
        norm = IARIC_NORM if _mp_cfg_kwargs["strategy"] == "iaric" else None
        score = composite_score(metrics, initial_equity, r_multiples=r_mult, norm=norm)
        return (
            name,
            score.total if not score.rejected else 0.0,
            metrics.net_profit,
            metrics.expectancy_dollar,
            metrics.trades_per_month,
            score.rejected,
            None,
        )
    except Exception as exc:
        return (name, 0.0, 0.0, 0.0, 0.0, True, f"{exc}")


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
    strategy: str
    tier: int
    base_mutations: dict
    base_score: float
    final_mutations: dict
    final_score: float
    kept_features: list[str]
    rounds: list[GreedyRound]
    final_trades: int = 0
    final_pf: float = 0.0
    final_dd_pct: float = 0.0
    final_return_pct: float = 0.0


def _save_checkpoint(
    checkpoint_path: Path,
    strategy: str,
    tier: int,
    base_mutations: dict,
    resume_signature: str,
    baseline_score: float,
    current_mutations: dict,
    current_score: float,
    kept_features: list[str],
    rounds: list[GreedyRound],
    remaining: list[tuple[str, dict]],
) -> None:
    """Save greedy loop state to a checkpoint file for resume."""
    data = {
        "checkpoint_version": _GREEDY_CHECKPOINT_VERSION,
        "resume_signature": resume_signature,
        "strategy": strategy,
        "tier": tier,
        "base_mutations": {k: _serialize_val(v) for k, v in base_mutations.items()},
        "baseline_score": baseline_score,
        "current_mutations": {k: _serialize_val(v) for k, v in current_mutations.items()},
        "current_score": current_score,
        "kept_features": kept_features,
        "rounds": [
            {
                "round_num": r.round_num,
                "candidates_tested": r.candidates_tested,
                "best_name": r.best_name,
                "best_score": r.best_score,
                "best_delta_pct": r.best_delta_pct,
                "kept": r.kept,
                "all_scores": r.all_scores,
            }
            for r in rounds
        ],
        "remaining": [
            (name, {k: _serialize_val(v) for k, v in muts.items()})
            for name, muts in remaining
        ],
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    try:
        tmp.replace(checkpoint_path)
    except PermissionError:
        # Windows: target may be locked by antivirus/indexer; fall back to copy
        import shutil
        shutil.copy2(tmp, checkpoint_path)
        tmp.unlink(missing_ok=True)


def _serialize_val(v):
    """Convert tuples to lists for JSON serialization."""
    if isinstance(v, tuple):
        return [_serialize_val(x) for x in v]
    if isinstance(v, list):
        return [_serialize_val(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (date, datetime, dt_time)):
        return v.isoformat()
    return v


def _checkpoint_signature(
    strategy: str,
    tier: int,
    base_mutations: dict,
    candidates: list[tuple[str, dict]],
    cfg_kwargs: dict,
) -> str:
    payload = {
        "algorithm_version": _GREEDY_ALGORITHM_VERSION,
        "strategy": strategy,
        "tier": tier,
        "base_mutations": {k: _serialize_val(v) for k, v in sorted(base_mutations.items())},
        "candidates": [
            (name, {k: _serialize_val(v) for k, v in sorted(muts.items())})
            for name, muts in candidates
        ],
        "initial_equity": cfg_kwargs["initial_equity"],
        "start_date": cfg_kwargs["start_date"],
        "end_date": cfg_kwargs["end_date"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_checkpoint(checkpoint_path: Path, *, resume_signature: str) -> dict | None:
    """Load checkpoint if it exists, return parsed data or None."""
    if not checkpoint_path.exists():
        return None
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if data.get("checkpoint_version") != _GREEDY_CHECKPOINT_VERSION:
            logger.info("Ignoring stale checkpoint %s: version mismatch", checkpoint_path)
            return None
        if data.get("resume_signature") != resume_signature:
            logger.info("Ignoring stale checkpoint %s: signature mismatch", checkpoint_path)
            return None
        return data
    except Exception as exc:
        logger.warning("Failed to load checkpoint %s: %s", checkpoint_path, exc)
        return None


def run_greedy(
    replay,
    strategy: str,
    tier: int,
    base_mutations: dict,
    candidates: list[tuple[str, dict]],
    initial_equity: float = 10_000.0,
    start_date: str = "2024-01-01",
    end_date: str = "2026-03-01",
    data_dir: str = "backtests/stock/data/raw",
    verbose: bool = True,
    max_workers: int = 1,
    checkpoint_path: Path | None = None,
    prune_delta: float = -0.10,
    prune_no_effect: bool = True,
    min_delta: float = 0.001,
) -> GreedyResult:
    """Run greedy forward selection to find optimal config.

    Args:
        replay: ResearchReplayEngine with data already loaded
        strategy: "iaric" or "alcb"
        tier: 1 or 2
        base_mutations: Starting mutations (e.g. full_recalibration)
        candidates: List of (name, mutations_dict) to test
        initial_equity: Starting equity
        start_date: Backtest start date
        end_date: Backtest end date
        data_dir: Path to bar data
        verbose: Print progress
        checkpoint_path: If provided, save/resume checkpoints at this path
        prune_delta: After round 1, drop candidates with delta worse than this
            threshold (default -0.10 = -10%). Set to None to disable pruning.
        prune_no_effect: After round 1, drop candidates with exactly 0% delta
            (no-ops that set parameters to their current values).
        min_delta: Minimum relative improvement to keep a candidate (default
            0.001 = 0.1%). Stops chasing noise in late rounds.

    Returns:
        GreedyResult with optimal mutations and round-by-round history
    """
    cfg_kwargs = dict(
        strategy=strategy, tier=tier, initial_equity=initial_equity,
        start_date=start_date, end_date=end_date, data_dir=Path(data_dir),
    )
    resume_signature = _checkpoint_signature(
        strategy, tier, base_mutations, candidates, cfg_kwargs,
    )

    if verbose:
        print(f"\n{'='*60}", flush=True)
        print(f"GREEDY FORWARD SELECTION: {strategy.upper()} T{tier}")
        print(f"{'='*60}")
        print(f"Base mutations: {len(base_mutations)} params")
        print(f"Candidate pool: {len(candidates)} features")
        if max_workers > 1:
            print(f"Parallel workers: {max_workers}")
        print(f"{'='*60}\n", flush=True)

    base_score_obj, base_metrics = _score_config(
        replay, base_mutations, initial_equity, cfg_kwargs,
    )

    if base_score_obj.rejected:
        print(f"WARNING: Base config rejected ({base_score_obj.reject_reason})")
        print("Proceeding anyway — greedy may overcome base limitations.\n")
        baseline_score = 0.0
    else:
        baseline_score = base_score_obj.total

    if verbose:
        print(f"Baseline: score={baseline_score:.4f}", flush=True)
        if base_metrics:
            print(f"  trades={base_metrics.total_trades}, "
                  f"PF={base_metrics.profit_factor:.2f}, "
                  f"DD={base_metrics.max_drawdown_pct:.1%}, "
                  f"return={base_metrics.net_profit / initial_equity:.1%}")
        print(flush=True)

    current_score_obj = base_score_obj
    current_metrics = base_metrics

    # --- Resume from checkpoint if available ---
    resumed = False
    if checkpoint_path is not None:
        ckpt = _load_checkpoint(checkpoint_path, resume_signature=resume_signature)
        if ckpt is not None:
            # Build a lookup of candidate names still remaining in checkpoint
            ckpt_remaining_names = {name for name, _ in ckpt["remaining"]}
            # Rebuild remaining from original candidates to preserve mutation dicts
            remaining_lookup = {name: muts for name, muts in candidates}
            _resumed_remaining = [
                (name, remaining_lookup[name])
                for name in ckpt_remaining_names
                if name in remaining_lookup
            ]
            if _resumed_remaining:
                current_mutations = ckpt["current_mutations"]
                current_score = ckpt["current_score"]
                current_score_obj, current_metrics = _score_config(
                    replay, current_mutations, initial_equity, cfg_kwargs,
                )
                kept_features = ckpt["kept_features"]
                rounds = [
                    GreedyRound(
                        round_num=r["round_num"],
                        candidates_tested=r["candidates_tested"],
                        best_name=r["best_name"],
                        best_score=r["best_score"],
                        best_delta_pct=r["best_delta_pct"],
                        kept=r["kept"],
                        all_scores=[tuple(s) for s in r["all_scores"]],
                    )
                    for r in ckpt["rounds"]
                ]
                remaining = _resumed_remaining
                baseline_score = ckpt["baseline_score"]
                resumed = True
                if verbose:
                    print(f"*** RESUMED from checkpoint: "
                          f"{len(rounds)} rounds done, "
                          f"{len(kept_features)} features kept, "
                          f"{len(remaining)} candidates remaining, "
                          f"score={current_score:.4f} ***\n", flush=True)

    # Create process pool for parallel evaluation
    pool = None
    if max_workers > 1:
        import multiprocessing as mp
        _cfg_ser = {**cfg_kwargs, "data_dir": str(cfg_kwargs["data_dir"])}
        if verbose:
            print(f"Initializing {max_workers} worker processes...", flush=True)
        pool = mp.Pool(
            processes=max_workers,
            initializer=_init_eval_worker,
            initargs=(str(cfg_kwargs["data_dir"]), _cfg_ser),
        )
        if verbose:
            print(f"  Workers ready.\n", flush=True)

    # Greedy loop — initialize fresh or keep resumed state
    if not resumed:
        current_mutations = dict(base_mutations)
        current_score = baseline_score
        remaining = list(candidates)
        kept_features: list[str] = []
        rounds: list[GreedyRound] = []

    round_num = rounds[-1].round_num if rounds else 0
    try:
        while remaining:
            round_num += 1
            if verbose:
                print(f"Round {round_num}: Testing {len(remaining)} candidates...", flush=True)

            round_scores: list[tuple[str, float, float]] = []
            round_evals: list[dict] = []
            t0 = time.time()

            if pool is not None:
                # Parallel evaluation
                tasks = [
                    (name, {**current_mutations, **cm}, initial_equity)
                    for name, cm in remaining
                ]
                results = pool.map(_eval_candidate_mp, tasks)
                for name, score_val, net_profit, expectancy_dollar, trades_per_month, rejected, err in results:
                    if err:
                        logger.warning("Candidate %s crashed: %s", name, err)
                    delta = (score_val - current_score) / current_score if current_score > 0 else 0.0
                    round_scores.append((name, score_val, delta))
                    round_evals.append({
                        "name": name,
                        "score_val": score_val,
                        "net_profit": net_profit,
                        "expectancy_dollar": expectancy_dollar,
                        "trades_per_month": trades_per_month,
                        "objective_valid": (not rejected and net_profit > 0 and expectancy_dollar > 0),
                    })
            else:
                for ci, (name, cand_mutations) in enumerate(remaining, 1):
                    merged = {**current_mutations, **cand_mutations}
                    t1 = time.time()
                    try:
                        score_obj, metrics = _score_config(
                            replay, merged, initial_equity, cfg_kwargs,
                        )
                        score_val = score_obj.total if not score_obj.rejected else 0.0
                        net_profit = metrics.net_profit if metrics else 0.0
                        expectancy_dollar = metrics.expectancy_dollar if metrics else 0.0
                        trades_per_month = metrics.trades_per_month if metrics else 0.0
                    except Exception:
                        logger.warning("Candidate %s crashed:\n%s", name, traceback.format_exc())
                        score_val = 0.0
                        score_obj = CompositeScore(0, 0, 0, 0, rejected=True)
                        net_profit = 0.0
                        expectancy_dollar = 0.0
                        trades_per_month = 0.0
                    delta = (score_val - current_score) / current_score if current_score > 0 else 0.0
                    round_scores.append((name, score_val, delta))
                    round_evals.append({
                        "name": name,
                        "score_val": score_val,
                        "net_profit": net_profit,
                        "expectancy_dollar": expectancy_dollar,
                        "trades_per_month": trades_per_month,
                        "objective_valid": (not score_obj.rejected and net_profit > 0 and expectancy_dollar > 0),
                    })
                    if verbose:
                        print(f"  [{ci}/{len(remaining)}] {name:30s} score={score_val:.4f} "
                              f"delta={delta:+.2%} ({time.time()-t1:.0f}s)", flush=True)

            elapsed = time.time() - t0

            # Sort by score descending
            round_scores.sort(key=lambda x: x[1], reverse=True)

            if verbose:
                for name, score_val, delta in round_scores:
                    marker = " ***" if score_val == round_scores[0][1] else ""
                    print(f"  {name:30s} score={score_val:.4f}  delta={delta:+.2%}{marker}")
                print(f"  ({elapsed:.1f}s)", flush=True)

            best_name, best_score, best_delta = round_scores[0]
            keep_candidate = best_score > current_score and best_delta >= min_delta

            if strategy == "alcb":
                current_eval = {
                    "name": "__CURRENT__",
                    "score_val": current_score,
                    "net_profit": current_metrics.net_profit if current_metrics else 0.0,
                    "expectancy_dollar": current_metrics.expectancy_dollar if current_metrics else 0.0,
                    "trades_per_month": current_metrics.trades_per_month if current_metrics else 0.0,
                    "objective_valid": (
                        current_score_obj is not None
                        and not current_score_obj.rejected
                        and current_metrics is not None
                        and current_metrics.net_profit > 0
                        and current_metrics.expectancy_dollar > 0
                    ),
                }
                objective_pool = [ev for ev in [current_eval, *round_evals] if ev["objective_valid"]]
                if objective_pool:
                    top_expectancy = max(ev["expectancy_dollar"] for ev in objective_pool)
                    floor = top_expectancy * 0.98
                    contenders = [ev for ev in objective_pool if ev["expectancy_dollar"] >= floor]
                    contenders.sort(
                        key=lambda ev: (
                            ev["trades_per_month"],
                            ev["score_val"],
                            ev["expectancy_dollar"],
                        ),
                        reverse=True,
                    )
                    selected = contenders[0]
                    best_name = selected["name"]
                    best_score = selected["score_val"]
                    best_delta = (
                        (best_score - current_score) / current_score
                        if current_score > 0 else 0.0
                    )
                    keep_candidate = best_name != "__CURRENT__"
                else:
                    best_name = "__CURRENT__"
                    best_score = current_score
                    best_delta = 0.0
                    keep_candidate = False

            if keep_candidate:
                prev_score = current_score
                # Merge best candidate into current
                best_cand_mutations = next(m for n, m in remaining if n == best_name)
                current_mutations = {**current_mutations, **best_cand_mutations}
                current_score_obj, current_metrics = _score_config(
                    replay, current_mutations, initial_equity, cfg_kwargs,
                )
                current_score = current_score_obj.total if not current_score_obj.rejected else 0.0
                kept_features.append(best_name)
                remaining = [(n, m) for n, m in remaining if n != best_name]

                # Prune after round 1
                if strategy != "alcb" and round_num >= 1:
                    scores_by_name = {n: d for n, _s, d in round_scores}
                    before_prune = len(remaining)

                    def _keep(n: str) -> bool:
                        d = scores_by_name.get(n, 0.0)
                        if prune_delta is not None and d < prune_delta:
                            return False
                        if prune_no_effect and abs(d) < 1e-6:
                            return False
                        return True

                    remaining = [(n, m) for n, m in remaining if _keep(n)]
                    pruned = before_prune - len(remaining)
                    if pruned > 0 and verbose:
                        print(f"  [pruned {pruned} candidates (delta < "
                              f"{prune_delta:+.0%} or no effect), "
                              f"{len(remaining)} remain]", flush=True)

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
                    print(f"  → KEEP {best_name} (score={best_score:.4f}, "
                          f"{best_delta:+.2%} vs {prev_score:.4f})\n", flush=True)

                # Checkpoint after each kept round
                if checkpoint_path is not None:
                    _save_checkpoint(
                        checkpoint_path, strategy, tier, base_mutations,
                        resume_signature,
                        baseline_score, current_mutations, current_score,
                        kept_features, rounds, remaining,
                    )
                    if verbose:
                        print(f"  [checkpoint saved: round {round_num}]\n", flush=True)
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
                    if strategy == "alcb":
                        print(f"  ??No candidate improves the ALCB objective. Stopping.\n", flush=True)
                    elif best_score > current_score:
                        print(f"  → Best candidate {best_name} ({best_delta:+.2%}) "
                              f"below min_delta ({min_delta:.1%}). Stopping.\n", flush=True)
                    else:
                        print(f"  → No candidate improves score. Stopping.\n", flush=True)
                # Final checkpoint (marks convergence)
                if checkpoint_path is not None:
                    _save_checkpoint(
                        checkpoint_path, strategy, tier, base_mutations,
                        resume_signature,
                        baseline_score, current_mutations, current_score,
                        kept_features, rounds, [],
                    )
                break
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Final scoring for summary metrics
    _, final_metrics = _score_config(
        replay, current_mutations, initial_equity, cfg_kwargs,
    )

    result = GreedyResult(
        strategy=strategy,
        tier=tier,
        base_mutations=base_mutations,
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
        result.final_return_pct = final_metrics.net_profit / initial_equity

    if verbose:
        _print_summary(result, baseline_score)

    return result


def _score_config(
    replay,
    mutations: dict,
    initial_equity: float,
    cfg_kwargs: dict,
) -> tuple[CompositeScore, object | None]:
    """Build config, run engine, return (score, metrics)."""
    config = _make_config(mutations, cfg_kwargs)
    trades, eq, ts = _run_engine(replay, cfg_kwargs["strategy"], cfg_kwargs["tier"], config)
    metrics = extract_metrics(trades, eq, ts, initial_equity)
    r_mult = compute_r_multiples(trades)
    norm = IARIC_NORM if cfg_kwargs["strategy"] == "iaric" else None
    score = composite_score(metrics, initial_equity, r_multiples=r_mult, norm=norm)
    return score, metrics


def _make_config(mutations: dict, cfg_kwargs: dict):
    """Build a config with mutations applied."""
    strategy = cfg_kwargs["strategy"]
    if strategy == "iaric":
        from backtests.stock.config_iaric import IARICBacktestConfig
        base = IARICBacktestConfig(
            start_date=cfg_kwargs["start_date"],
            end_date=cfg_kwargs["end_date"],
            initial_equity=cfg_kwargs["initial_equity"],
            tier=cfg_kwargs["tier"],
            data_dir=cfg_kwargs["data_dir"],
        )
        return mutate_iaric_config(base, mutations)
    elif strategy == "alcb":
        from backtests.stock.config_alcb import ALCBBacktestConfig
        base = ALCBBacktestConfig(
            start_date=cfg_kwargs["start_date"],
            end_date=cfg_kwargs["end_date"],
            initial_equity=cfg_kwargs["initial_equity"],
            tier=cfg_kwargs["tier"],
            data_dir=cfg_kwargs["data_dir"],
        )
        return mutate_alcb_config(base, mutations)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _run_engine(
    replay,
    strategy: str,
    tier: int,
    config,
) -> tuple[list[TradeRecord], np.ndarray, np.ndarray]:
    """Run the correct engine and return (trades, equity_curve, timestamps)."""
    if strategy == "iaric":
        from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine
        engine = IARICPullbackEngine(config, replay)
    elif strategy == "alcb":
        from backtests.stock.engine.alcb_engine import ALCBIntradayEngine
        engine = ALCBIntradayEngine(config, replay)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    result = engine.run()
    return result.trades, result.equity_curve, result.timestamps


def _print_summary(result: GreedyResult, baseline_score: float) -> None:
    """Print final summary."""
    print(f"{'='*60}")
    print(f"OPTIMAL CONFIG")
    print(f"{'='*60}")
    print(f"Strategy: {result.strategy.upper()} T{result.tier}")
    print(f"Base: full_recalibration")
    print(f"Added: {', '.join(result.kept_features) if result.kept_features else '(none)'}")
    print(f"Rounds: {len(result.rounds)} ({sum(1 for r in result.rounds if r.kept)} kept)")
    print()
    print(f"Final score:    {result.final_score:.4f}")
    print(f"Baseline score: {baseline_score:.4f}")
    if baseline_score > 0:
        print(f"Improvement:    {(result.final_score - baseline_score) / baseline_score:+.2%}")
    print()
    print(f"Trades: {result.final_trades}")
    print(f"PF:     {result.final_pf:.2f}")
    print(f"DD:     {result.final_dd_pct:.1%}")
    print(f"Return: {result.final_return_pct:.1%}")
    print()
    print("Final mutations:")
    for k, v in sorted(result.final_mutations.items()):
        print(f"  {k}: {v}")
    print(f"{'='*60}")


def save_result(result: GreedyResult, output_path: Path) -> None:
    """Save greedy result to JSON."""
    data = {
        "strategy": result.strategy,
        "tier": result.tier,
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
    output_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"\nResult saved to {output_path}")


# ---------------------------------------------------------------------------
# Predefined candidate pools
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# IARIC T3 Phase 7: Mean-Reversion Pullback-Buy
# ---------------------------------------------------------------------------
# Tier 3 is a completely new entry signal scanning the full S&P universe for
# short-term oversold pullbacks in uptrends. Independent of T1 sponsorship logic.

IARIC_T3_P7_BASE_MUTATIONS: dict = {
    "param_overrides.pb_rsi_period": 2,
    "param_overrides.pb_rsi_entry": 10.0,
    "param_overrides.pb_rsi_exit": 70.0,
    "param_overrides.pb_atr_stop_mult": 1.5,
    "param_overrides.pb_max_hold_days": 5,
    "param_overrides.pb_max_positions": 8,
    "param_overrides.pb_flow_gate": True,
    "param_overrides.base_risk_fraction": 0.005,
}

IARIC_T3_P7_CANDIDATES: list[tuple[str, dict]] = [
    # A. RSI Period (2)
    ("rsi_period_3", {"param_overrides.pb_rsi_period": 3}),
    ("rsi_period_5", {"param_overrides.pb_rsi_period": 5}),
    # B. RSI Entry Threshold (4)
    ("rsi_entry_5", {"param_overrides.pb_rsi_entry": 5.0}),
    ("rsi_entry_15", {"param_overrides.pb_rsi_entry": 15.0}),
    ("rsi_entry_20", {"param_overrides.pb_rsi_entry": 20.0}),
    ("rsi_entry_25", {"param_overrides.pb_rsi_entry": 25.0}),
    # C. RSI Exit Threshold (4)
    ("rsi_exit_50", {"param_overrides.pb_rsi_exit": 50.0}),
    ("rsi_exit_60", {"param_overrides.pb_rsi_exit": 60.0}),
    ("rsi_exit_80", {"param_overrides.pb_rsi_exit": 80.0}),
    ("rsi_exit_90", {"param_overrides.pb_rsi_exit": 90.0}),
    # D. Alternative Triggers (3)
    ("cdd_3", {"param_overrides.pb_cdd_min": 3}),
    ("cdd_4", {"param_overrides.pb_cdd_min": 4}),
    ("ma_zone", {"param_overrides.pb_ma_zone_entry": True}),
    # E. Stop Distance (2)
    ("atr_stop_1x", {"param_overrides.pb_atr_stop_mult": 1.0}),
    ("atr_stop_2x", {"param_overrides.pb_atr_stop_mult": 2.0}),
    # F. Hold Duration (3)
    ("hold_3d", {"param_overrides.pb_max_hold_days": 3}),
    ("hold_7d", {"param_overrides.pb_max_hold_days": 7}),
    ("hold_10d", {"param_overrides.pb_max_hold_days": 10}),
    # G. Profit Target (3)
    ("target_15r", {"param_overrides.pb_profit_target_r": 1.5}),
    ("target_2r", {"param_overrides.pb_profit_target_r": 2.0}),
    ("target_3r", {"param_overrides.pb_profit_target_r": 3.0}),
    # H. Position Limits (4)
    ("max_pos_4", {"param_overrides.pb_max_positions": 4}),
    ("max_pos_6", {"param_overrides.pb_max_positions": 6}),
    ("max_pos_10", {"param_overrides.pb_max_positions": 10}),
    ("max_pos_12", {"param_overrides.pb_max_positions": 12}),
    # I. Regime Gate (2)
    ("regime_b_and_above", {"param_overrides.pb_regime_gate": "B_and_above"}),
    ("regime_any", {"param_overrides.pb_regime_gate": "any"}),
    # J. Risk & Sizing (4)
    ("risk_0025", {"param_overrides.base_risk_fraction": 0.0025}),
    ("risk_003", {"param_overrides.base_risk_fraction": 0.003}),
    ("risk_0075", {"param_overrides.base_risk_fraction": 0.0075}),
    ("risk_010", {"param_overrides.base_risk_fraction": 0.010}),
    # K. Flow Gate (1)
    ("no_flow_gate", {"param_overrides.pb_flow_gate": False}),
    # L. Carry (4)
    ("carry_on", {"param_overrides.pb_carry_enabled": True}),
    ("carry_on_min_0", {"param_overrides.pb_carry_enabled": True,
                        "param_overrides.pb_carry_min_r": 0.0}),
    ("carry_on_min_05", {"param_overrides.pb_carry_enabled": True,
                         "param_overrides.pb_carry_min_r": 0.5}),
    ("carry_on_hold_10", {"param_overrides.pb_carry_enabled": True,
                          "param_overrides.pb_max_hold_days": 10}),
    # M. Combined Triggers (4)
    ("rsi5_cdd3", {"param_overrides.pb_rsi_entry": 5.0,
                   "param_overrides.pb_cdd_min": 3}),
    ("rsi15_mazone", {"param_overrides.pb_rsi_entry": 15.0,
                      "param_overrides.pb_ma_zone_entry": True}),
    ("rsi20_cdd3_mazone", {"param_overrides.pb_rsi_entry": 20.0,
                           "param_overrides.pb_cdd_min": 3,
                           "param_overrides.pb_ma_zone_entry": True}),
    ("rsi25_hold7_target2", {"param_overrides.pb_rsi_entry": 25.0,
                             "param_overrides.pb_max_hold_days": 7,
                             "param_overrides.pb_profit_target_r": 2.0}),
    # N. Trend Filter Sensitivity (2)
    ("trend_sma_20", {"param_overrides.pb_trend_sma": 20}),
    ("trend_slope_20", {"param_overrides.pb_trend_slope_lookback": 20}),
]
