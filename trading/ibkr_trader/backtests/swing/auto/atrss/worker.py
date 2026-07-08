"""ATRSS worker -- per-process init and candidate scoring.

Workers load data ONCE at pool creation (init_worker) and reuse across all phases.
Phase/weights/rejects are passed per-call via score_candidate args.

Supports two modes:
  - "independent": Each symbol runs its own engine (fast, original R1 mode).
  - "synchronized": All symbols step together with portfolio allocation (honest R9 mode).
"""
from __future__ import annotations

import io
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from backtests.shared.auto.types import ScoredCandidate

_worker_data = None
_worker_config = None
_worker_equity: float = 0.0
_worker_mode: str = "independent"
_worker_profile: str = "r1_independent"


def init_worker(
    data_dir_str: str,
    equity: float,
    mode: str = "independent",
    symbols: list[str] | None = None,
    data_symbols: list[str] | None = None,
    scoring_profile: str | None = None,
) -> None:
    """Initialize worker process: install aliases, load data, create base config.

    Called once per worker at pool creation. Data is loaded here and reused
    for all subsequent score_candidate calls across all phases.

    Args:
        data_dir_str: Path to data directory.
        equity: Initial equity.
        mode: "independent" or "synchronized".
        symbols: Baseline tradable symbols (default: ["QQQ", "GLD"]).
        data_symbols: Superset of symbols to load so symbol-sleeve
            experiments can be tested without changing baseline behavior.
        scoring_profile: Composite-score profile for this optimization round.
    """
    global _worker_data, _worker_config, _worker_equity, _worker_mode, _worker_profile

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.swing.config import AblationFlags, BacktestConfig, SlippageConfig

    # Suppress engine logger noise
    logging.getLogger("backtest.engine.backtest_engine").setLevel(logging.WARNING)

    _worker_equity = equity
    _worker_mode = mode
    _worker_profile = scoring_profile or ("r9_synchronized" if mode == "synchronized" else "r1_independent")
    data_dir = Path(data_dir_str)
    sym_list = symbols or ["QQQ", "GLD"]
    data_sym_list = data_symbols or sym_list

    _worker_config = BacktestConfig(
        symbols=sym_list,
        initial_equity=equity,
        fixed_qty=10,
        data_dir=data_dir,
        slippage=SlippageConfig(commission_per_contract=1.00),
        flags=AblationFlags(stall_exit=False),
    )

    from backtests.swing.data.replay_cache import load_atrss_replay_bundle

    _worker_data = load_atrss_replay_bundle(data_dir, symbols=tuple(data_sym_list)).data


def score_candidate(args: tuple) -> ScoredCandidate:
    """Score a single candidate mutation set.

    Args:
        args: (name, candidate_mutations, base_mutations, phase, scoring_weights, hard_rejects)

    Returns:
        ScoredCandidate with score, rejection status, and metrics.
    """
    name, candidate_muts, base_muts, phase, scoring_weights, hard_rejects = args

    try:
        from backtests.swing.engine.portfolio_engine import run_independent, run_synchronized
        from backtests.swing.auto.config_mutator import mutate_atrss_config

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_atrss_config(_worker_config, all_muts)

        if _worker_mode == "synchronized":
            result = run_synchronized(_worker_data, config)
        else:
            result = run_independent(_worker_data, config)

        from backtests.swing.auto.atrss.scoring import extract_atrss_metrics
        metrics = extract_atrss_metrics(result, _worker_equity)

        if phase > 0:
            from backtests.swing.auto.atrss.phase_scoring import score_phase_metrics
            score = score_phase_metrics(
                phase, metrics,
                weight_overrides=scoring_weights,
                hard_rejects=hard_rejects,
                profile=_worker_profile,
            )
        else:
            from .scoring import composite_score
            score = composite_score(metrics, hard_rejects=hard_rejects, profile=_worker_profile)

        return ScoredCandidate(
            name=name,
            score=score.total,
            rejected=score.rejected,
            reject_reason=score.reject_reason,
            metrics=asdict(metrics),
        )

    except Exception:
        return ScoredCandidate(
            name=name,
            score=0.0,
            rejected=True,
            reject_reason=f"Error: {traceback.format_exc()[-200:]}",
            metrics={},
        )
