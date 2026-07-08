"""Helix multiprocessing worker for phased auto-optimization.

Phase/weights/rejects are passed per-task so the worker pool can be
reused across phases without re-initialization (avoids expensive data
re-loading).
"""
from __future__ import annotations

import io
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from backtests.shared.auto.types import ScoredCandidate

_worker_data = None
_worker_config = None
_worker_equity: float = 0.0
_worker_start_date: str | None = None
_worker_end_date: str | None = None


def load_helix_worker_data(
    symbols: list[str],
    data_dir: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Load helix data -- usable from both worker init and plugin."""
    from backtests.swing.data.replay_cache import load_helix_replay_bundle

    return load_helix_replay_bundle(
        symbols,
        data_dir,
        start_date=start_date,
        end_date=end_date,
    ).data


def init_worker(
    data_dir_str: str,
    equity: float,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Initialize worker: load data once, reuse across all phases/tasks."""
    global _worker_data, _worker_config, _worker_equity, _worker_start_date, _worker_end_date

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.swing.config_helix import HelixBacktestConfig

    _worker_equity = equity
    _worker_config = HelixBacktestConfig(
        initial_equity=equity,
        data_dir=Path(data_dir_str),
        start_date=start_date,
        end_date=end_date,
        track_shadows=False,
    )
    _worker_start_date = start_date
    _worker_end_date = end_date
    _worker_data = load_helix_worker_data(
        _worker_config.symbols,
        _worker_config.data_dir,
        start_date=start_date,
        end_date=end_date,
    )


def score_candidate(args: tuple) -> ScoredCandidate:
    """Evaluate a single candidate.

    Phase/weights/rejects are passed per-task so the worker pool can be
    reused across phases without re-initialization.
    """
    name, candidate_muts, base_muts, phase, scoring_weights, hard_rejects = args

    try:
        from backtests.swing.engine.helix_portfolio_engine import run_helix_independent
        from backtests.swing.auto.helix.config_mutator import mutate_helix_config
        from backtests.swing.auto.helix.plugin import score_phase_metrics
        from backtests.swing.auto.helix.scoring import composite_score, extract_helix_metrics

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_helix_config(_worker_config, all_muts)
        result = run_helix_independent(_worker_data, config)
        metrics = extract_helix_metrics(result, _worker_equity)

        if phase > 0:
            score = score_phase_metrics(
                phase,
                metrics,
                weight_overrides=scoring_weights,
                hard_rejects=hard_rejects,
            )
        else:
            score = composite_score(metrics)

        if score.rejected:
            return ScoredCandidate(
                name=name,
                score=0.0,
                rejected=True,
                reject_reason=score.reject_reason,
                metrics=asdict(metrics),
            )
        return ScoredCandidate(name=name, score=score.total, metrics=asdict(metrics))

    except Exception:
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=traceback.format_exc())
