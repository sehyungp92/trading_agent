"""VdubusNQ worker -- multiprocessing-safe candidate evaluation."""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

from backtests.shared.auto.types import ScoredCandidate
from backtests.shared.auto.replay_bundle import ReplayBundle

logger = logging.getLogger(__name__)

_worker_data = None
_worker_config = None
_worker_data_dir_key: str | None = None


def init_worker(data_dir_str: str, equity: float) -> None:
    """Initialize worker: load data once, reuse across all phases/tasks.

    Phase, scoring weights, and hard rejects are passed per-task in
    score_candidate() so the pool can be reused across phases without
    re-initialization (avoids expensive data re-loading).
    """
    global _worker_data, _worker_config, _worker_data_dir_key

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # Suppress verbose engine logging in worker processes
    logging.getLogger("strategies.momentum.vdub").setLevel(logging.WARNING)
    logging.getLogger("backtests.momentum.engine.vdubus_engine").setLevel(logging.WARNING)

    from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig

    data_dir = Path(data_dir_str)
    _worker_config = VdubusBacktestConfig(
        initial_equity=equity,
        data_dir=data_dir,
        fixed_qty=10,
        flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
    )
    data_dir_key = str(data_dir.resolve())
    if _worker_data is None or _worker_data_dir_key != data_dir_key:
        _worker_data = load_worker_data("NQ", data_dir)
        _worker_data_dir_key = data_dir_key


def load_worker_data(symbol: str, data_dir: Path) -> ReplayBundle[dict]:
    """Load VdubusNQ bar data (same as cli._load_vdubus_data).

    Ensures optional 5m keys are present (engine.run requires them as kwargs).
    """
    from backtests.momentum.data.replay_cache import load_vdub_replay_bundle

    return load_vdub_replay_bundle(symbol, data_dir, include_5m=True)


def score_candidate(args: tuple) -> ScoredCandidate:
    """Evaluate a single candidate.

    Phase/weights/rejects are passed per-task so the worker pool can be
    reused across phases without re-initialization.
    """
    name, candidate_muts, base_muts, phase, scoring_weights, hard_rejects = args

    try:
        from dataclasses import asdict

        from backtests.momentum.data.replay_cache import replay_engine_kwargs
        from backtests.momentum.engine.vdubus_engine import VdubusEngine
        from backtests.momentum.auto.config_mutator import mutate_vdubus_config
        from backtests.momentum.auto.vdubus.plugin import score_phase_metrics
        from backtests.momentum.auto.vdubus.scoring import extract_vdubus_metrics

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_vdubus_config(_worker_config, all_muts)
        engine = VdubusEngine("NQ", config)
        result = engine.run(**replay_engine_kwargs(_worker_data))

        metrics = extract_vdubus_metrics(
            result.trades,
            list(result.equity_curve),
            list(result.time_series),
            _worker_config.initial_equity,
        )
        score = score_phase_metrics(
            phase,
            metrics,
            weight_overrides=scoring_weights,
            hard_rejects=hard_rejects,
        )

        metrics_dict = asdict(metrics)
        if score.rejected:
            return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=score.reject_reason, metrics=metrics_dict)
        return ScoredCandidate(name=name, score=score.total, metrics=metrics_dict)

    except Exception as exc:
        logger.error("Worker error for %s: %s", name, exc)
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=f"error: {exc}")
