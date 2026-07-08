from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

from backtests.shared.auto.types import ScoredCandidate

logger = logging.getLogger(__name__)

_worker_data = None
_worker_config = None
_worker_data_dir_key: str | None = None
_worker_replay_kwargs: dict | None = None


def init_worker(
    data_dir_str: str,
    equity: float,
) -> None:
    global _worker_data, _worker_config, _worker_data_dir_key, _worker_replay_kwargs

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.momentum.config_downturn import DownturnBacktestConfig

    data_dir = Path(data_dir_str)
    _worker_config = DownturnBacktestConfig(
        initial_equity=equity,
        data_dir=data_dir,
        track_signals=False,
        skip_parity_output=True,
        max_dd_abort=0.50,
    )
    data_dir_key = str(data_dir.resolve())
    if _worker_data is None or _worker_data_dir_key != data_dir_key:
        _worker_data = load_worker_data("NQ", data_dir)
        _worker_data_dir_key = data_dir_key
        from backtests.momentum.data.replay_cache import replay_engine_kwargs
        _worker_replay_kwargs = replay_engine_kwargs(_worker_data)


def load_worker_data(symbol: str, data_dir: Path):
    from backtests.momentum.data.replay_cache import load_replay_bundle

    return load_replay_bundle(
        symbol,
        data_dir,
        include_fifteen_min=True,
        include_thirty_min=True,
        include_hourly=True,
        include_four_hour=True,
        include_daily=True,
        include_daily_es=True,
    )


def score_candidate(args: tuple[str, dict, dict]) -> ScoredCandidate:
    name, candidate_muts, base_muts, phase, scoring_weights, hard_rejects = args

    try:
        from dataclasses import asdict

        from backtests.momentum.engine.downturn_engine import DownturnEngine
        from backtests.momentum.analysis.downturn_diagnostics import compute_downturn_metrics
        from backtests.momentum.auto.downturn.config_mutator import mutate_downturn_config
        from backtests.momentum.auto.downturn.plugin import score_phase_metrics

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_downturn_config(_worker_config, all_muts)
        engine = DownturnEngine("NQ", config)
        result = engine.run(**_worker_replay_kwargs)
        metrics = compute_downturn_metrics(result, _worker_data.data["daily"])
        score = score_phase_metrics(
            phase,
            metrics,
            weight_overrides=scoring_weights,
            hard_rejects=hard_rejects,
        )

        metrics_dict = asdict(metrics)
        if score.rejected:
            return ScoredCandidate(
                name=name,
                score=0.0,
                rejected=True,
                reject_reason=score.reject_reason,
                metrics=metrics_dict,
            )
        return ScoredCandidate(name=name, score=score.total, metrics=metrics_dict)

    except Exception as exc:
        logger.error("Worker error for %s: %s", name, exc)
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=f"error: {exc}")
