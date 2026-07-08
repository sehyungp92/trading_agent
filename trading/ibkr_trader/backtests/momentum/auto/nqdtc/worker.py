"""NQDTC worker -- multiprocessing-safe candidate evaluation."""
from __future__ import annotations

import io
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from backtests.shared.auto.types import ScoredCandidate
from backtests.shared.auto.replay_bundle import ReplayBundle

logger = logging.getLogger(__name__)

_worker_data = None
_worker_config = None
_worker_data_dir_key: str | None = None


def _parse_end_date(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def init_worker(data_dir_str: str, equity: float, end_date_iso: str | None = None) -> None:
    """Initialize worker: load data once, reuse across all phases/tasks.

    Phase, scoring weights, and hard rejects are passed per-task in
    score_candidate() so the pool can be reused across phases without
    re-initialization (avoids expensive data re-loading).
    """
    global _worker_data, _worker_config, _worker_data_dir_key

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # Suppress verbose box/engine logging (Box ACTIVE/DIRTY) in worker processes
    logging.getLogger("strategies.momentum.nqdtc.box").setLevel(logging.WARNING)
    logging.getLogger("backtests.momentum.engine.nqdtc_engine").setLevel(logging.WARNING)

    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig

    data_dir = Path(data_dir_str)
    _worker_config = NQDTCBacktestConfig(
        initial_equity=equity,
        data_dir=data_dir,
        end_date=_parse_end_date(end_date_iso),
        fixed_qty=10,
        track_signals=False,
        track_shadows=False,
        scoring_mode=True,
        max_dd_abort=0.50,
    )
    data_dir_key = str(data_dir.resolve())
    if _worker_data is None or _worker_data_dir_key != data_dir_key:
        _worker_data = load_worker_data("NQ", data_dir)
        _worker_data_dir_key = data_dir_key


def load_worker_data(symbol: str, data_dir: Path) -> ReplayBundle[dict]:
    """Load NQDTC bar data (same as cli._load_nqdtc_data)."""
    from backtests.momentum.data.replay_cache import load_replay_bundle

    bundle = load_replay_bundle(
        symbol,
        data_dir,
        include_fifteen_min=False,
        include_thirty_min=True,
        include_hourly=True,
        include_four_hour=True,
        include_daily=True,
        include_daily_es=True,
    )
    return ReplayBundle(
        data={
            "five_min_bars": bundle.data["five_min"],
            "thirty_min": bundle.data["thirty_min"],
            "hourly": bundle.data["hourly"],
            "four_hour": bundle.data["four_hour"],
            "daily": bundle.data["daily"],
            "thirty_min_idx_map": bundle.data["thirty_min_idx_map"],
            "hourly_idx_map": bundle.data["hourly_idx_map"],
            "four_hour_idx_map": bundle.data["four_hour_idx_map"],
            "daily_idx_map": bundle.data["daily_idx_map"],
            "daily_es": bundle.data.get("daily_es"),
            "daily_es_idx_map": bundle.data.get("daily_es_idx_map"),
        },
        cache_key=bundle.cache_key,
        cache_source_fingerprint=bundle.cache_source_fingerprint,
    )


def score_candidate(args: tuple) -> ScoredCandidate:
    """Evaluate a single candidate.

    Phase/weights/rejects are passed per-task so the worker pool can be
    reused across phases without re-initialization.
    """
    name, candidate_muts, base_muts, phase, scoring_weights, hard_rejects = args

    try:
        from dataclasses import asdict

        from backtests.momentum.data.replay_cache import replay_engine_kwargs
        from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
        from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
        from backtests.momentum.auto.nqdtc.plugin import score_phase_metrics
        from backtests.momentum.auto.nqdtc.scoring import extract_nqdtc_metrics

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_nqdtc_config(_worker_config, all_muts)
        engine = NQDTCEngine("MNQ", config)
        result = engine.run(**replay_engine_kwargs(_worker_data))

        metrics = extract_nqdtc_metrics(
            result.trades,
            result.equity_curve,
            result.timestamps,
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
