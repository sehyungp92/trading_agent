"""Fast screening worker for Helix alpha expansion optimization."""
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


def load_helix_worker_data(symbols: list[str], data_dir: Path) -> dict:
    from backtests.swing.data.replay_cache import load_helix_replay_bundle

    return load_helix_replay_bundle(symbols, data_dir).data


def init_worker(data_dir_str: str, equity: float) -> None:
    global _worker_data, _worker_config, _worker_equity

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.swing.config_helix import HelixBacktestConfig

    _worker_equity = equity
    _worker_config = HelixBacktestConfig(
        initial_equity=equity,
        data_dir=Path(data_dir_str),
        track_shadows=False,
    )
    _worker_data = load_helix_worker_data(_worker_config.symbols, _worker_config.data_dir)


def score_candidate(args: tuple) -> ScoredCandidate:
    name, candidate_muts, base_muts, phase, _scoring_weights, hard_rejects = args

    try:
        from backtests.swing.auto.helix.alpha_expansion.plugin import score_phase_metrics
        from backtests.swing.auto.helix.config_mutator import mutate_helix_config
        from backtests.swing.auto.helix.scoring import extract_helix_metrics
        from backtests.swing.engine.helix_portfolio_engine import run_helix_independent

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        config = mutate_helix_config(_worker_config, all_muts)
        result = run_helix_independent(_worker_data, config)
        metrics = extract_helix_metrics(result, _worker_equity)
        score = score_phase_metrics(phase, metrics, hard_rejects=hard_rejects)
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

    except Exception:
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=traceback.format_exc())
