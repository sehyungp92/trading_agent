from __future__ import annotations

import io
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from backtests.scalp.analysis.metrics import extract_metrics
from backtests.scalp.config_po3_reversal import Po3ReversalBacktestConfig
from backtests.scalp.engine.po3_reversal_engine import load_po3_reversal_data, run_po3_reversal_backtest
from backtests.shared.auto.replay_bundle import ReplayBundle
from backtests.shared.auto.types import ScoredCandidate

from .scoring import score_phase_metrics

logger = logging.getLogger(__name__)

_worker_bundle: ReplayBundle | None = None
_worker_config: Po3ReversalBacktestConfig | None = None
_worker_equity: float = 0.0
_worker_data_key: tuple[Path, str, str, str] | None = None


def load_worker_data(data_dir: Path, analysis_symbol: str, trade_symbol: str, confirmation_symbol: str) -> ReplayBundle:
    config = Po3ReversalBacktestConfig(
        analysis_symbol=analysis_symbol,
        trade_symbol=trade_symbol,
        confirmation_symbol=confirmation_symbol,
        data_dir=data_dir,
    )
    data = load_po3_reversal_data(config)
    from backtests.shared.auto.cache_keys import fingerprint_tree

    return ReplayBundle(data=data, cache_key=str(data_dir), cache_source_fingerprint=fingerprint_tree(data_dir, patterns=("*.parquet", "*.csv")))


def init_worker(
    data_dir_str: str,
    equity: float,
    analysis_symbol: str = "NQ",
    trade_symbol: str = "MNQ",
    confirmation_symbol: str = "ES",
) -> None:
    global _worker_bundle, _worker_config, _worker_equity, _worker_data_key
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    data_dir = Path(data_dir_str)
    analysis_symbol = analysis_symbol.upper()
    trade_symbol = trade_symbol.upper()
    confirmation_symbol = confirmation_symbol.upper()
    data_key = (data_dir, analysis_symbol, trade_symbol, confirmation_symbol)
    if _worker_bundle is None or _worker_data_key != data_key:
        _worker_bundle = load_worker_data(data_dir, analysis_symbol, trade_symbol, confirmation_symbol)
        _worker_data_key = data_key
    _worker_equity = equity
    _worker_config = Po3ReversalBacktestConfig(
        analysis_symbol=analysis_symbol,
        trade_symbol=trade_symbol,
        confirmation_symbol=confirmation_symbol,
        data_dir=data_dir,
        initial_equity=equity,
    )


def score_candidate(args: tuple[str, dict, dict, int, dict | None, dict | None]) -> ScoredCandidate:
    name, candidate_muts, base_muts, phase, scoring_weights, hard_rejects = args
    try:
        assert _worker_bundle is not None and _worker_config is not None
        mutations = dict(base_muts)
        mutations.update(candidate_muts)
        config = _mutate_config(_worker_config, mutations)
        result = run_po3_reversal_backtest(_worker_bundle.data, config)
        metrics_obj = extract_metrics(
            [trade for sr in result.symbol_results.values() for trade in sr.trades],
            result.combined_equity,
            result.combined_timestamps,
            _worker_equity,
        )
        score = score_phase_metrics(phase, metrics_obj, _worker_equity, weight_overrides=scoring_weights, hard_rejects=hard_rejects)
        return ScoredCandidate(name=name, score=score.total, rejected=score.rejected, reject_reason=score.reject_reason, metrics=asdict(metrics_obj))
    except Exception as exc:
        logger.exception("PO3 worker error for %s", name)
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=f"error: {exc}")


def _mutate_config(config: Po3ReversalBacktestConfig, mutations: dict) -> Po3ReversalBacktestConfig:
    import copy

    mutated = copy.deepcopy(config)
    for key, value in mutations.items():
        if key.startswith("flags."):
            setattr(mutated.flags, key.split(".", 1)[1], bool(value))
        elif key.startswith("param_overrides."):
            mutated.param_overrides[key.split(".", 1)[1]] = value
    return mutated
