from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from backtests.momentum.config import SlippageConfig
from backtests.momentum.config_regime import NqRegimeAblationFlags, NqRegimeBacktestConfig
from backtests.momentum.engine.regime_engine import load_nq_regime_data, run_nq_regime_backtest
from backtests.shared.auto.types import ScoredCandidate

from .scoring import composite_score

_DATA = None
_DATA_DIR = Path(".")
_INITIAL_EQUITY = 10_000.0
_ANALYSIS_SYMBOL = "NQ"
_TRADE_SYMBOL = "MNQ"


def init_worker(
    data_dir: str,
    initial_equity: float = 10_000.0,
    analysis_symbol: str = "NQ",
    trade_symbol: str = "MNQ",
) -> None:
    global _DATA, _DATA_DIR, _INITIAL_EQUITY, _ANALYSIS_SYMBOL, _TRADE_SYMBOL
    _DATA_DIR = Path(data_dir)
    _INITIAL_EQUITY = float(initial_equity)
    _ANALYSIS_SYMBOL = analysis_symbol.upper()
    _TRADE_SYMBOL = trade_symbol.upper()
    cfg = NqRegimeBacktestConfig(
        data_dir=_DATA_DIR,
        initial_equity=_INITIAL_EQUITY,
        analysis_symbol=_ANALYSIS_SYMBOL,
        trade_symbol=_TRADE_SYMBOL,
    )
    _DATA = load_nq_regime_data(cfg)


def score_candidate(args) -> ScoredCandidate:
    name, mutations, current_mutations, phase, scoring_weights, hard_rejects = args
    merged = {**(current_mutations or {}), **(mutations or {})}
    cfg = mutate_config(
        NqRegimeBacktestConfig(
            data_dir=_DATA_DIR,
            initial_equity=_INITIAL_EQUITY,
            analysis_symbol=_ANALYSIS_SYMBOL,
            trade_symbol=_TRADE_SYMBOL,
        ),
        merged,
    )
    data = _DATA if _DATA is not None else load_nq_regime_data(cfg)
    result = run_nq_regime_backtest(data, cfg)
    score = composite_score(result.metrics, scoring_weights, hard_rejects)
    return ScoredCandidate(
        name=name,
        score=score.score,
        rejected=score.rejected,
        reject_reason=score.reject_reason,
        metrics=dict(result.metrics),
    )


def mutate_config(base: NqRegimeBacktestConfig, mutations: dict[str, Any]) -> NqRegimeBacktestConfig:
    cfg = base
    flag_changes: dict[str, Any] = {}
    slippage_changes: dict[str, Any] = {}
    param_changes: dict[str, Any] = {}
    top_level: dict[str, Any] = {}
    for key, value in mutations.items():
        if key.startswith("flags."):
            flag_changes[key.split(".", 1)[1]] = value
        elif key.startswith("slippage."):
            slippage_changes[key.split(".", 1)[1]] = value
        elif key.startswith("param_overrides."):
            param_changes[key.split(".", 1)[1]] = value
        else:
            top_level[key] = value
    if flag_changes:
        valid = {field.name for field in fields(NqRegimeAblationFlags)}
        cfg = replace(cfg, flags=replace(cfg.flags, **{key: value for key, value in flag_changes.items() if key in valid}))
    if slippage_changes:
        valid = {field.name for field in fields(SlippageConfig)}
        cfg = replace(cfg, slippage=replace(cfg.slippage, **{key: value for key, value in slippage_changes.items() if key in valid}))
    if param_changes:
        cfg = replace(cfg, param_overrides={**cfg.param_overrides, **param_changes})
    if top_level:
        valid = {field.name for field in fields(NqRegimeBacktestConfig)}
        cfg = replace(cfg, **{key: value for key, value in top_level.items() if key in valid})
    return cfg

