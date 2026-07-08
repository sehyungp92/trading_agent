"""Anchored walk-forward optimizer for ATRSS phased auto runs.

This runner keeps ATRSS' own synchronized execution, candidate slates, and
phase scoring, then adds robust fold uplift and feature-cluster ablation so
folds guide expected out-of-sample return/frequency without becoming a blunt
realism veto.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[4]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtests.shared.auto.phase_state import PhaseState, _atomic_write_json
from backtests.shared.auto.plugin_utils import mutation_signature
from backtests.shared.auto.types import Experiment
from backtests.swing.auto.atrss.phase_scoring import score_phase_metrics
from backtests.swing.auto.atrss.phase_scoring import RISK_ALLOCATION_PHASE_HARD_REJECTS
from backtests.swing.auto.atrss.plugin import ATRSSPlugin
from backtests.swing.auto.atrss.scoring import ATRSSMetrics
from backtests.swing.auto.config_mutator import mutate_atrss_config
from backtests.swing.config import AblationFlags, BacktestConfig, SlippageConfig
from backtests.swing.data.cache import load_bars
from backtests.swing.data.replay_cache import load_atrss_replay_bundle
from backtests.swing.engine.portfolio_engine import PortfolioResult, run_synchronized


_DEV_WORKER_DATA = None
_DEV_WORKER_CONFIG: BacktestConfig | None = None
_DEV_WORKER_EQUITY: float = 0.0
_MISSING_PREVIOUS_VALUE = "__ATRSS_WFO_MISSING_PREVIOUS_VALUE__"


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str


@dataclass
class ScoredRun:
    name: str
    score: float
    rejected: bool
    reject_reason: str
    metrics: dict[str, float]
    mutations: dict[str, Any]


@dataclass
class ValidationSummary:
    passed: bool
    pass_rate: float
    robust_score_delta_pct: float
    fold_score_delta_std_pct: float
    median_score_delta_pct: float
    mean_score_delta_pct: float
    candidate_score_mean: float
    baseline_score_mean: float
    candidate_total_r: float
    baseline_total_r: float
    candidate_trades: int
    baseline_trades: int
    catastrophic_folds: int
    fold_results: list[dict[str, Any]]


@dataclass
class AcceptedFeature:
    name: str
    phase: int
    mutations: dict[str, Any]
    previous_values: dict[str, Any]


@dataclass(frozen=True)
class RepairCandidate:
    name: str
    stage: str
    mutations: dict[str, Any]
    intent: str
    source: str = ""


class ATRSSWindowScorer:
    def __init__(
        self,
        *,
        data_dir: Path,
        symbols: tuple[str, ...],
        initial_equity: float,
        train_start: str,
        dev_end: str,
        holdout_start: str,
        holdout_end: str,
    ) -> None:
        self.data_dir = data_dir
        self.symbols = symbols
        self.initial_equity = initial_equity
        self.train_start = train_start
        self.dev_end = dev_end
        self.holdout_start = holdout_start
        self.holdout_end = holdout_end
        self._data_cache: dict[tuple[str, tuple[str, ...], str, str], Any] = {}
        self._result_cache: dict[tuple[str, tuple[str, ...], str, str, str], PortfolioResult] = {}

    def score(
        self,
        *,
        name: str,
        mutations: dict[str, Any],
        phase: int,
        scoring_profile: str,
        hard_rejects: dict[str, float],
        data_start: str,
        data_end: str,
        score_start: str,
        score_end: str,
    ) -> ScoredRun:
        config = self._config_for(mutations)
        symbols = tuple(config.symbols)
        result = self._run_result(
            mutations=mutations,
            config=config,
            symbols=symbols,
            data_start=data_start,
            data_end=data_end,
        )
        metrics = metrics_from_result_window(
            result,
            initial_equity=self.initial_equity,
            window_start=score_start,
            window_end=score_end,
        )
        score = score_phase_metrics(
            phase,
            metrics,
            hard_rejects=hard_rejects,
            profile=scoring_profile,
        )
        return ScoredRun(
            name=name,
            score=score.total,
            rejected=score.rejected,
            reject_reason=score.reject_reason,
            metrics=asdict(metrics),
            mutations=dict(mutations),
        )

    def _config_for(self, mutations: dict[str, Any]) -> BacktestConfig:
        base = BacktestConfig(
            symbols=list(self.symbols),
            initial_equity=self.initial_equity,
            fixed_qty=10,
            data_dir=self.data_dir,
            slippage=SlippageConfig(commission_per_contract=1.00),
            flags=AblationFlags(stall_exit=False),
            track_shadows=False,
        )
        return mutate_atrss_config(base, mutations)

    def _run_result(
        self,
        *,
        mutations: dict[str, Any],
        config: BacktestConfig,
        symbols: tuple[str, ...],
        data_start: str,
        data_end: str,
    ) -> PortfolioResult:
        key = (
            mutation_signature(mutations),
            symbols,
            data_start,
            data_end,
            str(self.data_dir.resolve()),
        )
        cached = self._result_cache.get(key)
        if cached is not None:
            return cached
        data = self._data_for(symbols, data_start, data_end)
        result = run_synchronized(data, config)
        self._result_cache[key] = result
        return result

    def _data_for(self, symbols: tuple[str, ...], start: str, end: str):
        key = ("atrss", symbols, start, end)
        cached = self._data_cache.get(key)
        if cached is not None:
            return cached
        bundle = load_atrss_replay_bundle(
            self.data_dir,
            symbols=symbols,
            start_date=start,
            end_date=end,
        )
        self._data_cache[key] = bundle.data
        return bundle.data


class ParallelDevScorer:
    def __init__(
        self,
        *,
        data_dir: Path,
        symbols: tuple[str, ...],
        initial_equity: float,
        data_start: str,
        data_end: str,
        max_workers: int,
    ) -> None:
        self.data_dir = data_dir
        self.symbols = symbols
        self.initial_equity = initial_equity
        self.data_start = data_start
        self.data_end = data_end
        self.max_workers = max_workers
        self._pool: mp.pool.Pool | None = None
        self._local = ATRSSWindowScorer(
            data_dir=data_dir,
            symbols=symbols,
            initial_equity=initial_equity,
            train_start=data_start,
            dev_end=data_end,
            holdout_start=data_end,
            holdout_end=data_end,
        )

    def __enter__(self) -> "ParallelDevScorer":
        if self.max_workers > 1:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(
                processes=self.max_workers,
                initializer=_init_dev_worker,
                initargs=(
                    str(self.data_dir),
                    self.symbols,
                    self.initial_equity,
                    self.data_start,
                    self.data_end,
                ),
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._pool is not None:
            if exc_type is None:
                self._pool.close()
            else:
                self._pool.terminate()
            self._pool.join()
            self._pool = None

    def score_one(
        self,
        *,
        name: str,
        mutations: dict[str, Any],
        phase: int,
        scoring_profile: str,
        hard_rejects: dict[str, float],
        score_start: str,
        score_end: str,
    ) -> ScoredRun:
        return self.score_many(
            [(name, mutations)],
            phase=phase,
            scoring_profile=scoring_profile,
            hard_rejects=hard_rejects,
            score_start=score_start,
            score_end=score_end,
        )[0]

    def score_many(
        self,
        named_mutations: list[tuple[str, dict[str, Any]]],
        *,
        phase: int,
        scoring_profile: str,
        hard_rejects: dict[str, float],
        score_start: str,
        score_end: str,
    ) -> list[ScoredRun]:
        if not named_mutations:
            return []
        if self._pool is None:
            return [
                self._local.score(
                    name=name,
                    mutations=mutations,
                    phase=phase,
                    scoring_profile=scoring_profile,
                    hard_rejects=hard_rejects,
                    data_start=self.data_start,
                    data_end=self.data_end,
                    score_start=score_start,
                    score_end=score_end,
                )
                for name, mutations in named_mutations
            ]
        payloads = [
            (name, mutations, phase, scoring_profile, hard_rejects, score_start, score_end)
            for name, mutations in named_mutations
        ]
        return self._pool.map(_score_dev_worker, payloads)


def _init_dev_worker(
    data_dir_str: str,
    symbols: tuple[str, ...],
    equity: float,
    data_start: str,
    data_end: str,
) -> None:
    global _DEV_WORKER_DATA, _DEV_WORKER_CONFIG, _DEV_WORKER_EQUITY
    data_dir = Path(data_dir_str)
    _DEV_WORKER_EQUITY = equity
    _DEV_WORKER_CONFIG = BacktestConfig(
        symbols=list(symbols),
        initial_equity=equity,
        fixed_qty=10,
        data_dir=data_dir,
        slippage=SlippageConfig(commission_per_contract=1.00),
        flags=AblationFlags(stall_exit=False),
        track_shadows=False,
    )
    _DEV_WORKER_DATA = load_atrss_replay_bundle(
        data_dir,
        symbols=symbols,
        start_date=data_start,
        end_date=data_end,
    ).data


def _score_dev_worker(payload: tuple[Any, ...]) -> ScoredRun:
    name, mutations, phase, scoring_profile, hard_rejects, score_start, score_end = payload
    try:
        if _DEV_WORKER_CONFIG is None or _DEV_WORKER_DATA is None:
            raise RuntimeError("development worker was not initialised")
        config = mutate_atrss_config(_DEV_WORKER_CONFIG, mutations)
        result = run_synchronized(_DEV_WORKER_DATA, config)
        metrics = metrics_from_result_window(
            result,
            initial_equity=_DEV_WORKER_EQUITY,
            window_start=score_start,
            window_end=score_end,
        )
        score = score_phase_metrics(
            phase,
            metrics,
            hard_rejects=hard_rejects,
            profile=scoring_profile,
        )
        return ScoredRun(
            name=name,
            score=score.total,
            rejected=score.rejected,
            reject_reason=score.reject_reason,
            metrics=asdict(metrics),
            mutations=dict(mutations),
        )
    except Exception as exc:
        return ScoredRun(
            name=name,
            score=0.0,
            rejected=True,
            reject_reason=f"Error: {exc}",
            metrics={},
            mutations=dict(mutations),
        )


def metrics_from_result_window(
    result: PortfolioResult,
    *,
    initial_equity: float,
    window_start: str,
    window_end: str,
) -> ATRSSMetrics:
    start = _start_ts(window_start)
    end = _exclusive_end_ts(window_end)
    trades = []
    for symbol_result in result.symbol_results.values():
        for trade in symbol_result.trades:
            entry_time = _trade_time(trade.entry_time)
            if entry_time is not None and start <= entry_time < end:
                trades.append(trade)
    trades.sort(key=lambda item: item.entry_time)

    n_trades = len(trades)
    if n_trades:
        r_mults = np.array([float(t.r_multiple) for t in trades], dtype=np.float64)
        wins_mask = r_mults > 0
        n_wins = int(np.sum(wins_mask))
        win_rate = n_wins / n_trades
        gross_win_r = float(np.sum(r_mults[wins_mask])) if n_wins else 0.0
        gross_loss_r = abs(float(np.sum(r_mults[~wins_mask])))
        profit_factor = gross_win_r / gross_loss_r if gross_loss_r > 0 else 999.0
        total_r = float(np.sum(r_mults))
        avg_r = float(np.mean(r_mults))
        cum_r = np.cumsum(r_mults)
        peak_r = np.maximum.accumulate(cum_r)
        dd_r = peak_r - cum_r
        max_dd_r = float(np.max(dd_r)) if len(dd_r) else 0.0
        calmar_r = total_r / max_dd_r if max_dd_r > 0 else 0.0
        captures = [
            float(t.r_multiple) / float(t.mfe_r)
            for t in trades
            if float(t.mfe_r) > 0 and float(t.r_multiple) > 0
        ]
        mfe_capture = float(np.mean(captures)) if captures else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        total_r = 0.0
        avg_r = 0.0
        max_dd_r = 0.0
        calmar_r = 0.0
        mfe_capture = 0.0

    eq, eq_ts = _equity_window(result, start, end)
    if len(eq) > 1:
        peak_eq = np.maximum.accumulate(eq)
        dd_pct = (peak_eq - eq) / np.maximum(peak_eq, 1e-9)
        max_dd_pct = float(np.max(dd_pct))
        net_return_pct = float((eq[-1] - eq[0]) / initial_equity * 100.0)
        returns = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
        sharpe = (
            float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 7))
            if len(returns) > 1 and np.std(returns) > 0
            else 0.0
        )
        span_years = max((end - start).total_seconds() / (365.25 * 86400), 0.01)
        cagr = ((eq[-1] / max(eq[0], 1e-9)) ** (1 / span_years) - 1) if eq[-1] > 0 else 0.0
        calmar = cagr / max_dd_pct if max_dd_pct > 0 else 0.0
    else:
        max_dd_pct = 0.0
        net_return_pct = 0.0
        sharpe = 0.0
        calmar = 0.0

    window_days = max((end - start).total_seconds() / 86400, 1.0)
    trades_per_month = n_trades / max(window_days / 30.44, 1.0)

    return ATRSSMetrics(
        total_trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_dd_pct=max_dd_pct,
        calmar=calmar,
        sharpe=sharpe,
        net_return_pct=net_return_pct,
        total_r=total_r,
        max_dd_r=max_dd_r,
        calmar_r=calmar_r,
        avg_r=avg_r,
        mfe_capture=mfe_capture,
        trades_per_month=trades_per_month,
    )


def run_walk_forward(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve_output_dir(Path(args.output_dir) if args.output_dir else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = tuple(_parse_symbols(args.symbols))
    folds = build_quarterly_folds(
        train_start=args.train_start,
        first_validation_start=args.first_validation_start,
        dev_end=args.dev_end,
        fold_months=args.fold_months,
        step_months=args.step_months,
    )
    scorer = ATRSSWindowScorer(
        data_dir=Path(args.data_dir),
        symbols=symbols,
        initial_equity=float(args.equity),
        train_start=args.train_start,
        dev_end=args.dev_end,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
    )
    state = PhaseState(round_name="ATRSS anchored walk-forward")
    profile_reports: list[dict[str, Any]] = []
    cumulative_mutations: dict[str, Any] = {}

    run_spec = {
        "strategy": "atrss",
        "mode": "synchronized",
        "symbols": list(symbols),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_start": args.train_start,
        "dev_end": args.dev_end,
        "holdout_start": args.holdout_start,
        "holdout_end": args.holdout_end,
        "holdout_max_min_trades": args.holdout_max_min_trades,
        "folds": [asdict(fold) for fold in folds],
        "profile_sequence": _parse_profiles(args.profile_sequence),
        "min_delta": args.min_delta,
        "validation_pass_rate": args.validation_pass_rate,
        "validation_top_k": args.validation_top_k,
        "validation_min_delta": args.validation_min_delta,
        "development_weight": args.development_weight,
        "fold_weight": args.fold_weight,
        "fold_std_penalty": args.fold_std_penalty,
        "robust_min_delta": args.robust_min_delta,
        "validation_warmup_days": args.validation_warmup_days,
        "validation_max_fold_min_trades": args.validation_max_fold_min_trades,
        "enable_fold_ablation": args.enable_fold_ablation,
        "max_ablation_rounds": args.max_ablation_rounds,
        "ablation_min_dev_delta": args.ablation_min_dev_delta,
        "ablation_robust_min_delta": args.ablation_robust_min_delta,
        "max_catastrophic_folds": args.max_catastrophic_folds,
        "catastrophe_trade_floor": args.catastrophe_trade_floor,
        "catastrophe_min_pf": args.catastrophe_min_pf,
        "catastrophe_max_dd_mult": args.catastrophe_max_dd_mult,
        "catastrophe_relative_dd_floor": args.catastrophe_relative_dd_floor,
        "catastrophe_max_abs_dd": args.catastrophe_max_abs_dd,
        "allow_symbol_expansion": args.allow_symbol_expansion,
    }
    _atomic_write_json(run_spec, output_dir / "run_spec.json")

    for profile in _parse_profiles(args.profile_sequence):
        plugin = ATRSSPlugin(
            data_dir=Path(args.data_dir),
            initial_equity=float(args.equity),
            mode="synchronized",
            symbols=list(symbols),
            candidate_profile=profile,
        )
        plugin.initial_mutations = dict(cumulative_mutations)
        active_profile = plugin._set_active_candidate_profile(profile)
        profile_state = PhaseState(
            cumulative_mutations=dict(cumulative_mutations),
            round_name=f"ATRSS anchored WFO {active_profile}",
        )
        report = run_profile(
            profile=active_profile,
            plugin=plugin,
            state=profile_state,
            scorer=scorer,
            folds=folds,
            output_dir=output_dir,
            args=args,
        )
        profile_reports.append(report)
        cumulative_mutations = dict(profile_state.cumulative_mutations)
        state.cumulative_mutations = dict(cumulative_mutations)
        state.completed_phases = sorted(set(state.completed_phases + profile_state.completed_phases))
        state.phase_results.update(
            {
                int(f"{1 if active_profile == 'alpha' else 2}{phase}"): result
                for phase, result in profile_state.phase_results.items()
            }
        )
        _atomic_write_json(asdict_state(state), output_dir / "phase_state.json")
        _atomic_write_json(cumulative_mutations, output_dir / "optimized_config.json")

    final_profile = profile_reports[-1]["profile"] if profile_reports else "alpha"
    final_plugin = ATRSSPlugin(
        data_dir=Path(args.data_dir),
        initial_equity=float(args.equity),
        mode="synchronized",
        symbols=list(symbols),
        candidate_profile=final_profile,
    )
    final_plugin._set_active_candidate_profile(final_profile)
    holdout = scorer.score(
        name="holdout",
        mutations=cumulative_mutations,
        phase=4,
        scoring_profile=final_plugin._scoring_profile,
        hard_rejects=validation_hard_rejects(
            {"min_trades": 240, "max_dd_pct": 0.08, "min_pf": 2.2, "min_wr": 0.62},
            args.train_start,
            args.dev_end,
            args.holdout_start,
            args.holdout_end,
            max_min_trades=args.holdout_max_min_trades,
        ),
        data_start=args.train_start,
        data_end=args.holdout_end,
        score_start=args.holdout_start,
        score_end=args.holdout_end,
    )
    dev = scorer.score(
        name="development",
        mutations=cumulative_mutations,
        phase=4,
        scoring_profile=final_plugin._scoring_profile,
        hard_rejects={"min_trades": 1, "max_dd_pct": 1.0, "min_pf": 0.0, "min_wr": 0.0},
        data_start=args.train_start,
        data_end=args.dev_end,
        score_start=args.train_start,
        score_end=args.dev_end,
    )
    summary = {
        "output_dir": str(output_dir.resolve()),
        "optimized_config": cumulative_mutations,
        "profile_reports": profile_reports,
        "development": asdict(dev),
        "holdout": asdict(holdout),
    }
    _atomic_write_json(summary, output_dir / "summary.json")
    write_text_summary(output_dir / "summary.txt", summary)
    return summary


def run_incumbent_repair(args: argparse.Namespace) -> dict[str, Any]:
    """Refine the current ATRSS incumbent with granular ablation and OOS-targeted probes.

    This mode intentionally treats the 2026 window as selection OOS, not as an
    untouched holdout. It is meant for the user's stated goal here: maximize
    expected return and trade frequency while preserving development strength.
    """

    output_dir = _resolve_output_dir(
        Path(args.output_dir) if args.output_dir else None,
        prefix="atrss_incumbent_repair",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = tuple(_parse_symbols(args.symbols))
    folds = build_quarterly_folds(
        train_start=args.train_start,
        first_validation_start=args.first_validation_start,
        dev_end=args.dev_end,
        fold_months=args.fold_months,
        step_months=args.step_months,
    )
    scorer = ATRSSWindowScorer(
        data_dir=Path(args.data_dir),
        symbols=symbols,
        initial_equity=float(args.equity),
        train_start=args.train_start,
        dev_end=args.dev_end,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
    )
    incumbent_path = Path(args.incumbent_config)
    incumbent = load_config_json(incumbent_path)
    historical_features = build_historical_acceptance_features(Path(args.prior_rounds_dir))
    phase = 4
    scoring_profile = "r11_risk_allocation"
    hard_rejects = dict(RISK_ALLOCATION_PHASE_HARD_REJECTS.get(phase, {}))

    run_spec = {
        "strategy": "atrss",
        "mode": "incumbent_repair",
        "symbols": list(symbols),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "incumbent_config_path": str(incumbent_path),
        "incumbent_config": incumbent,
        "selection_oos_start": args.holdout_start,
        "selection_oos_end": args.holdout_end,
        "selection_oos_note": (
            "The OOS window is used for tuning/selection in this repair run; "
            "it is no longer an untouched holdout."
        ),
        "train_start": args.train_start,
        "dev_end": args.dev_end,
        "folds": [asdict(fold) for fold in folds],
        "stage_sequence": _parse_stage_sequence(args.repair_stage_sequence),
        "repair_validation_top_k": args.repair_validation_top_k,
        "repair_max_stage_rounds": args.repair_max_stage_rounds,
        "repair_objective_weights": {
            "development": args.repair_dev_weight,
            "folds": args.repair_fold_weight,
            "selection_oos_score": args.repair_oos_score_weight,
            "selection_oos_return": args.repair_oos_return_weight,
            "selection_oos_trades": args.repair_oos_trade_weight,
        },
        "historical_feature_count": len(historical_features),
    }
    _atomic_write_json(run_spec, output_dir / "run_spec.json")

    current = dict(incumbent)
    stage_reports: list[dict[str, Any]] = []
    with ParallelDevScorer(
        data_dir=scorer.data_dir,
        symbols=scorer.symbols,
        initial_equity=scorer.initial_equity,
        data_start=args.train_start,
        data_end=args.dev_end,
        max_workers=args.max_workers,
    ) as dev_scorer:
        baseline_dev = dev_scorer.score_one(
            name="incumbent_development",
            mutations=current,
            phase=phase,
            scoring_profile=scoring_profile,
            hard_rejects=hard_rejects,
            score_start=args.train_start,
            score_end=args.dev_end,
        )
        baseline_oos = score_selection_oos(
            scorer=scorer,
            mutations=current,
            scoring_profile=scoring_profile,
            hard_rejects=hard_rejects,
            args=args,
            name="incumbent_selection_oos",
        )

        print(
            "[incumbent repair] "
            f"dev={baseline_dev.score:.4f}, selection_oos={baseline_oos.score:.4f}, "
            f"features={len(historical_features)}, workers={args.max_workers}",
            flush=True,
        )

        for stage in _parse_stage_sequence(args.repair_stage_sequence):
            current, report = run_repair_stage(
                stage=stage,
                current_mutations=current,
                historical_features=historical_features,
                dev_scorer=dev_scorer,
                scorer=scorer,
                folds=folds,
                phase=phase,
                scoring_profile=scoring_profile,
                hard_rejects=hard_rejects,
                args=args,
            )
            stage_reports.append(report)
            _atomic_write_json(stage_reports, output_dir / "stage_reports.json")
            _atomic_write_json(current, output_dir / "optimized_config.json")

        final_dev = dev_scorer.score_one(
            name="final_development",
            mutations=current,
            phase=phase,
            scoring_profile=scoring_profile,
            hard_rejects=hard_rejects,
            score_start=args.train_start,
            score_end=args.dev_end,
        )

    final_oos = score_selection_oos(
        scorer=scorer,
        mutations=current,
        scoring_profile=scoring_profile,
        hard_rejects=hard_rejects,
        args=args,
        name="final_selection_oos",
    )
    summary = {
        "output_dir": str(output_dir.resolve()),
        "optimized_config": current,
        "baseline": {
            "development": asdict(baseline_dev),
            "selection_oos": asdict(baseline_oos),
        },
        "final": {
            "development": asdict(final_dev),
            "selection_oos": asdict(final_oos),
        },
        "stage_reports": stage_reports,
        "selection_oos_note": run_spec["selection_oos_note"],
    }
    _atomic_write_json(summary, output_dir / "summary.json")
    _atomic_write_json(current, output_dir / "optimized_config.json")
    write_repair_text_summary(output_dir / "summary.txt", summary)
    return summary


def run_repair_stage(
    *,
    stage: str,
    current_mutations: dict[str, Any],
    historical_features: list[AcceptedFeature],
    dev_scorer: ParallelDevScorer,
    scorer: ATRSSWindowScorer,
    folds: list[Fold],
    phase: int,
    scoring_profile: str,
    hard_rejects: dict[str, float],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = dict(current_mutations)
    rounds: list[dict[str, Any]] = []
    accepted_reports: list[dict[str, Any]] = []
    current_dev = dev_scorer.score_one(
        name=f"{stage}_baseline_development",
        mutations=current,
        phase=phase,
        scoring_profile=scoring_profile,
        hard_rejects=hard_rejects,
        score_start=args.train_start,
        score_end=args.dev_end,
    )
    current_oos = score_selection_oos(
        scorer=scorer,
        mutations=current,
        scoring_profile=scoring_profile,
        hard_rejects=hard_rejects,
        args=args,
        name=f"{stage}_baseline_selection_oos",
    )

    for round_num in range(1, args.repair_max_stage_rounds + 1):
        candidates = build_repair_candidates(
            stage=stage,
            current=current,
            historical_features=historical_features,
            args=args,
        )
        candidates = dedupe_repair_candidates(candidates, current)
        if not candidates:
            rounds.append(
                {
                    "round_num": round_num,
                    "candidate_count": 0,
                    "accepted": None,
                    "stop_reason": "no_candidates",
                }
            )
            break

        scored = dev_scorer.score_many(
            [(candidate.name, candidate.mutations) for candidate in candidates],
            phase=phase,
            scoring_profile=scoring_profile,
            hard_rejects=hard_rejects,
            score_start=args.train_start,
            score_end=args.dev_end,
        )
        score_by_name = {item.name: item for item in scored}
        preliminary: list[tuple[float, RepairCandidate, ScoredRun, dict[str, Any]]] = []
        prefilter_reports: list[dict[str, Any]] = []
        for candidate in candidates:
            dev = score_by_name[candidate.name]
            dev_delta = _delta_ratio(dev.score, current_dev.score)
            gate = repair_development_gate(current_dev.metrics, dev.metrics, args)
            prelim_score = repair_preliminary_score(current_dev.metrics, dev.metrics, dev_delta)
            report = {
                "candidate": asdict(candidate),
                "development": asdict(dev),
                "development_score_delta_pct": dev_delta * 100.0,
                "preliminary_score": prelim_score,
                "prefilter_passed": (not dev.rejected) and gate["passed"],
                "prefilter_reasons": gate["reasons"],
            }
            prefilter_reports.append(report)
            if dev.rejected or not gate["passed"]:
                continue
            preliminary.append((prelim_score, candidate, dev, report))

        ranked = sorted(preliminary, key=lambda item: item[0], reverse=True)
        evaluations: list[dict[str, Any]] = []
        evaluated_options: list[tuple[dict[str, Any], RepairCandidate, ScoredRun]] = []
        for _, candidate, dev, prefilter in ranked[: max(args.repair_validation_top_k, 1)]:
            dev_delta = _delta_ratio(dev.score, current_dev.score)
            validation = validate_candidate(
                candidate_name=candidate.name,
                phase=phase,
                scoring_profile=scoring_profile,
                hard_rejects=hard_rejects,
                current_mutations=current,
                candidate_mutations=candidate.mutations,
                development_delta=dev_delta,
                scorer=scorer,
                folds=folds,
                args=args,
            )
            candidate_oos = score_selection_oos(
                scorer=scorer,
                mutations=candidate.mutations,
                scoring_profile=scoring_profile,
                hard_rejects=hard_rejects,
                args=args,
                name=f"{stage}_{round_num}_{candidate.name}_selection_oos",
            )
            evaluation = build_repair_evaluation(
                candidate=candidate,
                current_dev=current_dev,
                candidate_dev=dev,
                current_oos=current_oos,
                candidate_oos=candidate_oos,
                validation=validation,
                args=args,
            )
            evaluation["prefilter"] = prefilter
            evaluations.append(evaluation)
            evaluated_options.append((evaluation, candidate, dev))
            print(
                f"  [{stage} round {round_num}] {candidate.name}: "
                f"obj {evaluation['objective_delta_pct']:.2f}% | "
                f"dev {evaluation['development_score_delta_pct']:.2f}% | "
                f"fold robust {validation.robust_score_delta_pct:.2f}% | "
                f"oos score {evaluation['selection_oos_score_delta_pct']:.2f}% | "
                f"oos trades {evaluation['selection_oos_trade_delta_pct']:.2f}%",
                flush=True,
            )

        eligible = [
            item
            for item in evaluated_options
            if item[0]["passed"] and item[0]["objective_delta"] >= args.repair_min_objective_delta
        ]
        if not eligible:
            best = max(evaluations, key=lambda item: item["objective_delta"], default=None)
            rounds.append(
                {
                    "round_num": round_num,
                    "candidate_count": len(candidates),
                    "prefiltered_count": len(ranked),
                    "evaluated_count": len(evaluations),
                    "accepted": None,
                    "best_candidate": best,
                    "prefilter_reports": prefilter_reports,
                    "evaluations": evaluations,
                    "stop_reason": "no_candidate_met_repair_acceptance",
                }
            )
            break

        best_eval, best_candidate, best_dev = max(
            eligible,
            key=lambda item: item[0]["objective_delta"],
        )
        current = dict(best_candidate.mutations)
        current_dev = best_dev
        current_oos = score_selection_oos(
            scorer=scorer,
            mutations=current,
            scoring_profile=scoring_profile,
            hard_rejects=hard_rejects,
            args=args,
            name=f"{stage}_{round_num}_accepted_selection_oos",
        )
        accepted_reports.append(best_eval)
        rounds.append(
            {
                "round_num": round_num,
                "candidate_count": len(candidates),
                "prefiltered_count": len(ranked),
                "evaluated_count": len(evaluations),
                "accepted": best_eval,
                "prefilter_reports": prefilter_reports,
                "evaluations": evaluations,
            }
        )
        print(
            f"  [{stage} round {round_num}] accepted {best_candidate.name} "
            f"(objective +{best_eval['objective_delta_pct']:.2f}%)",
            flush=True,
        )

    return current, {
        "stage": stage,
        "initial_mutations": dict(current_mutations),
        "final_mutations": dict(current),
        "accepted": accepted_reports,
        "rounds": rounds,
    }


def score_selection_oos(
    *,
    scorer: ATRSSWindowScorer,
    mutations: dict[str, Any],
    scoring_profile: str,
    hard_rejects: dict[str, float],
    args: argparse.Namespace,
    name: str,
) -> ScoredRun:
    return scorer.score(
        name=name,
        mutations=mutations,
        phase=4,
        scoring_profile=scoring_profile,
        hard_rejects=validation_hard_rejects(
            hard_rejects,
            args.train_start,
            args.dev_end,
            args.holdout_start,
            args.holdout_end,
            max_min_trades=args.holdout_max_min_trades,
        ),
        data_start=args.train_start,
        data_end=args.holdout_end,
        score_start=args.holdout_start,
        score_end=args.holdout_end,
    )


def run_profile(
    *,
    profile: str,
    plugin: ATRSSPlugin,
    state: PhaseState,
    scorer: ATRSSWindowScorer,
    folds: list[Fold],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    profile_dir = output_dir / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    phase_reports: list[dict[str, Any]] = []
    for phase in range(1, plugin.num_phases + 1):
        spec = plugin.get_phase_spec(phase, state)
        candidates = add_prior_seed_candidates(
            spec.candidates,
            profile=profile,
            prior_rounds_dir=Path(args.prior_rounds_dir),
        )
        candidates = filter_candidates(
            candidates,
            base_symbols=tuple(scorer.symbols),
            data_dir=scorer.data_dir,
            required_end=args.holdout_end,
            allow_symbol_expansion=args.allow_symbol_expansion,
        )
        current_mutations = dict(state.cumulative_mutations)
        remaining = list(candidates)
        kept: list[str] = []
        accepted_features: list[AcceptedFeature] = []
        rounds: list[dict[str, Any]] = []
        ablation_reports: list[dict[str, Any]] = []
        max_rounds = args.max_rounds if args.max_rounds is not None else len(remaining)
        with ParallelDevScorer(
            data_dir=scorer.data_dir,
            symbols=scorer.symbols,
            initial_equity=scorer.initial_equity,
            data_start=args.train_start,
            data_end=args.dev_end,
            max_workers=args.max_workers,
        ) as dev_scorer:
            base = dev_scorer.score_one(
                name="__baseline__",
                mutations=current_mutations,
                phase=phase,
                scoring_profile=plugin._scoring_profile,
                hard_rejects=spec.hard_rejects,
                score_start=args.train_start,
                score_end=args.dev_end,
            )
            current_score = base.score
            print(
                f"[{profile} phase {phase}] {spec.focus}: baseline={current_score:.4f}, "
                f"candidates={len(remaining)}, workers={args.max_workers}",
                flush=True,
            )

            for round_num in range(1, max_rounds + 1):
                if not remaining:
                    break
                scored = dev_scorer.score_many(
                    [
                        (candidate.name, {**current_mutations, **candidate.mutations})
                        for candidate in remaining
                    ],
                    phase=phase,
                    scoring_profile=plugin._scoring_profile,
                    hard_rejects=spec.hard_rejects,
                    score_start=args.train_start,
                    score_end=args.dev_end,
                )
                valid = [item for item in scored if not item.rejected]
                ranked = sorted(valid, key=lambda item: item.score, reverse=True)
                rejected_count = len(scored) - len(valid)
                accepted: ScoredRun | None = None
                accepted_validation: ValidationSummary | None = None
                validation_attempts: list[dict[str, Any]] = []
                for candidate_score in ranked[: max(args.validation_top_k, 1)]:
                    delta_ratio = _delta_ratio(candidate_score.score, current_score)
                    if candidate_score.score <= current_score or delta_ratio < args.min_delta:
                        break
                    candidate = next(item for item in remaining if item.name == candidate_score.name)
                    merged = {**current_mutations, **candidate.mutations}
                    validation = validate_candidate(
                        candidate_name=candidate.name,
                        phase=phase,
                        scoring_profile=plugin._scoring_profile,
                        hard_rejects=spec.hard_rejects,
                        current_mutations=current_mutations,
                        candidate_mutations=merged,
                        development_delta=delta_ratio,
                        scorer=scorer,
                        folds=folds,
                        args=args,
                    )
                    validation_attempts.append(
                        {
                            "name": candidate.name,
                            "dev_score": candidate_score.score,
                            "dev_delta_pct": delta_ratio * 100.0,
                            "validation": asdict(validation),
                        }
                    )
                    print(
                        f"  round {round_num}: {candidate.name} dev +{delta_ratio * 100:.2f}% "
                        f"robust {validation.robust_score_delta_pct:.2f}% "
                        f"fold mean {validation.mean_score_delta_pct:.2f}%",
                        flush=True,
                    )
                    if validation.passed:
                        accepted = candidate_score
                        accepted_validation = validation
                        break

                if accepted is None:
                    best_name = ranked[0].name if ranked else ""
                    best_score = ranked[0].score if ranked else current_score
                    rounds.append(
                        {
                            "round_num": round_num,
                            "candidates_tested": len(remaining),
                            "best_name": best_name,
                            "best_score": best_score,
                            "best_delta_pct": _delta_ratio(best_score, current_score) * 100.0,
                            "kept": False,
                            "rejected_count": rejected_count,
                            "validation_attempts": validation_attempts,
                            "stop_reason": "no_fold_validated_candidate",
                        }
                    )
                    break

                chosen = next(item for item in remaining if item.name == accepted.name)
                previous_values = {
                    key: current_mutations.get(key, _MISSING_PREVIOUS_VALUE)
                    for key in chosen.mutations
                    if current_mutations.get(key, _MISSING_PREVIOUS_VALUE) != chosen.mutations[key]
                }
                current_mutations.update(chosen.mutations)
                current_score = accepted.score
                kept.append(accepted.name)
                if previous_values:
                    accepted_features.append(
                        AcceptedFeature(
                            name=accepted.name,
                            phase=phase,
                            mutations={
                                key: chosen.mutations[key]
                                for key in previous_values
                            },
                            previous_values=previous_values,
                        )
                    )
                remaining = [item for item in remaining if item.name != accepted.name]
                rounds.append(
                    {
                        "round_num": round_num,
                        "candidates_tested": len(scored),
                        "best_name": accepted.name,
                        "best_score": accepted.score,
                        "best_delta_pct": _delta_ratio(accepted.score, base.score) * 100.0,
                        "kept": True,
                        "rejected_count": rejected_count,
                        "validation": asdict(accepted_validation) if accepted_validation else None,
                        "validation_attempts": validation_attempts,
                    }
                )

            final = dev_scorer.score_one(
                name=f"{profile}_phase_{phase}_final",
                mutations=current_mutations,
                phase=phase,
                scoring_profile=plugin._scoring_profile,
                hard_rejects=spec.hard_rejects,
                score_start=args.train_start,
                score_end=args.dev_end,
            )
            if args.enable_fold_ablation and accepted_features:
                current_mutations, final, ablation_reports = run_feature_ablation_prune(
                    profile=profile,
                    phase=phase,
                    current_mutations=current_mutations,
                    current_final=final,
                    accepted_features=accepted_features,
                    dev_scorer=dev_scorer,
                    scorer=scorer,
                    folds=folds,
                    scoring_profile=plugin._scoring_profile,
                    hard_rejects=spec.hard_rejects,
                    args=args,
                )
        phase_report = {
            "profile": profile,
            "phase": phase,
            "focus": spec.focus,
            "base_score": base.score,
            "final_score": final.score,
            "base_metrics": base.metrics,
            "final_metrics": final.metrics,
            "base_mutations": dict(state.cumulative_mutations),
            "final_mutations": dict(current_mutations),
            "new_mutations": {
                key: value
                for key, value in current_mutations.items()
                if state.cumulative_mutations.get(key) != value
            },
            "kept_features": kept,
            "accepted_feature_mutations": [asdict(item) for item in accepted_features],
            "fold_ablation": ablation_reports,
            "rounds": rounds,
            "candidate_count": len(candidates),
        }
        state.cumulative_mutations = dict(current_mutations)
        state.completed_phases.append(phase)
        state.phase_results[phase] = phase_report
        phase_reports.append(phase_report)
        _atomic_write_json(phase_report, profile_dir / f"phase_{phase}.json")
        _atomic_write_json(asdict_state(state), profile_dir / "phase_state.json")
        _atomic_write_json(state.cumulative_mutations, profile_dir / "optimized_config.json")
        print(
            f"[{profile} phase {phase}] final={final.score:.4f}, accepted={len(kept)}",
            flush=True,
        )
    return {
        "profile": profile,
        "phases": phase_reports,
        "final_mutations": dict(state.cumulative_mutations),
    }


def run_feature_ablation_prune(
    *,
    profile: str,
    phase: int,
    current_mutations: dict[str, Any],
    current_final: ScoredRun,
    accepted_features: list[AcceptedFeature],
    dev_scorer: ParallelDevScorer,
    scorer: ATRSSWindowScorer,
    folds: list[Fold],
    scoring_profile: str,
    hard_rejects: dict[str, float],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], ScoredRun, list[dict[str, Any]]]:
    """Remove accepted feature clusters only when fold-ablation says removal helps."""

    current = dict(current_mutations)
    final = current_final
    reports: list[dict[str, Any]] = []
    remaining = build_ablation_clusters(accepted_features)
    if not remaining:
        return current, final, reports

    for round_num in range(1, args.max_ablation_rounds + 1):
        attempts: list[dict[str, Any]] = []
        removable: list[dict[str, Any]] = []
        for feature in remaining:
            ablated = revert_feature(current, feature)
            if ablated == current:
                continue
            dev = dev_scorer.score_one(
                name=f"ablate_{feature.name}",
                mutations=ablated,
                phase=phase,
                scoring_profile=scoring_profile,
                hard_rejects=hard_rejects,
                score_start=args.train_start,
                score_end=args.dev_end,
            )
            dev_delta = _delta_ratio(dev.score, final.score)
            attempt = {
                "round_num": round_num,
                "feature": asdict(feature),
                "dev_score": dev.score,
                "dev_delta_pct": dev_delta * 100.0,
                "removed": False,
                "reason": "",
            }
            if dev.rejected:
                attempt["reason"] = f"dev_rejected: {dev.reject_reason}"
                attempts.append(attempt)
                continue
            if dev.score <= final.score or dev_delta < args.ablation_min_dev_delta:
                attempt["reason"] = "dev_uplift_too_small"
                attempts.append(attempt)
                continue

            validation = validate_candidate(
                candidate_name=f"ablate_{feature.name}",
                phase=phase,
                scoring_profile=scoring_profile,
                hard_rejects=hard_rejects,
                current_mutations=current,
                candidate_mutations=ablated,
                development_delta=dev_delta,
                scorer=scorer,
                folds=folds,
                args=args,
            )
            ablation_passed = (
                validation.robust_score_delta_pct >= args.ablation_robust_min_delta * 100.0
                and validation.catastrophic_folds <= args.max_catastrophic_folds
                and validation.candidate_trades >= max(
                    1,
                    int(validation.baseline_trades * args.validation_trade_floor),
                )
                and validation.candidate_total_r >= (
                    validation.baseline_total_r * args.validation_total_r_floor
                )
            )
            attempt.update(
                {
                    "validation": asdict(validation),
                    "ablation_passed": ablation_passed,
                    "reason": "passed" if ablation_passed else "fold_uplift_too_weak",
                }
            )
            attempts.append(attempt)
            if ablation_passed:
                removable.append(
                    {
                        "feature": feature,
                        "ablated_mutations": ablated,
                        "dev": dev,
                        "validation": validation,
                        "robust_delta_pct": validation.robust_score_delta_pct,
                    }
                )

        if not removable:
            reports.append(
                {
                    "round_num": round_num,
                    "removed_feature": None,
                    "attempts": attempts,
                    "stop_reason": "no_removal_improved_robust_uplift",
                }
            )
            break

        best = max(
            removable,
            key=lambda item: (
                item["robust_delta_pct"],
                _delta_ratio(item["dev"].score, final.score),
            ),
        )
        feature = best["feature"]
        current = dict(best["ablated_mutations"])
        final = best["dev"]
        reports.append(
            {
                "round_num": round_num,
                "removed_feature": asdict(feature),
                "dev_score": final.score,
                "robust_delta_pct": best["robust_delta_pct"],
                "validation": asdict(best["validation"]),
                "attempts": attempts,
            }
        )
        print(
            f"  ablation {profile} phase {phase}: removed {feature.name} "
            f"robust +{best['robust_delta_pct']:.2f}%",
            flush=True,
        )
        removed_keys = set(feature.mutations)
        remaining = [
            item
            for item in remaining
            if item.name != feature.name and not (set(item.mutations) & removed_keys)
        ]
        if not remaining:
            break

    return current, final, reports


def build_ablation_clusters(features: list[AcceptedFeature]) -> list[AcceptedFeature]:
    clusters: list[AcceptedFeature] = []
    for feature in features:
        grouped: dict[str, AcceptedFeature] = {}
        for key, value in feature.mutations.items():
            group = ablation_cluster_name(key)
            cluster_name = f"{feature.name}:{group}" if len(feature.mutations) > 1 else feature.name
            cluster = grouped.setdefault(
                cluster_name,
                AcceptedFeature(
                    name=cluster_name,
                    phase=feature.phase,
                    mutations={},
                    previous_values={},
                ),
            )
            cluster.mutations[key] = value
            cluster.previous_values[key] = feature.previous_values.get(key, _MISSING_PREVIOUS_VALUE)
        clusters.extend(grouped.values())
    return [cluster for cluster in clusters if cluster.mutations]


def ablation_cluster_name(key: str) -> str:
    if key == "symbols":
        return "symbol_universe"
    if key == "fixed_qty" or key.startswith("param_overrides.fixed_qty"):
        return "risk_sizing"
    if key in {
        "param_overrides.base_risk_pct",
        "param_overrides.max_portfolio_heat",
    } or key.startswith("param_overrides.dynamic_risk_"):
        return "risk_sizing"
    if key == "flags.addon_b" or key.startswith("param_overrides.addon_b_"):
        return "addon_b"
    if key == "flags.addon_a" or key.startswith("param_overrides.addon_a_"):
        return "addon_a"
    if key == "flags.early_stall_exit" or key.startswith("param_overrides.early_stall_"):
        return "early_stall_exit"
    if key == "flags.stall_exit" or key.startswith("param_overrides.stall_"):
        return "stall_exit"
    if key == "param_overrides.max_hold_hours":
        return "time_exit"
    if key.startswith("slippage.") or key.startswith("param_overrides.limit_"):
        return "execution"
    if key in {"flags.slippage_abort"} or key.startswith("param_overrides.max_entry_slip"):
        return "execution"
    if (
        key.startswith("param_overrides.adx_")
        or key.startswith("param_overrides.fast_confirm_")
        or key.startswith("param_overrides.confirm_days_")
        or key.startswith("param_overrides.shorts_enabled_")
        or key in {
            "param_overrides.recovery_tolerance_atr_trend",
            "flags.short_safety",
            "flags.quality_gate",
        }
    ):
        return "signal_regime"
    return key.replace(".", "_")


def revert_feature(current: dict[str, Any], feature: AcceptedFeature) -> dict[str, Any]:
    reverted = dict(current)
    for key in feature.mutations:
        previous = feature.previous_values.get(key, _MISSING_PREVIOUS_VALUE)
        if previous == _MISSING_PREVIOUS_VALUE:
            reverted.pop(key, None)
        else:
            reverted[key] = previous
    return reverted


def load_config_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected config object in {path}")
    return data


def build_historical_acceptance_features(prior_rounds_dir: Path) -> list[AcceptedFeature]:
    """Recover granular active mutation clusters from prior phased auto outputs."""

    features: list[AcceptedFeature] = []
    for round_num in (1, 2, 3):
        path = prior_rounds_dir / f"round_{round_num}" / "phase_state.json"
        if not path.exists():
            continue
        state = load_config_json(path)
        phase_results = state.get("phase_results", {})
        for phase_key in sorted(phase_results, key=lambda value: int(value)):
            report = phase_results[phase_key]
            base = report.get("base_mutations") or {}
            final = report.get("final_mutations") or {}
            changed_keys = [
                key
                for key, value in final.items()
                if base.get(key, _MISSING_PREVIOUS_VALUE) != value
            ]
            grouped: dict[str, AcceptedFeature] = {}
            for key in changed_keys:
                group = ablation_cluster_name(key)
                name = f"round_{round_num}_phase_{phase_key}_{group}"
                feature = grouped.setdefault(
                    name,
                    AcceptedFeature(
                        name=name,
                        phase=int(phase_key),
                        mutations={},
                        previous_values={},
                    ),
                )
                feature.mutations[key] = final[key]
                feature.previous_values[key] = base.get(key, _MISSING_PREVIOUS_VALUE)
            features.extend(grouped.values())
    return features


def build_repair_candidates(
    *,
    stage: str,
    current: dict[str, Any],
    historical_features: list[AcceptedFeature],
    args: argparse.Namespace,
) -> list[RepairCandidate]:
    if stage == "ablation":
        return build_incumbent_ablation_candidates(current, historical_features)
    if stage == "perturbation":
        return build_incumbent_perturbation_candidates(current)
    if stage == "targeted":
        return build_targeted_oos_candidates(current, args)
    raise ValueError(f"Unsupported repair stage: {stage}")


def build_incumbent_ablation_candidates(
    current: dict[str, Any],
    historical_features: list[AcceptedFeature],
) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []
    for feature in historical_features:
        active_keys = [
            key
            for key, value in feature.mutations.items()
            if current.get(key, _MISSING_PREVIOUS_VALUE) == value
        ]
        if not active_keys:
            continue
        active_feature = AcceptedFeature(
            name=feature.name,
            phase=feature.phase,
            mutations={key: feature.mutations[key] for key in active_keys},
            previous_values={
                key: feature.previous_values.get(key, _MISSING_PREVIOUS_VALUE)
                for key in active_keys
            },
        )
        if len(active_keys) > 1:
            ablated = revert_feature(current, active_feature)
            candidates.append(
                RepairCandidate(
                    name=f"ablate_cluster_{feature.name}",
                    stage="ablation",
                    mutations=ablated,
                    intent="Remove the active historical mutation cluster.",
                    source=feature.name,
                )
            )
        for key in active_keys:
            key_feature = AcceptedFeature(
                name=f"{feature.name}:{key}",
                phase=feature.phase,
                mutations={key: feature.mutations[key]},
                previous_values={key: feature.previous_values.get(key, _MISSING_PREVIOUS_VALUE)},
            )
            candidates.append(
                RepairCandidate(
                    name=f"ablate_key_{_short_key(key)}",
                    stage="ablation",
                    mutations=revert_feature(current, key_feature),
                    intent="Remove one accepted mutation while preserving the rest of the incumbent.",
                    source=feature.name,
                )
            )
    return candidates


def build_incumbent_perturbation_candidates(current: dict[str, Any]) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []
    stage = "perturbation"

    for value in (0.020, 0.021, 0.022, 0.023, 0.0235, 0.0245, 0.025, 0.026, 0.027):
        _add_patch_candidate(
            candidates,
            current,
            stage,
            f"base_risk_{int(round(value * 10000)):03d}bp",
            {"param_overrides.base_risk_pct": value},
            "Local risk percent sweep around the current dynamic-risk incumbent.",
        )

    for strong, weak in (
        (1.05, 0.85),
        (1.10, 0.85),
        (1.10, 0.80),
        (1.15, 0.75),
        (1.20, 0.80),
        (1.20, 0.75),
        (1.25, 0.70),
        (1.30, 0.70),
    ):
        _add_patch_candidate(
            candidates,
            current,
            stage,
            f"dynamic_scale_{int(strong * 100):03d}_{int(weak * 100):03d}",
            {
                "param_overrides.dynamic_risk_strong_trend_mult": strong,
                "param_overrides.dynamic_risk_weak_trend_mult": weak,
            },
            "Perturb dynamic risk scaling without changing signal geometry.",
        )

    for hours in (64, 72, 80, 96, 104, 120):
        _add_patch_candidate(
            candidates,
            current,
            stage,
            f"max_hold_{hours}h",
            {"param_overrides.max_hold_hours": hours},
            "Test whether capital-release timing improves return/frequency.",
        )

    for addon_r, size in (
        (1.00, 0.25),
        (1.25, 0.25),
        (1.25, 0.35),
        (1.50, 0.35),
        (1.50, 0.65),
        (1.75, 0.35),
        (1.75, 0.50),
        (2.00, 0.35),
    ):
        _add_patch_candidate(
            candidates,
            current,
            stage,
            f"addon_a_{int(addon_r * 100):03d}r_{int(size * 100):02d}",
            {
                "param_overrides.addon_a_r": addon_r,
                "param_overrides.addon_a_size_mult": size,
            },
            "Perturb add-on A winner scaling around the accepted setting.",
        )
    _add_patch_candidate(
        candidates,
        current,
        stage,
        "disable_addon_a",
        {"flags.addon_a": False},
        "Check whether add-on A complexity is still adding net expectancy.",
    )

    for adx in (12, 13, 15, 16):
        _add_patch_candidate(
            candidates,
            current,
            stage,
            f"adx_on_{adx}",
            {"param_overrides.adx_on": adx},
            "Local ADX regime threshold perturbation.",
        )
    for recov in (0.35, 0.45, 0.65, 0.75):
        _add_patch_candidate(
            candidates,
            current,
            stage,
            f"recovery_trend_{int(recov * 100):02d}",
            {"param_overrides.recovery_tolerance_atr_trend": recov},
            "Local pullback recovery tolerance perturbation.",
        )
    for adx, recov in ((13, 0.65), (12, 0.65), (15, 0.45)):
        _add_patch_candidate(
            candidates,
            current,
            stage,
            f"adx_{adx}_recovery_{int(recov * 100):02d}",
            {
                "param_overrides.adx_on": adx,
                "param_overrides.recovery_tolerance_atr_trend": recov,
            },
            "Joint local signal perturbation for frequency/quality balance.",
        )

    for hours, mfe in ((8, 0.20), (12, 0.30), (16, 0.30)):
        _add_patch_candidate(
            candidates,
            current,
            stage,
            f"early_stall_on_{hours}h_{int(mfe * 100):02d}",
            {
                "flags.early_stall_exit": True,
                "param_overrides.early_stall_check_hours": hours,
                "param_overrides.early_stall_mfe_threshold": mfe,
            },
            "Re-test partial early stall as a capital-release perturbation.",
        )

    for addon_r, size in ((2.50, 0.10), (2.50, 0.15), (3.00, 0.10), (3.00, 0.15)):
        _add_patch_candidate(
            candidates,
            current,
            stage,
            f"addon_b_{int(addon_r * 100):03d}r_{int(size * 100):02d}",
            {
                "flags.addon_b": True,
                "param_overrides.addon_b_r": addon_r,
                "param_overrides.addon_b_size_mult": size,
            },
            "Small add-on B winner lean-in perturbation.",
        )

    for heat in (0.050, 0.055, 0.065, 0.070):
        _add_patch_candidate(
            candidates,
            current,
            stage,
            f"heat_{int(heat * 1000):03d}",
            {"param_overrides.max_portfolio_heat": heat},
            "Portfolio heat guardrail perturbation.",
        )

    return candidates


def build_targeted_oos_candidates(current: dict[str, Any], args: argparse.Namespace) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []
    stage = "targeted"

    targeted_patches: list[tuple[str, dict[str, Any], str]] = [
        (
            "confirm_days_0",
            {"param_overrides.confirm_days_normal": 0},
            "Reduce confirmation delay to address low OOS trade frequency.",
        ),
        (
            "fast_confirm_score_45",
            {"param_overrides.fast_confirm_score": 45},
            "Loosen fast-confirm score threshold for more valid entries.",
        ),
        (
            "fast_confirm_score_50",
            {"param_overrides.fast_confirm_score": 50},
            "Slightly loosen fast-confirm score threshold.",
        ),
        (
            "fast_confirm_adx_18",
            {"param_overrides.fast_confirm_adx": 18},
            "Loosen fast-confirm ADX threshold.",
        ),
        (
            "confirm0_fast45",
            {"param_overrides.confirm_days_normal": 0, "param_overrides.fast_confirm_score": 45},
            "Combined confirmation-speed probe.",
        ),
        (
            "confirm0_fast50_adx18",
            {
                "param_overrides.confirm_days_normal": 0,
                "param_overrides.fast_confirm_score": 50,
                "param_overrides.fast_confirm_adx": 18,
            },
            "Balanced confirmation-speed combo.",
        ),
        (
            "pullback_lb_6",
            {"param_overrides.pullback_lookback": 6},
            "Shorten pullback lookback to catch faster OOS swings.",
        ),
        (
            "pullback_lb_12",
            {"param_overrides.pullback_lookback": 12},
            "Lengthen pullback lookback to admit slower OOS swings.",
        ),
        (
            "touch_tol_045",
            {"param_overrides.pullback_touch_tolerance_atr": 0.45},
            "Tighten pullback touch tolerance.",
        ),
        (
            "touch_tol_070",
            {"param_overrides.pullback_touch_tolerance_atr": 0.70},
            "Loosen pullback touch tolerance for OOS frequency.",
        ),
        (
            "touch_tol_085",
            {"param_overrides.pullback_touch_tolerance_atr": 0.85},
            "Aggressively loosen pullback touch tolerance.",
        ),
        (
            "touch_pct_015",
            {"param_overrides.pullback_touch_tolerance_pct": 0.0015},
            "Tighten price-percent pullback tolerance.",
        ),
        (
            "touch_pct_050",
            {"param_overrides.pullback_touch_tolerance_pct": 0.0050},
            "Loosen price-percent pullback tolerance.",
        ),
        (
            "recovery_trend_065",
            {"param_overrides.recovery_tolerance_atr_trend": 0.65},
            "Loosen trend recovery tolerance for more OOS entries.",
        ),
        (
            "recovery_strong_100",
            {"param_overrides.recovery_tolerance_atr_strong": 1.00},
            "Loosen strong-trend recovery tolerance.",
        ),
        (
            "touch70_recovery65",
            {
                "param_overrides.pullback_touch_tolerance_atr": 0.70,
                "param_overrides.recovery_tolerance_atr_trend": 0.65,
            },
            "Joint pullback looseness probe for frequency.",
        ),
        (
            "confirm0_touch70",
            {
                "param_overrides.confirm_days_normal": 0,
                "param_overrides.pullback_touch_tolerance_atr": 0.70,
            },
            "Speed plus pullback-looseness probe.",
        ),
        (
            "expiry_12h",
            {"param_overrides.order_expiry_hours": 12},
            "Shorten stale order lifetime.",
        ),
        (
            "expiry_24h",
            {"param_overrides.order_expiry_hours": 24},
            "Lengthen order lifetime to improve fill opportunity.",
        ),
        (
            "expiry_36h",
            {"param_overrides.order_expiry_hours": 36},
            "Aggressively lengthen order lifetime.",
        ),
        (
            "rank_score_per_risk",
            {"param_overrides.rank_mode": "score_per_risk"},
            "Re-rank simultaneous candidates by score per unit risk.",
        ),
        (
            "rank_stop_first",
            {"param_overrides.rank_mode": "stop_first"},
            "Re-rank simultaneous candidates by stop distance.",
        ),
        (
            "breakout_direct",
            {"param_overrides.breakout_direct_entry": True},
            "Target breakout conversion if pullback opportunities are sparse.",
        ),
        (
            "breakout_retrace_10_75",
            {
                "param_overrides.breakout_retrace_entry_frac": 0.10,
                "param_overrides.breakout_retrace_limit_frac": 0.75,
                "param_overrides.breakout_require_directional_candle": False,
            },
            "Looser breakout retrace conversion probe.",
        ),
        (
            "max_hold72_confirm0",
            {"param_overrides.max_hold_hours": 72, "param_overrides.confirm_days_normal": 0},
            "Free capital sooner while lowering entry delay.",
        ),
        (
            "max_hold80_touch70",
            {
                "param_overrides.max_hold_hours": 80,
                "param_overrides.pullback_touch_tolerance_atr": 0.70,
            },
            "Capital-release plus pullback-frequency combo.",
        ),
        (
            "early_stall16_max96",
            {
                "flags.early_stall_exit": True,
                "param_overrides.early_stall_check_hours": 16,
                "param_overrides.early_stall_mfe_threshold": 0.30,
                "param_overrides.max_hold_hours": 96,
            },
            "Softer early stall paired with slightly longer max hold.",
        ),
        (
            "enable_gld_shorts",
            {"param_overrides.shorts_enabled_GLD": 1},
            "Controlled GLD short-alpha probe using current QQQ/GLD data only.",
        ),
        (
            "enable_qqq_shorts",
            {"param_overrides.shorts_enabled_QQQ": 1},
            "Controlled QQQ short-alpha probe using current QQQ/GLD data only.",
        ),
        (
            "enable_etf_shorts",
            {"param_overrides.shorts_enabled_QQQ": 1, "param_overrides.shorts_enabled_GLD": 1},
            "Controlled ETF short-alpha probe using current QQQ/GLD data only.",
        ),
    ]
    for name, patch, intent in targeted_patches:
        _add_patch_candidate(candidates, current, stage, name, patch, intent)

    if args.allow_symbol_expansion:
        _add_patch_candidate(
            candidates,
            current,
            stage,
            "add_uso_sleeve",
            {"symbols": ["QQQ", "GLD", "USO"]},
            "Optional USO sleeve; disabled by default unless data coverage is refreshed.",
        )
    return candidates


def dedupe_repair_candidates(
    candidates: list[RepairCandidate],
    current: dict[str, Any],
) -> list[RepairCandidate]:
    current_sig = mutation_signature(current)
    seen: set[str] = set()
    deduped: list[RepairCandidate] = []
    for candidate in candidates:
        sig = mutation_signature(candidate.mutations)
        if sig == current_sig or sig in seen:
            continue
        seen.add(sig)
        deduped.append(candidate)
    return deduped


def _add_patch_candidate(
    candidates: list[RepairCandidate],
    current: dict[str, Any],
    stage: str,
    name: str,
    patch: dict[str, Any],
    intent: str,
    source: str = "",
) -> None:
    mutations = dict(current)
    changed = False
    for key, value in patch.items():
        if mutations.get(key, _MISSING_PREVIOUS_VALUE) != value:
            mutations[key] = value
            changed = True
    if not changed:
        return
    candidates.append(
        RepairCandidate(
            name=name,
            stage=stage,
            mutations=mutations,
            intent=intent,
            source=source,
        )
    )


def repair_development_gate(
    baseline: dict[str, float],
    candidate: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not candidate:
        return {"passed": False, "reasons": ["missing_candidate_metrics"]}
    if _metric_floor_failed(
        candidate.get("total_r", 0.0),
        baseline.get("total_r", 0.0),
        args.repair_dev_total_r_floor,
    ):
        reasons.append("development_total_r_floor")
    if _metric_floor_failed(
        candidate.get("net_return_pct", 0.0),
        baseline.get("net_return_pct", 0.0),
        args.repair_dev_net_return_floor,
    ):
        reasons.append("development_net_return_floor")
    if int(candidate.get("total_trades", 0)) < int(
        baseline.get("total_trades", 0) * args.repair_dev_trade_floor
    ):
        reasons.append("development_trade_floor")
    return {"passed": not reasons, "reasons": reasons}


def repair_preliminary_score(
    baseline: dict[str, float],
    candidate: dict[str, float],
    score_delta: float,
) -> float:
    return (
        score_delta
        + 0.30 * _metric_delta(candidate.get("net_return_pct", 0.0), baseline.get("net_return_pct", 0.0))
        + 0.25 * _metric_delta(candidate.get("total_r", 0.0), baseline.get("total_r", 0.0))
        + 0.20 * _metric_delta(candidate.get("total_trades", 0.0), baseline.get("total_trades", 0.0))
        + 0.10 * _metric_delta(candidate.get("trades_per_month", 0.0), baseline.get("trades_per_month", 0.0))
    )


def build_repair_evaluation(
    *,
    candidate: RepairCandidate,
    current_dev: ScoredRun,
    candidate_dev: ScoredRun,
    current_oos: ScoredRun,
    candidate_oos: ScoredRun,
    validation: ValidationSummary,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dev_delta = _delta_ratio(candidate_dev.score, current_dev.score)
    fold_delta = validation.robust_score_delta_pct / 100.0
    oos_score_delta = _delta_ratio(candidate_oos.score, current_oos.score)
    oos_return_delta = max(
        _metric_delta(
            candidate_oos.metrics.get("net_return_pct", 0.0),
            current_oos.metrics.get("net_return_pct", 0.0),
        ),
        _metric_delta(
            candidate_oos.metrics.get("total_r", 0.0),
            current_oos.metrics.get("total_r", 0.0),
        ),
    )
    oos_trade_delta = _metric_delta(
        candidate_oos.metrics.get("total_trades", 0.0),
        current_oos.metrics.get("total_trades", 0.0),
    )
    objective = (
        args.repair_dev_weight * dev_delta
        + args.repair_fold_weight * fold_delta
        + args.repair_oos_score_weight * oos_score_delta
        + args.repair_oos_return_weight * oos_return_delta
        + args.repair_oos_trade_weight * oos_trade_delta
    )
    reasons: list[str] = []
    if dev_delta < args.repair_dev_score_floor:
        reasons.append("development_score_floor")
    if not validation.passed:
        reasons.append("fold_validation_failed")
    if candidate_oos.rejected:
        reasons.append(f"selection_oos_rejected: {candidate_oos.reject_reason}")
    if _metric_floor_failed(
        candidate_oos.metrics.get("total_r", 0.0),
        current_oos.metrics.get("total_r", 0.0),
        args.repair_oos_total_r_floor,
    ):
        reasons.append("selection_oos_total_r_floor")
    if _metric_floor_failed(
        candidate_oos.metrics.get("net_return_pct", 0.0),
        current_oos.metrics.get("net_return_pct", 0.0),
        args.repair_oos_net_return_floor,
    ):
        reasons.append("selection_oos_net_return_floor")
    if int(candidate_oos.metrics.get("total_trades", 0)) < int(
        current_oos.metrics.get("total_trades", 0) * args.repair_oos_trade_floor
    ):
        reasons.append("selection_oos_trade_floor")
    if objective < args.repair_min_objective_delta:
        reasons.append("objective_too_small")

    return {
        "candidate": asdict(candidate),
        "passed": not reasons,
        "reasons": reasons,
        "objective_delta": objective,
        "objective_delta_pct": objective * 100.0,
        "development_score_delta_pct": dev_delta * 100.0,
        "fold_robust_delta_pct": validation.robust_score_delta_pct,
        "selection_oos_score_delta_pct": oos_score_delta * 100.0,
        "selection_oos_return_delta_pct": oos_return_delta * 100.0,
        "selection_oos_trade_delta_pct": oos_trade_delta * 100.0,
        "development": asdict(candidate_dev),
        "development_baseline": asdict(current_dev),
        "selection_oos": {
            "baseline": asdict(current_oos),
            "candidate": asdict(candidate_oos),
        },
        "fold_validation": asdict(validation),
    }


def _metric_floor_failed(candidate: float, baseline: float, floor: float) -> bool:
    if baseline > 0:
        return candidate < baseline * floor
    return candidate < baseline


def _metric_delta(candidate: float, baseline: float) -> float:
    if abs(baseline) > 1e-9:
        return (candidate - baseline) / abs(baseline)
    return candidate - baseline


def _short_key(key: str) -> str:
    return (
        key.replace("param_overrides.", "")
        .replace("flags.", "flag_")
        .replace("slippage.", "slip_")
        .replace(".", "_")
    )


def validate_candidate(
    *,
    candidate_name: str,
    phase: int,
    scoring_profile: str,
    hard_rejects: dict[str, float],
    current_mutations: dict[str, Any],
    candidate_mutations: dict[str, Any],
    development_delta: float,
    scorer: ATRSSWindowScorer,
    folds: list[Fold],
    args: argparse.Namespace,
) -> ValidationSummary:
    fold_results: list[dict[str, Any]] = []
    deltas: list[float] = []
    candidate_scores: list[float] = []
    baseline_scores: list[float] = []
    for fold in folds:
        fold_rejects = validation_hard_rejects(
            hard_rejects,
            args.train_start,
            args.dev_end,
            fold.validation_start,
            fold.validation_end,
            max_min_trades=args.validation_max_fold_min_trades,
        )
        baseline = scorer.score(
            name=f"{fold.name}_baseline",
            mutations=current_mutations,
            phase=phase,
            scoring_profile=scoring_profile,
            hard_rejects=fold_rejects,
            data_start=fold_data_start(fold, args),
            data_end=fold.validation_end,
            score_start=fold.validation_start,
            score_end=fold.validation_end,
        )
        candidate = scorer.score(
            name=f"{fold.name}_{candidate_name}",
            mutations=candidate_mutations,
            phase=phase,
            scoring_profile=scoring_profile,
            hard_rejects=fold_rejects,
            data_start=fold_data_start(fold, args),
            data_end=fold.validation_end,
            score_start=fold.validation_start,
            score_end=fold.validation_end,
        )
        delta = _delta_ratio(candidate.score, baseline.score)
        catastrophic = is_catastrophic_fold(
            baseline.metrics,
            candidate.metrics,
            args=args,
        )
        passed = (not candidate.rejected) and delta >= args.validation_min_delta
        deltas.append(delta)
        candidate_scores.append(candidate.score)
        baseline_scores.append(baseline.score)
        fold_results.append(
            {
                "fold": asdict(fold),
                "data_start": fold_data_start(fold, args),
                "passed": passed,
                "catastrophic": catastrophic,
                "score_delta_pct": delta * 100.0,
                "baseline": asdict(baseline),
                "candidate": asdict(candidate),
                "validation_hard_rejects": fold_rejects,
            }
        )

    pass_rate = sum(1 for item in fold_results if item["passed"]) / max(len(fold_results), 1)
    median_delta = float(np.median(deltas) * 100.0) if deltas else 0.0
    mean_delta = float(np.mean(deltas) * 100.0) if deltas else 0.0
    std_delta = float(np.std(deltas) * 100.0) if deltas else 0.0
    shrunk_fold_delta = (mean_delta / 100.0) - args.fold_std_penalty * (std_delta / 100.0)
    robust_delta = args.development_weight * development_delta + args.fold_weight * shrunk_fold_delta
    candidate_total_r = float(sum(item["candidate"]["metrics"].get("total_r", 0.0) for item in fold_results))
    baseline_total_r = float(sum(item["baseline"]["metrics"].get("total_r", 0.0) for item in fold_results))
    candidate_trades = int(sum(item["candidate"]["metrics"].get("total_trades", 0) for item in fold_results))
    baseline_trades = int(sum(item["baseline"]["metrics"].get("total_trades", 0) for item in fold_results))
    catastrophic_folds = sum(1 for item in fold_results if item["catastrophic"])
    passed = (
        robust_delta >= args.robust_min_delta
        and catastrophic_folds <= args.max_catastrophic_folds
        and candidate_trades >= max(1, int(baseline_trades * args.validation_trade_floor))
        and candidate_total_r >= baseline_total_r * args.validation_total_r_floor
    )
    return ValidationSummary(
        passed=passed,
        pass_rate=pass_rate,
        robust_score_delta_pct=robust_delta * 100.0,
        fold_score_delta_std_pct=std_delta,
        median_score_delta_pct=median_delta,
        mean_score_delta_pct=mean_delta,
        candidate_score_mean=float(np.mean(candidate_scores)) if candidate_scores else 0.0,
        baseline_score_mean=float(np.mean(baseline_scores)) if baseline_scores else 0.0,
        candidate_total_r=candidate_total_r,
        baseline_total_r=baseline_total_r,
        candidate_trades=candidate_trades,
        baseline_trades=baseline_trades,
        catastrophic_folds=catastrophic_folds,
        fold_results=fold_results,
    )


def is_catastrophic_fold(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    *,
    args: argparse.Namespace,
) -> bool:
    baseline_trades = max(int(baseline_metrics.get("total_trades", 0)), 1)
    candidate_trades = int(candidate_metrics.get("total_trades", 0))
    baseline_dd = float(baseline_metrics.get("max_dd_pct", 0.0))
    candidate_dd = float(candidate_metrics.get("max_dd_pct", 0.0))
    candidate_pf = float(candidate_metrics.get("profit_factor", 0.0))
    if candidate_trades < baseline_trades * args.catastrophe_trade_floor:
        return True
    if candidate_pf < args.catastrophe_min_pf:
        return True
    if candidate_dd > args.catastrophe_max_abs_dd:
        return True
    if (
        baseline_dd >= args.catastrophe_relative_dd_floor
        and candidate_dd > baseline_dd * args.catastrophe_max_dd_mult
    ):
        return True
    return False


def validation_hard_rejects(
    hard_rejects: dict[str, float],
    dev_start: str,
    dev_end: str,
    fold_start: str,
    fold_end: str,
    *,
    max_min_trades: int | None = None,
) -> dict[str, float]:
    dev_days = max((_exclusive_end_ts(dev_end) - _start_ts(dev_start)).total_seconds() / 86400, 1.0)
    fold_days = max((_exclusive_end_ts(fold_end) - _start_ts(fold_start)).total_seconds() / 86400, 1.0)
    scaled_min_trades = int(round(float(hard_rejects.get("min_trades", 1)) * fold_days / dev_days))
    min_trades = max(1, scaled_min_trades)
    if max_min_trades is not None:
        min_trades = min(max_min_trades, min_trades)
    return {
        "min_trades": min_trades,
        "max_dd_pct": min(0.20, float(hard_rejects.get("max_dd_pct", 0.10)) * 1.5),
        "min_pf": max(0.75, float(hard_rejects.get("min_pf", 1.0)) * 0.5),
        "min_wr": max(0.35, float(hard_rejects.get("min_wr", 0.50)) - 0.25),
    }


def fold_data_start(fold: Fold, args: argparse.Namespace) -> str:
    train_start = pd.Timestamp(args.train_start)
    validation_start = pd.Timestamp(fold.validation_start)
    warmup_start = validation_start - pd.Timedelta(days=args.validation_warmup_days)
    return max(train_start, warmup_start).strftime("%Y-%m-%d")


def build_quarterly_folds(
    *,
    train_start: str,
    first_validation_start: str,
    dev_end: str,
    fold_months: int,
    step_months: int,
) -> list[Fold]:
    folds: list[Fold] = []
    val_start = pd.Timestamp(first_validation_start)
    dev_end_ts = pd.Timestamp(dev_end)
    while val_start <= dev_end_ts:
        val_end = min(val_start + pd.DateOffset(months=fold_months) - pd.Timedelta(days=1), dev_end_ts)
        train_end = val_start - pd.Timedelta(days=1)
        folds.append(
            Fold(
                name=f"F{len(folds) + 1:02d}_{val_start.strftime('%Y%m%d')}_{val_end.strftime('%Y%m%d')}",
                train_start=train_start,
                train_end=train_end.strftime("%Y-%m-%d"),
                validation_start=val_start.strftime("%Y-%m-%d"),
                validation_end=val_end.strftime("%Y-%m-%d"),
            )
        )
        val_start = val_start + pd.DateOffset(months=step_months)
    return folds


def filter_candidates(
    candidates: list[Experiment],
    *,
    base_symbols: tuple[str, ...],
    data_dir: Path,
    required_end: str,
    allow_symbol_expansion: bool,
) -> list[Experiment]:
    filtered: list[Experiment] = []
    required = _start_ts(required_end)
    for candidate in candidates:
        symbols = tuple(candidate.mutations.get("symbols", base_symbols))
        if not allow_symbol_expansion and not set(symbols).issubset(set(base_symbols)):
            continue
        if any(data_coverage_end(data_dir, symbol) < required for symbol in symbols):
            continue
        filtered.append(candidate)
    return filtered


def add_prior_seed_candidates(
    candidates: list[Experiment],
    *,
    profile: str,
    prior_rounds_dir: Path,
) -> list[Experiment]:
    seeds: list[Experiment] = []
    if profile == "alpha":
        for round_num in (1, 2):
            mutations = load_prior_round_config(prior_rounds_dir, round_num)
            if mutations:
                seeds.append(Experiment(f"seed_round_{round_num}", mutations))
    elif profile == "risk":
        round_2 = load_prior_round_config(prior_rounds_dir, 2)
        round_3 = load_prior_round_config(prior_rounds_dir, 3)
        if round_3:
            seeds.append(Experiment("seed_round_3", round_3))
        if round_2 and round_3:
            overlay = {
                key: value
                for key, value in round_3.items()
                if round_2.get(key) != value
            }
            if overlay:
                seeds.append(Experiment("seed_round_3_risk_overlay", overlay))

    merged = [*seeds, *candidates]
    seen: set[tuple[str, str]] = set()
    deduped: list[Experiment] = []
    for candidate in merged:
        sig = (candidate.name, json.dumps(candidate.mutations, sort_keys=True, default=str))
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(candidate)
    return deduped


def load_prior_round_config(prior_rounds_dir: Path, round_num: int) -> dict[str, Any]:
    path = prior_rounds_dir / f"round_{round_num}" / "optimized_config.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


_COVERAGE_CACHE: dict[tuple[str, str], datetime] = {}


def data_coverage_end(data_dir: Path, symbol: str) -> datetime:
    key = (str(data_dir.resolve()), symbol)
    cached = _COVERAGE_CACHE.get(key)
    if cached is not None:
        return cached
    ends = []
    for timeframe in ("1h", "1d"):
        df = load_bars(data_dir / f"{symbol}_{timeframe}.parquet")
        if df.index.tz is None:
            idx = df.index.tz_localize("UTC")
        else:
            idx = df.index.tz_convert("UTC")
        ends.append(idx.max().to_pydatetime())
    end = min(ends)
    _COVERAGE_CACHE[key] = end
    return end


def write_text_summary(path: Path, summary: dict[str, Any]) -> None:
    holdout = summary["holdout"]
    dev = summary["development"]
    lines = [
        "ATRSS Anchored Walk-Forward Summary",
        "",
        f"Output: {summary['output_dir']}",
        f"Mutations: {len(summary['optimized_config'])}",
        "",
        "Development:",
        _format_score_line(dev),
        "",
        "Holdout:",
        _format_score_line(holdout),
        "",
        "Optimized config:",
    ]
    for key, value in sorted(summary["optimized_config"].items()):
        lines.append(f"  {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_repair_text_summary(path: Path, summary: dict[str, Any]) -> None:
    baseline_dev = summary["baseline"]["development"]
    baseline_oos = summary["baseline"]["selection_oos"]
    final_dev = summary["final"]["development"]
    final_oos = summary["final"]["selection_oos"]
    accepted = [
        item
        for stage in summary["stage_reports"]
        for item in stage.get("accepted", [])
    ]
    lines = [
        "ATRSS Incumbent Repair Summary",
        "",
        f"Output: {summary['output_dir']}",
        summary["selection_oos_note"],
        "",
        "Baseline Development:",
        _format_score_line(baseline_dev),
        "Final Development:",
        _format_score_line(final_dev),
        "",
        "Baseline Selection OOS:",
        _format_score_line(baseline_oos),
        "Final Selection OOS:",
        _format_score_line(final_oos),
        "",
        "Accepted repair candidates:",
    ]
    if accepted:
        for item in accepted:
            candidate = item["candidate"]
            lines.append(
                f"  {candidate['stage']} / {candidate['name']}: "
                f"objective +{item['objective_delta_pct']:.2f}%, "
                f"dev {item['development_score_delta_pct']:.2f}%, "
                f"fold {item['fold_robust_delta_pct']:.2f}%, "
                f"OOS score {item['selection_oos_score_delta_pct']:.2f}%"
            )
    else:
        lines.append("  None")
    lines.extend(["", "Optimized config:"])
    for key, value in sorted(summary["optimized_config"].items()):
        lines.append(f"  {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_score_line(item: dict[str, Any]) -> str:
    metrics = item["metrics"]
    return (
        f"score={item['score']:.4f}, trades={metrics.get('total_trades', 0)}, "
        f"PF={metrics.get('profit_factor', 0.0):.2f}, avgR={metrics.get('avg_r', 0.0):.2f}, "
        f"totalR={metrics.get('total_r', 0.0):.1f}, maxDD={metrics.get('max_dd_pct', 0.0) * 100:.2f}%"
    )


def asdict_state(state: PhaseState) -> dict[str, Any]:
    return {
        "current_phase": state.current_phase,
        "completed_phases": state.completed_phases,
        "cumulative_mutations": state.cumulative_mutations,
        "phase_results": state.phase_results,
        "phase_gate_results": state.phase_gate_results,
        "retry_count": state.retry_count,
        "scoring_retries": state.scoring_retries,
        "diagnostic_retries": state.diagnostic_retries,
        "phase_timestamps": state.phase_timestamps,
        "round_name": state.round_name,
    }


def _equity_window(
    result: PortfolioResult,
    start: datetime,
    end: datetime,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    if len(result.combined_equity) == 0 or len(result.combined_timestamps) == 0:
        return np.array([], dtype=np.float64), pd.DatetimeIndex([])
    idx = pd.DatetimeIndex(result.combined_timestamps)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    mask = (idx >= pd.Timestamp(start)) & (idx < pd.Timestamp(end))
    return result.combined_equity[mask], idx[mask]


def _trade_time(value: Any) -> datetime | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _start_ts(value: str) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _exclusive_end_ts(value: str) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if ts == ts.normalize():
        ts = ts + pd.Timedelta(days=1)
    return ts.to_pydatetime()


def _delta_ratio(score: float, baseline: float) -> float:
    if baseline > 0:
        return (score - baseline) / baseline
    return score - baseline


def _parse_symbols(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _parse_profiles(value: str) -> list[str]:
    profiles = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in profiles if item not in {"alpha", "risk"}]
    if invalid:
        raise ValueError(f"Unsupported profile(s): {', '.join(invalid)}")
    return profiles


def _parse_stage_sequence(value: str) -> list[str]:
    stages = [item.strip().lower() for item in value.split(",") if item.strip()]
    aliases = {
        "incumbent_ablation": "ablation",
        "oos_targeted": "targeted",
        "targeted_oos": "targeted",
    }
    normalized = [aliases.get(stage, stage) for stage in stages]
    invalid = [stage for stage in normalized if stage not in {"ablation", "perturbation", "targeted"}]
    if invalid:
        raise ValueError(f"Unsupported repair stage(s): {', '.join(invalid)}")
    return normalized


def _resolve_output_dir(output_dir: Path | None, *, prefix: str = "anchored_wfo") -> Path:
    if output_dir is not None:
        return output_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("backtests/output/swing/atrss") / f"{prefix}_{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ATRSS anchored walk-forward phased optimization")
    parser.add_argument("--mode", choices=["walk-forward", "incumbent-repair"], default="walk-forward")
    parser.add_argument("--data-dir", default="backtests/swing/data/raw")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--symbols", default="QQQ,GLD")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--train-start", default="2021-02-08")
    parser.add_argument("--dev-end", default="2025-12-31")
    parser.add_argument("--holdout-start", default="2026-01-01")
    parser.add_argument("--holdout-end", default="2026-05-01")
    parser.add_argument("--holdout-max-min-trades", type=int, default=3)
    parser.add_argument("--first-validation-start", default="2022-04-01")
    parser.add_argument("--fold-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--profile-sequence", default="alpha,risk")
    parser.add_argument("--prior-rounds-dir", default="backtests/output/swing/atrss")
    parser.add_argument("--incumbent-config", default="backtests/output/swing/atrss/round_3/optimized_config.json")
    parser.add_argument("--max-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--validation-top-k", type=int, default=5)
    parser.add_argument("--validation-warmup-days", type=int, default=180)
    parser.add_argument("--validation-max-fold-min-trades", type=int, default=3)
    parser.add_argument("--validation-pass-rate", type=float, default=0.60)
    parser.add_argument("--validation-min-delta", type=float, default=0.0)
    parser.add_argument("--validation-trade-floor", type=float, default=0.85)
    parser.add_argument("--validation-total-r-floor", type=float, default=0.90)
    parser.add_argument("--development-weight", type=float, default=0.60)
    parser.add_argument("--fold-weight", type=float, default=0.40)
    parser.add_argument("--fold-std-penalty", type=float, default=0.50)
    parser.add_argument("--robust-min-delta", type=float, default=0.003)
    parser.add_argument("--enable-fold-ablation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-ablation-rounds", type=int, default=3)
    parser.add_argument("--ablation-min-dev-delta", type=float, default=0.001)
    parser.add_argument("--ablation-robust-min-delta", type=float, default=0.0015)
    parser.add_argument("--max-catastrophic-folds", type=int, default=3)
    parser.add_argument("--catastrophe-trade-floor", type=float, default=0.50)
    parser.add_argument("--catastrophe-min-pf", type=float, default=0.75)
    parser.add_argument("--catastrophe-max-dd-mult", type=float, default=2.50)
    parser.add_argument("--catastrophe-relative-dd-floor", type=float, default=0.01)
    parser.add_argument("--catastrophe-max-abs-dd", type=float, default=0.08)
    parser.add_argument("--allow-symbol-expansion", action="store_true")
    parser.add_argument("--repair-stage-sequence", default="ablation,perturbation,targeted")
    parser.add_argument("--repair-max-stage-rounds", type=int, default=2)
    parser.add_argument("--repair-validation-top-k", type=int, default=6)
    parser.add_argument("--repair-min-objective-delta", type=float, default=0.002)
    parser.add_argument("--repair-dev-score-floor", type=float, default=-0.025)
    parser.add_argument("--repair-dev-total-r-floor", type=float, default=0.95)
    parser.add_argument("--repair-dev-net-return-floor", type=float, default=0.95)
    parser.add_argument("--repair-dev-trade-floor", type=float, default=0.90)
    parser.add_argument("--repair-oos-total-r-floor", type=float, default=0.95)
    parser.add_argument("--repair-oos-net-return-floor", type=float, default=0.95)
    parser.add_argument("--repair-oos-trade-floor", type=float, default=0.90)
    parser.add_argument("--repair-dev-weight", type=float, default=0.20)
    parser.add_argument("--repair-fold-weight", type=float, default=0.35)
    parser.add_argument("--repair-oos-score-weight", type=float, default=0.20)
    parser.add_argument("--repair-oos-return-weight", type=float, default=0.15)
    parser.add_argument("--repair-oos-trade-weight", type=float, default=0.10)
    return parser


def main() -> None:
    start = time.time()
    args = build_parser().parse_args()
    if args.mode == "incumbent-repair":
        summary = run_incumbent_repair(args)
        final_item = summary["final"]["selection_oos"]
    else:
        summary = run_walk_forward(args)
        final_item = summary["holdout"]
    elapsed = time.time() - start
    print(f"Complete in {elapsed / 60:.1f} min")
    print(f"Output: {summary['output_dir']}")
    print(_format_score_line(final_item))


if __name__ == "__main__":
    main()
