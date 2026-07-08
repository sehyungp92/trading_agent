"""Shared phased auto-optimization helpers for 15m ETF swing strategies."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from backtests.shared.auto.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    SharedPoolBatchEvaluator,
    create_process_pool,
    mutation_signature,
    resolve_worker_processes,
    shutdown_process_pool,
)
from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion, GreedyResult, ScoredCandidate


@dataclass(frozen=True)
class ETFMetrics:
    total_trades: int
    net_return_pct: float
    profit_factor: float
    avg_r: float
    total_r: float
    win_rate: float
    max_dd_pct: float
    sharpe: float
    trades_per_month: float
    return_per_trade_pct: float
    avg_mfe_r: float = 0.0
    median_mfe_r: float = 0.0
    avg_mae_r: float = 0.0
    mfe_capture: float = 0.0
    right_then_lost_rate: float = 0.0
    false_positive_rate: float = 0.0
    stop_rate: float = 1.0
    symbol_balance: float = 0.0
    active_years: float = 0.0
    top5_winner_share: float = 0.0
    avg_hold_bars: float = 0.0


@dataclass(frozen=True)
class ETFCompositeScore:
    total: float
    rejected: bool = False
    reject_reason: str = ""


def extract_etf_metrics(result: Any, initial_equity: float) -> ETFMetrics:
    trades = list(getattr(result, "trades", []))
    rs = np.asarray([float(getattr(t, "r_multiple", 0.0) or 0.0) for t in trades], dtype=float)
    wins = rs[rs > 0]
    losses = rs[rs < 0]
    gross_win = float(np.sum(wins)) if wins.size else 0.0
    gross_loss = abs(float(np.sum(losses))) if losses.size else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)
    mfe = np.asarray([float(getattr(t, "mfe_r", 0.0) or 0.0) for t in trades], dtype=float)
    mae = np.asarray([float(getattr(t, "mae_r", 0.0) or 0.0) for t in trades], dtype=float)
    holds = np.asarray([float(getattr(t, "bars_held", 0.0) or 0.0) for t in trades], dtype=float)
    mfe_positive = np.maximum(mfe, 0.0)
    captured_wins = np.maximum(rs, 0.0)
    mfe_capture = float(np.sum(captured_wins) / np.sum(mfe_positive)) if np.sum(mfe_positive) > 0 else 0.0
    right_then_lost = int(np.sum((mfe >= 1.0) & (rs <= 0.0))) if rs.size else 0
    false_positive = int(np.sum((mfe < 0.5) & (rs <= 0.0))) if rs.size else 0
    sorted_wins = np.sort(wins)[::-1] if wins.size else np.asarray([], dtype=float)
    top5_share = float(np.sum(sorted_wins[:5]) / gross_win) if gross_win > 0 else 0.0
    stop_flags = np.asarray(
        [str(getattr(t, "exit_reason", "") or "").upper() == "STOP" for t in trades],
        dtype=bool,
    )
    symbol_balance = _cohort_balance([str(getattr(t, "symbol", "") or "") for t in trades])
    active_years = len(
        {
            pd.Timestamp(getattr(t, "entry_time")).year
            for t in trades
            if getattr(t, "entry_time", None) is not None
        }
    )
    equity = np.asarray(getattr(result, "combined_equity", []), dtype=float)
    if equity.size:
        peak = np.maximum.accumulate(equity)
        dd = np.where(peak > 0, (peak - equity) / peak, 0.0)
        max_dd = float(np.max(dd) * 100.0)
        net_ret = float((equity[-1] - initial_equity) / initial_equity * 100.0)
        rets = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252 * 26)) if rets.size and np.std(rets) > 0 else 0.0
    else:
        max_dd = 0.0
        net_ret = 0.0
        sharpe = 0.0
    timestamps = list(getattr(result, "combined_timestamps", []))
    if len(timestamps) >= 2:
        idx = pd.to_datetime(timestamps)
        span_seconds = max((idx[-1] - idx[0]).total_seconds(), 0.0)
        months = max(span_seconds / (30.4375 * 24 * 3600), 1.0)
    else:
        months = 1.0
    total_trades = len(trades)
    return ETFMetrics(
        total_trades=total_trades,
        net_return_pct=net_ret,
        profit_factor=pf,
        avg_r=float(np.mean(rs)) if rs.size else 0.0,
        total_r=float(np.sum(rs)) if rs.size else 0.0,
        win_rate=float(np.mean(rs > 0)) if rs.size else 0.0,
        max_dd_pct=max_dd,
        sharpe=sharpe,
        trades_per_month=total_trades / months,
        return_per_trade_pct=net_ret / total_trades if total_trades else 0.0,
        avg_mfe_r=float(np.mean(mfe)) if mfe.size else 0.0,
        median_mfe_r=float(np.median(mfe)) if mfe.size else 0.0,
        avg_mae_r=float(np.mean(mae)) if mae.size else 0.0,
        mfe_capture=mfe_capture,
        right_then_lost_rate=right_then_lost / total_trades if total_trades else 0.0,
        false_positive_rate=false_positive / total_trades if total_trades else 0.0,
        stop_rate=float(np.mean(stop_flags)) if stop_flags.size else 1.0,
        symbol_balance=symbol_balance,
        active_years=float(active_years),
        top5_winner_share=top5_share,
        avg_hold_bars=float(np.mean(holds)) if holds.size else 0.0,
    )


DEFAULT_SCORING_WEIGHTS = {
    "expectancy": 0.25,
    "frequency": 0.22,
    "alpha_capture": 0.18,
    "false_positive_control": 0.13,
    "robustness": 0.10,
    "drawdown": 0.07,
    "execution_quality": 0.05,
}


def composite_score(
    metrics: ETFMetrics,
    hard_rejects: dict[str, float] | None = None,
    scoring_weights: dict[str, float] | None = None,
) -> ETFCompositeScore:
    rejects = hard_rejects or {}
    if metrics.total_trades < rejects.get("min_valid_trades", 1):
        return ETFCompositeScore(0.0, True, "no_trades")
    if metrics.max_dd_pct > rejects.get("max_dd_pct", 30.0):
        return ETFCompositeScore(0.0, True, "max_dd_pct")
    if metrics.net_return_pct < rejects.get("min_return_pct", -40.0):
        return ETFCompositeScore(0.0, True, "min_return_pct")
    if metrics.top5_winner_share > rejects.get("max_top5_winner_share", 1.01):
        return ETFCompositeScore(0.0, True, "top5_winner_share")
    quality_sample = rejects.get("quality_sample_trades", 12)
    if metrics.total_trades >= quality_sample and metrics.avg_mfe_r < rejects.get("min_avg_mfe_r", -1.0):
        return ETFCompositeScore(0.0, True, "avg_mfe_r")
    if metrics.total_trades >= quality_sample and metrics.false_positive_rate > rejects.get("max_false_positive_rate", 1.01):
        return ETFCompositeScore(0.0, True, "false_positive_rate")

    weights = _normalise_weights(scoring_weights or DEFAULT_SCORING_WEIGHTS)
    expectancy = (
        0.30 * _scale(metrics.net_return_pct, -5.0, 25.0)
        + 0.30 * _scale(metrics.avg_r, -0.50, 0.45)
        + 0.25 * _scale(metrics.profit_factor, 0.0, 1.80)
        + 0.15 * _scale(metrics.total_r, -10.0, 60.0)
    )
    frequency = (
        0.60 * _scale(metrics.trades_per_month, 0.10, 1.50)
        + 0.40 * _scale(float(metrics.total_trades), 6.0, 90.0)
    )
    alpha_capture = (
        0.40 * _scale(metrics.mfe_capture, 0.10, 0.55)
        + 0.20 * _scale(metrics.avg_mfe_r, 0.50, 2.00)
        + 0.15 * _scale(metrics.median_mfe_r, 0.25, 1.25)
        + 0.15 * (1.0 - _scale(metrics.false_positive_rate, 0.25, 0.75))
        + 0.10 * (1.0 - _scale(metrics.right_then_lost_rate, 0.00, 0.30))
    )
    false_positive_control = (
        0.55 * (1.0 - _scale(metrics.false_positive_rate, 0.0, 0.80))
        + 0.30 * (1.0 - _scale(metrics.stop_rate, 0.20, 1.00))
        + 0.15 * (1.0 - _scale(metrics.right_then_lost_rate, 0.0, 0.30))
    )
    robustness = (
        0.45 * _scale(metrics.symbol_balance, 0.0, 1.0)
        + 0.35 * _scale(metrics.active_years, 1.0, 4.0)
        + 0.20 * (1.0 - _scale(metrics.top5_winner_share, 0.60, 1.0))
    )
    drawdown = 1.0 - _scale(metrics.max_dd_pct, 0.0, 15.0)
    execution_quality = (
        0.40 * _scale(metrics.sharpe, -1.0, 2.0)
        + 0.35 * _scale(metrics.return_per_trade_pct, -0.15, 0.25)
        + 0.25 * _scale(metrics.win_rate, 0.15, 0.55)
    )
    components = {
        "alpha_capture": alpha_capture,
        "expectancy": expectancy,
        "frequency": frequency,
        "false_positive_control": false_positive_control,
        "robustness": robustness,
        "drawdown": drawdown,
        "execution_quality": execution_quality,
    }
    score = sum(weights.get(name, 0.0) * components.get(name, 0.0) for name in weights)
    return ETFCompositeScore(max(0.0, score * 100.0))


class ETFPhasePlugin:
    num_phases = 4
    initial_mutations: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        name: str,
        data_dir: Path,
        config_factory: Callable[..., Any],
        bundle_loader: Callable[..., Any],
        runner: Callable[[dict, Any], Any],
        candidates_fn: Callable[[int], list[tuple[str, dict[str, Any]]]],
        initial_equity: float = 100_000.0,
        max_workers: int | None = None,
        num_phases: int = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        self.name = name
        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.start_date = start_date
        self.end_date = end_date
        self.config_factory = config_factory
        self.bundle_loader = bundle_loader
        self.runner = runner
        self.candidates_fn = candidates_fn
        self.num_phases = num_phases
        self._cached_bundle = None
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._score_cache: dict[str, ScoredCandidate] = {}

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 35.0,
            "avg_r": 0.25,
            "trades_per_month": 1.5,
            "total_trades": 90.0,
            "profit_factor": 1.5,
            "max_dd_pct": 15.0,
        }

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        del state
        candidates = [Experiment(name=n, mutations=m) for n, m in self.candidates_fn(phase)]
        return PhaseSpec(
            focus={
                1: "SIGNAL_GATES",
                2: "ENTRY_CONFIRMATION",
                3: "EXITS_AND_RISK",
                4: "SIZING_FINETUNE",
            }.get(phase, "FINETUNE"),
            candidates=candidates,
            gate_criteria_fn=self._gate_criteria,
            scoring_weights=DEFAULT_SCORING_WEIGHTS,
            hard_rejects={
                "min_valid_trades": 1,
                "max_dd_pct": 30.0,
                "min_return_pct": -40.0,
                "quality_sample_trades": 12,
                "min_avg_mfe_r": 0.30,
                "max_false_positive_rate": 0.85,
            },
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=["net_return_pct", "avg_r", "trades_per_month", "total_trades", "avg_mfe_r", "max_dd_pct"],
                diagnostic_gap_fn=_etf_diagnostic_gaps,
                redesign_scoring_weights_fn=_redesign_etf_weights,
            ),
            max_rounds=12,
            prune_threshold=0.02,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        _indicator_cache: dict = {}

        def _evaluate(candidates: list[Experiment], current_mutations: dict[str, Any]) -> list[ScoredCandidate]:
            results: list[ScoredCandidate] = []
            for candidate in candidates:
                muts = dict(cumulative_mutations)
                muts.update(current_mutations)
                muts.update(candidate.mutations)
                metrics = ETFMetrics(**self.compute_final_metrics(muts, indicator_cache=_indicator_cache))
                score = composite_score(metrics, hard_rejects, scoring_weights=scoring_weights)
                results.append(
                    ScoredCandidate(
                        name=candidate.name,
                        score=0.0 if score.rejected else score.total,
                        rejected=score.rejected,
                        reject_reason=score.reject_reason,
                        metrics=metrics.__dict__,
                    )
                )
            return results

        local = CachedBatchEvaluator(
            _evaluate,
            cache=self._score_cache,
            metrics_cache=self._metrics_cache,
            signature_prefix=f"{self.name}:local",
        )
        if resolve_worker_processes(self.max_workers) <= 1:
            return local

        def _parallel_factory():
            worker_module = importlib.import_module(f"backtests.swing.auto.{self.name}.worker")
            pool = create_process_pool(
                self.max_workers,
                initializer=worker_module.init_worker,
                initargs=(str(self.data_dir), self.initial_equity, self.start_date, self.end_date),
                logger=logging.getLogger(__name__),
                description=f"{self.name} ETF auto",
            )
            return SharedPoolBatchEvaluator(
                pool,
                worker_fn=worker_module.score_candidate,
                build_args=lambda candidates, current_mutations: _worker_args(
                    candidates,
                    cumulative_mutations,
                    current_mutations,
                    phase,
                    scoring_weights,
                    hard_rejects,
                ),
                on_terminate=lambda: shutdown_process_pool(pool, force=True),
                on_close=lambda: shutdown_process_pool(pool, force=False),
                description=f"{self.name} phase {phase} candidate batch",
                logger=logging.getLogger(__name__),
            )

        parallel = ResilientBatchEvaluator(
            preferred_factory=_parallel_factory,
            fallback_factory=lambda: local,
            description=f"{self.name} parallel ETF evaluator",
            logger=logging.getLogger(__name__),
        )
        return CachedBatchEvaluator(
            parallel,
            cache=self._score_cache,
            metrics_cache=self._metrics_cache,
            signature_prefix=f"{self.name}:parallel",
        )

    def compute_final_metrics(self, mutations: dict[str, Any], *, indicator_cache: dict | None = None) -> dict[str, float]:
        cache_key = mutation_signature(mutations)
        cached = self._metrics_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        cfg = self.config_factory(initial_equity=self.initial_equity, data_dir=self.data_dir)
        if hasattr(cfg, "with_overrides"):
            cfg = cfg.with_overrides(mutations)
        result = self.runner(self._replay_bundle().data, cfg, indicator_cache=indicator_cache)
        metrics = extract_etf_metrics(result, self.initial_equity).__dict__
        self._metrics_cache[cache_key] = dict(metrics)
        return metrics

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result: GreedyResult) -> str:
        del state, greedy_result
        return (
            f"{self.name} phase {phase}: "
            f"net={metrics.get('net_return_pct', 0):+.2f}%, "
            f"avgR={metrics.get('avg_r', 0):+.2f}, "
            f"trades={metrics.get('total_trades', 0):.0f}, "
            f"trades/month={metrics.get('trades_per_month', 0):.2f}, "
            f"PF={metrics.get('profit_factor', 0):.2f}, "
            f"DD={metrics.get('max_dd_pct', 0):.2f}%"
        )

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result: GreedyResult) -> str:
        return self.run_phase_diagnostics(phase, state, metrics, greedy_result)

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        del state
        return EndOfRoundArtifacts(
            final_diagnostics_text=f"{self.name} ETF optimization completed.",
            dimension_reports={},
            overall_verdict="Review OOS robustness before promoting.",
        )

    def _replay_bundle(self):
        if self._cached_bundle is None:
            self._cached_bundle = self.bundle_loader(
                self.data_dir,
                start_date=self.start_date,
                end_date=self.end_date,
            )
        return self._cached_bundle

    @staticmethod
    def _gate_criteria(metrics: dict[str, float]) -> list[GateCriterion]:
        return [
            GateCriterion("net_return_pct", -10.0, float(metrics.get("net_return_pct", 0.0)), float(metrics.get("net_return_pct", 0.0)) >= -10.0),
            GateCriterion("avg_r", -1.25, float(metrics.get("avg_r", 0.0)), float(metrics.get("avg_r", 0.0)) >= -1.25),
            GateCriterion("max_dd_pct", 25.0, float(metrics.get("max_dd_pct", 0.0)), float(metrics.get("max_dd_pct", 0.0)) <= 25.0),
            GateCriterion("total_trades", 5.0, float(metrics.get("total_trades", 0.0)), float(metrics.get("total_trades", 0.0)) >= 5.0),
            GateCriterion("trades_per_month", 0.10, float(metrics.get("trades_per_month", 0.0)), float(metrics.get("trades_per_month", 0.0)) >= 0.10),
        ]


def _worker_args(
    candidates: list[Experiment],
    cumulative_mutations: dict[str, Any],
    current_mutations: dict[str, Any],
    phase: int,
    scoring_weights: dict[str, float] | None,
    hard_rejects: dict[str, float] | None,
) -> list[tuple]:
    base_muts = dict(cumulative_mutations)
    base_muts.update(current_mutations)
    return [
        (candidate.name, candidate.mutations, base_muts, phase, scoring_weights, hard_rejects)
        for candidate in candidates
    ]


def _cohort_balance(values: list[str]) -> float:
    clean = [value for value in values if value]
    if not clean:
        return 0.0
    counts = pd.Series(clean).value_counts()
    if len(counts) <= 1:
        return 0.0
    max_share = float(counts.max() / counts.sum())
    even_share = 1.0 / len(counts)
    return 1.0 - _scale(max_share, even_share, 1.0)


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    if not np.isfinite(value):
        return 0.0
    clipped = min(max(value, low), high)
    return (clipped - low) / (high - low)


def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {name: max(float(value), 0.0) for name, value in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        return dict(DEFAULT_SCORING_WEIGHTS)
    return {name: value / total for name, value in clean.items()}


def _etf_diagnostic_gaps(_phase: int, metrics: dict[str, float]) -> list[str]:
    gaps: list[str] = []
    if float(metrics.get("net_return_pct", 0.0)) <= 0.0:
        gaps.append("Expected return is still negative; inspect entry cohorts and exit giveback before promotion.")
    if float(metrics.get("avg_r", 0.0)) <= 0.0:
        gaps.append("Average R is not positive; the candidate is not yet capturing expectancy per trade.")
    if float(metrics.get("trades_per_month", 0.0)) < 0.75:
        gaps.append("Trade frequency is too low for stable phased optimisation; prefer candidates that add quality sample.")
    if float(metrics.get("total_trades", 0.0)) < 30.0:
        gaps.append("Closed trade sample is small; cohort diagnostics should be treated as directional only.")
    if float(metrics.get("avg_mfe_r", 0.0)) < 0.50:
        gaps.append("Average favourable excursion is weak; prefer candidates that improve MFE before simply adding trades.")
    if float(metrics.get("false_positive_rate", 0.0)) > 0.65:
        gaps.append("Never-worked losers remain too common; entry and false-breakout discrimination need priority.")
    if float(metrics.get("max_dd_pct", 0.0)) > 20.0:
        gaps.append("Drawdown is above the ETF optimisation guardrail; risk and exits need priority.")
    return gaps


def _redesign_etf_weights(
    _phase: int,
    current_weights: dict[str, float] | None,
    analysis,
    _gate_result,
) -> dict[str, float] | None:
    del analysis
    weights = {
        name: float((current_weights or DEFAULT_SCORING_WEIGHTS).get(name, value))
        for name, value in DEFAULT_SCORING_WEIGHTS.items()
    }
    return _normalise_weights(weights)


def run_plugin_cli(plugin_cls: type, prog: str) -> None:
    import argparse

    from backtests.shared.auto.phase_gates import evaluate_gate
    from backtests.shared.auto.phase_runner import PhaseRunner, _mutations_through_phase
    from backtests.shared.auto.phase_state import save_phase_state
    from backtests.shared.auto.round_manager import RoundManager

    parser = argparse.ArgumentParser(prog=prog)
    sub = parser.add_subparsers(dest="command")

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--data-dir", default="backtests/swing/data/raw")
        command.add_argument("--equity", type=float, default=100_000.0)
        command.add_argument("--round", type=int, default=None)
        command.add_argument("--start-date", default=None)
        command.add_argument("--end-date", default=None)
        command.add_argument(
            "--holdout-months",
            type=int,
            default=0,
            help="When --end-date is omitted, hold back this many months from the latest common 15m bar.",
        )
        command.add_argument("--max-workers", type=int, default=None)

    phase_run = sub.add_parser("phase-run")
    add_common(phase_run)
    phase_run.add_argument("--phase", type=int, required=True)
    phase_run.add_argument("--max-rounds", type=int, default=12)
    phase_run.add_argument("--min-delta", type=float, default=0.001)

    phase_auto = sub.add_parser("phase-auto")
    add_common(phase_auto)
    phase_auto.add_argument("--max-rounds", type=int, default=12)
    phase_auto.add_argument("--min-delta", type=float, default=0.001)
    phase_auto.add_argument("--max-retries", type=int, default=2)

    phase_gate = sub.add_parser("phase-gate")
    add_common(phase_gate)
    phase_gate.add_argument("--phase", type=int, required=True)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    name = prog.replace("-auto", "")
    manager = RoundManager("swing", name)
    end_date = args.end_date or _infer_holdout_end_date(Path(args.data_dir), args.holdout_months)

    def build_runner(for_write: bool = True) -> PhaseRunner:
        plugin = plugin_cls(
            Path(args.data_dir),
            initial_equity=args.equity,
            max_workers=getattr(args, "max_workers", None),
            start_date=args.start_date,
            end_date=end_date,
        )
        round_num, round_dir = manager.resolve_round(
            getattr(args, "round", None),
            for_write=for_write,
            expected_phases=plugin.num_phases if for_write else None,
        )
        if round_num > 1:
            plugin.initial_mutations = manager.get_previous_mutations(
                round_num,
                current_provenance=plugin.build_provenance(),
            )
        return PhaseRunner(
            plugin=plugin,
            output_dir=round_dir,
            max_rounds=getattr(args, "max_rounds", None),
            min_delta=getattr(args, "min_delta", 0.001),
            max_retries=getattr(args, "max_retries", 2),
            round_manager=manager,
            round_num=round_num,
        )

    if args.command == "phase-run":
        state = build_runner().run_phase(args.phase)
        result = state.phase_results.get(args.phase, {})
        print(f"Phase {args.phase} complete: {result.get('base_score', 0.0):.4f} -> {result.get('final_score', 0.0):.4f}")
    elif args.command == "phase-auto":
        state = build_runner().run_all_phases()
        print(f"{name} auto-optimization complete. Completed phases: {state.completed_phases}")
    elif args.command == "phase-gate":
        runner = build_runner(for_write=False)
        state = runner.load_state()
        muts = dict(getattr(runner.plugin, "initial_mutations", None) or {})
        muts.update(_mutations_through_phase(state, args.phase))
        metrics = runner.plugin.compute_final_metrics(muts)
        spec = runner.plugin.get_phase_spec(args.phase, state)
        gate = evaluate_gate(spec.gate_criteria_fn(metrics))
        state.record_gate(args.phase, {"passed": gate.passed, "criteria": [c.__dict__ for c in gate.criteria]})
        save_phase_state(state, runner.state_path)
        print(f"Phase {args.phase} gate: {'PASSED' if gate.passed else 'FAILED'}")


def _infer_holdout_end_date(data_dir: Path, holdout_months: int) -> str | None:
    if holdout_months <= 0:
        return None
    import pandas as pd

    ends = []
    for symbol in ("QQQ", "GLD"):
        path = data_dir / f"{symbol}_15m.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        if len(idx):
            ends.append(idx.max())
    if not ends:
        return None
    cutoff = min(ends) - pd.DateOffset(months=holdout_months)
    return pd.Timestamp(cutoff).isoformat()
