"""Round 8 TPC phased auto-optimisation plugin.

This plugin keeps the normal TPC strategy runner, replay bundle, and round-7
baseline lane. Round-8 candidates enable additive completed-30m pullback setups,
then test true 30m MA slope/transition context separately from EMA20 reclaim
geometry.
The holdout is not scored here; callers should pass an end_date that stops
before holdout.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from backtests.shared.auto.cache_keys import build_cache_key
from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.provenance import AutoRunProvenance, build_phase_auto_provenance
from backtests.shared.auto.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    SharedPoolBatchEvaluator,
    create_process_pool,
    shutdown_process_pool,
)
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion, ScoredCandidate
from backtests.swing.analysis.tpc_rejected_bar_forward import build_pb30_rejected_forward_report
from backtests.swing.data.replay_cache import load_tpc_replay_bundle

from .plugin import TPCPlugin, TPCScore
from .round8_candidates import get_round8_phase_candidates

logger = logging.getLogger(__name__)


ROUND8_SCORING_WEIGHTS = {
    "alpha_quality": 0.28,
    "false_positive_control": 0.22,
    "risk_quality": 0.17,
    "frequency_floor": 0.14,
    "symbol_balance": 0.11,
    "stability": 0.08,
}

ROUND8_HARD_REJECTS = {
    "min_valid_trades": 90.0,
    "min_trades_per_month": 1.60,
    "min_return_pct": 70.0,
    "min_total_r": 70.0,
    "min_avg_r": 0.42,
    "min_dollar_profit_factor": 1.35,
    "max_dd_pct": 18.0,
    "max_never_worked_rate": 0.34,
    "max_low_mfe_loss_rate": 0.44,
    "max_right_then_lost_rate": 0.12,
    "max_top5_winner_share": 0.46,
    "max_dollar_top5_winner_share": 0.43,
    "max_symbol_trade_share": 0.90,
    "min_qqq_trades": 16.0,
    "min_worst_year_pnl_pct": -14.0,
    "min_additive_trades": 5.0,
    "min_additive_avg_r": 0.12,
    "min_additive_total_r": 1.00,
    "max_additive_low_mfe_loss_rate": 0.50,
    "min_pb30_plain_trades": 5.0,
    "min_pb30_plain_avg_r": 0.12,
    "min_pb30_plain_total_r": 1.00,
    "max_pb30_plain_low_mfe_loss_rate": 0.50,
    "min_pb30_ema20_trades": 5.0,
    "min_pb30_ema20_avg_r": 0.12,
    "min_pb30_ema20_total_r": 1.00,
    "max_pb30_ema20_low_mfe_loss_rate": 0.50,
    "min_ema20_touch_trades": 5.0,
    "min_ema20_touch_avg_r": 0.12,
    "min_ema20_touch_total_r": 1.00,
    "max_ema20_touch_low_mfe_loss_rate": 0.50,
}


class _Round8SequentialBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float,
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
        start_date: str | None,
        end_date: str | None,
    ) -> None:
        self._data_dir = data_dir
        self._initial_equity = initial_equity
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._start_date = start_date
        self._end_date = end_date
        self._initialised = False

    def _ensure_init(self) -> None:
        if self._initialised:
            return
        from .round8_worker import init_worker

        init_worker(str(self._data_dir), self._initial_equity, self._start_date, self._end_date)
        self._initialised = True

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]) -> list[ScoredCandidate]:
        self._ensure_init()
        from .round8_worker import score_candidate

        return [
            score_candidate((candidate.name, candidate.mutations, current_mutations, self._phase, self._scoring_weights, self._hard_rejects))
            for candidate in candidates
        ]

    def close(self) -> None:
        pass


class Round8TPCPlugin(TPCPlugin):
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 100_000.0,
        max_workers: int | None = None,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        super().__init__(
            data_dir,
            initial_equity=initial_equity,
            max_workers=max_workers,
            num_phases=4,
            start_date=start_date,
            end_date=end_date,
            score_holdout=False,
        )
        self.candidates_fn = get_round8_phase_candidates

    def build_provenance(self) -> AutoRunProvenance:
        if self._provenance is None:
            repo_root = Path(__file__).resolve().parents[4]
            self._provenance = build_phase_auto_provenance(
                self.name,
                repo_root=repo_root,
                code_dirs=(
                    Path(__file__).resolve().parent,
                    repo_root / "strategies/swing/tpc",
                ),
                code_paths=(
                    Path(__file__).resolve(),
                    repo_root / "backtests/swing/engine/tpc_engine.py",
                    repo_root / "backtests/swing/config_tpc.py",
                    repo_root / "backtests/swing/data/replay_cache.py",
                ),
                data_dir=self.data_dir,
                selection_context={
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "initial_equity": self.initial_equity,
                    "num_phases": self.num_phases,
                    "scoring_weights": ROUND8_SCORING_WEIGHTS,
                    "hard_rejects": ROUND8_HARD_REJECTS,
                    "round_baseline_policy": "run_spec.baseline_mutations",
                },
            )
        return self._provenance

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 145.0,
            "avg_r": 0.80,
            "dollar_profit_factor": 2.25,
            "total_trades": 145.0,
            "trades_per_month": 2.60,
            "low_mfe_loss_rate": 0.30,
            "max_dd_pct": 14.0,
        }

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        del state
        candidates = [Experiment(name=name, mutations=mutations) for name, mutations in self.candidates_fn(phase)]
        return PhaseSpec(
            focus={
                1: "ADDITIVE_SCALED_30M_PULLBACK_CONTROL",
                2: "PB30_TRUE_MA_TRANSITION_FILTERS",
                3: "PB30_RECLAIM_VS_MA_TRANSITION_INTERACTIONS",
                4: "RISK_EXIT_AFTER_REJECTION_EVIDENCE",
            }.get(phase, "ROUND8_TPC"),
            candidates=candidates,
            gate_criteria_fn=self._round8_gate_criteria,
            scoring_weights=_validate_round8_weights(ROUND8_SCORING_WEIGHTS),
            hard_rejects=dict(ROUND8_HARD_REJECTS),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=[
                    "net_return_pct",
                    "avg_r",
                    "dollar_profit_factor",
                    "total_trades",
                    "additive_trade_count",
                    "additive_avg_r",
                    "low_mfe_loss_rate",
                    "max_dd_pct",
                ],
                diagnostic_gap_fn=_round8_diagnostic_gaps,
                min_effective_score_delta_pct=0.003,
            ),
            max_rounds=3,
            prune_threshold=0.035,
            reject_streak_limit=2,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        weights = _validate_round8_weights(scoring_weights or ROUND8_SCORING_WEIGHTS)
        evaluation_key = build_cache_key(
            "swing.tpc.round8.evaluation",
            source_fingerprint=self._replay_bundle().cache_source_fingerprint,
            extra={
                "phase": phase,
                "scoring_weights": weights,
                "hard_rejects": hard_rejects or {},
            },
        )

        def make_parallel():
            self._ensure_pool()
            from .round8_worker import score_candidate

            return SharedPoolBatchEvaluator(
                self._pool,
                worker_fn=score_candidate,
                build_args=lambda candidates, current_mutations: _round8_worker_args(
                    candidates,
                    cumulative_mutations,
                    current_mutations,
                    phase,
                    weights,
                    hard_rejects,
                ),
                on_terminate=self._on_pool_terminate,
                description=f"TPC round 8 phase {phase}",
                logger=logger,
            )

        def make_sequential():
            return _Round8SequentialBatchEvaluator(
                self.data_dir,
                self.initial_equity,
                phase,
                weights,
                hard_rejects,
                self.start_date,
                self.end_date,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"TPC round 8 phase {phase}", logger=logger)
        return CachedBatchEvaluator(
            raw,
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
            max_batch_size=max(1, self.max_workers),
        )

    def score_mutations(
        self,
        mutations: dict[str, Any],
        phase: int,
        hard_rejects: dict[str, float] | None,
        weights: dict[str, float] | None,
    ) -> tuple[dict[str, float], TPCScore]:
        del phase
        metrics = self.compute_train_metrics(mutations)
        return metrics, _round8_composite_score(metrics, hard_rejects, weights or ROUND8_SCORING_WEIGHTS)

    def run_phase_diagnostics(self, phase: int, state, metrics: dict[str, float], greedy_result) -> str:
        summary = (
            f"tpc round8 phase {phase}: "
            f"net={metrics.get('net_return_pct', 0):+.2f}%, "
            f"avgR={metrics.get('avg_r', 0):+.3f}, "
            f"totalR={metrics.get('total_r', 0):+.2f}, "
            f"trades={metrics.get('total_trades', 0):.0f}, "
            f"trades/month={metrics.get('trades_per_month', 0):.2f}, "
            f"$PF={metrics.get('dollar_profit_factor', 0):.2f}, "
            f"never-worked={metrics.get('never_worked_rate', 0):.0%}, "
            f"low-MFE losses={metrics.get('low_mfe_loss_rate', 0):.0%}, "
            f"additive={metrics.get('additive_trade_count', 0):.0f} trades/"
            f"{metrics.get('additive_avg_r', 0):+.3f} avgR/"
            f"{metrics.get('additive_total_r', 0):+.2f}R, "
            f"routes=pb30 {metrics.get('pb30_plain_trade_count', 0):.0f}/"
            f"pb30ema {metrics.get('pb30_ema20_trade_count', 0):.0f}/"
            f"ema1h {metrics.get('ema20_touch_trade_count', 0):.0f}, "
            f"QQQ={metrics.get('qqq_trade_count', 0):.0f}, "
            f"GLD share={metrics.get('gld_trade_share', 0):.0%}, "
            f"top5$={metrics.get('dollar_top5_winner_share', 0):.0%}, "
            f"worst-year={metrics.get('worst_year_pnl_pct', 0):+.2f}%, "
            f"DD={metrics.get('max_dd_pct', 0):.2f}%, "
            f"accepted={greedy_result.accepted_count}"
        )
        if phase in {2, 3}:
            try:
                summary += build_pb30_rejected_forward_report(
                    self._replay_bundle().data,
                    data_dir=self.data_dir,
                    initial_equity=self.initial_equity,
                    base_mutations=dict(getattr(state, "cumulative_mutations", {}) or {}),
                    candidate_mutations=self.candidates_fn(phase),
                    horizon_bars_15m=32,
                    max_candidates=16,
                )
            except Exception as exc:
                summary += f"\nRejected-bar forward MFE/MAE diagnostics failed: {exc}"
        return summary

    def build_end_of_round_artifacts(self, state) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        lines = [
            "TPC ROUND 8 FINAL TRAIN-ONLY DIAGNOSTICS",
            "",
            "Research scope: round-7 baseline plus additive completed-30m pullback, with true MA slope/transition filters tested separately from EMA20 reclaim geometry.",
            f"Training window end: {self.end_date or 'full available sample'}",
            "Holdout policy: excluded from scoring and diagnostics for this optimisation run.",
            "",
            "Score components: alpha_quality, false_positive_control, risk_quality, frequency_floor, symbol_balance, stability.",
            "Additive evidence gates apply only when an additive lane actually trades: each active additive route needs min 5 trades, avgR >= +0.12, totalR >= +1.00, low-MFE loss rate <= 50%.",
            "",
            _format_metric_line(metrics),
            "",
            "Kept round features:",
            *[f"- {name}" for phase in sorted(state.phase_results) for name in state.phase_results[phase].get("kept_features", [])],
            "",
            "Structural guardrails:",
            "- The canonical round-7 replay bundle remains the baseline; completed 30m bars are read through the normal bar_input.bars_30m path.",
            "- True 30m MA slope/transition filters qualify the completed-30m pullback route; EMA20 touch/reclaim geometry remains a separate context layer.",
            "- Hard rejects penalise tiny samples, weak PF/expectancy, high false positives, high drawdown, high top-winner concentration, poor worst-year behaviour, and additive lanes without enough direct evidence.",
        ]
        return EndOfRoundArtifacts(
            final_diagnostics_text="\n".join(lines) + "\n",
            dimension_reports={},
            overall_verdict="Train-only research result. Promote only after untouched holdout and live-path parity checks.",
        )

    def _ensure_pool(self) -> None:
        if self._pool is not None and not self._pool_dirty:
            return
        if self._pool is not None:
            shutdown_process_pool(self._pool, force=True)
        from .round8_worker import init_worker

        self._pool = create_process_pool(
            self.max_workers,
            initializer=init_worker,
            initargs=(str(self.data_dir), self.initial_equity, self.start_date, self.end_date),
            logger=logger,
            description="TPC round 8 evaluation",
        )
        self._pool_dirty = False

    def _replay_bundle(self):
        bundle = load_tpc_replay_bundle(
            self.data_dir,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        if self._cache_source_fingerprint != bundle.cache_source_fingerprint:
            self._metrics_cache.clear()
            self._holdout_metrics_cache.clear()
            self._evaluation_cache.clear()
            self.close_pool()
            self._cache_source_fingerprint = bundle.cache_source_fingerprint
        self._cached_bundle = bundle
        return bundle

    def _holdout_replay_bundle(self):
        return load_tpc_replay_bundle(
            self.data_dir,
            start_date=self.start_date,
            end_date=None,
        )

    @staticmethod
    def _round8_gate_criteria(metrics: dict[str, float]) -> list[GateCriterion]:
        return [
            GateCriterion("total_trades", 90.0, float(metrics.get("total_trades", 0.0)), float(metrics.get("total_trades", 0.0)) >= 90.0),
            GateCriterion("net_return_pct", 70.0, float(metrics.get("net_return_pct", 0.0)), float(metrics.get("net_return_pct", 0.0)) >= 70.0),
            GateCriterion("avg_r", 0.42, float(metrics.get("avg_r", 0.0)), float(metrics.get("avg_r", 0.0)) >= 0.42),
            GateCriterion("dollar_profit_factor", 1.35, float(metrics.get("dollar_profit_factor", 0.0)), float(metrics.get("dollar_profit_factor", 0.0)) >= 1.35),
            GateCriterion("low_mfe_loss_rate", 0.44, float(metrics.get("low_mfe_loss_rate", 1.0)), float(metrics.get("low_mfe_loss_rate", 1.0)) <= 0.44),
            GateCriterion("max_dd_pct", 18.0, float(metrics.get("max_dd_pct", 0.0)), float(metrics.get("max_dd_pct", 0.0)) <= 18.0),
        ]


def _round8_worker_args(
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


def _round8_composite_score(
    metrics: dict[str, float],
    hard_rejects: dict[str, float] | None,
    weights: dict[str, float],
) -> TPCScore:
    weights = _validate_round8_weights(weights)
    rejects = hard_rejects or {}
    checks = [
        (metrics.get("total_trades", 0.0) < rejects.get("min_valid_trades", 1.0), "min_valid_trades"),
        (metrics.get("trades_per_month", 0.0) < rejects.get("min_trades_per_month", 0.0), "min_trades_per_month"),
        (metrics.get("net_return_pct", 0.0) < rejects.get("min_return_pct", -99.0), "min_return_pct"),
        (metrics.get("total_r", 0.0) < rejects.get("min_total_r", -99.0), "min_total_r"),
        (metrics.get("avg_r", 0.0) < rejects.get("min_avg_r", -99.0), "min_avg_r"),
        (metrics.get("dollar_profit_factor", 0.0) < rejects.get("min_dollar_profit_factor", 0.0), "min_dollar_profit_factor"),
        (metrics.get("max_dd_pct", 0.0) > rejects.get("max_dd_pct", 99.0), "max_dd_pct"),
        (metrics.get("never_worked_rate", 0.0) > rejects.get("max_never_worked_rate", 1.0), "max_never_worked_rate"),
        (metrics.get("low_mfe_loss_rate", 0.0) > rejects.get("max_low_mfe_loss_rate", 1.0), "max_low_mfe_loss_rate"),
        (metrics.get("right_then_lost_rate", 0.0) > rejects.get("max_right_then_lost_rate", 1.0), "max_right_then_lost_rate"),
        (metrics.get("top5_winner_share", 0.0) > rejects.get("max_top5_winner_share", 1.0), "max_top5_winner_share"),
        (metrics.get("dollar_top5_winner_share", 0.0) > rejects.get("max_dollar_top5_winner_share", 1.0), "max_dollar_top5_winner_share"),
        (metrics.get("max_symbol_trade_share", 0.0) > rejects.get("max_symbol_trade_share", 1.0), "max_symbol_trade_share"),
        (metrics.get("qqq_trade_count", 0.0) < rejects.get("min_qqq_trades", 0.0), "min_qqq_trades"),
        (metrics.get("worst_year_pnl_pct", 0.0) < rejects.get("min_worst_year_pnl_pct", -99.0), "min_worst_year_pnl_pct"),
    ]
    for failed, reason in checks:
        if failed:
            return TPCScore(0.0, True, reason)
    additive_count = metrics.get("additive_trade_count", 0.0)
    if additive_count > 0:
        additive_checks = [
            (additive_count < rejects.get("min_additive_trades", 1.0), "min_additive_trades"),
            (metrics.get("additive_avg_r", 0.0) < rejects.get("min_additive_avg_r", -99.0), "min_additive_avg_r"),
            (metrics.get("additive_total_r", 0.0) < rejects.get("min_additive_total_r", -99.0), "min_additive_total_r"),
            (
                metrics.get("additive_low_mfe_loss_rate", 0.0)
                > rejects.get("max_additive_low_mfe_loss_rate", 1.0),
                "max_additive_low_mfe_loss_rate",
            ),
        ]
        for failed, reason in additive_checks:
            if failed:
                return TPCScore(0.0, True, reason)
    for route in ("pb30_plain", "pb30_ema20", "ema20_touch"):
        route_count = metrics.get(f"{route}_trade_count", 0.0)
        if route_count <= 0:
            continue
        route_checks = [
            (route_count < rejects.get(f"min_{route}_trades", rejects.get("min_additive_trades", 1.0)), f"min_{route}_trades"),
            (
                metrics.get(f"{route}_avg_r", 0.0)
                < rejects.get(f"min_{route}_avg_r", rejects.get("min_additive_avg_r", -99.0)),
                f"min_{route}_avg_r",
            ),
            (
                metrics.get(f"{route}_total_r", 0.0)
                < rejects.get(f"min_{route}_total_r", rejects.get("min_additive_total_r", -99.0)),
                f"min_{route}_total_r",
            ),
            (
                metrics.get(f"{route}_low_mfe_loss_rate", 0.0)
                > rejects.get(f"max_{route}_low_mfe_loss_rate", rejects.get("max_additive_low_mfe_loss_rate", 1.0)),
                f"max_{route}_low_mfe_loss_rate",
            ),
        ]
        for failed, reason in route_checks:
            if failed:
                return TPCScore(0.0, True, reason)

    total_trades = metrics.get("total_trades", 0.0)
    trades_per_month = metrics.get("trades_per_month", 0.0)
    max_dd = metrics.get("max_dd_pct", 0.0)
    return_to_dd = metrics.get("net_return_pct", 0.0) / max(max_dd, 1e-9)
    mfe_mae_ratio = metrics.get("avg_mfe_r", 0.0) / max(metrics.get("avg_mae_r", 0.0), 1e-9)
    components = {
        "alpha_quality": (
            0.22 * _scale(metrics.get("avg_r", 0.0), 0.42, 0.90)
            + 0.20 * _scale(metrics.get("total_r", 0.0), 70.0, 120.0)
            + 0.20 * _scale(metrics.get("dollar_profit_factor", 0.0), 1.35, 2.50)
            + 0.18 * _scale(metrics.get("net_return_pct", 0.0), 70.0, 150.0)
            + 0.12 * _scale(metrics.get("mfe_capture", 0.0), 0.25, 0.48)
            + 0.08 * _scale(mfe_mae_ratio, 1.0, 3.5)
        ),
        "false_positive_control": (
            0.40 * (1.0 - _scale(metrics.get("never_worked_rate", 1.0), 0.22, 0.34))
            + 0.35 * (1.0 - _scale(metrics.get("low_mfe_loss_rate", 1.0), 0.31, 0.44))
            + 0.15 * (1.0 - _scale(metrics.get("right_then_lost_rate", 1.0), 0.04, 0.12))
            + 0.10 * _scale(metrics.get("excellent_rate", 0.0), 0.48, 0.65)
        ),
        "risk_quality": (
            0.30 * (1.0 - _scale(max_dd, 10.0, 18.0))
            + 0.25 * _scale(return_to_dd, 4.0, 12.0)
            + 0.20 * _scale(metrics.get("dollar_profit_factor", 0.0), 1.35, 2.30)
            + 0.15 * (1.0 - _scale(metrics.get("dollar_top5_winner_share", 0.0), 0.25, 0.43))
            + 0.10 * _scale(metrics.get("sharpe", 0.0), 0.50, 1.20)
        ),
        "frequency_floor": (
            0.42 * _scale(trades_per_month, 1.60, 3.00)
            + 0.32 * _scale(total_trades, 90.0, 165.0)
            + 0.16 * _scale(metrics.get("excellent_trades_per_month", 0.0), 0.75, 1.55)
            + 0.10 * _scale(metrics.get("two_r_plus_rate", 0.0), 0.32, 0.52)
        ),
        "symbol_balance": (
            0.35 * (1.0 - _scale(metrics.get("max_symbol_trade_share", 1.0), 0.65, 0.90))
            + 0.25 * _scale(metrics.get("qqq_trade_count", 0.0), 16.0, 42.0)
            + 0.20 * _scale(metrics.get("qqq_excellent_trades", 0.0), 9.0, 24.0)
            + 0.20 * _scale(metrics.get("worst_symbol_avg_r", 0.0), -0.10, 0.90)
        ),
        "stability": (
            0.32 * _scale(metrics.get("worst_year_pnl_pct", 0.0), -12.0, 10.0)
            + 0.28 * (1.0 - _scale(metrics.get("top5_winner_share", 0.0), 0.30, 0.46))
            + 0.20 * (1.0 - _scale(metrics.get("dollar_top5_winner_share", 0.0), 0.25, 0.43))
            + 0.20 * _scale(total_trades, 90.0, 165.0)
        ),
    }
    clean_weights = _normalise_weights(weights)
    score = sum(clean_weights.get(name, 0.0) * max(0.0, min(components.get(name, 0.0), 1.0)) for name in clean_weights)
    return TPCScore(max(0.0, score * 100.0))


def _round8_diagnostic_gaps(_phase: int, metrics: dict[str, float]) -> list[str]:
    gaps: list[str] = []
    if metrics.get("total_trades", 0.0) < 120.0:
        gaps.append("Trade sample fell below the round-7 baseline; additive lanes should not win merely by starving the setup set.")
    if metrics.get("low_mfe_loss_rate", 1.0) > 0.38:
        gaps.append("Too many trades still fail before 1R MFE; favour pullback control or confirmation quality over extra supply.")
    if metrics.get("dollar_top5_winner_share", 0.0) > 0.38:
        gaps.append("Dollar winner concentration is high; reduce risk scaling or require broader symbol/year contribution.")
    if metrics.get("worst_year_pnl_pct", 0.0) < -10.0:
        gaps.append("Worst-year behaviour is fragile; the score should prefer stability over a concentrated return burst.")
    if metrics.get("max_symbol_trade_share", 1.0) > 0.90:
        gaps.append("One symbol dominates the evidence; require symbol-specific proof before treating the effect as portable alpha.")
    if 0.0 < metrics.get("additive_trade_count", 0.0) < ROUND8_HARD_REJECTS["min_additive_trades"]:
        gaps.append("Additive lane evidence is too thin; keep the route as a hypothesis until it contributes a larger sample.")
    if 0.0 < metrics.get("pb30_ema20_trade_count", 0.0) < ROUND8_HARD_REJECTS["min_pb30_ema20_trades"]:
        gaps.append("30m EMA-touch evidence is too thin; do not keep it unless it proves itself as a route, not as an interaction artifact.")
    if 0.0 < metrics.get("ema20_touch_trade_count", 0.0) < ROUND8_HARD_REJECTS["min_ema20_touch_trades"]:
        gaps.append("Standalone EMA20 touch evidence is too thin; keep it behind the controlled 30m pullback route.")
    return gaps


def _validate_round8_weights(weights: dict[str, float]) -> dict[str, float]:
    if len(weights) > 7:
        raise ValueError("Round 8 TPC score must not contain more than 7 components.")
    expected = frozenset(ROUND8_SCORING_WEIGHTS)
    actual = frozenset(weights)
    if actual != expected:
        raise ValueError(f"Round 8 TPC score components must be {sorted(expected)}, got {sorted(actual)}")
    return dict(weights)


def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {name: max(float(value), 0.0) for name, value in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        return dict(ROUND8_SCORING_WEIGHTS)
    return {name: value / total for name, value in clean.items()}


def _scale(value: float, low: float, high: float) -> float:
    if high <= low or not np.isfinite(value):
        return 0.0
    return (min(max(float(value), low), high) - low) / (high - low)


def _format_metric_line(metrics: dict[str, float]) -> str:
    return (
        f"net={metrics.get('net_return_pct', 0):+.2f}%, "
        f"avgR={metrics.get('avg_r', 0):+.3f}, "
        f"totalR={metrics.get('total_r', 0):+.2f}, "
        f"trades={metrics.get('total_trades', 0):.0f}, "
        f"trades/month={metrics.get('trades_per_month', 0):.2f}, "
        f"$PF={metrics.get('dollar_profit_factor', 0):.2f}, "
        f"additive={metrics.get('additive_trade_count', 0):.0f} trades/"
        f"{metrics.get('additive_avg_r', 0):+.3f} avgR/"
        f"{metrics.get('additive_total_r', 0):+.2f}R, "
        f"routes=pb30 {metrics.get('pb30_plain_trade_count', 0):.0f}/"
        f"pb30ema {metrics.get('pb30_ema20_trade_count', 0):.0f}/"
        f"ema1h {metrics.get('ema20_touch_trade_count', 0):.0f}, "
        f"low-MFE losses={metrics.get('low_mfe_loss_rate', 0):.0%}, "
        f"DD={metrics.get('max_dd_pct', 0):.2f}%"
    )
