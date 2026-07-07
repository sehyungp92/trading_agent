"""TPC phased auto-optimization plugin."""
from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtests.shared.auto.cache_keys import build_cache_key
from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.provenance import AutoRunProvenance, build_phase_auto_provenance
from backtests.shared.auto.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    SharedPoolBatchEvaluator,
    create_process_pool,
    mutation_signature,
    shutdown_process_pool,
)
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion
from backtests.swing.auto.etf_common import ETFPhasePlugin
from backtests.swing.config_tpc import TPCBacktestConfig
from backtests.swing.data.replay_cache import load_tpc_replay_bundle
from backtests.swing.engine.tpc_engine import run_tpc_independent

from .phase_candidates import get_phase_candidates

logger = logging.getLogger(__name__)


TPC_SCORING_WEIGHTS = {
    "false_positive_control": 0.30,
    "symbol_balance": 0.18,
    "alpha_quality": 0.18,
    "frequency_floor": 0.14,
    "expected_r": 0.12,
    "risk_quality": 0.08,
}
TPC_SCORE_COMPONENTS = frozenset(TPC_SCORING_WEIGHTS)


@dataclass(frozen=True)
class TPCScore:
    total: float
    rejected: bool = False
    reject_reason: str = ""


class _SequentialBatchEvaluator:
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
        from .worker import init_worker

        init_worker(str(self._data_dir), self._initial_equity, self._start_date, self._end_date)
        self._initialised = True

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        self._ensure_init()
        from .worker import score_candidate

        return [
            score_candidate((c.name, c.mutations, current_mutations, self._phase, self._scoring_weights, self._hard_rejects))
            for c in candidates
        ]

    def close(self) -> None:
        pass


class TPCPlugin(ETFPhasePlugin):
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 100_000.0,
        max_workers: int | None = None,
        *,
        num_phases: int = 7,
        start_date: str | None = None,
        end_date: str | None = None,
        score_holdout: bool = False,
    ) -> None:
        super().__init__(
            name="tpc",
            data_dir=data_dir,
            config_factory=TPCBacktestConfig,
            bundle_loader=load_tpc_replay_bundle,
            runner=run_tpc_independent,
            candidates_fn=get_phase_candidates,
            initial_equity=initial_equity,
            max_workers=max_workers,
            num_phases=num_phases,
            start_date=start_date,
            end_date=end_date,
        )
        self.max_workers = max(1, int(max_workers or 1))
        self._pool: mp.Pool | None = None
        self._pool_dirty = False
        self._evaluation_cache: dict[str, Any] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._holdout_metrics_cache: dict[str, dict[str, float]] = {}
        self._cache_source_fingerprint = ""
        self._holdout_warmup_15m: int | None = None
        self.score_holdout = bool(score_holdout)
        self._provenance: AutoRunProvenance | None = None

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
                    "score_holdout": self.score_holdout,
                    "scoring_weights": TPC_SCORING_WEIGHTS,
                    "round_baseline_policy": "run_spec.baseline_mutations",
                },
            )
        return self._provenance

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "false_positive_never_worked_rate": 0.26,
            "false_positive_low_mfe_loss_rate": 0.36,
            "right_then_lost_rate": 0.07,
            "max_symbol_trade_share": 0.78,
            "qqq_excellent_trades": 13.0,
            "total_r": 90.0,
            "trades_per_month": 2.30,
            "avg_r": 0.70,
            "dollar_profit_factor": 2.05,
        }

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        del state
        candidates = [Experiment(name=n, mutations=m) for n, m in self.candidates_fn(phase)]
        hard_rejects = {
            "min_valid_trades": 85,
            "min_trades_per_month": 1.50,
            "min_excellent_trades": 42,
            "max_dd_pct": 16.0,
            "min_return_pct": 55.0,
            "min_total_r": 70.0,
            "min_avg_r": 0.50,
            "min_profit_factor": 1.35,
            "min_dollar_profit_factor": 1.35,
            "max_never_worked_rate": 0.335,
            "max_low_mfe_loss_rate": 0.435,
            "max_right_then_lost_rate": 0.10,
            "max_top5_winner_share": 0.45,
            "max_dollar_top5_winner_share": 0.42,
            "max_symbol_trade_share": 0.88,
            "max_gld_trade_share": 0.88,
            "min_qqq_trades": 16,
            "min_worst_symbol_avg_r": -0.10,
            "min_worst_symbol_pnl_pct": -6.0,
            "min_worst_year_pnl_pct": -12.0,
        }
        return PhaseSpec(
            focus={
                1: "GLD_SIGNAL_DISCRIMINATION",
                2: "REGIME_ROOM_DISCRIMINATION",
                3: "SESSION_SYMBOL_BALANCE",
                4: "ENTRY_QUALITY_DISCRIMINATION",
                5: "PULLBACK_PURITY",
                6: "DISCRIMINATION_COMBOS",
                7: "QQQ_EXCELLENT_SUPPLY",
            }.get(phase, "TPC_FINETUNE"),
            candidates=candidates,
            gate_criteria_fn=self._tpc_oos_repair_gate_criteria if self.score_holdout and phase == 6 else self._tpc_gate_criteria,
            scoring_weights=_validate_tpc_score_weights(TPC_SCORING_WEIGHTS),
            hard_rejects=hard_rejects,
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=[
                    "false_positive_never_worked_rate",
                    "false_positive_low_mfe_loss_rate",
                    "right_then_lost_rate",
                    "max_symbol_trade_share",
                    "gld_trade_share",
                    "qqq_trade_count",
                    "qqq_excellent_trades",
                    "qqq_excellent_rate",
                    "total_r",
                    "avg_r",
                    "trades_per_month",
                ],
                diagnostic_gap_fn=_tpc_diagnostic_gaps,
                min_effective_score_delta_pct=0.005,
            ),
            max_rounds=4,
            prune_threshold=0.025,
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
        weights = _validate_tpc_score_weights(scoring_weights or TPC_SCORING_WEIGHTS)
        evaluation_key = build_cache_key(
            "swing.tpc.evaluation",
            source_fingerprint=self._replay_bundle().cache_source_fingerprint,
            extra={
                "phase": phase,
                "scoring_weights": weights,
                "hard_rejects": hard_rejects or {},
            },
        )

        def make_parallel():
            self._ensure_pool()
            from .worker import score_candidate

            return SharedPoolBatchEvaluator(
                self._pool,
                worker_fn=score_candidate,
                build_args=lambda candidates, current_mutations: _worker_args(
                    candidates,
                    cumulative_mutations,
                    current_mutations,
                    phase,
                    weights,
                    hard_rejects,
                ),
                on_terminate=self._on_pool_terminate,
                description=f"TPC phase {phase}",
                logger=logger,
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir,
                self.initial_equity,
                phase,
                weights,
                hard_rejects,
                self.start_date,
                self.end_date,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"TPC phase {phase}", logger=logger)
        return CachedBatchEvaluator(
            raw,
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
            max_batch_size=max(1, self.max_workers),
        )

    def compute_train_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        sig = mutation_signature(mutations)
        cached = self._metrics_cache.get(sig)
        if cached is not None:
            return dict(cached)

        cfg = self.config_factory(initial_equity=self.initial_equity, data_dir=self.data_dir)
        if hasattr(cfg, "with_overrides"):
            cfg = cfg.with_overrides(mutations)
        result = self.runner(self._replay_bundle().data, cfg, indicator_cache={})
        metrics = _extract_tpc_metrics(result, self.initial_equity)
        self._metrics_cache[sig] = dict(metrics)
        return metrics

    def compute_final_metrics(self, mutations: dict[str, Any], *, indicator_cache: dict | None = None) -> dict[str, float]:
        del indicator_cache
        metrics = self.compute_train_metrics(mutations)
        if self.end_date and self.score_holdout:
            return _with_oos_metrics(metrics, self.compute_holdout_metrics(mutations))
        return metrics

    def compute_holdout_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        if not self.end_date:
            return {}
        sig = f"holdout:{mutation_signature(mutations)}"
        cached = self._holdout_metrics_cache.get(sig)
        if cached is not None:
            return dict(cached)

        cfg = self.config_factory(initial_equity=self.initial_equity, data_dir=self.data_dir)
        holdout_mutations = dict(mutations)
        holdout_mutations["warmup_15m"] = self._infer_holdout_warmup_15m()
        if hasattr(cfg, "with_overrides"):
            cfg = cfg.with_overrides(holdout_mutations)
        result = self.runner(self._holdout_replay_bundle().data, cfg, indicator_cache={})
        metrics = _extract_tpc_metrics(result, self.initial_equity)
        self._holdout_metrics_cache[sig] = dict(metrics)
        return metrics

    def score_mutations(
        self,
        mutations: dict[str, Any],
        phase: int,
        hard_rejects: dict[str, float] | None,
        weights: dict[str, float] | None,
    ) -> tuple[dict[str, float], TPCScore]:
        clean_weights = weights or TPC_SCORING_WEIGHTS
        if phase == 6 and self.end_date and self.score_holdout:
            metrics = self.compute_final_metrics(mutations)
            return metrics, _tpc_oos_repair_score(metrics, hard_rejects, clean_weights)
        metrics = self.compute_train_metrics(mutations)
        return metrics, _tpc_composite_score(metrics, hard_rejects, clean_weights)

    def run_phase_diagnostics(self, phase: int, state, metrics: dict[str, float], greedy_result) -> str:
        del state
        if phase == 6 and self.score_holdout and "oos_net_return_pct" in metrics:
            return (
                f"tpc phase {phase}: "
                f"train net={metrics.get('net_return_pct', 0):+.2f}%, "
                f"train avgR={metrics.get('avg_r', 0):+.3f}, "
                f"train trades/month={metrics.get('trades_per_month', 0):.2f}, "
                f"train $PF={metrics.get('dollar_profit_factor', 0):.2f}, "
                f"train low-MFE losses={metrics.get('low_mfe_loss_rate', 0):.0%}, "
                f"OOS net={metrics.get('oos_net_return_pct', 0):+.2f}%, "
                f"OOS avgR={metrics.get('oos_avg_r', 0):+.3f}, "
                f"OOS trades={metrics.get('oos_total_trades', 0):.0f}, "
                f"OOS $PF={metrics.get('oos_dollar_profit_factor', 0):.2f}, "
                f"OOS low-MFE losses={metrics.get('oos_low_mfe_loss_rate', 0):.0%}, "
                f"OOS DD={metrics.get('oos_max_dd_pct', 0):.2f}%, "
                f"accepted={greedy_result.accepted_count}"
            )
        return (
            f"tpc phase {phase}: "
            f"avgR={metrics.get('avg_r', 0):+.3f}, "
            f"totalR={metrics.get('total_r', 0):+.2f}, "
            f"trades={metrics.get('total_trades', 0):.0f}, "
            f"trades/month={metrics.get('trades_per_month', 0):.2f}, "
            f"excellent={metrics.get('excellent_trades', 0):.0f}, "
            f"QQQ={metrics.get('qqq_trade_count', 0):.0f}, "
            f"QQQ excellent={metrics.get('qqq_excellent_trades', 0):.0f}, "
            f"GLD share={metrics.get('gld_trade_share', 0):.0%}, "
            f"$PF={metrics.get('dollar_profit_factor', 0):.2f}, "
            f"RPF={metrics.get('profit_factor', 0):.2f}, "
            f"MFE capture={metrics.get('mfe_capture', 0):+.2f}, "
            f"never-worked={metrics.get('never_worked_rate', 0):.0%}, "
            f"low-MFE losses={metrics.get('low_mfe_loss_rate', 0):.0%}, "
            f"DD={metrics.get('max_dd_pct', 0):.2f}%, "
            f"accepted={greedy_result.accepted_count}"
        )

    def build_end_of_round_artifacts(self, state) -> EndOfRoundArtifacts:
        from backtests.swing.analysis.etf_baseline import build_tpc_optimized_full_diagnostics

        diagnostics_text = build_tpc_optimized_full_diagnostics(
            state.cumulative_mutations,
            data_dir=self.data_dir,
            initial_equity=self.initial_equity,
            start_date=self.start_date,
            end_date=self.end_date,
            title="TPC FINAL OPTIMISED CONFIG FULL DIAGNOSTICS",
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=diagnostics_text,
            dimension_reports={},
            overall_verdict="Validate on untouched holdout and stress passive fills before promotion.",
        )

    def close_pool(self) -> None:
        shutdown_process_pool(self._pool)
        self._pool = None

    def _ensure_pool(self) -> None:
        if self._pool is not None and not self._pool_dirty:
            return
        if self._pool is not None:
            shutdown_process_pool(self._pool, force=True)
        from .worker import init_worker

        self._pool = create_process_pool(
            self.max_workers,
            initializer=init_worker,
            initargs=(str(self.data_dir), self.initial_equity, self.start_date, self.end_date),
            logger=logger,
            description="TPC evaluation",
        )
        self._pool_dirty = False

    def _on_pool_terminate(self) -> None:
        self._pool_dirty = True

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

    def _infer_holdout_warmup_15m(self) -> int:
        if self._holdout_warmup_15m is not None:
            return self._holdout_warmup_15m
        bundle = self._holdout_replay_bundle()
        if not bundle.data:
            self._holdout_warmup_15m = 1
            return self._holdout_warmup_15m
        primary = max(bundle.data, key=lambda symbol: len(bundle.data[symbol]["bars_15m"].closes))
        times = pd.DatetimeIndex(bundle.data[primary]["bars_15m"].times)
        if times.tz is None:
            times = times.tz_localize("UTC")
        else:
            times = times.tz_convert("UTC")
        cutoff = pd.Timestamp(self.end_date)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        self._holdout_warmup_15m = int(np.searchsorted(times.values, cutoff.to_datetime64(), side="left"))
        return self._holdout_warmup_15m

    @staticmethod
    def _tpc_gate_criteria(metrics: dict[str, float]) -> list[GateCriterion]:
        return [
            GateCriterion("total_trades", 85.0, float(metrics.get("total_trades", 0.0)), float(metrics.get("total_trades", 0.0)) >= 85.0),
            GateCriterion("trades_per_month", 1.50, float(metrics.get("trades_per_month", 0.0)), float(metrics.get("trades_per_month", 0.0)) >= 1.50),
            GateCriterion("excellent_trades", 42.0, float(metrics.get("excellent_trades", 0.0)), float(metrics.get("excellent_trades", 0.0)) >= 42.0),
            GateCriterion("total_r", 70.0, float(metrics.get("total_r", 0.0)), float(metrics.get("total_r", 0.0)) >= 70.0),
            GateCriterion("dollar_profit_factor", 1.35, float(metrics.get("dollar_profit_factor", 0.0)), float(metrics.get("dollar_profit_factor", 0.0)) >= 1.35),
            GateCriterion("avg_r", 0.50, float(metrics.get("avg_r", 0.0)), float(metrics.get("avg_r", 0.0)) >= 0.50),
            GateCriterion("false_positive_never_worked_rate", 0.335, float(metrics.get("never_worked_rate", 1.0)), float(metrics.get("never_worked_rate", 1.0)) <= 0.335),
            GateCriterion("false_positive_low_mfe_loss_rate", 0.435, float(metrics.get("low_mfe_loss_rate", 1.0)), float(metrics.get("low_mfe_loss_rate", 1.0)) <= 0.435),
            GateCriterion("right_then_lost_rate", 0.10, float(metrics.get("right_then_lost_rate", 1.0)), float(metrics.get("right_then_lost_rate", 1.0)) <= 0.10),
            GateCriterion("max_symbol_trade_share", 0.88, float(metrics.get("max_symbol_trade_share", 1.0)), float(metrics.get("max_symbol_trade_share", 1.0)) <= 0.88),
            GateCriterion("qqq_trade_count", 16.0, float(metrics.get("qqq_trade_count", 0.0)), float(metrics.get("qqq_trade_count", 0.0)) >= 16.0),
            GateCriterion("max_dd_pct", 16.0, float(metrics.get("max_dd_pct", 0.0)), float(metrics.get("max_dd_pct", 0.0)) <= 16.0),
        ]

    @staticmethod
    def _tpc_oos_repair_gate_criteria(metrics: dict[str, float]) -> list[GateCriterion]:
        return [
            GateCriterion("train_total_trades", 100.0, float(metrics.get("total_trades", 0.0)), float(metrics.get("total_trades", 0.0)) >= 100.0),
            GateCriterion("train_net_return_pct", 55.0, float(metrics.get("net_return_pct", 0.0)), float(metrics.get("net_return_pct", 0.0)) >= 55.0),
            GateCriterion("train_dollar_profit_factor", 1.25, float(metrics.get("dollar_profit_factor", 0.0)), float(metrics.get("dollar_profit_factor", 0.0)) >= 1.25),
            GateCriterion("train_trades_per_month", 1.90, float(metrics.get("trades_per_month", 0.0)), float(metrics.get("trades_per_month", 0.0)) >= 1.90),
            GateCriterion("oos_total_trades", 10.0, float(metrics.get("oos_total_trades", 0.0)), float(metrics.get("oos_total_trades", 0.0)) >= 10.0),
            GateCriterion("oos_net_return_pct", -6.0, float(metrics.get("oos_net_return_pct", -99.0)), float(metrics.get("oos_net_return_pct", -99.0)) >= -6.0),
            GateCriterion("oos_avg_r", -0.12, float(metrics.get("oos_avg_r", -99.0)), float(metrics.get("oos_avg_r", -99.0)) >= -0.12),
            GateCriterion("oos_dollar_profit_factor", 0.70, float(metrics.get("oos_dollar_profit_factor", 0.0)), float(metrics.get("oos_dollar_profit_factor", 0.0)) >= 0.70),
            GateCriterion("oos_low_mfe_loss_rate", 0.58, float(metrics.get("oos_low_mfe_loss_rate", 1.0)), float(metrics.get("oos_low_mfe_loss_rate", 1.0)) <= 0.58),
            GateCriterion("oos_max_dd_pct", 12.0, float(metrics.get("oos_max_dd_pct", 0.0)), float(metrics.get("oos_max_dd_pct", 0.0)) <= 12.0),
        ]


def _extract_tpc_metrics(result: Any, initial_equity: float) -> dict[str, float]:
    trades = list(getattr(result, "trades", []))
    rs = np.asarray([float(getattr(t, "r_multiple", 0.0) or 0.0) for t in trades], dtype=float)
    mfes = np.asarray([float(getattr(t, "mfe_r", 0.0) or 0.0) for t in trades], dtype=float)
    pnls = np.asarray([float(getattr(t, "pnl_dollars", 0.0) or 0.0) for t in trades], dtype=float)
    campaign_ids = [str(getattr(t, "campaign_id", "") or "") for t in trades]
    additive_mask = np.asarray([_is_tpc_additive_lane(campaign_id) for campaign_id in campaign_ids], dtype=bool)
    pb30_plain_mask = np.asarray(["-pb30-" in campaign_id for campaign_id in campaign_ids], dtype=bool)
    pb30_mask = np.asarray([plain or "-pb30_ema20_value_touch-" in campaign_id for plain, campaign_id in zip(pb30_plain_mask, campaign_ids)], dtype=bool)
    pb30_ema20_mask = np.asarray(["-pb30_ema20_value_touch-" in campaign_id for campaign_id in campaign_ids], dtype=bool)
    ema20_touch_mask = np.asarray(["-ema20_value_touch-" in campaign_id for campaign_id in campaign_ids], dtype=bool)
    wins = rs[rs > 0]
    losses = rs[rs < 0]
    gross_win = float(np.sum(wins)) if wins.size else 0.0
    gross_loss = abs(float(np.sum(losses))) if losses.size else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)
    dollar_wins = pnls[pnls > 0]
    dollar_losses = pnls[pnls < 0]
    gross_dollar_win = float(np.sum(dollar_wins)) if dollar_wins.size else 0.0
    gross_dollar_loss = abs(float(np.sum(dollar_losses))) if dollar_losses.size else 0.0
    dollar_pf = (
        gross_dollar_win / gross_dollar_loss
        if gross_dollar_loss > 0
        else (gross_dollar_win if gross_dollar_win > 0 else 0.0)
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
    net_return_per_month = net_ret / months
    total_pnl_dollars = float(np.sum(pnls)) if pnls.size else 0.0
    pnl_per_trade = total_pnl_dollars / total_trades if total_trades else 0.0
    worked_mask = mfes >= 1.0
    capture = float(np.mean(np.clip(rs[worked_mask] / np.maximum(mfes[worked_mask], 1e-9), -1.0, 1.0))) if np.any(worked_mask) else 0.0
    never_worked = float(np.mean((mfes < 0.5) & (rs <= 0.0))) if total_trades else 1.0
    low_mfe_loss = float(np.mean((mfes < 1.0) & (rs <= 0.0))) if total_trades else 1.0
    half_to_one_r_loss = float(np.mean((mfes >= 0.5) & (mfes < 1.0) & (rs <= 0.0))) if total_trades else 1.0
    right_then_lost = float(np.mean((mfes >= 1.0) & (rs <= 0.0))) if total_trades else 1.0
    excellent_mask = (rs > 0.0) & (mfes >= 1.0) & (np.asarray([float(getattr(t, "mae_r", 0.0) or 0.0) for t in trades], dtype=float) <= 1.10)
    excellent_trades = int(np.sum(excellent_mask)) if total_trades else 0
    excellent_rate = float(excellent_trades / total_trades) if total_trades else 0.0
    two_r_plus_rate = float(np.mean(mfes >= 2.0)) if total_trades else 0.0
    avg_mfe = float(np.mean(mfes)) if mfes.size else 0.0
    maes = np.asarray([float(getattr(t, "mae_r", 0.0) or 0.0) for t in trades], dtype=float)
    avg_mae = float(np.mean(maes)) if maes.size else 0.0
    top5_share = 0.0
    if wins.size:
        top = np.sort(wins)[-5:]
        top5_share = float(np.sum(top) / max(np.sum(wins), 1e-9))
    dollar_top5_share = 0.0
    if dollar_wins.size:
        dollar_top = np.sort(dollar_wins)[-5:]
        dollar_top5_share = float(np.sum(dollar_top) / max(np.sum(dollar_wins), 1e-9))
    trades_per_month = total_trades / months
    avg_r = float(np.mean(rs)) if rs.size else 0.0
    r_per_month = (float(np.sum(rs)) if rs.size else 0.0) / months
    entry_requests = sum(1 for event in getattr(result, "decision_stream", []) if event.get("code") == "ENTRY_REQUESTED")
    entry_fills = sum(1 for event in getattr(result, "decision_stream", []) if event.get("code") == "ENTRY_FILLED")
    order_terminals = sum(1 for event in getattr(result, "decision_stream", []) if event.get("code") == "ORDER_TERMINAL")
    request_fill_rate = entry_fills / max(entry_requests, 1)
    order_terminal_rate = order_terminals / max(entry_requests, 1)
    symbol_avgs: list[float] = []
    symbol_pnls: list[float] = []
    symbol_counts = {
        symbol: sum(1 for t in trades if str(getattr(t, "symbol", "") or "") == symbol)
        for symbol in sorted({str(getattr(t, "symbol", "") or "") for t in trades})
    }
    qqq_excellent_trades = float(
        sum(
            1
            for idx, trade in enumerate(trades)
            if str(getattr(trade, "symbol", "") or "") == "QQQ" and bool(excellent_mask[idx])
        )
    )
    gld_excellent_trades = float(
        sum(
            1
            for idx, trade in enumerate(trades)
            if str(getattr(trade, "symbol", "") or "") == "GLD" and bool(excellent_mask[idx])
        )
    )
    max_symbol_trade_share = max(symbol_counts.values(), default=0) / max(total_trades, 1)
    gld_trade_count = float(symbol_counts.get("GLD", 0))
    qqq_trade_count = float(symbol_counts.get("QQQ", 0))
    gld_trade_share = gld_trade_count / max(total_trades, 1)
    qqq_trade_share = qqq_trade_count / max(total_trades, 1)
    qqq_excellent_rate = qqq_excellent_trades / max(qqq_trade_count, 1.0)
    gld_excellent_rate = gld_excellent_trades / max(gld_trade_count, 1.0)
    for symbol in sorted({str(getattr(t, "symbol", "") or "") for t in trades}):
        vals = [float(getattr(t, "r_multiple", 0.0) or 0.0) for t in trades if str(getattr(t, "symbol", "") or "") == symbol]
        if vals:
            symbol_avgs.append(float(np.mean(vals)))
        dollar_vals = [float(getattr(t, "pnl_dollars", 0.0) or 0.0) for t in trades if str(getattr(t, "symbol", "") or "") == symbol]
        if dollar_vals:
            symbol_pnls.append(float(np.sum(dollar_vals)))
    year_avgs: list[float] = []
    year_pnls: list[float] = []
    for year in sorted({pd.Timestamp(getattr(t, "entry_time")).year for t in trades if getattr(t, "entry_time", None) is not None}):
        vals = [
            float(getattr(t, "r_multiple", 0.0) or 0.0)
            for t in trades
            if getattr(t, "entry_time", None) is not None and pd.Timestamp(getattr(t, "entry_time")).year == year
        ]
        if vals:
            year_avgs.append(float(np.mean(vals)))
        dollar_vals = [
            float(getattr(t, "pnl_dollars", 0.0) or 0.0)
            for t in trades
            if getattr(t, "entry_time", None) is not None and pd.Timestamp(getattr(t, "entry_time")).year == year
        ]
        if dollar_vals:
            year_pnls.append(float(np.sum(dollar_vals)))
    return {
        "total_trades": float(total_trades),
        "net_return_pct": net_ret,
        "net_return_per_month": net_return_per_month,
        "total_pnl_dollars": total_pnl_dollars,
        "pnl_per_trade": pnl_per_trade,
        "profit_factor": pf,
        "dollar_profit_factor": dollar_pf,
        "avg_r": avg_r,
        "total_r": float(np.sum(rs)) if rs.size else 0.0,
        "r_per_month": r_per_month,
        "win_rate": float(np.mean(rs > 0)) if rs.size else 0.0,
        "max_dd_pct": max_dd,
        "sharpe": sharpe,
        "trades_per_month": trades_per_month,
        "expectancy_frequency": avg_r * np.sqrt(max(trades_per_month, 0.0)),
        "mfe_capture": capture,
        "avg_mfe_r": avg_mfe,
        "avg_mae_r": avg_mae,
        "excellent_trades": float(excellent_trades),
        "excellent_trades_per_month": excellent_trades / months,
        "excellent_rate": excellent_rate,
        "two_r_plus_rate": two_r_plus_rate,
        "never_worked_rate": never_worked,
        "false_positive_never_worked_rate": never_worked,
        "low_mfe_loss_rate": low_mfe_loss,
        "false_positive_low_mfe_loss_rate": low_mfe_loss,
        "half_to_one_r_loss_rate": half_to_one_r_loss,
        "right_then_lost_rate": right_then_lost,
        "top5_winner_share": top5_share,
        "dollar_top5_winner_share": dollar_top5_share,
        "max_symbol_trade_share": max_symbol_trade_share,
        "dominant_symbol_trade_share": max_symbol_trade_share,
        "gld_trade_count": gld_trade_count,
        "qqq_trade_count": qqq_trade_count,
        "gld_excellent_trades": gld_excellent_trades,
        "qqq_excellent_trades": qqq_excellent_trades,
        "gld_excellent_rate": gld_excellent_rate,
        "qqq_excellent_rate": qqq_excellent_rate,
        "gld_trade_share": gld_trade_share,
        "max_gld_trade_share": gld_trade_share,
        "qqq_trade_share": qqq_trade_share,
        "entry_requests": float(entry_requests),
        "request_fill_rate": request_fill_rate,
        "order_terminal_rate": order_terminal_rate,
        "additive_trade_count": float(np.sum(additive_mask)) if total_trades else 0.0,
        "additive_avg_r": float(np.mean(rs[additive_mask])) if np.any(additive_mask) else 0.0,
        "additive_total_r": float(np.sum(rs[additive_mask])) if np.any(additive_mask) else 0.0,
        "additive_pnl_pct": (float(np.sum(pnls[additive_mask])) / initial_equity * 100.0) if np.any(additive_mask) else 0.0,
        "additive_low_mfe_loss_rate": float(np.mean((mfes[additive_mask] < 1.0) & (rs[additive_mask] <= 0.0))) if np.any(additive_mask) else 0.0,
        "pb30_trade_count": float(np.sum(pb30_mask)) if total_trades else 0.0,
        "pb30_avg_r": float(np.mean(rs[pb30_mask])) if np.any(pb30_mask) else 0.0,
        "pb30_total_r": float(np.sum(rs[pb30_mask])) if np.any(pb30_mask) else 0.0,
        "pb30_plain_trade_count": float(np.sum(pb30_plain_mask)) if total_trades else 0.0,
        "pb30_plain_avg_r": float(np.mean(rs[pb30_plain_mask])) if np.any(pb30_plain_mask) else 0.0,
        "pb30_plain_total_r": float(np.sum(rs[pb30_plain_mask])) if np.any(pb30_plain_mask) else 0.0,
        "pb30_plain_low_mfe_loss_rate": float(np.mean((mfes[pb30_plain_mask] < 1.0) & (rs[pb30_plain_mask] <= 0.0))) if np.any(pb30_plain_mask) else 0.0,
        "pb30_ema20_trade_count": float(np.sum(pb30_ema20_mask)) if total_trades else 0.0,
        "pb30_ema20_avg_r": float(np.mean(rs[pb30_ema20_mask])) if np.any(pb30_ema20_mask) else 0.0,
        "pb30_ema20_total_r": float(np.sum(rs[pb30_ema20_mask])) if np.any(pb30_ema20_mask) else 0.0,
        "pb30_ema20_low_mfe_loss_rate": float(np.mean((mfes[pb30_ema20_mask] < 1.0) & (rs[pb30_ema20_mask] <= 0.0))) if np.any(pb30_ema20_mask) else 0.0,
        "ema20_touch_trade_count": float(np.sum(ema20_touch_mask)) if total_trades else 0.0,
        "ema20_touch_avg_r": float(np.mean(rs[ema20_touch_mask])) if np.any(ema20_touch_mask) else 0.0,
        "ema20_touch_total_r": float(np.sum(rs[ema20_touch_mask])) if np.any(ema20_touch_mask) else 0.0,
        "ema20_touch_low_mfe_loss_rate": float(np.mean((mfes[ema20_touch_mask] < 1.0) & (rs[ema20_touch_mask] <= 0.0))) if np.any(ema20_touch_mask) else 0.0,
        "worst_symbol_avg_r": min(symbol_avgs) if symbol_avgs else 0.0,
        "worst_year_avg_r": min(year_avgs) if year_avgs else 0.0,
        "worst_symbol_pnl": min(symbol_pnls) if symbol_pnls else 0.0,
        "worst_symbol_pnl_pct": (min(symbol_pnls) / initial_equity * 100.0) if symbol_pnls else 0.0,
        "worst_year_pnl": min(year_pnls) if year_pnls else 0.0,
        "worst_year_pnl_pct": (min(year_pnls) / initial_equity * 100.0) if year_pnls else 0.0,
    }


def _is_tpc_additive_lane(campaign_id: str) -> bool:
    return any(
        marker in campaign_id
        for marker in (
            "-pb30-",
            "-pb30_ema20_value_touch-",
            "-ema20_value_touch-",
        )
    )


def _tpc_composite_score(
    metrics: dict[str, float],
    hard_rejects: dict[str, float] | None,
    weights: dict[str, float],
) -> TPCScore:
    weights = _validate_tpc_score_weights(weights)
    rejects = hard_rejects or {}
    checks = [
        (metrics.get("total_trades", 0.0) < rejects.get("min_valid_trades", 1.0), "min_valid_trades"),
        (metrics.get("trades_per_month", 0.0) < rejects.get("min_trades_per_month", 0.0), "min_trades_per_month"),
        (metrics.get("max_dd_pct", 0.0) > rejects.get("max_dd_pct", 99.0), "max_dd_pct"),
        (metrics.get("net_return_pct", 0.0) < rejects.get("min_return_pct", -99.0), "min_return_pct"),
        (metrics.get("total_r", 0.0) < rejects.get("min_total_r", -99.0), "min_total_r"),
        (metrics.get("avg_r", 0.0) < rejects.get("min_avg_r", -99.0), "min_avg_r"),
        (metrics.get("profit_factor", 0.0) < rejects.get("min_profit_factor", 0.0), "min_profit_factor"),
        (metrics.get("dollar_profit_factor", 0.0) < rejects.get("min_dollar_profit_factor", 0.0), "min_dollar_profit_factor"),
        (metrics.get("excellent_trades", 0.0) < rejects.get("min_excellent_trades", 0.0), "min_excellent_trades"),
        (metrics.get("never_worked_rate", 0.0) > rejects.get("max_never_worked_rate", 1.0), "max_never_worked_rate"),
        (metrics.get("low_mfe_loss_rate", 0.0) > rejects.get("max_low_mfe_loss_rate", 1.0), "max_low_mfe_loss_rate"),
        (metrics.get("right_then_lost_rate", 0.0) > rejects.get("max_right_then_lost_rate", 1.0), "max_right_then_lost_rate"),
        (metrics.get("top5_winner_share", 0.0) > rejects.get("max_top5_winner_share", 1.0), "max_top5_winner_share"),
        (metrics.get("dollar_top5_winner_share", 0.0) > rejects.get("max_dollar_top5_winner_share", 1.0), "max_dollar_top5_winner_share"),
        (metrics.get("max_symbol_trade_share", 0.0) > rejects.get("max_symbol_trade_share", 1.0), "max_symbol_trade_share"),
        (metrics.get("gld_trade_share", 0.0) > rejects.get("max_gld_trade_share", 1.0), "max_gld_trade_share"),
        (metrics.get("qqq_trade_count", 0.0) < rejects.get("min_qqq_trades", 0.0), "min_qqq_trades"),
        (metrics.get("worst_symbol_avg_r", 0.0) < rejects.get("min_worst_symbol_avg_r", -99.0), "min_worst_symbol_avg_r"),
        (metrics.get("worst_symbol_pnl_pct", 0.0) < rejects.get("min_worst_symbol_pnl_pct", -99.0), "min_worst_symbol_pnl_pct"),
        (metrics.get("worst_year_pnl_pct", 0.0) < rejects.get("min_worst_year_pnl_pct", -99.0), "min_worst_year_pnl_pct"),
    ]
    for failed, reason in checks:
        if failed:
            return TPCScore(0.0, True, reason)

    max_dd = metrics.get("max_dd_pct", 0.0)
    false_positive_control = (
        0.45 * (1.0 - _scale(metrics.get("never_worked_rate", 1.0), 0.24, 0.34))
        + 0.30 * (1.0 - _scale(metrics.get("low_mfe_loss_rate", 1.0), 0.34, 0.44))
        + 0.15 * (1.0 - _scale(metrics.get("right_then_lost_rate", 1.0), 0.04, 0.10))
        + 0.10 * _scale(metrics.get("excellent_rate", 0.0), 0.48, 0.60)
    )
    symbol_balance = (
        0.30 * (1.0 - _scale(metrics.get("max_symbol_trade_share", 1.0), 0.70, 0.86))
        + 0.22 * _scale(metrics.get("qqq_trade_count", 0.0), 18.0, 35.0)
        + 0.22 * _scale(metrics.get("qqq_excellent_trades", 0.0), 9.0, 18.0)
        + 0.14 * _scale(metrics.get("qqq_excellent_rate", 0.0), 0.45, 0.75)
        + 0.12 * (1.0 - _scale(metrics.get("dollar_top5_winner_share", 0.0), 0.28, 0.42))
    )
    robustness = (
        0.28 * (1.0 - _scale(metrics.get("dollar_top5_winner_share", 0.0), 0.28, 0.42))
        + 0.22 * _scale(metrics.get("worst_symbol_pnl_pct", 0.0), -5.00, 55.00)
        + 0.18 * _scale(metrics.get("worst_symbol_avg_r", 0.0), -0.10, 1.00)
        + 0.20 * _scale(metrics.get("worst_year_pnl_pct", 0.0), -12.00, 8.00)
        + 0.12 * (1.0 - _scale(max_dd, 10.00, 17.00))
    )
    components = {
        "false_positive_control": max(0.0, min(false_positive_control, 1.0)),
        "symbol_balance": max(0.0, min(symbol_balance, 1.0)),
        "alpha_quality": (
            0.25 * _scale(metrics.get("avg_r", 0.0), 0.45, 0.90)
            + 0.25 * _scale(metrics.get("dollar_profit_factor", 0.0), 1.35, 2.20)
            + 0.20 * _scale(metrics.get("net_return_pct", 0.0), 70.0, 145.0)
            + 0.15 * _scale(metrics.get("pnl_per_trade", 0.0), 650.0, 1150.0)
            + 0.15 * _scale(metrics.get("mfe_capture", 0.0), 0.28, 0.45)
        ),
        "frequency_floor": (
            0.40 * _scale(metrics.get("trades_per_month", 0.0), 1.50, 2.70)
            + 0.30 * _scale(metrics.get("total_trades", 0.0), 85.0, 140.0)
            + 0.20 * _scale(metrics.get("excellent_trades_per_month", 0.0), 0.75, 1.30)
            + 0.10 * _scale(metrics.get("two_r_plus_rate", 0.0), 0.35, 0.50)
        ),
        "expected_r": (
            0.35 * _scale(metrics.get("net_return_pct", 0.0), 70.0, 145.0)
            + 0.25 * _scale(metrics.get("total_r", 0.0), 65.0, 110.0)
            + 0.20 * _scale(metrics.get("net_return_per_month", 0.0), 1.25, 2.50)
            + 0.20 * _scale(metrics.get("two_r_plus_rate", 0.0), 0.35, 0.50)
        ),
        "risk_quality": (
            0.35 * _scale(metrics.get("dollar_profit_factor", 0.0), 1.35, 2.05)
            + 0.30 * max(0.0, min(robustness, 1.0))
            + 0.20 * _scale(metrics.get("sharpe", 0.0), 0.35, 1.00)
            + 0.15 * (1.0 - _scale(max_dd, 10.00, 17.00))
        ),
    }
    clean_weights = _normalise_weights(weights)
    score = sum(clean_weights.get(name, 0.0) * components.get(name, 0.0) for name in clean_weights)
    return TPCScore(max(0.0, score * 100.0))


def _with_oos_metrics(train_metrics: dict[str, float], oos_metrics: dict[str, float]) -> dict[str, float]:
    metrics = dict(train_metrics)
    for key, value in oos_metrics.items():
        metrics[f"oos_{key}"] = value
    return metrics


def _tpc_oos_repair_score(
    metrics: dict[str, float],
    hard_rejects: dict[str, float] | None,
    weights: dict[str, float],
) -> TPCScore:
    weights = _validate_tpc_score_weights(weights)
    rejects = hard_rejects or {}
    checks = [
        (metrics.get("total_trades", 0.0) < rejects.get("min_valid_trades", 1.0), "min_valid_trades"),
        (metrics.get("trades_per_month", 0.0) < rejects.get("min_trades_per_month", 0.0), "min_trades_per_month"),
        (metrics.get("net_return_pct", 0.0) < rejects.get("min_return_pct", -99.0), "min_return_pct"),
        (metrics.get("dollar_profit_factor", 0.0) < rejects.get("min_dollar_profit_factor", 0.0), "min_dollar_profit_factor"),
        (metrics.get("max_dd_pct", 0.0) > rejects.get("max_dd_pct", 99.0), "max_dd_pct"),
        (metrics.get("oos_total_trades", 0.0) < rejects.get("min_oos_valid_trades", 1.0), "min_oos_valid_trades"),
        (metrics.get("oos_net_return_pct", 0.0) < rejects.get("min_oos_return_pct", -99.0), "min_oos_return_pct"),
        (metrics.get("oos_avg_r", 0.0) < rejects.get("min_oos_avg_r", -99.0), "min_oos_avg_r"),
        (
            metrics.get("oos_dollar_profit_factor", 0.0) < rejects.get("min_oos_dollar_profit_factor", 0.0),
            "min_oos_dollar_profit_factor",
        ),
        (metrics.get("oos_max_dd_pct", 0.0) > rejects.get("max_oos_dd_pct", 99.0), "max_oos_dd_pct"),
        (
            metrics.get("oos_low_mfe_loss_rate", 0.0) > rejects.get("max_oos_low_mfe_loss_rate", 1.0),
            "max_oos_low_mfe_loss_rate",
        ),
    ]
    for failed, reason in checks:
        if failed:
            return TPCScore(0.0, True, reason)

    train_dd = metrics.get("max_dd_pct", 0.0)
    oos_dd = metrics.get("oos_max_dd_pct", 0.0)
    components = {
        "false_positive_control": (
            0.40 * (1.0 - _scale(metrics.get("oos_low_mfe_loss_rate", 1.0), 0.48, 0.58))
            + 0.25 * (1.0 - _scale(metrics.get("oos_never_worked_rate", 1.0), 0.40, 0.58))
            + 0.20 * (1.0 - _scale(metrics.get("low_mfe_loss_rate", 1.0), 0.36, 0.50))
            + 0.15 * (1.0 - _scale(metrics.get("oos_right_then_lost_rate", 1.0), 0.05, 0.20))
        ),
        "symbol_balance": (
            0.35 * (1.0 - _scale(metrics.get("oos_max_symbol_trade_share", 1.0), 0.65, 0.90))
            + 0.20 * (1.0 - _scale(metrics.get("max_symbol_trade_share", 1.0), 0.70, 0.88))
            + 0.18 * _scale(metrics.get("qqq_trade_count", 0.0), 16.0, 35.0)
            + 0.17 * _scale(metrics.get("qqq_excellent_trades", 0.0), 9.0, 18.0)
            + 0.10 * _scale(metrics.get("oos_qqq_trade_count", 0.0), 2.0, 8.0)
        ),
        "alpha_quality": (
            0.45 * _scale(metrics.get("oos_avg_r", 0.0), -0.12, 0.20)
            + 0.25 * _scale(metrics.get("oos_dollar_profit_factor", 0.0), 0.70, 1.10)
            + 0.20 * _scale(metrics.get("avg_r", 0.0), 0.10, 2.20)
            + 0.10 * _scale(metrics.get("oos_avg_mfe_r", 0.0), 1.00, 2.20)
        ),
        "frequency_floor": (
            0.45 * _scale(metrics.get("trades_per_month", 0.0), 1.90, 3.00)
            + 0.35 * _scale(metrics.get("oos_trades_per_month", 0.0), 2.00, 3.50)
            + 0.20 * _scale(metrics.get("total_trades", 0.0), 100.0, 160.0)
        ),
        "expected_r": (
            0.45 * _scale(metrics.get("total_r", 0.0), 175.0, 310.0)
            + 0.35 * _scale(metrics.get("r_per_month", 0.0), 3.20, 5.80)
            + 0.20 * _scale(metrics.get("oos_total_r", 0.0), -2.0, 8.0)
        ),
        "risk_quality": (
            0.30 * (1.0 - _scale(oos_dd, 7.0, 12.0))
            + 0.25 * (1.0 - _scale(train_dd, 12.0, 24.0))
            + 0.25 * _scale(metrics.get("net_return_pct", 0.0), 55.0, 150.0)
            + 0.20 * _scale(metrics.get("worst_symbol_avg_r", 0.0), -0.10, 1.00)
        ),
    }
    clean_weights = _normalise_weights(weights)
    score = sum(clean_weights.get(name, 0.0) * max(0.0, min(components.get(name, 0.0), 1.0)) for name in clean_weights)
    return TPCScore(max(0.0, score * 100.0))


def _scale(value: float, low: float, high: float) -> float:
    if high <= low or not np.isfinite(value):
        return 0.0
    return (min(max(float(value), low), high) - low) / (high - low)


def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {name: max(float(value), 0.0) for name, value in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        return dict(TPC_SCORING_WEIGHTS)
    return {name: value / total for name, value in clean.items()}


def _validate_tpc_score_weights(weights: dict[str, float]) -> dict[str, float]:
    components = frozenset(weights)
    if components != TPC_SCORE_COMPONENTS:
        missing = sorted(TPC_SCORE_COMPONENTS - components)
        extra = sorted(components - TPC_SCORE_COMPONENTS)
        raise ValueError(
            "TPC score weights must be strategy-bespoke and match exactly "
            f"{sorted(TPC_SCORE_COMPONENTS)}; missing={missing}, extra={extra}"
        )
    if len(weights) > 7:
        raise ValueError("TPC score must not contain more than 7 components.")
    return dict(weights)


def _tpc_diagnostic_gaps(_phase: int, metrics: dict[str, float]) -> list[str]:
    gaps: list[str] = []
    if metrics.get("never_worked_rate", 1.0) > 0.28 or metrics.get("low_mfe_loss_rate", 1.0) > 0.38:
        gaps.append("Core discrimination is still weak; prefer confirmation, value-hit, pullback-depth, room, and regime-quality filters over extra supply.")
    if metrics.get("max_symbol_trade_share", 1.0) > 0.80 or metrics.get("gld_trade_share", 1.0) > 0.80:
        gaps.append("GLD still dominates the sample; require GLD-specific proof and protected QQQ excellent-trade supply rather than accepting GLD-only gains.")
    if metrics.get("qqq_excellent_trades", 0.0) < 12:
        gaps.append("QQQ excellent-trade supply is thin; broaden QQQ only through score, room, confirmation, or source-quality protected paths.")
    if metrics.get("total_r", 0.0) < 75.0 or metrics.get("avg_r", 0.0) < 0.58:
        gaps.append("Discrimination filters are cutting too deeply or leaving expectancy thin; keep only filters that preserve total R and per-trade R.")
    if metrics.get("dollar_top5_winner_share", 0.0) > 0.38:
        gaps.append("Winner concentration is high; avoid letting one compounding tail event dominate the score.")
    if metrics.get("top5_winner_share", 0.0) > 0.45:
        gaps.append("R-multiple winner concentration is high; verify the score is not being improved by a smaller stop denominator rather than better trade economics.")
    if metrics.get("trades_per_month", 0.0) < 1.70:
        gaps.append("Frequency is below the useful optimization floor; reject filters that merely solve false positives by starving the strategy.")
    return gaps


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
