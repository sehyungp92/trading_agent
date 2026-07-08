from __future__ import annotations

from pathlib import Path
from typing import Any

from backtests.scalp.analysis.po3_reversal_diagnostics import po3_reversal_diagnostics
from backtests.scalp.config_po3_reversal import Po3ReversalBacktestConfig
from backtests.scalp.engine.po3_reversal_engine import load_po3_reversal_data, run_po3_reversal_backtest
from backtests.shared.auto.cache_keys import build_cache_key, fingerprint_tree
from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.plugin_utils import CachedBatchEvaluator, mutation_signature
from backtests.shared.auto.replay_bundle import ReplayBundle
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion

from .phase_candidates import BASE_MUTATIONS, PHASE_FOCUS, get_phase_candidates
from .scoring import PHASE_HARD_REJECTS, PHASE_WEIGHTS
from .worker import score_candidate


class _SequentialBatchEvaluator:
    def __init__(self, phase: int, scoring_weights, hard_rejects) -> None:
        self.phase = phase
        self.scoring_weights = scoring_weights
        self.hard_rejects = hard_rejects

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        return [
            score_candidate((candidate.name, candidate.mutations, current_mutations, self.phase, self.scoring_weights, self.hard_rejects))
            for candidate in candidates
        ]

    def close(self) -> None:
        return None


class Po3ReversalPlugin:
    name = "po3_reversal"
    num_phases = 4
    initial_mutations = dict(BASE_MUTATIONS)
    ultimate_targets = {"profit_factor": 1.3, "expectancy_dollar": 0.0, "trades_per_month": 4.0, "max_drawdown_pct": 0.15}

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 1,
        *,
        symbols=None,
        analysis_symbol: str = "NQ",
        trade_symbol: str = "MNQ",
        confirmation_symbol: str = "ES",
        num_phases: int = 4,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.analysis_symbol = analysis_symbol.upper()
        self.trade_symbol = trade_symbol.upper()
        self.confirmation_symbol = confirmation_symbol.upper()
        self.symbols = (self.trade_symbol,)
        self.num_phases = num_phases
        self._bundle: ReplayBundle | None = None
        self._last_fingerprint = ""
        self._evaluation_cache: dict[str, Any] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._context_cache: dict[str, Any] = {}
        self._last_context: dict[str, Any] = {}

    def _replay_bundle(self) -> ReplayBundle:
        data_fp = fingerprint_tree(self.data_dir, patterns=("*.parquet", "*.csv"))
        repo_root = Path(__file__).resolve().parents[4]
        source_fp = ":".join(
            [
                fingerprint_tree(repo_root / "strategies" / "scalp" / "po3_reversal", patterns=("*.py",)),
                fingerprint_tree(repo_root / "strategies" / "scalp" / "_shared", patterns=("*.py",)),
                fingerprint_tree(repo_root / "backtests" / "scalp", patterns=("*.py",)),
            ]
        )
        fingerprint = f"{data_fp}:{source_fp}"
        if self._bundle is None or fingerprint != self._last_fingerprint:
            self._evaluation_cache.clear()
            self._metrics_cache.clear()
            self._context_cache.clear()
            self._last_context = {}
            cfg = Po3ReversalBacktestConfig(
                analysis_symbol=self.analysis_symbol,
                trade_symbol=self.trade_symbol,
                symbols=list(self.symbols),
                confirmation_symbol=self.confirmation_symbol,
                data_dir=self.data_dir,
                initial_equity=self.initial_equity,
            )
            self._bundle = ReplayBundle(load_po3_reversal_data(cfg), str(self.data_dir), fingerprint)
            self._last_fingerprint = fingerprint
        return self._bundle

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        candidates = [Experiment(name, muts) for name, muts in get_phase_candidates(phase, state.cumulative_mutations)]
        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics: self._gate_criteria(phase, metrics),
            scoring_weights=PHASE_WEIGHTS.get(phase),
            hard_rejects=PHASE_HARD_REJECTS.get(phase, {}),
            analysis_policy=PhaseAnalysisPolicy(focus_metrics=focus_metrics),
            max_rounds=12,
        )

    def create_evaluate_batch(self, phase: int, cumulative_mutations: dict[str, Any], *, scoring_weights=None, hard_rejects=None):
        del cumulative_mutations
        bundle = self._replay_bundle()
        from .worker import init_worker

        init_worker(str(self.data_dir), self.initial_equity, self.analysis_symbol, self.trade_symbol, self.confirmation_symbol)
        signature_prefix = build_cache_key(
            "scalp.po3_reversal.evaluation",
            source_fingerprint=bundle.cache_source_fingerprint,
            extra={
                "phase": phase,
                "analysis_symbol": self.analysis_symbol,
                "trade_symbol": self.trade_symbol,
                "confirmation_symbol": self.confirmation_symbol,
                "symbols": list(self.symbols),
                "scoring_weights": scoring_weights or {},
                "hard_rejects": hard_rejects or {},
            },
        )
        return CachedBatchEvaluator(
            _SequentialBatchEvaluator(phase, scoring_weights, hard_rejects),
            cache=self._evaluation_cache,
            signature_prefix=signature_prefix,
            metrics_cache=self._metrics_cache,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        return dict(self._run_config(mutations)["metrics"])

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        del state
        return f"PO3 phase {phase}: score {greedy_result.base_score:.4f}->{greedy_result.final_score:.4f}, trades={metrics.get('total_trades', 0)}"

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        del phase, state, metrics
        context = self._run_config(greedy_result.final_mutations)
        return po3_reversal_diagnostics(context["trades"], context["metrics"])

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        context = self._run_config(state.cumulative_mutations)
        return EndOfRoundArtifacts(
            final_diagnostics_text=po3_reversal_diagnostics(context["trades"], context["metrics"]),
            dimension_reports={"po3_reversal": f"Trades={len(context['trades'])}, metrics={context['metrics']}"},
            overall_verdict=f"PO3 reversal finished with net ${context['metrics'].get('net_profit', 0.0):.2f}.",
        )

    def _run_config(self, mutations: dict[str, Any]) -> dict[str, Any]:
        key = mutation_signature(mutations)
        if key in self._context_cache:
            return self._context_cache[key]
        bundle = self._replay_bundle()
        cfg = Po3ReversalBacktestConfig(
            analysis_symbol=self.analysis_symbol,
            trade_symbol=self.trade_symbol,
            symbols=list(self.symbols),
            confirmation_symbol=self.confirmation_symbol,
            data_dir=self.data_dir,
            initial_equity=self.initial_equity,
        )
        for mut_key, value in mutations.items():
            if mut_key.startswith("flags."):
                setattr(cfg.flags, mut_key.split(".", 1)[1], bool(value))
            elif mut_key.startswith("param_overrides."):
                cfg.param_overrides[mut_key.split(".", 1)[1]] = value
        result = run_po3_reversal_backtest(bundle.data, cfg)
        trades = [trade for symbol_result in result.symbol_results.values() for trade in symbol_result.trades]
        context = {"result": result, "trades": trades, "metrics": dict(result.metrics)}
        self._context_cache[key] = context
        self._metrics_cache[key] = dict(result.metrics)
        self._last_context = context
        return context

    def _gate_criteria(self, phase: int, metrics: dict[str, float]) -> list[GateCriterion]:
        rejects = PHASE_HARD_REJECTS.get(phase, {})
        return [
            GateCriterion("min_trades", rejects.get("min_trades", 0), metrics.get("total_trades", 0), metrics.get("total_trades", 0) >= rejects.get("min_trades", 0)),
            GateCriterion("min_pf", rejects.get("min_pf", 0), metrics.get("profit_factor", 0), metrics.get("profit_factor", 0) >= rejects.get("min_pf", 0)),
            GateCriterion("max_dd_pct", rejects.get("max_dd_pct", 1), metrics.get("max_drawdown_pct", 0), metrics.get("max_drawdown_pct", 0) <= rejects.get("max_dd_pct", 1)),
        ]
