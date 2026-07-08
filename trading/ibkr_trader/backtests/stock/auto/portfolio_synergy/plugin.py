from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from backtests.shared.auto.cache_keys import build_cache_key
from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.provenance import AutoRunProvenance, build_phase_auto_provenance
from backtests.shared.auto.plugin_utils import CachedBatchEvaluator, mutation_signature
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion, ScoredCandidate

from .evaluator import evaluate_portfolio, load_evaluation_bundle, _latest_optimized_config_path, _load_stock_price_bars
from .phase_candidates import (
    DEFAULT_PROFILE,
    ROUND_NAME,
    SEED_PORTFOLIO_CONFIG,
    STRATEGY_ORDER,
    get_phase_candidates,
    get_phase_focus,
    get_phase_gates,
    get_round_targets,
    get_score_weights,
)
from .scoring import SCORE_COMPONENTS, score_portfolio_metrics

_DEFAULT_ROUND_TARGETS = get_round_targets(DEFAULT_PROFILE)

FOCUS_METRICS: dict[int, list[str]] = {
    1: ["active_strategy_count", "active_trades_per_month", "max_drawdown_pct"],
    2: ["active_trades_per_month", "total_r_per_month", "trade_capture_ratio"],
    3: ["net_return_pct", "max_strategy_risk_share", "max_drawdown_pct"],
    4: ["trade_capture_ratio", "positive_alpha_block_rate", "profit_factor"],
    5: ["candidate_discrimination", "positive_alpha_block_rate", "trade_capture_ratio"],
    6: ["max_drawdown_pct", "max_daily_loss_R", "max_weekly_loss_R"],
    7: ["net_return_pct", "active_trades_per_month", "positive_slices"],
}


def _latest_source_artifact(repo_root: Path, strategy: str) -> Path:
    strategy_dir = "alcb" if "ALCB" in strategy.upper() else "iaric"
    try:
        return _latest_optimized_config_path(repo_root, strategy_dir)
    except FileNotFoundError:
        return repo_root / "backtests" / "output" / "stock" / strategy_dir / "__missing_optimized_config__.json"


class _PortfolioBatchEvaluator:
    def __init__(
        self,
        plugin: "StockPortfolioSynergyPlugin",
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
    ):
        self._plugin = plugin
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        if self._plugin.max_workers <= 1 or len(candidates) <= 1:
            return [self._score_candidate(candidate, current_mutations) for candidate in candidates]

        scored: list[ScoredCandidate | None] = [None] * len(candidates)
        with ThreadPoolExecutor(max_workers=self._plugin.max_workers) as executor:
            futures = {
                executor.submit(self._score_candidate, candidate, current_mutations): idx
                for idx, candidate in enumerate(candidates)
            }
            for future in as_completed(futures):
                scored[futures[future]] = future.result()
        return [item for item in scored if item is not None]

    def _score_candidate(
        self,
        candidate: Experiment,
        current_mutations: dict[str, Any],
    ) -> ScoredCandidate:
        merged = dict(current_mutations or {})
        merged.update(candidate.mutations or {})
        try:
            metrics = self._plugin._compute_metrics_raw(merged)
            score = score_portfolio_metrics(
                metrics,
                scoring_weights=self._scoring_weights,
                hard_rejects=self._hard_rejects,
            )
            return ScoredCandidate(
                name=candidate.name,
                score=score.total,
                rejected=score.rejected,
                reject_reason=score.reject_reason,
                metrics=_with_score_metrics(metrics, score),
            )
        except Exception as exc:
            return ScoredCandidate(
                name=candidate.name,
                score=0.0,
                rejected=True,
                reject_reason=str(exc),
                metrics={},
            )

    def close(self) -> None:
        return None


class StockPortfolioSynergyPlugin:
    name = "stock_portfolio_synergy"
    num_phases = 7
    ultimate_targets = {
        "active_strategy_count": _DEFAULT_ROUND_TARGETS["min_active_strategies"],
        "active_trades_per_month": _DEFAULT_ROUND_TARGETS["min_active_trades_per_month"],
        "total_r_per_month": _DEFAULT_ROUND_TARGETS["min_total_r_per_month"],
        "profit_factor": _DEFAULT_ROUND_TARGETS["min_profit_factor"],
        "max_drawdown_pct": _DEFAULT_ROUND_TARGETS["target_max_drawdown_pct"],
        "trade_capture_ratio": _DEFAULT_ROUND_TARGETS["min_trade_capture_ratio"],
        "positive_slices": 4.0,
    }

    def __init__(
        self,
        data_dir: Path,
        *,
        start_date: str = "2024-01-01",
        end_date: str = "2026-03-01",
        initial_equity: float = 25_000.0,
        max_workers: int | None = 1,
        round_profile: str = DEFAULT_PROFILE,
    ):
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.initial_equity = float(initial_equity)
        self.max_workers = 1 if max_workers is None else max(1, min(2, int(max_workers)))
        self.round_profile = round_profile
        self.round_targets = get_round_targets(round_profile)
        self.score_weights = get_score_weights(round_profile)
        self.ultimate_targets = {
            "active_strategy_count": self.round_targets["min_active_strategies"],
            "active_trades_per_month": self.round_targets["min_active_trades_per_month"],
            "total_r_per_month": self.round_targets["min_total_r_per_month"],
            "profit_factor": self.round_targets["min_profit_factor"],
            "max_drawdown_pct": self.round_targets["target_max_drawdown_pct"],
            "trade_capture_ratio": self.round_targets["min_trade_capture_ratio"],
            "positive_slices": 4.0,
        }
        self.initial_mutations: dict[str, Any] | None = dict(SEED_PORTFOLIO_CONFIG)
        self.diagnostic_round_label = ROUND_NAME
        self._evaluation_data = None
        self._cached_bundle = None
        self._cache_source_fingerprint = ""
        self._evaluation_cache: dict[str, ScoredCandidate] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._price_bars_cache: dict[str, Any] | None = None
        self._metrics_lock = Lock()
        self._provenance: AutoRunProvenance | None = None

    def build_provenance(self) -> AutoRunProvenance:
        if self._provenance is None:
            repo_root = Path(__file__).resolve().parents[4]
            source_artifacts = {
                strategy: _latest_source_artifact(repo_root, strategy)
                for strategy in STRATEGY_ORDER
            }
            self._provenance = build_phase_auto_provenance(
                self.name,
                repo_root=repo_root,
                code_dirs=(Path(__file__).resolve().parent,),
                code_paths=(
                    repo_root / "backtests/stock/auto/portfolio_synergy/core/logic.py",
                    repo_root / "backtests/stock/auto/portfolio_synergy/core/state.py",
                    repo_root / "backtests/stock/auto/portfolio_synergy/evaluator.py",
                    repo_root / "libs/oms/risk/portfolio_rules.py",
                ),
                data_dir=self.data_dir,
                source_artifacts=source_artifacts,
                selection_context={
                    "round_profile": self.round_profile,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "initial_equity": self.initial_equity,
                    "score_components": SCORE_COMPONENTS,
                    "score_weights": self.score_weights,
                    "round_targets": self.round_targets,
                    "phase_gates": {
                        phase: get_phase_gates(phase, profile=self.round_profile)
                        for phase in range(1, self.num_phases + 1)
                    },
                    "round_baseline_policy": "run_spec.baseline_mutations",
                },
            )
        return self._provenance

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        del state
        candidates = [
            Experiment(name=item["name"], mutations=item["mutations"])
            for item in get_phase_candidates(phase, profile=self.round_profile)
        ]
        focus = get_phase_focus(phase, profile=self.round_profile)
        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics: self._gate_criteria(phase, metrics),
            scoring_weights=dict(self.score_weights),
            hard_rejects=self._hard_rejects_for_phase(phase),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=FOCUS_METRICS[phase],
                min_effective_score_delta_pct=0.003,
                diagnostic_gap_fn=self._diagnostic_gaps,
            ),
            max_rounds=8,
            prune_threshold=0.0,
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
        del cumulative_mutations
        if scoring_weights and len(scoring_weights) > 7:
            raise ValueError("Stock portfolio synergy scoring cannot use more than 7 components.")
        evaluation_key = build_cache_key(
            "stock.portfolio_synergy.evaluation",
            source_fingerprint=self._ensure_bundle().cache_source_fingerprint,
            extra={
                "phase": phase,
                "score_components": list(SCORE_COMPONENTS),
                "scoring_weights": scoring_weights or {},
                "hard_rejects": hard_rejects or {},
                "initial_equity": self.initial_equity,
                "start_date": self.start_date,
                "end_date": self.end_date,
            },
        )
        return CachedBatchEvaluator(
            _PortfolioBatchEvaluator(self, scoring_weights, hard_rejects),
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        sig = mutation_signature(mutations)
        with self._metrics_lock:
            cached = self._metrics_cache.get(sig)
        if cached is not None:
            score = score_portfolio_metrics(cached, scoring_weights=self.score_weights)
            metrics = _with_score_metrics(cached, score)
            with self._metrics_lock:
                self._metrics_cache[sig] = dict(metrics)
            return metrics
        metrics = self._compute_metrics_raw(mutations)
        score = score_portfolio_metrics(metrics, scoring_weights=self.score_weights)
        metrics = _with_score_metrics(metrics, score)
        with self._metrics_lock:
            self._metrics_cache[sig] = dict(metrics)
        return metrics

    def run_phase_diagnostics(
        self,
        phase: int,
        state: PhaseState,
        metrics: dict[str, float],
        greedy_result,
    ) -> str:
        del state
        return self._format_diagnostics(f"PHASE {phase} STOCK PORTFOLIO SYNERGY", metrics, greedy_result)

    def run_enhanced_diagnostics(
        self,
        phase: int,
        state: PhaseState,
        metrics: dict[str, float],
        greedy_result,
    ) -> str:
        return self.run_phase_diagnostics(phase, state, metrics, greedy_result)

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        diagnostics = self._format_diagnostics("FINAL STOCK PORTFOLIO SYNERGY DIAGNOSTICS", metrics, None)
        verdict = "PASS" if self._final_gate_passed(metrics) else "REVIEW"
        score_section = "\n".join(
            f"- {key}: {metrics.get(f'score_{key}', 0.0):.4f} (weight {self.score_weights[key]:.2f})"
            for key in SCORE_COMPONENTS
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=diagnostics,
            dimension_reports={"score_components": score_section},
            overall_verdict=verdict,
            extra_sections={
                "Score Components": score_section,
                "Execution Note": (
                    "Replay starts from latest active stock ALCB and IARIC optimized trades, "
                    "then applies portfolio-level dynamic allocation, routing, and heat controls."
                ),
            },
        )

    def _ensure_evaluation_data(self) -> None:
        self._evaluation_data = self._ensure_bundle().data

    def _ensure_bundle(self):
        bundle = load_evaluation_bundle(
            self.data_dir,
            initial_equity=self.initial_equity,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        if self._cache_source_fingerprint != bundle.cache_source_fingerprint:
            with self._metrics_lock:
                self._metrics_cache.clear()
            self._evaluation_cache.clear()
            self._price_bars_cache = None
            self._cache_source_fingerprint = bundle.cache_source_fingerprint
        self._cached_bundle = bundle
        self._evaluation_data = bundle.data
        return bundle

    def _ensure_price_bars(self) -> dict[str, Any]:
        if self._price_bars_cache is None:
            self._ensure_evaluation_data()
            assert self._evaluation_data is not None
            symbols = {
                trade.symbol
                for trade in (*self._evaluation_data.alcb_trades, *self._evaluation_data.iaric_trades)
            }
            self._price_bars_cache = _load_stock_price_bars(self.data_dir, symbols)
        return self._price_bars_cache

    def _compute_metrics_raw(self, mutations: dict[str, Any]) -> dict[str, float]:
        sig = mutation_signature(mutations)
        with self._metrics_lock:
            cached = self._metrics_cache.get(sig)
        if cached is not None and "score_total" not in cached:
            return dict(cached)
        self._ensure_evaluation_data()
        metrics = evaluate_portfolio(
            mutations,
            data_dir=self.data_dir,
            initial_equity=self.initial_equity,
            start_date=self.start_date,
            end_date=self.end_date,
            evaluation_data=self._evaluation_data,
            price_bars_by_symbol=self._ensure_price_bars(),
        )
        with self._metrics_lock:
            self._metrics_cache[sig] = dict(metrics)
        return metrics

    def _hard_rejects_for_phase(self, phase: int) -> dict[str, float]:
        gate = get_phase_gates(phase, profile=self.round_profile)
        rejects = {
            "max_drawdown_pct": gate.get("hard_max_drawdown_pct", self.round_targets["hard_max_drawdown_pct"]),
            "min_active_strategies": gate.get("min_active_strategies", 2.0),
        }
        if phase >= 4:
            rejects["min_profit_factor"] = 1.75
        if phase == 7:
            rejects["max_strategy_trade_share"] = self.round_targets["max_single_strategy_trade_share"]
        return rejects

    def _gate_criteria(self, phase: int, metrics: dict[str, float]) -> list[GateCriterion]:
        return [
            _criterion(name, target, metrics)
            for name, target in get_phase_gates(phase, profile=self.round_profile).items()
        ]

    def _diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        del phase
        gaps: list[str] = []
        if metrics.get("active_strategy_count", 0.0) < 2:
            gaps.append("Only one stock sleeve is contributing after routing.")
        if metrics.get("positive_alpha_block_rate", 0.0) > self.round_targets["max_positive_alpha_block_rate"]:
            gaps.append("Positive blocked candidates are too frequent; inspect ranking and heat scarcity.")
        if metrics.get("max_drawdown_pct", 0.0) > self.round_targets["target_max_drawdown_pct"]:
            gaps.append("Drawdown is above the controlled-aggressive target.")
        if metrics.get("candidate_discrimination", 0.0) < 0.58:
            gaps.append("Accepted candidates are not cleanly separating from blocked candidates.")
        return gaps

    def _format_diagnostics(self, title: str, metrics: dict[str, float], greedy_result) -> str:
        lines = [
            "=" * 78,
            title,
            "=" * 78,
            f"Round: {self.diagnostic_round_label}",
            f"Initial equity: ${self.initial_equity:,.0f}",
            f"Date range: {self.start_date} -> {self.end_date}",
            f"Risk stance: aggressive-controlled",
            f"Score components: {len(SCORE_COMPONENTS)} ({', '.join(SCORE_COMPONENTS)})",
            "",
            "Headline:",
            f"  Final equity: ${metrics.get('final_equity', 0.0):,.2f}",
            f"  Net PnL: ${metrics.get('net_pnl', 0.0):+,.2f}",
            f"  Net return: {metrics.get('net_return_pct', 0.0):+.2%}",
            f"  Active trades/month: {metrics.get('active_trades_per_month', 0.0):.2f}",
            f"  Total R/month: {metrics.get('total_r_per_month', 0.0):.2f}",
            f"  Profit factor: {metrics.get('profit_factor', 0.0):.2f}",
            f"  Win rate: {metrics.get('win_rate', 0.0):.2%}",
            f"  Max DD: {metrics.get('max_drawdown_pct', 0.0):.2%}",
            f"  Sharpe: {metrics.get('sharpe', 0.0):.2f}",
            f"  Sortino: {metrics.get('sortino', 0.0):.2f}",
            f"  Calmar: {metrics.get('calmar', 0.0):.2f}",
            f"  Risk basis: {metrics.get('risk_basis', 'realized_only')}",
            f"  Realized-only Max DD: {metrics.get('max_drawdown_pct_realized', metrics.get('max_drawdown_pct', 0.0)):.2%}",
            f"  Realized-only Calmar: {metrics.get('calmar_realized', metrics.get('calmar', 0.0)):.2f}",
            f"  Active sleeves: {metrics.get('active_strategy_count', 0.0):.0f}/2",
            "",
            "Sleeve contribution:",
        ]
        for strategy in STRATEGY_ORDER:
            lines.append(
                f"  {strategy:<12} trades={metrics.get(f'trades_{strategy}', 0.0):>6.0f} "
                f"pnl=${metrics.get(f'pnl_{strategy}', 0.0):>+12,.2f} "
                f"risk_share={metrics.get(f'risk_share_{strategy}', 0.0):>6.1%}"
            )
        lines.extend(
            [
                "",
                "Routing and blocked-candidate discrimination:",
                f"  Fired entries: {metrics.get('entry_signals_fired', 0.0):.0f}",
                f"  Accepted entries: {metrics.get('entries_accepted_by_portfolio', 0.0):.0f}",
                f"  Blocked entries: {metrics.get('entries_blocked_by_portfolio', 0.0):.0f}",
                f"  Trade capture ratio: {metrics.get('trade_capture_ratio', 0.0):.2%}",
                f"  Positive-alpha block rate: {metrics.get('positive_alpha_block_rate', 0.0):.2%}",
                f"  Blocked winners: {metrics.get('blocked_positive_count', 0.0):.0f} / "
                f"{metrics.get('entries_blocked_by_portfolio', 0.0):.0f} "
                f"({metrics.get('blocked_positive_fraction', 0.0):.2%})",
                f"  Accepted avg R: {metrics.get('accepted_avg_r', 0.0):+.3f}",
                f"  Blocked avg R: {metrics.get('blocked_avg_r', 0.0):+.3f}",
                f"  Candidate discrimination: {metrics.get('candidate_discrimination', 0.0):.3f}",
                f"  Max strategy trade share: {metrics.get('max_strategy_trade_share', 0.0):.2%}",
                f"  Max strategy risk share: {metrics.get('max_strategy_risk_share', 0.0):.2%}",
                f"  Positive slices: {metrics.get('positive_slices', 0.0):.0f}/4",
                "",
                "Risk governors:",
                f"  Max daily loss: {metrics.get('max_daily_loss_R', 0.0):.2f}R",
                f"  Max weekly loss: {metrics.get('max_weekly_loss_R', 0.0):.2f}R",
                "",
                "Score:",
            ]
        )
        for key in SCORE_COMPONENTS:
            lines.append(
                f"  {key:<22} component={metrics.get(f'score_{key}', 0.0):.4f} "
                f"weight={self.score_weights[key]:.2f}"
            )
        lines.append(f"  {'total':<22} {metrics.get('score_total', 0.0):.4f}")
        if greedy_result is not None:
            lines.extend(
                [
                    "",
                    "Greedy:",
                    f"  Base score: {greedy_result.base_score:.4f}",
                    f"  Final score: {greedy_result.final_score:.4f}",
                    f"  Accepted: {greedy_result.accepted_count}",
                    f"  Kept: {', '.join(greedy_result.kept_features) if greedy_result.kept_features else 'none'}",
                ]
            )
        return "\n".join(lines) + "\n"

    def _final_gate_passed(self, metrics: dict[str, float]) -> bool:
        return all(criterion.passed for criterion in self._gate_criteria(7, metrics))


def _criterion(name: str, target: float, metrics: dict[str, float]) -> GateCriterion:
    metric_name = _metric_for_gate(name)
    actual = float(metrics.get(metric_name, 0.0) or 0.0)
    lower_is_better = name.startswith(("max_", "hard_", "target_")) or "drawdown" in name
    passed = actual <= target if lower_is_better else actual >= target
    return GateCriterion(name=name, target=float(target), actual=actual, passed=passed)


def _metric_for_gate(name: str) -> str:
    mapping = {
        "min_active_strategies": "active_strategy_count",
        "min_active_trades_per_month": "active_trades_per_month",
        "min_total_r_per_month": "total_r_per_month",
        "min_profit_factor": "profit_factor",
        "target_max_drawdown_pct": "max_drawdown_pct",
        "hard_max_drawdown_pct": "max_drawdown_pct",
        "max_single_strategy_risk_share": "max_strategy_risk_share",
        "max_single_strategy_trade_share": "max_strategy_trade_share",
        "min_trade_capture_ratio": "trade_capture_ratio",
        "max_positive_alpha_block_rate": "positive_alpha_block_rate",
        "min_candidate_discrimination": "candidate_discrimination",
        "max_daily_loss_R": "max_daily_loss_R",
        "max_weekly_loss_R": "max_weekly_loss_R",
        "min_positive_slices": "positive_slices",
    }
    return mapping.get(name, name)


def _with_score_metrics(metrics: dict[str, float], score) -> dict[str, float]:
    return {
        **metrics,
        **{f"score_{key}": value for key, value in score.components.items()},
        "score_total": score.total,
    }
