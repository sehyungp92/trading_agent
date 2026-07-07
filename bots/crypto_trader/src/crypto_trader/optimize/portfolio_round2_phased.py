"""Portfolio round-2 phased optimizer.

This module adapts portfolio optimization to the shared phased-auto framework.
It scores real live-parity portfolio backtests on the development window only
and keeps the post-2026-04-20 period as validation evidence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.diagnostics import generate_diagnostics
from crypto_trader.backtest.metrics import metrics_to_dict
from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
from crypto_trader.optimize.config_mutator import apply_mutations, merge_mutations
from crypto_trader.optimize.types import (
    EndOfRoundArtifacts,
    EvaluateFn,
    Experiment,
    GateCriterion,
    GreedyResult,
    PhaseAnalysisPolicy,
    PhaseSpec,
    ScoredCandidate,
)
from crypto_trader.portfolio.backtest_runner import PortfolioBacktestResult, run_portfolio_backtest
from crypto_trader.portfolio.config import PortfolioConfig, StrategyAllocation
from crypto_trader.strategy.breakout.config import BreakoutConfig
from crypto_trader.strategy.momentum.config import MomentumConfig
from crypto_trader.strategy.trend.config import TrendConfig

SYMBOLS = ["BTC", "ETH", "SOL"]
STRATEGIES = ["momentum", "trend", "breakout"]
MAX_SCORE_COMPONENTS = 7

DEV_START = date(2026, 2, 25)
DEV_END = date(2026, 4, 20)
HOLDOUT_START = date(2026, 4, 21)
HOLDOUT_END = date(2026, 5, 23)
FULL_START = DEV_START
FULL_END = HOLDOUT_END

SCORING_WEIGHTS: dict[str, float] = {
    "return": 0.30,
    "frequency": 0.20,
    "edge_quality": 0.18,
    "capture": 0.10,
    "drawdown_resilience": 0.10,
    "rule_efficiency": 0.06,
    "strategy_balance": 0.06,
}

HARD_REJECTS: dict[str, tuple[str, float]] = {
    "total_trades": (">=", 42.0),
    "profit_factor": (">=", 1.55),
    "expectancy_r": (">=", 0.20),
    "max_drawdown_pct": ("<=", 8.50),
    "exit_efficiency": (">=", 0.30),
}

PHASE_NAMES: dict[int, str] = {
    1: "Signal Discrimination",
    2: "Capture And Exit Quality",
    3: "Frequency Expansion",
    4: "Portfolio Synergy",
    5: "Risk Scaling",
}

PHASE_OBJECTIVES: dict[int, str] = {
    1: "Remove or harden the weakest SOL and confirmation cohorts without collapsing frequency.",
    2: "Reduce giveback and improve exit capture with deployable strategy config knobs.",
    3: "Add real trades through guarded strategy-level frequency probes.",
    4: "Improve cross-strategy interaction so portfolio rules block crowding rather than good alpha.",
    5: "Lean risk aggressively but not recklessly after structure has been chosen.",
}


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(value, low), high)


def _exp(name: str, mutations: dict[str, Any]) -> Experiment:
    return Experiment(name=name, mutations=mutations)


def _strategy_path(strategy_id: str, root: Path) -> Path:
    return root / "output" / strategy_id / "round_3" / "optimized_config.json"


def _load_strategy_config(strategy_id: str, root: Path) -> Any:
    with open(_strategy_path(strategy_id, root), encoding="utf-8") as f:
        payload = json.load(f)["strategy"]
    if strategy_id == "momentum":
        return MomentumConfig.from_dict(payload)
    if strategy_id == "trend":
        return TrendConfig.from_dict(payload)
    if strategy_id == "breakout":
        return BreakoutConfig.from_dict(payload)
    raise ValueError(f"Unknown strategy: {strategy_id}")


def load_base_strategy_configs(root: Path) -> dict[str, Any]:
    return {strategy_id: _load_strategy_config(strategy_id, root) for strategy_id in STRATEGIES}


def load_base_portfolio_config(path: Path) -> PortfolioConfig:
    with open(path, encoding="utf-8") as f:
        return PortfolioConfig.from_dict(json.load(f))


def _bt_config(start: date, end: date, initial_equity: float) -> BacktestConfig:
    return build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=list(SYMBOLS),
        start_date=start,
        end_date=end,
        initial_equity=initial_equity,
    )


def _scale(value: float, multiplier: float, cap: float) -> float:
    return round(min(value * multiplier, cap), 8)


def _risk_scale_mutations(strategy_id: str, cfg: Any, multiplier: float) -> dict[str, Any]:
    if abs(multiplier - 1.0) < 1e-9:
        return {}

    if strategy_id == "momentum":
        risk = cfg.risk
        return {
            "risk.risk_pct_a": _scale(risk.risk_pct_a, multiplier, 0.030),
            "risk.risk_pct_b": _scale(risk.risk_pct_b, multiplier, 0.025),
            "risk.max_gross_risk": _scale(risk.max_gross_risk, max(1.0, multiplier), 0.060),
            "risk.max_correlated_risk": _scale(risk.max_correlated_risk, max(1.0, multiplier), 0.040),
        }

    if strategy_id == "trend":
        risk = cfg.risk
        limits = cfg.limits
        return {
            "risk.risk_pct_a": _scale(risk.risk_pct_a, multiplier, 0.028),
            "risk.risk_pct_b": _scale(risk.risk_pct_b, multiplier, 0.028),
            "risk.max_risk_pct": _scale(risk.max_risk_pct, max(1.0, multiplier), 0.030),
            "limits.max_correlated_risk_pct": _scale(
                limits.max_correlated_risk_pct, max(1.0, multiplier), 0.070
            ),
        }

    if strategy_id == "breakout":
        risk = cfg.risk
        limits = cfg.limits
        return {
            "risk.risk_pct_a_plus": _scale(risk.risk_pct_a_plus, multiplier, 0.018),
            "risk.risk_pct_a": _scale(risk.risk_pct_a, multiplier, 0.018),
            "risk.risk_pct_b": _scale(risk.risk_pct_b, multiplier, 0.032),
            "risk.max_risk_pct": _scale(risk.max_risk_pct, max(1.0, multiplier), 0.033),
            "limits.max_correlated_risk_pct": _scale(
                limits.max_correlated_risk_pct, max(1.0, multiplier), 0.030
            ),
        }

    raise ValueError(f"Unknown strategy: {strategy_id}")


def _split_policy_mutations(
    mutations: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, Any]]:
    strategy_mutations: dict[str, dict[str, Any]] = {strategy_id: {} for strategy_id in STRATEGIES}
    risk_scales: dict[str, float] = {strategy_id: 1.0 for strategy_id in STRATEGIES}
    portfolio_overrides: dict[str, Any] = {}

    for key, value in mutations.items():
        if key.startswith("strategy."):
            _, strategy_id, path = key.split(".", 2)
            if strategy_id not in STRATEGIES:
                raise ValueError(f"Unknown strategy mutation namespace: {key!r}")
            strategy_mutations[strategy_id][path] = value
        elif key.startswith("risk_scale."):
            _, strategy_id = key.split(".", 1)
            if strategy_id not in STRATEGIES:
                raise ValueError(f"Unknown risk scale namespace: {key!r}")
            risk_scales[strategy_id] = float(value)
        elif key.startswith("portfolio."):
            _, path = key.split(".", 1)
            portfolio_overrides[path] = value
        else:
            raise ValueError(f"Unknown portfolio optimization mutation key: {key!r}")

    return strategy_mutations, risk_scales, portfolio_overrides


def _apply_strategy_configs(
    base_configs: dict[str, Any],
    mutations: dict[str, Any],
) -> dict[str, Any]:
    strategy_mutations, risk_scales, _ = _split_policy_mutations(mutations)
    configs: dict[str, Any] = {}
    for strategy_id, cfg in base_configs.items():
        risk_mutations = _risk_scale_mutations(strategy_id, cfg, risk_scales[strategy_id])
        merged = merge_mutations(risk_mutations, strategy_mutations.get(strategy_id, {}))
        new_cfg = apply_mutations(cfg, merged) if merged else apply_mutations(cfg, {})
        new_cfg.symbols = list(SYMBOLS)
        configs[strategy_id] = new_cfg
    return configs


def _portfolio_config_from_mutations(
    base_config: PortfolioConfig,
    mutations: dict[str, Any],
) -> PortfolioConfig:
    _, risk_scales, portfolio_overrides = _split_policy_mutations(mutations)
    kwargs = base_config.to_dict()
    kwargs["strategies"] = tuple(
        StrategyAllocation(
            strategy_id=allocation.strategy_id,
            enabled=allocation.enabled,
            base_risk_pct=round(
                allocation.base_risk_pct * risk_scales.get(allocation.strategy_id, 1.0),
                8,
            ),
            max_concurrent=allocation.max_concurrent,
            daily_stop_R=allocation.daily_stop_R,
            priority=allocation.priority,
        )
        for allocation in base_config.strategies
    )
    kwargs.update(portfolio_overrides)
    if "dd_tiers" in kwargs:
        kwargs["dd_tiers"] = tuple(tuple(item) for item in kwargs["dd_tiers"])
    return PortfolioConfig(**kwargs)


def _strategy_balance(result: PortfolioBacktestResult) -> float:
    trades = len(result.all_trades)
    if trades <= 0:
        return 0.0
    shares = [
        len(result.per_strategy_trades.get(strategy_id, [])) / trades
        for strategy_id in STRATEGIES
    ]
    active = [share for share in shares if share > 0.0]
    if not active:
        return 0.0
    entropy = -sum(share * math.log(share) for share in active) / math.log(len(STRATEGIES))
    min_share_score = _clip(min(shares) / 0.10)
    return _clip(0.70 * entropy + 0.30 * min_share_score)


def _augment_metrics(metrics: dict[str, float], result: PortfolioBacktestResult) -> dict[str, float]:
    rule_checks = float(len(result.rule_events))
    approved = float(sum(1 for event in result.rule_events if event.approved))
    blocked = rule_checks - approved
    out = dict(metrics)
    out["rule_checks"] = rule_checks
    out["approved_entries"] = approved
    out["blocked_entries"] = blocked
    out["rule_approval_rate"] = approved / rule_checks * 100.0 if rule_checks else 100.0
    out["strategy_balance"] = _strategy_balance(result)
    return out


def _score_components(metrics: dict[str, float]) -> dict[str, float]:
    pf = float(metrics.get("profit_factor", 0.0))
    if math.isinf(pf):
        pf = 8.0
    expectancy = float(metrics.get("expectancy_r", 0.0))
    approval_rate = float(metrics.get("rule_approval_rate", 100.0))
    components = {
        "return": _clip(float(metrics.get("net_return_pct", 0.0)) / 75.0),
        "frequency": _clip(float(metrics.get("total_trades", 0.0)) / 58.0),
        "edge_quality": _clip(
            0.55 * _clip((pf - 1.0) / 4.0)
            + 0.45 * _clip((expectancy + 0.05) / 0.75)
        ),
        "capture": _clip(float(metrics.get("exit_efficiency", 0.0)) / 0.55),
        "drawdown_resilience": _clip(1.0 - float(metrics.get("max_drawdown_pct", 0.0)) / 8.5),
        "rule_efficiency": _clip((approval_rate - 82.0) / 16.0),
        "strategy_balance": _clip(float(metrics.get("strategy_balance", 0.0))),
    }
    if len(components) > MAX_SCORE_COMPONENTS:
        raise ValueError(
            f"Portfolio score has {len(components)} components, limit is {MAX_SCORE_COMPONENTS}"
        )
    return components


def _hard_reject_reason(
    metrics: dict[str, float],
    hard_rejects: dict[str, tuple[str, float]],
) -> str:
    failures: list[str] = []
    for metric, (operator, threshold) in hard_rejects.items():
        value = float(metrics.get(metric, 0.0))
        passed = (
            value >= threshold if operator == ">=" else
            value <= threshold if operator == "<=" else
            value > threshold if operator == ">" else
            value < threshold if operator == "<" else
            False
        )
        if not passed:
            failures.append(f"{metric} {value:.4g} fails {operator} {threshold:.4g}")
    return "; ".join(failures)


def _score_metrics(
    metrics: dict[str, float],
    scoring_weights: dict[str, float],
    hard_rejects: dict[str, tuple[str, float]],
) -> tuple[float, bool, str, dict[str, float]]:
    components = _score_components(metrics)
    score = sum(scoring_weights[name] * components[name] for name in scoring_weights)
    reject_reason = _hard_reject_reason(metrics, hard_rejects)
    metrics["score_components"] = components
    return score, bool(reject_reason), reject_reason, components


def _silence_worker_logging() -> None:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))


def _run_policy(
    *,
    root: Path,
    data_dir: Path,
    portfolio_config_path: Path,
    mutations: dict[str, Any],
    start: date,
    end: date,
) -> tuple[dict[str, float], PortfolioBacktestResult]:
    base_configs = load_base_strategy_configs(root)
    base_portfolio = load_base_portfolio_config(portfolio_config_path)
    strategy_configs = _apply_strategy_configs(base_configs, mutations)
    portfolio_config = _portfolio_config_from_mutations(base_portfolio, mutations)
    result = run_portfolio_backtest(
        portfolio_config=portfolio_config,
        strategy_configs=strategy_configs,
        backtest_config=_bt_config(start, end, portfolio_config.initial_equity),
        data_dir=data_dir,
    )
    metrics = _augment_metrics(metrics_to_dict(result.metrics), result)
    return metrics, result


def _evaluate_worker(args: tuple[Any, ...]) -> tuple[int, ScoredCandidate]:
    _silence_worker_logging()
    (
        idx,
        name,
        experiment_mutations,
        merged_mutations,
        root_str,
        data_dir_str,
        portfolio_config_path_str,
        start,
        end,
        scoring_weights,
        hard_rejects,
    ) = args
    try:
        metrics, _ = _run_policy(
            root=Path(root_str),
            data_dir=Path(data_dir_str),
            portfolio_config_path=Path(portfolio_config_path_str),
            mutations=merged_mutations,
            start=start,
            end=end,
        )
        score, rejected, reason, _ = _score_metrics(metrics, scoring_weights, hard_rejects)
        return idx, ScoredCandidate(
            experiment=Experiment(name=name, mutations=experiment_mutations),
            score=score,
            metrics=metrics,
            rejected=rejected,
            reject_reason=reason,
        )
    except Exception as exc:
        return idx, ScoredCandidate(
            experiment=Experiment(name=name, mutations=experiment_mutations),
            score=0.0,
            metrics={},
            rejected=True,
            reject_reason=f"Exception: {exc}",
        )


def _phase1_candidates() -> list[Experiment]:
    return [
        _exp("trend_sol_short_only", {"strategy.trend.symbol_filter.sol_direction": "short_only"}),
        _exp("momentum_disable_micro_shift", {"strategy.momentum.confirmation.enable_micro_shift": False}),
        _exp("momentum_weak_confirmations_need_three", {"strategy.momentum.confirmation.min_confluences_for_weak": 3}),
        _exp("momentum_sol_disabled", {"strategy.momentum.symbol_filter.sol_direction": "disabled"}),
        _exp("momentum_sol_short_probe", {"strategy.momentum.symbol_filter.sol_direction": "short_only"}),
        _exp("breakout_sol_short_only", {"strategy.breakout.symbol_filter.sol_direction": "short_only"}),
        _exp(
            "breakout_eth_long_only",
            {
                "strategy.breakout.symbol_filter.eth_direction": "long_only",
                "strategy.breakout.symbol_filter.eth_relaxed_body_direction": "long_only",
            },
        ),
        _exp("trend_raise_b_quality", {"strategy.trend.setup.min_setup_score_b": 1.45}),
    ]


def _phase2_candidates() -> list[Experiment]:
    return [
        _exp(
            "momentum_tighter_mfe_retrace",
            {
                "strategy.momentum.exits.mfe_retrace_trigger_r": 1.10,
                "strategy.momentum.exits.mfe_retrace_giveback_r": 0.60,
                "strategy.momentum.exits.mfe_retrace_min_bars": 4,
            },
        ),
        _exp("momentum_earlier_runner_trail", {"strategy.momentum.trail.runner_trigger_r": 1.20}),
        _exp(
            "momentum_proof_lock_less_noise",
            {
                "strategy.momentum.exits.proof_lock_trigger_r": 0.55,
                "strategy.momentum.exits.proof_lock_min_bars": 3,
            },
        ),
        _exp(
            "trend_mfe_lock_enable",
            {
                "strategy.trend.exits.mfe_lock_exit_enabled": True,
                "strategy.trend.exits.mfe_lock_trigger_r": 0.75,
                "strategy.trend.exits.mfe_lock_floor_r": 0.10,
                "strategy.trend.exits.mfe_lock_min_bars": 2,
            },
        ),
        _exp(
            "trend_scratch_less_eager",
            {
                "strategy.trend.exits.scratch_peak_r": 0.45,
                "strategy.trend.exits.scratch_floor_r": -0.05,
                "strategy.trend.exits.scratch_min_bars": 3,
            },
        ),
        _exp(
            "breakout_gentle_early_lock",
            {
                "strategy.breakout.exits.early_lock_enabled": True,
                "strategy.breakout.exits.early_lock_mfe_r": 0.45,
                "strategy.breakout.exits.early_lock_stop_r": 0.05,
            },
        ),
    ]


def _phase3_candidates() -> list[Experiment]:
    return [
        _exp(
            "breakout_relaxed_body_guarded_expand",
            {
                "strategy.breakout.setup.relaxed_body_min_confluences": 4,
                "strategy.breakout.setup.relaxed_body_min_room_r": 1.6,
                "strategy.breakout.setup.relaxed_body_risk_scale": 0.35,
            },
        ),
        _exp(
            "breakout_model2_wider_retest",
            {
                "strategy.breakout.confirmation.retest_max_bars": 4,
                "strategy.breakout.entry.max_bars_after_signal": 4,
            },
        ),
        _exp(
            "trend_reentry_more_patient",
            {
                "strategy.trend.reentry.max_reentries": 2,
                "strategy.trend.reentry.max_wait_bars": 8,
                "strategy.trend.reentry.risk_scale": 0.65,
            },
        ),
        _exp("momentum_reentry_faster", {"strategy.momentum.reentry.cooldown_bars": 2}),
        _exp(
            "breakout_frequency_combo",
            {
                "strategy.breakout.setup.relaxed_body_min_confluences": 4,
                "strategy.breakout.setup.relaxed_body_min_room_r": 1.6,
                "strategy.breakout.setup.relaxed_body_risk_scale": 0.35,
                "strategy.breakout.confirmation.retest_max_bars": 4,
                "strategy.breakout.entry.max_bars_after_signal": 4,
            },
        ),
    ]


def _phase4_candidates() -> list[Experiment]:
    return [
        _exp(
            "collision_cap_250",
            {
                "portfolio.symbol_collision": "cap",
                "portfolio.symbol_exposure_cap_R": 2.5,
            },
        ),
        _exp(
            "collision_cap_300",
            {
                "portfolio.symbol_collision": "cap",
                "portfolio.symbol_exposure_cap_R": 3.0,
            },
        ),
        _exp(
            "momentum_low_priority_headroom",
            {
                "portfolio.priority_headroom_R": 1.0,
                "portfolio.priority_reserve_threshold": 2,
            },
        ),
        _exp(
            "trend_breakout_priority_headroom",
            {
                "portfolio.priority_headroom_R": 0.75,
                "portfolio.priority_reserve_threshold": 1,
            },
        ),
        _exp(
            "aggressive_caps_with_fast_dd_guard",
            {
                "portfolio.heat_cap_R": 8.25,
                "portfolio.directional_cap_R": 5.5,
                "portfolio.max_total_positions": 11,
                "portfolio.dd_tiers": (
                    (0.06, 0.75),
                    (0.09, 0.50),
                    (0.12, 0.25),
                    (0.15, 0.00),
                ),
            },
        ),
    ]


def _phase5_candidates() -> list[Experiment]:
    return [
        _exp(
            "risk_all_110",
            {
                "risk_scale.momentum": 1.10,
                "risk_scale.trend": 1.10,
                "risk_scale.breakout": 1.10,
            },
        ),
        _exp(
            "risk_quality_weighted",
            {
                "risk_scale.momentum": 0.92,
                "risk_scale.trend": 1.18,
                "risk_scale.breakout": 1.12,
            },
        ),
        _exp(
            "risk_frequency_weighted",
            {
                "risk_scale.momentum": 1.08,
                "risk_scale.trend": 1.15,
                "risk_scale.breakout": 1.00,
            },
        ),
        _exp(
            "risk_breakout_alpha_probe",
            {
                "risk_scale.momentum": 0.90,
                "risk_scale.trend": 1.08,
                "risk_scale.breakout": 1.22,
            },
        ),
        _exp(
            "risk_all_115_dd_guarded",
            {
                "risk_scale.momentum": 1.15,
                "risk_scale.trend": 1.15,
                "risk_scale.breakout": 1.15,
                "portfolio.dd_tiers": (
                    (0.06, 0.75),
                    (0.09, 0.50),
                    (0.12, 0.25),
                    (0.15, 0.00),
                ),
            },
        ),
    ]


PHASE_CANDIDATES = {
    1: _phase1_candidates,
    2: _phase2_candidates,
    3: _phase3_candidates,
    4: _phase4_candidates,
    5: _phase5_candidates,
}


class PortfolioRound2PhasedPlugin:
    """Portfolio adapter for the shared phased-auto runner."""

    def __init__(
        self,
        *,
        root: Path,
        data_dir: Path,
        portfolio_config_path: Path,
        max_workers: int = 2,
    ) -> None:
        self.root = root
        self.data_dir = data_dir
        self.portfolio_config_path = portfolio_config_path
        self.max_workers = max_workers
        self._contract = self._build_contract()

    @property
    def name(self) -> str:
        return "portfolio_round2_phased"

    @property
    def num_phases(self) -> int:
        return 5

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 55.0,
            "total_trades": 52.0,
            "profit_factor": 2.0,
            "expectancy_r": 0.30,
            "exit_efficiency": 0.42,
            "max_drawdown_pct": 8.5,
        }

    @property
    def initial_mutations(self) -> dict[str, Any]:
        return {}

    @property
    def contract(self) -> dict[str, Any]:
        return dict(self._contract)

    def _build_contract(self) -> dict[str, Any]:
        source_hashes: dict[str, str] = {}
        for strategy_id in STRATEGIES:
            path = _strategy_path(strategy_id, self.root)
            source_hashes[strategy_id] = hashlib.sha256(path.read_bytes()).hexdigest()
        source_hashes["portfolio"] = hashlib.sha256(
            self.portfolio_config_path.read_bytes()
        ).hexdigest()
        payload = {
            "kind": "portfolio_phased_auto",
            "profile": "LIVE_PARITY_PROFILE",
            "symbols": list(SYMBOLS),
            "max_workers": self.max_workers,
            "development_window": {"start": str(DEV_START), "end": str(DEV_END)},
            "holdout_window": {"start": str(HOLDOUT_START), "end": str(HOLDOUT_END)},
            "full_window": {"start": str(FULL_START), "end": str(FULL_END)},
            "score_components": list(SCORING_WEIGHTS),
            "source_hashes": source_hashes,
        }
        payload["contract_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return payload

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        del state
        candidate_factory = PHASE_CANDIDATES.get(phase)
        candidates = candidate_factory() if candidate_factory else []
        max_rounds = 1 if phase == 5 else 3
        if phase == 4:
            max_rounds = 2
        policy = PhaseAnalysisPolicy(
            max_scoring_retries=0,
            max_diagnostic_retries=0,
            focus_metrics=[
                "net_return_pct",
                "total_trades",
                "profit_factor",
                "expectancy_r",
                "exit_efficiency",
            ],
        )
        return PhaseSpec(
            phase_num=phase,
            name=PHASE_NAMES.get(phase, f"Phase {phase}"),
            candidates=candidates,
            scoring_weights=dict(SCORING_WEIGHTS),
            hard_rejects=dict(HARD_REJECTS),
            gate_criteria=[
                GateCriterion("total_trades", ">=", 42.0),
                GateCriterion("profit_factor", ">=", 1.55),
                GateCriterion("expectancy_r", ">=", 0.20),
                GateCriterion("exit_efficiency", ">=", 0.30),
                GateCriterion("max_drawdown_pct", "<=", 8.5),
            ],
            analysis_policy=policy,
            min_delta=0.003,
            max_rounds=max_rounds,
            prune_threshold=0.0,
            focus=PHASE_OBJECTIVES.get(phase, ""),
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        scoring_weights: dict[str, float],
        hard_rejects: dict[str, tuple[str, float]],
    ) -> EvaluateFn:
        del phase, cumulative_mutations

        def evaluate_fn(
            candidates: list[Experiment],
            current_mutations: dict[str, Any],
        ) -> list[ScoredCandidate]:
            if not candidates:
                return []
            work_items = []
            for idx, experiment in enumerate(candidates):
                merged = merge_mutations(current_mutations, experiment.mutations)
                work_items.append((
                    idx,
                    experiment.name,
                    experiment.mutations,
                    merged,
                    str(self.root),
                    str(self.data_dir),
                    str(self.portfolio_config_path),
                    DEV_START,
                    DEV_END,
                    scoring_weights,
                    hard_rejects,
                ))

            if self.max_workers <= 1 or len(work_items) == 1:
                results = [_evaluate_worker(item) for item in work_items]
            else:
                results = []
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(_evaluate_worker, item): item[0] for item in work_items}
                    for future in as_completed(futures):
                        results.append(future.result())
            results.sort(key=lambda item: item[0])
            return [scored for _, scored in results]

        return evaluate_fn

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        metrics, _ = _run_policy(
            root=self.root,
            data_dir=self.data_dir,
            portfolio_config_path=self.portfolio_config_path,
            mutations=mutations,
            start=DEV_START,
            end=DEV_END,
        )
        score, rejected, reason, components = _score_metrics(
            metrics,
            dict(SCORING_WEIGHTS),
            dict(HARD_REJECTS),
        )
        metrics["immutable_score"] = score
        metrics["rejected"] = float(1 if rejected else 0)
        metrics["reject_reason"] = reason
        metrics["score_components"] = components
        return metrics

    def _format_metrics_line(self, label: str, metrics: dict[str, float]) -> str:
        return (
            f"{label}: trades={metrics.get('total_trades', 0):.0f}, "
            f"return={metrics.get('net_return_pct', 0):.2f}%, "
            f"PF={metrics.get('profit_factor', 0):.2f}, "
            f"expR={metrics.get('expectancy_r', 0):.3f}, "
            f"exit_eff={metrics.get('exit_efficiency', 0):.3f}, "
            f"DD={metrics.get('max_drawdown_pct', 0):.2f}%"
        )

    def run_phase_diagnostics(
        self,
        phase: int,
        state: Any,
        metrics: dict[str, float],
        greedy_result: GreedyResult,
    ) -> str:
        holdout_metrics, _ = _run_policy(
            root=self.root,
            data_dir=self.data_dir,
            portfolio_config_path=self.portfolio_config_path,
            mutations=greedy_result.final_mutations,
            start=HOLDOUT_START,
            end=HOLDOUT_END,
        )
        lines = [
            f"Portfolio phase {phase}: {PHASE_NAMES.get(phase, '')}",
            f"Objective: {PHASE_OBJECTIVES.get(phase, '')}",
            "Selection basis: development window only.",
            "Holdout is validation-only and is not part of the immutable score.",
            self._format_metrics_line("development", metrics),
            self._format_metrics_line("holdout", holdout_metrics),
            f"accepted: {', '.join(greedy_result.kept_features) or 'none'}",
            f"score_components: {json.dumps(metrics.get('score_components', {}), sort_keys=True)}",
        ]
        return "\n".join(lines)

    def run_enhanced_diagnostics(
        self,
        phase: int,
        state: Any,
        metrics: dict[str, float],
        greedy_result: GreedyResult,
    ) -> str:
        return self.run_phase_diagnostics(phase, state, metrics, greedy_result)

    def build_end_of_round_artifacts(self, state: Any) -> EndOfRoundArtifacts:
        mutations = dict(getattr(state, "cumulative_mutations", {}) or {})
        dev_metrics, _ = _run_policy(
            root=self.root,
            data_dir=self.data_dir,
            portfolio_config_path=self.portfolio_config_path,
            mutations=mutations,
            start=DEV_START,
            end=DEV_END,
        )
        holdout_metrics, _ = _run_policy(
            root=self.root,
            data_dir=self.data_dir,
            portfolio_config_path=self.portfolio_config_path,
            mutations=mutations,
            start=HOLDOUT_START,
            end=HOLDOUT_END,
        )
        full_metrics, full_result = _run_policy(
            root=self.root,
            data_dir=self.data_dir,
            portfolio_config_path=self.portfolio_config_path,
            mutations=mutations,
            start=FULL_START,
            end=FULL_END,
        )
        diagnostic_trades = [
            replace(trade, realized_r_multiple=None)
            for trade in full_result.all_trades
        ]
        diagnostics = (
            "# Portfolio round 2 diagnostics\n"
            "# Selection score used development only; holdout is validation-only.\n\n"
        )
        diagnostics += generate_diagnostics(
            diagnostic_trades,
            initial_equity=full_result.portfolio_config.initial_equity
            if full_result.portfolio_config else 25_000.0,
        )
        extra = {
            "windows": (
                f"development={DEV_START}..{DEV_END}, "
                f"holdout={HOLDOUT_START}..{HOLDOUT_END}, "
                f"full={FULL_START}..{FULL_END}"
            ),
            "development": self._format_metrics_line("development", dev_metrics),
            "holdout": self._format_metrics_line("holdout", holdout_metrics),
            "full": self._format_metrics_line("full", full_metrics),
        }
        verdict = (
            "Selected by development-window immutable score with post-2026-04-20 "
            "reported as validation evidence."
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=diagnostics,
            dimension_reports=extra,
            overall_verdict=verdict,
            extra_sections={"contract": json.dumps(self.contract, indent=2, default=str)},
        )

    def save_recommended_artifacts(
        self,
        mutations: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        base_configs = load_base_strategy_configs(self.root)
        base_portfolio = load_base_portfolio_config(self.portfolio_config_path)
        strategy_configs = _apply_strategy_configs(base_configs, mutations)
        portfolio_config = _portfolio_config_from_mutations(base_portfolio, mutations)

        config_dir = output_dir / "recommended_strategy_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        for strategy_id, cfg in strategy_configs.items():
            _write_json(config_dir / f"{strategy_id}.json", {"strategy": cfg.to_dict()})
        _write_json(output_dir / "recommended_portfolio_config.json", portfolio_config.to_dict())

        dev_metrics, _ = _run_policy(
            root=self.root,
            data_dir=self.data_dir,
            portfolio_config_path=self.portfolio_config_path,
            mutations=mutations,
            start=DEV_START,
            end=DEV_END,
        )
        holdout_metrics, _ = _run_policy(
            root=self.root,
            data_dir=self.data_dir,
            portfolio_config_path=self.portfolio_config_path,
            mutations=mutations,
            start=HOLDOUT_START,
            end=HOLDOUT_END,
        )
        full_metrics, full_result = _run_policy(
            root=self.root,
            data_dir=self.data_dir,
            portfolio_config_path=self.portfolio_config_path,
            mutations=mutations,
            start=FULL_START,
            end=FULL_END,
        )
        score, rejected, reason, components = _score_metrics(
            dev_metrics,
            dict(SCORING_WEIGHTS),
            dict(HARD_REJECTS),
        )
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection_basis": "development window only",
            "holdout_policy": "post-2026-04-20 excluded from optimization score",
            "immutable_score": score,
            "rejected": rejected,
            "reject_reason": reason,
            "score_components": components,
            "score_component_weights": dict(SCORING_WEIGHTS),
            "score_component_limit": MAX_SCORE_COMPONENTS,
            "mutations": dict(mutations),
            "portfolio_config": portfolio_config.to_dict(),
            "metrics": {
                "development": dev_metrics,
                "holdout": holdout_metrics,
                "full": full_metrics,
            },
            "rule_denials": _rule_denial_counts(full_result),
            "contract": self.contract,
        }
        _write_json(output_dir / "phase_auto_results.json", payload)
        return payload


def _rule_denial_counts(result: PortfolioBacktestResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in result.rule_events:
        if event.approved:
            continue
        reason = event.denial_reason or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
