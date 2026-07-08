"""Breakout canonical round 1: single-phase greedy run over legacy round-3 unions.

This canonical round 1 starts from the reconstructed pre-round-1 breakout
baseline and restricts the search space to the exact mutations present in the
historical breakout round-3 lineage.

The score is intentionally immutable and biased toward the user's stated
objective: maximize both expected returns and trading frequency, while keeping
drawdown contained enough that pure leverage changes do not dominate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.optimize.breakout_plugin import BreakoutPlugin
from crypto_trader.optimize.breakout_round3_pre_round1 import (
    SYMBOLS,
    build_backtest_config,
    build_pre_round1_config,
)
from crypto_trader.optimize.config_mutator import merge_mutations
from crypto_trader.optimize.greedy_optimizer import (
    _compute_identity,
    _load_checkpoint,
    _save_checkpoint,
)
from crypto_trader.optimize.parallel import evaluate_parallel
from crypto_trader.optimize.types import (
    EvaluateFn,
    Experiment,
    GateCriterion,
    GreedyResult,
    GreedyRound,
    PhaseAnalysisPolicy,
    PhaseSpec,
    ScoredCandidate,
)

log = structlog.get_logger("optimize.breakout_round4_trade_frequency")

ROUND3_SOURCE_DIRS: tuple[str, str] = ("round_3_breakout", "round_3")

# Returns + frequency dominate. Edge is kept, but with a wide ceiling so that
# low-sample PF spikes do not overpower trade count. Risk is a guardrail rather
# than the main target for this round.
IMMUTABLE_SCORING_WEIGHTS: dict[str, float] = {
    "returns": 0.40,
    "coverage": 0.35,
    "edge": 0.15,
    "risk": 0.10,
}

IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "returns": 8.0,
    "coverage": 20.0,
    "edge": 5.0,
    "risk": 15.0,
}

IMMUTABLE_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "total_trades": (">=", 10.0),
    "max_drawdown_pct": ("<=", 15.0),
}

IMMUTABLE_GATE_CRITERIA: list[GateCriterion] = [
    GateCriterion(metric="net_return_pct", operator=">", threshold=0.0),
    GateCriterion(metric="total_trades", operator=">=", threshold=12.0),
    GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=12.0),
]

_CANDIDATE_PRIORITY: dict[str, int] = {
    "confirmation.model1_require_direction_close": 10,
    "profile.lookback_bars": 20,
    "balance.min_bars_in_zone": 30,
    "balance.zone_width_atr": 40,
    "setup.body_ratio_min": 50,
    "trail.trail_activation_bars": 60,
    "exits.tp1_r": 70,
    "exits.time_stop_bars": 80,
    "stops.atr_mult": 90,
    "exits.be_buffer_r": 100,
    "limits.max_trades_per_day": 110,
    "limits.max_consecutive_losses": 120,
    "limits.max_daily_loss_pct": 130,
    "risk.risk_pct_a_plus": 140,
    "risk.risk_pct_a": 150,
    "risk.risk_pct_b": 160,
}


def _flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, dotted))
        else:
            flat[dotted] = value
    return flat


def _normalize_name_fragment(value: Any) -> str:
    text = str(value)
    return (
        text.replace(".", "_")
        .replace("-", "neg_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _experiment_name(key: str, value: Any) -> str:
    dotted = key.replace(".", "__")
    return f"{dotted}__{_normalize_name_fragment(value)}"


def _sort_value(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (1, float(value))
    return (2, str(value))


def _candidate_sort_key(candidate_meta: dict[str, Any]) -> tuple[Any, ...]:
    key = candidate_meta["key"]
    priority = _CANDIDATE_PRIORITY.get(key, 999)
    return (priority, key, *_sort_value(candidate_meta["value"]))


def _load_strategy_payload(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return dict(payload["strategy"])


def _delta_ratio(score: float, baseline: float) -> float:
    if baseline > 0:
        return (score - baseline) / baseline
    return score - baseline


def build_candidate_pool(
    output_dir: Path,
    baseline_config,
) -> tuple[list[Experiment], list[dict[str, Any]]]:
    """Build the canonical round-1 candidate union from both legacy branches."""
    baseline_flat = _flatten_dict(baseline_config.to_dict())
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for source_dir in ROUND3_SOURCE_DIRS:
        strategy_path = output_dir / source_dir / "optimized_config.json"
        strategy_flat = _flatten_dict(_load_strategy_payload(strategy_path))

        for key, value in strategy_flat.items():
            if baseline_flat.get(key) == value:
                continue

            identity = (key, json.dumps(value, sort_keys=True))
            entry = merged.setdefault(
                identity,
                {
                    "key": key,
                    "value": value,
                    "sources": [],
                    "source_paths": [],
                },
            )
            entry["sources"].append(source_dir)
            entry["source_paths"].append(str(strategy_path))

    candidate_meta = sorted(merged.values(), key=_candidate_sort_key)
    candidates = [
        Experiment(name=_experiment_name(meta["key"], meta["value"]), mutations={meta["key"]: meta["value"]})
        for meta in candidate_meta
    ]
    return candidates, candidate_meta


def run_greedy_without_pruning(
    candidates: list[Experiment],
    current_mutations: dict[str, Any],
    evaluate_fn: EvaluateFn,
    *,
    min_delta: float = 0.005,
    max_rounds: int = 20,
    prune_threshold: float = 0.05,
    checkpoint_path: Path | None = None,
    checkpoint_context: str | None = None,
    logger: Any = None,
) -> GreedyResult:
    """Round-4-specific greedy loop without permanent pruning side effects.

    The default framework permanently drops hard-rejected candidates and applies
    streak pruning after two rounds. For this single-phase union search that is
    too aggressive, because some round-3 mutations only become viable after a
    structural mutation is accepted first.
    """
    del prune_threshold, logger

    start_time = time.time()
    remaining = list(candidates)
    accepted: list[ScoredCandidate] = []
    rejected_by_name: dict[str, ScoredCandidate] = {}
    active_mutations = dict(current_mutations)
    rounds: list[GreedyRound] = []
    round_num = 0
    total_candidates = len(candidates)
    best_score = 0.0

    identity = _compute_identity(
        current_mutations,
        [candidate.name for candidate in candidates],
        checkpoint_context,
    )
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path, identity)
        if checkpoint:
            accepted = checkpoint["accepted"]
            active_mutations = checkpoint["mutations"]
            round_num = checkpoint["round"]
            rounds = checkpoint.get("rounds", [])
            best_score = checkpoint["best_score"]
            accepted_names = {sc.experiment.name for sc in accepted}
            remaining = [candidate for candidate in remaining if candidate.name not in accepted_names]
            log.info(
                "greedy_unpruned.resumed",
                round=round_num,
                remaining=len(remaining),
                accepted=len(accepted),
            )

    baseline_results = evaluate_fn([Experiment("__baseline__", {})], active_mutations)
    base_score = 0.0
    if baseline_results and not baseline_results[0].rejected:
        base_score = baseline_results[0].score
    if accepted:
        best_score = max(best_score, *(sc.score for sc in accepted), base_score)
    else:
        best_score = base_score

    while remaining and round_num < max_rounds:
        round_num += 1
        scored = evaluate_fn(remaining, active_mutations)

        viable = [sc for sc in scored if not sc.rejected]
        for sc in scored:
            if sc.experiment.name not in {item.experiment.name for item in accepted}:
                rejected_by_name[sc.experiment.name] = sc

        if not viable:
            rounds.append(
                GreedyRound(
                    round_num=round_num,
                    candidates_tested=len(scored),
                    best_name="(none)",
                    best_score=0.0,
                    best_delta_pct=0.0,
                    kept=False,
                    rejected_count=len(scored),
                )
            )
            break

        viable.sort(key=lambda sc: sc.score, reverse=True)
        best = viable[0]
        delta_pct = _delta_ratio(best.score, best_score) * 100.0
        kept = delta_pct >= (min_delta * 100.0)

        rounds.append(
            GreedyRound(
                round_num=round_num,
                candidates_tested=len(scored),
                best_name=best.experiment.name,
                best_score=best.score,
                best_delta_pct=delta_pct,
                kept=kept,
                rejected_count=len(scored) - len(viable),
            )
        )

        if not kept:
            break

        accepted.append(best)
        active_mutations = merge_mutations(active_mutations, best.experiment.mutations)
        best_score = best.score
        remaining = [c for c in remaining if c.name != best.experiment.name]
        rejected_by_name.pop(best.experiment.name, None)
        if checkpoint_path:
            _save_checkpoint(
                checkpoint_path,
                accepted,
                [],
                active_mutations,
                best_score,
                round_num,
                identity,
                rounds,
                checkpoint_context,
            )

    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            pass

    return GreedyResult(
        accepted_experiments=accepted,
        rejected_experiments=list(rejected_by_name.values()),
        final_mutations=active_mutations,
        final_score=best_score,
        rounds=rounds,
        base_score=base_score,
        kept_features=[sc.experiment.name for sc in accepted],
        total_candidates=total_candidates,
        accepted_count=len(accepted),
        elapsed_seconds=time.time() - start_time,
    )


class BreakoutRound4TradeFrequencyPlugin(BreakoutPlugin):
    """Single-phase greedy breakout optimizer biased toward returns + activity."""

    def __init__(
        self,
        *,
        backtest_config,
        base_config,
        data_dir: Path,
        source_output_dir: Path,
        max_workers: int | None = None,
    ) -> None:
        super().__init__(
            backtest_config=backtest_config,
            base_config=base_config,
            data_dir=data_dir,
            max_workers=max_workers,
        )
        self.source_output_dir = source_output_dir
        self._candidates, self._candidate_metadata = build_candidate_pool(
            source_output_dir,
            base_config,
        )

    @property
    def num_phases(self) -> int:
        return 1

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 6.0,
            "total_trades": 16.0,
            "profit_factor": 1.5,
            "max_drawdown_pct": 10.0,
            "sharpe_ratio": 1.2,
        }

    @property
    def candidate_metadata(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._candidate_metadata]

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        policy = PhaseAnalysisPolicy(
            max_scoring_retries=0,
            max_diagnostic_retries=0,
            focus_metrics=["net_return_pct", "total_trades"],
            diagnostic_gap_fn=lambda p, m: self._diagnostic_gap_fn(p, m),
            suggest_experiments_fn=lambda p, m, w, s: self._suggest_experiments_fn(p, m, w, s),
            decide_action_fn=lambda *args: self._decide_action_fn(*args),
            redesign_scoring_weights_fn=lambda *args: None,
            build_extra_analysis_fn=lambda p, m, s, g: self._build_extra_analysis_fn(p, m, s, g),
            format_extra_analysis_fn=lambda d: self._format_extra_analysis_fn(d),
        )

        return PhaseSpec(
            phase_num=phase,
            name="Round 3 Union Greedy",
            candidates=list(self._candidates),
            scoring_weights=dict(IMMUTABLE_SCORING_WEIGHTS),
            hard_rejects=dict(IMMUTABLE_HARD_REJECTS),
            min_delta=0.003,
            max_rounds=len(self._candidates),
            prune_threshold=-1.0,
            gate_criteria=list(IMMUTABLE_GATE_CRITERIA),
            gate_criteria_fn=lambda _m: list(IMMUTABLE_GATE_CRITERIA),
            analysis_policy=policy,
            focus="Return + Frequency Union",
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        scoring_weights: dict[str, float],
        hard_rejects: dict[str, tuple[str, float]],
    ) -> EvaluateFn:
        ceilings = IMMUTABLE_SCORING_CEILINGS

        def evaluate_fn(
            candidates: list[Experiment],
            current_mutations: dict[str, Any],
        ) -> list[ScoredCandidate]:
            return evaluate_parallel(
                candidates=candidates,
                current_mutations=current_mutations,
                cumulative_mutations=cumulative_mutations,
                base_config=self.base_config,
                backtest_config=self.backtest_config,
                data_dir=self.data_dir,
                scoring_weights=scoring_weights,
                hard_rejects=hard_rejects,
                phase=phase,
                max_workers=self.max_workers,
                strategy_type="breakout",
                ceilings=ceilings,
            )

        return evaluate_fn
