"""Dedicated momentum round-1 runner over the legacy round-3 endpoint union.

This canonical round 1 replay starts from the reconstructed pre-round-1
baseline, then runs a single unpruned greedy phase over only the final config
deltas implied by:

- output/momentum/round_3_momentum/optimized_config.json
- output/momentum/round_3/optimized_config.json

The candidate pool intentionally excludes any mutations that do not change the
pre-round-1 seed. That keeps the search faithful to "all mutations that make up
the optimized configs", including inherited prior-round values such as
``tp2_frac=0.4`` and ``risk_pct_b=0.014`` from the stronger
``round_3_momentum`` lineage.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
from crypto_trader.backtest.runner import run
from crypto_trader.cli import _configure_logging, _update_rounds_manifest
from crypto_trader.optimize.config_mutator import apply_mutations, merge_mutations
from crypto_trader.optimize.contracts import build_optimization_contract, run_optimization_preflight
from crypto_trader.optimize.evaluation import build_end_of_round_report
from crypto_trader.optimize.momentum_plugin import MomentumPlugin
from crypto_trader.optimize.parallel import evaluate_parallel
from crypto_trader.optimize.phase_analyzer import analyze_phase
from crypto_trader.optimize.phase_gates import evaluate_gate
from crypto_trader.optimize.phase_logging import PhaseLogger
from crypto_trader.optimize.phase_runner import PhaseRunner
from crypto_trader.optimize.phase_state import PhaseState, _atomic_write_json
from crypto_trader.optimize.scoring import composite_score
from crypto_trader.optimize.types import (
    EndOfRoundArtifacts,
    EvaluateFn,
    Experiment,
    GateCriterion,
    GreedyResult,
    GreedyRound,
    PhaseAnalysisPolicy,
    PhaseDecision,
    PhaseSpec,
    ScoredCandidate,
)
from crypto_trader.strategy.momentum.config import MomentumConfig

SYMBOLS: list[str] = ["BTC", "ETH", "SOL"]
TIMEFRAMES: list[str] = ["15m", "1h", "4h"]
SOURCE_ROUNDS: tuple[str, ...] = ("round_3_momentum", "round_3")
PRE_ROUND1_CONFIG_PATH = Path("config/momentum_round3_pre_round1.yaml")
OUTPUT_BASE = Path("output/momentum")
ROUND4_NAME = "round_1"
MAX_WORKERS = 2

# This score avoids the double-cap issue from round 3, where both endpoint
# configs saturated the returns and calmar dimensions. It stays immutable, but
# preserves enough headroom to distinguish a 17% return config from a 15% one.
ROUND4_SCORING_WEIGHTS: dict[str, float] = {
    "returns": 0.40,
    "edge": 0.25,
    "coverage": 0.15,
    "calmar": 0.10,
    "capture": 0.10,
}

ROUND4_SCORING_CEILINGS: dict[str, float] = {
    "returns": 20.0,
    "edge": 3.5,
    "coverage": 36.0,
    "calmar": 5.0,
}

ROUND4_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "max_drawdown_pct": ("<=", 50.0),
    "total_trades": (">=", 12.0),
    "profit_factor": (">=", 0.8),
}

ROUND4_GATE_CRITERIA: list[GateCriterion] = [
    GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=50.0),
    GateCriterion(metric="total_trades", operator=">=", threshold=12.0),
    GateCriterion(metric="profit_factor", operator=">=", threshold=0.8),
]

KEY_PRIORITY: dict[str, int] = {
    "trail.trail_buffer_tight": 10,
    "trail.trail_buffer_wide": 11,
    "trail.trail_activation_bars": 12,
    "trail.trail_r_ceiling": 13,
    "exits.soft_time_stop_min_r": 20,
    "exits.tp2_frac": 21,
    "setup.fib_high": 30,
    "setup.min_confluences_b": 31,
    "bias.min_1h_conditions": 40,
    "reentry.enabled": 41,
    "risk.risk_pct_b": 50,
}


def detect_common_window(
    data_dir: Path,
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> tuple[datetime, datetime]:
    """Return the common UTC window across candles and funding."""
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES

    per_symbol_starts: list[int] = []
    per_symbol_ends: list[int] = []

    for symbol in symbols:
        starts: list[int] = []
        ends: list[int] = []

        for timeframe in timeframes:
            path = data_dir / "candles" / symbol / f"{timeframe}.parquet"
            frame = pd.read_parquet(path, columns=["ts"])
            starts.append(int(frame["ts"].min()))
            ends.append(int(frame["ts"].max()))

        funding_path = data_dir / "funding" / f"{symbol}.parquet"
        funding = pd.read_parquet(funding_path, columns=["ts"])
        starts.append(int(funding["ts"].min()))
        ends.append(int(funding["ts"].max()))

        per_symbol_starts.append(max(starts))
        per_symbol_ends.append(min(ends))

    start_ms = max(per_symbol_starts)
    end_ms = min(per_symbol_ends)
    return (
        datetime.fromtimestamp(start_ms / 1000, tz=UTC),
        datetime.fromtimestamp(end_ms / 1000, tz=UTC),
    )


def build_pre_round1_config(config_path: Path = PRE_ROUND1_CONFIG_PATH) -> MomentumConfig:
    """Load the reconstructed pre-round-1 seed config."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return MomentumConfig.from_dict(raw.get("strategy", {}))


def build_backtest_config(data_dir: Path) -> tuple[BacktestConfig, dict[str, str]]:
    """Build a full-span backtest config for the current shared data window."""
    start_dt, end_dt = detect_common_window(data_dir)
    bt_cfg = build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=list(SYMBOLS),
        start_date=start_dt.date(),
        end_date=end_dt.date(),
    )
    metadata = {
        "common_start_utc": start_dt.isoformat(),
        "common_end_utc": end_dt.isoformat(),
        "start_date": bt_cfg.start_date.isoformat(),
        "end_date": bt_cfg.end_date.isoformat(),
    }
    return bt_cfg, metadata


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, sub_value in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            _flatten(next_prefix, sub_value, out)
    else:
        out[prefix] = value


def _sanitize_token(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text.replace("-", "neg_").replace(".", "_")
    return str(value).replace("-", "neg_").replace(".", "_").replace(" ", "_")


def _candidate_name(source: str, key: str, value: Any, shared: bool) -> str:
    prefix = "shared" if shared else ("r3m" if source == "round_3_momentum" else "r3")
    return f"{prefix}_{key.replace('.', '_')}__{_sanitize_token(value)}"


def _load_round_metrics(round_dir: Path) -> dict[str, float]:
    state = json.loads((round_dir / "phase_state.json").read_text(encoding="utf-8"))
    return state["phase_metrics"][str(max(int(p) for p in state["phase_metrics"]))]


def _load_round_diffs(
    base_config: MomentumConfig,
    round_dir: Path,
) -> dict[str, Any]:
    config_data = json.loads((round_dir / "optimized_config.json").read_text(encoding="utf-8"))
    final_config = MomentumConfig.from_dict(config_data["strategy"])

    base_flat: dict[str, Any] = {}
    final_flat: dict[str, Any] = {}
    _flatten("", base_config.to_dict(), base_flat)
    _flatten("", final_config.to_dict(), final_flat)

    return {
        key: final_flat[key]
        for key in sorted(final_flat)
        if base_flat.get(key) != final_flat[key]
    }


def build_union_candidates(
    base_config: MomentumConfig,
    output_base: Path = OUTPUT_BASE,
) -> tuple[list[Experiment], list[dict[str, Any]], dict[str, Any]]:
    """Build an ordered union of endpoint config deltas from the legacy variants."""
    source_diffs: dict[str, dict[str, Any]] = {}
    source_scores: dict[str, dict[str, Any]] = {}

    for source in SOURCE_ROUNDS:
        round_dir = output_base / source
        source_diffs[source] = _load_round_diffs(base_config, round_dir)
        metrics = _load_round_metrics(round_dir)
        score, rejected, reject_reason = composite_score(
            metrics,
            ROUND4_SCORING_WEIGHTS,
            ROUND4_HARD_REJECTS,
            ceilings=ROUND4_SCORING_CEILINGS,
        )
        source_scores[source] = {
            "score": score,
            "rejected": rejected,
            "reject_reason": reject_reason,
            "metrics": metrics,
        }

    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for source in SOURCE_ROUNDS:
        for key, value in source_diffs[source].items():
            token = (key, json.dumps(value, sort_keys=True, default=str))
            entry = seen.get(token)
            if entry is None:
                seen[token] = {
                    "key": key,
                    "value": value,
                    "sources": [source],
                }
            else:
                entry["sources"].append(source)

    metadata: list[dict[str, Any]] = []
    for source in SOURCE_ROUNDS:
        for key, value in source_diffs[source].items():
            token = (key, json.dumps(value, sort_keys=True, default=str))
            entry = seen[token]
            if entry.get("primary_source") is None:
                entry["primary_source"] = source
                entry["shared"] = len(entry["sources"]) > 1
                entry["order_rank"] = (
                    KEY_PRIORITY.get(key, 999),
                    SOURCE_ROUNDS.index(source),
                    key,
                    json.dumps(value, sort_keys=True, default=str),
                )
                entry["name"] = _candidate_name(source, key, value, entry["shared"])
                metadata.append(entry)

    metadata.sort(key=lambda item: item["order_rank"])
    candidates = [
        Experiment(item["name"], {item["key"]: item["value"]})
        for item in metadata
    ]

    candidate_info = [
        {
            "name": item["name"],
            "mutation": {item["key"]: item["value"]},
            "sources": item["sources"],
            "shared": item["shared"],
            "order_rank": list(item["order_rank"][:2]),
        }
        for item in metadata
    ]
    return candidates, candidate_info, source_scores


def run_greedy_unpruned(
    candidates: list[Experiment],
    current_mutations: dict[str, Any],
    evaluate_fn: EvaluateFn,
    *,
    min_delta: float = 0.001,
    logger: PhaseLogger | None = None,
    phase: int = 1,
) -> GreedyResult:
    """Run a plain greedy pass without delta or streak pruning.

    The canonical round-1 candidate pool is intentionally tiny and contains overrides that
    can become useful only after earlier selections. Keeping every viable
    candidate alive avoids throwing away later-beneficial endpoint mutations.
    """
    remaining = list(candidates)
    accepted: list[ScoredCandidate] = []
    rejected: list[ScoredCandidate] = []
    active_mutations = dict(current_mutations)
    rounds: list[GreedyRound] = []

    baseline = evaluate_fn([Experiment("__baseline__", {})], active_mutations)
    base_score = 0.0
    if baseline and not baseline[0].rejected:
        base_score = baseline[0].score
    best_score = base_score
    round_num = 0

    while remaining:
        round_num += 1
        scored = evaluate_fn(remaining, active_mutations)
        viable = [sc for sc in scored if not sc.rejected]
        round_rejected = [sc for sc in scored if sc.rejected]
        rejected.extend(round_rejected)

        rejected_names = {sc.experiment.name for sc in round_rejected}
        remaining = [c for c in remaining if c.name not in rejected_names]

        if not viable:
            rounds.append(
                GreedyRound(
                    round_num=round_num,
                    candidates_tested=len(scored),
                    best_name="(none)",
                    best_score=0.0,
                    best_delta_pct=0.0,
                    kept=False,
                    rejected_count=len(round_rejected),
                )
            )
            break

        viable.sort(key=lambda sc: sc.score, reverse=True)
        best = viable[0]
        delta_pct = (
            ((best.score - best_score) / best_score) * 100.0
            if best_score > 0
            else (best.score - best_score) * 100.0
        )
        kept = delta_pct >= (min_delta * 100.0)

        rounds.append(
            GreedyRound(
                round_num=round_num,
                candidates_tested=len(scored),
                best_name=best.experiment.name,
                best_score=best.score,
                best_delta_pct=delta_pct,
                kept=kept,
                rejected_count=len(round_rejected),
            )
        )
        if logger is not None:
            logger.log_greedy_round(
                phase,
                round_num,
                best.experiment.name,
                best.score,
                delta_pct,
            )

        if not kept:
            break

        accepted.append(best)
        active_mutations = merge_mutations(active_mutations, best.experiment.mutations)
        best_score = best.score
        remaining = [c for c in remaining if c.name != best.experiment.name]

    return GreedyResult(
        accepted_experiments=accepted,
        rejected_experiments=rejected,
        final_mutations=active_mutations,
        final_score=best_score,
        rounds=rounds,
        base_score=base_score,
        kept_features=[sc.experiment.name for sc in accepted],
        total_candidates=len(candidates),
        accepted_count=len(accepted),
    )


class MomentumRound4UnionPlugin(MomentumPlugin):
    """Single-phase momentum plugin restricted to the two round-3 endpoint deltas."""

    def __init__(
        self,
        backtest_config: BacktestConfig,
        base_config: MomentumConfig,
        candidates: list[Experiment],
        data_dir: Path,
        max_workers: int = MAX_WORKERS,
    ) -> None:
        super().__init__(backtest_config, base_config, data_dir=data_dir, max_workers=max_workers)
        self._candidates = candidates

    @property
    def name(self) -> str:
        return "momentum_pullback_round4_union"

    @property
    def num_phases(self) -> int:
        return 1

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 15.0,
            "total_trades": 24.0,
            "profit_factor": 3.0,
            "max_drawdown_pct": 10.0,
            "sharpe_ratio": 4.0,
            "calmar_ratio": 4.0,
        }

    def _decide_single_phase_action(
        self,
        phase: int,
        metrics: dict[str, float],
        state: Any,
        greedy_result: GreedyResult,
        gate_result: Any,
        current_weights: dict[str, float],
        goal_progress: dict[str, dict[str, Any]],
        max_scoring: int,
        max_diag: int,
    ) -> PhaseDecision:
        if gate_result.passed:
            reason = "Gate passed."
        elif greedy_result.accepted_count:
            reason = "Single-phase union run complete; no retries configured."
        else:
            reason = "No union mutations improved the score."
        return PhaseDecision(action="advance", reason=reason)

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        policy = PhaseAnalysisPolicy(
            max_scoring_retries=0,
            max_diagnostic_retries=0,
            focus_metrics=["net_return_pct", "profit_factor", "calmar_ratio", "total_trades"],
            diagnostic_gap_fn=lambda p, m: self._diagnostic_gap_fn(p, m),
            decide_action_fn=lambda *args: self._decide_single_phase_action(*args),
            redesign_scoring_weights_fn=lambda *args: None,
            build_extra_analysis_fn=lambda p, m, s, g: self._build_extra_analysis_fn(p, m, s, g),
            format_extra_analysis_fn=lambda d: self._format_extra_analysis_fn(d),
        )
        return PhaseSpec(
            phase_num=1,
            name="Round 1 Mutation Union",
            candidates=list(self._candidates),
            scoring_weights=dict(ROUND4_SCORING_WEIGHTS),
            hard_rejects=dict(ROUND4_HARD_REJECTS),
            gate_criteria=list(ROUND4_GATE_CRITERIA),
            gate_criteria_fn=lambda _m: list(ROUND4_GATE_CRITERIA),
            analysis_policy=policy,
            focus="Round 1 Mutation Union",
            min_delta=0.001,
            max_rounds=len(self._candidates),
            prune_threshold=0.0,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        scoring_weights: dict[str, float],
        hard_rejects: dict[str, tuple[str, float]],
    ) -> EvaluateFn:
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
                ceilings=ROUND4_SCORING_CEILINGS,
            )

        return evaluate_fn


def _phase_result_dict(
    focus: str,
    base_mutations: dict[str, Any],
    greedy_result: GreedyResult,
    metrics: dict[str, float],
    contract: dict[str, Any],
) -> dict[str, Any]:
    new_mutations = {
        key: value
        for key, value in greedy_result.final_mutations.items()
        if base_mutations.get(key) != value
    }
    return {
        "focus": focus,
        "base_mutations": dict(base_mutations),
        "final_mutations": dict(greedy_result.final_mutations),
        "base_score": greedy_result.base_score,
        "final_score": greedy_result.final_score,
        "kept_features": list(greedy_result.kept_features),
        "rounds": [
            {
                "round_num": round_data.round_num,
                "best_name": round_data.best_name,
                "best_score": round_data.best_score,
                "kept": round_data.kept,
            }
            for round_data in greedy_result.rounds
        ],
        "final_metrics": metrics,
        "contract_hash": contract.get("contract_hash", ""),
        "contract": contract,
        "accepted_count": greedy_result.accepted_count,
        "new_mutations": new_mutations,
        "suggested_experiments": [],
    }


def _write_metadata(
    output_dir: Path,
    *,
    window_metadata: dict[str, str],
    candidate_info: list[dict[str, Any]],
    source_scores: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    metadata = {
        "window": window_metadata,
        "contract_hash": contract.get("contract_hash", ""),
        "profile_hash": contract.get("profile_hash", ""),
        "contract": contract,
        "scoring_weights": ROUND4_SCORING_WEIGHTS,
        "scoring_ceilings": ROUND4_SCORING_CEILINGS,
        "hard_rejects": ROUND4_HARD_REJECTS,
        "source_scores": source_scores,
        "candidate_union": candidate_info,
    }
    _atomic_write_json(metadata, output_dir / "round_1_metadata.json")


def run_round4(
    *,
    data_dir: Path = Path("data"),
    output_base: Path = OUTPUT_BASE,
    max_workers: int = MAX_WORKERS,
) -> Path:
    """Execute the dedicated round-1 union run and return its output directory."""
    if not PRE_ROUND1_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing seed config: {PRE_ROUND1_CONFIG_PATH}")

    round_dir = output_base / ROUND4_NAME
    if round_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing round directory: {round_dir}")
    round_dir.mkdir(parents=True, exist_ok=True)

    base_config = build_pre_round1_config()
    backtest_config, window_metadata = build_backtest_config(data_dir)
    candidates, candidate_info, source_scores = build_union_candidates(base_config, output_base)
    if not candidates:
        raise RuntimeError("No union candidates were produced for round 1.")

    plugin = MomentumRound4UnionPlugin(
        backtest_config,
        base_config,
        candidates,
        data_dir=data_dir,
        max_workers=max_workers,
    )
    contract = build_optimization_contract(
        strategy_type="momentum",
        strategy_config=base_config,
        backtest_config=backtest_config,
        data_dir=data_dir,
        profile=LIVE_PARITY_PROFILE,
        plugin=plugin,
    )
    run_optimization_preflight(
        contract=contract,
        backtest_config=backtest_config,
        data_dir=data_dir,
        output_dir=round_dir,
        profile=LIVE_PARITY_PROFILE,
    )
    phase_logger = PhaseLogger(round_dir)
    state_path = round_dir / "phase_state.json"
    state = PhaseState(_path=state_path)
    state.ensure_contract(contract)
    state.start_phase(1)
    state.save(state_path)

    spec = plugin.get_phase_spec(1, state)
    phase_logger.log_phase_start(1, spec.name, len(spec.candidates))

    evaluate_fn = plugin.create_evaluate_batch(
        1,
        state.cumulative_mutations,
        scoring_weights=spec.scoring_weights,
        hard_rejects=spec.hard_rejects,
    )
    greedy_result = run_greedy_unpruned(
        spec.candidates,
        state.cumulative_mutations,
        evaluate_fn,
        min_delta=spec.min_delta,
        logger=phase_logger,
        phase=1,
    )

    metrics = plugin.compute_final_metrics(greedy_result.final_mutations)
    greedy_result.final_metrics = metrics

    gate_result = evaluate_gate(spec.gate_criteria, greedy_result)
    state.record_gate(
        1,
        {
            "passed": gate_result.passed,
            "failure_reasons": gate_result.failure_reasons,
            "failure_category": gate_result.failure_category,
        },
    )

    for scored in greedy_result.accepted_experiments:
        phase_logger.log_experiment_result(
            1,
            scored.experiment.name,
            scored.score,
            accepted=True,
        )
    for scored in greedy_result.rejected_experiments:
        phase_logger.log_experiment_result(
            1,
            scored.experiment.name,
            scored.score,
            accepted=False,
            rejected=scored.rejected,
            reject_reason=scored.reject_reason,
        )

    phase_logger.save_phase_output(
        1,
        "greedy",
        {
            "accepted": [sc.experiment.name for sc in greedy_result.accepted_experiments],
            "rejected": [sc.experiment.name for sc in greedy_result.rejected_experiments],
            "final_score": greedy_result.final_score,
            "base_score": greedy_result.base_score,
            "rounds": len(greedy_result.rounds),
            "elapsed_seconds": greedy_result.elapsed_seconds,
        },
    )

    phase_logger.log_gate_result(
        1,
        gate_result.passed,
        gate_result.failure_reasons,
        failure_category=gate_result.failure_category,
    )

    diagnostics_text = plugin.run_phase_diagnostics(1, state, metrics, greedy_result)
    phase_logger.save_phase_output(1, "diagnostics", diagnostics_text)

    analysis = analyze_phase(
        1,
        greedy_result,
        metrics,
        state,
        gate_result,
        ultimate_targets=plugin.ultimate_targets,
        policy=spec.analysis_policy,
        current_weights=spec.scoring_weights,
        max_scoring_retries=0,
        max_diagnostic_retries=0,
    )
    phase_logger.log_analysis(1, analysis.recommendation, analysis.summary)
    phase_logger.save_phase_output(1, "analysis", analysis.report or analysis.summary)

    phase_result = _phase_result_dict(
        spec.focus or spec.name,
        state.cumulative_mutations,
        greedy_result,
        metrics,
        contract,
    )
    state.advance_phase(1, greedy_result.final_mutations, phase_result)
    state.complete_phase(1)
    state.save(state_path)

    phase_logger.log_phase_end(
        1,
        spec.name,
        accepted=len(greedy_result.accepted_experiments),
        final_score=greedy_result.final_score,
        metrics=metrics,
    )
    phase_logger.update_progress(
        1,
        {
            "name": spec.name,
            "accepted": len(greedy_result.accepted_experiments),
            "final_score": greedy_result.final_score,
            "gate_passed": gate_result.passed,
        },
    )

    _write_metadata(
        round_dir,
        window_metadata=window_metadata,
        candidate_info=candidate_info,
        source_scores=source_scores,
        contract=contract,
    )

    runner = PhaseRunner(plugin, round_dir, contract=contract)
    runner.run_end_of_round(state)

    final_metrics = state.phase_metrics.get(1)
    _update_rounds_manifest(
        output_base,
        1,
        state.cumulative_mutations,
        final_metrics,
        contract=contract,
        phase_result=phase_result,
        gate_result=state.phase_gate_results.get(1, {}),
    )
    phase_logger.close()

    return round_dir


def main() -> None:
    _configure_logging()
    round_dir = run_round4()
    state = PhaseState.load(round_dir / "phase_state.json")
    final_metrics = state.phase_metrics[1]

    print("\n=== Optimization Complete (Round 1) ===")
    print(f"Output: {round_dir}")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Total mutations: {len(state.cumulative_mutations)}")
    for key, value in state.cumulative_mutations.items():
        print(f"  {key} = {value}")

    print("\nFinal metrics:")
    for metric in (
        "net_return_pct",
        "total_trades",
        "win_rate",
        "profit_factor",
        "max_drawdown_pct",
        "sharpe_ratio",
        "calmar_ratio",
    ):
        print(f"  {metric}: {final_metrics.get(metric, 0):.4f}")


if __name__ == "__main__":
    main()
