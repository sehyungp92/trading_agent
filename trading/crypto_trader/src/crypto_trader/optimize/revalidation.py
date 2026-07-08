"""Revalidation helpers for manifest replay, ablation, perturbation, and reruns."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date
import json
from pathlib import Path
from typing import Any

import yaml

from crypto_trader.backtest.analysis import (
    export_equity_curve,
    export_trade_journal,
    generate_report,
)
from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.metrics import metrics_to_dict
from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
from crypto_trader.backtest.runner import BacktestResult, run
from crypto_trader.data.store import ParquetStore
from crypto_trader.optimize.breakout_round3_pre_round1 import (
    build_pre_round1_config as build_breakout_pre_round1_config,
)
from crypto_trader.optimize.breakout_round4_trade_frequency import (
    run_greedy_without_pruning,
)
from crypto_trader.optimize.breakout_round6_phased import (
    BreakoutRound6PhasedPlugin,
    ROUND6_HARD_REJECTS,
    ROUND6_IMMUTABLE_SCORING_CEILINGS,
    ROUND6_IMMUTABLE_SCORING_WEIGHTS,
    ROUND6_PHASE_GATE_CRITERIA,
)
from crypto_trader.optimize.config_mutator import apply_mutations
from crypto_trader.optimize.contracts import build_optimization_contract, run_optimization_preflight
from crypto_trader.optimize.momentum_round4_union import (
    build_pre_round1_config as build_momentum_pre_round1_config,
)
from crypto_trader.optimize.momentum_round5_phased import (
    IMMUTABLE_HARD_REJECTS as MOMENTUM_HARD_REJECTS,
    IMMUTABLE_SCORING_CEILINGS as MOMENTUM_SCORING_CEILINGS,
    IMMUTABLE_SCORING_WEIGHTS as MOMENTUM_SCORING_WEIGHTS,
    MomentumRound5PhasedPlugin,
    PHASE_GATE_CRITERIA as MOMENTUM_PHASE_GATE_CRITERIA,
)
from crypto_trader.optimize.phase_gates import evaluate_gate
import crypto_trader.optimize.phase_runner as phase_runner_module
from crypto_trader.optimize.phase_runner import PhaseRunner
from crypto_trader.optimize.phase_state import PhaseState, _atomic_write_json
from crypto_trader.optimize.scoring import composite_score
from crypto_trader.optimize.trend_round7_plugin import (
    ROUND7_HARD_REJECTS,
    ROUND7_IMMUTABLE_SCORING_CEILINGS,
    ROUND7_PHASE_GATE_CRITERIA,
    ROUND7_SCORING_WEIGHTS,
    Round7TrendPlugin,
)
from crypto_trader.optimize.types import GateCriterion, GreedyResult
from crypto_trader.strategy.breakout.config import BreakoutConfig
from crypto_trader.strategy.momentum.config import MomentumConfig
from crypto_trader.strategy.trend.config import TrendConfig


MetricDict = dict[str, float]

_MOMENTUM_SPEC_PATH = Path("output/momentum/round_2/run_spec.json")
_TREND_SPEC_PATH = Path("output/trend/round_3/round3_context.json")
_BREAKOUT_SPEC_PATH = Path("output/breakout/round_3/run_spec.json")

_MOMENTUM_MANIFEST_PATH = Path("output/momentum/rounds_manifest.json")
_TREND_MANIFEST_PATH = Path("output/trend/rounds_manifest.json")
_BREAKOUT_MANIFEST_PATH = Path("output/breakout/rounds_manifest.json")

_TREND_PRE_ROUND1_PATH = Path("config/trend_pre_round1.yaml")

_SOURCE_METRIC_KEYS = (
    "total_trades",
    "win_rate",
    "profit_factor",
    "max_drawdown_pct",
    "sharpe_ratio",
    "calmar_ratio",
    "net_return_pct",
    "expectancy_r",
    "exit_efficiency",
    "realized_pnl_net",
    "terminal_mark_pnl_net",
    "net_profit",
    "total_fees",
    "funding_cost_total",
    "terminal_mark_count",
)

_MOMENTUM_STRUCTURAL_PREFIXES = (
    "setup.",
    "bias.",
    "confirmation.",
    "entry.",
    "session.",
    "filters.",
    "symbol_filter.",
    "reentry.",
)


@dataclass
class EvaluationSnapshot:
    """Serializable evaluation output for one mutation set."""

    label: str
    mutations: dict[str, Any]
    score: float
    rejected: bool
    reject_reason: str
    metrics: MetricDict
    phase_gates: dict[str, dict[str, Any]]
    final_phase: int
    final_phase_gate_passed: bool
    contract_hash: str = ""
    profile_hash: str = ""
    strategy_config_hash: str = ""
    portfolio_config_hash: str = ""
    data_window: dict[str, Any] = field(default_factory=dict)
    contract: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mutation_count"] = len(self.mutations)
        return payload


@dataclass
class EvaluationBundle:
    """Backtest result plus serializable snapshot."""

    snapshot: EvaluationSnapshot
    result: BacktestResult
    config: Any


@dataclass
class ManifestCheckpoint:
    """One cumulative checkpoint from a rounds manifest."""

    label: str
    mutations: dict[str, Any]
    manifest_round: int | None
    source_manifest_metrics: dict[str, float] | None = None


@dataclass
class StrategyRevalidationContext:
    """Strategy-specific runtime and scoring context."""

    root: Path
    strategy: str
    baseline_config: Any
    backtest_config: BacktestConfig
    symbols: list[str]
    manifest_path: Path
    scoring_weights: dict[str, float]
    scoring_ceilings: dict[str, float]
    hard_rejects: dict[str, tuple[str, float]]
    phase_gate_criteria: dict[int, list[GateCriterion]]
    spec_path: Path
    spec_payload: dict[str, Any]
    store: ParquetStore
    max_workers: int = 2
    contract: dict[str, Any] = field(default_factory=dict)

    @property
    def final_phase(self) -> int:
        return max(self.phase_gate_criteria)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(payload, path)


def normalize_hard_rejects(raw: dict[str, Any]) -> dict[str, tuple[str, float]]:
    """Normalize hard rejects from JSON artifacts into internal tuple form."""
    normalized: dict[str, tuple[str, float]] = {}
    for metric, value in raw.items():
        if isinstance(value, dict):
            normalized[metric] = (str(value["operator"]), float(value["threshold"]))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            normalized[metric] = (str(value[0]), float(value[1]))
        else:
            raise TypeError(f"Unsupported hard reject format for {metric!r}: {value!r}")
    return normalized


def parse_gate_criteria(raw: dict[str, Any]) -> dict[int, list[GateCriterion]]:
    """Parse phase gate criteria from JSON artifacts."""
    parsed: dict[int, list[GateCriterion]] = {}
    for phase_key, criteria in raw.items():
        phase = int(phase_key)
        parsed[phase] = [
            GateCriterion(
                metric=str(item["metric"]),
                operator=str(item["operator"]),
                threshold=float(item["threshold"]),
                weight=float(item.get("weight", 1.0)),
            )
            for item in criteria
        ]
    return parsed


def _profile_warmup_days(spec_payload: dict[str, Any]) -> int:
    return max(
        LIVE_PARITY_PROFILE.warmup_days,
        int(spec_payload.get("warmup_days", LIVE_PARITY_PROFILE.warmup_days)),
    )


def _build_revalidation_contract(
    *,
    root: Path,
    strategy: str,
    strategy_config: Any,
    backtest_config: BacktestConfig,
    scoring_weights: dict[str, float],
    scoring_ceilings: dict[str, float],
    hard_rejects: dict[str, tuple[str, float]],
    phase_gate_criteria: dict[int, list[GateCriterion]],
) -> dict[str, Any]:
    return build_optimization_contract(
        strategy_type=strategy,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        data_dir=root / "data",
        profile=LIVE_PARITY_PROFILE,
        scoring_weights=scoring_weights,
        scoring_ceilings=scoring_ceilings,
        hard_rejects=hard_rejects,
        gate_criteria=phase_gate_criteria,
    )


def build_manifest_checkpoints(manifest: dict[str, Any]) -> list[ManifestCheckpoint]:
    """Build cumulative checkpoints from a rounds manifest."""
    checkpoints: list[ManifestCheckpoint] = [
        ManifestCheckpoint(label="base", mutations={}, manifest_round=None),
    ]

    cumulative: dict[str, Any] = {}
    for round_entry in sorted(manifest.get("rounds", []), key=lambda item: int(item["round"])):
        cumulative = dict(cumulative)
        cumulative.update(round_entry.get("mutations", {}))
        source_metrics = {
            key: float(round_entry[key])
            for key in _SOURCE_METRIC_KEYS
            if key in round_entry
        }
        checkpoints.append(
            ManifestCheckpoint(
                label=f"round_{int(round_entry['round'])}_cumulative",
                mutations=dict(cumulative),
                manifest_round=int(round_entry["round"]),
                source_manifest_metrics=source_metrics or None,
            )
        )

    return checkpoints


def local_perturbation_values(key: str, value: Any) -> list[Any]:
    """Generate a small local perturbation sweep around one numeric mutation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return []

    generated: list[Any] = []

    if isinstance(value, int):
        steps = [-2, -1, 1, 2]
        if any(token in key for token in ("min_confluences", "max_reentries", "min_touches")):
            steps = [-1, 1]
        for step in steps:
            candidate = value + step
            if candidate <= 0:
                continue
            if candidate != value and candidate not in generated:
                generated.append(candidate)
        return generated

    absolute_steps: list[float] = []
    if any(token in key for token in ("tp1_frac", "tp2_frac", "runner_frac")):
        absolute_steps = [-0.10, -0.05, 0.05, 0.10]
    elif "risk_pct" in key:
        absolute_steps = [-0.003, -0.0015, 0.0015, 0.003]
    elif any(
        token in key
        for token in (
            "trail_buffer",
            "trail_activation_r",
            "scratch_peak_r",
            "scratch_floor_r",
            "be_buffer_r",
            "body_ratio",
            "relaxed_body_min",
            "relaxed_body_risk_scale",
            "fib_",
            "min_room",
            "room_r",
            "volume_ratio",
            "adx",
            "atr_mult",
            "time_stop_min_progress_r",
            "trail_r_ceiling",
        )
    ):
        absolute_steps = [-0.10, -0.05, 0.05, 0.10]

    if absolute_steps:
        for step in absolute_steps:
            candidate = round(float(value) + step, 6)
            if candidate <= 0:
                continue
            if any(token in key for token in ("tp1_frac", "tp2_frac", "runner_frac", "body_ratio", "relaxed_body_min", "fib_")):
                if candidate <= 0 or candidate >= 1:
                    continue
            if candidate != value and candidate not in generated:
                generated.append(candidate)
        return generated

    for multiplier in (0.90, 0.95, 1.05, 1.10):
        candidate = round(float(value) * multiplier, 6)
        if candidate <= 0:
            continue
        if candidate != value and candidate not in generated:
            generated.append(candidate)
    return generated


def build_momentum_cleaned_seed_mutations(
    winner_mutations: dict[str, Any],
    ablation_support: dict[str, bool],
) -> dict[str, Any]:
    """Keep only structural momentum mutations that still earn their place."""
    cleaned: dict[str, Any] = {}
    for key, value in winner_mutations.items():
        if not key.startswith(_MOMENTUM_STRUCTURAL_PREFIXES):
            continue
        if ablation_support.get(key, True):
            cleaned[key] = value
    return cleaned


def load_strategy_revalidation_context(
    root: Path,
    strategy: str,
) -> StrategyRevalidationContext:
    """Load baseline config plus current scoring/gate context for one strategy."""
    if strategy == "momentum":
        spec_path = root / _MOMENTUM_SPEC_PATH
        manifest_path = root / _MOMENTUM_MANIFEST_PATH
        spec_payload = _load_json(spec_path)
        baseline_config = build_momentum_pre_round1_config(root / "config" / "momentum_round3_pre_round1.yaml")
        backtest_config = build_backtest_config_from_profile(
            profile=LIVE_PARITY_PROFILE,
            symbols=list(spec_payload["symbols"]),
            start_date=date.fromisoformat(spec_payload["window"]["start_date"]),
            end_date=date.fromisoformat(spec_payload["window"]["end_date"]),
            warmup_days=_profile_warmup_days(spec_payload),
        )
        scoring_weights = dict(spec_payload["immutable_scoring_weights"])
        scoring_ceilings = dict(spec_payload["immutable_scoring_ceilings"])
        hard_rejects = normalize_hard_rejects(spec_payload["hard_rejects"])
        phase_gate_criteria = parse_gate_criteria(spec_payload["phase_gate_criteria"])
        contract = _build_revalidation_contract(
            root=root,
            strategy=strategy,
            strategy_config=baseline_config,
            backtest_config=backtest_config,
            scoring_weights=scoring_weights,
            scoring_ceilings=scoring_ceilings,
            hard_rejects=hard_rejects,
            phase_gate_criteria=phase_gate_criteria,
        )
        run_optimization_preflight(
            contract=contract,
            backtest_config=backtest_config,
            data_dir=root / "data",
            profile=LIVE_PARITY_PROFILE,
        )
        return StrategyRevalidationContext(
            root=root,
            strategy=strategy,
            baseline_config=baseline_config,
            backtest_config=backtest_config,
            symbols=list(spec_payload["symbols"]),
            manifest_path=manifest_path,
            scoring_weights=scoring_weights,
            scoring_ceilings=scoring_ceilings,
            hard_rejects=hard_rejects,
            phase_gate_criteria=phase_gate_criteria,
            spec_path=spec_path,
            spec_payload=spec_payload,
            store=ParquetStore(base_dir=root / "data"),
            max_workers=int(spec_payload.get("max_workers", 2)),
            contract=contract,
        )

    if strategy == "trend":
        spec_path = root / _TREND_SPEC_PATH
        manifest_path = root / _TREND_MANIFEST_PATH
        spec_payload = _load_json(spec_path)
        with open(root / _TREND_PRE_ROUND1_PATH, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        baseline_config = TrendConfig.from_dict(raw.get("strategy", raw))
        backtest_config = build_backtest_config_from_profile(
            profile=LIVE_PARITY_PROFILE,
            symbols=list(spec_payload["symbols"]),
            start_date=date.fromisoformat(spec_payload["measurement_start"]),
            end_date=date.fromisoformat(spec_payload["measurement_end"]),
            warmup_days=_profile_warmup_days(spec_payload),
        )
        scoring_weights = dict(spec_payload["scoring_weights"])
        scoring_ceilings = dict(spec_payload["immutable_scoring_ceilings"])
        hard_rejects = normalize_hard_rejects(spec_payload["hard_rejects"])
        phase_gate_criteria = parse_gate_criteria(spec_payload["phase_gate_criteria"])
        contract = _build_revalidation_contract(
            root=root,
            strategy=strategy,
            strategy_config=baseline_config,
            backtest_config=backtest_config,
            scoring_weights=scoring_weights,
            scoring_ceilings=scoring_ceilings,
            hard_rejects=hard_rejects,
            phase_gate_criteria=phase_gate_criteria,
        )
        run_optimization_preflight(
            contract=contract,
            backtest_config=backtest_config,
            data_dir=root / "data",
            profile=LIVE_PARITY_PROFILE,
        )
        return StrategyRevalidationContext(
            root=root,
            strategy=strategy,
            baseline_config=baseline_config,
            backtest_config=backtest_config,
            symbols=list(spec_payload["symbols"]),
            manifest_path=manifest_path,
            scoring_weights=scoring_weights,
            scoring_ceilings=scoring_ceilings,
            hard_rejects=hard_rejects,
            phase_gate_criteria=phase_gate_criteria,
            spec_path=spec_path,
            spec_payload=spec_payload,
            store=ParquetStore(base_dir=root / "data"),
            max_workers=int(spec_payload.get("max_workers", 2)),
            contract=contract,
        )

    if strategy == "breakout":
        spec_path = root / _BREAKOUT_SPEC_PATH
        manifest_path = root / _BREAKOUT_MANIFEST_PATH
        spec_payload = _load_json(spec_path)
        baseline_config = build_breakout_pre_round1_config()
        backtest_config = build_backtest_config_from_profile(
            profile=LIVE_PARITY_PROFILE,
            symbols=list(spec_payload["symbols"]),
            start_date=date.fromisoformat(spec_payload["window"]["start_date"]),
            end_date=date.fromisoformat(spec_payload["window"]["end_date"]),
            warmup_days=_profile_warmup_days(spec_payload),
        )
        scoring_weights = dict(spec_payload["immutable_scoring_weights"])
        scoring_ceilings = dict(spec_payload["immutable_scoring_ceilings"])
        hard_rejects = normalize_hard_rejects(spec_payload["hard_rejects"])
        phase_gate_criteria = parse_gate_criteria(spec_payload["phase_gate_criteria"])
        contract = _build_revalidation_contract(
            root=root,
            strategy=strategy,
            strategy_config=baseline_config,
            backtest_config=backtest_config,
            scoring_weights=scoring_weights,
            scoring_ceilings=scoring_ceilings,
            hard_rejects=hard_rejects,
            phase_gate_criteria=phase_gate_criteria,
        )
        run_optimization_preflight(
            contract=contract,
            backtest_config=backtest_config,
            data_dir=root / "data",
            profile=LIVE_PARITY_PROFILE,
        )
        return StrategyRevalidationContext(
            root=root,
            strategy=strategy,
            baseline_config=baseline_config,
            backtest_config=backtest_config,
            symbols=list(spec_payload["symbols"]),
            manifest_path=manifest_path,
            scoring_weights=scoring_weights,
            scoring_ceilings=scoring_ceilings,
            hard_rejects=hard_rejects,
            phase_gate_criteria=phase_gate_criteria,
            spec_path=spec_path,
            spec_payload=spec_payload,
            store=ParquetStore(base_dir=root / "data"),
            max_workers=int(spec_payload.get("max_workers", 2)),
            contract=contract,
        )

    raise ValueError(f"Unsupported strategy: {strategy}")


def _evaluate_phase_gates(
    phase_gate_criteria: dict[int, list[GateCriterion]],
    metrics: MetricDict,
) -> dict[str, dict[str, Any]]:
    gate_results: dict[str, dict[str, Any]] = {}
    synthetic = GreedyResult(
        accepted_experiments=[],
        rejected_experiments=[],
        final_mutations={},
        final_score=0.0,
        final_metrics=metrics,
    )
    for phase, criteria in sorted(phase_gate_criteria.items()):
        gate = evaluate_gate(criteria, synthetic)
        gate_results[str(phase)] = {
            "passed": gate.passed,
            "failure_reasons": list(gate.failure_reasons),
            "failure_category": gate.failure_category,
        }
    return gate_results


def _evaluate_mutations(
    context: StrategyRevalidationContext,
    label: str,
    mutations: dict[str, Any],
) -> EvaluationBundle:
    config = apply_mutations(context.baseline_config, mutations)
    contract = _build_revalidation_contract(
        root=context.root,
        strategy=context.strategy,
        strategy_config=config,
        backtest_config=context.backtest_config,
        scoring_weights=context.scoring_weights,
        scoring_ceilings=context.scoring_ceilings,
        hard_rejects=context.hard_rejects,
        phase_gate_criteria=context.phase_gate_criteria,
    )
    result = run(
        config,
        context.backtest_config,
        data_dir=context.root / "data",
        store=context.store,
        strategy_type=context.strategy,
    )
    metrics = metrics_to_dict(result.metrics)
    score, rejected, reject_reason = composite_score(
        metrics,
        weights=context.scoring_weights,
        hard_rejects=context.hard_rejects,
        ceilings=context.scoring_ceilings,
    )
    phase_gates = _evaluate_phase_gates(context.phase_gate_criteria, metrics)
    snapshot = EvaluationSnapshot(
        label=label,
        mutations=dict(mutations),
        score=score,
        rejected=rejected,
        reject_reason=reject_reason,
        metrics=metrics,
        phase_gates=phase_gates,
        final_phase=context.final_phase,
        final_phase_gate_passed=phase_gates[str(context.final_phase)]["passed"],
        contract_hash=contract.get("contract_hash", ""),
        profile_hash=contract.get("profile_hash", ""),
        strategy_config_hash=contract.get("strategy_config_hash", ""),
        portfolio_config_hash=contract.get("portfolio_config_hash", ""),
        data_window=contract.get("data_window", {}),
        contract=contract,
    )
    return EvaluationBundle(snapshot=snapshot, result=result, config=config)


def _write_strategy_config(
    path: Path,
    config: Any,
    *,
    contract: dict[str, Any] | None = None,
) -> None:
    payload = {"strategy": config.to_dict()}
    if contract is not None:
        payload["metadata"] = {
            "contract_hash": contract.get("contract_hash", ""),
            "profile_hash": contract.get("profile_hash", ""),
            "strategy_config_hash": contract.get("strategy_config_hash", ""),
            "portfolio_config_hash": contract.get("portfolio_config_hash", ""),
            "data_window": contract.get("data_window", {}),
            "data_fingerprint": contract.get("data_fingerprint", {}),
            "symbols": contract.get("symbols", []),
            "required_timeframes": contract.get("required_timeframes", []),
            "contract": contract,
        }
    _write_json(path, payload)


def _write_checkpoint_artifacts(
    bundle: EvaluationBundle,
    output_dir: Path,
    *,
    source_manifest_metrics: dict[str, float] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_strategy_config(
        output_dir / "optimized_config.json",
        bundle.config,
        contract=bundle.snapshot.contract,
    )
    generate_report(bundle.result, output_dir)
    export_equity_curve(bundle.result, output_dir)
    export_trade_journal(bundle.result, output_dir)

    summary = bundle.snapshot.to_dict()
    if source_manifest_metrics:
        summary["source_manifest_metrics"] = source_manifest_metrics
        summary["metric_deltas_vs_manifest"] = {
            key: summary["metrics"].get(key, 0.0) - value
            for key, value in source_manifest_metrics.items()
        }
    _write_json(output_dir / "summary.json", summary)


def run_manifest_replay(
    context: StrategyRevalidationContext,
    output_dir: Path,
) -> list[EvaluationBundle]:
    """Replay base and cumulative manifest checkpoints under the fixed engine."""
    manifest = _load_json(context.manifest_path)
    checkpoints = build_manifest_checkpoints(manifest)
    bundles: list[EvaluationBundle] = []

    for checkpoint in checkpoints:
        bundle = _evaluate_mutations(context, checkpoint.label, checkpoint.mutations)
        bundles.append(bundle)
        _write_checkpoint_artifacts(
            bundle,
            output_dir / checkpoint.label,
            source_manifest_metrics=checkpoint.source_manifest_metrics,
        )

    final_bundle = bundles[-1]
    _write_strategy_config(
        output_dir / "final_cumulative_config.json",
        final_bundle.config,
        contract=final_bundle.snapshot.contract,
    )
    _write_json(
        output_dir / "manifest_replay_summary.json",
        {
            "strategy": context.strategy,
            "manifest_path": str(context.manifest_path),
            "spec_path": str(context.spec_path),
            "contract_hash": context.contract.get("contract_hash", ""),
            "profile_hash": context.contract.get("profile_hash", ""),
            "contract": context.contract,
            "backtest_window": {
                "start_date": context.backtest_config.start_date.isoformat(),
                "end_date": context.backtest_config.end_date.isoformat(),
                "warmup_days": context.backtest_config.warmup_days,
                "symbols": context.symbols,
            },
            "checkpoints": [bundle.snapshot.to_dict() for bundle in bundles],
            "final_cumulative_label": final_bundle.snapshot.label,
        },
    )
    return bundles


def run_ablation(
    context: StrategyRevalidationContext,
    winner_bundle: EvaluationBundle,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Ablate each currently active accepted mutation one at a time."""
    results: list[dict[str, Any]] = []
    winner_snapshot = winner_bundle.snapshot

    for key, value in winner_snapshot.mutations.items():
        ablated_mutations = dict(winner_snapshot.mutations)
        ablated_mutations.pop(key)
        ablated = _evaluate_mutations(context, f"remove_{key}", ablated_mutations)
        supports_inclusion = (
            (winner_snapshot.final_phase_gate_passed and not ablated.snapshot.final_phase_gate_passed)
            or (not winner_snapshot.rejected and ablated.snapshot.rejected)
            or (ablated.snapshot.score < winner_snapshot.score - 1e-9)
        )
        results.append(
            {
                "mutation_key": key,
                "removed_value": value,
                "supports_inclusion": supports_inclusion,
                "score_delta": ablated.snapshot.score - winner_snapshot.score,
                "net_return_delta": ablated.snapshot.metrics["net_return_pct"] - winner_snapshot.metrics["net_return_pct"],
                "total_trades_delta": ablated.snapshot.metrics["total_trades"] - winner_snapshot.metrics["total_trades"],
                "profit_factor_delta": ablated.snapshot.metrics["profit_factor"] - winner_snapshot.metrics["profit_factor"],
                "max_drawdown_delta": ablated.snapshot.metrics["max_drawdown_pct"] - winner_snapshot.metrics["max_drawdown_pct"],
                "sharpe_delta": ablated.snapshot.metrics["sharpe_ratio"] - winner_snapshot.metrics["sharpe_ratio"],
                "exit_efficiency_delta": ablated.snapshot.metrics["exit_efficiency"] - winner_snapshot.metrics["exit_efficiency"],
                "snapshot": ablated.snapshot.to_dict(),
            }
        )

    results.sort(key=lambda item: (item["score_delta"], item["net_return_delta"]))
    _write_json(
        output_dir / "ablation_summary.json",
        {
            "strategy": context.strategy,
            "winner": winner_snapshot.to_dict(),
            "results": results,
        },
    )
    return results


def run_perturbation(
    context: StrategyRevalidationContext,
    winner_bundle: EvaluationBundle,
    ablation_results: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run a small local sweep around surviving accepted numeric mutations."""
    winner_snapshot = winner_bundle.snapshot
    perturbations: list[dict[str, Any]] = []

    surviving_numeric = [
        (item["mutation_key"], winner_snapshot.mutations[item["mutation_key"]])
        for item in ablation_results
        if item["supports_inclusion"]
        and item["mutation_key"] in winner_snapshot.mutations
        and isinstance(winner_snapshot.mutations[item["mutation_key"]], (int, float))
        and not isinstance(winner_snapshot.mutations[item["mutation_key"]], bool)
    ]

    for key, current_value in surviving_numeric:
        for variant in local_perturbation_values(key, current_value):
            candidate_mutations = dict(winner_snapshot.mutations)
            candidate_mutations[key] = variant
            candidate = _evaluate_mutations(context, f"{key}__{variant}", candidate_mutations)
            perturbations.append(
                {
                    "mutation_key": key,
                    "base_value": current_value,
                    "trial_value": variant,
                    "score_delta": candidate.snapshot.score - winner_snapshot.score,
                    "net_return_delta": candidate.snapshot.metrics["net_return_pct"] - winner_snapshot.metrics["net_return_pct"],
                    "total_trades_delta": candidate.snapshot.metrics["total_trades"] - winner_snapshot.metrics["total_trades"],
                    "profit_factor_delta": candidate.snapshot.metrics["profit_factor"] - winner_snapshot.metrics["profit_factor"],
                    "max_drawdown_delta": candidate.snapshot.metrics["max_drawdown_pct"] - winner_snapshot.metrics["max_drawdown_pct"],
                    "improves_score": candidate.snapshot.score > winner_snapshot.score + 1e-9,
                    "snapshot": candidate.snapshot.to_dict(),
                }
            )

    perturbations.sort(key=lambda item: item["score_delta"], reverse=True)
    _write_json(
        output_dir / "perturbation_summary.json",
        {
            "strategy": context.strategy,
            "winner": winner_snapshot.to_dict(),
            "surviving_numeric_mutations": [key for key, _ in surviving_numeric],
            "results": perturbations,
        },
    )
    return perturbations


def write_cleaned_seed(
    context: StrategyRevalidationContext,
    ablation_results: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Build a cleaned seed config and notes for the next canonical rerun."""
    manifest = _load_json(context.manifest_path)
    checkpoints = build_manifest_checkpoints(manifest)
    final_mutations = dict(checkpoints[-1].mutations)
    rounds = {checkpoint.manifest_round: checkpoint for checkpoint in checkpoints if checkpoint.manifest_round is not None}
    ablation_support = {
        item["mutation_key"]: bool(item["supports_inclusion"])
        for item in ablation_results
    }

    if context.strategy == "momentum":
        cleaned_mutations = build_momentum_cleaned_seed_mutations(final_mutations, ablation_support)
        notes = {
            "strategy": context.strategy,
            "seed_rule": "Keep only structural accepted mutations so trail/exit/risk tweaks must re-earn inclusion.",
            "kept_mutations": cleaned_mutations,
            "dropped_mutations": {
                key: value
                for key, value in final_mutations.items()
                if key not in cleaned_mutations
            },
        }
    elif context.strategy == "trend":
        round2 = rounds.get(2)
        if round2 is None:
            raise RuntimeError("Trend round 2 checkpoint is required for the cleaned seed.")
        round3_relaxations = {
            key: value
            for key, value in final_mutations.items()
            if key not in round2.mutations
        }
        cleaned_mutations = dict(round2.mutations)
        notes = {
            "strategy": context.strategy,
            "seed_rule": "Restart from the round-2 core so round-3 relaxations re-earn inclusion.",
            "kept_mutations": cleaned_mutations,
            "round3_relaxations": round3_relaxations,
            "round3_relaxation_ablation": {
                item["mutation_key"]: item["supports_inclusion"]
                for item in ablation_results
                if item["mutation_key"] in round3_relaxations
            },
        }
    elif context.strategy == "breakout":
        round2 = rounds.get(2)
        if round2 is None:
            raise RuntimeError("Breakout round 2 checkpoint is required for the cleaned seed.")
        round3_branch = {
            key: value
            for key, value in final_mutations.items()
            if key not in round2.mutations
        }
        cleaned_mutations = dict(round2.mutations)
        notes = {
            "strategy": context.strategy,
            "seed_rule": "Restart from the round-2 core so the round-3 relaxed-body branch must re-earn inclusion.",
            "kept_mutations": cleaned_mutations,
            "round3_noncore_mutations": round3_branch,
            "round3_branch_ablation": {
                item["mutation_key"]: item["supports_inclusion"]
                for item in ablation_results
                if item["mutation_key"] in round3_branch
            },
        }
    else:
        raise ValueError(f"Unsupported strategy: {context.strategy}")

    cleaned_config = apply_mutations(context.baseline_config, cleaned_mutations)
    contract = _build_revalidation_contract(
        root=context.root,
        strategy=context.strategy,
        strategy_config=cleaned_config,
        backtest_config=context.backtest_config,
        scoring_weights=context.scoring_weights,
        scoring_ceilings=context.scoring_ceilings,
        hard_rejects=context.hard_rejects,
        phase_gate_criteria=context.phase_gate_criteria,
    )
    notes["contract_hash"] = contract.get("contract_hash", "")
    notes["profile_hash"] = contract.get("profile_hash", "")
    notes["contract"] = contract
    seed_path = output_dir / "cleaned_seed_config.json"
    _write_strategy_config(seed_path, cleaned_config, contract=contract)
    _write_json(output_dir / "cleaned_seed_notes.json", notes)
    return seed_path, notes


@contextmanager
def _patched_run_greedy() -> Any:
    """Patch PhaseRunner to use the historical no-pruning greedy variant."""
    original = phase_runner_module.run_greedy
    phase_runner_module.run_greedy = run_greedy_without_pruning
    try:
        yield
    finally:
        phase_runner_module.run_greedy = original


def _load_strategy_json(path: Path, config_type: str) -> Any:
    payload = _load_json(path)
    strategy_payload = payload.get("strategy", payload)
    if config_type == "momentum":
        return MomentumConfig.from_dict(strategy_payload)
    if config_type == "trend":
        return TrendConfig.from_dict(strategy_payload)
    if config_type == "breakout":
        return BreakoutConfig.from_dict(strategy_payload)
    raise ValueError(f"Unsupported config type: {config_type}")


def _run_momentum_cleaned_seed_rerun(
    context: StrategyRevalidationContext,
    baseline_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base_config = _load_strategy_json(baseline_path, "momentum")
    plugin = MomentumRound5PhasedPlugin(
        backtest_config=context.backtest_config,
        base_config=base_config,
        data_dir=context.root / "data",
        max_workers=context.max_workers,
    )
    contract = build_optimization_contract(
        strategy_type="momentum",
        strategy_config=base_config,
        backtest_config=context.backtest_config,
        data_dir=context.root / "data",
        profile=LIVE_PARITY_PROFILE,
        plugin=plugin,
        scoring_weights=MOMENTUM_SCORING_WEIGHTS,
        scoring_ceilings=MOMENTUM_SCORING_CEILINGS,
        hard_rejects=MOMENTUM_HARD_REJECTS,
        gate_criteria=MOMENTUM_PHASE_GATE_CRITERIA,
    )
    runner = PhaseRunner(plugin, output_dir, contract=contract)
    state = PhaseState.load_or_create(output_dir / "phase_state.json")
    run_spec = {
        "strategy": "momentum",
        "baseline_seed": str(baseline_path),
        "rerun_reason": "Cleaned-seed rerun after manifest replay, ablation, and perturbation.",
        "contract_hash": contract.get("contract_hash", ""),
        "profile_hash": contract.get("profile_hash", ""),
        "contract": contract,
        "immutable_scoring_weights": MOMENTUM_SCORING_WEIGHTS,
        "immutable_scoring_ceilings": MOMENTUM_SCORING_CEILINGS,
        "hard_rejects": MOMENTUM_HARD_REJECTS,
        "phase_gate_criteria": {
            str(phase): [criterion.__dict__ for criterion in criteria]
            for phase, criteria in MOMENTUM_PHASE_GATE_CRITERIA.items()
        },
    }
    _write_json(output_dir / "run_spec.json", run_spec)

    with _patched_run_greedy():
        runner.run_all_phases(state)

    last_phase = max(state.phase_metrics) if state.phase_metrics else None
    final_metrics = state.phase_metrics.get(last_phase) if last_phase is not None else None
    summary = {
        "strategy": "momentum",
        "baseline_seed": str(baseline_path),
        "completed_phases": state.completed_phases,
        "cumulative_mutations": state.cumulative_mutations,
        "final_metrics": final_metrics,
        "contract_hash": contract.get("contract_hash", ""),
        "contract": contract,
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def _run_trend_cleaned_seed_rerun(
    context: StrategyRevalidationContext,
    baseline_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base_config = _load_strategy_json(baseline_path, "trend")
    plugin = Round7TrendPlugin(
        context.backtest_config,
        base_config,
        data_dir=context.root / "data",
        max_workers=context.max_workers,
    )
    contract = build_optimization_contract(
        strategy_type="trend",
        strategy_config=base_config,
        backtest_config=context.backtest_config,
        data_dir=context.root / "data",
        profile=LIVE_PARITY_PROFILE,
        plugin=plugin,
        scoring_weights=ROUND7_SCORING_WEIGHTS,
        scoring_ceilings=ROUND7_IMMUTABLE_SCORING_CEILINGS,
        hard_rejects=ROUND7_HARD_REJECTS,
        gate_criteria=ROUND7_PHASE_GATE_CRITERIA,
    )
    runner = PhaseRunner(
        plugin,
        output_dir,
        round_name="trend_cleaned_seed_rerun",
        contract=contract,
    )
    state = PhaseState.load_or_create(output_dir / "phase_state.json")

    run_spec = {
        "strategy": "trend",
        "baseline_seed": str(baseline_path),
        "rerun_reason": "Cleaned-seed rerun after manifest replay, ablation, and perturbation.",
        "contract_hash": contract.get("contract_hash", ""),
        "profile_hash": contract.get("profile_hash", ""),
        "contract": contract,
        "scoring_weights": ROUND7_SCORING_WEIGHTS,
        "immutable_scoring_ceilings": ROUND7_IMMUTABLE_SCORING_CEILINGS,
        "hard_rejects": ROUND7_HARD_REJECTS,
        "phase_gate_criteria": {
            str(phase): [criterion.__dict__ for criterion in criteria]
            for phase, criteria in ROUND7_PHASE_GATE_CRITERIA.items()
        },
    }
    _write_json(output_dir / "run_spec.json", run_spec)

    runner.run_all_phases(state)

    last_phase = max(state.phase_metrics) if state.phase_metrics else None
    final_metrics = state.phase_metrics.get(last_phase) if last_phase is not None else None
    summary = {
        "strategy": "trend",
        "baseline_seed": str(baseline_path),
        "completed_phases": state.completed_phases,
        "cumulative_mutations": state.cumulative_mutations,
        "final_metrics": final_metrics,
        "contract_hash": contract.get("contract_hash", ""),
        "contract": contract,
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def _run_breakout_cleaned_seed_rerun(
    context: StrategyRevalidationContext,
    baseline_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base_config = _load_strategy_json(baseline_path, "breakout")
    plugin = BreakoutRound6PhasedPlugin(
        backtest_config=context.backtest_config,
        base_config=base_config,
        data_dir=context.root / "data",
        max_workers=context.max_workers,
    )
    contract = build_optimization_contract(
        strategy_type="breakout",
        strategy_config=base_config,
        backtest_config=context.backtest_config,
        data_dir=context.root / "data",
        profile=LIVE_PARITY_PROFILE,
        plugin=plugin,
        scoring_weights=ROUND6_IMMUTABLE_SCORING_WEIGHTS,
        scoring_ceilings=ROUND6_IMMUTABLE_SCORING_CEILINGS,
        hard_rejects=ROUND6_HARD_REJECTS,
        gate_criteria=ROUND6_PHASE_GATE_CRITERIA,
    )
    runner = PhaseRunner(plugin, output_dir, contract=contract)
    state = PhaseState.load_or_create(output_dir / "phase_state.json")
    run_spec = {
        "strategy": "breakout",
        "baseline_seed": str(baseline_path),
        "rerun_reason": "Cleaned-seed rerun after manifest replay, ablation, and perturbation.",
        "contract_hash": contract.get("contract_hash", ""),
        "profile_hash": contract.get("profile_hash", ""),
        "contract": contract,
        "immutable_scoring_weights": ROUND6_IMMUTABLE_SCORING_WEIGHTS,
        "immutable_scoring_ceilings": ROUND6_IMMUTABLE_SCORING_CEILINGS,
        "hard_rejects": ROUND6_HARD_REJECTS,
        "phase_gate_criteria": {
            str(phase): [criterion.__dict__ for criterion in criteria]
            for phase, criteria in ROUND6_PHASE_GATE_CRITERIA.items()
        },
    }
    _write_json(output_dir / "run_spec.json", run_spec)

    with _patched_run_greedy():
        runner.run_all_phases(state)

    last_phase = max(state.phase_metrics) if state.phase_metrics else None
    final_metrics = state.phase_metrics.get(last_phase) if last_phase is not None else None
    summary = {
        "strategy": "breakout",
        "baseline_seed": str(baseline_path),
        "completed_phases": state.completed_phases,
        "cumulative_mutations": state.cumulative_mutations,
        "final_metrics": final_metrics,
        "contract_hash": contract.get("contract_hash", ""),
        "contract": contract,
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def run_cleaned_seed_rerun(
    context: StrategyRevalidationContext,
    baseline_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the latest canonical phased optimizer from a cleaned seed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if context.strategy == "momentum":
        return _run_momentum_cleaned_seed_rerun(context, baseline_path, output_dir)
    if context.strategy == "trend":
        return _run_trend_cleaned_seed_rerun(context, baseline_path, output_dir)
    if context.strategy == "breakout":
        return _run_breakout_cleaned_seed_rerun(context, baseline_path, output_dir)
    raise ValueError(f"Unsupported strategy: {context.strategy}")


def revalidate_strategy(
    root: Path,
    strategy: str,
    strategy_output_dir: Path,
    *,
    include_rerun: bool = True,
) -> dict[str, Any]:
    """Run the full revalidation sequence for one strategy."""
    context = load_strategy_revalidation_context(root, strategy)
    strategy_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_output_dir = strategy_output_dir / "manifest_replay"
    ablation_output_dir = strategy_output_dir / "ablation"
    perturb_output_dir = strategy_output_dir / "perturbation"
    cleaned_seed_output_dir = strategy_output_dir / "cleaned_seed"
    rerun_output_dir = strategy_output_dir / "cleaned_seed_rerun"

    manifest_bundles = run_manifest_replay(context, manifest_output_dir)
    winner_bundle = manifest_bundles[-1]
    ablation_results = run_ablation(context, winner_bundle, ablation_output_dir)
    perturbation_results = run_perturbation(context, winner_bundle, ablation_results, perturb_output_dir)
    cleaned_seed_path, cleaned_seed_notes = write_cleaned_seed(context, ablation_results, cleaned_seed_output_dir)

    rerun_summary = None
    if include_rerun:
        rerun_summary = run_cleaned_seed_rerun(context, cleaned_seed_path, rerun_output_dir)

    summary = {
        "strategy": strategy,
        "strategy_output_dir": str(strategy_output_dir),
        "manifest_replay_dir": str(manifest_output_dir),
        "ablation_dir": str(ablation_output_dir),
        "perturbation_dir": str(perturb_output_dir),
        "cleaned_seed_dir": str(cleaned_seed_output_dir),
        "cleaned_seed_config": str(cleaned_seed_path),
        "cleaned_seed_notes": cleaned_seed_notes,
        "winner": winner_bundle.snapshot.to_dict(),
        "ablation_survivors": [
            item["mutation_key"]
            for item in ablation_results
            if item["supports_inclusion"]
        ],
        "perturbation_improvements": [
            item
            for item in perturbation_results
            if item["improves_score"]
        ],
        "cleaned_seed_rerun": rerun_summary,
    }
    _write_json(strategy_output_dir / "strategy_summary.json", summary)
    return summary
