from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backtests.momentum.auto.portfolio_synergy.family_phase_auto import (
    SCORE_WEIGHTS,
    _load_mtm_price_bars_for_scoring,
    apply_portfolio_mutations,
    headline_mtm_metric_package,
    score_metrics,
)
from backtests.momentum.engine.family_portfolio_engine import (
    FamilyPortfolioBacktestConfig,
    FamilyPortfolioBacktester,
    FamilyPortfolioReplayBundle,
    build_family_replay_bundle,
    family_config_from_dict,
    family_config_to_dict,
)


@dataclass(frozen=True)
class DynamicCandidate:
    name: str
    description: str
    mutations: dict[str, Any]


@dataclass(frozen=True)
class DynamicEvaluation:
    name: str
    description: str
    score: float
    rejected: bool
    reject_reason: str
    soft_warnings: list[str]
    components: dict[str, float]
    metrics: dict[str, float]
    config: FamilyPortfolioBacktestConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run round 2 dynamic synergistic risk allocation search.",
    )
    parser.add_argument("--round-1-dir", default="backtests/output/momentum/portfolio_synergy/round_1")
    parser.add_argument("--output-dir", default="backtests/output/momentum/portfolio_synergy/round_2")
    parser.add_argument("--data-dir", default="backtests/momentum/data/raw")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.00001)
    args = parser.parse_args(argv)

    summary = run_dynamic_round2(
        round_1_dir=Path(args.round_1_dir),
        output_dir=Path(args.output_dir),
        data_dir=Path(args.data_dir),
        max_workers=args.max_workers,
        min_delta=args.min_delta,
    )
    print(f"Round 2 dynamic risk allocation complete: {args.output_dir}")
    print(f"Score components: {summary['score_component_count']}")
    print(f"Final score: {summary['final_score']:.4f}")
    print(f"Final metrics: {summary['final_metrics']}")


def run_dynamic_round2(
    *,
    round_1_dir: Path,
    output_dir: Path,
    data_dir: Path = Path("backtests/momentum/data/raw"),
    max_workers: int = 2,
    min_delta: float = 0.00001,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = family_config_from_dict(
        json.loads((round_1_dir / "optimized_portfolio_config.json").read_text(encoding="utf-8"))
    )
    with (round_1_dir / "strategy_trades.pkl").open("rb") as fh:
        trades_by_strategy = pickle.load(fh)
    with (output_dir / "strategy_trades.pkl").open("wb") as fh:
        pickle.dump(trades_by_strategy, fh)
    if (round_1_dir / "strategy_trade_manifest.json").exists():
        (output_dir / "strategy_trade_manifest.json").write_text(
            (round_1_dir / "strategy_trade_manifest.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    replay_bundle = build_family_replay_bundle(trades_by_strategy)
    price_bars = _load_mtm_price_bars_for_scoring(data_dir)

    current_config = base_config
    current_eval = evaluate_dynamic_config(
        "ROUND_1_STATIC_BASELINE",
        "Round 1 static optimizer result",
        current_config,
        replay_bundle,
        price_bars=price_bars,
    )
    phase_records: list[dict[str, Any]] = []
    _write_json(output_dir / "baseline.json", _evaluation_record(current_eval))

    for phase, candidates in sorted(PHASE_CANDIDATES.items()):
        evaluations = _evaluate_candidates(
            current_config,
            candidates,
            replay_bundle,
            max_workers=max_workers,
            price_bars=price_bars,
        )
        viable = [item for item in evaluations if not item.rejected]
        best = max(viable, key=_dynamic_selection_key, default=None)
        accepted = bool(best and best.score > current_eval.score + min_delta)
        if accepted and best is not None:
            current_config = best.config
            current_eval = best
        record = {
            "phase": phase,
            "accepted": accepted,
            "accepted_candidate": best.name if accepted and best is not None else None,
            "current_score": current_eval.score,
            "evaluations": [_evaluation_record(item) for item in evaluations],
        }
        phase_records.append(record)
        _write_json(output_dir / f"phase_{phase}.json", record)

    final_result = FamilyPortfolioBacktester(current_config).run_bundle(replay_bundle)
    final_metric_package = headline_mtm_metric_package(
        current_config,
        final_result,
        data_dir=data_dir,
        price_bars=price_bars,
    )
    final_scored = score_metrics(final_metric_package["headline_metrics"])
    summary = {
        "round": 2,
        "objective": "dynamic_synergistic_risk_allocation",
        "source_round_1": str(round_1_dir),
        "score_components": SCORE_WEIGHTS,
        "score_component_count": len(SCORE_WEIGHTS),
        "max_workers": max_workers,
        "min_delta": min_delta,
        "replay_architecture": final_result.replay_architecture,
        "replay_source_fingerprint": replay_bundle.source_fingerprint,
        "trade_outcome_count": len(replay_bundle.trade_outcomes),
        "decision_count": len(replay_bundle.decisions),
        "phases": phase_records,
        "final_score": final_scored["score"],
        "final_components": final_scored["components"],
        "final_rejected": final_scored["rejected"],
        "final_reject_reason": final_scored["reject_reason"],
        "final_soft_warnings": final_scored["soft_warnings"],
        "final_metrics": final_metric_package["headline_metrics"],
        "final_metrics_realized": final_metric_package["realized_metrics"],
        "final_diagnostic_equity": final_metric_package["diagnostic_equity"],
        "final_config": family_config_to_dict(current_config),
        "strategy_trade_counts": final_result.strategy_trade_counts,
        "strategy_blocked_counts": final_result.strategy_blocked_counts,
        "rule_blocks": final_result.rule_blocks,
    }
    _write_json(output_dir / "run_summary.json", summary)
    _write_json(output_dir / "optimized_portfolio_config.json", summary["final_config"])
    return summary


def evaluate_dynamic_config(
    name: str,
    description: str,
    config: FamilyPortfolioBacktestConfig,
    replay_bundle: FamilyPortfolioReplayBundle,
    *,
    price_bars: Any,
) -> DynamicEvaluation:
    result = FamilyPortfolioBacktester(config).run_bundle(replay_bundle)
    metric_package = headline_mtm_metric_package(config, result, price_bars=price_bars)
    headline_metrics = metric_package["headline_metrics"]
    scored = score_metrics(headline_metrics)
    return DynamicEvaluation(
        name=name,
        description=description,
        score=scored["score"],
        rejected=scored["rejected"],
        reject_reason=scored["reject_reason"],
        soft_warnings=scored["soft_warnings"],
        components=scored["components"],
        metrics=headline_metrics,
        config=config,
    )


def _evaluate_candidates(
    current_config: FamilyPortfolioBacktestConfig,
    candidates: list[DynamicCandidate],
    replay_bundle: FamilyPortfolioReplayBundle,
    *,
    max_workers: int,
    price_bars: Any,
) -> list[DynamicEvaluation]:
    worker_count = max(1, min(int(max_workers), 2))
    evaluations: list[DynamicEvaluation] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                evaluate_dynamic_config,
                candidate.name,
                candidate.description,
                apply_portfolio_mutations(current_config, candidate.mutations),
                replay_bundle,
                price_bars=price_bars,
            ): candidate
            for candidate in candidates
        }
        for future in as_completed(future_map):
            evaluations.append(future.result())
    evaluations.sort(key=_dynamic_selection_key, reverse=True)
    return evaluations


def _evaluation_record(evaluation: DynamicEvaluation) -> dict[str, Any]:
    return {
        "name": evaluation.name,
        "description": evaluation.description,
        "score": evaluation.score,
        "rejected": evaluation.rejected,
        "reject_reason": evaluation.reject_reason,
        "soft_warnings": evaluation.soft_warnings,
        "components": evaluation.components,
        "metrics": evaluation.metrics,
        "config": family_config_to_dict(evaluation.config),
    }


def _dynamic_base(
    *,
    max_trade_risk_R: float,
    min_trade_risk_R: float = 0.0,
    heat_pressure_threshold: float = 1.0,
    heat_pressure_mult: float = 1.0,
    same_direction_pressure_threshold: float = 1.0,
    same_direction_pressure_mult: float = 1.0,
    existing_position_mult: float = 1.0,
    strategy_multipliers: tuple[tuple[str, float], ...] = (),
) -> dict[str, Any]:
    return {
        "dynamic.enabled": True,
        "dynamic.fit_to_remaining_heat": True,
        "dynamic.fit_to_remaining_directional_cap": True,
        "dynamic.fit_to_remaining_family_cap": True,
        "dynamic.min_qty": 1,
        "dynamic.min_trade_risk_R": min_trade_risk_R,
        "dynamic.max_trade_risk_R": max_trade_risk_R,
        "dynamic.heat_pressure_threshold": heat_pressure_threshold,
        "dynamic.heat_pressure_mult": heat_pressure_mult,
        "dynamic.same_direction_pressure_threshold": same_direction_pressure_threshold,
        "dynamic.same_direction_pressure_mult": same_direction_pressure_mult,
        "dynamic.existing_position_mult": existing_position_mult,
        "dynamic.strategy_multipliers": strategy_multipliers,
    }


def _phase_candidates() -> dict[int, list[DynamicCandidate]]:
    return {
        1: [
            DynamicCandidate(
                "capacity_fit_conservative",
                "Fit entries to remaining caps with a conservative per-trade R clamp",
                _dynamic_base(max_trade_risk_R=1.50, heat_pressure_threshold=0.70, heat_pressure_mult=0.65, same_direction_pressure_threshold=0.70, same_direction_pressure_mult=0.75, existing_position_mult=0.90),
            ),
            DynamicCandidate(
                "capacity_fit_balanced",
                "Fit entries to remaining caps with a balanced per-trade R clamp",
                _dynamic_base(max_trade_risk_R=2.00, heat_pressure_threshold=0.75, heat_pressure_mult=0.75, same_direction_pressure_threshold=0.75, same_direction_pressure_mult=0.80, existing_position_mult=0.95),
            ),
            DynamicCandidate(
                "capacity_fit_aggressive",
                "Fit entries to remaining caps while preserving larger alpha bets",
                _dynamic_base(max_trade_risk_R=2.75, heat_pressure_threshold=0.85, heat_pressure_mult=0.85, same_direction_pressure_threshold=0.85, same_direction_pressure_mult=0.90),
            ),
            DynamicCandidate(
                "capacity_fit_frequency",
                "Smaller per-trade risk to maximize accepted signal count",
                _dynamic_base(max_trade_risk_R=1.20, heat_pressure_threshold=0.60, heat_pressure_mult=0.55, same_direction_pressure_threshold=0.60, same_direction_pressure_mult=0.65, existing_position_mult=0.80),
            ),
            DynamicCandidate(
                "capacity_fit_no_pressure_throttle",
                "Only shrink when required by the hard caps",
                _dynamic_base(max_trade_risk_R=3.25),
            ),
        ],
        2: [
            DynamicCandidate(
                "edge_balance_multipliers",
                "Lean into accepted-edge strategies and temper chronic cap pressure",
                {
                    "dynamic.strategy_multipliers": (
                        ("NQ_REGIME", 0.80),
                        ("VdubusNQ_v4", 1.10),
                        ("NQDTC_v2.1", 1.10),
                        ("DownturnDominator_v1", 1.00),
                    ),
                },
            ),
            DynamicCandidate(
                "blocked_alpha_recovery",
                "Recover blocked NQ edge through smaller but still meaningful entries",
                {
                    "dynamic.strategy_multipliers": (
                        ("NQ_REGIME", 0.90),
                        ("VdubusNQ_v4", 1.00),
                        ("NQDTC_v2.1", 1.00),
                        ("DownturnDominator_v1", 1.00),
                    ),
                },
            ),
            DynamicCandidate(
                "frequency_rebalance",
                "Prioritize accepted count and strategy breadth over large single tickets",
                {
                    "dynamic.max_trade_risk_R": 1.50,
                    "dynamic.strategy_multipliers": (
                        ("NQ_REGIME", 0.65),
                        ("VdubusNQ_v4", 0.85),
                        ("NQDTC_v2.1", 0.95),
                        ("DownturnDominator_v1", 0.95),
                    ),
                },
            ),
            DynamicCandidate(
                "aggressive_alpha_rebalance",
                "Aggressive but cap-fitted risk for all primary alpha sources",
                {
                    "dynamic.max_trade_risk_R": 2.75,
                    "dynamic.strategy_multipliers": (
                        ("NQ_REGIME", 1.05),
                        ("VdubusNQ_v4", 1.10),
                        ("NQDTC_v2.1", 1.05),
                        ("DownturnDominator_v1", 1.00),
                    ),
                },
            ),
        ],
        3: [
            DynamicCandidate("heat_6_25_contracts_22", "Slightly more shared capacity for dynamic fit", {"config.heat_cap_R": 6.25, "rules.max_family_contracts_mnq_eq": 22}),
            DynamicCandidate("heat_6_75_contracts_24", "Upper controlled-aggressive capacity probe", {"config.heat_cap_R": 6.75, "rules.max_family_contracts_mnq_eq": 24}),
            DynamicCandidate("direction_caps_5_75_6_25", "More asymmetric directional breathing room", {"rules.directional_cap_long_R": 5.75, "rules.directional_cap_short_R": 6.25}),
            DynamicCandidate("max_positions_7", "Allow one more family slot after dynamic fit", {"config.max_total_positions": 7}),
            DynamicCandidate("daily_stop_2_75", "Loosen daily stop back toward round-1 seed", {"config.portfolio_daily_stop_R": 2.75}),
            DynamicCandidate("daily_stop_2_00", "Tighter daily stop for aggressive dynamic entries", {"config.portfolio_daily_stop_R": 2.00}),
        ],
        4: [
            DynamicCandidate(
                "dynamic_guarded_capacity",
                "Best-effort guarded blend of dynamic fit, capacity, and edge balance",
                {
                    **_dynamic_base(
                        max_trade_risk_R=2.25,
                        heat_pressure_threshold=0.72,
                        heat_pressure_mult=0.72,
                        same_direction_pressure_threshold=0.72,
                        same_direction_pressure_mult=0.78,
                        existing_position_mult=0.92,
                        strategy_multipliers=(
                            ("NQ_REGIME", 0.85),
                            ("VdubusNQ_v4", 1.10),
                            ("NQDTC_v2.1", 1.05),
                            ("DownturnDominator_v1", 1.00),
                        ),
                    ),
                    "config.heat_cap_R": 6.25,
                    "rules.max_family_contracts_mnq_eq": 22,
                    "rules.directional_cap_long_R": 5.75,
                    "rules.directional_cap_short_R": 6.25,
                },
            ),
            DynamicCandidate(
                "dynamic_frequency_frontier",
                "Higher frequency with moderate individual ticket size",
                {
                    **_dynamic_base(
                        max_trade_risk_R=1.50,
                        heat_pressure_threshold=0.65,
                        heat_pressure_mult=0.65,
                        same_direction_pressure_threshold=0.65,
                        same_direction_pressure_mult=0.70,
                        existing_position_mult=0.85,
                        strategy_multipliers=(
                            ("NQ_REGIME", 0.75),
                            ("VdubusNQ_v4", 0.95),
                            ("NQDTC_v2.1", 1.00),
                            ("DownturnDominator_v1", 1.00),
                        ),
                    ),
                    "config.heat_cap_R": 6.75,
                    "rules.max_family_contracts_mnq_eq": 24,
                    "config.max_total_positions": 7,
                },
            ),
            DynamicCandidate(
                "dynamic_alpha_frontier",
                "More aggressive alpha extraction with cap-fit safeguards",
                {
                    **_dynamic_base(
                        max_trade_risk_R=3.00,
                        heat_pressure_threshold=0.85,
                        heat_pressure_mult=0.85,
                        same_direction_pressure_threshold=0.85,
                        same_direction_pressure_mult=0.90,
                        strategy_multipliers=(
                            ("NQ_REGIME", 1.00),
                            ("VdubusNQ_v4", 1.10),
                            ("NQDTC_v2.1", 1.05),
                            ("DownturnDominator_v1", 1.00),
                        ),
                    ),
                    "config.heat_cap_R": 6.75,
                    "rules.max_family_contracts_mnq_eq": 24,
                    "rules.directional_cap_long_R": 5.75,
                    "rules.directional_cap_short_R": 6.25,
                },
            ),
        ],
    }


PHASE_CANDIDATES = _phase_candidates()


def _dynamic_selection_key(evaluation: DynamicEvaluation) -> tuple[float, float, float, float, float, float]:
    metrics = evaluation.metrics
    return (
        evaluation.score,
        float(metrics.get("net_profit", 0.0) or 0.0),
        float(metrics.get("trades_per_month", 0.0) or 0.0),
        -float(metrics.get("max_drawdown_pct", 1.0) or 1.0),
        -float(metrics.get("block_rate", 1.0) or 1.0),
        float(metrics.get("profit_factor", 0.0) or 0.0),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
