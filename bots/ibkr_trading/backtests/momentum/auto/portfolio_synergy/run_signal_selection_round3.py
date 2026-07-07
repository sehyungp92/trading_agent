from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
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
    FamilyPortfolioResult,
    FamilySignalFilterCondition,
    FamilySignalFilterRule,
    _condition_matches,
    build_family_replay_bundle,
    family_config_from_dict,
    family_config_to_dict,
)


@dataclass(frozen=True)
class Round3Candidate:
    name: str
    description: str
    mutations: dict[str, Any]
    filters: tuple[FamilySignalFilterRule, ...] = ()


@dataclass(frozen=True)
class Round3Evaluation:
    name: str
    description: str
    score: float
    validation_score: float
    robust_pass: bool
    robust_reason: str
    metrics: dict[str, float]
    validation_metrics: dict[str, float]
    train_metrics: dict[str, float]
    rejected: bool
    reject_reason: str
    soft_warnings: list[str]
    components: dict[str, float]
    config: FamilyPortfolioBacktestConfig
    filter_sample_stats: list[dict[str, Any]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run round 3 signal-selection experiments for blocked-candidate discrimination.",
    )
    parser.add_argument("--round-2-dir", default="backtests/output/momentum/portfolio_synergy/round_2")
    parser.add_argument("--output-dir", default="backtests/output/momentum/portfolio_synergy/round_3")
    parser.add_argument("--data-dir", default="backtests/momentum/data/raw")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.00001)
    parser.add_argument("--validation-min-delta", type=float, default=0.00001)
    args = parser.parse_args(argv)
    summary = run_round3(
        round_2_dir=Path(args.round_2_dir),
        output_dir=Path(args.output_dir),
        data_dir=Path(args.data_dir),
        max_workers=args.max_workers,
        min_delta=args.min_delta,
        validation_min_delta=args.validation_min_delta,
    )
    print(f"Round 3 signal selection complete: {args.output_dir}")
    print(f"Score components: {summary['score_component_count']}")
    print(f"Final score: {summary['final_score']:.4f}")
    print(f"Validation score: {summary['final_validation_score']:.4f}")
    print(f"Final metrics: {summary['final_metrics']}")


def run_round3(
    *,
    round_2_dir: Path,
    output_dir: Path,
    data_dir: Path = Path("backtests/momentum/data/raw"),
    max_workers: int = 2,
    min_delta: float = 0.00001,
    validation_min_delta: float = 0.00001,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = family_config_from_dict(
        json.loads((round_2_dir / "optimized_portfolio_config.json").read_text(encoding="utf-8"))
    )
    with (round_2_dir / "strategy_trades.pkl").open("rb") as fh:
        trades_by_strategy = pickle.load(fh)
    with (output_dir / "strategy_trades.pkl").open("wb") as fh:
        pickle.dump(trades_by_strategy, fh)
    if (round_2_dir / "strategy_trade_manifest.json").exists():
        (output_dir / "strategy_trade_manifest.json").write_text(
            (round_2_dir / "strategy_trade_manifest.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    replay_bundle = build_family_replay_bundle(trades_by_strategy)
    price_bars = _load_mtm_price_bars_for_scoring(data_dir)

    split_time = _split_time(base_config, replay_bundle)
    current_config = base_config
    current_eval = evaluate_round3_config(
        "ROUND_2_DYNAMIC_BASELINE",
        "Round 2 dynamic allocation result",
        current_config,
        replay_bundle,
        split_time,
        baseline_validation_score=None,
        filter_rules=(),
        price_bars=price_bars,
    )
    phase_records: list[dict[str, Any]] = []
    _write_json(output_dir / "baseline.json", _evaluation_record(current_eval))

    for phase, candidates in sorted(PHASE_CANDIDATES.items()):
        evaluations = _evaluate_candidates(
            current_config,
            candidates,
            replay_bundle,
            split_time=split_time,
            current_eval=current_eval,
            max_workers=max_workers,
            validation_min_delta=validation_min_delta,
            price_bars=price_bars,
        )
        viable = [item for item in evaluations if not item.rejected and item.robust_pass]
        best = max(viable, key=_round3_selection_key, default=None)
        accepted = bool(
            best
            and best.score > current_eval.score + min_delta
            and best.validation_score > current_eval.validation_score + validation_min_delta
        )
        if accepted and best is not None:
            current_config = best.config
            current_eval = best
        record = {
            "phase": phase,
            "accepted": accepted,
            "accepted_candidate": best.name if accepted and best is not None else None,
            "current_score": current_eval.score,
            "current_validation_score": current_eval.validation_score,
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
    final_validation_result = FamilyPortfolioBacktester(
        replace(current_config, start_date=split_time, end_date=None)
    ).run_bundle(replay_bundle)
    final_validation_metric_package = headline_mtm_metric_package(
        replace(current_config, start_date=split_time, end_date=None),
        final_validation_result,
        data_dir=data_dir,
        price_bars=price_bars,
    )
    final_validation_scored = score_metrics(final_validation_metric_package["headline_metrics"])
    summary = {
        "round": 3,
        "objective": "blocked_candidate_winner_loser_discrimination",
        "source_round_2": str(round_2_dir),
        "split_time_utc": split_time.isoformat(),
        "anti_overfit_gates": {
            "time_split": "first 60pct train / last 40pct validation by candidate entry time",
            "validation_min_delta": validation_min_delta,
            "candidate_requirements": [
                "full_sample_score_improvement",
                "validation_score_improvement",
                "validation_net_profit_not_below_baseline_by_more_than_2pct",
                "validation_profit_factor_at_least_1_35",
                "validation_total_trades_at_least_50",
                "signal_filter_rules_need_50_full_and_20_validation_matches",
                "signal_filter_rules_rejected_on_positive_train_negative_validation_avgR",
            ],
        },
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
        "final_validation_score": final_validation_scored["score"],
        "final_components": final_scored["components"],
        "final_rejected": final_scored["rejected"],
        "final_reject_reason": final_scored["reject_reason"],
        "final_soft_warnings": final_scored["soft_warnings"],
        "final_metrics": final_metric_package["headline_metrics"],
        "final_metrics_realized": final_metric_package["realized_metrics"],
        "final_diagnostic_equity": final_metric_package["diagnostic_equity"],
        "final_validation_metrics": final_validation_metric_package["headline_metrics"],
        "final_validation_metrics_realized": final_validation_metric_package["realized_metrics"],
        "final_validation_diagnostic_equity": final_validation_metric_package["diagnostic_equity"],
        "final_config": family_config_to_dict(current_config),
        "strategy_trade_counts": final_result.strategy_trade_counts,
        "strategy_blocked_counts": final_result.strategy_blocked_counts,
        "rule_blocks": final_result.rule_blocks,
    }
    _write_json(output_dir / "run_summary.json", summary)
    _write_json(output_dir / "optimized_portfolio_config.json", summary["final_config"])
    _write_signal_selection_report(output_dir / "signal_selection_diagnostics.md", summary)
    return summary


def evaluate_round3_config(
    name: str,
    description: str,
    config: FamilyPortfolioBacktestConfig,
    replay_bundle: FamilyPortfolioReplayBundle,
    split_time: datetime,
    baseline_validation_score: float | None,
    baseline_validation_net_profit: float | None = None,
    validation_min_delta: float = 0.0,
    filter_rules: tuple[FamilySignalFilterRule, ...] = (),
    price_bars: Any = None,
) -> Round3Evaluation:
    full_result = FamilyPortfolioBacktester(config).run_bundle(replay_bundle)
    full_metric_package = headline_mtm_metric_package(config, full_result, price_bars=price_bars)
    full_metrics = full_metric_package["headline_metrics"]
    full_scored = score_metrics(full_metrics)
    train_result = FamilyPortfolioBacktester(replace(config, end_date=split_time)).run_bundle(replay_bundle)
    train_metric_package = headline_mtm_metric_package(
        replace(config, end_date=split_time),
        train_result,
        price_bars=price_bars,
    )
    validation_result = FamilyPortfolioBacktester(replace(config, start_date=split_time, end_date=None)).run_bundle(replay_bundle)
    validation_metric_package = headline_mtm_metric_package(
        replace(config, start_date=split_time, end_date=None),
        validation_result,
        price_bars=price_bars,
    )
    validation_metrics = validation_metric_package["headline_metrics"]
    validation_scored = score_metrics(validation_metrics)
    filter_sample_stats = _filter_sample_stats(full_result, split_time, filter_rules)
    robust_pass, robust_reason = _robust_gate(
        validation_metrics,
        validation_scored["score"],
        baseline_validation_score,
        baseline_validation_net_profit,
        validation_min_delta,
        filter_sample_stats,
    )
    return Round3Evaluation(
        name=name,
        description=description,
        score=full_scored["score"],
        validation_score=validation_scored["score"],
        robust_pass=robust_pass,
        robust_reason=robust_reason,
        metrics=full_metrics,
        validation_metrics=validation_metrics,
        train_metrics=train_metric_package["headline_metrics"],
        rejected=full_scored["rejected"],
        reject_reason=full_scored["reject_reason"],
        soft_warnings=full_scored["soft_warnings"],
        components=full_scored["components"],
        config=config,
        filter_sample_stats=filter_sample_stats,
    )


def _robust_gate(
    validation_metrics: dict[str, float],
    validation_score: float,
    baseline_validation_score: float | None,
    baseline_validation_net_profit: float | None,
    validation_min_delta: float,
    filter_sample_stats: list[dict[str, Any]],
) -> tuple[bool, str]:
    if baseline_validation_score is None:
        return True, "baseline"
    for stats in filter_sample_stats:
        if stats["total_count"] < 50 or stats["validation_count"] < 20 or stats["train_count"] < 20:
            return False, "signal_filter_sample_below_threshold"
        if stats["train_avg_r"] > 0 and stats["validation_avg_r"] < 0:
            return False, "signal_filter_train_validation_instability"
    if validation_score <= baseline_validation_score + validation_min_delta:
        return False, "validation_score_not_improved"
    if baseline_validation_net_profit is not None and baseline_validation_net_profit > 0:
        net_profit_floor = baseline_validation_net_profit * 0.98
        if validation_metrics.get("net_profit", 0.0) < net_profit_floor:
            return False, "validation_net_profit_below_baseline_floor"
    if validation_metrics.get("total_trades", 0.0) < 50:
        return False, "validation_trade_count_below_50"
    if validation_metrics.get("profit_factor", 0.0) < 1.35:
        return False, "validation_profit_factor_below_1_35"
    return True, ""


def _evaluate_candidates(
    current_config: FamilyPortfolioBacktestConfig,
    candidates: list[Round3Candidate],
    replay_bundle: FamilyPortfolioReplayBundle,
    *,
    split_time: datetime,
    current_eval: Round3Evaluation,
    max_workers: int,
    validation_min_delta: float,
    price_bars: Any,
) -> list[Round3Evaluation]:
    worker_count = max(1, min(int(max_workers), 2))
    evaluations: list[Round3Evaluation] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                evaluate_round3_config,
                candidate.name,
                candidate.description,
                _apply_candidate(current_config, candidate),
                replay_bundle,
                split_time,
                current_eval.validation_score,
                current_eval.validation_metrics.get("net_profit", 0.0),
                validation_min_delta,
                candidate.filters,
                price_bars=price_bars,
            ): candidate
            for candidate in candidates
        }
        for future in as_completed(future_map):
            evaluations.append(future.result())
    evaluations.sort(key=_round3_selection_key, reverse=True)
    return evaluations


def _apply_candidate(
    config: FamilyPortfolioBacktestConfig,
    candidate: Round3Candidate,
) -> FamilyPortfolioBacktestConfig:
    updated = apply_portfolio_mutations(config, candidate.mutations)
    if candidate.filters:
        updated = replace(
            updated,
            signal_filter_rules=tuple(updated.signal_filter_rules) + candidate.filters,
        )
    return updated


def _split_time(
    config: FamilyPortfolioBacktestConfig,
    replay_bundle: FamilyPortfolioReplayBundle,
) -> datetime:
    result = FamilyPortfolioBacktester(config).run_bundle(replay_bundle)
    candidates = sorted(
        [*result.trades, *result.blocked_trades],
        key=lambda trade: trade.entry_time,
    )
    return candidates[int(len(candidates) * 0.60)].entry_time


def _filter_sample_stats(
    result: FamilyPortfolioResult,
    split_time: datetime,
    filter_rules: tuple[FamilySignalFilterRule, ...],
) -> list[dict[str, Any]]:
    candidates = [*result.trades, *result.blocked_trades]
    stats: list[dict[str, Any]] = []
    for rule in filter_rules:
        matches = [
            trade
            for trade in candidates
            if rule.strategy_id == trade.strategy_id
            and all(_condition_matches(trade, condition) for condition in rule.conditions)
        ]
        train = [trade for trade in matches if trade.entry_time is not None and trade.entry_time < split_time]
        validation = [
            trade
            for trade in matches
            if trade.entry_time is not None and trade.entry_time >= split_time
        ]
        stats.append({
            "rule_name": rule.name,
            "strategy_id": rule.strategy_id,
            "total_count": len(matches),
            "train_count": len(train),
            "validation_count": len(validation),
            "blocked_count": sum(1 for trade in matches if not trade.portfolio_approved),
            "full_win_rate": _raw_win_rate(matches),
            "train_win_rate": _raw_win_rate(train),
            "validation_win_rate": _raw_win_rate(validation),
            "full_avg_r": _raw_avg_r(matches),
            "train_avg_r": _raw_avg_r(train),
            "validation_avg_r": _raw_avg_r(validation),
            "full_raw_pnl": float(sum(trade.raw_pnl_dollars for trade in matches)),
            "validation_raw_pnl": float(sum(trade.raw_pnl_dollars for trade in validation)),
        })
    return stats


def _raw_win_rate(trades: list) -> float:
    return float(sum(1 for trade in trades if trade.r_multiple > 0) / len(trades)) if trades else 0.0


def _raw_avg_r(trades: list) -> float:
    return float(sum(trade.r_multiple for trade in trades) / len(trades)) if trades else 0.0


def _rule(
    name: str,
    strategy_id: str,
    field_name: str,
    op: str,
    value: object,
) -> FamilySignalFilterRule:
    return FamilySignalFilterRule(
        name=name,
        strategy_id=strategy_id,
        conditions=(FamilySignalFilterCondition(field_name, op, value),),
    )


def _phase_candidates() -> dict[int, list[Round3Candidate]]:
    return {
        1: [
            Round3Candidate(
                "filter_nqdtc_score_2_5",
                "Block the only time-split weak NQDTC score bucket",
                {},
                (_rule("nqdtc_score_2_5", "NQDTC_v2.1", "metadata.score_at_entry", "eq", 2.5),),
            ),
            Round3Candidate(
                "filter_nq_regime_wide_ib",
                "Test NQ regime wide-IB subset that failed validation in diagnostics",
                {},
                (_rule("nq_regime_wide_ib", "NQ_REGIME", "metadata.ib_type", "eq", "wide"),),
            ),
            Round3Candidate(
                "filter_vdubus_close",
                "Test Vdubus close-window quality issue",
                {},
                (_rule("vdubus_close_window", "VdubusNQ_v4", "metadata.sub_window", "eq", "CLOSE"),),
            ),
        ],
        2: [
            Round3Candidate("vdubus_second_slot", "Allow high-throughput Vdubus second concurrent slot", {"allocation.VdubusNQ_v4.max_concurrent": 2}),
            Round3Candidate("nq_regime_third_slot", "Allow NQ regime third concurrent slot", {"allocation.NQ_REGIME.max_concurrent": 3}),
            Round3Candidate(
                "selective_extra_slots",
                "Add extra slots where blocked pools retained robust edge",
                {
                    "allocation.NQ_REGIME.max_concurrent": 3,
                    "allocation.VdubusNQ_v4.max_concurrent": 2,
                    "allocation.DownturnDominator_v1.max_concurrent": 2,
                    "allocation.NQDTC_v2.1.max_concurrent": 2,
                },
            ),
        ],
        3: [
            Round3Candidate(
                "capacity_6_25_contracts_22_risk_1_5",
                "Modest live-cap lift after extra slots",
                {
                    "allocation.NQ_REGIME.max_concurrent": 3,
                    "allocation.VdubusNQ_v4.max_concurrent": 2,
                    "allocation.DownturnDominator_v1.max_concurrent": 2,
                    "allocation.NQDTC_v2.1.max_concurrent": 2,
                    "config.heat_cap_R": 6.25,
                    "rules.max_family_contracts_mnq_eq": 22,
                    "rules.directional_cap_long_R": 6.25,
                    "rules.directional_cap_short_R": 6.75,
                },
            ),
            Round3Candidate(
                "capacity_6_75_contracts_24_risk_1_5",
                "Medium live-cap lift with unchanged per-trade cap",
                {
                    "allocation.NQ_REGIME.max_concurrent": 3,
                    "allocation.VdubusNQ_v4.max_concurrent": 2,
                    "allocation.DownturnDominator_v1.max_concurrent": 2,
                    "allocation.NQDTC_v2.1.max_concurrent": 2,
                    "config.heat_cap_R": 6.75,
                    "rules.max_family_contracts_mnq_eq": 24,
                    "rules.directional_cap_long_R": 6.75,
                    "rules.directional_cap_short_R": 7.25,
                },
            ),
            Round3Candidate(
                "capacity_7_25_contracts_26_risk_1_5",
                "Upper-medium live-cap lift with unchanged per-trade cap",
                {
                    "allocation.NQ_REGIME.max_concurrent": 3,
                    "allocation.VdubusNQ_v4.max_concurrent": 2,
                    "allocation.DownturnDominator_v1.max_concurrent": 2,
                    "allocation.NQDTC_v2.1.max_concurrent": 2,
                    "config.heat_cap_R": 7.25,
                    "rules.max_family_contracts_mnq_eq": 26,
                    "rules.directional_cap_long_R": 7.25,
                    "rules.directional_cap_short_R": 7.75,
                },
            ),
            Round3Candidate(
                "capacity_8_25_contracts_30_positions_7_risk_1_5",
                "Aggressive-but-controlled live-cap lift with seven total positions",
                {
                    "allocation.NQ_REGIME.max_concurrent": 3,
                    "allocation.VdubusNQ_v4.max_concurrent": 2,
                    "allocation.DownturnDominator_v1.max_concurrent": 2,
                    "allocation.NQDTC_v2.1.max_concurrent": 2,
                    "config.heat_cap_R": 8.25,
                    "config.max_total_positions": 7,
                    "rules.max_family_contracts_mnq_eq": 30,
                    "rules.directional_cap_long_R": 8.25,
                    "rules.directional_cap_short_R": 8.75,
                },
            ),
            Round3Candidate(
                "capacity_9_00_contracts_34_positions_7_risk_1_75",
                "Aggressive frontier with higher caps and moderate per-trade cap",
                {
                    "allocation.NQ_REGIME.max_concurrent": 3,
                    "allocation.VdubusNQ_v4.max_concurrent": 2,
                    "allocation.DownturnDominator_v1.max_concurrent": 2,
                    "allocation.NQDTC_v2.1.max_concurrent": 2,
                    "config.heat_cap_R": 9.0,
                    "config.max_total_positions": 7,
                    "rules.max_family_contracts_mnq_eq": 34,
                    "rules.directional_cap_long_R": 9.0,
                    "rules.directional_cap_short_R": 9.5,
                    "dynamic.max_trade_risk_R": 1.75,
                },
            ),
            Round3Candidate(
                "capacity_10_00_contracts_40_positions_8_risk_1_75",
                "High frontier that keeps risk dynamic rather than fixed-full size",
                {
                    "allocation.NQ_REGIME.max_concurrent": 3,
                    "allocation.VdubusNQ_v4.max_concurrent": 2,
                    "allocation.DownturnDominator_v1.max_concurrent": 2,
                    "allocation.NQDTC_v2.1.max_concurrent": 2,
                    "config.heat_cap_R": 10.0,
                    "config.max_total_positions": 8,
                    "rules.max_family_contracts_mnq_eq": 40,
                    "rules.directional_cap_long_R": 10.0,
                    "rules.directional_cap_short_R": 10.5,
                    "dynamic.max_trade_risk_R": 1.75,
                },
            ),
            Round3Candidate(
                "capacity_10_00_contracts_40_positions_8_risk_2_0",
                "Stress the high frontier with the larger per-trade cap",
                {
                    "allocation.NQ_REGIME.max_concurrent": 3,
                    "allocation.VdubusNQ_v4.max_concurrent": 2,
                    "allocation.DownturnDominator_v1.max_concurrent": 2,
                    "allocation.NQDTC_v2.1.max_concurrent": 2,
                    "config.heat_cap_R": 10.0,
                    "config.max_total_positions": 8,
                    "rules.max_family_contracts_mnq_eq": 40,
                    "rules.directional_cap_long_R": 10.0,
                    "rules.directional_cap_short_R": 10.5,
                    "dynamic.max_trade_risk_R": 2.0,
                },
            ),
        ],
        4: [
            Round3Candidate(
                "filter_nq_regime_wide_ib_after_capacity",
                "Retest NQ wide-IB filter after capacity pressure is reduced",
                {},
                (_rule("nq_regime_wide_ib", "NQ_REGIME", "metadata.ib_type", "eq", "wide"),),
            ),
            Round3Candidate(
                "filter_nqdtc_score_2_5_after_capacity",
                "Retest low-score NQDTC filter after capacity pressure is reduced",
                {},
                (_rule("nqdtc_score_2_5", "NQDTC_v2.1", "metadata.score_at_entry", "eq", 2.5),),
            ),
            Round3Candidate(
                "filter_vdubus_close_after_capacity",
                "Retest Vdubus close-window filter after capacity pressure is reduced",
                {},
                (_rule("vdubus_close_window", "VdubusNQ_v4", "metadata.sub_window", "eq", "CLOSE"),),
            ),
            Round3Candidate(
                "filter_nq_regime_wide_and_nqdtc_low_score",
                "Combined quality overlay using only entry-time metadata",
                {},
                (
                    _rule("nq_regime_wide_ib", "NQ_REGIME", "metadata.ib_type", "eq", "wide"),
                    _rule("nqdtc_score_2_5", "NQDTC_v2.1", "metadata.score_at_entry", "eq", 2.5),
                ),
            ),
        ],
    }


PHASE_CANDIDATES = _phase_candidates()


def _round3_selection_key(evaluation: Round3Evaluation) -> tuple[float, float, float, float, float, float, float, float]:
    metrics = evaluation.metrics
    validation_metrics = evaluation.validation_metrics
    return (
        1.0 if evaluation.robust_pass else 0.0,
        evaluation.score,
        evaluation.validation_score,
        float(validation_metrics.get("net_profit", 0.0) or 0.0),
        float(metrics.get("net_profit", 0.0) or 0.0),
        float(metrics.get("trades_per_month", 0.0) or 0.0),
        -float(metrics.get("max_drawdown_pct", 1.0) or 1.0),
        -float(metrics.get("block_rate", 1.0) or 1.0),
    )


def _evaluation_record(evaluation: Round3Evaluation) -> dict[str, Any]:
    return {
        "name": evaluation.name,
        "description": evaluation.description,
        "score": evaluation.score,
        "validation_score": evaluation.validation_score,
        "robust_pass": evaluation.robust_pass,
        "robust_reason": evaluation.robust_reason,
        "rejected": evaluation.rejected,
        "reject_reason": evaluation.reject_reason,
        "soft_warnings": evaluation.soft_warnings,
        "components": evaluation.components,
        "metrics": evaluation.metrics,
        "validation_metrics": evaluation.validation_metrics,
        "train_metrics": evaluation.train_metrics,
        "filter_sample_stats": evaluation.filter_sample_stats,
        "config": family_config_to_dict(evaluation.config),
    }


def _write_signal_selection_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Round 3 Signal Selection Diagnostics",
        "",
        f"Split time UTC: {summary['split_time_utc']}",
        f"Score components: {summary['score_component_count']}",
        f"Max workers: {summary['max_workers']}",
        "",
        "## Accepted Phases",
        "",
        "| Phase | Accepted | Candidate | Score | Validation Score |",
        "|---:|---|---|---:|---:|",
    ]
    for phase in summary["phases"]:
        candidate = phase["accepted_candidate"] or ""
        lines.append(
            f"| {phase['phase']} | {phase['accepted']} | {candidate} | "
            f"{phase['current_score']:.4f} | {phase['current_validation_score']:.4f} |"
        )

    lines.extend([
        "",
        "## Candidate Frontier",
        "",
        "| Phase | Candidate | Robust | Reason | Score | Val Score | Net Profit | Trades | Block Rate | WR | PF | Max DD | Val Net |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for phase in summary["phases"]:
        for evaluation in phase["evaluations"]:
            metrics = evaluation["metrics"]
            validation = evaluation["validation_metrics"]
            reason = evaluation["robust_reason"] or ""
            lines.append(
                f"| {phase['phase']} | {evaluation['name']} | {evaluation['robust_pass']} | {reason} | "
                f"{evaluation['score']:.4f} | {evaluation['validation_score']:.4f} | "
                f"${metrics['net_profit']:,.0f} | {metrics['total_trades']:.0f} | "
                f"{metrics['block_rate']:.1%} | {metrics['win_rate']:.1%} | "
                f"{metrics['profit_factor']:.2f} | {metrics['max_drawdown_pct']:.2%} | "
                f"${validation['net_profit']:,.0f} |"
            )

    filter_rows = [
        (phase["phase"], evaluation)
        for phase in summary["phases"]
        for evaluation in phase["evaluations"]
        if evaluation["filter_sample_stats"]
    ]
    if filter_rows:
        lines.extend([
            "",
            "## Signal Filter Sample Checks",
            "",
            "| Phase | Candidate | Rule | Full N | Train N | Val N | Train Avg R | Val Avg R | Raw Val PnL | Gate Note |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for phase, evaluation in filter_rows:
            for stats in evaluation["filter_sample_stats"]:
                gate_note = ""
                if (
                    stats["total_count"] < 50
                    or stats["train_count"] < 20
                    or stats["validation_count"] < 20
                ):
                    gate_note = "sample_below_threshold"
                elif stats["train_avg_r"] > 0 and stats["validation_avg_r"] < 0:
                    gate_note = "train_validation_instability"
                lines.append(
                    f"| {phase} | {evaluation['name']} | {stats['rule_name']} | "
                    f"{stats['total_count']} | {stats['train_count']} | {stats['validation_count']} | "
                    f"{stats['train_avg_r']:.2f} | {stats['validation_avg_r']:.2f} | "
                    f"${stats['validation_raw_pnl']:,.0f} | {gate_note} |"
                )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Round 3 only used entry-time metadata for signal filters.",
        "- Small or train/validation-unstable filter buckets were treated as overfit risks.",
        "- The accepted improvement came from letting robust blocked opportunity through shared live-style caps, not from a brittle low-sample exclusion rule.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
