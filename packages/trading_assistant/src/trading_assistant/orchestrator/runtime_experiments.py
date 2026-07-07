"""Runtime-owned experiment conclusion jobs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

METRIC_FIELD_BY_NAME = {
    "sharpe": "sharpe_ratio_30d",
    "sharpe_ratio": "sharpe_ratio_30d",
    "sharpe_ratio_30d": "sharpe_ratio_30d",
    "win_rate": "win_rate",
    "drawdown": "max_drawdown_pct",
    "max_drawdown": "max_drawdown_pct",
    "max_drawdown_pct": "max_drawdown_pct",
    "process_quality": "avg_process_quality",
    "avg_process_quality": "avg_process_quality",
    "composite": "composite_score",
    "composite_score": "composite_score",
    "pnl": "pnl_total",
    "pnl_total": "pnl_total",
    "net_pnl": "pnl_total",
    "calmar": "calmar_ratio_30d",
    "calmar_ratio": "calmar_ratio_30d",
    "calmar_ratio_30d": "calmar_ratio_30d",
    "profit_factor": "profit_factor",
    "expectancy": "expectancy",
    "expected_r": "expected_total_r",
    "expected_total_r": "expected_total_r",
    "trade_count": "trade_count",
}


def experiment_result_delta(result: Any, experiment: Any) -> float:
    metrics = {metric.variant_name: metric for metric in getattr(result, "variant_metrics", [])}
    variants = getattr(experiment, "variants", []) or []
    if len(variants) >= 2:
        control = metrics.get(variants[0].name)
        treatment = metrics.get(variants[1].name)
        metric_attr = {
            "pnl": "avg_pnl",
            "sharpe": "sharpe",
            "win_rate": "win_rate",
            "profit_factor": "profit_factor",
        }.get(getattr(experiment, "success_metric", "pnl"), "avg_pnl")
        if control is not None and treatment is not None:
            return float(
                (getattr(treatment, metric_attr, 0.0) or 0.0)
                - (getattr(control, metric_attr, 0.0) or 0.0)
            )
    return float(getattr(result, "effect_size", 0.0) or 0.0)


def structural_objective_delta(
    gt_before: Any,
    gt_after: Any,
    actual_values: list[float],
) -> float:
    before_composite = getattr(gt_before, "composite_score", None)
    after_composite = getattr(gt_after, "composite_score", None)
    if before_composite is not None and after_composite is not None:
        return float(after_composite - before_composite)
    values = [float(value) for value in actual_values if value is not None]
    return sum(values) / len(values) if values else 0.0


async def run_experiment_checks(
    *,
    experiment_manager: Any,
    structural_experiment_tracker: Any,
    event_stream: Any,
    memory_dir: Path,
    curated_dir: Path,
    suggestion_tracker: Any,
    hypothesis_library: Any,
    telegram_adapter: Any,
    lookup_proposal_id: Callable[..., str],
    record_proposal_outcome: Callable[..., None],
) -> None:
    if experiment_manager is not None:
        await _conclude_ab_experiments(
            experiment_manager=experiment_manager,
            event_stream=event_stream,
            memory_dir=memory_dir,
            suggestion_tracker=suggestion_tracker,
            hypothesis_library=hypothesis_library,
            telegram_adapter=telegram_adapter,
            lookup_proposal_id=lookup_proposal_id,
            record_proposal_outcome=record_proposal_outcome,
        )
    await _evaluate_structural_experiments(
        structural_experiment_tracker=structural_experiment_tracker,
        curated_dir=curated_dir,
        memory_dir=memory_dir,
        hypothesis_library=hypothesis_library,
        lookup_proposal_id=lookup_proposal_id,
        record_proposal_outcome=record_proposal_outcome,
    )


async def _conclude_ab_experiments(
    *,
    experiment_manager: Any,
    event_stream: Any,
    memory_dir: Path,
    suggestion_tracker: Any,
    hypothesis_library: Any,
    telegram_adapter: Any,
    lookup_proposal_id: Callable[..., str],
    record_proposal_outcome: Callable[..., None],
) -> None:
    try:
        for experiment in experiment_manager.get_active():
            if not experiment_manager.check_auto_conclusion(experiment.experiment_id):
                continue
            result = experiment_manager.analyze_experiment(experiment.experiment_id)
            experiment_manager.conclude_experiment(experiment.experiment_id, result)
            logger.info(
                "Auto-concluded experiment %s: %s",
                experiment.experiment_id,
                result.recommendation,
            )
            event_stream.broadcast("experiment_concluded", {
                "experiment_id": experiment.experiment_id,
                "recommendation": result.recommendation,
                "winner": result.winner,
                "p_value": result.p_value,
            })
            proposal_verdict = {
                "adopt_treatment": "positive",
                "keep_control": "negative",
            }.get(result.recommendation, "inconclusive")
            record_proposal_outcome(
                proposal_id=lookup_proposal_id(
                    suggestion_id=getattr(experiment, "source_suggestion_id", "") or "",
                    experiment_id=experiment.experiment_id,
                ),
                objective_delta=experiment_result_delta(result, experiment),
                verdict=proposal_verdict,
                measurement_path=memory_dir / "findings" / "experiment_results.jsonl",
            )
            _record_experiment_lifecycle(
                experiment=experiment,
                result=result,
                suggestion_tracker=suggestion_tracker,
                hypothesis_library=hypothesis_library,
            )
            if telegram_adapter is not None:
                await _send_experiment_result(telegram_adapter, experiment, result)
    except Exception:
        logger.exception("Experiment check failed")


def _record_experiment_lifecycle(
    *,
    experiment: Any,
    result: Any,
    suggestion_tracker: Any,
    hypothesis_library: Any,
) -> None:
    if result.recommendation == "adopt_treatment":
        if getattr(experiment, "source_suggestion_id", None) and suggestion_tracker:
            try:
                suggestion_tracker.accept(experiment.source_suggestion_id)
                logger.info(
                    "Auto-accepted suggestion %s (experiment %s passed)",
                    experiment.source_suggestion_id,
                    experiment.experiment_id,
                )
            except Exception:
                logger.warning(
                    "Failed to auto-accept suggestion %s",
                    experiment.source_suggestion_id,
                )
        if getattr(experiment, "hypothesis_id", None):
            try:
                hypothesis_library.record_outcome(experiment.hypothesis_id, positive=True)
                logger.info("Recorded positive outcome for hypothesis %s", experiment.hypothesis_id)
            except Exception:
                logger.warning("Failed to record hypothesis outcome %s", experiment.hypothesis_id)
    elif result.recommendation == "keep_control" and getattr(experiment, "hypothesis_id", None):
        try:
            hypothesis_library.record_outcome(experiment.hypothesis_id, positive=False)
        except Exception:
            pass


async def _send_experiment_result(telegram_adapter: Any, experiment: Any, result: Any) -> None:
    try:
        from trading_assistant.comms.telegram_renderer import TelegramRenderer

        text = TelegramRenderer().render_experiment_result(experiment, result)
        await telegram_adapter.send_message(text)
    except Exception:
        logger.warning("Failed to send experiment result notification")


async def _evaluate_structural_experiments(
    *,
    structural_experiment_tracker: Any,
    curated_dir: Path,
    memory_dir: Path,
    hypothesis_library: Any,
    lookup_proposal_id: Callable[..., str],
    record_proposal_outcome: Callable[..., None],
) -> None:
    if structural_experiment_tracker is None:
        return
    try:
        from trading_assistant.skills.ground_truth_computer import GroundTruthComputer

        ground_truth = GroundTruthComputer(curated_dir)
        for experiment in structural_experiment_tracker.get_evaluable_experiments():
            try:
                _evaluate_structural_experiment(
                    experiment=experiment,
                    ground_truth=ground_truth,
                    structural_experiment_tracker=structural_experiment_tracker,
                    memory_dir=memory_dir,
                    hypothesis_library=hypothesis_library,
                    lookup_proposal_id=lookup_proposal_id,
                    record_proposal_outcome=record_proposal_outcome,
                )
            except Exception:
                logger.exception(
                    "Failed to evaluate structural experiment %s",
                    experiment.experiment_id,
                )
    except Exception:
        logger.exception("Structural experiment check failed")


def _evaluate_structural_experiment(
    *,
    experiment: Any,
    ground_truth: Any,
    structural_experiment_tracker: Any,
    memory_dir: Path,
    hypothesis_library: Any,
    lookup_proposal_id: Callable[..., str],
    record_proposal_outcome: Callable[..., None],
) -> None:
    activated_date = experiment.activated_at.strftime("%Y-%m-%d") if experiment.activated_at else ""
    if not activated_date:
        return
    gt_before = ground_truth.compute_snapshot(experiment.bot_id, activated_date)
    gt_after = ground_truth.compute_snapshot(
        experiment.bot_id,
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    max_min_trades = (
        max((getattr(criteria, "minimum_trade_count", 20) or 20) for criteria in experiment.acceptance_criteria)
        if experiment.acceptance_criteria
        else 20
    )
    actual_trades = getattr(gt_after, "trade_count", None)
    if actual_trades is not None and actual_trades < max_min_trades:
        logger.info(
            "Experiment %s: insufficient trades (%d < %d), skipping evaluation",
            experiment.experiment_id,
            actual_trades,
            max_min_trades,
        )
        return

    criteria_met: list[bool] = []
    actual_values: list[float] = []
    for criteria in experiment.acceptance_criteria:
        field = METRIC_FIELD_BY_NAME.get(criteria.metric, criteria.metric)
        before_value = getattr(gt_before, field, None)
        after_value = getattr(gt_after, field, None)
        if before_value is None or after_value is None:
            criteria_met.append(False)
            actual_values.append(0.0)
            continue
        delta = after_value - before_value
        actual_values.append(round(delta, 4))
        if criteria.direction == "improve":
            passed = delta >= criteria.minimum_change
        else:
            passed = delta >= -criteria.minimum_change
        if passed and getattr(criteria, "baseline_value", None) is not None:
            passed = after_value is not None and after_value >= criteria.baseline_value
        criteria_met.append(passed)

    structural_experiment_tracker.resolve(experiment.experiment_id, criteria_met, actual_values)
    passed = all(criteria_met) if criteria_met else False
    logger.info(
        "Resolved structural experiment %s: %s",
        experiment.experiment_id,
        "PASSED" if passed else "FAILED",
    )

    if experiment.hypothesis_id:
        try:
            hypothesis_library.record_outcome(experiment.hypothesis_id, positive=passed)
        except Exception:
            logger.warning("Failed to record hypothesis outcome for %s", experiment.hypothesis_id)
    record_proposal_outcome(
        proposal_id=lookup_proposal_id(
            suggestion_id=experiment.suggestion_id or "",
            experiment_id=experiment.experiment_id,
        ),
        objective_delta=structural_objective_delta(gt_before, gt_after, actual_values),
        verdict="positive" if passed else "negative",
        measurement_path=memory_dir / "findings" / "structural_experiments.jsonl",
    )
