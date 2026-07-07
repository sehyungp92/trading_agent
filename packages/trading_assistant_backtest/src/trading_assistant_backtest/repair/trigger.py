"""Selection-OOS repair trigger."""

from __future__ import annotations

from typing import Any


def material_underperformance(latest_oos_delta: float, *, threshold: float = -0.05) -> bool:
    return latest_oos_delta <= threshold


DEFAULT_REPAIR_THRESHOLDS = {
    "objective_drop_threshold": -0.05,
    "drawdown_increase_threshold": 0.05,
    "trade_count_collapse_ratio": 0.50,
    "cost_sensitivity_threshold": -0.001,
    "under_trading_min_trades": 1,
}


def evaluate_selection_oos_repair_trigger(
    *,
    run_id: str,
    incumbent: dict[str, Any],
    candidate: dict[str, Any] | None,
    fold_profile: dict[str, Any],
    force_trigger: bool = False,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    limits = {**DEFAULT_REPAIR_THRESHOLDS, **(thresholds or {})}
    candidate_payload = candidate or {}
    fold_mean = float(fold_profile.get("mean_objective_score") or 0.0)
    fold_trade_mean = float(fold_profile.get("mean_trade_count") or 0.0)
    fold_drawdown_mean = float(fold_profile.get("mean_max_drawdown") or 0.0)
    observed_score = float(
        candidate_payload.get("objective_score", incumbent.get("objective_score", 0.0)) or 0.0
    )
    observed_trades = float(
        candidate_payload.get("trade_count", incumbent.get("trade_count", 0)) or 0.0
    )
    observed_drawdown = float(
        candidate_payload.get("max_drawdown", incumbent.get("max_drawdown", 0.0)) or 0.0
    )
    objective_degradation = observed_score - fold_mean
    drawdown_increase = observed_drawdown - fold_drawdown_mean
    trade_ratio = observed_trades / fold_trade_mean if fold_trade_mean > 0 else 1.0
    sparse_sample = observed_trades < float(limits["under_trading_min_trades"])
    profile_missing = not fold_profile or "mean_objective_score" not in fold_profile
    reasons: list[str] = []
    if force_trigger:
        reasons.append("explicit smoke_repair mode requested repair evidence")
    if profile_missing and not force_trigger:
        reasons.append("selection-OOS repair skipped because no in-sample fold profile was available")
    else:
        if objective_degradation <= float(limits["objective_drop_threshold"]):
            reasons.append("selection-OOS objective materially trails the in-sample fold profile")
        if drawdown_increase >= float(limits["drawdown_increase_threshold"]):
            reasons.append("selection-OOS drawdown materially exceeds the in-sample fold profile")
        if fold_trade_mean > 0 and trade_ratio <= float(limits["trade_count_collapse_ratio"]):
            reasons.append("selection-OOS trade count collapsed versus in-sample folds")
    if sparse_sample and not reasons:
        reasons.append("selection-OOS sparse sample is within tolerance; repair not triggered")
    triggered = force_trigger or (
        not profile_missing
        and any(
            reason
            for reason in reasons
            if "within tolerance" not in reason and "skipped" not in reason
        )
    )
    return {
        "schema_version": "selection_oos_repair_trigger_v1",
        "run_id": run_id,
        "triggered": triggered,
        "status": "triggered" if triggered else "not_triggered",
        "thresholds": limits,
        "incumbent_selection_oos": incumbent,
        "candidate_selection_oos": candidate_payload,
        "expected_is_fold_score_band": {
            "mean_objective_score": fold_mean,
            "min_objective_score": float(fold_profile.get("min_objective_score") or 0.0),
            "max_objective_score": float(fold_profile.get("max_objective_score") or 0.0),
            "mean_trade_count": fold_trade_mean,
            "mean_max_drawdown": fold_drawdown_mean,
        },
        "measured_degradation": {
            "objective_delta_vs_fold_mean": objective_degradation,
            "drawdown_increase_vs_fold_mean": drawdown_increase,
            "trade_count_ratio_vs_fold_mean": trade_ratio,
        },
        "sample_size_caveat": (
            "selection-OOS sparse sample"
            if sparse_sample
            else ""
        ),
        "reasons": list(dict.fromkeys(reasons)),
    }
