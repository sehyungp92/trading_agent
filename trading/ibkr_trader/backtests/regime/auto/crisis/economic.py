"""Economic replay and action-layer optimization for crisis detection.

The threshold optimizer answers "did the detector fire?" This module answers
"did acting on the detector improve the portfolio after costs/opportunity
cost?" It intentionally freezes the R5 detector thresholds and searches over
the action layer: WARNING/CRISIS multipliers plus optional advisory/shock/grind
pre-action tightening.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtests.regime.analysis.metrics import compute_metrics
from backtests.regime.crisis_validation import CRISIS_PERIODS, run_crisis_detector
from regime.crisis import config as C


DEFAULT_SIGNALS_PATH = Path("backtests/regime/auto/output/optimized_signals.parquet")
INITIAL_EQUITY = 100_000.0
PORTFOLIO_SCORE_WEIGHTS = {
    "regime_proxy": 0.60,
    "balanced_60_40": 0.20,
    "spy": 0.20,
}
SLEEVE_SCORE_WEIGHTS = {
    "equity_only": 0.15,
    "overlay_qqq_gld_60_40": 0.20,
    "overlay_qqq_gld_50_50": 0.10,
    "risk_on_with_hedges": 0.25,
    "crisis_stack": 0.15,
    "long_short_macro": 0.10,
    "defensive_mix": 0.05,
}
SLEEVE_PORTFOLIOS = {
    "equity_only": {"equity_beta": 1.00},
    "overlay_qqq_gld_60_40": {"qqq_proxy": 0.60, "gld": 0.40},
    "overlay_qqq_gld_50_50": {"qqq_proxy": 0.50, "gld": 0.50},
    "risk_on_with_hedges": {"equity_beta": 0.70, "gld": 0.20, "short_spy": 0.10},
    "crisis_stack": {"equity_beta": 0.55, "gld": 0.25, "short_spy": 0.20},
    "long_short_macro": {"equity_beta": 0.60, "gld": 0.10, "short_spy": 0.30},
    "defensive_mix": {"equity_beta": 0.30, "gld": 0.50, "short_spy": 0.20},
}


@dataclass(frozen=True)
class CrisisEconomicPolicy:
    """Exposure multipliers used by the economic replay."""

    name: str = "current_live"
    warning_mult: float = C.RISK_MULT_WARNING
    crisis_mult: float = C.RISK_MULT_CRISIS
    advisory_mult: float = 1.00
    shock_mult: float = C.STRESS_FORMATION_RISK_MULT_SHOCK
    grind_mult: float = C.STRESS_FORMATION_RISK_MULT_GRIND
    credit_impulse_mult: float = C.STRESS_FORMATION_RISK_MULT_CREDIT_IMPULSE
    credit_bridge_warning_mult: float = C.ACTION_CREDIT_BRIDGE_WARNING_RISK_MULT
    stress_regime_warning_mult: float = C.ACTION_WARNING_RISK_MULT_STRESS_REGIME
    defensive_regime_warning_mult: float = C.ACTION_WARNING_RISK_MULT_DEFENSIVE_REGIME

    def with_updates(self, name: str, updates: dict[str, float]) -> "CrisisEconomicPolicy":
        data = asdict(self)
        data.update(updates)
        data["name"] = name
        return CrisisEconomicPolicy(**data)


@dataclass(frozen=True)
class CrisisSleeveEconomicPolicy:
    """Sleeve-aware action multipliers for economic replay.

    Equity beta proxies QQQ/NQ/stock-beta longs. GLD is replayed separately
    because HMM already rotates the swing overlay toward GLD in stress regimes.
    Short exposure is replayed separately to test whether validated shorts
    should keep sizing capacity during confirmed WARNING/CRISIS.
    """

    name: str = "current_symmetric"
    equity_warning_mult: float = C.RISK_MULT_WARNING
    equity_crisis_mult: float = C.RISK_MULT_CRISIS
    gld_warning_mult: float = C.RISK_MULT_WARNING
    gld_crisis_mult: float = C.RISK_MULT_CRISIS
    short_warning_mult: float = C.RISK_MULT_WARNING
    short_crisis_mult: float = C.RISK_MULT_CRISIS

    def with_name(self, name: str) -> "CrisisSleeveEconomicPolicy":
        data = asdict(self)
        data["name"] = name
        return CrisisSleeveEconomicPolicy(**data)


@dataclass(frozen=True)
class EconomicScore:
    """Return-quality score centered around 0.5 for no improvement."""

    calmar_component: float
    sortino_component: float
    maxdd_component: float
    crisis_dd_component: float
    cagr_component: float
    recovery_component: float
    friction_component: float
    total: float
    rejected: bool = False
    reject_reason: str = ""


def load_replay_inputs(
    data_dir: Path,
    *,
    signals_path: Path = DEFAULT_SIGNALS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load market data, returns, and optional regime proxy signals."""
    market_df = pd.read_parquet(data_dir / "market_df.parquet")
    strat_ret_df = pd.read_parquet(data_dir / "strat_ret_df.parquet")
    if signals_path.exists():
        signals_df = pd.read_parquet(signals_path)
    else:
        signals_df = pd.DataFrame()
    return market_df, strat_ret_df, signals_df


def build_base_portfolios(
    strat_ret_df: pd.DataFrame,
    signals_df: pd.DataFrame | None = None,
) -> dict[str, pd.Series]:
    """Build proxy portfolio return series for economic overlay testing.

    The primary portfolio is the latest optimized regime allocation proxy when
    weekly `w_*` columns are available. SPY and 60/40 are included as robustness
    proxies so the crisis action layer is not only fit to one allocation path.
    """
    returns = strat_ret_df.copy().fillna(0.0)
    portfolios: dict[str, pd.Series] = {}

    if signals_df is not None and not signals_df.empty:
        weight_cols = [c for c in signals_df.columns if c.startswith("w_")]
        assets = [c[2:] for c in weight_cols if c[2:] in returns.columns]
        if assets:
            weights = signals_df[[f"w_{asset}" for asset in assets]].copy()
            weights.columns = assets
            daily_weights = weights.reindex(returns.index).ffill()
            # Apply weights one trading day after the signal date to avoid
            # same-close lookahead in this economic replay.
            daily_weights = daily_weights.shift(1).ffill().fillna(0.0)
            row_sum = daily_weights.sum(axis=1).replace(0.0, np.nan)
            daily_weights = daily_weights.div(row_sum, axis=0).fillna(0.0)
            portfolios["regime_proxy"] = (daily_weights[assets] * returns[assets]).sum(axis=1)

    if "regime_proxy" not in portfolios:
        portfolios["regime_proxy"] = _weighted_returns(
            returns, {"SPY": 0.30, "EFA": 0.10, "TLT": 0.40, "GLD": 0.10, "CASH": 0.10},
        )
    portfolios["balanced_60_40"] = _weighted_returns(returns, {"SPY": 0.60, "TLT": 0.40})
    portfolios["spy"] = returns.get("SPY", pd.Series(0.0, index=returns.index))
    return portfolios


def evaluate_policy(
    *,
    alerts_df: pd.DataFrame,
    base_portfolios: dict[str, pd.Series],
    cash_returns: pd.Series,
    policy: CrisisEconomicPolicy,
    portfolio_score_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Replay a crisis action policy over one or more base portfolios."""
    weights = portfolio_score_weights or PORTFOLIO_SCORE_WEIGHTS
    exposure = build_exposure_series(alerts_df, policy)
    cash = cash_returns.reindex(exposure.index).fillna(0.0)

    portfolio_results: dict[str, Any] = {}
    weighted_score = 0.0
    score_weight_sum = 0.0
    rejected_reasons: list[str] = []

    for name, base_ret in base_portfolios.items():
        aligned_base = base_ret.reindex(exposure.index).fillna(0.0)
        overlay_ret = exposure * aligned_base + (1.0 - exposure) * cash
        turnover = exposure.diff().abs().fillna(0.0)

        base_metrics = _metrics_for_returns(aligned_base, turnover * 0.0)
        overlay_metrics = _metrics_for_returns(overlay_ret, turnover)
        period = _period_diagnostics(aligned_base, overlay_ret)
        score = economic_score(base_metrics, overlay_metrics, period, exposure)
        if score.rejected:
            rejected_reasons.append(f"{name}: {score.reject_reason}")

        portfolio_results[name] = {
            "base_metrics": base_metrics,
            "overlay_metrics": overlay_metrics,
            "period_diagnostics": period,
            "score": asdict(score),
            "deltas": _metric_deltas(base_metrics, overlay_metrics),
        }
        w = float(weights.get(name, 0.0))
        weighted_score += w * score.total
        score_weight_sum += w

    aggregate_score = weighted_score / score_weight_sum if score_weight_sum else 0.0
    action_day_share = float((exposure < 0.999).mean())
    if action_day_share > 0.45:
        rejected_reasons.append(f"action_day_share {action_day_share:.1%} > 45%")

    return {
        "policy": asdict(policy),
        "score": 0.0 if rejected_reasons else aggregate_score,
        "rejected": bool(rejected_reasons),
        "reject_reason": "; ".join(rejected_reasons),
        "action_day_share": action_day_share,
        "avg_exposure": float(exposure.mean()),
        "min_exposure": float(exposure.min()),
        "n_exposure_transitions": int((exposure.diff().abs().fillna(0.0) > 1e-12).sum()),
        "portfolio_results": portfolio_results,
    }


def economic_score(
    base_metrics: dict[str, float],
    overlay_metrics: dict[str, float],
    period: dict[str, float],
    exposure: pd.Series,
) -> EconomicScore:
    """Score economic overlay quality relative to no-overlay baseline."""
    cagr_delta = overlay_metrics["cagr"] - base_metrics["cagr"]
    maxdd_reduction = base_metrics["max_drawdown_pct"] - overlay_metrics["max_drawdown_pct"]

    if overlay_metrics["max_drawdown_pct"] > base_metrics["max_drawdown_pct"] + 0.01:
        return EconomicScore(0, 0, 0, 0, 0, 0, 0, 0, True, "max_drawdown worsened > 1pp")
    if cagr_delta < -0.02 and maxdd_reduction < 0.005:
        return EconomicScore(0, 0, 0, 0, 0, 0, 0, 0, True, "CAGR drag without DD relief")

    calmar_c = _centered_component(overlay_metrics["calmar"] - base_metrics["calmar"], 0.50)
    sortino_c = _centered_component(overlay_metrics["sortino"] - base_metrics["sortino"], 0.75)
    maxdd_c = _centered_component(maxdd_reduction, 0.05)
    crisis_dd_c = _centered_component(period["avg_crisis_dd_reduction"], 0.08)
    cagr_c = _centered_component(cagr_delta, 0.03)
    recovery_c = _centered_component(
        base_metrics["max_drawdown_duration"] - overlay_metrics["max_drawdown_duration"],
        120.0,
    )
    friction_penalty = (
        min(float(exposure.diff().abs().sum()) / 20.0, 0.5)
        + min(period["avg_missed_rebound_drag"] / 0.05, 0.5)
    )
    friction_c = _clip01(1.0 - friction_penalty)

    total = (
        0.30 * calmar_c
        + 0.20 * sortino_c
        + 0.20 * maxdd_c
        + 0.10 * crisis_dd_c
        + 0.10 * cagr_c
        + 0.05 * recovery_c
        + 0.05 * friction_c
    )
    return EconomicScore(
        calmar_component=calmar_c,
        sortino_component=sortino_c,
        maxdd_component=maxdd_c,
        crisis_dd_component=crisis_dd_c,
        cagr_component=cagr_c,
        recovery_component=recovery_c,
        friction_component=friction_c,
        total=total,
    )


def build_exposure_series(
    alerts_df: pd.DataFrame,
    policy: CrisisEconomicPolicy,
) -> pd.Series:
    """Convert crisis/advisory/stress states into daily exposure multipliers."""
    exposure = pd.Series(1.0, index=alerts_df.index, dtype=float)
    action = alerts_df.get("portfolio_action_level_int", alerts_df["alert_level_int"])
    advisory = alerts_df.get("advisory_level_int", pd.Series(0, index=alerts_df.index))

    exposure.loc[(action < 2) & (advisory >= 1)] = np.minimum(
        exposure.loc[(action < 2) & (advisory >= 1)],
        policy.advisory_mult,
    )

    if "stress_formation_mode" in alerts_df.columns:
        mode = alerts_df["stress_formation_mode"].fillna("")
        shock_mask = (action < 2) & mode.str.contains("shock", regex=False)
        grind_mask = (action < 2) & mode.str.contains("grind", regex=False)
        credit_impulse_mask = (
            (action < 2) & mode.str.contains("credit_impulse", regex=False)
        )
        exposure.loc[shock_mask] = np.minimum(exposure.loc[shock_mask], policy.shock_mult)
        exposure.loc[grind_mask] = np.minimum(exposure.loc[grind_mask], policy.grind_mult)
        exposure.loc[credit_impulse_mask] = np.minimum(
            exposure.loc[credit_impulse_mask],
            policy.credit_impulse_mult,
        )

    warning_mask = action == 2
    exposure.loc[warning_mask] = policy.warning_mult
    if "stress_formation_mode" in alerts_df.columns:
        mode = alerts_df["stress_formation_mode"].fillna("")
        warning_count = alerts_df.get(
            "primary_warning_count",
            pd.Series(C.WARNING_MIN_PRIMARY, index=alerts_df.index),
        )
        crisis_count = alerts_df.get(
            "primary_crisis_count",
            pd.Series(0, index=alerts_df.index),
        )
        bridge_warning_mask = (
            warning_mask
            & mode.str.contains("credit_impulse", regex=False)
            & (warning_count < C.WARNING_MIN_PRIMARY)
            & (crisis_count <= 0)
        )
        exposure.loc[bridge_warning_mask] = np.maximum(
            exposure.loc[bridge_warning_mask],
            policy.credit_bridge_warning_mult,
        )

    regime = _regime_series(alerts_df)
    if regime is not None:
        stress_warning_mask = warning_mask & (regime == "S")
        defensive_warning_mask = warning_mask & (regime == "D")
        exposure.loc[stress_warning_mask] = np.maximum(
            exposure.loc[stress_warning_mask],
            policy.stress_regime_warning_mult,
        )
        exposure.loc[defensive_warning_mask] = np.maximum(
            exposure.loc[defensive_warning_mask],
            policy.defensive_regime_warning_mult,
        )

    exposure.loc[action >= 3] = policy.crisis_mult
    return exposure.clip(lower=0.0, upper=1.0)


def optimize_action_policy(
    *,
    alerts_df: pd.DataFrame,
    base_portfolios: dict[str, pd.Series],
    cash_returns: pd.Series,
    min_delta: float = 0.001,
    max_rounds: int = 8,
) -> dict[str, Any]:
    """Greedy action-layer optimization, with thresholds held fixed."""
    current = CrisisEconomicPolicy()
    base_eval = evaluate_policy(
        alerts_df=alerts_df,
        base_portfolios=base_portfolios,
        cash_returns=cash_returns,
        policy=current,
    )
    current_score = float(base_eval["score"])
    current_eval = base_eval
    rounds: list[dict[str, Any]] = []
    accepted: list[str] = []

    for round_num in range(1, max_rounds + 1):
        best_name = ""
        best_policy = current
        best_eval: dict[str, Any] | None = None
        best_score = -math.inf
        rejected = 0
        for name, updates in _policy_candidates(current):
            candidate = current.with_updates(name, updates)
            result = evaluate_policy(
                alerts_df=alerts_df,
                base_portfolios=base_portfolios,
                cash_returns=cash_returns,
                policy=candidate,
            )
            if result["rejected"]:
                rejected += 1
                continue
            score = float(result["score"])
            if score > best_score:
                best_name = name
                best_policy = candidate
                best_eval = result
                best_score = score

        delta = best_score - current_score
        kept = bool(best_eval is not None and delta >= min_delta)
        rounds.append({
            "round": round_num,
            "best_name": best_name,
            "best_score": best_score if best_eval is not None else None,
            "delta": delta if best_eval is not None else None,
            "kept": kept,
            "rejected_count": rejected,
        })
        if not kept:
            break
        current = best_policy
        current_score = best_score
        current_eval = best_eval
        accepted.append(best_name)

    scenarios = evaluate_standard_scenarios(
        alerts_df=alerts_df,
        base_portfolios=base_portfolios,
        cash_returns=cash_returns,
    )
    return {
        "baseline_policy": asdict(CrisisEconomicPolicy()),
        "optimized_policy": asdict(current),
        "base_score": float(base_eval["score"]),
        "optimized_score": float(current_score),
        "accepted": accepted,
        "rounds": rounds,
        "optimized_evaluation": current_eval,
        "standard_scenarios": scenarios,
    }


def evaluate_standard_scenarios(
    *,
    alerts_df: pd.DataFrame,
    base_portfolios: dict[str, pd.Series],
    cash_returns: pd.Series,
) -> dict[str, Any]:
    """Evaluate the A-E scenarios from the design recommendation."""
    policies = [
        CrisisEconomicPolicy(
            name="A_no_crisis",
            warning_mult=1.0,
            crisis_mult=1.0,
            advisory_mult=1.0,
            shock_mult=1.0,
            grind_mult=1.0,
            credit_impulse_mult=1.0,
        ),
        CrisisEconomicPolicy(
            name="B_threshold_only",
            shock_mult=1.0,
            grind_mult=1.0,
            credit_impulse_mult=1.0,
        ),
        CrisisEconomicPolicy(name="C_current_live"),
        CrisisEconomicPolicy(name="D_advisory_light", advisory_mult=0.95),
        CrisisEconomicPolicy(
            name="E_legacy_r5_threshold",
            warning_mult=0.75,
            crisis_mult=0.50,
            shock_mult=1.0,
            grind_mult=1.0,
            credit_impulse_mult=1.0,
        ),
    ]
    return {
        p.name: evaluate_policy(
            alerts_df=alerts_df,
            base_portfolios=base_portfolios,
            cash_returns=cash_returns,
            policy=p,
        )
        for p in policies
    }


def evaluate_sleeve_policy(
    *,
    alerts_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    cash_returns: pd.Series,
    policy: CrisisSleeveEconomicPolicy,
    sleeve_portfolios: dict[str, dict[str, float]] | None = None,
    portfolio_score_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Replay a sleeve-aware crisis action policy.

    This complements ``evaluate_policy`` by separating equity-beta longs, GLD,
    and short-beta exposure. It is intentionally proxy-based: historical live
    strategy trade streams are not required, so it can run on the same raw
    regime/crisis data used by phase-auto.
    """
    portfolios = sleeve_portfolios or SLEEVE_PORTFOLIOS
    weights = portfolio_score_weights or SLEEVE_SCORE_WEIGHTS
    exposures = build_sleeve_exposure_map(alerts_df, policy)
    sleeve_returns = build_sleeve_proxy_returns(strat_ret_df).reindex(alerts_df.index).fillna(0.0)
    cash = cash_returns.reindex(alerts_df.index).fillna(0.0)

    portfolio_results: dict[str, Any] = {}
    weighted_score = 0.0
    score_weight_sum = 0.0
    rejected_reasons: list[str] = []
    aggregate_exposure = pd.Series(0.0, index=alerts_df.index, dtype=float)
    aggregate_weight = 0.0

    for name, sleeve_weights in portfolios.items():
        base_ret = pd.Series(0.0, index=alerts_df.index, dtype=float)
        overlay_ret = pd.Series(0.0, index=alerts_df.index, dtype=float)
        effective_exposure = pd.Series(0.0, index=alerts_df.index, dtype=float)
        abs_weight_sum = sum(abs(float(w)) for w in sleeve_weights.values())
        if abs_weight_sum <= 0.0:
            continue

        for sleeve, sleeve_weight in sleeve_weights.items():
            sleeve_ret = sleeve_returns.get(sleeve)
            if sleeve_ret is None:
                continue
            exposure = exposures.get(sleeve)
            if exposure is None:
                exposure = pd.Series(1.0, index=alerts_df.index, dtype=float)
            w = float(sleeve_weight)
            base_ret = base_ret + w * sleeve_ret
            overlay_ret = overlay_ret + w * (exposure * sleeve_ret + (1.0 - exposure) * cash)
            effective_exposure = effective_exposure + (abs(w) / abs_weight_sum) * exposure

        turnover = effective_exposure.diff().abs().fillna(0.0)
        base_metrics = _metrics_for_returns(base_ret, turnover * 0.0)
        overlay_metrics = _metrics_for_returns(overlay_ret, turnover)
        period = _period_diagnostics(base_ret, overlay_ret)
        score = economic_score(base_metrics, overlay_metrics, period, effective_exposure)
        if score.rejected:
            rejected_reasons.append(f"{name}: {score.reject_reason}")

        portfolio_results[name] = {
            "base_metrics": base_metrics,
            "overlay_metrics": overlay_metrics,
            "period_diagnostics": period,
            "score": asdict(score),
            "deltas": _metric_deltas(base_metrics, overlay_metrics),
            "avg_exposure": float(effective_exposure.mean()),
            "min_exposure": float(effective_exposure.min()),
        }

        w = float(weights.get(name, 0.0))
        weighted_score += w * score.total
        score_weight_sum += w
        aggregate_exposure = aggregate_exposure + w * effective_exposure
        aggregate_weight += w

    aggregate_score = weighted_score / score_weight_sum if score_weight_sum else 0.0
    if aggregate_weight > 0:
        aggregate_exposure = aggregate_exposure / aggregate_weight
    action_day_share = float((aggregate_exposure < 0.999).mean())
    if action_day_share > 0.45:
        rejected_reasons.append(f"action_day_share {action_day_share:.1%} > 45%")

    return {
        "policy": asdict(policy),
        "score": 0.0 if rejected_reasons else aggregate_score,
        "rejected": bool(rejected_reasons),
        "reject_reason": "; ".join(rejected_reasons),
        "action_day_share": action_day_share,
        "avg_exposure": float(aggregate_exposure.mean()),
        "min_exposure": float(aggregate_exposure.min()),
        "n_exposure_transitions": int(
            (aggregate_exposure.diff().abs().fillna(0.0) > 1e-12).sum()
        ),
        "portfolio_results": portfolio_results,
    }


def optimize_sleeve_action_policy(
    *,
    alerts_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    cash_returns: pd.Series,
) -> dict[str, Any]:
    """Evaluate the fixed sleeve-aware candidate set and select the best."""
    scenarios = evaluate_sleeve_standard_scenarios(
        alerts_df=alerts_df,
        strat_ret_df=strat_ret_df,
        cash_returns=cash_returns,
    )
    current = scenarios["current_symmetric"]
    ranked = sorted(
        scenarios.items(),
        key=lambda item: float(item[1]["score"]),
        reverse=True,
    )
    best_name, best_eval = ranked[0]
    accepted = [] if best_name == "current_symmetric" else [best_name]
    return {
        "baseline_policy": current["policy"],
        "optimized_policy": best_eval["policy"],
        "base_score": float(current["score"]),
        "optimized_score": float(best_eval["score"]),
        "accepted": accepted,
        "rounds": [{
            "round": 1,
            "best_name": best_name,
            "best_score": float(best_eval["score"]),
            "delta": float(best_eval["score"] - current["score"]),
            "kept": best_name != "current_symmetric",
            "rejected_count": sum(1 for _, item in ranked if item["rejected"]),
        }],
        "optimized_evaluation": best_eval,
        "standard_scenarios": scenarios,
        "ranked_scenarios": [
            {
                "name": name,
                "score": float(item["score"]),
                "rejected": bool(item["rejected"]),
                "action_day_share": float(item["action_day_share"]),
                "avg_exposure": float(item["avg_exposure"]),
            }
            for name, item in ranked
        ],
    }


def evaluate_sleeve_standard_scenarios(
    *,
    alerts_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    cash_returns: pd.Series,
) -> dict[str, Any]:
    """Evaluate sleeve-aware crisis action candidates."""
    policies = [
        CrisisSleeveEconomicPolicy(name="current_symmetric"),
        CrisisSleeveEconomicPolicy(
            name="conservative_asym",
            gld_warning_mult=0.90,
            gld_crisis_mult=0.60,
            short_warning_mult=0.95,
            short_crisis_mult=0.75,
        ),
        CrisisSleeveEconomicPolicy(
            name="balanced_asym",
            gld_warning_mult=0.90,
            gld_crisis_mult=0.75,
            short_warning_mult=1.00,
            short_crisis_mult=0.85,
        ),
        CrisisSleeveEconomicPolicy(
            name="preserve_defensive",
            gld_warning_mult=0.95,
            gld_crisis_mult=0.85,
            short_warning_mult=1.00,
            short_crisis_mult=1.00,
        ),
        CrisisSleeveEconomicPolicy(
            name="preserve_gld_short",
            gld_warning_mult=1.00,
            gld_crisis_mult=1.00,
            short_warning_mult=1.00,
            short_crisis_mult=1.00,
        ),
        CrisisSleeveEconomicPolicy(
            name="equity_current_short_preserved",
            short_warning_mult=1.00,
            short_crisis_mult=1.00,
        ),
        CrisisSleeveEconomicPolicy(
            name="equity_current_gld_preserved",
            gld_warning_mult=1.00,
            gld_crisis_mult=1.00,
        ),
        CrisisSleeveEconomicPolicy(
            name="strong_equity_keep_def",
            equity_warning_mult=0.60,
            equity_crisis_mult=0.25,
            gld_warning_mult=0.95,
            gld_crisis_mult=0.85,
            short_warning_mult=1.00,
            short_crisis_mult=1.00,
        ),
    ]
    return {
        p.name: evaluate_sleeve_policy(
            alerts_df=alerts_df,
            strat_ret_df=strat_ret_df,
            cash_returns=cash_returns,
            policy=p,
        )
        for p in policies
    }


def build_sleeve_proxy_returns(strat_ret_df: pd.DataFrame) -> pd.DataFrame:
    """Build proxy sleeve return streams from the regime raw data."""
    returns = strat_ret_df.copy().fillna(0.0)
    idx = returns.index
    spy = returns.get("SPY", pd.Series(0.0, index=idx))
    efa = returns.get("EFA", pd.Series(0.0, index=idx))
    gld = returns.get("GLD", pd.Series(0.0, index=idx))
    qqq_proxy = returns.get("QQQ", spy)
    equity_beta = 0.85 * spy + 0.15 * efa
    return pd.DataFrame(
        {
            "equity_beta": equity_beta,
            "qqq_proxy": qqq_proxy,
            "gld": gld,
            "short_spy": -spy,
            "short_equity": -equity_beta,
        },
        index=idx,
    )


def build_sleeve_exposure_map(
    alerts_df: pd.DataFrame,
    policy: CrisisSleeveEconomicPolicy,
) -> dict[str, pd.Series]:
    """Build per-sleeve exposure multipliers for a sleeve policy."""
    equity_exposure = _build_action_exposure_series(
        alerts_df,
        warning_mult=policy.equity_warning_mult,
        crisis_mult=policy.equity_crisis_mult,
    )
    gld_exposure = _build_action_exposure_series(
        alerts_df,
        warning_mult=policy.gld_warning_mult,
        crisis_mult=policy.gld_crisis_mult,
    )
    short_exposure = _build_action_exposure_series(
        alerts_df,
        warning_mult=policy.short_warning_mult,
        crisis_mult=policy.short_crisis_mult,
    )
    return {
        "equity_beta": equity_exposure,
        "qqq_proxy": equity_exposure,
        "gld": gld_exposure,
        "short_spy": short_exposure,
        "short_equity": short_exposure,
    }


def build_economic_report(result: dict[str, Any]) -> str:
    """Build a compact markdown report for saved artifacts."""
    lines = [
        "# Crisis Action-Layer Economic Optimization",
        "",
        "Thresholds are frozen at the latest optimized crisis detector config.",
        "This run scores portfolio economics: Calmar, Sortino, max drawdown, crisis drawdown, CAGR, recovery, and rebound/turnover drag.",
        "",
        "## Optimized Policy",
        "",
        "```json",
        json.dumps(result["optimized_policy"], indent=2),
        "```",
        "",
        f"Base score: {result['base_score']:.6f}",
        f"Optimized score: {result['optimized_score']:.6f}",
        f"Accepted: {', '.join(result['accepted']) if result['accepted'] else 'none'}",
        "",
        "## Standard Scenarios",
        "",
        "| Scenario | Score | Rejected | Action Days | Avg Exposure | Regime Calmar Delta | Regime CAGR Delta | Regime MaxDD Delta |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["standard_scenarios"].items():
        regime = item["portfolio_results"]["regime_proxy"]
        deltas = regime["deltas"]
        lines.append(
            f"| {name} | {item['score']:.6f} | {item['rejected']} | "
            f"{item['action_day_share']:.1%} | {item['avg_exposure']:.3f} | "
            f"{deltas['calmar']:+.3f} | {deltas['cagr']:+.2%} | "
            f"{deltas['max_drawdown_pct']:+.2%} |"
        )
    lines.extend([
        "",
        "## Greedy Rounds",
        "",
        "| Round | Best | Score | Delta | Kept | Rejected |",
        "|---:|---|---:|---:|---|---:|",
    ])
    for r in result["rounds"]:
        score = "" if r["best_score"] is None else f"{r['best_score']:.6f}"
        delta = "" if r["delta"] is None else f"{r['delta']:+.6f}"
        lines.append(
            f"| {r['round']} | {r['best_name']} | {score} | {delta} | "
            f"{r['kept']} | {r['rejected_count']} |"
        )
    return "\n".join(lines) + "\n"


def build_sleeve_economic_report(result: dict[str, Any]) -> str:
    """Build a markdown report for sleeve-aware action optimization."""
    lines = [
        "# Crisis Sleeve-Aware Economic Optimization",
        "",
        "Thresholds are frozen at the latest optimized crisis detector config.",
        "This replay separates equity-beta longs, GLD, and short-beta exposure. QQQ is proxied by SPY when QQQ returns are unavailable in the raw regime data.",
        "",
        "## Optimized Policy",
        "",
        "```json",
        json.dumps(result["optimized_policy"], indent=2),
        "```",
        "",
        f"Base score: {result['base_score']:.6f}",
        f"Optimized score: {result['optimized_score']:.6f}",
        f"Accepted: {', '.join(result['accepted']) if result['accepted'] else 'none'}",
        "",
        "## Candidate Ranking",
        "",
        "| Scenario | Score | Rejected | Action Days | Avg Exposure | Risk-On Calmar Delta | Risk-On CAGR Delta | Risk-On MaxDD Delta |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    scenarios = result["standard_scenarios"]
    for item in result["ranked_scenarios"]:
        name = item["name"]
        risk_on = scenarios[name]["portfolio_results"]["risk_on_with_hedges"]
        deltas = risk_on["deltas"]
        lines.append(
            f"| {name} | {item['score']:.6f} | {item['rejected']} | "
            f"{item['action_day_share']:.1%} | {item['avg_exposure']:.3f} | "
            f"{deltas['calmar']:+.3f} | {deltas['cagr']:+.2%} | "
            f"{deltas['max_drawdown_pct']:+.2%} |"
        )

    best = result["optimized_evaluation"]
    current = scenarios["current_symmetric"]
    lines.extend([
        "",
        "## Current vs Optimized",
        "",
        "| Portfolio Proxy | Current Score | Optimized Score | Current Calmar Delta | Optimized Calmar Delta | Current CAGR Delta | Optimized CAGR Delta | Current MaxDD Delta | Optimized MaxDD Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, cur_result in current["portfolio_results"].items():
        best_result = best["portfolio_results"][name]
        cur_delta = cur_result["deltas"]
        best_delta = best_result["deltas"]
        lines.append(
            f"| {name} | {cur_result['score']['total']:.6f} | "
            f"{best_result['score']['total']:.6f} | "
            f"{cur_delta['calmar']:+.3f} | {best_delta['calmar']:+.3f} | "
            f"{cur_delta['cagr']:+.2%} | {best_delta['cagr']:+.2%} | "
            f"{cur_delta['max_drawdown_pct']:+.2%} | "
            f"{best_delta['max_drawdown_pct']:+.2%} |"
        )
    return "\n".join(lines) + "\n"


def _policy_candidates(current: CrisisEconomicPolicy) -> list[tuple[str, dict[str, float]]]:
    candidates: list[tuple[str, dict[str, float]]] = []
    for val in (0.65, 0.70, 0.75, 0.80, 0.85):
        if val != current.warning_mult:
            candidates.append((f"warning_{val:.2f}", {"warning_mult": val}))
    for val in (0.30, 0.40, 0.50, 0.60):
        if val != current.crisis_mult:
            candidates.append((f"crisis_{val:.2f}", {"crisis_mult": val}))
    for val in (1.00, 0.98, 0.95, 0.90):
        if val != current.advisory_mult:
            candidates.append((f"advisory_{val:.2f}", {"advisory_mult": val}))
    for val in (1.00, 0.95, 0.90, 0.80, 0.75):
        if val != current.shock_mult:
            candidates.append((f"shock_{val:.2f}", {"shock_mult": val}))
    for val in (1.00, 0.98, 0.95, 0.90):
        if val != current.grind_mult:
            candidates.append((f"grind_{val:.2f}", {"grind_mult": val}))
    for val in (1.00, 0.95, 0.90, 0.85, 0.80, 0.75):
        if val != current.credit_impulse_mult:
            candidates.append((
                f"credit_impulse_{val:.2f}",
                {"credit_impulse_mult": val},
            ))
    for val in (0.65, 0.70, 0.75, 0.80, 0.85):
        if val != current.credit_bridge_warning_mult:
            candidates.append((
                f"credit_bridge_warning_{val:.2f}",
                {"credit_bridge_warning_mult": val},
            ))
    for val in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        if val != current.stress_regime_warning_mult:
            candidates.append((
                f"stress_regime_warning_{val:.2f}",
                {"stress_regime_warning_mult": val},
            ))
    for val in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        if val != current.defensive_regime_warning_mult:
            candidates.append((
                f"defensive_regime_warning_{val:.2f}",
                {"defensive_regime_warning_mult": val},
            ))
    candidates.extend([
        ("gentler_action", {"warning_mult": 0.85, "crisis_mult": 0.60}),
        ("stronger_action", {"warning_mult": 0.65, "crisis_mult": 0.40}),
        ("shock_grind_light", {"shock_mult": 0.90, "grind_mult": 0.95}),
        ("advisory_shock_light", {"advisory_mult": 0.98, "shock_mult": 0.90}),
        ("credit_impulse_light", {"credit_impulse_mult": 0.90}),
    ])
    return candidates


def _build_action_exposure_series(
    alerts_df: pd.DataFrame,
    *,
    warning_mult: float,
    crisis_mult: float,
) -> pd.Series:
    return build_exposure_series(
        alerts_df,
        CrisisEconomicPolicy(
            warning_mult=warning_mult,
            crisis_mult=crisis_mult,
        ),
    )


def _weighted_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    out = pd.Series(0.0, index=returns.index)
    for asset, weight in weights.items():
        if asset in returns.columns:
            out = out + returns[asset].fillna(0.0) * float(weight)
    return out


def _regime_series(alerts_df: pd.DataFrame) -> pd.Series | None:
    for col in ("hmm_regime", "regime", "macro_regime"):
        if col in alerts_df.columns:
            return alerts_df[col].fillna("").astype(str).str.upper()
    return None


def _metrics_for_returns(daily_returns: pd.Series, turnover: pd.Series) -> dict[str, float]:
    daily_returns = daily_returns.fillna(0.0)
    equity = INITIAL_EQUITY * (1.0 + daily_returns).cumprod()
    metrics = compute_metrics(
        equity,
        daily_returns,
        turnover.reindex(daily_returns.index).fillna(0.0),
        int((turnover.abs() > 1e-12).sum()),
    )
    return asdict(metrics)


def _period_diagnostics(base_ret: pd.Series, overlay_ret: pd.Series) -> dict[str, float]:
    crisis_dd_reductions: list[float] = []
    missed_rebounds: list[float] = []
    for _, (start, end, period_type) in CRISIS_PERIODS.items():
        if period_type == "C":
            continue
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        base_period = base_ret.loc[(base_ret.index >= start_ts) & (base_ret.index <= end_ts)]
        overlay_period = overlay_ret.loc[(overlay_ret.index >= start_ts) & (overlay_ret.index <= end_ts)]
        if not base_period.empty and not overlay_period.empty:
            crisis_dd_reductions.append(
                _max_drawdown(base_period) - _max_drawdown(overlay_period)
            )

        rebound_start = end_ts + pd.Timedelta(days=1)
        rebound_end = end_ts + pd.Timedelta(days=60)
        base_rebound = base_ret.loc[(base_ret.index >= rebound_start) & (base_ret.index <= rebound_end)]
        overlay_rebound = overlay_ret.loc[(overlay_ret.index >= rebound_start) & (overlay_ret.index <= rebound_end)]
        if not base_rebound.empty and not overlay_rebound.empty:
            base_cum = float((1.0 + base_rebound).prod() - 1.0)
            overlay_cum = float((1.0 + overlay_rebound).prod() - 1.0)
            missed_rebounds.append(max(0.0, base_cum - overlay_cum))

    return {
        "avg_crisis_dd_reduction": float(np.mean(crisis_dd_reductions)) if crisis_dd_reductions else 0.0,
        "worst_crisis_dd_reduction": float(np.min(crisis_dd_reductions)) if crisis_dd_reductions else 0.0,
        "avg_missed_rebound_drag": float(np.mean(missed_rebounds)) if missed_rebounds else 0.0,
        "max_missed_rebound_drag": float(np.max(missed_rebounds)) if missed_rebounds else 0.0,
    }


def _metric_deltas(base: dict[str, float], overlay: dict[str, float]) -> dict[str, float]:
    keys = ["total_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown_pct", "max_drawdown_duration"]
    return {key: float(overlay[key] - base[key]) for key in keys}


def _max_drawdown(daily_returns: pd.Series) -> float:
    equity = (1.0 + daily_returns.fillna(0.0)).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(-drawdown.min()) if len(drawdown) else 0.0


def _centered_component(delta: float, full_scale: float) -> float:
    if full_scale <= 0:
        return 0.5
    return _clip01(0.5 + 0.5 * (float(delta) / full_scale))


def _clip01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)
