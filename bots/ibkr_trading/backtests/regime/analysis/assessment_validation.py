"""Helpers for assessment Steps 6 and 7 validation flows."""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd


def compute_spy_allocation_range_bp(signals: pd.DataFrame) -> float:
    """Compute the SPY allocation spread across dominant-regime buckets in bps."""
    alloc_col = "w_SPY" if "w_SPY" in signals.columns else "pi_SPY"
    if alloc_col not in signals.columns or not {"P_G", "P_R", "P_S", "P_D"}.issubset(signals.columns):
        return 0.0

    dominant = signals[["P_G", "P_R", "P_S", "P_D"]].idxmax(axis=1).str.removeprefix("P_")
    grouped = signals[[alloc_col]].groupby(dominant).mean()
    if len(grouped) < 2:
        return 0.0
    return float((grouped[alloc_col].max() - grouped[alloc_col].min()) * 10_000.0)


def compute_period_dominant_regime(
    signals: pd.DataFrame,
    start: str,
    end: str,
) -> str:
    """Return the modal dominant regime over the requested window."""
    if not {"P_G", "P_R", "P_S", "P_D"}.issubset(signals.columns):
        return "?"

    period = signals.loc[start:end]
    if period.empty:
        return "?"
    dominant = period[["P_G", "P_R", "P_S", "P_D"]].idxmax(axis=1).str.removeprefix("P_")
    return str(dominant.mode().iloc[0]) if not dominant.mode().empty else str(dominant.iloc[-1])


def summarize_calibration_candidate(signals: pd.DataFrame, result) -> dict:
    """Summarize the Step 6 acceptance metrics for one candidate run."""
    probs = signals[["P_G", "P_R", "P_S", "P_D"]]
    avg_p_dom = float(probs.max(axis=1).mean()) if not probs.empty else 0.0
    spy_range_bp = compute_spy_allocation_range_bp(signals)
    regime_2022 = compute_period_dominant_regime(signals, "2022-01-01", "2022-10-31")

    return {
        "avg_p_dom": avg_p_dom,
        "spy_allocation_range_bp": spy_range_bp,
        "sharpe": float(result.metrics.sharpe),
        "max_drawdown_pct": float(result.metrics.max_drawdown_pct),
        "regime_2022": regime_2022,
    }


def calibration_candidate_passes(summary: dict) -> bool:
    """Check the Step 6 acceptance thresholds."""
    return (
        0.70 <= summary["avg_p_dom"] <= 0.90
        and summary["spy_allocation_range_bp"] >= 200.0
        and summary["sharpe"] >= 1.40
        and summary["max_drawdown_pct"] <= 0.10
        and summary["regime_2022"] in {"S", "D"}
    )


def find_first_risk_off_alert(
    signals: pd.DataFrame,
    start: str,
    end: str,
    threshold: float = 0.5,
) -> Optional[str]:
    """Find the first risk-off scanner alert date inside a window."""
    if "shift_prob" not in signals.columns or "shift_dir" not in signals.columns:
        return None

    alerts = signals.loc[start:end]
    active_mask = (alerts["shift_prob"] > threshold) & (alerts["shift_dir"] == "risk_off")
    for dt, active in active_mask.items():
        if active:
            return str(dt.date())
    return None


def compute_p_crisis_snapshot(signals: pd.DataFrame, date: str) -> float:
    """Get the most recent p_crisis value at or before a date."""
    if "p_crisis" not in signals.columns:
        return math.nan

    upto = signals.loc[:date, "p_crisis"]
    if upto.empty:
        return math.nan
    return float(upto.iloc[-1])


def compute_peak_p_crisis(signals: pd.DataFrame, start: str, end: str) -> float:
    """Get the maximum p_crisis over a slice."""
    if "p_crisis" not in signals.columns:
        return math.nan

    period = signals.loc[start:end, "p_crisis"]
    if period.empty:
        return math.nan
    return float(period.max())


def compute_slice_return_and_max_dd(result, start: str, end: str) -> tuple[float, float]:
    """Compute return and within-slice max drawdown from a portfolio result."""
    eq = result.equity_curve
    window = eq.loc[start:end]
    if window.empty:
        return 0.0, 0.0

    start_ts = pd.Timestamp(start)
    prior = eq.loc[:start_ts]
    base = float(prior.iloc[-1]) if not prior.empty else float(window.iloc[0])
    total_return = float(window.iloc[-1] / base - 1.0)

    extended = pd.concat(
        [
            pd.Series([base], index=[window.index[0] - pd.Timedelta(days=1)]),
            window,
        ]
    )
    drawdown = extended / extended.cummax() - 1.0
    max_dd = float(-drawdown.min())
    return total_return, max_dd


def summarize_2022_validation(
    signals: pd.DataFrame,
    result,
    scanner_threshold: float = 0.5,
) -> dict:
    """Build the Step 7 summary metrics for a single scenario."""
    ret_2022, max_dd_2022 = compute_slice_return_and_max_dd(
        result,
        "2022-01-01",
        "2022-12-31",
    )

    return {
        "first_jan_alert": find_first_risk_off_alert(
            signals,
            "2022-01-01",
            "2022-01-31",
            threshold=scanner_threshold,
        ),
        "p_crisis_feb25": compute_p_crisis_snapshot(signals, "2022-02-25"),
        "p_crisis_peak_feb": compute_peak_p_crisis(signals, "2022-01-01", "2022-02-28"),
        "return_2022": ret_2022,
        "max_dd_2022": max_dd_2022,
        "max_dd_full_window": float(result.metrics.max_drawdown_pct),
    }


def validate_step7_outcome(
    full_stack: dict,
    scanner_off: dict,
    crisis_off: dict,
    r3_reference: dict,
) -> dict:
    """Apply the Step 7 pass/fail rules to the four scenario summaries."""
    alert = full_stack["first_jan_alert"]
    alert_ok = alert is not None and "2022-01-07" <= alert <= "2022-01-21"
    crisis_ok = full_stack["p_crisis_feb25"] >= 0.5
    dd_2022_ok = (
        full_stack["max_dd_2022"] < scanner_off["max_dd_2022"]
        and full_stack["max_dd_2022"] < crisis_off["max_dd_2022"]
    )
    dd_full_window_ok = (
        full_stack["max_dd_full_window"] < scanner_off["max_dd_full_window"]
        and full_stack["max_dd_full_window"] < crisis_off["max_dd_full_window"]
    )
    dd_ok = dd_2022_ok and dd_full_window_ok
    return_ok = full_stack["return_2022"] > r3_reference["return_2022"]

    return {
        "alert_ok": alert_ok,
        "crisis_ok": crisis_ok,
        "dd_2022_ok": dd_2022_ok,
        "dd_full_window_ok": dd_full_window_ok,
        "dd_ok": dd_ok,
        "return_ok": return_ok,
        "passed": alert_ok and crisis_ok and dd_ok and return_ok,
    }
