"""Historical validation of the crisis detection system.

Pure-function interface: run_crisis_detector(market_df, strat_ret_df) returns
a daily DataFrame of alert levels. No side effects, no persistence.

Labeled periods (9 total):
  7 crises  (type "D" sharp / "S" slow) -- detected at WARNING+ (level >= 2)
  2 corrections (type "C")              -- detected at WATCH+   (level >= 1)

Acceptance criteria:
  - All 7 crises detected (WARNING+)
  - GFC, COVID: detection <= 3 days
  - FP rate at WARNING < 5%
  - FP rate at CRISIS < 2%
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from regime.crisis import config as C
from regime.crisis.actions import stress_formation_risk_multiplier
from regime.crisis.context import CrisisContext, _risk_mult_for_level, _dd_mult_for_level
from regime.crisis.detector import (
    compute_alert_level,
    compute_advisory_level,
    is_hard_credit_impulse_warning_candidate,
)
from regime.crisis.hysteresis import HysteresisTracker
from regime.crisis.indicators import compute_indicators

logger = logging.getLogger(__name__)


# Labeled periods: 7 crises (D/S) + 2 corrections (C).
# Type "D" = sharp drawdown, "S" = slow grind, "C" = correction (WATCH-level).
CRISIS_PERIODS = {
    "GFC": ("2008-09-01", "2009-03-01", "D"),
    "Euro Crisis": ("2011-07-01", "2011-12-31", "D"),
    "2015 China": ("2015-08-01", "2016-03-01", "D"),
    "2018 Q4": ("2018-10-01", "2019-01-01", "D"),
    "COVID": ("2020-02-15", "2020-04-15", "D"),
    "2022 Inflation": ("2022-01-01", "2022-10-01", "S"),
    "2023 SVB": ("2023-03-01", "2023-05-31", "C"),
    "2025 Tariff": ("2025-03-01", "2025-05-31", "D"),
    "2026 Iran": ("2026-02-01", "2026-03-27", "C"),
}


def run_crisis_detector(
    market_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    vix3m_series: pd.Series | None = None,
    use_hysteresis: bool = True,
) -> pd.DataFrame:
    """Run crisis detector over full history, return daily DataFrame.

    Args:
        market_df: Must have VIX, SPREAD, SLOPE_10Y2Y columns (from regime data).
        strat_ret_df: Must have SPY, TLT columns (daily returns).
        vix3m_series: Optional VIX3M for term structure (confirming indicator).
        use_hysteresis: Whether to apply sticky de-escalation.

    Returns:
        DataFrame indexed by date with columns:
            alert_level, alert_level_int, risk_multiplier, dd_tier_multiplier,
            vix, credit_spread_bps, yield_curve_slope, spy_tlt_corr,
            spy_10d_return, primary_watch_count, primary_warning_count,
            primary_crisis_count, dominant_channel
    """
    # Determine date range from intersection of both DataFrames
    common_dates = market_df.index.intersection(strat_ret_df.index)
    if common_dates.empty:
        return pd.DataFrame()

    # Need at least 90 days of history for yield curve lookback
    start_idx = max(C.SLOPE_INVERSION_LOOKBACK, C.CORR_WINDOW, 20)
    if len(common_dates) <= start_idx:
        return pd.DataFrame()

    tracker = HysteresisTracker() if use_hysteresis else None
    rows: list[dict] = []

    for date in common_dates[start_idx:]:
        indicators = compute_indicators(
            market_df, strat_ret_df, date=date, vix3m_series=vix3m_series,
        )

        raw_level_str, raw_level_int = compute_alert_level(indicators)
        bridge_candidate = is_hard_credit_impulse_warning_candidate(indicators)

        if tracker is not None:
            raw_level_int = tracker.apply_hard_credit_impulse_bridge(
                raw_level_int,
                bridge_candidate,
            )
            raw_level_str = C.ALERT_LEVELS[raw_level_int]
            final_level_int = tracker.update(raw_level_int)
        else:
            final_level_int = raw_level_int

        final_level_str = C.ALERT_LEVELS[final_level_int]
        advisory_level_str, advisory_level_int, advisory_reason = compute_advisory_level(
            indicators,
            final_level_int,
        )
        formation_risk_mult = stress_formation_risk_multiplier(
            indicators.stress_formation_mode,
            indicators.stress_formation_score,
        )
        portfolio_action_level_int = (
            final_level_int if final_level_int >= 2
            else 1 if formation_risk_mult < 1.0
            else 0
        )
        portfolio_action_level_str = C.ALERT_LEVELS[portfolio_action_level_int]

        rows.append({
            "date": date,
            "alert_level": final_level_str,
            "alert_level_int": final_level_int,
            "raw_level_int": raw_level_int,
            "advisory_level": advisory_level_str,
            "advisory_level_int": advisory_level_int,
            "advisory_reason": advisory_reason,
            "portfolio_action_level": portfolio_action_level_str,
            "portfolio_action_level_int": portfolio_action_level_int,
            "risk_multiplier": (
                _risk_mult_for_level(final_level_int)
                if final_level_int >= 2 else formation_risk_mult
            ),
            "dd_tier_multiplier": _dd_mult_for_level(final_level_int),
            "vix": indicators.vix.value,
            "credit_spread_bps": indicators.credit_spread.value,
            "yield_curve_slope": indicators.yield_curve.value,
            "spy_tlt_corr": indicators.spy_tlt_corr.value,
            "spy_3d_return": indicators.spy_3d_return,
            "spy_5d_return": indicators.spy_5d_return,
            "spy_10d_return": indicators.spy_10d_return,
            "spy_20d_return": indicators.spy_20d_return,
            "vix_3d_change": indicators.vix_3d_change,
            "credit_spread_20d_change_bps": indicators.credit_spread_20d_change_bps,
            "stress_formation_score": indicators.stress_formation_score,
            "stress_formation_mode": indicators.stress_formation_mode,
            "stress_formation_reason": indicators.stress_formation_reason,
            "hard_credit_impulse_warning_candidate": bridge_candidate,
            "hard_credit_impulse_warning_days": (
                tracker.hard_credit_impulse_warning_days
                if tracker is not None else int(bridge_candidate)
            ),
            "vix_level_int": indicators.vix.level,
            "credit_spread_level_int": indicators.credit_spread.level,
            "yield_curve_level_int": indicators.yield_curve.level,
            "spy_tlt_corr_level_int": indicators.spy_tlt_corr.level,
            "spy_drawdown_level_int": (
                indicators.spy_drawdown.level if indicators.spy_drawdown is not None else 0
            ),
            "primary_watch_count": indicators.watch_count,
            "primary_warning_count": indicators.warning_count,
            "primary_crisis_count": indicators.crisis_count,
            "dominant_channel": indicators.dominant_channel,
            "spy_dd_level": indicators.spy_drawdown.level if indicators.spy_drawdown is not None else 0,
        })

    result = pd.DataFrame(rows).set_index("date")
    return result


def validate_detection_latency(
    alerts_df: pd.DataFrame,
    crisis_periods: dict[str, tuple[str, str, str]] | None = None,
) -> dict[str, dict]:
    """Measure detection latency for labeled crisis periods.

    Returns dict of {crisis_name: {start, detected_at, latency_days, detected}}.
    """
    if crisis_periods is None:
        crisis_periods = CRISIS_PERIODS

    results = {}
    for name, (start, end, expected_type) in crisis_periods.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        # Only analyze if we have data for this period
        period_alerts = alerts_df.loc[
            (alerts_df.index >= start_ts) & (alerts_df.index <= end_ts)
        ]
        if period_alerts.empty:
            results[name] = {
                "start": start, "end": end,
                "detected": False, "latency_days": None,
                "reason": "no data in period",
            }
            continue

        # Corrections (type "C") detected at WATCH+ (level >= 1);
        # crises (type "D"/"S") detected at WARNING+ (level >= 2).
        detect_threshold = 1 if expected_type == "C" else 2
        threshold_label = "WATCH" if expected_type == "C" else "WARNING"
        qualifying_days = period_alerts[
            period_alerts["alert_level_int"] >= detect_threshold
        ]
        if qualifying_days.empty:
            results[name] = {
                "start": start, "end": end,
                "detected": False, "latency_days": None,
                "max_level": int(period_alerts["alert_level_int"].max()),
                "reason": f"never reached {threshold_label}",
                "period_type": expected_type,
            }
            continue

        detected_at = qualifying_days.index[0]
        latency = (detected_at - start_ts).days

        results[name] = {
            "start": start,
            "end": end,
            "detected": True,
            "detected_at": str(detected_at.date()),
            "latency_days": latency,
            "max_level": int(period_alerts["alert_level_int"].max()),
            "peak_level": C.ALERT_LEVELS[int(period_alerts["alert_level_int"].max())],
            "period_type": expected_type,
        }

    return results


_CHANNEL_LEVEL_COLUMNS = {
    "VIX": "vix_level_int",
    "CREDIT_SPREAD": "credit_spread_level_int",
    "YIELD_CURVE": "yield_curve_level_int",
    "SPY_TLT_CORR": "spy_tlt_corr_level_int",
    "SPY_DRAWDOWN": "spy_drawdown_level_int",
}


def build_event_channel_chronology(
    alerts_df: pd.DataFrame,
    crisis_periods: dict[str, tuple[str, str, str]] | None = None,
) -> dict[str, dict]:
    """Explain event-level detection timing by channel.

    For each labeled event, returns first dates for channel WATCH/WARNING/CRISIS,
    first raw and final conjunction dates, and the channel that most likely
    acted as the confirmation bottleneck at detection time.
    """
    if crisis_periods is None:
        crisis_periods = CRISIS_PERIODS

    chronology: dict[str, dict] = {}
    for name, (start, end, expected_type) in crisis_periods.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        period_alerts = alerts_df.loc[
            (alerts_df.index >= start_ts) & (alerts_df.index <= end_ts)
        ]
        detect_threshold = 1 if expected_type == "C" else 2
        threshold_label = "WATCH" if detect_threshold == 1 else "WARNING"

        if period_alerts.empty:
            chronology[name] = {
                "start": start,
                "end": end,
                "period_type": expected_type,
                "detected": False,
                "reason": "no data in period",
            }
            continue

        first_final = _first_index_at_or_above(
            period_alerts,
            "alert_level_int",
            detect_threshold,
        )
        first_raw = _first_index_at_or_above(
            period_alerts,
            "raw_level_int",
            detect_threshold,
        )
        first_advisory = _first_index_at_or_above(
            period_alerts,
            "advisory_level_int",
            1,
        )
        first_action = _first_index_at_or_above(
            period_alerts,
            "portfolio_action_level_int",
            1,
        )
        first_stress = _first_index_at_or_above(
            period_alerts,
            "stress_formation_score",
            C.STRESS_FORMATION_MIN_SCORE,
        )
        bridge_persist_days = getattr(
            C,
            "HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS",
            0,
        )
        first_hard_bridge = (
            _first_index_at_or_above(
                period_alerts,
                "hard_credit_impulse_warning_days",
                bridge_persist_days,
            )
            if bridge_persist_days > 0 else None
        )

        channel_firsts: dict[str, dict[str, str | None]] = {}
        active_confirmation: list[tuple[str, pd.Timestamp]] = []
        for channel, col in _CHANNEL_LEVEL_COLUMNS.items():
            if col not in period_alerts.columns:
                continue
            watch_at = _first_index_at_or_above(period_alerts, col, 1)
            warning_at = _first_index_at_or_above(period_alerts, col, 2)
            crisis_at = _first_index_at_or_above(period_alerts, col, 3)
            channel_firsts[channel] = {
                "WATCH": _date_str(watch_at),
                "WARNING": _date_str(warning_at),
                "CRISIS": _date_str(crisis_at),
            }
            confirm_at = watch_at if detect_threshold == 1 else warning_at
            if first_final is not None and confirm_at is not None and confirm_at <= first_final:
                active_confirmation.append((channel, confirm_at))

        if (
            detect_threshold >= 2
            and first_final is not None
            and first_hard_bridge is not None
            and first_hard_bridge <= first_final
        ):
            active_confirmation.append(
                ("HARD_CREDIT_IMPULSE_BRIDGE", first_hard_bridge),
            )

        active_confirmation.sort(key=lambda item: item[1])
        bottleneck_channel = ""
        bottleneck_date: pd.Timestamp | None = None
        if active_confirmation:
            bottleneck_channel, bottleneck_date = active_confirmation[-1]

        chronology[name] = {
            "start": start,
            "end": end,
            "period_type": expected_type,
            "detect_threshold": threshold_label,
            "detected": first_final is not None,
            "detected_at": _date_str(first_final),
            "latency_days": (
                int((first_final - start_ts).days) if first_final is not None else None
            ),
            "first_raw_conjunction_at": _date_str(first_raw),
            "raw_latency_days": (
                int((first_raw - start_ts).days) if first_raw is not None else None
            ),
            "first_external_advisory_at": _date_str(first_advisory),
            "external_advisory_latency_days": (
                int((first_advisory - start_ts).days) if first_advisory is not None else None
            ),
            "first_portfolio_action_at": _date_str(first_action),
            "portfolio_action_latency_days": (
                int((first_action - start_ts).days) if first_action is not None else None
            ),
            "first_stress_formation_at": _date_str(first_stress),
            "stress_formation_latency_days": (
                int((first_stress - start_ts).days) if first_stress is not None else None
            ),
            "first_hard_credit_impulse_bridge_at": _date_str(first_hard_bridge),
            "hard_credit_impulse_bridge_latency_days": (
                int((first_hard_bridge - start_ts).days)
                if first_hard_bridge is not None else None
            ),
            "max_level": int(period_alerts["alert_level_int"].max()),
            "peak_level": C.ALERT_LEVELS[int(period_alerts["alert_level_int"].max())],
            "bottleneck_channel": bottleneck_channel,
            "bottleneck_date": _date_str(bottleneck_date),
            "channels_confirming_by_detection": [
                {"channel": ch, "first_confirmed_at": _date_str(ts)}
                for ch, ts in active_confirmation
            ],
            "channel_firsts": channel_firsts,
        }

    return chronology


def _first_index_at_or_above(
    df: pd.DataFrame,
    column: str,
    threshold: int,
) -> pd.Timestamp | None:
    if column not in df.columns:
        return None
    qualifying = df[df[column] >= threshold]
    if qualifying.empty:
        return None
    return pd.Timestamp(qualifying.index[0])


def _date_str(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return str(pd.Timestamp(value).date())


def compute_false_positive_rates(
    alerts_df: pd.DataFrame,
    crisis_periods: dict[str, tuple[str, str, str]] | None = None,
) -> dict[str, float]:
    """Compute FP rates for each alert level.

    A day is a false positive if it's at WARNING/CRISIS but NOT within
    a labeled crisis period.

    Returns dict with keys: watch_rate, warning_rate, crisis_rate.
    """
    if crisis_periods is None:
        crisis_periods = CRISIS_PERIODS

    # Build mask of labeled crisis days
    crisis_mask = pd.Series(False, index=alerts_df.index)
    for name, (start, end, _) in crisis_periods.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        crisis_mask |= (alerts_df.index >= start_ts) & (alerts_df.index <= end_ts)

    non_crisis_days = alerts_df[~crisis_mask]
    total_non_crisis = len(non_crisis_days)

    if total_non_crisis == 0:
        return {"watch_rate": 0.0, "warning_rate": 0.0, "crisis_rate": 0.0}

    watch_fp = (non_crisis_days["alert_level_int"] >= 1).sum()
    warning_fp = (non_crisis_days["alert_level_int"] >= 2).sum()
    crisis_fp = (non_crisis_days["alert_level_int"] >= 3).sum()

    return {
        "watch_rate": float(watch_fp / total_non_crisis),
        "warning_rate": float(warning_fp / total_non_crisis),
        "crisis_rate": float(crisis_fp / total_non_crisis),
        "total_non_crisis_days": total_non_crisis,
        "watch_fp_days": int(watch_fp),
        "warning_fp_days": int(warning_fp),
        "crisis_fp_days": int(crisis_fp),
    }


def run_full_validation(
    market_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    vix3m_series: pd.Series | None = None,
) -> dict:
    """Run complete validation suite and return summary.

    Checks:
      1. Detection latency for labeled crises
      2. False positive rates
      3. Alert level distribution
    """
    alerts = run_crisis_detector(market_df, strat_ret_df, vix3m_series)

    if alerts.empty:
        return {"error": "No data available for validation"}

    latency = validate_detection_latency(alerts)
    fp_rates = compute_false_positive_rates(alerts)
    chronology = build_event_channel_chronology(alerts)

    # Alert level distribution
    level_counts = alerts["alert_level"].value_counts().to_dict()
    advisory_counts = alerts["advisory_level"].value_counts().to_dict()
    total_days = len(alerts)
    level_pcts = {k: f"{v / total_days:.1%}" for k, v in level_counts.items()}
    advisory_pcts = {k: f"{v / total_days:.1%}" for k, v in advisory_counts.items()}

    # Acceptance criteria checks
    criteria = {
        "warning_fp_rate_under_5pct": fp_rates["warning_rate"] < 0.05,
        "crisis_fp_rate_under_2pct": fp_rates["crisis_rate"] < 0.02,
    }

    # Count crises (D/S) and corrections (C) separately
    crisis_periods = CRISIS_PERIODS
    crises_detected = sum(
        1 for name, result in latency.items()
        if result.get("detected") and crisis_periods.get(name, ("", "", ""))[2] != "C"
    )
    total_crises = sum(1 for v in crisis_periods.values() if v[2] != "C")
    criteria["all_crises_detected"] = crises_detected == total_crises

    for name, result in latency.items():
        if result.get("detected"):
            if name in ("GFC", "COVID"):
                criteria[f"{name}_detection_under_3d"] = result["latency_days"] <= 3

    return {
        "date_range": f"{alerts.index[0].date()} to {alerts.index[-1].date()}",
        "total_days": total_days,
        "level_distribution": level_counts,
        "level_percentages": level_pcts,
        "advisory_level_distribution": advisory_counts,
        "advisory_level_percentages": advisory_pcts,
        "detection_latency": latency,
        "event_channel_chronology": chronology,
        "false_positive_rates": fp_rates,
        "acceptance_criteria": criteria,
        "all_criteria_pass": all(criteria.values()),
    }
