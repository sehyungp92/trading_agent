"""Crisis detection composite scoring -- early/action-aware components.

Components (BASE_WEIGHTS):
  detection_speed     : fast hard WARNING/CRISIS detection of labeled crises
  early_action_speed  : fast external advisory / portfolio action lead time
  fp_control          : low hard false positive rates
  coverage            : detect all 7 labeled crisis periods (D/S types)
  severity            : reach appropriate alert levels during crises
  stability           : avoid excessive alert level transitions
  calibration         : correct proportion of time at elevated levels
  recovery_speed      : fast de-escalation after crises end (R3)
  preaction_quality   : useful but not saturated WATCH pre-action layer

Corrections (type "C") are tracked but do not affect the composite score.
They use WATCH+ threshold (not WARNING+) and are reported separately.

Hard rejects (configurable per-phase): max_warning_fp_rate, max_crisis_fp_rate,
min_crises_detected, max_avg_latency, max_avg_recovery_days,
max_advisory_fp_rate, max_preaction_fp_rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtests.regime.crisis_validation import (
    CRISIS_PERIODS,
    validate_detection_latency,
    compute_false_positive_rates,
)


@dataclass
class CrisisMetrics:
    """Crisis detection performance metrics for scoring and diagnostics."""

    # Detection performance (crises = D/S types only)
    avg_latency: float = 0.0
    max_latency: float = 0.0
    avg_action_latency: float = 0.0
    max_action_latency: float = 0.0
    avg_advisory_latency: float = 0.0
    max_advisory_latency: float = 0.0
    gfc_latency: float = -1.0       # -1 = not detected
    covid_latency: float = -1.0     # -1 = not detected
    crises_detected: int = 0
    total_crises: int = 7           # D/S periods in CRISIS_PERIODS

    # Correction detection (C type -- WATCH+ threshold)
    corrections_detected: int = 0
    total_corrections: int = 2      # C periods in CRISIS_PERIODS
    correction_avg_latency: float = 0.0

    # False positive rates
    warning_fp_rate: float = 0.0
    crisis_fp_rate: float = 0.0
    advisory_fp_rate: float = 0.0
    preaction_fp_rate: float = 0.0

    # Severity
    avg_peak_level: float = 0.0

    # Stability
    total_transitions: int = 0
    transitions_per_year: float = 0.0

    # Alert level distribution
    pct_normal: float = 0.0
    pct_watch: float = 0.0
    pct_warning: float = 0.0
    pct_crisis: float = 0.0
    pct_advisory_watch: float = 0.0
    pct_preaction_watch: float = 0.0

    # Recovery speed (R3)
    avg_recovery_days: float = 0.0   # avg days at WARNING+ in 60d post-crisis window
    max_recovery_days: float = 0.0   # worst single crisis recovery


@dataclass(frozen=True)
class CrisisCompositeScore:
    """Frozen composite score for crisis detection and early action."""

    detection_speed: float = 0.0
    early_action_speed: float = 0.0
    fp_control: float = 0.0
    coverage: float = 0.0
    severity: float = 0.0
    stability: float = 0.0
    calibration: float = 0.0
    recovery_speed: float = 0.0
    preaction_quality: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


BASE_WEIGHTS: dict[str, float] = {
    "detection_speed": 0.15,
    "early_action_speed": 0.15,
    "fp_control": 0.16,
    "coverage": 0.14,
    "severity": 0.11,
    "stability": 0.07,
    "calibration": 0.07,
    "recovery_speed": 0.08,
    "preaction_quality": 0.07,
}


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def extract_crisis_metrics(alerts_df: pd.DataFrame) -> CrisisMetrics:
    """Extract CrisisMetrics from a crisis detector output DataFrame."""
    if alerts_df.empty:
        return CrisisMetrics()

    advisory_levels = alerts_df.get(
        "advisory_level_int",
        alerts_df["alert_level_int"],
    )
    action_levels = alerts_df.get(
        "portfolio_action_level_int",
        alerts_df["alert_level_int"],
    )

    # Detection latency -- separate crises (D/S) from corrections (C)
    latency_results = validate_detection_latency(alerts_df)
    crisis_latencies: list[float] = []
    action_latencies: list[float] = []
    advisory_latencies: list[float] = []
    correction_latencies: list[float] = []
    peak_levels: list[float] = []
    gfc_latency = -1.0
    covid_latency = -1.0

    total_crises = sum(1 for v in CRISIS_PERIODS.values() if v[2] != "C")
    total_corrections = sum(1 for v in CRISIS_PERIODS.values() if v[2] == "C")

    for name, result in latency_results.items():
        period_type = CRISIS_PERIODS.get(name, ("", "", "D"))[2]
        if result.get("detected"):
            lat = float(result["latency_days"])
            if period_type == "C":
                correction_latencies.append(lat)
            else:
                crisis_latencies.append(lat)
                peak_levels.append(float(result.get("max_level", 0)))
            if name == "GFC":
                gfc_latency = lat
            elif name == "COVID":
                covid_latency = lat

    crises_detected = len(crisis_latencies)
    corrections_detected = len(correction_latencies)
    avg_latency = (
        sum(crisis_latencies) / len(crisis_latencies)
        if crisis_latencies
        else 99.0
    )
    max_latency = max(crisis_latencies) if crisis_latencies else 99.0
    correction_avg_latency = (
        sum(correction_latencies) / len(correction_latencies)
        if correction_latencies
        else 99.0
    )
    avg_peak_level = (
        sum(peak_levels) / len(peak_levels) if peak_levels else 0.0
    )

    for name, (s, e, period_type) in CRISIS_PERIODS.items():
        if period_type == "C":
            continue
        start_ts = pd.Timestamp(s)
        end_ts = pd.Timestamp(e)
        period = alerts_df.loc[
            (alerts_df.index >= start_ts) & (alerts_df.index <= end_ts)
        ]
        if period.empty:
            continue

        period_action = action_levels.loc[period.index]
        action_days = period_action[period_action >= 1]
        if not action_days.empty:
            action_latencies.append(float((action_days.index[0] - start_ts).days))

        period_advisory = advisory_levels.loc[period.index]
        advisory_days = period_advisory[period_advisory >= 1]
        if not advisory_days.empty:
            advisory_latencies.append(float((advisory_days.index[0] - start_ts).days))

    avg_action_latency = (
        sum(action_latencies) / len(action_latencies)
        if action_latencies else 99.0
    )
    max_action_latency = max(action_latencies) if action_latencies else 99.0
    avg_advisory_latency = (
        sum(advisory_latencies) / len(advisory_latencies)
        if advisory_latencies else 99.0
    )
    max_advisory_latency = max(advisory_latencies) if advisory_latencies else 99.0

    # False positive rates
    fp_rates = compute_false_positive_rates(alerts_df)
    warning_fp_rate = fp_rates["warning_rate"]
    crisis_fp_rate = fp_rates["crisis_rate"]

    crisis_mask = pd.Series(False, index=alerts_df.index)
    for _, (start, end, _) in CRISIS_PERIODS.items():
        crisis_mask |= (
            (alerts_df.index >= pd.Timestamp(start))
            & (alerts_df.index <= pd.Timestamp(end))
        )
    non_crisis = ~crisis_mask
    if int(non_crisis.sum()) > 0:
        advisory_fp_rate = float((advisory_levels.loc[non_crisis] >= 1).mean())
        preaction_fp_rate = float((action_levels.loc[non_crisis] >= 1).mean())
    else:
        advisory_fp_rate = 0.0
        preaction_fp_rate = 0.0

    # Transitions: count changes in alert_level_int
    levels = alerts_df["alert_level_int"].values
    transitions = sum(
        1 for i in range(1, len(levels)) if levels[i] != levels[i - 1]
    )
    total_days = len(alerts_df)
    years = total_days / 252.0  # trading days per year
    transitions_per_year = transitions / years if years > 0 else 0.0

    # Level distribution
    level_counts = alerts_df["alert_level_int"].value_counts()
    pct_normal = float(level_counts.get(0, 0)) / total_days if total_days else 0.0
    pct_watch = float(level_counts.get(1, 0)) / total_days if total_days else 0.0
    pct_warning = float(level_counts.get(2, 0)) / total_days if total_days else 0.0
    pct_crisis = float(level_counts.get(3, 0)) / total_days if total_days else 0.0
    pct_advisory_watch = (
        float((advisory_levels == 1).sum()) / total_days if total_days else 0.0
    )
    pct_preaction_watch = (
        float((action_levels == 1).sum()) / total_days if total_days else 0.0
    )

    # Recovery speed: days at WARNING+ in 60d post-crisis window
    recovery_days_list: list[int] = []
    for name, (s, e, period_type) in CRISIS_PERIODS.items():
        if period_type == "C":
            continue
        start_ts = pd.Timestamp(s)
        end_ts = pd.Timestamp(e)
        period_alerts = alerts_df.loc[
            (alerts_df.index >= start_ts) & (alerts_df.index <= end_ts)
        ]
        if period_alerts.empty or (period_alerts["alert_level_int"] >= 2).sum() == 0:
            continue
        post_start = end_ts + pd.Timedelta(days=1)
        post_end = end_ts + pd.Timedelta(days=60)
        post_alerts = alerts_df.loc[
            (alerts_df.index >= post_start) & (alerts_df.index <= post_end)
        ]
        if not post_alerts.empty:
            recovery_days_list.append(int((post_alerts["alert_level_int"] >= 2).sum()))

    avg_recovery_days = (
        sum(recovery_days_list) / len(recovery_days_list)
        if recovery_days_list else 0.0
    )
    max_recovery_days = float(max(recovery_days_list)) if recovery_days_list else 0.0

    return CrisisMetrics(
        avg_latency=avg_latency,
        max_latency=max_latency,
        avg_action_latency=avg_action_latency,
        max_action_latency=max_action_latency,
        avg_advisory_latency=avg_advisory_latency,
        max_advisory_latency=max_advisory_latency,
        gfc_latency=gfc_latency,
        covid_latency=covid_latency,
        crises_detected=crises_detected,
        total_crises=total_crises,
        corrections_detected=corrections_detected,
        total_corrections=total_corrections,
        correction_avg_latency=correction_avg_latency,
        warning_fp_rate=warning_fp_rate,
        crisis_fp_rate=crisis_fp_rate,
        advisory_fp_rate=advisory_fp_rate,
        preaction_fp_rate=preaction_fp_rate,
        avg_peak_level=avg_peak_level,
        total_transitions=transitions,
        transitions_per_year=transitions_per_year,
        pct_normal=pct_normal,
        pct_watch=pct_watch,
        pct_warning=pct_warning,
        pct_crisis=pct_crisis,
        pct_advisory_watch=pct_advisory_watch,
        pct_preaction_watch=pct_preaction_watch,
        avg_recovery_days=avg_recovery_days,
        max_recovery_days=max_recovery_days,
    )


def composite_score(
    metrics: CrisisMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> CrisisCompositeScore:
    """Compute the immutable 7-component composite score for crisis detection."""
    w = dict(BASE_WEIGHTS)
    if weight_overrides:
        w.update(weight_overrides)

    hr = hard_rejects or {}

    # Hard rejects
    if metrics.warning_fp_rate > hr.get("max_warning_fp_rate", 0.10):
        return CrisisCompositeScore(
            rejected=True,
            reject_reason=f"high_warning_fp ({metrics.warning_fp_rate:.3f})",
        )
    if metrics.crisis_fp_rate > hr.get("max_crisis_fp_rate", 0.05):
        return CrisisCompositeScore(
            rejected=True,
            reject_reason=f"high_crisis_fp ({metrics.crisis_fp_rate:.3f})",
        )
    if metrics.advisory_fp_rate > hr.get("max_advisory_fp_rate", 0.99):
        return CrisisCompositeScore(
            rejected=True,
            reject_reason=f"high_advisory_fp ({metrics.advisory_fp_rate:.3f})",
        )
    if metrics.preaction_fp_rate > hr.get("max_preaction_fp_rate", 0.99):
        return CrisisCompositeScore(
            rejected=True,
            reject_reason=f"high_preaction_fp ({metrics.preaction_fp_rate:.3f})",
        )
    if metrics.crises_detected < hr.get("min_crises_detected", 7):
        return CrisisCompositeScore(
            rejected=True,
            reject_reason=f"low_coverage ({metrics.crises_detected}/{metrics.total_crises})",
        )
    if metrics.avg_latency > hr.get("max_avg_latency", 99.0):
        return CrisisCompositeScore(
            rejected=True,
            reject_reason=f"slow_detection ({metrics.avg_latency:.1f}d)",
        )
    if metrics.avg_recovery_days > hr.get("max_avg_recovery_days", 99.0):
        return CrisisCompositeScore(
            rejected=True,
            reject_reason=f"slow_recovery ({metrics.avg_recovery_days:.1f}d)",
        )

    # --- Components ---

    # Detection speed: 1.0 at 0d latency, 0.0 at 60d+
    # Wide range (60d) so optimizer can distinguish 15d vs 50d in early phases;
    # ultimate target of <=20d scores ~0.67.
    detection_speed_c = _clip01(1.0 - metrics.avg_latency / 60.0)

    # Early action speed: reward advisory/action lead time, but keep the
    # strongest incentive on risk-bearing action rather than noisy visibility.
    action_speed_c = _clip01(1.0 - metrics.avg_action_latency / 45.0)
    advisory_speed_c = _clip01(1.0 - metrics.avg_advisory_latency / 45.0)
    action_lead_c = _clip01((metrics.avg_latency - metrics.avg_action_latency) / 20.0)
    advisory_lead_c = _clip01(
        (metrics.avg_latency - metrics.avg_advisory_latency) / 30.0
    )
    early_action_speed_c = (
        0.45 * action_speed_c
        + 0.25 * advisory_speed_c
        + 0.20 * action_lead_c
        + 0.10 * advisory_lead_c
    )

    # FP control: blend of warning and crisis FP containment.
    # Wide normalization range (0.50/0.25) so the optimizer can distinguish
    # 20% from 45% FP in early phases when conjunction is being relaxed.
    # At ultimate targets (5%/2%), scores are still 0.90/0.92.
    fp_warn_c = _clip01(1.0 - metrics.warning_fp_rate / 0.50)
    fp_crisis_c = _clip01(1.0 - metrics.crisis_fp_rate / 0.25)
    fp_control_c = 0.6 * fp_warn_c + 0.4 * fp_crisis_c

    # Coverage: fraction of labeled crises detected (D/S types only, not corrections)
    coverage_c = _clip01(metrics.crises_detected / max(metrics.total_crises, 1))

    # Severity: average peak alert level during detected crises (max=3 for CRISIS)
    severity_c = _clip01(metrics.avg_peak_level / 3.0)

    # Stability: penalize excessive state transitions
    stability_c = _clip01(1.0 - metrics.transitions_per_year / 30.0)

    # Calibration: elevated time should be ~7% (WARNING + CRISIS)
    target_elevated = 0.07
    actual_elevated = metrics.pct_warning + metrics.pct_crisis
    calibration_c = _clip01(
        1.0 - abs(actual_elevated - target_elevated) / target_elevated
    )

    # Recovery speed: 1.0 at 0 post-crisis elevated days, 0.0 at 30+
    recovery_speed_c = _clip01(1.0 - metrics.avg_recovery_days / 30.0)

    # Pre-action should exist, but it should remain a scarce early-risk tool.
    # A 2-8% WATCH action band gets full volume credit; larger saturation
    # progressively loses credit.
    if metrics.pct_preaction_watch <= 0.0:
        preaction_volume_c = 0.0
    elif metrics.pct_preaction_watch < 0.02:
        preaction_volume_c = _clip01(metrics.pct_preaction_watch / 0.02)
    elif metrics.pct_preaction_watch <= 0.08:
        preaction_volume_c = 1.0
    else:
        preaction_volume_c = _clip01(1.0 - (metrics.pct_preaction_watch - 0.08) / 0.12)
    preaction_fp_c = _clip01(1.0 - metrics.preaction_fp_rate / 0.20)
    advisory_fp_c = _clip01(1.0 - metrics.advisory_fp_rate / 0.60)
    preaction_quality_c = (
        0.50 * preaction_volume_c
        + 0.35 * preaction_fp_c
        + 0.15 * advisory_fp_c
    )

    total = (
        w["detection_speed"] * detection_speed_c
        + w.get("early_action_speed", 0.0) * early_action_speed_c
        + w["fp_control"] * fp_control_c
        + w["coverage"] * coverage_c
        + w["severity"] * severity_c
        + w["stability"] * stability_c
        + w["calibration"] * calibration_c
        + w.get("recovery_speed", 0.0) * recovery_speed_c
        + w.get("preaction_quality", 0.0) * preaction_quality_c
    )

    return CrisisCompositeScore(
        detection_speed=detection_speed_c,
        early_action_speed=early_action_speed_c,
        fp_control=fp_control_c,
        coverage=coverage_c,
        severity=severity_c,
        stability=stability_c,
        calibration=calibration_c,
        recovery_speed=recovery_speed_c,
        preaction_quality=preaction_quality_c,
        total=total,
    )
