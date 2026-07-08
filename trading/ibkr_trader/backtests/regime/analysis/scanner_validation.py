"""Acceptance criteria validation for the Leading Indicator Scanner (Layer 2).

Three criteria:
1. Lead time: scanner should flag risk-off 2-4 weeks before HMM transitions to S/D
2. False positive rate: <20% of risk-off alerts should be false alarms
3. Jan 2022 detection: scanner should detect risk-off within first 3 weeks of Jan 2022
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from regime.config import REGIMES


def _extract_alert_episodes(
    signals: pd.DataFrame,
    threshold: float,
    direction: str = "risk_off",
) -> list[dict[str, pd.Timestamp]]:
    """Collapse consecutive alert rows into alert episodes."""
    if "shift_prob" not in signals.columns or "shift_dir" not in signals.columns:
        return []

    active_mask = (signals["shift_prob"] > threshold) & (signals["shift_dir"] == direction)
    episodes: list[dict[str, pd.Timestamp]] = []
    start: pd.Timestamp | None = None
    prev_dt: pd.Timestamp | None = None

    for dt, active in active_mask.items():
        if active and start is None:
            start = dt
        elif not active and start is not None:
            episodes.append({"start": start, "end": prev_dt or dt})
            start = None
        prev_dt = dt

    if start is not None:
        episodes.append({"start": start, "end": prev_dt or start})

    return episodes


def validate_scanner(
    signals: pd.DataFrame,
    threshold: float = 0.5,
    lookahead_weeks: int = 8,
) -> dict:
    """Validate scanner against acceptance criteria.

    Args:
        signals: DataFrame with columns P_G, P_R, P_S, P_D, shift_prob, shift_dir.

    Returns:
        dict with lead_time_median_weeks, false_positive_rate, jan_2022_detected,
        transitions_analyzed, verdict, and per-criterion details.
    """
    # Identify dominant regime per week
    regime_cols = [f"P_{r}" for r in REGIMES]
    available = [c for c in regime_cols if c in signals.columns]
    dom = signals[available].idxmax(axis=1).str.replace("P_", "")

    # Find transitions (week where dominant regime changes)
    transitions = []
    prev_regime = None
    for dt, regime in dom.items():
        if prev_regime is not None and regime != prev_regime:
            transitions.append({"date": dt, "from": prev_regime, "to": regime})
        prev_regime = regime

    # Risk-off transitions (to S or D)
    risk_off_transitions = [t for t in transitions if t["to"] in ("S", "D")]
    alert_episodes = _extract_alert_episodes(signals, threshold=threshold)

    # --- Criterion 1: Lead time ---
    lead_times = []
    for trans in risk_off_transitions:
        trans_date = trans["date"]
        recent_alerts = [
            episode
            for episode in alert_episodes
            if episode["start"] < trans_date
            and episode["start"] >= trans_date - pd.DateOffset(weeks=lookahead_weeks)
        ]
        if recent_alerts:
            first_flag = recent_alerts[0]["start"]
            lead_days = (trans_date - first_flag).days
            lead_weeks = lead_days / 7.0
            lead_times.append(lead_weeks)

    lead_median = float(np.median(lead_times)) if lead_times else 0.0

    # --- Criterion 2: False positive rate ---
    total_alerts = len(alert_episodes)
    false_alarms = 0
    for episode in alert_episodes:
        alert_dt = episode["start"]
        future_transition = any(
            alert_dt < trans["date"] <= alert_dt + pd.DateOffset(weeks=lookahead_weeks)
            for trans in risk_off_transitions
        )
        current_regime = dom.get(alert_dt, "")
        if not future_transition and current_regime not in ("S", "D"):
            false_alarms += 1

    fpr = false_alarms / total_alerts if total_alerts > 0 else 0.0

    # --- Criterion 3: Jan 2022 detection ---
    jan_2022_detected = False
    jan_2022_first_week = None
    jan_2022_episodes = [
        episode
        for episode in alert_episodes
        if pd.Timestamp("2022-01-01") <= episode["start"] <= pd.Timestamp("2022-01-21")
    ]
    if jan_2022_episodes:
        jan_2022_detected = True
        jan_2022_first_week = str(jan_2022_episodes[0]["start"].date())

    # --- Verdict ---
    lead_ok = 2.0 <= lead_median <= 4.0 if lead_times else False
    fpr_ok = fpr < 0.20
    jan_ok = jan_2022_detected

    verdict = "PASS" if (lead_ok and fpr_ok and jan_ok) else "FAIL"

    return {
        "lead_time_median_weeks": lead_median,
        "false_positive_rate": fpr,
        "jan_2022_detected": jan_2022_detected,
        "jan_2022_first_week": jan_2022_first_week,
        "transitions_analyzed": len(risk_off_transitions),
        "total_risk_off_alerts": total_alerts,
        "threshold": threshold,
        "lead_times": lead_times,
        "verdict": verdict,
        "criteria": {
            "lead_time": {
                "target": "2-4 weeks",
                "actual": f"{lead_median:.1f} weeks",
                "passed": lead_ok,
            },
            "fpr": {
                "target": "<20%",
                "actual": f"{fpr:.1%}",
                "passed": fpr_ok,
            },
            "jan_2022": {
                "target": True,
                "actual": jan_2022_detected,
                "passed": jan_ok,
            },
        },
    }
