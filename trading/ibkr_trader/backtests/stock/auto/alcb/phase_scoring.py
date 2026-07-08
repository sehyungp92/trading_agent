from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import time
from typing import Any
from zoneinfo import ZoneInfo


_ET = ZoneInfo("America/New_York")
_BUCKET_1000_START = time(10, 0)
_BUCKET_1000_END = time(10, 30)
_LATE_BUCKET_START = time(11, 30)
_NEUTRAL_STRUCTURAL_SCORE = 0.50


IMMUTABLE_SCORING_WEIGHTS: dict[str, float] = {
    "expected_total_r": 0.24,
    "trades_per_month": 0.20,
    "expectancy_dollar": 0.13,
    "profit_factor": 0.12,
    "net_profit": 0.10,
    "profit_protection": 0.09,
    "signal_quality": 0.06,
    "timing_quality": 0.03,
    "mfe_capture_efficiency": 0.02,
    "inv_dd": 0.01,
}

PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    1: {
        "expected_total_r": 0.23,
        "trades_per_month": 0.17,
        "profit_factor": 0.12,
        "expectancy_dollar": 0.12,
        "signal_quality": 0.14,
        "score_monotonicity": 0.10,
        "sizing_alignment": 0.07,
        "timing_quality": 0.03,
        "inv_dd": 0.02,
    },
    2: {
        "expected_total_r": 0.22,
        "net_profit": 0.16,
        "profit_protection": 0.16,
        "short_hold_24_drag_inverse": 0.14,
        "mfe_capture_efficiency": 0.10,
        "profit_factor": 0.08,
        "trades_per_month": 0.08,
        "expectancy_dollar": 0.06,
    },
    3: {
        "expected_total_r": 0.21,
        "net_profit": 0.16,
        "profit_protection": 0.14,
        "flow_mfe_exit_inverse": 0.12,
        "mfe_capture_efficiency": 0.11,
        "long_hold_capture": 0.08,
        "profit_factor": 0.08,
        "trades_per_month": 0.05,
        "expectancy_dollar": 0.05,
    },
    4: {
        "trades_per_month": 0.27,
        "expected_total_r": 0.22,
        "net_profit": 0.15,
        "expectancy_dollar": 0.09,
        "profit_factor": 0.08,
        "late_entry_quality": 0.07,
        "timing_quality": 0.05,
        "signal_quality": 0.04,
        "inv_dd": 0.03,
    },
    5: {
        "expected_total_r": 0.22,
        "trades_per_month": 0.16,
        "profit_factor": 0.12,
        "expectancy_dollar": 0.10,
        "timing_quality": 0.10,
        "extended_avwap_inverse": 0.10,
        "rvol_selectivity": 0.07,
        "net_profit": 0.08,
        "inv_dd": 0.05,
    },
    6: {
        "expected_total_r": 0.24,
        "trades_per_month": 0.17,
        "net_profit": 0.14,
        "profit_factor": 0.12,
        "expectancy_dollar": 0.10,
        "signal_quality": 0.07,
        "timing_quality": 0.06,
        "sizing_alignment": 0.05,
        "profit_protection": 0.03,
        "inv_dd": 0.02,
    },
    7: {
        "expected_total_r": 0.23,
        "net_profit": 0.15,
        "profit_protection": 0.17,
        "short_hold_24_drag_inverse": 0.16,
        "mfe_capture_efficiency": 0.10,
        "profit_factor": 0.08,
        "trades_per_month": 0.07,
        "expectancy_dollar": 0.04,
    },
    8: {
        "expected_total_r": 0.24,
        "trades_per_month": 0.18,
        "net_profit": 0.14,
        "expectancy_dollar": 0.12,
        "profit_factor": 0.11,
        "signal_quality": 0.07,
        "timing_quality": 0.06,
        "profit_protection": 0.06,
        "inv_dd": 0.02,
    },
}

NORMALIZATION_RANGES: dict[str, tuple[float, float]] = {
    "expected_total_r": (115.0, 155.0),
    "net_profit": (9000.0, 12500.0),
    "trades_per_month": (19.0, 24.5),
    "expectancy": (0.20, 0.32),
    "expectancy_dollar": (16.0, 25.0),
    "profit_factor": (1.85, 2.40),
    "mfe_capture_efficiency": (0.76, 0.90),
    "profit_protection": (0.60, 0.80),
    "short_hold_24_drag_inverse": (0.35, 0.60),
    "flow_mfe_exit_inverse": (0.78, 0.92),
    "inv_dd": (0.70, 0.84),
    "entry_quality": (0.60, 1.00),
    "signal_quality": (0.55, 0.76),
    "timing_quality": (0.45, 0.70),
    "extended_avwap_inverse": (0.45, 0.78),
    "bar9_inverse": (0.35, 0.78),
    "late_entry_quality": (0.50, 0.90),
    "score_monotonicity": (0.40, 0.70),
    "rvol_selectivity": (0.60, 0.88),
    "sizing_alignment": (0.85, 1.05),
    "long_hold_capture": (0.78, 1.00),
}


def merge_alcb_metrics(performance_metrics: Any, trades: list[Any]) -> dict[str, float]:
    metrics = asdict(performance_metrics) if is_dataclass(performance_metrics) else dict(performance_metrics)
    metrics.update(compute_alcb_phase_metrics(trades))
    return enrich_alcb_phase_metrics(metrics)


def score_alcb_phase(
    phase: int,
    metrics: dict[str, float],
    scoring_weights: dict[str, float] | None = None,
) -> float:
    weights = dict(PHASE_SCORING_WEIGHTS.get(phase, IMMUTABLE_SCORING_WEIGHTS))
    if scoring_weights:
        weights.update(scoring_weights)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    enriched = enrich_alcb_phase_metrics(metrics)
    return sum(
        (weight / total_weight) * _normalize(metric_name, enriched)
        for metric_name, weight in weights.items()
    )


def enrich_alcb_phase_metrics(metrics: dict[str, float]) -> dict[str, float]:
    enriched = dict(metrics)
    max_dd = float(metrics.get("max_drawdown_pct", 0.0))
    total_trades = float(metrics.get("total_trades", 0.0))
    expectancy_r = float(metrics.get("expectancy", 0.0))
    short_hold_total_r = float(metrics.get("short_hold_total_r", 0.0))
    short_hold_24_total_r = float(metrics.get("short_hold_24_total_r", short_hold_total_r))
    flow_reversal_short_total_r = float(metrics.get("flow_reversal_short_total_r", 0.0))
    flow_reversal_total_r = float(metrics.get("flow_reversal_total_r", flow_reversal_short_total_r))
    mfe_conviction_total_r = float(metrics.get("mfe_conviction_total_r", 0.0))
    early_1000_short_total_r = float(metrics.get("early_1000_short_total_r", 0.0))
    long_hold_total_r = float(metrics.get("long_hold_total_r", 0.0))
    carry_total_r = float(metrics.get("carry_total_r", 0.0))
    positive_total_r = float(metrics.get("positive_total_r", 0.0))

    enriched["inv_dd"] = _clip01(1.0 - max_dd / 0.12)
    enriched["expected_total_r"] = expectancy_r * total_trades
    short_hold_drag_inv = _clip01(1.0 - abs(min(short_hold_total_r, 0.0)) / 180.0)
    short_hold_24_drag_inv = _clip01(1.0 - abs(min(short_hold_24_total_r, 0.0)) / 240.0)
    flow_rev_short_inv = _clip01(1.0 - abs(min(flow_reversal_short_total_r, 0.0)) / 140.0)
    flow_rev_inv = _clip01(1.0 - abs(min(flow_reversal_total_r, 0.0)) / 90.0)
    mfe_conviction_inv = _clip01(1.0 - abs(min(mfe_conviction_total_r, 0.0)) / 75.0)
    flow_mfe_exit_inv = (flow_rev_inv + mfe_conviction_inv) / 2.0
    early_1000_drag_inv = _clip01(1.0 - abs(min(early_1000_short_total_r, 0.0)) / 150.0)
    enriched["short_hold_drag_inverse"] = short_hold_drag_inv
    enriched["short_hold_24_drag_inverse"] = short_hold_24_drag_inv
    enriched["flow_reversal_short_inverse"] = flow_rev_short_inv
    enriched["flow_reversal_inverse"] = flow_rev_inv
    enriched["mfe_conviction_inverse"] = mfe_conviction_inv
    enriched["flow_mfe_exit_inverse"] = flow_mfe_exit_inv
    enriched["early_1000_drag_inverse"] = early_1000_drag_inv
    enriched["long_hold_capture"] = (
        _clip01(max(long_hold_total_r, 0.0) / 80.0)
        + _clip01(max(long_hold_total_r, 0.0) / max(positive_total_r, 1.0))
    ) / 2.0
    enriched["carry_capture"] = (
        _clip01(max(carry_total_r, 0.0) / 12.0)
        + _clip01(max(carry_total_r, 0.0) / max(positive_total_r, 1.0))
    ) / 2.0
    enriched["profit_protection"] = (
        short_hold_24_drag_inv
        + flow_mfe_exit_inv
        + early_1000_drag_inv
    ) / 3.0
    return enriched


def _acc_avg(sum_r: float, n: int) -> float:
    return sum_r / n if n > 0 else 0.0


def _acc_shrunk(sum_r: float, n: int, prior: float, strength: float) -> float:
    if n == 0:
        return prior
    return (sum_r + prior * strength) / (n + strength)


def compute_alcb_phase_metrics(trades: list[Any]) -> dict[str, float]:
    total = len(trades)
    if total == 0:
        return _empty_phase_metrics()

    # --- Single-pass accumulators ---
    all_r_sum = 0.0
    positive_total_r = 0.0
    winner_r_sum = 0.0
    winner_mfe_sum = 0.0
    winner_n = 0

    # Hold-time buckets
    short_hold_r = 0.0; short_hold_n = 0
    short_hold_24_r = 0.0; short_hold_24_n = 0
    mid_hold_7_24_r = 0.0; mid_hold_7_24_n = 0
    short_flow_r = 0.0; short_flow_n = 0
    flow_reversal_r = 0.0; flow_reversal_n = 0
    mfe_conviction_r = 0.0; mfe_conviction_n = 0
    early_1000_short_r = 0.0; early_1000_short_n = 0
    long_hold_r = 0.0; long_hold_n = 0
    carry_r = 0.0; carry_n = 0

    # Entry-type buckets
    or_r = 0.0; or_n = 0
    combined_r = 0.0; combined_n = 0
    pdh_r = 0.0; pdh_n = 0
    reclaim_r = 0.0; reclaim_n = 0
    tight_or_r = 0.0; tight_or_n = 0

    # RVOL buckets
    mid_rvol_r = 0.0; mid_rvol_n = 0
    mid_high_rvol_r = 0.0; mid_high_rvol_n = 0
    strong_rvol_r = 0.0; strong_rvol_n = 0
    high_rvol_r = 0.0; high_rvol_n = 0

    # AVWAP premium buckets
    slight_prem_r = 0.0; slight_prem_n = 0
    extended_prem_r = 0.0; extended_prem_n = 0

    # Timing buckets
    bar9_r = 0.0; bar9_n = 0
    late_r = 0.0; late_n = 0

    # Momentum score buckets
    score_r = {4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0}
    score_n = {4: 0, 5: 0, 6: 0, 7: 0}

    winner_risk_sum = 0.0; winner_risk_n = 0
    loser_risk_sum = 0.0; loser_risk_n = 0

    _OR_ENTRY_TYPES = {"OR_BREAKOUT", "OR_RECLAIM", "AVWAP_RECLAIM"}
    _PDH_ENTRY_TYPES = {"PDH_BREAKOUT", "PDH_RECLAIM"}

    for trade in trades:
        r = float(trade.r_multiple)
        all_r_sum += r
        positive_total_r += max(r, 0.0)

        if r > 0:
            winner_r_sum += r
            winner_mfe_sum += _meta_float(trade, "mfe_r", 0.0)
            winner_n += 1
            winner_risk_sum += _risk_dollars(trade)
            winner_risk_n += 1
        elif r < 0:
            loser_risk_sum += _risk_dollars(trade)
            loser_risk_n += 1

        # Hold bars -- extract once
        hb = _hold_bars(trade)

        # Timezone conversion -- extract once, reuse for 1000-bucket + bar9 + late
        entry_et = _entry_time_et(trade)
        exit_reason = _normalize_exit(getattr(trade, "exit_reason", ""))

        if hb <= 6:
            short_hold_r += r; short_hold_n += 1
            if exit_reason == "FLOW_REVERSAL":
                short_flow_r += r; short_flow_n += 1
            if entry_et is not None:
                entry_t = entry_et.timetz().replace(tzinfo=None)
                if _BUCKET_1000_START <= entry_t < _BUCKET_1000_END:
                    early_1000_short_r += r; early_1000_short_n += 1
        if hb <= 24:
            short_hold_24_r += r; short_hold_24_n += 1
        if hb > 24:
            long_hold_r += r; long_hold_n += 1
        if 7 <= hb <= 24:
            mid_hold_7_24_r += r; mid_hold_7_24_n += 1
        if exit_reason == "FLOW_REVERSAL":
            flow_reversal_r += r; flow_reversal_n += 1
        if exit_reason == "MFE_CONVICTION":
            mfe_conviction_r += r; mfe_conviction_n += 1

        # Carry
        if _is_carry_trade(trade):
            carry_r += r; carry_n += 1

        # Entry type -- extract once
        et = _entry_type(trade)
        if et in _OR_ENTRY_TYPES:
            or_r += r; or_n += 1
        if et.startswith("COMBINED"):
            combined_r += r; combined_n += 1
        if et in _PDH_ENTRY_TYPES:
            pdh_r += r; pdh_n += 1
        if "RECLAIM" in et:
            reclaim_r += r; reclaim_n += 1

        # OR width
        if _or_width_pct(trade) < 0.2:
            tight_or_r += r; tight_or_n += 1

        # RVOL -- extract once
        rv = _rvol(trade)
        if 1.5 <= rv < 2.0:
            mid_rvol_r += r; mid_rvol_n += 1
        if 2.0 <= rv < 3.0:
            mid_high_rvol_r += r; mid_high_rvol_n += 1
        if rv >= 2.0:
            strong_rvol_r += r; strong_rvol_n += 1
        if rv >= 3.0:
            high_rvol_r += r; high_rvol_n += 1

        # AVWAP premium -- extract once
        avwap_prem = _avwap_premium_pct(trade)
        if 0.0 < avwap_prem <= 0.005:
            slight_prem_r += r; slight_prem_n += 1
        if avwap_prem > 0.005:
            extended_prem_r += r; extended_prem_n += 1

        # Timing -- reuse entry_et extracted above
        if entry_et is not None:
            minutes_from_open = (entry_et.hour * 60 + entry_et.minute) - 570
            if minutes_from_open >= 0 and minutes_from_open // 5 + 1 == 9:
                bar9_r += r; bar9_n += 1
            if entry_et.timetz().replace(tzinfo=None) >= _LATE_BUCKET_START:
                late_r += r; late_n += 1

        # Momentum score -- extract once
        ms = _momentum_score(trade)
        if ms >= 7:
            score_r[7] += r; score_n[7] += 1
        elif ms == 6:
            score_r[6] += r; score_n[6] += 1
        elif ms == 5:
            score_r[5] += r; score_n[5] += 1
        elif ms == 4:
            score_r[4] += r; score_n[4] += 1

    # --- Derived metrics from accumulators ---
    global_avg_r = all_r_sum / total

    or_avg_r = _acc_avg(or_r, or_n)
    combined_avg_r = _acc_avg(combined_r, combined_n)
    pdh_avg_r = _acc_avg(pdh_r, pdh_n)
    reclaim_avg_r = _acc_avg(reclaim_r, reclaim_n)
    mid_high_rvol_avg_r = _acc_avg(mid_high_rvol_r, mid_high_rvol_n)
    strong_rvol_avg_r = _acc_avg(strong_rvol_r, strong_rvol_n)
    high_rvol_avg_r = _acc_avg(high_rvol_r, high_rvol_n)

    tight_or_inverse = _clip01(1.0 - abs(min(tight_or_r, 0.0)) / 45.0)
    mid_rvol_inverse = _clip01(1.0 - abs(min(mid_rvol_r, 0.0)) / 40.0)
    combined_inverse = _clip01(1.0 - abs(min(combined_avg_r, 0.0)) / 0.12)
    pdh_inverse = _clip01(1.0 - abs(min(pdh_avg_r, 0.0)) / 0.12)
    or_edge = _clip01((or_avg_r + 0.02) / 0.10)
    strong_rvol_edge = _clip01((strong_rvol_avg_r + 0.02) / 0.10)
    high_rvol_edge = (
        strong_rvol_edge
        + _clip01((high_rvol_avg_r + 0.02) / 0.10)
        + mid_rvol_inverse
    ) / 3.0
    entry_quality = (
        or_edge + combined_inverse + pdh_inverse
        + tight_or_inverse + mid_rvol_inverse + high_rvol_edge
    ) / 6.0

    mfe_capture_eff = winner_r_sum / winner_mfe_sum if winner_mfe_sum > 0 else 0.0
    winner_avg_risk = _acc_avg(winner_risk_sum, winner_risk_n)
    loser_avg_risk = _acc_avg(loser_risk_sum, loser_risk_n)
    sizing_alignment = (
        _clip01(winner_avg_risk / loser_avg_risk)
        if loser_avg_risk > 0 else _NEUTRAL_STRUCTURAL_SCORE
    )

    slight_premium_avg_r = _acc_shrunk(slight_prem_r, slight_prem_n, global_avg_r, 12.0)
    extended_premium_avg_r = _acc_shrunk(extended_prem_r, extended_prem_n, global_avg_r, 18.0)
    extended_raw = _clip01(1.0 - max(0.0, slight_premium_avg_r - extended_premium_avg_r) / 0.18)
    extended_avwap_inverse = _with_neutral_sample_floor(
        extended_raw,
        sample_size=extended_prem_n,
        full_weight_at=24,
    )

    bar9_avg_r = _acc_shrunk(bar9_r, bar9_n, global_avg_r, 12.0)
    bar9_raw = _clip01((bar9_avg_r + 0.08) / 0.18)
    bar9_inverse = _with_neutral_sample_floor(
        bar9_raw,
        sample_size=bar9_n,
        full_weight_at=24,
    )

    late_avg_r = _acc_shrunk(late_r, late_n, global_avg_r, 14.0)
    late_raw = _clip01((late_avg_r + 0.02) / 0.20)
    late_entry_quality = _with_neutral_sample_floor(
        late_raw,
        sample_size=late_n,
        full_weight_at=40,
    )

    score_bucket_avgs = {
        s: _acc_shrunk(score_r[s], score_n[s], global_avg_r, 12.0)
        for s in (4, 5, 6, 7)
    }
    score_monotonicity_components = []
    for lower, higher in ((4, 5), (5, 6), (6, 7)):
        gap = score_bucket_avgs[higher] - score_bucket_avgs[lower]
        raw = _clip01((gap + 0.03) / 0.12)
        score_monotonicity_components.append(
            _with_neutral_sample_floor(
                raw,
                sample_size=min(score_n[lower], score_n[higher]),
                full_weight_at=35,
            )
        )
    score_monotonicity = (
        sum(score_monotonicity_components) / len(score_monotonicity_components)
        if score_monotonicity_components else _NEUTRAL_STRUCTURAL_SCORE
    )

    rvol_selectivity_raw = _clip01(((high_rvol_avg_r - mid_high_rvol_avg_r) + 0.02) / 0.16)
    rvol_selectivity = _with_neutral_sample_floor(
        rvol_selectivity_raw,
        sample_size=min(high_rvol_n, mid_high_rvol_n),
        full_weight_at=35,
    )
    timing_quality = (extended_avwap_inverse + bar9_inverse + late_entry_quality) / 3.0
    signal_quality = (score_monotonicity * 0.70) + (rvol_selectivity * 0.30)

    return {
        "short_hold_total_r": short_hold_r,
        "short_hold_24_total_r": short_hold_24_r,
        "mid_hold_7_24_total_r": mid_hold_7_24_r,
        "flow_reversal_short_total_r": short_flow_r,
        "flow_reversal_total_r": flow_reversal_r,
        "mfe_conviction_total_r": mfe_conviction_r,
        "early_1000_short_total_r": early_1000_short_r,
        "long_hold_total_r": long_hold_r,
        "carry_total_r": carry_r,
        "positive_total_r": positive_total_r,
        "or_avg_r": or_avg_r,
        "combined_avg_r": combined_avg_r,
        "pdh_avg_r": pdh_avg_r,
        "reclaim_avg_r": reclaim_avg_r,
        "reclaim_share": reclaim_n / total if total > 0 else 0.0,
        "mid_high_rvol_avg_r": mid_high_rvol_avg_r,
        "strong_rvol_avg_r": strong_rvol_avg_r,
        "high_rvol_avg_r": high_rvol_avg_r,
        "tight_or_inverse": tight_or_inverse,
        "mid_rvol_inverse": mid_rvol_inverse,
        "combined_inverse": combined_inverse,
        "pdh_inverse": pdh_inverse,
        "or_edge": or_edge,
        "high_rvol_edge": high_rvol_edge,
        "entry_quality": entry_quality,
        "signal_quality": signal_quality,
        "timing_quality": timing_quality,
        "rvol_selectivity": rvol_selectivity,
        "short_hold_share": short_hold_n / total if total > 0 else 0.0,
        "short_hold_24_share": short_hold_24_n / total if total > 0 else 0.0,
        "mid_hold_7_24_share": mid_hold_7_24_n / total if total > 0 else 0.0,
        "flow_reversal_share": flow_reversal_n / total if total > 0 else 0.0,
        "mfe_conviction_share": mfe_conviction_n / total if total > 0 else 0.0,
        "long_hold_share": long_hold_n / total if total > 0 else 0.0,
        "carry_share": carry_n / total if total > 0 else 0.0,
        "mfe_capture_efficiency": mfe_capture_eff,
        "winner_avg_risk_dollars": winner_avg_risk,
        "loser_avg_risk_dollars": loser_avg_risk,
        "sizing_alignment": sizing_alignment,
        "slight_premium_avg_r": slight_premium_avg_r,
        "extended_premium_avg_r": extended_premium_avg_r,
        "extended_avwap_inverse": extended_avwap_inverse,
        "bar9_avg_r": bar9_avg_r,
        "bar9_inverse": bar9_inverse,
        "late_avg_r": late_avg_r,
        "late_entry_quality": late_entry_quality,
        "score_4_avg_r": score_bucket_avgs[4],
        "score_5_avg_r": score_bucket_avgs[5],
        "score_6_avg_r": score_bucket_avgs[6],
        "score_7_avg_r": score_bucket_avgs[7],
        "score_monotonicity": score_monotonicity,
    }


def _empty_phase_metrics() -> dict[str, float]:
    """Return zeroed metrics dict for empty trade lists."""
    return {
        "short_hold_total_r": 0.0, "short_hold_24_total_r": 0.0,
        "mid_hold_7_24_total_r": 0.0, "flow_reversal_short_total_r": 0.0,
        "flow_reversal_total_r": 0.0, "mfe_conviction_total_r": 0.0,
        "early_1000_short_total_r": 0.0, "long_hold_total_r": 0.0,
        "carry_total_r": 0.0, "positive_total_r": 0.0,
        "or_avg_r": 0.0, "combined_avg_r": 0.0, "pdh_avg_r": 0.0,
        "reclaim_avg_r": 0.0, "reclaim_share": 0.0,
        "mid_high_rvol_avg_r": 0.0, "strong_rvol_avg_r": 0.0,
        "high_rvol_avg_r": 0.0, "tight_or_inverse": 1.0,
        "mid_rvol_inverse": 1.0, "combined_inverse": 1.0,
        "pdh_inverse": 1.0, "or_edge": 0.0, "high_rvol_edge": 0.0,
        "entry_quality": 0.0, "signal_quality": _NEUTRAL_STRUCTURAL_SCORE,
        "timing_quality": _NEUTRAL_STRUCTURAL_SCORE,
        "rvol_selectivity": _NEUTRAL_STRUCTURAL_SCORE,
        "short_hold_share": 0.0, "short_hold_24_share": 0.0,
        "mid_hold_7_24_share": 0.0, "flow_reversal_share": 0.0,
        "mfe_conviction_share": 0.0, "long_hold_share": 0.0,
        "carry_share": 0.0, "mfe_capture_efficiency": 0.0,
        "winner_avg_risk_dollars": 0.0, "loser_avg_risk_dollars": 0.0,
        "sizing_alignment": _NEUTRAL_STRUCTURAL_SCORE,
        "slight_premium_avg_r": 0.0, "extended_premium_avg_r": 0.0,
        "extended_avwap_inverse": _NEUTRAL_STRUCTURAL_SCORE,
        "bar9_avg_r": 0.0, "bar9_inverse": _NEUTRAL_STRUCTURAL_SCORE,
        "late_avg_r": 0.0, "late_entry_quality": _NEUTRAL_STRUCTURAL_SCORE,
        "score_4_avg_r": 0.0, "score_5_avg_r": 0.0,
        "score_6_avg_r": 0.0, "score_7_avg_r": 0.0,
        "score_monotonicity": _NEUTRAL_STRUCTURAL_SCORE,
    }


def _entry_type(trade: Any) -> str:
    if getattr(trade, "entry_type", None):
        return str(trade.entry_type)
    metadata = getattr(trade, "metadata", None) or {}
    return str(metadata.get("entry_type", "UNKNOWN"))


def _hold_bars(trade: Any) -> int:
    try:
        return int(getattr(trade, "hold_bars", 0))
    except (TypeError, ValueError):
        return 0


def _rvol(trade: Any) -> float:
    return _meta_float(trade, "rvol_at_entry", 0.0)


def _or_width_pct(trade: Any) -> float:
    or_high = _meta_float(trade, "or_high", 0.0)
    or_low = _meta_float(trade, "or_low", 0.0)
    if or_high <= 0:
        return 0.0
    return ((or_high - or_low) / or_high) * 100.0


def _meta_float(trade: Any, key: str, default: float = 0.0) -> float:
    metadata = getattr(trade, "metadata", None) or {}
    try:
        return float(metadata.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _risk_dollars(trade: Any) -> float:
    try:
        risk_per_share = float(getattr(trade, "risk_per_share", 0.0))
        quantity = float(getattr(trade, "quantity", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, risk_per_share * quantity)


def _momentum_score(trade: Any) -> int:
    metadata = getattr(trade, "metadata", None) or {}
    raw_value = metadata.get("momentum_score", getattr(trade, "momentum_score", 0))
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return 0


def _avwap_premium_pct(trade: Any) -> float:
    avwap = _meta_float(trade, "avwap_at_entry", 0.0)
    entry_price = float(getattr(trade, "entry_price", 0.0))
    if avwap <= 0 or entry_price <= 0:
        return 0.0
    return (entry_price - avwap) / avwap



def _normalize(metric_name: str, metrics: dict[str, float]) -> float:
    value = float(metrics.get(metric_name, 0.0))
    lo, hi = NORMALIZATION_RANGES.get(metric_name, (0.0, 1.0))
    if hi <= lo:
        return 0.0
    return _clip01((value - lo) / (hi - lo))


def _normalize_exit(exit_reason: str | None) -> str:
    return str(exit_reason or "").strip().upper()


def _entry_time_et(trade: Any):
    fill_time = getattr(trade, "fill_time", None) or getattr(trade, "entry_time", None)
    if fill_time is None:
        metadata = getattr(trade, "metadata", None) or {}
        fill_time = metadata.get("fill_time")
    if fill_time is None:
        return None
    try:
        return fill_time.astimezone(_ET)
    except Exception:
        return fill_time


def _entry_bar_number(trade: Any) -> int:
    entry_dt = _entry_time_et(trade)
    if entry_dt is None:
        return 0
    minutes_from_open = (entry_dt.hour * 60 + entry_dt.minute) - 570
    if minutes_from_open < 0:
        return 0
    return minutes_from_open // 5 + 1


def _in_1000_bucket(trade: Any) -> bool:
    entry_dt = _entry_time_et(trade)
    if entry_dt is None:
        return False
    entry_t = entry_dt.timetz().replace(tzinfo=None)
    return _BUCKET_1000_START <= entry_t < _BUCKET_1000_END


def _is_late_trade(trade: Any) -> bool:
    entry_dt = _entry_time_et(trade)
    if entry_dt is None:
        return False
    entry_t = entry_dt.timetz().replace(tzinfo=None)
    return entry_t >= _LATE_BUCKET_START


def _is_carry_trade(trade: Any) -> bool:
    entry_time = getattr(trade, "entry_time", None)
    exit_time = getattr(trade, "exit_time", None)
    if entry_time is None or exit_time is None:
        return False
    return exit_time.date() > entry_time.date()


def _with_neutral_sample_floor(value: float, *, sample_size: int, full_weight_at: int) -> float:
    if full_weight_at <= 0:
        return _clip01(value)
    weight = _clip01(float(sample_size) / float(full_weight_at))
    return float((_NEUTRAL_STRUCTURAL_SCORE * (1.0 - weight)) + (_clip01(value) * weight))


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
