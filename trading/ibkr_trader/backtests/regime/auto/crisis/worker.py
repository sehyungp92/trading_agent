"""Crisis detection worker -- vectorized fast evaluation.

Pre-computes all indicator time series once in init_worker(), then
applies threshold classification + conjunction + hysteresis per candidate
in ~2-5ms (vs ~65s for the naive config-patching approach).

Approach:
  1. init_worker(): load parquets, forward-fill, pre-compute VIX/spread/slope/
     slope_20d_change/spy_10d_return as numpy arrays.  Lazily cache rolling
     SPY-TLT correlation and rolling was-inverted flags per window/lookback.
  2. _fast_evaluate(): vectorized threshold classification -> conjunction ->
     sequential hysteresis loop (~5600 iterations) -> inline metrics.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtests.shared.auto.types import ScoredCandidate

logger = logging.getLogger(__name__)

_INTEGER_PARAMS = frozenset({
    "CORR_WINDOW", "SLOPE_INVERSION_LOOKBACK",
    "WATCH_MIN_PRIMARY", "WARNING_MIN_PRIMARY",
    "CRISIS_MIN_PRIMARY", "CRISIS_ALT_WARNING",
    "HYBRID_WARNING_MIN_CRISIS", "HYBRID_WARNING_MIN_PRIMARY",
    "DEESCALATE_CRISIS_DAYS", "DEESCALATE_WARNING_DAYS", "DEESCALATE_WATCH_DAYS",
    "ACCEL_DEESCALATE_NORMAL_DAYS",
    "ADVISORY_WATCH_MIN_PRIMARY", "ADVISORY_WATCH_MIN_WARNING",
    "ADVISORY_WATCH_MIN_CRISIS", "STRESS_FORMATION_MIN_SCORE",
    "GRIND_VIX_PERSIST_DAYS",
    "HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS",
    "HARD_CREDIT_IMPULSE_WARNING_MIN_PRIMARY",
})

# ---------------------------------------------------------------------------
# Pre-computed state (populated by init_worker)
# ---------------------------------------------------------------------------
_n: int = 0
_dates: np.ndarray | None = None           # datetime64[ns]
_vix: np.ndarray | None = None             # float64, ffilled
_spread_bps: np.ndarray | None = None      # float64, converted to bps
_slope: np.ndarray | None = None           # float64, ffilled
_slope_20d_change: np.ndarray | None = None  # float64
_slope_negative: np.ndarray | None = None  # bool (for inversion cache)
_spy_3d_return: np.ndarray | None = None   # float64
_spy_5d_return: np.ndarray | None = None   # float64
_spy_10d_return: np.ndarray | None = None  # float64
_spy_20d_return: np.ndarray | None = None  # float64
_vix_3d_change: np.ndarray | None = None   # float64
_spread_20d_change_bps: np.ndarray | None = None  # float64
_spy_ret_pd: pd.Series | None = None       # for rolling correlation
_tlt_ret_pd: pd.Series | None = None
_corr_cache: dict[int, np.ndarray] = {}    # window -> correlation array
_inversion_cache: dict[int, np.ndarray] = {}  # lookback -> was_inverted array
_crisis_mask: np.ndarray | None = None     # bool (labeled crisis periods)
_defaults: dict[str, Any] = {}             # config defaults snapshot


def init_worker(data_dir: str) -> None:
    """Pre-compute all indicator time series from parquet data."""
    global _n, _dates, _vix, _spread_bps, _slope, _slope_20d_change
    global _slope_negative, _spy_3d_return, _spy_5d_return, _spy_10d_return
    global _spy_20d_return, _vix_3d_change, _spread_20d_change_bps
    global _spy_ret_pd, _tlt_ret_pd
    global _corr_cache, _inversion_cache, _crisis_mask, _defaults

    import regime.crisis.config as C

    data_path = Path(data_dir)
    market_df = pd.read_parquet(data_path / "market_df.parquet")
    strat_ret_df = pd.read_parquet(data_path / "strat_ret_df.parquet")

    # Snapshot config defaults (used as fallback in _param)
    _defaults = {
        k: getattr(C, k)
        for k in dir(C)
        if k.isupper() and not k.startswith("ALERT") and hasattr(C, k)
    }

    # Common dates between market and return data
    common_dates = market_df.index.intersection(strat_ret_df.index)
    _n = len(common_dates)
    _dates = common_dates.values

    # --- Pre-compute static series ---

    # VIX (forward-fill = equivalent to _safe_last per-date behavior)
    _vix = (market_df["VIX"].reindex(common_dates).ffill()
            .values.astype(np.float64))
    _vix_3d_change = np.zeros(_n, dtype=np.float64)
    if _n > 3:
        _vix_3d_change[3:] = _vix[3:] - _vix[:_n - 3]

    # Credit Spread with bps conversion (FRED data < 50 means pct points)
    spread_raw = (market_df["SPREAD"].reindex(common_dates).ffill()
                  .values.astype(np.float64))
    _spread_bps = np.where(spread_raw < 50, spread_raw * 100.0, spread_raw)
    _spread_20d_change_bps = np.zeros(_n, dtype=np.float64)
    if _n > 20:
        _spread_20d_change_bps[20:] = _spread_bps[20:] - _spread_bps[:_n - 20]

    # Yield curve slope
    _slope = (market_df["SLOPE_10Y2Y"].reindex(common_dates).ffill()
              .values.astype(np.float64))
    _slope_negative = _slope < 0

    # Slope 20d change: slope[i] - slope[i-20]
    _slope_20d_change = np.zeros(_n, dtype=np.float64)
    if _n > 20:
        _slope_20d_change[20:] = _slope[20:] - _slope[:_n - 20]

    # SPY cumulative returns ending at each date.
    spy_ret = (strat_ret_df["SPY"].reindex(common_dates).fillna(0.0)
               .values.astype(np.float64))
    _spy_3d_return = _rolling_compound_return(spy_ret, 3)
    _spy_5d_return = _rolling_compound_return(spy_ret, 5)
    _spy_10d_return = _rolling_compound_return(spy_ret, 10)
    _spy_20d_return = _rolling_compound_return(spy_ret, 20)

    # SPY-TLT returns (pandas Series) for rolling correlation
    _spy_ret_pd = strat_ret_df["SPY"].reindex(common_dates)
    _tlt_ret_pd = strat_ret_df["TLT"].reindex(common_dates)

    # Clear and pre-populate caches for common parameter values
    _corr_cache = {}
    _inversion_cache = {}
    for w in (10, 15, 20):
        _get_corr(w)
    for lb in (60, 75, 90, 120):
        _get_inversion(lb)

    # Build crisis period mask for FP rate computation
    from backtests.regime.crisis_validation import CRISIS_PERIODS
    _crisis_mask = np.zeros(_n, dtype=bool)
    for _, (start, end, _) in CRISIS_PERIODS.items():
        _crisis_mask |= (
            (_dates >= np.datetime64(start)) & (_dates <= np.datetime64(end))
        )

    logger.info(
        "Worker initialized: %d dates, %d crisis days",
        _n, int(_crisis_mask.sum()),
    )


# ---------------------------------------------------------------------------
# Lazy caches for parameterized series
# ---------------------------------------------------------------------------

def _rolling_compound_return(returns: np.ndarray, window: int) -> np.ndarray:
    """Compound returns over the trailing ``window`` days ending at each index."""
    out = np.full(_n, np.nan, dtype=np.float64)
    if window <= 0 or _n <= window:
        return out
    cumret = np.cumprod(1.0 + returns)
    out[window:] = cumret[window:] / cumret[:_n - window] - 1.0
    return out


def _rolling_all_ge(values: np.ndarray, window: int, threshold: float) -> np.ndarray:
    """True when all values in the trailing window are at least threshold."""
    out = np.zeros(_n, dtype=bool)
    if window <= 0 or _n < window:
        return out
    ok = (~np.isnan(values)) & (values >= threshold)
    cum = np.cumsum(ok.astype(np.int32))
    prev = np.concatenate(([0], cum[:-window]))
    out[window - 1:] = (cum[window - 1:] - prev) == window
    return out


def _get_corr(window: int) -> np.ndarray:
    """Get or compute rolling SPY-TLT correlation for given window."""
    if window not in _corr_cache:
        c = _spy_ret_pd.rolling(window, min_periods=window).corr(_tlt_ret_pd)
        _corr_cache[window] = c.values.astype(np.float64)
    return _corr_cache[window]


def _get_inversion(lookback: int) -> np.ndarray:
    """Get or compute rolling was-inverted flag for given lookback.

    was_inverted[i] = True if slope < 0 at any point in [i-lookback+1, i].
    Uses cumsum trick: rolling sum of negative flags > 0 means inversion.
    """
    if lookback not in _inversion_cache:
        neg = _slope_negative.astype(np.float64)
        cum = np.cumsum(neg)
        rolling_sum = np.zeros(_n, dtype=np.float64)
        lim = min(lookback, _n)
        rolling_sum[:lim] = cum[:lim]
        if _n > lookback:
            rolling_sum[lookback:] = cum[lookback:] - cum[:_n - lookback]
        _inversion_cache[lookback] = rolling_sum > 0
    return _inversion_cache[lookback]


# ---------------------------------------------------------------------------
# Fast vectorized evaluation
# ---------------------------------------------------------------------------

def _param(muts: dict, key: str) -> Any:
    """Read param from mutations dict, falling back to config defaults."""
    if key in muts:
        v = muts[key]
        return int(v) if key in _INTEGER_PARAMS else float(v)
    return _defaults.get(key, 0)


def _fast_evaluate(all_muts: dict[str, Any]) -> dict[str, float]:
    """Run vectorized crisis detection and return metrics dict.

    Steps:
      1. Read threshold/structural params from mutations (or defaults)
      2. Classify each channel (vectorized numpy boolean masks)
      3. Apply conjunction logic (vectorized)
      4. Apply hysteresis (sequential loop, ~5600 iterations)
      5. Compute detection latency, FP rates, transitions, distribution
    """
    n = _n

    # --- Read all 27 parameters ---
    vix_watch = _param(all_muts, "VIX_WATCH")
    vix_warning = _param(all_muts, "VIX_WARNING")
    vix_crisis = _param(all_muts, "VIX_CRISIS")

    spread_watch = _param(all_muts, "SPREAD_WATCH_BPS")
    spread_warning = _param(all_muts, "SPREAD_WARNING_BPS")
    spread_crisis = _param(all_muts, "SPREAD_CRISIS_BPS")

    slope_watch_thresh = _param(all_muts, "SLOPE_WATCH_THRESHOLD")
    slope_warn = _param(all_muts, "SLOPE_STEEPEN_WARNING")
    slope_cris = _param(all_muts, "SLOPE_STEEPEN_CRISIS")
    slope_lookback = int(_param(all_muts, "SLOPE_INVERSION_LOOKBACK"))

    corr_window = int(_param(all_muts, "CORR_WINDOW"))
    corr_watch = _param(all_muts, "CORR_WATCH")
    corr_warning = _param(all_muts, "CORR_WARNING")
    corr_crisis_thr = _param(all_muts, "CORR_CRISIS")
    corr_spy_dd = _param(all_muts, "CORR_CRISIS_SPY_DD")

    watch_min = int(_param(all_muts, "WATCH_MIN_PRIMARY"))
    warning_min = int(_param(all_muts, "WARNING_MIN_PRIMARY"))
    crisis_min = int(_param(all_muts, "CRISIS_MIN_PRIMARY"))
    crisis_alt = int(_param(all_muts, "CRISIS_ALT_WARNING"))

    deesc_crisis_d = int(_param(all_muts, "DEESCALATE_CRISIS_DAYS"))
    deesc_warning_d = int(_param(all_muts, "DEESCALATE_WARNING_DAYS"))
    deesc_watch_d = int(_param(all_muts, "DEESCALATE_WATCH_DAYS"))

    hybrid_crisis_min = int(_param(all_muts, "HYBRID_WARNING_MIN_CRISIS"))
    hybrid_warning_min = int(_param(all_muts, "HYBRID_WARNING_MIN_PRIMARY"))
    spy_dd_watch = _param(all_muts, "SPY_DD_WATCH")
    spy_dd_warning = _param(all_muts, "SPY_DD_WARNING")
    spy_dd_crisis = _param(all_muts, "SPY_DD_CRISIS")

    accel_normal_d = int(_param(all_muts, "ACCEL_DEESCALATE_NORMAL_DAYS"))

    advisory_watch_min = int(_param(all_muts, "ADVISORY_WATCH_MIN_PRIMARY"))
    advisory_warning_min = int(_param(all_muts, "ADVISORY_WATCH_MIN_WARNING"))
    advisory_crisis_min = int(_param(all_muts, "ADVISORY_WATCH_MIN_CRISIS"))

    stress_min_score = int(_param(all_muts, "STRESS_FORMATION_MIN_SCORE"))
    shock_spy_3d = _param(all_muts, "SHOCK_SPY_3D_RETURN")
    shock_spy_5d = _param(all_muts, "SHOCK_SPY_5D_RETURN")
    shock_vix_3d = _param(all_muts, "SHOCK_VIX_3D_CHANGE")
    shock_min_vix = _param(all_muts, "SHOCK_MIN_VIX")
    shock_corr_min = _param(all_muts, "SHOCK_CORR_MIN")
    shock_corr_spy_5d = _param(all_muts, "SHOCK_CORR_SPY_5D_RETURN")

    grind_spread_20d = _param(all_muts, "GRIND_SPREAD_20D_CHANGE_BPS")
    grind_spy_20d = _param(all_muts, "GRIND_SPY_20D_RETURN")
    grind_vix_min = _param(all_muts, "GRIND_VIX_MIN")
    grind_vix_persist_days = int(_param(all_muts, "GRIND_VIX_PERSIST_DAYS"))
    grind_spread_confirm = _param(all_muts, "GRIND_SPREAD_CONFIRM_BPS")
    grind_spy_confirm_20d = _param(all_muts, "GRIND_SPY_CONFIRM_20D_RETURN")
    credit_impulse_spread = _param(all_muts, "CREDIT_IMPULSE_SPREAD_BPS")
    credit_impulse_spy_3d = _param(all_muts, "CREDIT_IMPULSE_SPY_3D_RETURN")
    credit_impulse_min_vix = _param(all_muts, "CREDIT_IMPULSE_MIN_VIX")
    hard_credit_impulse_persist_days = int(_param(
        all_muts,
        "HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS",
    ))
    hard_credit_impulse_min_primary = int(_param(
        all_muts,
        "HARD_CREDIT_IMPULSE_WARNING_MIN_PRIMARY",
    ))

    start_idx = max(slope_lookback, corr_window, 20)

    # --- 1. Channel classification (vectorized) ---

    # VIX
    vix_nan = np.isnan(_vix)
    vl = np.where(
        vix_nan, 0,
        np.where(_vix >= vix_crisis, 3,
        np.where(_vix >= vix_warning, 2,
        np.where(_vix >= vix_watch, 1, 0))))

    # Credit Spread
    sp_nan = np.isnan(_spread_bps)
    sl = np.where(
        sp_nan, 0,
        np.where(_spread_bps >= spread_crisis, 3,
        np.where(_spread_bps >= spread_warning, 2,
        np.where(_spread_bps >= spread_watch, 1, 0))))

    # Yield Curve (layered: watch -> warning -> crisis, higher overwrites)
    was_inv = _get_inversion(slope_lookback)
    ok = ~np.isnan(_slope)
    yl = np.zeros(n, dtype=np.int32)
    yl = np.where(ok & (_slope <= slope_watch_thresh), 1, yl)
    yl = np.where(
        ok & was_inv & (_slope_20d_change >= slope_warn), 2, yl)
    yl = np.where(
        ok & was_inv & (_slope_20d_change >= slope_cris), 3, yl)

    # SPY-TLT Correlation
    corr_arr = _get_corr(corr_window)
    cn = np.isnan(corr_arr)
    cl = np.where(
        cn, 0,
        np.where(
            (corr_arr >= corr_crisis_thr) & (_spy_10d_return <= corr_spy_dd),
            3,
        np.where(corr_arr >= corr_warning, 2,
        np.where(corr_arr >= corr_watch, 1, 0))))

    # SPY Drawdown (5th primary channel -- thresholds are negative)
    sd_nan = np.isnan(_spy_10d_return)
    sdl = np.where(
        sd_nan, 0,
        np.where(_spy_10d_return <= spy_dd_crisis, 3,
        np.where(_spy_10d_return <= spy_dd_warning, 2,
        np.where(_spy_10d_return <= spy_dd_watch, 1, 0))))

    # --- 2. Conjunction (vectorized) ---
    w_cnt = ((vl >= 1).astype(np.int32) + (sl >= 1).astype(np.int32) +
             (yl >= 1).astype(np.int32) + (cl >= 1).astype(np.int32) +
             (sdl >= 1).astype(np.int32))
    wa_cnt = ((vl >= 2).astype(np.int32) + (sl >= 2).astype(np.int32) +
              (yl >= 2).astype(np.int32) + (cl >= 2).astype(np.int32) +
              (sdl >= 2).astype(np.int32))
    cr_cnt = ((vl >= 3).astype(np.int32) + (sl >= 3).astype(np.int32) +
              (yl >= 3).astype(np.int32) + (cl >= 3).astype(np.int32) +
              (sdl >= 3).astype(np.int32))

    raw = np.zeros(n, dtype=np.int32)
    raw = np.where(w_cnt >= watch_min, 1, raw)
    raw = np.where(wa_cnt >= warning_min, 2, raw)
    # Hybrid WARNING: when any channel at CRISIS, lower WARNING conjunction
    raw = np.where(
        (cr_cnt >= hybrid_crisis_min) & (wa_cnt >= hybrid_warning_min),
        np.maximum(raw, 2), raw)
    raw = np.where(
        (cr_cnt >= crisis_min) | (wa_cnt >= crisis_alt), 3, raw)

    # --- 2b. Early advisory / stress-formation pre-action layer ---
    shock_score = (
        (_spy_3d_return <= shock_spy_3d).astype(np.int32)
        + (_spy_5d_return <= shock_spy_5d).astype(np.int32)
        + ((_vix >= shock_min_vix) & (_vix_3d_change >= shock_vix_3d)).astype(np.int32)
        + (
            (corr_arr >= shock_corr_min)
            & (_spy_5d_return <= shock_corr_spy_5d)
        ).astype(np.int32)
    )
    vix_persistent = _rolling_all_ge(_vix, grind_vix_persist_days, grind_vix_min)
    grind_score = (
        (_spread_20d_change_bps >= grind_spread_20d).astype(np.int32)
        + (_spy_20d_return <= grind_spy_20d).astype(np.int32)
        + vix_persistent.astype(np.int32)
        + (
            (_spread_bps >= grind_spread_confirm)
            & (_spy_20d_return <= grind_spy_confirm_20d)
        ).astype(np.int32)
    )
    credit_impulse_active = (
        (_spread_bps >= credit_impulse_spread)
        & (_spy_3d_return <= credit_impulse_spy_3d)
        & (_vix >= credit_impulse_min_vix)
    )
    credit_impulse_score = np.where(credit_impulse_active, 2, 0)
    stress_active = (
        np.maximum(np.maximum(shock_score, grind_score), credit_impulse_score)
        >= stress_min_score
    )

    if hard_credit_impulse_persist_days > 0:
        hard_bridge_candidate = (
            credit_impulse_active
            & (wa_cnt >= hard_credit_impulse_min_primary)
        )
        hard_bridge_confirmed = _rolling_all_ge(
            hard_bridge_candidate.astype(np.float64),
            hard_credit_impulse_persist_days,
            1.0,
        )
        raw = np.where(hard_bridge_confirmed, np.maximum(raw, 2), raw)

    # --- 3. Hysteresis (sequential -- state machine can't vectorize) ---
    raw_list = raw[start_idx:].tolist()
    num_eval = len(raw_list)
    final = [0] * num_eval

    cur = 0
    db = 0
    db_normal = 0  # consecutive days raw==0
    for i in range(num_eval):
        rl = raw_list[i]
        # Track consecutive all-normal days
        if rl == 0:
            db_normal += 1
        else:
            db_normal = 0

        if rl > cur:
            cur = rl
            db = 0
        elif rl == cur:
            db = 0
        else:
            db += 1
            # Accelerated: raw NORMAL for N+ days -> jump to NORMAL
            if accel_normal_d > 0 and rl == 0 and db_normal >= accel_normal_d and cur > 0:
                cur = 0
                db = 0
            else:
                req = (deesc_crisis_d if cur == 3 else
                       deesc_warning_d if cur == 2 else
                       deesc_watch_d if cur == 1 else 0)
                if db >= req:
                    cur = max(cur - 1, rl)
                    db = 0
        final[i] = cur

    final_arr = np.array(final, dtype=np.int32)
    final_full = np.zeros(n, dtype=np.int32)
    final_full[start_idx:] = final_arr

    advisory_full = np.zeros(n, dtype=np.int32)
    advisory_full = np.where(final_full >= 2, final_full, advisory_full)
    advisory_watch = (
        (cr_cnt >= advisory_crisis_min)
        | (wa_cnt >= advisory_warning_min)
        | stress_active
        | (w_cnt >= advisory_watch_min)
    )
    advisory_full = np.where(
        (final_full < 2) & advisory_watch,
        1,
        advisory_full,
    ).astype(np.int32)

    action_full = np.where(
        final_full >= 2,
        final_full,
        np.where(stress_active, 1, 0),
    ).astype(np.int32)

    # --- 4. Metrics computation ---
    return _compute_metrics(
        final_arr,
        start_idx,
        advisory_levels=advisory_full[start_idx:],
        action_levels=action_full[start_idx:],
    )


# ---------------------------------------------------------------------------
# Metrics extraction (inline, avoids DataFrame overhead)
# ---------------------------------------------------------------------------

def _compute_metrics(
    levels: np.ndarray,
    start_idx: int,
    advisory_levels: np.ndarray | None = None,
    action_levels: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute CrisisMetrics from final alert level array.

    Separates crises (D/S types, detected at WARNING+) from corrections
    (C type, detected at WATCH+).
    """
    from backtests.regime.crisis_validation import CRISIS_PERIODS

    ne = len(levels)
    if ne == 0:
        return _empty_metrics()

    if advisory_levels is None or len(advisory_levels) != ne:
        advisory_levels = levels
    if action_levels is None or len(action_levels) != ne:
        action_levels = levels

    eval_dates = _dates[start_idx:]

    # --- Detection latency (separate crises from corrections) ---
    crisis_lats: list[float] = []
    action_lats: list[float] = []
    advisory_lats: list[float] = []
    correction_lats: list[float] = []
    peaks: list[float] = []
    gfc_lat = -1.0
    covid_lat = -1.0

    total_crises = sum(1 for v in CRISIS_PERIODS.values() if v[2] != "C")
    total_corrections = sum(1 for v in CRISIS_PERIODS.values() if v[2] == "C")

    for name, (s, e, period_type) in CRISIS_PERIODS.items():
        s64 = np.datetime64(s)
        e64 = np.datetime64(e)
        mask = (eval_dates >= s64) & (eval_dates <= e64)
        period = levels[mask]
        if len(period) == 0:
            continue

        # Corrections use WATCH+ (level >= 1); crises use WARNING+ (level >= 2)
        detect_threshold = 1 if period_type == "C" else 2
        qualifying = period >= detect_threshold
        if not qualifying.any():
            continue

        idx_in_period = int(np.argmax(qualifying))
        all_idx = np.where(mask)[0]
        det_date = eval_dates[all_idx[idx_in_period]]
        lat = float((det_date - s64) / np.timedelta64(1, "D"))

        if period_type == "C":
            correction_lats.append(lat)
        else:
            crisis_lats.append(lat)
            peaks.append(float(period.max()))
            if name == "GFC":
                gfc_lat = lat
            elif name == "COVID":
                covid_lat = lat

        if period_type != "C":
            action_period = action_levels[mask]
            action_qualifying = action_period >= 1
            if action_qualifying.any():
                idx_in_period = int(np.argmax(action_qualifying))
                all_idx = np.where(mask)[0]
                det_date = eval_dates[all_idx[idx_in_period]]
                action_lats.append(float((det_date - s64) / np.timedelta64(1, "D")))

            advisory_period = advisory_levels[mask]
            advisory_qualifying = advisory_period >= 1
            if advisory_qualifying.any():
                idx_in_period = int(np.argmax(advisory_qualifying))
                all_idx = np.where(mask)[0]
                det_date = eval_dates[all_idx[idx_in_period]]
                advisory_lats.append(float((det_date - s64) / np.timedelta64(1, "D")))

    n_crisis_det = len(crisis_lats)
    n_corr_det = len(correction_lats)
    avg_lat = sum(crisis_lats) / n_crisis_det if n_crisis_det else 99.0
    max_lat = max(crisis_lats) if crisis_lats else 99.0
    avg_action_lat = sum(action_lats) / len(action_lats) if action_lats else 99.0
    max_action_lat = max(action_lats) if action_lats else 99.0
    avg_advisory_lat = (
        sum(advisory_lats) / len(advisory_lats) if advisory_lats else 99.0
    )
    max_advisory_lat = max(advisory_lats) if advisory_lats else 99.0
    corr_avg_lat = (
        sum(correction_lats) / n_corr_det if n_corr_det else 99.0
    )
    avg_peak = sum(peaks) / len(peaks) if peaks else 0.0

    # --- False positive rates ---
    cm = _crisis_mask[start_idx:]
    nc = ~cm
    tnc = int(nc.sum())
    if tnc > 0:
        wfp = float(((levels >= 2) & nc).sum()) / tnc
        cfp = float(((levels >= 3) & nc).sum()) / tnc
        advisory_fp = float(((advisory_levels >= 1) & nc).sum()) / tnc
        preaction_fp = float(((action_levels >= 1) & nc).sum()) / tnc
    else:
        wfp = cfp = 0.0
        advisory_fp = preaction_fp = 0.0

    # --- Recovery speed: days at WARNING+ in 60-day post-crisis window ---
    recovery_days_list: list[int] = []
    for name, (s, e, period_type) in CRISIS_PERIODS.items():
        if period_type == "C":
            continue
        s64, e64 = np.datetime64(s), np.datetime64(e)
        # Only measure recovery for detected crises
        during_mask = (eval_dates >= s64) & (eval_dates <= e64)
        during_levels = levels[during_mask]
        if len(during_levels) == 0 or (during_levels >= 2).sum() == 0:
            continue
        post_mask = (eval_dates > e64) & (eval_dates <= e64 + np.timedelta64(60, "D"))
        post_levels = levels[post_mask]
        if len(post_levels) > 0:
            recovery_days_list.append(int(np.sum(post_levels >= 2)))

    avg_recovery_days = (
        sum(recovery_days_list) / len(recovery_days_list)
        if recovery_days_list else 0.0
    )
    max_recovery_days = float(max(recovery_days_list)) if recovery_days_list else 0.0

    # --- Transitions ---
    trans = int(np.sum(levels[1:] != levels[:-1])) if ne > 1 else 0
    yrs = ne / 252.0
    tpy = trans / yrs if yrs > 0 else 0.0

    # --- Distribution ---
    t = float(ne)
    return {
        "avg_latency": avg_lat,
        "max_latency": max_lat,
        "avg_action_latency": avg_action_lat,
        "max_action_latency": max_action_lat,
        "avg_advisory_latency": avg_advisory_lat,
        "max_advisory_latency": max_advisory_lat,
        "gfc_latency": gfc_lat,
        "covid_latency": covid_lat,
        "crises_detected": n_crisis_det,
        "total_crises": total_crises,
        "corrections_detected": n_corr_det,
        "total_corrections": total_corrections,
        "correction_avg_latency": corr_avg_lat,
        "warning_fp_rate": wfp,
        "crisis_fp_rate": cfp,
        "advisory_fp_rate": advisory_fp,
        "preaction_fp_rate": preaction_fp,
        "avg_peak_level": avg_peak,
        "total_transitions": trans,
        "transitions_per_year": tpy,
        "pct_normal": float(np.sum(levels == 0)) / t,
        "pct_watch": float(np.sum(levels == 1)) / t,
        "pct_warning": float(np.sum(levels == 2)) / t,
        "pct_crisis": float(np.sum(levels == 3)) / t,
        "pct_advisory_watch": float(np.sum(advisory_levels == 1)) / t,
        "pct_preaction_watch": float(np.sum(action_levels == 1)) / t,
        "avg_recovery_days": avg_recovery_days,
        "max_recovery_days": max_recovery_days,
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "avg_latency": 99.0, "max_latency": 99.0,
        "avg_action_latency": 99.0, "max_action_latency": 99.0,
        "avg_advisory_latency": 99.0, "max_advisory_latency": 99.0,
        "gfc_latency": -1.0, "covid_latency": -1.0,
        "crises_detected": 0, "total_crises": 7,
        "corrections_detected": 0, "total_corrections": 2,
        "correction_avg_latency": 99.0,
        "warning_fp_rate": 0.0, "crisis_fp_rate": 0.0,
        "advisory_fp_rate": 0.0, "preaction_fp_rate": 0.0,
        "avg_peak_level": 0.0, "total_transitions": 0, "transitions_per_year": 0.0,
        "pct_normal": 0.0, "pct_watch": 0.0, "pct_warning": 0.0, "pct_crisis": 0.0,
        "pct_advisory_watch": 0.0, "pct_preaction_watch": 0.0,
        "avg_recovery_days": 30.0, "max_recovery_days": 60.0,
    }


# ---------------------------------------------------------------------------
# Monotonicity validation
# ---------------------------------------------------------------------------

def _check_monotonicity(all_muts: dict[str, Any]) -> str | None:
    """Pre-evaluation reject for threshold ordering violations."""
    def _val(key: str) -> float:
        if key in all_muts:
            return float(all_muts[key])
        return float(_defaults.get(key, 0.0))

    # Strict: WATCH < WARNING < CRISIS
    for triple in [
        ("VIX_WATCH", "VIX_WARNING", "VIX_CRISIS"),
        ("SPREAD_WATCH_BPS", "SPREAD_WARNING_BPS", "SPREAD_CRISIS_BPS"),
    ]:
        vals = [_val(k) for k in triple]
        for i in range(len(vals) - 1):
            if vals[i] >= vals[i + 1]:
                return (
                    f"monotonicity: {triple[i]}={vals[i]:.2f} "
                    f">= {triple[i+1]}={vals[i+1]:.2f}"
                )

    w = _val("SLOPE_STEEPEN_WARNING")
    c = _val("SLOPE_STEEPEN_CRISIS")
    if w >= c:
        return f"monotonicity: SLOPE_STEEPEN_WARNING={w:.2f} >= CRISIS={c:.2f}"

    # Non-strict for correlation (allows equal)
    for a, b in [("CORR_WATCH", "CORR_WARNING"), ("CORR_WARNING", "CORR_CRISIS")]:
        va, vb = _val(a), _val(b)
        if va > vb:
            return f"monotonicity: {a}={va:.2f} > {b}={vb:.2f}"

    # SPY DD: reversed ordering (less negative = watch, more negative = crisis)
    spy_w = _val("SPY_DD_WATCH")
    spy_wa = _val("SPY_DD_WARNING")
    spy_cr = _val("SPY_DD_CRISIS")
    if spy_w <= spy_wa:
        return f"monotonicity: SPY_DD_WATCH={spy_w:.4f} <= SPY_DD_WARNING={spy_wa:.4f}"
    if spy_wa <= spy_cr:
        return f"monotonicity: SPY_DD_WARNING={spy_wa:.4f} <= SPY_DD_CRISIS={spy_cr:.4f}"

    stress_min = int(_val("STRESS_FORMATION_MIN_SCORE"))
    if stress_min < 1 or stress_min > 4:
        return f"bounds: STRESS_FORMATION_MIN_SCORE={stress_min} outside [1, 4]"

    persist_days = int(_val("GRIND_VIX_PERSIST_DAYS"))
    if persist_days < 1 or persist_days > 20:
        return f"bounds: GRIND_VIX_PERSIST_DAYS={persist_days} outside [1, 20]"

    hard_bridge_days = int(_val("HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS"))
    if hard_bridge_days < 0 or hard_bridge_days > 20:
        return (
            "bounds: HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS="
            f"{hard_bridge_days} outside [0, 20]"
        )

    hard_bridge_min = int(_val("HARD_CREDIT_IMPULSE_WARNING_MIN_PRIMARY"))
    if hard_bridge_min < 1 or hard_bridge_min > 3:
        return (
            "bounds: HARD_CREDIT_IMPULSE_WARNING_MIN_PRIMARY="
            f"{hard_bridge_min} outside [1, 3]"
        )

    shock_3d = _val("SHOCK_SPY_3D_RETURN")
    shock_5d = _val("SHOCK_SPY_5D_RETURN")
    if shock_3d <= shock_5d:
        return (
            f"monotonicity: SHOCK_SPY_3D_RETURN={shock_3d:.4f} "
            f"<= SHOCK_SPY_5D_RETURN={shock_5d:.4f}"
        )

    grind_20d = _val("GRIND_SPY_20D_RETURN")
    grind_confirm = _val("GRIND_SPY_CONFIRM_20D_RETURN")
    if grind_confirm <= grind_20d:
        return (
            f"monotonicity: GRIND_SPY_CONFIRM_20D_RETURN={grind_confirm:.4f} "
            f"<= GRIND_SPY_20D_RETURN={grind_20d:.4f}"
        )

    credit_impulse_spy_3d = _val("CREDIT_IMPULSE_SPY_3D_RETURN")
    if credit_impulse_spy_3d >= 0:
        return (
            "bounds: CREDIT_IMPULSE_SPY_3D_RETURN="
            f"{credit_impulse_spy_3d:.4f} must be negative"
        )

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_candidate(args: tuple) -> ScoredCandidate:
    """Evaluate a candidate using vectorized fast path."""
    name, candidate_muts, base_muts, phase, scoring_weights, hard_rejects = args

    try:
        from backtests.regime.auto.crisis.scoring import CrisisMetrics, composite_score

        all_muts = dict(base_muts)
        all_muts.update(candidate_muts)

        violation = _check_monotonicity(all_muts)
        if violation:
            return ScoredCandidate(
                name=name, score=0.0, rejected=True, reject_reason=violation,
            )

        metrics_dict = _fast_evaluate(all_muts)
        _INT_FIELDS = {
            "crises_detected", "total_crises", "total_transitions",
            "corrections_detected", "total_corrections",
        }
        metrics = CrisisMetrics(**{
            k: (int(v) if k in _INT_FIELDS else float(v))
            for k, v in metrics_dict.items()
        })
        result = composite_score(metrics, scoring_weights, hard_rejects)

        if result.rejected:
            return ScoredCandidate(
                name=name, score=0.0, rejected=True,
                reject_reason=result.reject_reason, metrics=metrics_dict,
            )
        return ScoredCandidate(
            name=name, score=result.total, metrics=metrics_dict,
        )

    except Exception as exc:
        logger.error("Worker error for %s: %s", name, exc)
        return ScoredCandidate(
            name=name, score=0.0, rejected=True, reject_reason=f"error: {exc}",
        )
