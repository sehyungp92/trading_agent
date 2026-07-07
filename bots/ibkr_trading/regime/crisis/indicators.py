"""Crisis indicator computation from market data.

All indicators use absolute levels, NOT z-scores. This is the fundamental
design difference from the failed stress HMM approach.

Primary data sources (all already available in existing regime data):
- VIX: market_df["VIX"] (FRED VIXCLS)
- Credit Spread: market_df["SPREAD"] (FRED BAMLH0A0HYM2)
- Yield Curve: market_df["SLOPE_10Y2Y"] (FRED T10Y2Y)
- SPY/TLT returns: strat_ret_df columns or computed from IBKR bars
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime.crisis import config as C

logger = logging.getLogger(__name__)


@dataclass
class ChannelReading:
    """Single indicator channel reading with tiered alert level."""
    name: str
    value: float
    level: int          # 0=normal, 1=watch, 2=warning, 3=crisis
    detail: str = ""    # human-readable explanation


@dataclass
class CrisisIndicators:
    """Computed indicator readings for a single date."""
    vix: ChannelReading
    credit_spread: ChannelReading
    yield_curve: ChannelReading
    spy_tlt_corr: ChannelReading

    # Surfaced intermediate values for CrisisContext observability
    yield_curve_20d_change: float = 0.0
    spy_3d_return: float = 0.0
    spy_5d_return: float = 0.0
    spy_10d_return: float = 0.0
    spy_20d_return: float = 0.0
    vix_3d_change: float = 0.0
    credit_spread_20d_change_bps: float = 0.0
    stress_formation_score: int = 0
    stress_formation_mode: str = ""
    stress_formation_reason: str = ""

    # Confirming (VIX term structure only; SPY drawdown is now primary)
    vix_term_structure: ChannelReading | None = None
    spy_drawdown: ChannelReading | None = None

    @property
    def primary_channels(self) -> list[ChannelReading]:
        channels = [self.vix, self.credit_spread, self.yield_curve, self.spy_tlt_corr]
        if self.spy_drawdown is not None:
            channels.append(self.spy_drawdown)
        return channels

    @property
    def watch_count(self) -> int:
        return sum(1 for ch in self.primary_channels if ch.level >= 1)

    @property
    def warning_count(self) -> int:
        return sum(1 for ch in self.primary_channels if ch.level >= 2)

    @property
    def crisis_count(self) -> int:
        return sum(1 for ch in self.primary_channels if ch.level >= 3)

    @property
    def dominant_channel(self) -> str:
        """Return the name of the channel with the highest alert level."""
        best = max(self.primary_channels, key=lambda ch: (ch.level, abs(ch.value)))
        return best.name if best.level > 0 else ""


def _classify_vix(vix: float) -> ChannelReading:
    """Classify VIX level into alert tiers."""
    if np.isnan(vix):
        return ChannelReading("VIX", 0.0, 0, "VIX data unavailable")
    if vix >= C.VIX_CRISIS:
        return ChannelReading("VIX", vix, 3, f"VIX={vix:.1f} >= {C.VIX_CRISIS}")
    if vix >= C.VIX_WARNING:
        return ChannelReading("VIX", vix, 2, f"VIX={vix:.1f} >= {C.VIX_WARNING}")
    if vix >= C.VIX_WATCH:
        return ChannelReading("VIX", vix, 1, f"VIX={vix:.1f} >= {C.VIX_WATCH}")
    return ChannelReading("VIX", vix, 0, f"VIX={vix:.1f} < {C.VIX_WATCH}")


def _classify_credit_spread(spread_bps: float) -> ChannelReading:
    """Classify HY credit spread (basis points) into alert tiers."""
    if np.isnan(spread_bps):
        return ChannelReading("CREDIT_SPREAD", 0.0, 0, "Spread data unavailable")
    # FRED BAMLH0A0HYM2 is in percentage points (e.g. 4.5 = 450 bps)
    bps = spread_bps * 100.0 if spread_bps < 50 else spread_bps
    if bps >= C.SPREAD_CRISIS_BPS:
        return ChannelReading("CREDIT_SPREAD", bps, 3, f"Spread={bps:.0f}bps >= {C.SPREAD_CRISIS_BPS}")
    if bps >= C.SPREAD_WARNING_BPS:
        return ChannelReading("CREDIT_SPREAD", bps, 2, f"Spread={bps:.0f}bps >= {C.SPREAD_WARNING_BPS}")
    if bps >= C.SPREAD_WATCH_BPS:
        return ChannelReading("CREDIT_SPREAD", bps, 1, f"Spread={bps:.0f}bps >= {C.SPREAD_WATCH_BPS}")
    return ChannelReading("CREDIT_SPREAD", bps, 0, f"Spread={bps:.0f}bps < {C.SPREAD_WATCH_BPS}")


def _classify_yield_curve(
    slope: float,
    slope_20d_change: float,
    was_inverted_within_lookback: bool,
) -> ChannelReading:
    """Classify yield curve slope into alert tiers.

    Watch: persistent deep inversion.
    Warning/Crisis: rapid steepening after inversion (recession signal).
    """
    if np.isnan(slope):
        return ChannelReading("YIELD_CURVE", 0.0, 0, "Slope data unavailable")

    # Crisis: aggressive steepening after inversion
    if (was_inverted_within_lookback
            and slope_20d_change >= C.SLOPE_STEEPEN_CRISIS):
        return ChannelReading(
            "YIELD_CURVE", slope, 3,
            f"Rapid steepening: 20d change={slope_20d_change:+.2f}% "
            f"(>={C.SLOPE_STEEPEN_CRISIS}) after inversion",
        )

    # Warning: moderate steepening after inversion
    if (was_inverted_within_lookback
            and slope_20d_change >= C.SLOPE_STEEPEN_WARNING):
        return ChannelReading(
            "YIELD_CURVE", slope, 2,
            f"Un-inversion: 20d change={slope_20d_change:+.2f}% "
            f"(>={C.SLOPE_STEEPEN_WARNING}) after inversion",
        )

    # Watch: deep inversion
    if slope <= C.SLOPE_WATCH_THRESHOLD:
        return ChannelReading(
            "YIELD_CURVE", slope, 1,
            f"Deep inversion: slope={slope:+.2f}% <= {C.SLOPE_WATCH_THRESHOLD}",
        )

    return ChannelReading("YIELD_CURVE", slope, 0, f"Slope={slope:+.2f}% normal")


def _classify_spy_tlt_corr(
    corr: float,
    spy_10d_return: float,
) -> ChannelReading:
    """Classify SPY-TLT correlation into alert tiers.

    Positive correlation = stocks and bonds falling together.
    Crisis requires both high correlation AND SPY drawdown.
    """
    if np.isnan(corr):
        return ChannelReading("SPY_TLT_CORR", 0.0, 0, "Correlation data unavailable")

    # Crisis: high positive correlation + SPY selling off
    if corr >= C.CORR_CRISIS and spy_10d_return <= C.CORR_CRISIS_SPY_DD:
        return ChannelReading(
            "SPY_TLT_CORR", corr, 3,
            f"Corr={corr:.2f} >= {C.CORR_CRISIS} AND SPY 10d={spy_10d_return:.1%}",
        )

    # Warning: high positive correlation
    if corr >= C.CORR_WARNING:
        return ChannelReading(
            "SPY_TLT_CORR", corr, 2,
            f"Corr={corr:.2f} >= {C.CORR_WARNING}",
        )

    # Watch: moderate positive correlation
    if corr >= C.CORR_WATCH:
        return ChannelReading(
            "SPY_TLT_CORR", corr, 1,
            f"Corr={corr:.2f} >= {C.CORR_WATCH}",
        )

    return ChannelReading("SPY_TLT_CORR", corr, 0, f"Corr={corr:.2f} normal")


def _classify_vix_term_structure(ratio: float) -> ChannelReading:
    """VIX/VIX3M ratio -- confirming only."""
    if np.isnan(ratio) or ratio <= 0:
        return ChannelReading("VIX_TERM_STRUCTURE", 0.0, 0, "VIX3M unavailable")
    if ratio >= C.VIX_TERM_STRUCTURE_THRESHOLD:
        return ChannelReading(
            "VIX_TERM_STRUCTURE", ratio, 1,
            f"Backwardation: VIX/VIX3M={ratio:.2f} >= {C.VIX_TERM_STRUCTURE_THRESHOLD}",
        )
    return ChannelReading("VIX_TERM_STRUCTURE", ratio, 0, f"VIX/VIX3M={ratio:.2f} contango")


def _classify_spy_drawdown(dd: float) -> ChannelReading:
    """SPY 10d drawdown -- 5th primary channel."""
    if np.isnan(dd):
        return ChannelReading("SPY_DRAWDOWN", 0.0, 0, "SPY data unavailable")
    if dd <= C.SPY_DD_CRISIS:
        return ChannelReading("SPY_DRAWDOWN", dd, 3, f"SPY 10d DD={dd:.1%} <= {C.SPY_DD_CRISIS}")
    if dd <= C.SPY_DD_WARNING:
        return ChannelReading("SPY_DRAWDOWN", dd, 2, f"SPY 10d DD={dd:.1%} <= {C.SPY_DD_WARNING}")
    if dd <= C.SPY_DD_WATCH:
        return ChannelReading("SPY_DRAWDOWN", dd, 1, f"SPY 10d DD={dd:.1%} <= {C.SPY_DD_WATCH}")
    return ChannelReading("SPY_DRAWDOWN", dd, 0, f"SPY 10d DD={dd:.1%} normal")


def compute_indicators(
    market_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    date: pd.Timestamp | None = None,
    vix3m_series: pd.Series | None = None,
) -> CrisisIndicators:
    """Compute all crisis indicators for a single date.

    Args:
        market_df: Must have columns VIX, SPREAD, SLOPE_10Y2Y (from FRED).
        strat_ret_df: Must have SPY, TLT columns (daily returns).
        date: Target date. If None, uses latest available date.
        vix3m_series: Optional VIX3M series for term structure (confirming).

    Returns:
        CrisisIndicators with all channel readings.
    """
    if date is None:
        date = market_df.index[-1]

    # Align to the target date (use latest available <= date)
    mkt = market_df.loc[:date]
    ret = strat_ret_df.loc[:date]

    if mkt.empty:
        return _empty_indicators()

    # ── VIX ───────────────────────────────────────────────────────────
    vix_val = _safe_last(mkt, "VIX")
    vix_3d_change = 0.0
    vix_persistent = False
    if "VIX" in mkt.columns:
        vix_series = mkt["VIX"].dropna()
        if len(vix_series) >= 4:
            vix_3d_change = float(vix_series.iloc[-1] - vix_series.iloc[-4])
        if len(vix_series) >= C.GRIND_VIX_PERSIST_DAYS:
            vix_persistent = bool(
                (vix_series.iloc[-C.GRIND_VIX_PERSIST_DAYS:] >= C.GRIND_VIX_MIN).all()
            )
    vix_ch = _classify_vix(vix_val)

    # ── Credit Spread ─────────────────────────────────────────────────
    spread_val = _safe_last(mkt, "SPREAD")
    spread_20d_change_bps = 0.0
    if "SPREAD" in mkt.columns:
        spread_series = mkt["SPREAD"].dropna()
        if len(spread_series) >= 21:
            spread_now = _spread_to_bps(float(spread_series.iloc[-1]))
            spread_then = _spread_to_bps(float(spread_series.iloc[-21]))
            spread_20d_change_bps = spread_now - spread_then
    spread_ch = _classify_credit_spread(spread_val)

    # ── Yield Curve ───────────────────────────────────────────────────
    slope_val = _safe_last(mkt, "SLOPE_10Y2Y")
    # 20-day change in slope
    slope_20d_change = 0.0
    if "SLOPE_10Y2Y" in mkt.columns and len(mkt) >= 20:
        slope_series = mkt["SLOPE_10Y2Y"].dropna()
        if len(slope_series) >= 20:
            slope_20d_change = float(slope_series.iloc[-1] - slope_series.iloc[-20])

    # Check if curve was inverted within lookback window
    was_inverted = False
    if "SLOPE_10Y2Y" in mkt.columns:
        lookback_start = max(0, len(mkt) - C.SLOPE_INVERSION_LOOKBACK)
        slope_window = mkt["SLOPE_10Y2Y"].iloc[lookback_start:].dropna()
        was_inverted = bool((slope_window < 0).any())

    slope_ch = _classify_yield_curve(slope_val, slope_20d_change, was_inverted)

    # ── SPY 10d Return (shared by correlation and SPY DD channel) ──────
    spy_3d_return = np.nan
    spy_5d_return = np.nan
    spy_10d_return = np.nan
    spy_20d_return = np.nan
    if not ret.empty and "SPY" in ret.columns:
        spy_ret = ret["SPY"].dropna()
        spy_3d_return = _compound_recent_return(spy_ret, 3)
        spy_5d_return = _compound_recent_return(spy_ret, 5)
        spy_10d_return = _compound_recent_return(spy_ret, 10)
        spy_20d_return = _compound_recent_return(spy_ret, 20)

    # ── SPY-TLT Correlation ───────────────────────────────────────────
    spy_tlt_corr = np.nan
    if not ret.empty and "SPY" in ret.columns and "TLT" in ret.columns:
        spy_ret_c = ret["SPY"].dropna()
        tlt_ret = ret["TLT"].dropna()
        common = spy_ret_c.index.intersection(tlt_ret.index)
        if len(common) >= C.CORR_WINDOW:
            spy_aligned = spy_ret_c.loc[common].iloc[-C.CORR_WINDOW:]
            tlt_aligned = tlt_ret.loc[common].iloc[-C.CORR_WINDOW:]
            if spy_aligned.std() > 1e-10 and tlt_aligned.std() > 1e-10:
                spy_tlt_corr = float(spy_aligned.corr(tlt_aligned))

    corr_ch = _classify_spy_tlt_corr(spy_tlt_corr, spy_10d_return)

    # ── Confirming: VIX Term Structure ────────────────────────────────
    vix_ts_ch = None
    if vix3m_series is not None and not vix3m_series.empty:
        vix3m_val = _safe_last_series(vix3m_series, date)
        ratio = vix_val / vix3m_val if vix3m_val > 0 and not np.isnan(vix_val) else np.nan
        vix_ts_ch = _classify_vix_term_structure(ratio)

    # ── SPY Drawdown (5th primary channel) ─────────────────────────────
    spy_dd_ch = _classify_spy_drawdown(spy_10d_return)
    stress_score, stress_mode, stress_reason = _compute_stress_formation(
        spy_3d_return=spy_3d_return,
        spy_5d_return=spy_5d_return,
        spy_20d_return=spy_20d_return,
        vix_level=vix_val,
        vix_3d_change=vix_3d_change,
        vix_persistent=vix_persistent,
        spread_bps=spread_ch.value,
        spread_20d_change_bps=spread_20d_change_bps,
        spy_tlt_corr=spy_tlt_corr,
    )

    return CrisisIndicators(
        vix=vix_ch,
        credit_spread=spread_ch,
        yield_curve=slope_ch,
        spy_tlt_corr=corr_ch,
        yield_curve_20d_change=slope_20d_change,
        spy_3d_return=float(spy_3d_return) if not np.isnan(spy_3d_return) else 0.0,
        spy_5d_return=float(spy_5d_return) if not np.isnan(spy_5d_return) else 0.0,
        spy_10d_return=float(spy_10d_return) if not np.isnan(spy_10d_return) else 0.0,
        spy_20d_return=float(spy_20d_return) if not np.isnan(spy_20d_return) else 0.0,
        vix_3d_change=vix_3d_change,
        credit_spread_20d_change_bps=spread_20d_change_bps,
        stress_formation_score=stress_score,
        stress_formation_mode=stress_mode,
        stress_formation_reason=stress_reason,
        vix_term_structure=vix_ts_ch,
        spy_drawdown=spy_dd_ch,
    )


def _spread_to_bps(spread: float) -> float:
    """Convert HY spread to basis points when source is percentage points."""
    if np.isnan(spread):
        return 0.0
    return spread * 100.0 if spread < 50 else spread


def _compound_recent_return(series: pd.Series, window: int) -> float:
    """Compound the most recent ``window`` simple returns."""
    if len(series) < window:
        return np.nan
    return float((1.0 + series.iloc[-window:]).prod() - 1.0)


def _compute_stress_formation(
    *,
    spy_3d_return: float,
    spy_5d_return: float,
    spy_20d_return: float,
    vix_level: float,
    vix_3d_change: float,
    vix_persistent: bool,
    spread_bps: float,
    spread_20d_change_bps: float,
    spy_tlt_corr: float,
) -> tuple[int, str, str]:
    """Score early shock/grind stress formation without forcing action."""
    shock_reasons: list[str] = []
    grind_reasons: list[str] = []
    credit_impulse_reasons: list[str] = []

    if not np.isnan(spy_3d_return) and spy_3d_return <= C.SHOCK_SPY_3D_RETURN:
        shock_reasons.append(f"SPY 3d={spy_3d_return:.1%}")
    if not np.isnan(spy_5d_return) and spy_5d_return <= C.SHOCK_SPY_5D_RETURN:
        shock_reasons.append(f"SPY 5d={spy_5d_return:.1%}")
    if (
        not np.isnan(vix_level)
        and vix_level >= C.SHOCK_MIN_VIX
        and vix_3d_change >= C.SHOCK_VIX_3D_CHANGE
    ):
        shock_reasons.append(f"VIX 3d change={vix_3d_change:+.1f}")
    if (
        not np.isnan(spy_tlt_corr)
        and not np.isnan(spy_5d_return)
        and spy_tlt_corr >= C.SHOCK_CORR_MIN
        and spy_5d_return <= C.SHOCK_CORR_SPY_5D_RETURN
    ):
        shock_reasons.append(f"SPY/TLT corr={spy_tlt_corr:.2f} with SPY 5d={spy_5d_return:.1%}")

    if spread_20d_change_bps >= C.GRIND_SPREAD_20D_CHANGE_BPS:
        grind_reasons.append(f"spread 20d change={spread_20d_change_bps:+.0f}bps")
    if not np.isnan(spy_20d_return) and spy_20d_return <= C.GRIND_SPY_20D_RETURN:
        grind_reasons.append(f"SPY 20d={spy_20d_return:.1%}")
    if vix_persistent:
        grind_reasons.append(f"VIX >= {C.GRIND_VIX_MIN:.0f} for {C.GRIND_VIX_PERSIST_DAYS}d")
    if (
        spread_bps >= C.GRIND_SPREAD_CONFIRM_BPS
        and not np.isnan(spy_20d_return)
        and spy_20d_return <= C.GRIND_SPY_CONFIRM_20D_RETURN
    ):
        grind_reasons.append(f"spread={spread_bps:.0f}bps with SPY 20d={spy_20d_return:.1%}")

    if (
        spread_bps >= C.CREDIT_IMPULSE_SPREAD_BPS
        and not np.isnan(spy_3d_return)
        and spy_3d_return <= C.CREDIT_IMPULSE_SPY_3D_RETURN
        and not np.isnan(vix_level)
        and vix_level >= C.CREDIT_IMPULSE_MIN_VIX
    ):
        credit_impulse_reasons.append(
            f"credit impulse: spread={spread_bps:.0f}bps, "
            f"SPY 3d={spy_3d_return:.1%}, VIX={vix_level:.1f}"
        )

    shock_score = len(shock_reasons)
    grind_score = len(grind_reasons)
    credit_impulse_score = 2 if credit_impulse_reasons else 0
    score = max(shock_score, grind_score, credit_impulse_score)
    if score < C.STRESS_FORMATION_MIN_SCORE:
        return score, "", ""

    modes: list[str] = []
    reasons: list[str] = []
    if shock_score == score and shock_score >= C.STRESS_FORMATION_MIN_SCORE:
        modes.append("shock")
        reasons.extend(shock_reasons)
    if grind_score == score and grind_score >= C.STRESS_FORMATION_MIN_SCORE:
        modes.append("grind")
        reasons.extend(grind_reasons)
    if credit_impulse_score == score:
        modes.append("credit_impulse")
        reasons.extend(credit_impulse_reasons)
    return score, "+".join(modes), "; ".join(reasons)


def _safe_last(df: pd.DataFrame, col: str) -> float:
    """Get last non-NaN value from a column, or NaN if unavailable."""
    if col not in df.columns:
        return np.nan
    s = df[col].dropna()
    return float(s.iloc[-1]) if len(s) > 0 else np.nan


def _safe_last_series(s: pd.Series, date: pd.Timestamp) -> float:
    """Get last value from series up to date."""
    subset = s.loc[:date].dropna()
    return float(subset.iloc[-1]) if len(subset) > 0 else np.nan


def _empty_indicators() -> CrisisIndicators:
    """Return all-normal indicators when data is unavailable."""
    return CrisisIndicators(
        vix=ChannelReading("VIX", 0.0, 0, "No data"),
        credit_spread=ChannelReading("CREDIT_SPREAD", 0.0, 0, "No data"),
        yield_curve=ChannelReading("YIELD_CURVE", 0.0, 0, "No data"),
        spy_tlt_corr=ChannelReading("SPY_TLT_CORR", 0.0, 0, "No data"),
        spy_drawdown=ChannelReading("SPY_DRAWDOWN", 0.0, 0, "No data"),
    )
