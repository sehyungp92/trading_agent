"""ATRSS composite scoring -- 7 active components, immutable across phases.

Two scoring profiles:
  - r1_independent: Original scales calibrated for independent-account mode.
  - r9_synchronized: Rescaled for honest synchronized/fee-net conditions.

Active components (sum to 1.0):
  - return_quality (0.38): blended total_r and net_return_pct
  - profit_factor (0.12)
  - risk_efficiency (0.16): blended calmar_r and inverse drawdown
  - frequency (0.18)
  - mfe_capture (0.08)
  - avg_r (0.06)
  - win_rate (0.02)

Hard rejects are phase-specific (see phase_scoring.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Weights -- immutable across all phases and profiles
# ---------------------------------------------------------------------------
W_RETURN = 0.38
W_PF = 0.12
W_RISK = 0.16
W_FREQUENCY = 0.18
W_MFE_CAPTURE = 0.08
W_AVG_R = 0.06
W_WIN_RATE = 0.02

# Within-component blends preserve the old return/risk balance without
# exposing more optimizer objectives than the seven active components.
RETURN_NET_R_BLEND = 0.24 / W_RETURN
RETURN_NET_RETURN_BLEND = 0.14 / W_RETURN
RISK_CALMAR_BLEND = 0.10 / W_RISK
RISK_INV_DD_BLEND = 0.06 / W_RISK


# ---------------------------------------------------------------------------
# Scoring profiles -- component normalization scales
# ---------------------------------------------------------------------------
# Each profile maps component name -> (divisor_or_params) used in normalization.
# Format: {component: (numerator_offset, denominator, denominator_offset)}
#   normalized = (raw - numerator_offset) / denominator
# For special components (net_r, inv_dd), parameters are handled directly.

SCORING_PROFILES: dict[str, dict] = {
    "r1_independent": {
        # net_r: log(1 + R/100) / log(5)  -- 400R -> 1.0
        "net_r_log_divisor": 100.0,
        "net_r_log_base": 5.0,
        # profit_factor: (PF - 1) / 8  -- PF=9 -> 1.0
        "pf_offset": 1.0,
        "pf_divisor": 8.0,
        # calmar_r: calmar / 80
        "calmar_r_divisor": 80.0,
        # inv_dd: (1/dd - 10) / 90  -- dd=1.1% -> 1.0
        "inv_dd_offset": 10.0,
        "inv_dd_divisor": 90.0,
        # frequency: tpm / 6
        "freq_divisor": 6.0,
        # mfe_capture: capture / 0.90
        "capture_divisor": 0.90,
        # win_rate: wr / 0.85
        "wr_divisor": 0.85,
    },
    "r9_synchronized": {
        # Calibrated from Phase 0 triage: vanilla both_vanilla scores ~0.50.
        # Actuals: total_r=190.8, PF=4.47, calmar_r=40.9, DD=2.07%,
        #          TPM=5.0, MFE_capture=0.654, WR=74.1%
        # net_r: log(1 + R/150) / log(5)  -- vanilla ~0.51
        "net_r_log_divisor": 150.0,
        "net_r_log_base": 5.0,
        # profit_factor: (PF - 1) / 7  -- vanilla ~0.50
        "pf_offset": 1.0,
        "pf_divisor": 7.0,
        # calmar_r: calmar / 80  -- vanilla ~0.51
        "calmar_r_divisor": 80.0,
        # inv_dd: (1/dd - 5) / 85  -- vanilla ~0.51
        "inv_dd_offset": 5.0,
        "inv_dd_divisor": 85.0,
        # frequency: tpm / 10  -- vanilla ~0.50
        "freq_divisor": 7.5,
        # mfe_capture: capture / 1.30  -- vanilla ~0.50
        "capture_divisor": 0.80,
        # win_rate: WR is a guardrail more than an optimizer target.
        "wr_divisor": 0.85,
        # fee-net return and expectancy targets for R2 alpha expansion.
        "net_return_divisor": 40.0,
        "avg_r_divisor": 1.20,
    },
    "r11_risk_allocation": {
        # Round 3 profile: same honest synchronized metrics, with the return
        # scale widened enough for dynamic risk candidates above the fixed-size
        # reference to remain distinguishable while DD stays a hard guardrail.
        "net_r_log_divisor": 150.0,
        "net_r_log_base": 5.0,
        "pf_offset": 1.0,
        "pf_divisor": 7.0,
        "calmar_r_divisor": 90.0,
        "inv_dd_offset": 5.0,
        "inv_dd_divisor": 70.0,
        "freq_divisor": 7.5,
        "capture_divisor": 0.80,
        "wr_divisor": 0.85,
        "net_return_divisor": 180.0,
        "avg_r_divisor": 1.20,
        "weights": {
            "return": 0.48,
            "pf": 0.12,
            "risk": 0.22,
            "frequency": 0.04,
            "mfe_capture": 0.04,
            "avg_r": 0.06,
            "win_rate": 0.04,
        },
    },
}


# Profile-aware default hard rejects -- used when caller doesn't pass explicit rejects.
_DEFAULT_HARD_REJECTS: dict[str, dict[str, float]] = {
    "r1_independent": {"min_trades": 100, "max_dd_pct": 0.07, "min_pf": 2.0, "min_wr": 0.55},
    "r9_synchronized": {"min_trades": 220, "max_dd_pct": 0.06, "min_pf": 1.8, "min_wr": 0.58},
    "r11_risk_allocation": {"min_trades": 240, "max_dd_pct": 0.08, "min_pf": 4.0, "min_wr": 0.75},
}


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _span_seconds(timestamps) -> float:
    if timestamps is None or len(timestamps) < 2:
        return 0.0
    delta = timestamps[-1] - timestamps[0]
    if hasattr(delta, "total_seconds"):
        return float(delta.total_seconds())
    if isinstance(delta, np.timedelta64):
        return float(delta / np.timedelta64(1, "s"))
    if isinstance(delta, (int, float, np.integer, np.floating)):
        # Some portfolio runners store epoch-second offsets as numeric arrays.
        return float(delta)
    return 0.0


@dataclass(frozen=True)
class ATRSSMetrics:
    """Metrics extracted from an ATRSS backtest run."""
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    calmar: float = 0.0
    sharpe: float = 0.0
    net_return_pct: float = 0.0
    total_r: float = 0.0
    max_dd_r: float = 0.0
    calmar_r: float = 0.0
    avg_r: float = 0.0
    mfe_capture: float = 0.0
    trades_per_month: float = 0.0


@dataclass(frozen=True)
class ATRSSCompositeScore:
    """Frozen composite score for ATRSS optimization."""
    return_component: float = 0.0
    pf_component: float = 0.0
    risk_component: float = 0.0
    frequency_component: float = 0.0
    mfe_capture_component: float = 0.0
    avg_r_component: float = 0.0
    win_rate_component: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""

    @property
    def net_r_component(self) -> float:
        """Compatibility alias for older reports."""
        return self.return_component

    @property
    def net_return_component(self) -> float:
        """Compatibility alias for older reports."""
        return self.return_component

    @property
    def calmar_r_component(self) -> float:
        """Compatibility alias for older reports."""
        return self.risk_component

    @property
    def inv_dd_component(self) -> float:
        """Compatibility alias for older reports."""
        return self.risk_component


def composite_score(
    metrics: ATRSSMetrics,
    weights: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
    profile: str = "r1_independent",
) -> ATRSSCompositeScore:
    """Compute the ATRSS composite score.

    Args:
        metrics: ATRSS metrics from backtests.swing.
        weights: Optional weight overrides (unused -- ATRSS uses fixed weights).
        hard_rejects: Phase-specific hard reject thresholds.
        profile: Scoring profile name ("r1_independent" or "r9_synchronized").

    Returns:
        ATRSSCompositeScore with component values and total.
    """
    rejects = hard_rejects or {}
    defaults = _DEFAULT_HARD_REJECTS.get(profile, _DEFAULT_HARD_REJECTS["r1_independent"])
    min_trades = int(rejects.get("min_trades", defaults["min_trades"]))
    max_dd = rejects.get("max_dd_pct", defaults["max_dd_pct"])
    min_pf = rejects.get("min_pf", defaults["min_pf"])
    min_wr = rejects.get("min_wr", defaults["min_wr"])

    if metrics.total_trades < min_trades:
        return ATRSSCompositeScore(
            rejected=True,
            reject_reason=f"Too few trades: {metrics.total_trades} < {min_trades}",
        )
    if metrics.max_dd_pct > max_dd:
        return ATRSSCompositeScore(
            rejected=True,
            reject_reason=f"Max DD too high: {metrics.max_dd_pct:.1%} > {max_dd:.1%}",
        )
    if metrics.profit_factor < min_pf:
        return ATRSSCompositeScore(
            rejected=True,
            reject_reason=f"PF too low: {metrics.profit_factor:.2f} < {min_pf:.2f}",
        )
    if metrics.win_rate < min_wr:
        return ATRSSCompositeScore(
            rejected=True,
            reject_reason=f"WR too low: {metrics.win_rate:.1%} < {min_wr:.0%}",
        )

    # --- Component normalization (all clipped to [0, 1]) ---
    p = SCORING_PROFILES.get(profile, SCORING_PROFILES["r1_independent"])

    # Weights are immutable within a scoring profile. R11 intentionally has a
    # separate fixed profile to evaluate capital deployment without changing
    # the R9 alpha-discovery score.
    profile_weights = p.get("weights", {})
    w_return = float(profile_weights.get("return", W_RETURN))
    w_pf = float(profile_weights.get("pf", W_PF))
    w_risk = float(profile_weights.get("risk", W_RISK))
    w_freq = float(profile_weights.get("frequency", W_FREQUENCY))
    w_capture = float(profile_weights.get("mfe_capture", W_MFE_CAPTURE))
    w_avg_r = float(profile_weights.get("avg_r", W_AVG_R))
    w_wr = float(profile_weights.get("win_rate", W_WIN_RATE))

    return_net_r_blend = float(profile_weights.get("return_net_r_blend", RETURN_NET_R_BLEND))
    return_net_return_blend = float(profile_weights.get("return_net_return_blend", RETURN_NET_RETURN_BLEND))
    risk_calmar_blend = float(profile_weights.get("risk_calmar_blend", RISK_CALMAR_BLEND))
    risk_inv_dd_blend = float(profile_weights.get("risk_inv_dd_blend", RISK_INV_DD_BLEND))

    # net_r: log(1 + total_r / log_divisor) / log(log_base)
    net_r_raw = _clip01(
        math.log(1.0 + max(metrics.total_r, 0.0) / p["net_r_log_divisor"])
        / math.log(p["net_r_log_base"])
    )

    # profit_factor: (pf - offset) / divisor
    pf_raw = _clip01((metrics.profit_factor - p["pf_offset"]) / p["pf_divisor"])

    # fee-net return: net return % / divisor
    net_return_raw = _clip01(max(metrics.net_return_pct, 0.0) / p.get("net_return_divisor", 40.0))

    # calmar_r: calmar_r / divisor
    calmar_r_raw = _clip01(metrics.calmar_r / p["calmar_r_divisor"])

    # inv_dd: (1/dd - offset) / divisor
    dd_frac = max(metrics.max_dd_pct, 0.001)
    inv_dd_raw = _clip01(((1.0 / dd_frac) - p["inv_dd_offset"]) / p["inv_dd_divisor"])

    # frequency: tpm / divisor
    freq_raw = _clip01(metrics.trades_per_month / p["freq_divisor"])

    # mfe_capture: capture / divisor
    capture_raw = _clip01(metrics.mfe_capture / p["capture_divisor"])

    # avg_r: expectancy / divisor
    avg_r_raw = _clip01(max(metrics.avg_r, 0.0) / p.get("avg_r_divisor", 1.20))

    # win_rate: wr / divisor
    wr_raw = _clip01(metrics.win_rate / p["wr_divisor"])

    return_blend_total = max(return_net_r_blend + return_net_return_blend, 1e-9)
    return_raw = _clip01(
        (return_net_r_blend * net_r_raw + return_net_return_blend * net_return_raw)
        / return_blend_total
    )
    risk_blend_total = max(risk_calmar_blend + risk_inv_dd_blend, 1e-9)
    risk_raw = _clip01(
        (risk_calmar_blend * calmar_r_raw + risk_inv_dd_blend * inv_dd_raw)
        / risk_blend_total
    )

    total = (
        w_return * return_raw
        + w_pf * pf_raw
        + w_risk * risk_raw
        + w_freq * freq_raw
        + w_capture * capture_raw
        + w_avg_r * avg_r_raw
        + w_wr * wr_raw
    )

    return ATRSSCompositeScore(
        return_component=return_raw,
        pf_component=pf_raw,
        risk_component=risk_raw,
        frequency_component=freq_raw,
        mfe_capture_component=capture_raw,
        avg_r_component=avg_r_raw,
        win_rate_component=wr_raw,
        total=total,
    )


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

def extract_atrss_metrics(
    result,  # PortfolioResult
    initial_equity: float,
) -> ATRSSMetrics:
    """Extract ATRSSMetrics from a PortfolioResult.

    Uses TradeRecord fields directly (r_multiple, mfe_r, pnl_dollars)
    and PortfolioResult.combined_equity for equity-based metrics.
    """
    all_trades = []
    for sr in result.symbol_results.values():
        all_trades.extend(sr.trades)
    all_trades.sort(key=lambda t: t.entry_time)

    n_trades = len(all_trades)
    if n_trades == 0:
        return ATRSSMetrics()

    # R-based metrics from TradeRecord.r_multiple
    r_mults = np.array([t.r_multiple for t in all_trades])
    wins_mask = r_mults > 0
    n_wins = int(np.sum(wins_mask))
    win_rate = n_wins / n_trades
    gross_win_r = float(np.sum(r_mults[wins_mask])) if n_wins > 0 else 0.0
    gross_loss_r = abs(float(np.sum(r_mults[~wins_mask])))
    profit_factor = gross_win_r / gross_loss_r if gross_loss_r > 0 else 999.0
    total_r = float(np.sum(r_mults))
    avg_r = float(np.mean(r_mults))

    # Max DD in R
    cum_r = np.cumsum(r_mults)
    peak_r = np.maximum.accumulate(cum_r)
    dd_r = peak_r - cum_r
    max_dd_r = float(np.max(dd_r)) if len(dd_r) > 0 else 0.0
    calmar_r = total_r / max_dd_r if max_dd_r > 0 else 0.0

    # Equity-based metrics from combined_equity
    eq = result.combined_equity
    if len(eq) > 1:
        peak_eq = np.maximum.accumulate(eq)
        dd_pct = (peak_eq - eq) / np.maximum(peak_eq, 1e-9)
        max_dd_pct = float(np.max(dd_pct))
        net_return_pct = (eq[-1] - initial_equity) / initial_equity * 100

        # Sharpe: annualized from bar-level returns
        returns = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 7))
        else:
            sharpe = 0.0

        # CAGR-based calmar
        ts = result.combined_timestamps
        if len(ts) > 1:
            span_secs = _span_seconds(ts)
            span_years = span_secs / (365.25 * 86400)
            cagr = ((eq[-1] / initial_equity) ** (1 / max(span_years, 0.01)) - 1) if span_years > 0 else 0.0
            calmar = cagr / max_dd_pct if max_dd_pct > 0 else 0.0
        else:
            calmar = 0.0
    else:
        max_dd_pct = 0.0
        net_return_pct = 0.0
        sharpe = 0.0
        calmar = 0.0

    # MFE capture: avg(r_multiple / mfe_r) for winning trades with mfe > 0
    captures = []
    for t in all_trades:
        if t.mfe_r > 0 and t.r_multiple > 0:
            captures.append(t.r_multiple / t.mfe_r)
    mfe_capture = float(np.mean(captures)) if captures else 0.0

    # Trades per month
    if n_trades >= 2:
        span_days = (all_trades[-1].entry_time - all_trades[0].entry_time).total_seconds() / 86400
        tpm = n_trades / max(span_days / 30.44, 1.0)
    else:
        tpm = 0.0

    return ATRSSMetrics(
        total_trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_dd_pct=max_dd_pct,
        calmar=calmar,
        sharpe=sharpe,
        net_return_pct=net_return_pct,
        total_r=total_r,
        max_dd_r=max_dd_r,
        calmar_r=calmar_r,
        avg_r=avg_r,
        mfe_capture=mfe_capture,
        trades_per_month=tpm,
    )
