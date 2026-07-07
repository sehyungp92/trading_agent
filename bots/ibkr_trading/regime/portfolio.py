"""Sections 4-6: Risk budgets, correlation-adjusted weights, ventilator."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MetaConfig
from .utils import budget, clip_and_renorm


# ---------------------------------------------------------------------------
# Section 4: Default regime budgets
# ---------------------------------------------------------------------------

def default_regime_budgets(
    cols,
    cfg: "MetaConfig | None" = None,
) -> Tuple[dict, pd.Series]:
    if cfg is not None:
        regime_budgets = {
            "G": budget(cols, SPY=cfg.budget_G_spy, EFA=cfg.budget_G_efa,
                        TLT=cfg.budget_G_tlt, GLD=cfg.budget_G_gld, CASH=cfg.budget_G_cash),
            "R": budget(cols, SPY=cfg.budget_R_spy, EFA=cfg.budget_R_efa, GLD=cfg.budget_R_gld,
                        TLT=cfg.budget_R_tlt, CASH=cfg.budget_R_cash),
            "S": budget(cols, GLD=cfg.budget_S_gld, CASH=cfg.budget_S_cash, SPY=cfg.budget_S_spy,
                        EFA=cfg.budget_S_efa, TLT=cfg.budget_S_tlt),
            "D": budget(cols, TLT=cfg.budget_D_tlt, CASH=cfg.budget_D_cash, GLD=cfg.budget_D_gld,
                        SPY=cfg.budget_D_spy, EFA=cfg.budget_D_efa),
        }
        w_neutral = budget(cols, SPY=cfg.budget_neutral_spy, EFA=cfg.budget_neutral_efa,
                           TLT=cfg.budget_neutral_tlt, GLD=cfg.budget_neutral_gld,
                           CASH=cfg.budget_neutral_cash)
    else:
        regime_budgets = {
            "G": budget(cols, SPY=0.40, EFA=0.10, TLT=0.05, GLD=0.05, CASH=0.40),
            "R": budget(cols, SPY=0.35, EFA=0.15, GLD=0.30, TLT=0.00, CASH=0.20),
            "S": budget(cols, GLD=0.50, CASH=0.30, SPY=0.10, EFA=0.05, TLT=0.05),
            "D": budget(cols, TLT=0.50, CASH=0.30, GLD=0.10, SPY=0.05, EFA=0.05),
        }
        w_neutral = budget(cols, SPY=0.20, EFA=0.10, TLT=0.25, GLD=0.25, CASH=0.20)
    return regime_budgets, w_neutral


# ---------------------------------------------------------------------------
# Section 5: Shrunk covariance + risk budgeting
# ---------------------------------------------------------------------------

def estimate_shrunk_covariance(
    ret_df: pd.DataFrame, cfg: MetaConfig
) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage on trailing window of daily returns.
    Returns annualised covariance matrix."""
    from sklearn.covariance import LedoitWolf

    recent = ret_df.iloc[-cfg.cov_window :].dropna(axis=1, how="any")
    lw = LedoitWolf().fit(recent.values)
    cov_daily = pd.DataFrame(
        lw.covariance_, index=recent.columns, columns=recent.columns
    )
    return cov_daily * cfg.ann_factor


def weights_from_risk_budget(
    b: pd.Series,
    cov_annual: pd.DataFrame,
    cfg: MetaConfig,
) -> pd.Series:
    """Solve for w such that each asset's risk contribution ~ b_i x total_risk.
    Uses iterative proportional fitting on marginal contributions.
    Falls back to diagonal if covariance is degenerate."""
    sleeves = b.index.tolist()
    cov = cov_annual.reindex(index=sleeves, columns=sleeves).fillna(0.0).values.copy()
    bv = b.values.copy()

    # Ensure sigma floor on diagonal
    floor_var = cfg.sigma_floor_annual**2
    for i in range(len(sleeves)):
        cov[i, i] = max(cov[i, i], floor_var)

    # Initialise from diagonal approximation
    diag_vol = np.sqrt(np.diag(cov)).clip(min=cfg.sigma_floor_annual)
    w = bv / diag_vol
    w = np.clip(w, 0, None)
    s = w.sum()
    if s < 1e-12:
        w = np.ones(len(sleeves)) / len(sleeves)
    else:
        w /= s

    # Iterative solve
    for _ in range(80):
        mc = cov @ w  # marginal risk contribution (unnormalised)
        mc = np.clip(mc, 1e-10, None)
        w_new = bv / mc
        w_new = np.clip(w_new, 0, None)
        s = w_new.sum()
        if s < 1e-12:
            break
        w_new /= s
        if np.max(np.abs(w_new - w)) < 1e-7:
            w = w_new
            break
        w = w_new

    result = pd.Series(w, index=sleeves)
    return clip_and_renorm(result, wmax=cfg.per_strat_max)


# ---------------------------------------------------------------------------
# Section 5 (cont): Blending and confidence fallback
# ---------------------------------------------------------------------------

def blend_policy_portfolios(
    P_grsd: np.ndarray,
    wG: pd.Series,
    wR: pd.Series,
    wS: pd.Series,
    wD: pd.Series,
) -> pd.Series:
    return P_grsd[0] * wG + P_grsd[1] * wR + P_grsd[2] * wS + P_grsd[3] * wD


def confidence_fallback(
    w_active: pd.Series, w_neutral: pd.Series, conf: float
) -> pd.Series:
    w = conf * w_active + (1.0 - conf) * w_neutral
    w = w.clip(lower=0.0)
    return w / w.sum()


def smooth_weights(
    w_new: pd.Series,
    w_prev: Optional[pd.Series],
    alpha: float = 1.0,
) -> pd.Series:
    """EMA blend for allocation signal stability.
    alpha=1.0 = no smoothing (backward-compatible).
    alpha=0.3 = 30% new, 70% previous (strong smoothing).
    """
    if w_prev is None or alpha >= 1.0:
        return w_new
    w = alpha * w_new + (1.0 - alpha) * w_prev
    w = w.clip(lower=0)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Section 6: Anticipatory correlation ventilator
# ---------------------------------------------------------------------------

def apply_ventilator(
    w: pd.Series,
    p_crisis: float,
    hist_ret: pd.DataFrame,
    cfg: MetaConfig,
    exempt_state: Optional[Dict[str, int]] = None,
    spy_col: str = "SPY",
) -> Tuple[pd.Series, Dict[str, int]]:
    if exempt_state is None:
        exempt_state = {}

    v = float(np.clip(1.0 - cfg.ventilator_lambda * p_crisis, cfg.ventilator_vmin, 1.0))

    need_long = cfg.rho_long_window + 5
    if len(hist_ret) < need_long or spy_col not in hist_ret.columns:
        # Insufficient data — ventilate all risk-on uniformly
        for s in cfg.risk_on_set:
            if s in w.index and s != cfg.cash_col:
                w[s] *= v
        w[cfg.cash_col] = max(0.0, 1.0 - w.drop(cfg.cash_col).sum())
        return w, exempt_state

    # Short-window and long-window correlations to SPY
    rho_short = hist_ret.iloc[-cfg.rho_short_window :].corr()[spy_col]
    rho_long = hist_ret.iloc[-cfg.rho_long_window :].corr()[spy_col]

    for s in cfg.risk_on_set:
        if s not in w.index or s == cfg.cash_col:
            continue

        rho_s = rho_short.get(s, np.nan)
        rho_l = rho_long.get(s, np.nan)

        if pd.isna(rho_s) or pd.isna(rho_l):
            w[s] *= v
            exempt_state[s] = 0
            continue

        delta_rho = rho_s - rho_l  # positive = becoming more correlated

        # Optional P&L confirmation for exemption
        pnl_ok = True
        if cfg.pnl_confirm_days > 0 and s in hist_ret.columns:
            n = min(cfg.pnl_confirm_days, len(hist_ret))
            pnl = (1.0 + hist_ret[s].iloc[-n:]).prod() - 1.0
            pnl_ok = pnl >= 0.0

        prev_exempt = int(exempt_state.get(s, 0))

        # -- Anticipatory logic --
        if delta_rho >= cfg.delta_rho_threshold:
            new_exempt = 0
        elif delta_rho <= cfg.delta_rho_exempt and pnl_ok:
            new_exempt = 1
        else:
            new_exempt = prev_exempt

        exempt_state[s] = new_exempt

        if new_exempt == 0:
            w[s] *= v

    # Cash absorbs residual
    non_cash = w.drop(cfg.cash_col).clip(lower=0.0, upper=cfg.per_strat_max)
    s_nc = non_cash.sum()
    if s_nc > 1.0:
        non_cash /= s_nc
    w.update(non_cash)
    w[cfg.cash_col] = max(0.0, 1.0 - w.drop(cfg.cash_col).sum())
    return w, exempt_state
