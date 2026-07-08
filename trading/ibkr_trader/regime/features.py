"""Section 1: Observation matrix (4-8 features depending on config flags)."""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

from .config import MetaConfig
from .utils import rolling_zscore

logger = logging.getLogger(__name__)


def build_observation_matrix(
    macro_df: pd.DataFrame,
    market_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    cfg: MetaConfig,
    growth_feature: str = "growth_feature",
    inflation_feature: str = "inflation_feature",
) -> Tuple[pd.DataFrame, int, int]:
    """Build z-scored observation matrix for HMM ingestion.

    Feature count varies with config flags (4 base + up to 4 optional).

    Returns:
        Xz: DataFrame of z-scored features aligned to daily index
        g_idx: column index of growth feature (for alignment)
        i_idx: column index of inflation feature (for alignment)
    """
    # 1-2: macro features (always included)
    growth_inf = macro_df[[growth_feature, inflation_feature]]

    features = [growth_inf]

    # 3: equity-bond correlation (conditional)
    if not cfg.drop_eq_bond_corr:
        spy_ret = strat_ret_df["SPY"]
        tlt_ret = strat_ret_df["TLT"]
        eq_bond_corr = spy_ret.rolling(60, min_periods=30).corr(tlt_ret).rename("eq_bond_corr")
        features.append(eq_bond_corr)

    # 4: yield curve slope (always)
    slope = market_df["SLOPE_10Y2Y"].rename("yield_curve_slope")
    features.append(slope)

    # 5: credit spread level (always)
    credit = market_df["SPREAD"].rename("credit_spread_level")
    features.append(credit)

    # 6: momentum breadth (conditional)
    if not cfg.drop_momentum_breadth:
        non_cash = strat_ret_df.drop(columns=[cfg.cash_col], errors="ignore")
        mom_63 = non_cash.rolling(63).sum()
        breadth = (mom_63 > 0).mean(axis=1).rename("momentum_breadth")
        features.append(breadth)

    # 7: commodity index (optional, Phase 2+)
    if cfg.use_commodity_feature:
        if "DBC" in market_df.columns:
            commodity = market_df["DBC"].rename("commodity_index")
            features.append(commodity)
        else:
            logger.warning("use_commodity_feature=True but 'DBC' not in market_df; run download first")

    # 8: real rates (optional, Phase 2+)
    if cfg.use_real_rates_feature:
        if "REAL_RATE_10Y" in market_df.columns:
            real_rate = market_df["REAL_RATE_10Y"].rename("real_rate")
            features.append(real_rate)
        else:
            logger.warning("use_real_rates_feature=True but 'REAL_RATE_10Y' not in market_df; run download first")

    # 9: VIX level (captures vol regime — currently only in crisis overlay)
    if cfg.use_vix_feature:
        if "VIX" in market_df.columns:
            vix = market_df["VIX"].rename("vix_level")
            features.append(vix)
        else:
            logger.warning("use_vix_feature=True but 'VIX' not in market_df")

    # 10: Realized equity volatility (different from implied VIX)
    if cfg.use_realized_vol_feature:
        if "SPY" in strat_ret_df.columns:
            realized_vol = strat_ret_df["SPY"].rolling(21).std().rename("realized_vol")
            features.append(realized_vol)
        else:
            logger.warning("use_realized_vol_feature=True but 'SPY' not in strat_ret_df")

    # 11: Cross-asset momentum divergence (trend regime indicator)
    if cfg.use_trend_divergence_feature:
        non_cash = strat_ret_df.drop(columns=[cfg.cash_col], errors="ignore")
        mom_21 = non_cash.rolling(21).sum()
        trend_div = mom_21.std(axis=1).rename("trend_divergence")
        features.append(trend_div)

    raw = pd.concat(features, axis=1)
    Xz = rolling_zscore(raw, cfg.z_window, cfg.z_minp).dropna()

    g_idx = list(Xz.columns).index(growth_feature)
    i_idx = list(Xz.columns).index(inflation_feature)
    return Xz, g_idx, i_idx
