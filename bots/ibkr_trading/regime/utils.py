"""Shared utilities for the MR-AWQ Meta Allocator v2."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(df: pd.DataFrame, window: int, minp: int) -> pd.DataFrame:
    mu = df.rolling(window, min_periods=minp).mean()
    sd = df.rolling(window, min_periods=minp).std(ddof=0).replace(0, np.nan)
    return (df - mu) / sd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ewma_vol(daily_returns: pd.Series, span: int) -> pd.Series:
    return np.sqrt(daily_returns.pow(2).ewm(span=span, adjust=False).mean())


def clip_and_renorm(w: pd.Series, wmax: float, eps: float = 1e-12) -> pd.Series:
    w2 = w.clip(lower=0.0, upper=wmax)
    s = w2.sum()
    if s < eps:
        w2[:] = 1.0 / len(w2)
        return w2
    return w2 / s


def max_drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def dd_scalar(dd: float, ladder) -> float:
    sc = 1.0
    for thr, v in ladder:
        if dd <= thr:
            sc = min(sc, v)
    return float(sc)


def budget(cols, **kwargs) -> pd.Series:
    s = pd.Series(0.0, index=cols)
    for k, v in kwargs.items():
        s[k] = v
    assert s.sum() > 0, "Budget sums to 0."
    return s / s.sum()


def compute_avg_pairwise_corr(ret_df: pd.DataFrame, window: int) -> pd.Series:
    out = []
    for t in range(len(ret_df)):
        if t < window:
            out.append(np.nan)
            continue
        x = ret_df.iloc[t - window : t].dropna(axis=1, how="any")
        if x.shape[1] < 2:
            out.append(np.nan)
            continue
        c = x.corr().values
        n = c.shape[0]
        out.append(float(np.nanmean(c[~np.eye(n, dtype=bool)])))
    return pd.Series(out, index=ret_df.index, name="avg_pairwise_corr")
