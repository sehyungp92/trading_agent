"""Section 7: Unified leverage governor (decoupled from confidence)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MetaConfig
from .utils import dd_scalar, ewma_vol, max_drawdown


def compute_leverage(
    w: pd.Series,
    hist_ret: pd.DataFrame,
    cfg: MetaConfig,
) -> float:
    """Downside-vol targeting + total-vol cap + DD ladder.
    Target vol is FIXED (not modulated by confidence)."""
    cols = [c for c in w.index if c in hist_ret.columns]
    R = hist_ret[cols].copy()
    r_meta = (R * w[cols].values).sum(axis=1)

    # Downside vol (semi-deviation)
    r_neg = r_meta.clip(upper=0.0)
    sig_down_daily = np.sqrt(
        (r_neg**2).ewm(span=cfg.ewma_downside_span, adjust=False).mean()
    ).iloc[-1]

    # Total vol
    sig_tot_daily = ewma_vol(r_meta, span=cfg.ewma_total_span).iloc[-1]

    # Fixed target vol (the key change from v1)
    sig_star_daily = cfg.base_target_vol_annual / np.sqrt(cfg.ann_factor)

    eps = 1e-12
    L_down = min(cfg.L_max, sig_star_daily / max(sig_down_daily, eps))
    L_cap = cfg.kappa_totalvol_cap * (sig_star_daily / max(sig_tot_daily, eps))
    L_vol = min(L_down, L_cap)

    # DD scalar
    equity = (1.0 + r_meta).cumprod()
    dd = max_drawdown(equity).iloc[-1]
    s_dd = dd_scalar(float(dd), cfg.dd_ladder)

    # Anti-stacking
    s_vol = float(np.clip(L_vol / max(cfg.L_max, eps), 0.0, 1.0))
    s_min = min(s_vol, s_dd)
    penalty = cfg.gamma * np.median([1.0 - s_vol, 1.0 - s_dd])
    s = max(cfg.s_floor, s_min - penalty)

    return float(np.clip(L_vol * s, 0.0, cfg.L_max))
