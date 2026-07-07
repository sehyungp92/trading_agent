"""Idle capital deployment study.

Evaluates overlay strategies for the ~80-90% of capital that sits
uninvested during the active strategy's operation.

Usage:
    python -m backtest.analysis.idle_capital_study
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ── Indicator helpers (self-contained, mirrors strategy/indicators.py) ─────

def ema(series: np.ndarray, period: int) -> np.ndarray:
    """EMA with SMA seed."""
    out = np.full_like(series, np.nan, dtype=float)
    if len(series) < period:
        return out
    out[period - 1] = np.mean(series[:period])
    k = 2.0 / (period + 1)
    for i in range(period, len(series)):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray:
    """Wilder-smoothed ATR."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray:
    """Wilder's ADX."""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
    atr_arr = atr(high, low, close, period)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    # Wilder smooth +DM and -DM
    sm_plus = np.full(n, np.nan)
    sm_minus = np.full(n, np.nan)
    if n < period + 1:
        return np.full(n, np.nan)
    sm_plus[period] = np.sum(plus_dm[1:period + 1])
    sm_minus[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        sm_plus[i] = sm_plus[i - 1] - sm_plus[i - 1] / period + plus_dm[i]
        sm_minus[i] = sm_minus[i - 1] - sm_minus[i - 1] / period + minus_dm[i]
    for i in range(period, n):
        if atr_arr[i] is not None and atr_arr[i] > 0 and not np.isnan(atr_arr[i]):
            plus_di[i] = 100.0 * sm_plus[i] / (atr_arr[i] * period)
            minus_di[i] = 100.0 * sm_minus[i] / (atr_arr[i] * period)
    dx = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            s = plus_di[i] + minus_di[i]
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / s if s > 0 else 0
    adx_out = np.full(n, np.nan)
    # First ADX = mean of first `period` DX values
    start = period
    valid_dx = []
    idx = start
    while idx < n and len(valid_dx) < period:
        if not np.isnan(dx[idx]):
            valid_dx.append(dx[idx])
        idx += 1
    if len(valid_dx) == period:
        adx_out[idx - 1] = np.mean(valid_dx)
        for i in range(idx, n):
            if not np.isnan(dx[i]):
                adx_out[i] = (adx_out[i - 1] * (period - 1) + dx[i]) / period
    return adx_out


def sma(series: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    out = np.full_like(series, np.nan, dtype=float)
    for i in range(period - 1, len(series)):
        out[i] = np.mean(series[i - period + 1: i + 1])
    return out


# ── Regime classification (mirrors ATRSS logic) ──────────────────────────

def compute_regime(adx_arr: np.ndarray, adx_on: float = 20.0,
                   adx_off: float = 15.0) -> np.ndarray:
    """Returns regime array: 2=STRONG_TREND, 1=TREND, 0=RANGE."""
    n = len(adx_arr)
    regime = np.zeros(n, dtype=int)
    on = False
    for i in range(n):
        if np.isnan(adx_arr[i]):
            regime[i] = 0
            continue
        if adx_arr[i] >= adx_on:
            on = True
        elif adx_arr[i] < adx_off:
            on = False
        if on and adx_arr[i] >= 25:
            regime[i] = 2  # STRONG_TREND
        elif on:
            regime[i] = 1  # TREND
        else:
            regime[i] = 0  # RANGE
    return regime


# ── Strategy definitions ─────────────────────────────────────────────────

def buy_and_hold(prices: dict[str, np.ndarray], equity: float,
                 weights: dict[str, float]) -> np.ndarray:
    """Equal or weighted buy-and-hold across symbols."""
    n = len(next(iter(prices.values())))
    shares = {}
    for sym, w in weights.items():
        shares[sym] = (equity * w) / prices[sym][0]
    eq = np.zeros(n)
    for i in range(n):
        eq[i] = sum(shares[sym] * prices[sym][i] for sym in shares)
    return eq


def strategy_sma_crossover(prices: dict[str, np.ndarray], equity: float,
                           weights: dict[str, float],
                           fast: int = 50, slow: int = 200) -> np.ndarray:
    """SMA crossover per symbol. Invested when fast > slow, cash otherwise."""
    n = len(next(iter(prices.values())))
    eq = np.full(n, equity)
    # Track per-symbol allocation
    cash = equity
    positions: dict[str, float] = {}  # sym -> shares held

    for sym in weights:
        positions[sym] = 0.0

    sma_f = {sym: sma(p, fast) for sym, p in prices.items()}
    sma_s = {sym: sma(p, slow) for sym, p in prices.items()}

    for i in range(1, n):
        # Check signals at close, execute next bar (simplification: same bar)
        for sym in weights:
            sf = sma_f[sym][i]
            ss = sma_s[sym][i]
            if np.isnan(sf) or np.isnan(ss):
                continue
            alloc = equity * weights[sym]  # target allocation per symbol
            if sf > ss and positions[sym] == 0:
                # Buy
                shares = alloc / prices[sym][i]
                positions[sym] = shares
                cash -= shares * prices[sym][i]
            elif sf <= ss and positions[sym] > 0:
                # Sell
                cash += positions[sym] * prices[sym][i]
                positions[sym] = 0.0

        # Mark to market
        mtm = cash
        for sym in weights:
            mtm += positions[sym] * prices[sym][i]
        eq[i] = mtm
        # Update target allocation based on current equity
        equity = mtm

    return eq


def strategy_ema_crossover(prices: dict[str, np.ndarray], equity: float,
                           weights: dict[str, float],
                           fast: int = 9, slow: int = 21) -> np.ndarray:
    """EMA crossover per symbol (matches ATRSS EMA periods)."""
    n = len(next(iter(prices.values())))
    eq = np.full(n, equity)
    cash = equity
    positions: dict[str, float] = {}
    for sym in weights:
        positions[sym] = 0.0

    ema_f = {sym: ema(p, fast) for sym, p in prices.items()}
    ema_s = {sym: ema(p, slow) for sym, p in prices.items()}

    for i in range(1, n):
        for sym in weights:
            ef = ema_f[sym][i]
            es = ema_s[sym][i]
            if np.isnan(ef) or np.isnan(es):
                continue
            if ef > es and positions[sym] == 0:
                alloc_pct = weights[sym]
                shares = (eq[i - 1] * alloc_pct) / prices[sym][i]
                positions[sym] = shares
                cash -= shares * prices[sym][i]
            elif ef <= es and positions[sym] > 0:
                cash += positions[sym] * prices[sym][i]
                positions[sym] = 0.0

        mtm = cash
        for sym in weights:
            mtm += positions[sym] * prices[sym][i]
        eq[i] = mtm

    return eq


def strategy_regime_gated(prices: dict[str, np.ndarray],
                          highs: dict[str, np.ndarray],
                          lows: dict[str, np.ndarray],
                          equity: float,
                          weights: dict[str, float]) -> np.ndarray:
    """Regime-gated exposure using ADX regime detection (ATRSS logic).

    - STRONG_TREND (ADX≥25): 100% invested in that symbol
    - TREND (ADX≥20):        75% invested
    - RANGE (ADX<15):         0% invested (cash)

    Uses EMA 9/21 for direction (long only when fast > slow).
    """
    n = len(next(iter(prices.values())))
    eq = np.full(n, equity)
    cash = equity
    positions: dict[str, float] = {}
    for sym in weights:
        positions[sym] = 0.0

    adx_arr = {}
    regime_arr = {}
    ema_f = {}
    ema_s = {}
    for sym in weights:
        adx_arr[sym] = adx(highs[sym], lows[sym], prices[sym])
        regime_arr[sym] = compute_regime(adx_arr[sym])
        ema_f[sym] = ema(prices[sym], 9)
        ema_s[sym] = ema(prices[sym], 21)

    for i in range(1, n):
        current_eq = eq[i - 1]
        for sym in weights:
            r = regime_arr[sym][i]
            ef = ema_f[sym][i]
            es = ema_s[sym][i]
            if np.isnan(ef) or np.isnan(es):
                continue

            # Direction gate: long only when EMA fast > slow
            bullish = ef > es

            # Target exposure based on regime
            if r == 2 and bullish:      # STRONG_TREND
                target_pct = weights[sym] * 1.0
            elif r == 1 and bullish:    # TREND
                target_pct = weights[sym] * 0.75
            else:                        # RANGE or bearish
                target_pct = 0.0

            target_shares = (current_eq * target_pct) / prices[sym][i]
            delta = target_shares - positions[sym]

            if abs(delta) > 0.01:
                cash -= delta * prices[sym][i]
                positions[sym] = target_shares

        mtm = cash
        for sym in weights:
            mtm += positions[sym] * prices[sym][i]
        eq[i] = mtm

    return eq


def strategy_regime_gated_longshort(prices: dict[str, np.ndarray],
                                     highs: dict[str, np.ndarray],
                                     lows: dict[str, np.ndarray],
                                     equity: float,
                                     weights: dict[str, float]) -> np.ndarray:
    """Regime-gated with short capability in bear trends.

    - STRONG_TREND + bullish:  100% long
    - TREND + bullish:          75% long
    - STRONG_TREND + bearish:   50% short (conservative)
    - TREND + bearish:          25% short
    - RANGE:                     0% (cash)
    """
    n = len(next(iter(prices.values())))
    eq = np.full(n, equity)
    cash = equity
    positions: dict[str, float] = {}
    for sym in weights:
        positions[sym] = 0.0

    adx_arr = {}
    regime_arr = {}
    ema_f = {}
    ema_s = {}
    for sym in weights:
        adx_arr[sym] = adx(highs[sym], lows[sym], prices[sym])
        regime_arr[sym] = compute_regime(adx_arr[sym])
        ema_f[sym] = ema(prices[sym], 9)
        ema_s[sym] = ema(prices[sym], 21)

    for i in range(1, n):
        current_eq = eq[i - 1]
        for sym in weights:
            r = regime_arr[sym][i]
            ef = ema_f[sym][i]
            es = ema_s[sym][i]
            if np.isnan(ef) or np.isnan(es):
                continue

            bullish = ef > es

            if r == 2 and bullish:
                target_pct = weights[sym] * 1.0
            elif r == 1 and bullish:
                target_pct = weights[sym] * 0.75
            elif r == 2 and not bullish:
                target_pct = -weights[sym] * 0.50
            elif r == 1 and not bullish:
                target_pct = -weights[sym] * 0.25
            else:
                target_pct = 0.0

            target_shares = (current_eq * target_pct) / prices[sym][i]
            delta = target_shares - positions[sym]

            if abs(delta) > 0.01:
                cash -= delta * prices[sym][i]
                positions[sym] = target_shares

        mtm = cash
        for sym in weights:
            mtm += positions[sym] * prices[sym][i]
        eq[i] = mtm

    return eq


def strategy_dual_momentum(prices: dict[str, np.ndarray], equity: float,
                           lookback: int = 63) -> np.ndarray:
    """Dual momentum: invest in whichever of QQQ/GLD has stronger momentum.

    If both negative, go to cash. Rebalance monthly (every 21 bars).
    """
    syms = list(prices.keys())
    n = len(prices[syms[0]])
    eq = np.full(n, equity)
    cash = equity
    held_sym: str | None = None
    held_shares = 0.0

    for i in range(lookback, n):
        # Rebalance every 21 bars
        if (i - lookback) % 21 != 0:
            # Mark to market
            mtm = cash + (held_shares * prices[held_sym][i] if held_sym else 0)
            eq[i] = mtm
            continue

        # Compute momentum (rate of change over lookback)
        mom = {}
        for sym in syms:
            mom[sym] = (prices[sym][i] / prices[sym][i - lookback] - 1) * 100

        # Sell current
        if held_sym is not None:
            cash += held_shares * prices[held_sym][i]
            held_shares = 0.0
            held_sym = None

        # Pick best momentum if positive
        best_sym = max(mom, key=mom.get)
        if mom[best_sym] > 0:
            held_sym = best_sym
            held_shares = cash / prices[held_sym][i]
            cash = 0.0

        eq[i] = cash + (held_shares * prices[held_sym][i] if held_sym else 0)

    return eq


def strategy_vol_weighted(prices: dict[str, np.ndarray],
                          highs: dict[str, np.ndarray],
                          lows: dict[str, np.ndarray],
                          equity: float,
                          rebal_period: int = 21) -> np.ndarray:
    """Inverse-volatility weighted: allocate more to lower-vol assets.

    Always invested, rebalances monthly. Lower-vol assets get more weight.
    """
    syms = list(prices.keys())
    n = len(prices[syms[0]])
    eq = np.full(n, equity)
    atr_arr = {sym: atr(highs[sym], lows[sym], prices[sym], 21) for sym in syms}

    cash = 0.0
    positions: dict[str, float] = {sym: 0.0 for sym in syms}

    for i in range(21, n):
        if (i - 21) % rebal_period == 0:
            # Liquidate
            current_eq = cash
            for sym in syms:
                current_eq += positions[sym] * prices[sym][i]
            cash = current_eq
            for sym in syms:
                positions[sym] = 0.0

            # Compute inverse-vol weights
            inv_vols = {}
            for sym in syms:
                a = atr_arr[sym][i]
                if np.isnan(a) or a <= 0:
                    inv_vols[sym] = 0
                else:
                    # Normalize ATR by price for comparability
                    inv_vols[sym] = 1.0 / (a / prices[sym][i])
            total = sum(inv_vols.values())
            if total > 0:
                for sym in syms:
                    w = inv_vols[sym] / total
                    positions[sym] = (cash * w) / prices[sym][i]
                cash = 0.0

        # Mark to market
        mtm = cash
        for sym in syms:
            mtm += positions[sym] * prices[sym][i]
        eq[i] = mtm

    return eq


def strategy_core_satellite(core_eq: np.ndarray,
                            active_pnl: float,
                            core_pct: float = 0.70) -> np.ndarray:
    """Core + Satellite: core_pct in B&H overlay, rest in active strategy.

    We scale the B&H equity curve and add the active strategy's
    contribution proportionally.
    """
    n = len(core_eq)
    # Scale core to core_pct of initial
    init = core_eq[0]
    core_scaled = core_eq * core_pct
    # Active PnL scaled to satellite allocation
    sat_pct = 1.0 - core_pct
    sat_base = init * sat_pct
    # Assume active PnL is distributed linearly for simplicity
    active_daily_pnl = active_pnl * sat_pct / n
    sat_curve = sat_base + np.arange(n) * active_daily_pnl
    return core_scaled + sat_curve


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(eq: np.ndarray, n_days: int) -> dict:
    """Compute standard performance metrics."""
    eq = eq[~np.isnan(eq)]
    if len(eq) < 2:
        return {}
    years = n_days / 365.25
    total_ret = (eq[-1] / eq[0] - 1) * 100
    ann_ret = ((eq[-1] / eq[0]) ** (1 / years) - 1) * 100
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = float(np.min(dd))
    rets = np.diff(eq) / eq[:-1]
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Time invested (non-cash)
    # Approximate: bars where equity changed
    invested_bars = np.sum(np.abs(rets) > 1e-10)
    pct_invested = invested_bars / len(rets) * 100

    return {
        "init": eq[0],
        "final": eq[-1],
        "total_pnl": eq[-1] - eq[0],
        "total_ret": total_ret,
        "ann_ret": ann_ret,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
        "pct_invested": pct_invested,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    data_dir = Path("backtest/data/raw")
    init_equity = 10_000.0

    # Load daily data
    daily = {}
    for sym in ["QQQ", "GLD"]:
        df = pd.read_parquet(data_dir / f"{sym}_1d.parquet")
        df.columns = df.columns.str.lower()
        daily[sym] = df

    # Align dates
    common_idx = daily["QQQ"].index.intersection(daily["GLD"].index)
    for sym in daily:
        daily[sym] = daily[sym].loc[common_idx]

    n_days = len(common_idx)
    prices = {sym: daily[sym]["close"].values for sym in daily}
    highs = {sym: daily[sym]["high"].values for sym in daily}
    lows = {sym: daily[sym]["low"].values for sym in daily}

    ew = {"QQQ": 0.5, "GLD": 0.5}

    # ── Run all strategies ───────────────────────────────────────────────

    results = {}

    # 1. Buy & Hold (baseline)
    eq = buy_and_hold(prices, init_equity, ew)
    results["Buy & Hold (EW)"] = eq

    # 2. SMA 50/200 crossover
    eq = strategy_sma_crossover(prices, init_equity, ew, 50, 200)
    results["SMA 50/200 Cross"] = eq

    # 3. SMA 10/50 crossover (faster)
    eq = strategy_sma_crossover(prices, init_equity, ew, 10, 50)
    results["SMA 10/50 Cross"] = eq

    # 4. EMA 9/21 crossover (matches ATRSS periods)
    eq = strategy_ema_crossover(prices, init_equity, ew, 9, 21)
    results["EMA 9/21 Cross"] = eq

    # 5. EMA 13/48 crossover (matches Helix periods)
    eq = strategy_ema_crossover(prices, init_equity, ew, 13, 48)
    results["EMA 13/48 Cross"] = eq

    # 6. Regime-gated (long only) — uses ATRSS ADX regime detection
    eq = strategy_regime_gated(prices, highs, lows, init_equity, ew)
    results["Regime-Gated (L)"] = eq

    # 7. Regime-gated (long/short)
    eq = strategy_regime_gated_longshort(prices, highs, lows, init_equity, ew)
    results["Regime-Gated (L/S)"] = eq

    # 8. Dual momentum (QQQ vs GLD, 63-day lookback)
    eq = strategy_dual_momentum(prices, init_equity, 63)
    results["Dual Momentum 63d"] = eq

    # 9. Dual momentum (shorter, 21-day)
    eq = strategy_dual_momentum(prices, init_equity, 21)
    results["Dual Momentum 21d"] = eq

    # 10. Inverse-vol weighted (always invested, rebalanced monthly)
    eq = strategy_vol_weighted(prices, highs, lows, init_equity, 21)
    results["Inv-Vol Weighted"] = eq

    # 11. Core(70%) + Satellite(30% active strategy)
    bh_eq = results["Buy & Hold (EW)"]
    eq = strategy_core_satellite(bh_eq, 3600, core_pct=0.70)
    results["Core70 + Sat30"] = eq

    # 12. Core(50%) + Satellite(50% active strategy)
    eq = strategy_core_satellite(bh_eq, 3600, core_pct=0.50)
    results["Core50 + Sat50"] = eq

    # 13. Active strategy alone (from backtest results)
    # Linearly approximate the equity curve for table inclusion
    active_eq = np.linspace(init_equity, init_equity + 3600, n_days)
    results["Active Only"] = active_eq

    # 14. Regime-gated + T-bills on cash portion
    # Estimate: regime-gated is invested ~60% of time, idle earns 4.5% annualized
    regime_eq = results["Regime-Gated (L)"].copy()
    # Add risk-free return on cash portion (approximate)
    rfr_daily = (1 + 0.045) ** (1/252) - 1  # 4.5% annualized
    regime_plus_rf = regime_eq.copy()
    regime_rets = np.diff(regime_eq) / regime_eq[:-1]
    for i in range(1, len(regime_plus_rf)):
        # If equity didn't change much, we're mostly in cash → earn risk-free
        if abs(regime_rets[i-1]) < 1e-8:
            regime_plus_rf[i] = regime_plus_rf[i-1] * (1 + rfr_daily)
        else:
            regime_plus_rf[i] = regime_plus_rf[i-1] * (1 + regime_rets[i-1])
    results["Regime + T-Bills"] = regime_plus_rf

    # ── Print comparison table ───────────────────────────────────────────

    print()
    print("=" * 115)
    print(f"IDLE CAPITAL DEPLOYMENT STUDY - QQQ + GLD, ${init_equity:,.0f}")
    print(f"Period: {common_idx[0].date()} to {common_idx[-1].date()} ({n_days} trading days)")
    print("=" * 115)
    print(f"{'Strategy':<24} {'Final':>9} {'PnL':>9} {'Tot%':>7} {'Ann%':>7} "
          f"{'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7} {'Inv%':>6}")
    print("-" * 115)

    all_metrics = {}
    for name, eq in results.items():
        m = compute_metrics(eq, n_days)
        all_metrics[name] = m
        if not m:
            continue
        marker = ""
        if name == "Buy & Hold (EW)":
            marker = " <-- B&H"
        elif name == "Active Only":
            marker = " <-- current"
        print(f"{name:<24} ${m['final']:>8,.0f} ${m['total_pnl']:>+8,.0f} "
              f"{m['total_ret']:>+6.1f}% {m['ann_ret']:>+6.1f}% "
              f"{m['max_dd']:>6.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f} "
              f"{m['pct_invested']:>5.0f}%{marker}")

    print("-" * 115)

    # ── Highlight best performers by category ────────────────────────────

    # Exclude "Active Only" from overlay comparisons since it's approximate
    overlay_names = [n for n in all_metrics if n != "Active Only" and n != "Buy & Hold (EW)"]

    best_calmar_name = max(overlay_names, key=lambda n: all_metrics[n].get("calmar", 0))
    best_sharpe_name = max(overlay_names, key=lambda n: all_metrics[n].get("sharpe", 0))
    best_pnl_name = max(overlay_names, key=lambda n: all_metrics[n].get("total_pnl", 0))
    best_dd_name = max(overlay_names, key=lambda n: all_metrics[n].get("max_dd", -999))

    bh = all_metrics["Buy & Hold (EW)"]

    print(f"\n{'CATEGORY WINNERS':}")
    print(f"  Best Calmar (risk-adj):  {best_calmar_name} "
          f"({all_metrics[best_calmar_name]['calmar']:.2f} vs B&H {bh['calmar']:.2f})")
    print(f"  Best Sharpe:             {best_sharpe_name} "
          f"({all_metrics[best_sharpe_name]['sharpe']:.2f} vs B&H {bh['sharpe']:.2f})")
    print(f"  Best PnL:                {best_pnl_name} "
          f"(${all_metrics[best_pnl_name]['total_pnl']:+,.0f} vs B&H ${bh['total_pnl']:+,.0f})")
    print(f"  Lowest MaxDD:            {best_dd_name} "
          f"({all_metrics[best_dd_name]['max_dd']:.1f}% vs B&H {bh['max_dd']:.1f}%)")

    # ── Strategies that beat B&H on BOTH return AND drawdown ─────────────

    print(f"\n{'STRATEGIES THAT BEAT B&H ON RETURN':}")
    for name in overlay_names:
        m = all_metrics[name]
        if m["total_ret"] > bh["total_ret"]:
            dd_vs = "better" if m["max_dd"] > bh["max_dd"] else "worse"
            print(f"  {name}: +{m['total_ret']:.1f}% return, "
                  f"{m['max_dd']:.1f}% DD ({dd_vs} than B&H's {bh['max_dd']:.1f}%)")

    print(f"\n{'STRATEGIES THAT BEAT B&H CALMAR WITH >50% OF B&H RETURN':}")
    for name in overlay_names:
        m = all_metrics[name]
        if m["calmar"] > bh["calmar"] and m["ann_ret"] > bh["ann_ret"] * 0.5:
            print(f"  {name}: Calmar {m['calmar']:.2f}, Ann.Ret {m['ann_ret']:+.1f}%, "
                  f"MaxDD {m['max_dd']:.1f}%")

    print()
    print("=" * 115)


if __name__ == "__main__":
    main()
