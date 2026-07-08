"""Priority 2: Stop Loss Calibration Analysis.

Analyzes CLOSE_STOP trades to determine if stop losses could be
tightened or loosened for better risk-adjusted returns.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from backtests.stock.models import TradeRecord


def _hdr(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


def iaric_stop_calibration(trades: list[TradeRecord]) -> str:
    """Analyze CLOSE_STOP trades for stop loss calibration insights."""
    lines = [_hdr("STOP-1  Stop Loss Calibration Analysis")]

    stopped = [t for t in trades if t.exit_reason == "CLOSE_STOP"]
    non_stopped = [t for t in trades if t.exit_reason != "CLOSE_STOP"]

    if not stopped:
        lines.append("  No CLOSE_STOP trades found.")
        return "\n".join(lines)

    n_stopped = len(stopped)
    n_total = len(trades)
    lines.append(f"  Stopped trades: {n_stopped}/{n_total} ({n_stopped/n_total:.1%})")
    lines.append(f"  Total stop loss: ${sum(t.pnl_net for t in stopped):,.2f}")
    lines.append(f"  Avg stop R: {np.mean([t.r_multiple for t in stopped]):+.3f}")

    # --- MAE distribution of stopped trades ---
    lines.append(f"\n  MAE Distribution (stopped trades):")
    mae_rs = []
    for t in stopped:
        m = t.metadata.get("mae_r", 0) if t.metadata else 0
        if m == 0 and t.risk_per_share > 0 and t.max_adverse > 0:
            m = (t.entry_price - t.max_adverse) / t.risk_per_share
        mae_rs.append(m)

    if mae_rs:
        arr = np.array(mae_rs)
        for pct, label in [(25, "P25"), (50, "P50"), (75, "P75"), (90, "P90"), (100, "Max")]:
            lines.append(f"    {label}: {np.percentile(arr, pct):+.3f}R")
        lines.append(f"    Mean MAE: {np.mean(arr):+.3f}R")

    # --- MFE of stopped trades (were they ever winners?) ---
    lines.append(f"\n  MFE Distribution (stopped trades -- were they ever winning?):")
    mfe_rs = []
    for t in stopped:
        m = t.metadata.get("mfe_r", 0) if t.metadata else 0
        if m == 0 and t.risk_per_share > 0 and t.max_favorable > 0:
            m = (t.max_favorable - t.entry_price) / t.risk_per_share
        mfe_rs.append(m)

    if mfe_rs:
        arr = np.array(mfe_rs)
        winners_before_stop = sum(1 for m in mfe_rs if m > 0.5)
        lines.append(f"    Had MFE > 0.5R before stopping: {winners_before_stop}/{n_stopped} ({winners_before_stop/n_stopped:.1%})")
        lines.append(f"    Had MFE > 1.0R before stopping: {sum(1 for m in mfe_rs if m > 1.0)}/{n_stopped}")
        for pct, label in [(25, "P25"), (50, "P50"), (75, "P75"), (90, "P90"), (100, "Max")]:
            lines.append(f"    {label}: {np.percentile(arr, pct):+.3f}R")

    # --- Time to stop ---
    lines.append(f"\n  Time-to-Stop Distribution:")
    hold_hours = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in stopped]
    if hold_hours:
        arr = np.array(hold_hours)
        early = sum(1 for h in hold_hours if h < 2)
        mid = sum(1 for h in hold_hours if 2 <= h < 5)
        late = sum(1 for h in hold_hours if h >= 5)
        lines.append(f"    Early (<2h): {early} ({early/n_stopped:.1%})")
        lines.append(f"    Mid (2-5h): {mid} ({mid/n_stopped:.1%})")
        lines.append(f"    Late (>5h): {late} ({late/n_stopped:.1%})")
        lines.append(f"    Mean hold: {np.mean(arr):.1f}h  |  Median: {np.median(arr):.1f}h")

    # --- Stop distance analysis ---
    lines.append(f"\n  Stop Distance Analysis:")
    stop_distances = []
    for t in stopped:
        if t.risk_per_share > 0 and t.entry_price > 0:
            stop_distances.append(t.risk_per_share / t.entry_price * 100)
    if stop_distances:
        arr = np.array(stop_distances)
        lines.append(f"    Mean stop distance: {np.mean(arr):.2f}% of entry price")
        lines.append(f"    Median stop distance: {np.median(arr):.2f}%")

    # --- Simulated stop alternatives ---
    lines.append(f"\n  Stop Distance Optimization (simulated using MFE/MAE):")
    lines.append(f"    {'Mult':>6s} {'Saved':>8s} {'Lost Winners':>14s} {'Net Impact':>12s}")
    lines.append(f"    {'-'*44}")

    # For each hypothetical stop multiplier, estimate impact
    # A tighter stop would save money on deep losers but might have stopped some winners early
    all_mfe_rs = []
    all_mae_rs = []
    for t in trades:
        mfe = t.metadata.get("mfe_r", 0) if t.metadata else 0
        mae = t.metadata.get("mae_r", 0) if t.metadata else 0
        if mfe == 0 and t.risk_per_share > 0 and t.max_favorable > 0:
            mfe = (t.max_favorable - t.entry_price) / t.risk_per_share
        if mae == 0 and t.risk_per_share > 0 and t.max_adverse > 0:
            mae = (t.entry_price - t.max_adverse) / t.risk_per_share
        all_mfe_rs.append(mfe)
        all_mae_rs.append(mae)

    for mult in [0.50, 0.75, 1.00, 1.25, 1.50]:
        # Trades that would have been stopped at this tighter level
        would_stop = sum(1 for mae in all_mae_rs if mae >= mult)
        # Of current winners, how many had MAE deeper than this mult?
        current_winners = [i for i, t in enumerate(trades) if t.is_winner]
        lost_winners = sum(1 for i in current_winners if all_mae_rs[i] >= mult)
        # Saved = stopped trades with MAE > mult that now exit earlier at -mult R instead of actual R
        saved_r = 0.0
        for i, t in enumerate(trades):
            if t.exit_reason == "CLOSE_STOP" and all_mae_rs[i] >= mult and t.r_multiple < -mult:
                saved_r += (-mult - t.r_multiple)  # positive = savings
        lines.append(f"    {mult:5.2f}x  {saved_r:+7.2f}R  {lost_winners:>14d}  {'see note':>12s}")

    lines.append(f"\n    Note: 'Lost Winners' = current winners whose MAE exceeded the mult.")
    lines.append(f"    These might have been stopped out prematurely at a tighter level.")

    # --- Stopped vs non-stopped comparison by sector ---
    lines.append(f"\n  Sector Breakdown of Stopped Trades:")
    sector_stops: dict[str, int] = defaultdict(int)
    sector_total: dict[str, int] = defaultdict(int)
    for t in trades:
        sector_total[t.sector] += 1
        if t.exit_reason == "CLOSE_STOP":
            sector_stops[t.sector] += 1

    lines.append(f"    {'Sector':<20s} {'Stopped':>8s} {'Total':>6s} {'Rate':>8s}")
    lines.append(f"    {'-'*44}")
    for sector in sorted(sector_total, key=lambda s: sector_stops.get(s, 0) / max(sector_total[s], 1), reverse=True):
        s = sector_stops.get(sector, 0)
        t = sector_total[sector]
        lines.append(f"    {sector:<20s} {s:>8d} {t:>6d} {s/t:>7.1%}")

    return "\n".join(lines)
