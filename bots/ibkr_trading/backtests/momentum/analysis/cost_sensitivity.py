"""Cost sensitivity analysis — how robust is edge to transaction costs.

Tests profitability at various slippage/commission multipliers to determine
whether the strategy's edge is robust or fragile.
"""
from __future__ import annotations

import numpy as np


def cost_sensitivity_report(trades: list, strategy: str = "momentum") -> str:
    """Generate cost sensitivity analysis.

    Args:
        trades: Trade records with pnl_dollars, pnl_points, commission,
                r_multiple, direction, qty fields.
        strategy: Strategy name for header.
    """
    lines = ["=" * 60]
    lines.append(f"  {strategy.upper()} COST SENSITIVITY REPORT")
    lines.append("=" * 60)
    lines.append("")

    if not trades:
        lines.append("  No trades to analyze.")
        return "\n".join(lines)

    # ── Current Cost Impact ──
    lines.append("  A. CURRENT COST IMPACT")
    lines.append("  " + "-" * 40)

    total_commission = sum(getattr(t, 'commission', 0.0) for t in trades)
    # Gross profit = net pnl + total commission (since pnl_dollars is net)
    total_net_pnl = sum(getattr(t, 'pnl_dollars', 0.0) for t in trades)
    total_gross = total_net_pnl + total_commission

    r_arr = np.array([getattr(t, 'r_multiple', 0.0) for t in trades])
    total_r = float(np.sum(r_arr))
    cost_per_r = total_commission / abs(total_r) if total_r != 0 else 0
    cost_pct = total_commission / total_gross * 100 if total_gross > 0 else 0

    lines.append(f"    Total commission:        ${total_commission:,.0f}")
    lines.append(f"    Gross profit:            ${total_gross:+,.0f}")
    lines.append(f"    Net profit:              ${total_net_pnl:+,.0f}")
    lines.append(f"    Commission as % gross:   {cost_pct:.1f}%")
    lines.append(f"    Effective cost per R:     ${cost_per_r:,.0f}")
    lines.append(f"    Avg commission/trade:     ${total_commission/len(trades):,.2f}")

    # ── Slippage Sensitivity ──
    lines.append("")
    lines.append("  B. SLIPPAGE SENSITIVITY")
    lines.append("  " + "-" * 40)
    lines.append("    Recomputing PnL at various cost multipliers:")
    lines.append("")

    multipliers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    header = f"    {'Mult':>5s} {'Net PnL':>12s} {'PF':>6s} {'WR%':>6s} {'AvgR':>7s} {'Sharpe':>7s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    base_commission_per_trade = [getattr(t, 'commission', 0.0) for t in trades]
    base_gross_per_trade = [getattr(t, 'pnl_dollars', 0.0) + getattr(t, 'commission', 0.0) for t in trades]

    breakeven_mult = None

    for mult in multipliers:
        adjusted_pnl = []
        for i, t in enumerate(trades):
            base_comm = base_commission_per_trade[i]
            extra_cost = base_comm * (mult - 1.0)  # Additional cost beyond base
            adj = base_gross_per_trade[i] - base_comm * mult
            adjusted_pnl.append(adj)

        adj_arr = np.array(adjusted_pnl)
        net_pnl = float(np.sum(adj_arr))
        pos = adj_arr[adj_arr > 0]
        neg = adj_arr[adj_arr <= 0]
        pf = float(np.sum(pos) / abs(np.sum(neg))) if len(neg) > 0 and np.sum(neg) != 0 else float('inf')
        wr = float(np.mean(adj_arr > 0)) * 100

        # Approximate R adjustment
        avg_base_r = float(np.mean(r_arr))
        avg_cost_impact = total_commission * (mult - 1.0) / len(trades)
        # Rough avg R (not exact but directional)
        if total_r != 0:
            r_dollar_value = total_net_pnl / total_r
            adj_avg_r = avg_base_r - (avg_cost_impact / r_dollar_value if r_dollar_value != 0 else 0)
        else:
            adj_avg_r = 0

        # Sharpe from adjusted PnL
        if len(adj_arr) > 1 and np.std(adj_arr) > 0:
            sharpe = np.mean(adj_arr) / np.std(adj_arr) * np.sqrt(len(adj_arr) / 2)  # Rough annualization
        else:
            sharpe = 0

        marker = " ◄ current" if mult == 1.0 else ""
        pf_str = f"{pf:.2f}" if pf < 100 else "INF"
        lines.append(
            f"    {mult:5.1f}x {net_pnl:+12,.0f} {pf_str:>6s} {wr:5.1f}% "
            f"{adj_avg_r:+7.3f} {sharpe:7.2f}{marker}"
        )

        if net_pnl <= 0 and breakeven_mult is None:
            breakeven_mult = mult

    # ── Breakeven Slippage ──
    lines.append("")
    lines.append("  C. BREAKEVEN ANALYSIS")
    lines.append("  " + "-" * 40)

    if breakeven_mult is not None:
        lines.append(f"    Breakeven at:          {breakeven_mult:.1f}x current costs")
    else:
        lines.append(f"    Breakeven at:          > 3.0x current costs (very robust)")

    # Binary search for more precise breakeven
    if total_gross > 0 and total_commission > 0:
        # breakeven: gross - mult * commission = 0 → mult = gross / commission
        precise_be = total_gross / total_commission
        lines.append(f"    Precise breakeven:     {precise_be:.2f}x")

    # ── Verdict ──
    lines.append("")
    if breakeven_mult is None or (total_gross > 0 and total_gross / total_commission > 2.0):
        verdict = "ROBUST — profitable at 2x+ current costs"
    elif total_gross > 0 and total_gross / total_commission > 1.5:
        verdict = "MODERATE — profitable at 1.5x but stressed at 2x"
    else:
        verdict = "FRAGILE — edge erodes quickly with increased costs"
    lines.append(f"    Verdict: {verdict}")

    return "\n".join(lines)
