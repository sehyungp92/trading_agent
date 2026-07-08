"""NQDTC phase diagnostics -- capture ratio, burst stats, direction asymmetry."""
from __future__ import annotations

from typing import Any

from .scoring import NQDTCMetrics


def _net_trade_pnl(trade: Any) -> float:
    return float(getattr(trade, "pnl_dollars", 0.0) or 0.0) - float(getattr(trade, "commission", 0.0) or 0.0)


def generate_phase_diagnostics(
    phase: int,
    metrics: NQDTCMetrics,
    greedy_result: dict | None,
    state_dict: dict | None,
    all_trades: list | None = None,
    force_all_modules: bool = False,
) -> str:
    """Generate NQDTC-specific phase diagnostics report."""
    lines: list[str] = []
    lines.append(f"{'='*60}")
    lines.append(f"NQDTC Phase {phase} Diagnostics")
    lines.append(f"{'='*60}")

    # D1: Core performance (always)
    lines.append("\n--- D1: Core Performance ---")
    lines.append(f"Total trades:    {metrics.total_trades}")
    lines.append(f"Win rate:        {metrics.win_rate:.1%}")
    lines.append(f"Profit factor:   {metrics.profit_factor:.2f}")
    lines.append(f"Net return:      {metrics.net_return_pct:.1f}%")
    lines.append(f"Robust return:   {metrics.robust_net_return_pct:.1f}% (ex largest winner)")
    lines.append(f"Max drawdown:    {metrics.max_dd_pct:.2%}")
    lines.append(f"Calmar:          {metrics.calmar:.2f}")
    lines.append(f"Sharpe:          {metrics.sharpe:.2f}")
    lines.append(f"Sortino:         {metrics.sortino:.2f}")
    lines.append(f"Avg R:           {metrics.avg_r:.3f}")

    # D2: Exit efficiency (always)
    lines.append("\n--- D2: Exit Efficiency ---")
    lines.append(f"Capture ratio:   {metrics.capture_ratio:.3f} (winners exit_R / MFE)")
    lines.append(f"TP1 hit rate:    {metrics.tp1_hit_rate:.1%}")
    lines.append(f"TP2 hit rate:    {metrics.tp2_hit_rate:.1%}")
    lines.append(f"Avg winner R:    {metrics.avg_winner_r:.3f}")
    lines.append(f"Avg loser R:     {metrics.avg_loser_r:.3f}")
    lines.append(f"Avg MFE R:       {metrics.avg_mfe_r:.3f}")
    lines.append(f"Avg hold hours:  {metrics.avg_hold_hours:.1f}")
    lines.append(f"Largest win:     {metrics.largest_winner_r:.3f}R, "
                 f"{metrics.largest_win_pnl_share:.1%} of net profit")

    # D3: Regime analysis (always -- critical for post-audit regime filtering)
    lines.append("\n--- D3: Regime Analysis ---")
    lines.append(f"Range regime:    {metrics.range_regime_pct:.1%} of trades")
    if all_trades:
        _regime_breakdown(lines, all_trades)
    else:
        lines.append("(No trade records available)")

    # D4: Session/direction asymmetry (phase >= 2 or force)
    if phase >= 2 or force_all_modules:
        lines.append("\n--- D4: Session/Direction Asymmetry ---")
        lines.append(f"ETH short WR:    {metrics.eth_short_wr:.1%} ({metrics.eth_short_trades} trades)")

        if all_trades:
            _session_direction_breakdown(lines, all_trades)

    # D5: Trade clustering (phase >= 2 or force)
    if phase >= 2 or force_all_modules:
        lines.append("\n--- D5: Trade Clustering ---")
        lines.append(f"Burst trade pct: {metrics.burst_trade_pct:.1%}")

        if all_trades:
            _burst_analysis(lines, all_trades)

    # D6: Greedy result summary (always)
    if greedy_result:
        lines.append("\n--- D6: Greedy Result ---")
        lines.append(f"Base score:      {greedy_result.get('base_score', 0):.4f}")
        lines.append(f"Final score:     {greedy_result.get('final_score', 0):.4f}")
        lines.append(f"Accepted:        {greedy_result.get('accepted_count', 0)}")
        lines.append(f"Total candidates:{greedy_result.get('total_candidates', 0)}")
        kept = greedy_result.get("kept_features", [])
        if kept:
            lines.append(f"Kept features:   {', '.join(kept)}")

    lines.append(f"\n{'='*60}")
    return "\n".join(lines)


def get_diagnostic_gaps(phase: int, metrics: NQDTCMetrics) -> list[str]:
    """Identify diagnostic gaps for the current phase."""
    gaps: list[str] = []

    if metrics.capture_ratio < 0.42:
        gaps.append("Low MFE capture ratio -- exits leave alpha on the table")
    if metrics.burst_trade_pct > 0.15:
        gaps.append("High burst clustering -- correlated entries drag performance")
    if metrics.eth_short_wr < 0.40 and metrics.eth_short_trades > 30:
        gaps.append("ETH shorts underperforming -- session-direction filter needed")
    if metrics.total_trades < 120:
        gaps.append("Low trade frequency -- signal gates may be too restrictive")
    if metrics.max_dd_pct > 0.20:
        gaps.append("Elevated drawdown -- risk controls or regime filtering needed")
    if metrics.range_regime_pct < 0.40:
        gaps.append("Low Range regime concentration -- regime filtering may help")
    if metrics.tp1_hit_rate < 0.10:
        gaps.append("Very low TP1 hit rate -- TP target may be too aggressive")
    if metrics.profit_factor < 1.6:
        gaps.append("Low profit factor -- edge may be degraded by filters")
    if metrics.largest_win_pnl_share > 0.30:
        gaps.append("High largest-winner concentration -- return may be outlier-dependent")
    if metrics.robust_net_return_pct < 220:
        gaps.append("Low robust return after removing largest winner")

    return gaps


def _session_direction_breakdown(lines: list[str], trades: list) -> None:
    """Break down performance by session x direction."""
    buckets: dict[str, list] = {}
    for t in trades:
        key = f"{t.session}_{'LONG' if t.direction == 1 else 'SHORT'}"
        buckets.setdefault(key, []).append(t)

    for key in sorted(buckets.keys()):
        group = buckets[key]
        count = len(group)
        wins = sum(1 for t in group if t.r_multiple > 0)
        wr = wins / count if count else 0
        avg_r = sum(t.r_multiple for t in group) / count if count else 0
        total_pnl = sum(_net_trade_pnl(t) for t in group)
        lines.append(f"  {key:15s}: {count:3d} trades, WR={wr:.0%}, avgR={avg_r:+.3f}, PnL=${total_pnl:+,.0f}")


def _burst_analysis(lines: list[str], trades: list) -> None:
    """Analyze trade bursts (3+ within 4h)."""
    sorted_trades = sorted(trades, key=lambda t: t.entry_time)
    burst_trades: list = []
    isolated_trades: list = []

    for i, t in enumerate(sorted_trades):
        cluster_size = 1
        for j in range(i + 1, len(sorted_trades)):
            if (sorted_trades[j].entry_time - t.entry_time).total_seconds() <= 14400:
                cluster_size += 1
            else:
                break
        if cluster_size >= 3:
            burst_trades.append(t)
        else:
            isolated_trades.append(t)

    if burst_trades:
        burst_avg_r = sum(t.r_multiple for t in burst_trades) / len(burst_trades)
        burst_pnl = sum(_net_trade_pnl(t) for t in burst_trades)
        lines.append(f"  Burst trades:    {len(burst_trades)}, avgR={burst_avg_r:+.3f}, PnL=${burst_pnl:+,.0f}")
    if isolated_trades:
        iso_avg_r = sum(t.r_multiple for t in isolated_trades) / len(isolated_trades)
        iso_pnl = sum(_net_trade_pnl(t) for t in isolated_trades)
        lines.append(f"  Isolated trades: {len(isolated_trades)}, avgR={iso_avg_r:+.3f}, PnL=${iso_pnl:+,.0f}")


def _regime_breakdown(lines: list[str], trades: list) -> None:
    """Break down performance by composite regime."""
    buckets: dict[str, list] = {}
    for t in trades:
        buckets.setdefault(t.composite_regime, []).append(t)

    for regime in sorted(buckets.keys()):
        group = buckets[regime]
        count = len(group)
        wins = sum(1 for t in group if t.r_multiple > 0)
        wr = wins / count if count else 0
        avg_r = sum(t.r_multiple for t in group) / count if count else 0
        total_pnl = sum(_net_trade_pnl(t) for t in group)
        lines.append(f"  {regime:12s}: {count:3d} trades, WR={wr:.0%}, avgR={avg_r:+.3f}, PnL=${total_pnl:+,.0f}")
