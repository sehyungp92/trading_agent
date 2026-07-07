"""Comprehensive strategy diagnostics — 17-section deep analysis.

Modeled on reference diagnostic frameworks (alcb 30-section, iaric 26-section).
Operates on Trade objects and PerformanceMetrics from BacktestResult.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from crypto_trader.backtest.metrics import PerformanceMetrics
from crypto_trader.core.models import SetupGrade, Side, TerminalMark, Trade


# ── Structured Diagnostic Extraction ─────────────────────────────────


@dataclass
class DiagnosticInsights:
    """Structured diagnostic data extracted from a trade list.

    Reusable by any strategy plugin for adaptive phase analysis.
    """

    n_trades: int
    win_rate: float
    mean_r: float
    profit_factor: float

    per_confirmation: dict[str, dict[str, float]]  # {type: {n, wr, avg_r, total_r, pnl}}
    per_asset: dict[str, dict[str, float]]  # {sym: {n, wr, avg_r, long_wr, short_wr, ...}}
    exit_attribution: dict[str, dict[str, float]]  # {reason: {n, wr, avg_r, total_r, pnl_share}}
    mfe_capture: dict[str, float]  # {avg_mfe_r, avg_mae_r, avg_capture_pct, ...}
    direction: dict[str, dict[str, float]]  # {long: {n, wr, avg_r}, short: ...}
    confluence: dict[int, dict[str, float]]  # {count: {n, wr, avg_r}}
    grade: dict[str, dict[str, float]]  # {A: {n, wr, avg_r}, B: ...}
    duration: dict[str, float]  # {avg_bars, avg_hours}
    concentration: dict[str, float]  # {top1_pct, top20_pct}
    r_stats: dict[str, float]  # {mean, median, std, skew}
    worst_trades: list[dict[str, Any]]  # top 5 by R (most negative)
    best_trades: list[dict[str, Any]]  # top 5 by R (most positive)


def extract_diagnostic_insights(trades: list[Trade]) -> DiagnosticInsights:
    """Extract structured diagnostic data from a trade list.

    Independent of the text-based generate_diagnostics() — pure data extraction
    reusable by any strategy plugin's callbacks.
    """
    if not trades:
        return DiagnosticInsights(
            n_trades=0, win_rate=0.0, mean_r=0.0, profit_factor=0.0,
            per_confirmation={}, per_asset={}, exit_attribution={},
            mfe_capture={}, direction={}, confluence={}, grade={},
            duration={"avg_bars": 0.0, "avg_hours": 0.0},
            concentration={"top1_pct": 0.0, "top20_pct": 0.0},
            r_stats={"mean": 0.0, "median": 0.0, "std": 0.0, "skew": 0.0},
            worst_trades=[], best_trades=[],
        )

    n = len(trades)
    winners = [t for t in trades if t.net_pnl > 0]
    wr = len(winners) / n * 100.0
    rs = [_safe_r(t) for t in trades]
    mean_r = float(np.mean(rs))
    gross_p = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_l = abs(sum(t.net_pnl for t in trades if t.net_pnl <= 0))
    pf = gross_p / gross_l if gross_l > 0 else float("inf") if gross_p > 0 else 0.0

    # Per-confirmation
    per_confirmation: dict[str, dict[str, float]] = {}
    by_conf: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_conf[t.confirmation_type or "unknown"].append(t)
    for ctype, group in by_conf.items():
        c_rs = [_safe_r(t) for t in group]
        c_winners = [t for t in group if t.net_pnl > 0]
        per_confirmation[ctype] = {
            "n": len(group),
            "wr": len(c_winners) / len(group) * 100.0,
            "avg_r": float(np.mean(c_rs)),
            "total_r": sum(c_rs),
            "pnl": sum(t.net_pnl for t in group),
        }

    # Per-asset
    per_asset: dict[str, dict[str, float]] = {}
    by_sym: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_sym[t.symbol].append(t)
    for sym, group in by_sym.items():
        s_rs = [_safe_r(t) for t in group]
        s_winners = [t for t in group if t.net_pnl > 0]
        longs = [t for t in group if t.direction == Side.LONG]
        shorts = [t for t in group if t.direction == Side.SHORT]
        long_wr = (sum(1 for t in longs if t.net_pnl > 0) / len(longs) * 100.0) if longs else 0.0
        short_wr = (sum(1 for t in shorts if t.net_pnl > 0) / len(shorts) * 100.0) if shorts else 0.0
        long_avg_r = float(np.mean([_safe_r(t) for t in longs])) if longs else 0.0
        short_avg_r = float(np.mean([_safe_r(t) for t in shorts])) if shorts else 0.0
        per_asset[sym] = {
            "n": len(group),
            "wr": len(s_winners) / len(group) * 100.0,
            "avg_r": float(np.mean(s_rs)),
            "long_wr": long_wr,
            "short_wr": short_wr,
            "long_avg_r": long_avg_r,
            "short_avg_r": short_avg_r,
        }

    # Exit attribution
    exit_attribution: dict[str, dict[str, float]] = {}
    total_pnl = sum(t.net_pnl for t in trades)
    by_exit: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_exit[t.exit_reason or "unknown"].append(t)
    for reason, group in by_exit.items():
        e_rs = [_safe_r(t) for t in group]
        e_winners = [t for t in group if t.net_pnl > 0]
        e_pnl = sum(t.net_pnl for t in group)
        exit_attribution[reason] = {
            "n": len(group),
            "wr": len(e_winners) / len(group) * 100.0,
            "avg_r": float(np.mean(e_rs)),
            "total_r": sum(e_rs),
            "pnl_share": e_pnl / total_pnl if total_pnl != 0 else 0.0,
        }

    # MFE capture — winners-only for capture/giveback (consistent with exit_efficiency)
    has_mfe = [t for t in trades if t.mfe_r is not None and t.mfe_r > 0]
    winners_with_mfe = [t for t in winners if t.mfe_r is not None and t.mfe_r > 0]
    if winners_with_mfe:
        caps = [_safe_r(t) / t.mfe_r for t in winners_with_mfe]
        avg_cap = float(np.mean(caps))
        avg_give = 1.0 - avg_cap
    else:
        avg_cap = 0.0
        avg_give = 0.0
    losers_with_mfe = [t for t in trades if t.net_pnl <= 0 and t.mfe_r is not None and t.mfe_r > 0]
    losers_total = [t for t in trades if t.net_pnl <= 0]
    mfe_capture = {
        "avg_mfe_r": float(np.mean([t.mfe_r for t in has_mfe])) if has_mfe else 0.0,
        "avg_mae_r": float(np.mean([t.mae_r or 0 for t in trades])),
        "avg_capture_pct": avg_cap,
        "avg_giveback_pct": avg_give,
        "losers_with_mfe_pct": len(losers_with_mfe) / len(losers_total) * 100.0 if losers_total else 0.0,
    }

    # Direction
    direction_data: dict[str, dict[str, float]] = {}
    for side in [Side.LONG, Side.SHORT]:
        group = [t for t in trades if t.direction == side]
        if group:
            d_rs = [_safe_r(t) for t in group]
            d_winners = [t for t in group if t.net_pnl > 0]
            direction_data[side.value.lower()] = {
                "n": len(group),
                "wr": len(d_winners) / len(group) * 100.0,
                "avg_r": float(np.mean(d_rs)),
            }

    # Confluence
    confluence_data: dict[int, dict[str, float]] = {}
    by_confl: dict[int, list[Trade]] = defaultdict(list)
    for t in trades:
        c = len(t.confluences_used) if t.confluences_used else 0
        by_confl[c].append(t)
    for count, group in sorted(by_confl.items()):
        co_rs = [_safe_r(t) for t in group]
        co_winners = [t for t in group if t.net_pnl > 0]
        confluence_data[count] = {
            "n": len(group),
            "wr": len(co_winners) / len(group) * 100.0,
            "avg_r": float(np.mean(co_rs)),
        }

    # Grade
    grade_data: dict[str, dict[str, float]] = {}
    for grade_val in [SetupGrade.A, SetupGrade.B]:
        group = [t for t in trades if t.setup_grade == grade_val]
        if group:
            g_rs = [_safe_r(t) for t in group]
            g_winners = [t for t in group if t.net_pnl > 0]
            grade_data[grade_val.value] = {
                "n": len(group),
                "wr": len(g_winners) / len(group) * 100.0,
                "avg_r": float(np.mean(g_rs)),
            }

    # Duration
    bars = [t.bars_held for t in trades]
    hours = [_hold_hours(t) for t in trades]
    duration_data = {
        "avg_bars": float(np.mean(bars)),
        "avg_hours": float(np.mean(hours)),
    }

    # Concentration
    if total_pnl > 0 and winners:
        sorted_w = sorted(winners, key=lambda t: t.net_pnl, reverse=True)
        top1_pct = sorted_w[0].net_pnl / total_pnl * 100.0
        top_n = max(1, len(sorted_w) // 5)
        top20_pnl = sum(t.net_pnl for t in sorted_w[:top_n])
        top20_pct = top20_pnl / total_pnl * 100.0
    else:
        top1_pct = 0.0
        top20_pct = 0.0
    concentration_data = {"top1_pct": top1_pct, "top20_pct": top20_pct}

    # R-stats
    r_stats_data = {
        "mean": float(np.mean(rs)),
        "median": float(np.median(rs)),
        "std": float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0,
        "skew": _skew(rs),
    }

    # Worst/best trades
    def _trade_summary(t: Trade) -> dict[str, Any]:
        return {
            "symbol": t.symbol,
            "direction": t.direction.value,
            "r_multiple": _safe_r(t),
            "pnl": t.net_pnl,
            "exit_reason": t.exit_reason,
            "confirmation_type": t.confirmation_type,
            "bars_held": t.bars_held,
            "mfe_r": t.mfe_r,
            "mae_r": t.mae_r,
        }

    sorted_by_r = sorted(trades, key=lambda t: _safe_r(t))
    worst = [_trade_summary(t) for t in sorted_by_r[:min(5, n)]]
    best = [_trade_summary(t) for t in reversed(sorted_by_r[-min(5, n):])]

    return DiagnosticInsights(
        n_trades=n,
        win_rate=wr,
        mean_r=mean_r,
        profit_factor=pf,
        per_confirmation=per_confirmation,
        per_asset=per_asset,
        exit_attribution=exit_attribution,
        mfe_capture=mfe_capture,
        direction=direction_data,
        confluence=confluence_data,
        grade=grade_data,
        duration=duration_data,
        concentration=concentration_data,
        r_stats=r_stats_data,
        worst_trades=worst,
        best_trades=best,
    )


# ── Helpers ──────────────────────────────────────────────────────────


def _hdr(title: str) -> str:
    """Section header with separator."""
    return f"\n{'═' * 70}\n  {title}\n{'═' * 70}"


def _group_stats(trades: list[Trade]) -> str:
    """Standard stats block: n, WR, mean R, median R, PF, total R."""
    n = len(trades)
    if n == 0:
        return "  n=0"
    winners = [t for t in trades if t.net_pnl > 0]
    wr = len(winners) / n
    rs = _reporting_rs(trades)
    mean_r = float(np.mean(rs)) if rs else 0.0
    median_r = float(np.median(rs)) if rs else 0.0
    total_r = sum(rs)
    gross_p = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_l = abs(sum(t.net_pnl for t in trades if t.net_pnl <= 0))
    pf = gross_p / gross_l if gross_l > 0 else float("inf") if gross_p > 0 else 0.0
    return (
        f"  n={n}, WR={wr:.1%}, Mean R={mean_r:+.3f}, "
        f"Median R={median_r:+.3f}, PF={pf:.2f}, Total R={total_r:+.2f}"
    )


def _ordered_symbols(
    trades: list[Trade],
    expected_symbols: list[str] | None = None,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for sym in expected_symbols or []:
        if sym not in seen:
            ordered.append(sym)
            seen.add(sym)
    for sym in sorted({t.symbol for t in trades}):
        if sym not in seen:
            ordered.append(sym)
            seen.add(sym)
    return ordered


def _trade_signal_variant(trade: Trade) -> str:
    variant = getattr(trade, "signal_variant", None)
    if isinstance(variant, str) and variant.strip():
        return variant.strip()
    return "core"


def _compact_bucket_stats(trades: list[Trade]) -> str:
    if not trades:
        return "n= 0 WR= -- R=  --"
    winners = sum(1 for t in trades if t.net_pnl > 0)
    wr = winners / len(trades)
    avg_r = float(np.mean([_safe_r(t) for t in trades]))
    return f"n={len(trades):>2d} WR={wr:>3.0%} R={avg_r:>+5.2f}"


def _format_counterfactual_r(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):+.2f}R"
    return "--"


def _hold_hours(t: Trade) -> float:
    """Hours between entry and exit."""
    if t.entry_time and t.exit_time:
        return (t.exit_time - t.entry_time).total_seconds() / 3600
    return 0.0


def _reporting_r(t: Trade) -> float | None:
    economic_r = getattr(t, "economic_r_multiple", None)
    if economic_r is not None:
        return float(economic_r)
    realized_r = getattr(t, "realized_r_multiple", None)
    if realized_r is not None:
        return float(realized_r)
    geometric_r = getattr(t, "r_multiple", None)
    if geometric_r is not None:
        return float(geometric_r)
    return None


def _reporting_rs(trades: list[Trade]) -> list[float]:
    rs: list[float] = []
    for trade in trades:
        reporting_r = _reporting_r(trade)
        if reporting_r is not None:
            rs.append(reporting_r)
    return rs


def _safe_r(t: Trade) -> float:
    reporting_r = _reporting_r(t)
    return reporting_r if reporting_r is not None else 0.0


def _geometric_r(t: Trade) -> float:
    return t.r_multiple if t.r_multiple is not None else 0.0


def _terminal_mark_total(terminal_marks: list[TerminalMark] | None) -> float:
    if not terminal_marks:
        return 0.0
    return float(sum(mark.unrealized_pnl_net for mark in terminal_marks))


def _pnl_pct_text(pnl: float, initial_equity: float) -> str:
    if initial_equity > 0:
        return f"({pnl / initial_equity * 100:+.2f}%)"
    return "(n/a)"


def _append_terminal_mark_summary(lines: list[str], terminal_marks: list[TerminalMark] | None) -> None:
    if not terminal_marks:
        return

    total_pnl = _terminal_mark_total(terminal_marks)
    lines.append("")
    lines.append(
        f"  Terminal marks: {len(terminal_marks)} open position(s), "
        f"net liquidation P&L ${total_pnl:,.2f}"
    )
    for mark in terminal_marks:
        r_text = f"{mark.unrealized_r_at_mark:+.3f}R" if mark.unrealized_r_at_mark is not None else "n/a"
        lines.append(
            f"    {mark.symbol:<8s} {mark.direction.value:<5s} "
            f"qty={mark.qty:.6f} raw=${mark.mark_price_raw:,.2f} "
            f"net=${mark.mark_price_net_liquidation:,.2f} "
            f"pnl=${mark.unrealized_pnl_net:,.2f} r={r_text}"
        )


# ── Section 1: Overview ──────────────────────────────────────────────


def _s01_overview(
    trades: list[Trade],
    initial_equity: float,
    terminal_marks: list[TerminalMark] | None = None,
    performance_metrics: PerformanceMetrics | None = None,
) -> str:
    lines = [_hdr("1. Overview")]
    terminal_marks = terminal_marks or []
    n = len(trades)
    if n == 0 and not terminal_marks:
        lines.append("  No realized trades or terminal marks.")
        return "\n".join(lines)

    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]
    rs = [_safe_r(t) for t in trades]
    pnls = [t.net_pnl for t in trades]
    realized_pnl = sum(pnls)
    terminal_pnl = _terminal_mark_total(terminal_marks)
    total_pnl = realized_pnl + terminal_pnl
    mean_r = float(np.mean(rs)) if rs else 0.0
    median_r = float(np.median(rs)) if rs else 0.0
    total_r = sum(rs)
    gross_p = sum(t.net_pnl for t in winners)
    gross_l = abs(sum(t.net_pnl for t in losers))
    pf = gross_p / gross_l if gross_l > 0 else float("inf") if gross_p > 0 else 0.0
    avg_hold_h = float(np.mean([_hold_hours(t) for t in trades])) if trades else 0.0

    lines.append(f"  Closed Trades:  {n}  (W:{len(winners)} / L:{len(losers)})")
    if n > 0:
        lines.append(f"  Win Rate:       {len(winners)/n:.1%}")
        lines.append(f"  Profit Factor:  {pf:.2f}")
        lines.append(f"  Mean R:         {mean_r:+.3f}")
        lines.append(f"  Median R:       {median_r:+.3f}")
        lines.append(f"  Total R:        {total_r:+.2f}")
        lines.append(
            f"  Avg Hold:       {avg_hold_h:.1f}h  "
            f"({float(np.mean([t.bars_held for t in trades])):.1f} bars)"
        )
        if performance_metrics is not None:
            lines.append(f"  Max DD:         {performance_metrics.max_drawdown_pct:.2f}%")
            lines.append(f"  Sharpe Ratio:   {performance_metrics.sharpe_ratio:.2f}")
            lines.append(f"  Calmar Ratio:   {performance_metrics.calmar_ratio:.2f}")
    else:
        lines.append("  Win Rate:       n/a (no realized exits)")
        lines.append("  Profit Factor:  n/a (no realized exits)")

    lines.append(f"  Realized P&L:   ${realized_pnl:,.2f}  {_pnl_pct_text(realized_pnl, initial_equity)}")
    if terminal_marks:
        lines.append(f"  Terminal Marks: {len(terminal_marks)} open position(s)  ${terminal_pnl:,.2f}")
    lines.append(f"  Net Liq P&L:    ${total_pnl:,.2f}  {_pnl_pct_text(total_pnl, initial_equity)}")

    # Quick health check
    lines.append("")
    flags = []
    if n < 30:
        flags.append(f"LOW SAMPLE (n={n}) — statistical significance limited")
    if n > 0 and pf < 1.0:
        flags.append("UNPROFITABLE — profit factor < 1.0")
    if n > 0 and mean_r < 0:
        flags.append(f"NEGATIVE EXPECTANCY — mean R = {mean_r:+.3f}")
    if n > 0 and len(winners) / n < 0.4:
        flags.append(f"LOW WIN RATE ({len(winners)/n:.0%}) — needs high payoff ratio to compensate")

    if terminal_marks:
        flags.append(f"OPEN EXPOSURE AT SAMPLE END ({len(terminal_marks)} marked position(s))")

    if flags:
        lines.append("  ⚠ Flags:")
        for f in flags:
            lines.append(f"    - {f}")
    else:
        lines.append("  ✓ No critical flags")

    return "\n".join(lines)


# ── Section 2: Winner vs Loser Profiles ──────────────────────────────


def _s02_winner_loser_profiles(trades: list[Trade]) -> str:
    lines = [_hdr("2. Winner vs Loser Profiles")]
    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]

    if not winners or not losers:
        lines.append("  Need both winners and losers for comparison.")
        return "\n".join(lines)

    def _avg(ts: list[Trade], fn) -> float:
        vals = [fn(t) for t in ts]
        return float(np.mean(vals)) if vals else 0.0

    lines.append(f"  {'Metric':<28s} {'Winners':>10s} {'Losers':>10s} {'Delta':>10s}")
    lines.append("  " + "-" * 62)

    metrics: list[tuple[str, Any]] = [
        ("Avg R", lambda ts: _avg(ts, _safe_r)),
        ("Avg P&L ($)", lambda ts: _avg(ts, lambda t: t.net_pnl)),
        ("Avg Hold (h)", lambda ts: _avg(ts, _hold_hours)),
        ("Avg Bars Held", lambda ts: _avg(ts, lambda t: t.bars_held)),
        ("Avg MFE R", lambda ts: _avg(ts, lambda t: t.mfe_r or 0)),
        ("Avg MAE R", lambda ts: _avg(ts, lambda t: t.mae_r or 0)),
    ]

    for name, fn in metrics:
        w_val = fn(winners)
        l_val = fn(losers)
        delta = w_val - l_val
        lines.append(f"  {name:<28s} {w_val:>10.3f} {l_val:>10.3f} {delta:>+10.3f}")

    # Grade distribution
    lines.append("")
    lines.append("  Setup Grade Distribution:")
    for grade in [SetupGrade.A, SetupGrade.B]:
        w_count = sum(1 for t in winners if t.setup_grade == grade)
        l_count = sum(1 for t in losers if t.setup_grade == grade)
        lines.append(f"    {grade.value}-grade:  W={w_count}  L={l_count}")

    # Confirmation distribution
    lines.append("")
    lines.append("  Confirmation Distribution:")
    all_confirms = set(t.confirmation_type for t in trades if t.confirmation_type)
    for ctype in sorted(all_confirms):
        w_count = sum(1 for t in winners if t.confirmation_type == ctype)
        l_count = sum(1 for t in losers if t.confirmation_type == ctype)
        lines.append(f"    {ctype:>25s}:  W={w_count}  L={l_count}")

    return "\n".join(lines)


# ── Section 3: MFE/MAE Capture Efficiency ────────────────────────────


def _s03_mfe_capture(trades: list[Trade]) -> str:
    lines = [_hdr("3. MFE/MAE & Capture Efficiency")]

    has_mfe = [t for t in trades if t.mfe_r is not None and t.mfe_r > 0]
    if not has_mfe:
        lines.append("  (no MFE data available)")
        return "\n".join(lines)

    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]

    winner_has_mfe = [t for t in winners if t.mfe_r is not None and t.mfe_r > 0]
    all_efficiencies = [_safe_r(t) / t.mfe_r for t in has_mfe]
    winner_efficiencies = [_safe_r(t) / t.mfe_r for t in winner_has_mfe]
    winner_givebacks = [t.mfe_r - _safe_r(t) for t in winner_has_mfe]

    lines.append(f"  Overall ({len(has_mfe)} trades with MFE data):")
    lines.append(f"    Avg MFE:               {float(np.mean([t.mfe_r for t in has_mfe])):.3f}R")
    lines.append(f"    Avg MAE:               {float(np.mean([t.mae_r or 0 for t in trades])):.3f}R")
    lines.append(
        f"    Winner capture efficiency: {float(np.mean(winner_efficiencies)):.1%}"
        if winner_efficiencies else
        "    Winner capture efficiency: n/a"
    )
    lines.append(f"    All-trades capture:       {float(np.mean(all_efficiencies)):.1%}")
    lines.append(
        f"    Winner avg giveback:      {float(np.mean(winner_givebacks)):.3f}R"
        if winner_givebacks else
        "    Winner avg giveback:      n/a"
    )

    # By exit reason
    by_exit: dict[str, list[Trade]] = defaultdict(list)
    for t in has_mfe:
        by_exit[t.exit_reason or "unknown"].append(t)

    lines.append("")
    lines.append("  Capture by Exit Reason:")
    lines.append(f"    {'Exit Reason':<22s} {'n':>4s} {'Avg MFE':>8s} {'Avg R':>8s} {'Capture':>8s} {'Giveback':>9s}")
    lines.append("    " + "-" * 64)

    for reason, group in sorted(by_exit.items(), key=lambda x: -len(x[1])):
        mfes = [t.mfe_r for t in group]
        rs = [_safe_r(t) for t in group]
        caps = [r / m if m > 0 else 0 for r, m in zip(rs, mfes)]
        gives = [m - r for r, m in zip(rs, mfes)]
        lines.append(
            f"    {reason:<22s} {len(group):>4d} "
            f"{float(np.mean(mfes)):>8.3f} {float(np.mean(rs)):>+8.3f} "
            f"{float(np.mean(caps)):>7.0%} {float(np.mean(gives)):>+9.3f}"
        )

    # Winner giveback analysis
    if winners:
        w_has_mfe = [t for t in winners if t.mfe_r is not None and t.mfe_r > 0]
        if w_has_mfe:
            w_gives = [t.mfe_r - _safe_r(t) for t in w_has_mfe]
            lines.append("")
            lines.append(f"  Winner Giveback: avg {float(np.mean(w_gives)):+.3f}R "
                         f"(range {min(w_gives):.3f} to {max(w_gives):.3f})")

    # Loser MFE (did losers ever go positive?)
    if losers:
        l_has_mfe = [t for t in losers if t.mfe_r is not None and t.mfe_r > 0]
        if l_has_mfe:
            lines.append(f"  Losers with positive MFE: {len(l_has_mfe)}/{len(losers)} "
                         f"(avg peak {float(np.mean([t.mfe_r for t in l_has_mfe])):.3f}R before reversal)")

    return "\n".join(lines)


# ── Section 4: Stop Calibration ──────────────────────────────────────


def _s04_stop_calibration(trades: list[Trade]) -> str:
    lines = [_hdr("4. Stop Calibration")]

    stop_trades = [t for t in trades if t.exit_reason and "stop" in t.exit_reason.lower()]
    non_stop = [t for t in trades if t not in stop_trades]

    lines.append(f"  Stop exits: {len(stop_trades)}/{len(trades)} "
                 f"({len(stop_trades)/len(trades):.0%})" if trades else "  N/A")

    if stop_trades:
        lines.append(f"    {_group_stats(stop_trades)}")
        mae_at_stop = [t.mae_r for t in stop_trades if t.mae_r is not None]
        if mae_at_stop:
            lines.append(f"    Avg MAE at stop: {float(np.mean(mae_at_stop)):.3f}R")
            lines.append(f"    Median MAE:      {float(np.median(mae_at_stop)):.3f}R")

    if non_stop:
        lines.append(f"\n  Non-stop exits: {len(non_stop)}/{len(trades)}")
        lines.append(f"    {_group_stats(non_stop)}")

    # Were stops too tight?
    all_mae = [t.mae_r for t in trades if t.mae_r is not None]
    if all_mae:
        pct_hit_1r = sum(1 for m in all_mae if m <= -1.0) / len(all_mae)
        pct_hit_half = sum(1 for m in all_mae if m <= -0.5) / len(all_mae)
        lines.append("")
        lines.append("  MAE Distribution (stop tightness check):")
        lines.append(f"    MAE <= -1.0R:  {pct_hit_1r:.0%} of trades")
        lines.append(f"    MAE <= -0.5R:  {pct_hit_half:.0%} of trades")
        avg_mae = float(np.mean(all_mae))
        if avg_mae < -0.5:
            lines.append("    → Stops may be too wide (high avg MAE)")
        elif pct_hit_1r < 0.1 and len(stop_trades) / len(trades) > 0.5:
            lines.append("    → Stops rarely hit -1R but many stop exits — consider if trailing is too tight")

    return "\n".join(lines)


# ── Section 5: Exit Reason Attribution ───────────────────────────────


def _s05_exit_attribution(
    trades: list[Trade],
    terminal_marks: list[TerminalMark] | None = None,
) -> str:
    lines = [_hdr("5. Exit Reason Attribution")]
    terminal_marks = terminal_marks or []

    by_exit: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_exit[t.exit_reason or "unknown"].append(t)

    if trades:
        lines.append(f"  {'Exit Reason':<22s} {'n':>4s} {'WR':>6s} {'Avg R':>8s} {'Med R':>8s} {'PF':>6s} {'$ P&L':>10s} {'Share':>6s}")
        lines.append("  " + "-" * 74)

        total_pnl = sum(t.net_pnl for t in trades)
        for reason, group in sorted(by_exit.items(), key=lambda x: -len(x[1])):
            winners = [t for t in group if t.net_pnl > 0]
            rs = [_safe_r(t) for t in group]
            wr = len(winners) / len(group)
            mean_r = float(np.mean(rs))
            median_r = float(np.median(rs))
            gp = sum(t.net_pnl for t in group if t.net_pnl > 0)
            gl = abs(sum(t.net_pnl for t in group if t.net_pnl <= 0))
            pf = gp / gl if gl > 0 else float("inf") if gp > 0 else 0.0
            pnl = sum(t.net_pnl for t in group)
            share = pnl / total_pnl if total_pnl != 0 else 0

            lines.append(
                f"  {reason:<22s} {len(group):>4d} {wr:>5.0%} "
                f"{mean_r:>+8.3f} {median_r:>+8.3f} {pf:>6.2f} "
                f"${pnl:>9,.2f} {share:>5.0%}"
            )
    else:
        lines.append("  No realized exits.")

    _append_terminal_mark_summary(lines, terminal_marks)

    return "\n".join(lines)


# ── Section 6: Streak Analysis ───────────────────────────────────────


def _s06_streaks(trades: list[Trade]) -> str:
    lines = [_hdr("6. Streak Analysis")]
    if not trades:
        return "\n".join(lines)

    results = [t.net_pnl > 0 for t in trades]
    max_win = max_loss = cur_win = cur_loss = 0
    worst_consec_loss_r = 0.0
    cur_loss_r = 0.0
    best_consec_win_r = 0.0
    cur_win_r = 0.0

    for i, is_win in enumerate(results):
        if is_win:
            cur_win += 1
            cur_win_r += _safe_r(trades[i])
            cur_loss = 0
            cur_loss_r = 0.0
            max_win = max(max_win, cur_win)
            best_consec_win_r = max(best_consec_win_r, cur_win_r)
        else:
            cur_loss += 1
            cur_loss_r += _safe_r(trades[i])
            cur_win = 0
            cur_win_r = 0.0
            max_loss = max(max_loss, cur_loss)
            worst_consec_loss_r = min(worst_consec_loss_r, cur_loss_r)

    lines.append(f"  Max win streak:   {max_win}  (best run: {best_consec_win_r:+.2f}R)")
    lines.append(f"  Max loss streak:  {max_loss}  (worst run: {worst_consec_loss_r:+.2f}R)")

    # 1st half vs 2nd half comparison
    mid = len(trades) // 2
    if mid > 0:
        first_half_r = float(np.mean([_safe_r(t) for t in trades[:mid]]))
        second_half_r = float(np.mean([_safe_r(t) for t in trades[mid:]]))
        if second_half_r > first_half_r + 0.05:
            trend = "IMPROVING"
        elif second_half_r < first_half_r - 0.05:
            trend = "DEGRADING"
        else:
            trend = "STABLE"
        lines.append(f"  Trend: {trend} (1st half: {first_half_r:+.3f}R, 2nd half: {second_half_r:+.3f}R)")

    # Sequence visualization
    lines.append("")
    lines.append("  Sequence: " + "".join("W" if r else "L" for r in results))

    return "\n".join(lines)


# ── Section 7: R-Curve Drawdown ──────────────────────────────────────


def _s07_drawdown(trades: list[Trade]) -> str:
    lines = [_hdr("7. R-Curve Drawdown")]
    if not trades:
        return "\n".join(lines)

    rs = np.array([_safe_r(t) for t in trades])
    cum = np.cumsum(rs)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak

    max_dd_r = float(np.min(dd))
    max_dd_idx = int(np.argmin(dd))
    peak_at_dd = float(peak[max_dd_idx])

    lines.append(f"  Max R-drawdown:  {max_dd_r:+.2f}R  (trade #{max_dd_idx + 1})")
    lines.append(f"  Peak before DD:  {peak_at_dd:+.2f}R")
    lines.append(f"  Final cum R:     {float(cum[-1]):+.2f}R")

    # Recovery analysis
    if max_dd_r < 0 and max_dd_idx < len(trades) - 1:
        recovered = False
        for i in range(max_dd_idx + 1, len(trades)):
            if cum[i] >= peak_at_dd:
                recovery_trades = i - max_dd_idx
                lines.append(f"  Recovery:        {recovery_trades} trades to recover from max DD")
                recovered = True
                break
        if not recovered:
            lines.append(f"  Recovery:        NOT YET — still {float(cum[-1] - peak_at_dd):+.2f}R below peak")

    # Drawdown episodes (any DD > 0.5R)
    episodes = []
    in_dd = False
    dd_start = 0
    for i, d in enumerate(dd):
        if d < 0 and not in_dd:
            in_dd = True
            dd_start = i
        elif (d >= 0 or i == len(dd) - 1) and in_dd:
            depth = float(np.min(dd[dd_start:i + 1]))
            duration = i - dd_start
            if depth < -0.3:
                episodes.append((dd_start + 1, i + 1, depth, duration))
            in_dd = False

    if episodes:
        lines.append("")
        lines.append("  Drawdown Episodes (> 0.3R):")
        for start, end, depth, dur in episodes:
            lines.append(f"    Trades #{start}-#{end}: {depth:+.2f}R over {dur} trades")

    return "\n".join(lines)


# ── Section 8: Rolling Expectancy ────────────────────────────────────


def _s08_rolling_expectancy(trades: list[Trade]) -> str:
    lines = [_hdr("8. Rolling Expectancy")]
    window = min(10, len(trades))
    if len(trades) < 5:
        lines.append("  (need >= 5 trades for rolling analysis)")
        return "\n".join(lines)

    rs = [_safe_r(t) for t in trades]
    rolling = []
    for i in range(window - 1, len(rs)):
        rolling.append(float(np.mean(rs[i - window + 1: i + 1])))

    lines.append(f"  Window: {window}-trade rolling average")
    lines.append(f"  Current:  {rolling[-1]:+.3f}R")
    lines.append(f"  Best:     {max(rolling):+.3f}R  (at trade #{rolling.index(max(rolling)) + window})")
    lines.append(f"  Worst:    {min(rolling):+.3f}R  (at trade #{rolling.index(min(rolling)) + window})")

    # Sparkline (ASCII chart)
    n_bins = min(len(rolling), 20)
    if n_bins >= 3:
        step = max(1, len(rolling) // n_bins)
        sampled = [rolling[i] for i in range(0, len(rolling), step)]
        min_v = min(sampled)
        max_v = max(sampled)
        rng = max_v - min_v if max_v != min_v else 1.0
        chart_h = 5
        lines.append("")
        lines.append("  Rolling R sparkline:")
        for row in range(chart_h, -1, -1):
            threshold = min_v + (row / chart_h) * rng
            bar = "  "
            for v in sampled:
                bar += "█" if v >= threshold else " "
            label = f"{threshold:+.2f}" if row in (0, chart_h) else "      "
            lines.append(f"    {label:>6s} |{bar}")

    return "\n".join(lines)


# ── Section 9: Per-Asset Deep Dive ───────────────────────────────────


def _s09_per_asset(
    trades: list[Trade],
    expected_symbols: list[str] | None = None,
) -> str:
    lines = [_hdr("9. Per-Asset Breakdown")]

    by_sym: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_sym[t.symbol].append(t)

    for sym in _ordered_symbols(trades, expected_symbols):
        group = by_sym.get(sym, [])
        lines.append(f"\n  {sym}:")
        lines.append(f"    {_group_stats(group)}")
        # Direction split
        longs = [t for t in group if t.direction == Side.LONG]
        shorts = [t for t in group if t.direction == Side.SHORT]
        if longs:
            lines.append(f"    Long:  n={len(longs)}, WR={sum(1 for t in longs if t.net_pnl > 0)/len(longs):.0%}, "
                         f"avg R={float(np.mean([_safe_r(t) for t in longs])):+.3f}")
        else:
            lines.append("    Long:  n=0, WR=--, avg R=--")
        if shorts:
            lines.append(f"    Short: n={len(shorts)}, WR={sum(1 for t in shorts if t.net_pnl > 0)/len(shorts):.0%}, "
                         f"avg R={float(np.mean([_safe_r(t) for t in shorts])):+.3f}")
        else:
            lines.append("    Short: n=0, WR=--, avg R=--")
        core = [t for t in group if _trade_signal_variant(t) == "core"]
        relaxed = [t for t in group if _trade_signal_variant(t) == "relaxed_body"]
        lines.append(f"    Core:         {_compact_bucket_stats(core)}")
        lines.append(f"    Relaxed body: {_compact_bucket_stats(relaxed)}")
        # MFE/MAE
        mfes = [t.mfe_r for t in group if t.mfe_r is not None]
        maes = [t.mae_r for t in group if t.mae_r is not None]
        if mfes:
            lines.append(f"    Avg MFE: {float(np.mean(mfes)):.3f}R, Avg MAE: {float(np.mean(maes)):.3f}R")

    return "\n".join(lines)


# ── Section 10: Direction Analysis ───────────────────────────────────


def _s10_direction(trades: list[Trade]) -> str:
    lines = [_hdr("10. Direction Analysis")]

    for direction in [Side.LONG, Side.SHORT]:
        group = [t for t in trades if t.direction == direction]
        lines.append(f"\n  {direction.value}:")
        lines.append(f"    {_group_stats(group)}")

        # Exit reason split
        by_exit: dict[str, int] = defaultdict(int)
        for t in group:
            by_exit[t.exit_reason or "unknown"] += 1
        exits_str = ", ".join(f"{k}={v}" for k, v in sorted(by_exit.items(), key=lambda x: -x[1])) or "n/a"
        lines.append(f"    Exits: {exits_str}")

    return "\n".join(lines)


# ── Section 11: Confirmation Type Monotonicity ───────────────────────


def _s11_confirmation(trades: list[Trade]) -> str:
    lines = [_hdr("11. Confirmation Type Analysis")]

    by_type: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_type[t.confirmation_type or "unknown"].append(t)

    lines.append(f"  {'Type':<25s} {'n':>4s} {'WR':>6s} {'Avg R':>8s} {'Med R':>8s} {'Total R':>8s} {'$ P&L':>10s}")
    lines.append("  " + "-" * 68)

    for ctype, group in sorted(by_type.items(), key=lambda x: -len(x[1])):
        winners = [t for t in group if t.net_pnl > 0]
        rs = [_safe_r(t) for t in group]
        wr = len(winners) / len(group)
        lines.append(
            f"  {ctype:<25s} {len(group):>4d} {wr:>5.0%} "
            f"{float(np.mean(rs)):>+8.3f} {float(np.median(rs)):>+8.3f} "
            f"{sum(rs):>+8.2f} ${sum(t.net_pnl for t in group):>9,.2f}"
        )

    return "\n".join(lines)


# ── Section 12: Confluence Monotonicity ──────────────────────────────


def _s12_confluence(trades: list[Trade]) -> str:
    lines = [_hdr("12. Confluence Count → Outcome")]

    by_count: dict[int, list[Trade]] = defaultdict(list)
    for t in trades:
        c = len(t.confluences_used) if t.confluences_used else 0
        by_count[c].append(t)

    lines.append(f"  {'Confluences':>12s} {'n':>4s} {'WR':>6s} {'Avg R':>8s} {'Total R':>8s}")
    lines.append("  " + "-" * 44)

    prev_wr = None
    monotonic = True
    for count in sorted(by_count.keys()):
        group = by_count[count]
        winners = [t for t in group if t.net_pnl > 0]
        wr = len(winners) / len(group)
        rs = [_safe_r(t) for t in group]
        lines.append(
            f"  {count:>12d} {len(group):>4d} {wr:>5.0%} "
            f"{float(np.mean(rs)):>+8.3f} {sum(rs):>+8.2f}"
        )
        if prev_wr is not None and wr < prev_wr - 0.05:
            monotonic = False
        prev_wr = wr

    lines.append("")
    if monotonic:
        lines.append("  ✓ Monotonic: more confluences → higher WR (signal quality is predictive)")
    else:
        lines.append("  ✗ Non-monotonic: confluence count is NOT reliably predictive")

    return "\n".join(lines)


# ── Section 13: Session & Timing ─────────────────────────────────────


def _s13_timing(trades: list[Trade]) -> str:
    lines = [_hdr("13. Session & Timing Patterns")]

    # Hour-of-day
    by_hour: dict[int, list[Trade]] = defaultdict(list)
    for t in trades:
        if t.entry_time:
            by_hour[t.entry_time.hour].append(t)

    if by_hour:
        lines.append("  Entry Hour (UTC):")
        lines.append(f"    {'Hour':>6s} {'n':>4s} {'WR':>6s} {'Avg R':>8s} {'$ P&L':>10s}")
        lines.append("    " + "-" * 38)
        for hour in sorted(by_hour.keys()):
            group = by_hour[hour]
            winners = [t for t in group if t.net_pnl > 0]
            rs = [_safe_r(t) for t in group]
            wr = len(winners) / len(group)
            lines.append(
                f"    {hour:>4d}h {len(group):>4d} {wr:>5.0%} "
                f"{float(np.mean(rs)):>+8.3f} ${sum(t.net_pnl for t in group):>9,.2f}"
            )

    # Day-of-week
    by_dow: dict[str, list[Trade]] = defaultdict(list)
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for t in trades:
        if t.entry_time:
            by_dow[dow_names[t.entry_time.weekday()]].append(t)

    if by_dow:
        lines.append("")
        lines.append("  Day of Week:")
        lines.append(f"    {'Day':>6s} {'n':>4s} {'WR':>6s} {'Avg R':>8s} {'$ P&L':>10s}")
        lines.append("    " + "-" * 38)
        for dow in dow_names:
            if dow not in by_dow:
                continue
            group = by_dow[dow]
            winners = [t for t in group if t.net_pnl > 0]
            rs = [_safe_r(t) for t in group]
            wr = len(winners) / len(group)
            lines.append(
                f"    {dow:>6s} {len(group):>4d} {wr:>5.0%} "
                f"{float(np.mean(rs)):>+8.3f} ${sum(t.net_pnl for t in group):>9,.2f}"
            )

    # Session breakdown (repeated for completeness with full stats)
    def _session(hour: int) -> str:
        if 0 <= hour < 8:
            return "Asia"
        elif 8 <= hour < 13:
            return "London"
        elif 13 <= hour < 16:
            return "Overlap"
        elif 16 <= hour < 21:
            return "New York"
        return "Off-hours"

    by_session: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        if t.entry_time:
            by_session[_session(t.entry_time.hour)].append(t)

    if by_session:
        lines.append("")
        lines.append("  Session Performance:")
        for session in ["Asia", "London", "Overlap", "New York", "Off-hours"]:
            if session not in by_session:
                continue
            group = by_session[session]
            lines.append(f"    {session}:")
            lines.append(f"      {_group_stats(group)}")

    return "\n".join(lines)


# ── Section 14: Trade Duration ───────────────────────────────────────


def _s14_duration(trades: list[Trade]) -> str:
    lines = [_hdr("14. Trade Duration Analysis")]

    bars = [t.bars_held for t in trades]
    hours = [_hold_hours(t) for t in trades]

    lines.append(f"  Bars held:  avg={float(np.mean(bars)):.1f}, "
                 f"median={float(np.median(bars)):.0f}, "
                 f"range=[{min(bars)}, {max(bars)}]")
    lines.append(f"  Hours:      avg={float(np.mean(hours)):.1f}h, "
                 f"median={float(np.median(hours)):.1f}h")

    # Duration vs outcome
    short_trades = [t for t in trades if t.bars_held <= 2]
    medium_trades = [t for t in trades if 2 < t.bars_held <= 5]
    long_trades = [t for t in trades if t.bars_held > 5]

    lines.append("")
    lines.append("  Duration Buckets:")
    for label, group in [("≤2 bars", short_trades), ("3-5 bars", medium_trades), (">5 bars", long_trades)]:
        if group:
            rs = [_safe_r(t) for t in group]
            wr = sum(1 for t in group if t.net_pnl > 0) / len(group)
            lines.append(f"    {label:<10s}: n={len(group)}, WR={wr:.0%}, avg R={float(np.mean(rs)):+.3f}")

    return "\n".join(lines)


# ── Section 15: Concentration Risk ───────────────────────────────────


def _s15_concentration(trades: list[Trade]) -> str:
    lines = [_hdr("15. Concentration & Dependency Risk")]

    winners = [t for t in trades if t.net_pnl > 0]
    total_pnl = sum(t.net_pnl for t in trades)

    if not winners or total_pnl <= 0:
        lines.append("  (no profitable trades to analyze)")
        return "\n".join(lines)

    # Top trade dependency
    sorted_by_pnl = sorted(winners, key=lambda t: t.net_pnl, reverse=True)
    top1_pnl = sorted_by_pnl[0].net_pnl
    top1_share = top1_pnl / total_pnl * 100 if total_pnl > 0 else 0

    lines.append(f"  Largest single winner: ${top1_pnl:,.2f} ({top1_share:.0f}% of total profit)")
    lines.append(f"    Symbol: {sorted_by_pnl[0].symbol}, "
                 f"R={_safe_r(sorted_by_pnl[0]):+.2f}, "
                 f"Direction: {sorted_by_pnl[0].direction.value}")

    # Top N% contribution
    top_n = max(1, len(sorted_by_pnl) // 5)  # top 20%
    top_pnl = sum(t.net_pnl for t in sorted_by_pnl[:top_n])
    lines.append(f"  Top {top_n} trade(s) ({top_n}/{len(winners)} = {top_n/len(winners):.0%}): "
                 f"${top_pnl:,.2f} ({top_pnl/total_pnl*100:.0f}% of total profit)")

    # Symbol concentration
    by_sym: dict[str, float] = defaultdict(float)
    for t in trades:
        by_sym[t.symbol] += t.net_pnl
    dominant_sym = max(by_sym.items(), key=lambda x: x[1])
    lines.append(f"  Symbol concentration: {dominant_sym[0]} contributes "
                 f"${dominant_sym[1]:,.2f} ({dominant_sym[1]/total_pnl*100:.0f}% of total)")

    # Risk assessment
    lines.append("")
    risk_level = "LOW"
    risk_reasons = []
    if top1_share > 50:
        risk_level = "HIGH"
        risk_reasons.append(f"single trade = {top1_share:.0f}% of profit")
    elif top1_share > 30:
        risk_level = "MODERATE"
        risk_reasons.append(f"single trade = {top1_share:.0f}% of profit")

    if len(by_sym) == 1:
        risk_level = "HIGH"
        risk_reasons.append("all profit from single symbol")
    elif dominant_sym[1] / total_pnl > 0.7:
        if risk_level != "HIGH":
            risk_level = "MODERATE"
        risk_reasons.append(f"{dominant_sym[0]} = {dominant_sym[1]/total_pnl*100:.0f}% of profit")

    lines.append(f"  Concentration Risk: {risk_level}")
    for r in risk_reasons:
        lines.append(f"    - {r}")

    return "\n".join(lines)


# ── Section 16: Worst Trades Autopsy ─────────────────────────────────


def _s16_worst_trades(trades: list[Trade]) -> str:
    lines = [_hdr("16. Worst Trades Autopsy")]

    n_worst = min(5, len(trades))
    worst = sorted(trades, key=lambda t: _safe_r(t))[:n_worst]

    for i, t in enumerate(worst):
        lines.append(f"\n  #{i+1} worst: {t.symbol} {t.direction.value}")
        lines.append(f"    R-multiple:    {_safe_r(t):+.3f}")
        lines.append(f"    P&L:           ${t.net_pnl:,.2f}")
        lines.append(f"    Entry/Exit:    ${t.entry_price:,.2f} → ${t.exit_price:,.2f}")
        lines.append(f"    Bars held:     {t.bars_held}")
        lines.append(f"    Hold time:     {_hold_hours(t):.1f}h")
        lines.append(f"    Exit reason:   {t.exit_reason}")
        lines.append(f"    Grade:         {t.setup_grade.value if t.setup_grade else 'N/A'}")
        lines.append(f"    Confirmation:  {t.confirmation_type or 'N/A'}")
        confluences = ", ".join(t.confluences_used) if t.confluences_used else "none"
        lines.append(f"    Confluences:   {confluences}")
        if t.mfe_r is not None:
            lines.append(f"    MFE/MAE:       +{t.mfe_r:.3f}R / {t.mae_r:.3f}R")
        if t.entry_time:
            lines.append(f"    Entry time:    {t.entry_time.strftime('%Y-%m-%d %H:%M')} UTC")

    # Common patterns among worst trades
    lines.append("\n  Common Patterns in Worst Trades:")
    exit_reasons = [t.exit_reason for t in worst]
    confirms = [t.confirmation_type for t in worst if t.confirmation_type]
    symbols = [t.symbol for t in worst]
    grades = [t.setup_grade for t in worst if t.setup_grade]

    from collections import Counter
    for label, items in [("Exit reasons", exit_reasons), ("Confirmations", confirms),
                         ("Symbols", symbols)]:
        counts = Counter(items)
        dominant = counts.most_common(1)[0] if counts else ("N/A", 0)
        if dominant[1] > 1:
            lines.append(f"    {label}: {dominant[0]} appears {dominant[1]}x")

    return "\n".join(lines)


# ── Section 17: Interaction Tables ───────────────────────────────────


def _s17_interactions(
    trades: list[Trade],
    expected_symbols: list[str] | None = None,
) -> str:
    lines = [_hdr("17. Interaction Analysis")]
    symbols = _ordered_symbols(trades, expected_symbols)

    # Grade × Direction
    lines.append("  Grade x Direction:")
    lines.append(f"    {'':>10s} {'Long':>18s} {'Short':>18s}")
    lines.append("    " + "-" * 48)
    for grade in [SetupGrade.A, SetupGrade.B]:
        row = f"    {grade.value + '-grade':>10s}"
        for direction in [Side.LONG, Side.SHORT]:
            group = [t for t in trades if t.setup_grade == grade and t.direction == direction]
            if group:
                wr = sum(1 for t in group if t.net_pnl > 0) / len(group)
                avg_r = float(np.mean([_safe_r(t) for t in group]))
                row += f"  n={len(group):>2d} WR={wr:>3.0%} R={avg_r:>+.2f}"
            else:
                row += f"  {'--':>17s}"
        lines.append(row)

    # Confirmation × Symbol
    confirms = sorted(set(t.confirmation_type for t in trades if t.confirmation_type))
    if confirms and symbols:
        lines.append("")
        lines.append("  Confirmation x Symbol:")
        header = f"    {'':>25s}" + "".join(f" {s:>12s}" for s in symbols)
        lines.append(header)
        lines.append("    " + "-" * (25 + 13 * len(symbols)))
        for ctype in confirms:
            row = f"    {ctype:>25s}"
            for sym in symbols:
                group = [t for t in trades if t.confirmation_type == ctype and t.symbol == sym]
                if group:
                    wr = sum(1 for t in group if t.net_pnl > 0) / len(group)
                    avg_r = float(np.mean([_safe_r(t) for t in group]))
                    row += f"  {len(group):>2d}/{wr:>3.0%}/{avg_r:>+.2f}"
                else:
                    row += f" {'--':>12s}"
            lines.append(row)

    for variant, title in [("core", "Core"), ("relaxed_body", "Relaxed Body")]:
        lines.append("")
        lines.append(f"  Symbol x Direction x Signal Variant ({title}):")
        lines.append(f"    {'Symbol':<8s} {'Long':>20s} {'Short':>20s}")
        lines.append("    " + "-" * 52)
        for sym in symbols:
            row = f"    {sym:<8s}"
            for direction in [Side.LONG, Side.SHORT]:
                group = [
                    t for t in trades
                    if t.symbol == sym
                    and t.direction == direction
                    and _trade_signal_variant(t) == variant
                ]
                row += f"  {_compact_bucket_stats(group):>20s}"
            lines.append(row)

    return "\n".join(lines)


# ── Section 18: Friction Analysis ──────────────────────────────────


def _s18_friction(trades: list[Trade]) -> str:
    lines = [_hdr("18. Friction Analysis (Commission + Funding)")]

    if not trades:
        lines.append("  No trades.")
        return "\n".join(lines)

    comms = [t.commission for t in trades]
    funds = [t.funding_paid for t in trades]
    total_comm = sum(comms)
    total_fund = sum(funds)
    total_friction = total_comm + total_fund
    total_pnl = sum(t.net_pnl for t in trades)
    gross_pnl = total_pnl + total_friction  # P&L before friction

    lines.append(f"  Total commissions:   ${total_comm:>10,.2f}  "
                 f"(avg ${total_comm/len(trades):,.2f}/trade)")
    lines.append(f"  Total funding:       ${total_fund:>10,.2f}  "
                 f"(avg ${total_fund/len(trades):,.2f}/trade)")
    lines.append(f"  Total friction:      ${total_friction:>10,.2f}")
    lines.append(f"  Net P&L:             ${total_pnl:>10,.2f}")
    lines.append(f"  Gross P&L (pre-fric):${gross_pnl:>10,.2f}")

    if abs(gross_pnl) > 0:
        drag_pct = total_friction / abs(gross_pnl) * 100
        lines.append(f"  Friction as % of gross: {drag_pct:.1f}%")

    # Per-symbol friction
    by_sym: dict[str, dict[str, float]] = defaultdict(lambda: {"comm": 0.0, "fund": 0.0, "n": 0})
    for t in trades:
        by_sym[t.symbol]["comm"] += t.commission
        by_sym[t.symbol]["fund"] += t.funding_paid
        by_sym[t.symbol]["n"] += 1

    if len(by_sym) > 1:
        lines.append("")
        lines.append("  Per-Symbol Friction:")
        lines.append(f"    {'Symbol':<6s} {'n':>4s} {'Comm':>10s} {'Funding':>10s} {'Total':>10s}")
        lines.append("    " + "-" * 44)
        for sym in sorted(by_sym.keys()):
            d = by_sym[sym]
            lines.append(f"    {sym:<6s} {int(d['n']):>4d} "
                         f"${d['comm']:>9,.2f} ${d['fund']:>9,.2f} "
                         f"${d['comm']+d['fund']:>9,.2f}")

    # Funding direction analysis
    adverse = [t for t in trades if t.funding_paid > 0]
    favorable = [t for t in trades if t.funding_paid < 0]
    if adverse or favorable:
        lines.append("")
        lines.append(f"  Funding direction: {len(adverse)} adverse, "
                     f"{len(favorable)} favorable, "
                     f"{len(trades) - len(adverse) - len(favorable)} neutral")

    return "\n".join(lines)


# ── Section 19: Weekly P&L Calendar ─────────────────────────────────


def _s19_weekly_pnl(trades: list[Trade]) -> str:
    lines = [_hdr("19. Weekly P&L Calendar")]

    if not trades:
        lines.append("  No trades.")
        return "\n".join(lines)

    # Group by ISO week
    by_week: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        if t.exit_time:
            iso = t.exit_time.isocalendar()
            by_week[f"{iso[0]}-W{iso[1]:02d}"].append(t)

    if not by_week:
        lines.append("  No exit timestamps.")
        return "\n".join(lines)

    lines.append(f"  {'Week':<10s} {'n':>3s} {'WR':>5s} {'R':>8s} {'P&L':>10s} "
                 f"{'Cum P&L':>10s} {'Bar':>20s}")
    lines.append("  " + "-" * 70)

    cum_pnl = 0.0
    max_weekly = 0.0
    for week in sorted(by_week.keys()):
        group = by_week[week]
        week_pnl = sum(t.net_pnl for t in group)
        cum_pnl += week_pnl
        rs = [_safe_r(t) for t in group]
        wr = sum(1 for t in group if t.net_pnl > 0) / len(group)
        max_weekly = max(max_weekly, abs(week_pnl))

    # Second pass with bar chart
    cum_pnl = 0.0
    for week in sorted(by_week.keys()):
        group = by_week[week]
        week_pnl = sum(t.net_pnl for t in group)
        cum_pnl += week_pnl
        rs = [_safe_r(t) for t in group]
        wr = sum(1 for t in group if t.net_pnl > 0) / len(group)
        bar_len = int(abs(week_pnl) / max_weekly * 15) if max_weekly > 0 else 0
        bar = ("+" * bar_len) if week_pnl >= 0 else ("-" * bar_len)
        lines.append(
            f"  {week:<10s} {len(group):>3d} {wr:>4.0%} "
            f"{sum(rs):>+7.2f} ${week_pnl:>9,.2f} "
            f"${cum_pnl:>9,.2f}  {bar}"
        )

    # Summary
    weekly_pnls = [sum(t.net_pnl for t in by_week[w]) for w in sorted(by_week.keys())]
    pos_weeks = sum(1 for p in weekly_pnls if p > 0)
    neg_weeks = sum(1 for p in weekly_pnls if p <= 0)
    lines.append("")
    lines.append(f"  Positive weeks: {pos_weeks}/{len(weekly_pnls)}  "
                 f"Negative weeks: {neg_weeks}/{len(weekly_pnls)}")
    if weekly_pnls:
        lines.append(f"  Best week: ${max(weekly_pnls):,.2f}  "
                     f"Worst week: ${min(weekly_pnls):,.2f}")

    return "\n".join(lines)


# ── Section 20: Best Trades Autopsy ─────────────────────────────────


def _s20_best_trades(trades: list[Trade]) -> str:
    lines = [_hdr("20. Best Trades Autopsy")]

    winners = [t for t in trades if t.net_pnl > 0]
    if not winners:
        lines.append("  No winning trades.")
        return "\n".join(lines)

    n_best = min(5, len(winners))
    best = sorted(winners, key=lambda t: _safe_r(t), reverse=True)[:n_best]

    for i, t in enumerate(best):
        lines.append(f"\n  #{i+1} best: {t.symbol} {t.direction.value}")
        lines.append(f"    R-multiple:    {_safe_r(t):+.3f}")
        lines.append(f"    P&L:           ${t.net_pnl:,.2f}")
        lines.append(f"    Entry/Exit:    ${t.entry_price:,.2f} → ${t.exit_price:,.2f}")
        lines.append(f"    Bars held:     {t.bars_held}")
        lines.append(f"    Hold time:     {_hold_hours(t):.1f}h")
        lines.append(f"    Exit reason:   {t.exit_reason}")
        lines.append(f"    Grade:         {t.setup_grade.value if t.setup_grade else 'N/A'}")
        lines.append(f"    Confirmation:  {t.confirmation_type or 'N/A'}")
        confluences = ", ".join(t.confluences_used) if t.confluences_used else "none"
        lines.append(f"    Confluences:   {confluences}")
        if t.mfe_r is not None:
            eff = _safe_r(t) / t.mfe_r * 100 if t.mfe_r > 0 else 0.0
            lines.append(f"    MFE/MAE:       +{t.mfe_r:.3f}R / {t.mae_r:.3f}R  (captured {eff:.0f}%)")
        if t.entry_time:
            lines.append(f"    Entry time:    {t.entry_time.strftime('%Y-%m-%d %H:%M')} UTC")

    # Common patterns among best trades
    lines.append("\n  Common Patterns in Best Trades:")
    from collections import Counter
    for label, items in [
        ("Exit reasons", [t.exit_reason for t in best]),
        ("Confirmations", [t.confirmation_type for t in best if t.confirmation_type]),
        ("Symbols", [t.symbol for t in best]),
    ]:
        counts = Counter(items)
        dominant = counts.most_common(1)[0] if counts else ("N/A", 0)
        if dominant[1] > 1:
            lines.append(f"    {label}: {dominant[0]} appears {dominant[1]}x")

    return "\n".join(lines)


# ── Section 21: Risk & Sizing Analysis ──────────────────────────────


def _s21_risk_sizing(trades: list[Trade]) -> str:
    lines = [_hdr("21. Risk & Sizing Analysis")]

    if not trades:
        lines.append("  No trades.")
        return "\n".join(lines)

    # Position value at entry
    position_values = [t.entry_price * t.qty for t in trades]
    lines.append(f"  Position value at entry:")
    lines.append(f"    Avg:    ${float(np.mean(position_values)):>10,.2f}")
    lines.append(f"    Median: ${float(np.median(position_values)):>10,.2f}")
    lines.append(f"    Range:  ${min(position_values):>10,.2f} — ${max(position_values):>10,.2f}")

    # R vs Dollar divergence
    lines.append("")
    lines.append("  Geometric R-Multiple vs Dollar P&L Alignment:")
    mismatches = []
    for t in trades:
        r = _geometric_r(t)
        if (r > 0 and t.net_pnl <= 0) or (r < 0 and t.net_pnl > 0) or (r == 0 and abs(t.net_pnl) > 1):
            mismatches.append(t)

    if mismatches:
        lines.append(f"    {len(mismatches)} trade(s) where R and $ P&L disagree:")
        for t in mismatches:
            lines.append(f"      {t.symbol} {t.direction.value}: "
                         f"R={_geometric_r(t):+.3f}, P&L=${t.net_pnl:,.2f}")
    else:
        lines.append("    All trades: R and $ P&L directionally consistent")

    # Dollar impact vs R impact
    rs = [_geometric_r(t) for t in trades]
    pnls = [t.net_pnl for t in trades]
    if len(trades) >= 3:
        # Rank correlation (Spearman-like)
        r_ranks = np.argsort(np.argsort(rs)).astype(float)
        p_ranks = np.argsort(np.argsort(pnls)).astype(float)
        n = len(trades)
        d_sq = sum((r_ranks[i] - p_ranks[i]) ** 2 for i in range(n))
        rho = 1 - 6 * d_sq / (n * (n * n - 1))
        lines.append(f"    R vs $ rank correlation: {rho:.3f} "
                     f"({'strong' if abs(rho) > 0.8 else 'moderate' if abs(rho) > 0.5 else 'weak'})")

    # Largest dollar losses vs R losses
    worst_by_dollar = sorted(trades, key=lambda t: t.net_pnl)[:3]
    worst_by_r = sorted(trades, key=lambda t: _geometric_r(t))[:3]
    dollar_ids = set(id(t) for t in worst_by_dollar)
    r_ids = set(id(t) for t in worst_by_r)
    overlap = len(dollar_ids & r_ids)
    lines.append(f"    Worst-3 overlap ($ vs R): {overlap}/3 "
                 f"({'sizing consistent' if overlap >= 2 else 'sizing may distort'})")

    return "\n".join(lines)


# ── Section 22: Entry Method Analysis ───────────────────────────────


def _s22_entry_method(trades: list[Trade]) -> str:
    lines = [_hdr("22. Entry Method Analysis")]

    by_method: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_method[t.entry_method or "unknown"].append(t)

    if len(by_method) <= 1 and "unknown" in by_method:
        lines.append("  No entry method data recorded.")
        return "\n".join(lines)

    lines.append(f"  {'Method':<20s} {'n':>4s} {'WR':>6s} {'Avg R':>8s} "
                 f"{'Med R':>8s} {'PF':>6s} {'$ P&L':>10s}")
    lines.append("  " + "-" * 66)

    for method, group in sorted(by_method.items(), key=lambda x: -len(x[1])):
        winners = [t for t in group if t.net_pnl > 0]
        rs = [_safe_r(t) for t in group]
        wr = len(winners) / len(group)
        gross_p = sum(t.net_pnl for t in group if t.net_pnl > 0)
        gross_l = abs(sum(t.net_pnl for t in group if t.net_pnl <= 0))
        pf = gross_p / gross_l if gross_l > 0 else float("inf") if gross_p > 0 else 0.0
        lines.append(
            f"  {method:<20s} {len(group):>4d} {wr:>5.0%} "
            f"{float(np.mean(rs)):>+8.3f} {float(np.median(rs)):>+8.3f} "
            f"{pf:>6.2f} ${sum(t.net_pnl for t in group):>9,.2f}"
        )

    # MFE comparison by method
    methods_with_mfe = {}
    for method, group in by_method.items():
        winner_group = [t for t in group if t.net_pnl > 0]
        mfes = [t.mfe_r for t in winner_group if t.mfe_r is not None and t.mfe_r > 0]
        if mfes:
            caps = [_safe_r(t) / t.mfe_r for t in winner_group if t.mfe_r and t.mfe_r > 0]
            methods_with_mfe[method] = float(np.mean(caps))

    if len(methods_with_mfe) > 1:
        lines.append("")
        lines.append("  Winner Capture Efficiency by Method:")
        for method, eff in sorted(methods_with_mfe.items(), key=lambda x: -x[1]):
            lines.append(f"    {method:<20s} {eff:.1%}")

    return "\n".join(lines)


# ── Verdict ──────────────────────────────────────────────────────────


def _s23_blocked_relaxed_body_audit(
    trades: list[Trade],
    expected_symbols: list[str] | None = None,
    diagnostic_context: dict[str, Any] | None = None,
) -> str:
    lines = [_hdr("23. Blocked Relaxed-Body Audit")]
    diagnostic_context = diagnostic_context or {}

    audit_payload = diagnostic_context.get("blocked_relaxed_body_audit")
    raw_signals = diagnostic_context.get("blocked_relaxed_body_signals")

    signals: list[dict[str, Any]] = []
    source = ""
    match_window_bars: Any = None
    if isinstance(audit_payload, dict):
        candidate_signals = audit_payload.get("signals", [])
        if isinstance(candidate_signals, list):
            signals = [row for row in candidate_signals if isinstance(row, dict)]
        source = str(audit_payload.get("source", "") or "")
        match_window_bars = audit_payload.get("match_window_bars")
    elif isinstance(raw_signals, list):
        signals = [row for row in raw_signals if isinstance(row, dict)]

    if not signals:
        lines.append("  No blocked relaxed-body signals recorded.")
        return "\n".join(lines)

    symbols = _ordered_symbols(trades, expected_symbols)
    if source:
        lines.append(f"  Counterfactual source: {source}")
    if isinstance(match_window_bars, int):
        lines.append(f"  Counterfactual match window: {match_window_bars} bars")

    lines.append("")
    lines.append("  Symbol x Direction summary:")
    lines.append(
        f"    {'Symbol':<8s} {'Dir':<6s} {'Blocked':>7s} {'Matched':>7s} "
        f"{'Unmatched':>9s} {'WR':>6s} {'Avg R':>8s}"
    )
    lines.append("  " + "-" * 62)
    for sym in symbols:
        for direction in [Side.LONG.value, Side.SHORT.value]:
            group = [
                row for row in signals
                if str(row.get("symbol", "")).upper() == sym.upper()
                and str(row.get("direction", "")).upper() == direction
            ]
            matched = [
                row for row in group
                if str(row.get("counterfactual_status", "")).lower() == "matched_trade"
            ]
            matched_rs = [
                float(row["counterfactual_r_multiple"])
                for row in matched
                if isinstance(row.get("counterfactual_r_multiple"), (int, float))
            ]
            matched_winners = sum(1 for value in matched_rs if value > 0)
            wr_text = f"{matched_winners / len(matched_rs):.0%}" if matched_rs else "--"
            avg_r_text = f"{float(np.mean(matched_rs)):+.2f}" if matched_rs else "--"
            lines.append(
                f"    {sym:<8s} {direction:<6s} {len(group):>7d} {len(matched):>7d} "
                f"{len(group) - len(matched):>9d} {wr_text:>6s} {avg_r_text:>8s}"
            )

    lines.append("")
    lines.append("  Detailed blocked signals:")
    def _signal_sort_key(row: dict[str, Any]) -> tuple[datetime, str, str]:
        signal_time = row.get("signal_time")
        if isinstance(signal_time, datetime):
            normalized = (
                signal_time.astimezone(timezone.utc)
                if signal_time.tzinfo is not None
                else signal_time.replace(tzinfo=timezone.utc)
            )
        else:
            normalized = datetime.min.replace(tzinfo=timezone.utc)
        return (
            normalized,
            str(row.get("symbol", "")),
            str(row.get("direction", "")),
        )

    for row in sorted(signals, key=_signal_sort_key):
        signal_time = row.get("signal_time")
        if isinstance(signal_time, datetime):
            signal_time_text = signal_time.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
        else:
            signal_time_text = str(signal_time or "n/a")
        direction = str(row.get("direction", ""))
        symbol = str(row.get("symbol", ""))
        blocked_rule = str(row.get("blocked_rule", ""))
        conf_count = row.get("confluence_count")
        body_ratio = row.get("body_ratio")
        room_r = row.get("room_r")
        status = str(row.get("counterfactual_status", "no_counterfactual_trade"))
        body_ratio_text = f"{float(body_ratio):.2f}" if isinstance(body_ratio, (int, float)) else "--"
        room_r_text = f"{float(room_r):.2f}" if isinstance(room_r, (int, float)) else "--"
        if status == "matched_trade":
            entry_time = row.get("counterfactual_entry_time")
            if isinstance(entry_time, datetime):
                entry_time_text = entry_time.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
            else:
                entry_time_text = str(entry_time or "n/a")
            lines.append(
                "    "
                f"{signal_time_text} {symbol} {direction} "
                f"conf={conf_count} body={body_ratio_text} room={room_r_text} "
                f"rule={blocked_rule} -> matched {entry_time_text} "
                f"{_format_counterfactual_r(row.get('counterfactual_r_multiple'))} "
                f"{row.get('counterfactual_exit_reason', 'unknown')}"
            )
        else:
            lines.append(
                "    "
                f"{signal_time_text} {symbol} {direction} "
                f"conf={conf_count} body={body_ratio_text} room={room_r_text} "
                f"rule={blocked_rule} -> no counterfactual trade"
            )

    return "\n".join(lines)


def _verdict(
    trades: list[Trade],
    initial_equity: float,
    terminal_marks: list[TerminalMark] | None = None,
) -> str:
    lines = [_hdr("VERDICT & RECOMMENDATIONS")]
    terminal_marks = terminal_marks or []

    if not trades and not terminal_marks:
        lines.append("  No trades or terminal marks to assess.")
        return "\n".join(lines)

    n = len(trades)
    winners = [t for t in trades if t.net_pnl > 0]
    rs = [_safe_r(t) for t in trades]
    realized_pnl = sum(t.net_pnl for t in trades)
    terminal_pnl = _terminal_mark_total(terminal_marks)
    total_pnl = realized_pnl + terminal_pnl
    wr = len(winners) / n if n > 0 else 0.0
    mean_r = float(np.mean(rs)) if rs else 0.0
    gross_p = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_l = abs(sum(t.net_pnl for t in trades if t.net_pnl <= 0))
    pf = gross_p / gross_l if gross_l > 0 else float("inf") if gross_p > 0 else 0.0

    lines.append(
        f"  Net liquidation P&L: ${total_pnl:,.2f} "
        f"{_pnl_pct_text(total_pnl, initial_equity)}"
    )
    lines.append(f"  Realized closed-trade P&L: ${realized_pnl:,.2f}")
    if terminal_marks:
        lines.append(
            f"  Terminal marked open-position P&L: ${terminal_pnl:,.2f} "
            f"across {len(terminal_marks)} position(s)"
        )

    # Strengths
    strengths = []
    if n > 0 and wr > 0.65:
        strengths.append(f"High win rate ({wr:.0%})")
    if n > 0 and pf > 2.0:
        strengths.append(f"Strong profit factor ({pf:.2f})")
    if n > 0 and mean_r > 0.2:
        strengths.append(f"Good expectancy ({mean_r:+.3f}R)")
    if terminal_marks and terminal_pnl > 0:
        strengths.append(f"Open exposure added ${terminal_pnl:,.2f} at sample end")

    # Per-confirmation check
    by_conf: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_conf[t.confirmation_type or "unknown"].append(t)
    for ctype, group in by_conf.items():
        c_wr = sum(1 for t in group if t.net_pnl > 0) / len(group)
        if c_wr > 0.75 and len(group) >= 3:
            strengths.append(f"{ctype} confirmation strong ({c_wr:.0%} WR, n={len(group)})")

    # Confluence monotonicity check
    by_confl: dict[int, list[Trade]] = defaultdict(list)
    for t in trades:
        by_confl[len(t.confluences_used) if t.confluences_used else 0].append(t)
    confl_wrs = [(c, sum(1 for t in g if t.net_pnl > 0) / len(g)) for c, g in sorted(by_confl.items())]
    if len(confl_wrs) >= 2 and all(confl_wrs[i][1] <= confl_wrs[i+1][1] for i in range(len(confl_wrs)-1)):
        strengths.append("Confluence count is monotonically predictive")

    # Weaknesses
    weaknesses = []
    if n < 30:
        weaknesses.append(f"Low sample size (n={n}) — results not statistically robust")
    if n > 0 and mean_r < 0.1:
        weaknesses.append(f"Low expectancy ({mean_r:+.3f}R) — small edge")

    if terminal_marks and terminal_pnl < 0:
        weaknesses.append(f"Open exposure detracted ${abs(terminal_pnl):,.2f} at sample end")

    # MFE capture check
    has_mfe = [t for t in winners if t.mfe_r and t.mfe_r > 0]
    if has_mfe:
        caps = [_safe_r(t) / t.mfe_r for t in has_mfe if t.mfe_r > 0]
        avg_cap = float(np.mean(caps))
        if avg_cap < 0.4:
            weaknesses.append(f"Low capture efficiency ({avg_cap:.0%}) — leaving R on the table")
        gives = [t.mfe_r - _safe_r(t) for t in has_mfe]
        avg_give = float(np.mean(gives))
        if avg_give > 0.3:
            weaknesses.append(f"High giveback ({avg_give:+.3f}R avg) — exits too late or trail too slow")

    # Underperforming confirmations
    for ctype, group in by_conf.items():
        c_wr = sum(1 for t in group if t.net_pnl > 0) / len(group)
        c_avg_r = float(np.mean([_safe_r(t) for t in group]))
        if c_wr < 0.5 and len(group) >= 3:
            weaknesses.append(f"{ctype} confirmation underperforms ({c_wr:.0%} WR, {c_avg_r:+.3f}R)")

    # Underperforming symbols
    by_sym: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_sym[t.symbol].append(t)
    for sym, group in by_sym.items():
        sym_pnl = sum(t.net_pnl for t in group)
        if sym_pnl < 0 and len(group) >= 3:
            sym_wr = sum(1 for t in group if t.net_pnl > 0) / len(group)
            weaknesses.append(f"{sym} is net negative (${sym_pnl:,.2f}, {sym_wr:.0%} WR)")

    # Concentration check
    if total_pnl > 0:
        sorted_w = sorted(winners, key=lambda t: t.net_pnl, reverse=True)
        top1_share = sorted_w[0].net_pnl / total_pnl if sorted_w else 0
        if top1_share > 0.5:
            weaknesses.append(f"Profit concentrated — top trade = {top1_share:.0%} of total")

    # Friction check
    total_friction = sum(t.commission + t.funding_paid for t in trades)
    if total_friction > 0 and abs(total_pnl) > 0:
        friction_ratio = total_friction / abs(total_pnl)
        if friction_ratio > 0.3:
            weaknesses.append(f"High friction drag ({friction_ratio:.0%} of |P&L|) — "
                              f"${total_friction:,.2f} in commissions + funding")

    # Recommendations
    recommendations = []
    if mean_r < 0.15 and avg_cap < 0.5 if has_mfe else False:
        recommendations.append("Consider widening trail to capture more MFE")
    for ctype, group in by_conf.items():
        c_wr = sum(1 for t in group if t.net_pnl > 0) / len(group)
        c_avg_r = float(np.mean([_safe_r(t) for t in group]))
        if c_wr < 0.45 and c_avg_r < 0 and len(group) >= 3:
            recommendations.append(f"Consider filtering or tightening {ctype} entries")
    for sym, group in by_sym.items():
        sym_pnl = sum(t.net_pnl for t in group)
        if sym_pnl < -20 and len(group) >= 5:
            recommendations.append(f"Review {sym} — consistent underperformance")
    if n < 30:
        recommendations.append("Extend data period to increase sample size before drawing conclusions")
    if not recommendations:
        recommendations.append("No critical changes recommended — continue monitoring")

    if terminal_marks:
        recommendations.append(
            "Keep account-level comparisons on net liquidation value and treat terminal marks as unrealized."
        )

    # Output
    lines.append("")
    lines.append("  STRENGTHS:")
    if strengths:
        for s in strengths:
            lines.append(f"    + {s}")
    else:
        lines.append("    (none identified)")

    lines.append("")
    lines.append("  WEAKNESSES:")
    if weaknesses:
        for w in weaknesses:
            lines.append(f"    - {w}")
    else:
        lines.append("    (none identified)")

    lines.append("")
    lines.append("  RECOMMENDATIONS:")
    for r in recommendations:
        lines.append(f"    → {r}")

    return "\n".join(lines)


# ── R-Distribution (enhanced) ────────────────────────────────────────


def _r_distribution(trades: list[Trade]) -> str:
    lines = [_hdr("R-Multiple Distribution")]
    rs = [_safe_r(t) for t in trades]
    if not rs:
        return "\n".join(lines)

    buckets = [
        ("< -1.0", lambda r: r < -1.0),
        ("-1.0 to -0.5", lambda r: -1.0 <= r < -0.5),
        ("-0.5 to 0", lambda r: -0.5 <= r < 0),
        ("0 to 0.5", lambda r: 0 <= r < 0.5),
        ("0.5 to 1.0", lambda r: 0.5 <= r < 1.0),
        ("1.0 to 2.0", lambda r: 1.0 <= r < 2.0),
        ("> 2.0", lambda r: r >= 2.0),
    ]

    max_count = max(sum(1 for r in rs if fn(r)) for _, fn in buckets)
    scale = 40 / max_count if max_count > 0 else 1

    for label, fn in buckets:
        count = sum(1 for r in rs if fn(r))
        bar = "█" * int(count * scale)
        lines.append(f"  {label:>14s} | {count:>3d} {bar}")

    lines.append("")
    lines.append(f"  Mean: {float(np.mean(rs)):+.3f}  Median: {float(np.median(rs)):+.3f}  "
                 f"Std: {float(np.std(rs)):.3f}  Skew: {_skew(rs):+.2f}")

    return "\n".join(lines)


def _skew(values: list[float]) -> float:
    """Compute skewness."""
    if len(values) < 3:
        return 0.0
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if std == 0:
        return 0.0
    return float(n / ((n - 1) * (n - 2)) * sum(((v - mean) / std) ** 3 for v in values))


# ── Diagnostic Module Registry ──────────────────────────────────────

# Each module groups related section functions for phase-targeted diagnostics.
# Functions that require initial_equity are marked in _EQUITY_SECTIONS.
DIAGNOSTIC_MODULES: dict[str, list[str]] = {
    "D1": ["_s03_mfe_capture", "_s04_stop_calibration", "_s14_duration"],
    "D2": ["_s05_exit_attribution", "_s16_worst_trades", "_s20_best_trades"],
    "D3": ["_s07_drawdown", "_s06_streaks", "_s21_risk_sizing", "_s15_concentration", "_s08_rolling_expectancy", "_s18_friction"],
    "D4": ["_s11_confirmation", "_s12_confluence", "_s22_entry_method", "_s09_per_asset", "_s23_blocked_relaxed_body_audit"],
    "D5": ["_s10_direction", "_s13_timing", "_s17_interactions", "_s19_weekly_pnl"],
    "D6": ["_s01_overview", "_s02_winner_loser_profiles", "_r_distribution", "_verdict"],
}

# Section functions that take (trades, initial_equity) instead of just (trades)
_EQUITY_SECTIONS = {"_s01_overview", "_verdict"}

# Lookup from function name to actual callable
_SECTION_FUNCTIONS: dict[str, Any] = {
    "_s01_overview": _s01_overview,
    "_s02_winner_loser_profiles": _s02_winner_loser_profiles,
    "_s03_mfe_capture": _s03_mfe_capture,
    "_s04_stop_calibration": _s04_stop_calibration,
    "_s05_exit_attribution": _s05_exit_attribution,
    "_s06_streaks": _s06_streaks,
    "_s07_drawdown": _s07_drawdown,
    "_s08_rolling_expectancy": _s08_rolling_expectancy,
    "_s09_per_asset": _s09_per_asset,
    "_s10_direction": _s10_direction,
    "_s11_confirmation": _s11_confirmation,
    "_s12_confluence": _s12_confluence,
    "_s13_timing": _s13_timing,
    "_s14_duration": _s14_duration,
    "_s15_concentration": _s15_concentration,
    "_s16_worst_trades": _s16_worst_trades,
    "_s17_interactions": _s17_interactions,
    "_s18_friction": _s18_friction,
    "_s19_weekly_pnl": _s19_weekly_pnl,
    "_s20_best_trades": _s20_best_trades,
    "_s21_risk_sizing": _s21_risk_sizing,
    "_s22_entry_method": _s22_entry_method,
    "_s23_blocked_relaxed_body_audit": _s23_blocked_relaxed_body_audit,
    "_r_distribution": _r_distribution,
    "_verdict": _verdict,
}


def _render_section(
    fn_name: str,
    trades: list[Trade],
    initial_equity: float,
    terminal_marks: list[TerminalMark] | None,
    performance_metrics: PerformanceMetrics | None = None,
    expected_symbols: list[str] | None = None,
    diagnostic_context: dict[str, Any] | None = None,
) -> str:
    fn = _SECTION_FUNCTIONS.get(fn_name)
    if fn is None:
        return ""
    if fn_name == "_s01_overview":
        return fn(trades, initial_equity, terminal_marks, performance_metrics)
    if fn_name == "_s05_exit_attribution":
        return fn(trades, terminal_marks)
    if fn_name == "_s09_per_asset":
        return fn(trades, expected_symbols)
    if fn_name == "_s17_interactions":
        return fn(trades, expected_symbols)
    if fn_name == "_s23_blocked_relaxed_body_audit":
        return fn(trades, expected_symbols, diagnostic_context)
    if fn_name == "_verdict":
        return fn(trades, initial_equity, terminal_marks)
    if fn_name in _EQUITY_SECTIONS:
        return fn(trades, initial_equity)
    return fn(trades)


def generate_phase_diagnostics(
    trades: list[Trade],
    modules: list[str],
    initial_equity: float = 10_000.0,
    title: str = "",
    terminal_marks: list[TerminalMark] | None = None,
    performance_metrics: PerformanceMetrics | None = None,
    expected_symbols: list[str] | None = None,
    diagnostic_context: dict[str, Any] | None = None,
) -> str:
    """Generate targeted diagnostic report using only specified modules.

    Args:
        trades: Trade list to diagnose.
        modules: List of module IDs (e.g. ["D1", "D6"]).
        initial_equity: Starting equity for overview/verdict sections.
        title: Optional title override.

    D6 (overview) is always included regardless of requested modules.
    Sections are deduplicated when multiple modules share functions.
    """
    terminal_marks = terminal_marks or []
    if not trades and not terminal_marks:
        return "No trades to diagnose."
    if not trades:
        diag_title = title or "Phase Diagnostics"
        parts = [
            f"{'=' * 70}",
            f"  {diag_title.upper()}",
            f"  0 realized trades | {len(terminal_marks)} terminal marks",
            f"{'=' * 70}",
            _s01_overview(trades, initial_equity, terminal_marks, performance_metrics),
            _s05_exit_attribution(trades, terminal_marks),
            _verdict(trades, initial_equity, terminal_marks),
            "",
        ]
        return "\n".join(parts)

    # Always include D6
    requested = set(modules)
    requested.add("D6")

    # Collect unique section function names — D6 (overview) first, then rest sorted,
    # but _verdict always last (it summarizes everything).
    seen: set[str] = set()
    section_names: list[str] = []
    ordered = ["D6"] + sorted(requested - {"D6"})
    for mod_id in ordered:
        for fn_name in DIAGNOSTIC_MODULES.get(mod_id, []):
            if fn_name not in seen:
                seen.add(fn_name)
                section_names.append(fn_name)
    if "_verdict" in section_names:
        section_names.remove("_verdict")
        section_names.append("_verdict")

    # Build output
    diag_title = title or "Phase Diagnostics"
    parts = [
        f"{'=' * 70}",
        f"  {diag_title.upper()}",
        f"  {len(trades)} realized trades | {len(terminal_marks)} terminal marks | modules: {', '.join(sorted(requested))}",
        f"{'=' * 70}",
    ]

    for fn_name in section_names:
        section = _render_section(
            fn_name,
            trades,
            initial_equity,
            terminal_marks,
            performance_metrics,
            expected_symbols,
            diagnostic_context,
        )
        if section:
            parts.append(section)

    parts.append("")
    return "\n".join(parts)


# ── Main Entry Point ────────────────────────────────────────────────


def generate_diagnostics(
    trades: list[Trade],
    initial_equity: float = 10_000.0,
    title: str = "Strategy Diagnostics",
    terminal_marks: list[TerminalMark] | None = None,
    performance_metrics: PerformanceMetrics | None = None,
    expected_symbols: list[str] | None = None,
    diagnostic_context: dict[str, Any] | None = None,
) -> str:
    """Generate comprehensive diagnostic report from trade list.

    Returns a text report with 22 numbered sections plus verdict.
    """
    terminal_marks = terminal_marks or []
    if not trades and not terminal_marks:
        return "No trades to diagnose."
    if not trades:
        sections = [
            f"{'=' * 70}",
            f"  {title.upper()}",
            f"  0 realized trades | {len(terminal_marks)} terminal marks | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
            f"{'=' * 70}",
            _s01_overview(trades, initial_equity, terminal_marks, performance_metrics),
            _s05_exit_attribution(trades, terminal_marks),
            _s23_blocked_relaxed_body_audit(trades, expected_symbols, diagnostic_context),
            _verdict(trades, initial_equity, terminal_marks),
            "",
        ]
        return "\n".join(sections)

    sections = [
        f"{'=' * 70}",
        f"  {title.upper()}",
        f"  {len(trades)} realized trades | {len(terminal_marks)} terminal marks | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
        f"{'=' * 70}",
        _s01_overview(trades, initial_equity, terminal_marks, performance_metrics),
        _r_distribution(trades),
        _s02_winner_loser_profiles(trades),
        _s03_mfe_capture(trades),
        _s04_stop_calibration(trades),
        _s05_exit_attribution(trades, terminal_marks),
        _s06_streaks(trades),
        _s07_drawdown(trades),
        _s08_rolling_expectancy(trades),
        _s09_per_asset(trades, expected_symbols),
        _s10_direction(trades),
        _s11_confirmation(trades),
        _s12_confluence(trades),
        _s13_timing(trades),
        _s14_duration(trades),
        _s15_concentration(trades),
        _s16_worst_trades(trades),
        _s17_interactions(trades, expected_symbols),
        _s18_friction(trades),
        _s19_weekly_pnl(trades),
        _s20_best_trades(trades),
        _s21_risk_sizing(trades),
        _s22_entry_method(trades),
        _s23_blocked_relaxed_body_audit(trades, expected_symbols, diagnostic_context),
        _verdict(trades, initial_equity, terminal_marks),
        "",
    ]
    return "\n".join(sections)
