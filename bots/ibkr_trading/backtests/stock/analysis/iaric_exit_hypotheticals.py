"""IARIC exit hypotheticals -- replay trades under alternative exit rules.

For each completed trade, replay 5m bars from entry to EOD under alternative rules:

| Rule           | Variants                                    |
|----------------|---------------------------------------------|
| Time stop      | 30min / 45min(default) / 60min / 90min / none |
| Partial R      | 1.0R / 1.5R(default) / 2.0R / 2.5R / none  |
| Partial frac   | 25% / 33% / 50%(default)                    |
| Carry          | always flatten / current / any-profitable   |

Output per variant: n, WR%, mean R, PF, total R, delta_vs_actual, MFE_capture%
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from backtests.stock.models import TradeRecord


def _hdr(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


@dataclass
class HypotheticalResult:
    """Result from replaying a single trade under alternative rules."""
    symbol: str
    trade_date: date
    original_r: float
    hypothetical_r: float
    exit_reason: str
    bars_held: int
    mfe_r: float


def _replay_trade_bars(
    bars_data: list[tuple[float, float, float, float]],  # [(open, high, low, close), ...]
    entry_price: float,
    stop_price: float,
    risk_per_share: float,
    time_stop_bars: int | None,
    partial_r: float | None,
    partial_fraction: float,
) -> HypotheticalResult:
    """Replay a series of 5m bars under specified exit rules.

    Returns the hypothetical outcome.
    """
    rps = max(risk_per_share, 0.01)
    max_price = entry_price
    min_price = entry_price
    partial_taken = False
    partial_r_captured = 0.0

    for i, (o, h, l, c) in enumerate(bars_data):
        max_price = max(max_price, h)
        min_price = min(min_price, l)
        bars = i + 1

        # Stop hit
        if l <= stop_price:
            sim_r = (stop_price - entry_price) / rps
            if partial_taken:
                sim_r = partial_fraction * partial_r_captured + (1 - partial_fraction) * sim_r
            return HypotheticalResult(
                symbol="", trade_date=date.min, original_r=0,
                hypothetical_r=sim_r, exit_reason="STOP_HIT",
                bars_held=bars, mfe_r=(max_price - entry_price) / rps,
            )

        # Time stop
        if time_stop_bars is not None and bars >= time_stop_bars and c <= entry_price:
            sim_r = (c - entry_price) / rps
            if partial_taken:
                sim_r = partial_fraction * partial_r_captured + (1 - partial_fraction) * sim_r
            return HypotheticalResult(
                symbol="", trade_date=date.min, original_r=0,
                hypothetical_r=sim_r, exit_reason="TIME_STOP",
                bars_held=bars, mfe_r=(max_price - entry_price) / rps,
            )

        # Partial take
        if partial_r is not None and not partial_taken:
            if h >= entry_price + partial_r * rps:
                partial_taken = True
                partial_r_captured = partial_r

    # EOD flatten
    if not bars_data:
        return HypotheticalResult(
            symbol="", trade_date=date.min, original_r=0,
            hypothetical_r=0, exit_reason="NO_DATA",
            bars_held=0, mfe_r=0,
        )

    final_close = bars_data[-1][3]
    sim_r = (final_close - entry_price) / rps
    if partial_taken:
        runner_r = (final_close - entry_price) / rps
        sim_r = partial_fraction * partial_r_captured + (1 - partial_fraction) * runner_r

    return HypotheticalResult(
        symbol="", trade_date=date.min, original_r=0,
        hypothetical_r=sim_r, exit_reason="EOD_FLATTEN",
        bars_held=len(bars_data), mfe_r=(max_price - entry_price) / rps,
    )


def _variant_table(
    variant_name: str,
    variant_values: list[tuple[str, dict]],
    trades: list[TradeRecord],
    get_bars_fn,
) -> str:
    """Run all variants for a given rule dimension and produce summary table."""
    lines = []
    lines.append(f"\n  {variant_name}")
    lines.append(f"  {'Variant':<16s} {'N':>5s} {'WR%':>6s} {'Mean R':>8s} {'PF':>6s}"
                 f" {'Total R':>9s} {'ΔvsActual':>10s} {'MFE Cap%':>9s}")
    lines.append("  " + "-" * 72)

    actual_total_r = sum(t.r_multiple for t in trades)

    for label, params in variant_values:
        results: list[HypotheticalResult] = []

        for t in trades:
            entry_date = t.entry_time.date() if hasattr(t.entry_time, 'date') else t.entry_time
            bars = get_bars_fn(t.symbol, entry_date, t.entry_time)
            if not bars:
                continue

            res = _replay_trade_bars(
                bars_data=bars,
                entry_price=t.entry_price,
                stop_price=t.entry_price - t.risk_per_share,
                risk_per_share=t.risk_per_share,
                **params,
            )
            res.symbol = t.symbol
            res.trade_date = entry_date
            res.original_r = t.r_multiple
            results.append(res)

        if not results:
            lines.append(f"  {label:<16s} {'--':>5s}")
            continue

        n = len(results)
        rs = [r.hypothetical_r for r in results]
        wins = sum(1 for r in rs if r > 0)
        wr = wins / n
        mean_r = float(np.mean(rs))
        total_r = sum(rs)
        gross_p = sum(r for r in rs if r > 0)
        gross_l = abs(sum(r for r in rs if r < 0))
        pf = gross_p / gross_l if gross_l > 0 else float("inf") if gross_p > 0 else 0.0
        delta = total_r - actual_total_r
        mfe_caps = [r.hypothetical_r / r.mfe_r if r.mfe_r > 0 else 0 for r in results]
        avg_mfe_cap = float(np.mean(mfe_caps)) if mfe_caps else 0

        lines.append(
            f"  {label:<16s} {n:>5} {wr:>5.0%} {mean_r:>+8.3f} {pf:>6.2f}"
            f" {total_r:>+9.2f} {delta:>+10.2f} {avg_mfe_cap:>8.0%}"
        )

    return "\n".join(lines)


def iaric_exit_hypotheticals(
    trades: list[TradeRecord],
    replay_engine=None,
    config=None,
) -> str:
    """Generate the IARIC exit hypotheticals report.

    Parameters
    ----------
    trades : list[TradeRecord]
        Completed IARIC trades.
    replay_engine : ResearchReplayEngine, optional
        For bar data access. If None, uses simplified simulation.
    config : IARICBacktestConfig, optional
        For settings reference.
    """
    if not trades:
        return "No trades to analyze."

    lines = [_hdr("Exit Hypotheticals")]
    lines.append(f"  Trades: {len(trades)}")
    lines.append(f"  Actual total R: {sum(t.r_multiple for t in trades):+.2f}")

    # Build bar accessor
    def get_bars(symbol: str, trade_date, entry_time) -> list[tuple[float, float, float, float]]:
        """Get 5m bars from entry to EOD."""
        if replay_engine is None:
            return []
        bars_df = replay_engine.get_5m_bars_for_date(symbol, trade_date)
        if bars_df is None or bars_df.empty:
            return []
        # Filter to bars after entry
        result = []
        for row in bars_df.itertuples():
            bar_dt = row.Index.to_pydatetime()
            if bar_dt >= entry_time - timedelta(minutes=5):
                result.append((float(row.open), float(row.high), float(row.low), float(row.close)))
        return result

    # Time stop variants
    time_stop_variants = [
        ("30min", {"time_stop_bars": 6, "partial_r": 1.5, "partial_fraction": 0.50}),
        ("45min (default)", {"time_stop_bars": 9, "partial_r": 1.5, "partial_fraction": 0.50}),
        ("60min", {"time_stop_bars": 12, "partial_r": 1.5, "partial_fraction": 0.50}),
        ("90min", {"time_stop_bars": 18, "partial_r": 1.5, "partial_fraction": 0.50}),
        ("No time stop", {"time_stop_bars": None, "partial_r": 1.5, "partial_fraction": 0.50}),
    ]

    # Partial R variants
    partial_r_variants = [
        ("1.0R partial", {"time_stop_bars": 9, "partial_r": 1.0, "partial_fraction": 0.50}),
        ("1.5R (default)", {"time_stop_bars": 9, "partial_r": 1.5, "partial_fraction": 0.50}),
        ("2.0R partial", {"time_stop_bars": 9, "partial_r": 2.0, "partial_fraction": 0.50}),
        ("2.5R partial", {"time_stop_bars": 9, "partial_r": 2.5, "partial_fraction": 0.50}),
        ("No partial", {"time_stop_bars": 9, "partial_r": None, "partial_fraction": 0.50}),
    ]

    # Partial fraction variants
    partial_frac_variants = [
        ("25% partial", {"time_stop_bars": 9, "partial_r": 1.5, "partial_fraction": 0.25}),
        ("33% partial", {"time_stop_bars": 9, "partial_r": 1.5, "partial_fraction": 0.33}),
        ("50% (default)", {"time_stop_bars": 9, "partial_r": 1.5, "partial_fraction": 0.50}),
    ]

    if replay_engine is not None:
        lines.append(_variant_table("Time Stop Variants", time_stop_variants, trades, get_bars))
        lines.append(_variant_table("Partial R Variants", partial_r_variants, trades, get_bars))
        lines.append(_variant_table("Partial Fraction Variants", partial_frac_variants, trades, get_bars))
    else:
        lines.append("\n  (No replay engine -- using simplified R-based simulation)")

        # Simplified analysis without bar data: just compare exit reason distribution
        lines.append(f"\n  Exit reason distribution:")
        reasons: dict[str, int] = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / len(trades)
            group = [t for t in trades if t.exit_reason == reason]
            mean_r = float(np.mean([t.r_multiple for t in group]))
            lines.append(f"    {reason:<20s}: {count:>4} ({pct:.0%}), Mean R={mean_r:+.3f}")

        # Simple what-if: remove time stops
        time_stop_trades = [t for t in trades if t.exit_reason == "TIME_STOP"]
        if time_stop_trades:
            ts_r = sum(t.r_multiple for t in time_stop_trades)
            lines.append(f"\n  Time stop impact: {len(time_stop_trades)} trades, {ts_r:+.2f}R")
            lines.append(f"  If removed: would save {abs(ts_r):.2f}R" if ts_r < 0 else
                        f"  Time stops are net positive ({ts_r:+.2f}R)")

    return "\n".join(lines)
