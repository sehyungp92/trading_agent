"""Investigation 5: Hold-time vs R analysis.

Buckets trades by hold duration and computes mean R, win rate, mean MFE
per bucket to identify optimal hold duration targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HoldTimeBucket:
    """Stats for one hold-time bucket."""
    label: str
    lo: int
    hi: int  # exclusive upper bound (999999 = unbounded)
    count: int = 0
    wins: int = 0
    win_rate: float = 0.0
    mean_r: float = 0.0
    median_r: float = 0.0
    total_r: float = 0.0
    mean_mfe: float = 0.0
    mean_mae: float = 0.0
    mfe_capture_pct: float = 0.0
    mean_bars: float = 0.0


# Default bucket boundaries (in RTH bars held)
DEFAULT_BUCKETS = [
    ("0-20h", 0, 20),
    ("20-40h", 20, 40),
    ("40-80h", 40, 80),
    ("80-160h", 80, 160),
    ("160+h", 160, 999999),
]


def hold_time_analysis(
    trades: list,
    buckets: list[tuple[str, int, int]] | None = None,
) -> list[HoldTimeBucket]:
    """Bucket trades by hold time and compute per-bucket statistics.

    Parameters
    ----------
    trades : list of TradeRecord
        Completed trades with bars_held, r_multiple, mfe_r, mae_r fields.
    buckets : optional list of (label, lo, hi) tuples
        Custom bucket boundaries. Defaults to DEFAULT_BUCKETS.

    Returns
    -------
    list of HoldTimeBucket with computed statistics.
    """
    if buckets is None:
        buckets = DEFAULT_BUCKETS

    results: list[HoldTimeBucket] = []

    for label, lo, hi in buckets:
        bucket_trades = [t for t in trades if lo <= t.bars_held < hi]
        b = HoldTimeBucket(label=label, lo=lo, hi=hi)

        if not bucket_trades:
            results.append(b)
            continue

        rs = np.array([t.r_multiple for t in bucket_trades])
        mfes = np.array([t.mfe_r for t in bucket_trades])
        maes = np.array([t.mae_r for t in bucket_trades])
        holds = np.array([t.bars_held for t in bucket_trades])

        b.count = len(bucket_trades)
        b.wins = int(np.sum(rs > 0))
        b.win_rate = b.wins / b.count * 100
        b.mean_r = float(np.mean(rs))
        b.median_r = float(np.median(rs))
        b.total_r = float(np.sum(rs))
        b.mean_mfe = float(np.mean(mfes))
        b.mean_mae = float(np.mean(maes))
        b.mfe_capture_pct = (b.mean_r / b.mean_mfe * 100) if b.mean_mfe > 0 else 0.0
        b.mean_bars = float(np.mean(holds))

        results.append(b)

    return results


def format_hold_time_report(
    all_results: dict[str, list[HoldTimeBucket]],
) -> str:
    """Format hold-time analysis results as a printable report.

    Parameters
    ----------
    all_results : dict mapping symbol -> list of HoldTimeBucket
    """
    lines = ["=" * 80, "INVESTIGATION 5: HOLD-TIME vs R ANALYSIS", "=" * 80]

    for symbol, buckets in all_results.items():
        lines.append(f"\n--- {symbol} ---")
        lines.append(
            f"{'Bucket':<12} {'Count':>6} {'WR%':>6} {'MeanR':>8} "
            f"{'MedR':>8} {'TotalR':>8} {'MFE':>8} {'MAE':>8} "
            f"{'MFE Cap%':>9} {'AvgBars':>8}"
        )
        lines.append("-" * 95)

        total_count = 0
        total_r = 0.0

        for b in buckets:
            lines.append(
                f"{b.label:<12} {b.count:>6} {b.win_rate:>5.1f}% "
                f"{b.mean_r:>+8.3f} {b.median_r:>+8.3f} "
                f"{b.total_r:>+8.2f} {b.mean_mfe:>8.3f} "
                f"{b.mean_mae:>8.3f} {b.mfe_capture_pct:>8.1f}% "
                f"{b.mean_bars:>8.1f}"
            )
            total_count += b.count
            total_r += b.total_r

        lines.append("-" * 95)
        lines.append(f"{'TOTAL':<12} {total_count:>6} {'':>7} {'':>8} {'':>8} {total_r:>+8.2f}")

        # Identify best bucket
        profitable_buckets = [b for b in buckets if b.count > 0 and b.mean_r > 0]
        if profitable_buckets:
            best = max(profitable_buckets, key=lambda x: x.mean_r)
            lines.append(f"\n  Best mean R: {best.label} ({best.mean_r:+.3f}R, "
                         f"WR={best.win_rate:.1f}%, MFE capture={best.mfe_capture_pct:.1f}%)")

        # Identify where most total R comes from
        if buckets:
            top_total = max(buckets, key=lambda x: x.total_r)
            if top_total.count > 0:
                lines.append(f"  Most total R: {top_total.label} ({top_total.total_r:+.2f}R "
                             f"from {top_total.count} trades)")

    return "\n".join(lines)
