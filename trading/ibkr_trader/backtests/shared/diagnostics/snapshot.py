from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class BucketSummary:
    label: str
    count: int
    win_rate: float
    avg_r: float
    net_pnl: float
    profit_factor: float


def trade_net_pnl(trade: Any) -> float:
    pnl = getattr(trade, "pnl_dollars", None)
    if pnl is None:
        pnl = getattr(trade, "pnl", 0.0)
    commission = getattr(trade, "commission", 0.0) or 0.0
    return float(pnl) - float(commission)


def summarize_groups(
    trades: list[Any],
    key_fn: Callable[[Any], Any],
    *,
    min_count: int = 1,
) -> list[BucketSummary]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        label = key_fn(trade)
        if label in (None, "", "UNKNOWN"):
            continue
        grouped[str(label)].append(trade)

    summaries: list[BucketSummary] = []
    for label, bucket in grouped.items():
        if len(bucket) < min_count:
            continue
        net_pnls = [trade_net_pnl(trade) for trade in bucket]
        wins = [pnl for pnl in net_pnls if pnl > 0]
        losses = [pnl for pnl in net_pnls if pnl < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        r_multiples = [float(getattr(trade, "r_multiple", 0.0) or 0.0) for trade in bucket]
        win_rate = sum(1 for r_multiple in r_multiples if r_multiple > 0) / len(bucket)
        avg_r = sum(r_multiples) / len(bucket)
        net_pnl = sum(net_pnls)
        summaries.append(
            BucketSummary(
                label=label,
                count=len(bucket),
                win_rate=win_rate,
                avg_r=avg_r,
                net_pnl=net_pnl,
                profit_factor=profit_factor,
            )
        )

    return sorted(summaries, key=lambda item: (-item.net_pnl, -item.avg_r, -item.count, item.label))


def _is_clean_strength(summary: BucketSummary) -> bool:
    return summary.net_pnl > 0 and summary.avg_r > 0 and summary.profit_factor > 1.0


def best_bucket(summaries: list[BucketSummary], *, min_count: int = 3) -> BucketSummary | None:
    eligible = [
        summary
        for summary in summaries
        if summary.count >= min_count and _is_clean_strength(summary)
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (item.net_pnl, item.avg_r, item.profit_factor, item.win_rate, item.count),
    )


def worst_bucket(summaries: list[BucketSummary], *, min_count: int = 3) -> BucketSummary | None:
    eligible = [summary for summary in summaries if summary.count >= min_count]
    if not eligible:
        eligible = summaries
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (item.net_pnl, item.avg_r, item.win_rate, -item.count, item.label),
    )


def concentration_share(trades: list[Any], *, top_n: int = 5, positive: bool = True) -> float:
    values: list[float] = []
    for trade in trades:
        pnl = trade_net_pnl(trade)
        if positive and pnl > 0:
            values.append(pnl)
        elif not positive and pnl < 0:
            values.append(pnl)
    if not values:
        return 0.0
    magnitudes = sorted((abs(value) for value in values), reverse=True)
    total = sum(magnitudes)
    if total <= 0:
        return 0.0
    return sum(magnitudes[:top_n]) / total


def format_bucket(summary: BucketSummary | None) -> str:
    if summary is None:
        return "N/A"
    pf = "inf" if summary.profit_factor == float("inf") else f"{summary.profit_factor:.2f}"
    return (
        f"{summary.label} (n={summary.count}, WR={summary.win_rate*100:.1f}%, "
        f"avgR={summary.avg_r:+.2f}, fee-net=${summary.net_pnl:+,.0f}, PF={pf})"
    )


def render_snapshot(
    title: str,
    strengths: list[str],
    weaknesses: list[str],
    *,
    notes: list[str] | None = None,
    width: int = 72,
) -> str:
    lines = ["=" * width, f"  {title}", "=" * width, ""]
    lines.append("  Strengths")
    for item in strengths:
        lines.append(f"    - {item}")
    if not strengths:
        lines.append("    - N/A")
    lines.append("")
    lines.append("  Weaknesses")
    for item in weaknesses:
        lines.append(f"    - {item}")
    if not weaknesses:
        lines.append("    - N/A")
    if notes:
        lines.append("")
        lines.append("  Notes")
        for item in notes:
            lines.append(f"    - {item}")
    return "\n".join(lines)


def build_group_snapshot(
    title: str,
    trades: list[Any],
    groups: list[tuple[str, Callable[[Any], Any]]],
    *,
    min_count: int = 3,
    top_n: int = 5,
    width: int = 72,
) -> str:
    strengths: list[str] = []
    weaknesses: list[str] = []
    notes: list[str] = []

    for group_name, key_fn in groups:
        summaries = summarize_groups(trades, key_fn, min_count=min_count)
        if not summaries:
            continue
        best = best_bucket(summaries, min_count=min_count)
        worst = worst_bucket(summaries, min_count=min_count)
        if best is not None and best.net_pnl > 0:
            strengths.append(f"Best {group_name}: {format_bucket(best)}")
        if worst is not None and worst.net_pnl < 0:
            weaknesses.append(f"Worst {group_name}: {format_bucket(worst)}")

    net_pnls = [trade_net_pnl(trade) for trade in trades]
    positive_trade_count = sum(1 for pnl in net_pnls if pnl > 0)
    negative_trade_count = sum(1 for pnl in net_pnls if pnl < 0)

    top_share = concentration_share(trades, top_n=top_n, positive=True)
    if top_share >= 0.6:
        weaknesses.append(
            f"Positive fee-net PnL is concentrated: top {top_n} winners drive {top_share:.0%} of gains."
        )
    elif top_share > 0:
        strengths.append(
            f"Positive fee-net PnL is reasonably distributed: top {top_n} winners drive {top_share:.0%} of gains."
        )

    loss_share = concentration_share(trades, top_n=top_n, positive=False)
    if negative_trade_count > top_n and loss_share >= 0.6:
        weaknesses.append(
            f"Losses are concentrated: top {top_n} losers drive {loss_share:.0%} of losses."
        )

    if net_pnls:
        losers = sum(1 for pnl in net_pnls if pnl <= 0)
        notes.append(
            f"Fee-net trade count: {len(net_pnls)} with {losers} non-positive outcomes and "
            f"{len(net_pnls) - losers} positive outcomes."
        )
        if positive_trade_count == 0:
            notes.append("No fee-net positive trades were recorded in this run.")

    return render_snapshot(
        title,
        strengths,
        weaknesses,
        notes=notes,
        width=width,
    )
