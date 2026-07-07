from __future__ import annotations

from collections import Counter


def po3_reversal_diagnostics(trades: list, metrics: dict[str, float]) -> str:
    tiers = Counter(getattr(trade, "tier", "") for trade in trades)
    exits = Counter(getattr(trade, "exit_reason", "") for trade in trades)
    return "\n".join(
        [
            "PO3 REVERSAL DIAGNOSTICS",
            f"Trades: {int(metrics.get('total_trades', 0))}",
            f"Net PnL: {metrics.get('net_profit', 0.0):+.2f}",
            f"Expectancy: {metrics.get('expectancy_dollar', 0.0):+.2f}",
            f"Profit factor: {metrics.get('profit_factor', 0.0):.2f}",
            f"Max drawdown: {metrics.get('max_drawdown_pct', 0.0):.2%}",
            f"Tiers: {_fmt_counter(tiers)}",
            f"Exits: {_fmt_counter(exits)}",
        ]
    )


def _fmt_counter(counter: Counter) -> str:
    return ", ".join(f"{key or 'unknown'}={value}" for key, value in counter.most_common()) or "none"

