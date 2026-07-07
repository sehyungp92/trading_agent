from __future__ import annotations

from collections import Counter


def ivb_auction_diagnostics(trades: list, metrics: dict[str, float]) -> str:
    modules = Counter(getattr(trade, "module", "") for trade in trades)
    triggers = Counter(getattr(trade, "trigger", "") for trade in trades)
    return "\n".join(
        [
            "IVB AUCTION DIAGNOSTICS",
            f"Trades: {int(metrics.get('total_trades', 0))}",
            f"Net PnL: {metrics.get('net_profit', 0.0):+.2f}",
            f"Expectancy: {metrics.get('expectancy_dollar', 0.0):+.2f}",
            f"Profit factor: {metrics.get('profit_factor', 0.0):.2f}",
            f"Max drawdown: {metrics.get('max_drawdown_pct', 0.0):.2%}",
            f"Modules: {_fmt_counter(modules)}",
            f"Triggers: {_fmt_counter(triggers)}",
        ]
    )


def _fmt_counter(counter: Counter) -> str:
    return ", ".join(f"{key or 'unknown'}={value}" for key, value in counter.most_common()) or "none"

