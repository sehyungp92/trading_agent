"""Matplotlib visualizations for stock backtest results."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from backtests.stock.models import TradeRecord


def plot_equity_curve(
    equity_curve: np.ndarray,
    timestamps: np.ndarray | None = None,
    title: str = "Equity Curve",
    save_path: Path | None = None,
) -> None:
    """Plot equity curve with drawdown overlay."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)

    x = timestamps if timestamps is not None and len(timestamps) == len(equity_curve) else np.arange(len(equity_curve))

    # Equity curve
    ax1.plot(x, equity_curve, linewidth=1.2, color="#2196F3")
    ax1.fill_between(x, equity_curve[0], equity_curve, alpha=0.1, color="#2196F3")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=equity_curve[0], color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    # Drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd_pct = (equity_curve - peak) / np.where(peak > 0, peak, 1.0) * 100
    ax2.fill_between(x, dd_pct, 0, alpha=0.4, color="#F44336")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date" if timestamps is not None else "Bar")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_trade_distribution(
    trades: list[TradeRecord],
    title: str = "Trade P&L Distribution",
    save_path: Path | None = None,
) -> None:
    """Plot histogram of trade P&L in R-multiples."""
    import matplotlib.pyplot as plt

    if not trades:
        return

    r_multiples = [t.r_multiple for t in trades]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4CAF50" if r > 0 else "#F44336" for r in r_multiples]
    bins = np.linspace(min(r_multiples) - 0.5, max(r_multiples) + 0.5, 40)
    ax.hist(r_multiples, bins=bins, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(x=np.mean(r_multiples), color="#FF9800", linestyle="-", linewidth=1.5, label=f"Mean: {np.mean(r_multiples):.2f}R")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("R-Multiple")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_monthly_returns(
    trades: list[TradeRecord],
    title: str = "Monthly Returns",
    save_path: Path | None = None,
) -> None:
    """Plot monthly P&L as bar chart."""
    import matplotlib.pyplot as plt

    if not trades:
        return

    monthly: dict[str, float] = {}
    for t in trades:
        key = t.entry_time.strftime("%Y-%m")
        monthly[key] = monthly.get(key, 0) + t.pnl_net

    months = sorted(monthly.keys())
    values = [monthly[m] for m in months]
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in values]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(months)), values, color=colors, edgecolor="white", alpha=0.8)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("P&L ($)")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_sector_attribution(
    trades: list[TradeRecord],
    title: str = "Sector P&L Attribution",
    save_path: Path | None = None,
) -> None:
    """Horizontal bar chart of P&L by sector."""
    import matplotlib.pyplot as plt

    if not trades:
        return

    sectors: dict[str, float] = {}
    for t in trades:
        sec = t.sector or "UNKNOWN"
        sectors[sec] = sectors.get(sec, 0) + t.pnl_net

    sorted_sectors = sorted(sectors.items(), key=lambda x: x[1])
    names = [s[0] for s in sorted_sectors]
    values = [s[1] for s in sorted_sectors]
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
    ax.barh(names, values, color=colors, edgecolor="white", alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("P&L ($)")
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
