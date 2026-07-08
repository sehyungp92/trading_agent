"""Candlestick charts with trade entry/exit arrows for backtest visualization.

Uses raw matplotlib (no mplfinance dependency).
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_candlesticks(
    ax: plt.Axes,
    times: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> None:
    """Render OHLC candles using bar() for bodies + vlines() for wicks."""
    n = len(times)
    if n == 0:
        return

    # Convert datetime64 to matplotlib date numbers
    # Handle both datetime64 and datetime objects
    if hasattr(times[0], 'astype') or np.issubdtype(times.dtype, np.datetime64):
        # numpy datetime64 array
        dates = mdates.date2num(times.astype("datetime64[ms]").astype("O"))
    else:
        dates = mdates.date2num(times)

    # Determine bar width based on data density
    if n > 1:
        median_gap = np.median(np.diff(dates))
        bar_width = median_gap * 0.6
    else:
        bar_width = 0.5

    green_mask = closes >= opens
    red_mask = ~green_mask

    # Bodies
    body_heights = np.abs(closes - opens)
    body_bottoms = np.minimum(opens, closes)

    if green_mask.any():
        ax.bar(
            dates[green_mask],
            body_heights[green_mask],
            bottom=body_bottoms[green_mask],
            width=bar_width,
            color="#26a69a",
            edgecolor="#26a69a",
            linewidth=0.5,
        )
    if red_mask.any():
        ax.bar(
            dates[red_mask],
            body_heights[red_mask],
            bottom=body_bottoms[red_mask],
            width=bar_width,
            color="#ef5350",
            edgecolor="#ef5350",
            linewidth=0.5,
        )

    # Wicks (high-low lines)
    ax.vlines(dates, lows, highs, color="black", linewidth=0.3)


def _draw_trade_arrows(
    ax: plt.Axes,
    bar_times: np.ndarray,
    trades: list,
    highs: np.ndarray,
    lows: np.ndarray,
    daily: bool = False,
) -> None:
    """Draw trade arrows distinguishing long vs short trades.

    Long:  blue ^ entry (below low), blue v exit (above high)
    Short: orange v entry (above high), orange ^ exit (below low)

    Uses np.searchsorted for time matching. When daily=True, normalizes
    trade times to date-level for matching against daily bars.
    """
    if len(bar_times) == 0 or not trades:
        return

    # Build sorted time array for searchsorted
    if np.issubdtype(bar_times.dtype, np.datetime64):
        if daily:
            sorted_times = bar_times.astype("datetime64[D]")
        else:
            sorted_times = bar_times.astype("datetime64[ms]")
        dates_num = mdates.date2num(bar_times.astype("datetime64[ms]").astype("O"))
    else:
        sorted_times = bar_times
        dates_num = mdates.date2num(bar_times)

    price_range = np.max(highs) - np.min(lows)
    arrow_offset = price_range * 0.015

    def _find_idx(trade_time):
        if daily:
            target = np.datetime64(
                trade_time.date() if hasattr(trade_time, 'date') else trade_time, "D"
            )
        else:
            target = (np.datetime64(trade_time, "ms")
                      if not isinstance(trade_time, np.datetime64) else trade_time)
        idx = np.searchsorted(sorted_times, target)
        return idx if 0 <= idx < len(bar_times) else None

    # Separate into 4 groups: long entry/exit, short entry/exit
    long_entry_xs, long_entry_ys = [], []
    long_exit_xs, long_exit_ys = [], []
    short_entry_xs, short_entry_ys = [], []
    short_exit_xs, short_exit_ys = [], []

    for t in trades:
        is_long = getattr(t, 'direction', 1) == 1

        if t.entry_time is not None:
            idx = _find_idx(t.entry_time)
            if idx is not None:
                if is_long:
                    long_entry_xs.append(dates_num[idx])
                    long_entry_ys.append(lows[idx] - arrow_offset)
                else:
                    short_entry_xs.append(dates_num[idx])
                    short_entry_ys.append(highs[idx] + arrow_offset)

        if t.exit_time is not None:
            idx = _find_idx(t.exit_time)
            if idx is not None:
                if is_long:
                    long_exit_xs.append(dates_num[idx])
                    long_exit_ys.append(highs[idx] + arrow_offset)
                else:
                    short_exit_xs.append(dates_num[idx])
                    short_exit_ys.append(lows[idx] - arrow_offset)

    if long_entry_xs:
        ax.scatter(long_entry_xs, long_entry_ys, marker="^", color="#1565c0",
                   s=30, zorder=5, label="Long entry")
    if long_exit_xs:
        ax.scatter(long_exit_xs, long_exit_ys, marker="v", color="#1565c0",
                   s=20, zorder=5, label="Long exit", alpha=0.6)
    if short_entry_xs:
        ax.scatter(short_entry_xs, short_entry_ys, marker="v", color="#e65100",
                   s=30, zorder=5, label="Short entry")
    if short_exit_xs:
        ax.scatter(short_exit_xs, short_exit_ys, marker="^", color="#e65100",
                   s=20, zorder=5, label="Short exit", alpha=0.6)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_backtest_charts(
    symbol: str,
    daily,       # NumpyBars (duck-typed)
    hourly,      # NumpyBars (duck-typed)
    trades: list,
    output_dir: Path,
    strategy_label: str = "",
) -> list[Path]:
    """Generate daily and hourly candlestick chart PNGs with trade arrows.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "QQQ").
    daily : NumpyBars
        Full daily bar data.
    hourly : NumpyBars
        Full hourly bar data.
    trades : list
        List of TradeRecord or HelixTradeRecord (duck-typed).
    output_dir : Path
        Directory to save PNGs.
    strategy_label : str
        Label for filenames (e.g. "helix", "atrss").

    Returns
    -------
    list[Path]
        Paths of generated PNG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label = f"_{strategy_label}" if strategy_label else ""
    saved: list[Path] = []

    # --- Daily chart: full period ---
    if daily is not None and len(daily) > 0:
        fig, ax = plt.subplots(figsize=(20, 8))
        _draw_candlesticks(ax, daily.times, daily.opens, daily.highs,
                           daily.lows, daily.closes)
        _draw_trade_arrows(ax, daily.times, trades, daily.highs, daily.lows,
                           daily=True)

        ax.set_title(f"{symbol} Daily{' (' + strategy_label + ')' if strategy_label else ''}",
                     fontsize=14)
        ax.set_ylabel("Price")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate(rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        path = output_dir / f"{symbol}{label}_daily.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info("Saved daily chart: %s", path)

    # --- Hourly chart: last 1 year only ---
    if hourly is not None and len(hourly) > 0:
        # Compute 1-year cutoff
        last_time = hourly.times[-1]
        if np.issubdtype(hourly.times.dtype, np.datetime64):
            one_year_ago = last_time - np.timedelta64(365, "D")
            mask = hourly.times >= one_year_ago
        else:
            one_year_ago = last_time - np.timedelta64(365, "D")
            mask = hourly.times >= one_year_ago

        h_times = hourly.times[mask]
        h_opens = hourly.opens[mask]
        h_highs = hourly.highs[mask]
        h_lows = hourly.lows[mask]
        h_closes = hourly.closes[mask]

        if len(h_times) > 0:
            fig, ax = plt.subplots(figsize=(20, 8))
            _draw_candlesticks(ax, h_times, h_opens, h_highs, h_lows, h_closes)
            _draw_trade_arrows(ax, h_times, trades, h_highs, h_lows, daily=False)

            ax.set_title(
                f"{symbol} Hourly (last 1yr)"
                f"{' (' + strategy_label + ')' if strategy_label else ''}",
                fontsize=14,
            )
            ax.set_ylabel("Price")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            fig.autofmt_xdate(rotation=45)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")

            path = output_dir / f"{symbol}{label}_hourly.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
            logger.info("Saved hourly chart: %s", path)

    return saved
