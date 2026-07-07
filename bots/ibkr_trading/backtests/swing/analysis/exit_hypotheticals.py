"""Investigation 1: MFE-based exit hypothetical simulator.

Takes existing trade records + hourly bar data and computes hypothetical P&L
under different exit rules — without modifying the backtest engine. Replays
each trade forward from its entry bar to simulate alternative exits.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from strategies.swing.atrss.config import SymbolConfig
from strategies.swing.atrss.models import DailyState, Direction
from backtests.swing.data.preprocessing import NumpyBars

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exit rule definitions
# ---------------------------------------------------------------------------

@dataclass
class ExitRule:
    """Describes one hypothetical exit strategy."""
    name: str
    # Fixed R target: exit 100% at this R (0 = disabled)
    fixed_r_target: float = 0.0
    # Partial: exit this fraction at partial_r, trail rest with trail_atr_mult
    partial_frac: float = 0.0
    partial_r: float = 0.0
    trail_atr_mult: float = 0.0
    # Regime downgrade exit (exit on STRONG→TREND or TREND→RANGE)
    regime_downgrade_exit: bool = False
    # Time-based forced exit: exit at this many bars if < forced_exit_min_r
    forced_exit_hours: int = 0
    forced_exit_min_r: float = 0.0


# Standard exit rule library
STANDARD_RULES: list[ExitRule] = [
    ExitRule(name="ACTUAL", fixed_r_target=0),  # placeholder for actual results
    ExitRule(name="Exit@1.0R", fixed_r_target=1.0),
    ExitRule(name="Exit@1.5R", fixed_r_target=1.5),
    ExitRule(name="Exit@2.0R", fixed_r_target=2.0),
    ExitRule(name="Exit@2.5R", fixed_r_target=2.5),
    ExitRule(name="Exit@3.0R", fixed_r_target=3.0),
    ExitRule(name="50%@1.5R+Trail1.5xATRh", partial_frac=0.5, partial_r=1.5, trail_atr_mult=1.5),
    ExitRule(name="50%@1.5R+Trail2.0xATRh", partial_frac=0.5, partial_r=1.5, trail_atr_mult=2.0),
    ExitRule(name="RegimeDowngrade", regime_downgrade_exit=True),
    ExitRule(name="TimeExit@40h", forced_exit_hours=40, forced_exit_min_r=1.0),
    ExitRule(name="TimeExit@80h", forced_exit_hours=80, forced_exit_min_r=1.0),
    ExitRule(name="TimeExit@120h", forced_exit_hours=120, forced_exit_min_r=1.0),
    ExitRule(name="TimeExit@200h", forced_exit_hours=200, forced_exit_min_r=1.0),
]


@dataclass
class HypotheticalResult:
    """Result of simulating one trade under one exit rule."""
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    bars_held: int = 0
    exit_reason: str = ""
    pnl_dollars: float = 0.0


@dataclass
class ExitRuleStats:
    """Aggregated stats for one exit rule across all trades."""
    rule_name: str = ""
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    mean_r: float = 0.0
    total_r: float = 0.0
    profit_factor: float = 0.0
    net_pnl: float = 0.0
    mean_mfe: float = 0.0
    mfe_capture_pct: float = 0.0
    mean_bars_held: float = 0.0
    median_r: float = 0.0


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate_exit_hypotheticals(
    trades: list,
    hourly: NumpyBars,
    daily_states: dict[int, DailyState],
    daily_idx_map: np.ndarray,
    cfg: SymbolConfig,
    rules: list[ExitRule] | None = None,
) -> dict[str, ExitRuleStats]:
    """Simulate hypothetical exits for a set of trades.

    Parameters
    ----------
    trades : list of TradeRecord
        Completed trades from the backtest engine.
    hourly : NumpyBars
        Hourly OHLCV data for the symbol.
    daily_states : dict mapping daily bar index -> DailyState
        Daily state computed during the backtest.
    daily_idx_map : np.ndarray
        Maps each hourly bar index to its corresponding daily bar index.
    cfg : SymbolConfig
        Symbol configuration (for tick_size, multiplier).
    rules : optional list of ExitRule
        Exit rules to test. Defaults to STANDARD_RULES.

    Returns
    -------
    dict mapping rule name -> ExitRuleStats with aggregated results.
    """
    if rules is None:
        rules = STANDARD_RULES

    results_by_rule: dict[str, list[HypotheticalResult]] = {r.name: [] for r in rules}

    for trade in trades:
        # Find the entry bar index
        entry_idx = _find_bar_index(hourly.times, trade.entry_time)
        if entry_idx < 0:
            continue

        risk_per_unit = abs(trade.entry_price - trade.initial_stop)
        if risk_per_unit <= 0:
            continue

        for rule in rules:
            if rule.name == "ACTUAL":
                # Record actual trade results
                results_by_rule["ACTUAL"].append(HypotheticalResult(
                    r_multiple=trade.r_multiple,
                    mfe_r=trade.mfe_r,
                    bars_held=trade.bars_held,
                    exit_reason=trade.exit_reason,
                    pnl_dollars=trade.pnl_dollars,
                ))
                continue

            hyp = _simulate_one_trade(
                trade, entry_idx, risk_per_unit,
                hourly, daily_states, daily_idx_map,
                cfg, rule,
            )
            results_by_rule[rule.name].append(hyp)

    # Aggregate stats per rule
    stats: dict[str, ExitRuleStats] = {}
    for rule_name, results in results_by_rule.items():
        stats[rule_name] = _compute_rule_stats(rule_name, results)

    return stats


def _find_bar_index(times: np.ndarray, trade_time) -> int:
    """Find the index in times closest to trade_time."""
    if trade_time is None:
        return -1

    if hasattr(trade_time, 'timestamp'):
        # datetime -> numpy datetime64
        ts = np.datetime64(int(trade_time.timestamp() * 1e9), 'ns')
    else:
        ts = np.datetime64(trade_time, 'ns')

    idx = np.searchsorted(times, ts, side='right') - 1
    if idx < 0:
        idx = 0
    if idx >= len(times):
        return -1

    # Allow up to 2-hour tolerance for matching
    diff = abs(int(times[idx]) - int(ts))
    if diff > 2 * 3600 * 1e9:  # 2 hours in nanoseconds
        # Try idx+1
        if idx + 1 < len(times):
            diff2 = abs(int(times[idx + 1]) - int(ts))
            if diff2 < diff:
                idx = idx + 1
                diff = diff2
        if diff > 2 * 3600 * 1e9:
            return -1

    return idx


def _simulate_one_trade(
    trade,
    entry_idx: int,
    risk_per_unit: float,
    hourly: NumpyBars,
    daily_states: dict[int, DailyState],
    daily_idx_map: np.ndarray,
    cfg: SymbolConfig,
    rule: ExitRule,
) -> HypotheticalResult:
    """Simulate a single trade under a hypothetical exit rule."""
    direction = trade.direction
    entry_price = trade.entry_price
    initial_stop = trade.initial_stop

    mfe_price = entry_price
    mfe_r = 0.0
    current_stop = initial_stop

    # Partial exit tracking
    partial_done = False
    partial_r_captured = 0.0
    remaining_frac = 1.0

    # Trail stop for partial exit remainder
    trail_high = entry_price if direction == 1 else entry_price

    # Previous regime for downgrade detection
    prev_regime = None

    for j in range(entry_idx + 1, len(hourly.closes)):
        bars_held = j - entry_idx
        h_high = hourly.highs[j]
        h_low = hourly.lows[j]
        h_close = hourly.closes[j]

        # Update MFE
        if direction == 1:  # LONG
            if h_high > mfe_price:
                mfe_price = h_high
            cur_mfe = (mfe_price - entry_price) / risk_per_unit
            cur_r = (h_close - entry_price) / risk_per_unit
            intrabar_max_r = (h_high - entry_price) / risk_per_unit
        else:  # SHORT
            if h_low < mfe_price:
                mfe_price = h_low
            cur_mfe = (entry_price - mfe_price) / risk_per_unit
            cur_r = (entry_price - h_close) / risk_per_unit
            intrabar_max_r = (entry_price - h_low) / risk_per_unit
        mfe_r = max(mfe_r, cur_mfe)

        # --- Fixed R target exit ---
        if rule.fixed_r_target > 0 and intrabar_max_r >= rule.fixed_r_target:
            return HypotheticalResult(
                r_multiple=rule.fixed_r_target,
                mfe_r=mfe_r,
                bars_held=bars_held,
                exit_reason=f"TARGET_{rule.fixed_r_target}R",
                pnl_dollars=rule.fixed_r_target * risk_per_unit * cfg.multiplier * trade.qty,
            )

        # --- Partial exit at partial_r, trail remainder ---
        if rule.partial_frac > 0 and not partial_done and intrabar_max_r >= rule.partial_r:
            partial_done = True
            partial_r_captured = rule.partial_r * rule.partial_frac
            remaining_frac = 1.0 - rule.partial_frac
            # Initialize trail from this point
            trail_high = h_high if direction == 1 else h_low

        if partial_done and rule.trail_atr_mult > 0:
            # Compute hourly ATR-based trailing stop for remainder
            atrh = _estimate_hourly_atr(hourly, j)
            if direction == 1:
                if h_high > trail_high:
                    trail_high = h_high
                trail_stop = trail_high - rule.trail_atr_mult * atrh
                if h_low <= trail_stop:
                    trail_r = (trail_stop - entry_price) / risk_per_unit
                    blended_r = partial_r_captured + trail_r * remaining_frac
                    return HypotheticalResult(
                        r_multiple=blended_r,
                        mfe_r=mfe_r,
                        bars_held=bars_held,
                        exit_reason=f"PARTIAL+TRAIL",
                        pnl_dollars=blended_r * risk_per_unit * cfg.multiplier * trade.qty,
                    )
            else:
                if h_low < trail_high:
                    trail_high = h_low
                trail_stop = trail_high + rule.trail_atr_mult * atrh
                if h_high >= trail_stop:
                    trail_r = (entry_price - trail_stop) / risk_per_unit
                    blended_r = partial_r_captured + trail_r * remaining_frac
                    return HypotheticalResult(
                        r_multiple=blended_r,
                        mfe_r=mfe_r,
                        bars_held=bars_held,
                        exit_reason=f"PARTIAL+TRAIL",
                        pnl_dollars=blended_r * risk_per_unit * cfg.multiplier * trade.qty,
                    )

        # --- Regime downgrade exit ---
        if rule.regime_downgrade_exit and j < len(daily_idx_map):
            d_idx = int(daily_idx_map[j])
            d = daily_states.get(d_idx)
            if d is not None:
                cur_regime = d.regime.value if hasattr(d.regime, 'value') else str(d.regime)
                if prev_regime is not None and _is_downgrade(prev_regime, cur_regime):
                    return HypotheticalResult(
                        r_multiple=cur_r,
                        mfe_r=mfe_r,
                        bars_held=bars_held,
                        exit_reason="REGIME_DOWNGRADE",
                        pnl_dollars=cur_r * risk_per_unit * cfg.multiplier * trade.qty,
                    )
                prev_regime = cur_regime

        # --- Time-based forced exit ---
        if rule.forced_exit_hours > 0 and bars_held >= rule.forced_exit_hours:
            if cur_r < rule.forced_exit_min_r:
                return HypotheticalResult(
                    r_multiple=cur_r,
                    mfe_r=mfe_r,
                    bars_held=bars_held,
                    exit_reason=f"TIME_EXIT_{rule.forced_exit_hours}h",
                    pnl_dollars=cur_r * risk_per_unit * cfg.multiplier * trade.qty,
                )

        # --- Original stop hit (all rules still respect the initial stop) ---
        if direction == 1 and h_low <= initial_stop:
            stop_r = (initial_stop - entry_price) / risk_per_unit
            if partial_done:
                blended_r = partial_r_captured + stop_r * remaining_frac
            else:
                blended_r = stop_r
            return HypotheticalResult(
                r_multiple=blended_r,
                mfe_r=mfe_r,
                bars_held=bars_held,
                exit_reason="STOP",
                pnl_dollars=blended_r * risk_per_unit * cfg.multiplier * trade.qty,
            )
        if direction == -1 and h_high >= initial_stop:
            stop_r = (entry_price - initial_stop) / risk_per_unit
            if partial_done:
                blended_r = partial_r_captured + stop_r * remaining_frac
            else:
                blended_r = stop_r
            return HypotheticalResult(
                r_multiple=blended_r,
                mfe_r=mfe_r,
                bars_held=bars_held,
                exit_reason="STOP",
                pnl_dollars=blended_r * risk_per_unit * cfg.multiplier * trade.qty,
            )

    # End of data — use close
    last_close = hourly.closes[-1]
    if direction == 1:
        final_r = (last_close - entry_price) / risk_per_unit
    else:
        final_r = (entry_price - last_close) / risk_per_unit
    if partial_done:
        final_r = partial_r_captured + final_r * remaining_frac

    return HypotheticalResult(
        r_multiple=final_r,
        mfe_r=mfe_r,
        bars_held=len(hourly.closes) - entry_idx,
        exit_reason="END_OF_DATA",
        pnl_dollars=final_r * risk_per_unit * cfg.multiplier * trade.qty,
    )


def _estimate_hourly_atr(hourly: NumpyBars, bar_idx: int, period: int = 48) -> float:
    """Estimate hourly ATR at bar_idx using a simple lookback."""
    start = max(0, bar_idx - period)
    if start >= bar_idx:
        return 1.0  # fallback
    highs = hourly.highs[start:bar_idx + 1]
    lows = hourly.lows[start:bar_idx + 1]
    closes = hourly.closes[start:bar_idx + 1]

    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    return float(np.mean(tr)) if len(tr) > 0 else 1.0


_REGIME_RANK = {"STRONG_TREND": 3, "TREND": 2, "RANGE": 1}


def _is_downgrade(prev: str, cur: str) -> bool:
    """Check if regime transitioned to a weaker state."""
    return _REGIME_RANK.get(cur, 0) < _REGIME_RANK.get(prev, 0)


def _compute_rule_stats(rule_name: str, results: list[HypotheticalResult]) -> ExitRuleStats:
    """Aggregate hypothetical results into summary stats."""
    stats = ExitRuleStats(rule_name=rule_name)
    if not results:
        return stats

    rs = np.array([r.r_multiple for r in results])
    mfes = np.array([r.mfe_r for r in results])
    bars = np.array([r.bars_held for r in results])
    pnls = np.array([r.pnl_dollars for r in results])

    stats.trades = len(results)
    stats.wins = int(np.sum(rs > 0))
    stats.win_rate = stats.wins / stats.trades * 100
    stats.mean_r = float(np.mean(rs))
    stats.median_r = float(np.median(rs))
    stats.total_r = float(np.sum(rs))
    stats.net_pnl = float(np.sum(pnls))
    stats.mean_mfe = float(np.mean(mfes))
    stats.mfe_capture_pct = (stats.mean_r / stats.mean_mfe * 100) if stats.mean_mfe > 0 else 0.0
    stats.mean_bars_held = float(np.mean(bars))

    gross_profit = float(np.sum(rs[rs > 0]))
    gross_loss = float(np.abs(np.sum(rs[rs < 0])))
    stats.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return stats


def format_exit_hypotheticals_report(
    all_stats: dict[str, dict[str, ExitRuleStats]],
) -> str:
    """Format exit hypothetical results as a printable report.

    Parameters
    ----------
    all_stats : dict mapping symbol -> (rule_name -> ExitRuleStats)
    """
    lines = ["=" * 100, "INVESTIGATION 1: EXIT HYPOTHETICALS", "=" * 100]

    for symbol, rule_stats in all_stats.items():
        lines.append(f"\n--- {symbol} ---")
        lines.append(
            f"{'Rule':<28} {'Trades':>6} {'WR%':>6} {'MeanR':>8} "
            f"{'MedR':>8} {'TotalR':>8} {'PF':>6} {'NetPnL':>10} "
            f"{'MFE':>8} {'MFECap%':>8} {'AvgBars':>8}"
        )
        lines.append("-" * 118)

        # Sort: ACTUAL first, then by total_r descending
        sorted_rules = sorted(
            rule_stats.values(),
            key=lambda s: (s.rule_name != "ACTUAL", -s.total_r),
        )

        actual_r = None
        for s in sorted_rules:
            if s.rule_name == "ACTUAL":
                actual_r = s.total_r

            pf_str = f"{s.profit_factor:>6.2f}" if s.profit_factor < 100 else "   inf"
            lines.append(
                f"{s.rule_name:<28} {s.trades:>6} {s.win_rate:>5.1f}% "
                f"{s.mean_r:>+8.3f} {s.median_r:>+8.3f} "
                f"{s.total_r:>+8.2f} {pf_str} {s.net_pnl:>+10.2f} "
                f"{s.mean_mfe:>8.3f} {s.mfe_capture_pct:>7.1f}% "
                f"{s.mean_bars_held:>8.1f}"
            )

        # Highlight best alternative
        alternatives = [s for s in sorted_rules if s.rule_name != "ACTUAL" and s.trades > 0]
        if alternatives and actual_r is not None:
            best = max(alternatives, key=lambda s: s.total_r)
            delta = best.total_r - actual_r
            lines.append(f"\n  Best alternative: {best.rule_name}")
            lines.append(f"  Delta vs ACTUAL: {delta:+.2f}R "
                         f"(WR {best.win_rate:.1f}%, PF {best.profit_factor:.2f}, "
                         f"MFE capture {best.mfe_capture_pct:.1f}%)")

    return "\n".join(lines)
