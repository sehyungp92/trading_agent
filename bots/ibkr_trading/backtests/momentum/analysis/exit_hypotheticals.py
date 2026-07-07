"""Momentum exit hypotheticals — what-if exit rule testing.

For each trade, replays bars forward from entry applying alternative
exit rules. Answers: "Would a different exit strategy improve results?"

Supports NQDTC (5-min) and Vdubus (15-min) strategies.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ExitRule:
    """Defines an alternative exit rule to test."""
    name: str
    should_exit: callable  # (bar_idx, entry_price, direction, r_base, mfe_r, bars_held, bar_h, bar_l, bar_c) -> bool
    description: str = ""


@dataclass
class HypotheticalResult:
    """Result of applying one exit rule to one trade."""
    rule_name: str
    exit_price: float = 0.0
    exit_bar: int = 0
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    bars_held: int = 0


def _build_nqdtc_rules() -> list[ExitRule]:
    """Exit rules to test for NQDTC strategy."""
    rules = []

    # Flat TP targets
    for target in [1.0, 1.5, 2.0]:
        t = target
        rules.append(ExitRule(
            name=f"FLAT_TP_{t:.0f}R" if t == int(t) else f"FLAT_TP_{t}R",
            should_exit=lambda idx, ep, d, rb, mfe, bh, h, l, c, _t=t: (
                ((h - ep) / rb if d == 1 else (ep - l) / rb) >= _t if rb > 0 else False
            ),
        ))

    # No chandelier trailing
    rules.append(ExitRule(
        name="NO_CHANDELIER",
        should_exit=lambda idx, ep, d, rb, mfe, bh, h, l, c: False,
        description="Disable chandelier trailing stop",
    ))

    # Early chandelier (activate at 0.5R instead of 1R)
    rules.append(ExitRule(
        name="EARLY_CHANDELIER",
        should_exit=lambda idx, ep, d, rb, mfe, bh, h, l, c: mfe >= 0.5 and (
            (mfe - ((c - ep) / rb if d == 1 else (ep - c) / rb)) > 0.4 if rb > 0 else False
        ),
    ))

    # No stale exit
    rules.append(ExitRule(
        name="NO_STALE",
        should_exit=lambda idx, ep, d, rb, mfe, bh, h, l, c: False,
        description="Remove stale bar exit rule",
    ))

    # Max loss caps
    for cap in [2.0, 3.0]:
        c_val = cap
        rules.append(ExitRule(
            name=f"MAX_LOSS_{c_val:.0f}R",
            should_exit=lambda idx, ep, d, rb, mfe, bh, h, l, c, _c=c_val: (
                ((ep - l) / rb if d == 1 else (h - ep) / rb) >= _c if rb > 0 else False
            ),
        ))

    return rules


def _build_vdubus_rules() -> list[ExitRule]:
    """Exit rules to test for Vdubus strategy."""
    rules = []

    for target in [1.0, 1.5, 2.0]:
        t = target
        rules.append(ExitRule(
            name=f"FLAT_{t:.0f}R" if t == int(t) else f"FLAT_{t}R",
            should_exit=lambda idx, ep, d, rb, mfe, bh, h, l, c, _t=t: (
                ((h - ep) / rb if d == 1 else (ep - l) / rb) >= _t if rb > 0 else False
            ),
        ))

    rules.append(ExitRule(
        name="TIGHT_TRAIL",
        should_exit=lambda idx, ep, d, rb, mfe, bh, h, l, c: mfe >= 0.5 and (
            (mfe - ((c - ep) / rb if d == 1 else (ep - c) / rb)) > 0.3 if rb > 0 else False
        ),
    ))

    return rules


def _simulate_trade(
    trade,
    bar_opens: np.ndarray,
    bar_highs: np.ndarray,
    bar_lows: np.ndarray,
    bar_closes: np.ndarray,
    bar_times: np.ndarray,
    rule: ExitRule,
    point_value: float,
) -> HypotheticalResult | None:
    """Replay one trade forward applying an alternative exit rule."""
    entry_price = trade.entry_price
    direction = trade.direction
    stop0 = getattr(trade, 'initial_stop', None) or getattr(trade, 'stop0', 0.0)
    r_base = abs(entry_price - stop0)
    if r_base <= 0:
        return None

    # Find entry bar
    entry_ts = trade.entry_time
    if hasattr(entry_ts, 'value'):
        entry_ns = entry_ts.value
    elif isinstance(entry_ts, np.datetime64):
        entry_ns = entry_ts
    else:
        entry_ns = np.datetime64(entry_ts, 'ns')

    start_idx = int(np.searchsorted(bar_times, entry_ns, side='left'))
    if start_idx >= len(bar_times):
        return None

    # Find exit bar (actual trade end)
    exit_ts = trade.exit_time
    if hasattr(exit_ts, 'value'):
        exit_ns = exit_ts.value
    elif isinstance(exit_ts, np.datetime64):
        exit_ns = exit_ts
    else:
        exit_ns = np.datetime64(exit_ts, 'ns')
    end_idx = int(np.searchsorted(bar_times, exit_ns, side='right'))
    # Allow some buffer beyond actual exit
    max_idx = min(end_idx + 50, len(bar_closes))

    mfe_r = 0.0
    current_stop = stop0

    for i in range(start_idx, max_idx):
        h = float(bar_highs[i])
        l = float(bar_lows[i])
        c = float(bar_closes[i])
        bars_held = i - start_idx

        # Update MFE
        if direction == 1:
            cur_mfe = (h - entry_price) / r_base
        else:
            cur_mfe = (entry_price - l) / r_base
        mfe_r = max(mfe_r, cur_mfe)

        # Check stop first
        if direction == 1 and l <= current_stop:
            pnl = (current_stop - entry_price) / r_base
            return HypotheticalResult(
                rule_name=rule.name, exit_price=current_stop,
                exit_bar=i, r_multiple=pnl, mfe_r=mfe_r, bars_held=bars_held,
            )
        elif direction == -1 and h >= current_stop:
            pnl = (entry_price - current_stop) / r_base
            return HypotheticalResult(
                rule_name=rule.name, exit_price=current_stop,
                exit_bar=i, r_multiple=pnl, mfe_r=mfe_r, bars_held=bars_held,
            )

        # Check rule exit
        r_now = (c - entry_price) / r_base if direction == 1 else (entry_price - c) / r_base
        if rule.should_exit(i, entry_price, direction, r_base, mfe_r, bars_held, h, l, c):
            return HypotheticalResult(
                rule_name=rule.name, exit_price=c,
                exit_bar=i, r_multiple=r_now, mfe_r=mfe_r, bars_held=bars_held,
            )

    # End of data
    last_c = float(bar_closes[max_idx - 1]) if max_idx > 0 else entry_price
    r_final = (last_c - entry_price) / r_base if direction == 1 else (entry_price - last_c) / r_base
    return HypotheticalResult(
        rule_name=rule.name, exit_price=last_c,
        exit_bar=max_idx - 1, r_multiple=r_final, mfe_r=mfe_r,
        bars_held=max_idx - start_idx,
    )


def exit_hypotheticals_report(
    trades: list,
    bar_data: tuple,
    bar_times: np.ndarray,
    strategy: str,
    point_value: float = 2.0,
) -> str:
    """Generate exit hypotheticals comparison report.

    Args:
        trades: Completed trade records.
        bar_data: Tuple of (opens, highs, lows, closes, volumes) numpy arrays.
        bar_times: Corresponding timestamps array.
        strategy: One of "nqdtc", "vdubus".
        point_value: Dollar value per point (default 2.0 for MNQ).
    """
    lines = ["=" * 60]
    lines.append(f"  {strategy.upper()} EXIT HYPOTHETICALS REPORT")
    lines.append("=" * 60)
    lines.append("")

    if not trades:
        lines.append("  No trades to analyze.")
        return "\n".join(lines)

    opens, highs, lows, closes, *_ = bar_data

    # Build rules for strategy
    if strategy.lower() == "nqdtc":
        rules = _build_nqdtc_rules()
    elif strategy.lower() == "vdubus":
        rules = _build_vdubus_rules()
    else:
        lines.append(f"  Unknown strategy: {strategy}")
        return "\n".join(lines)

    # Actual baseline
    actual_r = np.array([t.r_multiple for t in trades])
    actual_wr = float(np.mean(actual_r > 0)) * 100
    actual_mean_r = float(np.mean(actual_r))
    actual_total_r = float(np.sum(actual_r))
    pos_r = actual_r[actual_r > 0]
    neg_r = actual_r[actual_r <= 0]
    actual_pf = float(np.sum(pos_r) / abs(np.sum(neg_r))) if len(neg_r) > 0 and np.sum(neg_r) != 0 else float('inf')
    actual_mfe = float(np.mean([t.mfe_r for t in trades]))

    # Test each rule
    rule_results: dict[str, list[HypotheticalResult]] = {}
    for rule in rules:
        results = []
        for trade in trades:
            res = _simulate_trade(
                trade, opens, highs, lows, closes, bar_times, rule, point_value,
            )
            if res is not None:
                results.append(res)
        rule_results[rule.name] = results

    # Build comparison table
    header = (
        f"  {'Rule':22s} {'Trades':>6s} {'WR%':>6s} {'MeanR':>7s} "
        f"{'TotalR':>8s} {'PF':>6s} {'MFEcap':>7s} {'vs ACT':>8s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    # Actual row
    lines.append(
        f"  {'ACTUAL':22s} {len(trades):6d} {actual_wr:5.1f}% "
        f"{actual_mean_r:+7.3f} {actual_total_r:+8.1f} "
        f"{actual_pf:6.2f} {actual_mfe:6.2f}%     —"
    )

    best_rule = None
    best_delta = 0.0

    for rule in rules:
        results = rule_results[rule.name]
        if not results:
            continue
        r_arr = np.array([r.r_multiple for r in results])
        wr = float(np.mean(r_arr > 0)) * 100
        mean_r = float(np.mean(r_arr))
        total_r = float(np.sum(r_arr))
        pr = r_arr[r_arr > 0]
        nr = r_arr[r_arr <= 0]
        pf = float(np.sum(pr) / abs(np.sum(nr))) if len(nr) > 0 and np.sum(nr) != 0 else float('inf')
        mfe_arr = np.array([r.mfe_r for r in results])
        mfe_cap = float(np.mean(r_arr / np.maximum(mfe_arr, 0.01))) * 100

        delta = total_r - actual_total_r
        if delta > best_delta:
            best_delta = delta
            best_rule = rule.name

        delta_str = f"{delta:+8.1f}"
        lines.append(
            f"  {rule.name:22s} {len(results):6d} {wr:5.1f}% "
            f"{mean_r:+7.3f} {total_r:+8.1f} "
            f"{pf:6.2f} {mfe_cap:6.1f}% {delta_str}"
        )

    lines.append("")
    if best_rule and best_delta > 0:
        lines.append(f"  Best alternative: {best_rule} (+{best_delta:.1f}R vs ACTUAL)")
    else:
        lines.append("  ACTUAL exit rules are optimal — no alternative improves total R")

    lines.append("")
    lines.append("  MFEcap = avg(R / MFE) — how much of favorable excursion is captured")
    lines.append("  vs ACT = difference in total R vs actual exit rules")

    return "\n".join(lines)
