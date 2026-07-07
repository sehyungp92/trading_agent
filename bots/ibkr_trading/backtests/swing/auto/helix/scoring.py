"""Helix composite scoring -- seven frontier-scaled components.

The score is intentionally calibrated as distance-to-target, not minimum
viability. A 270-trade, sub-40% win-rate incumbent should retain credit for
real expectancy, but it should not score like a nearly solved strategy.

Components (default weights):
  - net_profit     (43%): return/R expectancy, gated by win-rate/frequency quality
  - win_rate       (18%): win-rate conversion, ramped from 45% to 55%
  - frequency      (6%): trade count, ramped from 300 to 420 trades
  - winning_trades (11%): absolute winners, ramped from 150 to 230
  - pf             (10%): profit factor, ramped from 1.2 to 3.5
  - exit_quality   (8%): payoff/tail quality, not a standalone objective
  - inv_dd         (4%): R drawdown, full credit near 4R and zero near 14R

Hard rejects are supplied by the plugin as baseline-aware guardrails.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_SCORE_WEIGHTS = {
    "net_profit": 0.43,
    "win_rate": 0.18,
    "frequency": 0.06,
    "winning_trades": 0.11,
    "pf": 0.10,
    "exit_quality": 0.08,
    "inv_dd": 0.04,
}

# Base weights -- sum to 1.0
W_NET_PROFIT = 0.43
W_WIN_RATE = 0.18
W_FREQUENCY = 0.06
W_WINNING_TRADES = 0.11
W_PF = 0.10
W_EXIT_QUALITY = 0.08
W_INV_DD = 0.04

NET_RETURN_FLOOR_PCT = 55.0
NET_RETURN_TARGET_PCT = 150.0
TOTAL_R_FLOOR = 80.0
TOTAL_R_TARGET = 170.0
WIN_RATE_FLOOR = 45.0
WIN_RATE_TARGET = 55.0
TRADE_COUNT_FLOOR = 300.0
TRADE_COUNT_TARGET = 420.0
WINNING_TRADES_FLOOR = 150.0
WINNING_TRADES_TARGET = 230.0
PF_FLOOR = 1.20
PF_TARGET = 3.50
SIDE_QUALITY_FLOOR = 0.90
SIDE_QUALITY_TARGET = 3.50
EXIT_EFFICIENCY_FLOOR = 0.30
EXIT_EFFICIENCY_TARGET = 0.65
WASTE_RATIO_FLOOR = 0.55
WASTE_RATIO_TARGET = 0.85
TAIL_PCT_FLOOR = 0.55
TAIL_PCT_TARGET = 0.85
AVG_WIN_R_FLOOR = 1.50
AVG_WIN_R_TARGET = 2.50
BIG_WINNER_R_FLOOR = 180.0
BIG_WINNER_R_TARGET = 230.0
R_DD_TARGET = 4.0
R_DD_FAIL = 14.0
MIN_SIDE_SAMPLE = 30


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _scale(value: float, floor: float, target: float) -> float:
    if target <= floor:
        return 0.0
    return _clip01((value - floor) / (target - floor))


def _resolve_weights(weights: dict[str, float] | None) -> dict[str, float]:
    resolved = dict(DEFAULT_SCORE_WEIGHTS)
    if weights:
        updates: dict[str, float] = {}
        aliases = {
            "expected_return": "net_profit",
            "total_r": "net_profit",
            "exit_efficiency": "exit_quality",
            "waste_ratio": "exit_quality",
            "tail_preservation": "exit_quality",
            "payoff_quality": "exit_quality",
            "side_quality": "pf",
        }
        for key, value in weights.items():
            canonical = aliases.get(key, key)
            if canonical not in resolved:
                continue
            if canonical == "exit_quality" and key != "exit_quality":
                updates[canonical] = updates.get(canonical, 0.0) + float(value)
            else:
                updates[canonical] = float(value)
        resolved.update(updates)

    total = sum(value for value in resolved.values() if value > 0)
    if total <= 0:
        return dict(DEFAULT_SCORE_WEIGHTS)
    return {key: max(value, 0.0) / total for key, value in resolved.items()}


@dataclass(frozen=True)
class HelixCompositeScore:
    """Frozen score with exactly seven weighted components."""

    net_profit_component: float = 0.0
    win_rate_component: float = 0.0
    frequency_component: float = 0.0
    winning_trades_component: float = 0.0
    pf_component: float = 0.0
    exit_quality_component: float = 0.0
    inv_dd_component: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""

    @property
    def exit_efficiency_component(self) -> float:
        return self.exit_quality_component

    @property
    def waste_ratio_component(self) -> float:
        return self.exit_quality_component

    @property
    def side_quality_component(self) -> float:
        return self.pf_component


@dataclass
class HelixMetrics:
    """Metrics extracted from a Helix backtest run."""
    total_trades: int = 0
    profit_factor: float = 0.0
    net_return_pct: float = 0.0
    max_r_dd: float = 0.0
    exit_efficiency: float = 0.0
    waste_ratio: float = 0.0
    tail_pct: float = 0.0
    bull_pf: float = 0.0
    bear_pf: float = 0.0
    min_regime_pf: float = 0.0
    long_pf: float = 0.0
    short_pf: float = 0.0
    min_side_pf: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    class_d_short_pf: float = 0.0
    class_d_short_trades: int = 0
    total_r: float = 0.0
    gross_win_r: float = 0.0
    gross_loss_r: float = 0.0
    stale_r: float = 0.0
    short_hold_r: float = 0.0
    big_winner_r: float = 0.0
    sharpe: float = 0.0
    calmar_r: float = 0.0
    win_rate: float = 0.0
    winning_trades: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0


def composite_score(
    metrics: HelixMetrics,
    weights: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> HelixCompositeScore:
    """Compute the Helix composite score."""
    hr = hard_rejects or {}
    min_trades = hr.get("min_trades", 200)
    min_pf = hr.get("min_pf", 1.2)
    max_dd = hr.get("max_r_dd", 25.0)
    min_tail = hr.get("min_tail_pct", 0.30)
    min_rpf = hr.get("min_regime_pf", 0.80)
    min_side_pf = hr.get("min_side_pf", 0.0)
    min_win_rate = hr.get("min_win_rate", 0.0)
    min_winning_trades = hr.get("min_winning_trades", 0.0)

    # Hard rejects
    if metrics.total_trades < min_trades:
        return HelixCompositeScore(
            rejected=True,
            reject_reason=f"Too few trades: {metrics.total_trades} < {min_trades:.0f}",
        )
    if metrics.profit_factor < min_pf:
        return HelixCompositeScore(
            rejected=True,
            reject_reason=f"PF too low: {metrics.profit_factor:.2f} < {min_pf}",
        )
    if metrics.max_r_dd > max_dd:
        return HelixCompositeScore(
            rejected=True,
            reject_reason=f"Max R DD too high: {metrics.max_r_dd:.1f} > {max_dd}",
        )
    if metrics.tail_pct < min_tail:
        return HelixCompositeScore(
            rejected=True,
            reject_reason=f"Tail preservation too low: {metrics.tail_pct:.2f} < {min_tail}",
        )
    if metrics.min_regime_pf < min_rpf:
        return HelixCompositeScore(
            rejected=True,
            reject_reason=f"Min regime PF too low: {metrics.min_regime_pf:.2f} < {min_rpf}",
        )
    if min_side_pf > 0 and metrics.min_side_pf < min_side_pf:
        return HelixCompositeScore(
            rejected=True,
            reject_reason=f"Min side PF too low: {metrics.min_side_pf:.2f} < {min_side_pf}",
        )
    if min_win_rate > 0 and metrics.win_rate < min_win_rate:
        return HelixCompositeScore(
            rejected=True,
            reject_reason=f"Win rate too low: {metrics.win_rate:.2f}% < {min_win_rate:.2f}%",
        )
    winning_trades = metrics.winning_trades or (metrics.total_trades * metrics.win_rate / 100.0)
    if min_winning_trades > 0 and winning_trades < min_winning_trades:
        return HelixCompositeScore(
            rejected=True,
            reject_reason=f"Winning trades too low: {winning_trades:.1f} < {min_winning_trades:.1f}",
        )

    w = _resolve_weights(weights)

    # Components
    wr_c = _scale(metrics.win_rate, WIN_RATE_FLOOR, WIN_RATE_TARGET)
    fr_c = _scale(float(metrics.total_trades), TRADE_COUNT_FLOOR, TRADE_COUNT_TARGET)
    wt_c = _scale(winning_trades, WINNING_TRADES_FLOOR, WINNING_TRADES_TARGET)
    raw_return_c = _clip01(
        0.55 * _scale(metrics.net_return_pct, NET_RETURN_FLOOR_PCT, NET_RETURN_TARGET_PCT)
        + 0.45 * _scale(metrics.total_r, TOTAL_R_FLOOR, TOTAL_R_TARGET)
    )
    quality_score = 0.40 * wr_c + 0.25 * fr_c + 0.35 * wt_c
    return_quality_gate = _clip01(0.35 + 0.65 * min(1.0, quality_score / 0.50))
    np_c = raw_return_c * return_quality_gate
    pf_c = _scale(metrics.profit_factor, PF_FLOOR, PF_TARGET)
    exit_eff_c = _scale(metrics.exit_efficiency, EXIT_EFFICIENCY_FLOOR, EXIT_EFFICIENCY_TARGET)
    waste_c = _scale(metrics.waste_ratio, WASTE_RATIO_FLOOR, WASTE_RATIO_TARGET)
    tail_c = _scale(metrics.tail_pct, TAIL_PCT_FLOOR, TAIL_PCT_TARGET)
    avg_win_c = _scale(metrics.avg_win_r, AVG_WIN_R_FLOOR, AVG_WIN_R_TARGET)
    big_winner_c = _scale(metrics.big_winner_r, BIG_WINNER_R_FLOOR, BIG_WINNER_R_TARGET)
    exit_quality_c = _clip01(
        0.25 * exit_eff_c
        + 0.15 * waste_c
        + 0.25 * tail_c
        + 0.25 * avg_win_c
        + 0.10 * big_winner_c
    )
    dd_c = _scale(R_DD_FAIL - metrics.max_r_dd, 0.0, R_DD_FAIL - R_DD_TARGET)

    total = (
        w["net_profit"] * np_c
        + w["win_rate"] * wr_c
        + w["frequency"] * fr_c
        + w["winning_trades"] * wt_c
        + w["pf"] * pf_c
        + w["exit_quality"] * exit_quality_c
        + w["inv_dd"] * dd_c
    )

    return HelixCompositeScore(
        net_profit_component=np_c,
        win_rate_component=wr_c,
        frequency_component=fr_c,
        winning_trades_component=wt_c,
        pf_component=pf_c,
        exit_quality_component=exit_quality_c,
        inv_dd_component=dd_c,
        total=total,
    )


def _trade_net_pnl(trade) -> float:
    if hasattr(trade, "net_pnl_dollars"):
        return float(getattr(trade, "net_pnl_dollars", 0.0) or 0.0)
    return float(getattr(trade, "pnl_dollars", 0.0) or 0.0) - float(getattr(trade, "commission", 0.0) or 0.0)


def _trade_net_r(trade) -> float:
    if hasattr(trade, "net_r_multiple"):
        return float(getattr(trade, "net_r_multiple", 0.0) or 0.0)
    return float(getattr(trade, "r_multiple", 0.0) or 0.0)


def _is_long_trade(trade) -> bool:
    direction = getattr(trade, "direction", 0)
    if hasattr(direction, "name"):
        return str(direction.name).upper() == "LONG"
    if hasattr(direction, "value"):
        direction = direction.value
    if isinstance(direction, str):
        return direction.upper().endswith("LONG") or direction.upper() == "1"
    try:
        return int(direction) > 0
    except (TypeError, ValueError):
        return False


def _pf_from_trades(trades: list) -> float:
    pnls = [_trade_net_pnl(t) for t in trades]
    win = sum(pnl for pnl in pnls if pnl > 0)
    loss = abs(sum(pnl for pnl in pnls if pnl < 0))
    return win / loss if loss > 0 else 999.0


def extract_helix_metrics(
    result,
    initial_equity: float,
) -> HelixMetrics:
    """Extract HelixMetrics from a Helix portfolio result."""
    all_trades = []
    for sr in result.symbol_results.values():
        all_trades.extend(sr.trades)

    if not all_trades:
        return HelixMetrics()

    wins = [t for t in all_trades if _trade_net_r(t) > 0]
    losses = [t for t in all_trades if _trade_net_r(t) <= 0]
    net_pnls = [_trade_net_pnl(t) for t in all_trades]

    gross_win = sum(_trade_net_r(t) for t in wins) if wins else 0.0
    gross_loss = abs(sum(_trade_net_r(t) for t in losses)) if losses else 0.0
    net_gross_win = sum(pnl for pnl in net_pnls if pnl > 0)
    net_gross_loss = abs(sum(pnl for pnl in net_pnls if pnl < 0))
    pf = net_gross_win / net_gross_loss if net_gross_loss > 0 else 999.0
    total_r = sum(_trade_net_r(t) for t in all_trades)

    # Regime-specific PF
    bull_trades = [t for t in all_trades if getattr(t, "regime_at_entry", "") == "BULL"]
    bear_trades = [t for t in all_trades if getattr(t, "regime_at_entry", "") == "BEAR"]

    bull_pf = _pf_from_trades(bull_trades) if bull_trades else 999.0
    bear_pf = _pf_from_trades(bear_trades) if bear_trades else 999.0
    min_regime_pf = min(bull_pf, bear_pf)

    # Direction-specific PF. Under-sampled sides are scored as weak rather
    # than perfect, so the optimizer cannot manufacture quality by deleting a side.
    long_trades = [t for t in all_trades if _is_long_trade(t)]
    short_trades = [t for t in all_trades if not _is_long_trade(t)]
    long_pf = _pf_from_trades(long_trades) if len(long_trades) >= MIN_SIDE_SAMPLE else 0.0
    short_pf = _pf_from_trades(short_trades) if len(short_trades) >= MIN_SIDE_SAMPLE else 0.0
    min_side_pf = min(long_pf, short_pf)

    class_d_short_trades = [
        t for t in short_trades
        if str(getattr(t, "setup_class", "")).upper().endswith("D")
    ]
    class_d_short_pf = (
        _pf_from_trades(class_d_short_trades)
        if len(class_d_short_trades) >= MIN_SIDE_SAMPLE
        else 0.0
    )

    # Exit efficiency: aggregate sum_R / sum_MFE_pos
    sum_r = total_r
    sum_mfe_pos = sum(t.mfe_r for t in all_trades if t.mfe_r > 0)
    exit_eff = sum_r / sum_mfe_pos if sum_mfe_pos > 0 else 0.0

    # Waste ratio: 1 - (|stale_R| + |short_hold_R|) / gross_win_R
    stale_trades = [t for t in all_trades if getattr(t, "exit_reason", "") == "STALE"]
    short_hold_trades = [t for t in all_trades if getattr(t, "bars_held", 0) <= 10 and _trade_net_r(t) < 0]
    stale_r = abs(sum(_trade_net_r(t) for t in stale_trades)) if stale_trades else 0.0
    short_hold_r = abs(sum(_trade_net_r(t) for t in short_hold_trades)) if short_hold_trades else 0.0
    waste = (stale_r + short_hold_r) / gross_win if gross_win > 0 else 0.0
    waste_ratio = max(0.0, 1.0 - waste)

    # Tail preservation: big winners (>=3R) as pct of gross win R
    big_winners = [t for t in wins if _trade_net_r(t) >= 3.0]
    big_winner_r = sum(_trade_net_r(t) for t in big_winners) if big_winners else 0.0
    tail_pct = big_winner_r / gross_win if gross_win > 0 else 0.0

    # Max R drawdown
    cum_r = np.cumsum([_trade_net_r(t) for t in all_trades])
    peak_r = np.maximum.accumulate(cum_r)
    r_dd = peak_r - cum_r
    max_r_dd = float(np.max(r_dd)) if len(r_dd) > 0 else 0.0

    # Net return
    eq = result.combined_equity
    if len(eq) > 0:
        net_ret = (eq[-1] - initial_equity) / initial_equity * 100
    else:
        net_ret = 0.0

    # Sharpe
    sharpe = 0.0
    if len(eq) > 2:
        hourly_returns = np.diff(eq) / eq[:-1]
        hourly_returns = hourly_returns[~np.isnan(hourly_returns)]
        if len(hourly_returns) > 1:
            mu = float(np.mean(hourly_returns))
            sigma = float(np.std(hourly_returns))
            if sigma > 0:
                sharpe = mu / sigma * math.sqrt(252.0 * 7.0)

    # Calmar (R-based)
    calmar_r = total_r / max_r_dd if max_r_dd > 0 else 0.0

    # Win rate
    wr = len(wins) / len(all_trades) * 100 if all_trades else 0.0
    avg_win = sum(_trade_net_r(t) for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(_trade_net_r(t) for t in losses) / len(losses) if losses else 0.0

    return HelixMetrics(
        total_trades=len(all_trades),
        profit_factor=pf,
        net_return_pct=net_ret,
        max_r_dd=max_r_dd,
        exit_efficiency=max(0.0, min(1.0, exit_eff)),
        waste_ratio=waste_ratio,
        tail_pct=tail_pct,
        bull_pf=bull_pf,
        bear_pf=bear_pf,
        min_regime_pf=min_regime_pf,
        long_pf=long_pf,
        short_pf=short_pf,
        min_side_pf=min_side_pf,
        long_trades=len(long_trades),
        short_trades=len(short_trades),
        class_d_short_pf=class_d_short_pf,
        class_d_short_trades=len(class_d_short_trades),
        total_r=total_r,
        gross_win_r=gross_win,
        gross_loss_r=gross_loss,
        stale_r=stale_r,
        short_hold_r=short_hold_r,
        big_winner_r=big_winner_r,
        sharpe=sharpe,
        calmar_r=calmar_r,
        win_rate=wr,
        winning_trades=float(len(wins)),
        avg_win_r=avg_win,
        avg_loss_r=avg_loss,
    )
