"""Helix Full Diagnostics -- comprehensive analysis with default (optimized) config."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Setup path and aliases
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backtests.swing.config_helix import HelixBacktestConfig
from backtests.swing.engine.helix_portfolio_engine import load_helix_data, run_helix_synchronized
from backtests.swing.analysis.metrics import compute_metrics, compute_sharpe, compute_sortino, compute_max_drawdown
from backtests.shared.diagnostics.snapshot import build_group_snapshot
from backtests.swing.analysis.optimized_baseline import (
    load_phase_mutation_source,
    summarize_optimizer_reference,
)
from backtests.swing.auto.config_mutator import mutate_helix_config

import numpy as np

DATA_DIR = Path("backtests/swing/data/raw")
INITIAL_EQUITY = 10_000.0
DEFAULT_OUTPUT = Path("backtests/swing/auto/output/helix_full_diagnostics.txt")

CRISIS_WINDOWS = [
    ("2022 Bear", datetime(2022, 1, 3), datetime(2022, 10, 13)),
    ("SVB", datetime(2023, 3, 8), datetime(2023, 3, 15)),
    ("Aug 2024 Unwind", datetime(2024, 8, 1), datetime(2024, 8, 5)),
    ("Tariff Shock", datetime(2025, 2, 21), datetime(2025, 4, 7)),
    ("Mar 2026 Slow Burn", datetime(2026, 3, 5), datetime(2026, 3, 27)),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pf(wins_total: float, losses_total: float) -> float:
    if losses_total == 0:
        return float("inf") if wins_total > 0 else 0.0
    return wins_total / abs(losses_total)


def _trade_net_pnl(trade) -> float:
    if hasattr(trade, "net_pnl_dollars"):
        return float(getattr(trade, "net_pnl_dollars", 0.0) or 0.0)
    return float(trade.pnl_dollars) - float(getattr(trade, "commission", 0.0) or 0.0)


def _trade_net_r(trade) -> float:
    if hasattr(trade, "net_r_multiple"):
        return float(getattr(trade, "net_r_multiple", 0.0) or 0.0)
    return float(getattr(trade, "r_multiple", 0.0) or 0.0)


def _wr(trades: list) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if _trade_net_r(t) > 0) / len(trades) * 100


def _avg_r(trades: list) -> float:
    if not trades:
        return 0.0
    return sum(_trade_net_r(t) for t in trades) / len(trades)


def _total_r(trades: list) -> float:
    return sum(_trade_net_r(t) for t in trades)


def _risk_basis_summary(trades: list) -> dict:
    rows = []
    for t in trades:
        target = float(getattr(t, "target_initial_risk_dollars", 0.0) or 0.0)
        actual = float(getattr(t, "actual_initial_risk_dollars", 0.0) or 0.0)
        util = float(getattr(t, "risk_utilization", 0.0) or 0.0)
        if target > 0 and actual > 0:
            rows.append((target, actual, util))
    if not rows:
        return {
            "basis": "actual_initial_risk_dollars_after_fill_rounding_and_caps",
            "trades_with_basis": 0,
        }
    targets = [r[0] for r in rows]
    actuals = [r[1] for r in rows]
    utils = [r[2] for r in rows]
    return {
        "basis": "actual_initial_risk_dollars_after_fill_rounding_and_caps",
        "trades_with_basis": len(rows),
        "target_initial_risk_total": float(sum(targets)),
        "actual_initial_risk_total": float(sum(actuals)),
        "mean_utilization_pct": float(statistics.mean(utils) * 100.0),
        "median_utilization_pct": float(statistics.median(utils) * 100.0),
        "min_utilization_pct": float(min(utils) * 100.0),
        "max_utilization_pct": float(max(utils) * 100.0),
        "underfilled_98pct_count": int(sum(1 for u in utils if u < 0.98)),
        "underfilled_90pct_count": int(sum(1 for u in utils if u < 0.90)),
        "overfilled_102pct_count": int(sum(1 for u in utils if u > 1.02)),
    }


def _compute_pf(trades: list) -> float:
    wins = sum(_trade_net_pnl(t) for t in trades if _trade_net_pnl(t) > 0)
    losses = sum(_trade_net_pnl(t) for t in trades if _trade_net_pnl(t) < 0)
    return _pf(wins, losses)


def _trade_time(t, attr: str) -> datetime | None:
    val = getattr(t, attr, None)
    if val is None:
        return None
    if hasattr(val, "astype"):
        try:
            return val.astype("datetime64[s]").astype(datetime)
        except Exception:
            pass
    if isinstance(val, datetime):
        return val.replace(tzinfo=None) if val.tzinfo else val
    return None


def _print_trade_table(trades: list, label: str, f=None) -> None:
    n = len(trades)
    if n == 0:
        line = f"  {label:30s}  n={n:3d}  (no trades)"
    else:
        wr = _wr(trades)
        ar = _avg_r(trades)
        tr = _total_r(trades)
        pf = _compute_pf(trades)
        pnl = sum(_trade_net_pnl(t) for t in trades)
        line = (f"  {label:30s}  n={n:3d}  WR={wr:5.1f}%  avgR={ar:+.2f}  "
                f"totR={tr:+.2f}  PF={pf:.2f}  PnL=${pnl:+,.0f}")
    print(line)
    if f:
        f.write(line + "\n")


def _out(text: str, f=None) -> None:
    print(text)
    if f:
        f.write(text + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-result",
        default="current",
        help="Phase result to load from phase_state.json (1-4) or 'current' for cumulative mutations.",
    )
    parser.add_argument(
        "--state-path",
        default=str(
            Path(__file__).resolve().parent.parent / "auto" / "helix" / "output" / "phase_state.json"
        ),
        help="Path to the Helix lineage phase_state.json file.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-json", default="", help="Optional path to save a machine-readable summary.")
    parser.add_argument("--title", default="", help="Optional report title override.")
    parser.add_argument("--lineage-label", default="Helix", help="Display label for the report header.")
    parser.add_argument("--start-date", default=None, help="Optional inclusive data start date.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive data end date.")
    parser.add_argument(
        "--equity",
        type=float,
        default=INITIAL_EQUITY,
        help="Initial equity for the replay. Use the same value across strategies for like-for-like diagnostics.",
    )
    args = parser.parse_args()
    initial_equity = float(args.equity)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = load_phase_mutation_source(args.state_path, args.phase_result)

    base_config = HelixBacktestConfig(
        initial_equity=initial_equity,
        data_dir=DATA_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    config = mutate_helix_config(base_config, source.mutations) if source.mutations else base_config
    data = load_helix_data(
        config.symbols,
        config.data_dir,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    result = run_helix_synchronized(data, config)

    all_trades = []
    for sym, sr in result.symbol_results.items():
        for t in sr.trades:
            t.symbol = sym
            all_trades.append(t)

    all_trades.sort(key=lambda t: _trade_time(t, "entry_time") or datetime.min)
    snapshot = build_group_snapshot(
        "Swing Helix Strength / Weakness Snapshot",
        all_trades,
        [
            ("symbol", lambda trade: getattr(trade, "symbol", None)),
            ("setup class", lambda trade: getattr(trade, "setup_class", None)),
            ("exit reason", lambda trade: getattr(trade, "exit_reason", None)),
        ],
        min_count=5,
        width=80,
    )
    report_title = args.title or f"{args.lineage_label.upper()} FULL DIAGNOSTICS ({source.phase_label})"
    optimizer_lines = summarize_optimizer_reference(source.optimizer_reference)
    summary_payload = {
        "strategy": "helix",
        "lineage": args.lineage_label,
        "phase_result": source.phase_result,
        "phase_label": source.phase_label,
        "execution_mode": "synchronized",
        "initial_equity": initial_equity,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "state_path": str(source.state_path),
        "mutations": source.mutations,
    }

    with open(output_path, "w", encoding="utf-8") as f:

        _out("=" * 80, f)
        _out(report_title, f)
        _out(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", f)
        _out(f"Initial Equity: ${initial_equity:,.0f}", f)
        if args.start_date or args.end_date:
            _out(f"Data window: {args.start_date or 'beginning'} through {args.end_date or 'end'}", f)
        _out(f"Symbols: {', '.join(config.symbols)}", f)
        _out("Execution mode: synchronized/shared-capital", f)
        _out("Comparable basis: current-code final optimized config replay", f)
        _out(f"Mutation source: {source.state_path}", f)
        for key, value in sorted(source.mutations.items()):
            _out(f"  {key}: {value}", f)
        if optimizer_lines:
            _out("", f)
            for line in optimizer_lines:
                _out(line, f)
        _out("=" * 80, f)
        _out("", f)
        _out(snapshot, f)
        _out("", f)

        # -- SUMMARY --
        total_pnl = sum(_trade_net_pnl(t) for t in all_trades)
        total_r = _total_r(all_trades)
        _out(f"Total trades: {len(all_trades)}", f)
        _out(f"Total PnL: ${total_pnl:,.2f}", f)
        _out(f"Total R (fee-net): {total_r:+.2f}", f)
        _out(f"Win Rate: {_wr(all_trades):.1f}%", f)
        _out(f"Profit Factor: {_compute_pf(all_trades):.2f}", f)
        avg_win = _avg_r([t for t in all_trades if _trade_net_r(t) > 0])
        avg_loss = _avg_r([t for t in all_trades if _trade_net_r(t) <= 0])
        _out(f"Avg Win R: {avg_win:+.2f}  Avg Loss R: {avg_loss:+.2f}  "
             f"Win/Loss Ratio: {abs(avg_win/avg_loss) if avg_loss else 0:.2f}", f)
        _out(f"Total Commission: ${sum(t.commission for t in all_trades):,.2f}", f)
        risk_basis = _risk_basis_summary(all_trades)
        if risk_basis.get("trades_with_basis", 0):
            _out("Risk Basis: actual initial dollars-at-risk after fill, rounding, and caps", f)
            _out(
                "Sizing Utilization: "
                f"mean={risk_basis['mean_utilization_pct']:.1f}% "
                f"median={risk_basis['median_utilization_pct']:.1f}% "
                f"min={risk_basis['min_utilization_pct']:.1f}% "
                f"max={risk_basis['max_utilization_pct']:.1f}% "
                f"under98={risk_basis['underfilled_98pct_count']} "
                f"under90={risk_basis['underfilled_90pct_count']} "
                f"over102={risk_basis['overfilled_102pct_count']}",
                f,
            )
        summary_payload.update(
            {
                "total_trades": len(all_trades),
                "total_pnl": total_pnl,
                "total_r": total_r,
                "win_rate_pct": _wr(all_trades),
                "profit_factor": _compute_pf(all_trades),
                "avg_win_r": avg_win,
                "avg_loss_r": avg_loss,
                "total_commission": float(sum(t.commission for t in all_trades)),
                "r_basis": "fee_net_actual_initial_risk",
                "gross_total_r": float(sum(getattr(t, "r_multiple", 0.0) for t in all_trades)),
                "risk_basis": risk_basis,
            }
        )

        # Equity curve metrics
        if result.combined_equity is not None and len(result.combined_equity) > 0:
            eq = result.combined_equity
            dd_pct, dd_dollar = compute_max_drawdown(eq)
            sharpe = compute_sharpe(eq)
            sortino = compute_sortino(eq)
            final_eq = eq[-1] if len(eq) > 0 else initial_equity
            ret_pct = (final_eq - initial_equity) / initial_equity * 100
            summary_payload.update(
                {
                    "final_equity": float(final_eq),
                    "net_return_pct": float(ret_pct),
                    "max_drawdown_fraction": float(dd_pct),
                    "max_drawdown_pct": float(dd_pct * 100.0),
                    "max_drawdown_dollars": float(dd_dollar),
                    "sharpe": float(sharpe),
                    "sortino": float(sortino),
                }
            )
            _out(f"\nFinal Equity: ${final_eq:,.2f}", f)
            _out(f"Net Return: {ret_pct:+.1f}%", f)
            _out(f"Max Drawdown: {dd_pct * 100.0:.2f}% (${dd_dollar:,.0f})", f)
            _out(f"Sharpe: {sharpe:.2f}  Sortino: {sortino:.2f}", f)
            if dd_pct > 0:
                calmar = ret_pct / (dd_pct * 100.0)
                summary_payload["calmar"] = float(calmar)
                _out(f"Calmar: {calmar:.2f}", f)
        _out("", f)

        # A) Per-symbol
        _out("=" * 80, f)
        _out("A) PER-SYMBOL TRADE SUMMARY", f)
        _out("=" * 80, f)
        by_symbol = defaultdict(list)
        for t in all_trades:
            by_symbol[t.symbol].append(t)
        for sym in sorted(by_symbol):
            _print_trade_table(by_symbol[sym], sym, f)
            sr = result.symbol_results.get(sym)
            if sr:
                _out(f"     Setups: detected={sr.setups_detected} armed={sr.setups_armed} "
                     f"filled={sr.setups_filled} expired={sr.setups_expired}  "
                     f"Fill rate: {sr.setups_filled/sr.setups_detected*100 if sr.setups_detected else 0:.1f}%", f)
        _out("", f)

        # B) Per-class (A vs D)
        _out("=" * 80, f)
        _out("B) PER-CLASS BREAKDOWN (A = divergence, D = momentum)", f)
        _out("=" * 80, f)
        by_class = defaultdict(list)
        for t in all_trades:
            by_class[t.setup_class or "UNKNOWN"].append(t)
        for cls in sorted(by_class):
            _print_trade_table(by_class[cls], f"Class {cls}", f)
        _out("", f)

        # C) Per-regime
        _out("=" * 80, f)
        _out("C) PER-REGIME BREAKDOWN", f)
        _out("=" * 80, f)
        by_regime = defaultdict(list)
        for t in all_trades:
            by_regime[t.regime_at_entry or "UNKNOWN"].append(t)
        for regime in sorted(by_regime):
            _print_trade_table(by_regime[regime], regime, f)

        # Regime time vs trade allocation
        _out("\n  Regime time distribution:", f)
        for sym in sorted(result.symbol_results):
            sr = result.symbol_results[sym]
            total_days = sr.regime_days_bull + sr.regime_days_bear + sr.regime_days_chop
            if total_days:
                _out(f"    {sym}: BULL={sr.regime_days_bull}d ({sr.regime_days_bull/total_days*100:.0f}%) "
                     f"BEAR={sr.regime_days_bear}d ({sr.regime_days_bear/total_days*100:.0f}%) "
                     f"CHOP={sr.regime_days_chop}d ({sr.regime_days_chop/total_days*100:.0f}%)", f)
        _out("", f)

        # D) Per-direction
        _out("=" * 80, f)
        _out("D) PER-DIRECTION BREAKDOWN", f)
        _out("=" * 80, f)
        for direction, dir_label in [(1, "LONG"), (-1, "SHORT")]:
            dt = [t for t in all_trades if t.direction == direction]
            _print_trade_table(dt, dir_label, f)
        _out("", f)

        # E) Regime x Direction cross-tab
        _out("=" * 80, f)
        _out("E) REGIME x DIRECTION CROSS-TAB", f)
        _out("=" * 80, f)
        for regime in ["BULL", "BEAR", "CHOP"]:
            for direction, dir_label in [(1, "LONG"), (-1, "SHORT")]:
                cell = [t for t in all_trades
                        if t.regime_at_entry == regime and t.direction == direction]
                _print_trade_table(cell, f"{regime}-{dir_label}", f)

        # Counter-regime highlight
        counter = [t for t in all_trades
                   if (t.regime_at_entry == "BULL" and t.direction == -1)
                   or (t.regime_at_entry == "BEAR" and t.direction == 1)]
        if counter:
            _out(f"\n  ** Counter-regime trades: {len(counter)}, "
                 f"avgR={_avg_r(counter):+.2f}, totR={_total_r(counter):+.2f}, "
                 f"PnL=${sum(_trade_net_pnl(t) for t in counter):+,.0f}", f)
        _out("", f)

        # F) 4H Regime alignment
        _out("=" * 80, f)
        _out("F) 4H REGIME ALIGNMENT", f)
        _out("=" * 80, f)
        has_4h = any(getattr(t, 'regime_4h_at_entry', '') for t in all_trades)
        if has_4h:
            agreed = [t for t in all_trades
                      if t.regime_at_entry == getattr(t, 'regime_4h_at_entry', '')]
            disagreed = [t for t in all_trades
                         if t.regime_at_entry != getattr(t, 'regime_4h_at_entry', '')
                         and getattr(t, 'regime_4h_at_entry', '')]
            _print_trade_table(agreed, "Daily-4H AGREED", f)
            _print_trade_table(disagreed, "Daily-4H DISAGREED", f)

            _out(f"\n  Cross-tab:", f)
            _out(f"  {'Daily':8s} {'4H':8s} {'Count':>6s} {'AvgR':>7s} {'WR':>6s}", f)
            for daily in ["BULL", "BEAR", "CHOP"]:
                for r4h in ["BULL", "BEAR", "CHOP"]:
                    cell = [t for t in all_trades
                            if t.regime_at_entry == daily
                            and getattr(t, 'regime_4h_at_entry', '') == r4h]
                    if cell:
                        _out(f"  {daily:8s} {r4h:8s} {len(cell):6d} "
                             f"{_avg_r(cell):+7.3f} {_wr(cell):5.0f}%", f)
        else:
            _out("  No 4H regime data.", f)
        _out("", f)

        # G) Origin timeframe (1H vs 4H)
        _out("=" * 80, f)
        _out("G) ORIGIN TIMEFRAME BREAKDOWN", f)
        _out("=" * 80, f)
        by_tf = defaultdict(list)
        for t in all_trades:
            by_tf[t.origin_tf or "UNKNOWN"].append(t)
        for tf in sorted(by_tf):
            _print_trade_table(by_tf[tf], f"TF: {tf}", f)
        _out("", f)

        # H) Exit reason analysis
        _out("=" * 80, f)
        _out("H) EXIT REASON ANALYSIS", f)
        _out("=" * 80, f)
        by_exit = defaultdict(list)
        for t in all_trades:
            by_exit[t.exit_reason or "UNKNOWN"].append(t)
        for reason in sorted(by_exit):
            _print_trade_table(by_exit[reason], reason, f)
        _out("", f)

        # I) Partial exit analysis
        _out("=" * 80, f)
        _out("I) PARTIAL EXIT & ADD-ON ANALYSIS", f)
        _out("=" * 80, f)
        partial_1 = [t for t in all_trades if t.qty_partial_1 > 0]
        partial_2 = [t for t in all_trades if t.qty_partial_2 > 0]
        adds = [t for t in all_trades if t.add_on_qty > 0]

        _out(f"  Trades hitting +2.5R partial: {len(partial_1)}/{len(all_trades)} "
             f"({len(partial_1)/len(all_trades)*100:.1f}%)", f)
        if partial_1:
            _out(f"    Avg final R: {_avg_r(partial_1):+.2f}  "
                 f"Avg MFE: {np.mean([t.mfe_r for t in partial_1]):.2f}R", f)

        _out(f"  Trades hitting +5R partial:   {len(partial_2)}/{len(all_trades)} "
             f"({len(partial_2)/len(all_trades)*100:.1f}%)", f)
        if partial_2:
            _out(f"    Avg final R: {_avg_r(partial_2):+.2f}  "
                 f"Avg MFE: {np.mean([t.mfe_r for t in partial_2]):.2f}R", f)

        _out(f"  Trades with add-ons:          {len(adds)}/{len(all_trades)} "
             f"({len(adds)/len(all_trades)*100:.1f}%)", f)
        if adds:
            _out(f"    Avg final R: {_avg_r(adds):+.2f}  "
                 f"Avg add-on qty: {np.mean([t.add_on_qty for t in adds]):.0f}", f)
        _out("", f)

        # J) Stop efficiency & MFE analysis
        _out("=" * 80, f)
        _out("J) STOP EFFICIENCY & MFE ANALYSIS", f)
        _out("=" * 80, f)

        stop_dists = [abs(t.entry_price - t.initial_stop) for t in all_trades]
        stop_pcts = [d / t.entry_price * 100 for t, d in zip(all_trades, stop_dists) if t.entry_price]
        _out(f"  Stop distance (pts): mean={np.mean(stop_dists):.2f}  "
             f"median={np.median(stop_dists):.2f}  max={np.max(stop_dists):.2f}", f)
        if stop_pcts:
            _out(f"  Stop distance (%%):  mean={np.mean(stop_pcts):.2f}%  "
                 f"median={np.median(stop_pcts):.2f}%", f)

        # MFE capture for winners
        winners = [t for t in all_trades if _trade_net_r(t) > 0]
        if winners:
            captures = [_trade_net_r(t) / t.mfe_r for t in winners if t.mfe_r > 0]
            if captures:
                _out(f"\n  MFE capture (winners, n={len(winners)}):", f)
                _out(f"    Mean: {np.mean(captures):.1%}  Median: {np.median(captures):.1%}", f)
                _out(f"    Captures < 50%: {sum(1 for c in captures if c < 0.5)} "
                     f"({sum(1 for c in captures if c < 0.5)/len(captures)*100:.0f}%)", f)

        # Loser classification
        losers = [t for t in all_trades if _trade_net_r(t) <= 0]
        if losers:
            right_then_stopped = [t for t in losers if t.mfe_r >= 0.5]
            immediately_wrong = [t for t in losers if t.mfe_r < 0.5]
            _out(f"\n  Loser classification (n={len(losers)}):", f)
            _out(f"    Right-then-stopped (MFE >= 0.5R): {len(right_then_stopped)} "
                 f"({len(right_then_stopped)/len(losers)*100:.0f}%)", f)
            if right_then_stopped:
                _out(f"      Avg MFE: {np.mean([t.mfe_r for t in right_then_stopped]):.2f}R  "
                     f"Avg final R: {_avg_r(right_then_stopped):+.2f}", f)
            _out(f"    Immediately wrong (MFE < 0.5R):   {len(immediately_wrong)} "
                 f"({len(immediately_wrong)/len(losers)*100:.0f}%)", f)

            stopped_at_1r = sum(1 for t in losers if -1.1 <= _trade_net_r(t) <= -0.9)
            _out(f"    Stopped at ~1R: {stopped_at_1r} ({stopped_at_1r/len(losers)*100:.0f}%)", f)
        _out("", f)

        # K) Divergence magnitude analysis (Class A)
        _out("=" * 80, f)
        _out("K) DIVERGENCE MAGNITUDE ANALYSIS (Class A)", f)
        _out("=" * 80, f)
        class_a = [t for t in all_trades if t.setup_class == "A"]
        if class_a:
            div_vals = np.array([getattr(t, 'div_mag_norm', 0.0) for t in class_a])
            rs = np.array([_trade_net_r(t) for t in class_a])
            if div_vals.sum() > 0:
                _out(f"  n={len(class_a)}  div_mag: mean={np.mean(div_vals):.4f}  "
                     f"median={np.median(div_vals):.4f}  max={np.max(div_vals):.4f}", f)
                try:
                    edges = np.percentile(div_vals, [0, 25, 50, 75, 100])
                    _out(f"  {'Quartile':10s} {'Range':>20s} {'Count':>6s} {'AvgR':>7s} {'WR':>6s}", f)
                    for i in range(4):
                        lo, hi = edges[i], edges[i + 1]
                        mask = (div_vals >= lo) & (div_vals < hi) if i < 3 else (div_vals >= lo)
                        if mask.sum():
                            _out(f"  Q{i+1}  {lo:8.4f}-{hi:8.4f}  {mask.sum():6d}  "
                                 f"{np.mean(rs[mask]):+7.3f}  {np.mean(rs[mask]>0)*100:5.0f}%", f)
                except Exception:
                    pass
                if len(div_vals) >= 5:
                    corr = np.corrcoef(div_vals, rs)[0, 1]
                    _out(f"  Correlation(DivMag, R): {corr:+.3f}", f)
            else:
                _out("  No divergence magnitude data.", f)
        else:
            _out("  No Class A trades.", f)
        _out("", f)

        # L) ADX analysis
        _out("=" * 80, f)
        _out("L) ADX AT ENTRY ANALYSIS", f)
        _out("=" * 80, f)
        adx_vals = [t.adx_at_entry for t in all_trades if t.adx_at_entry > 0]
        if adx_vals:
            _out(f"  n={len(adx_vals)}  mean={np.mean(adx_vals):.1f}  "
                 f"median={np.median(adx_vals):.1f}  "
                 f"min={np.min(adx_vals):.1f}  max={np.max(adx_vals):.1f}", f)

            # Bucket by ADX strength
            for label, lo, hi in [("Weak (<20)", 0, 20), ("Moderate (20-30)", 20, 30),
                                   ("Strong (30-40)", 30, 40), ("Very Strong (>40)", 40, 999)]:
                bucket = [t for t in all_trades if lo <= t.adx_at_entry < hi and t.adx_at_entry > 0]
                if bucket:
                    _print_trade_table(bucket, label, f)

            adx_arr = np.array([t.adx_at_entry for t in all_trades if t.adx_at_entry > 0])
            r_arr = np.array([_trade_net_r(t) for t in all_trades if t.adx_at_entry > 0])
            if len(adx_arr) >= 5:
                corr = np.corrcoef(adx_arr, r_arr)[0, 1]
                _out(f"\n  Correlation(ADX, R): {corr:+.3f}", f)
        else:
            _out("  No ADX data.", f)
        _out("", f)

        # M) Crisis window analysis
        _out("=" * 80, f)
        _out("M) CRISIS WINDOW ANALYSIS", f)
        _out("=" * 80, f)
        for cname, cstart, cend in CRISIS_WINDOWS:
            crisis_trades = [t for t in all_trades
                            if (et := _trade_time(t, "entry_time")) and cstart <= et <= cend]
            n = len(crisis_trades)
            tr = _total_r(crisis_trades)
            first_entry = min((_trade_time(t, "entry_time") for t in crisis_trades), default=None)
            first_str = first_entry.strftime("%Y-%m-%d %H:%M") if first_entry else "N/A"
            _out(f"  {cname:25s}  {cstart.date()} -> {cend.date()}  "
                 f"trades={n:3d}  totR={tr:+.2f}  first_entry={first_str}", f)
            for t in crisis_trades:
                et_t = _trade_time(t, "entry_time")
                xt = _trade_time(t, "exit_time")
                dir_l = "LONG" if t.direction == 1 else "SHORT"
                _out(f"    {t.symbol:5s} {t.setup_class:3s} {dir_l:5s} {t.regime_at_entry:6s} "
                     f"entry={et_t.strftime('%Y-%m-%d %H:%M') if et_t else 'N/A':16s} "
                     f"R={_trade_net_r(t):+.2f}  PnL=${_trade_net_pnl(t):+,.0f}  bars={t.bars_held}", f)
        _out("", f)

        # N) Trade gap analysis
        _out("=" * 80, f)
        _out("N) TRADE GAP ANALYSIS (top 10 longest gaps)", f)
        _out("=" * 80, f)
        gaps = []
        for i in range(len(all_trades) - 1):
            xt = _trade_time(all_trades[i], "exit_time")
            et_next = _trade_time(all_trades[i + 1], "entry_time")
            if xt and et_next:
                gap_days = (et_next - xt).total_seconds() / 86400
                gaps.append((gap_days, xt, et_next, all_trades[i], all_trades[i + 1]))

        gaps.sort(key=lambda x: -x[0])
        for rank, (gap_days, xt, et_next, t_prev, t_next) in enumerate(gaps[:10], 1):
            _out(f"  #{rank:2d}  gap={gap_days:6.1f}d  "
                 f"exit={xt.strftime('%Y-%m-%d'):10s} ({t_prev.symbol:5s}) -> "
                 f"entry={et_next.strftime('%Y-%m-%d'):10s} ({t_next.symbol:5s})", f)
        _out("", f)

        # O) Quarterly trade distribution
        _out("=" * 80, f)
        _out("O) QUARTERLY TRADE DISTRIBUTION", f)
        _out("=" * 80, f)
        by_quarter = defaultdict(list)
        for t in all_trades:
            et_t = _trade_time(t, "entry_time")
            if et_t:
                q = (et_t.month - 1) // 3 + 1
                key = f"{et_t.year}-Q{q}"
                by_quarter[key].append(t)
        for qk in sorted(by_quarter):
            qt = by_quarter[qk]
            n = len(qt)
            tr = _total_r(qt)
            wr = _wr(qt)
            bar = "#" * min(n, 50)
            _out(f"  {qk:8s}  n={n:3d}  WR={wr:5.1f}%  totR={tr:+6.2f}  {bar}", f)
        _out("", f)

        # P) Cumulative R curve & drawdown
        _out("=" * 80, f)
        _out("P) CUMULATIVE R CURVE & DRAWDOWN", f)
        _out("=" * 80, f)
        sorted_by_exit = sorted(all_trades, key=lambda t: t.exit_time or datetime.min)
        rs = [_trade_net_r(t) for t in sorted_by_exit]
        cum_r = np.cumsum(rs)

        _out(f"  Total R: {cum_r[-1]:+.2f}  Peak R: {np.max(cum_r):+.2f}", f)
        running_max = np.maximum.accumulate(cum_r)
        drawdowns = cum_r - running_max
        max_dd = np.min(drawdowns)
        max_dd_idx = int(np.argmin(drawdowns))
        _out(f"  Max R drawdown: {max_dd:+.2f} (after trade #{max_dd_idx + 1})", f)

        in_dd = drawdowns < 0
        if in_dd.any():
            longest = current = 0
            for v in in_dd:
                if v:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
            _out(f"  Longest drawdown: {longest} trades", f)

        # Month-by-month
        month_r: dict[str, float] = defaultdict(float)
        month_count: dict[str, int] = defaultdict(int)
        for t in sorted_by_exit:
            if t.exit_time:
                key = t.exit_time.strftime("%Y-%m")
                month_r[key] += _trade_net_r(t)
                month_count[key] += 1

        if month_r:
            _out(f"\n  Month-by-month R:", f)
            _out(f"  {'Month':>7s} {'Trades':>6s} {'R':>8s} {'CumR':>8s}", f)
            cum = 0.0
            for month in sorted(month_r):
                r = month_r[month]
                cum += r
                _out(f"  {month:>7s} {month_count[month]:6d} {r:+8.2f} {cum:+8.2f}", f)
            monthly_rs = list(month_r.values())
            pos_months = sum(1 for r in monthly_rs if r > 0)
            _out(f"\n  Positive months: {pos_months}/{len(monthly_rs)} "
                 f"({pos_months/len(monthly_rs)*100:.0f}%)", f)
            _out(f"  Best month:  {max(monthly_rs):+.2f}R", f)
            _out(f"  Worst month: {min(monthly_rs):+.2f}R", f)
        _out("", f)

        # Q) Streak analysis
        _out("=" * 80, f)
        _out("Q) WIN/LOSS STREAK ANALYSIS", f)
        _out("=" * 80, f)
        outcomes = [1 if _trade_net_r(t) > 0 else 0 for t in sorted_by_exit]
        win_streaks = []
        loss_streaks = []
        current_type = outcomes[0]
        current_len = 1
        for o in outcomes[1:]:
            if o == current_type:
                current_len += 1
            else:
                (win_streaks if current_type == 1 else loss_streaks).append(current_len)
                current_type = o
                current_len = 1
        (win_streaks if current_type == 1 else loss_streaks).append(current_len)

        if win_streaks:
            _out(f"  Win streaks:  max={max(win_streaks)}  avg={np.mean(win_streaks):.1f}  "
                 f"count={len(win_streaks)}", f)
        if loss_streaks:
            _out(f"  Loss streaks: max={max(loss_streaks)}  avg={np.mean(loss_streaks):.1f}  "
                 f"count={len(loss_streaks)}", f)

        # Recovery after 3+ loss streak
        if loss_streaks and max(loss_streaks) >= 3:
            recovery_rs = []
            sc = 0
            for i, o in enumerate(outcomes):
                if o == 0:
                    sc += 1
                else:
                    if sc >= 3:
                        recovery_rs.append(_trade_net_r(sorted_by_exit[i]))
                    sc = 0
            if recovery_rs:
                _out(f"  Post-3+-loss recovery: {len(recovery_rs)} trades, "
                     f"WR={np.mean([r>0 for r in recovery_rs])*100:.0f}%, "
                     f"avgR={np.mean(recovery_rs):+.3f}", f)
        _out("", f)

        # R) Hold time analysis
        _out("=" * 80, f)
        _out("R) HOLD TIME ANALYSIS", f)
        _out("=" * 80, f)
        bars = [t.bars_held for t in all_trades]
        _out(f"  Bars held: mean={np.mean(bars):.1f}  median={np.median(bars):.0f}  "
             f"min={np.min(bars)}  max={np.max(bars)}", f)

        # Hold time vs outcome
        for label, lo, hi in [("Short (1-10)", 1, 11), ("Medium (11-30)", 11, 31),
                               ("Long (31-60)", 31, 61), ("Very Long (>60)", 61, 99999)]:
            bucket = [t for t in all_trades if lo <= t.bars_held < hi]
            if bucket:
                _print_trade_table(bucket, label, f)

        # Correlation
        if len(bars) >= 5:
            corr = np.corrcoef(bars, [_trade_net_r(t) for t in all_trades])[0, 1]
            _out(f"\n  Correlation(bars_held, R): {corr:+.3f}", f)
        _out("", f)

        # S) Size multiplier analysis
        _out("=" * 80, f)
        _out("S) SIZE MULTIPLIER ANALYSIS", f)
        _out("=" * 80, f)
        by_size = defaultdict(list)
        for t in all_trades:
            mult = getattr(t, 'setup_size_mult', 1.0)
            if mult != 1.0:
                by_size[f"mult={mult:.1f}"].append(t)
            else:
                by_size["mult=1.0 (default)"].append(t)
        for label in sorted(by_size):
            _print_trade_table(by_size[label], label, f)
        _out("", f)

        # T) Top 10 best and worst trades
        _out("=" * 80, f)
        _out("T) TOP 10 BEST & WORST TRADES", f)
        _out("=" * 80, f)
        by_r = sorted(all_trades, key=_trade_net_r, reverse=True)

        _out("  Best trades:", f)
        for i, t in enumerate(by_r[:10], 1):
            et_t = _trade_time(t, "entry_time")
            _out(f"    #{i:2d}  {t.symbol:5s} {t.setup_class:3s} {t.regime_at_entry:6s} "
                 f"{et_t.strftime('%Y-%m-%d') if et_t else 'N/A':10s} "
                 f"R={_trade_net_r(t):+7.2f}  PnL=${_trade_net_pnl(t):+,.0f}  "
                 f"MFE={t.mfe_r:.2f}R  bars={t.bars_held}", f)

        _out("\n  Worst trades:", f)
        for i, t in enumerate(reversed(by_r[-10:]), 1):
            et_t = _trade_time(t, "entry_time")
            _out(f"    #{i:2d}  {t.symbol:5s} {t.setup_class:3s} {t.regime_at_entry:6s} "
                 f"{et_t.strftime('%Y-%m-%d') if et_t else 'N/A':10s} "
                 f"R={_trade_net_r(t):+7.2f}  PnL=${_trade_net_pnl(t):+,.0f}  "
                 f"MFE={t.mfe_r:.2f}R  MAE={t.mae_r:.2f}R  reason={t.exit_reason}", f)
        _out("", f)

        # ===================================================================
        # NEW HIGH-VALUE DIAGNOSTICS
        # ===================================================================

        # U) Expectancy stability (rolling window)
        _out("=" * 80, f)
        _out("U) EXPECTANCY STABILITY (rolling 30-trade window)", f)
        _out("=" * 80, f)
        window = 30
        if len(sorted_by_exit) >= window:
            rolling_exp = []
            rolling_wr = []
            for i in range(len(sorted_by_exit) - window + 1):
                chunk = sorted_by_exit[i:i + window]
                rolling_exp.append(_avg_r(chunk))
                rolling_wr.append(_wr(chunk))

            _out(f"  Rolling expectancy (n={len(rolling_exp)} windows):", f)
            _out(f"    Mean: {np.mean(rolling_exp):+.3f}  Stdev: {np.std(rolling_exp):.3f}", f)
            _out(f"    Min:  {np.min(rolling_exp):+.3f}  Max: {np.max(rolling_exp):+.3f}", f)

            # Identify drawdown zones
            neg_windows = [(i, rolling_exp[i]) for i in range(len(rolling_exp)) if rolling_exp[i] < 0]
            if neg_windows:
                _out(f"    Negative expectancy windows: {len(neg_windows)}/{len(rolling_exp)} "
                     f"({len(neg_windows)/len(rolling_exp)*100:.0f}%)", f)
                # Find contiguous negative zones
                zones = []
                z_start = None
                for i, exp in enumerate(rolling_exp):
                    if exp < 0 and z_start is None:
                        z_start = i
                    elif exp >= 0 and z_start is not None:
                        zones.append((z_start, i - 1, min(rolling_exp[z_start:i])))
                        z_start = None
                if z_start is not None:
                    zones.append((z_start, len(rolling_exp)-1, min(rolling_exp[z_start:])))

                _out(f"    Negative expectancy zones: {len(zones)}", f)
                for z_s, z_e, worst in zones:
                    t_s = _trade_time(sorted_by_exit[z_s], "entry_time")
                    t_e = _trade_time(sorted_by_exit[min(z_e + window - 1, len(sorted_by_exit)-1)], "exit_time")
                    _out(f"      Trades {z_s+1}-{z_e+window}: "
                         f"{t_s.strftime('%Y-%m-%d') if t_s else '?'} to "
                         f"{t_e.strftime('%Y-%m-%d') if t_e else '?'}  "
                         f"worst avgR={worst:+.3f}", f)
            else:
                _out(f"    No negative expectancy windows -- consistently positive", f)

            _out(f"\n  Rolling WR:", f)
            _out(f"    Mean: {np.mean(rolling_wr):.1f}%  Min: {np.min(rolling_wr):.1f}%  "
                 f"Max: {np.max(rolling_wr):.1f}%", f)
        else:
            _out(f"  Insufficient trades for rolling analysis (need {window}, have {len(sorted_by_exit)})", f)
        _out("", f)

        # V) MFE/MAE efficiency frontier
        _out("=" * 80, f)
        _out("V) MFE/MAE EFFICIENCY FRONTIER", f)
        _out("=" * 80, f)
        mfe_vals = np.array([t.mfe_r for t in all_trades])
        mae_vals = np.array([t.mae_r for t in all_trades])
        r_vals = np.array([_trade_net_r(t) for t in all_trades])

        _out(f"  MFE: mean={np.mean(mfe_vals):.2f}R  median={np.median(mfe_vals):.2f}R  "
             f"max={np.max(mfe_vals):.2f}R", f)
        _out(f"  MAE: mean={np.mean(mae_vals):.2f}R  median={np.median(mae_vals):.2f}R  "
             f"max={np.max(mae_vals):.2f}R", f)

        # Edge ratio: avg MFE / avg MAE for winners vs losers
        if winners and losers:
            w_mfe = np.mean([t.mfe_r for t in winners])
            w_mae = np.mean([t.mae_r for t in winners])
            l_mfe = np.mean([t.mfe_r for t in losers])
            l_mae = np.mean([t.mae_r for t in losers])
            _out(f"\n  Edge ratios:", f)
            _out(f"    Winners: MFE/MAE = {w_mfe/(abs(w_mae) or 1):.2f}  "
                 f"(MFE={w_mfe:.2f}R, MAE={w_mae:.2f}R)", f)
            _out(f"    Losers:  MFE/MAE = {l_mfe/(abs(l_mae) or 1):.2f}  "
                 f"(MFE={l_mfe:.2f}R, MAE={l_mae:.2f}R)", f)

        # What if we tightened stops? Simulate different stop levels
        # NOTE: mae_r is stored as POSITIVE magnitude (max adverse excursion)
        _out(f"\n  Stop tightening simulation (what-if):", f)
        _out(f"  {'Stop':>8s} {'Saved':>6s} {'Lost':>6s} {'NetR':>8s} {'NewWR':>6s} {'Delta':>8s}", f)
        baseline_r = _total_r(all_trades)
        for stop_mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            saved = lost = 0
            new_rs = []
            for t in all_trades:
                if t.mae_r >= stop_mult:
                    # MAE exceeded this stop level -- would have been stopped
                    new_rs.append(-stop_mult)
                    if _trade_net_r(t) > -stop_mult:
                        lost += 1  # lost a trade that recovered
                    else:
                        saved += 1  # saved from worse loss
                else:
                    new_rs.append(_trade_net_r(t))
            net_r = sum(new_rs)
            new_wr = sum(1 for r in new_rs if r > 0) / len(new_rs) * 100
            _out(f"  {stop_mult:8.2f}R {saved:6d} {lost:6d} {net_r:+8.2f} {new_wr:5.1f}% "
                 f"{net_r - baseline_r:+8.2f}", f)
        _out("", f)

        # W) Regime transition analysis
        _out("=" * 80, f)
        _out("W) REGIME TRANSITION ANALYSIS (daily vs 4H disagreement)", f)
        _out("=" * 80, f)
        regime_changes = []
        for t in all_trades:
            r4h = getattr(t, 'regime_4h_at_entry', '')
            if r4h and r4h != t.regime_at_entry:
                regime_changes.append(t)

        if regime_changes:
            _print_trade_table(regime_changes, "Regime-disagreement trades", f)
            transitions = defaultdict(list)
            for t in regime_changes:
                key = f"{t.regime_at_entry}->{getattr(t, 'regime_4h_at_entry', '?')}"
                transitions[key].append(t)
            for tk in sorted(transitions):
                _print_trade_table(transitions[tk], f"  {tk}", f)
        else:
            _out("  No regime-disagreement trades detected.", f)
        _out("", f)

        # X) Losing trade detail (worst 20)
        _out("=" * 80, f)
        _out("X) LOSING TRADE DETAIL (worst 20)", f)
        _out("=" * 80, f)
        if losers:
            losers_sorted = sorted(losers, key=_trade_net_r)
            _out(f"  Total losers: {len(losers)}/{len(all_trades)} "
                 f"({len(losers)/len(all_trades)*100:.0f}%)", f)
            _out(f"  Total loss: ${sum(_trade_net_pnl(t) for t in losers):+,.0f}", f)
            _out(f"\n  {'#':>3s} {'Entry':12s} {'Sym':5s} {'Cls':3s} {'Dir':5s} "
                 f"{'Regime':6s} {'ADX':>5s} {'R':>7s} {'MFE':>6s} {'MAE':>6s} "
                 f"{'Reason':15s} {'Bars':>4s}", f)
            _out("  " + "-" * 90, f)
            for i, t in enumerate(losers_sorted[:20], 1):
                et_t = _trade_time(t, "entry_time")
                dir_l = "LONG" if t.direction == 1 else "SHORT"
                _out(f"  {i:3d} {et_t.strftime('%Y-%m-%d') if et_t else 'N/A':12s} "
                     f"{t.symbol:5s} {t.setup_class:3s} {dir_l:5s} "
                     f"{t.regime_at_entry:6s} {t.adx_at_entry:5.1f} {_trade_net_r(t):+7.3f} "
                     f"{t.mfe_r:6.2f} {t.mae_r:6.2f} {t.exit_reason:15s} {t.bars_held:4d}", f)
        _out("", f)

        # Y) Filter summary (from shadow tracker)
        _out("=" * 80, f)
        _out("Y) SIGNAL FUNNEL / FILTER SUMMARY", f)
        _out("=" * 80, f)
        if result.filter_summary:
            for sym in sorted(result.filter_summary):
                fs = result.filter_summary[sym]
                _out(f"  {sym}:", f)
                if isinstance(fs, dict):
                    for k, v in sorted(fs.items()):
                        _out(f"    {k}: {v}", f)
                else:
                    _out(f"    {fs}", f)
        else:
            _out("  No filter summary data.", f)
        _out("", f)

        # ===================================================================
        # ADVANCED DIAGNOSTICS
        # ===================================================================

        # Z) Profit concentration (Pareto analysis)
        _out("=" * 80, f)
        _out("Z) PROFIT CONCENTRATION (Pareto analysis)", f)
        _out("=" * 80, f)
        by_pnl_desc = sorted(all_trades, key=_trade_net_r, reverse=True)
        total_gross_r = sum(_trade_net_r(t) for t in all_trades if _trade_net_r(t) > 0)
        if total_gross_r > 0:
            cum_gross = 0.0
            thresholds = {50: None, 75: None, 90: None}
            for i, t in enumerate(by_pnl_desc, 1):
                if _trade_net_r(t) > 0:
                    cum_gross += _trade_net_r(t)
                    pct = cum_gross / total_gross_r * 100
                    for th in thresholds:
                        if thresholds[th] is None and pct >= th:
                            thresholds[th] = i
            for th, count in thresholds.items():
                if count:
                    _out(f"  Top {count} trades ({count/len(all_trades)*100:.0f}%) "
                         f"generate {th}% of gross R", f)

            # How much R comes from trades > 3R?
            big_winners = [t for t in all_trades if _trade_net_r(t) >= 3.0]
            if big_winners:
                big_r = _total_r(big_winners)
                _out(f"\n  Big winners (>=3R): {len(big_winners)} trades, "
                     f"totR={big_r:+.2f} ({big_r/total_gross_r*100:.0f}% of gross wins)", f)
            small_winners = [t for t in all_trades if 0 < _trade_net_r(t) < 1.0]
            if small_winners:
                sm_r = _total_r(small_winners)
                _out(f"  Small winners (<1R): {len(small_winners)} trades, "
                     f"totR={sm_r:+.2f} ({sm_r/total_gross_r*100:.0f}% of gross wins)", f)
        _out("", f)

        # AA) Right-then-stopped deep dive
        _out("=" * 80, f)
        _out("AA) RIGHT-THEN-STOPPED DEEP DIVE", f)
        _out("=" * 80, f)
        right_then_stopped = [t for t in losers if t.mfe_r >= 0.5]
        if right_then_stopped:
            _out(f"  {len(right_then_stopped)} trades went right (MFE >= 0.5R) then lost", f)
            rts_leaked = sum(t.mfe_r - _trade_net_r(t) for t in right_then_stopped)
            _out(f"  Total R leaked (MFE - final R): {rts_leaked:+.2f}R", f)
            _out(f"  Avg MFE before reversal: {np.mean([t.mfe_r for t in right_then_stopped]):.2f}R", f)
            _out(f"  Avg final R: {_avg_r(right_then_stopped):+.2f}", f)

            _out(f"\n  By exit reason:", f)
            by_reason = defaultdict(list)
            for t in right_then_stopped:
                by_reason[t.exit_reason].append(t)
            for reason in sorted(by_reason, key=lambda r: -len(by_reason[r])):
                rt = by_reason[reason]
                leaked = sum(t.mfe_r - _trade_net_r(t) for t in rt)
                _out(f"    {reason:15s}  n={len(rt):3d}  avgMFE={np.mean([t.mfe_r for t in rt]):.2f}R  "
                     f"avgR={_avg_r(rt):+.2f}  leaked={leaked:+.2f}R", f)

            _out(f"\n  By MFE reached:", f)
            for mfe_lo, mfe_hi, label in [(0.5, 1.0, "0.5-1.0R"), (1.0, 2.0, "1.0-2.0R"),
                                           (2.0, 3.0, "2.0-3.0R"), (3.0, 99, "3.0R+")]:
                bucket = [t for t in right_then_stopped if mfe_lo <= t.mfe_r < mfe_hi]
                if bucket:
                    leaked = sum(t.mfe_r - _trade_net_r(t) for t in bucket)
                    _out(f"    MFE {label:8s}  n={len(bucket):3d}  avgR={_avg_r(bucket):+.2f}  "
                         f"leaked={leaked:+.2f}R", f)

            _out(f"\n  By class:", f)
            for cls in ["A", "B", "C", "D"]:
                ct = [t for t in right_then_stopped if t.setup_class == cls]
                if ct:
                    _out(f"    Class {cls}: n={len(ct):3d}  avgMFE={np.mean([t.mfe_r for t in ct]):.2f}R  "
                         f"avgR={_avg_r(ct):+.2f}", f)

            _out(f"\n  By hold time:", f)
            for lo, hi, label in [(1, 11, "1-10 bars"), (11, 31, "11-30 bars"),
                                   (31, 61, "31-60 bars"), (61, 999, "60+ bars")]:
                bucket = [t for t in right_then_stopped if lo <= t.bars_held < hi]
                if bucket:
                    _out(f"    {label:12s}  n={len(bucket):3d}  avgMFE={np.mean([t.mfe_r for t in bucket]):.2f}R  "
                         f"avgR={_avg_r(bucket):+.2f}", f)
        _out("", f)

        # AB) Class x Symbol cross-tab
        _out("=" * 80, f)
        _out("AB) CLASS x SYMBOL CROSS-TAB", f)
        _out("=" * 80, f)
        _out(f"  {'Class':5s} {'Symbol':6s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} "
             f"{'TotR':>8s} {'PF':>6s} {'PnL':>10s}", f)
        _out("  " + "-" * 62, f)
        for cls in ["A", "B", "C", "D"]:
            for sym in sorted(by_symbol):
                cell = [t for t in all_trades if t.setup_class == cls and t.symbol == sym]
                if not cell:
                    continue
                wr = _wr(cell)
                avg_r = _avg_r(cell)
                tot_r = _total_r(cell)
                pf = _compute_pf(cell)
                pnl = sum(_trade_net_pnl(t) for t in cell)
                _out(f"  {cls:5s} {sym:6s} {len(cell):6d} {wr:5.0f}% {avg_r:+7.3f} "
                     f"{tot_r:+8.2f} {pf:6.2f} {pnl:+10,.0f}", f)
            _out("", f)

        # AC) Annual performance summary
        _out("=" * 80, f)
        _out("AC) ANNUAL PERFORMANCE SUMMARY", f)
        _out("=" * 80, f)
        by_year = defaultdict(list)
        for t in all_trades:
            et_t = _trade_time(t, "entry_time")
            if et_t:
                by_year[et_t.year].append(t)

        _out(f"  {'Year':>4s} {'Trades':>6s} {'WR':>6s} {'AvgR':>7s} {'TotR':>8s} "
             f"{'PF':>6s} {'PnL':>10s} {'MaxDD_R':>8s}", f)
        _out("  " + "-" * 62, f)
        for year in sorted(by_year):
            yt = by_year[year]
            wr = _wr(yt)
            avg_r = _avg_r(yt)
            tot_r = _total_r(yt)
            pf = _compute_pf(yt)
            pnl = sum(_trade_net_pnl(t) for t in yt)
            # Year max drawdown in R
            yr_sorted = sorted(yt, key=lambda t: t.exit_time or datetime.min)
            yr_cum = np.cumsum([_trade_net_r(t) for t in yr_sorted])
            yr_peak = np.maximum.accumulate(yr_cum)
            yr_dd = float(np.min(yr_cum - yr_peak)) if len(yr_cum) > 0 else 0.0
            _out(f"  {year:4d} {len(yt):6d} {wr:5.0f}% {avg_r:+7.3f} {tot_r:+8.2f} "
                 f"{pf:6.2f} {pnl:+10,.0f} {yr_dd:+8.2f}", f)
        _out("", f)

        # AD) Exit timing what-if (partial target sensitivity)
        _out("=" * 80, f)
        _out("AD) EXIT TIMING WHAT-IF (partial target sensitivity)", f)
        _out("=" * 80, f)
        _out("  Simulates what if winners were exited at different R thresholds:", f)
        _out(f"  {'Target':>8s} {'HitRate':>8s} {'CapR':>8s} {'vs_Actual':>10s}", f)
        for target in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            # How many trades reached this MFE?
            reached = [t for t in all_trades if t.mfe_r >= target]
            not_reached = [t for t in all_trades if t.mfe_r < target]
            if not reached:
                continue
            # If we exited at target R: reached trades get target R, others keep actual
            sim_r = sum(target for _ in reached) + sum(_trade_net_r(t) for t in not_reached)
            actual_r = _total_r(all_trades)
            hit_rate = len(reached) / len(all_trades) * 100
            _out(f"  {target:8.1f}R {hit_rate:7.1f}% {sim_r:+8.2f} {sim_r - actual_r:+10.2f}", f)
        _out("", f)

        # AE) Symbol-level equity consistency
        _out("=" * 80, f)
        _out("AE) PER-SYMBOL ANNUAL CONSISTENCY", f)
        _out("=" * 80, f)
        for sym in sorted(by_symbol):
            sym_trades = by_symbol[sym]
            sym_by_year = defaultdict(list)
            for t in sym_trades:
                et_t = _trade_time(t, "entry_time")
                if et_t:
                    sym_by_year[et_t.year].append(t)

            _out(f"  {sym}:", f)
            _out(f"    {'Year':>4s} {'N':>4s} {'WR':>6s} {'AvgR':>7s} {'TotR':>8s} {'PF':>6s}", f)
            pos_years = 0
            for year in sorted(sym_by_year):
                yt = sym_by_year[year]
                wr = _wr(yt)
                avg_r = _avg_r(yt)
                tot_r = _total_r(yt)
                pf = _compute_pf(yt)
                if tot_r > 0:
                    pos_years += 1
                _out(f"    {year:4d} {len(yt):4d} {wr:5.0f}% {avg_r:+7.3f} "
                     f"{tot_r:+8.2f} {pf:6.2f}", f)
            total_years = len(sym_by_year)
            _out(f"    Positive years: {pos_years}/{total_years}", f)
            _out("", f)

        _out("=" * 80, f)
        _out("HELIX DIAGNOSTICS COMPLETE", f)
        _out("=" * 80, f)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved summary to {summary_path}")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
