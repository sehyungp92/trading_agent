"""ATRSS Full Diagnostics -- comprehensive analysis with optimized config.

Runs the ATRSS backtest with the phased-auto-optimized config and generates
a complete diagnostic report using all 22 analysis functions.

Usage:
    python -m backtests.swing.analysis.atrss_full_diagnostics
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Setup path and aliases
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backtests.swing.analysis.atrss_diagnostics import (
    atrss_addon_analysis,
    atrss_adx_edge_analysis,
    atrss_bias_alignment,
    atrss_breakout_arm_diagnostic,
    atrss_crisis_window_analysis,
    atrss_entry_type_drilldown,
    atrss_exit_analysis,
    atrss_filter_rejection_detail,
    atrss_losing_trade_detail,
    atrss_mfe_cohort_segmentation,
    atrss_monthly_returns,
    atrss_order_fill_rate,
    atrss_position_occupancy,
    atrss_profit_concentration,
    atrss_r_curve,
    atrss_regime_time_report,
    atrss_right_then_stopped,
    atrss_rolling_edge,
    atrss_signal_funnel,
    atrss_stop_efficiency,
    atrss_streak_analysis,
    atrss_time_analysis,
)
from backtests.swing.analysis.metrics import (
    compute_buy_and_hold,
    compute_max_drawdown,
    compute_metrics,
    compute_sharpe,
    compute_sortino,
)
from backtests.swing.analysis.reports import (
    behavior_report,
    buy_and_hold_report,
    diagnostic_report,
    performance_report,
)
from backtests.swing.config import AblationFlags, BacktestConfig, SlippageConfig
from backtests.swing.engine.portfolio_engine import PortfolioResult, run_synchronized
from backtests.shared.diagnostics.snapshot import build_group_snapshot
from backtests.swing.analysis.optimized_baseline import (
    load_phase_mutation_source,
    summarize_optimizer_reference,
)
from backtests.swing.auto.config_mutator import mutate_atrss_config

import numpy as np

DATA_DIR = Path("backtests/swing/data/raw")
INITIAL_EQUITY = 10_000.0
DEFAULT_OUTPUT = Path("backtests/swing/auto/output/atrss_full_diagnostics.txt")


def _out(text: str, f=None) -> None:
    print(text)
    if f is not None:
        f.write(text + "\n")


def _trade_net_pnl(trade) -> float:
    return float(trade.pnl_dollars) - float(getattr(trade, "commission", 0.0) or 0.0)


def _net_profit_factor(trades: list) -> float:
    gross_profit = sum(_trade_net_pnl(t) for t in trades if _trade_net_pnl(t) > 0)
    gross_loss = abs(sum(_trade_net_pnl(t) for t in trades if _trade_net_pnl(t) < 0))
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def _load_data(symbols: list[str], data_dir: Path):
    from backtests.swing.data.cache import load_bars
    from backtests.swing.data.preprocessing import (
        align_daily_to_hourly,
        build_numpy_arrays,
        filter_rth,
        normalize_timezone,
    )
    from backtests.swing.engine.portfolio_engine import PortfolioData

    data = PortfolioData()
    for sym in symbols:
        hourly_path = data_dir / f"{sym}_1h.parquet"
        daily_path = data_dir / f"{sym}_1d.parquet"
        if not hourly_path.exists() or not daily_path.exists():
            print(f"WARNING: Missing data for {sym}, skipping")
            continue
        h_df = normalize_timezone(load_bars(hourly_path))
        h_df = filter_rth(h_df)
        d_df = normalize_timezone(load_bars(daily_path))
        data.hourly[sym] = build_numpy_arrays(h_df)
        data.daily[sym] = build_numpy_arrays(d_df)
        data.daily_idx_maps[sym] = align_daily_to_hourly(h_df, d_df)
        print(f"  Loaded {sym}: {len(h_df)} hourly, {len(d_df)} daily bars")
    return data


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
            Path(__file__).resolve().parent.parent / "auto" / "atrss" / "output" / "phase_state.json"
        ),
        help="Path to the ATRSS phase_state.json file.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-json", default="", help="Optional path to save a machine-readable summary.")
    parser.add_argument("--title", default="", help="Optional report title override.")
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
    mutations = source.mutations
    print(f"Optimized mutations ({len(mutations)}): {mutations}")

    # Base config (matches plugin.py baseline)
    base_config = BacktestConfig(
        symbols=["QQQ", "GLD"],
        initial_equity=initial_equity,
        data_dir=DATA_DIR,
        flags=AblationFlags(stall_exit=False),
        slippage=SlippageConfig(commission_per_contract=1.00),
        fixed_qty=10,
    )

    # Apply phased-auto mutations
    config = mutate_atrss_config(base_config, mutations) if mutations else base_config

    print(f"Running ATRSS backtest: symbols={config.symbols}, "
          f"equity=${config.initial_equity:,.0f}, fixed_qty={config.fixed_qty}")
    print(f"Loading data from {DATA_DIR}...")
    data = _load_data(config.symbols, DATA_DIR)
    print("Running backtest...")
    result = run_synchronized(data, config)

    # Collect all trades
    all_trades: list = []
    for sym, sr in result.symbol_results.items():
        for t in sr.trades:
            t.symbol = sym
            all_trades.append(t)
    all_trades.sort(key=lambda t: t.entry_time or datetime.min)

    print(f"Total trades: {len(all_trades)}")
    snapshot = build_group_snapshot(
        "ATRSS Strength / Weakness Snapshot",
        all_trades,
        [
            ("symbol", lambda trade: getattr(trade, "symbol", None)),
            ("entry type", lambda trade: getattr(trade, "entry_type", None)),
            ("exit reason", lambda trade: getattr(trade, "exit_reason", None)),
        ],
        min_count=5,
    )
    report_title = args.title or f"ATRSS FULL DIAGNOSTICS ({source.phase_label})"
    optimizer_lines = summarize_optimizer_reference(source.optimizer_reference)
    summary_payload = {
        "strategy": "atrss",
        "phase_result": source.phase_result,
        "phase_label": source.phase_label,
        "execution_mode": "synchronized",
        "initial_equity": initial_equity,
        "state_path": str(source.state_path),
        "mutations": mutations,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        _out("=" * 80, f)
        _out(report_title, f)
        _out(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", f)
        _out(f"Initial Equity: ${initial_equity:,.0f}", f)
        _out(f"Symbols: {', '.join(config.symbols)}", f)
        _out("Execution mode: synchronized/shared-capital", f)
        _out("Comparable basis: current-code final optimized config replay", f)
        _out(f"Mutation source: {source.state_path}", f)
        _out(f"Base: stall_exit=False, commission=$1.00/contract", f)
        for mk, mv in sorted(mutations.items()):
            _out(f"  {mk}: {mv}", f)
        if optimizer_lines:
            _out("", f)
            for line in optimizer_lines:
                _out(line, f)
        _out("=" * 80, f)
        _out("", f)
        _out(snapshot, f)
        _out("", f)

        # ==================================================================
        # AGGREGATE SUMMARY
        # ==================================================================
        if all_trades:
            total_pnl = sum(_trade_net_pnl(t) for t in all_trades)
            total_r = sum(t.r_multiple for t in all_trades)
            n_wins = sum(1 for t in all_trades if t.r_multiple > 0)
            wr = n_wins / len(all_trades) * 100

            # Build combined equity for Sharpe/Sortino
            eq = result.combined_equity

            pf = _net_profit_factor(all_trades)
            final_equity = float(eq[-1]) if len(eq) else initial_equity + total_pnl
            net_return_pct = (final_equity - initial_equity) / initial_equity * 100.0
            open_mtm_pnl = final_equity - initial_equity - total_pnl
            max_dd_fraction, max_dd_dollar = compute_max_drawdown(eq) if len(eq) > 1 else (0, 0)

            sharpe = compute_sharpe(eq) if len(eq) > 1 else 0.0
            sortino = compute_sortino(eq) if len(eq) > 1 else 0.0
            summary_payload.update(
                {
                    "total_trades": len(all_trades),
                    "win_rate_pct": wr,
                    "profit_factor": pf,
                    "equity_basis": "mark_to_market",
                    "total_r": total_r,
                    "total_pnl": total_pnl,
                    "open_mtm_pnl": open_mtm_pnl,
                    "final_equity": final_equity,
                    "net_return_pct": net_return_pct,
                    "max_drawdown_fraction": max_dd_fraction,
                    "max_drawdown_pct": max_dd_fraction * 100.0,
                    "max_drawdown_dollars": max_dd_dollar,
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "avg_r": float(np.mean([t.r_multiple for t in all_trades])),
                    "avg_hold_bars": float(np.mean([t.bars_held for t in all_trades])),
                }
            )

            _out("--- AGGREGATE SUMMARY ---", f)
            _out(f"  Total trades:    {len(all_trades)}", f)
            _out(f"  Win rate:        {wr:.1f}%", f)
            _out(f"  Profit factor:   {pf:.2f}", f)
            _out(f"  Total R:         {total_r:+.2f}", f)
            _out(f"  Total PnL:       ${total_pnl:+,.2f}", f)
            _out(f"  Open MTM PnL:    ${open_mtm_pnl:+,.2f}", f)
            _out(f"  Final equity:    ${final_equity:,.2f}", f)
            _out(f"  Net return:      {net_return_pct:+.1f}%", f)
            _out(f"  Max drawdown:    {max_dd_fraction:.2%} (${max_dd_dollar:,.2f})", f)
            _out(f"  Sharpe:          {sharpe:.2f}", f)
            _out(f"  Sortino:         {sortino:.2f}", f)
            _out(f"  Avg R:           {np.mean([t.r_multiple for t in all_trades]):+.3f}", f)
            _out(f"  Avg hold (bars): {np.mean([t.bars_held for t in all_trades]):.1f}", f)
            _out("", f)

        # ==================================================================
        # PER-SYMBOL SUMMARY TABLE
        # ==================================================================
        _out("=" * 80, f)
        _out("PER-SYMBOL SUMMARY", f)
        _out("=" * 80, f)
        _out(f"  {'Symbol':>6s} {'N':>5s} {'WR':>5s} {'PF':>6s} {'AvgR':>7s} "
             f"{'TotR':>8s} {'PnL':>10s} {'MaxDD':>7s} {'Sharpe':>7s}", f)
        _out("  " + "-" * 70, f)

        for sym in sorted(result.symbol_results):
            sr = result.symbol_results[sym]
            trades = sr.trades
            if not trades:
                _out(f"  {sym:>6s}    -- no trades --", f)
                continue
            n = len(trades)
            wr_s = sum(1 for t in trades if t.r_multiple > 0) / n * 100
            pf_s = _net_profit_factor(trades)
            avg_r = np.mean([t.r_multiple for t in trades])
            tot_r = sum(t.r_multiple for t in trades)
            pnl_s = sum(_trade_net_pnl(t) for t in trades)
            dd_pct, _ = compute_max_drawdown(sr.equity_curve) if len(sr.equity_curve) > 1 else (0, 0)
            sh = compute_sharpe(sr.equity_curve) if len(sr.equity_curve) > 1 else 0.0
            _out(f"  {sym:>6s} {n:5d} {wr_s:4.0f}% {min(pf_s, 99.99):6.2f} {avg_r:+7.3f} "
                 f"{tot_r:+8.2f} ${pnl_s:+9,.0f} {dd_pct:6.2%} {sh:7.2f}", f)
        _out("", f)

        # ==================================================================
        # PER-SYMBOL DETAILED REPORTS
        # ==================================================================
        for sym in sorted(result.symbol_results):
            sr = result.symbol_results[sym]
            if not sr.trades:
                continue

            _out("=" * 80, f)
            _out(f"DETAILED REPORT: {sym}", f)
            _out("=" * 80, f)

            multiplier = 1.0  # ETFs
            pnls = np.array([t.pnl_dollars for t in sr.trades])
            risks = np.array([abs(t.entry_price - t.initial_stop) * multiplier * t.qty
                              for t in sr.trades])
            holds = np.array([t.bars_held for t in sr.trades])
            comms = np.array([t.commission for t in sr.trades])

            metrics = compute_metrics(
                pnls, risks, holds, comms,
                sr.equity_curve, sr.timestamps, config.initial_equity,
            )

            _out(performance_report(sr, metrics), f)
            _out(behavior_report(sr.trades), f)
            _out(diagnostic_report(sr), f)

            # Buy & hold comparison
            if sym in data.daily:
                daily_closes = data.daily[sym].closes
                if len(sr.timestamps) >= 2 and len(daily_closes) >= 2:
                    delta = sr.timestamps[-1] - sr.timestamps[0]
                    if hasattr(delta, 'total_seconds'):
                        years = delta.total_seconds() / (365.25 * 24 * 3600)
                    elif isinstance(delta, np.timedelta64):
                        years = float(delta / np.timedelta64(1, 's')) / (365.25 * 24 * 3600)
                    else:
                        years = float(delta) / (365.25 * 24 * 3600)
                    qty = config.fixed_qty or 1
                    bh = compute_buy_and_hold(
                        sym, daily_closes, years,
                        qty=qty,
                        multiplier=multiplier,
                        initial_equity=config.initial_equity,
                    )
                    _out(buy_and_hold_report(bh, metrics), f)

            # Signal funnel (per-symbol)
            if sr.funnel is not None:
                _out(atrss_signal_funnel(sr.funnel, len(sr.trades)), f)
                _out(atrss_regime_time_report(sr.funnel, sr), f)
                _out(atrss_breakout_arm_diagnostic(sr.funnel), f)

            # Order fill rate (per-symbol)
            if sr.order_metadata:
                _out(atrss_order_fill_rate(sr.order_metadata), f)

            # Bias alignment (per-symbol)
            _out(atrss_bias_alignment(sr.trades, sr), f)
            _out("", f)

        # ==================================================================
        # AGGREGATE DIAGNOSTICS (all trades pooled)
        # ==================================================================
        _out("=" * 80, f)
        _out("AGGREGATE DIAGNOSTICS (all symbols pooled)", f)
        _out("=" * 80, f)
        _out("", f)

        _out(atrss_entry_type_drilldown(all_trades), f)
        _out("", f)
        _out(atrss_exit_analysis(all_trades), f)
        _out("", f)
        _out(atrss_stop_efficiency(all_trades), f)
        _out("", f)
        _out(atrss_time_analysis(all_trades), f)
        _out("", f)
        _out(atrss_r_curve(all_trades), f)
        _out("", f)
        _out(atrss_streak_analysis(all_trades), f)
        _out("", f)
        _out(atrss_addon_analysis(all_trades), f)
        _out("", f)
        _out(atrss_position_occupancy(all_trades), f)
        _out("", f)
        _out(atrss_mfe_cohort_segmentation(all_trades), f)
        _out("", f)

        # Filter rejection detail (from shadow tracker)
        if result.filter_summary:
            for sym in sorted(result.filter_summary):
                fs = result.filter_summary[sym]
                if isinstance(fs, dict):
                    _out(f"\n--- Filter Rejections: {sym} ---", f)
                    _out(atrss_filter_rejection_detail(fs), f)
        _out("", f)

        _out(atrss_losing_trade_detail(all_trades), f)
        _out("", f)

        # ==================================================================
        # ADVANCED DIAGNOSTICS (new functions)
        # ==================================================================
        _out("=" * 80, f)
        _out("ADVANCED DIAGNOSTICS", f)
        _out("=" * 80, f)
        _out("", f)

        _out(atrss_crisis_window_analysis(all_trades), f)
        _out("", f)
        _out(atrss_rolling_edge(all_trades), f)
        _out("", f)
        _out(atrss_profit_concentration(all_trades), f)
        _out("", f)
        _out(atrss_right_then_stopped(all_trades), f)
        _out("", f)
        _out(atrss_monthly_returns(all_trades), f)
        _out("", f)
        _out(atrss_adx_edge_analysis(all_trades), f)
        _out("", f)

        _out("=" * 80, f)
        _out("ATRSS DIAGNOSTICS COMPLETE", f)
        _out("=" * 80, f)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved summary to {summary_path}")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
