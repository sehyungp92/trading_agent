"""Reports and exports for backtest results."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import structlog

log = structlog.get_logger()


def generate_report(result: object, output_dir: Path | str = Path("output")) -> Path:
    """Generate a markdown summary report."""
    from crypto_trader.backtest.runner import BacktestResult

    res: BacktestResult = result  # type: ignore
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "backtest_report.md"

    m = res.metrics
    lines = [
        "# Backtest Report",
        "",
        f"**Trades:** {m.total_trades}",
        f"**Net Profit:** ${m.net_profit:,.2f}",
        f"**Net Return:** {m.net_return_pct:.2f}%",
        f"**Win Rate:** {m.win_rate:.1f}%",
        f"**Profit Factor:** {m.profit_factor:.2f}",
        f"**Max Drawdown:** {m.max_drawdown_pct:.2f}%",
        f"**Sharpe Ratio:** {m.sharpe_ratio:.2f}",
        f"**Sortino Ratio:** {m.sortino_ratio:.2f}",
        f"**Calmar Ratio:** {m.calmar_ratio:.2f}",
        "",
        "## R-Multiple Stats",
        f"**Avg Winner R:** {m.avg_winner_r:.2f}",
        f"**Avg Loser R:** {m.avg_loser_r:.2f}",
        f"**Expectancy R:** {m.expectancy_r:.2f}",
        "",
        "## Execution",
        f"**Avg Bars Held:** {m.avg_bars_held:.1f}",
        f"**Avg MAE R:** {m.avg_mae_r:.2f}",
        f"**Avg MFE R:** {m.avg_mfe_r:.2f}",
        f"**Exit Efficiency:** {m.exit_efficiency:.2f}",
        "",
        "## Setup Breakdown",
        f"**A-Grade Win Rate:** {m.a_setup_win_rate:.1f}%",
        f"**B-Grade Win Rate:** {m.b_setup_win_rate:.1f}%",
        f"**Long Win Rate:** {m.long_win_rate:.1f}%",
        f"**Short Win Rate:** {m.short_win_rate:.1f}%",
        f"**Funding Cost:** ${m.funding_cost_total:,.2f}",
        "",
        "## Per-Asset",
    ]
    for sym, data in m.per_asset.items():
        lines.append(f"- **{sym}**: {data['trades']} trades, "
                      f"{data['win_rate']:.1f}% WR, ${data['net_profit']:,.2f}")

    if m.per_session:
        lines.append("")
        lines.append("## Per-Session")
        for session, data in m.per_session.items():
            lines.append(f"- **{session}**: {data['trades']} trades, "
                          f"{data['win_rate']:.1f}% WR, ${data['net_profit']:,.2f}")

    # New diagnostics sections
    lines.append("")
    lines.append("## Trade Quality")
    lines.append(f"**Edge Ratio (MFE/MAE):** {m.edge_ratio:.2f}")
    lines.append(f"**Payoff Ratio:** {m.payoff_ratio:.2f}")
    lines.append(f"**Recovery Factor:** {m.recovery_factor:.2f}")
    lines.append(f"**Profit Concentration (top 20%):** {m.profit_concentration:.1f}%")
    lines.append(f"**Max Consecutive Wins:** {m.max_consecutive_wins}")
    lines.append(f"**Max Consecutive Losses:** {m.max_consecutive_losses}")

    if m.r_distribution:
        lines.append("")
        lines.append("## R-Multiple Distribution")
        for bucket, count in m.r_distribution.items():
            bar = "#" * count
            lines.append(f"- **{bucket}**: {count} {bar}")

    if m.per_confirmation:
        lines.append("")
        lines.append("## Per-Confirmation Type")
        for ctype, data in m.per_confirmation.items():
            lines.append(f"- **{ctype}**: {data['trades']} trades, "
                          f"{data['win_rate']:.1f}% WR, avg R={data['avg_r']:.2f}, "
                          f"${data['net_profit']:,.2f}")

    if m.per_confluence_count:
        lines.append("")
        lines.append("## Per-Confluence Count")
        for count, data in sorted(m.per_confluence_count.items()):
            lines.append(f"- **{count} confluences**: {data['trades']} trades, "
                          f"{data['win_rate']:.1f}% WR, avg R={data['avg_r']:.2f}, "
                          f"${data['net_profit']:,.2f}")

    if m.per_exit_reason:
        lines.append("")
        lines.append("## Per-Exit Reason")
        for reason, data in m.per_exit_reason.items():
            lines.append(f"- **{reason}**: {data['trades']} trades, "
                          f"{data['win_rate']:.1f}% WR, avg R={data['avg_r']:.2f}, "
                          f"${data['net_profit']:,.2f}")

    if m.weekly_returns:
        lines.append("")
        lines.append("## Weekly Returns")
        for w in m.weekly_returns:
            sign = "+" if w["pnl"] >= 0 else ""
            lines.append(f"- **{w['week']}**: {sign}${w['pnl']:,.2f}")

    path.write_text("\n".join(lines))
    log.info("analysis.report_saved", path=str(path))
    return path


def export_equity_curve(result: object, output_dir: Path | str = Path("output")) -> Path:
    """Export equity curve to CSV."""
    from crypto_trader.backtest.runner import BacktestResult

    res: BacktestResult = result  # type: ignore
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "equity_curve.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "equity"])
        for ts, eq in res.equity_curve:
            writer.writerow([ts.isoformat() if isinstance(ts, datetime) else ts, f"{eq:.2f}"])

    log.info("analysis.equity_curve_saved", path=str(path), rows=len(res.equity_curve))
    return path


def export_trade_journal(result: object, output_dir: Path | str = Path("output")) -> Path:
    """Export full journal to CSV."""
    from crypto_trader.backtest.runner import BacktestResult

    res: BacktestResult = result  # type: ignore
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "journal.csv"
    res.journal.export_csv(path)
    log.info("analysis.journal_saved", path=str(path))
    return path


def print_summary(result: object) -> None:
    """Print a console-friendly summary."""
    from crypto_trader.backtest.runner import BacktestResult

    res: BacktestResult = result  # type: ignore
    m = res.metrics

    print("\n" + "=" * 60)
    print("  BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Total Trades:    {m.total_trades}")
    print(f"  Net Profit:      ${m.net_profit:>12,.2f}")
    print(f"  Net Return:      {m.net_return_pct:>11.2f}%")
    print(f"  Win Rate:        {m.win_rate:>11.1f}%")
    print(f"  Profit Factor:   {m.profit_factor:>11.2f}")
    print(f"  Max Drawdown:    {m.max_drawdown_pct:>11.2f}%")
    print(f"  Sharpe Ratio:    {m.sharpe_ratio:>11.2f}")
    print(f"  Expectancy R:    {m.expectancy_r:>11.2f}")
    print(f"  Avg Bars Held:   {m.avg_bars_held:>11.1f}")
    print("-" * 60)
    print(f"  Edge Ratio:      {m.edge_ratio:>11.2f}")
    print(f"  Payoff Ratio:    {m.payoff_ratio:>11.2f}")
    print(f"  Recovery Factor: {m.recovery_factor:>11.2f}")
    print(f"  Max Win Streak:  {m.max_consecutive_wins:>11}")
    print(f"  Max Loss Streak: {m.max_consecutive_losses:>11}")
    print("-" * 60)

    if m.per_asset:
        print("  Per-Asset:")
        for sym, data in m.per_asset.items():
            print(f"    {sym:>5}: {data['trades']:>3} trades, "
                  f"{data['win_rate']:>5.1f}% WR, ${data['net_profit']:>10,.2f}")

    if m.per_session:
        print("-" * 60)
        print("  Per-Session:")
        for session, data in m.per_session.items():
            print(f"    {session:>10}: {data['trades']:>3} trades, "
                  f"{data['win_rate']:>5.1f}% WR, ${data['net_profit']:>10,.2f}")

    if m.per_exit_reason:
        print("-" * 60)
        print("  Per-Exit Reason:")
        for reason, data in m.per_exit_reason.items():
            print(f"    {reason:>20}: {data['trades']:>3} trades, "
                  f"{data['win_rate']:>5.1f}% WR, R={data['avg_r']:>5.2f}")

    if m.per_confirmation:
        print("-" * 60)
        print("  Per-Confirmation:")
        for ctype, data in m.per_confirmation.items():
            print(f"    {ctype:>20}: {data['trades']:>3} trades, "
                  f"{data['win_rate']:>5.1f}% WR, R={data['avg_r']:>5.2f}")

    if m.r_distribution:
        print("-" * 60)
        print("  R-Distribution:")
        for bucket, count in m.r_distribution.items():
            bar = "#" * count
            print(f"    {bucket:>14}: {count:>3} {bar}")

    print("=" * 60 + "\n")
