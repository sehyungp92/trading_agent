"""Portfolio vs individual strategy comparison.

Runs each strategy individually AND together in portfolio mode with
round 3 optimized configs, then compares trade counts and metrics.
Saves full diagnostics to output/portfolio/.
"""

import json
import logging
import sys
from datetime import date
from pathlib import Path

# Suppress debug logging
logging.basicConfig(level=logging.WARNING)
import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto_trader.backtest.diagnostics import generate_diagnostics
from crypto_trader.backtest.metrics import compute_metrics, metrics_to_dict
from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
from crypto_trader.backtest.runner import run as run_individual
from crypto_trader.portfolio.backtest_runner import run_portfolio_backtest
from crypto_trader.portfolio.config import PortfolioConfig, StrategyAllocation


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/portfolio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Config paths
CONFIG_PATHS = {
    "momentum": Path("output/momentum/round_3/optimized_config.json"),
    "trend": Path("output/trend/round_3/optimized_config.json"),
    "breakout": Path("output/breakout/round_3/optimized_config.json"),
}

# Common backtest params — covers full available data range
BACKTEST_CONFIG = build_backtest_config_from_profile(
    profile=LIVE_PARITY_PROFILE,
    start_date=date(2025, 12, 1),   # warmup will push load earlier
    end_date=date(2026, 4, 18),
    symbols=["BTC", "ETH", "SOL"],
)


def load_strategy_config(strategy_id: str, path: Path):
    """Load an optimized config JSON and return the config object."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Unwrap "strategy" key if present
    if "strategy" in data:
        data = data["strategy"]

    if strategy_id == "momentum":
        from crypto_trader.strategy.momentum.config import MomentumConfig
        return MomentumConfig.from_dict(data)
    elif strategy_id == "trend":
        from crypto_trader.strategy.trend.config import TrendConfig
        return TrendConfig.from_dict(data)
    elif strategy_id == "breakout":
        from crypto_trader.strategy.breakout.config import BreakoutConfig
        return BreakoutConfig.from_dict(data)
    else:
        raise ValueError(f"Unknown strategy: {strategy_id}")


def run_individual_strategies():
    """Run each strategy individually, return trade lists + metrics."""
    results = {}
    for strategy_id, config_path in CONFIG_PATHS.items():
        print(f"\n{'='*60}")
        print(f"Running {strategy_id} individually...")
        print(f"{'='*60}")

        config = load_strategy_config(strategy_id, config_path)
        config.symbols = BACKTEST_CONFIG.symbols

        bt_result = run_individual(
            strategy_config=config,
            backtest_config=BACKTEST_CONFIG,
            data_dir=DATA_DIR,
            strategy_type=strategy_id,
        )

        md = metrics_to_dict(bt_result.metrics)
        results[strategy_id] = {
            "trades": bt_result.trades,
            "metrics": bt_result.metrics,
            "metrics_dict": md,
        }

        wr = md.get('win_rate', 0)
        # win_rate in metrics_to_dict is already a percentage (e.g. 58.33)
        print(f"  Trades: {int(md.get('total_trades', 0))}")
        print(f"  Win rate: {wr:.1f}%")
        print(f"  PF: {md.get('profit_factor', 0):.2f}")
        print(f"  Net return: {md.get('net_return_pct', 0):.2f}%")
        print(f"  Max DD: {md.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Sharpe: {md.get('sharpe_ratio', 0):.2f}")

    return results


def run_portfolio():
    """Run all strategies together in portfolio mode."""
    print(f"\n{'='*60}")
    print("Running PORTFOLIO (all 3 strategies together)...")
    print(f"{'='*60}")

    # Load all configs
    strategy_configs = {}
    for strategy_id, config_path in CONFIG_PATHS.items():
        strategy_configs[strategy_id] = load_strategy_config(strategy_id, config_path)

    # Portfolio config with sensible defaults
    portfolio_config = PortfolioConfig(
        initial_equity=BACKTEST_CONFIG.initial_equity,
        strategies=tuple(
            StrategyAllocation(strategy_id=sid)
            for sid in strategy_configs
        ),
    )

    result = run_portfolio_backtest(
        portfolio_config=portfolio_config,
        strategy_configs=strategy_configs,
        backtest_config=BACKTEST_CONFIG,
        data_dir=DATA_DIR,
    )

    return result


def save_diagnostics(portfolio_result, individual_results):
    """Generate and save full diagnostics."""
    lines = []
    lines.append("=" * 80)
    lines.append("PORTFOLIO vs INDIVIDUAL STRATEGY COMPARISON")
    lines.append("=" * 80)
    lines.append(f"Date range: {BACKTEST_CONFIG.start_date} to {BACKTEST_CONFIG.end_date}")
    lines.append(f"Warmup: {BACKTEST_CONFIG.warmup_days} days")
    lines.append(f"Symbols: {', '.join(BACKTEST_CONFIG.symbols)}")
    lines.append(f"Initial equity: ${BACKTEST_CONFIG.initial_equity:,.0f}")
    lines.append("")

    # Individual strategy summary
    lines.append("-" * 80)
    lines.append("INDIVIDUAL STRATEGY RESULTS (run independently)")
    lines.append("-" * 80)
    total_individual_trades = 0
    for sid, data in individual_results.items():
        md = data["metrics_dict"]
        n = int(md.get("total_trades", 0))
        total_individual_trades += n
        lines.append(f"\n  {sid.upper()}:")
        lines.append(f"    Trades: {n}")
        lines.append(f"    Win Rate: {md.get('win_rate', 0):.1f}%")
        lines.append(f"    Profit Factor: {md.get('profit_factor', 0):.2f}")
        lines.append(f"    Net Return: {md.get('net_return_pct', 0):.2f}%")
        lines.append(f"    Max DD: {md.get('max_drawdown_pct', 0):.2f}%")
        lines.append(f"    Sharpe: {md.get('sharpe_ratio', 0):.2f}")

    lines.append(f"\n  TOTAL INDIVIDUAL TRADES: {total_individual_trades}")

    # Portfolio summary
    lines.append("")
    lines.append("-" * 80)
    lines.append("PORTFOLIO RESULTS (all strategies sharing one account)")
    lines.append("-" * 80)
    pm = portfolio_result.metrics
    pmd = metrics_to_dict(pm)
    total_portfolio_trades = sum(
        len(trades) for trades in portfolio_result.per_strategy_trades.values()
    )

    lines.append(f"\n  Combined:")
    lines.append(f"    Total Trades: {total_portfolio_trades}")
    lines.append(f"    Win Rate: {pmd.get('win_rate', 0):.1f}%")
    lines.append(f"    Profit Factor: {pmd.get('profit_factor', 0):.2f}")
    lines.append(f"    Net Return: {pmd.get('net_return_pct', 0):.2f}%")
    lines.append(f"    Max DD: {pmd.get('max_drawdown_pct', 0):.2f}%")
    lines.append(f"    Sharpe: {pmd.get('sharpe_ratio', 0):.2f}")

    lines.append(f"\n  Per-strategy breakdown:")
    for sid, trades in portfolio_result.per_strategy_trades.items():
        indiv_n = int(individual_results.get(sid, {}).get("metrics_dict", {}).get("total_trades", 0))
        diff = len(trades) - indiv_n
        diff_str = f" ({diff:+d})" if diff != 0 else " (=)"
        lines.append(f"    {sid}: {len(trades)} trades{diff_str}")

    # Rule events analysis
    lines.append("")
    lines.append("-" * 80)
    lines.append("PORTFOLIO RULE EVENTS")
    lines.append("-" * 80)
    n_approved = sum(1 for e in portfolio_result.rule_events if e.approved)
    n_blocked = sum(1 for e in portfolio_result.rule_events if not e.approved)
    lines.append(f"  Total checks: {len(portfolio_result.rule_events)}")
    lines.append(f"  Approved: {n_approved}")
    lines.append(f"  Blocked: {n_blocked}")

    if n_blocked > 0:
        # Group blocked by reason
        reasons = {}
        for e in portfolio_result.rule_events:
            if not e.approved:
                reason = e.denial_reason or "unknown"
                reasons[reason] = reasons.get(reason, 0) + 1

        lines.append(f"\n  Block reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(f"    {reason}: {count}")

        # Blocked by strategy
        blocked_by_strat = {}
        for e in portfolio_result.rule_events:
            if not e.approved:
                blocked_by_strat[e.strategy_id] = blocked_by_strat.get(e.strategy_id, 0) + 1
        lines.append(f"\n  Blocked per strategy:")
        for sid, count in sorted(blocked_by_strat.items()):
            lines.append(f"    {sid}: {count}")

    # Trade count reconciliation
    lines.append("")
    lines.append("-" * 80)
    lines.append("TRADE COUNT RECONCILIATION")
    lines.append("-" * 80)
    lines.append(f"  Individual total: {total_individual_trades}")
    lines.append(f"  Portfolio total:  {total_portfolio_trades}")
    diff = total_portfolio_trades - total_individual_trades
    if diff == 0:
        lines.append(f"  Difference: 0 (PERFECT MATCH)")
    else:
        lines.append(f"  Difference: {diff:+d}")
        lines.append(f"\n  Explanation of differences:")
        if n_blocked > 0:
            lines.append(f"    - {n_blocked} entry attempts blocked by portfolio rules")
        remaining = abs(diff) - n_blocked
        if remaining > 0:
            lines.append(f"    - {remaining} cascading effect(s) from blocked trades")
            lines.append(f"      (blocked entries alter subsequent state, causing signal divergence)")

    # Full diagnostics
    if portfolio_result.all_trades:
        lines.append("")
        lines.append("=" * 80)
        lines.append("FULL PORTFOLIO DIAGNOSTICS")
        lines.append("=" * 80)
        diag = generate_diagnostics(
            portfolio_result.all_trades,
            initial_equity=BACKTEST_CONFIG.initial_equity,
        )
        lines.append(diag)

    report = "\n".join(lines)

    # Save
    output_path = OUTPUT_DIR / "portfolio_diagnostics.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nDiagnostics saved to: {output_path}")

    # Also save per-strategy diagnostics from portfolio run
    for sid, trades in portfolio_result.per_strategy_trades.items():
        if trades:
            diag = generate_diagnostics(trades, initial_equity=BACKTEST_CONFIG.initial_equity)
            strat_path = OUTPUT_DIR / f"{sid}_portfolio_diagnostics.txt"
            with open(strat_path, "w", encoding="utf-8") as f:
                f.write(f"# {sid.upper()} — Portfolio Mode\n")
                f.write(f"# Trades: {len(trades)}\n\n")
                f.write(diag)
            print(f"  {sid} diagnostics: {strat_path}")

    # Save rule events log
    events_path = OUTPUT_DIR / "rule_events.jsonl"
    with open(events_path, "w", encoding="utf-8") as f:
        for e in portfolio_result.rule_events:
            f.write(json.dumps({
                "timestamp": str(e.timestamp),
                "strategy": e.strategy_id,
                "symbol": e.symbol,
                "direction": e.direction,
                "risk_R": e.risk_R,
                "approved": e.approved,
                "denial_reason": e.denial_reason,
                "size_multiplier": e.size_multiplier,
            }) + "\n")
    print(f"  Rule events: {events_path}")

    return report


def main():
    print("Portfolio vs Individual Strategy Comparison")
    print(f"Using round 3 optimized configs")
    print(f"Date range: {BACKTEST_CONFIG.start_date} to {BACKTEST_CONFIG.end_date}")
    print(f"Warmup: {BACKTEST_CONFIG.warmup_days} days\n")

    # Check configs exist
    for sid, path in CONFIG_PATHS.items():
        if not path.exists():
            print(f"ERROR: Config not found: {path}")
            sys.exit(1)

    # Run individual strategies
    individual_results = run_individual_strategies()

    # Run portfolio
    portfolio_result = run_portfolio()

    # Save diagnostics and comparison
    report = save_diagnostics(portfolio_result, individual_results)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_indiv = sum(
        int(r["metrics_dict"].get("total_trades", 0)) for r in individual_results.values()
    )
    total_port = sum(
        len(t) for t in portfolio_result.per_strategy_trades.values()
    )
    print(f"Individual total trades: {total_indiv}")
    print(f"Portfolio total trades:  {total_port}")
    n_blocked = sum(1 for e in portfolio_result.rule_events if not e.approved)
    print(f"Portfolio blocks:        {n_blocked}")
    print(f"Trade count diff:        {total_port - total_indiv:+d}")


if __name__ == "__main__":
    main()
