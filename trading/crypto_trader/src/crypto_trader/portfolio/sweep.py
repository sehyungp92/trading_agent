"""Portfolio parameter sweep — fast discovery of optimal portfolio rules.

Follows the reference pattern: run strategies independently to get trade lists,
then re-run portfolio overlay with different configs (milliseconds per variant).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import structlog

from crypto_trader.core.models import Side, Trade
from crypto_trader.portfolio.config import PortfolioConfig
from crypto_trader.portfolio.manager import PortfolioManager
from crypto_trader.portfolio.state import PortfolioState

log = structlog.get_logger()


@dataclass(frozen=True)
class SweepVariant:
    """A portfolio config variant to test."""

    name: str
    description: str
    config_factory: Callable[[PortfolioConfig], PortfolioConfig]


@dataclass
class SweepResult:
    """Result of running one sweep variant."""

    variant_name: str
    n_approved: int
    n_blocked: int
    total_R: float
    max_dd_pct: float
    net_pnl: float
    final_equity: float
    deltas: dict[str, float] = field(default_factory=dict)  # vs baseline


@dataclass
class _TradeEntry:
    """Internal: a trade with strategy attribution for sweep replay."""

    strategy_id: str
    trade: Trade
    risk_R: float  # risk of this trade in R-units


def run_sweep(
    baseline_config: PortfolioConfig,
    strategy_trade_lists: dict[str, list[Trade]],
    variants: list[SweepVariant],
    initial_equity: float | None = None,
) -> list[SweepResult]:
    """Run portfolio sweep across variants.

    Args:
        baseline_config: The baseline portfolio configuration
        strategy_trade_lists: {strategy_id: [Trade]} from independent backtests
        variants: List of config variants to test
        initial_equity: Override initial equity (default: from baseline_config)

    Returns:
        List of SweepResult, one per variant (baseline is variants[0] if included)
    """
    equity = initial_equity or baseline_config.initial_equity

    # Build unified trade timeline
    timeline = _build_timeline(strategy_trade_lists)

    results = []

    # Run baseline first
    baseline_result = _simulate_overlay(baseline_config, timeline, equity)
    baseline_result.variant_name = "__baseline__"
    results.append(baseline_result)

    # Run each variant
    for variant in variants:
        config = variant.config_factory(baseline_config)
        result = _simulate_overlay(config, timeline, equity)
        result.variant_name = variant.name

        # Compute deltas vs baseline
        result.deltas = {
            "n_approved": result.n_approved - baseline_result.n_approved,
            "n_blocked": result.n_blocked - baseline_result.n_blocked,
            "total_R": result.total_R - baseline_result.total_R,
            "max_dd_pct": result.max_dd_pct - baseline_result.max_dd_pct,
            "net_pnl": result.net_pnl - baseline_result.net_pnl,
        }

        results.append(result)

    return results


def find_winners(
    results: list[SweepResult],
    min_delta_R: float = 0.0,
    max_dd_increase: float = 0.03,
) -> list[SweepResult]:
    """Filter sweep results for variants that improve on baseline.

    Args:
        results: Full sweep results (first entry is baseline)
        min_delta_R: Minimum total R improvement to qualify
        max_dd_increase: Maximum allowed increase in max DD
    """
    if not results:
        return []

    baseline = results[0]
    winners = []

    for r in results[1:]:
        delta_R = r.total_R - baseline.total_R
        dd_increase = r.max_dd_pct - baseline.max_dd_pct

        if delta_R >= min_delta_R and dd_increase <= max_dd_increase:
            winners.append(r)

    return sorted(winners, key=lambda r: r.total_R, reverse=True)


def build_combined_variant(
    winners: list[SweepResult],
    variants: list[SweepVariant],
) -> SweepVariant | None:
    """Chain winning variant factories into a combined variant.

    Returns None if no winners.
    """
    if not winners:
        return None

    winner_names = {w.variant_name for w in winners}
    winning_variants = [v for v in variants if v.name in winner_names]

    if not winning_variants:
        return None

    def combined_factory(config: PortfolioConfig) -> PortfolioConfig:
        result = config
        for v in winning_variants:
            result = v.config_factory(result)
        return result

    names = " + ".join(v.name for v in winning_variants)
    return SweepVariant(
        name=f"combined({names})",
        description="Combined winning variants",
        config_factory=combined_factory,
    )


def format_sweep_table(results: list[SweepResult]) -> str:
    """Format sweep results as a readable table."""
    if not results:
        return "No results."

    lines = []
    header = f"{'Variant':<30} {'Approved':>8} {'Blocked':>8} {'TotalR':>8} {'MaxDD%':>8} {'NetPnL':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        delta_str = ""
        if r.deltas:
            dr = r.deltas.get("total_R", 0)
            delta_str = f" ({dr:+.2f}R)" if dr else ""

        lines.append(
            f"{r.variant_name:<30} {r.n_approved:>8} {r.n_blocked:>8} "
            f"{r.total_R:>8.2f} {r.max_dd_pct:>7.2%} {r.net_pnl:>10.2f}{delta_str}"
        )

    return "\n".join(lines)


def _build_timeline(
    strategy_trade_lists: dict[str, list[Trade]],
) -> list[_TradeEntry]:
    """Merge trades from all strategies into chronological order."""
    entries = []
    for strategy_id, trades in strategy_trade_lists.items():
        for trade in trades:
            # Each trade risks 1.0R by definition (R-multiple already normalizes)
            risk_R = 1.0
            entries.append(_TradeEntry(
                strategy_id=strategy_id,
                trade=trade,
                risk_R=risk_R,
            ))

    entries.sort(key=lambda e: e.trade.entry_time)
    return entries


def _simulate_overlay(
    config: PortfolioConfig,
    timeline: list[_TradeEntry],
    initial_equity: float,
) -> SweepResult:
    """Simulate portfolio rules over a trade timeline.

    This is a fast replay — no actual backtest, just rule checks and P&L accounting.
    P&L is applied at trade exit time (not entry) for correct equity/DD tracking.
    """
    state = PortfolioState(
        equity=initial_equity,
        peak_equity=initial_equity,
    )
    manager = PortfolioManager(config=config, state=state)

    n_approved = 0
    n_blocked = 0
    total_R = 0.0
    max_dd_pct = 0.0
    equity = initial_equity

    # Track approved trades pending exit
    open_trades: list[tuple[_TradeEntry, float]] = []  # (entry, size_multiplier)

    for entry in timeline:
        trade = entry.trade

        # Daily reset
        manager.maybe_reset_daily(trade.entry_time.date())

        # Close any open trades that exit before this entry
        equity, max_dd_pct, total_R = _close_before(
            open_trades, trade.entry_time, state, manager, equity, max_dd_pct, total_R,
        )

        # Check portfolio rules
        result = manager.check_entry(
            strategy_id=entry.strategy_id,
            symbol=trade.symbol,
            direction=trade.direction,
            new_risk_R=entry.risk_R,
        )

        if result.approved:
            n_approved += 1
            scaled_risk = entry.risk_R * result.size_multiplier

            manager.register_entry(
                strategy_id=entry.strategy_id,
                symbol=trade.symbol,
                direction=trade.direction,
                risk_R=scaled_risk,
                entry_time=trade.entry_time,
            )
            # Defer P&L to exit time
            open_trades.append((entry, result.size_multiplier))
        else:
            n_blocked += 1

    # Close remaining trades
    equity, max_dd_pct, total_R = _close_before(
        open_trades,
        datetime.max.replace(tzinfo=timezone.utc),
        state, manager, equity, max_dd_pct, total_R,
    )

    return SweepResult(
        variant_name="",
        n_approved=n_approved,
        n_blocked=n_blocked,
        total_R=total_R,
        max_dd_pct=max_dd_pct,
        net_pnl=equity - initial_equity,
        final_equity=equity,
    )


def _close_before(
    open_trades: list[tuple[_TradeEntry, float]],
    before: datetime,
    state: PortfolioState,
    manager: PortfolioManager,
    equity: float,
    max_dd_pct: float,
    total_R: float,
) -> tuple[float, float, float]:
    """Close trades that exit before a given time. Returns updated (equity, max_dd, total_R)."""
    remaining = []
    for entry, multiplier in open_trades:
        trade = entry.trade
        exit_time = trade.exit_time
        if exit_time is None or exit_time < before:
            # Apply P&L at exit time
            r_mult = trade.r_multiple or 0.0
            total_R += r_mult * multiplier
            equity += trade.net_pnl * multiplier
            state.update_equity(equity)

            dd = state.dd_pct()
            if dd > max_dd_pct:
                max_dd_pct = dd

            pnl_R = r_mult * multiplier
            manager.register_exit(
                strategy_id=entry.strategy_id,
                symbol=trade.symbol,
                pnl_R=pnl_R,
            )
        else:
            remaining.append((entry, multiplier))
    open_trades.clear()
    open_trades.extend(remaining)
    return equity, max_dd_pct, total_R
