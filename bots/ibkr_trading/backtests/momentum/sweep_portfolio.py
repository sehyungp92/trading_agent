"""Portfolio configuration sweep.

Runs each strategy engine once, then re-runs the portfolio overlay with
different configs to find optimal parameter settings.  Each variant takes
milliseconds since it just walks pre-computed trade lists.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from backtests.momentum.analysis.metrics import (
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
)
from backtests.momentum.config_portfolio import PortfolioBacktestConfig
from backtests.momentum.engine.portfolio_engine import PortfolioBacktester, PortfolioResult
from libs.oms.config.portfolio_config import PortfolioConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SweepVariant:
    """A single parameter variant to test."""
    name: str
    description: str
    config_factory: Callable[[PortfolioConfig], PortfolioConfig]


@dataclass
class SweepResult:
    """Metrics for one sweep variant."""
    variant_name: str
    n_approved: int = 0
    n_blocked: int = 0
    total_R: float = 0.0
    r_capture_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    cagr: float = 0.0
    max_dd_pct: float = 0.0
    net_pnl: float = 0.0
    # Deltas vs baseline
    delta_R: float = 0.0
    delta_sharpe: float = 0.0
    delta_pnl: float = 0.0
    delta_max_dd: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _replace_strategy(
    config: PortfolioConfig,
    strategy_id: str,
    **kwargs,
) -> PortfolioConfig:
    """Replace fields on a specific strategy within a frozen PortfolioConfig."""
    new_strategies = tuple(
        replace(s, **kwargs) if s.strategy_id == strategy_id else s
        for s in config.strategies
    )
    return replace(config, strategies=new_strategies)


def _extract_metrics(
    result: PortfolioResult,
    iso_total_R: float,
    variant_name: str,
) -> SweepResult:
    """Extract key metrics from a PortfolioResult."""
    trades = result.trades
    n_approved = len(trades)
    n_blocked = len(result.blocked_trades)

    if not trades:
        return SweepResult(variant_name=variant_name)

    pnls = np.array([t.adjusted_pnl for t in trades])
    net_pnl = float(np.sum(pnls))

    r_mults = np.array([t.r_multiple * t.size_multiplier for t in trades])
    total_R = float(np.sum(r_mults))
    r_capture_pct = (total_R / iso_total_R * 100) if iso_total_R > 0 else 0.0

    ec = result.equity_curve
    max_dd_pct, _ = compute_max_drawdown(ec)
    sharpe = compute_sharpe(ec, periods_per_year=252)
    sortino = compute_sortino(ec, periods_per_year=252)

    if len(ec) >= 2 and len(result.equity_timestamps) >= 2:
        span = result.equity_timestamps[-1] - result.equity_timestamps[0]
        years = span.total_seconds() / (365.25 * 24 * 3600)
    else:
        years = 1.0
    final_equity = float(ec[-1]) if len(ec) > 0 else result.initial_equity
    cagr = compute_cagr(result.initial_equity, final_equity, years)

    return SweepResult(
        variant_name=variant_name,
        n_approved=n_approved,
        n_blocked=n_blocked,
        total_R=total_R,
        r_capture_pct=r_capture_pct,
        sharpe=sharpe,
        sortino=sortino,
        cagr=cagr,
        max_dd_pct=max_dd_pct,
        net_pnl=net_pnl,
    )


# ---------------------------------------------------------------------------
# Sweep variants
# ---------------------------------------------------------------------------

def build_sweep_variants() -> list[SweepVariant]:
    """Build the list of parameter variants to test."""
    variants: list[SweepVariant] = []

    # 1a. nqdtc_oppose_size_mult 0.0 -> 0.25
    variants.append(SweepVariant(
        name="oppose_0.25",
        description="Quarter-size Vdubus when NQDTC opposes (was: block)",
        config_factory=lambda c: replace(c, nqdtc_oppose_size_mult=0.25),
    ))

    # 1b. nqdtc_oppose_size_mult 0.0 -> 0.50
    variants.append(SweepVariant(
        name="oppose_0.50",
        description="Half-size Vdubus when NQDTC opposes (was: block)",
        config_factory=lambda c: replace(c, nqdtc_oppose_size_mult=0.50),
    ))

    # 4a. Directional cap 2.5 -> 3.0
    variants.append(SweepVariant(
        name="dir_cap_3.0",
        description="Raise directional cap 2.5R -> 3.0R",
        config_factory=lambda c: replace(c, directional_cap_R=3.0),
    ))

    # 4b. Directional cap 2.5 -> 3.5
    variants.append(SweepVariant(
        name="dir_cap_3.5",
        description="Raise directional cap 2.5R -> 3.5R (match heat cap)",
        config_factory=lambda c: replace(c, directional_cap_R=3.5),
    ))

    # 5. Vdubus daily stop 2.5 -> 3.0
    variants.append(SweepVariant(
        name="vdubus_stop_3.0",
        description="Raise Vdubus daily_stop_R 2.5 -> 3.0",
        config_factory=lambda c: _replace_strategy(c, "Vdubus", daily_stop_R=3.0),
    ))

    # 6. NQDTC daily stop 2.5 -> 3.0
    variants.append(SweepVariant(
        name="nqdtc_stop_3.0",
        description="Raise NQDTC daily_stop_R 2.5 -> 3.0",
        config_factory=lambda c: _replace_strategy(c, "NQDTC", daily_stop_R=3.0),
    ))

    # 7. Relaxed drawdown tiers
    variants.append(SweepVariant(
        name="dd_tiers_relaxed",
        description="Relax DD tiers: (8/12/15)% -> (10/15/20)%",
        config_factory=lambda c: replace(c, dd_tiers=(
            (0.10, 1.00),
            (0.15, 0.50),
            (0.20, 0.25),
            (1.00, 0.00),
        )),
    ))

    return variants


# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------

def run_sweep(
    baseline_config: PortfolioConfig,
    nqdtc_trades: list | None,
    vdubus_trades: list | None,
    variants: list[SweepVariant],
    iso_total_R: float,
    bt_config_template: PortfolioBacktestConfig,
) -> list[SweepResult]:
    """Run baseline + all variants, return results list.

    Args:
        baseline_config: The baseline PortfolioConfig (e.g. 10k_v4).
        nqdtc_trades: Raw trade lists from independent engine runs.
        vdubus_trades: Raw trade lists from independent engine runs.
        variants: List of SweepVariant to test.
        iso_total_R: Sum of isolated R-multiples across all strategies.
        bt_config_template: Template PortfolioBacktestConfig for shared fields.
    """
    results: list[SweepResult] = []

    def _run_one(pc: PortfolioConfig, name: str) -> SweepResult:
        cfg = replace(bt_config_template, portfolio=pc)
        bt = PortfolioBacktester(cfg)
        pr = bt.run(
            nqdtc_trades=nqdtc_trades,
            vdubus_trades=vdubus_trades,
        )
        return _extract_metrics(pr, iso_total_R, name)

    # Baseline
    baseline = _run_one(baseline_config, "BASELINE")
    results.append(baseline)

    # Each variant
    for v in variants:
        modified_config = v.config_factory(baseline_config)
        sr = _run_one(modified_config, v.name)
        # Compute deltas
        sr.delta_R = sr.total_R - baseline.total_R
        sr.delta_sharpe = sr.sharpe - baseline.sharpe
        sr.delta_pnl = sr.net_pnl - baseline.net_pnl
        sr.delta_max_dd = sr.max_dd_pct - baseline.max_dd_pct
        results.append(sr)

    return results


# ---------------------------------------------------------------------------
# Combined winners
# ---------------------------------------------------------------------------

def find_winners(
    results: list[SweepResult],
    min_delta_R: float = 0.0,
    max_sharpe_loss: float = 0.10,
    max_dd_increase: float = 0.03,
) -> list[str]:
    """Find variant names that pass winner criteria.

    Criteria:
    - delta_R > min_delta_R (captures more R)
    - delta_sharpe >= -max_sharpe_loss (doesn't destroy Sharpe)
    - delta_max_dd <= max_dd_increase (doesn't blow up drawdown)
    """
    winners = []
    for sr in results:
        if sr.variant_name == "BASELINE":
            continue
        if (
            sr.delta_R > min_delta_R
            and sr.delta_sharpe >= -max_sharpe_loss
            and sr.delta_max_dd <= max_dd_increase
        ):
            winners.append(sr.variant_name)
    return winners


def build_combined_variant(
    winner_names: list[str],
    variants: list[SweepVariant],
) -> SweepVariant | None:
    """Chain winning variant config_factories into a single combined variant."""
    winner_variants = [v for v in variants if v.name in winner_names]
    if not winner_variants:
        return None

    def combined_factory(c: PortfolioConfig) -> PortfolioConfig:
        for v in winner_variants:
            c = v.config_factory(c)
        return c

    return SweepVariant(
        name="COMBINED",
        description="Combined winners: " + ", ".join(winner_names),
        config_factory=combined_factory,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_sweep_table(results: list[SweepResult]) -> str:
    """Format sweep results as a comparison table."""
    lines = [
        "",
        "=" * 120,
        "PORTFOLIO PARAMETER SWEEP RESULTS",
        "=" * 120,
        "",
    ]

    header = (
        f"{'Variant':<22} {'Appr':>5} {'Blkd':>5} {'TotalR':>8} "
        f"{'R-cap%':>7} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} "
        f"{'MaxDD%':>7} {'PnL':>11}  {'dR':>7} {'dSharpe':>8} {'dPnL':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for sr in results:
        is_baseline = sr.variant_name == "BASELINE"
        dr = "--" if is_baseline else f"{sr.delta_R:+.1f}"
        ds = "--" if is_baseline else f"{sr.delta_sharpe:+.3f}"
        dp = "--" if is_baseline else f"${sr.delta_pnl:+,.0f}"

        lines.append(
            f"{sr.variant_name:<22} {sr.n_approved:>5} {sr.n_blocked:>5} "
            f"{sr.total_R:>+7.1f}R {sr.r_capture_pct:>6.1f}% "
            f"{sr.sharpe:>7.2f} {sr.sortino:>8.2f} {sr.cagr:>6.1%} "
            f"{sr.max_dd_pct:>6.1%} ${sr.net_pnl:>+10,.0f}  "
            f"{dr:>7} {ds:>8} {dp:>10}"
        )

    lines.append("")
    return "\n".join(lines)


def format_winners_summary(
    winner_names: list[str],
    combined_result: SweepResult | None,
) -> str:
    """Format the combined-winners analysis."""
    lines = [
        "",
        "=" * 120,
        "COMBINED WINNERS ANALYSIS",
        "=" * 120,
        "",
    ]

    if not winner_names:
        lines.append("No variants passed winner criteria.")
        return "\n".join(lines)

    lines.append(f"Winners: {', '.join(winner_names)}")
    lines.append("")

    if combined_result:
        lines.append("Combined result:")
        lines.append(
            f"  Approved: {combined_result.n_approved}  "
            f"Blocked: {combined_result.n_blocked}  "
            f"Total R: {combined_result.total_R:+.1f}  "
            f"R-capture: {combined_result.r_capture_pct:.1f}%"
        )
        lines.append(
            f"  Sharpe: {combined_result.sharpe:.2f}  "
            f"Sortino: {combined_result.sortino:.2f}  "
            f"CAGR: {combined_result.cagr:.1%}  "
            f"MaxDD: {combined_result.max_dd_pct:.1%}"
        )
        lines.append(
            f"  PnL: ${combined_result.net_pnl:+,.0f}  "
            f"dR: {combined_result.delta_R:+.1f}  "
            f"dSharpe: {combined_result.delta_sharpe:+.3f}  "
            f"dPnL: ${combined_result.delta_pnl:+,.0f}"
        )

    lines.append("")
    return "\n".join(lines)
