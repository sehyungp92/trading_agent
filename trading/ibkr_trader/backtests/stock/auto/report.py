"""Markdown report generation from auto-backtesting results."""
from __future__ import annotations

from datetime import datetime

from backtests.stock.auto.results_tracker import ExperimentResult
from backtests.stock.auto.scoring import CompositeScore


def generate_report(
    results: list[ExperimentResult],
    baselines: dict[tuple[str, int], CompositeScore] | None = None,
) -> str:
    """Generate a markdown report from experiment results."""
    lines = [
        "# Auto Backtesting Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Baselines
    if baselines:
        lines.append("## Baselines")
        lines.append("")
        lines.append("| Strategy | Tier | Score | Calmar | PF | InvDD | NetPnL |")
        lines.append("|----------|------|-------|--------|-----|-------|--------|")
        for (strategy, tier), score in sorted(baselines.items()):
            lines.append(
                f"| {strategy.upper()} | {tier} | {score.total:.4f} | "
                f"{score.calmar_component:.3f} | {score.pf_component:.3f} | "
                f"{score.inv_dd_component:.3f} | {score.net_profit_component:.3f} |"
            )
        lines.append("")

    # Summary counts
    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    lines.append("## Summary")
    lines.append("")
    lines.append(f"Total experiments: {len(results)}")
    for status in ["APPROVE", "TEST_FURTHER", "DISCARD", "UNWIRED", "CRASH"]:
        if status in status_counts:
            lines.append(f"- **{status}**: {status_counts[status]}")
    lines.append("")

    # Ablation contribution ranking
    ablations = [r for r in results if r.type == "ABLATION" and r.status != "CRASH"]
    if ablations:
        lines.append("## Ablation Contribution Ranking")
        lines.append("")
        lines.append("Components sorted by impact (|delta|). Removing a valuable component")
        lines.append("should _decrease_ score (negative delta).")
        lines.append("")
        lines.append("| Rank | Experiment | Strategy | Delta | Status |")
        lines.append("|------|-----------|----------|-------|--------|")

        # Sort by absolute delta descending
        ranked = sorted(ablations, key=lambda r: abs(r.delta_pct), reverse=True)
        for i, r in enumerate(ranked, 1):
            if r.status == "UNWIRED":
                lines.append(
                    f"| {i} | {r.experiment_id} | {r.strategy.upper()} | "
                    f"0.00% | UNWIRED |"
                )
            else:
                lines.append(
                    f"| {i} | {r.experiment_id} | {r.strategy.upper()} | "
                    f"{r.delta_pct:+.2%} | {r.status} |"
                )
        lines.append("")

    # Unwired flag inventory
    unwired = [r for r in results if r.status == "UNWIRED"]
    if unwired:
        lines.append("## Unwired Flag Inventory")
        lines.append("")
        lines.append("These ablation flags had delta=0, meaning they are not checked")
        lines.append("in the engine code. Wiring them could unlock additional performance.")
        lines.append("")
        for r in unwired:
            lines.append(f"- `{r.experiment_id}` ({r.strategy.upper()}): {r.description}")
        lines.append("")

    # Parameter sweep results
    sweeps = [r for r in results if r.type == "PARAM_SWEEP" and r.status != "CRASH"]
    if sweeps:
        lines.append("## Parameter Sweep Results")
        lines.append("")
        lines.append("| Experiment | Strategy | Delta | Score | Status |")
        lines.append("|-----------|----------|-------|-------|--------|")
        for r in sorted(sweeps, key=lambda r: r.delta_pct, reverse=True)[:20]:
            lines.append(
                f"| {r.experiment_id} | {r.strategy.upper()} | "
                f"{r.delta_pct:+.2%} | {r.experiment_score:.4f} | {r.status} |"
            )
        lines.append("")

    # Interaction results
    interactions = [r for r in results if r.type == "INTERACTION" and r.status != "CRASH"]
    if interactions:
        lines.append("## Interaction Effects")
        lines.append("")
        lines.append("| Experiment | Strategy | Delta | Status |")
        lines.append("|-----------|----------|-------|--------|")
        for r in sorted(interactions, key=lambda r: r.delta_pct, reverse=True):
            lines.append(
                f"| {r.experiment_id} | {r.strategy.upper()} | "
                f"{r.delta_pct:+.2%} | {r.status} |"
            )
        lines.append("")

    # Portfolio results
    portfolio = [r for r in results if r.type == "PORTFOLIO"]
    if portfolio:
        lines.append("## Portfolio Integration")
        lines.append("")
        lines.append("| Experiment | Delta | Score | Status |")
        lines.append("|-----------|-------|-------|--------|")
        for r in sorted(portfolio, key=lambda r: r.delta_pct, reverse=True):
            lines.append(
                f"| {r.experiment_id} | {r.delta_pct:+.2%} | "
                f"{r.experiment_score:.4f} | {r.status} |"
            )
        lines.append("")

    # Recommended changes
    approved = [r for r in results if r.status == "APPROVE"]
    if approved:
        lines.append("## Recommended Changes")
        lines.append("")
        lines.append("These experiments passed all robustness checks and showed >= 5% improvement:")
        lines.append("")
        for r in sorted(approved, key=lambda r: r.delta_pct, reverse=True):
            lines.append(f"1. **{r.experiment_id}** ({r.strategy.upper()}): "
                         f"{r.description} → {r.delta_pct:+.2%}")
        lines.append("")

    # Test further
    test_further = [r for r in results if r.status == "TEST_FURTHER"]
    if test_further:
        lines.append("## Needs Further Testing")
        lines.append("")
        for r in sorted(test_further, key=lambda r: r.delta_pct, reverse=True):
            lines.append(f"- **{r.experiment_id}** ({r.strategy.upper()}): "
                         f"{r.description} → {r.delta_pct:+.2%}")
        lines.append("")

    return "\n".join(lines)
