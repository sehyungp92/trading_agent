"""Markdown report generation from auto-backtesting results."""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime

from backtests.swing.auto.experiments import Experiment
from backtests.swing.auto.results_tracker import ExperimentResult
from backtests.swing.auto.scoring import CompositeScore


def generate_report(
    results: list[ExperimentResult],
    baselines: dict[str, CompositeScore] | None = None,
    experiments: list[Experiment] | None = None,
) -> str:
    """Generate a markdown report from experiment results."""
    lines = [
        "# Swing Auto Backtesting Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Baselines
    if baselines:
        lines.append("## Baselines")
        lines.append("")
        lines.append("| Strategy | Score | Calmar | PF | InvDD | NetPnL |")
        lines.append("|----------|-------|--------|-----|-------|--------|")
        for strategy, score in sorted(baselines.items()):
            if score.rejected:
                lines.append(
                    f"| {strategy.upper()} | REJECTED | — | — | — | — |"
                )
            else:
                lines.append(
                    f"| {strategy.upper()} | {score.total:.4f} | "
                    f"{score.calmar_component:.3f} | {score.pf_component:.3f} | "
                    f"{score.inv_dd_component:.3f} | {score.net_profit_component:.3f} |"
                )
        lines.append("")

    # Summary
    if not results:
        lines.append("No experiments run yet.")
        return "\n".join(lines)

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
    portfolio = [r for r in results if r.type == "PORTFOLIO" and r.status != "CRASH"]
    if portfolio:
        lines.append("## Portfolio Experiments")
        lines.append("")
        lines.append("| Experiment | Delta | Score | Status |")
        lines.append("|-----------|-------|-------|--------|")
        for r in sorted(portfolio, key=lambda r: r.delta_pct, reverse=True):
            lines.append(
                f"| {r.experiment_id} | "
                f"{r.delta_pct:+.2%} | {r.experiment_score:.4f} | {r.status} |"
            )
        lines.append("")

    # Recommended changes
    approved = [r for r in results if r.status == "APPROVE"]
    if approved:
        lines.append("## Recommended Changes (APPROVE)")
        lines.append("")
        for r in sorted(approved, key=lambda r: r.delta_pct, reverse=True):
            lines.append(
                f"- **{r.experiment_id}** ({r.strategy.upper()}): "
                f"{r.description} — delta {r.delta_pct:+.2%}"
            )
        lines.append("")

    # Needs further testing
    test_further = [r for r in results if r.status == "TEST_FURTHER"]
    if test_further:
        lines.append("## Needs Further Testing (TEST_FURTHER)")
        lines.append("")
        for r in sorted(test_further, key=lambda r: r.delta_pct, reverse=True):
            lines.append(
                f"- **{r.experiment_id}** ({r.strategy.upper()}): "
                f"{r.description} — delta {r.delta_pct:+.2%}"
            )
        lines.append("")

    # Recommended configurations
    lines.extend(_recommended_configs(results, baselines, experiments))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recommended Configuration Builder
# ---------------------------------------------------------------------------

def _recommended_configs(
    results: list[ExperimentResult],
    baselines: dict[str, CompositeScore] | None,
    experiments: list[Experiment] | None,
) -> list[str]:
    """Build recommended configuration sections per strategy and portfolio."""
    lines: list[str] = []
    lines.append("## Recommended Configurations")
    lines.append("")

    # Build experiment lookup for mutations
    exp_by_id: dict[str, Experiment] = {}
    if experiments:
        exp_by_id = {e.id: e for e in experiments}

    # Group results by strategy
    by_strategy: dict[str, list[ExperimentResult]] = defaultdict(list)
    for r in results:
        by_strategy[r.strategy].append(r)

    # Individual strategies first, then portfolio
    strategy_order = ["atrss", "helix"]
    for strategy in strategy_order:
        if strategy not in by_strategy:
            continue
        strat_results = by_strategy[strategy]
        baseline = baselines.get(strategy) if baselines else None
        lines.extend(_strategy_recommendation(strategy, strat_results, baseline, exp_by_id))

    # Portfolio
    if "portfolio" in by_strategy:
        lines.extend(_portfolio_recommendation(by_strategy["portfolio"], baselines, exp_by_id))

    return lines


def _strategy_recommendation(
    strategy: str,
    results: list[ExperimentResult],
    baseline: CompositeScore | None,
    exp_by_id: dict[str, Experiment],
) -> list[str]:
    """Build recommendation for a single strategy."""
    lines: list[str] = []
    name = strategy.upper()
    lines.append(f"### {name}")
    lines.append("")

    if baseline:
        if baseline.rejected:
            lines.append(f"**Baseline**: REJECTED ({baseline.reject_reason})")
            lines.append("")
            lines.append("No recommendations — baseline did not pass minimum thresholds.")
            lines.append("")
            return lines
        lines.append(
            f"**Baseline**: score={baseline.total:.4f} "
            f"(calmar={baseline.calmar_component:.3f}, PF={baseline.pf_component:.3f}, "
            f"invDD={baseline.inv_dd_component:.3f}, netPnL={baseline.net_profit_component:.3f})"
        )
        lines.append("")

    ablations = [r for r in results if r.type == "ABLATION"]
    sweeps = [r for r in results if r.type == "PARAM_SWEEP"]
    interactions = [r for r in results if r.type == "INTERACTION"]

    # --- Essential components (ablation delta < -1%) ---
    essential = [r for r in ablations if r.delta_pct < -0.01 and r.status != "UNWIRED"]
    if essential:
        lines.append("**Essential components** (removing these hurts performance):")
        lines.append("")
        for r in sorted(essential, key=lambda r: r.delta_pct):
            exp = exp_by_id.get(r.experiment_id)
            flag_str = ""
            if exp and exp.mutations:
                flag_str = f" — `{'`, `'.join(f'{k}={v}' for k, v in exp.mutations.items())}`"
            lines.append(f"- {r.description}: **{r.delta_pct:+.2%}**{flag_str}")
        lines.append("")

    # --- Potentially harmful (ablation delta > +1%, removing it helped) ---
    harmful = [r for r in ablations if r.delta_pct > 0.01 and r.status not in ("UNWIRED", "CRASH")]
    if harmful:
        lines.append("**Consider removing** (disabling these improved performance):")
        lines.append("")
        for r in sorted(harmful, key=lambda r: r.delta_pct, reverse=True):
            exp = exp_by_id.get(r.experiment_id)
            flag_str = ""
            if exp and exp.mutations:
                flag_str = f" — `{'`, `'.join(f'{k}={v}' for k, v in exp.mutations.items())}`"
            lines.append(f"- {r.description}: **{r.delta_pct:+.2%}**{flag_str}")
        lines.append("")

    # --- Unwired ---
    unwired = [r for r in ablations if r.status == "UNWIRED"]
    if unwired:
        lines.append(f"**Unwired** ({len(unwired)} flags not connected in engine — wire for potential gains):")
        lines.append("")
        for r in unwired:
            lines.append(f"- `{r.experiment_id}`: {r.description}")
        lines.append("")

    # --- Best param sweep values per parameter group ---
    if sweeps:
        lines.append("**Parameter recommendations** (best value per parameter group):")
        lines.append("")

        # Group sweeps by parameter group (strip trailing value from experiment ID)
        param_groups: dict[str, list[ExperimentResult]] = defaultdict(list)
        for r in sweeps:
            # e.g. "ps_helix_chand_1.5" → group "ps_helix_chand"
            group = _param_group_key(r.experiment_id)
            param_groups[group].append(r)

        for group, group_results in sorted(param_groups.items()):
            best = max(group_results, key=lambda r: r.experiment_score)
            baseline_score = best.baseline_score

            # Check if any value beat baseline
            if best.experiment_score > baseline_score * 1.001:
                exp = exp_by_id.get(best.experiment_id)
                mut_str = ""
                if exp and exp.mutations:
                    mut_str = ", ".join(f"`{k}={v}`" for k, v in exp.mutations.items())
                lines.append(
                    f"- **{best.description}**: score={best.experiment_score:.4f} "
                    f"({best.delta_pct:+.2%}) {mut_str} {'[ROBUST]' if best.robust else ''}"
                )
            else:
                # All values were equal or worse — keep default
                descs = [r.description for r in group_results]
                common = _common_prefix(descs)
                lines.append(f"- {common.strip()}: **keep default** (no improvement found)")

        lines.append("")

    # --- Approved / test_further summary ---
    approved = [r for r in results if r.status == "APPROVE"]
    test_further = [r for r in results if r.status == "TEST_FURTHER"]
    if approved or test_further:
        lines.append("**Action items**:")
        lines.append("")
        for r in approved:
            exp = exp_by_id.get(r.experiment_id)
            mut_str = ""
            if exp and exp.mutations:
                mut_str = " → " + ", ".join(f"`{k}={v}`" for k, v in exp.mutations.items())
            lines.append(f"- APPLY: {r.description} ({r.delta_pct:+.2%}){mut_str}")
        for r in test_further:
            exp = exp_by_id.get(r.experiment_id)
            mut_str = ""
            if exp and exp.mutations:
                mut_str = " → " + ", ".join(f"`{k}={v}`" for k, v in exp.mutations.items())
            lines.append(f"- TEST MORE: {r.description} ({r.delta_pct:+.2%}){mut_str}")
        lines.append("")

    # If nothing actionable, note that
    if not essential and not harmful and not approved and not test_further:
        lines.append("**Verdict**: Current configuration is optimal — no changes recommended.")
        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def _portfolio_recommendation(
    results: list[ExperimentResult],
    baselines: dict[str, CompositeScore] | None,
    exp_by_id: dict[str, Experiment],
) -> list[str]:
    """Build recommendation for portfolio-level experiments."""
    lines: list[str] = []
    lines.append("### Portfolio")
    lines.append("")

    # Show portfolio baseline if available
    baseline_score = results[0].baseline_score if results else 0.0
    if baseline_score == 0.0:
        lines.append("**Baseline**: score=0.0000 (rejected or insufficient trades)")
        lines.append("")
        lines.append(
            "Portfolio baseline was rejected. All portfolio experiments scored relative "
            "to a zero baseline, making delta comparisons meaningless. "
            "Investigate portfolio engine configuration (trade count, drawdown thresholds)."
        )
        lines.append("")
        return lines

    lines.append(f"**Baseline**: score={baseline_score:.4f}")
    lines.append("")

    # Group by parameter
    param_groups: dict[str, list[ExperimentResult]] = defaultdict(list)
    for r in results:
        group = _param_group_key(r.experiment_id)
        param_groups[group].append(r)

    lines.append("**Portfolio parameter recommendations**:")
    lines.append("")
    for group, group_results in sorted(param_groups.items()):
        best = max(group_results, key=lambda r: r.experiment_score)
        if best.experiment_score > baseline_score * 1.001:
            exp = exp_by_id.get(best.experiment_id)
            mut_str = ""
            if exp and exp.mutations:
                mut_str = ", ".join(f"`{k}={v}`" for k, v in exp.mutations.items())
            status = "ROBUST" if best.robust else best.status
            lines.append(
                f"- **{best.description}**: score={best.experiment_score:.4f} "
                f"({best.delta_pct:+.2%}) {mut_str} [{status}]"
            )
        else:
            descs = [r.description for r in group_results]
            common = _common_prefix(descs)
            lines.append(f"- {common.strip()}: **keep default** (no improvement)")
    lines.append("")

    approved = [r for r in results if r.status in ("APPROVE", "TEST_FURTHER")]
    if approved:
        lines.append("**Action items**:")
        lines.append("")
        for r in sorted(approved, key=lambda r: r.delta_pct, reverse=True):
            exp = exp_by_id.get(r.experiment_id)
            mut_str = ""
            if exp and exp.mutations:
                mut_str = " → " + ", ".join(f"`{k}={v}`" for k, v in exp.mutations.items())
            lines.append(f"- {r.status}: {r.description} ({r.delta_pct:+.2%}){mut_str}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _param_group_key(experiment_id: str) -> str:
    """Extract parameter group from experiment ID by stripping trailing numeric value.

    Examples:
        ps_helix_chand_1.5 → ps_helix_chand
        ps_brk_score_3 → ps_brk_score
        pf_atrss_risk_2.0 → pf_atrss_risk
        ps_helix_stale_1h_8 → ps_helix_stale_1h
        ps_brk_chop_stale_-1 → ps_brk_chop_stale
    """
    # Strip trailing _<number> (including negative and decimal)
    return re.sub(r'_-?[\d.]+$', '', experiment_id)


def _common_prefix(strings: list[str]) -> str:
    """Find the longest common prefix of description strings."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix) and prefix:
            # Trim to last space
            idx = prefix.rfind(" ")
            if idx > 0:
                prefix = prefix[:idx]
            else:
                prefix = ""
    return prefix if prefix else strings[0]
