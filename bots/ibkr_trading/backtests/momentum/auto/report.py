"""Report generation for momentum auto-backtesting results."""
from __future__ import annotations

import logging
import re
from datetime import datetime

from backtests.momentum.auto.scoring import CompositeScore

logger = logging.getLogger(__name__)


def generate_report(
    results: list[object],
    baselines: dict[str, CompositeScore],
    experiments: list[object],
) -> str:
    """Generate a comprehensive markdown report from experiment results.

    Args:
        results: List of ExperimentResult from results_tracker.
        baselines: Dict of strategy -> CompositeScore baselines.
        experiments: Full experiment list for cross-referencing.

    Returns:
        Markdown-formatted report string.
    """
    lines: list[str] = []
    lines.append("# Momentum Auto-Research Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"\nTotal experiments: {len(results)}")
    lines.append("")

    # Section 1: Baselines
    lines.append("## Baselines\n")
    lines.append("| Strategy | Score | Components (Calmar / PF / InvDD / NetProfit) |")
    lines.append("|----------|-------|----------------------------------------------|")
    for strategy in sorted(baselines.keys()):
        s = baselines[strategy]
        if s.rejected:
            lines.append(
                f"| {strategy} | **REJECTED** ({s.reject_reason}) | "
                f"- / - / - / - |"
            )
        else:
            lines.append(
                f"| {strategy} | {s.total:.4f} | "
                f"{s.calmar_component:.3f} / {s.pf_component:.3f} / "
                f"{s.inv_dd_component:.3f} / {s.net_profit_component:.3f} |"
            )
    lines.append("")

    # Section 2: Summary by status
    lines.append("## Status Summary\n")
    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    for status in ["APPROVE", "TEST_FURTHER", "DISCARD", "UNWIRED", "CRASH"]:
        count = status_counts.get(status, 0)
        if count > 0:
            lines.append(f"| {status} | {count} |")
    lines.append("")

    # Section 3: Approved experiments (best finds)
    approved = [r for r in results if r.status == "APPROVE"]
    if approved:
        lines.append("## Approved Experiments\n")
        lines.append("| Experiment | Strategy | Delta | Score | Description |")
        lines.append("|------------|----------|-------|-------|-------------|")
        for r in sorted(approved, key=lambda x: -x.delta_pct):
            lines.append(
                f"| {r.experiment_id} | {r.strategy} | "
                f"{r.delta_pct:+.2%} | {r.experiment_score:.4f} | "
                f"{r.description} |"
            )
        lines.append("")

    # Section 4: Test Further
    test_further = [r for r in results if r.status == "TEST_FURTHER"]
    if test_further:
        lines.append("## Test Further\n")
        lines.append("| Experiment | Strategy | Delta | Score | Description |")
        lines.append("|------------|----------|-------|-------|-------------|")
        for r in sorted(test_further, key=lambda x: -x.delta_pct)[:30]:
            lines.append(
                f"| {r.experiment_id} | {r.strategy} | "
                f"{r.delta_pct:+.2%} | {r.experiment_score:.4f} | "
                f"{r.description} |"
            )
        if len(test_further) > 30:
            lines.append(f"\n... and {len(test_further) - 30} more")
        lines.append("")

    # Section 5: Ablation ranking (most impactful flags)
    ablations = [r for r in results if r.type == "ABLATION"]
    if ablations:
        lines.append("## Ablation Impact Ranking\n")
        lines.append("Negative delta = flag is ESSENTIAL (disabling hurts performance).")
        lines.append("Positive delta = flag is HARMFUL (disabling improves performance).\n")

        for strategy in sorted(set(r.strategy for r in ablations)):
            strat_abl = [r for r in ablations if r.strategy == strategy]
            strat_abl.sort(key=lambda x: x.delta_pct)

            lines.append(f"### {strategy}\n")
            lines.append("| Flag | Delta | Status | Verdict |")
            lines.append("|------|-------|--------|---------|")
            for r in strat_abl:
                flag_name = r.experiment_id.replace(f"abl_{strategy}_", "")
                if r.status == "UNWIRED":
                    verdict = "NOT WIRED"
                elif r.delta_pct < -0.05:
                    verdict = "ESSENTIAL"
                elif r.delta_pct < -0.02:
                    verdict = "VALUABLE"
                elif r.delta_pct > 0.05:
                    verdict = "HARMFUL"
                elif r.delta_pct > 0.02:
                    verdict = "REVIEW"
                else:
                    verdict = "NEUTRAL"
                lines.append(f"| {flag_name} | {r.delta_pct:+.2%} | {r.status} | {verdict} |")
            lines.append("")

    # Section 6: Unwired flags inventory
    unwired = [r for r in results if r.status == "UNWIRED"]
    if unwired:
        lines.append("## Unwired Flags (not checked in engine)\n")
        for r in sorted(unwired, key=lambda x: (x.strategy, x.experiment_id)):
            flag_name = r.experiment_id.replace(f"abl_{r.strategy}_", "")
            lines.append(f"- **{r.strategy}**: `{flag_name}`")
        lines.append("")

    # Section 7: Parameter sweep winners
    sweeps = [r for r in results if r.type == "PARAM_SWEEP" and r.delta_pct > 0.01]
    if sweeps:
        lines.append("## Top Parameter Sweeps\n")
        lines.append("| Experiment | Strategy | Delta | Score |")
        lines.append("|------------|----------|-------|-------|")
        for r in sorted(sweeps, key=lambda x: -x.delta_pct)[:25]:
            lines.append(
                f"| {r.experiment_id} | {r.strategy} | "
                f"{r.delta_pct:+.2%} | {r.experiment_score:.4f} |"
            )
        lines.append("")

    # Section 8: Interaction results
    interactions = [r for r in results if r.type == "INTERACTION" and r.delta_pct > 0.01]
    if interactions:
        lines.append("## Top Interactions\n")
        lines.append("| Experiment | Strategy | Delta | Score |")
        lines.append("|------------|----------|-------|-------|")
        for r in sorted(interactions, key=lambda x: -x.delta_pct)[:15]:
            lines.append(
                f"| {r.experiment_id} | {r.strategy} | "
                f"{r.delta_pct:+.2%} | {r.experiment_score:.4f} |"
            )
        lines.append("")

    # Section 9: Portfolio experiments
    portfolio_results = [r for r in results if r.strategy == "portfolio"]
    if portfolio_results:
        lines.append("## Portfolio Experiments\n")
        lines.append("| Experiment | Delta | Score | Status |")
        lines.append("|------------|-------|-------|--------|")
        for r in sorted(portfolio_results, key=lambda x: -x.delta_pct):
            lines.append(
                f"| {r.experiment_id} | {r.delta_pct:+.2%} | "
                f"{r.experiment_score:.4f} | {r.status} |"
            )
        lines.append("")

    # Section 10: Recommended config changes
    lines.append("## Recommended Config Changes\n")
    lines.extend(_recommended_configs(results, experiments))
    lines.append("")

    # Section 10b: Portfolio recommendations
    if portfolio_results:
        lines.append("### portfolio\n")
        positive_port = [r for r in portfolio_results if r.delta_pct > 0.02]
        if positive_port:
            lines.append("**Positive portfolio experiments**:")
            groups: dict[str, list] = {}
            for r in sorted(positive_port, key=lambda x: -x.delta_pct):
                group_key = _param_group_key(r.experiment_id)
                groups.setdefault(group_key, []).append(r)
            for group, group_results in sorted(groups.items()):
                best = max(group_results, key=lambda x: x.delta_pct)
                exp = {e.id: e for e in experiments}.get(best.experiment_id)
                mut_str = ""
                if exp and exp.mutations:
                    mut_str = " — " + ", ".join(f"`{k}={v}`" for k, v in exp.mutations.items())
                lines.append(f"  - `{best.experiment_id}` (delta={best.delta_pct:+.2%}){mut_str}")
        else:
            lines.append("No portfolio experiments beat baseline by >2%.")
        lines.append("")

    # Section 11: Crashes
    crashes = [r for r in results if r.status == "CRASH"]
    if crashes:
        lines.append("## Crashes\n")
        for r in crashes:
            lines.append(f"- `{r.experiment_id}` ({r.strategy}): {r.description}")
        lines.append("")

    return "\n".join(lines)


def _recommended_configs(
    results: list,
    experiments: list,
) -> list[str]:
    """Generate per-strategy config recommendations."""
    lines: list[str] = []
    experiments_map = {e.id: e for e in experiments}

    strategies = sorted(set(r.strategy for r in results if r.strategy != "portfolio"))

    for strategy in strategies:
        lines.append(f"### {strategy}\n")

        strat_results = [r for r in results if r.strategy == strategy]

        # Essential flags (ablation caused big negative delta)
        essential = [
            r for r in strat_results
            if r.type == "ABLATION" and r.delta_pct < -0.02 and r.status != "UNWIRED"
        ]
        if essential:
            lines.append("**Essential flags** (DO NOT disable):")
            for r in sorted(essential, key=lambda x: x.delta_pct):
                flag = r.experiment_id.replace(f"abl_{strategy}_", "")
                lines.append(f"  - `{flag}` (delta={r.delta_pct:+.2%})")

        # Harmful flags (ablation caused positive delta)
        harmful = [
            r for r in strat_results
            if r.type == "ABLATION" and r.delta_pct > 0.02 and r.status != "UNWIRED"
        ]
        if harmful:
            lines.append("**Harmful flags** (consider disabling):")
            for r in sorted(harmful, key=lambda x: -x.delta_pct):
                flag = r.experiment_id.replace(f"abl_{strategy}_", "")
                lines.append(f"  - `{flag}` (delta={r.delta_pct:+.2%})")

        # Best sweep values
        best_sweeps = [
            r for r in strat_results
            if r.type == "PARAM_SWEEP" and r.delta_pct > 0.02
        ]
        if best_sweeps:
            lines.append("**Best parameter values**:")
            # Group by param name
            groups: dict[str, list] = {}
            for r in best_sweeps:
                group_key = _param_group_key(r.experiment_id)
                groups.setdefault(group_key, []).append(r)
            for group, group_results in sorted(groups.items()):
                best = max(group_results, key=lambda x: x.delta_pct)
                exp = experiments_map.get(best.experiment_id)
                mut_str = ""
                if exp and exp.mutations:
                    mut_str = " — " + ", ".join(f"`{k}={v}`" for k, v in exp.mutations.items())
                lines.append(f"  - `{best.experiment_id}` (delta={best.delta_pct:+.2%}){mut_str}")

        # Best interactions
        best_ints = [
            r for r in strat_results
            if r.type == "INTERACTION" and r.delta_pct > 0.02
        ]
        if best_ints:
            lines.append("**Best interactions**:")
            for r in sorted(best_ints, key=lambda x: -x.delta_pct)[:5]:
                lines.append(f"  - `{r.experiment_id}` (delta={r.delta_pct:+.2%})")

        lines.append("")

    return lines


def _param_group_key(experiment_id: str) -> str:
    """Group sweep experiment IDs by parameter name.

    e.g. "sweep_nqdtc_SCORE_MIN_4" -> "sweep_nqdtc_SCORE_MIN"
    Handles negative values, decimals, and trailing text.
    """
    return re.sub(r"_-?[\d.]+$", "", experiment_id)
