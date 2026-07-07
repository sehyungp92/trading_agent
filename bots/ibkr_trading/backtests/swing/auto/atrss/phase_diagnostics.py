"""ATRSS per-phase diagnostic output."""
from __future__ import annotations

from typing import Any

from .scoring import ATRSSMetrics


def format_phase_diagnostics(
    phase: int,
    metrics: ATRSSMetrics,
    baseline_metrics: ATRSSMetrics | None = None,
) -> list[str]:
    """Format diagnostic output for a completed phase."""
    lines = [f"--- ATRSS Phase {phase} Diagnostics ---"]
    lines.append("")

    # Core metrics
    lines.append(f"  Trades:        {metrics.total_trades}")
    lines.append(f"  Win Rate:      {metrics.win_rate:.1%}")
    lines.append(f"  Profit Factor: {metrics.profit_factor:.2f}")
    lines.append(f"  Total R:       {metrics.total_r:+.1f}")
    lines.append(f"  Max DD (pct):  {metrics.max_dd_pct:.2%}")
    lines.append(f"  Max DD (R):    {metrics.max_dd_r:.2f}")
    lines.append(f"  Calmar (R):    {metrics.calmar_r:.1f}")
    lines.append(f"  Sharpe:        {metrics.sharpe:.2f}")
    lines.append(f"  MFE Capture:   {metrics.mfe_capture:.3f}")
    lines.append(f"  Trades/Month:  {metrics.trades_per_month:.1f}")
    lines.append(f"  Avg R:         {metrics.avg_r:+.3f}")

    # Comparison with baseline
    if baseline_metrics:
        lines.append("")
        lines.append("  vs Baseline:")
        _compare(lines, "Trades", baseline_metrics.total_trades, metrics.total_trades)
        _compare(lines, "PF", baseline_metrics.profit_factor, metrics.profit_factor, fmt=".2f")
        _compare(lines, "Total R", baseline_metrics.total_r, metrics.total_r, fmt=".1f")
        _compare(lines, "WR", baseline_metrics.win_rate, metrics.win_rate, fmt=".1%", pct=True)
        _compare(lines, "DD pct", baseline_metrics.max_dd_pct, metrics.max_dd_pct, fmt=".2%", lower_better=True, pct=True)
        _compare(lines, "MFE Cap", baseline_metrics.mfe_capture, metrics.mfe_capture, fmt=".3f")
        _compare(lines, "TPM", baseline_metrics.trades_per_month, metrics.trades_per_month, fmt=".1f")

    # Phase-specific diagnostics
    if phase == 1:
        lines.extend(_phase_1_diagnostics(metrics))
    elif phase == 2:
        lines.extend(_phase_2_diagnostics(metrics))
    elif phase == 3:
        lines.extend(_phase_3_diagnostics(metrics))
    elif phase == 4:
        lines.extend(_phase_4_diagnostics(metrics))

    return lines


def _compare(
    lines: list[str],
    label: str,
    old: float,
    new: float,
    fmt: str = "",
    lower_better: bool = False,
    pct: bool = False,
) -> None:
    if old == 0:
        return
    if pct:
        delta = new - old
        direction = "v" if (delta < 0) != lower_better else "^"
    else:
        delta_pct = (new - old) / abs(old) * 100 if old != 0 else 0
        direction = "v" if (delta_pct < 0) != lower_better else "^"

    old_str = f"{old:{fmt}}" if fmt else str(old)
    new_str = f"{new:{fmt}}" if fmt else str(new)
    lines.append(f"    {label:12s}: {old_str} -> {new_str} {direction}")


def _phase_1_diagnostics(metrics: ATRSSMetrics) -> list[str]:
    lines = ["", "  Phase 1 Focus (Exit Cleanup):"]
    if metrics.mfe_capture > 0.40:
        lines.append("    [OK] MFE capture above 40%")
    else:
        lines.append(f"    [!] MFE capture low: {metrics.mfe_capture:.3f}")
    return lines


def _phase_2_diagnostics(metrics: ATRSSMetrics) -> list[str]:
    lines = ["", "  Phase 2 Focus (Signal & Filtering):"]
    if metrics.trades_per_month > 4.0:
        lines.append(f"    [OK] Frequency improved: {metrics.trades_per_month:.1f} TPM")
    else:
        lines.append(f"    [!] Frequency still low: {metrics.trades_per_month:.1f} TPM")
    return lines


def _phase_3_diagnostics(metrics: ATRSSMetrics) -> list[str]:
    lines = ["", "  Phase 3 Focus (Entry & Fill):"]
    if metrics.total_trades > 200:
        lines.append(f"    [OK] Trade count: {metrics.total_trades}")
    else:
        lines.append(f"    [!] Trade count still low: {metrics.total_trades}")
    return lines


def _phase_4_diagnostics(metrics: ATRSSMetrics) -> list[str]:
    lines = ["", "  Phase 4 Focus (Sizing & Fine-tune):"]
    if metrics.calmar_r > 40:
        lines.append(f"    [OK] Calmar R: {metrics.calmar_r:.1f}")
    else:
        lines.append(f"    [!] Calmar R: {metrics.calmar_r:.1f} (target: 40+)")
    return lines


def generate_phase_diagnostics(
    phase: int,
    metrics: ATRSSMetrics,
    greedy_result: dict[str, Any] | None = None,
    force_all_phases: bool = False,
) -> str:
    """Generate full diagnostic text for a phase (returned as str for PhaseRunner).

    Args:
        phase: Phase number (1-4).
        metrics: Current ATRSSMetrics.
        greedy_result: Dict from greedy_result_to_dict (optional).
        force_all_phases: If True, include diagnostics for all phases (enhanced mode).
    """
    lines = format_phase_diagnostics(phase, metrics)

    if greedy_result:
        lines.append("")
        lines.append("  Greedy result:")
        lines.append(f"    Base score:  {greedy_result.get('base_score', 0.0):.4f}")
        lines.append(f"    Final score: {greedy_result.get('final_score', 0.0):.4f}")
        kept = greedy_result.get("kept_features", [])
        lines.append(f"    Accepted: {len(kept)} experiments")
        for feat in kept:
            if isinstance(feat, dict):
                lines.append(f"      + {feat.get('name', '?')} ({feat.get('delta', 0.0):+.4f})")
            else:
                lines.append(f"      + {feat}")

    if force_all_phases:
        lines.append("")
        lines.append("=== All-phase summary ===")
        for p in range(1, 5):
            focus_label = {
                1: "Exit Cleanup", 2: "Signal & Filtering",
                3: "Entry & Fill", 4: "Sizing & Fine-tune",
            }.get(p, "?")
            lines.append(f"  Phase {p} ({focus_label}):")
            lines.extend(_phase_specific_lines(p, metrics))

    return "\n".join(lines)


def _phase_specific_lines(phase: int, metrics: ATRSSMetrics) -> list[str]:
    """Compact phase-specific summary lines."""
    if phase == 1:
        return [f"    MFE capture: {metrics.mfe_capture:.3f}, PF: {metrics.profit_factor:.2f}"]
    elif phase == 2:
        return [f"    TPM: {metrics.trades_per_month:.1f}, WR: {metrics.win_rate:.1%}, trades: {metrics.total_trades}"]
    elif phase == 3:
        return [f"    Trades: {metrics.total_trades}, TPM: {metrics.trades_per_month:.1f}"]
    elif phase == 4:
        return [f"    Calmar R: {metrics.calmar_r:.1f}, Total R: {metrics.total_r:+.1f}"]
    return []
