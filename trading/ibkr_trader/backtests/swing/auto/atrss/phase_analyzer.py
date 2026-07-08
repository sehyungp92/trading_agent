"""ATRSS phase analysis -- post-phase recommendations."""
from __future__ import annotations

from typing import Any

from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.types import Experiment, GateResult, GreedyResult, PhaseAnalysis


def get_diagnostic_gaps(phase: int, metrics: dict[str, float]) -> list[str]:
    """Identify diagnostic gaps based on phase metrics."""
    gaps = []

    if phase == 1:
        if metrics.get("mfe_capture", 0) < 0.35:
            gaps.append("Exit efficiency still below 35% -- consider profit floor variants")
        if metrics.get("total_r", 0) < 130:
            gaps.append("Total R below 130 -- TP levels may need wider sweep")
    elif phase == 2:
        if metrics.get("trades_per_month", 0) < 4.0:
            gaps.append("Frequency still below 4 TPM -- regime thresholds may need wider sweep")
        if metrics.get("win_rate", 0) < 0.65:
            gaps.append("Win rate below 65% -- quality gate or confirmation tuning needed")
    elif phase == 3:
        if metrics.get("total_trades", 0) < 200:
            gaps.append("Trade count below 200 -- fill rate improvements insufficient")
    elif phase == 4:
        if metrics.get("calmar_r", 0) < 35:
            gaps.append("Calmar R below 35 -- sizing adjustments underperforming")

    return gaps


def suggest_experiments(
    phase: int,
    metrics: dict[str, float],
    gaps: list[str],
    state: PhaseState,
) -> list[Experiment]:
    """Suggest additional experiments based on diagnostic gaps."""
    suggestions: list[Experiment] = []

    if phase == 1:
        # If MFE capture is still low, try intermediate TP values
        if metrics.get("mfe_capture", 0) < 0.35:
            suggestions.append(Experiment(
                name="tp1_r_090",
                mutations={"param_overrides.tp1_r": 0.90},
            ))
            suggestions.append(Experiment(
                name="tp2_r_175",
                mutations={"param_overrides.tp2_r": 1.75},
            ))
    elif phase == 2:
        # If frequency still low, try more aggressive regime relaxation
        if metrics.get("trades_per_month", 0) < 3.5:
            suggestions.append(Experiment(
                name="adx_on_13",
                mutations={"param_overrides.adx_on": 13},
            ))
    elif phase == 3:
        # If fill rate still low, try market orders (no slip abort + wide tolerance)
        if metrics.get("total_trades", 0) < 180:
            suggestions.append(Experiment(
                name="wide_entry",
                mutations={
                    "flags.slippage_abort": False,
                    "param_overrides.max_entry_slip_atr": 0.75,
                },
            ))

    return suggestions
