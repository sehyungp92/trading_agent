from __future__ import annotations

from .phase_state import PhaseState
from .types import EndOfRoundArtifacts

EVALUATION_DIMENSIONS = (
    "signal_extraction",
    "signal_discrimination",
    "entry_mechanism",
    "trade_management",
    "exit_mechanism",
)

_DIMENSION_TITLES = {
    "signal_extraction": "Signal Extraction / Alpha Capture",
    "signal_discrimination": "Signal Discrimination",
    "entry_mechanism": "Entry Mechanism",
    "trade_management": "Trade Management",
    "exit_mechanism": "Exit Mechanism",
}


def _title_for_dimension(name: str) -> str:
    return _DIMENSION_TITLES.get(name, name.replace("_", " ").title())


def build_end_of_round_report(
    strategy_name: str,
    state: PhaseState,
    artifacts: EndOfRoundArtifacts,
) -> str:
    dimension_reports = artifacts.dimension_reports
    lines = [
        "=" * 70,
        f"{strategy_name.upper()} END-OF-ROUND EVALUATION",
        "=" * 70,
        "",
        "Phase Progression Summary:",
    ]

    for phase in sorted(state.phase_results):
        result = state.phase_results[phase]
        lines.append(
            f"  Phase {phase}: score {result.get('base_score', 0.0):.4f} -> "
            f"{result.get('final_score', 0.0):.4f} "
            f"({len(result.get('kept_features', []))} accepted)"
        )

    lines.extend(["", "Cumulative Mutations Applied:"])
    if state.cumulative_mutations:
        for key, value in sorted(state.cumulative_mutations.items()):
            lines.append(f"  {key}: {value}")
    else:
        lines.append("  (none)")

    for dimension in EVALUATION_DIMENSIONS:
        lines.extend(["", _title_for_dimension(dimension), dimension_reports.get(dimension, "No report provided.")])

    extra_dimensions = [name for name in artifacts.extra_sections if name not in EVALUATION_DIMENSIONS]
    for dimension in extra_dimensions:
        lines.extend(["", _title_for_dimension(dimension), artifacts.extra_sections.get(dimension, "No report provided.")])

    verdict = artifacts.overall_verdict or ("Ready for the next research round." if state.completed_phases else "Round incomplete.")
    lines.extend(["", "Overall Verdict", verdict])
    return "\n".join(lines)
