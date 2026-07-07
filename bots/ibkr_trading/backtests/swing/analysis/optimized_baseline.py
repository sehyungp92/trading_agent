"""Shared helpers for truthful swing baseline diagnostics."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PhaseMutationSource:
    """Resolved mutation source for a saved phase baseline."""

    state_path: Path
    phase_result: str
    mutations: dict[str, Any]
    phase_label: str
    optimizer_reference: dict[str, Any] | None

    @property
    def canonical_phase_tag(self) -> str:
        if self.phase_result == "current":
            return "current"
        return f"phase_{self.phase_result}_final"


def load_phase_mutation_source(
    state_path: str | Path,
    phase_result: str = "current",
) -> PhaseMutationSource:
    """Load mutations plus reference metrics from a saved phase_state.json."""
    path = Path(state_path)
    state = json.loads(path.read_text(encoding="utf-8"))
    phase_results = state.get("phase_results", {})

    if phase_result == "current":
        mutations = dict(state.get("cumulative_mutations", {}))
        current_phase = state.get("current_phase")
        reference = phase_results.get(str(current_phase), phase_results.get(current_phase))
        phase_label = "CURRENT OPTIMIZED BASELINE"
    else:
        phase_key = str(phase_result)
        reference = phase_results.get(phase_key, phase_results.get(phase_result))
        if not isinstance(reference, dict):
            raise KeyError(f"No phase result '{phase_result}' found in {path}")
        mutations = dict(reference.get("final_mutations", {}))
        phase_label = f"PHASE {phase_result} FINAL BASELINE"

    if not mutations and not path.exists():
        raise FileNotFoundError(path)

    return PhaseMutationSource(
        state_path=path,
        phase_result=phase_result,
        mutations=mutations,
        phase_label=phase_label,
        optimizer_reference=reference if isinstance(reference, dict) else None,
    )


def summarize_optimizer_reference(reference: dict[str, Any] | None) -> list[str]:
    """Build short report lines for independent optimizer reference metrics."""
    if not reference:
        return []
    metrics = reference.get("final_metrics", {})
    score = reference.get("final_score")

    lines = ["Optimizer reference (historical independent fast-path phase-search basis; not the current full replay headline):"]
    if score is not None:
        lines.append(f"  Score: {float(score):.4f}")
    if metrics:
        pf = metrics.get("profit_factor")
        net_return = metrics.get("net_return_pct")
        dd_pct = metrics.get("max_dd_pct")
        dd_r = metrics.get("max_r_dd")
        trades = metrics.get("total_trades")
        if trades is not None:
            lines.append(f"  Trades: {int(trades)}")
        if pf is not None:
            lines.append(f"  Profit factor: {float(pf):.2f}")
        if net_return is not None:
            lines.append(f"  Net return: {float(net_return):+.1f}%")
        if dd_pct is not None:
            dd_value = float(dd_pct)
            if abs(dd_value) <= 1.0:
                lines.append(f"  Max drawdown: {dd_value:.2%}")
            else:
                lines.append(f"  Max drawdown: {dd_value:.2f}%")
        elif dd_r is not None:
            lines.append(f"  Max R drawdown: {float(dd_r):.2f}R")
    return lines
