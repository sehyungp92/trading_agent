"""Helix R2 Exit Alpha plugin -- subclasses HelixPlugin with exit-focused optimization.

Key differences from R1:
  - initial_mutations = R1 final config (all phases build on top)
  - Exit-weighted scoring (30% exit_efficiency, 15% waste_ratio)
  - Tighter gates (starts from strong R1 baseline)
  - 4 phases: TRAILING_JOINT, R_BAND, CLASS_SPECIFIC, FINETUNE
"""
from __future__ import annotations

from typing import Any

from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.plugin_utils import (
    deserialize_experiments,
)
from backtests.shared.auto.types import Experiment, GateCriterion

from backtests.swing.auto.helix.plugin import HelixPlugin

from .phase_candidates import get_phase_candidates
from .phase_gates import GATE_FN, gate_criteria_phase_4


# ---------------------------------------------------------------------------
# R1 final mutations (baseline for all R2 phases)
# ---------------------------------------------------------------------------

ROUND_1_MUTATIONS: dict[str, Any] = {
    "param_overrides.ADX_UPPER_GATE": 40,
    "flags.disable_class_a": True,
    "param_overrides.CLASS_B_MOM_LOOKBACK": 5,
    "param_overrides.TRAIL_STALL_ONSET": 6,
    "param_overrides.CLASS_B_BAIL_BARS": 10,
    "param_overrides.ADD_RISK_FRAC": 0.84,
    "param_overrides.ADD_1H_R": 0.9,
    "param_overrides.ADD_4H_R": 0.4,
}


# ---------------------------------------------------------------------------
# Exit-focused scoring weights
# ---------------------------------------------------------------------------

_EXIT_WEIGHTS = {
    "exit_efficiency": 0.30, "waste_ratio": 0.15, "net_profit": 0.18,
    "pf": 0.10, "tail_preservation": 0.12, "inv_dd": 0.08, "frequency": 0.07,
}
PHASE_WEIGHTS: dict[int, dict[str, float]] = {
    1: _EXIT_WEIGHTS, 2: _EXIT_WEIGHTS, 3: _EXIT_WEIGHTS,
    4: {
        "exit_efficiency": 0.20, "waste_ratio": 0.12, "net_profit": 0.18,
        "pf": 0.14, "tail_preservation": 0.14, "inv_dd": 0.12, "frequency": 0.10,
    },
}

_HARD_REJECTS = {"min_trades": 200, "min_pf": 1.2, "max_r_dd": 25.0, "min_tail_pct": 0.30, "min_regime_pf": 0.80}
PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {p: _HARD_REJECTS for p in range(1, 5)}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("TRAILING_JOINT + STOP_REPLAY", ["exit_efficiency", "waste_ratio", "profit_factor"]),
    2: ("R_BAND_TRAILING", ["exit_efficiency", "net_return_pct", "tail_pct"]),
    3: ("CLASS_SPECIFIC", ["exit_efficiency", "waste_ratio", "net_return_pct"]),
    4: ("FINETUNE", ["calmar_r", "net_return_pct", "exit_efficiency"]),
}

ULTIMATE_TARGETS = {
    "net_return_pct": 150.0,
    "profit_factor": 3.0,
    "max_r_dd": 7.0,
    "exit_efficiency": 0.50,
    "total_trades": 400.0,
}


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class HelixExitAlphaPlugin(HelixPlugin):
    """Helix R2 exit alpha capture -- inherits HelixPlugin infrastructure."""

    name = "helix_exit_alpha"
    num_phases = 4
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations = ROUND_1_MUTATIONS

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        prior_phase = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior_phase.get("suggested_experiments", []))
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_phase_candidates(
                phase,
                prior_mutations=state.cumulative_mutations if phase == 4 else None,
                suggested_experiments=[(e.name, e.mutations) for e in suggested] or None,
            )
        ]

        p3_metrics = state.get_phase_metrics(3) if phase == 4 else None

        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics, _p=phase, _p3=p3_metrics: self._gate_criteria(_p, metrics, _p3),
            scoring_weights=PHASE_WEIGHTS.get(phase),
            hard_rejects=PHASE_HARD_REJECTS.get(phase, {}),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=0.01,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                suggest_experiments_fn=self.suggest_experiments,
                redesign_scoring_weights_fn=self.redesign_scoring_weights,
                build_extra_analysis_fn=self.build_analysis_extra,
                format_extra_analysis_fn=self.format_analysis_extra,
            ),
            max_rounds=20,
            prune_threshold=0.05,
        )

    def _gate_criteria(
        self,
        phase: int,
        metrics: dict[str, float],
        prior_phase_metrics: dict[str, float] | None = None,
    ) -> list[GateCriterion]:
        if phase == 4:
            return gate_criteria_phase_4(metrics, prior_phase_metrics)
        gate_fn = GATE_FN.get(phase)
        if gate_fn is None:
            return []
        return gate_fn(metrics)
