"""StrategyPlugin protocol — interface that strategy-specific plugins implement."""

from __future__ import annotations

from typing import Any, Protocol

from crypto_trader.optimize.types import (
    EndOfRoundArtifacts,
    EvaluateFn,
    GreedyResult,
    PhaseSpec,
)


class StrategyPlugin(Protocol):
    """Protocol that strategy-specific optimization plugins must implement."""

    @property
    def name(self) -> str: ...

    @property
    def num_phases(self) -> int: ...

    @property
    def ultimate_targets(self) -> dict[str, float]:
        """Final performance targets across all phases."""
        ...

    @property
    def initial_mutations(self) -> dict[str, Any]:
        """Starting mutations (empty for default config)."""
        ...

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        """Return the specification for the given phase."""
        ...

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        scoring_weights: dict[str, float],
        hard_rejects: dict[str, tuple[str, float]],
    ) -> EvaluateFn:
        """Create an evaluate function for the greedy optimizer.

        The returned function takes (candidates, current_mutations) and returns
        scored candidates. The plugin's closure handles merging mutations and
        running the backtest.
        """
        ...

    def compute_final_metrics(
        self,
        mutations: dict[str, Any],
    ) -> dict[str, float]:
        """Run walk-forward validation and return test-set metrics."""
        ...

    def run_phase_diagnostics(
        self,
        phase: int,
        state: Any,
        metrics: dict[str, float],
        greedy_result: GreedyResult,
    ) -> str:
        """Run diagnostics for the phase. Returns text summary."""
        ...

    def run_enhanced_diagnostics(
        self,
        phase: int,
        state: Any,
        metrics: dict[str, float],
        greedy_result: GreedyResult,
    ) -> str:
        """Run enhanced diagnostics with deeper analysis. Returns text summary."""
        ...

    def build_end_of_round_artifacts(
        self,
        state: Any,
    ) -> EndOfRoundArtifacts:
        """Build end-of-round evaluation artifacts."""
        ...
