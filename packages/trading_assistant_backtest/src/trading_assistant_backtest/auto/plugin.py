"""Strategy plugin protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from trading_assistant_backtest.auto.types import Candidate, CandidateEvaluation, PhaseSpec
from trading_assistant_backtest.contract_models import (
    DataBundleManifest,
    DecisionParityReport,
    MonthlyRunManifest,
)
from trading_assistant_backtest.replay.types import ReplayResult, WindowSpec


class MonthlyStrategyPlugin(Protocol):
    plugin_id: str
    strategy_id: str
    family: str
    supported_symbols: list[str]
    supported_timeframes: list[str]

    def load_baseline(
        self, manifest: MonthlyRunManifest, data_bundle: DataBundleManifest
    ) -> Any: ...
    def run_incumbent(self, window: WindowSpec, baseline: Any) -> ReplayResult: ...
    def run_diagnostics(self, replay: ReplayResult) -> Any: ...
    def build_phase_specs(
        self, diagnostics: Any, experiment_plan: Any, search_brief: Any
    ) -> list[PhaseSpec]: ...
    def evaluate_candidate(
        self, candidate: Candidate, window: WindowSpec
    ) -> CandidateEvaluation: ...
    def build_repair_candidates(
        self, failure_analysis: Any, round_chain: Any
    ) -> list[Candidate]: ...
    def build_confirmatory_variants(self, primary: Candidate, context: Any) -> list[Candidate]: ...
    def write_round_n_plus_1(self, candidate: Candidate, output_dir: Path) -> Any: ...
    def run_decision_parity(
        self, candidate: Candidate, fixtures: list[Path]
    ) -> DecisionParityReport: ...
