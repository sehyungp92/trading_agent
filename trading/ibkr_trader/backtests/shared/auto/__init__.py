"""Shared phased auto-optimization infrastructure."""

from .evaluation import EVALUATION_DIMENSIONS, build_end_of_round_report
from .greedy_optimizer import load_greedy_checkpoint, run_greedy, save_greedy_result
from .phase_analyzer import analyze_phase
from .phase_gates import FAILURE_CATEGORIES, evaluate_gate
from .phase_logging import PhaseLogger
from .phase_runner import PhaseRunner
from .phase_state import PhaseState, load_phase_state, save_phase_state
from .plugin import PhaseAnalysisPolicy, PhaseSpec, StrategyPlugin
from .plugin_utils import CachedBatchEvaluator, ResilientBatchEvaluator, mutation_signature
from .round_manager import RoundManager
from .types import (
    EndOfRoundArtifacts,
    Experiment,
    GateCriterion,
    GateResult,
    GreedyResult,
    GreedyRound,
    PhaseAnalysis,
    PhaseDecision,
    ScoredCandidate,
)

__all__ = [
    "EVALUATION_DIMENSIONS",
    "EndOfRoundArtifacts",
    "Experiment",
    "FAILURE_CATEGORIES",
    "GateCriterion",
    "GateResult",
    "GreedyResult",
    "GreedyRound",
    "CachedBatchEvaluator",
    "PhaseAnalysis",
    "PhaseAnalysisPolicy",
    "PhaseDecision",
    "PhaseLogger",
    "PhaseRunner",
    "PhaseSpec",
    "PhaseState",
    "ResilientBatchEvaluator",
    "RoundManager",
    "ScoredCandidate",
    "StrategyPlugin",
    "analyze_phase",
    "build_end_of_round_report",
    "evaluate_gate",
    "load_greedy_checkpoint",
    "load_phase_state",
    "mutation_signature",
    "run_greedy",
    "save_greedy_result",
    "save_phase_state",
]
