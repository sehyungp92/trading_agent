"""Data containers for the optimization framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Experiment:
    """A single candidate mutation to evaluate."""

    name: str
    mutations: dict[str, Any]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Experiment):
            return NotImplemented
        return self.name == other.name


@dataclass
class ScoredCandidate:
    """Result of evaluating one experiment."""

    experiment: Experiment
    score: float
    metrics: dict[str, float]
    rejected: bool = False
    reject_reason: str = ""


@dataclass
class GateCriterion:
    """A single gate condition to check."""

    metric: str
    operator: str  # ">=", "<=", ">", "<"
    threshold: float
    weight: float = 1.0


@dataclass
class GateResult:
    """Result of evaluating phase gate criteria."""

    passed: bool
    criteria_results: list[tuple[GateCriterion, float, bool]]
    failure_reasons: list[str]
    failure_category: str | None = None


@dataclass
class GreedyRound:
    """Structured record of one greedy forward-selection round."""

    round_num: int
    candidates_tested: int
    best_name: str
    best_score: float
    best_delta_pct: float
    kept: bool
    rejected_count: int = 0


@dataclass
class GreedyResult:
    """Output of the greedy forward-selection optimizer."""

    accepted_experiments: list[ScoredCandidate]
    rejected_experiments: list[ScoredCandidate]
    final_mutations: dict[str, Any]
    final_score: float
    final_metrics: dict[str, float] = field(default_factory=dict)
    rounds: list[GreedyRound] = field(default_factory=list)
    base_score: float = 0.0
    kept_features: list[str] = field(default_factory=list)
    total_candidates: int = 0
    accepted_count: int = 0
    elapsed_seconds: float = 0.0


# ── Callback type aliases for PhaseAnalysisPolicy ────────────────────

DiagnosticGapFn = Callable[[int, dict[str, float]], list[str]]
SuggestExperimentsFn = Callable[
    [int, dict[str, float], list[str], Any], list[Experiment]
]
RedesignScoringWeightsFn = Callable[
    [int, dict[str, float], dict[str, float], list[str], list[str]],
    dict[str, float] | None,
]
BuildExtraAnalysisFn = Callable[
    [int, dict[str, float], Any, Any], dict[str, Any]
]
FormatExtraAnalysisFn = Callable[[dict[str, Any]], str]
DecideActionFn = Callable[..., Any]  # returns PhaseDecision | None


@dataclass
class PhaseAnalysisPolicy:
    """Controls how phase analysis generates recommendations."""

    max_scoring_retries: int = 2
    max_diagnostic_retries: int = 1
    focus_metrics: list[str] = field(default_factory=list)
    min_effective_score_delta_pct: float = 0.01

    # Optional callback hooks
    diagnostic_gap_fn: DiagnosticGapFn | None = None
    suggest_experiments_fn: SuggestExperimentsFn | None = None
    redesign_scoring_weights_fn: RedesignScoringWeightsFn | None = None
    build_extra_analysis_fn: BuildExtraAnalysisFn | None = None
    format_extra_analysis_fn: FormatExtraAnalysisFn | None = None
    decide_action_fn: DecideActionFn | None = None


@dataclass
class PhaseDecision:
    """Result of the decide_action callback."""

    action: str  # "improve_scoring", "improve_diagnostics", "advance"
    reason: str
    scoring_weight_overrides: dict[str, float] | None = None
    scoring_assessment_override: str | None = None
    extra_diagnostic_gaps: list[str] = field(default_factory=list)
    extra_suggested_experiments: list[Experiment] = field(default_factory=list)


@dataclass
class PhaseAnalysis:
    """Result of analyzing a completed phase."""

    phase: int
    recommendation: str  # "advance", "improve_scoring", "improve_diagnostics"
    summary: str
    goal_progress: dict[str, dict]
    scoring_assessment: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    diagnostic_gaps: list[str] = field(default_factory=list)
    suggested_experiments: list[Experiment] = field(default_factory=list)
    recommendation_reason: str = ""
    report: str = ""
    scoring_weight_overrides: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EndOfRoundArtifacts:
    """End-of-round evaluation artifacts with text-based dimension reports."""

    final_diagnostics_text: str = ""
    dimension_reports: dict[str, str] = field(default_factory=dict)
    overall_verdict: str = ""
    extra_sections: dict[str, str] = field(default_factory=dict)


@dataclass
class PhaseSpec:
    """Specification for a single optimization phase."""

    phase_num: int
    name: str
    candidates: list[Experiment]
    scoring_weights: dict[str, float]
    hard_rejects: dict[str, tuple[str, float]]  # metric -> (operator, threshold)
    gate_criteria: list[GateCriterion]
    gate_criteria_fn: Callable[[dict[str, float]], list[GateCriterion]] | None = None
    analysis_policy: PhaseAnalysisPolicy = field(default_factory=PhaseAnalysisPolicy)
    min_delta: float = 0.005
    focus: str = ""
    max_rounds: int | None = None
    prune_threshold: float | None = None


# Type alias for the evaluate function
EvaluateFn = Callable[[list[Experiment], dict[str, Any]], list[ScoredCandidate]]
