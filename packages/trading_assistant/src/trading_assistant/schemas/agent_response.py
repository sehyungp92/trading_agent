# schemas/agent_response.py
"""Structured response schemas for parsed Claude analysis outputs.

Every Claude invocation should emit a structured block that gets parsed into
ParsedAnalysis. This enables programmatic tracking, validation, and feedback.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from trading_assistant.schemas.repo_changes import FileChange

logger = logging.getLogger(__name__)


# Shared mapping from suggestion category → suggestion tier.
# Used in _record_agent_suggestions (handlers.py) and suggestion_scorer.py.
CATEGORY_TO_TIER: dict[str, str] = {
    "exit_timing": "parameter",
    "filter_threshold": "filter",
    "stop_loss": "parameter",
    "signal": "parameter",
    "structural": "hypothesis",
    "position_sizing": "parameter",
    "regime_gate": "filter",
    "funding_threshold": "filter",
    "leverage_cap": "parameter",
    "confluence_count": "filter",
    "setup_grade_filter": "filter",
    "portfolio_allocation": "portfolio",
    "portfolio_risk_cap": "portfolio",
    "portfolio_coordination": "portfolio",
    "portfolio_drawdown_tier": "portfolio",
}

# Categories accepted by AgentSuggestion. Empty string = unspecified (allowed for
# backward compatibility with prompts that don't require it). "uncategorized" is
# the canonical bucket for values the model emits that aren't in CATEGORY_TO_TIER.
_KNOWN_SUGGESTION_CATEGORIES: frozenset[str] = frozenset(
    list(CATEGORY_TO_TIER) + ["", "uncategorized"]
)


class AgentPrediction(BaseModel):
    """A specific, measurable prediction about a bot's future performance."""

    bot_id: str
    strategy_id: Optional[str] = None  # specific strategy this prediction targets; None = bot-wide
    metric: Literal["pnl", "win_rate", "drawdown", "sharpe"]
    direction: Literal["improve", "decline", "stable"]
    confidence: float = Field(ge=0.0, le=1.0)
    timeframe_days: int = 7
    reasoning: str = ""


class AgentSuggestion(BaseModel):
    """A structured suggestion extracted from Claude's analysis."""

    suggestion_id: str = ""
    bot_id: str
    strategy_id: Optional[str] = None  # specific strategy this suggestion targets; None = bot-wide
    category: str = ""  # exit_timing, filter_threshold, stop_loss, signal, structural, position_sizing, regime_gate
    title: str
    expected_impact: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_summary: str = ""
    proposed_value: Optional[float] = None  # numeric value for parameter suggestions
    target_param: Optional[str] = None  # parameter name this suggestion targets
    engine: str = ""  # engine tag if suggestion targets specific engine (e.g., "REVERSAL")
    ablation_flag: str = ""  # flag name if ablation toggle (e.g., "fade_oscillation_gate")
    regime_condition: str = ""  # regime name if regime-conditional (e.g., "volatile")

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, v: Any) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        if s in _KNOWN_SUGGESTION_CATEGORIES:
            return s
        logger.warning(
            "AgentSuggestion: unknown category %r — mapping to 'uncategorized'", s,
        )
        return "uncategorized"


class StructuralProposal(BaseModel):
    """A proposal for structural changes to a bot's architecture."""

    hypothesis_id: Optional[str] = None
    linked_suggestion_id: Optional[str] = None
    bot_id: str
    affected_strategy_id: str = ""
    affected_config_version: str = ""
    title: str
    description: str = ""
    reversibility: Literal["easy", "moderate", "hard"] = "moderate"
    evidence: str = ""
    evidence_paths: list[str] = Field(default_factory=list)
    capability_labels: list[str] = Field(default_factory=list)
    evidence_authority: str = ""
    risk_classification: str = ""
    objective_impact: dict[str, float] = Field(default_factory=dict)
    replay_or_experiment_plan: str = ""
    rollback_plan: str = ""
    routing: str = ""
    estimated_complexity: Literal["low", "medium", "high"] = "medium"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    file_changes: list[FileChange] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    acceptance_criteria: list[dict] = Field(default_factory=list)


class ParsedAnalysis(BaseModel):
    """Result of parsing Claude's structured output block."""

    predictions: list[AgentPrediction] = Field(default_factory=list)
    suggestions: list[AgentSuggestion] = Field(default_factory=list)
    structural_proposals: list[StructuralProposal] = Field(default_factory=list)
    portfolio_proposals: list[Any] = Field(default_factory=list)  # list[PortfolioProposal] — untyped to avoid circular import
    raw_report: str = ""
    parse_success: bool = True
    fallback_used: bool = False
    raw_structured: Optional[dict] = None  # Full parsed JSON from STRUCTURED_OUTPUT block
    dropped_counts: dict[str, int] = Field(default_factory=dict)
