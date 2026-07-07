"""Schemas for agent runtime provider selection and preferences."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class AgentProvider(str, Enum):
    CLAUDE_MAX = "claude_max"
    CODEX_PRO = "codex_pro"
    ZAI_CODING_PLAN = "zai_coding_plan"
    OPENROUTER = "openrouter"


class AgentWorkflow(str, Enum):
    DAILY_ANALYSIS = "daily_analysis"
    WEEKLY_ANALYSIS = "weekly_analysis"
    MONTHLY_VALIDATION = "monthly_validation"
    MONTHLY_MODEL_REVIEW = "monthly_model_review"
    MONTHLY_VERIFIER = "monthly_verifier"
    TRIAGE = "triage"


class AgentSelection(BaseModel):
    provider: AgentProvider
    model: str | None = None

    @model_validator(mode="after")
    def _normalize_model(self) -> AgentSelection:
        if self.model is not None:
            trimmed = self.model.strip()
            self.model = trimmed or None
        return self


class FallbackEntry(BaseModel):
    provider: AgentProvider
    model: str | None = None


class WorkflowTuning(BaseModel):
    timeout_seconds: int | None = None
    max_turns: int | None = None
    allowed_tools: list[str] | None = None


class AgentPreferences(BaseModel):
    default: AgentSelection = Field(
        default_factory=lambda: AgentSelection(provider=AgentProvider.CODEX_PRO)
    )
    overrides: dict[AgentWorkflow, AgentSelection | None] = Field(default_factory=dict)
    fallback_chain: list[FallbackEntry] = Field(default_factory=list)
    workflow_tuning: dict[AgentWorkflow, WorkflowTuning] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_wfo(cls, data):
        if not isinstance(data, dict):
            return data
        cleaned = dict(data)
        for key in ("overrides", "workflow_tuning"):
            value = cleaned.get(key)
            if isinstance(value, dict) and "wfo" in value:
                value = dict(value)
                value.pop("wfo", None)
                cleaned[key] = value
        return cleaned


class ProviderReadiness(BaseModel):
    provider: AgentProvider
    available: bool
    runtime: str
    reason: str = ""


class AgentPreferencesView(BaseModel):
    default: AgentSelection
    overrides: dict[AgentWorkflow, AgentSelection | None] = Field(default_factory=dict)
    effective: dict[AgentWorkflow, AgentSelection] = Field(default_factory=dict)
    providers: list[ProviderReadiness] = Field(default_factory=list)
