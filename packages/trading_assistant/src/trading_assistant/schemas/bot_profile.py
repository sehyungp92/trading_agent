"""Shared bot repository and configuration profile schemas."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from trading_assistant.schemas.parameter_definition import ParameterDefinition


class BotConfigProfile(BaseModel):
    """Configuration profile for a single bot."""

    bot_id: str
    repo_url: str = ""
    repo_dir: str = ""
    default_branch: str = "main"
    strategy_version: Optional[str] = None
    config_version: Optional[str] = None
    code_sha: Optional[str] = None
    allowed_edit_paths: list[str] = Field(default_factory=list)
    structural_context_paths: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    parameters: list[ParameterDefinition] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)

    def get_parameter(self, param_name: str) -> Optional[ParameterDefinition]:
        for param in self.parameters:
            if param.param_name == param_name:
                return param
        return None

    def get_parameters_by_category(self, category: str) -> list[ParameterDefinition]:
        return [param for param in self.parameters if param.category == category]


__all__ = ["BotConfigProfile"]
