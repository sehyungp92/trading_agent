"""Shared bot parameter definitions."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, model_validator


class ParameterType(str, Enum):
    YAML_FIELD = "YAML_FIELD"
    PYTHON_CONSTANT = "PYTHON_CONSTANT"
    TOML_FIELD = "TOML_FIELD"
    JSON_FIELD = "JSON_FIELD"


class ParameterDefinition(BaseModel):
    """A tunable parameter in a bot's configuration."""

    param_name: str
    bot_id: str
    strategy_id: Optional[str] = None
    param_type: ParameterType
    file_path: str
    yaml_key: Optional[str] = None
    python_path: Optional[str] = None
    current_value: Any = None
    valid_range: Optional[tuple[float, float]] = None
    valid_values: Optional[list[Any]] = None
    value_type: Literal["int", "float", "bool", "str"] = "float"
    category: str = ""
    is_safety_critical: bool = False

    @model_validator(mode="after")
    def _check_type_fields(self) -> ParameterDefinition:
        if self.param_type == ParameterType.YAML_FIELD and not self.yaml_key:
            raise ValueError("yaml_key is required when param_type is YAML_FIELD")
        if self.param_type == ParameterType.PYTHON_CONSTANT and not self.python_path:
            raise ValueError("python_path is required when param_type is PYTHON_CONSTANT")
        if self.param_type == ParameterType.TOML_FIELD and not self.python_path:
            raise ValueError(
                "python_path (used as TOML dotted path) is required when "
                "param_type is TOML_FIELD"
            )
        if self.param_type == ParameterType.JSON_FIELD and not self.python_path:
            raise ValueError(
                "python_path (used as JSON dotted path) is required when "
                "param_type is JSON_FIELD"
            )
        if self.valid_range is not None and self.valid_range[0] >= self.valid_range[1]:
            raise ValueError("valid_range[0] must be less than valid_range[1]")
        return self


__all__ = ["ParameterDefinition", "ParameterType"]
