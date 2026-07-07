"""Models for generated effective live config evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


CONFIG_MERGE_ORDER = (
    "strategy_defaults",
    "latest_optimized_config",
    "venue_runtime_overlay",
    "environment_deployment_overlay",
    "environment_secrets_uncommitted",
)


class SourceFileReference(BaseModel):
    role: str
    path: str
    sha256: str
    canonical_json_sha256: str = ""


class PromotionReference(BaseModel):
    strategy_id: str
    path: str
    sha256: str
    canonical_json_sha256: str
    promotion_state: str


class EffectiveConfigSnapshot(BaseModel):
    schema_version: str = "effective_live_config_artifact.v1"
    bot_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    merge_order: tuple[str, ...] = CONFIG_MERGE_ORDER
    source_files: list[SourceFileReference]
    promotion_manifests: list[PromotionReference]
    materialized_config: dict[str, Any] = Field(default_factory=dict)
    materialized_config_hash: str
    effective_config_hash: str
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_snapshot(self) -> "EffectiveConfigSnapshot":
        if self.merge_order != CONFIG_MERGE_ORDER:
            raise ValueError("effective config artifact merge order does not match policy")
        if not self.bot_id.strip():
            raise ValueError("effective config artifact bot_id is required")
        if not self.source_files:
            raise ValueError("effective config artifact requires source files")
        if not self.promotion_manifests:
            raise ValueError("effective config artifact requires promotion manifests")
        if not self.materialized_config:
            raise ValueError("effective config artifact requires materialized runtime config")
        if not self.materialized_config_hash.strip():
            raise ValueError("effective config artifact materialized hash is required")
        if not self.effective_config_hash.strip():
            raise ValueError("effective config artifact hash is required")
        return self
