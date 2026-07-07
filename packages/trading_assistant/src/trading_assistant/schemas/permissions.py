"""Permission gate data models."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, Field


class PermissionTier(IntEnum):
    """Higher value = more restrictive. Used for comparison."""
    AUTO = 0
    REQUIRES_APPROVAL = 1
    REQUIRES_DOUBLE_APPROVAL = 2


class PermissionCheckResult(BaseModel):
    tier: PermissionTier
    allowed: bool  # True only for AUTO tier
    flagged_files: list[str] = Field(default_factory=list)
    reason: str = ""
