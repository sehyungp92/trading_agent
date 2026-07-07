"""Task registry data models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRecord(BaseModel):
    id: str
    type: str
    agent: str
    status: TaskStatus = TaskStatus.PENDING
    context_files: list[str] = []
    run_folder: str = ""
    retries: int = 0
    max_retries: int = 3
    result_summary: str = ""
    error: str = ""
    source_event_id: str = ""
    source_action_type: str = ""
    subagent_id: str = ""
    notify_on_complete: bool = True
    notify_channels: list[str] = Field(default_factory=lambda: ["telegram"])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
