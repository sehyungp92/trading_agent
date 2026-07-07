# schemas/prompt_package.py
"""Structured prompt package for all Claude analysis invocations."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class PromptPackage(BaseModel):
    """Unified prompt package returned by all assemblers."""
    system_prompt: str = ""
    task_prompt: str = ""
    data: dict = {}
    instructions: str = ""
    corrections: list[dict] = []
    context_files: list[str] = []
    skill_context: str = ""
    metadata: dict = {}
    assembled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
