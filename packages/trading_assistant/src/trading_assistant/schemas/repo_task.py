"""Shared schemas for repo mutation task execution."""
from __future__ import annotations

from pydantic import BaseModel


class RepoTaskContext(BaseModel):
    """Execution context for a repo mutation task."""

    task_id: str
    repo_url: str = ""
    repo_dir: str = ""
    default_branch: str = "main"
    repo_cache_dir: str = ""
    worktree_dir: str = ""
    artifact_dir: str = ""


__all__ = ["RepoTaskContext"]
