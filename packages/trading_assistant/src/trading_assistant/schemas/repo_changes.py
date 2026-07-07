"""Shared repository change and pull-request schemas."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from trading_assistant.schemas.repo_task import RepoTaskContext


class ChangeKind(str, Enum):
    PARAMETER_CHANGE = "parameter_change"
    BUG_FIX = "bug_fix"
    STRUCTURAL_CHANGE = "structural_change"
    ROLLBACK = "rollback"


class FileChangeMode(str, Enum):
    FILE_REPLACE = "file_replace"
    YAML_FIELD = "yaml_field"
    PYTHON_CONSTANT = "python_constant"
    TOML_FIELD = "toml_field"
    JSON_FIELD = "json_field"
    UNIFIED_DIFF = "unified_diff"


class FileChange(BaseModel):
    """A single file modification."""

    file_path: str
    original_content: str = ""
    new_content: str = ""
    change_mode: FileChangeMode = FileChangeMode.FILE_REPLACE
    metadata: dict[str, Any] = Field(default_factory=dict)
    patch: str = ""
    diff_preview: str = ""


class PRRequest(BaseModel):
    """Request to create a PR in a bot repository."""

    approval_request_id: str
    suggestion_id: str
    bot_id: str
    repo_dir: str
    branch_name: str
    title: str
    body: str = ""
    change_kind: ChangeKind = ChangeKind.PARAMETER_CHANGE
    draft: bool = False
    verification_commands: list[str] = Field(default_factory=list)
    repo_task: Optional[RepoTaskContext] = None
    file_changes: list[FileChange] = Field(default_factory=list)


class GitHubIssueRequest(BaseModel):
    """Request to create or deduplicate a GitHub issue."""

    bot_id: str
    title: str
    body: str
    repo_dir: str
    labels: list[str] = Field(default_factory=list)
    dedupe_key: str = ""
    repo_task: Optional[RepoTaskContext] = None
    change_kind: ChangeKind = ChangeKind.BUG_FIX


class GitHubIssueResult(BaseModel):
    """Result of a GitHub issue creation attempt."""

    success: bool
    issue_url: Optional[str] = None
    issue_number: Optional[int] = None
    error: Optional[str] = None
    existing_issue_url: Optional[str] = None


class PreflightResult(BaseModel):
    """Result of pre-flight checks before PR creation."""

    passed: bool = True
    checks: list[dict] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ReviewState(str, Enum):
    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    COMMENTED = "COMMENTED"
    PENDING = "PENDING"


class PRReviewStatus(BaseModel):
    """Status of PR reviews from GitHub."""

    pr_number: int
    pr_url: str = ""
    review_state: ReviewState = ReviewState.PENDING
    reviewers: list[str] = Field(default_factory=list)
    actionable_comments: list[str] = Field(default_factory=list)
    needs_attention: bool = False
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PRResult(BaseModel):
    """Result of a PR creation attempt."""

    success: bool
    pr_url: Optional[str] = None
    branch_name: str = ""
    error: Optional[str] = None
    pr_number: Optional[int] = None
    preflight: Optional[PreflightResult] = None
    existing_pr_url: Optional[str] = None
    diff_summary: list[str] = Field(default_factory=list)


__all__ = [
    "ChangeKind",
    "FileChange",
    "FileChangeMode",
    "GitHubIssueRequest",
    "GitHubIssueResult",
    "PRRequest",
    "PRResult",
    "PRReviewStatus",
    "PreflightResult",
    "ReviewState",
]
