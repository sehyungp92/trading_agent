"""Local clean-checkout manager for pinned live strategy repos."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LiveRepoCheckoutSpec:
    repo_url: str
    commit_sha: str
    checkout_root: Path

    def __post_init__(self) -> None:
        if not self.repo_url.strip():
            raise ValueError("repo_url is required")
        if not self.commit_sha.strip():
            raise ValueError("commit_sha is required")


class LiveRepoCloneManager:
    def __init__(self, checkout_root: Path) -> None:
        self.checkout_root = checkout_root.resolve()

    def checkout_path(self, repo_url: str, commit_sha: str) -> Path:
        repo_name = Path(repo_url).name or repo_url.rstrip("/\\").replace("\\", "/").split("/")[-1]
        safe_repo = re.sub(r"[^A-Za-z0-9_.-]+", "_", repo_name)
        if safe_repo.endswith(".git"):
            safe_repo = safe_repo[:-4]
        suffix = hashlib.sha256(repo_url.encode("utf-8")).hexdigest()[:10]
        return self.checkout_root / f"{safe_repo}_{suffix}_{commit_sha[:12]}"

    def prepare(self, spec: LiveRepoCheckoutSpec) -> Path:
        source = Path(spec.repo_url)
        if source.exists():
            errors = validate_clean_checkout(source, spec.commit_sha)
            if errors:
                raise RuntimeError("; ".join(errors))
            return source.resolve()
        target = self.checkout_path(spec.repo_url, spec.commit_sha)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.resolve().relative_to(self.checkout_root)
        if (target / ".git").exists():
            _git(["fetch", "--all", "--tags", "--prune"], target)
        else:
            _git(["clone", "--no-checkout", spec.repo_url, str(target)], self.checkout_root)
        _git(["checkout", "--detach", spec.commit_sha], target)
        errors = validate_clean_checkout(target, spec.commit_sha)
        if errors:
            raise RuntimeError("; ".join(errors))
        return target


def validate_clean_checkout(repo_path: Path, expected_commit_sha: str) -> list[str]:
    repo = repo_path.resolve()
    worktree = _worktree_root(repo)
    if worktree is None:
        return [f"live repo checkout is not a git repo: {repo}"]
    errors: list[str] = []
    head = _git(["rev-parse", "HEAD"], worktree).strip()
    if head != expected_commit_sha:
        errors.append(f"live repo HEAD {head} does not match expected {expected_commit_sha}")
    if (repo / ".git").exists():
        status = _git(["status", "--porcelain"], repo).strip()
        if status:
            errors.append("live repo checkout has uncommitted changes")
    return errors


def validate_pinned_head(repo_path: Path, expected_commit_sha: str) -> list[str]:
    """Validate only the pinned HEAD, leaving cleanliness to approval audit."""

    repo = repo_path.resolve()
    worktree = _worktree_root(repo)
    if worktree is None:
        return [f"live repo checkout is not a git repo: {repo}"]
    head = _git(["rev-parse", "HEAD"], worktree).strip()
    if head != expected_commit_sha:
        return [f"live repo HEAD {head} does not match expected {expected_commit_sha}"]
    return []


def _worktree_root(path: Path) -> Path | None:
    try:
        return Path(_git(["rev-parse", "--show-toplevel"], path).strip()).resolve()
    except RuntimeError:
        return None


def _git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout
