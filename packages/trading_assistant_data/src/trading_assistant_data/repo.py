"""Repository path helpers."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

GIT_COMMIT_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")


def resolve_repo_root(path: Path | None = None) -> Path:
    candidate = Path(path) if path else Path.cwd()
    candidate = candidate.resolve()
    if (candidate / "pyproject.toml").exists() and candidate.name == "trading_assistant_data":
        return candidate
    if (candidate / "packages" / "trading_assistant_data" / "pyproject.toml").exists():
        return (candidate / "packages" / "trading_assistant_data").resolve()
    if (candidate / "trading_assistant_data" / "pyproject.toml").exists():
        return (candidate / "trading_assistant_data").resolve()
    return candidate


def git_commit_sha(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def is_git_commit_sha(value: str) -> bool:
    return bool(GIT_COMMIT_RE.fullmatch(str(value or "").strip()))


def git_commit_exists(repo_root: Path, commit_sha: str) -> bool:
    if not is_git_commit_sha(commit_sha):
        return False
    try:
        completed = subprocess.run(
            ["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return False
    return completed.returncode == 0


def git_dirty_paths(repo_root: Path, paths: list[Path]) -> list[str]:
    rel_paths = [_rel(path, repo_root) for path in paths]
    if not rel_paths:
        return []
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--", *rel_paths],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception:
        return rel_paths
    if completed.returncode != 0:
        return rel_paths
    dirty: list[str] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        dirty.append(line[3:].strip().strip('"') or line.strip())
    return dirty


def git_branch(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")
