"""Managed repo caches and disposable worktrees for bot repo tasks."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from trading_assistant.schemas.bot_profile import BotConfigProfile
from trading_assistant.schemas.repo_task import RepoTaskContext


class RepoWorkspaceManager:
    """Maintains bare repo caches plus disposable task worktrees."""

    def __init__(self, cache_root: Path, task_root: Path) -> None:
        self._cache_root = Path(cache_root)
        self._task_root = Path(task_root)

    def prepare_workspace(
        self,
        profile: BotConfigProfile,
        task_id: str,
    ) -> RepoTaskContext:
        artifact_dir = self._task_root / task_id
        worktree_dir = artifact_dir / "worktree"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        source_dir, is_bare = self._ensure_source_repo(profile)
        ref = self._sync_source_repo(source_dir, profile.default_branch, is_bare)
        self._add_worktree(source_dir, worktree_dir, ref, is_bare)

        context = RepoTaskContext(
            task_id=task_id,
            repo_url=profile.repo_url,
            repo_dir=profile.repo_dir,
            default_branch=profile.default_branch,
            repo_cache_dir=str(source_dir if is_bare else ""),
            worktree_dir=str(worktree_dir),
            artifact_dir=str(artifact_dir),
        )
        (artifact_dir / "repo_task.json").write_text(
            context.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return context

    def cleanup(self, context: RepoTaskContext) -> None:
        worktree_dir = Path(context.worktree_dir)
        if not worktree_dir.exists():
            return

        source_dir = Path(context.repo_cache_dir or context.repo_dir)
        is_bare = bool(context.repo_cache_dir)
        try:
            self._run_git_worktree_remove(source_dir, worktree_dir, is_bare)
        except RuntimeError:
            shutil.rmtree(worktree_dir, ignore_errors=True)

    def collect_context_files(
        self,
        profile: BotConfigProfile,
        worktree_dir: Path | str,
        preferred_paths: list[str] | None = None,
        max_files: int = 20,
    ) -> list[str]:
        root = Path(worktree_dir)
        ordered_patterns: list[str] = []
        for path in preferred_paths or []:
            if path and path not in ordered_patterns:
                ordered_patterns.append(path)
        for pattern in profile.structural_context_paths:
            if pattern and pattern not in ordered_patterns:
                ordered_patterns.append(pattern)

        collected: list[str] = []
        seen: set[str] = set()
        for pattern in ordered_patterns:
            for relative_path in self._expand_pattern(root, pattern):
                if relative_path in seen:
                    continue
                seen.add(relative_path)
                collected.append(relative_path)
                if len(collected) >= max_files:
                    return collected
        return collected

    def _ensure_source_repo(self, profile: BotConfigProfile) -> tuple[Path, bool]:
        if profile.repo_url:
            self._cache_root.mkdir(parents=True, exist_ok=True)
            cache_dir = self._cache_root / self._cache_name(profile)
            if not cache_dir.exists():
                self._run(
                    ["git", "clone", "--mirror", profile.repo_url, str(cache_dir)],
                    cwd=self._cache_root.parent,
                )
            else:
                self._run(
                    ["git", "--git-dir", str(cache_dir), "remote", "set-url", "origin", profile.repo_url],
                    cwd=self._cache_root,
                )
            return cache_dir, True

        if profile.repo_dir:
            repo_dir = Path(profile.repo_dir)
            if not repo_dir.exists():
                raise FileNotFoundError(f"Legacy repo_dir does not exist: {repo_dir}")
            return repo_dir, False

        raise ValueError(f"Bot {profile.bot_id} has neither repo_url nor repo_dir configured")

    def _sync_source_repo(
        self,
        source_dir: Path,
        default_branch: str,
        is_bare: bool,
    ) -> str:
        if is_bare:
            self._run(
                ["git", "--git-dir", str(source_dir), "fetch", "--prune", "origin"],
                cwd=self._cache_root,
            )
            preferred_ref = f"refs/heads/{default_branch}"
            verify = self._run(
                ["git", "--git-dir", str(source_dir), "show-ref", "--verify", preferred_ref],
                cwd=self._cache_root,
                check=False,
            )
            if verify.returncode == 0:
                return preferred_ref

            head_ref = self._run(
                ["git", "--git-dir", str(source_dir), "symbolic-ref", "HEAD"],
                cwd=self._cache_root,
                check=False,
            )
            if head_ref.returncode == 0 and head_ref.stdout.strip():
                return head_ref.stdout.strip()

            fallback = self._run(
                [
                    "git",
                    "--git-dir",
                    str(source_dir),
                    "for-each-ref",
                    "--format=%(refname)",
                    "refs/heads",
                ],
                cwd=self._cache_root,
                check=False,
            )
            first_ref = next(
                (line.strip() for line in fallback.stdout.splitlines() if line.strip()),
                "",
            )
            if first_ref:
                return first_ref
            raise RuntimeError(f"No branch refs found in mirror cache {source_dir}")

        fetch_result = self._run(
            ["git", "-C", str(source_dir), "fetch", "--prune", "origin"],
            cwd=source_dir,
            check=False,
        )
        if fetch_result.returncode == 0:
            return f"origin/{default_branch}"
        return default_branch

    def _add_worktree(
        self,
        source_dir: Path,
        worktree_dir: Path,
        ref: str,
        is_bare: bool,
    ) -> None:
        if worktree_dir.exists():
            shutil.rmtree(worktree_dir, ignore_errors=True)

        if is_bare:
            cmd = [
                "git",
                "--git-dir",
                str(source_dir),
                "worktree",
                "add",
                "--detach",
                str(worktree_dir),
                ref,
            ]
            cwd = self._cache_root
        else:
            cmd = [
                "git",
                "-C",
                str(source_dir),
                "worktree",
                "add",
                "--detach",
                str(worktree_dir),
                ref,
            ]
            cwd = source_dir
        self._run(cmd, cwd=cwd)

    def _run_git_worktree_remove(
        self,
        source_dir: Path,
        worktree_dir: Path,
        is_bare: bool,
    ) -> None:
        if is_bare:
            cmd = [
                "git",
                "--git-dir",
                str(source_dir),
                "worktree",
                "remove",
                "--force",
                str(worktree_dir),
            ]
            cwd = self._cache_root
        else:
            cmd = [
                "git",
                "-C",
                str(source_dir),
                "worktree",
                "remove",
                "--force",
                str(worktree_dir),
            ]
            cwd = source_dir
        self._run(cmd, cwd=cwd)

    @staticmethod
    def _cache_name(profile: BotConfigProfile) -> str:
        source = profile.repo_url or profile.repo_dir or profile.bot_id
        source = source.rstrip("/").replace("\\", "/")
        tail = source.split("/")[-1]
        tail = tail[:-4] if tail.endswith(".git") else tail
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", tail).strip("-")
        return normalized or profile.bot_id

    @staticmethod
    def _expand_pattern(root: Path, pattern: str) -> list[str]:
        direct = root / pattern
        paths: list[Path] = []
        if direct.is_file():
            paths.append(direct)
        elif direct.is_dir():
            paths.extend(path for path in direct.rglob("*") if path.is_file())
        else:
            paths.extend(path for path in root.glob(pattern) if path.exists())

        expanded: list[str] = []
        for path in paths:
            if path.is_dir():
                expanded.extend(
                    str(child.relative_to(root)).replace("\\", "/")
                    for child in path.rglob("*")
                    if child.is_file()
                )
            elif path.is_file():
                expanded.append(str(path.relative_to(root)).replace("\\", "/"))
        return expanded

    @staticmethod
    def _run(
        args: list[str],
        cwd: Path,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {' '.join(args)}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}",
            )
        return result
