from __future__ import annotations

import subprocess
from pathlib import Path

from trading_assistant.schemas.bot_profile import BotConfigProfile
from trading_assistant.skills.repo_workspace import RepoWorkspaceManager


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo: Path, file_text: str = "value: 1\n") -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "config.yaml").write_text(file_text, encoding="utf-8")
    _git(repo, "add", "config.yaml")
    _git(repo, "commit", "-m", "init")


def test_prepare_workspace_uses_cached_mirror_and_cleans_up(tmp_path: Path):
    origin = tmp_path / "origin"
    _init_repo(origin)

    manager = RepoWorkspaceManager(
        cache_root=tmp_path / "cache",
        task_root=tmp_path / "runs" / "repo_tasks",
    )
    profile = BotConfigProfile(
        bot_id="bot1",
        repo_url=str(origin),
        default_branch="main",
    )

    context = manager.prepare_workspace(profile, "task-1")
    worktree = Path(context.worktree_dir)
    assert worktree.exists()
    assert (worktree / "config.yaml").read_text(encoding="utf-8") == "value: 1\n"
    assert Path(context.repo_cache_dir).exists()

    (worktree / "config.yaml").write_text("value: 2\n", encoding="utf-8")
    assert (origin / "config.yaml").read_text(encoding="utf-8") == "value: 1\n"

    manager.cleanup(context)
    assert not worktree.exists()
    assert (Path(context.artifact_dir) / "repo_task.json").exists()


def test_prepare_workspace_fetches_new_commits(tmp_path: Path):
    origin = tmp_path / "origin"
    _init_repo(origin, "value: 1\n")

    manager = RepoWorkspaceManager(
        cache_root=tmp_path / "cache",
        task_root=tmp_path / "runs" / "repo_tasks",
    )
    profile = BotConfigProfile(
        bot_id="bot1",
        repo_url=str(origin),
        default_branch="main",
    )

    first = manager.prepare_workspace(profile, "task-1")
    manager.cleanup(first)

    (origin / "config.yaml").write_text("value: 3\n", encoding="utf-8")
    _git(origin, "add", "config.yaml")
    _git(origin, "commit", "-m", "update")

    second = manager.prepare_workspace(profile, "task-2")
    assert (Path(second.worktree_dir) / "config.yaml").read_text(encoding="utf-8") == "value: 3\n"
    manager.cleanup(second)


def test_collect_context_files_uses_structural_context_paths(tmp_path: Path):
    origin = tmp_path / "origin"
    _init_repo(origin, "value: 1\n")
    (origin / "strategies").mkdir()
    (origin / "strategies" / "alpha.py").write_text("ALPHA = 1\n", encoding="utf-8")
    (origin / "tests").mkdir()
    (origin / "tests" / "test_alpha.py").write_text("def test_alpha():\n    assert True\n", encoding="utf-8")
    _git(origin, "add", "strategies/alpha.py", "tests/test_alpha.py")
    _git(origin, "commit", "-m", "add context files")

    manager = RepoWorkspaceManager(
        cache_root=tmp_path / "cache",
        task_root=tmp_path / "runs" / "repo_tasks",
    )
    profile = BotConfigProfile(
        bot_id="bot1",
        repo_url=str(origin),
        default_branch="main",
        structural_context_paths=["strategies/*"],
    )

    context = manager.prepare_workspace(profile, "task-ctx")
    files = manager.collect_context_files(
        profile,
        context.worktree_dir,
        preferred_paths=["tests/test_alpha.py"],
    )

    assert files[0] == "tests/test_alpha.py"
    assert "strategies/alpha.py" in files
    manager.cleanup(context)
