"""Tests for PRBuilder subprocess timeout (Task 6)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading_assistant.skills.github_pr import PRBuilder


@pytest.fixture
def builder() -> PRBuilder:
    return PRBuilder(dry_run=True)


def _make_mock_process(
    *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0, hang: bool = False
) -> AsyncMock:
    """Create a mock async subprocess process."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.wait = AsyncMock()

    async def _communicate():
        if hang:
            await asyncio.sleep(3600)  # will be cancelled by timeout
        return stdout, stderr

    proc.communicate = _communicate
    return proc


class TestSubprocessTimeout:
    """6a: Timeout on _run_cmd."""

    async def test_timeout_kills_process(self, tmp_path):
        builder = PRBuilder(dry_run=False)
        proc = _make_mock_process(hang=True)

        with patch("trading_assistant.skills.github_pr.asyncio.create_subprocess_exec", return_value=proc):
            code, stdout, stderr = await builder._run_cmd(
                ["python", "-c", "import time; time.sleep(10)"],
                cwd=tmp_path,
                timeout_seconds=0,
            )
        assert code == 1
        assert "timed out" in stderr
        proc.kill.assert_called_once()

    async def test_normal_command_succeeds(self, tmp_path):
        builder = PRBuilder(dry_run=False)
        proc = _make_mock_process(stdout=b"hello\n", returncode=0)

        with patch("trading_assistant.skills.github_pr.asyncio.create_subprocess_exec", return_value=proc):
            code, stdout, stderr = await builder._run_cmd(
                ["python", "-c", "print('hello')"],
                cwd=tmp_path,
                timeout_seconds=10,
            )
        assert code == 0
        assert "hello" in stdout


class TestPublicAPI:
    """6b: Public run_gh_command API."""

    async def test_run_gh_command_delegates_to_run_cmd(self, tmp_path):
        builder = PRBuilder(dry_run=False)
        proc = _make_mock_process(stdout=b"via_public\n", returncode=0)

        with patch("trading_assistant.skills.github_pr.asyncio.create_subprocess_exec", return_value=proc):
            code, stdout, stderr = await builder.run_gh_command(
                ["python", "-c", "print('via_public')"],
                cwd=tmp_path,
                timeout_seconds=10,
            )
        assert code == 0
        assert "via_public" in stdout

    async def test_run_gh_command_timeout(self, tmp_path):
        builder = PRBuilder(dry_run=False)
        proc = _make_mock_process(hang=True)

        with patch("trading_assistant.skills.github_pr.asyncio.create_subprocess_exec", return_value=proc):
            code, _, stderr = await builder.run_gh_command(
                ["python", "-c", "import time; time.sleep(10)"],
                cwd=tmp_path,
                timeout_seconds=0,
            )
        assert code == 1
        assert "timed out" in stderr
