from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch
import os

import pytest
import aiosqlite

os.environ.setdefault("ALLOW_UNAUTHENTICATED_LOCAL", "true")
os.environ.setdefault("BIND_HOST", "127.0.0.1")
os.environ.setdefault("ENVIRONMENT", "development")

# Import shared fixtures so they are available to all test files.
# pytest auto-discovers fixtures defined in conftest.py or imported here.
from tests.fixtures import (  # noqa: F401
    data_dir,
    event_stream,
    memory_dir,
    memory_dir_with_policies,
    mock_event_stream,
    sample_package,
    session_store,
)


@pytest.fixture(autouse=True)
def _block_provider_auth_subprocess():
    """Block CLI preflight subprocess.run calls in provider_auth to avoid 5s timeouts.

    Only intercepts claude/codex CLI preflight checks (--help, auth status).
    All other subprocess.run calls pass through to the real implementation.
    """
    import subprocess as _subprocess_mod

    _real_run = _subprocess_mod.run

    def _guarded_run(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and len(cmd) > 0:
            exe = str(cmd[0]).lower()
            args_lower = [str(a).lower() for a in cmd[1:]]
            # Block only claude/codex CLI preflight checks
            is_cli_preflight = ("claude" in exe or "codex" in exe) and (
                "--help" in args_lower or "auth" in args_lower
            )
            if is_cli_preflight:
                return CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="",
                    stderr="mocked: blocked by test fixture",
                )
        return _real_run(cmd, *args, **kwargs)

    with patch(
        "trading_assistant.orchestrator.provider_auth.subprocess.run",
        side_effect=_guarded_run,
    ):
        yield


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Provide a temporary SQLite database path."""
    return tmp_path / "test.db"


@pytest.fixture
async def tmp_db(tmp_db_path: Path) -> aiosqlite.Connection:
    """Provide an initialized temporary SQLite connection."""
    async with aiosqlite.connect(tmp_db_path) as db:
        yield db
