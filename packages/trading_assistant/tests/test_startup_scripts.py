"""Tests for Windows startup helper scripts."""

from __future__ import annotations

import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


START_COMMON = Path(__file__).resolve().parents[1] / "scripts" / "start-common.ps1"
START_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "start.ps1"
START_RELAY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "start-relay.ps1"
INSTALL_STARTUP_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install-startup.ps1"


def _run_powershell(command: str) -> str:
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_start_script_passes_uvicorn_host_to_bind_env():
    source = START_SCRIPT.read_text(encoding="utf-8")
    assert "$env:BIND_HOST = $UvicornHost" in source
    assert "$env:UVICORN_HOST = $UvicornHost" in source
    assert '"--host", $UvicornHost' in source
    assert 'Import-AssistantEnvFile -EnvFile (Join-Path $ProjectRoot ".env")' in source


def test_relay_start_script_runs_local_relay_from_monorepo():
    source = START_RELAY_SCRIPT.read_text(encoding="utf-8")
    assert 'Import-AssistantEnvFile -EnvFile (Join-Path $ProjectRoot ".env")' in source
    assert '"trading_assistant.relay_ingress.app:app"' in source
    assert '"--app-dir", $RelayAppDir' in source
    assert '$env:RELAY_DB_PATH' in source
    assert 'RELAY_NETWORK_MODE' in source
    assert 'RELAY_HOST must not be loopback' in source
    assert '"prod", "production"' in source
    assert 'trading\\ibkr_trading' not in source


def test_install_startup_registers_relay_and_orchestrator_tasks():
    source = INSTALL_STARTUP_SCRIPT.read_text(encoding="utf-8")
    assert "TradingAssistantRelayAutoStart" in source
    assert "TradingAssistantAutoStart" in source
    assert "start-relay.ps1" in source
    assert "start.ps1" in source


def test_duplicate_start_detection_requires_ready_json_not_startup_grace():
    source = START_COMMON.read_text(encoding="utf-8")
    body = source.split("function Test-OrchestratorAlreadyRunning", 1)[1].split(
        "function ",
        1,
    )[0]

    assert "StartupGraceSeconds" not in body
    assert "StartTime" not in body
    assert "Test-OrchestratorHealthy -Url $HealthUrl" in body


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only startup script tests")
def test_resolve_interpreter_prefers_dotvenv(tmp_path: Path):
    project_root = tmp_path
    dot_venv = project_root / ".venv" / "Scripts"
    venv = project_root / "venv" / "Scripts"
    dot_venv.mkdir(parents=True)
    venv.mkdir(parents=True)
    (dot_venv / "pythonw.exe").write_bytes(b"")
    (venv / "pythonw.exe").write_bytes(b"")

    output = _run_powershell(
        f". '{START_COMMON}'; Resolve-OrchestratorPythonw -ProjectRoot '{project_root}'"
    )

    assert output.endswith(str(dot_venv / "pythonw.exe"))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only startup script tests")
def test_resolve_interpreter_falls_back_to_venv(tmp_path: Path):
    project_root = tmp_path
    venv = project_root / "venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "pythonw.exe").write_bytes(b"")

    output = _run_powershell(
        f". '{START_COMMON}'; Resolve-OrchestratorPythonw -ProjectRoot '{project_root}'"
    )

    assert output.endswith(str(venv / "pythonw.exe"))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only startup script tests")
def test_healthy_existing_process_prevents_duplicate_start(tmp_path: Path):
    pid_file = tmp_path / "trading_assistant.orchestrator.pid"

    class _HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/ready":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):  # noqa: A003
            return

    server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        output = _run_powershell(
            ". '{0}'; "
            "Set-Content -Path '{1}' -Value $PID -Encoding ASCII; "
            "Test-OrchestratorAlreadyRunning -PidFile '{1}' -HealthUrl 'http://127.0.0.1:{2}/ready'".format(
                START_COMMON,
                pid_file,
                server.server_port,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert output == "True"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only startup script tests")
def test_degraded_readiness_is_not_healthy(tmp_path: Path):
    class _HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/ready":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"degraded"}')
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):  # noqa: A003
            return

    server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        output = _run_powershell(
            ". '{0}'; Test-OrchestratorHealthy -Url 'http://127.0.0.1:{1}/ready'".format(
                START_COMMON,
                server.server_port,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert output == "False"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only startup script tests")
def test_supervisor_lock_allows_only_one_owner(tmp_path: Path):
    lock_file = tmp_path / "trading_assistant.orchestrator.supervisor.lock"

    output = _run_powershell(
        ". '{0}'; "
        "$first = Enter-OrchestratorSupervisorLock -LockFile '{1}'; "
        "$second = Enter-OrchestratorSupervisorLock -LockFile '{1}'; "
        "try {{ Write-Output ([bool]$first); Write-Output ([bool]$second) }} "
        "finally {{ "
        "if ($second) {{ Exit-OrchestratorSupervisorLock -LockHandle $second -LockFile '{1}' }}; "
        "if ($first) {{ Exit-OrchestratorSupervisorLock -LockHandle $first -LockFile '{1}' }} "
        "}}".format(
            START_COMMON,
            lock_file,
        )
    )

    assert output.splitlines() == ["True", "False"]
