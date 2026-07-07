from __future__ import annotations

from pathlib import Path

from trading_assistant.orchestrator.preflight import run_preflight


_ENV_KEYS = [
    "ALLOW_UNAUTHENTICATED_LOCAL",
    "BIND_HOST",
    "UVICORN_HOST",
    "ENVIRONMENT",
    "APP_ENV",
    "DEPLOYMENT_ENV",
    "ORCHESTRATOR_API_KEY",
    "BOT_IDS",
    "RELAY_URL",
    "RELAY_API_KEY",
    "DIRECT_INGEST_ONLY",
    "DATA_DIR",
]


def _clear_env(monkeypatch, tmp_path: Path) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))


def _statuses(result):
    return {check.name: check.status for check in result.checks}


def test_preflight_fails_missing_auth(monkeypatch, tmp_path: Path):
    _clear_env(monkeypatch, tmp_path)

    result = run_preflight(dotenv_path=tmp_path / ".env")

    assert result.ok is False
    assert _statuses(result)["auth"] == "fail"


def test_preflight_accepts_explicit_loopback_local_dev(monkeypatch, tmp_path: Path):
    _clear_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED_LOCAL", "true")
    monkeypatch.setenv("BIND_HOST", "127.0.0.1")

    result = run_preflight(dotenv_path=tmp_path / ".env")

    assert result.ok is True
    assert _statuses(result)["auth"] == "ok"


def test_preflight_fails_production_without_bots(monkeypatch, tmp_path: Path):
    _clear_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ORCHESTRATOR_API_KEY", "secret")
    monkeypatch.setenv("DIRECT_INGEST_ONLY", "true")

    result = run_preflight(dotenv_path=tmp_path / ".env")

    assert result.ok is False
    assert _statuses(result)["production_runtime"] == "fail"


def test_preflight_fails_production_relay_without_api_key(monkeypatch, tmp_path: Path):
    _clear_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ORCHESTRATOR_API_KEY", "secret")
    monkeypatch.setenv("BOT_IDS", "bot1")
    monkeypatch.setenv("RELAY_URL", "https://relay.example")

    result = run_preflight(dotenv_path=tmp_path / ".env")

    assert result.ok is False
    assert _statuses(result)["production_runtime"] == "fail"
