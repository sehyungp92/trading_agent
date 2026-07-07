"""Side-effect-light deployment preflight for the orchestrator runtime."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from trading_assistant.orchestrator.config import AppConfig
from trading_assistant.orchestrator.data_paths import resolve_data_dirs
from trading_assistant.orchestrator.provider_auth import ProviderAuthChecker
from trading_assistant.orchestrator.runtime_validation import (
    RuntimeConfigError,
    runtime_config_summary,
    validate_auth_config,
    validate_production_runtime_config,
)
from trading_assistant.paths import package_root
from trading_assistant.schemas.agent_preferences import AgentProvider


@dataclass
class PreflightCheck:
    name: str
    status: str
    message: str
    details: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status != "fail"


@dataclass
class PreflightResult:
    dotenv_path: str
    checks: list[PreflightCheck]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.ok,
            "dotenv_path": self.dotenv_path,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "message": check.message,
                    "details": check.details,
                }
                for check in self.checks
            ],
        }


def default_dotenv_path() -> Path:
    return package_root() / ".env"


def _check(name: str, status: str, message: str, **details: object) -> PreflightCheck:
    return PreflightCheck(name=name, status=status, message=message, details=details)


def _writable_path_check(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".orchestrator-preflight-write-test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)


def _selected_provider_names(config: AppConfig) -> list[str]:
    providers = {
        config.agent_default_provider,
        config.daily_agent_provider,
        config.weekly_agent_provider,
        config.monthly_validation_agent_provider,
        config.monthly_model_review_agent_provider,
        config.monthly_verifier_agent_provider,
        config.triage_agent_provider,
    }
    return sorted(provider for provider in providers if provider)


def run_preflight(dotenv_path: str | Path | None = None) -> PreflightResult:
    env_path = Path(dotenv_path) if dotenv_path is not None else default_dotenv_path()
    checks: list[PreflightCheck] = [
        _check(
            "env_source",
            "ok" if env_path.exists() else "warn",
            "resolved orchestrator .env path",
            path=str(env_path),
            exists=env_path.exists(),
        )
    ]

    try:
        config = AppConfig.from_env(dotenv_path=env_path)
    except Exception as exc:
        checks.append(_check("config_load", "fail", str(exc)))
        return PreflightResult(dotenv_path=str(env_path), checks=checks)

    checks.append(_check("config_load", "ok", "AppConfig loaded", **runtime_config_summary(config)))

    try:
        validate_auth_config(config)
        checks.append(_check("auth", "ok", "orchestrator auth policy is safe"))
    except RuntimeConfigError as exc:
        checks.append(_check("auth", "fail", str(exc)))

    try:
        validate_production_runtime_config(config)
        checks.append(
            _check(
                "production_runtime",
                "ok",
                "production runtime requirements satisfied"
                if config.is_production
                else "not in production mode",
            )
        )
    except RuntimeConfigError as exc:
        checks.append(_check("production_runtime", "fail", str(exc)))

    ingest_ok = bool(config.relay_url or config.direct_ingest_only)
    checks.append(
        _check(
            "ingest_mode",
            "ok" if ingest_ok or not config.is_production else "fail",
            "relay polling configured"
            if config.relay_url
            else (
                "direct ingest explicitly selected"
                if config.direct_ingest_only
                else "no relay configured; acceptable outside production"
            ),
            relay_url_configured=bool(config.relay_url),
            direct_ingest_only=config.direct_ingest_only,
        )
    )

    if config.relay_url:
        checks.append(
            _check(
                "relay_auth",
                "ok" if config.relay_api_key or not config.is_production else "fail",
                "relay API key configured"
                if config.relay_api_key
                else "relay API key missing; acceptable outside production only",
            )
        )

    try:
        data_dir = Path(config.data_dir)
        data_dirs = resolve_data_dirs(data_dir)
        for candidate in (data_dir, data_dirs.raw_data_dir, data_dirs.curated_dir):
            _writable_path_check(candidate)
        checks.append(
            _check(
                "data_dirs",
                "ok",
                "data directories are writable",
                data_dir=str(data_dir),
                raw_data_dir=str(data_dirs.raw_data_dir),
                curated_dir=str(data_dirs.curated_dir),
            )
        )
    except Exception as exc:
        checks.append(_check("data_dirs", "fail", str(exc)))

    try:
        import apscheduler  # noqa: F401

        checks.append(_check("scheduler_dependency", "ok", "APScheduler import succeeded"))
    except Exception as exc:
        checks.append(_check("scheduler_dependency", "fail", f"APScheduler import failed: {exc}"))

    providers = _selected_provider_names(config)
    provider_failures: list[str] = []
    provider_details: dict[str, object] = {"selected_providers": providers}
    if providers:
        checker = ProviderAuthChecker(
            claude_command=config.claude_command,
            claude_command_args=config.claude_command_args,
            codex_command=config.codex_command,
            codex_command_args=config.codex_command_args,
            zai_api_key=config.zai_api_key,
            openrouter_api_key=config.openrouter_api_key,
        )
        readiness: dict[str, dict[str, object]] = {}
        for provider_name in providers:
            try:
                provider = AgentProvider(provider_name)
            except ValueError:
                provider_failures.append(f"invalid provider: {provider_name}")
                continue
            status = checker.get_provider_status(provider)
            readiness[provider_name] = status.model_dump(mode="json")
            if not status.available:
                provider_failures.append(f"{provider_name}: {status.reason or 'not available'}")
        provider_details["readiness"] = readiness
    checks.append(
        _check(
            "provider_readiness",
            "fail" if provider_failures else "ok",
            "; ".join(provider_failures) if provider_failures else (
                "explicit providers are ready" if providers else "no explicit provider override selected"
            ),
            **provider_details,
        )
    )

    return PreflightResult(dotenv_path=str(env_path), checks=checks)


def _format_human(result: PreflightResult) -> str:
    lines = [
        f"Orchestrator preflight: {'PASS' if result.ok else 'FAIL'}",
        f".env path: {result.dotenv_path}",
    ]
    for check in result.checks:
        lines.append(f"[{check.status.upper()}] {check.name}: {check.message}")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate orchestrator deployment settings.")
    parser.add_argument("--dotenv", default=None, help="Override orchestrator .env path")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = run_preflight(dotenv_path=args.dotenv)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_format_human(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
