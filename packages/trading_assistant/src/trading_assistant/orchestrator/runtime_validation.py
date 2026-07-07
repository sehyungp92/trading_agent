"""Runtime configuration validation shared by app startup and preflight."""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_assistant.orchestrator.config import AppConfig


class RuntimeConfigError(RuntimeError):
    """Raised when runtime configuration is unsafe for the selected mode."""


def is_loopback_bind_host(host: str) -> bool:
    value = normalize_bind_host(host)
    if value == "loopback":
        return True
    if value == "public":
        return False
    if not value:
        return False
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def normalize_bind_host(host: str) -> str:
    """Normalize host declarations for bind-host consistency checks."""
    value = (host or "").strip().lower()
    if value in {"localhost"}:
        return "loopback"
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value:
        return ""
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return value
    if ip.is_unspecified:
        return "public"
    if ip.is_loopback:
        return "loopback"
    return str(ip)


def validate_bind_host_consistency(config: AppConfig) -> None:
    bind_host = getattr(config, "configured_bind_host", "") or getattr(config, "bind_host", "") or ""
    uvicorn_host = getattr(config, "uvicorn_host", "") or ""
    if not bind_host or not uvicorn_host:
        return

    bind_normalized = normalize_bind_host(bind_host)
    uvicorn_normalized = normalize_bind_host(uvicorn_host)
    if bind_normalized != uvicorn_normalized:
        raise RuntimeConfigError(
            "BIND_HOST and UVICORN_HOST disagree. Refusing to start because "
            "the launcher bind host must match auth policy. "
            f"BIND_HOST={bind_host!r}, UVICORN_HOST={uvicorn_host!r}."
        )


def validate_auth_config(config: AppConfig) -> None:
    """Fail closed unless local unauthenticated mode is explicit and loopback."""
    validate_bind_host_consistency(config)
    if config.orchestrator_api_key:
        return
    if (
        config.allow_unauthenticated_local
        and config.bind_host_explicit
        and is_loopback_bind_host(config.bind_host)
        and not config.is_production
    ):
        return

    hint = (
        "Set ORCHESTRATOR_API_KEY, or for local-only development set "
        "ALLOW_UNAUTHENTICATED_LOCAL=true and explicitly set BIND_HOST or "
        "UVICORN_HOST to 127.0.0.1/localhost."
    )
    raise RuntimeConfigError(
        "Orchestrator refusing to start without ORCHESTRATOR_API_KEY. "
        f"{hint} Current BIND_HOST={config.bind_host!r}, "
        f"BIND_HOST_EXPLICIT={config.bind_host_explicit!r}, "
        f"ENVIRONMENT={config.environment!r}."
    )


def validate_production_runtime_config(config: AppConfig) -> None:
    """Require production settings that prevent silent no-work deployments."""
    if not config.is_production:
        return

    errors: list[str] = []
    if not config.bot_ids:
        errors.append("BOT_IDS must include at least one bot in production")
    if not config.relay_url and not config.direct_ingest_only:
        errors.append(
            "production requires RELAY_URL or DIRECT_INGEST_ONLY=true for an "
            "explicit direct-ingest deployment"
        )
    if config.relay_url and not config.relay_api_key:
        errors.append("RELAY_API_KEY is required when RELAY_URL is set in production")

    if errors:
        raise RuntimeConfigError("; ".join(errors))


def runtime_config_summary(config: AppConfig) -> dict[str, object]:
    return {
        "environment": config.environment,
        "auth_enabled": bool(config.orchestrator_api_key),
        "allow_unauthenticated_local": config.allow_unauthenticated_local,
        "bind_host": config.bind_host,
        "bind_host_explicit": config.bind_host_explicit,
        "uvicorn_host": getattr(config, "uvicorn_host", ""),
        "configured_bind_host": getattr(config, "configured_bind_host", ""),
        "configured_bot_count": len(config.bot_ids),
        "relay_configured": bool(config.relay_url),
        "relay_auth_configured": bool(config.relay_api_key),
        "direct_ingest_only": config.direct_ingest_only,
    }
