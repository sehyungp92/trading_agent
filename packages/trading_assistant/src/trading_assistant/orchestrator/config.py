"""Application configuration reads from environment variables and local .env."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from json import JSONDecodeError
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, model_validator

from trading_assistant.paths import memory_root, package_root
from trading_assistant.schemas.bot_config import BotConfig
from trading_assistant.schemas.strategy_profile import StrategyRegistry

logger = logging.getLogger(__name__)


def _default_dotenv_path() -> Path:
    return package_root() / ".env"


def _load_dotenv_defaults(dotenv_path: str | Path | None = None) -> dict[str, str]:
    """Load .env pairs with normal precedence semantics.

    Values read from disk are defaults only; existing process environment
    variables still win because callers overlay ``os.environ`` on top.
    """
    path = Path(dotenv_path) if dotenv_path is not None else _default_dotenv_path()
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("Could not read .env from %s", path)
        return values

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            logger.warning("Ignoring malformed .env line in %s: %r", path, raw_line)
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _parse_command_args(raw: str, env_name: str) -> list[str]:
    """Parse a JSON-array command-args env var into a list of strings."""
    if not raw.strip():
        return []
    try:
        import json

        parsed = json.loads(raw)
    except JSONDecodeError:
        logger.warning("Ignoring invalid %s JSON: %r", env_name, raw)
        return []

    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        logger.warning("Ignoring %s because it must be a JSON array of strings", env_name)
        return []
    return parsed


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()] if raw else []


def _parse_key_value_csv(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in _parse_csv(raw):
        separator = "=" if "=" in item else ":"
        if separator not in item:
            logger.warning("Ignoring malformed key/value config entry: %r", item)
            continue
        key, value = (part.strip() for part in item.split(separator, 1))
        if key and value:
            values[key] = value
    return values


def _parse_bot_timezones(raw: str, bot_ids: list[str]) -> dict[str, BotConfig]:
    """Parse BOT_TIMEZONES env var into BotConfig dict.

    Format: ``bot_id:Asia/Seoul,bot_id2:US/Eastern``.
    Bots in ``bot_ids`` but not in the mapping get default UTC config.
    """
    configs: dict[str, BotConfig] = {}

    if raw:
        for pair in raw.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if ":" not in pair:
                logger.warning("Ignoring malformed BOT_TIMEZONES entry: %r", pair)
                continue
            parts = pair.split(":", 2)
            bot_id = parts[0].strip()
            tz = parts[1].strip() if len(parts) > 1 else ""
            close_time = parts[2].strip() if len(parts) > 2 else ""
            if not bot_id or not tz:
                continue
            try:
                kwargs: dict = {"bot_id": bot_id, "timezone": tz}
                if close_time:
                    kwargs["market_close_local"] = close_time
                configs[bot_id] = BotConfig(**kwargs)
            except Exception as exc:
                logger.warning("Invalid timezone config for %s: %s", bot_id, exc)

    for bid in bot_ids:
        if bid not in configs:
            configs[bid] = BotConfig(bot_id=bid)

    return configs


class AppConfig(BaseModel):
    """Configuration loaded from environment variables."""

    bot_ids: list[str] = []
    bot_configs: dict[str, BotConfig] = {}
    claude_command: str = "claude"
    claude_command_args: list[str] = []
    codex_command: str = "codex"
    codex_command_args: list[str] = []
    zai_api_key: str = ""
    openrouter_api_key: str = ""
    agent_default_provider: str = ""
    agent_default_model: str = ""
    daily_agent_provider: str = ""
    daily_agent_model: str = ""
    weekly_agent_provider: str = ""
    weekly_agent_model: str = ""
    monthly_validation_agent_provider: str = ""
    monthly_validation_agent_model: str = ""
    monthly_model_review_agent_provider: str = ""
    monthly_model_review_agent_model: str = ""
    monthly_verifier_agent_provider: str = ""
    monthly_verifier_agent_model: str = ""
    triage_agent_provider: str = ""
    triage_agent_model: str = ""
    relay_url: str = ""
    relay_hmac_secret: str = ""
    relay_api_key: str = ""
    orchestrator_api_key: str = ""
    allow_unauthenticated_local: bool = False
    # Effective launcher bind host used for auth policy. When UVICORN_HOST is
    # present it is the process-level host, while configured_bind_host keeps
    # the raw BIND_HOST policy value for mismatch validation.
    bind_host: str = "127.0.0.1"
    bind_host_explicit: bool = True
    uvicorn_host: str = ""
    configured_bind_host: str = ""
    environment: str = "development"
    direct_ingest_only: bool = False
    # Worker throughput controls (P1-2). The default 200/30s lets the worker
    # drain ~tens of thousands of events per minute when the queue is hot.
    worker_batch_size: int = 200
    worker_drain_seconds: float = 30.0
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_bot_token: str = ""
    discord_channel_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    email_from: str = ""
    email_to: str = ""
    data_dir: str = "data"
    memory_dir: str = ""
    log_level: str = "INFO"
    bot_config_dir: str = "data/bot_configs"
    bot_repo_dir: str = "."
    bot_repo_cache_dir: str = "runs/repo_cache"
    github_token: str = ""
    strategy_registry: StrategyRegistry = StrategyRegistry()
    autonomous_enabled: bool = False
    adaptive_thresholds_enabled: bool = False
    deployment_monitoring_enabled: bool = False
    ab_testing_enabled: bool = False
    learning_review_mode: Literal["disabled", "deterministic", "llm_review"] = "deterministic"
    learning_review_disabled_workflows: list[str] = []
    market_data_root: str = ""
    backtest_repo_path: str = ""
    backtest_artifact_root: str = ""
    monthly_validation_enabled: bool = False
    monthly_validation_mode: Literal["disabled", "shadow", "approval_gated"] = "disabled"
    monthly_approval_scope_allowlist: list[str] = []
    monthly_approval_scope_map: dict[str, str] = {}
    monthly_deployment_metadata_install_report_paths: list[str] = []
    monthly_operational_evidence_path: str = ""
    monthly_relay_ingest_evidence_path: str = ""
    monthly_vps_host_id: str = ""
    monthly_assistant_host_id: str = "local"
    monthly_validation_day_of_month: int = 2
    monthly_validation_hour: int = 3
    monthly_validation_minute: int = 0
    monthly_optimizer_sequence_enabled: bool = True
    monthly_backtest_command: list[str] = []
    monthly_workflow_contract_path: str = ""
    monthly_workflow_contract_version: str = ""
    monthly_strategy_plugin_contract_path: str = ""
    market_data_sync_day_of_month: int = 1
    market_data_sync_hour: int = 1
    market_data_sync_minute: int = 0
    backtest_command_timeout_seconds: int = 3600
    backtest_max_parallel_strategies: int = 1
    market_data_required_coverage_ratio: float = 0.95
    telemetry_required_lineage_ratio: float = 0.95

    @model_validator(mode="after")
    def _normalize_monthly_validation_flag(self) -> AppConfig:
        if self.monthly_validation_mode != "disabled":
            self.monthly_validation_enabled = True
        self.environment = (self.environment or "development").strip().lower()
        return self

    @property
    def is_production(self) -> bool:
        return self.environment in {"prod", "production"}

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> AppConfig:
        """Build config from process env layered over local .env defaults."""
        env = _load_dotenv_defaults(dotenv_path)
        env.update(os.environ)

        bot_ids_raw = env.get("BOT_IDS", "")
        bot_ids = [b.strip() for b in bot_ids_raw.split(",") if b.strip()] if bot_ids_raw else []

        bot_configs = _parse_bot_timezones(env.get("BOT_TIMEZONES", ""), bot_ids)

        from trading_assistant.orchestrator.strategy_registry_loader import load_strategy_registry

        strategy_profiles_path = env.get("STRATEGY_PROFILES_PATH", "")
        strategy_registry = load_strategy_registry(
            Path(strategy_profiles_path) if strategy_profiles_path else None
        )

        bind_host_env = env.get("BIND_HOST", "")
        uvicorn_host_env = env.get("UVICORN_HOST", "")
        bind_host_explicit = bool(bind_host_env or uvicorn_host_env)
        environment = (
            env.get("ENVIRONMENT")
            or env.get("APP_ENV")
            or env.get("DEPLOYMENT_ENV")
            or "development"
        )

        config = cls(
            bot_ids=bot_ids,
            bot_configs=bot_configs,
            strategy_registry=strategy_registry,
            claude_command=env.get("CLAUDE_COMMAND", "claude"),
            claude_command_args=_parse_command_args(
                env.get("CLAUDE_COMMAND_ARGS", ""),
                "CLAUDE_COMMAND_ARGS",
            ),
            codex_command=env.get("CODEX_COMMAND", "codex"),
            codex_command_args=_parse_command_args(
                env.get("CODEX_COMMAND_ARGS", ""),
                "CODEX_COMMAND_ARGS",
            ),
            zai_api_key=env.get("ZAI_API_KEY", ""),
            openrouter_api_key=env.get("OPENROUTER_API_KEY", ""),
            agent_default_provider=env.get("AGENT_PROVIDER", ""),
            agent_default_model=env.get("AGENT_MODEL", ""),
            daily_agent_provider=env.get("DAILY_AGENT_PROVIDER", ""),
            daily_agent_model=env.get("DAILY_AGENT_MODEL", ""),
            weekly_agent_provider=env.get("WEEKLY_AGENT_PROVIDER", ""),
            weekly_agent_model=env.get("WEEKLY_AGENT_MODEL", ""),
            monthly_validation_agent_provider=env.get("MONTHLY_VALIDATION_AGENT_PROVIDER", ""),
            monthly_validation_agent_model=env.get("MONTHLY_VALIDATION_AGENT_MODEL", ""),
            monthly_model_review_agent_provider=env.get("MONTHLY_MODEL_REVIEW_AGENT_PROVIDER", ""),
            monthly_model_review_agent_model=env.get("MONTHLY_MODEL_REVIEW_AGENT_MODEL", ""),
            monthly_verifier_agent_provider=env.get("MONTHLY_VERIFIER_AGENT_PROVIDER", ""),
            monthly_verifier_agent_model=env.get("MONTHLY_VERIFIER_AGENT_MODEL", ""),
            triage_agent_provider=env.get("TRIAGE_AGENT_PROVIDER", ""),
            triage_agent_model=env.get("TRIAGE_AGENT_MODEL", ""),
            relay_url=env.get("RELAY_URL", ""),
            relay_hmac_secret=env.get("RELAY_HMAC_SECRET", ""),
            relay_api_key=env.get("RELAY_API_KEY", ""),
            orchestrator_api_key=env.get("ORCHESTRATOR_API_KEY", ""),
            allow_unauthenticated_local=_parse_bool(
                env.get("ALLOW_UNAUTHENTICATED_LOCAL", "false"),
            ),
            bind_host=uvicorn_host_env or bind_host_env or "127.0.0.1",
            bind_host_explicit=bind_host_explicit,
            uvicorn_host=uvicorn_host_env,
            configured_bind_host=bind_host_env,
            environment=environment,
            direct_ingest_only=_parse_bool(env.get("DIRECT_INGEST_ONLY", "false")),
            worker_batch_size=int(env.get("WORKER_BATCH_SIZE", "200")),
            worker_drain_seconds=float(env.get("WORKER_DRAIN_SECONDS", "30")),
            telegram_bot_token=env.get("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=env.get("TELEGRAM_CHAT_ID", ""),
            discord_bot_token=env.get("DISCORD_BOT_TOKEN", ""),
            discord_channel_id=env.get("DISCORD_CHANNEL_ID", ""),
            smtp_host=env.get("SMTP_HOST", ""),
            smtp_port=int(env.get("SMTP_PORT", "587")),
            smtp_user=env.get("SMTP_USER", ""),
            smtp_pass=env.get("SMTP_PASS", ""),
            email_from=env.get("EMAIL_FROM", ""),
            email_to=env.get("EMAIL_TO", ""),
            data_dir=env.get("DATA_DIR", "data"),
            memory_dir=env.get("MEMORY_DIR", ""),
            log_level=env.get("LOG_LEVEL", "INFO"),
            bot_config_dir=env.get("BOT_CONFIG_DIR", "data/bot_configs"),
            bot_repo_dir=env.get("BOT_REPO_DIR", "."),
            bot_repo_cache_dir=env.get("BOT_REPO_CACHE_DIR", "runs/repo_cache"),
            github_token=env.get("GITHUB_TOKEN", ""),
            autonomous_enabled=env.get("AUTONOMOUS_ENABLED", "false").lower() in ("true", "1", "yes"),
            adaptive_thresholds_enabled=env.get("ADAPTIVE_THRESHOLDS_ENABLED", "false").lower() in ("true", "1", "yes"),
            deployment_monitoring_enabled=env.get("DEPLOYMENT_MONITORING_ENABLED", "false").lower() in ("true", "1", "yes"),
            ab_testing_enabled=env.get("AB_TESTING_ENABLED", "false").lower() in ("true", "1", "yes"),
            learning_review_mode=env.get("LEARNING_REVIEW_MODE", "deterministic"),
            learning_review_disabled_workflows=_parse_csv(
                env.get("LEARNING_REVIEW_DISABLED_WORKFLOWS", ""),
            ),
            market_data_root=env.get("MARKET_DATA_ROOT", ""),
            backtest_repo_path=env.get("BACKTEST_REPO_PATH", ""),
            backtest_artifact_root=env.get("BACKTEST_ARTIFACT_ROOT", ""),
            monthly_validation_enabled=_parse_bool(env.get("MONTHLY_VALIDATION_ENABLED", "false")),
            monthly_validation_mode=env.get("MONTHLY_VALIDATION_MODE", "disabled"),
            monthly_approval_scope_allowlist=_parse_csv(
                env.get("MONTHLY_APPROVAL_SCOPE_ALLOWLIST", ""),
            ),
            monthly_approval_scope_map=_parse_key_value_csv(
                env.get("MONTHLY_APPROVAL_SCOPE_MAP", ""),
            ),
            monthly_deployment_metadata_install_report_paths=_parse_csv(
                env.get("MONTHLY_DEPLOYMENT_METADATA_INSTALL_REPORTS", ""),
            ),
            monthly_operational_evidence_path=env.get("MONTHLY_OPERATIONAL_EVIDENCE_PATH", ""),
            monthly_relay_ingest_evidence_path=env.get("MONTHLY_RELAY_INGEST_EVIDENCE_PATH", ""),
            monthly_vps_host_id=env.get("MONTHLY_VPS_HOST_ID", ""),
            monthly_assistant_host_id=env.get("MONTHLY_ASSISTANT_HOST_ID", "local"),
            monthly_validation_day_of_month=int(env.get("MONTHLY_VALIDATION_DAY_OF_MONTH", "2")),
            monthly_validation_hour=int(env.get("MONTHLY_VALIDATION_HOUR", "3")),
            monthly_validation_minute=int(env.get("MONTHLY_VALIDATION_MINUTE", "0")),
            monthly_optimizer_sequence_enabled=_parse_bool(
                env.get("MONTHLY_OPTIMIZER_SEQUENCE_ENABLED", "true"),
            ),
            monthly_backtest_command=_parse_command_args(
                env.get("MONTHLY_BACKTEST_COMMAND", ""),
                "MONTHLY_BACKTEST_COMMAND",
            ),
            monthly_workflow_contract_path=env.get("MONTHLY_WORKFLOW_CONTRACT_PATH", ""),
            monthly_workflow_contract_version=env.get("MONTHLY_WORKFLOW_CONTRACT_VERSION", ""),
            monthly_strategy_plugin_contract_path=env.get("MONTHLY_STRATEGY_PLUGIN_CONTRACT_PATH", ""),
            market_data_sync_day_of_month=int(env.get("MARKET_DATA_SYNC_DAY_OF_MONTH", "1")),
            market_data_sync_hour=int(env.get("MARKET_DATA_SYNC_HOUR", "1")),
            market_data_sync_minute=int(env.get("MARKET_DATA_SYNC_MINUTE", "0")),
            backtest_command_timeout_seconds=int(env.get("BACKTEST_COMMAND_TIMEOUT_SECONDS", "3600")),
            backtest_max_parallel_strategies=int(env.get("BACKTEST_MAX_PARALLEL_STRATEGIES", "1")),
            market_data_required_coverage_ratio=float(env.get("MARKET_DATA_REQUIRED_COVERAGE_RATIO", "0.95")),
            telemetry_required_lineage_ratio=float(env.get("TELEMETRY_REQUIRED_LINEAGE_RATIO", "0.95")),
        )

        config._validate_provider_secrets()
        config._validate_monthly_paths()
        return config

    # Providers that require an API key in the environment to function. Other
    # providers (e.g. claude_max, codex_pro) authenticate via the runtime CLI.
    # ClassVar so Pydantic doesn't wrap it as a private attribute / FieldInfo.
    _PROVIDER_REQUIRED_SECRET: ClassVar[dict[str, str]] = {
        "zai_coding_plan": "zai_api_key",
        "openrouter": "openrouter_api_key",
    }

    def _validate_provider_secrets(self) -> None:
        """Fail fast at startup if a selected provider lacks its required secret."""
        selected: list[tuple[str, str]] = []
        for label, name in (
            ("AGENT_PROVIDER", self.agent_default_provider),
            ("DAILY_AGENT_PROVIDER", self.daily_agent_provider),
            ("WEEKLY_AGENT_PROVIDER", self.weekly_agent_provider),
            ("MONTHLY_VALIDATION_AGENT_PROVIDER", self.monthly_validation_agent_provider),
            ("MONTHLY_MODEL_REVIEW_AGENT_PROVIDER", self.monthly_model_review_agent_provider),
            ("MONTHLY_VERIFIER_AGENT_PROVIDER", self.monthly_verifier_agent_provider),
            ("TRIAGE_AGENT_PROVIDER", self.triage_agent_provider),
        ):
            if name:
                selected.append((label, name))

        missing: list[str] = []
        seen: set[str] = set()
        for label, provider in selected:
            attr = self._PROVIDER_REQUIRED_SECRET.get(provider)
            if attr is None:
                continue
            if not getattr(self, attr, ""):
                env_name = attr.upper()
                if env_name in seen:
                    continue
                seen.add(env_name)
                missing.append(
                    f"{label}={provider!r} requires ${env_name} (currently empty)"
                )
        if missing:
            raise ValueError(
                "Provider secret validation failed:\n  - "
                + "\n  - ".join(missing)
            )

    def _validate_monthly_paths(self) -> None:
        if self.monthly_validation_mode != "disabled":
            self.monthly_validation_enabled = True
        if not self.monthly_validation_enabled:
            return
        if not self.backtest_repo_path.strip():
            logger.warning(
                "Disabling monthly validation: BACKTEST_REPO_PATH is empty",
            )
            self.monthly_validation_enabled = False
            self.monthly_validation_mode = "disabled"
            return
        if not Path(self.backtest_repo_path).is_dir():
            logger.warning(
                "Disabling monthly validation: BACKTEST_REPO_PATH is not a directory: %s",
                self.backtest_repo_path,
            )
            self.monthly_validation_enabled = False
            self.monthly_validation_mode = "disabled"
        if self.market_data_root and not Path(self.market_data_root).exists():
            logger.warning(
                "Monthly validation can run only in diagnostics mode until MARKET_DATA_ROOT exists: %s",
                self.market_data_root,
            )


def resolve_runtime_memory_dir(
    config: AppConfig,
    *,
    db_dir: str | Path | None = None,
    db_dir_explicit: bool = False,
) -> Path:
    """Resolve the memory root used by runtime context and projections.

    Default startup should read and write the checked package memory so loop
    contracts, work logs, and performance ledgers validated by workspace checks
    are the same artifacts agents see. Explicit app-factory ``db_dir`` overrides
    keep tests and isolated harnesses self-contained unless ``MEMORY_DIR`` is set.
    """

    if config.memory_dir.strip():
        return _package_relative_path(config.memory_dir)
    if db_dir_explicit and db_dir is not None:
        return Path(db_dir) / "memory"
    return memory_root()


def runtime_scheduler_config_from_app_config(config: AppConfig):
    from trading_assistant.orchestrator.scheduler import SchedulerConfig

    return SchedulerConfig(
        market_data_sync_day_of_month=config.market_data_sync_day_of_month,
        market_data_sync_hour=config.market_data_sync_hour,
        market_data_sync_minute=config.market_data_sync_minute,
        monthly_validation_day_of_month=config.monthly_validation_day_of_month,
        monthly_validation_hour=config.monthly_validation_hour,
        monthly_validation_minute=config.monthly_validation_minute,
    )


def runtime_scheduler_config_from_env(env: Mapping[str, str] | None = None):
    from trading_assistant.orchestrator.scheduler import SchedulerConfig

    merged = _load_dotenv_defaults()
    merged.update(os.environ)
    if env:
        merged.update({key: str(value) for key, value in env.items()})
    defaults = SchedulerConfig()
    return SchedulerConfig(
        market_data_sync_day_of_month=_int_env(
            merged,
            "MARKET_DATA_SYNC_DAY_OF_MONTH",
            defaults.market_data_sync_day_of_month,
        ),
        market_data_sync_hour=_int_env(merged, "MARKET_DATA_SYNC_HOUR", defaults.market_data_sync_hour),
        market_data_sync_minute=_int_env(
            merged,
            "MARKET_DATA_SYNC_MINUTE",
            defaults.market_data_sync_minute,
        ),
        monthly_validation_day_of_month=_int_env(
            merged,
            "MONTHLY_VALIDATION_DAY_OF_MONTH",
            defaults.monthly_validation_day_of_month,
        ),
        monthly_validation_hour=_int_env(
            merged,
            "MONTHLY_VALIDATION_HOUR",
            defaults.monthly_validation_hour,
        ),
        monthly_validation_minute=_int_env(
            merged,
            "MONTHLY_VALIDATION_MINUTE",
            defaults.monthly_validation_minute,
        ),
    )


def runtime_memory_dir_from_env(
    env: Mapping[str, str] | None = None,
    *,
    db_dir_explicit: bool = False,
) -> Path:
    merged = _load_dotenv_defaults()
    merged.update(os.environ)
    if env:
        merged.update({key: str(value) for key, value in env.items()})
    config = AppConfig(
        data_dir=merged.get("DATA_DIR", "data"),
        memory_dir=merged.get("MEMORY_DIR", ""),
    )
    return resolve_runtime_memory_dir(
        config,
        db_dir=config.data_dir,
        db_dir_explicit=db_dir_explicit,
    )


def _package_relative_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else package_root() / path


def _int_env(env: Mapping[str, str], name: str, default: int) -> int:
    return int(env.get(name, str(default)))
