"""Runtime shell and preflight checks for the monorepo scaffold."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import signal
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from libs.broker_ibkr.session import UnifiedIBSession
from libs.config.capital_allocation import resolve_strategy_capital_allocation
from libs.config.loader import (
    _resolve_env_vars,
    load_contracts,
    load_event_calendar,
    load_portfolio_config,
    load_routes,
    load_strategy_registry,
)
from libs.config.models import PortfolioConfig, StrategyRegistryConfig
from libs.config.registry import build_registry_artifact
from libs.oms.persistence.db_config import get_environment
from libs.runtime.active_config import (
    ActiveRuntimeConfigRecord,
    active_config_expiry,
    build_account_runtime_config,
    upsert_active_runtime_config,
)
from strategies.contracts import RuntimeContext
from strategies.stock.readiness import validate_stock_readiness

logger = logging.getLogger(__name__)

# Family coordinator registry (lazy imports to avoid circular deps)
_FAMILY_COORDINATORS: dict[str, str] = {
    "swing": "strategies.swing.coordinator.SwingFamilyCoordinator",
    "momentum": "strategies.momentum.coordinator.MomentumFamilyCoordinator",
    "stock": "strategies.stock.coordinator.StockFamilyCoordinator",
}

_PAPER_PORTS = {4002, 7497}
_LIVE_PORTS = {4001, 7496}
_ACCOUNT_PLACEHOLDER_TOKENS = ("PLACEHOLDER", "YOUR_ACCOUNT", "CHANGEME")
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


def _ib_mode_port_mismatch(runtime_env: str, port: int) -> bool:
    return (
        (runtime_env == "live" and port in _PAPER_PORTS)
        or (runtime_env == "paper" and port in _LIVE_PORTS)
    )


def _redact_account_id(account_id: str | None) -> str:
    value = str(account_id or "").strip()
    if not value:
        return "<blank>"
    if value.startswith("${") and value.endswith("}"):
        return value
    if len(value) <= 4:
        return "***"
    return f"{value[:2]}...{value[-2:]}"


def _is_placeholder_account_id(account_id: str | None) -> bool:
    value = str(account_id or "").strip().upper()
    return value.startswith("${") or any(
        token in value for token in _ACCOUNT_PLACEHOLDER_TOKENS
    )


def _ib_mode_account_ok(runtime_env: str, account_id: str | None) -> bool:
    value = str(account_id or "").strip().upper()
    if runtime_env == "paper":
        return value.startswith("DU") and not _is_placeholder_account_id(value)
    if runtime_env == "live":
        return (
            value.startswith("U")
            and not value.startswith("DU")
            and not _is_placeholder_account_id(value)
        )
    return True


def _ib_mode_account_detail(runtime_env: str, account_id: str | None) -> str:
    detail = f"env={runtime_env} account_id={_redact_account_id(account_id)}"
    if runtime_env == "paper":
        return f"{detail} expected_prefix=DU"
    if runtime_env == "live":
        return f"{detail} expected_prefix=U"
    return detail


def _import_coordinator(family: str) -> type:
    """Dynamically import a family coordinator class."""
    dotted = _FAMILY_COORDINATORS[family]
    module_path, class_name = dotted.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str


class RuntimeShell:
    """Loads monorepo runtime metadata and optionally starts IB connectivity."""

    def __init__(self, config_dir: str | Path):
        self.config_dir = Path(config_dir)
        self.registry = None
        self.portfolio = None
        self.contracts = None
        self.routes = None
        self.event_calendar = None
        self.session: UnifiedIBSession | None = None
        self._effective_config_hash = ""

    def load(self) -> None:
        self.registry = load_strategy_registry(self.config_dir)
        self.portfolio = load_portfolio_config(self.config_dir)
        self.contracts = load_contracts(self.config_dir)
        self.routes = load_routes(self.config_dir)
        self.event_calendar = load_event_calendar(self.config_dir)

    def _require_loaded(self) -> None:
        """Verify config was loaded. Raises RuntimeError instead of assert."""
        for attr in ("registry", "portfolio", "contracts", "routes", "event_calendar"):
            if getattr(self, attr) is None:
                raise RuntimeError(f"config not loaded — call load() first (missing: {attr})")

    def _ib_mode_checks(self, runtime_env: str) -> list[PreflightCheck]:
        if self.registry is None:
            return []
        checks: list[PreflightCheck] = []
        for group_name, group_cfg in self.registry.connection_groups.items():
            port = getattr(group_cfg, "port", 0)
            checks.append(
                PreflightCheck(
                    name=f"ib-mode-port:{group_name}",
                    ok=not _ib_mode_port_mismatch(runtime_env, port),
                    detail=f"env={runtime_env} port={port}",
                )
            )
            account_id = getattr(group_cfg, "account_id", None)
            checks.append(
                PreflightCheck(
                    name=f"ib-mode-account:{group_name}",
                    ok=_ib_mode_account_ok(runtime_env, account_id),
                    detail=_ib_mode_account_detail(runtime_env, account_id),
                )
            )
        return checks

    def run_preflight(self) -> list[PreflightCheck]:
        self.load()
        self._require_loaded()

        checks: list[PreflightCheck] = []
        runtime_env = get_environment()
        enabled = self.registry.enabled_strategies(live=runtime_env == "live")
        checks.append(
            PreflightCheck(
                name="registry-load",
                ok=True,
                detail=f"Loaded {len(self.registry.strategies)} strategies across {len(self.registry.connection_groups)} groups",
            )
        )
        checks.append(
            PreflightCheck(
                name="enabled-strategies",
                ok=bool(enabled),
                detail=f"{len(enabled)} strategies enabled",
            )
        )
        checks.extend(self._ib_mode_checks(runtime_env))

        missing_contracts: list[str] = []
        missing_routes: list[str] = []
        for manifest in enabled:
            for symbol in manifest.symbols:
                if symbol not in self.contracts:
                    missing_contracts.append(f"{manifest.strategy_id}:{symbol}")
                if symbol not in self.routes:
                    missing_routes.append(f"{manifest.strategy_id}:{symbol}")
        checks.append(
            PreflightCheck(
                name="contract-coverage",
                ok=not missing_contracts,
                detail="all manifest symbols resolved"
                if not missing_contracts
                else ", ".join(missing_contracts),
            )
        )
        checks.append(
            PreflightCheck(
                name="route-coverage",
                ok=not missing_routes,
                detail="all manifest symbols routed"
                if not missing_routes
                else ", ".join(missing_routes),
            )
        )

        _, stock_failures = validate_stock_readiness(
            self.registry,
            live=runtime_env == "live",
        )
        checks.extend(
            PreflightCheck(
                name=failure.check_name,
                ok=False,
                detail=failure.detail,
            )
            for failure in stock_failures
        )

        family_total = sum(self.portfolio.capital.family_allocations.values())
        checks.append(
            PreflightCheck(
                name="family-allocation-sum",
                ok=math.isclose(family_total, 1.0, abs_tol=1e-9),
                detail=f"family allocation total={family_total:.6f}",
            )
        )

        # Dynamic per-family allocation check for families with explicit strategy_allocations
        families_with_explicit: dict[str, list[str]] = {}
        for manifest in enabled:
            if manifest.strategy_id in self.portfolio.capital.strategy_allocations:
                families_with_explicit.setdefault(manifest.family, []).append(manifest.strategy_id)
        for family, strategy_ids in families_with_explicit.items():
            family_total = sum(
                self.portfolio.capital.strategy_allocations.get(sid, 0.0)
                for sid in strategy_ids
            )
            checks.append(
                PreflightCheck(
                    name=f"family-allocation-sum:{family}",
                    ok=math.isclose(family_total, 1.0, abs_tol=1e-9),
                    detail=f"{family} enabled strategy allocation total={family_total:.6f} ({', '.join(strategy_ids)})",
                )
            )

        for manifest in enabled:
            allocation = resolve_strategy_capital_allocation(
                manifest.strategy_id,
                raw_nav=self.portfolio.capital.allocation_check_equity,
                registry=self.registry,
                portfolio=self.portfolio,
            )
            checks.append(
                PreflightCheck(
                    name=f"allocation:{manifest.strategy_id}",
                    ok=allocation.allocated_nav > 0,
                    detail=f"allocated_nav={allocation.allocated_nav:.2f}",
                )
            )

        artifact = build_registry_artifact(self.registry)
        checks.append(
            PreflightCheck(
                name="registry-artifact",
                ok=len(artifact["strategies"]) == len(self.registry.strategies),
                detail=f"artifact strategies={len(artifact['strategies'])}",
            )
        )

        # Family cross-validation: every enabled strategy's family must have a family_allocation entry
        families_used = {m.family for m in enabled}
        families_configured = set(self.portfolio.capital.family_allocations.keys())
        missing_families = families_used - families_configured
        extra_families = families_configured - families_used
        family_detail_parts: list[str] = []
        if missing_families:
            family_detail_parts.append(f"missing allocations for: {sorted(missing_families)}")
        if extra_families:
            family_detail_parts.append(f"unreferenced families: {sorted(extra_families)}")
        checks.append(
            PreflightCheck(
                name="family-allocation-coverage",
                ok=not missing_families,
                detail="; ".join(family_detail_parts) if family_detail_parts else "all families covered",
            )
        )

        checks.append(
            PreflightCheck(
                name="event-calendar",
                ok=True,
                detail=f"{len(self.event_calendar.windows)} blackout windows configured",
            )
        )
        return checks

    def _run_sync_preflight_or_raise(self) -> list[PreflightCheck]:
        checks = self.run_preflight()
        for check in checks:
            level = logging.INFO if check.ok else logging.ERROR
            logger.log(
                level,
                "SYNC PREFLIGHT %s: %s -- %s",
                "OK" if check.ok else "FAIL",
                check.name,
                check.detail,
            )
        failures = [check for check in checks if not check.ok]
        if failures:
            raise RuntimeError(f"Runtime sync preflight failed: {len(failures)} check(s)")
        return checks

    async def _run_async_preflight(
        self,
        connect_ib: bool,
        families: set[str],
        require_instrumentation: bool = False,
    ) -> list[PreflightCheck]:
        """Run async preflight checks before heavy startup.

        Checks:
          1a. Coordinator imports (CRITICAL)
          1b. Database connectivity (CRITICAL in paper/live)
          1c. IB Gateway reachability (CRITICAL when connect_ib=True)
          1d. Instrumentation config/evidence path readiness
        """
        checks: list[PreflightCheck] = []

        # 1a. Coordinator imports
        for family in sorted(families):
            if family not in _FAMILY_COORDINATORS:
                checks.append(PreflightCheck(
                    name=f"import:{family}",
                    ok=False,
                    detail=f"No coordinator registered for family '{family}'",
                ))
                continue
            try:
                _import_coordinator(family)
                checks.append(PreflightCheck(
                    name=f"import:{family}",
                    ok=True,
                    detail=f"Coordinator for '{family}' imported successfully",
                ))
            except Exception as exc:
                checks.append(PreflightCheck(
                    name=f"import:{family}",
                    ok=False,
                    detail=f"Coordinator import failed: {exc}",
                ))

        checks.extend(self._ib_mode_checks(get_environment()))

        # 1b. Database connectivity
        try:
            from libs.oms.persistence.db_config import DBConfig
            db_config = DBConfig.from_env()
            if db_config is not None:
                import asyncpg
                conn = await asyncio.wait_for(
                    asyncpg.connect(dsn=db_config.to_dsn()),
                    timeout=5.0,
                )
                await conn.execute("SELECT 1")
                await conn.close()
                checks.append(PreflightCheck(
                    name="database",
                    ok=True,
                    detail="Database reachable",
                ))
            else:
                env = get_environment()
                db_required = env in ("paper", "live")
                checks.append(PreflightCheck(
                    name="database",
                    ok=not db_required,
                    detail=f"No DB config (env={env})"
                    + (" -- required for paper/live" if db_required else " -- OK for dev/backtest"),
                ))
        except Exception as exc:
            checks.append(PreflightCheck(
                name="database",
                ok=False,
                detail=f"Database unreachable: {exc}",
            ))

        if "stock" in families and self.registry is not None:
            _, stock_failures = validate_stock_readiness(
                self.registry,
                live=get_environment() == "live",
            )
            checks.extend(
                PreflightCheck(
                    name=failure.check_name,
                    ok=False,
                    detail=failure.detail,
                )
                for failure in stock_failures
            )

        # 1c. IB Gateway reachability (async to avoid blocking the event loop)
        if connect_ib and self.registry is not None:
            for group_name, group_cfg in self.registry.connection_groups.items():
                host = getattr(group_cfg, "host", "127.0.0.1")
                port = getattr(group_cfg, "port", 4002)
                try:
                    _reader, _writer = await asyncio.wait_for(
                        asyncio.open_connection(host, port),
                        timeout=5.0,
                    )
                    _writer.close()
                    await _writer.wait_closed()
                    checks.append(PreflightCheck(
                        name=f"ib-gateway:{group_name}",
                        ok=True,
                        detail=f"IB Gateway reachable at {host}:{port}",
                    ))
                except Exception as exc:
                    checks.append(PreflightCheck(
                        name=f"ib-gateway:{group_name}",
                        ok=False,
                        detail=f"IB Gateway unreachable at {host}:{port}: {exc}",
                    ))

        # 1d. Instrumentation config/evidence path readiness
        for family in sorted(families):
            config_path = (
                _WORKSPACE_ROOT / "strategies" / family / "instrumentation" / "config" / "instrumentation_config.yaml"
            )
            if not config_path.exists():
                checks.append(PreflightCheck(
                    name=f"instr-config:{family}",
                    ok=not require_instrumentation,
                    detail=(
                        f"{config_path} missing"
                        if require_instrumentation
                        else "Instrumentation config missing; strategies will use defaults"
                    ),
                ))
                continue
            try:
                import yaml
                with open(config_path, encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                checks.append(PreflightCheck(
                    name=f"instr-config:{family}",
                    ok=True,
                    detail="Instrumentation config parsed OK",
                ))
                checks.extend(
                    self._instrumentation_readiness_checks(
                        family,
                        config,
                        require_instrumentation=require_instrumentation,
                    )
                )
            except Exception as exc:
                checks.append(PreflightCheck(
                    name=f"instr-config:{family}",
                    ok=not require_instrumentation,
                    detail=(
                        f"Instrumentation config parse error"
                        f"{' (fatal with required instrumentation)' if require_instrumentation else ' (non-fatal)'}: {exc}"
                    ),
                ))
                logger.warning("Instrumentation config for %s unparseable: %s", family, exc)

        return checks

    def _instrumentation_readiness_checks(
        self,
        family: str,
        config: dict[str, Any],
        *,
        require_instrumentation: bool,
    ) -> list[PreflightCheck]:
        from trading_contracts.relay_acceptance import (
            probe_relay_acceptance,
            validate_hmac_secret,
            validate_relay_url,
        )

        sidecar = config.get("sidecar") if isinstance(config.get("sidecar"), dict) else {}
        bot_id = str(config.get("bot_id") or family).strip()
        relay_url = os.environ.get("INSTRUMENTATION_RELAY_URL") or str(
            sidecar.get("relay_url") or ""
        ).strip()
        relay_errors = validate_relay_url(
            relay_url,
            allow_loopback=self._allow_loopback_relay(),
        )
        relay_ok = not relay_errors

        hmac_env = str(sidecar.get("hmac_secret_env") or "INSTRUMENTATION_HMAC_SECRET").strip()
        hmac_secret = os.environ.get(hmac_env, "")
        hmac_errors = validate_hmac_secret(hmac_secret, field_name=hmac_env)
        hmac_ok = not hmac_errors

        checks = [
            PreflightCheck(
                name=f"instrumentation-relay:{family}",
                ok=relay_ok or not require_instrumentation,
                detail=(
                    f"Relay URL configured: {relay_url}"
                    if relay_ok
                    else f"Valid sidecar relay_url is required for {family}: {'; '.join(relay_errors)}"
                ),
            ),
            PreflightCheck(
                name=f"instrumentation-hmac:{family}",
                ok=hmac_ok or not require_instrumentation,
                detail=(
                    f"{hmac_env} configured"
                    if hmac_ok
                    else f"{hmac_env} is required when paper/live instrumentation is required: {'; '.join(hmac_errors)}"
                ),
            ),
        ]

        if require_instrumentation and relay_ok and hmac_ok:
            from trading_contracts.relay_acceptance import bot_exact_ack_api_key

            api_key = bot_exact_ack_api_key(os.environ)
            effective_hash = self._effective_config_hash or os.environ.get(
                "IBKR_EFFECTIVE_CONFIG_HASH",
                "unverified-effective-config",
            )
            relay_probe = probe_relay_acceptance(
                relay_url=relay_url,
                hmac_secret=hmac_secret,
                bot_id=bot_id,
                runtime_instance_id=f"ibkr-{family}:{effective_hash[:12]}",
                effective_config_hash=effective_hash,
                deployment_id=f"ibkr:{family}:{effective_hash[:16]}",
                source=f"ibkr-{family}-runtime-preflight",
                timeout_seconds=float(os.environ.get("RELAY_PROBE_TIMEOUT_SECONDS", "10")),
                confirm_health=True,
                require_exact_ack=bool(api_key),
                relay_api_key=api_key,
            )
            checks.append(
                PreflightCheck(
                    name=f"instrumentation-relay-acceptance:{family}",
                    ok=relay_probe.ok,
                    detail=(
                        f"assistant relay accepted heartbeat event_id={relay_probe.event_id}"
                        if relay_probe.ok
                        else f"assistant relay acceptance failed: {relay_probe.error}"
                    ),
                )
            )

        for check_name, raw_path in (
            ("instrumentation-data-dir", config.get("data_dir")),
            ("instrumentation-buffer-dir", sidecar.get("buffer_dir")),
        ):
            if not raw_path:
                checks.append(PreflightCheck(
                    name=f"{check_name}:{family}",
                    ok=not require_instrumentation,
                    detail=f"{check_name} missing from instrumentation config",
                ))
                continue
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = _WORKSPACE_ROOT / path
            parent = path if path.exists() else path.parent
            checks.append(PreflightCheck(
                name=f"{check_name}:{family}",
                ok=(parent.exists() and os.access(parent, os.W_OK)) or not require_instrumentation,
                detail=(
                    f"{path} parent writable"
                    if parent.exists() and os.access(parent, os.W_OK)
                    else f"{path} parent missing or not writable"
                ),
            ))
        return checks

    @staticmethod
    def _allow_loopback_relay() -> bool:
        mode = os.environ.get("RELAY_NETWORK_MODE", "").strip().lower()
        return os.environ.get("ALLOW_LOOPBACK_RELAY") == "1" or mode in {
            "local_direct",
            "private_interface",
            "secure_tunnel",
            "tunnel",
        }

    def _effective_config_preflight_checks(self, effective_config_path: str | Path | None) -> list[PreflightCheck]:
        if not effective_config_path:
            return [
                PreflightCheck(
                    name="effective-config:ibkr",
                    ok=False,
                    detail="--effective-config is required in paper/live runtime mode",
                )
            ]
        path = Path(effective_config_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return [
                PreflightCheck(
                    name="effective-config:ibkr",
                    ok=False,
                    detail=f"could not read {path}: {exc}",
                )
            ]

        checks: list[PreflightCheck] = []
        bot_id = str(payload.get("bot_id") or "")
        checks.append(
            PreflightCheck(
                name="effective-config:ibkr",
                ok=bot_id == "ibkr",
                detail=f"bot_id={bot_id or '<missing>'} path={path}",
            )
        )
        effective_hash = str(payload.get("effective_config_hash") or "")
        materialized_hash = str(payload.get("materialized_config_hash") or "")
        if effective_hash:
            self._effective_config_hash = effective_hash
        checks.append(
            PreflightCheck(
                name="effective-config-hash:ibkr",
                ok=bool(effective_hash and materialized_hash),
                detail=f"effective_hash={effective_hash[:12]} materialized_hash={materialized_hash[:12]}",
            )
        )

        source_files = payload.get("source_files")
        if not isinstance(source_files, list):
            checks.append(
                PreflightCheck(
                    name="effective-config-sources:ibkr",
                    ok=False,
                    detail="source_files must be a list",
                )
            )
            return checks

        roles = {str(item.get("role") or "") for item in source_files if isinstance(item, dict)}
        required_roles = {"ibkr_strategies", "ibkr_portfolio"}
        missing_roles = sorted(required_roles - roles)
        checks.append(
            PreflightCheck(
                name="effective-config-source-roles:ibkr",
                ok=not missing_roles,
                detail=(
                    "required source roles present"
                    if not missing_roles
                    else f"missing required source roles: {missing_roles}"
                ),
            )
        )

        root = self._repo_root()
        for item in source_files:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "<unknown>")
            raw_source_path = str(item.get("path") or "")
            expected_hash = str(item.get("sha256") or item.get("file_sha256") or "")
            source_path = Path(raw_source_path)
            if not source_path.is_absolute():
                source_path = root / source_path
            if not source_path.exists():
                checks.append(
                    PreflightCheck(
                        name=f"effective-config-source:{role}",
                        ok=False,
                        detail=f"{source_path} missing",
                    )
                )
                continue
            actual_hash = self._file_sha256(source_path)
            checks.append(
                PreflightCheck(
                    name=f"effective-config-source:{role}",
                    ok=bool(expected_hash) and actual_hash == expected_hash,
                    detail=f"{raw_source_path} sha256={actual_hash[:12]}",
                )
            )
        checks.append(self._loaded_runtime_equivalence_check(payload.get("materialized_config")))
        return checks

    def _loaded_runtime_equivalence_check(self, materialized: Any) -> PreflightCheck:
        if not isinstance(materialized, dict):
            return PreflightCheck(
                name="effective-config-loaded-runtime:ibkr",
                ok=False,
                detail="materialized_config must be present for loaded runtime comparison",
            )
        try:
            expected = self._expected_loaded_runtime_config(materialized)
            registry = self.registry or load_strategy_registry(self.config_dir)
            portfolio = self.portfolio or load_portfolio_config(self.config_dir)
            expected_strategy_ids = set(expected["strategies"])
            actual_strategy_ids = set(registry.strategies)
            missing = sorted(expected_strategy_ids - actual_strategy_ids)
            extra = sorted(actual_strategy_ids - expected_strategy_ids)
            if missing or extra:
                raise ValueError(
                    f"loaded registry strategy roster mismatch: missing={missing} extra={extra}"
                )
            actual = self._loaded_runtime_config(
                registry=registry,
                portfolio=portfolio,
                strategy_ids=expected_strategy_ids,
            )
        except Exception as exc:
            return PreflightCheck(
                name="effective-config-loaded-runtime:ibkr",
                ok=False,
                detail=f"loaded runtime comparison failed: {exc}",
            )
        actual_hash = self._canonical_json_hash(actual)
        expected_hash = self._canonical_json_hash(expected)
        return PreflightCheck(
            name="effective-config-loaded-runtime:ibkr",
            ok=actual == expected,
            detail=f"loaded_hash={actual_hash[:12]} expected_hash={expected_hash[:12]}",
        )

    @staticmethod
    def _expected_loaded_runtime_config(materialized: dict[str, Any]) -> dict[str, Any]:
        strategy_records = materialized.get("strategies")
        if not isinstance(strategy_records, list):
            raise ValueError("materialized_config.strategies must be a list")
        strategies: dict[str, Any] = {}
        for record in strategy_records:
            if not isinstance(record, dict):
                continue
            strategy_id = str(record.get("strategy_id") or "").strip()
            if strategy_id:
                strategies[strategy_id] = record.get("effective_config") or {}
        registry = StrategyRegistryConfig.model_validate(
            {
                "connection_groups": _resolve_env_vars(materialized.get("connection_groups") or {}),
                "strategies": _resolve_env_vars(strategies),
            }
        )
        portfolio = PortfolioConfig.model_validate(
            _resolve_env_vars(materialized.get("portfolio") or {})
        )
        return {
            "connection_groups": RuntimeShell._model_map_dump(registry.connection_groups),
            "portfolio": portfolio.model_dump(mode="json"),
            "strategies": RuntimeShell._model_map_dump(registry.strategies),
        }

    def _loaded_runtime_config(
        self,
        *,
        registry: StrategyRegistryConfig,
        portfolio: PortfolioConfig,
        strategy_ids: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "connection_groups": self._model_map_dump(registry.connection_groups),
            "portfolio": portfolio.model_dump(mode="json"),
            "strategies": {
                strategy_id: registry.strategies[strategy_id].model_dump(mode="json")
                for strategy_id in sorted(strategy_ids)
            },
        }

    @staticmethod
    def _model_map_dump(values: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
            for key, value in sorted(values.items())
        }

    @staticmethod
    def _canonical_json_hash(payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _repo_root() -> Path:
        env_root = os.environ.get("REPO_ROOT")
        if env_root:
            return Path(env_root).resolve()
        return _WORKSPACE_ROOT.parent.parent.resolve()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    async def run(
        self,
        shadow: bool = False,
        connect_ib: bool = False,
        once: bool = False,
        family_filter: str | None = None,
        effective_config_path: str | Path | None = None,
        allow_no_db: bool = False,
        allow_partial_families: bool = False,
        allow_no_instrumentation: bool = False,
    ) -> None:
        runtime_env = get_environment()
        if runtime_env in {"paper", "live"}:
            self._run_sync_preflight_or_raise()
        else:
            self.load()
        self._require_loaded()

        enabled = self.registry.enabled_strategies(live=runtime_env == "live")
        logger.info(
            "Runtime shell loaded %d enabled strategies across %d connection groups (env=%s)%s",
            len(enabled),
            len(self.registry.connection_groups),
            runtime_env,
            " in shadow mode" if shadow else "",
        )

        # ------------------------------------------------------------------
        # 0. Filter by family if requested (before preflight)
        # ------------------------------------------------------------------
        if family_filter:
            enabled = [m for m in enabled if m.family == family_filter]
            if not enabled:
                raise RuntimeError(f"No enabled strategies for family={family_filter!r}")
            logger.info("Family filter active: running %d strategies for '%s'", len(enabled), family_filter)

        require_instrumentation = (
            runtime_env in {"paper", "live"}
            and not allow_no_instrumentation
        )

        effective_config_required = (
            runtime_env in {"paper", "live"}
            and bool(enabled)
            and not (once and not connect_ib and effective_config_path is None)
        )
        if effective_config_path is not None or effective_config_required:
            effective_checks = self._effective_config_preflight_checks(effective_config_path)
            for c in effective_checks:
                lvl = logging.INFO if c.ok else logging.ERROR
                logger.log(lvl, "PREFLIGHT %s: %s -- %s", "OK" if c.ok else "FAIL", c.name, c.detail)
            effective_failures = [c for c in effective_checks if not c.ok]
            if effective_failures:
                raise RuntimeError(
                    f"Effective config preflight failed: {len(effective_failures)} check(s)"
                )

        # ------------------------------------------------------------------
        # 1. Async preflight (fail-fast before heavy startup)
        # ------------------------------------------------------------------
        enabled_families = {m.family for m in enabled}
        checks = await self._run_async_preflight(
            connect_ib=connect_ib,
            families=enabled_families,
            require_instrumentation=require_instrumentation,
        )
        for c in checks:
            lvl = logging.INFO if c.ok else logging.WARNING
            logger.log(lvl, "PREFLIGHT %s: %s -- %s", "OK" if c.ok else "FAIL", c.name, c.detail)
        critical_prefixes = {
            "import",
            "database",
            "ib-gateway",
            "ib-mode-account",
            "ib-mode-port",
        }
        if "stock" in enabled_families and (
            runtime_env in {"paper", "live"} or enabled_families == {"stock"}
        ):
            critical_prefixes.update({
                "stock-account-config",
                "stock-artifact-readiness",
            })
        if require_instrumentation:
            critical_prefixes.update({
                "instr-config",
                "instrumentation-relay",
                "instrumentation-hmac",
                "instrumentation-relay-acceptance",
                "instrumentation-data-dir",
                "instrumentation-buffer-dir",
            })

        critical_failures = [
            c for c in checks
            if not c.ok and c.name.split(":")[0] in critical_prefixes
        ]
        if critical_failures:
            for c in critical_failures:
                logger.error("PREFLIGHT FAIL: %s -- %s", c.name, c.detail)
            raise RuntimeError(f"Preflight failed: {len(critical_failures)} critical check(s)")

        # ------------------------------------------------------------------
        # 2. Connect broker
        # ------------------------------------------------------------------
        if connect_ib:
            strategy_group_map = {
                manifest.strategy_id: manifest.connection_group for manifest in enabled
            }
            self.session = UnifiedIBSession(self.registry.connection_groups, strategy_group_map)
            await self.session.start()
            await self.session.wait_ready()
            logger.info("Unified IB session connected for all configured groups")
            await self.session.verify_streaming_data(runtime_env=runtime_env)

        if once:
            return

        # ------------------------------------------------------------------
        # 3. Bootstrap database
        # ------------------------------------------------------------------
        db_pool = None
        account_gate = None
        trade_recorder = None
        heartbeat = None
        try:
            from libs.services.bootstrap import bootstrap_database
            bootstrap_ctx = await bootstrap_database()
            db_pool = bootstrap_ctx.pool
            trade_recorder = bootstrap_ctx.trade_recorder
            heartbeat = bootstrap_ctx.heartbeat
            logger.info("Database bootstrapped")
        except Exception as exc:
            if allow_no_db:
                logger.warning("Database bootstrap failed (--allow-no-db): %s", exc)
            else:
                raise RuntimeError(
                    f"Database bootstrap failed (portfolio rules require DB). "
                    f"Use --allow-no-db to start without DB. Error: {exc}"
                ) from exc

        if db_pool is not None:
            # CFG-1: wire portfolio.yaml.risk.* through to the gate.
            # Previously the gate was constructed without the YAML values and
            # silently fell through to dataclass defaults — making the YAML
            # keys dead config. Now editing portfolio.yaml has the effect
            # operators expect.
            risk_cfg = self.portfolio.risk
            try:
                from libs.risk.account_risk_gate import AccountRiskGate
                _account_urd = float(
                    os.environ.get(
                        "ACCOUNT_UNIT_RISK_DOLLARS",
                        str(risk_cfg.account_urd_dollars),
                    )
                )
                if _account_urd <= 0:
                    raise ValueError("account_urd_dollars must be > 0")
                account_gate = AccountRiskGate(
                    db_pool,
                    heat_cap_R=risk_cfg.heat_cap_R,
                    daily_stop_R=risk_cfg.portfolio_daily_stop_R,
                    weekly_stop_R=risk_cfg.portfolio_weekly_stop_R,
                    account_urd=_account_urd,
                    global_standdown=risk_cfg.global_standdown,
                )
                account_id = next(
                    (
                        str(group.account_id).strip()
                        for group in self.registry.connection_groups.values()
                        if getattr(group, "account_id", None)
                    ),
                    "default",
                )
                account_payload = build_account_runtime_config(
                    account_id=account_id,
                    heat_cap_R=risk_cfg.heat_cap_R,
                    portfolio_daily_stop_R=risk_cfg.portfolio_daily_stop_R,
                    portfolio_weekly_stop_R=risk_cfg.portfolio_weekly_stop_R,
                    global_standdown=risk_cfg.global_standdown,
                    account_urd=_account_urd,
                )
                await upsert_active_runtime_config(
                    db_pool,
                    ActiveRuntimeConfigRecord(
                        account_id=account_id,
                        config_scope="account",
                        scope_id=account_id,
                        runtime_env=get_environment(),
                        payload=account_payload,
                        expires_at=active_config_expiry(),
                    ),
                )
                logger.info(
                    "AccountRiskGate active: heat=%.1fR=$%.0f, "
                    "daily_stop=%.1fR=$%.0f, weekly_stop=%.1fR=$%.0f, "
                    "global_standdown=%s, urd=$%.0f",
                    risk_cfg.heat_cap_R, risk_cfg.heat_cap_R * _account_urd,
                    risk_cfg.portfolio_daily_stop_R,
                    risk_cfg.portfolio_daily_stop_R * _account_urd,
                    risk_cfg.portfolio_weekly_stop_R,
                    risk_cfg.portfolio_weekly_stop_R * _account_urd,
                    risk_cfg.global_standdown,
                    _account_urd,
                )
            except Exception as exc:
                # In paper/live, deletion of the only cross-family heat cap is
                # a hard fail; in dev it stays a warning so unit tests with no
                # AccountRiskGate dependency still run.
                if get_environment() in ("paper", "live"):
                    raise RuntimeError(
                        f"AccountRiskGate init failed in {get_environment()} mode "
                        f"- refusing to start without cross-family risk gate: {exc}"
                    ) from exc
                logger.warning(
                    "AccountRiskGate init failed (non-fatal in dev): %s", exc,
                )

        # ------------------------------------------------------------------
        # 3.5  Regime service
        # ------------------------------------------------------------------
        regime_service = None
        market_calendar = None
        if connect_ib and self.session:
            try:
                from regime.live import RegimeService
                from libs.config.market_calendar import MarketCalendar
                market_calendar = MarketCalendar()
                regime_service = RegimeService(
                    ib_session=self.session,
                    market_calendar=market_calendar,
                )
                await regime_service.start()
                logger.info("Regime service started: %s", regime_service.get_context())
            except Exception as exc:
                logger.error("Regime service init failed: %s", exc, exc_info=True)
                raise RuntimeError(
                    "Regime service startup failed; live deployment requires "
                    "a usable HMM regime backlog after cache + IBKR/FRED backfill"
                ) from exc

        # ------------------------------------------------------------------
        # 3.6  Crisis detection service
        # ------------------------------------------------------------------
        crisis_service = None
        if connect_ib and self.session:
            try:
                from regime.crisis.service import CrisisService
                crisis_service = CrisisService(
                    ib_session=self.session,
                    market_calendar=market_calendar,
                )
                await crisis_service.start()
                logger.info("Crisis service started: %s", crisis_service.get_context().alert_level if crisis_service.get_context() else "loading")
            except Exception as exc:
                logger.error("Crisis service init failed: %s", exc, exc_info=True)
                raise RuntimeError(
                    "Crisis service startup failed; live deployment requires "
                    "a usable early-regime/crisis backlog after cache + IBKR/FRED backfill"
                ) from exc

        # ------------------------------------------------------------------
        # 4. Group strategies by family and build coordinators
        # ------------------------------------------------------------------
        families: dict[str, list] = {}
        for manifest in enabled:
            families.setdefault(manifest.family, []).append(manifest)

        coordinators: list[Any] = []
        for family, manifests in families.items():
            if family not in _FAMILY_COORDINATORS:
                logger.error("No coordinator registered for family '%s', skipping", family)
                continue

            # Build RuntimeContext for this family
            ctx = RuntimeContext(
                manifest=manifests[0],  # primary manifest (coordinator reads all from registry)
                registry=self.registry,
                portfolio=self.portfolio,
                session=self.session,
                market_data=None,
                oms=None,  # coordinators build their own OMS
                state_store=None,
                instrumentation=None,
                contracts=self.contracts,
                health={},
                logger=logging.getLogger(f"runtime.{family}"),
                clock=None,
                db_pool=db_pool,
                account_gate=account_gate,
                family_coordinator=None,
                regime_service=regime_service,
                crisis_service=crisis_service,
                trade_recorder=trade_recorder,
                heartbeat=heartbeat,
                require_instrumentation=require_instrumentation,
            )

            try:
                coordinator_cls = _import_coordinator(family)
                coordinator = coordinator_cls(ctx)
                coordinators.append(coordinator)
                logger.info(
                    "Coordinator created for family '%s' (%d strategies)",
                    family, len(manifests),
                )
            except Exception as exc:
                logger.error("Failed to create coordinator for '%s': %s", family, exc, exc_info=True)
                # RUNTIME-1: in paper/live, refuse to start with a missing
                # family by default. Operators expecting a 3-family runtime
                # would otherwise silently end up with 1-2 families running
                # and an under-allocated AccountRiskGate.
                if (
                    runtime_env in ("paper", "live")
                    and not allow_partial_families
                ):
                    raise RuntimeError(
                        f"Coordinator '{family}' failed to construct in "
                        f"{runtime_env} mode. Pass --allow-partial-families "
                        f"to permit degraded startup."
                    ) from exc

        # ------------------------------------------------------------------
        # 5. Start all coordinators (must run before regime apply so that
        #    _base_portfolio_rules is initialised inside start())
        # ------------------------------------------------------------------
        started_coordinators: list[Any] = []
        for coordinator in coordinators:
            try:
                await coordinator.start()
                started_coordinators.append(coordinator)
                logger.info("Family '%s' coordinator started", coordinator.family_id)
            except Exception as exc:
                logger.error(
                    "Coordinator '%s' failed to start: %s",
                    getattr(coordinator, "family_id", "?"), exc, exc_info=True,
                )
                # RUNTIME-1: same gate as the create loop above. Running
                # paper/live with a partial family set silently mis-allocates
                # the AccountRiskGate. Default = strict; opt-in for
                # development workflows that need a single-family run.
                if (
                    runtime_env in ("paper", "live")
                    and not allow_partial_families
                ):
                    raise RuntimeError(
                        f"Coordinator '{getattr(coordinator, 'family_id', '?')}' "
                        f"failed to start in {runtime_env} mode. Pass "
                        f"--allow-partial-families to permit degraded startup."
                    ) from exc
        coordinators = started_coordinators

        # ------------------------------------------------------------------
        # 5b. Load and apply initial regime context AFTER coordinators started
        # ------------------------------------------------------------------
        regime_task: asyncio.Task | None = None
        try:
            regime_ctx = None
            if regime_service is not None:
                regime_ctx = regime_service.get_context()
            if regime_ctx is None:
                from regime.persistence import load_regime_context
                regime_ctx = load_regime_context()
            for coordinator in coordinators:
                if hasattr(coordinator, "apply_regime"):
                    try:
                        coordinator.apply_regime(regime_ctx)
                    except Exception as exc:
                        logger.error("Initial regime apply failed for %s: %s",
                                    getattr(coordinator, "family_id", "?"), exc)
            logger.info(
                "Regime context applied: regime=%s, confidence=%.3f, "
                "computed_at=%s, data_as_of=%s",
                regime_ctx.regime,
                regime_ctx.regime_confidence,
                regime_ctx.computed_at or "unknown",
                getattr(regime_ctx, "data_as_of", "") or "unknown",
            )
        except Exception as exc:
            logger.warning("Regime context load failed (non-fatal): %s", exc)

        # 5c. Load and apply initial crisis context AFTER regime
        last_crisis_action_level_int: int | None = None
        try:
            crisis_ctx = None
            if crisis_service is not None:
                crisis_ctx = crisis_service.get_context()
            if crisis_ctx is None:
                from regime.crisis.persistence import load_crisis_context
                crisis_ctx = load_crisis_context()
            last_crisis_action_level_int = crisis_ctx.portfolio_action_level_int
            for coordinator in coordinators:
                if hasattr(coordinator, "apply_crisis"):
                    try:
                        coordinator.apply_crisis(crisis_ctx)
                    except Exception as exc:
                        logger.error("Initial crisis apply failed for %s: %s",
                                    getattr(coordinator, "family_id", "?"), exc)
            logger.info(
                "Crisis context applied: internal=%s advisory=%s action=%s "
                "(risk_mult=%.2f, data_as_of=%s)",
                crisis_ctx.alert_level,
                crisis_ctx.advisory_level,
                crisis_ctx.portfolio_action_level,
                crisis_ctx.risk_multiplier,
                getattr(crisis_ctx, "data_as_of", "") or "unknown",
            )
        except Exception as exc:
            logger.warning("Crisis context load failed (non-fatal): %s", exc)

        if not coordinators:
            logger.error("No coordinators started successfully — shutting down")
            if db_pool is not None:
                await db_pool.close()
            if self.session is not None:
                await self.session.stop()
            return

        async def _apply_crisis_context(ctx: Any, source: str) -> None:
            """Apply live crisis context, refreshing HMM first on hard escalation."""
            nonlocal last_crisis_action_level_int
            prev_action = last_crisis_action_level_int
            last_crisis_action_level_int = ctx.portfolio_action_level_int
            crisis_escalated = (
                ctx.portfolio_action_level_int >= 2
                and (prev_action is None or prev_action < 2)
            )

            if crisis_escalated and regime_service is not None:
                try:
                    regime_ctx = await regime_service.compute_now()
                    for coordinator in coordinators:
                        if hasattr(coordinator, "apply_regime"):
                            try:
                                coordinator.apply_regime(regime_ctx)
                            except Exception as exc:
                                logger.error(
                                    "Crisis-triggered regime refresh failed for %s: %s",
                                    getattr(coordinator, "family_id", "?"), exc,
                                )
                    logger.info(
                        "Crisis-triggered regime refresh: %s "
                        "(confidence=%.3f, computed_at=%s, data_as_of=%s)",
                        regime_ctx.regime,
                        regime_ctx.regime_confidence,
                        regime_ctx.computed_at or "unknown",
                        getattr(regime_ctx, "data_as_of", "") or "unknown",
                    )
                except Exception as exc:
                    logger.error("Crisis-triggered regime recompute failed: %s", exc)

            for coordinator in coordinators:
                if hasattr(coordinator, "apply_crisis"):
                    try:
                        coordinator.apply_crisis(ctx)
                    except Exception as exc:
                        logger.error("Crisis apply failed for %s: %s",
                                    getattr(coordinator, "family_id", "?"), exc)
            logger.info(
                "Crisis signal delivered (%s): internal=%s advisory=%s action=%s "
                "(risk_mult=%.2f, dominant=%s, data_as_of=%s)",
                source,
                ctx.alert_level,
                ctx.advisory_level,
                ctx.portfolio_action_level,
                ctx.risk_multiplier,
                ctx.dominant_channel,
                getattr(ctx, "data_as_of", "") or "unknown",
            )

        if crisis_service is not None:
            crisis_service.add_listener(
                lambda ctx: _apply_crisis_context(ctx, "live_service")
            )
            logger.info("Crisis live service listener registered for downstream delivery")

        # ------------------------------------------------------------------
        # 6. Run until shutdown signal
        # ------------------------------------------------------------------
        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            logger.info("Shutdown signal received")
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                pass  # Windows

        active_families = [getattr(c, "family_id", "?") for c in coordinators]
        logger.info("Runtime active — families: %s — press Ctrl+C to stop", active_families)

        # 6b. Start weekly regime refresh task
        async def _regime_refresh_loop() -> None:
            """Reload regime context weekly. Checks hourly, refreshes Friday 17:00+ ET."""
            from zoneinfo import ZoneInfo
            ET = ZoneInfo("America/New_York")
            last_refresh_date = None
            while True:
                await asyncio.sleep(3600)
                if stop_event.is_set():
                    break
                now_et = datetime.now(ET)
                today = now_et.date()
                if now_et.weekday() != 4 or now_et.hour < 17:
                    continue
                if last_refresh_date == today:
                    continue
                last_refresh_date = today
                try:
                    # Prefer live service context over disk
                    ctx = None
                    if regime_service is not None:
                        ctx = regime_service.get_context()
                    if ctx is None:
                        from regime.persistence import load_regime_context
                        ctx = load_regime_context()

                    # Staleness circuit breaker: escalate to ERROR if >14 days old
                    if ctx.computed_at:
                        try:
                            from datetime import timezone as _tz
                            age = datetime.now(_tz.utc) - datetime.fromisoformat(ctx.computed_at)
                            if age.days > 14:
                                logger.error(
                                    "Regime context is %d days stale (computed_at=%s) -- "
                                    "check RegimeService or data pipeline",
                                    age.days, ctx.computed_at,
                                )
                        except (ValueError, TypeError):
                            pass

                    for coordinator in coordinators:
                        if hasattr(coordinator, "apply_regime"):
                            try:
                                coordinator.apply_regime(ctx)
                            except Exception as exc:
                                logger.error("Regime refresh failed for %s: %s",
                                            getattr(coordinator, "family_id", "?"), exc)
                    logger.info(
                        "Weekly regime refresh: %s "
                        "(confidence=%.3f, computed_at=%s, data_as_of=%s)",
                        ctx.regime,
                        ctx.regime_confidence,
                        ctx.computed_at or "unknown",
                        getattr(ctx, "data_as_of", "") or "unknown",
                    )
                except Exception as exc:
                    logger.error("Regime refresh loop error: %s", exc)

        regime_task = asyncio.create_task(_regime_refresh_loop(), name="regime_refresh")

        # 6c. Start daily crisis refresh task
        crisis_task: asyncio.Task | None = None

        async def _crisis_refresh_loop() -> None:
            """Fallback disk-based crisis refresh when the live service is unavailable."""
            from zoneinfo import ZoneInfo
            ET = ZoneInfo("America/New_York")
            last_refresh_date = None
            while True:
                await asyncio.sleep(3600)
                if stop_event.is_set():
                    break
                now_et = datetime.now(ET)
                today = now_et.date()
                # Weekdays only, after 17:00 ET
                if now_et.weekday() >= 5:
                    continue
                if now_et.hour < 17:
                    continue
                if last_refresh_date == today:
                    continue
                last_refresh_date = today
                try:
                    from regime.crisis.persistence import load_crisis_context
                    ctx = load_crisis_context()
                    await _apply_crisis_context(ctx, "disk_fallback")
                except Exception as exc:
                    logger.error("Crisis refresh loop error: %s", exc)

        if crisis_service is None:
            crisis_task = asyncio.create_task(_crisis_refresh_loop(), name="crisis_refresh")
        else:
            logger.info("Crisis live service scheduler active; runtime polling disabled")

        try:
            await stop_event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass

        # ------------------------------------------------------------------
        # 7. Graceful shutdown (reverse order)
        # ------------------------------------------------------------------
        logger.info("Shutting down ...")

        if crisis_task is not None:
            crisis_task.cancel()
            with suppress(asyncio.CancelledError):
                await crisis_task

        if crisis_service is not None:
            try:
                await crisis_service.stop()
                logger.info("Crisis service stopped")
            except Exception as exc:
                logger.warning("Crisis service stop error: %s", exc)

        if regime_task is not None:
            regime_task.cancel()
            with suppress(asyncio.CancelledError):
                await regime_task

        if regime_service is not None:
            try:
                await regime_service.stop()
                logger.info("Regime service stopped")
            except Exception as exc:
                logger.warning("Regime service stop error: %s", exc)

        for coordinator in reversed(coordinators):
            try:
                await coordinator.stop()
                logger.info("Family '%s' coordinator stopped", getattr(coordinator, "family_id", "?"))
            except Exception as exc:
                logger.warning("Coordinator stop error: %s", exc, exc_info=True)

        if db_pool is not None:
            try:
                await db_pool.close()
                logger.info("Database pool closed")
            except Exception as exc:
                logger.warning("DB pool close error: %s", exc)

        if self.session is not None:
            await self.session.stop()
            logger.info("IB session disconnected")

        logger.info("Runtime shutdown complete")
