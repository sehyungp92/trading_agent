"""Deterministic lineage and version helpers for instrumentation events."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


_SECRET_KEY_FRAGMENTS = (
    "secret",
    "password",
    "passwd",
    "token",
    "api_key",
    "apikey",
    "private_key",
    "hmac",
)
_ACCOUNT_KEYS = {
    "account",
    "account_id",
    "broker_account",
    "broker_account_id",
    "ibkr_account",
    "ibkr_account_id",
}


@dataclass(frozen=True)
class LineageContext:
    bot_id: str
    strategy_id: str = ""
    family_id: str = ""
    portfolio_id: str = "default"
    account_alias: str = ""
    strategy_version: str = ""
    config_version: str = ""
    portfolio_config_version: str = ""
    risk_config_version: str = ""
    allocation_version: str = ""
    strategy_registry_version: str = ""
    deployment_id: str = ""
    parameter_set_id: str = ""
    experiment_id: str = ""
    variant_id: str = ""
    code_sha: str = ""
    trace_id: str = ""
    proposal_ids: tuple[str, ...] = ()
    suggestion_ids: tuple[str, ...] = ()
    source_weekly_signal_ids: tuple[str, ...] = ()
    strategy_change_record_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    monthly_search_brief_id: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, (datetime, Path)):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _strip_private_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _strip_private_keys(v)
            for k, v in value.items()
            if not str(k).startswith("_")
        }
    if isinstance(value, list):
        return [_strip_private_keys(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation suitable for hashing."""
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(prefix: str, value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}"


def redact_config(value: Any) -> Any:
    """Redact secrets and raw broker account ids before hashing or emission."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)

    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, raw in value.items():
            key_str = str(key)
            key_l = key_str.lower()
            if key_l == "account_alias":
                redacted[key_str] = redact_config(raw)
            elif key_l in _ACCOUNT_KEYS or any(fragment in key_l for fragment in _SECRET_KEY_FRAGMENTS):
                redacted[key_str] = "<redacted>"
            else:
                redacted[key_str] = redact_config(raw)
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact_config(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(redact_config(item) for item in value)
    return _json_safe(value)


def compute_code_sha(repo_root: str | Path | None = None) -> str:
    """Return the current git SHA, or a fail-open placeholder."""
    env_sha = os.environ.get("CODE_SHA") or os.environ.get("GIT_SHA") or os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha[:40]

    root = Path(repo_root) if repo_root is not None else _repo_root()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        sha = result.stdout.strip()
        if result.returncode == 0 and sha:
            return sha[:40]
    except Exception:
        pass
    return "unknown"


def compute_strategy_registry_version(registry: Any) -> str:
    try:
        from libs.config.registry import build_registry_artifact

        value = build_registry_artifact(registry)
    except Exception:
        value = registry
    return stable_hash("registry_", redact_config(value))


def compute_portfolio_config_version(portfolio_config: Any) -> str:
    return stable_hash("pcfg_", redact_config(portfolio_config))


def compute_allocation_version(portfolio_config: Any, registry: Any) -> str:
    portfolio = _json_safe(portfolio_config)
    registry_value = _json_safe(registry)
    capital = portfolio.get("capital", {}) if isinstance(portfolio, dict) else {}
    strategies = registry_value.get("strategies", {}) if isinstance(registry_value, dict) else {}
    enabled = {
        sid: {
            "family": item.get("family"),
            "enabled": item.get("enabled", True),
            "allocation": item.get("allocation", {}),
        }
        for sid, item in sorted(strategies.items())
        if isinstance(item, dict) and item.get("enabled", True)
    }
    return stable_hash("alloc_", redact_config({"capital": capital, "enabled_strategies": enabled}))


def compute_risk_config_version(
    portfolio_config: Any,
    portfolio_rules_config: Any = None,
    registry: Any = None,
) -> str:
    portfolio = _json_safe(portfolio_config)
    registry_value = _json_safe(registry)
    strategies = registry_value.get("strategies", {}) if isinstance(registry_value, dict) else {}
    strategy_risk = {
        sid: {
            "family": item.get("family"),
            "enabled": item.get("enabled", True),
            "risk": item.get("risk", {}),
        }
        for sid, item in sorted(strategies.items())
        if isinstance(item, dict)
    }
    value = {
        "portfolio_risk": portfolio.get("risk", {}) if isinstance(portfolio, dict) else {},
        "drawdown_tiers": portfolio.get("drawdown_tiers", []) if isinstance(portfolio, dict) else [],
        "coordination": portfolio.get("coordination", {}) if isinstance(portfolio, dict) else {},
        "portfolio_rules_config": _strip_private_keys(_json_safe(portfolio_rules_config or {})),
        "strategy_risk": strategy_risk,
    }
    return stable_hash("risk_", redact_config(value))


def compute_strategy_config_version(
    strategy_manifest: Any,
    effective_strategy_config: Any,
    lineage_parts: Any = None,
) -> str:
    return stable_hash(
        "cfg_",
        redact_config(
            {
                "strategy_manifest": strategy_manifest or {},
                "effective_strategy_config": effective_strategy_config or {},
                "lineage_parts": lineage_parts or {},
            }
        ),
    )


def _env_first(env: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = env.get(name)
        if value:
            return str(value)
    return default


def resolve_deployment_id(lineage: LineageContext, env: Mapping[str, str] | None = None) -> str:
    env = env or os.environ
    explicit = _env_first(env, "DEPLOYMENT_ID", "INSTRUMENTATION_DEPLOYMENT_ID")
    if explicit:
        return explicit
    now = datetime.now(timezone.utc)
    suffix = stable_hash(
        "",
        {
            "bot_id": lineage.bot_id,
            "strategy_id": lineage.strategy_id,
            "family_id": lineage.family_id,
            "config_version": lineage.config_version,
            "code_sha": lineage.code_sha,
            "started_at": now.isoformat(),
        },
        length=12,
    )
    return f"dep_{now.strftime('%Y_%m_%d')}_{suffix}"


def lineage_from_runtime(
    *,
    bot_id: str,
    strategy_id: str = "",
    family_id: str = "",
    portfolio_id: str = "",
    account_alias: str = "",
    strategy_version: str = "",
    config_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    strategy_manifest: Any = None,
    effective_strategy_config: Any = None,
    portfolio_config: Any = None,
    portfolio_rules_config: Any = None,
    parameter_set: Any = None,
    experiment_id: str = "",
    variant_id: str = "",
    extras: dict[str, Any] | None = None,
) -> LineageContext:
    """Build a fail-open lineage context from runtime config and repo config files."""
    env = env or os.environ
    root = Path(repo_root) if repo_root is not None else _repo_root()
    cfg_dir = Path(config_dir) if config_dir is not None else root / "config"

    registry = None
    if portfolio_config is None or strategy_manifest is None:
        try:
            from libs.config.loader import load_portfolio_config, load_strategy_registry

            if portfolio_config is None:
                portfolio_config = load_portfolio_config(cfg_dir)
            registry = load_strategy_registry(cfg_dir)
            if strategy_manifest is None and strategy_id:
                strategy_manifest = registry.strategies.get(strategy_id)
        except Exception:
            registry = None

    if registry is None:
        try:
            from libs.config.loader import load_strategy_registry

            registry = load_strategy_registry(cfg_dir)
        except Exception:
            registry = {}

    manifest_safe = _json_safe(strategy_manifest or {})
    if not family_id:
        family_id = str(manifest_safe.get("family", "") if isinstance(manifest_safe, dict) else "")
    if not strategy_version:
        artifact_config = manifest_safe.get("artifact_config", {}) if isinstance(manifest_safe, dict) else {}
        strategy_version = (
            _env_first(env, "STRATEGY_VERSION", "INSTRUMENTATION_STRATEGY_VERSION")
            or str(artifact_config.get("version", "") if isinstance(artifact_config, dict) else "")
            or (f"{strategy_id}.unversioned" if strategy_id else "")
        )

    portfolio_id = portfolio_id or _env_first(env, "PORTFOLIO_ID", "TRADING_PORTFOLIO_ID", default="paper_default")
    account_alias = account_alias or _env_first(
        env,
        "ACCOUNT_ALIAS",
        "TRADING_ACCOUNT_ALIAS",
        "BROKER_ACCOUNT_ALIAS",
        default="",
    )
    code_sha = compute_code_sha(root)

    strategy_registry_version = compute_strategy_registry_version(registry)
    portfolio_config_version = compute_portfolio_config_version(portfolio_config or {})
    allocation_version = compute_allocation_version(portfolio_config or {}, registry)
    risk_config_version = compute_risk_config_version(portfolio_config or {}, portfolio_rules_config, registry)

    effective_strategy_config = effective_strategy_config or {
        "manifest": manifest_safe,
        "runtime": {
            "strategy_id": strategy_id,
            "family_id": family_id,
            "bot_id": bot_id,
        },
    }
    parameter_source = parameter_set if parameter_set is not None else effective_strategy_config
    parameter_set_id = _env_first(env, "PARAMETER_SET_ID", "INSTRUMENTATION_PARAMETER_SET_ID")
    if not parameter_set_id:
        parameter_set_id = stable_hash("param_", redact_config(parameter_source))

    lineage_parts = {
        "portfolio_config_version": portfolio_config_version,
        "risk_config_version": risk_config_version,
        "allocation_version": allocation_version,
        "strategy_registry_version": strategy_registry_version,
    }
    config_version = compute_strategy_config_version(
        strategy_manifest,
        effective_strategy_config,
        lineage_parts,
    )

    base = LineageContext(
        bot_id=bot_id,
        strategy_id=strategy_id,
        family_id=family_id,
        portfolio_id=portfolio_id,
        account_alias=account_alias,
        strategy_version=strategy_version,
        config_version=config_version,
        portfolio_config_version=portfolio_config_version,
        risk_config_version=risk_config_version,
        allocation_version=allocation_version,
        strategy_registry_version=strategy_registry_version,
        parameter_set_id=parameter_set_id,
        experiment_id=experiment_id or _env_first(env, "EXPERIMENT_ID"),
        variant_id=variant_id or _env_first(env, "EXPERIMENT_VARIANT", "VARIANT_ID"),
        code_sha=code_sha,
        extras=extras or {},
    )
    deployment_id = resolve_deployment_id(base, env)
    trace_id = _env_first(env, "TRACE_ID", "INSTRUMENTATION_TRACE_ID")
    if not trace_id:
        trace_id = stable_hash("trace_", {"deployment_id": deployment_id, "strategy_id": strategy_id})
    return replace(base, deployment_id=deployment_id, trace_id=trace_id)


def lineage_from_config(
    config: Mapping[str, Any],
    *,
    family_id: str = "",
    strategy_id: str = "",
    portfolio_rules_config: Any = None,
) -> LineageContext:
    """Build lineage from an instrumentation config dict."""
    existing = config.get("lineage")
    if isinstance(existing, LineageContext):
        return existing
    if isinstance(existing, Mapping):
        values = {k: v for k, v in existing.items() if k in LineageContext.__dataclass_fields__}
        values["bot_id"] = str(values.get("bot_id") or config.get("bot_id") or "")
        if family_id and not values.get("family_id"):
            values["family_id"] = family_id
        if strategy_id and not values.get("strategy_id"):
            values["strategy_id"] = strategy_id
        for key in (
            "proposal_ids",
            "suggestion_ids",
            "source_weekly_signal_ids",
            "strategy_change_record_ids",
            "candidate_ids",
        ):
            if key in values and not isinstance(values[key], tuple):
                values[key] = tuple(str(item) for item in values[key] or ())
        return LineageContext(**values)

    strategy = strategy_id or str(config.get("strategy_id") or "")
    return lineage_from_runtime(
        bot_id=str(config.get("bot_id") or ""),
        strategy_id=strategy,
        family_id=family_id or str(config.get("family_id") or ""),
        portfolio_id=str(config.get("portfolio_id") or ""),
        account_alias=str(config.get("account_alias") or ""),
        strategy_version=str(config.get("strategy_version") or ""),
        effective_strategy_config=redact_config(
            {
                key: value
                for key, value in dict(config).items()
                if key not in {"lineage", "sidecar", "logging"}
            }
        ),
        portfolio_rules_config=portfolio_rules_config,
        experiment_id=str(config.get("experiment_id") or ""),
        variant_id=str(config.get("experiment_variant") or config.get("variant_id") or ""),
    )


def lineage_to_payload(lineage: LineageContext | Mapping[str, Any] | None) -> dict[str, Any]:
    if lineage is None:
        return {}
    if isinstance(lineage, LineageContext):
        data = dataclasses.asdict(lineage)
    else:
        data = dict(lineage)
    data = {key: _json_safe(value) for key, value in data.items()}
    extras = data.pop("extras", {}) or {}
    if isinstance(extras, Mapping):
        for key, value in extras.items():
            data.setdefault(str(key), _json_safe(value))
    return data


def merge_lineage(
    payload: Mapping[str, Any],
    lineage: LineageContext | Mapping[str, Any] | None,
    scope: str,
    schema_version: str,
    event_type: str,
) -> dict[str, Any]:
    from .event_contract import merge_lineage as _merge_lineage

    return _merge_lineage(payload, lineage, scope, schema_version, event_type)
