"""Generate canonical promotion manifests and materialized effective live configs."""

from __future__ import annotations

import ast
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trading_contracts.canonical import canonical_json_sha256, file_sha256, load_json

from trading_config.models import CONFIG_MERGE_ORDER, EffectiveConfigSnapshot


CRYPTO_PUBLIC_LIVE_CONFIG_EXCLUDED_FIELDS = {
    "wallet_address",
    "private_key",
    "relay_url",
    "relay_secret",
    "postgres_dsn",
}


@dataclass(frozen=True)
class BotConfigSpec:
    bot_id: str
    canonical_dir: str
    generated_path: str
    source_files: tuple[tuple[str, str], ...]
    notes: tuple[str, ...] = ()


BOT_SPECS = (
    BotConfigSpec(
        bot_id="ibkr",
        canonical_dir="contracts/promotions/ibkr",
        generated_path="deployments/ibkr/generated/strategies.effective.json",
        source_files=(("ibkr_strategies", "bots/ibkr_trading/config/strategies.yaml"),),
    ),
    BotConfigSpec(
        bot_id="crypto",
        canonical_dir="contracts/promotions/crypto",
        generated_path="deployments/crypto/generated/live_config.effective.json",
        source_files=(
            ("crypto_live_config", "bots/crypto_trader/config/live_config.example.json"),
            ("crypto_strategy_breakout", "bots/crypto_trader/config/strategies/breakout.json"),
            ("crypto_strategy_momentum", "bots/crypto_trader/config/strategies/momentum.json"),
            ("crypto_strategy_trend", "bots/crypto_trader/config/strategies/trend.json"),
        ),
        notes=(
            "Portfolio round 3 uses explicit supersession evidence for the missing legacy rounds_manifest.json.",
        ),
    ),
    BotConfigSpec(
        bot_id="k_stock",
        canonical_dir="contracts/promotions/k_stock",
        generated_path="deployments/k_stock/generated/olr_kalcb.effective.json",
        source_files=(
            ("k_stock_kalcb_live", "bots/k_stock_trader/config/kalcb.yaml"),
            ("k_stock_kalcb_optimizer", "bots/k_stock_trader/config/optimization/kalcb.yaml"),
            ("k_stock_olr_optimizer", "bots/k_stock_trader/config/optimization/olr.yaml"),
            (
                "k_stock_olr_universe",
                "bots/k_stock_trader/config/olr_kalcb/olr_deployment_universe_103.yaml",
            ),
            ("k_stock_kalcb_defaults", "bots/k_stock_trader/strategy_kalcb/config.py"),
            (
                "k_stock_kalcb_phase_base",
                "bots/k_stock_trader/backtests/strategies/kalcb/phase_candidates.py",
            ),
        ),
        notes=("KALCB frontier.size must remain aligned at 103 across live/default/optimizer sources.",),
    ),
)


def generate_effective_configs(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    generated: list[dict[str, str]] = []
    for spec in BOT_SPECS:
        promotions = _canonical_promotions(root, spec)
        snapshot = build_effective_snapshot(root, spec, promotions)
        output = root / spec.generated_path
        _preserve_generated_at_if_hash_unchanged(output, snapshot)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        generated.append({"bot_id": spec.bot_id, "path": spec.generated_path})
    return {"generated": generated}


def _canonical_promotions(root: Path, spec: BotConfigSpec) -> list[Path]:
    canonical_dir = root / spec.canonical_dir
    paths = sorted(canonical_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"canonical promotion manifests are missing for {spec.bot_id}: "
            f"{canonical_dir.relative_to(root).as_posix()}"
        )
    return paths


def build_effective_snapshot(
    root: Path,
    spec: BotConfigSpec,
    promotion_paths: list[Path],
) -> EffectiveConfigSnapshot:
    source_files = [_source_reference(root, role, path) for role, path in spec.source_files]
    promotions = [_promotion_reference(root, path) for path in promotion_paths]
    materialized = _materialized_config(root, spec, promotion_paths)
    materialized_hash = canonical_json_sha256(materialized)
    hash_payload = {
        "bot_id": spec.bot_id,
        "merge_order": CONFIG_MERGE_ORDER,
        "source_files": source_files,
        "promotion_manifests": promotions,
        "materialized_config_hash": materialized_hash,
    }
    return EffectiveConfigSnapshot(
        bot_id=spec.bot_id,
        source_files=source_files,
        promotion_manifests=promotions,
        materialized_config=materialized,
        materialized_config_hash=materialized_hash,
        effective_config_hash=canonical_json_sha256(hash_payload),
        notes=list(spec.notes),
    )


def _preserve_generated_at_if_hash_unchanged(output: Path, snapshot: EffectiveConfigSnapshot) -> None:
    if not output.exists():
        return
    try:
        existing = EffectiveConfigSnapshot.model_validate(load_json(output))
    except Exception:
        return
    if existing.effective_config_hash == snapshot.effective_config_hash:
        snapshot.generated_at = existing.generated_at


def _source_reference(root: Path, role: str, relative_path: str) -> dict[str, str]:
    path = root / relative_path
    record = {
        "role": role,
        "path": relative_path,
        "sha256": file_sha256(path),
        "canonical_json_sha256": "",
    }
    if path.suffix.lower() == ".json":
        record["canonical_json_sha256"] = canonical_json_sha256(load_json(path))
    return record


def _promotion_reference(root: Path, path: Path) -> dict[str, str]:
    payload = load_json(path)
    return {
        "strategy_id": str(payload.get("strategy_id") or path.stem),
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "canonical_json_sha256": canonical_json_sha256(payload),
        "promotion_state": str(payload.get("promotion_state") or ""),
    }


def _materialized_config(root: Path, spec: BotConfigSpec, promotion_paths: list[Path]) -> dict[str, Any]:
    promotions = [load_json(path) for path in promotion_paths]
    base = {
        "schema_version": "materialized_effective_config.v1",
        "bot_id": spec.bot_id,
        "merge_order": list(CONFIG_MERGE_ORDER),
    }
    if spec.bot_id == "ibkr":
        return {**base, **_materialize_ibkr(root, promotions)}
    if spec.bot_id == "crypto":
        return {**base, **_materialize_crypto(root, promotions)}
    if spec.bot_id == "k_stock":
        return {**base, **_materialize_k_stock(root, promotions)}
    raise ValueError(f"unsupported bot_id={spec.bot_id!r}")


def _materialize_ibkr(root: Path, promotions: list[dict[str, Any]]) -> dict[str, Any]:
    runtime = _load_yaml(root / "bots/ibkr_trading/config/strategies.yaml")
    strategies = runtime.get("strategies", {}) if isinstance(runtime, dict) else {}
    return {
        "connection_groups": runtime.get("connection_groups", {}) if isinstance(runtime, dict) else {},
        "strategies": [
            _strategy_record(
                root,
                promotion,
                strategies.get(str(promotion.get("strategy_id")), {}),
            )
            for promotion in promotions
        ],
    }


def _materialize_crypto(root: Path, promotions: list[dict[str, Any]]) -> dict[str, Any]:
    live_config = load_json(root / "bots/crypto_trader/config/live_config.example.json")
    strategy_dir = root / "bots/crypto_trader/config/strategies"
    strategy_configs = {
        path.stem: load_json(path)
        for path in sorted(strategy_dir.glob("*.json"))
        if ".pre_" not in path.name
    }
    records = []
    for promotion in promotions:
        strategy_id = str(promotion.get("strategy_id"))
        if strategy_id == "portfolio_round_3":
            records.append({
                "strategy_id": strategy_id,
                "promotion_state": promotion.get("promotion_state", ""),
                "portfolio_runtime_config": {
                    key: live_config.get(key)
                    for key in (
                        "strategy_configs",
                        "portfolio_config_path",
                        "deployment_manifest_path",
                    )
                },
                "promotion_portfolio_round": promotion.get("portfolio_round", {}),
            })
            continue
        records.append(_strategy_record(root, promotion, strategy_configs.get(strategy_id, {})))
    return {
        "live_config": live_config,
        "runtime_config_contract": _crypto_runtime_config_contract(live_config),
        "strategies": records,
    }


def _crypto_runtime_config_contract(live_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "crypto_runtime_config_contract.v1",
        "mounted_config_path": "config/live_config.json",
        "public_live_config_sha256": canonical_json_sha256(
            _crypto_public_live_config(live_config),
        ),
        "public_hash_excludes": sorted(CRYPTO_PUBLIC_LIVE_CONFIG_EXCLUDED_FIELDS),
        "required_non_empty_fields": [
            "wallet_address",
            "private_key",
            "relay_url",
            "relay_secret",
            "bot_id",
        ],
        "sidecar_forwarding_required": True,
    }


def _crypto_public_live_config(live_config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in live_config.items()
        if key not in CRYPTO_PUBLIC_LIVE_CONFIG_EXCLUDED_FIELDS
    }


def _materialize_k_stock(root: Path, promotions: list[dict[str, Any]]) -> dict[str, Any]:
    live_kalcb = _load_yaml(root / "bots/k_stock_trader/config/kalcb.yaml")
    optimizer_kalcb = _load_yaml(root / "bots/k_stock_trader/config/optimization/kalcb.yaml")
    optimizer_olr = _load_yaml(root / "bots/k_stock_trader/config/optimization/olr.yaml")
    runtime_by_strategy = {
        "kalcb": live_kalcb,
        "olr": optimizer_olr,
        "olr_kalcb_portfolio": {
            "kalcb": live_kalcb,
            "olr_optimizer": optimizer_olr,
        },
    }
    return {
        "frontier_size_evidence": _kalcb_frontier_size_evidence(root),
        "optimizer_base": {"kalcb": optimizer_kalcb, "olr": optimizer_olr},
        "strategies": [
            _strategy_record(
                root,
                promotion,
                runtime_by_strategy.get(str(promotion.get("strategy_id")), {}),
            )
            for promotion in promotions
        ],
    }


def _strategy_record(
    root: Path,
    promotion: dict[str, Any],
    runtime_overlay: dict[str, Any],
) -> dict[str, Any]:
    optimizer_path = str((promotion.get("optimizer_round") or {}).get("optimized_config_path") or "")
    optimized = load_json(root / optimizer_path) if optimizer_path and (root / optimizer_path).exists() else {}
    optimized_values = _optimized_runtime_values(optimized)
    effective = _deep_merge(optimized_values, runtime_overlay)
    record = {
        "strategy_id": str(promotion.get("strategy_id") or ""),
        "promotion_state": str(promotion.get("promotion_state") or ""),
        "approval_status": str((promotion.get("approval") or {}).get("status") or ""),
        "optimizer_round": promotion.get("optimizer_round") or {},
        "runtime_overlay": runtime_overlay,
        "latest_optimized_config": optimized_values,
        "effective_config": effective,
        "effective_config_hash": canonical_json_sha256(effective),
    }
    if optimizer_path:
        record["optimized_config_path"] = optimizer_path
    overridden = _overridden_values(optimized_values, effective)
    if overridden:
        record["runtime_overrides"] = overridden
    return record


def _optimized_runtime_values(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("strategy"), dict):
        return deepcopy(payload["strategy"])
    if isinstance(payload.get("mutations"), dict):
        return _undot(payload["mutations"])
    config = payload.get("config") if isinstance(payload.get("config"), dict) else payload
    return _undot(config)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return value if isinstance(value, dict) else {}


def _deep_merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = deepcopy(base)
        for key, value in overlay.items():
            merged[key] = _deep_merge(merged.get(key), value)
        return merged
    return deepcopy(overlay if overlay is not None else base)


def _undot(values: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or "." not in key:
            result[key] = value
            continue
        cursor = result
        parts = key.split(".")
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                break
            cursor = child
        else:
            cursor[parts[-1]] = value
    return result


def _overridden_values(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for path, old_value in _iter_leaf_paths(before):
        new_value = _get_path(after, path)
        if new_value != old_value:
            overrides[".".join(path)] = {"optimized": old_value, "effective": new_value}
    return overrides


def _iter_leaf_paths(value: Any, prefix: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_leaf_paths(child, (*prefix, str(key)))
    elif prefix:
        yield prefix, value


def _get_path(value: Any, path: tuple[str, ...]) -> Any:
    cursor = value
    for part in path:
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _kalcb_frontier_size_evidence(root: Path) -> dict[str, Any]:
    live_path = root / "bots/k_stock_trader/config/kalcb.yaml"
    optimizer_path = root / "bots/k_stock_trader/config/optimization/kalcb.yaml"
    universe_path = root / "bots/k_stock_trader/config/olr_kalcb/olr_deployment_universe_103.yaml"
    default_path = root / "bots/k_stock_trader/strategy_kalcb/config.py"
    phase_base_path = root / "bots/k_stock_trader/backtests/strategies/kalcb/phase_candidates.py"
    values = {
        "live_config": _get_path(_load_yaml(live_path), ("kalcb", "frontier", "size")),
        "optimizer_base_mutation": _get_path(
            _load_yaml(optimizer_path),
            ("initial_mutations", "kalcb.frontier.size"),
        ),
        "strategy_default": _class_attribute(default_path, "KALCBConfig", "frontier_size"),
        "phase_optimizer_base": _module_dict_value(
            phase_base_path,
            "BASE_MUTATIONS",
            "kalcb.frontier.size",
        ),
        "deployment_universe": _get_path(_load_yaml(universe_path), ("symbol_count",)),
    }
    return {
        "values": values,
        "status": "aligned" if set(values.values()) == {103} else "blocked_alignment_finding",
        "paths": {
            "live_config": live_path.relative_to(root).as_posix(),
            "optimizer_base_mutation": optimizer_path.relative_to(root).as_posix(),
            "strategy_default": default_path.relative_to(root).as_posix(),
            "phase_optimizer_base": phase_base_path.relative_to(root).as_posix(),
            "deployment_universe": universe_path.relative_to(root).as_posix(),
        },
    }


def _class_attribute(path: Path, class_name: str, attr_name: str) -> Any:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            target = None
            value = None
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                target, value = item.target.id, item.value
            elif isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                target, value = item.targets[0].id, item.value
            if target == attr_name and value is not None:
                return ast.literal_eval(value)
    return None


def _module_dict_value(path: Path, dict_name: str, key: str) -> Any:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == dict_name
            for target in node.targets
        ):
            payload = ast.literal_eval(node.value)
            return payload.get(key) if isinstance(payload, dict) else None
    return None
