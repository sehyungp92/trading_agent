"""Metadata registry and optional factory registration for strategies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Awaitable, Callable

from .models import StrategyManifest, StrategyRegistryConfig

StrategyFactory = Callable[..., Awaitable[object]]
STRATEGY_FACTORIES: dict[str, StrategyFactory] = {}


def register(strategy_id: str) -> Callable[[StrategyFactory], StrategyFactory]:
    """Register a future runtime factory for a strategy id."""

    def decorator(factory_fn: StrategyFactory) -> StrategyFactory:
        existing = STRATEGY_FACTORIES.get(strategy_id)
        if existing is not None:
            raise ValueError(
                f"Duplicate factory for {strategy_id!r}: "
                f"existing={existing.__module__}.{existing.__qualname__}, "
                f"new={factory_fn.__module__}.{factory_fn.__qualname__}"
            )
        STRATEGY_FACTORIES[strategy_id] = factory_fn
        return factory_fn

    return decorator


def get_factory(strategy_id: str) -> StrategyFactory | None:
    return STRATEGY_FACTORIES.get(strategy_id)


def build_registry_artifact(registry: StrategyRegistryConfig) -> dict:
    strategies: list[dict] = []
    for manifest in registry.strategies.values():
        strategies.append(_manifest_to_artifact(manifest))
    return {
        "connection_groups": sorted(registry.connection_groups.keys()),
        "strategies": sorted(strategies, key=lambda item: item["strategy_id"]),
    }


def write_registry_artifact(registry: StrategyRegistryConfig, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_registry_artifact(registry)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _manifest_to_artifact(manifest: StrategyManifest) -> dict:
    return {
        "strategy_id": manifest.strategy_id,
        "system_id": manifest.system_id,
        "family": manifest.family,
        "display_name": manifest.display_name,
        "connection_group": manifest.connection_group,
        "enabled": manifest.enabled,
        "paper_mode": manifest.paper_mode,
        "asset_class": manifest.asset_class,
        "symbols": manifest.symbols,
        "dashboard_metadata": manifest.dashboard_metadata,
        "deployment_tags": manifest.deployment_tags,
    }

