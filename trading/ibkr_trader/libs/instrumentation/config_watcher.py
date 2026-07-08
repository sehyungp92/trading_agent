"""Shared module/YAML config watcher for strategy instrumentation."""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config_snapshot import default_yaml_watch_paths, flatten_mapping, snapshot_yaml_file
from .event_contract import enrich_payload
from .lineage import (
    compute_portfolio_config_version,
    lineage_to_payload,
    stable_hash,
)


logger = logging.getLogger("instrumentation.config_watcher")

SnapshotModuleFn = Callable[[Any], dict[str, Any]]

_SAFETY_CRITICAL_PARAMS = {
    "risk_per_trade",
    "max_position_size",
    "kill_switch_enabled",
    "trailing_stop_pct",
    "max_drawdown_pct",
    "leverage_limit",
    "HEAT_CAP_R",
    "DAILY_STOP_R",
    "WEEKLY_STOP_R",
    "MAX_POSITION_R",
}
_SAFETY_CRITICAL_PARAM_KEYS = {param.lower() for param in _SAFETY_CRITICAL_PARAMS}


class ModuleConfigWatcher:
    """Watch Python config modules and YAML inputs, emitting parameter_change events."""

    def __init__(
        self,
        *,
        bot_id: str,
        config_modules: list[str],
        data_dir: str | Path,
        lineage: dict | object | None = None,
        yaml_paths: list[str | Path] | None = None,
        config_dir: str | Path | None = None,
        snapshot_module_fn: SnapshotModuleFn,
        auto_baseline: bool = True,
    ) -> None:
        self.bot_id = bot_id
        self._bot_id = bot_id
        self._config_modules = list(config_modules)
        self.config_modules = self._config_modules
        self._data_dir = Path(data_dir) / "config_changes"
        self.data_dir = self._data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._lineage = lineage or {"bot_id": bot_id}
        self._snapshot_module_fn = snapshot_module_fn
        self._previous: dict[str, dict[str, Any]] = {}
        self._yaml_previous: dict[str, dict[str, Any]] = {}
        self._yaml_paths = [
            Path(path)
            for path in (
                yaml_paths
                if yaml_paths is not None
                else default_yaml_watch_paths(config_dir)
            )
        ]
        self.yaml_paths = self._yaml_paths

        if auto_baseline:
            self.take_baseline()

    def take_baseline(self) -> None:
        """Snapshot the currently watched sources."""
        for module_name in self._config_modules:
            self._previous[module_name] = self._snapshot_module(module_name)
        for path in self._yaml_paths:
            self._yaml_previous[str(path)] = self._snapshot_yaml(path)

    def _snapshot_module(self, module_name: str) -> dict[str, Any]:
        try:
            module = importlib.import_module(module_name)
            try:
                importlib.reload(module)
            except Exception:
                pass
            return self._snapshot_module_fn(module)
        except Exception:
            return {}

    @staticmethod
    def _snapshot_yaml(path: Path) -> dict[str, Any]:
        return flatten_mapping(snapshot_yaml_file(path))

    def check(self) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        for module_name in self._config_modules:
            current = self._snapshot_module(module_name)
            previous = self._previous.get(module_name, {})
            changes.extend(
                self._diff_source(
                    source_name=module_name,
                    current=current,
                    previous=previous,
                    change_source="hot_reload",
                )
            )
            self._previous[module_name] = current

        for path in self._yaml_paths:
            source_name = str(path)
            current = self._snapshot_yaml(path)
            previous = self._yaml_previous.get(source_name, {})
            changes.extend(
                self._diff_source(
                    source_name=source_name,
                    current=current,
                    previous=previous,
                    change_source="config_file",
                )
            )
            self._yaml_previous[source_name] = current

        return changes

    def _diff_source(
        self,
        *,
        source_name: str,
        current: dict[str, Any],
        previous: dict[str, Any],
        change_source: str,
    ) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        for key in set(current.keys()) | set(previous.keys()):
            old_value = previous.get(key)
            new_value = current.get(key)
            if old_value == new_value:
                continue
            event = self._make_event(
                source_name,
                key,
                old_value,
                new_value,
                change_source=change_source,
                config_file=source_name,
                source_before=previous,
                source_after=current,
            )
            changes.append(event)
            self._write_event(event)
        return changes

    def _make_event(
        self,
        module_name: str,
        param_name: str,
        old_value: Any,
        new_value: Any,
        *,
        change_source: str = "hot_reload",
        config_file: str | None = None,
        source_before: dict[str, Any] | None = None,
        source_after: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        source_name = config_file or module_name
        raw = f"{self._bot_id}|{timestamp}|parameter_change|{source_name}:{param_name}"
        lineage = lineage_to_payload(self._lineage)
        before_snapshot = source_before if source_before is not None else {param_name: old_value}
        after_snapshot = source_after if source_after is not None else {param_name: new_value}
        version_fields = _version_fields(
            source_name,
            param_name,
            before_snapshot,
            after_snapshot,
            lineage,
            before_graph=self._effective_watch_graph(source_name, before_snapshot),
            after_graph=self._effective_watch_graph(source_name, after_snapshot),
        )
        return {
            "bot_id": self._bot_id,
            "timestamp": timestamp,
            "event_id": hashlib.sha256(raw.encode()).hexdigest()[:16],
            "event_type": "parameter_change",
            "module": module_name,
            "config_file": source_name,
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
            **version_fields,
            "change_source": change_source,
            "is_safety_critical": _is_safety_critical(param_name),
            "approval_id": lineage.get("approval_id", ""),
            "proposal_ids": list(lineage.get("proposal_ids") or ()),
            "suggestion_ids": list(lineage.get("suggestion_ids") or ()),
            "rollback_of": lineage.get("rollback_of", ""),
        }

    def _write_event(self, event: dict[str, Any]) -> None:
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = self._data_dir / f"config_changes_{today}.jsonl"
            payload = enrich_payload(
                event,
                lineage=self._lineage,
                event_type="parameter_change",
                scope="strategy",
            )
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str) + "\n")
        except Exception as exc:
            logger.debug("Failed to write parameter change event: %s", exc)

    @staticmethod
    def _is_safety_critical(param_name: str) -> bool:
        return _is_safety_critical(param_name)

    def _effective_watch_graph(self, source_name: str, replacement: dict[str, Any]) -> dict[str, Any]:
        modules = {key: dict(value or {}) for key, value in self._previous.items()}
        yaml_sources = {key: dict(value or {}) for key, value in self._yaml_previous.items()}
        if source_name in yaml_sources or Path(source_name).suffix.lower() in {".yaml", ".yml"}:
            yaml_sources[source_name] = dict(replacement or {})
        else:
            modules[source_name] = dict(replacement or {})
        lineage = lineage_to_payload(self._lineage)
        return {
            "bot_id": self._bot_id,
            "strategy_id": lineage.get("strategy_id", ""),
            "family_id": lineage.get("family_id", ""),
            "portfolio_id": lineage.get("portfolio_id", ""),
            "modules": modules,
            "yaml_sources": yaml_sources,
        }


def _is_safety_critical(param_name: str) -> bool:
    lowered = param_name.lower()
    if lowered in _SAFETY_CRITICAL_PARAM_KEYS:
        return True
    return any(
        token in lowered
        for token in ("risk", "stop", "drawdown", "heat", "position", "kill", "leverage")
    )


def _version_fields(
    source_name: str,
    param_name: str,
    before_snapshot: dict[str, Any],
    after_snapshot: dict[str, Any],
    lineage: dict[str, Any],
    *,
    before_graph: dict[str, Any] | None = None,
    after_graph: dict[str, Any] | None = None,
) -> dict[str, str]:
    before_source = {"source": source_name, "snapshot": before_snapshot}
    after_source = {"source": source_name, "snapshot": after_snapshot}
    before_effective = before_graph or before_source
    after_effective = after_graph or after_source
    fields = {
        "config_version_before": stable_hash("cfg_", _domain_graph(before_effective, "config")),
        "config_version_after": stable_hash("cfg_", _domain_graph(after_effective, "config")),
        "portfolio_config_version_before": str(lineage.get("portfolio_config_version", "") or ""),
        "portfolio_config_version_after": str(lineage.get("portfolio_config_version", "") or ""),
        "risk_config_version_before": str(lineage.get("risk_config_version", "") or ""),
        "risk_config_version_after": str(lineage.get("risk_config_version", "") or ""),
        "allocation_version_before": str(lineage.get("allocation_version", "") or ""),
        "allocation_version_after": str(lineage.get("allocation_version", "") or ""),
    }

    source_key = Path(source_name).name.lower()
    risk_relevant = _is_safety_critical(param_name) or source_key in {
        "portfolio.yaml",
        "strategies.yaml",
        "sector_map.yaml",
        "contracts.yaml",
        "event_calendar.yaml",
        "routing.yaml",
    }
    if source_key == "portfolio.yaml":
        fields["portfolio_config_version_before"] = compute_portfolio_config_version(
            _domain_graph(before_effective, "portfolio")
        )
        fields["portfolio_config_version_after"] = compute_portfolio_config_version(
            _domain_graph(after_effective, "portfolio")
        )
    if risk_relevant:
        fields["risk_config_version_before"] = stable_hash("risk_", _domain_graph(before_effective, "risk"))
        fields["risk_config_version_after"] = stable_hash("risk_", _domain_graph(after_effective, "risk"))
    if source_key in {"portfolio.yaml", "strategies.yaml"}:
        fields["allocation_version_before"] = stable_hash("alloc_", _domain_graph(before_effective, "allocation"))
        fields["allocation_version_after"] = stable_hash("alloc_", _domain_graph(after_effective, "allocation"))
    return fields


def _domain_graph(graph: dict[str, Any], domain: str) -> dict[str, Any]:
    yaml_sources = dict(graph.get("yaml_sources", {})) if isinstance(graph, dict) else {}
    modules = dict(graph.get("modules", {})) if isinstance(graph, dict) else {}
    if not yaml_sources and not modules:
        return graph

    def yaml_by_name(names: set[str]) -> dict[str, Any]:
        return {
            key: value
            for key, value in sorted(yaml_sources.items())
            if Path(key).name.lower() in names
        }

    base = {
        "bot_id": graph.get("bot_id", ""),
        "strategy_id": graph.get("strategy_id", ""),
        "family_id": graph.get("family_id", ""),
        "portfolio_id": graph.get("portfolio_id", ""),
    }
    if domain == "portfolio":
        return {
            **base,
            "yaml_sources": yaml_by_name({"portfolio.yaml", "strategies.yaml"}),
        }
    if domain == "risk":
        return {
            **base,
            "modules": modules,
            "yaml_sources": yaml_by_name({
                "portfolio.yaml",
                "strategies.yaml",
                "sector_map.yaml",
                "contracts.yaml",
                "event_calendar.yaml",
                "routing.yaml",
            }),
        }
    if domain == "allocation":
        return {
            **base,
            "yaml_sources": yaml_by_name({"portfolio.yaml", "strategies.yaml"}),
        }
    return {**base, "modules": modules, "yaml_sources": yaml_sources}


__all__ = ["ModuleConfigWatcher"]
