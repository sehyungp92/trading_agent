"""Parameter change detection for swing instrumentation."""
from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from libs.instrumentation.config_watcher import ModuleConfigWatcher
from libs.instrumentation.lineage import lineage_from_config

from .config_snapshot import _make_json_safe, snapshot_config_module


@dataclass
class ParameterChangeEvent:
    """A single parameter change detection."""

    bot_id: str
    param_name: str
    old_value: Any
    new_value: Any
    change_source: str = "pr_merge"
    timestamp: str = ""
    config_file: str = ""
    commit_sha: Optional[str] = None
    pr_url: Optional[str] = None
    event_id: str = ""
    event_metadata: dict = field(default_factory=dict)
    config_version_before: str = ""
    config_version_after: str = ""
    portfolio_config_version_before: str = ""
    portfolio_config_version_after: str = ""
    risk_config_version_before: str = ""
    risk_config_version_after: str = ""
    allocation_version_before: str = ""
    allocation_version_after: str = ""
    is_safety_critical: bool = False
    approval_id: str = ""
    proposal_ids: list[str] = field(default_factory=list)
    suggestion_ids: list[str] = field(default_factory=list)
    rollback_of: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigWatcher(ModuleConfigWatcher):
    """Watch swing config modules and YAML files for parameter changes."""

    def __init__(
        self,
        config: dict[str, Any],
        config_modules: Optional[list[str]] = None,
        yaml_paths: Optional[list[Path]] = None,
    ) -> None:
        self.data_source_id = config.get("data_source_id", "ibkr_execution")
        lineage = lineage_from_config(
            config,
            family_id="swing",
            strategy_id=config.get("strategy_id", ""),
        )
        super().__init__(
            bot_id=config["bot_id"],
            config_modules=config_modules or ["strategy.config"],
            data_dir=config["data_dir"],
            lineage=lineage,
            yaml_paths=yaml_paths,
            snapshot_module_fn=_snapshot_swing_module,
            auto_baseline=False,
        )
        self._baseline = self._previous
        self._yaml_baseline = self._yaml_previous

    def _emit_change(
        self,
        param_name: str,
        old_value: Any,
        new_value: Any,
        config_file: str,
        change_source: str,
    ) -> ParameterChangeEvent:
        event = self._make_event(
            config_file,
            param_name,
            old_value,
            new_value,
            change_source=change_source,
            config_file=config_file,
        )
        self._write_event(event)
        return event

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
    ) -> ParameterChangeEvent:
        event = super()._make_event(
            module_name,
            param_name,
            old_value,
            new_value,
            change_source=change_source,
            config_file=config_file,
            source_before=source_before,
            source_after=source_after,
        )
        return ParameterChangeEvent(
            bot_id=str(event.get("bot_id") or self.bot_id),
            param_name=str(event.get("param_name") or param_name),
            old_value=event.get("old_value"),
            new_value=event.get("new_value"),
            change_source=str(event.get("change_source") or change_source),
            timestamp=str(event.get("timestamp") or ""),
            config_file=str(event.get("config_file") or config_file or module_name),
            event_id=str(event.get("event_id") or ""),
            config_version_before=str(event.get("config_version_before") or ""),
            config_version_after=str(event.get("config_version_after") or ""),
            portfolio_config_version_before=str(event.get("portfolio_config_version_before") or ""),
            portfolio_config_version_after=str(event.get("portfolio_config_version_after") or ""),
            risk_config_version_before=str(event.get("risk_config_version_before") or ""),
            risk_config_version_after=str(event.get("risk_config_version_after") or ""),
            allocation_version_before=str(event.get("allocation_version_before") or ""),
            allocation_version_after=str(event.get("allocation_version_after") or ""),
            is_safety_critical=bool(event.get("is_safety_critical")),
            approval_id=str(event.get("approval_id") or ""),
            proposal_ids=list(event.get("proposal_ids") or []),
            suggestion_ids=list(event.get("suggestion_ids") or []),
            rollback_of=str(event.get("rollback_of") or ""),
        )

    def _write_event(self, event: ParameterChangeEvent | dict[str, Any]) -> None:
        payload = event.to_dict() if isinstance(event, ParameterChangeEvent) else event
        super()._write_event(payload)


def _snapshot_swing_module(module: Any) -> dict[str, Any]:
    result = snapshot_config_module(module)
    symbol_config = getattr(module, "SymbolConfig", None)
    fields = getattr(symbol_config, "__dataclass_fields__", None)
    if fields:
        defaults: dict[str, Any] = {}
        for item in fields.values():
            if item.default is not MISSING:
                defaults[item.name] = _make_json_safe(item.default)
            elif item.default_factory is not MISSING:
                defaults[item.name] = _make_json_safe(item.default_factory())
        result["__SymbolConfig_defaults__"] = defaults
    return result
