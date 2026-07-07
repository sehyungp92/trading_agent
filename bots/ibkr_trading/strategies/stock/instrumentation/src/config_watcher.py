"""Parameter change detection for stock instrumentation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from libs.instrumentation.config_watcher import ModuleConfigWatcher

from .config_snapshot import snapshot_config_module


class ConfigWatcher(ModuleConfigWatcher):
    """Monitor stock config modules and shared YAML inputs."""

    def __init__(
        self,
        bot_id: str,
        config_modules: list[str],
        data_dir: str | Path,
        lineage: dict | object | None = None,
        yaml_paths: Optional[list[str | Path]] = None,
        config_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            bot_id=bot_id,
            config_modules=config_modules,
            data_dir=data_dir,
            lineage=lineage,
            yaml_paths=yaml_paths,
            config_dir=config_dir,
            snapshot_module_fn=snapshot_config_module,
        )
