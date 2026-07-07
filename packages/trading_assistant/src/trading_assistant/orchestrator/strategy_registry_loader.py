"""Load strategy profiles from YAML into a StrategyRegistry."""
from __future__ import annotations

import logging
from pathlib import Path

from trading_assistant.paths import data_root
from trading_assistant.schemas.strategy_profile import StrategyRegistry

logger = logging.getLogger(__name__)

_DEFAULT_PATH = data_root() / "strategy_profiles.yaml"


def load_strategy_registry(path: Path | None = None) -> StrategyRegistry:
    """Load strategy profiles from YAML. Returns empty registry if file missing."""
    target = path or _DEFAULT_PATH
    if not target.exists():
        logger.info("Strategy profiles not found at %s — using empty registry", target)
        return StrategyRegistry()

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed — cannot load strategy profiles")
        return StrategyRegistry()

    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to parse strategy profiles from %s", target, exc_info=True)
        return StrategyRegistry()

    if not isinstance(raw, dict):
        logger.warning("Strategy profiles YAML is not a mapping — using empty registry")
        return StrategyRegistry()

    try:
        return StrategyRegistry.model_validate(raw)
    except Exception:
        logger.warning("Failed to validate strategy profiles", exc_info=True)
        return StrategyRegistry()
