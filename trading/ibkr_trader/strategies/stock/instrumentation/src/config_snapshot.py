import enum
from types import ModuleType
from typing import Any


def snapshot_config_module(module: ModuleType) -> dict[str, Any]:
    """Capture all uppercase constants from a config module, making them JSON-safe."""
    result = {}
    for name in dir(module):
        if not name.isupper() or name.startswith("_"):
            continue
        val = getattr(module, name)
        if callable(val) and not isinstance(val, type):
            continue
        result[name] = _make_json_safe(val)
    return result


def _make_json_safe(val: Any) -> Any:
    """Convert non-JSON-serializable types to safe representations."""
    if isinstance(val, enum.Enum):
        return val.value
    if isinstance(val, (set, frozenset)):
        return sorted(val)
    if isinstance(val, dict):
        return {str(k): _make_json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_make_json_safe(v) for v in val]
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    return str(val)
