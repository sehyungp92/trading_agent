from __future__ import annotations

from collections import deque
from dataclasses import MISSING, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Mapping, Union, get_args, get_origin, get_type_hints

_TYPED_DICT_ITEMS = "__codex_typed_dict_items__"


def snapshot_dataclass(instance: Any) -> dict[str, Any]:
    return {
        field.name: snapshot_value(getattr(instance, field.name))
        for field in fields(instance)
    }


def restore_dataclass(cls: type[Any], payload: Mapping[str, Any]) -> Any:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in payload:
            if field.default is not MISSING or field.default_factory is not MISSING:
                continue
            kwargs[field.name] = None
            continue
        kwargs[field.name] = restore_value(hints.get(field.name, Any), payload[field.name])
    return cls(**kwargs)


def snapshot_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return snapshot_dataclass(value)
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: snapshot_value(item) for key, item in value.items()}
        return {
            _TYPED_DICT_ITEMS: [
                [snapshot_value(key), snapshot_value(item)]
                for key, item in value.items()
            ]
        }
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [snapshot_value(item) for item in value]
    return value


def restore_value(expected_type: Any, value: Any) -> Any:
    if value is None:
        return None

    if expected_type in (Any, object):
        return value

    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is None:
        return _restore_leaf(expected_type, value)

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(restore_value(args[0], item) for item in value)
        if args:
            restored_items = []
            for index, item in enumerate(value):
                item_type = args[index] if index < len(args) else Any
                restored_items.append(restore_value(item_type, item))
            return tuple(restored_items)
        return tuple(restore_value(Any, item) for item in value)

    if origin in (list, set, frozenset, deque):
        item_type = args[0] if args else Any
        restored = [restore_value(item_type, item) for item in value]
        if origin is set:
            return set(restored)
        if origin is frozenset:
            return frozenset(restored)
        if origin is deque:
            return deque(restored)
        return restored

    if origin is dict:
        key_type = args[0] if args else Any
        value_type = args[1] if len(args) > 1 else Any
        if isinstance(value, Mapping) and _TYPED_DICT_ITEMS in value:
            return {
                restore_value(key_type, key): restore_value(value_type, item)
                for key, item in value[_TYPED_DICT_ITEMS]
            }
        return {
            restore_value(key_type, key): restore_value(value_type, item)
            for key, item in value.items()
        }

    if origin in (Union, UnionType):
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return restore_value(candidate, value)
            except Exception:
                continue
        return value

    return value


def _restore_leaf(expected_type: Any, value: Any) -> Any:
    if expected_type in (str, int, float):
        return expected_type(value)
    if expected_type is bool:
        return value if isinstance(value, bool) else bool(value)
    if expected_type is datetime:
        return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if expected_type is Path:
        return value if isinstance(value, Path) else Path(value)
    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        return value if isinstance(value, expected_type) else expected_type(value)
    if isinstance(expected_type, type) and is_dataclass(expected_type):
        return restore_dataclass(expected_type, value)
    return value
