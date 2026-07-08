"""Small helpers for strategy state snapshot round trips."""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, get_args, get_origin, get_type_hints


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def dataclass_from_plain(cls: type, payload: dict[str, Any]) -> Any:
    kwargs = {}
    hints = get_type_hints(cls)
    for field in fields(cls):
        if field.name not in payload:
            continue
        kwargs[field.name] = _coerce(payload[field.name], hints.get(field.name, field.type))
    return cls(**kwargs)


def _coerce(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is tuple and isinstance(value, list):
        return tuple(value)
    if origin in (list, tuple):
        return value
    if origin is dict:
        return value
    if origin is not None and type(None) in args:
        if value is None:
            return None
        concrete = next((arg for arg in args if arg is not type(None)), Any)
        return _coerce(value, concrete)
    if isinstance(annotation, type) and is_dataclass(annotation) and isinstance(value, dict):
        return dataclass_from_plain(annotation, value)
    if isinstance(annotation, type) and issubclass(annotation, Enum) and value is not None:
        return annotation(value)
    return value
