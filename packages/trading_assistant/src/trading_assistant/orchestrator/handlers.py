"""Compatibility facade for orchestrator action handlers."""

from __future__ import annotations

from typing import Any

from trading_assistant.orchestrator._handler_implementation import Handlers as _HandlerImplementation
from trading_assistant.orchestrator.loops.daily_analysis import (
    INSTRUMENTATION_READINESS_THRESHOLD as _INSTRUMENTATION_READINESS_THRESHOLD,
    MIN_TRADES_FOR_ANALYSIS as _MIN_TRADES_FOR_ANALYSIS,
)


class _HandlersFacadeMeta(type):
    def __getattr__(cls, name: str) -> Any:
        return _implementation_attr(name)


class Handlers(metaclass=_HandlersFacadeMeta):
    """Logic-free facade over the extracted handler implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        object.__setattr__(self, "_impl", _HandlerImplementation(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        impl = self.__dict__.get("_impl")
        if impl is not None:
            return getattr(impl, name)
        return _implementation_attr(name, self)

    def __setattr__(self, name: str, value: Any) -> None:
        impl = self.__dict__.get("_impl")
        if name == "_impl" or impl is None:
            object.__setattr__(self, name, value)
            return
        if name.startswith("_") and hasattr(impl, name):
            setattr(impl, name, value)
            return
        object.__setattr__(self, name, value)


__all__ = [
    "Handlers",
    "_INSTRUMENTATION_READINESS_THRESHOLD",
    "_MIN_TRADES_FOR_ANALYSIS",
]


def _implementation_attr(name: str, instance: Any | None = None) -> Any:
    descriptor = _HandlerImplementation.__dict__.get(name)
    if isinstance(descriptor, staticmethod):
        return descriptor.__get__(None, _HandlerImplementation)
    if isinstance(descriptor, classmethod):
        return descriptor.__get__(_HandlerImplementation, _HandlerImplementation)
    attr = getattr(_HandlerImplementation, name)
    if instance is not None and hasattr(attr, "__get__"):
        return attr.__get__(instance, type(instance))
    return attr
