"""Execution adapters - public interface for OMS."""

__all__ = ["IBKRExecutionAdapter", "OrderNotFoundError"]


def __getattr__(name: str):
    if name in ("IBKRExecutionAdapter", "OrderNotFoundError"):
        from .execution_adapter import IBKRExecutionAdapter, OrderNotFoundError
        if name == "IBKRExecutionAdapter":
            return IBKRExecutionAdapter
        return OrderNotFoundError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
