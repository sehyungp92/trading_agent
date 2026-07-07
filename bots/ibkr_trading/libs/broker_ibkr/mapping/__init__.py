"""Contract and order mapping utilities."""
from .exchange_routes import ExchangeRouter
from .order_flags import AllowedOrderFlags, EXCHANGE_FLAGS

__all__ = [
    "ContractFactory",
    "ContractResolutionError",
    "ExchangeRouter",
    "AllowedOrderFlags",
    "EXCHANGE_FLAGS",
    "OrderMapper",
]


def __getattr__(name: str):
    if name == "ContractFactory":
        from .contract_factory import ContractFactory
        return ContractFactory
    elif name == "ContractResolutionError":
        from .contract_factory import ContractResolutionError
        return ContractResolutionError
    elif name == "OrderMapper":
        from .order_mapper import OrderMapper
        return OrderMapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
