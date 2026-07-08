"""Exchange routing resolution."""
from ..config.schemas import ExchangeRoute


class ExchangeRouter:
    """Resolves canonical symbols to IB exchange routing strings."""

    def __init__(self, routes: dict[str, ExchangeRoute]):
        self._routes = routes

    def get_exchange(self, symbol: str) -> str:
        return self._routes[symbol].exchange

    def get_primary_exchange(self, symbol: str) -> str | None:
        return self._routes[symbol].primary_exchange

    def get_trading_class(self, symbol: str) -> str | None:
        return self._routes[symbol].trading_class

    def has_route(self, symbol: str) -> bool:
        return symbol in self._routes
