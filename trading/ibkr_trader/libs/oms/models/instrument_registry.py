"""Global instrument registry for hydrating orders from DB."""
import logging
from typing import Optional
from .instrument import Instrument

logger = logging.getLogger(__name__)


class InstrumentRegistry:
    """Strategies register instruments at startup. Repository uses for hydration."""

    _instruments: dict[str, Instrument] = {}

    @classmethod
    def register(cls, instrument: Instrument) -> None:
        cls._instruments[instrument.symbol] = instrument
        if instrument.root != instrument.symbol:
            cls._instruments[instrument.root] = instrument

    @classmethod
    def get(cls, symbol: str) -> Optional[Instrument]:
        return cls._instruments.get(symbol)

    @classmethod
    def get_or_raise(cls, symbol: str) -> Instrument:
        inst = cls._instruments.get(symbol)
        if not inst:
            raise KeyError(f"Unknown instrument: {symbol}")
        return inst

    @classmethod
    def clear(cls) -> None:
        cls._instruments.clear()
