"""Price and size rounding for Hyperliquid order submission."""

from __future__ import annotations

import math


def round_price(price: float, tick_size: float) -> float:
    """Round price down to the nearest tick size."""
    return round(math.floor(price / tick_size) * tick_size, 10)


def round_size(size: float, lot_size: float) -> float:
    """Round size down to the nearest lot size."""
    return round(math.floor(size / lot_size) * lot_size, 10)
