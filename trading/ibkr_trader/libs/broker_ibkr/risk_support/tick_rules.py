"""Tick size and quantity rounding utilities."""
import math


def round_to_tick(price: float, tick_size: float, direction: str = "nearest") -> float:
    """Round price to valid tick increment.

    Args:
        price: The price to round
        tick_size: The minimum tick increment
        direction: 'nearest', 'up', or 'down'

    Returns:
        Price rounded to valid tick
    """
    if direction == "up":
        return math.ceil(price / tick_size) * tick_size
    elif direction == "down":
        return math.floor(price / tick_size) * tick_size
    else:
        return round(price / tick_size) * tick_size


def round_qty(qty: float, min_qty: float = 1.0) -> int:
    """Round quantity to integer, enforce minimum."""
    return max(int(math.floor(qty)), int(min_qty))


def validate_price(price: float, tick_size: float) -> bool:
    """Check if price conforms to tick size."""
    remainder = abs(price % tick_size)
    return remainder < tick_size * 0.01 or remainder > tick_size * 0.99
