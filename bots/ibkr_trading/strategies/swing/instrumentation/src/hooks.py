"""Async-safe instrumentation hook helpers.

Wraps instrumentation calls so they never crash or block trading.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger("instrumentation.hooks")


def safe_instrument(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call *func* synchronously, swallowing exceptions.

    Returns the function result on success, None on failure.
    Used for lightweight calls (file appends, cache lookups).
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.debug("Instrumentation call %s failed: %s", func.__name__, e)
        return None


async def async_safe_instrument(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run *func* in the default executor (thread pool), swallowing exceptions.

    Used for heavier operations (regime classification with TA computation)
    that should not block the event loop.
    """
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    except Exception as e:
        logger.debug("Async instrumentation call %s failed: %s", func.__name__, e)
        return None
