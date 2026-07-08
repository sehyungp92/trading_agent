"""Allow running as: python -m strategy_2"""
from .main import _setup_logging, main
import asyncio

_setup_logging()
try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
