"""Allow running as: python -m strategies.momentum.vdub"""
from strategies.momentum.vdub.main import _setup_logging, main
import asyncio

_setup_logging()
try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
