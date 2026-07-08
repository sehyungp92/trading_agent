"""Allow running as: python -m strategies.momentum.nqdtc"""
from strategies.momentum.nqdtc.main import _setup_logging, main
import asyncio

_setup_logging()
try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
