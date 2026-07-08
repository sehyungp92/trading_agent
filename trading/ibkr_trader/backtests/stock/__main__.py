"""Entry point for ``python -m backtests.stock``."""
import multiprocessing

multiprocessing.freeze_support()

from backtests.stock.cli import main  # noqa: E402

main()
