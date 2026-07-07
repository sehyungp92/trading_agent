"""TPC phased auto CLI."""
from __future__ import annotations

from backtests.swing.auto.etf_common import run_plugin_cli
from .plugin import TPCPlugin


def main() -> None:
    run_plugin_cli(TPCPlugin, "tpc-auto")


if __name__ == "__main__":
    main()
