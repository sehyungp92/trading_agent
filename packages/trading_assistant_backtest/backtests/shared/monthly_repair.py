"""Compatibility module for `python -m backtests.shared.monthly_repair`."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from trading_assistant_backtest.monthly import main as monthly_main

    return monthly_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
