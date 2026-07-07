"""Monthly backtest execution entrypoints."""

__all__ = ["MonthlyExecution"]


def __getattr__(name: str) -> object:
    if name == "MonthlyExecution":
        from trading_assistant_backtest.monthly_execution.runner import MonthlyExecution

        return MonthlyExecution
    raise AttributeError(name)
