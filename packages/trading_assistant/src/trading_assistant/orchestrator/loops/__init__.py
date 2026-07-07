"""Loop-shaped orchestrator adapters."""

from trading_assistant.orchestrator.loops.daily_analysis import DailyAnalysisLoop
from trading_assistant.orchestrator.loops.monthly_validation import MonthlyValidationLoop
from trading_assistant.orchestrator.loops.weekly_analysis import WeeklyAnalysisLoop

__all__ = ["DailyAnalysisLoop", "MonthlyValidationLoop", "WeeklyAnalysisLoop"]
