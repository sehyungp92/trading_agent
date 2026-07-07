"""Detector catalog interfaces for strategy analysis."""

from trading_assistant.analysis.detectors.catalog import DEFAULT_DETECTOR_CATALOG, DetectorCatalog
from trading_assistant.analysis.detectors.signal_decay import (
    evaluate_alpha_decay,
    evaluate_signal_decay,
)
from trading_assistant.analysis.detectors.time_of_day import evaluate_time_of_day_patterns

__all__ = [
    "DEFAULT_DETECTOR_CATALOG",
    "DetectorCatalog",
    "evaluate_alpha_decay",
    "evaluate_signal_decay",
    "evaluate_time_of_day_patterns",
]
