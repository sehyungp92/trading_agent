"""Structured signal confluence factor capture."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SignalFactor:
    """One component contributing to an entry signal's overall strength."""
    factor_name: str
    factor_value: float
    threshold: float
    contribution: float

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "factor_value": self.factor_value,
            "threshold": self.threshold,
            "contribution": self.contribution,
        }


def build_signal_factors(factors: list[SignalFactor]) -> list[dict]:
    """Convert SignalFactor list to list of dicts for TradeEvent.signal_factors."""
    return [f.to_dict() for f in factors]
