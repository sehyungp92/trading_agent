"""Market-data requirements catalog for monthly replay."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MarketDataRequirement:
    bot_id: str
    strategy_id: str
    market: str
    symbol: str
    timeframe: str
    source: str = "filesystem"
    required_for_authoritative_validation: bool = True


class MarketDataCatalog:
    """Resolve replay data requirements from strategy profiles or explicit rows."""

    def __init__(self, requirements: Iterable[MarketDataRequirement] | None = None) -> None:
        self._requirements = list(requirements or [])

    @classmethod
    def from_strategy_registry(
        cls,
        registry: object,
        *,
        default_source: str = "filesystem",
        default_timeframe: str = "1m",
    ) -> MarketDataCatalog:
        requirements: list[MarketDataRequirement] = []
        strategies = getattr(registry, "strategies", {}) or {}
        for strategy_id, profile in strategies.items():
            bot_id = getattr(profile, "bot_id", "")
            market = getattr(profile, "asset_class", "") or "unknown"
            for symbol in getattr(profile, "symbols", []) or []:
                requirements.append(MarketDataRequirement(
                    bot_id=bot_id,
                    strategy_id=strategy_id,
                    market=market,
                    symbol=str(symbol),
                    timeframe=default_timeframe,
                    source=default_source,
                ))
        return cls(requirements)

    def for_strategy(self, bot_id: str, strategy_id: str = "") -> list[MarketDataRequirement]:
        return [
            requirement for requirement in self._requirements
            if requirement.bot_id == bot_id
            and (not strategy_id or requirement.strategy_id == strategy_id)
        ]

    def all_requirements(self) -> list[MarketDataRequirement]:
        return list(self._requirements)

    @staticmethod
    def manifest_path(root: Path, requirement: MarketDataRequirement, run_month: str) -> Path:
        return (
            Path(root)
            / "manifests"
            / requirement.source
            / requirement.market
            / requirement.symbol
            / requirement.timeframe
            / f"{run_month}.coverage_manifest.json"
        )

    @staticmethod
    def strategy_manifest_path(root: Path, bot_id: str, strategy_id: str, run_month: str) -> Path:
        return (
            Path(root)
            / "manifests"
            / bot_id
            / strategy_id
            / f"{run_month}.coverage_manifest.json"
        )
