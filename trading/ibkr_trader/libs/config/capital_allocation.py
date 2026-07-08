"""Config-driven capital allocation helpers for the unified runtime scaffold."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .models import PortfolioConfig, StrategyManifest, StrategyRegistryConfig


class CapitalAllocationError(ValueError):
    """Raised when capital allocation configuration is inconsistent."""


@dataclass(frozen=True)
class StrategyCapitalAllocation:
    strategy_id: str
    family: str
    raw_nav: float
    family_fraction: float
    strategy_fraction_within_family: float
    allocated_nav: float

    @property
    def total_fraction(self) -> float:
        return self.family_fraction * self.strategy_fraction_within_family

    @property
    def capital_pct(self) -> float:
        return self.total_fraction * 100.0

    def assert_positive_allocated_nav(self) -> None:
        if not math.isfinite(self.raw_nav) or self.raw_nav <= 0:
            raise RuntimeError(f"{self.strategy_id} requires a positive raw NAV; got {self.raw_nav!r}.")
        if not math.isfinite(self.allocated_nav) or self.allocated_nav <= 0:
            raise RuntimeError(
                f"{self.strategy_id} requires a positive allocated NAV; got {self.allocated_nav!r}."
            )


def resolve_strategy_capital_allocation(
    strategy_id: str,
    raw_nav: float,
    registry: StrategyRegistryConfig,
    portfolio: PortfolioConfig,
) -> StrategyCapitalAllocation:
    manifest = registry.strategies[strategy_id]
    family_fraction = _resolve_family_fraction(manifest.family, portfolio)
    within_family_fraction = _resolve_strategy_fraction_within_family(manifest, registry, portfolio)
    allocated_nav = raw_nav * family_fraction * within_family_fraction
    return StrategyCapitalAllocation(
        strategy_id=manifest.strategy_id,
        family=manifest.family,
        raw_nav=float(raw_nav),
        family_fraction=family_fraction,
        strategy_fraction_within_family=within_family_fraction,
        allocated_nav=allocated_nav,
    )


def _resolve_family_fraction(family: str, portfolio: PortfolioConfig) -> float:
    family_allocations = portfolio.capital.family_allocations
    if family not in family_allocations:
        raise CapitalAllocationError(f"Missing family allocation for {family!r}")
    fraction = family_allocations[family]
    if not math.isfinite(fraction) or fraction <= 0:
        raise CapitalAllocationError(f"Family allocation for {family!r} must be > 0")
    return fraction


def _resolve_strategy_fraction_within_family(
    manifest: StrategyManifest,
    registry: StrategyRegistryConfig,
    portfolio: PortfolioConfig,
) -> float:
    explicit = portfolio.capital.strategy_allocations.get(manifest.strategy_id)
    if explicit is not None:
        if not math.isfinite(explicit) or explicit <= 0:
            raise CapitalAllocationError(
                f"Strategy allocation for {manifest.strategy_id!r} must be > 0"
            )
        return explicit

    family_members = [
        item for item in registry.enabled_strategies() if item.family == manifest.family
    ]
    if not family_members:
        raise CapitalAllocationError(f"No enabled strategies found for family {manifest.family!r}")
    return 1.0 / len(family_members)

