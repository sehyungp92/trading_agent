from pathlib import Path

import pytest

from libs.config.capital_allocation import resolve_strategy_capital_allocation
from libs.config.loader import load_portfolio_config, load_strategy_registry


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_family_default_split_for_swing_strategy() -> None:
    registry = load_strategy_registry(CONFIG_DIR)
    portfolio = load_portfolio_config(CONFIG_DIR)
    family_fraction = 1 / 3

    allocation = resolve_strategy_capital_allocation("ATRSS", 100_000.0, registry, portfolio)

    assert allocation.family == "swing"
    assert allocation.family_fraction == pytest.approx(family_fraction)
    # Swing has 3 enabled strategies (ATRSS, Helix, TPC), so fallback split is 1/3.
    assert allocation.strategy_fraction_within_family == pytest.approx(1 / 3, rel=1e-6)
    assert allocation.allocated_nav == pytest.approx(100_000.0 * family_fraction / 3, rel=1e-6)


def test_stock_strategy_allocations_fall_back_to_equal_split() -> None:
    registry = load_strategy_registry(CONFIG_DIR)
    portfolio = load_portfolio_config(CONFIG_DIR)
    family_fraction = 1 / 3

    iaric = resolve_strategy_capital_allocation("IARIC_v1", 100_000.0, registry, portfolio)
    alcb = resolve_strategy_capital_allocation("ALCB_v1", 100_000.0, registry, portfolio)

    assert iaric.family == "stock"
    assert iaric.family_fraction == pytest.approx(family_fraction)
    assert iaric.strategy_fraction_within_family == pytest.approx(0.5)
    assert iaric.allocated_nav == pytest.approx(100_000.0 * family_fraction * 0.5, rel=1e-6)

    assert alcb.family == "stock"
    assert alcb.family_fraction == pytest.approx(family_fraction)
    assert alcb.strategy_fraction_within_family == pytest.approx(0.5)
    assert alcb.allocated_nav == pytest.approx(100_000.0 * family_fraction * 0.5, rel=1e-6)
