from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from libs.config.loader import load_portfolio_config, load_strategy_registry
from strategies.core.capital import resolve_plugin_nav

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_resolve_plugin_nav_uses_explicit_strategy_nav_in_live(monkeypatch) -> None:
    monkeypatch.setattr("strategies.core.capital.get_environment", lambda: "live")
    ctx = SimpleNamespace(
        portfolio=SimpleNamespace(
            capital=SimpleNamespace(strategy_navs={"TPC": 10_000.0})
        )
    )

    assert resolve_plugin_nav(ctx, "TPC") == 10_000.0


def test_resolve_plugin_nav_rejects_implicit_live_nominal_equity(monkeypatch) -> None:
    monkeypatch.setattr("strategies.core.capital.get_environment", lambda: "live")
    ctx = SimpleNamespace(
        registry=SimpleNamespace(strategies={}),
        portfolio=SimpleNamespace(
            capital=SimpleNamespace(
                allocation_check_equity=100_000.0,
                paper_initial_equity=30_000.0,
            )
        ),
    )

    with pytest.raises(RuntimeError, match="no explicit runtime NAV"):
        resolve_plugin_nav(ctx, "TPC")


def test_resolve_plugin_nav_derives_paper_nav_from_paper_seed(monkeypatch) -> None:
    monkeypatch.setattr("strategies.core.capital.get_environment", lambda: "paper")
    ctx = SimpleNamespace(
        registry=load_strategy_registry(CONFIG_DIR),
        portfolio=load_portfolio_config(CONFIG_DIR),
    )

    assert resolve_plugin_nav(ctx, "TPC") == pytest.approx(10_000.0 / 3.0)
    assert resolve_plugin_nav(ctx, "IARIC_v1") == pytest.approx(5_000.0)
