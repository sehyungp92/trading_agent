from pathlib import Path

from backtests.swing.auto.experiments import build_experiment_queue
from backtests.swing.config import BacktestConfig
from backtests.swing.config_helix import HelixBacktestConfig
from backtests.swing.config_regime import RegimeConfig
from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.optimization.helix_param_space import _BTC_SYMS as HELIX_BTC_SYMS
from backtests.swing.optimization.helix_param_space import _helix_symbols
from backtests.swing.optimization.param_space import _BTC_SYMS as ATRSS_BTC_SYMS
from libs.config.loader import load_strategy_registry
from strategies.swing.akc_helix.config import SYMBOLS as HELIX_SYMBOLS
from strategies.swing.atrss.config import SYMBOLS as ATRSS_SYMBOLS


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
EXPECTED_SWING_SYMBOLS = {"QQQ", "GLD"}


def test_swing_registry_excludes_retired_strategies_and_uses_current_symbols() -> None:
    registry = load_strategy_registry(CONFIG_DIR)

    assert "S5_PB" not in registry.strategies
    assert "S5_DUAL" not in registry.strategies
    assert set(registry.strategies["AKC_HELIX"].symbols) == EXPECTED_SWING_SYMBOLS


def test_live_swing_defaults_use_current_symbols() -> None:
    assert set(HELIX_SYMBOLS) == EXPECTED_SWING_SYMBOLS


def test_backtest_swing_defaults_use_current_symbols() -> None:
    assert set(BacktestConfig().symbols) == EXPECTED_SWING_SYMBOLS
    assert set(HelixBacktestConfig().symbols) == EXPECTED_SWING_SYMBOLS
    assert set(RegimeConfig().symbols) == EXPECTED_SWING_SYMBOLS
    unified = UnifiedBacktestConfig()
    assert set(unified.atrss_symbols) == EXPECTED_SWING_SYMBOLS
    assert set(unified.helix_symbols) == EXPECTED_SWING_SYMBOLS


def test_unified_backtest_priorities_match_live_subset() -> None:
    unified = UnifiedBacktestConfig()

    assert [
        unified.atrss.strategy_id,
        unified.helix.strategy_id,
    ] == ["ATRSS", "AKC_HELIX"]
    assert unified.atrss.priority < unified.helix.priority


def test_swing_auto_experiments_exclude_retired_strategy_overlay() -> None:
    experiments = build_experiment_queue("all")

    assert {exp.strategy for exp in experiments}.issubset(
        {"atrss", "helix", "portfolio"},
    )
    assert all("s5" not in exp.id for exp in experiments)
    assert all(
        not any(key.startswith(("s5_pb", "s5_dual")) for key in exp.mutations)
        for exp in experiments
    )
    assert all(
        set(exp.mutations.get("overlay_symbols", [])) <= EXPECTED_SWING_SYMBOLS
        for exp in experiments
    )


def test_swing_optimization_defaults_use_current_symbols() -> None:
    assert set(_helix_symbols()) == EXPECTED_SWING_SYMBOLS
    assert set(ATRSS_SYMBOLS) == EXPECTED_SWING_SYMBOLS
    assert HELIX_BTC_SYMS.isdisjoint(EXPECTED_SWING_SYMBOLS)
    assert ATRSS_BTC_SYMS.isdisjoint(EXPECTED_SWING_SYMBOLS)
