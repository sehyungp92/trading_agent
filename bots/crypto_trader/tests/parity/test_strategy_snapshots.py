"""Strategy snapshot/restore round-trip tests for live restart parity."""

from crypto_trader.core.models import SetupGrade, Side
from crypto_trader.strategy.breakout.balance import BalanceZone
from crypto_trader.strategy.breakout.exits import BreakoutExitState
from crypto_trader.strategy.breakout.strategy import BreakoutStrategy, _PositionMeta as BreakoutMeta
from crypto_trader.strategy.momentum.exits import PositionExitState
from crypto_trader.strategy.momentum.strategy import MomentumStrategy, _PositionMeta as MomentumMeta
from crypto_trader.strategy.trend.exits import TrendExitState
from crypto_trader.strategy.trend.setup import TrendSetupResult
from crypto_trader.strategy.trend.strategy import (
    TrendStrategy,
    _PendingTrendSetup,
    _PositionMeta as TrendMeta,
)


def test_momentum_strategy_snapshot_round_trip() -> None:
    strategy = MomentumStrategy()
    strategy._position_meta["BTC"] = MomentumMeta(
        setup_grade=SetupGrade.A,
        confluences=("ema",),
        entry_price=100.0,
    )
    strategy._exit_manager._states["BTC"] = PositionExitState(
        entry_price=100.0,
        stop_distance=5.0,
        original_qty=1.0,
        remaining_qty=0.7,
    )
    strategy._trail_manager._current_stops["BTC"] = 98.0

    restored = MomentumStrategy()
    restored.restore_state(strategy.snapshot_state())

    assert restored._position_meta["BTC"].setup_grade == SetupGrade.A
    assert restored._position_meta["BTC"].confluences == ("ema",)
    assert restored._exit_manager.get_state("BTC").remaining_qty == 0.7
    assert restored._trail_manager._current_stops["BTC"] == 98.0


def test_trend_strategy_snapshot_round_trip() -> None:
    strategy = TrendStrategy()
    strategy._position_meta["ETH"] = TrendMeta(setup_grade=SetupGrade.B, is_reentry=True)
    strategy._exit_manager._states["ETH"] = TrendExitState(
        entry_price=100.0,
        stop_distance=4.0,
        original_qty=2.0,
        remaining_qty=1.0,
        direction=Side.SHORT,
    )
    strategy._pending_setups["ETH"] = _PendingTrendSetup(
        setup=TrendSetupResult(
            grade=SetupGrade.B,
            direction=Side.SHORT,
            impulse_start=120.0,
            impulse_end=100.0,
            impulse_atr_move=2.0,
            pullback_depth=0.4,
            confluences=("h1_ema_zone",),
            zone_price=105.0,
            room_r=1.5,
            stop_level=110.0,
        ),
        created_h1_bar_index=10,
        regime_tier="A",
    )

    restored = TrendStrategy()
    restored.restore_state(strategy.snapshot_state())

    assert restored._position_meta["ETH"].is_reentry is True
    assert restored._exit_manager.get_state("ETH").direction == Side.SHORT
    assert restored._pending_setups["ETH"].setup.direction == Side.SHORT
    assert restored._pending_setups["ETH"].setup.confluences == ("h1_ema_zone",)


def test_breakout_strategy_snapshot_round_trip() -> None:
    strategy = BreakoutStrategy()
    zone = BalanceZone(
        center=100.0,
        upper=110.0,
        lower=90.0,
        bars_in_zone=20,
        touches=3,
        formation_bar_idx=10,
        volume_contracting=True,
        width_atr=1.2,
    )
    strategy._position_meta["SOL"] = BreakoutMeta(
        setup_grade=SetupGrade.C,
        balance_zone=zone,
        signal_variant="relaxed",
    )
    strategy._exit_manager._states["SOL"] = BreakoutExitState(
        entry_price=100.0,
        stop_distance=3.0,
        original_qty=3.0,
        remaining_qty=2.0,
        direction=Side.LONG,
        balance_upper=110.0,
        balance_lower=90.0,
    )
    strategy._trail_manager._last_stops["SOL"] = 97.0

    restored = BreakoutStrategy()
    restored.restore_state(strategy.snapshot_state())

    assert restored._position_meta["SOL"].signal_variant == "relaxed"
    assert restored._position_meta["SOL"].balance_zone.upper == 110.0
    assert restored._exit_manager.get_state("SOL").direction == Side.LONG
    assert restored._trail_manager._last_stops["SOL"] == 97.0
