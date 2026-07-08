from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import Final

from strategies.scalp._shared.nq_contract import FuturesSpec, spec_for
from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry

STRATEGY_ID: Final[str] = "NQ_REGIME"
DISPLAY_NAME: Final[str] = "NQ Regime Engine"
ANALYSIS_SYMBOL: Final[str] = "NQ"
TRADE_SYMBOL: Final[str] = "MNQ"
POINT_VALUE: Final[float] = 2.0
TICK_SIZE: Final[float] = 0.25

ENABLE_STRUCTURAL_EXPANSION: Final[bool] = True
ENABLE_LIQUIDITY_REVERSION: Final[bool] = True
ENABLE_SECOND_WIND: Final[bool] = True

PREMARKET_START_ET: Final[time] = time(8, 30)
RTH_OPEN_ET: Final[time] = time(9, 30)
EARLY_SWEEP_START_ET: Final[time] = time(9, 45)
IB_END_ET: Final[time] = time(10, 0)
PRIMARY_WINDOW_END_ET: Final[time] = time(11, 45)
LUNCH_END_ET: Final[time] = time(13, 15)
PM_WINDOW_START_ET: Final[time] = time(13, 30)
LATE_PM_START_ET: Final[time] = time(14, 45)
NO_NEW_ENTRIES_AFTER_ET: Final[time] = time(15, 15)
HARD_FLATTEN_ET: Final[time] = time(15, 45)
EMERGENCY_FLATTEN_ET: Final[time] = time(15, 55)

IB_NARROW_MAX: Final[float] = 40.0
IB_WIDE_MIN: Final[float] = 80.0

REGIME_MIN_CONFIDENCE: Final[float] = 0.65
REGIME_MIN_MARGIN: Final[float] = 0.15
REGIME_FAILURE_CONFIRM_BARS: Final[int] = 1
ROUTE_ALLOW_A_PLUS_FALLBACK: Final[bool] = False
ROUTE_FALLBACK_MIN_SCORE: Final[int] = 10
ROUTE_CANDIDATE_LED_ENABLED: Final[bool] = True
ROUTE_CANDIDATE_LED_MIN_SCORE: Final[int] = 9
ROUTE_CANDIDATE_LED_MIN_ROOM_R: Final[float] = 0.25
SECOND_WIND_CANDIDATE_LED_ENABLED: Final[bool] = True
SECOND_WIND_CANDIDATE_LED_MIN_SCORE: Final[int] = 8
SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R: Final[float] = 1.5
SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE: Final[float] = 0.55

RISK_PCT_A_PLUS: Final[float] = 0.0075
RISK_PCT_A: Final[float] = 0.005
RISK_PCT_B: Final[float] = 0.0025
RISK_PCT_POST_NEWS: Final[float] = 0.0025
RISK_REDUCTION_AFTER_LOSS: Final[float] = 0.5
BASE_RISK_PCT: Final[float] = 0.006
DAILY_STOP_R: Final[float] = 3.0
HEAT_CAP_R: Final[float] = 1.5
PORTFOLIO_DAILY_STOP_R: Final[float] = 2.75

MAX_TRADES_PER_DAY: Final[int] = 4
MAX_FULL_RISK_TRADES: Final[int] = 3
MAX_CONCURRENT_POSITIONS: Final[int] = 1
MAX_LOSSES_PER_DAY: Final[int] = 2
MAX_DAILY_REALIZED_R_LOSS: Final[float] = -2.0

STRUCTURAL_MIN_SCORE: Final[int] = 8
STRUCTURAL_A_PLUS_SCORE: Final[int] = 10
REVERSION_MIN_SCORE: Final[int] = 8
REVERSION_A_SCORE: Final[int] = 8
REVERSION_A_PLUS_SCORE: Final[int] = 11
SECOND_WIND_MIN_SCORE: Final[int] = 8
SECOND_WIND_A_SCORE: Final[int] = 8
SECOND_WIND_A_PLUS_SCORE: Final[int] = 10

REVERSION_STANDARD_STOP_CAP: Final[float] = 10.0
REVERSION_A_PLUS_STOP_CAP: Final[float] = 12.0
SECOND_WIND_STOP_CAP: Final[float] = 30.0

TARGET_ROOM_MIN_R: Final[float] = 0.5
TARGET_ROOM_STRONG_R: Final[float] = 2.0

ENTRY_TTL_RETEST_MINUTES: Final[int] = 120
ENTRY_TTL_MOMENTUM_MINUTES: Final[int] = 15
TARGET1_QTY_FRACTION: Final[float] = 0.40
TARGET2_QTY_FRACTION: Final[float] = 0.30
MOVE_STOP_TO_BE_ON_T1: Final[bool] = True

PROFIT_FLOOR_ENABLED: Final[bool] = True
PROFIT_FLOOR_TRIGGER_R: Final[float] = 0.5
PROFIT_FLOOR_LOCK_R: Final[float] = 0.25
MFE_RATCHET_ENABLED: Final[bool] = True
MFE_RATCHET_TRIGGER_R: Final[float] = 1.0
MFE_RATCHET_FLOOR_PCT: Final[float] = 0.65
TIME_STOP_ENABLED: Final[bool] = False
TIME_STOP_BARS: Final[int] = 6
TIME_STOP_MIN_MFE_R: Final[float] = 0.50
REVERSION_VWAP_REACTION_EXIT_ENABLED: Final[bool] = False
STRUCTURAL_FAILURE_EXIT_ENABLED: Final[bool] = False
SECOND_WIND_EMA_TRAIL_EXIT_ENABLED: Final[bool] = True

STRUCTURAL_ENTRY_MODE: Final[str] = "structure_shift"
STRUCTURAL_ACCEPTANCE_ENTRY_ENABLED: Final[bool] = True
STRUCTURAL_RETEST_OFFSET_TICKS: Final[int] = 1
STRUCTURAL_STOP_ENTRY_OFFSET_TICKS: Final[int] = 1
STRUCTURAL_ADAPTIVE_RETEST_PREFERS_FVG: Final[bool] = True
STRUCTURAL_MIDPOINT_RETEST_ENABLED: Final[bool] = True
STRUCTURAL_FVG_RETEST_ENABLED: Final[bool] = False
STRUCTURAL_FVG_RETEST_MAX_GAP_PTS: Final[float] = 80.0
STRUCTURAL_RETEST_ENTRY_MAX_DISTANCE_R: Final[float] = 2.50
STRUCTURAL_HYBRID_CLOSE_MIN_SCORE: Final[int] = 8
STRUCTURAL_HYBRID_CLOSE_MAX_STOP_PTS: Final[float] = 45.0
STRUCTURAL_STOP_MODEL: Final[str] = "recent_5m"
STRUCTURAL_MIN_STOP_PTS: Final[float] = 10.0
STRUCTURAL_MAX_STOP_PTS: Final[float] = 45.0
STRUCTURAL_CONTINUATION_ENABLED: Final[bool] = False
STRUCTURAL_CONTINUATION_MIN_SCORE: Final[int] = 8
STRUCTURAL_MIN_BODY_PCT: Final[float] = 0.55
STRUCTURAL_MIN_CLOSE_LOCATION: Final[float] = 0.65
STRUCTURAL_BLOCK_NARROW_IB: Final[bool] = False
STRUCTURAL_BLOCK_NORMAL_IB: Final[bool] = False
STRUCTURAL_BLOCK_WIDE_IB: Final[bool] = False
STRUCTURAL_REQUIRE_SECOND_ACCEPTANCE: Final[bool] = False
STRUCTURAL_BLOCK_OPPOSITE_PM_BREAKOUT: Final[bool] = True
STRUCTURAL_MIN_ENTRY_MINUTE_ET: Final[int] = 0
STRUCTURAL_MAX_ENTRY_MINUTE_ET: Final[int] = 24 * 60
STRUCTURAL_LONG_MIN_SCORE: Final[int] = 8
STRUCTURAL_SHORT_MIN_SCORE: Final[int] = 10
STRUCTURAL_TARGET_ROOM_MAX_R: Final[float] = 999.0
STRUCTURAL_ALLOW_MIN_MICRO_SIZE: Final[bool] = True
STRUCTURAL_MIN_MICRO_MAX_RISK_PCT: Final[float] = 0.01
STRUCTURAL_CONTINUATION_REQUIRE_ACTIVE_BREAK: Final[bool] = False
STRUCTURAL_CONTINUATION_REQUIRE_15M_ACCEPTANCE: Final[bool] = False
STRUCTURAL_CONTINUATION_MAX_AGE_MINUTES: Final[int] = 24 * 60
STRUCTURAL_CONTINUATION_MIN_ROOM_R: Final[float] = 0.0
STRUCTURAL_CONTINUATION_MAX_ROOM_R: Final[float] = 999.0
STRUCTURAL_CONTINUATION_MIN_VOLUME_MULTIPLE: Final[float] = 0.0
STRUCTURAL_CONTINUATION_MIN_CLOSE_LOCATION: Final[float] = 0.55
STRUCTURAL_CONTINUATION_REQUIRE_TREND: Final[bool] = False
STRUCTURAL_CONTINUATION_ENTRY_MODE: Final[str] = "close"
STRUCTURAL_CONTINUATION_ENTRY_OFFSET_TICKS: Final[int] = 1
STRUCTURAL_PULLBACK_RECLAIM_ENABLED: Final[bool] = True
STRUCTURAL_PULLBACK_RECLAIM_MIN_SCORE: Final[int] = 8
STRUCTURAL_PULLBACK_RECLAIM_MAX_AGE_MINUTES: Final[int] = 240
STRUCTURAL_PULLBACK_RECLAIM_MAX_PULLBACK_BARS: Final[int] = 6
STRUCTURAL_PULLBACK_RECLAIM_BAND_ATR_MULT: Final[float] = 0.35
STRUCTURAL_PULLBACK_RECLAIM_MAX_BAND_PTS: Final[float] = 16.0
STRUCTURAL_PULLBACK_RECLAIM_MIN_CLOSE_LOCATION: Final[float] = 0.55
STRUCTURAL_PULLBACK_RECLAIM_MIN_ROOM_R: Final[float] = 1.0
STRUCTURAL_PULLBACK_RECLAIM_MAX_ROOM_R: Final[float] = 999.0
STRUCTURAL_PULLBACK_RECLAIM_MIN_VOLUME_MULTIPLE: Final[float] = 0.8
STRUCTURAL_PULLBACK_RECLAIM_REQUIRE_TREND: Final[bool] = False
STRUCTURAL_PULLBACK_RECLAIM_ENTRY_MODE: Final[str] = "close"
STRUCTURAL_PULLBACK_RECLAIM_ENTRY_OFFSET_TICKS: Final[int] = 1
STRUCTURAL_FAST_FAILURE_EXIT_ENABLED: Final[bool] = False
STRUCTURAL_TIME_STOP_ENABLED: Final[bool] = False
STRUCTURAL_TIME_STOP_BARS: Final[int] = 3
STRUCTURAL_TIME_STOP_MIN_MFE_R: Final[float] = 0.50
STRUCTURAL_PROFIT_FLOOR_ENABLED: Final[bool] = True
STRUCTURAL_PROFIT_FLOOR_TRIGGER_R: Final[float] = 0.5
STRUCTURAL_PROFIT_FLOOR_LOCK_R: Final[float] = 0.1
STRUCTURAL_MFE_RATCHET_ENABLED: Final[bool] = False
STRUCTURAL_MFE_RATCHET_TRIGGER_R: Final[float] = 1.0
STRUCTURAL_MFE_RATCHET_FLOOR_PCT: Final[float] = 0.50
STRUCTURAL_TARGET1_QTY_FRACTION: Final[float] = TARGET1_QTY_FRACTION
STRUCTURAL_TARGET2_QTY_FRACTION: Final[float] = TARGET2_QTY_FRACTION
STRUCTURAL_NARROW_TARGET1_R: Final[float] = 1.0
STRUCTURAL_NARROW_TARGET2_R: Final[float] = 1.5
STRUCTURAL_NARROW_TARGET3_R: Final[float] = 2.5
STRUCTURAL_NORMAL_TARGET1_R: Final[float] = 0.75
STRUCTURAL_NORMAL_TARGET2_R: Final[float] = 1.25
STRUCTURAL_NORMAL_TARGET3_R: Final[float] = 2.0
STRUCTURAL_WIDE_TARGET1_R: Final[float] = 0.5
STRUCTURAL_WIDE_TARGET2_R: Final[float] = 1.0
STRUCTURAL_WIDE_TARGET3_R: Final[float] = 1.5

REVERSION_ENTRY_MODEL: Final[str] = "swept_level_retest"
REVERSION_RETEST_OFFSET_TICKS: Final[int] = 0
REVERSION_ADAPTIVE_MARKET_MIN_SCORE: Final[int] = 12
REVERSION_ADAPTIVE_MARKET_MAX_PENETRATION_PTS: Final[float] = 8.0
REVERSION_ADAPTIVE_MARKET_MIN_ROOM_R: Final[float] = 2.0
REVERSION_ALLOW_EARLY_SWEEP: Final[bool] = False
REVERSION_ENABLE_SWING_LEVELS: Final[bool] = True
REVERSION_SWING_LOOKBACK_BARS: Final[int] = 48
REVERSION_SWING_RADIUS: Final[int] = 1
REVERSION_SWING_MAX_LEVELS_PER_SIDE: Final[int] = 6
REVERSION_MIN_VALUE_FACTORS: Final[int] = 0
REVERSION_MIN_PENETRATION_PTS: Final[float] = 2.0
REVERSION_MAX_PENETRATION_PTS: Final[float] = 12.0
REVERSION_PROFIT_FLOOR_ENABLED: Final[bool] = True
REVERSION_PROFIT_FLOOR_TRIGGER_R: Final[float] = 0.75
REVERSION_PROFIT_FLOOR_LOCK_R: Final[float] = 0.25
REVERSION_MFE_RATCHET_ENABLED: Final[bool] = False
REVERSION_MFE_RATCHET_TRIGGER_R: Final[float] = 1.0
REVERSION_MFE_RATCHET_FLOOR_PCT: Final[float] = 0.50
REVERSION_VWAP_TOUCH_EXIT_ENABLED: Final[bool] = True
REVERSION_TIME_STOP_ENABLED: Final[bool] = False
REVERSION_TIME_STOP_BARS: Final[int] = 6
REVERSION_TIME_STOP_MIN_MFE_R: Final[float] = 0.50
REVERSION_TARGET1_QTY_FRACTION: Final[float] = TARGET1_QTY_FRACTION
REVERSION_TARGET2_QTY_FRACTION: Final[float] = TARGET2_QTY_FRACTION

ALLOW_LATE_PM_REVERSION: Final[bool] = True

SECOND_WIND_ENTRY_MODEL: Final[str] = "trigger_midpoint"
SECOND_WIND_ATR_STOP_MULT: Final[float] = 0.75
SECOND_WIND_MIN_SQUEEZE_BARS: Final[int] = 3
SECOND_WIND_TRIGGER_CLOSE_MIN: Final[float] = 0.67
SECOND_WIND_ENTRY_STOP_INVALIDATION_ENABLED: Final[bool] = True
SECOND_WIND_MIN_VOLUME_MULTIPLE: Final[float] = 0.8
SECOND_WIND_MAX_STOP_PTS: Final[float] = 30.0
SECOND_WIND_MIN_ENTRY_MINUTE_ET: Final[int] = 13 * 60 + 30
SECOND_WIND_MAX_ENTRY_MINUTE_ET: Final[int] = 15 * 60 + 15
SECOND_WIND_VWAP_RECLAIM_ENABLED: Final[bool] = True
SECOND_WIND_MICRO_COMPRESSION_ENABLED: Final[bool] = True
SECOND_WIND_RANGE_ACCEPTANCE_ENABLED: Final[bool] = True
SECOND_WIND_SECOND_LEG_ENABLED: Final[bool] = False
SECOND_WIND_MIN_PM_SCORE: Final[float] = 0.55
SECOND_WIND_PM_TRANSITION_MIN_SCORE: Final[float] = 0.55
SECOND_WIND_VWAP_RECLAIM_MIN_SCORE: Final[int] = 9
SECOND_WIND_VWAP_RECLAIM_MIN_PM_SCORE: Final[float] = 0.6
SECOND_WIND_VWAP_RECLAIM_MIN_VOLUME_MULTIPLE: Final[float] = 1.3
SECOND_WIND_VWAP_RECLAIM_MIN_CLOSE_LOCATION: Final[float] = 0.7
SECOND_WIND_VWAP_RECLAIM_MIN_RECLAIM_ATR: Final[float] = 0.08
SECOND_WIND_VWAP_RECLAIM_MIN_RECLAIM_PTS: Final[float] = 0.0
SECOND_WIND_VWAP_RECLAIM_MAX_RECLAIM_ATR: Final[float] = 999.0
SECOND_WIND_VWAP_RECLAIM_REQUIRE_PM_TRANSITION: Final[bool] = False
SECOND_WIND_VWAP_RECLAIM_REQUIRE_EMA_ALIGNMENT: Final[bool] = True
SECOND_WIND_SECOND_LEG_MIN_SCORE: Final[int] = 0
SECOND_WIND_SECOND_LEG_MIN_PM_SCORE: Final[float] = 0.0
SECOND_WIND_SECOND_LEG_MIN_VOLUME_MULTIPLE: Final[float] = 0.0
SECOND_WIND_SECOND_LEG_MIN_CLOSE_LOCATION: Final[float] = 0.0
SECOND_WIND_SECOND_LEG_MIN_BREAKOUT_ATR: Final[float] = 0.0
SECOND_WIND_SECOND_LEG_MIN_BREAKOUT_PTS: Final[float] = 0.0
SECOND_WIND_SECOND_LEG_REQUIRE_PM_TRANSITION: Final[bool] = False
SECOND_WIND_SECOND_LEG_REQUIRE_IMPULSE: Final[bool] = False
SECOND_WIND_SECOND_LEG_PULLBACK_BUFFER_ATR_MULT: Final[float] = 0.35
SECOND_WIND_SECOND_LEG_PULLBACK_BUFFER_MIN_PTS: Final[float] = 1.0
SECOND_WIND_MICRO_RANGE_ATR_MULT: Final[float] = 0.85
SECOND_WIND_PULLBACK_LOOKBACK_BARS: Final[int] = 6
SECOND_WIND_PROFIT_FLOOR_ENABLED: Final[bool] = False
SECOND_WIND_PROFIT_FLOOR_TRIGGER_R: Final[float] = 1.00
SECOND_WIND_PROFIT_FLOOR_LOCK_R: Final[float] = 0.50
SECOND_WIND_MFE_RATCHET_ENABLED: Final[bool] = False
SECOND_WIND_MFE_RATCHET_TRIGGER_R: Final[float] = 1.50
SECOND_WIND_MFE_RATCHET_FLOOR_PCT: Final[float] = 0.50
SECOND_WIND_TIME_STOP_ENABLED: Final[bool] = False
SECOND_WIND_TIME_STOP_BARS: Final[int] = 4
SECOND_WIND_TIME_STOP_MIN_MFE_R: Final[float] = 0.50
SECOND_WIND_EMA_TRAIL_REQUIRES_PARTIAL: Final[bool] = True
SECOND_WIND_TARGET1_QTY_FRACTION: Final[float] = TARGET1_QTY_FRACTION
SECOND_WIND_TARGET2_QTY_FRACTION: Final[float] = TARGET2_QTY_FRACTION


class TradeSide(str, Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"

    @property
    def action_side(self) -> str:
        if self is TradeSide.SHORT:
            return "SELL"
        return "BUY"

    @property
    def exit_action_side(self) -> str:
        if self is TradeSide.SHORT:
            return "BUY"
        return "SELL"

    @property
    def sign(self) -> int:
        if self is TradeSide.LONG:
            return 1
        if self is TradeSide.SHORT:
            return -1
        return 0


class ModuleId(str, Enum):
    NONE = "none"
    STRUCTURAL_EXPANSION = "structural_expansion"
    LIQUIDITY_REVERSION = "liquidity_reversion"
    SECOND_WIND = "second_wind"


class Grade(str, Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    INVALID = "Invalid"


class IBType(str, Enum):
    UNCLASSIFIED = "unclassified"
    NARROW = "narrow"
    NORMAL = "normal"
    WIDE = "wide"


@dataclass(frozen=True, slots=True)
class StrategyRuntimeSettings:
    analysis_symbol: str = ANALYSIS_SYMBOL
    trade_symbol: str = TRADE_SYMBOL
    initial_equity: float = 100_000.0
    prefer_micro: bool = True
    max_contracts: int | None = None
    enable_structural_expansion: bool = ENABLE_STRUCTURAL_EXPANSION
    enable_liquidity_reversion: bool = ENABLE_LIQUIDITY_REVERSION
    enable_second_wind: bool = ENABLE_SECOND_WIND

    @property
    def trade_spec(self) -> FuturesSpec:
        return spec_for(self.trade_symbol)


def classify_ib_range(points: float) -> IBType:
    if points <= 0:
        return IBType.UNCLASSIFIED
    if points < IB_NARROW_MAX:
        return IBType.NARROW
    if points > IB_WIDE_MIN:
        return IBType.WIDE
    return IBType.NORMAL


def build_instruments() -> dict[str, Instrument]:
    instruments: dict[str, Instrument] = {}
    for symbol in (ANALYSIS_SYMBOL, TRADE_SYMBOL):
        spec = spec_for(symbol)
        instrument = Instrument(
            symbol=symbol,
            root=symbol,
            venue="CME",
            tick_size=spec.tick,
            tick_value=spec.tick_value,
            multiplier=spec.point_value,
            point_value=spec.point_value,
            currency="USD",
            contract_expiry="",
            sec_type="FUT",
            trading_class=symbol,
        )
        InstrumentRegistry.register(instrument)
        instruments[symbol] = instrument
    return instruments
