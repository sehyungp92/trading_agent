"""TPC backtest adapter."""
from __future__ import annotations

from backtests.swing.config_etf_base import ETFSlippageConfig
from backtests.swing.engine.etf_engine_base import ETFBacktestResult, ETFStrategyBacktestEngine
from strategies.swing.tpc import STRATEGY_ID
from strategies.swing.tpc import indicators
from strategies.swing.tpc.config import SYMBOL_CONFIGS
from strategies.swing.tpc.core import logic
from strategies.swing.tpc.core.state import TPCBarInput, TPCCoreState, TPCFill, TPCOrderUpdate


def run_tpc_independent(replay_data: dict, config=None, **kwargs) -> ETFBacktestResult:
    cfgs = dict(getattr(config, "symbol_configs", SYMBOL_CONFIGS))
    requested_symbols = tuple(getattr(config, "symbols", tuple(cfgs)) or tuple(cfgs))
    cfgs = {symbol: cfgs[symbol] for symbol in requested_symbols if symbol in cfgs}
    engine = ETFStrategyBacktestEngine(
        strategy_id=STRATEGY_ID,
        configs=cfgs,
        core_logic=logic,
        state_factory=TPCCoreState,
        bar_input_factory=TPCBarInput,
        fill_factory=TPCFill,
        order_update_factory=TPCOrderUpdate,
        indicator_module=indicators,
        slippage=getattr(config, "slippage", ETFSlippageConfig()) if config else ETFSlippageConfig(),
        initial_equity=getattr(config, "initial_equity", 100_000.0) if config else 100_000.0,
        warmup_15m=getattr(config, "warmup_15m", 2_000) if config else 2_000,
        indicator_cache=kwargs.get("indicator_cache"),
    )
    return engine.run(replay_data)
