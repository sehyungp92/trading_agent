"""TPC backtest configuration."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from backtests.swing.config_etf_base import ETFSlippageConfig
from strategies.swing.tpc.config import SYMBOL_CONFIGS, TPCSymbolConfig


@dataclass(frozen=True)
class TPCBacktestConfig:
    initial_equity: float = 100_000.0
    data_dir: Path = Path("backtests/swing/data/raw")
    symbols: tuple[str, ...] = ("QQQ", "GLD")
    warmup_15m: int = 2_000
    slippage: ETFSlippageConfig = field(default_factory=ETFSlippageConfig)
    symbol_configs: dict[str, TPCSymbolConfig] = field(default_factory=lambda: dict(SYMBOL_CONFIGS))
    param_overrides: dict[str, Any] = field(default_factory=dict)

    def with_overrides(self, overrides: dict[str, Any]) -> "TPCBacktestConfig":
        configs = dict(self.symbol_configs)
        warmup = self.warmup_15m
        symbols = self.symbols
        for key, value in overrides.items():
            if key == "warmup_15m":
                warmup = int(value)
            elif key == "symbols":
                symbols = tuple(str(symbol) for symbol in value)
            elif "." in key:
                sym, field_name = key.split(".", 1)
                if sym == "all":
                    for name, cfg in list(configs.items()):
                        if hasattr(cfg, field_name):
                            configs[name] = replace(cfg, **{field_name: value})
                elif sym in configs and hasattr(configs[sym], field_name):
                    configs[sym] = replace(configs[sym], **{field_name: value})
        return replace(self, warmup_15m=warmup, symbols=symbols, symbol_configs=configs, param_overrides={**self.param_overrides, **overrides})
