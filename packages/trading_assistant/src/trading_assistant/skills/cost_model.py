# skills/cost_model.py
"""Transaction cost model — computes realistic fees and slippage.

Supports three slippage models:
  - fixed: constant bps per trade
  - spread_proportional: slippage = spread in bps
  - empirical: regime-specific bps loaded from historical data

Cost multiplier support for sensitivity testing at 1x, 1.5x, 2x costs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from trading_assistant.schemas.cost_model import CostModelConfig, SlippageModel

logger = logging.getLogger(__name__)


@dataclass
class TradeCosts:
    """Breakdown of transaction costs for a single trade."""

    fees: float = 0.0
    slippage: float = 0.0

    @property
    def total(self) -> float:
        return self.fees + self.slippage


class CostModel:
    """Computes transaction costs given config and trade details."""

    def __init__(self, config: CostModelConfig) -> None:
        self._config = config
        self._empirical_stats: dict[str, float] | None = None
        if config.slippage_model == SlippageModel.EMPIRICAL and config.slippage_source:
            self._empirical_stats = self._load_stats(config.slippage_source)

    @classmethod
    def from_slippage_export(
        cls, regime_bps: dict[str, float], config: CostModelConfig
    ) -> "CostModel":
        """Create a CostModel with empirical slippage from SlippageAnalyzer export."""
        config = config.model_copy(update={"slippage_model": SlippageModel.EMPIRICAL})
        instance = cls(config)
        instance._empirical_stats = regime_bps
        return instance

    def compute_costs(
        self,
        entry_price: float,
        position_size: float,
        spread_bps: float = 0.0,
        regime: str = "",
        cost_multiplier: float = 1.0,
    ) -> TradeCosts:
        """Compute round-trip transaction costs for a single trade."""
        notional = entry_price * position_size
        if notional == 0 or cost_multiplier == 0:
            return TradeCosts()

        fees = notional * self._config.fees_per_trade_bps / 10_000 * 2
        slippage = self._compute_slippage(notional, spread_bps, regime)

        return TradeCosts(
            fees=fees * cost_multiplier,
            slippage=slippage * cost_multiplier,
        )

    def _compute_slippage(self, notional: float, spread_bps: float, regime: str) -> float:
        if self._config.slippage_model == SlippageModel.FIXED:
            return notional * self._config.fixed_slippage_bps / 10_000 * 2

        if self._config.slippage_model == SlippageModel.SPREAD_PROPORTIONAL:
            return notional * spread_bps / 10_000 * 2

        if self._config.slippage_model == SlippageModel.EMPIRICAL:
            bps = self._get_empirical_bps(regime)
            return notional * bps / 10_000 * 2

        return 0.0

    def _get_empirical_bps(self, regime: str) -> float:
        if self._empirical_stats is None:
            # Fallback to fixed slippage if stats unavailable
            return self._config.fixed_slippage_bps
        return self._empirical_stats.get(regime, self._empirical_stats.get("default", 0.0))

    @staticmethod
    def _load_stats(path_str: str) -> dict[str, float] | None:
        path = Path(path_str)
        if not path.exists():
            logger.warning("Slippage stats file not found: %s — falling back to fixed", path)
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load slippage stats from %s", path)
            return None
