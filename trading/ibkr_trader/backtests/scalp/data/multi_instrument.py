from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from libs.config.completed_bar_policy import (
    align_completed_daily_session_indices,
    align_completed_higher_timeframe_indices,
)

from .preprocessing import NumpyBars, load_bar_data


@dataclass
class ScalpMultiInstrumentData:
    analysis_symbol: str = "NQ"
    confirmation_symbol: str = "ES"
    analysis: dict[str, NumpyBars] = field(default_factory=dict)
    confirmation: dict[str, NumpyBars] = field(default_factory=dict)
    analysis_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    confirmation_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def trade_symbol(self) -> str:
        return self.analysis_symbol

    @property
    def trade(self) -> dict[str, NumpyBars]:
        return self.analysis

    @property
    def trade_idx_maps(self) -> dict[str, np.ndarray]:
        return self.analysis_idx_maps


def load_po3_data(
    data_dir: str | Path,
    *,
    analysis_symbol: str = "NQ",
    confirmation_symbol: str = "ES",
    trade_symbol: str | None = None,
) -> ScalpMultiInstrumentData:
    analysis_symbol = (trade_symbol or analysis_symbol).upper()
    confirmation_symbol = confirmation_symbol.upper()
    analysis = load_bar_data(data_dir, analysis_symbol)
    confirmation = load_bar_data(data_dir, confirmation_symbol)
    primary = analysis.get("1m", NumpyBars())
    return ScalpMultiInstrumentData(
        analysis_symbol=analysis_symbol,
        confirmation_symbol=confirmation_symbol,
        analysis=analysis,
        confirmation=confirmation,
        analysis_idx_maps=_index_maps(primary, analysis),
        confirmation_idx_maps=_index_maps(primary, confirmation),
    )


def _index_maps(primary: NumpyBars, frames: dict[str, NumpyBars]) -> dict[str, np.ndarray]:
    maps: dict[str, np.ndarray] = {}
    for timeframe, bars in frames.items():
        if timeframe == "1m":
            maps[timeframe] = np.arange(len(primary), dtype=np.int64)
        elif timeframe == "daily":
            maps[timeframe] = align_completed_daily_session_indices(primary.times, bars.times)
        else:
            maps[timeframe] = align_completed_higher_timeframe_indices(primary.times, bars.times)
    return maps
