"""Split replay runner seam."""

from __future__ import annotations

from trading_assistant_backtest.replay.types import ReplayResult, WindowSpec


def deterministic_empty_replay(run_id: str, window: WindowSpec) -> ReplayResult:
    return ReplayResult(run_id=run_id, window=window)
