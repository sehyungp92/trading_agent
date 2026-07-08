from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from backtests.shared.auto.cache_keys import build_cache_key, fingerprint_paths, stable_signature
from backtests.shared.auto.replay_bundle import ReplayBundle
from backtests.shared.auto.round_manager import RoundManager
from backtests.stock.analysis.metrics import compute_cagr, compute_max_drawdown
from backtests.stock.auto.alcb.time_utils import hydrate_time_mutations
from backtests.stock.auto.config_mutator import mutate_alcb_config, mutate_iaric_config
from backtests.stock.config_alcb import ALCBBacktestConfig
from backtests.stock.config_iaric import IARICBacktestConfig
from backtests.stock.data.cache import bar_path, load_bars
from backtests.stock.engine.alcb_engine import ALCBIntradayEngine
from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine
from backtests.stock.models import TradeRecord
from .core.state import PortfolioPosition

from .core.logic import replay_trade_streams, run_portfolio_replay
from .phase_candidates import INITIAL_EQUITY, SEED_PORTFOLIO_CONFIG

__all__ = [
    "StrategyTradeBundle",
    "build_effective_portfolio_config",
    "evaluate_portfolio",
    "load_evaluation_bundle",
    "load_evaluation_data",
    "replay_trade_streams",
]


@dataclass(frozen=True)
class StrategyTradeBundle:
    alcb_trades: tuple[TradeRecord, ...]
    iaric_trades: tuple[TradeRecord, ...]


def load_evaluation_data(
    data_dir: Path,
    *,
    initial_equity: float = INITIAL_EQUITY,
    start_date: str = "2024-01-01",
    end_date: str = "2026-03-01",
) -> StrategyTradeBundle:
    return load_evaluation_bundle(
        data_dir,
        initial_equity=initial_equity,
        start_date=start_date,
        end_date=end_date,
    ).data


def load_evaluation_bundle(
    data_dir: Path,
    *,
    initial_equity: float = INITIAL_EQUITY,
    start_date: str = "2024-01-01",
    end_date: str = "2026-03-01",
) -> ReplayBundle[StrategyTradeBundle]:
    return _load_evaluation_bundle_cached(str(Path(data_dir)), float(initial_equity), start_date, end_date)


@lru_cache(maxsize=4)
def _load_evaluation_bundle_cached(
    data_dir_str: str,
    initial_equity: float,
    start_date: str,
    end_date: str,
) -> ReplayBundle[StrategyTradeBundle]:
    from backtests.stock.data.replay_cache import load_research_replay_bundle

    data_dir = Path(data_dir_str)
    repo_root = _repo_root()
    replay_bundle = load_research_replay_bundle(data_dir)
    mutation_paths = _latest_strategy_mutation_paths(repo_root)
    source_fingerprint = stable_signature(
        {
            "stock_replay": replay_bundle.cache_source_fingerprint,
            "strategy_configs": fingerprint_paths(mutation_paths, root=repo_root),
            "initial_equity": initial_equity,
            "start_date": start_date,
            "end_date": end_date,
        }
    )
    cache_key = build_cache_key(
        "stock.portfolio_synergy.strategy_trade_bundle",
        source_fingerprint=source_fingerprint,
        extra={
            "data_dir": str(data_dir.resolve()),
            "initial_equity": initial_equity,
            "start_date": start_date,
            "end_date": end_date,
        },
    )
    replay = replay_bundle.data
    alcb_mutations, iaric_mutations = _load_latest_strategy_mutations(repo_root)

    alcb_cfg = mutate_alcb_config(
        ALCBBacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_equity=initial_equity,
            tier=2,
            data_dir=data_dir,
        ),
        hydrate_time_mutations(alcb_mutations),
    )
    iaric_cfg = mutate_iaric_config(
        IARICBacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_equity=initial_equity,
            tier=3,
            data_dir=data_dir,
        ),
        iaric_mutations,
    )

    alcb_result = ALCBIntradayEngine(alcb_cfg, replay).run()
    iaric_result = IARICPullbackEngine(iaric_cfg, replay).run()
    return ReplayBundle(
        data=StrategyTradeBundle(
            alcb_trades=tuple(alcb_result.trades),
            iaric_trades=tuple(iaric_result.trades),
        ),
        cache_key=cache_key,
        cache_source_fingerprint=source_fingerprint,
    )


def evaluate_portfolio(
    mutations: dict[str, Any] | None,
    *,
    data_dir: Path,
    initial_equity: float = INITIAL_EQUITY,
    start_date: str = "2024-01-01",
    end_date: str = "2026-03-01",
    evaluation_data: StrategyTradeBundle | None = None,
    price_bars_by_symbol: dict[str, Any] | None = None,
) -> dict[str, float]:
    effective = build_effective_portfolio_config(mutations, initial_equity=initial_equity)
    data = evaluation_data or load_evaluation_data(
        data_dir,
        initial_equity=initial_equity,
        start_date=start_date,
        end_date=end_date,
    )
    result = run_portfolio_replay(data.alcb_trades, data.iaric_trades, effective)
    metrics = dict(result.metrics)
    metrics.update(
        _headline_mtm_metrics(
            result.state.accepted_positions,
            metrics,
            initial_equity=initial_equity,
            data_dir=data_dir,
            price_bars_by_symbol=price_bars_by_symbol,
        )
    )
    return metrics


def _headline_mtm_metrics(
    positions: list[PortfolioPosition],
    realized_metrics: dict[str, float],
    *,
    initial_equity: float,
    data_dir: Path,
    price_bars_by_symbol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    realized_dd = float(realized_metrics.get("max_drawdown_pct", 0.0) or 0.0)
    realized_calmar = float(realized_metrics.get("calmar", 0.0) or 0.0)
    mtm = _stock_portfolio_mtm_metrics(
        positions,
        initial_equity=initial_equity,
        data_dir=data_dir,
        price_bars_by_symbol=price_bars_by_symbol,
    )
    mtm_dd = float(mtm.get("max_drawdown_pct", realized_dd) or 0.0)
    mtm_calmar = float(mtm.get("calmar", realized_calmar) or 0.0)
    use_mtm = mtm.get("risk_basis") == "bar_close_mark_to_market"
    return {
        "risk_basis": mtm.get("risk_basis", "realized_only"),
        "max_drawdown_pct": mtm_dd if use_mtm else realized_dd,
        "calmar": mtm_calmar if use_mtm else realized_calmar,
        "max_drawdown_pct_mtm": mtm_dd,
        "calmar_mtm": mtm_calmar,
        "max_drawdown_pct_realized": realized_dd,
        "calmar_realized": realized_calmar,
        "mtm_points": float(mtm.get("points", 0.0) or 0.0),
    }


def _stock_portfolio_mtm_metrics(
    positions: list[PortfolioPosition],
    *,
    initial_equity: float,
    data_dir: Path | None = None,
    price_bars_by_symbol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    accepted = [
        position for position in positions
        if position.entry_time is not None
        and position.exit_time is not None
        and position.entry_price > 0
        and position.quantity > 0
    ]
    final_equity = float(initial_equity) + sum(position.pnl for position in accepted)
    if not accepted:
        return {
            "risk_basis": "bar_close_mark_to_market",
            "final_equity": final_equity,
            "max_drawdown_pct": 0.0,
            "max_drawdown_dollar": 0.0,
            "calmar": 0.0,
            "points": 0,
            "priced_symbol_count": 0,
            "accepted_symbol_count": 0,
            "missing_price_symbols": [],
        }

    accepted_symbols = {position.symbol for position in accepted}
    bars_by_symbol = price_bars_by_symbol or _load_stock_price_bars(
        data_dir or Path("backtests/stock/data/raw"),
        accepted_symbols,
    )
    close_series: dict[str, Any] = {}
    start = min(_aware_utc(position.entry_time) for position in accepted)
    end = max(_aware_utc(position.exit_time) for position in accepted)
    for symbol in accepted_symbols:
        frame = bars_by_symbol.get(symbol)
        if frame is None or "close" not in frame:
            continue
        series = _normalized_close_series(frame)
        series = series[(series.index >= start) & (series.index <= end)]
        if len(series) > 0:
            close_series[symbol] = series
    if not close_series:
        return {
            "risk_basis": "realized_only_mtm_unavailable",
            "final_equity": final_equity,
            "max_drawdown_pct": 0.0,
            "max_drawdown_dollar": 0.0,
            "calmar": 0.0,
            "points": 0,
            "priced_symbol_count": 0,
            "accepted_symbol_count": len(accepted_symbols),
            "missing_price_symbols": sorted(accepted_symbols),
        }

    import pandas as pd

    timeline = pd.Index([])
    for series in close_series.values():
        timeline = timeline.union(series.index)
    timeline = timeline.sort_values()
    if len(timeline) == 0:
        return {
            "risk_basis": "realized_only_mtm_unavailable",
            "final_equity": final_equity,
            "max_drawdown_pct": 0.0,
            "max_drawdown_dollar": 0.0,
            "calmar": 0.0,
            "points": 0,
            "priced_symbol_count": len(close_series),
            "accepted_symbol_count": len(accepted_symbols),
            "missing_price_symbols": sorted(accepted_symbols.difference(close_series)),
        }
    close_on_timeline = {
        symbol: series.reindex(timeline, method="ffill").to_numpy(dtype=float)
        for symbol, series in close_series.items()
    }
    events: list[tuple[datetime, int, int]] = []
    for idx, position in enumerate(accepted):
        events.append((_aware_utc(position.exit_time), 0, idx))
        events.append((_aware_utc(position.entry_time), 1, idx))
    events.sort(key=lambda item: (item[0], item[1]))

    realized_equity = float(initial_equity)
    open_ids: set[int] = set()
    event_idx = 0
    values: list[float] = [float(initial_equity)]
    timestamps: list[datetime] = [_aware_utc(timeline[0].to_pydatetime())]
    for step, ts in enumerate(timeline):
        ts_dt = _aware_utc(ts.to_pydatetime())
        while event_idx < len(events) and events[event_idx][0] <= ts_dt:
            _, event_type, position_idx = events[event_idx]
            position = accepted[position_idx]
            if event_type == 0:
                open_ids.discard(position_idx)
                realized_equity += position.pnl
            elif _aware_utc(position.exit_time) > ts_dt:
                open_ids.add(position_idx)
            event_idx += 1
        unrealized = 0.0
        for position_idx in open_ids:
            position = accepted[position_idx]
            prices = close_on_timeline.get(position.symbol)
            if prices is None:
                continue
            close_price = float(prices[step])
            if not np.isfinite(close_price):
                continue
            unrealized += (
                (close_price - position.entry_price)
                * float(position.direction)
                * position.quantity
            )
        values.append(realized_equity + unrealized)
        timestamps.append(ts_dt)

    while event_idx < len(events):
        event_time, event_type, position_idx = events[event_idx]
        if event_type == 0:
            open_ids.discard(position_idx)
            realized_equity += accepted[position_idx].pnl
            values.append(realized_equity)
            timestamps.append(event_time)
        event_idx += 1

    curve = np.asarray(values, dtype=float)
    max_dd_pct, max_dd_dollar = compute_max_drawdown(curve)
    span_seconds = max((timestamps[-1] - timestamps[0]).total_seconds(), 0.0) if len(timestamps) >= 2 else 0.0
    years = span_seconds / (365.25 * 24 * 3600) if span_seconds > 0 else 1.0
    cagr = compute_cagr(float(initial_equity), final_equity, years)
    missing_symbols = accepted_symbols.difference(close_series)
    risk_basis = (
        "bar_close_mark_to_market"
        if not missing_symbols
        else "partial_bar_close_mark_to_market"
    )
    return {
        "risk_basis": risk_basis,
        "final_equity": final_equity,
        "max_drawdown_pct": float(max_dd_pct),
        "max_drawdown_dollar": float(max_dd_dollar),
        "cagr": float(cagr),
        "calmar": float(cagr / max(max_dd_pct, 1e-9)),
        "points": int(len(curve)),
        "price_source": str(data_dir) if data_dir is not None else "in_memory",
        "priced_symbol_count": len(close_series),
        "accepted_symbol_count": len(accepted_symbols),
        "missing_price_symbols": sorted(missing_symbols),
    }


def _load_stock_price_bars(data_dir: Path, symbols: set[str]) -> dict[str, Any]:
    bars: dict[str, Any] = {}
    for symbol in symbols:
        path = bar_path(data_dir, symbol, "5m")
        if path.exists():
            bars[symbol] = load_bars(path)
    return bars


def _normalized_close_series(frame):
    import pandas as pd

    series = frame["close"].copy()
    series.index = pd.to_datetime(series.index, utc=True)
    return series.sort_index()


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def build_effective_portfolio_config(
    mutations: dict[str, Any] | None,
    *,
    initial_equity: float = INITIAL_EQUITY,
) -> dict[str, Any]:
    config = deepcopy(SEED_PORTFOLIO_CONFIG)
    config["initial_equity"] = float(initial_equity)
    for key, value in (mutations or {}).items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            _deep_merge(config[key], value)
        elif key in config:
            config[key] = deepcopy(value)
        elif "." in key:
            _set_path(config, key.split("."), value)
        else:
            config[key] = deepcopy(value)
    return config


def _load_latest_strategy_mutations(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    alcb_path, iaric_path = _latest_strategy_mutation_paths(repo_root)
    alcb = _read_json(alcb_path)
    iaric = _read_json(iaric_path)
    return dict(alcb), dict(iaric)


def _latest_strategy_mutation_paths(repo_root: Path) -> tuple[Path, Path]:
    return (
        _latest_optimized_config_path(repo_root, "alcb"),
        _latest_optimized_config_path(repo_root, "iaric"),
    )


def _latest_optimized_config_path(repo_root: Path, strategy: str) -> Path:
    strategy_dir = repo_root / "backtests" / "output" / "stock" / strategy
    manager = RoundManager("stock", strategy, base_dir=repo_root / "backtests" / "output")
    if manager.manifest_path.exists():
        latest_round = manager.get_latest_round()
        if latest_round < 1:
            raise FileNotFoundError(f"No active manifest round for stock/{strategy} under {strategy_dir}")
        path = manager.optimized_config_path(manager.round_path(latest_round))
        if not path.exists():
            raise FileNotFoundError(f"Active manifest round {latest_round} is missing {path}")
        return path

    candidates: list[tuple[int, Path]] = []
    for path in strategy_dir.glob("round_*/optimized_config.json"):
        try:
            round_num = int(path.parent.name.removeprefix("round_"))
        except ValueError:
            continue
        candidates.append((round_num, path))
    if not candidates:
        raise FileNotFoundError(f"No optimized stock config found for {strategy} under {strategy_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_merge(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(target.get(key), dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def _set_path(target: dict[str, Any], parts: list[str], value: Any) -> None:
    cursor = target
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = deepcopy(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
