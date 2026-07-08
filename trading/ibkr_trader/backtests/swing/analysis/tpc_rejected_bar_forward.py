"""Rejected-bar forward MFE/MAE diagnostics for TPC 30m pullback research."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backtests.swing.config_etf_base import ETFSlippageConfig
from backtests.swing.config_tpc import TPCBacktestConfig
from backtests.swing.engine.etf_engine_base import ETFStrategyBacktestEngine
from strategies.swing._shared.models import Direction
from strategies.swing.tpc import STRATEGY_ID, gates, indicators
from strategies.swing.tpc.config import TPCSymbolConfig
from strategies.swing.tpc.core import logic
from strategies.swing.tpc.core.state import TPCBarInput, TPCCoreState, TPCFill, TPCOrderUpdate


@dataclass(frozen=True)
class ForwardOutcome:
    symbol: str
    mfe_r: float
    mae_r: float
    terminal_r: float


@dataclass(frozen=True)
class RejectedForwardDiagnostic:
    name: str
    base_count: int
    kept_count: int
    rejected_count: int
    rejected_stats: dict[str, float]
    kept_stats: dict[str, float]
    rejected_by_symbol: dict[str, int]


def build_pb30_rejected_forward_report(
    replay_data: dict[str, dict[str, Any]],
    *,
    data_dir: Path,
    initial_equity: float,
    base_mutations: dict[str, Any],
    candidate_mutations: Iterable[tuple[str, dict[str, Any]]],
    horizon_bars_15m: int = 32,
    max_candidates: int = 12,
) -> str:
    """Compare base PB30 bars with bars rejected by candidate filters.

    This is diagnostics-only analytics. It uses the shared TPC setup lane for
    selection, then measures fixed-horizon forward excursion from the next
    completed 15m bar open so the diagnostic does not grant same-bar fills.
    """

    base_config = _config_with_overrides(data_dir, initial_equity, base_mutations)
    engine, prepared = _prepared_engine(base_config, replay_data, initial_equity)
    base_setups = _collect_pb30_setups(engine, prepared, base_config, initial_equity, horizon_bars_15m)
    if not base_setups:
        return "Rejected-bar forward MFE/MAE: no base PB30 setup bars found."

    base_forwards = {
        key: outcome
        for key, setup_info in base_setups.items()
        if (outcome := _forward_outcome(prepared[setup_info["symbol"]], setup_info, horizon_bars_15m)) is not None
    }
    if not base_forwards:
        return "Rejected-bar forward MFE/MAE: base PB30 setup bars had no forward horizon."

    diagnostics: list[RejectedForwardDiagnostic] = []
    for idx, (name, mutations) in enumerate(candidate_mutations):
        if idx >= max_candidates:
            break
        merged = dict(base_mutations)
        merged.update(mutations)
        candidate_config = _config_with_overrides(data_dir, initial_equity, merged)
        candidate_engine, candidate_prepared = _prepared_engine(candidate_config, replay_data, initial_equity)
        candidate_setups = _collect_pb30_setups(
            candidate_engine,
            candidate_prepared,
            candidate_config,
            initial_equity,
            horizon_bars_15m,
        )
        candidate_keys = set(candidate_setups)
        base_keys = list(base_forwards)
        rejected_keys = [key for key in base_keys if key not in candidate_keys]
        kept_keys = [key for key in base_keys if key in candidate_keys]
        diagnostics.append(
            RejectedForwardDiagnostic(
                name=name,
                base_count=len(base_keys),
                kept_count=len(kept_keys),
                rejected_count=len(rejected_keys),
                rejected_stats=_summarise_outcomes(base_forwards[key] for key in rejected_keys),
                kept_stats=_summarise_outcomes(base_forwards[key] for key in kept_keys),
                rejected_by_symbol=_symbol_counts(base_forwards[key] for key in rejected_keys),
            )
        )

    return _format_report(diagnostics, horizon_bars_15m)


def _config_with_overrides(data_dir: Path, initial_equity: float, mutations: dict[str, Any]) -> TPCBacktestConfig:
    config = TPCBacktestConfig(
        initial_equity=initial_equity,
        data_dir=data_dir,
        slippage=ETFSlippageConfig(),
    )
    return config.with_overrides(mutations)


def _prepared_engine(
    config: TPCBacktestConfig,
    replay_data: dict[str, dict[str, Any]],
    initial_equity: float,
) -> tuple[ETFStrategyBacktestEngine, dict[str, dict[str, Any]]]:
    engine = ETFStrategyBacktestEngine(
        strategy_id=STRATEGY_ID,
        configs=dict(config.symbol_configs),
        core_logic=logic,
        state_factory=TPCCoreState,
        bar_input_factory=TPCBarInput,
        fill_factory=TPCFill,
        order_update_factory=TPCOrderUpdate,
        indicator_module=indicators,
        slippage=config.slippage,
        initial_equity=initial_equity,
        warmup_15m=config.warmup_15m,
        indicator_cache={},
    )
    prepared = {
        symbol: engine._prepare_symbol(symbol, payload)
        for symbol, payload in replay_data.items()
        if symbol in config.symbol_configs
    }
    return engine, prepared


def _collect_pb30_setups(
    engine: ETFStrategyBacktestEngine,
    prepared: dict[str, dict[str, Any]],
    config: TPCBacktestConfig,
    initial_equity: float,
    horizon_bars_15m: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    setups: dict[tuple[str, int], dict[str, Any]] = {}
    for symbol, payload in prepared.items():
        cfg = config.symbol_configs.get(symbol)
        if cfg is None or not cfg.pb30_pullback_enabled:
            continue
        bars_15m = payload["bars_15m"]
        stop = max(int(config.warmup_15m), 1)
        final_i = max(stop, len(bars_15m) - horizon_bars_15m - 1)
        for i in range(stop, final_i):
            setup = _pb30_setup_at(engine, payload, symbol, i, cfg, initial_equity)
            if setup is None:
                continue
            setups[(symbol, i)] = {
                "symbol": symbol,
                "bar_index": i,
                "direction": setup.direction,
                "entry_order_type": setup.entry_order_type,
                "planned_entry": float(setup.entry_price),
                "stop_price": float(setup.stop_price),
            }
    return setups


def _pb30_setup_at(
    engine: ETFStrategyBacktestEngine,
    payload: dict[str, Any],
    symbol: str,
    i: int,
    cfg: TPCSymbolConfig,
    initial_equity: float,
):
    bar_input = engine._bar_input(symbol, payload, i, initial_equity)
    if bar_input is None or bar_input.bar_15m is None:
        return None
    bar = bar_input.bar_15m
    if not gates.session_filter(bar.timestamp, cfg) or not gates.news_filter(bar.timestamp, cfg):
        return None
    direction, grade, _reason = gates.regime_direction(bar_input, cfg)
    if direction == Direction.FLAT:
        return None
    if direction == Direction.LONG and not cfg.longs_enabled:
        return None
    if direction == Direction.SHORT and (not cfg.shorts_enabled or (cfg.shorts_require_a_plus and grade.value != "a_plus")):
        return None
    lane_cfg = logic._pb30_lane_config(cfg, cfg.pb30_entry_order_model or cfg.entry_order_model)
    return logic._evaluate_setup_lane(
        TPCCoreState(),
        bar_input,
        lane_cfg,
        direction,
        grade,
        lane_name="pb30",
        pullback_timeframe="30m",
    )


def _forward_outcome(
    payload: dict[str, Any],
    setup_info: dict[str, Any],
    horizon_bars_15m: int,
) -> ForwardOutcome | None:
    bars = payload["bars_15m"]
    signal_i = int(setup_info["bar_index"])
    entry_i = signal_i + 1
    if entry_i >= len(bars):
        return None
    end_i = min(len(bars) - 1, signal_i + max(1, int(horizon_bars_15m)))
    if end_i < entry_i:
        return None
    if str(setup_info.get("entry_order_type", "MARKET")).upper() == "MARKET":
        entry = float(bars.opens[entry_i])
    else:
        entry = float(setup_info["planned_entry"])
    stop = float(setup_info["stop_price"])
    risk = abs(entry - stop)
    if not np.isfinite(entry) or not np.isfinite(stop) or risk <= 0:
        return None
    highs = bars.highs[entry_i : end_i + 1]
    lows = bars.lows[entry_i : end_i + 1]
    terminal_close = float(bars.closes[end_i])
    direction = setup_info["direction"]
    if direction == Direction.LONG:
        mfe = (float(np.nanmax(highs)) - entry) / risk
        mae = (entry - float(np.nanmin(lows))) / risk
        terminal = (terminal_close - entry) / risk
    else:
        mfe = (entry - float(np.nanmin(lows))) / risk
        mae = (float(np.nanmax(highs)) - entry) / risk
        terminal = (entry - terminal_close) / risk
    return ForwardOutcome(
        symbol=str(setup_info["symbol"]),
        mfe_r=float(mfe),
        mae_r=float(mae),
        terminal_r=float(terminal),
    )


def _summarise_outcomes(outcomes: Iterable[ForwardOutcome]) -> dict[str, float]:
    items = list(outcomes)
    if not items:
        return {
            "count": 0.0,
            "avg_mfe_r": 0.0,
            "avg_mae_r": 0.0,
            "avg_terminal_r": 0.0,
            "hit_1r_rate": 0.0,
            "hit_stop_rate": 0.0,
        }
    mfes = np.asarray([item.mfe_r for item in items], dtype=float)
    maes = np.asarray([item.mae_r for item in items], dtype=float)
    terminals = np.asarray([item.terminal_r for item in items], dtype=float)
    return {
        "count": float(len(items)),
        "avg_mfe_r": float(np.mean(mfes)),
        "avg_mae_r": float(np.mean(maes)),
        "avg_terminal_r": float(np.mean(terminals)),
        "hit_1r_rate": float(np.mean(mfes >= 1.0)),
        "hit_stop_rate": float(np.mean(maes >= 1.0)),
    }


def _symbol_counts(outcomes: Iterable[ForwardOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in outcomes:
        counts[item.symbol] = counts.get(item.symbol, 0) + 1
    return dict(sorted(counts.items()))


def _format_report(diagnostics: list[RejectedForwardDiagnostic], horizon_bars_15m: int) -> str:
    lines = [
        "",
        "Rejected-bar forward MFE/MAE diagnostics:",
        (
            f"  Cohort: base PB30 setup bars; forward window={horizon_bars_15m} completed 15m bars; "
            "entry proxy=next 15m open for market entries; no same-bar fill credit."
        ),
    ]
    for item in diagnostics:
        rejected = item.rejected_stats
        kept = item.kept_stats
        rejection_rate = item.rejected_count / max(item.base_count, 1)
        symbol_bits = ", ".join(f"{symbol}:{count}" for symbol, count in item.rejected_by_symbol.items()) or "none"
        lines.append(
            "  "
            f"{item.name}: rejected {item.rejected_count}/{item.base_count} ({rejection_rate:.0%}), "
            f"rej MFE/MAE/terminal={rejected['avg_mfe_r']:+.2f}/"
            f"{rejected['avg_mae_r']:.2f}/{rejected['avg_terminal_r']:+.2f}R, "
            f"rej hit1R/stop={rejected['hit_1r_rate']:.0%}/{rejected['hit_stop_rate']:.0%}; "
            f"kept {item.kept_count}, kept MFE/MAE/terminal={kept['avg_mfe_r']:+.2f}/"
            f"{kept['avg_mae_r']:.2f}/{kept['avg_terminal_r']:+.2f}R; "
            f"symbols={symbol_bits}"
        )
    return "\n".join(lines)
