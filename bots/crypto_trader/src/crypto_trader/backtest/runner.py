"""Backtest runner — wires components and executes strategy over historical data."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.metrics import PerformanceMetrics, compute_metrics
from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.core.clock import SimClock
from crypto_trader.core.engine import StrategyEngine
from crypto_trader.core.events import EventBus
from crypto_trader.core.models import TerminalMark, TimeFrame, Trade
from crypto_trader.data.historical_feed import HistoricalFeed
from crypto_trader.data.store import ParquetStore
from crypto_trader.exchange.funding import FundingHelper
from crypto_trader.exchange.meta import AssetMeta

log = structlog.get_logger()


def _create_strategy(strategy_type: str, strategy_config, bot_id: str = ""):
    """Factory: create strategy instance, timeframes list, and primary TF."""
    if strategy_type == "trend":
        from crypto_trader.strategy.trend.config import TrendConfig
        from crypto_trader.strategy.trend.strategy import TrendStrategy
        cfg = strategy_config if isinstance(strategy_config, TrendConfig) else TrendConfig()
        return TrendStrategy(cfg, bot_id=bot_id), [TimeFrame.M15, TimeFrame.H1, TimeFrame.D1], TimeFrame.M15
    elif strategy_type == "breakout":
        from crypto_trader.strategy.breakout.config import BreakoutConfig
        from crypto_trader.strategy.breakout.strategy import BreakoutStrategy
        cfg = strategy_config if isinstance(strategy_config, BreakoutConfig) else BreakoutConfig()
        return BreakoutStrategy(cfg, bot_id=bot_id), [TimeFrame.M30, TimeFrame.H4], TimeFrame.M30
    else:
        from crypto_trader.strategy.momentum.config import MomentumConfig
        from crypto_trader.strategy.momentum.strategy import MomentumStrategy
        cfg = strategy_config if isinstance(strategy_config, MomentumConfig) else MomentumConfig()
        return MomentumStrategy(cfg, bot_id=bot_id), [TimeFrame.M15, TimeFrame.H1, TimeFrame.H4], TimeFrame.M15


@dataclass
class BacktestResult:
    trades: list[Trade]
    terminal_marks: list[TerminalMark]
    equity_curve: list[tuple[datetime, float]]
    metrics: PerformanceMetrics
    journal: object  # TradeJournal — strategy-agnostic
    config: BacktestConfig | None = None
    diagnostic_context: dict[str, object] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    train: BacktestResult
    test: BacktestResult


@dataclass
class SplitContinuationResult:
    """Backtest split where OOS continues from the exact IS checkpoint."""

    in_sample: BacktestResult
    out_of_sample: BacktestResult
    stitched: BacktestResult
    checkpoint: dict[str, Any] = field(default_factory=dict)


@dataclass
class _BacktestComponents:
    strategy: Any
    broker: SimBroker
    engine: StrategyEngine
    feed: HistoricalFeed
    config: BacktestConfig


@dataclass
class _MetricBrokerView:
    _initial_equity: float
    _closed_trades: list[Trade]
    _terminal_marks: list[TerminalMark]
    _equity_history: list[tuple[datetime, float]]
    _liquidation_equity_history: list[tuple[datetime, float]]

    @property
    def initial_equity(self) -> float:
        return self._initial_equity


def _to_utc_start(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.combine(value, dt_time.min, tzinfo=timezone.utc)


def _to_utc_end(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.combine(value, dt_time.max, tzinfo=timezone.utc)


def _measurement_start(value: date | datetime | None) -> datetime | None:
    if value is None:
        return None
    return _to_utc_start(value)


def _warmup_start(config: BacktestConfig) -> date | datetime | None:
    actual_start = config.start_date
    if config.warmup_days > 0 and actual_start is not None:
        return actual_start - timedelta(days=config.warmup_days)
    return actual_start


def _build_components(
    strategy_config,
    backtest_config: BacktestConfig,
    data_dir: Path = Path("data"),
    meta_path: Path | None = None,
    store=None,
    strategy_type: str = "momentum",
    *,
    feed_start: date | datetime | None = None,
    feed_end: date | datetime | None = None,
) -> _BacktestComponents:
    symbols = backtest_config.symbols or strategy_config.symbols
    backtest_config.symbols = symbols
    strategy_config.symbols = symbols

    if store is None:
        store = ParquetStore(base_dir=data_dir)

    asset_meta = None
    if meta_path and meta_path.exists():
        asset_meta = AssetMeta.from_cache(meta_path)

    funding_helpers: dict[str, FundingHelper] = {}
    if backtest_config.apply_funding:
        for sym in symbols:
            df = store.load_funding(sym)
            if df is not None and not df.empty:
                funding_helpers[sym] = FundingHelper(df)

    strategy, timeframes, primary_tf = _create_strategy(strategy_type, strategy_config)

    feed = HistoricalFeed(
        store=store,
        symbols=symbols,
        timeframes=timeframes,
        start_date=feed_start if feed_start is not None else _warmup_start(backtest_config),
        end_date=feed_end if feed_end is not None else backtest_config.end_date,
        primary_timeframe=primary_tf,
    )

    broker = SimBroker(
        initial_equity=backtest_config.initial_equity,
        taker_fee_bps=backtest_config.taker_fee_bps,
        maker_fee_bps=backtest_config.maker_fee_bps,
        slippage_bps=backtest_config.slippage_bps,
        spread_bps=backtest_config.spread_bps,
        asset_meta=asset_meta,
        funding_helpers=funding_helpers if funding_helpers else None,
    )

    engine = StrategyEngine(
        strategy=strategy,
        broker=broker,
        feed=feed,
        clock=SimClock(),
        events=EventBus(),
        config=backtest_config,
        primary_timeframe=primary_tf,
    )
    return _BacktestComponents(
        strategy=strategy,
        broker=broker,
        engine=engine,
        feed=feed,
        config=backtest_config,
    )


def _save_journal(strategy: Any) -> None:
    if hasattr(strategy, "_journal") and hasattr(strategy._journal, "save"):
        strategy._journal.save()


def _trim_equity_to_measurement_start(
    broker: SimBroker,
    measurement_start: datetime | None,
) -> None:
    if measurement_start is None:
        return
    filtered_equity = [
        (ts, eq) for ts, eq in broker._equity_history
        if ts >= measurement_start
    ]
    filtered_liquidation_equity = [
        (ts, eq) for ts, eq in broker._liquidation_equity_history
        if ts >= measurement_start
    ]
    if filtered_equity:
        broker._equity_history = filtered_equity
    if filtered_liquidation_equity:
        broker._liquidation_equity_history = filtered_liquidation_equity
    initial_curve = filtered_liquidation_equity or filtered_equity
    if initial_curve:
        broker._initial_equity = initial_curve[0][1]


def _diagnostic_context(strategy: Any) -> dict[str, object]:
    export_diag_context = getattr(strategy, "export_diagnostic_context", None)
    if not callable(export_diag_context):
        return {}
    try:
        payload = export_diag_context()
        if isinstance(payload, dict):
            return payload
    except Exception:
        log.exception("backtest.export_diagnostic_context.failed")
    return {}


def _result_from_components(
    components: _BacktestComponents,
    terminal_marks: list[TerminalMark],
    *,
    trim_start: datetime | None = None,
) -> BacktestResult:
    broker = components.broker
    if trim_start is not None:
        _trim_equity_to_measurement_start(broker, trim_start)
    metrics = compute_metrics(broker)
    return BacktestResult(
        trades=broker._closed_trades,
        terminal_marks=terminal_marks,
        equity_curve=broker._liquidation_equity_history or broker._equity_history,
        metrics=metrics,
        journal=components.strategy.journal,
        config=components.config,
        diagnostic_context=_diagnostic_context(components.strategy),
    )


def _history_from_split(
    history: list[tuple[datetime, float]],
    split_at: datetime,
    checkpoint_equity: float,
) -> list[tuple[datetime, float]]:
    segment = [(ts, eq) for ts, eq in history if ts >= split_at]
    return [(split_at, checkpoint_equity), *segment]


def _out_of_sample_result_from_stitched(
    components: _BacktestComponents,
    terminal_marks: list[TerminalMark],
    *,
    split_at: datetime,
    checkpoint_equity: float,
    checkpoint_liquidation_equity: float,
    closed_trade_count: int,
) -> BacktestResult:
    broker = components.broker
    segment_broker = _MetricBrokerView(
        _initial_equity=checkpoint_liquidation_equity,
        _closed_trades=broker._closed_trades[closed_trade_count:],
        _terminal_marks=terminal_marks,
        _equity_history=_history_from_split(
            broker._equity_history,
            split_at,
            checkpoint_equity,
        ),
        _liquidation_equity_history=_history_from_split(
            broker._liquidation_equity_history,
            split_at,
            checkpoint_liquidation_equity,
        ),
    )
    return BacktestResult(
        trades=segment_broker._closed_trades,
        terminal_marks=terminal_marks,
        equity_curve=(
            segment_broker._liquidation_equity_history
            or segment_broker._equity_history
        ),
        metrics=compute_metrics(segment_broker),
        journal=components.strategy.journal,
        config=components.config,
        diagnostic_context=_diagnostic_context(components.strategy),
    )


def run(
    strategy_config,
    backtest_config: BacktestConfig,
    data_dir: Path = Path("data"),
    meta_path: Path | None = None,
    store=None,
    strategy_type: str = "momentum",
) -> BacktestResult:
    """Run a full backtest with the specified strategy."""
    components = _build_components(
        strategy_config,
        backtest_config,
        data_dir=data_dir,
        meta_path=meta_path,
        store=store,
        strategy_type=strategy_type,
    )

    log.info(
        "backtest.start",
        symbols=components.config.symbols,
        start=str(backtest_config.start_date),
        end=str(backtest_config.end_date),
        equity=backtest_config.initial_equity,
    )

    components.engine.run()
    terminal_marks = components.engine.mark_open_positions()

    # Re-save journal so persisted artifacts include the final realized trades.
    _save_journal(components.strategy)

    # Trim warmup-period equity only — entries are blocked before measurement start.
    trim_start = (
        _measurement_start(backtest_config.start_date)
        if backtest_config.warmup_days > 0
        else None
    )
    return _result_from_components(
        components,
        terminal_marks,
        trim_start=trim_start,
    )


def run_split_continuation(
    strategy_config,
    backtest_config: BacktestConfig,
    split_date: date | datetime,
    data_dir: Path = Path("data"),
    meta_path: Path | None = None,
    store=None,
    strategy_type: str = "momentum",
) -> SplitContinuationResult:
    """Run IS, checkpoint at ``split_date``, then continue OOS from that state."""
    if backtest_config.start_date is None or backtest_config.end_date is None:
        raise ValueError("start_date and end_date required for split continuation")

    split_at = _to_utc_start(split_date)
    if split_at <= _to_utc_start(backtest_config.start_date):
        raise ValueError("split_date must be after start_date")
    if split_at > _to_utc_end(backtest_config.end_date):
        raise ValueError("split_date must be on or before end_date")

    base_strategy_config = deepcopy(strategy_config)
    pre_components = _build_components(
        deepcopy(base_strategy_config),
        deepcopy(backtest_config),
        data_dir=data_dir,
        meta_path=meta_path,
        store=store,
        strategy_type=strategy_type,
    )

    log.info(
        "backtest.split.start",
        symbols=pre_components.config.symbols,
        start=str(backtest_config.start_date),
        split=split_at.isoformat(),
        end=str(backtest_config.end_date),
        equity=backtest_config.initial_equity,
    )

    pre_components.strategy.on_init(pre_components.engine._ctx)
    for bar in pre_components.feed:
        if bar.timestamp >= split_at:
            break
        pre_components.engine._process_single_bar(bar)

    state = pre_components.engine.snapshot_state()
    closed_trade_count = len(pre_components.broker._closed_trades)
    checkpoint_equity = pre_components.broker.get_equity()
    checkpoint_liquidation_equity = (
        pre_components.broker._liquidation_equity_history[-1][1]
        if pre_components.broker._liquidation_equity_history
        else checkpoint_equity
    )

    pre_components.strategy.on_shutdown(pre_components.engine._ctx)
    pre_terminal_marks = pre_components.engine.mark_open_positions()
    _save_journal(pre_components.strategy)
    trim_start = (
        _measurement_start(backtest_config.start_date)
        if backtest_config.warmup_days > 0
        else None
    )
    in_sample = _result_from_components(
        pre_components,
        pre_terminal_marks,
        trim_start=trim_start,
    )

    oos_config = deepcopy(backtest_config)
    oos_config.start_date = split_at
    oos_config.warmup_days = 0
    oos_components = _build_components(
        deepcopy(base_strategy_config),
        oos_config,
        data_dir=data_dir,
        meta_path=meta_path,
        store=store,
        strategy_type=strategy_type,
        feed_start=split_at,
        feed_end=backtest_config.end_date,
    )

    oos_components.strategy.on_init(oos_components.engine._ctx)
    oos_components.engine.restore_state(state)
    for bar in oos_components.feed:
        oos_components.engine._process_single_bar(bar)
    oos_components.strategy.on_shutdown(oos_components.engine._ctx)
    terminal_marks = oos_components.engine.mark_open_positions()
    _save_journal(oos_components.strategy)

    stitched = _result_from_components(
        oos_components,
        terminal_marks,
        trim_start=trim_start,
    )
    out_of_sample = _out_of_sample_result_from_stitched(
        oos_components,
        terminal_marks,
        split_at=split_at,
        checkpoint_equity=checkpoint_equity,
        checkpoint_liquidation_equity=checkpoint_liquidation_equity,
        closed_trade_count=closed_trade_count,
    )

    return SplitContinuationResult(
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        stitched=stitched,
        checkpoint={
            "split_at": split_at,
            "checkpoint_equity": checkpoint_equity,
            "checkpoint_liquidation_equity": checkpoint_liquidation_equity,
            "closed_trade_count": closed_trade_count,
            "bar_count": state.get("runtime", {}).get("bar_count"),
            "fill_count": state.get("runtime", {}).get("fill_count"),
        },
    )


def run_walk_forward(
    strategy_config,
    backtest_config: BacktestConfig,
    data_dir: Path = Path("data"),
    meta_path: Path | None = None,
    strategy_type: str = "momentum",
) -> WalkForwardResult:
    """Run walk-forward analysis: train on first portion, test on remainder."""
    if not backtest_config.start_date or not backtest_config.end_date:
        raise ValueError("start_date and end_date required for walk-forward")

    total_days = (backtest_config.end_date - backtest_config.start_date).days
    train_days = int(total_days * backtest_config.train_pct)

    split_date = backtest_config.start_date + timedelta(days=train_days)

    # Train
    train_cfg = BacktestConfig(
        symbols=backtest_config.symbols,
        start_date=backtest_config.start_date,
        end_date=split_date,
        initial_equity=backtest_config.initial_equity,
        taker_fee_bps=backtest_config.taker_fee_bps,
        maker_fee_bps=backtest_config.maker_fee_bps,
        slippage_bps=backtest_config.slippage_bps,
        spread_bps=backtest_config.spread_bps,
        apply_funding=backtest_config.apply_funding,
        warmup_days=backtest_config.warmup_days,
    )
    train_result = run(strategy_config, train_cfg, data_dir, meta_path, strategy_type=strategy_type)

    # Test
    test_cfg = BacktestConfig(
        symbols=backtest_config.symbols,
        start_date=split_date,
        end_date=backtest_config.end_date,
        initial_equity=backtest_config.initial_equity,
        taker_fee_bps=backtest_config.taker_fee_bps,
        maker_fee_bps=backtest_config.maker_fee_bps,
        slippage_bps=backtest_config.slippage_bps,
        spread_bps=backtest_config.spread_bps,
        apply_funding=backtest_config.apply_funding,
        warmup_days=backtest_config.warmup_days,
    )
    test_result = run(strategy_config, test_cfg, data_dir, meta_path, strategy_type=strategy_type)

    return WalkForwardResult(train=train_result, test=test_result)
