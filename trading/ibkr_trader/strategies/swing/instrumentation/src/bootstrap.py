"""Bootstrap instrumentation — factory that reads config and builds all services.

Usage::

    from strategies.swing.instrumentation.src.bootstrap import bootstrap_instrumentation, bootstrap_kit

    # Create context directly
    ctx = bootstrap_instrumentation(symbols=["QQQ", "SPY"])
    ctx.start()

    # Or create a Kit facade (recommended for most use cases)
    kit = bootstrap_kit(strategy_id="ATRSS", symbols=["QQQ", "SPY"])
    kit._ctx.start()
    trade = kit.log_entry(...)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("instrumentation.bootstrap")

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "instrumentation_config.yaml"


def _resolve_applied_portfolio_rules_config(get_applied_config) -> object | None:
    if not callable(get_applied_config):
        return None
    try:
        return get_applied_config()
    except Exception as exc:
        logger.warning("Failed to read applied portfolio rules config for instrumentation lineage: %s", exc)
        return None


def bootstrap_instrumentation(
    symbols: list[str] | None = None,
    data_provider=None,
    strategy_id: str | None = None,
    initial_equity: float = 100_000,
    coordinator=None,
    get_regime_ctx=None,
    get_applied_config=None,
    pg_store=None,
) -> "InstrumentationContext":
    """Create an InstrumentationContext with all services wired up.

    Args:
        symbols: Active trading symbols (populates market_snapshots.symbols).
        data_provider: Optional data source for snapshots/regime. None is fine —
            snapshots degrade gracefully to zeros.
        strategy_id: Optional strategy identifier to override config bot_id.
            If provided, sets config["bot_id"] = strategy_id.

    Returns:
        Fully wired InstrumentationContext ready for ``ctx.start()``.
    """
    from .context import InstrumentationContext
    from .market_snapshot import MarketSnapshotService
    from .trade_logger import TradeLogger
    from .missed_opportunity import MissedOpportunityLogger
    from .process_scorer import ProcessScorer
    from .daily_snapshot import DailySnapshotBuilder
    from .regime_classifier import RegimeClassifier
    from .sidecar import Sidecar
    from libs.instrumentation.lineage import lineage_from_config

    config = _load_config()
    if strategy_id:
        config["bot_id"] = strategy_id
        config["strategy_id"] = strategy_id
    config["family_id"] = "swing"
    lineage = lineage_from_config(
        config,
        family_id="swing",
        strategy_id=config.get("strategy_id", strategy_id or ""),
        portfolio_rules_config=_resolve_applied_portfolio_rules_config(get_applied_config),
    )
    config["lineage"] = lineage

    # Populate symbols into config
    if symbols:
        config.setdefault("market_snapshots", {})["symbols"] = list(symbols)

    snapshot_service = MarketSnapshotService(config, data_provider=data_provider)
    trade_logger = TradeLogger(config, snapshot_service, coordinator=coordinator)
    missed_logger = MissedOpportunityLogger(config, snapshot_service)
    process_scorer = ProcessScorer()
    regime_classifier = RegimeClassifier(data_provider=data_provider)
    sidecar = Sidecar(config)

    from .drawdown_tracker import DrawdownTracker
    from .overnight_gap_tracker import OvernightGapTracker
    from .coordination_logger import CoordinationLogger
    from .order_logger import OrderLogger
    from .indicator_logger import IndicatorLogger
    from .filter_logger import FilterLogger
    from .orderbook_logger import OrderBookLogger
    from .experiment_registry import ExperimentRegistry
    drawdown_tracker = DrawdownTracker(initial_equity=initial_equity)
    gap_tracker = OvernightGapTracker()
    coordination_logger = CoordinationLogger(config)
    order_logger = OrderLogger(config)
    indicator_logger = IndicatorLogger(config)
    filter_logger = FilterLogger(config)
    orderbook_logger = OrderBookLogger(config)

    experiments_path = Path(config["data_dir"]).parent / "config" / "experiments.yaml"
    experiment_registry = ExperimentRegistry(config_path=experiments_path)
    daily_builder = DailySnapshotBuilder(config, experiment_registry=experiment_registry, get_regime_ctx=get_regime_ctx, get_applied_config=get_applied_config)

    post_exit_tracker = None
    if data_provider is not None:
        try:
            from .post_exit_tracker import PostExitTracker
            post_exit_tracker = PostExitTracker(
                data_dir=config["data_dir"],
                data_provider=data_provider,
            )
        except Exception as e:
            logger.warning("PostExitTracker creation failed: %s", e)

    ctx = InstrumentationContext(
        snapshot_service=snapshot_service,
        trade_logger=trade_logger,
        missed_logger=missed_logger,
        process_scorer=process_scorer,
        daily_builder=daily_builder,
        regime_classifier=regime_classifier,
        sidecar=sidecar,
        drawdown_tracker=drawdown_tracker,
        overnight_gap_tracker=gap_tracker,
        coordination_logger=coordination_logger,
        order_logger=order_logger,
        indicator_logger=indicator_logger,
        filter_logger=filter_logger,
        orderbook_logger=orderbook_logger,
        experiment_registry=experiment_registry,
        post_exit_tracker=post_exit_tracker,
        bot_id=config.get("bot_id", "swing_multi_01"),
        data_dir=config["data_dir"],
        lineage=lineage,
        get_regime_ctx=get_regime_ctx,
        get_applied_config=get_applied_config,
        pg_store=pg_store,
    )

    logger.info(
        "Instrumentation bootstrapped: symbols=%s, data_dir=%s",
        symbols, ctx.data_dir,
    )
    return ctx


def bootstrap_kit(
    strategy_id: str,
    symbols: list[str] | None = None,
    data_provider=None,
    initial_equity: float = 100_000,
    shared_ctx: "InstrumentationContext | None" = None,
    coordinator=None,
) -> "InstrumentationKit":
    """Create an InstrumentationKit with all services wired up.

    Convenience wrapper: bootstraps context + wraps in Kit facade.

    Args:
        strategy_id: Strategy identifier (used as bot_id and for scoring).
        symbols: Active trading symbols.
        data_provider: Optional data source for snapshots/regime.
        shared_ctx: Optional shared InstrumentationContext. When provided,
            the kit reuses the shared context's services (snapshot_service,
            regime_classifier, sidecar, drawdown_tracker, etc.) instead of
            creating redundant instances. The kit gets its own TradeLogger
            and MissedOpportunityLogger (with strategy-specific bot_id) but
            shares everything else. The sidecar is NOT duplicated — only the
            shared context should start/stop the sidecar thread.

    Returns:
        InstrumentationKit ready for log_entry/log_exit calls.
        When using shared_ctx, do NOT call kit._ctx.start() — the shared
        context owns the sidecar lifecycle.
    """
    from .kit import InstrumentationKit

    if shared_ctx is not None:
        ctx = _bootstrap_kit_from_shared(strategy_id, shared_ctx, coordinator=coordinator)
    else:
        ctx = bootstrap_instrumentation(
            symbols=symbols,
            data_provider=data_provider,
            strategy_id=strategy_id,
            initial_equity=initial_equity,
            coordinator=coordinator,
        )
    return InstrumentationKit(ctx, strategy_id=strategy_id)


def _bootstrap_kit_from_shared(
    strategy_id: str,
    shared_ctx: "InstrumentationContext",
    coordinator=None,
) -> "InstrumentationContext":
    """Create a lightweight per-strategy context that reuses shared services.

    Gets its own TradeLogger and MissedOpportunityLogger (strategy-specific
    bot_id) but shares snapshot_service, regime_classifier, sidecar, etc.
    The sidecar is set to None so starting this context is a no-op.
    """
    from .context import InstrumentationContext
    from .trade_logger import TradeLogger
    from .missed_opportunity import MissedOpportunityLogger

    config = _load_config()
    config["bot_id"] = strategy_id
    config["strategy_id"] = strategy_id
    config["family_id"] = "swing"
    from libs.instrumentation.lineage import lineage_from_config
    lineage = lineage_from_config(
        config,
        family_id="swing",
        strategy_id=strategy_id,
        portfolio_rules_config=_resolve_applied_portfolio_rules_config(shared_ctx.get_applied_config),
    )
    config["lineage"] = lineage

    # Own loggers with strategy-specific bot_id
    trade_logger = TradeLogger(config, shared_ctx.snapshot_service, coordinator=coordinator)
    missed_logger = MissedOpportunityLogger(config, shared_ctx.snapshot_service)

    return InstrumentationContext(
        snapshot_service=shared_ctx.snapshot_service,
        trade_logger=trade_logger,
        missed_logger=missed_logger,
        process_scorer=shared_ctx.process_scorer,
        daily_builder=shared_ctx.daily_builder,
        regime_classifier=shared_ctx.regime_classifier,
        sidecar=None,  # shared context owns the single sidecar
        post_exit_tracker=None,  # shared context owns the backfill thread
        drawdown_tracker=shared_ctx.drawdown_tracker,
        overnight_gap_tracker=shared_ctx.overnight_gap_tracker,
        coordination_logger=shared_ctx.coordination_logger,
        order_logger=shared_ctx.order_logger,
        indicator_logger=shared_ctx.indicator_logger,
        filter_logger=shared_ctx.filter_logger,
        orderbook_logger=shared_ctx.orderbook_logger,
        experiment_registry=shared_ctx.experiment_registry,
        pg_store=shared_ctx.pg_store,
        bot_id=strategy_id,
        data_dir=shared_ctx.data_dir,
        lineage=lineage,
        get_regime_ctx=shared_ctx.get_regime_ctx,
        get_applied_config=shared_ctx.get_applied_config,
    )


def _load_config() -> dict:
    """Load instrumentation_config.yaml, applying defaults for any missing keys."""
    if _CONFIG_PATH.exists():
        try:
            import yaml
            with open(_CONFIG_PATH) as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Failed to load %s: %s — using defaults", _CONFIG_PATH, e)
            config = {}
    else:
        config = {}

    _default_data_dir = str(Path(__file__).resolve().parent.parent / "data")
    config.setdefault("bot_id", "swing_multi_01")
    config.setdefault("family_id", "swing")
    config.setdefault("data_dir", _default_data_dir)
    config.setdefault("data_source_id", "ibkr_execution")
    config["portfolio_id"] = config.get("portfolio_id") or os.environ.get("PORTFOLIO_ID") or "paper_default"
    config["account_alias"] = (
        config.get("account_alias")
        or os.environ.get("ACCOUNT_ALIAS")
        or os.environ.get("TRADING_ACCOUNT_ALIAS")
        or os.environ.get("BROKER_ACCOUNT_ALIAS")
        or "paper_ibkr_1"
    )
    config.setdefault("market_snapshots", {})
    config["market_snapshots"].setdefault("interval_seconds", 60)
    config["market_snapshots"].setdefault("symbols", [])
    config.setdefault("sidecar", {})
    config["sidecar"].setdefault("relay_url", "http://127.0.0.1:8000/events")
    config["sidecar"].setdefault("batch_size", 50)
    config["sidecar"].setdefault("retry_max", 5)
    config["sidecar"].setdefault("poll_interval_seconds", 60)
    return config
