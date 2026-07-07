# tests/factories.py
"""Shared test data factories — canonical constructors for test objects.

Eliminates duplication of _make_trade(), _make_missed(), _make_handlers(), etc.
across 50+ test files.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from trading_assistant.schemas.events import (
    MissedOpportunityEvent,
    TradeEvent,
)
from trading_assistant.schemas.prompt_package import PromptPackage


# ---------------------------------------------------------------------------
# TradeEvent factory
# ---------------------------------------------------------------------------

_UNSET = object()  # sentinel for optional override


def make_trade(
    trade_id: str = "t1",
    bot_id: str = "bot1",
    pnl: float = 100.0,
    pair: str = "BTC/USDT",
    side: str = "LONG",
    entry_price: float = 100.0,
    exit_price: float | None = None,
    entry_time: datetime | str | None = None,
    exit_time: datetime | str | None = None,
    position_size: float = 1.0,
    pnl_pct: float | object = _UNSET,
    market_regime: str = "",
    process_quality_score: int = 100,
    root_causes: list[str] | None = None,
    entry_signal: str = "",
    exit_reason: str = "",
    spread_at_entry: float = 0.0,
    signal_factors: list[dict] | None = None,
    post_exit_1h_price: float | None = None,
    post_exit_4h_price: float | None = None,
    entry_fill_details: dict | None = None,
    exit_fill_details: dict | None = None,
    atr_at_entry: float = 0.0,
    active_filters: list[str] | None = None,
    signal_evolution: list[dict] | None = None,
    **kwargs,
) -> TradeEvent:
    """Create a TradeEvent with sensible defaults.

    All specialized _make_trade() variants across the test suite can be replaced
    by calling this with the relevant keyword arguments.
    """
    # Auto-derive exit_price from pnl if not given
    if exit_price is None:
        exit_price = entry_price + pnl

    # Parse string datetimes
    if isinstance(entry_time, str):
        entry_time = datetime.fromisoformat(entry_time)
    if isinstance(exit_time, str):
        exit_time = datetime.fromisoformat(exit_time)

    # Default times
    if entry_time is None:
        entry_time = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
    if exit_time is None:
        exit_time = datetime(2026, 3, 1, 11, 0, 0, tzinfo=timezone.utc)

    if pnl_pct is _UNSET:
        pnl_pct = (pnl / entry_price * 100) if entry_price else 0.0

    return TradeEvent(
        trade_id=trade_id,
        bot_id=bot_id,
        pair=pair,
        side=side,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        position_size=position_size,
        pnl=pnl,
        pnl_pct=pnl_pct,
        market_regime=market_regime,
        process_quality_score=process_quality_score,
        root_causes=root_causes if root_causes is not None else [],
        entry_signal=entry_signal,
        exit_reason=exit_reason,
        spread_at_entry=spread_at_entry,
        signal_factors=signal_factors,
        post_exit_1h_price=post_exit_1h_price,
        post_exit_4h_price=post_exit_4h_price,
        entry_fill_details=entry_fill_details,
        exit_fill_details=exit_fill_details,
        atr_at_entry=atr_at_entry,
        active_filters=active_filters if active_filters is not None else [],
        signal_evolution=signal_evolution,
        **kwargs,
    )


def make_trades(
    n: int = 10,
    bot_id: str = "bot1",
    pnl_base: float = 100.0,
) -> list[TradeEvent]:
    """Generate a batch of realistic trade events."""
    trades: list[TradeEvent] = []
    for i in range(n):
        win = i % 3 != 0
        pnl = pnl_base if win else -pnl_base * 0.5
        trades.append(make_trade(
            trade_id=f"t{i}",
            bot_id=bot_id,
            pnl=pnl,
            market_regime="trending",
            exit_reason="TAKE_PROFIT" if win else "STOP_LOSS",
            root_causes=["normal_win"] if win else ["normal_loss"],
            process_quality_score=85,
        ))
    return trades


# ---------------------------------------------------------------------------
# MissedOpportunityEvent factory
# ---------------------------------------------------------------------------

def make_missed(
    bot_id: str = "bot1",
    pair: str = "BTC/USDT",
    signal: str = "momentum",
    blocked_by: str = "",
    outcome_24h: float = 50.0,
    confidence: float = 0.8,
    hypothetical_entry: float = 100.0,
    assumption_tags: list[str] | None = None,
    margin_pct: float | None = None,
    **kwargs,
) -> MissedOpportunityEvent:
    """Create a MissedOpportunityEvent with sensible defaults."""
    return MissedOpportunityEvent(
        bot_id=bot_id,
        pair=pair,
        signal=signal,
        blocked_by=blocked_by,
        hypothetical_entry=hypothetical_entry,
        outcome_24h=outcome_24h,
        confidence=confidence,
        assumption_tags=assumption_tags if assumption_tags is not None else [],
        margin_pct=margin_pct,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Error event factory (returns dict, matching existing test patterns)
# ---------------------------------------------------------------------------

def make_error_event(
    event_id: str = "err1",
    bot_id: str = "bot1",
    severity: str = "HIGH",
    error_type: str = "ConnectionError",
    **kwargs,
) -> dict:
    """Create an error event dict."""
    base = {
        "event_id": event_id,
        "bot_id": bot_id,
        "severity": severity,
        "error_type": error_type,
        "message": f"Test {error_type}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# PromptPackage factory
# ---------------------------------------------------------------------------

def make_sample_package(
    system_prompt: str = "You are a trading analyst.",
    task_prompt: str = "Analyze today's performance.",
    data: dict | None = None,
    instructions: str = "",
    **kwargs,
) -> PromptPackage:
    """Create a PromptPackage with sensible defaults."""
    return PromptPackage(
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        data=data if data is not None else {"summary": {"pnl": 100}},
        instructions=instructions,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Handlers factory
# ---------------------------------------------------------------------------

def make_handlers(
    tmp_path: Path,
    agent_runner: object | None = None,
    event_stream: object | None = None,
    dispatcher: object | None = None,
    notification_prefs: object | None = None,
    suggestion_tracker: object | None = None,
    bots: list[str] | None = None,
    create_policy_files: bool = True,
    **kwargs,
):
    """Create a Handlers instance with standard test defaults.

    Returns ``(handlers, agent_runner, event_stream)`` tuple so callers
    can inspect mocks.
    """
    from trading_assistant.orchestrator.event_stream import EventStream
    from trading_assistant.orchestrator.handlers import Handlers
    from trading_assistant.skills.suggestion_tracker import SuggestionTracker

    memory_dir = tmp_path / "memory"
    findings_dir = memory_dir / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)
    policies_dir = memory_dir / "policies" / "v1"
    policies_dir.mkdir(parents=True, exist_ok=True)

    if create_policy_files:
        (policies_dir / "agent.md").write_text("Agent policy")
        (policies_dir / "trading_rules.md").write_text("Rules")
        (policies_dir / "soul.md").write_text("Soul")

    if agent_runner is None:
        agent_runner = MagicMock()
        agent_runner.invoke = AsyncMock()
    if event_stream is None:
        event_stream = EventStream()
    if dispatcher is None:
        dispatcher = AsyncMock()
    if notification_prefs is None:
        notification_prefs = MagicMock()
    if suggestion_tracker is None:
        suggestion_tracker = SuggestionTracker(store_dir=findings_dir)
    if bots is None:
        bots = ["bot1", "bot2"]

    curated_dir = kwargs.pop("curated_dir", tmp_path / "data" / "curated")
    runs_dir = kwargs.pop("runs_dir", tmp_path / "runs")
    source_root = kwargs.pop("source_root", tmp_path)

    handlers = Handlers(
        agent_runner=agent_runner,
        event_stream=event_stream,
        dispatcher=dispatcher,
        notification_prefs=notification_prefs,
        curated_dir=curated_dir,
        memory_dir=memory_dir,
        runs_dir=runs_dir,
        source_root=source_root,
        bots=bots,
        suggestion_tracker=suggestion_tracker,
        **kwargs,
    )
    return handlers, agent_runner, event_stream


# ---------------------------------------------------------------------------
# AppConfig factory
# ---------------------------------------------------------------------------

def make_config(**overrides) -> object:
    """Create an AppConfig with test defaults."""
    from trading_assistant.orchestrator.config import AppConfig

    defaults = {
        "bot_ids": ["bot1", "bot2"],
        "relay_url": "http://localhost:9000",
        "data_dir": "/tmp/test_data",
    }
    defaults.update(overrides)
    return AppConfig(**defaults)
