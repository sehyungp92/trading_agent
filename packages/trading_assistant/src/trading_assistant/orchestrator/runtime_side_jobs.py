"""Runtime-owned scheduler side jobs."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading_assistant.orchestrator.config import AppConfig
from trading_assistant.orchestrator.tz_utils import bot_trading_date
from trading_assistant.schemas.notifications import NotificationPreferences
from trading_assistant.skills.proactive_scanner import ProactiveScanner

logger = logging.getLogger(__name__)


async def run_morning_scan(
    *,
    config: AppConfig,
    queue: Any,
    curated_dir: Path,
    dispatcher: Any,
    notification_preferences: NotificationPreferences,
    bot_ids: list[str] | None = None,
    scheduled_for: datetime | None = None,
) -> None:
    """Gather overnight data and dispatch morning scan notifications."""
    reference_time = (scheduled_for or datetime.now(timezone.utc)).astimezone(timezone.utc)
    scan_bots = bot_ids or config.bot_ids
    errors: list[dict] = []
    unusual_losses: list[dict] = []

    try:
        dead_letters = await queue.get_dead_letters(limit=20)
        for item in dead_letters:
            if isinstance(item, dict):
                errors.append(item)
    except Exception:
        logger.warning("Morning scan: could not read dead-letter queue")

    for bot_id in scan_bots:
        bot_config = config.bot_configs.get(bot_id) if config.bot_configs else None
        yesterday = bot_trading_date(
            bot_config.timezone if bot_config else "UTC",
            reference_time - timedelta(days=1),
        )
        summary_path = curated_dir / yesterday / bot_id / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            pnl = summary.get("net_pnl", 0)
            if pnl < 0:
                unusual_losses.append({
                    "bot_id": bot_id,
                    "date": yesterday,
                    "pnl": pnl,
                    "reason": f"Loss on {yesterday}",
                })
        except Exception:
            logger.warning("Morning scan: could not read summary for %s/%s", bot_id, yesterday)

    result = ProactiveScanner().morning_scan(
        events=[],
        errors=errors,
        unusual_losses=unusual_losses,
    )
    for payload in result.payloads:
        try:
            await dispatcher.dispatch(payload, notification_preferences, reference_time.hour)
        except Exception:
            logger.exception("Morning scan dispatch failed")


async def run_evening_report(
    *,
    config: AppConfig,
    curated_dir: Path,
    dispatcher: Any,
    notification_preferences: NotificationPreferences,
    bot_ids: list[str] | None = None,
    scheduled_for: datetime | None = None,
) -> None:
    """Check if daily report is ready and send evening notification."""
    reference_time = (scheduled_for or datetime.now(timezone.utc)).astimezone(timezone.utc)
    scan_bots = bot_ids or config.bot_ids
    daily_report_ready = False
    for bot_id in scan_bots:
        bot_config = config.bot_configs.get(bot_id) if config.bot_configs else None
        bot_date = bot_trading_date(bot_config.timezone if bot_config else "UTC", reference_time)
        if (curated_dir / bot_date / bot_id / "summary.json").exists():
            daily_report_ready = True
            break

    result = ProactiveScanner().evening_scan(
        date=reference_time.strftime("%Y-%m-%d"),
        daily_report_ready=daily_report_ready,
    )
    for payload in result.payloads:
        try:
            await dispatcher.dispatch(payload, notification_preferences, reference_time.hour)
        except Exception:
            logger.exception("Evening report dispatch failed")


async def run_market_data_sync(
    *,
    config: AppConfig,
    db_path: Path,
    event_stream: Any,
    scheduled_for: datetime | None = None,
) -> None:
    if not config.monthly_validation_enabled:
        return
    from trading_assistant.orchestrator.market_data_jobs import MarketDataSyncJob
    from trading_assistant.skills.monthly_validation_orchestrator import latest_completed_month

    run_month = latest_completed_month(scheduled_for)
    job = MarketDataSyncJob(
        market_data_root=(
            Path(config.market_data_root) if config.market_data_root else db_path / "market_data"
        ),
        strategy_registry=config.strategy_registry,
        event_stream=event_stream,
        required_coverage_ratio=config.market_data_required_coverage_ratio,
    )
    await asyncio.to_thread(job.run, run_month=run_month, bot_ids=config.bot_ids)


async def run_lineage_audit(
    *,
    config: AppConfig,
    curated_dir: Path,
    memory_dir: Path,
    proposal_ledger: Any,
    scheduled_for: datetime | None = None,
) -> None:
    from trading_assistant.orchestrator.lineage_audit import LineageAuditor
    from trading_assistant.skills.monthly_validation_orchestrator import (
        latest_completed_month,
        month_window,
    )

    run_month = latest_completed_month(scheduled_for)
    window_start, window_end = month_window(run_month)
    auditor = LineageAuditor(
        curated_dir=curated_dir,
        findings_dir=memory_dir / "findings",
        required_lineage_ratio=config.telemetry_required_lineage_ratio,
        proposal_ledger=proposal_ledger,
    )
    for bot_id in config.bot_ids:
        await asyncio.to_thread(
            auditor.audit,
            bot_id=bot_id,
            window_start=window_start,
            window_end=window_end,
        )
