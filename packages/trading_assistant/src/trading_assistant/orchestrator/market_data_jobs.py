"""Scheduled market-data coverage sync/readiness jobs."""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.schemas.market_data_manifest import MarketDataManifest
from trading_assistant.skills.coverage_manifest_writer import CoverageManifestWriter
from trading_assistant.skills.market_data_catalog import MarketDataCatalog, MarketDataRequirement
from trading_assistant.skills.market_data_sync import (
    BotUploadedParquetAdapter,
    FileSystemParquetAdapter,
    KisMarketDataAdapter,
    MarketDataAdapter,
    MarketDataSyncRequest,
)
from trading_assistant.skills.monthly_validation_orchestrator import latest_completed_month, month_window

logger = logging.getLogger(__name__)


class MarketDataSyncJob:
    """Validate canonical data coverage and write manifests for monthly replay."""

    def __init__(
        self,
        *,
        market_data_root: Path,
        strategy_registry: object | None = None,
        event_stream: EventStream | None = None,
        required_coverage_ratio: float = 0.95,
    ) -> None:
        self.market_data_root = Path(market_data_root)
        self.catalog = MarketDataCatalog.from_strategy_registry(strategy_registry)
        self.event_stream = event_stream
        self.writer = CoverageManifestWriter(required_coverage_ratio=required_coverage_ratio)

    def run(
        self,
        *,
        run_month: str | None = None,
        bot_ids: list[str] | None = None,
    ) -> dict:
        run_month = run_month or latest_completed_month()
        selected_bots = set(bot_ids or [])
        requirements = [
            requirement for requirement in self.catalog.all_requirements()
            if not selected_bots or requirement.bot_id in selected_bots
        ]
        summary = {
            "run_month": run_month,
            "requirements": len(requirements),
            "manifests_written": 0,
            "authoritative": 0,
            "blocked": 0,
            "errors": [],
        }
        self._emit("market_data_sync_progress", {**summary, "stage": "started"})

        grouped: dict[tuple[str, str], list[MarketDataManifest]] = defaultdict(list)
        for requirement in requirements:
            manifest = self._sync_requirement(requirement, run_month, summary)
            grouped[(requirement.bot_id, requirement.strategy_id)].append(manifest)

        for (bot_id, strategy_id), manifests in sorted(grouped.items()):
            self._write_strategy_manifest(bot_id, strategy_id, run_month, manifests, summary)

        status = "ok" if summary["blocked"] == 0 else "blocked"
        self._emit("market_data_sync_done", {**summary, "status": status})
        return summary

    def _sync_requirement(
        self,
        requirement: MarketDataRequirement,
        run_month: str,
        summary: dict,
    ) -> MarketDataManifest:
        start_ts, end_ts, expected_bars = _month_request_window(run_month, requirement.timeframe)
        request = MarketDataSyncRequest(
            market=requirement.market,
            symbol=requirement.symbol,
            timeframe=requirement.timeframe,
            source=requirement.source,
            start_ts=start_ts,
            end_ts=end_ts,
            expected_bars=expected_bars,
            destination_path=(
                self.market_data_root
                / requirement.source
                / requirement.market
                / requirement.symbol
                / requirement.timeframe
                / f"{run_month}.parquet"
            ),
        )
        adapter = self._adapter_for(requirement.source)
        result = adapter.sync(request)
        manifest = result.manifest or adapter.validate_coverage(request)
        if result.error and result.error not in manifest.blocking_reasons:
            manifest.blocking_reasons.append(result.error)
            manifest.usable_for_authoritative_validation = False
        output_path = MarketDataCatalog.manifest_path(self.market_data_root, requirement, run_month)
        self.writer.write(manifest, output_path)
        summary["manifests_written"] += 1
        if manifest.usable_for_authoritative_validation:
            summary["authoritative"] += 1
        else:
            summary["blocked"] += 1
            summary["errors"].append(
                f"{requirement.bot_id}/{requirement.strategy_id}/{requirement.symbol}: "
                + "; ".join(manifest.blocking_reasons or ["not authoritative"])
            )
        return manifest

    def _write_strategy_manifest(
        self,
        bot_id: str,
        strategy_id: str,
        run_month: str,
        manifests: list[MarketDataManifest],
        summary: dict,
    ) -> None:
        if not manifests:
            return
        start_ts = min(manifest.start_ts for manifest in manifests)
        end_ts = max(manifest.end_ts for manifest in manifests)
        expected_bars = sum(manifest.expected_bars for manifest in manifests)
        actual_bars = sum(manifest.actual_bars for manifest in manifests)
        blocking_reasons = [
            f"{manifest.symbol}: {reason}"
            for manifest in manifests
            for reason in (manifest.blocking_reasons or [])
        ]
        composite = MarketDataManifest(
            source="composite",
            market=",".join(sorted({manifest.market for manifest in manifests if manifest.market})),
            symbol=",".join(sorted({manifest.symbol for manifest in manifests if manifest.symbol})),
            timeframe=",".join(sorted({manifest.timeframe for manifest in manifests if manifest.timeframe})),
            start_ts=start_ts,
            end_ts=end_ts,
            expected_bars=expected_bars,
            actual_bars=actual_bars,
            coverage_ratio=(actual_bars / expected_bars if expected_bars else 0.0),
            usable_for_authoritative_validation=not blocking_reasons and all(
                manifest.usable_for_authoritative_validation for manifest in manifests
            ),
            blocking_reasons=blocking_reasons,
        )
        output_path = MarketDataCatalog.strategy_manifest_path(
            self.market_data_root,
            bot_id,
            strategy_id,
            run_month,
        )
        self.writer.write(composite, output_path)
        summary["manifests_written"] += 1

    def _adapter_for(self, source: str) -> MarketDataAdapter:
        if source == "bot_uploaded":
            return BotUploadedParquetAdapter(self.market_data_root, writer=self.writer)
        if source == "kis":
            return KisMarketDataAdapter()
        return FileSystemParquetAdapter(self.market_data_root, writer=self.writer)

    def _emit(self, event_type: str, data: dict) -> None:
        if self.event_stream is not None:
            self.event_stream.broadcast(event_type, data)


def _month_request_window(run_month: str, timeframe: str) -> tuple[datetime, datetime, int]:
    start_date, _ = month_window(run_month)
    start_ts = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_exclusive = _next_month_start(start_ts)
    expected_bars = max(
        1,
        int((end_exclusive - start_ts).total_seconds() // (_timeframe_minutes(timeframe) * 60)),
    )
    return start_ts, end_exclusive, expected_bars


def _next_month_start(value: datetime) -> datetime:
    if value.month == 12:
        return datetime(value.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(value.year, value.month + 1, 1, tzinfo=timezone.utc)


def _timeframe_minutes(timeframe: str) -> int:
    raw = timeframe.strip().lower()
    if raw.endswith("m") and raw[:-1].isdigit():
        return max(1, int(raw[:-1]))
    if raw.endswith("h") and raw[:-1].isdigit():
        return max(1, int(raw[:-1]) * 60)
    if raw.endswith("d") and raw[:-1].isdigit():
        return max(1, int(raw[:-1]) * 24 * 60)
    logger.warning("Unknown timeframe %r; assuming 1m bars", timeframe)
    return 1
