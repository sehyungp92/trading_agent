"""Canonical market-data sync adapter interfaces and local adapters."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from trading_assistant.schemas.market_data_manifest import MarketDataManifest
from trading_assistant.skills.coverage_manifest_writer import CoverageManifestWriter


@dataclass(frozen=True)
class MarketDataSyncRequest:
    market: str
    symbol: str
    timeframe: str
    source: str
    start_ts: datetime
    end_ts: datetime
    expected_bars: int
    destination_path: Path


@dataclass
class MarketDataSyncResult:
    success: bool
    data_path: Path
    manifest: MarketDataManifest | None = None
    warnings: list[str] = field(default_factory=list)
    error: str = ""


class MarketDataAdapter(Protocol):
    source_id: str

    def sync(self, request: MarketDataSyncRequest) -> MarketDataSyncResult: ...

    def validate_coverage(self, request: MarketDataSyncRequest) -> MarketDataManifest: ...


class FileSystemParquetAdapter:
    """Adapter for already-downloaded parquet under a filesystem root."""

    source_id = "filesystem"

    def __init__(self, root: Path, writer: CoverageManifestWriter | None = None) -> None:
        self.root = Path(root)
        self.writer = writer or CoverageManifestWriter()

    def sync(self, request: MarketDataSyncRequest) -> MarketDataSyncResult:
        source_path = self.root / request.source / request.market / request.symbol / request.timeframe / f"{request.start_ts:%Y-%m}.parquet"
        if not source_path.exists():
            return MarketDataSyncResult(False, request.destination_path, error=f"source missing: {source_path}")
        request.destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != request.destination_path.resolve():
            shutil.copy2(source_path, request.destination_path)
        manifest = self.validate_coverage(request)
        return MarketDataSyncResult(True, request.destination_path, manifest=manifest)

    def validate_coverage(self, request: MarketDataSyncRequest) -> MarketDataManifest:
        return self.writer.build_manifest(
            data_path=request.destination_path,
            source=request.source,
            market=request.market,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_ts=request.start_ts,
            end_ts=request.end_ts,
            expected_bars=request.expected_bars,
        )


class BotUploadedParquetAdapter(FileSystemParquetAdapter):
    source_id = "bot_uploaded"


class KisMarketDataAdapter:
    """KIS placeholder: real endpoint capability is environment/account-specific."""

    source_id = "kis"

    def sync(self, request: MarketDataSyncRequest) -> MarketDataSyncResult:
        return MarketDataSyncResult(
            success=False,
            data_path=request.destination_path,
            error="KIS historical sync is not configured in this environment",
            warnings=["strategy remains diagnostics-only until KIS historical coverage is verified"],
        )

    def validate_coverage(self, request: MarketDataSyncRequest) -> MarketDataManifest:
        writer = CoverageManifestWriter()
        return writer.build_manifest(
            data_path=request.destination_path,
            source=request.source,
            market=request.market,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_ts=request.start_ts,
            end_ts=request.end_ts,
            expected_bars=request.expected_bars,
        )
