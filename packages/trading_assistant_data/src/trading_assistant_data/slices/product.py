"""Canonical data slice product contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading_assistant_data.manifests import MarketDataManifest
from trading_assistant_data.slices.authority import (
    SliceAuthorityStatus,
    authority_status_for_manifest,
)
from trading_assistant_data.slices.coverage import SliceCoverageReport, coverage_report
from trading_assistant_data.slices.writer import slice_manifest_path, write_slice_manifest


@dataclass(frozen=True)
class SliceRequest:
    market: str
    source: str
    symbol: str
    timeframe: str
    snapshot: str = ""


@dataclass(frozen=True)
class CanonicalSlice:
    request: SliceRequest
    canonical_paths: list[Path]
    manifest: MarketDataManifest
    timestamps: Iterable | None = None
    exchange_timestamps: Iterable | None = None


@dataclass(frozen=True)
class SliceWrite:
    manifest_path: Path
    canonical_paths: list[Path]
    manifest: MarketDataManifest


class DataSliceProduct:
    """Facade for canonical slice writes, coverage, and authority views."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = Path(repo_root)

    def write_slice(self, canonical: CanonicalSlice) -> SliceWrite:
        manifest_path = write_slice_manifest(self._repo_root, canonical.manifest)
        return SliceWrite(
            manifest_path=manifest_path,
            canonical_paths=canonical.canonical_paths,
            manifest=canonical.manifest,
        )

    def manifest_path(self, manifest: MarketDataManifest) -> Path:
        return slice_manifest_path(self._repo_root, manifest)

    def coverage_report(self, *, calendar, canonical: CanonicalSlice) -> SliceCoverageReport:
        if canonical.timestamps is None:
            raise ValueError("CanonicalSlice.timestamps is required for coverage_report")
        return coverage_report(
            timestamps=canonical.timestamps,
            timeframe=canonical.manifest.timeframe,
            calendar=calendar,
            exchange_timestamps=canonical.exchange_timestamps,
        )

    def coverage_for_request(
        self,
        *,
        request: SliceRequest,
        timestamps: Iterable,
        calendar,
        exchange_timestamps: Iterable | None = None,
    ) -> SliceCoverageReport:
        timestamp_list = list(timestamps)
        if not timestamp_list:
            raise ValueError("timestamps is required for coverage_for_request")
        return self.coverage_report(
            calendar=calendar,
            canonical=CanonicalSlice(
                request=request,
                canonical_paths=[],
                manifest=MarketDataManifest(
                    source=request.source,
                    market=request.market,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    start_ts=min(timestamp_list),
                    end_ts=max(timestamp_list),
                ),
                timestamps=timestamp_list,
                exchange_timestamps=exchange_timestamps,
            ),
        )

    def authority_status(self, manifest: MarketDataManifest) -> SliceAuthorityStatus:
        return authority_status_for_manifest(manifest)


def timestamps_sorted_unique_utc(timestamps: Iterable) -> bool:
    values = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True))
    return bool(values.is_monotonic_increasing and values.is_unique)
