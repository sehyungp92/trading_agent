"""Coverage manifest writer for canonical market data."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable

from trading_assistant.schemas.market_data_manifest import MarketDataManifest, MissingRange


class CoverageManifestWriter:
    """Create and persist market-data coverage manifests."""

    def __init__(self, required_coverage_ratio: float = 0.95) -> None:
        self.required_coverage_ratio = required_coverage_ratio

    def build_manifest(
        self,
        *,
        data_path: Path,
        source: str,
        market: str,
        symbol: str,
        timeframe: str,
        start_ts: datetime,
        end_ts: datetime,
        expected_bars: int,
        actual_bars: int | None = None,
        missing_ranges: Iterable[MissingRange | dict] | None = None,
        timezone_name: str = "UTC",
        source_version: str = "",
        adjustment_policy: str = "",
        fee_model_version: str = "",
        slippage_model_version: str = "",
        session_calendar: str = "",
    ) -> MarketDataManifest:
        checksum = file_checksum(data_path) if Path(data_path).exists() else ""
        actual = actual_bars if actual_bars is not None else infer_bar_count(data_path)
        coverage_ratio = actual / expected_bars if expected_bars else 0.0
        ranges = [
            item if isinstance(item, MissingRange) else MissingRange.model_validate(item)
            for item in (missing_ranges or [])
        ]
        blocking_reasons: list[str] = []
        if not Path(data_path).exists():
            blocking_reasons.append(f"data file missing: {data_path}")
        if expected_bars <= 0:
            blocking_reasons.append("expected_bars must be positive")
        if coverage_ratio < self.required_coverage_ratio:
            blocking_reasons.append(
                f"coverage {coverage_ratio:.3f} below required {self.required_coverage_ratio:.3f}"
            )
        if ranges:
            blocking_reasons.append("missing bar ranges present")
        for label, value in (
            ("session_calendar", session_calendar),
            ("fee_model_version", fee_model_version),
            ("slippage_model_version", slippage_model_version),
            ("adjustment_policy", adjustment_policy),
        ):
            if not str(value or "").strip():
                blocking_reasons.append(f"{label} is required for authoritative validation")

        return MarketDataManifest(
            source=source,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            expected_bars=expected_bars,
            actual_bars=actual,
            coverage_ratio=coverage_ratio,
            missing_ranges=ranges,
            session_calendar=session_calendar,
            timezone=timezone_name,
            checksum=checksum,
            source_version=source_version,
            adjustment_policy=adjustment_policy,
            fee_model_version=fee_model_version,
            slippage_model_version=slippage_model_version,
            usable_for_authoritative_validation=not blocking_reasons,
            blocking_reasons=blocking_reasons,
        )

    def write(self, manifest: MarketDataManifest, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return output_path


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_bar_count(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if path.suffix.lower() == ".csv":
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return max(0, len(lines) - 1)
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd  # type: ignore

            return int(len(pd.read_parquet(path)))
        except Exception:
            return 0
    return 0
