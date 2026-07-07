"""Manifest models compatible with trading_assistant schemas."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .repo import is_git_commit_sha


class MarketDataUsability(str, Enum):
    AUTHORITATIVE = "authoritative"
    DIAGNOSTICS_ONLY = "diagnostics_only"
    BLOCKED = "blocked"


class MissingRange(BaseModel):
    start_ts: datetime
    end_ts: datetime
    reason: str = ""


class MarketDataManifest(BaseModel):
    """Coverage and provenance for one market-data slice."""

    manifest_id: str = ""
    source: str
    market: str = ""
    symbol: str
    timeframe: str
    start_ts: datetime
    end_ts: datetime
    expected_bars: int = 0
    actual_bars: int = 0
    coverage_ratio: float = 0.0
    missing_ranges: list[MissingRange] = Field(default_factory=list)
    session_calendar: str = ""
    timezone: str = "UTC"
    checksum: str = ""
    schema_version: str = "market_data_manifest_v1"
    source_version: str = ""
    adjustment_policy: str = ""
    fee_model_version: str = ""
    slippage_model_version: str = ""
    lineage: dict[str, str] = Field(default_factory=dict)
    usable_for_authoritative_validation: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _normalize(self) -> "MarketDataManifest":
        if self.end_ts < self.start_ts:
            raise ValueError("end_ts must be >= start_ts")
        if self.expected_bars < 0 or self.actual_bars < 0:
            raise ValueError("bar counts must be non-negative")
        if self.expected_bars and not self.coverage_ratio:
            self.coverage_ratio = self.actual_bars / self.expected_bars
        self.coverage_ratio = max(0.0, min(float(self.coverage_ratio), 1.0))
        if not self.manifest_id:
            raw = "|".join(
                [
                    self.source,
                    self.market,
                    self.symbol,
                    self.timeframe,
                    self.start_ts.isoformat(),
                    self.end_ts.isoformat(),
                    self.checksum,
                ]
            )
            self.manifest_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        if self.blocking_reasons:
            self.usable_for_authoritative_validation = False
        return self

    @property
    def usability(self) -> MarketDataUsability:
        if self.usable_for_authoritative_validation:
            return MarketDataUsability.AUTHORITATIVE
        if self.actual_bars > 0:
            return MarketDataUsability.DIAGNOSTICS_ONLY
        return MarketDataUsability.BLOCKED


class DataBundleStatus(str, Enum):
    AUTHORITATIVE = "authoritative"
    DIAGNOSTICS_ONLY = "diagnostics_only"
    BLOCKED = "blocked"


class DataBundleSlice(BaseModel):
    """One market-data slice included in a monthly runner bundle."""

    manifest_path: str
    manifest_id: str = ""
    source: str = ""
    market: str = ""
    symbol: str
    timeframe: str
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    checksum: str = ""
    calendar: str = ""
    authoritative: bool = False


class DataBundleManifest(BaseModel):
    """Frozen repo-level data contract consumed by external monthly runners."""

    bundle_id: str = ""
    data_repo_path: str = ""
    data_repo_commit_sha: str = ""
    data_repo_branch: str = ""
    slice_manifests: list[DataBundleSlice]
    bundle_checksum: str = ""
    calendars: list[str] = Field(default_factory=list)
    fee_model_version: str = ""
    slippage_model_version: str = ""
    adjustment_policy: str = ""
    status: DataBundleStatus = DataBundleStatus.DIAGNOSTICS_ONLY
    diagnostics_only_reason: str = ""
    schema_version: str = "data_bundle_manifest_v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _normalize(self) -> "DataBundleManifest":
        if not self.slice_manifests:
            raise ValueError("data bundle requires at least one slice manifest")
        if not self.calendars:
            self.calendars = sorted({item.calendar for item in self.slice_manifests if item.calendar})
        if not self.bundle_checksum:
            raw = "|".join(
                [
                    self.data_repo_commit_sha,
                    self.fee_model_version,
                    self.slippage_model_version,
                    self.adjustment_policy,
                    *[
                        "|".join(
                            [
                                item.manifest_id,
                                item.symbol,
                                item.timeframe,
                                item.checksum,
                            ]
                        )
                        for item in self.slice_manifests
                    ],
                ]
            )
            self.bundle_checksum = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if not self.bundle_id:
            raw = "|".join(
                [
                    self.data_repo_commit_sha,
                    self.bundle_checksum,
                    ",".join(f"{item.symbol}:{item.timeframe}" for item in self.slice_manifests),
                ]
            )
            self.bundle_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        if self.status == DataBundleStatus.AUTHORITATIVE:
            missing = self.authoritative_contract_errors()
            if missing:
                raise ValueError(
                    "authoritative data bundle missing required fields: "
                    + ", ".join(sorted(missing))
                )
        if self.status != DataBundleStatus.AUTHORITATIVE and not self.diagnostics_only_reason:
            self.diagnostics_only_reason = self.status.value
        return self

    def authoritative_contract_errors(self) -> list[str]:
        missing: list[str] = []
        for attr in (
            "bundle_checksum",
            "fee_model_version",
            "slippage_model_version",
            "adjustment_policy",
        ):
            if not str(getattr(self, attr, "") or "").strip():
                missing.append(attr)
        if not self.data_repo_commit_sha or not is_git_commit_sha(self.data_repo_commit_sha):
            missing.append("data_repo_commit_sha")
        if not self.calendars:
            missing.append("calendars")
        for index, item in enumerate(self.slice_manifests):
            if not item.checksum:
                missing.append(f"slice_manifests[{index}].checksum")
            if not item.calendar:
                missing.append(f"slice_manifests[{index}].calendar")
            if not item.authoritative:
                missing.append(f"slice_manifests[{index}].authoritative")
        return missing

    @property
    def usable_for_authoritative_validation(self) -> bool:
        return (
            self.status == DataBundleStatus.AUTHORITATIVE
            and not self.authoritative_contract_errors()
        )


def load_market_manifest(path: Path) -> MarketDataManifest:
    return MarketDataManifest.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def load_bundle_manifest(path: Path) -> DataBundleManifest:
    return DataBundleManifest.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def write_model(path: Path, model: BaseModel) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = model.model_dump_json(indent=2)
    last_error: OSError | None = None
    for attempt in range(5):
        tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
        try:
            tmp.write_text(rendered, encoding="utf-8")
            tmp.replace(path)
            return path
        except OSError as exc:
            last_error = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            time.sleep(0.1 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return path
