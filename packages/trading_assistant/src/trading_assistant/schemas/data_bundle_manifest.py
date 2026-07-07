"""Repo-level market-data bundle manifest schema."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


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
            self.calendars = sorted({
                item.calendar for item in self.slice_manifests if item.calendar
            })
        if not self.bundle_checksum:
            raw = "|".join([
                self.data_repo_commit_sha,
                self.fee_model_version,
                self.slippage_model_version,
                self.adjustment_policy,
                *[
                    "|".join([
                        item.manifest_id,
                        item.symbol,
                        item.timeframe,
                        item.checksum,
                    ])
                    for item in self.slice_manifests
                ],
            ])
            self.bundle_checksum = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if not self.bundle_id:
            raw = "|".join([
                self.data_repo_commit_sha,
                self.bundle_checksum,
                ",".join(f"{item.symbol}:{item.timeframe}" for item in self.slice_manifests),
            ])
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
            "data_repo_commit_sha",
            "bundle_checksum",
            "fee_model_version",
            "slippage_model_version",
            "adjustment_policy",
        ):
            if not str(getattr(self, attr, "") or "").strip():
                missing.append(attr)
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
        return self.status == DataBundleStatus.AUTHORITATIVE and not self.authoritative_contract_errors()
