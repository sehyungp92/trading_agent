"""Runtime and deployment contract envelopes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DeploymentMetadata(BaseModel):
    """Deployment identity required before paper/live runtime startup."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = "deployment_metadata_v1"
    bot_id: str
    strategy_id: str = ""
    environment: str = "development"
    runtime_mode: str = "dry_run"
    image_version: str = ""
    image_digest: str = ""
    git_commit: str
    git_branch: str = ""
    config_hash: str
    promotion_hash: str = ""
    contract_hash: str = ""
    strategy_version: str = ""
    telemetry_schema_version: str = ""
    runtime_entrypoint: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_identity(self) -> "DeploymentMetadata":
        missing = [
            field_name
            for field_name in (
                "bot_id",
                "git_commit",
                "config_hash",
                "runtime_entrypoint",
            )
            if not str(getattr(self, field_name, "") or "").strip()
        ]
        if missing:
            raise ValueError("deployment metadata missing required fields: " + ", ".join(missing))
        if self.runtime_mode in {"paper", "live"}:
            required_live_fields = [
                "image_version",
                "promotion_hash",
                "contract_hash",
                "telemetry_schema_version",
            ]
            missing_live = [
                field_name
                for field_name in required_live_fields
                if not str(getattr(self, field_name, "") or "").strip()
            ]
            if missing_live:
                raise ValueError(
                    "paper/live deployment metadata missing required fields: "
                    + ", ".join(missing_live)
                )
        return self


class ReadinessCheck(BaseModel):
    name: str
    passed: bool
    severity: str = "error"
    message: str = ""
    evidence_path: str = ""

    @model_validator(mode="after")
    def _validate_name(self) -> "ReadinessCheck":
        if not self.name.strip():
            raise ValueError("readiness check name is required")
        return self


class RuntimeReadinessReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = "runtime_readiness_report_v1"
    bot_id: str
    runtime_mode: str = "dry_run"
    ready: bool
    checks: list[ReadinessCheck] = Field(default_factory=list)
    deployment_metadata: DeploymentMetadata | None = None
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_report(self) -> "RuntimeReadinessReport":
        failed_checks = [check for check in self.checks if not check.passed and check.severity == "error"]
        if self.ready and (failed_checks or self.errors):
            raise ValueError("runtime readiness cannot be ready with error checks or errors")
        return self


class TelemetryEventEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = "telemetry_event_envelope_v1"
    event_id: str
    bot_id: str
    event_type: str
    source: str = ""
    exchange_timestamp: datetime | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    priority: int = 5
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_envelope(self) -> "TelemetryEventEnvelope":
        missing = [
            field_name
            for field_name in ("event_id", "bot_id", "event_type")
            if not str(getattr(self, field_name, "") or "").strip()
        ]
        if missing:
            raise ValueError("telemetry event envelope missing required fields: " + ", ".join(missing))
        if self.priority < 0:
            raise ValueError("telemetry event priority cannot be negative")
        return self
