"""Strategy plugin and replay-adapter contract schema."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class StrategyPluginMaturity(str, Enum):
    DIAGNOSTIC = "diagnostic"
    SHADOW_VALIDATED = "shadow_validated"
    APPROVAL_READY = "approval_ready"


class StrategyPluginContract(BaseModel):
    """Machine-readable live strategy to backtest adapter contract."""

    plugin_id: str
    live_repo_path: str = ""
    live_repo_commit_sha: str = ""
    backtest_adapter_path: str
    backtest_adapter_commit_sha: str = ""
    config_schema_version: str
    decision_api_version: str
    required_telemetry_schemas: list[str] = Field(default_factory=list)
    supported_symbols: list[str] = Field(default_factory=list)
    supported_timeframes: list[str] = Field(default_factory=list)
    parity_fixture_set: list[str] = Field(default_factory=list)
    maturity: StrategyPluginMaturity = StrategyPluginMaturity.DIAGNOSTIC
    contract_id: str = ""
    contract_version: str = "strategy_plugin_contract_v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _normalize(self) -> "StrategyPluginContract":
        self.supported_symbols = sorted({
            symbol.strip().upper() for symbol in self.supported_symbols if symbol.strip()
        })
        self.supported_timeframes = sorted({
            timeframe.strip() for timeframe in self.supported_timeframes if timeframe.strip()
        })
        self.required_telemetry_schemas = sorted({
            schema.strip() for schema in self.required_telemetry_schemas if schema.strip()
        })
        self.parity_fixture_set = [
            path.strip() for path in self.parity_fixture_set if path.strip()
        ]
        missing: list[str] = []
        if not self.plugin_id:
            missing.append("plugin_id")
        if not self.backtest_adapter_path:
            missing.append("backtest_adapter_path")
        if not self.config_schema_version:
            missing.append("config_schema_version")
        if not self.decision_api_version:
            missing.append("decision_api_version")
        if missing:
            raise ValueError("strategy plugin contract missing required fields: " + ", ".join(missing))
        if self.maturity in {
            StrategyPluginMaturity.SHADOW_VALIDATED,
            StrategyPluginMaturity.APPROVAL_READY,
        }:
            mature_missing = self.maturity_contract_errors()
            if mature_missing:
                raise ValueError(
                    "mature strategy plugin contract missing required fields: "
                    + ", ".join(sorted(mature_missing))
                )
        if not self.contract_id:
            raw = "|".join([
                self.plugin_id,
                self.live_repo_commit_sha,
                self.backtest_adapter_path,
                self.backtest_adapter_commit_sha,
                self.config_schema_version,
                self.decision_api_version,
                self.maturity.value,
            ])
            self.contract_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return self

    def maturity_contract_errors(self) -> list[str]:
        missing: list[str] = []
        for attr in (
            "live_repo_path",
            "live_repo_commit_sha",
            "backtest_adapter_path",
            "backtest_adapter_commit_sha",
            "config_schema_version",
            "decision_api_version",
        ):
            if not str(getattr(self, attr, "") or "").strip():
                missing.append(attr)
        for attr in (
            "required_telemetry_schemas",
            "supported_symbols",
            "supported_timeframes",
            "parity_fixture_set",
        ):
            if not getattr(self, attr):
                missing.append(attr)
        return missing

    @property
    def eligible_for_optimizer(self) -> bool:
        return self.maturity in {
            StrategyPluginMaturity.SHADOW_VALIDATED,
            StrategyPluginMaturity.APPROVAL_READY,
        } and not self.maturity_contract_errors()

    @property
    def eligible_for_approval(self) -> bool:
        return self.maturity == StrategyPluginMaturity.APPROVAL_READY and not self.maturity_contract_errors()
