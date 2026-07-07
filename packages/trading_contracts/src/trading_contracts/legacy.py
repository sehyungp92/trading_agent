"""Compatibility readers for existing pre-shared-contract artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_contracts.canonical import load_json
from trading_contracts.models import RoundsManifest, StrategyPluginContract
from trading_contracts.runtime import DeploymentMetadata


class LegacyRoundsManifest(BaseModel):
    """Tolerant reader for IBKR, crypto, and K-stock optimizer round manifests."""

    model_config = ConfigDict(extra="allow")

    family: str = ""
    schema_version: int | str | None = None
    generated_at_utc: str = ""
    baseline_reset: dict[str, Any] | None = None
    rounds: list[dict[str, Any]]

    @model_validator(mode="after")
    def _validate_rounds(self) -> "LegacyRoundsManifest":
        if not self.rounds:
            raise ValueError("legacy rounds manifest requires at least one round")
        missing_round_ids = [
            index
            for index, row in enumerate(self.rounds)
            if "round" not in row and "round_id" not in row
        ]
        if missing_round_ids:
            raise ValueError(
                "legacy rounds missing round identifiers at indexes: "
                + ", ".join(str(index) for index in missing_round_ids)
            )
        return self


class CryptoOptimizerContractV1(BaseModel):
    """Compatibility wrapper for crypto optimizer contract v1 artifacts."""

    model_config = ConfigDict(extra="allow")

    schema_version: int | str = "optimizer_contract_v1"
    strategy_id: str = ""
    contract_version: str = ""


class CryptoDeploymentManifestV1(BaseModel):
    """Compatibility wrapper for crypto deployment manifest schema version 1."""

    model_config = ConfigDict(extra="allow")

    schema_version: int | str = 1
    deployment_id: str = ""
    generated_at: str = ""


class KStockOlrKalcbStrategyPluginContractV1(StrategyPluginContract):
    """Named compatibility adapter for the existing K-stock OLR/KALCB bridge contract."""

    @model_validator(mode="after")
    def _validate_k_stock_contract(self) -> "KStockOlrKalcbStrategyPluginContractV1":
        if self.contract_version not in {
            "strategy_plugin_contract_v1",
            "k_stock_olr_kalcb_strategy_plugin_contract_v1",
        }:
            raise ValueError("unexpected K-stock OLR/KALCB strategy plugin contract version")
        return self


class StrategyPromotionManifest(BaseModel):
    """Draft or approved strategy promotion evidence."""

    model_config = ConfigDict(extra="allow")

    schema_version: str
    bot_id: str
    strategy_id: str
    venue: str = ""
    promotion_state: str = "draft"
    baseline_id: str | None = ""
    baseline_status: str | None = ""
    optimizer_round: dict[str, Any] = Field(default_factory=dict)
    source_live_config: dict[str, Any] = Field(default_factory=dict)
    approval: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_promotion(self) -> "StrategyPromotionManifest":
        if not self.schema_version.startswith("strategy_promotion_manifest."):
            raise ValueError("promotion manifest schema_version must start with strategy_promotion_manifest.")
        missing = [
            field_name
            for field_name in ("bot_id", "strategy_id", "promotion_state")
            if not str(getattr(self, field_name, "") or "").strip()
        ]
        if missing:
            raise ValueError("promotion manifest missing required fields: " + ", ".join(missing))
        if self.baseline_id is None:
            self.baseline_id = ""
        if self.baseline_status is None:
            self.baseline_status = ""
        if self.promotion_state not in {
            "draft",
            "approved",
            "disabled",
            "not_promoted",
            "draft_portfolio_bundle_supersession",
        }:
            raise ValueError("promotion manifest has unknown promotion_state")
        legacy_paths = list(_iter_legacy_reference_paths(self.model_dump(mode="json")))
        if legacy_paths:
            raise ValueError(
                "promotion manifest contains legacy reference paths: "
                + ", ".join(legacy_paths[:5])
            )
        return self


def _iter_legacy_reference_paths(value: Any, location: str = "$") -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if isinstance(child, str) and _is_path_field(key) and _is_legacy_reference_path(child):
                yield f"{child_location}={child}"
                continue
            yield from _iter_legacy_reference_paths(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_legacy_reference_paths(child, f"{location}[{index}]")


def _is_path_field(key: str) -> bool:
    return key == "path" or key.endswith("_path") or key.endswith("_root")


def _is_legacy_reference_path(value: str) -> bool:
    normalized = value.replace("\\", "/").strip()
    token = "_ref" "erences"
    return normalized.startswith(f"{token}/") or normalized.startswith(f"../../../{token}/")


def load_rounds_manifest(path: str | Path) -> RoundsManifest | LegacyRoundsManifest:
    payload = load_json(path)
    if isinstance(payload, dict) and "current_round_id" in payload:
        return RoundsManifest.model_validate(payload)
    return LegacyRoundsManifest.model_validate(payload)


def validate_rounds_manifest(path: str | Path) -> RoundsManifest | LegacyRoundsManifest:
    return load_rounds_manifest(path)


def validate_plugin_contract(path: str | Path) -> StrategyPluginContract:
    payload = load_json(path)
    if isinstance(payload, dict) and payload.get("plugin_id") == "k-stock-olr-kalcb":
        return KStockOlrKalcbStrategyPluginContractV1.model_validate(payload)
    return StrategyPluginContract.model_validate(payload)


def validate_promotion_manifest(path: str | Path) -> StrategyPromotionManifest:
    return StrategyPromotionManifest.model_validate(load_json(path))


def validate_deployment_manifest(path: str | Path) -> DeploymentMetadata | CryptoDeploymentManifestV1:
    payload = load_json(path)
    if isinstance(payload, dict) and str(payload.get("schema_version", "")) in {"1", "1.0"}:
        return CryptoDeploymentManifestV1.model_validate(payload)
    return DeploymentMetadata.model_validate(payload)
