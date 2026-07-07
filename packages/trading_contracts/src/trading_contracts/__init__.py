"""Shared contract surface for the trading workspace."""

from trading_contracts.models import (
    BacktestArtifactIndex,
    ConfirmatoryRerank,
    DataBundleManifest,
    DecisionParityCheck,
    DecisionParityReport,
    DecisionParityStatus,
    MonthlyRunManifest,
    RoundManifestRecord,
    RoundsManifest,
    StrategyPluginContract,
    StrategyPluginMaturity,
)

from trading_contracts.canonical import canonical_json_sha256, canonical_json_text, file_sha256
from trading_contracts.legacy import (
    CryptoDeploymentManifestV1,
    CryptoOptimizerContractV1,
    KStockOlrKalcbStrategyPluginContractV1,
    LegacyRoundsManifest,
    StrategyPromotionManifest,
    load_rounds_manifest,
    validate_deployment_manifest,
    validate_plugin_contract,
    validate_promotion_manifest,
    validate_rounds_manifest,
)
from trading_contracts.runtime import (
    DeploymentMetadata,
    ReadinessCheck,
    RuntimeReadinessReport,
    TelemetryEventEnvelope,
)

__all__ = [
    "BacktestArtifactIndex",
    "ConfirmatoryRerank",
    "CryptoDeploymentManifestV1",
    "CryptoOptimizerContractV1",
    "DataBundleManifest",
    "DecisionParityCheck",
    "DecisionParityReport",
    "DecisionParityStatus",
    "DeploymentMetadata",
    "KStockOlrKalcbStrategyPluginContractV1",
    "LegacyRoundsManifest",
    "MonthlyRunManifest",
    "ReadinessCheck",
    "RoundManifestRecord",
    "RoundsManifest",
    "RuntimeReadinessReport",
    "StrategyPluginContract",
    "StrategyPluginMaturity",
    "StrategyPromotionManifest",
    "TelemetryEventEnvelope",
    "canonical_json_sha256",
    "canonical_json_text",
    "file_sha256",
    "load_rounds_manifest",
    "validate_deployment_manifest",
    "validate_plugin_contract",
    "validate_promotion_manifest",
    "validate_rounds_manifest",
]
