"""JSON Schema generation for shared contract models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel
from trading_contracts.models import (
    BacktestArtifactIndex,
    ConfirmatoryRerank,
    DataBundleManifest,
    DecisionParityReport,
    MonthlyRunManifest,
    RoundsManifest,
    StrategyPluginContract,
)

from trading_contracts.legacy import LegacyRoundsManifest, StrategyPromotionManifest
from trading_contracts.runtime import DeploymentMetadata, RuntimeReadinessReport, TelemetryEventEnvelope


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "backtest_artifact_index.schema.json": BacktestArtifactIndex,
    "confirmatory_rerank.schema.json": ConfirmatoryRerank,
    "data_bundle_manifest.schema.json": DataBundleManifest,
    "decision_parity_report.schema.json": DecisionParityReport,
    "deployment_metadata.schema.json": DeploymentMetadata,
    "legacy_rounds_manifest.schema.json": LegacyRoundsManifest,
    "monthly_run_manifest.schema.json": MonthlyRunManifest,
    "rounds_manifest.schema.json": RoundsManifest,
    "runtime_readiness_report.schema.json": RuntimeReadinessReport,
    "strategy_plugin_contract.schema.json": StrategyPluginContract,
    "strategy_promotion_manifest.schema.json": StrategyPromotionManifest,
    "telemetry_event_envelope.schema.json": TelemetryEventEnvelope,
}


def generate_schemas(output_dir: str | Path, names: Iterable[str] | None = None) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = set(names or SCHEMA_MODELS)
    written: list[Path] = []
    for filename, model in sorted(SCHEMA_MODELS.items()):
        if filename not in selected:
            continue
        path = output / filename
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written
