"""Experiment A/B tracking infrastructure (#20).

Provides structured experiment metadata definitions and a registry
that loads from YAML config. The experiment_id/experiment_variant fields
already exist on TradeEvent and propagate through the system; this adds
the registry and definition layer.

Supports both the legacy dict YAML format and the orchestrator-generated
list format (Phase 2C).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("instrumentation.experiment")

_DEFAULT_CONFIG_PATH = Path("instrumentation/config/experiments.yaml")


@dataclass
class ExperimentMetadata:
    """Definition of a single A/B experiment."""
    experiment_id: str
    hypothesis: str = ""
    variants: list[str] = field(default_factory=list)
    start_date: str = ""
    strategy_type: str = ""
    primary_metric: str = "sharpe"
    secondary_metrics: list[str] = field(default_factory=lambda: ["win_rate", "avg_pnl"])
    end_date: Optional[str] = None
    min_trades_per_variant: int = 30
    # Phase 2C: orchestrator-compatible fields
    variant_params: dict[str, dict] = field(default_factory=dict)
    variant_allocations: dict[str, float] = field(default_factory=dict)
    status: str = "active"
    title: str = ""
    success_metric: str = "pnl"
    max_duration_days: int = 30
    allocation_method: str = "hash"


class ExperimentRegistry:
    """Loads experiment definitions from YAML and provides lookup.

    Supports two YAML formats:
    1. Legacy (dict): ``{experiments: {exp_id: {hypothesis, variants, ...}}}``
    2. Orchestrator (list): ``{experiments: [{experiment_id, variants: {name: {params, allocation_pct}}, ...}]}``

    Usage:
        registry = ExperimentRegistry()
        exp = registry.get("exp_001")
        active = registry.active_experiments()
        variant = registry.assign_variant("exp_001", trade_id)
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._path = config_path or _DEFAULT_CONFIG_PATH
        self._experiments: dict[str, ExperimentMetadata] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            logger.debug("No experiments config at %s", self._path)
            return
        try:
            with open(self._path) as f:
                data = yaml.safe_load(f) or {}

            raw = data.get("experiments", data)  # fallback to root dict for legacy

            if isinstance(raw, list):
                # Orchestrator format: list of experiment dicts
                self._load_orchestrator_format(raw)
            elif isinstance(raw, dict):
                # Legacy format: {exp_id: {hypothesis, variants, ...}}
                self._load_legacy_format(raw)

            if self._experiments:
                logger.info("Loaded %d experiment definitions", len(self._experiments))
        except Exception as e:
            logger.warning("Failed to load experiments config: %s", e)

    def _load_orchestrator_format(self, raw: list) -> None:
        """Parse orchestrator-generated format (list of experiment dicts)."""
        for exp_data in raw:
            exp_id = exp_data.get("experiment_id", "")
            if not exp_id:
                continue

            variants_raw = exp_data.get("variants", {})
            variant_names = []
            variant_params = {}
            variant_allocations = {}

            if isinstance(variants_raw, dict):
                for vname, vdata in variants_raw.items():
                    variant_names.append(vname)
                    if isinstance(vdata, dict):
                        variant_params[vname] = vdata.get("params", {})
                        variant_allocations[vname] = vdata.get("allocation_pct", 50.0)
            elif isinstance(variants_raw, list):
                variant_names = variants_raw

            self._experiments[exp_id] = ExperimentMetadata(
                experiment_id=exp_id,
                hypothesis=exp_data.get("hypothesis", ""),
                variants=variant_names,
                variant_params=variant_params,
                variant_allocations=variant_allocations,
                primary_metric=exp_data.get("primary_metric", "sharpe"),
                start_date=exp_data.get("start_date", ""),
                end_date=exp_data.get("end_date"),
                min_trades_per_variant=exp_data.get("min_trades_per_variant", 30),
                strategy_type=exp_data.get("strategy_type", ""),
                status=exp_data.get("status", "active"),
                title=exp_data.get("title", ""),
                success_metric=exp_data.get("success_metric", "pnl"),
                max_duration_days=exp_data.get("max_duration_days", 30),
                allocation_method=exp_data.get("allocation_method", "hash"),
            )

    def _load_legacy_format(self, raw: dict) -> None:
        """Parse legacy dict format: {exp_id: {hypothesis, variants, ...}}."""
        for exp_id, exp_data in raw.items():
            if not isinstance(exp_data, dict):
                continue
            self._experiments[exp_id] = ExperimentMetadata(
                experiment_id=exp_id,
                hypothesis=exp_data.get("hypothesis", ""),
                variants=exp_data.get("variants", []),
                start_date=exp_data.get("start_date", ""),
                strategy_type=exp_data.get("strategy_type", ""),
                primary_metric=exp_data.get("primary_metric", "sharpe"),
                secondary_metrics=exp_data.get("secondary_metrics", ["win_rate", "avg_pnl"]),
                end_date=exp_data.get("end_date"),
                min_trades_per_variant=exp_data.get("min_trades_per_variant", 30),
            )

    def get(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        return self._experiments.get(experiment_id)

    def active_experiments(self, as_of: Optional[str] = None) -> list[ExperimentMetadata]:
        """Return experiments that are currently active (started, not ended)."""
        ref = as_of or date.today().isoformat()
        result = []
        for exp in self._experiments.values():
            if exp.status not in ("active", ""):
                continue
            if exp.start_date and exp.start_date <= ref:
                if exp.end_date is None or exp.end_date >= ref:
                    result.append(exp)
        return result

    def assign_variant(self, experiment_id: str, trade_id: str) -> str:
        """Deterministic variant assignment via hash.

        Same algorithm across all bots for reproducible results.
        """
        exp = self.get(experiment_id)
        if exp is None or not exp.variants:
            return ""

        raw = f"{trade_id}|{experiment_id}"
        hash_val = int(hashlib.sha256(raw.encode()).hexdigest(), 16)

        if exp.variant_allocations:
            # Use explicit allocation percentages
            bucket = hash_val % 10000
            total_alloc = sum(exp.variant_allocations.values())
            if total_alloc <= 0:
                total_alloc = len(exp.variants) * 50.0

            cumulative = 0.0
            for vname in exp.variants:
                alloc = exp.variant_allocations.get(vname, 100.0 / len(exp.variants))
                cumulative += alloc / total_alloc * 10000
                if bucket < cumulative:
                    return vname
            return exp.variants[-1]
        else:
            # Equal allocation
            idx = hash_val % len(exp.variants)
            return exp.variants[idx]

    def get_variant_params(self, experiment_id: str, variant_name: str) -> dict:
        """Get parameter overrides for a variant."""
        exp = self.get(experiment_id)
        if exp is None:
            return {}
        return exp.variant_params.get(variant_name, {})

    def export_active(self, as_of: str | None = None) -> dict:
        """Export active experiment metadata for downstream consumption.

        Returns dict keyed by experiment_id with hypothesis, variants, and metrics.
        Suitable for inclusion in DailySnapshot.active_experiments.
        """
        active = self.active_experiments(as_of=as_of)
        ref = as_of or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return {
            exp.experiment_id: {
                "hypothesis": exp.hypothesis,
                "variants": exp.variants,
                "variant_params": exp.variant_params if exp.variant_params else None,
                "variant_allocations": exp.variant_allocations if exp.variant_allocations else None,
                "primary_metric": exp.primary_metric,
                "secondary_metrics": exp.secondary_metrics,
                "start_date": exp.start_date,
                "end_date": exp.end_date,
                "min_trades_per_variant": exp.min_trades_per_variant,
                "strategy_type": exp.strategy_type,
                "status": "active" if exp.end_date is None or exp.end_date >= ref
                    else "concluded",
                "title": exp.title if exp.title else None,
                "allocation_method": exp.allocation_method,
            }
            for exp in active
        }

    def all_experiments(self) -> list[ExperimentMetadata]:
        return list(self._experiments.values())
