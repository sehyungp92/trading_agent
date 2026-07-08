"""ExperimentRegistry — manages A/B experiment definitions and variant assignment.

Parses experiments from YAML config (legacy dict format or orchestrator list format),
provides deterministic variant assignment based on trade_id hashing, and exports
active experiments for DailySnapshot enrichment.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("instrumentation.experiment_registry")


@dataclass
class Experiment:
    """A single experiment definition."""

    experiment_id: str
    hypothesis: str = ""
    variants: list[str] = field(default_factory=lambda: ["control", "treatment"])
    primary_metric: str = "sharpe"
    start_date: str = ""
    end_date: str = ""
    strategy_type: str = ""          # "" = all strategies, "coordinator", "ATRSS", etc.
    concluded: bool = False
    variant_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    # variant_params example: {"tight_2atr": {"daily_mult": 2.0}, "control": {}}


class ExperimentRegistry:
    """Loads experiment definitions and provides variant assignment.

    Supports two config formats:

    1. Legacy dict (current swing_trader experiments.yaml)::

        exp_coord_tight_stop:
          hypothesis: "..."
          variants: ["control", "tight_2atr"]
          strategy_type: "coordinator"

    2. Orchestrator list format::

        - experiment_id: "exp_coord_tight_stop"
          hypothesis: "..."
          variants: ["control", "tight_2atr"]
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self._experiments: dict[str, Experiment] = {}
        self._load()

    def _load(self) -> None:
        """Load experiments from YAML config."""
        if self.config_path is None or not self.config_path.exists():
            return

        try:
            import yaml
            with open(self.config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if not raw:
                return

            if isinstance(raw, dict):
                # Legacy dict format: {exp_id: {hypothesis, variants, ...}}
                for exp_id, exp_data in raw.items():
                    if not isinstance(exp_data, dict):
                        continue
                    self._experiments[exp_id] = Experiment(
                        experiment_id=exp_id,
                        hypothesis=exp_data.get("hypothesis", ""),
                        variants=exp_data.get("variants", ["control", "treatment"]),
                        primary_metric=exp_data.get("primary_metric", "sharpe"),
                        start_date=str(exp_data.get("start_date", "")),
                        end_date=str(exp_data.get("end_date", "")),
                        strategy_type=exp_data.get("strategy_type", ""),
                        concluded=exp_data.get("concluded", False),
                        variant_params=exp_data.get("variant_params", {}),
                    )
            elif isinstance(raw, list):
                # Orchestrator list format: [{experiment_id, hypothesis, ...}]
                for exp_data in raw:
                    if not isinstance(exp_data, dict):
                        continue
                    exp_id = exp_data.get("experiment_id", "")
                    if not exp_id:
                        continue
                    self._experiments[exp_id] = Experiment(
                        experiment_id=exp_id,
                        hypothesis=exp_data.get("hypothesis", ""),
                        variants=exp_data.get("variants", ["control", "treatment"]),
                        primary_metric=exp_data.get("primary_metric", "sharpe"),
                        start_date=str(exp_data.get("start_date", "")),
                        end_date=str(exp_data.get("end_date", "")),
                        strategy_type=exp_data.get("strategy_type", ""),
                        concluded=exp_data.get("concluded", False),
                        variant_params=exp_data.get("variant_params", {}),
                    )
        except Exception as e:
            logger.warning("Failed to load experiments from %s: %s", self.config_path, e)

    def reload(self) -> None:
        """Reload experiments from disk."""
        self._experiments.clear()
        self._load()

    def active_experiments(self) -> list[Experiment]:
        """Return all non-concluded experiments."""
        today_str = date.today().isoformat()
        result = []
        for exp in self._experiments.values():
            if exp.concluded:
                continue
            if exp.end_date and exp.end_date < today_str:
                continue
            result.append(exp)
        return result

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get a specific experiment by ID."""
        return self._experiments.get(experiment_id)

    def assign_variant(self, experiment_id: str, trade_id: str) -> str:
        """Deterministic variant assignment based on trade_id hash.

        Uses SHA256(experiment_id + trade_id) mod len(variants) to assign.
        Consistent: same (experiment_id, trade_id) always gets same variant.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None or not exp.variants:
            return "control"

        raw = f"{experiment_id}|{trade_id}"
        h = int(hashlib.sha256(raw.encode()).hexdigest(), 16)
        idx = h % len(exp.variants)
        return exp.variants[idx]

    def get_variant_params(self, experiment_id: str, variant: str) -> dict[str, Any]:
        """Get parameter overrides for a specific variant."""
        exp = self._experiments.get(experiment_id)
        if exp is None:
            return {}
        return dict(exp.variant_params.get(variant, {}))

    def export_active(self) -> dict[str, dict]:
        """Export active experiments for DailySnapshot enrichment."""
        result = {}
        for exp in self.active_experiments():
            result[exp.experiment_id] = {
                "hypothesis": exp.hypothesis,
                "variants": exp.variants,
                "primary_metric": exp.primary_metric,
                "start_date": exp.start_date,
                "strategy_type": exp.strategy_type,
            }
        return result
